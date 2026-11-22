# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 layers 0..N-1 run free on one 4x4 stage (SP on axis 1) over an FP64 chain's prompt, chunk by chunk.

The device half of ``device_chain.py`` (its docstring has the scoring). The teacher-fed runs
(``test_kda_chain_feed.py``, ``test_moe_teacher_feed.py``, ``test_dsa_teacher_feed.py``) feed each layer the
chain's FP64 input; here no layer is fed: the production stage runtime of the chain's layers (the K3.1 depth
ladder's path: ``k31_stage.window_params`` + ``ADAPTER.build_runtime`` + ``prefill_chunk``, eager, one slot) embeds
the chain's token ids on device and every layer reads the device's own previous layers, with the KDA carry and the
DSA caches carried across 2560-token chunks as chunked prefill carries them. The device embedding is a lookup of the
checkpoint's bf16 table, the chain's embedding exactly.

``$TT_DC_CHAIN`` picks the reference and so the layers (``device_chain.chain_taps``): the 1M chain
``k31-l0-3-fp64-1m`` (layers 0-3, the default) or the deep chain ``k31-l0-7-fp64-256k`` (layers 0-7). Each layer's
taps go to ``$TT_DC_OUT`` in the teacher-feed formats:

- ``kda-l<l>/``: the KDA output (``attn_out``, layers >= 1; the chains have no layer-0 KDA output) and the carry
  every ``$TT_DC_STATE_EVERY`` rows;
- ``moe-l<l>/``: the MoE output and the device routing (``device_routing``, as the MoE teacher feed reads it);
- ``dsa-l<l>/``: the attention output, the top-k and, at the end, the kvpe and index-key caches;
- ``stream/``: what feeds each layer and what each layer leaves: the KDA inputs (after ``input_layernorm``), the
  MoE inputs (after ``post_attention_layernorm``), the DSA layers' AttnRes reads, and the live residual after every
  layer but the last.

Against the 1M chain each run directory's ``input`` names the teacher-fed input of the same layer, whose manifest
names the chain the teacher-feed scorers read; against the deep chain the manifests name the chain (``chain``) and
``device_chain.py score-deep`` scores every tap. The device reads neither. Skips unless ``$TT_DC_OUT`` is set.

    TT_DC_OUT=/mnt/.../k31-1m-device-chain scripts/run_safe_pytest.sh \\
        "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_device_chain.py::test_k31_layers_run_free[blackhole-4x4-sp1]"
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.runners.runner_utils import MTP_PAD_TOKEN_ID
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST, provenance, token_ids_sha256
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage
from models.demos.deepseek_v3_d_p.tests.kimi_k3.device_chain import CHAIN, INPUTS, TRACE, chain_taps, load_token_ids
from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import CACHE_LEN, cache_rows, compose
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import chunk_path, finish_run, start_run
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_kda_chain_feed import read_carry
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_moe_teacher_feed import device_routing
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import k31_topk_rows

PLACEMENT_4X4 = [p for p in k31_stage.PLACEMENT if p.id == "4x4-sp1"]
OUT = os.environ.get("TT_DC_OUT")


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.skipif(not OUT, reason="set TT_DC_OUT (the run's output dir)")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENT_4X4, indirect=True)
def test_k31_layers_run_free(mesh_device, device_params):
    out = Path(OUT)
    chain = Path(os.environ.get("TT_DC_CHAIN", CHAIN))
    chain_manifest = json.loads((chain / MANIFEST).read_text())
    spec = chain_taps(chain)
    num_layers, kda_fed = spec.num_layers, tuple(l for l in spec.kda if l)
    count = int(os.environ.get("TT_DC_COUNT", chain_manifest["count"]))
    state_every = int(os.environ.get("TT_DC_STATE_EVERY", 20480))
    token_ids = load_token_ids(Path(os.environ.get("TT_DC_TRACE", TRACE)), chain_manifest["count"])
    assert token_ids_sha256(token_ids.tolist()) == chain_manifest["token_ids_sha256"], "not the chain's tokens"
    mesh_shape = tuple(mesh_device.shape)
    layout = k31_stage.STAGE_LAYOUTS[mesh_shape]
    chunk = layout.chunk
    assert state_every % chunk == 0, f"state reads every {state_every} rows must end on {chunk}-row chunks"
    max_seq_len = -(-count // CACHE_LEN) * CACHE_LEN
    # The window is the whole model as the runtime sees it (first and last rank): a rank that hands off mid-block
    # (layers 0-11 are AttnRes block 0) has no socket spec; the last rank's final norm is not scored.
    hf_config, params = k31_stage.window_params(
        mesh_shape, 0, num_layers, max_seq_len, is_first_rank=True, is_last_rank=True
    )
    common = {
        "count": count,
        "chunk": chunk,
        "max_seq_len": max_seq_len,
        "mesh": list(mesh_shape),
        "sp_axis": layout.sp_axis,
        "fed": f"free: layers 0-{num_layers - 1} chained on device from the token ids",
        "chain": str(chain),
        "weight_cache": str(params.weight_cache_path),
        **provenance(),
    }
    dirs = spec.dirs(out)
    manifests = {}
    for name, path in dirs.items():
        manifests[name] = {**common, "tap": name}
        if name != "stream":
            manifests[name]["layer"] = int(name.split("-l")[1])
        if name in INPUTS and chain == Path(CHAIN):
            layer, source = INPUTS[name]
            built = json.loads((source / MANIFEST).read_text())
            keep = ("trace", "trace_commit", "input", "checkpoint", "rebuild", "chain")
            manifests[name].update(input=str(source), input_manifest={k: built[k] for k in keep if k in built})
        start_run(path, manifests[name])

    runtime = k31_stage.ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    caches = k31_stage.ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime.compile(caches)
    model = runtime.model
    schedule = model.schedule
    mla = [schedule.is_mla(l) for l in range(num_layers)]
    assert mla == [l in spec.dsa for l in range(num_layers)], f"device layers' attention {mla} is not the chain's"
    kda_slab = {l: i for i, l in enumerate(schedule.rank_kda_layer_ids())}  # the KDA state slabs' layer axis

    taps: dict[str, torch.Tensor] = {}
    rows = {"n": 0}  # this chunk's real rows

    def keep(name, tensor):
        taps[name] = compose(mesh_device, tensor, rows["n"]).to(torch.bfloat16)

    restore = []
    for local, block in enumerate(model.layers):
        attention, layer = block.attention, local

        def attention_forward(normed, ctx, _inner=attention.forward, _layer=layer):
            if _layer in kda_fed:
                keep(f"kda_input_{_layer}", normed)
            out_ = _inner(normed, ctx)
            if _layer in kda_fed or _layer in spec.dsa:
                keep(f"attn_out_{_layer}", out_)
            return out_

        restore.append((attention, "forward"))
        attention.forward = attention_forward
        if layer in spec.dsa:
            mla_module = attention.mla

            def mla_forward(*args, _inner=mla_module.forward, _layer=layer, **kwargs):
                out_, indices = _inner(*args, return_indexer_indices=True, **kwargs)
                topk = k31_topk_rows(mesh_device, indices, chunk, layout.sp_axis, layout.tp_axis)[: rows["n"]]
                taps[f"topk_{_layer}"] = torch.where(topk >= 2**31, -1, topk).to(torch.int32)
                ttnn.deallocate(indices)
                return out_

            restore.append((mla_module, "forward"))
            mla_module.forward = mla_forward
        if layer in spec.moe:

            def ffn_path(ffn_norm_out, *, _inner=block._ffn_path, _layer=layer, _block=block, **kwargs):
                keep(f"moe_input_{_layer}", ffn_norm_out)
                out_ = _inner(ffn_norm_out, **kwargs)
                keep(f"moe_out_{_layer}", out_)
                ids, weights = device_routing(_block.ffn, _block.ffn._block(chunk), rows["n"])
                taps[f"expert_ids_{_layer}"] = ids.to(torch.int16)
                taps[f"expert_weights_{_layer}"] = weights
                return out_

            restore.append((block, "_ffn_path"))
            block._ffn_path = ffn_path

    def dsa_input_tap(global_idx, hidden):  # fires inside layer global_idx + 1 with its AttnRes read
        if global_idx + 1 in spec.dsa:
            keep(f"dsa_input_{global_idx + 1}", hidden)

    def layer_tap(local_idx, hidden):  # the live residual after each layer
        if local_idx < num_layers - 1:
            keep(f"live_{local_idx}", hidden)

    inner_forward = model.forward

    def tapped_forward(*args, on_layer_hidden=None, **kwargs):  # the runtime's own hook (the drafter's) still fires
        def both(global_idx, hidden, _runtime_hook=on_layer_hidden):
            dsa_input_tap(global_idx, hidden)
            if _runtime_hook is not None:
                _runtime_hook(global_idx, hidden)

        return inner_forward(*args, layer_tap=layer_tap, on_layer_hidden=both, **kwargs)

    model.forward = tapped_forward
    restore.append((model, "forward"))

    expected = {*spec.stream_keys, *(f"attn_out_{l}" for l in (*kda_fed, *spec.dsa))}
    expected |= {f"topk_{l}" for l in spec.dsa} | {f"{k}_{l}" for l in spec.moe for k in ("moe_out", "expert_ids")}
    num_chunks = -(-count // chunk)
    states = {l: [] for l in spec.kda}
    began = time.perf_counter()
    try:
        for c in range(num_chunks):
            lo, hi = c * chunk, min((c + 1) * chunk, count)
            rows["n"] = hi - lo
            ids = token_ids[lo:hi].tolist()
            activation = runtime.make_chunk_input(ids + [MTP_PAD_TOKEN_ID] * (chunk - len(ids)))
            activation = runtime.prefill_chunk(activation, caches, slot_id=0, actual_start=lo, actual_end=hi)
            if activation is not None:
                ttnn.deallocate(activation)
            ttnn.synchronize_device(mesh_device)
            missing = expected - taps.keys()
            assert not missing, f"chunk {c}: taps {sorted(missing)} never fired"
            for name, value in taps.items():
                if value.is_floating_point():
                    assert torch.isfinite(value.float()).all(), f"non-finite {name} rows in chunk {c}"
            base = {"start": lo, "end": hi}
            for l in kda_fed:
                torch.save({**base, "attn_out": taps[f"attn_out_{l}"]}, chunk_path(dirs[f"kda-l{l}"], c))
            for l in spec.moe:
                moe = {"moe_out": taps[f"moe_out_{l}"], "expert_ids": taps[f"expert_ids_{l}"]}
                torch.save(
                    {**base, **moe, "expert_weights": taps[f"expert_weights_{l}"]}, chunk_path(dirs[f"moe-l{l}"], c)
                )
            for l in spec.dsa:
                dsa = {"attn_out": taps[f"attn_out_{l}"], "topk": taps[f"topk_{l}"]}
                torch.save({**base, **dsa}, chunk_path(dirs[f"dsa-l{l}"], c))
            torch.save({**base, **{k: taps[k] for k in spec.stream_keys}}, chunk_path(dirs["stream"], c))
            taps.clear()
            if hi % state_every == 0:
                for l in spec.kda:
                    name = f"state_{hi:08d}.pt"
                    torch.save(
                        read_carry(mesh_device, caches.kda_states, kda_slab[l], layout.sp_axis, layout.tp_axis),
                        dirs[f"kda-l{l}"] / name,
                    )
                    states[l].append({"row": hi, "file": name})
            logger.info(
                f"DEVICE_CHAIN chunk {c + 1}/{num_chunks} rows [{lo}, {hi}) done, {time.perf_counter() - began:.1f} s"
            )
        kvpe = cache_rows(mesh_device, caches.kvpe, chunk, count)
        index = cache_rows(mesh_device, caches.index, chunk, count)
        for l in spec.dsa:
            slot = schedule.kv_slot_of_local[l]
            for name, values in (("kvpe", kvpe[slot]), ("index", index[slot])):
                assert torch.isfinite(values).all(), f"non-finite rows in layer {l}'s {name} cache"
            torch.save(
                {"kvpe": kvpe[slot].to(torch.bfloat16), "index": index[slot].to(torch.bfloat16)},
                dirs[f"dsa-l{l}"] / "caches.pt",
            )
    finally:
        for owner, attribute in restore:
            vars(owner).pop(attribute, None)  # the instance attribute; the class's method shows through again
        if caches.kda_states is not None:
            caches.kda_states.deallocate()
        ttnn.deallocate(caches.kvpe.storage)
        if caches.index is not None:
            ttnn.deallocate(caches.index)
        runtime.release_trace()
    seconds = time.perf_counter() - began
    for name, path in dirs.items():
        layer = int(name[5:]) if name.startswith("kda-l") else None
        extra = {"states": states[layer]} if layer is not None else {}
        finish_run(path, {**manifests[name], **extra}, seconds)
    logger.info(
        f"DEVICE_CHAIN done: layers 0-{num_layers - 1}, {num_chunks} chunks, {count} rows, {seconds:.1f} s -> {out}"
    )
