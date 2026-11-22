# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 layers 0-3 run free on one 4x4 stage (SP on axis 1) over the FP64 chain's prompt, chunk by chunk.

The device half of ``device_chain.py`` (its docstring has the scoring). The teacher-fed runs
(``test_kda_chain_feed.py``, ``test_moe_teacher_feed.py``, ``test_dsa_teacher_feed.py``) feed each layer the
chain's FP64 input; here no layer is fed: the production stage runtime of layers 0-3 (the K3.1 depth ladder's
path: ``k31_stage.window_params`` + ``ADAPTER.build_runtime`` + ``prefill_chunk``, eager, one slot) embeds the
chain's token ids on device and every layer reads the device's own previous layers, with the KDA carry and the DSA
caches carried across 2560-token chunks as chunked prefill carries them. The device embedding is a lookup of the
checkpoint's bf16 table, the chain's embedding exactly.

Each layer's taps go to ``$TT_DC_OUT`` in the teacher-feed formats, so each teacher-feed scorer scores them
against the chain unchanged:

- ``kda-l{1,2}/``: the KDA output (``attn_out``) and the carry every ``$TT_DC_STATE_EVERY`` rows; ``kda-l0/``
  the carry only (the chain has no layer-0 KDA output, only its state);
- ``moe-l{1,2}/``: the MoE output and the device routing (``device_routing``, as the MoE teacher feed reads it);
- ``dsa-l3/``: the attention output, the top-k and, at the end, the kvpe and index-key caches;
- ``stream/``: what feeds each layer and what each layer leaves, for the chain's inputs: the KDA inputs of layers
  1-2 (after ``input_layernorm``), the MoE inputs of layers 1-2 (after ``post_attention_layernorm``), the DSA
  layer's AttnRes read (``dsa_input_3``), and the live residual after layers 0-2.

Each run directory's ``input`` names the teacher-fed input of the same layer: its manifest names the reference
(chain) the scorers read. The device does not read it. Skips unless ``$TT_DC_OUT`` is set.

    TT_DC_OUT=/mnt/.../k31-1m-device-chain scripts/run_safe_pytest.sh \\
        "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_device_chain.py::test_k31_layers_0_3_run_free[blackhole-4x4-sp1]"
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
from models.demos.deepseek_v3_d_p.tests.kimi_k3.device_chain import (
    CHAIN,
    INPUTS,
    STREAM_KEYS,
    TRACE,
    load_token_ids,
    tap_dirs,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.synthetic import CACHE_LEN, cache_rows, compose
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import chunk_path, finish_run, start_run
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_kda_chain_feed import read_carry
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_moe_teacher_feed import device_routing
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import k31_topk_rows

PLACEMENT_4X4 = [p for p in k31_stage.PLACEMENT if p.id == "4x4-sp1"]
OUT = os.environ.get("TT_DC_OUT")
NUM_LAYERS = 4


@pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3.1 runs on Blackhole only")
@pytest.mark.skipif(not OUT, reason="set TT_DC_OUT (the run's output dir)")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", PLACEMENT_4X4, indirect=True)
def test_k31_layers_0_3_run_free(mesh_device, device_params):
    out = Path(OUT)
    chain = Path(os.environ.get("TT_DC_CHAIN", CHAIN))
    chain_manifest = json.loads((chain / MANIFEST).read_text())
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
    # (layer 3 is inside AttnRes block 0) has no socket spec; the last rank's final norm is not scored.
    hf_config, params = k31_stage.window_params(
        mesh_shape, 0, NUM_LAYERS, max_seq_len, is_first_rank=True, is_last_rank=True
    )
    common = {
        "count": count,
        "chunk": chunk,
        "max_seq_len": max_seq_len,
        "mesh": list(mesh_shape),
        "sp_axis": layout.sp_axis,
        "fed": "free: layers 0-3 chained on device from the token ids",
        "chain": str(chain),
        "weight_cache": str(params.weight_cache_path),
        **provenance(),
    }
    dirs = tap_dirs(out)
    manifests = {}
    for name, path in dirs.items():
        manifests[name] = {**common, "tap": name}
        if name == "kda-l0":
            manifests[name]["layer"] = 0
        if name in INPUTS:
            layer, source = INPUTS[name]
            built = json.loads((source / MANIFEST).read_text())
            keep = ("trace", "trace_commit", "input", "checkpoint", "rebuild", "chain")
            manifests[name].update(
                layer=layer, input=str(source), input_manifest={k: built[k] for k in keep if k in built}
            )
        start_run(path, manifests[name])

    runtime = k31_stage.ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    caches = k31_stage.ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime.compile(caches)
    model = runtime.model
    schedule = model.schedule
    assert [schedule.is_mla(l) for l in range(NUM_LAYERS)] == [False, False, False, True], "K3.1 layers 0-3"

    taps: dict[str, torch.Tensor] = {}
    rows = {"n": 0}  # this chunk's real rows

    def keep(name, tensor):
        taps[name] = compose(mesh_device, tensor, rows["n"]).to(torch.bfloat16)

    restore = []
    for local, block in enumerate(model.layers):
        attention, layer = block.attention, local

        def attention_forward(normed, ctx, _inner=attention.forward, _layer=layer):
            if _layer in (1, 2):
                keep(f"kda_input_{_layer}", normed)
            out_ = _inner(normed, ctx)
            if _layer in (1, 2, 3):
                keep(f"attn_out_{_layer}", out_)
            return out_

        restore.append((attention, "forward"))
        attention.forward = attention_forward
        if layer == 3:
            mla = attention.mla

            def mla_forward(*args, _inner=mla.forward, **kwargs):
                out_, indices = _inner(*args, return_indexer_indices=True, **kwargs)
                topk = k31_topk_rows(mesh_device, indices, chunk, layout.sp_axis, layout.tp_axis)[: rows["n"]]
                taps["topk_3"] = torch.where(topk >= 2**31, -1, topk).to(torch.int32)
                ttnn.deallocate(indices)
                return out_

            restore.append((mla, "forward"))
            mla.forward = mla_forward
        if layer in (1, 2):

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

    def on_layer_hidden(global_idx, hidden):  # fires inside layer global_idx + 1 with its AttnRes read
        if global_idx == 2:
            keep("dsa_input_3", hidden)

    def layer_tap(local_idx, hidden):  # the live residual after each layer
        if local_idx < 3:
            keep(f"live_{local_idx}", hidden)

    inner_forward = model.forward

    def tapped_forward(*args, **kwargs):
        return inner_forward(*args, layer_tap=layer_tap, on_layer_hidden=on_layer_hidden, **kwargs)

    model.forward = tapped_forward
    restore.append((model, "forward"))

    num_chunks = -(-count // chunk)
    states = {l: [] for l in (0, 1, 2)}
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
            missing = {*STREAM_KEYS, "attn_out_1", "attn_out_2", "attn_out_3", "topk_3"} - taps.keys()
            assert not missing, f"chunk {c}: taps {sorted(missing)} never fired"
            for name, value in taps.items():
                if value.is_floating_point():
                    assert torch.isfinite(value.float()).all(), f"non-finite {name} rows in chunk {c}"
            base = {"start": lo, "end": hi}
            for l in (1, 2):
                torch.save({**base, "attn_out": taps[f"attn_out_{l}"]}, chunk_path(dirs[f"kda-l{l}"], c))
                moe = {"moe_out": taps[f"moe_out_{l}"], "expert_ids": taps[f"expert_ids_{l}"]}
                torch.save(
                    {**base, **moe, "expert_weights": taps[f"expert_weights_{l}"]}, chunk_path(dirs[f"moe-l{l}"], c)
                )
            torch.save({**base, "attn_out": taps["attn_out_3"], "topk": taps["topk_3"]}, chunk_path(dirs["dsa-l3"], c))
            torch.save({**base, **{k: taps[k] for k in STREAM_KEYS}}, chunk_path(dirs["stream"], c))
            taps.clear()
            if hi % state_every == 0:
                for l in (0, 1, 2):
                    name = f"state_{hi:08d}.pt"
                    torch.save(
                        read_carry(mesh_device, caches.kda_states, l, layout.sp_axis, layout.tp_axis),
                        dirs[f"kda-l{l}"] / name,
                    )
                    states[l].append({"row": hi, "file": name})
            logger.info(
                f"DEVICE_CHAIN chunk {c + 1}/{num_chunks} rows [{lo}, {hi}) done, {time.perf_counter() - began:.1f} s"
            )
        slot = schedule.kv_slot_of_local[3]
        kvpe = cache_rows(mesh_device, caches.kvpe, chunk, count)[slot]
        index = cache_rows(mesh_device, caches.index, chunk, count)[slot]
        for name, values in (("kvpe", kvpe), ("index", index)):
            assert torch.isfinite(values).all(), f"non-finite rows in the {name} cache"
        torch.save({"kvpe": kvpe.to(torch.bfloat16), "index": index.to(torch.bfloat16)}, dirs["dsa-l3"] / "caches.pt")
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
        layer = int(name[-1]) if name.startswith("kda-l") else None
        extra = {"states": states[layer]} if layer is not None else {}
        finish_run(path, {**manifests[name], **extra}, seconds)
    logger.info(f"DEVICE_CHAIN done: layers 0-3, {num_chunks} chunks, {count} rows, {seconds:.1f} s -> {out}")
