# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline acceptance (method/ACCEPTANCE.md): full-depth, full-width, real-weight prefill on the spec's
8x4 mesh with the golden trace's exact token ids; every layer's carried state vs the CPU trace.

  PREFILL_CHUNKED=0  one-shot: the whole trace in one forward        (recipe P1)
  PREFILL_CHUNKED=1  multi-chunk at the spec's chunk_size, cache-read (recipe P2)

Per-layer rows (``layer_pcc``): attention layers compare K and V ([1, 4, T, 256], post-RoPE K / raw V);
Gated-DeltaNet layers have no per-token K/V — their cache *is* the carried state, so the row's ``k`` is
the recurrent state ([1, 48, 128, 128]) and ``v`` the conv state ([1, 10240, 3]), the two tensors the
trace stores for those layers. The report is printed (``PCC REPORT {...}``) before the PCC assert, so a
failing run still shows every layer, and written to ``PREFILL_ACCEPTANCE_OUT`` only after it passes.

Trace formats: token ids either inline in ``metadata.json`` (``token_ids``) or, for
agentic-prefill-goldens traces, the first ``n_tokens`` of the shared ``token_cache`` file it points to.
(pattern: minimax_m3/tests/galaxy_prefill_kv_pcc.py)
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.qwen_3_8_27b.config import Qwen38Config
from models.demos.qwen_3_8_27b.reference import golden
from models.demos.qwen_3_8_27b.reference.checkpoint import CheckpointReader, checkpoint_dir
from models.demos.qwen_3_8_27b.tests.common import pcc
from models.demos.qwen_3_8_27b.tt.common import weight_cache_dir
from models.demos.qwen_3_8_27b.tt.kv_cache import cache_capacity
from models.demos.qwen_3_8_27b.tt.mesh import fabric_name
from models.demos.qwen_3_8_27b.tt.model import CheckpointWeights, TtQwen38Model, gather_hidden
from models.demos.qwen_3_8_27b.tt.runtime import TtPrefillRuntime


def _golden(trace_dir: Path, layer: int, is_full: bool):
    with safe_open(str(trace_dir / "kv_cache" / f"layer_{layer}.safetensors"), "pt") as f:
        if is_full:
            return f.get_tensor(f"key_cache_layer_{layer}").float(), f.get_tensor(f"value_cache_layer_{layer}").float()
        return f.get_tensor(f"recurrent_state_layer_{layer}").float(), f.get_tensor(f"conv_state_layer_{layer}").float()


@pytest.mark.timeout(7200)  # cold weight conversion + 64-layer read-back exceed the repo's 300 s default
def test_prefill_kv(mesh, mesh_config, ccl_manager, spec):
    chunked = os.environ.get("PREFILL_CHUNKED", "0") == "1"
    trace_dir = Path(os.environ["PREFILL_TRACE_DIR"])
    out_path = os.environ.get("PREFILL_ACCEPTANCE_OUT")
    hf_dir = checkpoint_dir()

    # dimensions come from the checkpoint's own config.json (not from constants)
    cfg = Qwen38Config.from_hf_json(hf_dir / "config.json")
    meta = json.loads((trace_dir / "metadata.json").read_text())
    token_ids = golden.trace_token_ids(trace_dir)
    T = token_ids.numel()
    chunk = spec.chunk_size
    assert meta["num_layers"] == cfg.num_hidden_layers, "trace depth != model depth"
    assert T > chunk, f"trace has {T} tokens; must exceed chunk_size {chunk} to exercise multi-chunk"
    assert T <= spec.max_seq_len
    assert T % chunk == 0, "the trace fills whole chunks (no padded tail) — keeps one-shot and chunked comparable"
    assert tuple(mesh.shape) == spec.mesh_shape

    t0 = time.time()
    model = TtQwen38Model(
        mesh_config, ccl_manager, cfg, spec, CheckpointWeights(CheckpointReader(hf_dir)), cache=weight_cache_dir(mesh)
    )
    logger.info(f"model built in {time.time() - t0:.0f}s ({len(model.layers)} layers)")
    capacity = cache_capacity(spec.max_seq_len, [chunk, T])
    rt = TtPrefillRuntime(model, chunk_size=chunk, max_seq_len=spec.max_seq_len, capacity=capacity)
    caches = rt.allocate_caches()

    t0 = time.time()
    if chunked:
        outs = []
        for s in range(0, T, chunk):
            o = rt.prefill_chunk(rt.make_chunk_input(token_ids[s : s + chunk]), caches, 0, s, s + chunk)
            outs.append(gather_hidden(o, mesh_config))
            ttnn.deallocate(o)
        hidden = torch.cat(outs, dim=1)
        period = chunk
    else:
        o = rt.prefill_one_shot(token_ids, caches, 0)
        hidden = gather_hidden(o, mesh_config)
        ttnn.deallocate(o)
        period = T
    ttnn.synchronize_device(mesh)
    elapsed = time.time() - t0
    logger.info(
        f"prefill {'chunked' if chunked else 'one-shot'}: {T} tokens in {elapsed:.1f}s ({T / elapsed:.0f} tok/s)"
    )

    with safe_open(str(trace_dir / "final_hidden.safetensors"), "pt") as f:
        e2e = pcc(hidden, f.get_tensor("final_hidden").float())
    logger.info(f"e2e final_hidden PCC = {e2e:.6f}")

    rows = []
    for layer in range(cfg.num_hidden_layers):
        is_full = cfg.is_full_attention(layer)
        got_a, got_b = model.read_layer_state(caches, layer, n_tokens=T, period=period)
        want_a, want_b = _golden(trace_dir, layer, is_full)
        assert got_a.shape == want_a.shape and got_b.shape == want_b.shape, (layer, got_a.shape, want_a.shape)
        pa, pb = pcc(got_a, want_a), pcc(got_b, want_b)
        rows.append({"layer": layer, "k": pa, "v": pb})
        kind = "K/V" if is_full else "recurrent/conv"
        flag = "" if min(pa, pb) >= spec.pcc_target else "  (below pcc_target)"
        logger.info(f"layer {layer:2d} {cfg.layer_types[layer]:17s} {kind:15s} PCC {pa:.6f} / {pb:.6f}{flag}")
        print(f"layer {layer:2d} {cfg.layer_types[layer]} {kind} PCC {pa:.6f} / {pb:.6f}{flag}")

    worst = min(min(r["k"], r["v"]) for r in rows)
    logger.info(f"min per-layer PCC = {worst:.6f} (lower bound {spec.pcc_lower_bound}, target {spec.pcc_target})")
    report = {
        "mode": "chunked" if chunked else "one_shot",
        "python": sys.executable,
        "trace": str(trace_dir),
        "num_layers": len(model.layers),
        "hidden_size": cfg.hidden_size,
        "seq_len": T,
        "chunk_size": chunk,
        "sp": mesh_config.sp,
        "tp": mesh_config.tp,
        "target_hw": spec.target_hw,
        "fabric": fabric_name(),
        "e2e_final_hidden_pcc": e2e,
        "min_layer_pcc": worst,
        "gdn_layer_state_mapping": {"k": "recurrent_state", "v": "conv_state"},
        "layer_pcc": rows,
    }
    # printed before the asserts so a failing run still shows every layer
    print("PCC REPORT " + json.dumps(report), flush=True)
    for r in rows:
        for key in ("k", "v"):
            assert math.isfinite(r[key]), f"layer {r['layer']} {key} PCC is not finite"
            assert (
                r[key] >= spec.pcc_lower_bound
            ), f"layer {r['layer']} {key}: PCC {r[key]:.6f} < pcc_lower_bound {spec.pcc_lower_bound}"

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
        logger.info(f"acceptance report -> {out_path}")
