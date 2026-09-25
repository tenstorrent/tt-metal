# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline acceptance (method/ACCEPTANCE.md; recipe P1 one-shot / P2 chunked).

A real, full-depth (88 layers), full-width (hidden 12288) prefill on the spec's 8x4 Blackhole Galaxy with
the real checkpoint, fed the golden trace's exact token ids; every layer's K and V read back from the
device KV cache and PCC'd against the saved CPU trace. Nothing is reduced, skipped or substituted.

Env (all required except PREFILL_CHUNKED, default 0):
  PREFILL_SPEC            spec JSON: mesh, chunk size, dataformats, pcc_lower_bound
  PREFILL_HF_MODEL        checkpoint dir (HF_MODEL accepted as fallback); dims come from its config.json
  PREFILL_TRACE_DIR       golden trace (metadata.json + kv_cache/layer_N.safetensors)
  PREFILL_CHUNKED         0 = one-shot (one chunk over the whole prompt), 1 = spec chunk_size chunks
  PREFILL_ACCEPTANCE_OUT  JSON report path, written after the PCC asserts
Optional: MISTRAL_TT_CACHE (tilized weight cache root), MISTRAL_FORCE_LOAD_WEIGHTS=1, MISTRAL_FABRIC.
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

import ttnn
from models.demos.mistral_medium_3_5_128b.config import (
    MistralMediumConfig,
    load_prefill_spec,
    pcc_thresholds,
    resolve_dataformats,
    ttnn_dtype,
)
from models.demos.mistral_medium_3_5_128b.tt.common import dtype_tag
from models.demos.mistral_medium_3_5_128b.tt.fabric import fabric_mode
from models.demos.mistral_medium_3_5_128b.tt.kv_validation import layer_kv_pcc, trace_token_ids
from models.demos.mistral_medium_3_5_128b.tt.runtime import PrefillRuntime, PrefillRuntimeConfig
from models.demos.mistral_medium_3_5_128b.tt.weights import mark_cache_complete, select_weights, weight_cache_dir


def _required(*names):
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    raise AssertionError(f"acceptance test needs {' or '.join(names)}")


@pytest.mark.timeout(7200)
def test_prefill_kv(galaxy_mesh):
    spec_path = _required("PREFILL_SPEC")
    ckpt = _required("PREFILL_HF_MODEL", "HF_MODEL")
    trace = _required("PREFILL_TRACE_DIR")
    out_path = _required("PREFILL_ACCEPTANCE_OUT")
    mode_flag = os.environ.get("PREFILL_CHUNKED", "0")
    assert mode_flag in ("0", "1"), f"PREFILL_CHUNKED must be 0 or 1, got {mode_flag!r}"
    chunked = mode_flag == "1"

    spec = load_prefill_spec(spec_path)
    target_pcc, lower_bound = pcc_thresholds(spec)
    sp, tp = spec["parallelism"]["sp"], spec["parallelism"]["tp"]
    assert spec["target_hw"] == "bh_galaxy" and "blackhole" in ttnn.get_arch_name().lower(), ttnn.get_arch_name()
    assert tuple(galaxy_mesh.shape) == (sp, tp), f"mesh {tuple(galaxy_mesh.shape)} != spec SP x TP ({sp}, {tp})"
    cfg = MistralMediumConfig.from_json(Path(ckpt) / "config.json")

    token_ids, meta = trace_token_ids(trace)
    n = len(token_ids)
    chunk_size = spec["shapes"]["chunk_size"]
    assert meta.get("num_layers", cfg.num_hidden_layers) == cfg.num_hidden_layers and not meta.get("reduced_depth")
    assert chunk_size < n <= spec["shapes"]["max_seq_len"], f"trace of {n} tokens must span >1 chunk of {chunk_size}"
    if chunked:
        run_chunk = chunk_size
        capacity = PrefillRuntime.pad_to(n, chunk_size)
    else:
        capacity = PrefillRuntime.pad_to(n, ttnn.TILE_SIZE * sp)
        run_chunk = capacity  # one chunk over the whole prompt

    df = resolve_dataformats(spec)
    dtypes = {k: ttnn_dtype(v) for k, v in df.items()}
    cache_dir = weight_cache_dir(ckpt, tuple(galaxy_mesh.shape))
    tag = f"L0-{cfg.num_hidden_layers}_emb1d_attn{dtype_tag(dtypes['attention'])}_mlp" + "".join(
        dtype_tag(dtypes[k]) for k in ("mlp_gate", "mlp_up", "mlp_down")
    )
    weights, from_cache = select_weights(ckpt, cache_dir, tag)
    logger.info(
        f"[acceptance] mode={'chunked' if chunked else 'one_shot'} n_tokens={n} chunk={run_chunk} capacity={capacity} "
        f"fabric={fabric_mode()} weights={'tilized cache ' if from_cache else 'checkpoint -> '}{cache_dir} dtypes={df}"
    )

    runtime = PrefillRuntime(
        galaxy_mesh,
        cfg,
        weights,
        PrefillRuntimeConfig(
            num_layers=cfg.num_hidden_layers,
            max_seq_len=capacity,
            chunk_size=run_chunk,
            dtypes=dtypes,
            cache_dtype=dtypes["kv_cache"],
            weight_cache_path=str(cache_dir),
            pad_token_id=cfg.pad_token_id,
        ),
    )
    if not from_cache:
        mark_cache_complete(cache_dir, tag)
    kv_cache = runtime.allocate_kv_cache()
    dram = ttnn.get_memory_view(galaxy_mesh, ttnn.BufferType.DRAM)
    logger.info(f"[acceptance] DRAM free per bank after build: {dram.total_bytes_free_per_bank / 2**30:.3f} GiB")

    t0 = time.perf_counter()
    n_chunks = runtime.prefill(token_ids, kv_cache, slot_id=0)
    prefill_s = time.perf_counter() - t0
    if not chunked:
        assert n_chunks == 1
    else:
        assert n_chunks == math.ceil(n / chunk_size) > 1

    k_blk, v_blk = runtime.read_slot_kv(kv_cache, 0)
    rows = layer_kv_pcc(
        k_blk, v_blk, trace, n_tokens=n, sp=sp, chunk_size=run_chunk, max_seq_len=capacity, head_dim=cfg.head_dim
    )
    for r in rows:
        note = "" if min(r["k"], r["v"]) >= target_pcc else f"  [below pcc_target {target_pcc}]"
        logger.info(f"[acceptance] layer {r['layer']:2d}: K={r['k']:.6f} V={r['v']:.6f}{note}")
    min_k, min_v = min(r["k"] for r in rows), min(r["v"] for r in rows)
    logger.info(
        f"[acceptance] {n_chunks} chunk(s), prefill {prefill_s:.1f}s ({n / prefill_s:.0f} tok/s), build "
        f"{runtime.build_seconds:.1f}s; worst K {min_k:.6f} V {min_v:.6f} (bound {lower_bound}, target {target_pcc})"
    )

    assert [r["layer"] for r in rows] == list(range(cfg.num_hidden_layers))
    for r in rows:
        for c in ("k", "v"):
            assert math.isfinite(r[c]), f"layer {r['layer']} {c} PCC is not finite"
            assert r[c] >= lower_bound, f"layer {r['layer']} {c} PCC {r[c]:.6f} < pcc_lower_bound {lower_bound}"
    assert torch.isfinite(k_blk).all() and torch.isfinite(v_blk).all()

    report = {
        "mode": "chunked" if chunked else "one_shot",
        "python": sys.executable,
        "num_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "seq_len": n,
        "chunk_size": chunk_size,
        "sp": sp,
        "tp": tp,
        "target_hw": spec["target_hw"],
        "layer_pcc": rows,
        "n_chunks": n_chunks,
        "kv_capacity": capacity,
        "fabric": fabric_mode(),
        "pcc_lower_bound": lower_bound,
        "pcc_target": target_pcc,
        "min_k": min_k,
        "min_v": min_v,
        "prefill_seconds": prefill_s,
        "weights_from_tilized_cache": from_cache,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1) + "\n")
