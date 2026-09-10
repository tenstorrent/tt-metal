#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Depth decoder head split / merge alternatives (stage 07): ``nlp_create_qkv_heads`` runs on 2 cores for the
``[2, 1, 32, 12288]`` depth QKV tensor (113 us) and ``nlp_concat_heads`` 38 us; compare with reshape + permute
(multi-core generic permute) and with ``ttnn.transformer.split_query_key_value_and_split_heads``.

    with_hw_lock timeout 600 $MM3_PY $MM3_MODEL_DIR/scripts/probe_depth_heads.py
"""

import json
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn

MODEL_DIR = Path(__file__).resolve().parents[1]
B, S, H, D = 2, 32, 16, 256


def timed(dev, fn, ops=8, reps=5):
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    outs = [fn() for _ in range(ops)]
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / (reps * ops)
    out = outs[0]
    res = [ttnn.to_torch(o).float() for o in (out if isinstance(out, (tuple, list)) else (out,))]
    ttnn.release_trace(dev, tid)
    return dt, res


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=30_000_000)
    dev.enable_program_cache()
    res = {}
    try:
        g = torch.Generator().manual_seed(0)
        qkv_t = torch.randn(B, 1, S, 3 * H * D, generator=g).to(torch.bfloat16)
        qkv = ttnn.from_torch(
            qkv_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        # reference split
        dt, (q0, k0, v0) = timed(
            dev,
            lambda: ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False, memory_config=ttnn.L1_MEMORY_CONFIG
            ),
        )
        res["nlp_create_qkv_heads_us"] = dt * 1e6
        logger.info(f"nlp_create_qkv_heads: {dt*1e6:.1f} us")

        def permute_split():
            outs = []
            for i in range(3):
                part = ttnn.slice(
                    qkv, [0, 0, 0, i * H * D], [B, 1, S, (i + 1) * H * D], memory_config=ttnn.L1_MEMORY_CONFIG
                )
                part = ttnn.reshape(part, (B, S, H, D))
                outs.append(ttnn.permute(part, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG))
            return outs

        try:
            dt, (q1, k1, v1) = timed(dev, permute_split)
            res["slice_reshape_permute_us"] = dt * 1e6
            res["permute_matches"] = bool(torch.equal(q0, q1) and torch.equal(k0, k1) and torch.equal(v0, v1))
            logger.info(f"slice+reshape+permute: {dt*1e6:.1f} us, matches {res['permute_matches']}")
        except Exception as exc:  # noqa: BLE001
            res["slice_reshape_permute_error"] = repr(exc)[:200]
            logger.warning(repr(exc)[:200])

        try:
            dt, outs = timed(
                dev,
                lambda: ttnn.transformer.split_query_key_value_and_split_heads(
                    ttnn.reshape(qkv, (B, S, 3 * H * D)),
                    num_heads=H,
                    transpose_key=False,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                ),
            )
            res["split_qkv_and_split_heads_us"] = dt * 1e6
            res["split_qkv_matches"] = bool(
                torch.equal(q0, outs[0]) and torch.equal(k0, outs[1]) and torch.equal(v0, outs[2])
            )
            logger.info(f"split_query_key_value_and_split_heads: {dt*1e6:.1f} us, matches {res['split_qkv_matches']}")
        except Exception as exc:  # noqa: BLE001
            res["split_qkv_error"] = repr(exc)[:200]
            logger.warning(repr(exc)[:200])

        # merge
        attn = ttnn.from_torch(
            torch.randn(B, H, S, D, generator=g).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        dt, (m0,) = timed(dev, lambda: ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.L1_MEMORY_CONFIG))
        res["nlp_concat_heads_us"] = dt * 1e6
        logger.info(f"nlp_concat_heads: {dt*1e6:.1f} us")
        try:
            dt, (m1,) = timed(
                dev,
                lambda: ttnn.reshape(
                    ttnn.permute(attn, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG), (B, 1, S, H * D)
                ),
            )
            res["permute_merge_us"] = dt * 1e6
            res["permute_merge_matches"] = bool(torch.equal(m0, m1))
            logger.info(f"permute merge: {dt*1e6:.1f} us, matches {res['permute_merge_matches']}")
        except Exception as exc:  # noqa: BLE001
            res["permute_merge_error"] = repr(exc)[:200]
            logger.warning(repr(exc)[:200])
        try:
            dt, (m2,) = timed(
                dev, lambda: ttnn.transformer.concatenate_heads(attn, memory_config=ttnn.L1_MEMORY_CONFIG)
            )
            res["concatenate_heads_us"] = dt * 1e6
            res["concatenate_heads_matches"] = bool(torch.equal(m0.reshape(B, S, H * D), m2.reshape(B, S, H * D)))
            logger.info(f"transformer.concatenate_heads: {dt*1e6:.1f} us, matches {res['concatenate_heads_matches']}")
        except Exception as exc:  # noqa: BLE001
            res["concatenate_heads_error"] = repr(exc)[:200]
            logger.warning(repr(exc)[:200])
    finally:
        ttnn.close_mesh_device(dev)
    d = MODEL_DIR / "doc" / "optimize" / "sweeps"
    d.mkdir(parents=True, exist_ok=True)
    (d / "depth_heads.json").write_text(json.dumps(res, indent=2) + "\n")
    logger.info(json.dumps(res))


if __name__ == "__main__":
    main()
