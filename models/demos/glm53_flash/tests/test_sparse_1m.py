# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""1 M-position traced sparse decode (OPT_SPARSE_1M deliverable 2, attention level): the layer-3 DSA attention
with its paged latent cache and pooled-key index allocated for ``max_batch`` users x 1 048 640 positions, user 0
seeded synthetically to N = 16384 / 131072 / 1048576 positions exactly as ``test_cache_1m`` does (same host twin,
same HF references), then at every N:

* correctness (1 user): two decode steps at N, N+1 with real attention-input rows through the new traced path
  (``SparseDecodeMeta``) vs the restricted HF attention forward (identical to the full forward up to summation
  order, ``test_cache_1m.test_host_reference_matches_hf``), the selected-id overlap, and the same step through the
  eager reference path (``GLM53_SPARSE_DECODE=eager``: dense flash-MLA bounded to 2050 + per-user sparse_sdpa over
  the whole-prefix copy) for the before/after timing on this build;
* timing: the DSA attention decode step captured as a trace and replayed for 1, 4 and 8 users of the chip at
  position N (users beyond 0 carry unseeded zero caches: the step's cost does not depend on the content).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_sparse_1m.py -s --timeout 7000

Knobs: GLM53_CACHE1M_N (curve points), GLM53_CACHE1M_CHUNK (seed chunk), GLM53_SPARSE1M_USERS (default 1,4,8).
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import load_config
from models.demos.glm53_flash.tests.test_cache_1m import (
    CHUNK,
    CURVE,
    LAYER,
    HostContext,
    _attn_keys,
    _check_chunk,
    _decode_rows,
    _overlap,
    _x_stats,
    hf_decode_step,
    synthetic_rows,
)
from models.demos.glm53_flash.tests.test_dsa import _hf_attention
from models.demos.glm53_flash.tests.utils import log_dram, pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import SparseDecodeMeta, TtDSA, TtLatentCache, decode_position_tensors

USERS = [int(v) for v in os.environ.get("GLM53_SPARSE1M_USERS", "1,4,8").split(",")]
REPLAYS = 20


def _traced_step_ms(device, fn, replays: int = REPLAYS) -> float:
    """Capture ``fn`` (a decode step on persistent inputs) as a trace and time its replays (ms per step)."""
    fn()  # compile
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(replays):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    dt = (time.perf_counter() - t0) / replays * 1e3
    ttnn.release_trace(device, tid)
    ttnn.deallocate(out)
    return dt


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200_000_000, "l1_small_size": 32768}], indirect=True)
def test_sparse_1m(mesh_device, monkeypatch):
    assert CURVE == sorted(CURVE) and all(n % 128 == 0 for n in CURVE)
    n_max, B_max = CURVE[-1], max(USERS)
    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot), name_filter=_attn_keys)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    P, K = cfg.index_kpool, cfg.index_select_pools
    hf = _hf_attention(text, {k: v for k, v in raw.items() if k.startswith("self_attn.")})
    mu, sigma = _x_stats(raw, cfg)
    rows = _decode_rows(raw, cfg, 3)

    log_dram(mesh_device, "before build")
    t0 = time.perf_counter()
    attn = TtDSA(
        mesh_device,
        cfg,
        {k[len("self_attn.") :]: v for k, v in raw.items() if k.startswith("self_attn.")},
        layer_idx=LAYER,
    )
    # one cache per batch size (the paged cache ops take the decode batch == the cache's user slots): the
    # 1-user cache is seeded and checked, the others time the step on unseeded (zero) caches
    caches = {B: TtLatentCache(mesh_device, cfg, max_batch=B, max_seq_len=n_max + 64) for B in sorted(set(USERS) | {1})}
    cache = caches[1]
    log_dram(
        mesh_device,
        f"attention + caches for {sorted(caches)} users x {n_max + 64} positions built ({time.perf_counter() - t0:.0f} s)",
    )
    host = HostContext(hf, text, LAYER, n_max + 8)

    table = []
    filled, chunk_index = 0, 0
    for N in CURVE:
        while filled < N:
            T = min(CHUNK, N - filled)
            x = synthetic_rows(mu, sigma, chunk_index, T)
            c_h, k_h, g_h = host.extend(x)
            x_tt = ttnn.from_torch(
                x.reshape(1, 1, T, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
            )
            attn.prefill_state(x_tt, cache, 0, T, start=filled)
            ttnn.deallocate(x_tt)
            p_lat, p_pool = _check_chunk(cache, hf, cfg, filled, c_h, k_h, g_h)
            assert p_lat > 0.995 and p_pool > 0.995, (chunk_index, p_lat, p_pool)
            filled += T
            chunk_index += 1
        logger.info(f"N={N} seeded (latent / pooled-key rows vs host checked per chunk)")

        # ---- correctness, 1 user: new traced path vs HF restricted, and the eager reference path
        cache_r = host.make_cache(N)
        for s in range(2):
            a = N + s
            row = rows[s]
            x_tt = ttnn.from_torch(
                row.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
            )
            pos, pool_ids, ring_slots = decode_position_tensors(mesh_device, [a], P)
            meta = SparseDecodeMeta.from_positions(mesh_device, [a], cfg)
            y_new = attn.decode_forward(x_tt, cache, pos, pool_ids, ring_slots, sparse_meta=meta)
            y_n = ttnn.to_torch(y_new).reshape(-1).float()
            ttnn.deallocate(y_new)
            dense_pos = decode_position_tensors(mesh_device, [min(a, attn.dense_exact_max_pos)], P)[0]
            y_old = attn.decode_forward(
                x_tt, cache, pos, pool_ids, ring_slots, host_positions=[a], dense_positions=dense_pos
            )
            y_o = ttnn.to_torch(y_old).reshape(-1).float()
            ttnn.deallocate(y_old)
            pools = attn.selected_pools(x_tt, cache, 0, a)
            n, t = (a + 1) // P, (a + 1) % P
            assert all(0 <= p < n for p in pools)
            tt_set = {P * p + j for p in pools for j in range(P)} | {P * n + j for j in range(t)}
            y_r, ids_r = hf_decode_step(hf, cache_r, row, full=False)
            p_new, p_old, p_same = pcc(y_r, y_n), pcc(y_r, y_o), pcc(y_o, y_n)
            ov = _overlap(ids_r, tt_set)
            logger.info(
                f"N={N} pos {a}: new traced path vs HF-restricted pcc {p_new:.6f} (eager reference path {p_old:.6f}, "
                f"new vs eager {p_same:.6f}); ids HF {len(ids_r)} TT {len(tt_set)} overlap {ov:.4f}"
            )
            assert p_new > 0.995 and ov > 0.8
            ttnn.deallocate(x_tt)

        # ---- timing: the DSA attention decode step, traced, for B users at position N
        for B in USERS:
            cache_b = caches[B]
            x_b = ttnn.from_torch(
                torch.cat([rows[2].reshape(1, 1, 1, -1)] * B, dim=2),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
            pos, pool_ids, ring_slots = decode_position_tensors(mesh_device, [N + 2] * B, P)
            meta = SparseDecodeMeta.from_positions(mesh_device, [N + 2] * B, cfg)
            ms_new = _traced_step_ms(
                mesh_device, lambda: attn.decode_forward(x_b, cache_b, pos, pool_ids, ring_slots, sparse_meta=meta)
            )
            ms_old = float("nan")
            if B == 1:
                # the eager reference path is not traceable (host geometry): warm wall time per step
                dense_pos = decode_position_tensors(mesh_device, [min(N + 2, attn.dense_exact_max_pos)], P)[0]
                for i in range(4):
                    if i == 1:
                        ttnn.synchronize_device(mesh_device)
                        t0 = time.perf_counter()
                    y = attn.decode_forward(
                        x_b, cache_b, pos, pool_ids, ring_slots, host_positions=[N + 2], dense_positions=dense_pos
                    )
                    ttnn.deallocate(y)
                ttnn.synchronize_device(mesh_device)
                ms_old = (time.perf_counter() - t0) / 3 * 1e3
            table.append((N, B, ms_new, ms_old))
            logger.info(
                f"N={N} B={B}: DSA attention decode step, traced {ms_new:.3f} ms/step"
                + (f"; eager reference path {ms_old:.1f} ms" if B == 1 else "")
            )
            ttnn.deallocate(x_b)
        log_dram(mesh_device, f"N={N} done")

    logger.info("DSA attention decode step (layer 3, one chip, traced new path; eager reference path for 1 user):")
    logger.info("| positions | users | traced ms/step | eager reference ms/step |")
    for N, B, ms_new, ms_old in table:
        logger.info(f"| {N} | {B} | {ms_new:.3f} | {'-' if ms_old != ms_old else f'{ms_old:.1f}'} |")
