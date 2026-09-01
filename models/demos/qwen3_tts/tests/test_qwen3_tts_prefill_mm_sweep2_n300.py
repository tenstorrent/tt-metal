# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Can anything beat the current prefill config for gate/up and o_proj on N300?

The shared (ungated) prefill tuning picks its 1D grids with ``find_1d_mcast_grid`` and was
validated on N150. On N300 the TP=2 split halves N, and two of the four matmul families come
out slower than the auto-routing they replaced:

    N300, per layer          before          now
      mlp gate  seq=64        68 us          87 us
      mlp up    seq=64        67 us          87 us
      o_proj    seq=128       40 us          48 us

On N150 the same code is right — gate/up there is 2048x6144 and runs at 210 GB/s (73 % of
DRAM peak) on the full grid. It is the N300 shape (2048x3072) that lands badly at 145 GB/s.

This sweep asks whether a different config wins on N300, measuring **the same experiment the
model runs**, which the first sweep did not:

  * gate/up receives the width-sharded RMSNorm output on the full grid — `decoder_layer`
    hands `self.mlp()` the sharded tensor directly in prefill. Any candidate that wants a
    different in0 layout must pay the reshard, so the reshard is inside the timed region.
  * o_proj receives an L1-interleaved tensor from `nlp_concat_heads`.

The reference is the config the model builds today, not auto-routing.

    pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_prefill_mm_sweep2_n300.py
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch

import ttnn
from models.demos.qwen3_tts.tt.linear_1d_program_config import find_1d_mcast_grid, make_linear_1d_program_config

TILE = 32
REPS = int(os.environ.get("QWEN3_TTS_SWEEP_REPS", "16"))
REPLAYS = int(os.environ.get("QWEN3_TTS_SWEEP_REPLAYS", "20"))
ROUNDS = int(os.environ.get("QWEN3_TTS_SWEEP_ROUNDS", "3"))
_TRACE_REGION = 100_000_000
PCC_FLOOR = 0.999
WIN_PCT = 5.0

HIDDEN = 2048  # 1.7B talker
LOCAL_INTERMEDIATE = 3072  # 6144 // tp=2
LOCAL_HIDDEN = 1024  # 8 local heads x 128


@dataclass(frozen=True)
class Case:
    name: str
    m: int
    k: int
    n: int
    in0_wshard_cores: Optional[int]  # in0 layout the model actually delivers; None = L1 interleaved
    note: str


CASES = [
    # gate/up: in0 is the width-sharded LN output on the full 64-core grid.
    Case("s64_gate_up", 64, HIDDEN, LOCAL_INTERMEDIATE, 64, "mlp gate / up (x2 per layer)"),
    Case("s128_gate_up", 128, HIDDEN, LOCAL_INTERMEDIATE, 64, "mlp gate / up (x2 per layer)"),
    # o_proj: in0 is L1 interleaved out of nlp_concat_heads.
    Case("s64_wo", 64, LOCAL_HIDDEN, HIDDEN, None, "attention o_proj"),
    Case("s128_wo", 128, LOCAL_HIDDEN, HIDDEN, None, "attention o_proj"),
]


@pytest.fixture(scope="module")
def device():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2), l1_small_size=32768, trace_region_size=_TRACE_REGION)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _kcfg(fp32: bool):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )


def _wshard(m: int, k: int, cores: int, gx: int) -> Optional[ttnn.MemoryConfig]:
    if k // TILE % cores:
        return None
    rows, cols = math.ceil(cores / gx), min(cores, gx)
    if rows * cols != cores:
        return None
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (m, k // cores), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _subblocks(pm: int, pn: int, cap: int):
    out = [
        (h, w)
        for h in range(1, min(cap, pm) + 1)
        if pm % h == 0
        for w in range(1, min(cap // h, pn) + 1)
        if pn % w == 0
    ]
    out.sort(key=lambda hw: -(hw[0] * hw[1]))
    return out[:4]


@dataclass
class Cand:
    label: str
    program_config: object
    target_memcfg: object  # layout the matmul wants for in0; reshard cost is timed
    fp32: bool


def build(case: Case, gx: int, gy: int) -> list[Cand]:
    m_t, k_t, n_t = case.m // TILE, case.k // TILE, case.n // TILE
    L1 = ttnn.L1_MEMORY_CONFIG
    native = _wshard(case.m, case.k, case.in0_wshard_cores, gx) if case.in0_wshard_cores else L1
    out: list[Cand] = []

    # --- reference: exactly what the model builds today ---
    if case.n == LOCAL_INTERMEDIATE:  # gate/up -> full grid, per upstream
        ref_pc = make_linear_1d_program_config(case.m, case.k, case.n, gx, gy, True)
        ref_label = f"CURRENT 1D full-grid {gx}x{gy}"
    else:  # o_proj -> find_1d_mcast_grid
        rgx, rgy = find_1d_mcast_grid(case.k, case.n, gx, gy)
        ref_pc = make_linear_1d_program_config(case.m, case.k, case.n, rgx, rgy, True)
        ref_label = f"CURRENT 1D find_grid {rgx}x{rgy}"
    out.append(Cand(ref_label, ref_pc, native, True))

    # --- auto-routing, in the model's native layout and resharded to L1 ---
    out.append(Cand("auto / native in0", None, native, True))
    if native is not L1:
        out.append(Cand("auto / S2I to L1 (reshard timed)", None, L1, True))

    for fp32 in (True, False):
        cap, tag = (4, "fp32acc") if fp32 else (8, "fp16acc")
        # --- 1D mcast over a range of core counts, in0 resharded to match ---
        for cores in sorted({gx * gy, 48, 32, 24, 16, 12, 8}, reverse=True):
            if cores > gx * gy or n_t % cores:
                continue
            pn, pk = n_t // cores, math.ceil(k_t / cores)
            mc = _wshard(case.m, case.k, cores, gx) or L1
            for h, w in _subblocks(m_t, pn, cap):
                out.append(
                    Cand(
                        f"1D c{cores} ibw{pk} sb{h}x{w}/{tag}" + ("" if mc is not L1 else " (L1 in0)"),
                        ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(gx, gy),
                            in0_block_w=pk,
                            out_subblock_h=h,
                            out_subblock_w=w,
                            per_core_M=m_t,
                            per_core_N=pn,
                            fuse_batch=True,
                            fused_activation=None,
                            mcast_in0=True,
                        ),
                        mc,
                        fp32,
                    )
                )
        # --- 2D mcast block-sharded: gy must divide M tiles (only 2 or 4 here) ---
        for g_y in [g for g in (1, 2, 4) if g <= min(gy, m_t) and m_t % g == 0]:
            pm = m_t // g_y
            for g_x in (gx, gx // 2):
                if g_x < 1 or k_t % g_x or n_t % g_x:
                    continue
                pn = n_t // g_x
                bs = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g_x - 1, g_y - 1))}),
                        (case.m // g_y, case.k // g_x),
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
                for ibw in [d for d in (1, 2, 4, 8) if (k_t // g_x) % d == 0]:
                    for h, w in _subblocks(pm, pn, cap)[:2]:
                        out.append(
                            Cand(
                                f"2D block {g_x}x{g_y} ibw{ibw} sb{h}x{w}/{tag}",
                                ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                                    compute_with_storage_grid_size=(g_x, g_y),
                                    in0_block_w=ibw,
                                    out_subblock_h=h,
                                    out_subblock_w=w,
                                    per_core_M=pm,
                                    per_core_N=pn,
                                    transpose_mcast=False,
                                    fused_activation=None,
                                ),
                                bs,
                                fp32,
                            )
                        )
    return out


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    ac, bc = a - a.mean(), b - b.mean()
    d = ac.norm() * bc.norm()
    return 0.0 if d < 1e-12 else (ac * bc).sum().item() / d.item()


def _time_us(device, fn: Callable) -> float:
    warm = fn()
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        cap = [fn() for _ in range(REPS)]
    finally:
        ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    s = []
    try:
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            s.append((time.perf_counter() - t0) / (REPLAYS * REPS) * 1e6)
    finally:
        ttnn.release_trace(device, tid)
        for o in cap:
            ttnn.deallocate(o)
    s.sort()
    return s[len(s) // 2]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_beat_current_prefill_config(device, case: Case):
    from models.demos.qwen3_tts.tt.mesh_utils import is_n300, to_torch

    if not is_n300(device):
        pytest.skip("N300-only")
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y

    torch.manual_seed(0)
    at = torch.randn(1, 1, case.m, case.k, dtype=torch.bfloat16)
    wt = torch.randn(1, 1, case.k, case.n, dtype=torch.bfloat16)
    ref = (at.float() @ wt.float()).squeeze(0).squeeze(0)

    native = _wshard(case.m, case.k, case.in0_wshard_cores, gx) if case.in0_wshard_cores else ttnn.L1_MEMORY_CONFIG
    w = ttnn.from_torch(
        wt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    act = ttnn.from_torch(at, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=native)

    rows, current_us, skipped = [], None, 0
    for c in build(case, gx, gy):
        kw = dict(memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_kcfg(c.fp32), dtype=ttnn.bfloat16)
        if c.program_config is not None:
            kw["program_config"] = c.program_config
        needs_reshard = c.target_memcfg != native

        def fn(c=c, kw=kw, needs_reshard=needs_reshard):
            # The reshard is part of the candidate's cost: the model would have to pay it.
            a = ttnn.to_memory_config(act, c.target_memcfg) if needs_reshard else act
            o = ttnn.linear(a, w, **kw)
            if needs_reshard:
                ttnn.deallocate(a)
            return o

        try:
            o = fn()
            ttnn.synchronize_device(device)
            p = _pcc(ref, to_torch(o, device=device).float())
            ttnn.deallocate(o)
            us = _time_us(device, fn)
        except Exception:
            skipped += 1
            continue
        rows.append((us, p, c.label, needs_reshard))
        if c.label.startswith("CURRENT"):
            current_us = us

    assert current_us is not None, "the current config failed to run"
    ok = sorted(r for r in rows if r[1] >= PCC_FLOOR)

    print(
        f"\n### {case.name} — {case.note}   M={case.m} K={case.k} N={case.n}"
        f"   in0={'wshard/' + str(case.in0_wshard_cores) + 'c' if case.in0_wshard_cores else 'L1 interleaved'}"
    )
    print(
        f"    {len(rows)} of {len(rows)+skipped} candidates ran; reference = current model config "
        f"({current_us:.1f} us)"
    )
    print(f"    {'us':>7} {'vs current':>11}  {'PCC':>8}  reshard  config")
    for us, p, lbl, rs in ok[:10]:
        print(f"    {us:7.1f} {100*(us-current_us)/current_us:+10.1f}%  {p:.5f}  {'yes' if rs else ' - ':^7}  {lbl}")
    best_us, best_p, best_lbl, _ = ok[0]
    win = 100 * (current_us - best_us) / current_us
    verdict = f"BEATS current by {win:.1f} %" if win > WIN_PCT else "nothing beats the current config"
    print(f"    -> {verdict}   (best: {best_lbl})")
