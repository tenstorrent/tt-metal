# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Memory-layout + program-config sweep for the TextEncoder CNN LayerNorm.

The CNN stack runs ``Conv1d -> channel-LayerNorm -> LeakyReLU`` three times. In the Tracy report
the LayerNorm is ~10.6% of device time (12 us x 3) yet runs on only 3 cores because it executes on
a DRAM-interleaved ``[1, 1, B*T, C]`` activation (normalize over the last dim ``C``). With B=2, T=48
the shape is ``[1, 1, 96, 512]`` => M=96 (3 tiles), N=512 (16 tiles) — plenty of width to spread the
reduction across many cores via a block/width-sharded LayerNorm.

This sweep isolates that one op at the production shape and compares:
  - interleaved DRAM (current production baseline) and interleaved L1 (default program config), vs
  - ``LayerNormShardedMultiCoreProgramConfig`` over every block-sharded grid that tiles [3, 16].

For each case it measures true device-kernel time (profiler read-back, 1 warmup + 1 timed) and PCC
vs a torch reference, and prints the fastest PCC-passing config.

NOTE — read before wiring the winner into ``tt_text_encoder.py``:
  The conv that feeds the LayerNorm emits a *sharded* output today (perf log: Conv2d -> Sharded-
  ToInterleaved -> LayerNorm). A sharded LayerNorm can consume that directly and drop the
  ShardedToInterleaved, but its output is sharded too, so the following LeakyReLU/next-conv must
  accept (or re-interleave) it. Account for any added InterleavedToSharded/ShardedToInterleaved
  (~1 us each, host-dispatched) when comparing against the DRAM baseline on this dispatch-bound model.

REQUIRED ENV (test skips otherwise):
  - ``TT_METAL_DEVICE_PROFILER=1``
  - ``TT_METAL_PROFILER_MID_RUN_DUMP=1``

Run:
    export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
    pytest -s models/experimental/kokoro/perf/test_layernorm_text_encoder_perf_sweep.py -v
    # override the shape:  KOKORO_LN_SHAPE=96x512
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc

from .test_matmul_decoder_perf_sweep import _device_kernel_us, _drain_profiler

TILE = 32
_PCC_TARGET = 0.99
_LN_EPS = 1e-5


@dataclass(frozen=True)
class LNCase:
    label: str
    in_mem: str  # "dram" | "l1" | "block"
    grid_x: int = 0
    grid_y: int = 0
    block_h: int = 0
    block_w: int = 0
    subblock_w: int = 0

    @property
    def num_cores(self) -> int:
        return self.grid_x * self.grid_y


def _largest_subblock_w(block_w: int, cap: int = 4) -> int:
    """Largest divisor of ``block_w`` that is <= ``cap`` (fp32_dest_acc_en holds <=4 tiles)."""
    for d in range(min(block_w, cap), 0, -1):
        if block_w % d == 0:
            return d
    return 1


def _make_cases(m_tiles: int, n_tiles: int, gx_max: int, gy_max: int) -> list[LNCase]:
    cases: list[LNCase] = [
        LNCase("interleaved-dram", "dram"),
        LNCase("interleaved-l1", "l1"),
    ]
    # Block-sharded: gy splits the M (row) tiles, gx splits the N (col / reduction) tiles. Every
    # combination that divides [m_tiles, n_tiles] evenly and fits the live grid.
    for gy in range(1, m_tiles + 1):
        if m_tiles % gy or gy > gy_max:
            continue
        for gx in range(1, n_tiles + 1):
            if n_tiles % gx or gx > gx_max:
                continue
            if gx * gy <= 1:
                continue  # 1 core == no parallelism
            block_h = m_tiles // gy
            block_w = n_tiles // gx
            sub_w = _largest_subblock_w(block_w)
            cases.append(
                LNCase(
                    label=f"block_{gx}x{gy}_bh{block_h}_bw{block_w}_sw{sub_w}",
                    in_mem="block",
                    grid_x=gx,
                    grid_y=gy,
                    block_h=block_h,
                    block_w=block_w,
                    subblock_w=sub_w,
                )
            )
    return cases


def _block_sharded_mem_config(case: LNCase, m: int, n: int) -> ttnn.MemoryConfig:
    return ttnn.create_sharded_memory_config(
        shape=[1, 1, m, n],
        core_grid=ttnn.CoreGrid(y=case.grid_y, x=case.grid_x),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def _program_config(case: LNCase):
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(case.grid_x, case.grid_y),
        subblock_w=case.subblock_w,
        block_h=case.block_h,
        block_w=case.block_w,
        inplace=False,
    )


@dataclass
class LNResult:
    case: LNCase
    dev_us: float
    pcc: float
    pcc_pass: bool
    err: Optional[str] = None


def _shape() -> tuple[int, int]:
    s = os.getenv("KOKORO_LN_SHAPE", "96x512")
    m, n = s.lower().split("x")
    return int(m), int(n)


def test_text_encoder_layernorm_perf_sweep(device):
    if os.getenv("TT_METAL_DEVICE_PROFILER") is None:
        pytest.skip("device-time sweep needs a profiler build + TT_METAL_DEVICE_PROFILER=1")
    if os.getenv("TT_METAL_PROFILER_MID_RUN_DUMP") is None:
        pytest.skip("set TT_METAL_PROFILER_MID_RUN_DUMP=1 so ReadDeviceProfiler flushes mid-run")

    M, N = _shape()
    assert M % TILE == 0 and N % TILE == 0, f"shape {M}x{N} must be tile-aligned"
    m_tiles, n_tiles = M // TILE, N // TILE

    grid = device.compute_with_storage_grid_size()
    gx_max, gy_max = grid.x, grid.y
    cases = _make_cases(m_tiles, n_tiles, gx_max, gy_max)

    torch.manual_seed(0)
    x = torch.randn((1, 1, M, N), dtype=torch.bfloat16)
    gamma = torch.randn((N,), dtype=torch.bfloat16)
    beta = torch.randn((N,), dtype=torch.bfloat16)
    ref = torch.nn.functional.layer_norm(x.float(), (N,), gamma.float(), beta.float(), eps=_LN_EPS)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    # gamma/beta as the TT LayerNorm wants them (row-major, tile-friendly 1xN handled internally).
    w_tt = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    results: list[LNResult] = []
    for case in cases:
        in_t = warm = timed = None
        try:
            if case.in_mem == "dram":
                in_mem = ttnn.DRAM_MEMORY_CONFIG
                prog = None
                out_mem = ttnn.DRAM_MEMORY_CONFIG
            elif case.in_mem == "l1":
                in_mem = ttnn.L1_MEMORY_CONFIG
                prog = None
                out_mem = ttnn.L1_MEMORY_CONFIG
            else:
                in_mem = _block_sharded_mem_config(case, M, N)
                prog = _program_config(case)
                out_mem = in_mem

            in_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem)

            def _run():
                return ttnn.layer_norm(
                    in_t,
                    weight=w_tt,
                    bias=b_tt,
                    epsilon=_LN_EPS,
                    compute_kernel_config=compute_kernel_config,
                    memory_config=out_mem,
                    program_config=prog,
                )

            warm = _run()
            ttnn.synchronize_device(device)
            ttnn.deallocate(warm)
            warm = None
            _drain_profiler(device)
            ttnn.synchronize_device(device)

            timed = _run()
            ttnn.synchronize_device(device)
            dev_us = _device_kernel_us(device)
            if dev_us is None:
                raise RuntimeError("profiler produced no device data — is this a profiler build?")

            out = ttnn.to_torch(timed).float()
            pcc_pass, pcc = comp_pcc(ref, out, _PCC_TARGET)
            results.append(LNResult(case, dev_us, float(pcc), bool(pcc_pass)))
        except Exception as e:
            results.append(LNResult(case, float("inf"), 0.0, False, str(e).strip().splitlines()[0][:90]))
        finally:
            for t in (timed, warm, in_t):
                if t is not None:
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass
            ttnn.synchronize_device(device)

    passing = [r for r in results if r.pcc_pass]
    best = min(passing, key=lambda r: r.dev_us) if passing else None
    baseline = next((r for r in results if r.case.label == "interleaved-dram"), None)

    logger.info(
        f"=== TextEncoder LayerNorm sweep  shape={M}x{N} ({m_tiles}x{n_tiles} tiles)  grid={gx_max}x{gy_max} ==="
    )
    logger.info(f"{'config':>26} {'cores':>5} {'dev_us':>9} {'PCC':>8} {'result':>7}  note")
    for r in sorted(results, key=lambda r: r.dev_us):
        c = r.case
        if r.err:
            logger.info(f"{c.label:>26} {c.num_cores:>5} {'-':>9} {'-':>8} {'ERROR':>7}  {r.err}")
            continue
        tag = "best" if r is best else ("baseline" if r is baseline else "")
        logger.info(
            f"{c.label:>26} {c.num_cores:>5} {r.dev_us:>9.2f} {r.pcc:>8.4f} "
            f"{('PASS' if r.pcc_pass else 'FAIL'):>7}  {tag}"
        )

    if best is not None and baseline is not None and baseline.dev_us != float("inf"):
        logger.info(
            f"FASTEST PCC-PASS: {best.case.label} -> {best.dev_us:.2f}us (PCC={best.pcc:.4f}) | "
            f"baseline interleaved-dram={baseline.dev_us:.2f}us | speedup={baseline.dev_us / best.dev_us:.2f}x"
        )

    ttnn.deallocate(w_tt)
    ttnn.deallocate(b_tt)
    assert passing, f"No LayerNorm config reached PCC>={_PCC_TARGET} — sweep harness broken."
