# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Program-config + memory-layout sweep for a SINGLE matmul: M=32, K=3072, N=6144.

Target: P150 / **Blackhole**. Tunes the text_prefill hot matmul flagged in the
device-perf report:

    MatmulDeviceOperation  32 x 3072 x 6144   HiFi2  BF16 x BFP8 => BF16   (~16.6 TFLOPs, ~50% util)

Fixes the shape / dtype-chain / fidelity and sweeps program-config knobs AND memory
layouts, to read off the fastest variant that still passes PCC.

M=32 => Mt=1 (single M-tile): no real 2D block-shard (gy|Mt=1 => gy=1), and the
DRAM-bank-sharded weights family ("DS", decode-weights path) becomes legal.

Blackhole-specific (queried at RUNTIME, not hardcoded like the Wormhole sample):
  - DRAM bank count via ``device.dram_grid_size()``.
  - compute grid via ``device.compute_with_storage_grid_size()`` (Blackhole > 8x8), so
    grids are generated to fit the actual device.

Measurement — host wall-clock, NOT the device profiler.
  On this build ``ttnn.ReadDeviceProfiler`` only flushes ``profile_log_device.csv`` at
  device close, so the Wormhole sample's per-op ``rm`` + read does not work here. Instead
  each config is warmed up (program-cache miss + JIT), then ``_TIMED_ITERS`` matmuls are
  enqueued back-to-back with a SINGLE ``synchronize_device`` at the end. With a deep queue
  the host dispatch overlaps device execution, so elapsed/iters approximates device time
  — accurate enough to RANK configs (it's slightly inflated by fixed overhead vs the pure
  kernel duration in the perf report, but the relative ordering/speedup is what matters).
  Runs with a plain ``pytest`` — no profiler build / Tracy required.

Fixed (NOT swept):
  - shape       : M=32, K=3072, N=6144  -> Mt=1, Kt=96, Nt=192.
  - dtype-chain : in0=bf16 (activations), in1=bfp8_b (weights), out=bf16.
  - fidelity    : HiFi2, fp32_dest_acc_en=True (=> out_subblock h*w<=4), packer_l1_acc=True.

Output: one aligned line per config (device us, achieved TFLOPs, PCC, PASS/FAIL/ERROR),
then a summary naming the fastest PCC-passing variant and its speedup vs the baseline.
The test FAILS if no config both passes PCC and beats the baseline.
"""

import math
from dataclasses import dataclass
from time import perf_counter

import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc

# --- fixed shape -----------------------------------------------------------
M = 32
K = 3072
N = 6144
TILE = 32
MT = M // TILE  # 1
KT = K // TILE  # 96
NT = N // TILE  # 192

# --- fixed dtype-chain + fidelity ------------------------------------------
IN0_DTYPE = ttnn.bfloat16
IN1_DTYPE = ttnn.bfloat8_b
OUT_DTYPE = ttnn.bfloat16
MATH_FIDELITY = ttnn.MathFidelity.HiFi2

# --- measurement -----------------------------------------------------------
_WARMUP_ITERS = 5
_TIMED_ITERS = 100

# PCC pass target keyed by weight (in1) dtype.  bfp8_b weights at K=3072 (bf16 out)
# comfortably clear 0.99.
_PCC_TARGETS = {
    ttnn.bfloat16: 0.999,
    ttnn.bfloat8_b: 0.99,
    ttnn.bfloat4_b: 0.95,
}
_PCC_TARGET = _PCC_TARGETS[IN1_DTYPE]

# Must-beat device time [us].  None -> use the measured time of the baseline row.
_BASELINE_US = None

# fp32 dest reg holds <=4 tiles when fp32_dest_acc_en=True, so out_subblock h*w must be <=4.
_SUBBLOCK_CHOICES = [(2, 2), (4, 1), (1, 4), (3, 1), (1, 3), (2, 1), (1, 2), (1, 1)]


def _subblock(per_core_m, per_core_n, out_sharded):
    """Largest valid (out_subblock_h, out_subblock_w) with h*w<=4 (fp32 dest)."""
    if out_sharded:
        for w in (4, 3, 2, 1):
            if per_core_n % w == 0:
                return 1, w
        return 1, 1
    for h, w in _SUBBLOCK_CHOICES:
        if per_core_m % h == 0 and per_core_n % w == 0:
            return h, w
    return 1, 1


@dataclass
class MatmulCase:
    family: str  # "1D" | "DS"
    grid_x: int
    grid_y: int
    in0_block_w: int
    in0: str  # "l1" | "dram" | "ws"
    in1: str  # "dram" (1D) | "ds" (DS — bank-sharded weights)
    out: str  # "l1" | "dram" | "ws"
    is_baseline: bool = False

    def __post_init__(self):
        assert KT % self.in0_block_w == 0, f"{self.label}: in0_block_w must divide Kt={KT}"
        assert (
            self.kt_per_core % self.in0_block_w == 0
        ), f"{self.label}: in0_block_w must divide Kt/core={self.kt_per_core}"
        if self.family == "1D":
            assert self.in1 == "dram", f"{self.label}: 1D weights must be DRAM interleaved"
            assert NT % self.num_cores == 0, f"{self.label}: {self.num_cores} cores must divide Nt={NT}"
            if self.in0 == "ws":
                assert KT % self.num_cores == 0, f"{self.label}: width-sharded in0 needs cores | Kt={KT}"
            assert self.in0 in ("l1", "dram", "ws"), f"{self.label}: 1D in0 must be l1/dram/ws"
            assert self.out in ("l1", "dram", "ws"), f"{self.label}: 1D out must be l1/dram/ws"
        elif self.family == "DS":
            # DS (DRAM-bank-sharded weights) derives its compute grid from the in0 width-shard,
            # which may be 2D — production QKV uses dram_shard_core_grid_for_k (a 2D CoreGrid).
            assert (self.in0, self.in1, self.out) == ("ws", "ds", "ws"), f"{self.label}: DS must be ws/ds/ws"
            assert NT % self.num_cores == 0, f"{self.label}: {self.num_cores} cores must divide Nt={NT}"
            assert KT % self.num_cores == 0, f"{self.label}: {self.num_cores} cores must divide Kt={KT}"
        else:
            raise AssertionError(f"{self.label}: unknown family {self.family}")

    @property
    def num_cores(self):
        return self.grid_x * self.grid_y

    @property
    def per_core_M(self):
        return MT

    @property
    def per_core_N(self):
        return NT // self.num_cores

    @property
    def out_sharded(self):
        return self.out == "ws"

    @property
    def kt_per_core(self):
        if self.in0 == "ws":  # width-sharded in0 splits K across all cores (1D & DS)
            return KT // self.num_cores
        return KT

    @property
    def subblock(self):
        return _subblock(self.per_core_M, self.per_core_N, self.out_sharded)

    @property
    def layout(self):
        return f"{self.in0}/{self.in1}/{self.out}"

    @property
    def label(self):
        return f"{self.family} {self.layout} {self.grid_x}x{self.grid_y} w{self.in0_block_w}"


def _grid_for_cores(cores, max_gx, max_gy):
    """A (gx, gy) with gx*gy == cores that fits the device grid; prefer wide rows."""
    for gx in range(min(cores, max_gx), 0, -1):
        if cores % gx == 0 and cores // gx <= max_gy:
            return gx, cores // gx
    return None


def _divisors(n, lim):
    return [d for d in range(1, lim + 1) if n % d == 0]


def _valid_w(kt_per_core, candidates=(8, 4, 2)):
    ws = [w for w in candidates if kt_per_core % w == 0]
    return ws or [1]


def _make_cases(max_gx, max_gy):
    """Build the sweep for the device's actual grid (Kt=96, Nt=192)."""
    cases = []
    max_cores = max_gx * max_gy
    cores_1d = [c for c in _divisors(NT, max_cores) if c >= 4 and _grid_for_cores(c, max_gx, max_gy)]
    g = math.gcd(KT, NT)  # 96
    cores_wsK = [c for c in _divisors(g, max_cores) if c >= 4 and _grid_for_cores(c, max_gx, max_gy)]

    baseline_cores = 32 if 32 in cores_1d else cores_1d[len(cores_1d) // 2]

    # 1D, in0 L1, weights DRAM (baseline family)
    for c in cores_1d:
        gx, gy = _grid_for_cores(c, max_gx, max_gy)
        for w in (2, 4, 8):
            cases.append(MatmulCase("1D", gx, gy, w, "l1", "dram", "l1", is_baseline=(c == baseline_cores and w == 8)))
    # 1D, in0 streamed from DRAM
    for c in cores_1d:
        gx, gy = _grid_for_cores(c, max_gx, max_gy)
        cases.append(MatmulCase("1D", gx, gy, 8, "dram", "dram", "l1"))
    # 1D, output to DRAM
    for c in cores_1d:
        gx, gy = _grid_for_cores(c, max_gx, max_gy)
        cases.append(MatmulCase("1D", gx, gy, 8, "l1", "dram", "dram"))
    # 1D, output L1 width-sharded
    for c in cores_1d:
        gx, gy = _grid_for_cores(c, max_gx, max_gy)
        for w in (2, 4, 8):
            cases.append(MatmulCase("1D", gx, gy, w, "l1", "dram", "ws"))
    # 1D, in0 + out L1 width-sharded (in0 splits K: w | Kt/cores)
    for c in cores_wsK:
        gx, gy = _grid_for_cores(c, max_gx, max_gy)
        for w in _valid_w(KT // c):
            cases.append(MatmulCase("1D", gx, gy, w, "ws", "dram", "ws"))
    # DS: DRAM-bank-sharded weights (decode path; M=32 only) — the PRODUCTION QKV family.
    # in0 width-sharded across a (possibly 2D) grid of num_cores | gcd(Kt,Nt)=96; weights
    # DRAM-bank-sharded. Tests the production family across 4..48 cores (production uses 12).
    for c in cores_wsK:
        gx, gy = _grid_for_cores(c, max_gx, max_gy)
        for w in _valid_w(KT // c):
            cases.append(MatmulCase("DS", gx, gy, w, "ws", "ds", "ws", is_baseline=False))
    return cases


@dataclass
class CaseResult:
    case: MatmulCase
    dev_us: float
    tflops: float
    pcc: float
    pcc_pass: bool
    err: str = ""


def _mem_in0(case):
    if case.in0 == "l1":
        return ttnn.L1_MEMORY_CONFIG
    if case.in0 == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.create_sharded_memory_config(
        (1, 1, M, K),
        core_grid=ttnn.CoreGrid(y=case.grid_y, x=case.grid_x),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def _n_padded(num_banks):
    return math.ceil(N / (TILE * num_banks)) * (TILE * num_banks)


def _mem_in1(case, device):
    if case.in1 == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    # "ds": DRAM width-sharded across ALL DRAM banks (queried from the device).
    dram = device.dram_grid_size()
    num_banks = dram.x * dram.y
    n_padded = _n_padded(num_banks)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(grid, [K, n_padded // num_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _mem_out(case):
    if case.out == "l1":
        return ttnn.L1_MEMORY_CONFIG
    if case.out == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.MemoryConfig(memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type=ttnn.BufferType.L1)


def _program_config(case):
    if case.family == "DS":
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=case.in0_block_w,
            per_core_M=case.per_core_M,
            per_core_N=case.per_core_N,
            fused_activation=None,
        )
    sh, sw = case.subblock
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(case.grid_x, case.grid_y),
        in0_block_w=case.in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=case.per_core_M,
        per_core_N=case.per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _time_matmul_us(device, in0, in1, program_config, out_mem_cfg, cck):
    """Host wall-clock device time per matmul: warm + queued iters + single sync."""
    for _ in range(_WARMUP_ITERS):  # absorb program-cache miss + JIT
        w = ttnn.matmul(
            in0,
            in1,
            program_config=program_config,
            memory_config=out_mem_cfg,
            compute_kernel_config=cck,
            dtype=OUT_DTYPE,
        )
        ttnn.deallocate(w)
    ttnn.synchronize_device(device)
    t0 = perf_counter()
    for _ in range(_TIMED_ITERS):
        o = ttnn.matmul(
            in0,
            in1,
            program_config=program_config,
            memory_config=out_mem_cfg,
            compute_kernel_config=cck,
            dtype=OUT_DTYPE,
        )
        ttnn.deallocate(o)  # enqueued; keeps L1/DRAM bounded over the loop
    ttnn.synchronize_device(device)
    return (perf_counter() - t0) * 1e6 / _TIMED_ITERS


def test_matmul_32x3072x6144_sweep(device):
    torch.manual_seed(0)

    grid = device.compute_with_storage_grid_size()
    cases = _make_cases(grid.x, grid.y)
    dram = device.dram_grid_size()
    logger.info(f"device compute grid = {grid.x}x{grid.y} ({grid.x * grid.y} cores), dram banks = {dram.x * dram.y}")

    torch_input_a = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    torch_input_b = torch.randn((1, 1, K, N), dtype=torch.bfloat16)
    torch_output = torch.matmul(torch_input_a, torch_input_b)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(  # generic struct, valid on Blackhole
        math_fidelity=MATH_FIDELITY,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    results = []
    for case in cases:
        in0 = in1 = chk = None
        try:
            in0 = ttnn.from_torch(
                torch_input_a, dtype=IN0_DTYPE, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_mem_in0(case)
            )
            in1 = ttnn.from_torch(
                torch_input_b,
                dtype=IN1_DTYPE,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_mem_in1(case, device),
            )
            program_config = _program_config(case)
            out_mem_cfg = _mem_out(case)

            # Correctness: one matmul -> PCC vs torch.
            chk = ttnn.matmul(
                in0,
                in1,
                program_config=program_config,
                memory_config=out_mem_cfg,
                compute_kernel_config=compute_kernel_config,
                dtype=OUT_DTYPE,
            )
            out = ttnn.to_torch(chk)
            assert out.shape == torch_output.shape, f"{out.shape} != {torch_output.shape}"
            pcc_pass, pcc = comp_pcc(torch_output, out, _PCC_TARGET)
            ttnn.deallocate(chk)
            chk = None

            # Timing: host wall-clock over a queued loop.
            dev_us = _time_matmul_us(device, in0, in1, program_config, out_mem_cfg, compute_kernel_config)
            tflops = 2 * M * K * N / 1e6 / dev_us
            results.append(CaseResult(case, dev_us, tflops, float(pcc), bool(pcc_pass)))
        except Exception as e:  # OOM / FATAL / shape mismatch — record, keep sweeping
            results.append(CaseResult(case, float("inf"), 0.0, 0.0, False, str(e).strip().splitlines()[0][:90]))
        finally:
            for t in (chk, in1, in0):
                if t is not None:
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass
            ttnn.synchronize_device(device)

    # --- pick winners ------------------------------------------------------
    baseline = next(r for r in results if r.case.is_baseline)
    assert not baseline.err, f"baseline config {baseline.case.label} failed: {baseline.err}"
    must_beat_us = _BASELINE_US if _BASELINE_US is not None else baseline.dev_us
    passing = [r for r in results if r.pcc_pass]
    winners = [r for r in passing if r.dev_us < must_beat_us]
    best = min(passing, key=lambda r: r.dev_us) if passing else None

    # --- report ------------------------------------------------------------
    logger.info(f"matmul sweep M={M} K={K} N={N} in0={IN0_DTYPE} in1={IN1_DTYPE} fidelity={MATH_FIDELITY}")
    logger.info(
        f"{'family':>6} {'in0/in1/out':>12} {'grid':>5} {'cores':>5} {'ibw':>4} {'pcM':>3} {'pcN':>4} "
        f"{'sub':>5} {'dev_us':>9} {'TFLOPs':>7} {'PCC':>8} {'result':>7}  note"
    )
    for r in results:
        c = r.case
        sh, sw = c.subblock
        if r.err:
            metrics = f"{'-':>9} {'-':>7} {'-':>8} {'ERROR':>7}"
            note = r.err
        else:
            metrics = f"{r.dev_us:>9.2f} {r.tflops:>7.2f} {r.pcc:>8.4f} {('PASS' if r.pcc_pass else 'FAIL'):>7}"
            note = "baseline" if c.is_baseline else ("best" if r is best else "")
        logger.info(
            f"{c.family:>6} {c.layout:>12} {f'{c.grid_x}x{c.grid_y}':>5} {c.num_cores:>5} {c.in0_block_w:>4} "
            f"{c.per_core_M:>3} {c.per_core_N:>4} {f'{sh}x{sw}':>5} {metrics}  {note}"
        )

    if best is not None:
        logger.info(
            f"FASTEST PCC-PASS: {best.case.label} — {best.dev_us:.2f}us, {best.tflops:.2f} TFLOPs, PCC={best.pcc:.4f} "
            f"| must-beat={must_beat_us:.2f}us | speedup={must_beat_us / best.dev_us:.2f}x"
        )
    else:
        logger.info(f"FASTEST PCC-PASS: none (no config reached PCC>={_PCC_TARGET}) | must-beat={must_beat_us:.2f}us")

    assert winners, (
        f"No config both passed PCC>={_PCC_TARGET} and beat the baseline {must_beat_us:.2f}us "
        f"(baseline row {baseline.case.label}={baseline.dev_us:.2f}us). Baseline is already optimal among the swept configs."
    )
