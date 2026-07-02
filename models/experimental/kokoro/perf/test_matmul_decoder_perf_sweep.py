# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Program-config + memory-layout sweep for SLOW decoder matmuls (``decoder.txt``).

Reusable harness modeled on ``seamless_m4t_v2_large/tests/perf/test_matmul_perf_report_sweep.py``.
Shapes and per-shape dtype/fidelity are lifted from the Kokoro decoder Tracy report
(``decoder.txt`` at repo root): all are small-M (32–224) HiFi4 matmuls on Blackhole.

Fixed per shape (NOT swept) — taken from the report's dtype / Math Fidelity columns:
  - FP32 style-linear / iSTFT rows: in0=fp32, in1=fp32, out=fp32, HiFi4.
  - Generator conv-stack rows: in0=fp32, in1=fp32, out=bf16, HiFi4.
  - fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False (decoder default).

Swept axes (enumerated from shape + live device grid):
  - family: 1D_in0 | 1D_in1 | 2D | dram
  - compute grid (gx, gy), in0_block_w, in0/out memory layout (l1/dram/ws/hs/bs)

Must-beat: the report's average device time per call (4th field in each shape entry),
overridable with ``MATMUL_BASELINE_US``.  A shape FAILS if no config both passes PCC and
beats that baseline.

Measurement: true device kernel time via ``ttnn.ReadDeviceProfiler`` +
``get_device_data_generate_report`` (one warmup + one timed run per config).

REQUIRED ENV (test skips otherwise):
  - ``TT_METAL_DEVICE_PROFILER=1``
  - ``TT_METAL_PROFILER_MID_RUN_DUMP=1``

Run one shape:
    export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
    KOKORO_MATMUL_SHAPE=32x128x128 pytest -s \\
      models/experimental/kokoro/tests/perf/test_matmul_decoder_perf_sweep.py -v

Run all decoder shapes:
    export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
    pytest -s models/experimental/kokoro/tests/perf/test_matmul_decoder_perf_sweep.py -v
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from tracy.common import PROFILER_LOGS_DIR
from tracy.process_ops_logs import get_device_data_generate_report

TILE = 32

_PCC_TARGETS = {
    ttnn.bfloat16: 0.99,
    ttnn.float32: 0.999,
}

_MAX_GRIDS_PER_FAMILY = 3
_IN0_BLOCK_W_CHOICES = (1, 2, 4, 8)

# fp32 dest reg holds <=4 tiles when fp32_dest_acc_en=True.
_SUBBLOCK_CHOICES = [(2, 2), (4, 1), (1, 4), (3, 1), (1, 3), (2, 1), (1, 2), (1, 1)]


@dataclass(frozen=True)
class ReportShape:
    """One SLOW ``MatmulDeviceOperation M x K x N`` row from ``decoder.txt``."""

    M: int
    K: int
    N: int
    report_us: float | None
    in0_dtype: ttnn.DataType
    in1_dtype: ttnn.DataType
    out_dtype: ttnn.DataType
    math_fidelity: ttnn.MathFidelity
    tag: str = ""

    @property
    def sweep_id(self) -> str:
        return f"{self.M}x{self.K}x{self.N}" + (f"_{self.tag}" if self.tag else "")


# (M, K, N, report_device_us) from decoder.txt — report_us is the *average* device
# time per call (total device time / op count).  Shapes are ordered by total device
# contribution in the full decoder forward.
_REPORT_SHAPES: list[ReportShape] = [
    ReportShape(32, 6624, 3008, 288.0, ttnn.float32, ttnn.float32, ttnn.float32, ttnn.MathFidelity.HiFi4, "istft"),
    ReportShape(32, 128, 128, 6.0, ttnn.float32, ttnn.float32, ttnn.float32, ttnn.MathFidelity.HiFi4, "adain_linear"),
    ReportShape(32, 1120, 1024, 34.0, ttnn.float32, ttnn.float32, ttnn.bfloat16, ttnn.MathFidelity.HiFi4, "gen_noise"),
    ReportShape(96, 1120, 1024, 45.0, ttnn.float32, ttnn.float32, ttnn.bfloat16, ttnn.MathFidelity.HiFi4, "gen_resblk"),
    ReportShape(
        96, 544, 1024, 25.0, ttnn.float32, ttnn.float32, ttnn.bfloat16, ttnn.MathFidelity.HiFi4, "gen_upsample"
    ),
    ReportShape(32, 544, 1024, 18.0, ttnn.float32, ttnn.float32, ttnn.bfloat16, ttnn.MathFidelity.HiFi4, "decode_cond"),
    ReportShape(
        224, 1120, 512, 69.0, ttnn.float32, ttnn.float32, ttnn.bfloat16, ttnn.MathFidelity.HiFi4, "gen_resblk_wide"
    ),
    ReportShape(32, 1120, 512, 33.0, ttnn.float32, ttnn.float32, ttnn.bfloat16, ttnn.MathFidelity.HiFi4, "gen_mid"),
    ReportShape(32, 512, 64, 8.0, ttnn.float32, ttnn.float32, ttnn.float32, ttnn.MathFidelity.HiFi4, "gen_post"),
]


def _filtered_shapes() -> list[ReportShape]:
    only = os.getenv("KOKORO_MATMUL_SHAPE")
    if not only:
        return _REPORT_SHAPES
    return [s for s in _REPORT_SHAPES if only in s.sweep_id or only == f"{s.M}x{s.K}x{s.N}"]


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _largest_div_leq(n: int, cap: int) -> int:
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def _grids_for_cores(cores: int, gx_max: int, gy_max: int) -> list[tuple[int, int]]:
    out = []
    for gx in range(1, gx_max + 1):
        if cores % gx:
            continue
        gy = cores // gx
        if gy <= gy_max:
            out.append((gx, gy))
    return out


def _subblock(per_core_m: int, per_core_n: int, out_sharded: bool) -> tuple[int, int]:
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
    M: int
    K: int
    N: int
    family: str
    grid_x: int
    grid_y: int
    in0_block_w: int
    in0: str
    out: str

    @property
    def Mt(self) -> int:
        return self.M // TILE

    @property
    def Kt(self) -> int:
        return self.K // TILE

    @property
    def Nt(self) -> int:
        return self.N // TILE

    @property
    def num_cores(self) -> int:
        return self.grid_x * self.grid_y

    @property
    def mcast_in0(self) -> bool:
        return self.family == "1D_in0"

    @property
    def per_core_M(self) -> int:
        if self.family in ("1D_in0", "dram"):
            return self.Mt
        if self.family == "1D_in1":
            return self.Mt // self.num_cores
        return self.Mt // self.grid_y

    @property
    def per_core_N(self) -> int:
        if self.family == "1D_in0":
            return self.Nt // self.num_cores
        if self.family == "1D_in1":
            return self.Nt
        if self.family == "dram":
            return self.Nt // self.num_cores
        return self.Nt // self.grid_x

    @property
    def kt_per_core(self) -> int:
        if self.family == "dram":
            return self.Kt // self.num_cores
        if self.family == "2D" and self.in0 == "bs":
            return self.Kt // self.grid_x
        if self.family == "1D_in0" and self.in0 == "ws":
            return self.Kt // self.num_cores
        return self.Kt

    @property
    def out_sharded(self) -> bool:
        return self.out in ("ws", "hs", "bs")

    @property
    def subblock(self) -> tuple[int, int]:
        return _subblock(self.per_core_M, self.per_core_N, self.out_sharded)

    @property
    def layout(self) -> str:
        return f"{self.in0}/dram/{self.out}"

    @property
    def label(self) -> str:
        return f"{self.family} {self.layout} {self.grid_x}x{self.grid_y} w{self.in0_block_w}"

    def feasible(self, gx_max: int, gy_max: int, max_cores: int) -> bool:
        if not (1 <= self.grid_x <= gx_max and 1 <= self.grid_y <= gy_max):
            return False
        if self.num_cores > max_cores:
            return False
        if self.per_core_M < 1 or self.per_core_N < 1:
            return False
        if self.kt_per_core < 1 or self.kt_per_core % self.in0_block_w:
            return False
        if self.family == "1D_in0":
            if self.Nt % self.num_cores:
                return False
            if self.in0 == "ws" and self.Kt % self.num_cores:
                return False
            return self.in0 in ("l1", "dram", "ws") and self.out in ("l1", "dram", "ws")
        if self.family == "1D_in1":
            if self.Mt % self.num_cores:
                return False
            return self.in0 in ("l1", "dram", "hs") and self.out in ("l1", "dram", "hs")
        if self.family == "2D":
            if self.Mt % self.grid_y or self.Nt % self.grid_x:
                return False
            if self.in0 == "bs" and self.Kt % self.grid_x:
                return False
            return self.in0 in ("l1", "dram", "bs") and self.out in ("l1", "dram", "bs")
        if self.family == "dram":
            if self.Nt % self.num_cores or self.Kt % self.num_cores:
                return False
            return self.in0 == "ws" and self.out == "ws"
        return False


_LAYOUTS = {
    "1D_in0": [("dram", "l1"), ("l1", "l1"), ("dram", "ws"), ("ws", "ws")],
    "1D_in1": [("dram", "l1"), ("l1", "l1"), ("dram", "hs"), ("hs", "hs")],
    "2D": [("dram", "l1"), ("l1", "l1"), ("dram", "bs"), ("bs", "bs")],
    "dram": [("ws", "ws")],
}


def _candidate_grids(
    family: str, Mt: int, Kt: int, Nt: int, gx_max: int, gy_max: int, max_cores: int
) -> list[tuple[int, int]]:
    cands: set[tuple[int, int]] = set()
    if family == "1D_in0":
        core_opts = [c for c in _divisors(Nt) if c <= max_cores]
    elif family == "1D_in1":
        core_opts = [c for c in _divisors(Mt) if c <= max_cores]
    elif family == "dram":
        core_opts = [c for c in _divisors(math.gcd(Kt, Nt)) if c <= max_cores]
    else:
        for gy in _divisors(Mt):
            if gy > gy_max:
                continue
            for gx in _divisors(Nt):
                if gx <= gx_max and gx * gy <= max_cores:
                    cands.add((gx, gy))
        core_opts = []
    for c in core_opts:
        cands.update(_grids_for_cores(c, gx_max, gy_max))
    return sorted(cands, key=lambda g: (-(g[0] * g[1]), -g[0]))[:_MAX_GRIDS_PER_FAMILY]


def _make_cases(M: int, K: int, N: int, gx_max: int, gy_max: int, max_cores: int) -> list[MatmulCase]:
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    cases: list[MatmulCase] = []
    for family in ("1D_in0", "1D_in1", "2D", "dram"):
        grids = _candidate_grids(family, Mt, Kt, Nt, gx_max, gy_max, max_cores)
        for gx, gy in grids:
            for in0, out in _LAYOUTS[family]:
                if family == "dram":
                    kt_pc = Kt // (gx * gy)
                    bws = [_largest_div_leq(kt_pc, max(1, kt_pc // 4))]
                else:
                    kt_pc = Kt // (gx * gy) if in0 in ("ws", "bs") else Kt
                    bws = [w for w in _IN0_BLOCK_W_CHOICES if kt_pc % w == 0]
                for w in bws:
                    case = MatmulCase(M, K, N, family, gx, gy, w, in0, out)
                    if case.feasible(gx_max, gy_max, max_cores):
                        cases.append(case)
    return cases


@dataclass
class CaseResult:
    case: MatmulCase
    dev_us: float
    tflops: float
    pcc: float
    pcc_pass: bool
    err: str = ""


def _pad_to_dram_banks(n: int, num_banks: int) -> int:
    lcm = TILE * num_banks
    rem = n % lcm
    return n if rem == 0 else n + (lcm - rem)


def _mem_in0(case: MatmulCase, device: ttnn.Device) -> ttnn.MemoryConfig:
    if case.in0 == "l1":
        return ttnn.L1_MEMORY_CONFIG
    if case.in0 == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    strategy = {
        "ws": ttnn.ShardStrategy.WIDTH,
        "hs": ttnn.ShardStrategy.HEIGHT,
        "bs": ttnn.ShardStrategy.BLOCK,
    }[case.in0]
    return ttnn.create_sharded_memory_config(
        (1, 1, case.M, case.K),
        core_grid=ttnn.CoreGrid(y=case.grid_y, x=case.grid_x),
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def _mem_in1(case: MatmulCase, device: ttnn.Device) -> ttnn.MemoryConfig:
    if case.family != "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    num_banks = device.dram_grid_size().x
    n_padded = _pad_to_dram_banks(case.N, num_banks)
    shard_shape = [case.K, n_padded // num_banks]
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _mem_out(case: MatmulCase, device: ttnn.Device) -> ttnn.MemoryConfig:
    if case.out == "l1":
        return ttnn.L1_MEMORY_CONFIG
    if case.out == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if case.family == "dram":
        return ttnn.create_sharded_memory_config(
            (1, 1, case.M, case.N),
            core_grid=ttnn.CoreGrid(y=case.grid_y, x=case.grid_x),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    layout = {
        "ws": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "hs": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "bs": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    }[case.out]
    return ttnn.MemoryConfig(memory_layout=layout, buffer_type=ttnn.BufferType.L1)


def _program_config(case: MatmulCase):
    sh, sw = case.subblock
    grid = ttnn.CoreCoord(case.grid_x, case.grid_y)
    if case.family == "dram":
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=case.in0_block_w,
            per_core_M=case.per_core_M,
            per_core_N=case.per_core_N,
            fused_activation=None,
        )
    if case.family in ("1D_in0", "1D_in1"):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=case.in0_block_w,
            out_subblock_h=sh,
            out_subblock_w=sw,
            per_core_M=case.per_core_M,
            per_core_N=case.per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=case.mcast_in0,
        )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=case.in0_block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=case.per_core_M,
        out_block_w=case.per_core_N,
        per_core_M=case.per_core_M,
        per_core_N=case.per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )


def _drain_profiler(device: ttnn.Device) -> None:
    ttnn.ReadDeviceProfiler(device)
    try:
        get_device_data_generate_report(PROFILER_LOGS_DIR, None, None, None, export_csv=False, cleanup_device_log=True)
    except Exception:
        pass


def _device_kernel_us(device: ttnn.Device) -> float | None:
    ttnn.ReadDeviceProfiler(device)
    data = get_device_data_generate_report(
        PROFILER_LOGS_DIR, None, None, None, export_csv=False, cleanup_device_log=True
    )
    durations = [float(d["DEVICE KERNEL DURATION [ns]"]) for d in data if "DEVICE KERNEL DURATION [ns]" in d]
    if not durations:
        return None
    return sum(durations) / len(durations) / 1e3


def _torch_dtype(tt_dtype: ttnn.DataType):
    if tt_dtype == ttnn.float32:
        return torch.float32
    if tt_dtype == ttnn.bfloat16:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {tt_dtype}")


@pytest.mark.parametrize("shape", _filtered_shapes(), ids=[s.sweep_id for s in _filtered_shapes()])
def test_matmul_decoder_perf_report_sweep(device, shape: ReportShape):
    if os.getenv("TT_METAL_DEVICE_PROFILER") is None:
        pytest.skip(
            "device-time sweep needs a profiler build: rebuild with --enable-profiler "
            "and set TT_METAL_DEVICE_PROFILER=1"
        )
    if os.getenv("TT_METAL_PROFILER_MID_RUN_DUMP") is None:
        pytest.skip(
            "set TT_METAL_PROFILER_MID_RUN_DUMP=1 so ReadDeviceProfiler flushes " "profile_log_device.csv mid-run"
        )

    M, K, N = shape.M, shape.K, shape.N
    grid = device.compute_with_storage_grid_size()
    gx_max, gy_max = grid.x, grid.y
    max_cores = gx_max * gy_max
    cases = _make_cases(M, K, N, gx_max, gy_max, max_cores)
    assert cases, f"no feasible configs generated for {M}x{K}x{N} on a {gx_max}x{gy_max} grid"

    pcc_target = _PCC_TARGETS[shape.out_dtype]
    torch_in0 = _torch_dtype(shape.in0_dtype)
    torch_in1 = _torch_dtype(shape.in1_dtype)
    torch_out = _torch_dtype(shape.out_dtype)

    torch.manual_seed(0)
    torch_input_a = torch.randn((1, 1, M, K), dtype=torch_in0)
    torch_input_b = torch.randn((1, 1, K, N), dtype=torch_in1)
    torch_output = torch.matmul(torch_input_a.float(), torch_input_b.float()).to(torch_out)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=shape.math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    results: list[CaseResult] = []
    for case in cases:
        in0 = in1 = warm = timed = None
        try:
            in0 = ttnn.from_torch(
                torch_input_a,
                dtype=shape.in0_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_mem_in0(case, device),
            )
            in1 = ttnn.from_torch(
                torch_input_b,
                dtype=shape.in1_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_mem_in1(case, device),
            )
            program_config = _program_config(case)
            out_mem_cfg = _mem_out(case, device)

            warm = ttnn.matmul(
                in0,
                in1,
                program_config=program_config,
                memory_config=out_mem_cfg,
                compute_kernel_config=compute_kernel_config,
                dtype=shape.out_dtype,
            )
            ttnn.synchronize_device(device)
            ttnn.deallocate(warm)
            warm = None
            _drain_profiler(device)
            ttnn.synchronize_device(device)

            timed = ttnn.matmul(
                in0,
                in1,
                program_config=program_config,
                memory_config=out_mem_cfg,
                compute_kernel_config=compute_kernel_config,
                dtype=shape.out_dtype,
            )
            ttnn.synchronize_device(device)
            dev_us = _device_kernel_us(device)
            if dev_us is None:
                raise RuntimeError("profiler produced no device data — is this a profiler build?")

            out = ttnn.to_torch(timed).float()
            assert out.shape == torch_output.shape, f"{out.shape} != {torch_output.shape}"
            pcc_pass, pcc = comp_pcc(torch_output.float(), out, pcc_target)
            tflops = 2 * M * K * N / 1e6 / dev_us
            results.append(CaseResult(case, dev_us, tflops, float(pcc), bool(pcc_pass)))
        except Exception as e:
            results.append(CaseResult(case, float("inf"), 0.0, 0.0, False, str(e).strip().splitlines()[0][:90]))
        finally:
            for t in (timed, warm, in1, in0):
                if t is not None:
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass
            ttnn.synchronize_device(device)

    env_us = os.getenv("MATMUL_BASELINE_US")
    must_beat_us = float(env_us) if env_us is not None else shape.report_us
    passing = [r for r in results if r.pcc_pass]
    winners = [r for r in passing if must_beat_us is None or r.dev_us < must_beat_us]
    best = min(passing, key=lambda r: r.dev_us) if passing else None

    logger.info(
        f"decoder matmul sweep {shape.sweep_id} in0={shape.in0_dtype} in1={shape.in1_dtype} "
        f"out={shape.out_dtype} fidelity={shape.math_fidelity} grid={gx_max}x{gy_max}"
    )
    logger.info(
        f"{'family':>7} {'in0/in1/out':>12} {'grid':>5} {'cores':>5} {'ibw':>4} {'pcM':>4} {'pcN':>4} "
        f"{'sub':>5} {'dev_us':>9} {'TFLOPs':>7} {'PCC':>8} {'result':>7}  note"
    )
    for r in sorted(results, key=lambda r: r.dev_us):
        c = r.case
        sh, sw = c.subblock
        if r.err:
            metrics = f"{'-':>9} {'-':>7} {'-':>8} {'ERROR':>7}"
            note = r.err
        else:
            metrics = f"{r.dev_us:>9.2f} {r.tflops:>7.2f} {r.pcc:>8.4f} {('PASS' if r.pcc_pass else 'FAIL'):>7}"
            note = "best" if r is best else ""
        logger.info(
            f"{c.family:>7} {c.layout:>12} {f'{c.grid_x}x{c.grid_y}':>5} {c.num_cores:>5} {c.in0_block_w:>4} "
            f"{c.per_core_M:>4} {c.per_core_N:>4} {f'{sh}x{sw}':>5} {metrics}  {note}"
        )

    must_beat_str = f"{must_beat_us:.2f}us" if must_beat_us is not None else "none (no baseline)"
    if best is not None:
        speedup = f" | speedup={must_beat_us / best.dev_us:.2f}x" if must_beat_us is not None else ""
        logger.info(
            f"FASTEST PCC-PASS: {best.case.label} -> {best.dev_us:.2f}us, {best.tflops:.2f} TFLOPs, "
            f"PCC={best.pcc:.4f} | must-beat={must_beat_str}{speedup}"
        )
    else:
        logger.info(f"FASTEST PCC-PASS: none (no config reached PCC>={pcc_target}) | must-beat={must_beat_str}")

    assert passing, f"No config reached PCC>={pcc_target} for {shape.sweep_id} — sweep harness or shape is broken."
    if must_beat_us is not None and not winners:
        logger.warning(
            f"No swept config beat the report baseline {must_beat_us:.2f}us for {shape.sweep_id}; "
            f"fastest PCC-pass = {best.dev_us:.2f}us ({best.case.label}). "
            f"The production program config may already be optimal, or the baseline includes "
            f"layout/reshard overhead not modeled in this isolated matmul."
        )
    if os.getenv("MATMUL_STRICT_BASELINE"):
        assert winners, (
            f"No config both passed PCC>={pcc_target} and beat the report time {must_beat_us:.2f}us for "
            f"{shape.sweep_id}. The report config is already optimal among the swept configs "
            f"(fastest PCC-pass = {best.dev_us:.2f}us)."
        )
