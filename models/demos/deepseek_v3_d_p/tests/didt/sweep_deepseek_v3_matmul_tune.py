# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sweep matmul program-config parameters for Deepseek V3 prefill and print timing.

Requires Tracy for device-level profiling. Set the env vars before running pytest:

Usage:
  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
  pytest models/demos/deepseek_v3_d_p/tests/didt/sweep_deepseek_v3_matmul_tune.py -v -s --didt-workload-iterations 1000
  pytest ... -k "dense_mlp_w1" --didt-workload-iterations 100
  pytest ... --timeout=7200

DRAM tensors, 11×10 grid. MLA/Gate: HiFi2; MoE: LoFi. Timing and core count
come from Tracy device profiler data. Default timeout 3600s.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Iterator

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, skip_for_wormhole_b0
from models.demos.deepseek_v3_d_p.tests.deepseek_v3_matmul_config import (
    DENSE_MLP_MATMUL_PARAMS,
    GATE_MATMUL_CONFIG,
    GRID_SIZE,
    MLA_MATMUL_PARAMS,
    ROUTED_EXPERT_MATMUL_PARAMS,
    SHARED_EXPERT_MATMUL_PARAMS,
    TILE_SIZE,
)
from tests.didt.op_test_base import OpParameter, OpTestBase
from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time

# ---------------------------------------------------------------------------
# Compute utilization: ideal cycles vs actual cycles (same as test_benchmark.py)
# ---------------------------------------------------------------------------

# Cycles per tile for each math fidelity (Tensix matmul on 2 tiles).
CYCLES_PER_TILE = {
    ttnn.MathFidelity.LoFi: 16,
    ttnn.MathFidelity.HiFi2: 32,
    ttnn.MathFidelity.HiFi3: 48,
    ttnn.MathFidelity.HiFi4: 64,
}
TILE_H = TILE_W = 32


# Device frequency (MHz). Used to convert duration_ns -> cycles. Blackhole 1350, Wormhole 1000.
def _device_freq_mhz() -> float:
    return 1350.0 if is_blackhole() else 1000.0


def compute_utilization_pct(
    M: int,
    K: int,
    N: int,
    duration_ns: int,
    num_cores: int,
    math_fidelity: Any,
) -> float:
    """
    Compute utilization % vs theoretical peak (ideal cycles / actual cycles).
    num_cores is the grid size used for ideal_cycles (sweep uses full GRID_SIZE, e.g. 110,
    so utilization is comparable across configs that may use fewer cores).
    Formula matches tests/ttnn/unit_tests/benchmarks/test_benchmark.py:
      ideal_cycle = M*K*N / (tile_h*tile_w*32) * cycle_per_tile / num_cores
      inference_cycle = duration_sec * device_freq_Hz = (duration_ns*1e-9) * (freq_MHz*1e6)
      utilization = ideal_cycle / inference_cycle * 100
    """
    cycle_per_tile = CYCLES_PER_TILE.get(math_fidelity, CYCLES_PER_TILE[ttnn.MathFidelity.LoFi])
    ideal_cycles = (M * K * N) / (TILE_H * TILE_W * 32) * cycle_per_tile / num_cores
    duration_sec = duration_ns * 1e-9
    freq_hz = _device_freq_mhz() * 1e6
    inference_cycles = duration_sec * freq_hz
    if inference_cycles <= 0:
        return 0.0
    return (ideal_cycles / inference_cycles) * 100.0


# ---------------------------------------------------------------------------
# Helpers: divisors and parameter sweep generation
# ---------------------------------------------------------------------------


def _divisors_up_to(n: int, max_val: int) -> list[int]:
    """Return sorted list of divisors of n that are <= max_val."""
    out = []
    for d in range(1, min(int(math.isqrt(n)) + 1, max_val + 1)):
        if n % d == 0:
            if d <= max_val:
                out.append(d)
            comp = n // d
            if comp != d and comp <= max_val:
                out.append(comp)
    return sorted(set(out))


def _sweep_program_config_params(
    M: int, K: int, N: int, grid_size: tuple[int, int] = GRID_SIZE
) -> Iterator[dict[str, Any]]:
    """Yield dicts of (in0_block_w, out_subblock_h, out_subblock_w) and derived per_core_* for valid configs."""
    grid_x, grid_y = grid_size
    M_tiles = math.ceil(M / TILE_SIZE)
    K_tiles = math.ceil(K / TILE_SIZE)
    N_tiles = math.ceil(N / TILE_SIZE)
    per_core_M = math.ceil(M_tiles / grid_y)
    per_core_N = math.ceil(N_tiles / grid_x)

    # in0_block_w: divisors of K_tiles; cap to avoid L1 overflow (e.g. 32)
    in0_block_w_candidates = _divisors_up_to(K_tiles, 32)
    if not in0_block_w_candidates:
        in0_block_w_candidates = [1]

    # out_subblock_h must divide per_core_M; out_subblock_w must divide per_core_N; product <= 8
    out_subblock_h_candidates = _divisors_up_to(per_core_M, 8)
    out_subblock_w_candidates = _divisors_up_to(per_core_N, 8)
    if not out_subblock_h_candidates:
        out_subblock_h_candidates = [1]
    if not out_subblock_w_candidates:
        out_subblock_w_candidates = [1]

    for in0_block_w in in0_block_w_candidates:
        for out_subblock_h in out_subblock_h_candidates:
            for out_subblock_w in out_subblock_w_candidates:
                if out_subblock_h * out_subblock_w > 8:
                    continue
                if per_core_M % out_subblock_h != 0 or per_core_N % out_subblock_w != 0:
                    continue
                yield {
                    "in0_block_w": in0_block_w,
                    "out_subblock_h": out_subblock_h,
                    "out_subblock_w": out_subblock_w,
                    "per_core_M": per_core_M,
                    "per_core_N": per_core_N,
                    "grid_size": grid_size,
                }


def _make_program_config(
    M: int, K: int, N: int, params: dict[str, Any]
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """Build MatmulMultiCoreReuseMultiCastProgramConfig from sweep params."""
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=params["grid_size"],
        in0_block_w=params["in0_block_w"],
        out_subblock_h=params["out_subblock_h"],
        out_subblock_w=params["out_subblock_w"],
        per_core_M=params["per_core_M"],
        per_core_N=params["per_core_N"],
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------


@dataclass
class MatmulWorkload:
    """Single matmul workload: shapes, weight dtype, and math fidelity."""

    workload_id: str
    M: int
    K: int
    N: int
    batch: int
    in1_dtype: Any
    math_fidelity: Any


def _moe_workload(M: int, K: int, N: int, workload_id: str) -> MatmulWorkload:
    """MoE matmul: bfloat4_b, LoFi."""
    return MatmulWorkload(
        workload_id=workload_id,
        M=M,
        K=K,
        N=N,
        batch=1,
        in1_dtype=ttnn.DataType.BFLOAT4_B,
        math_fidelity=ttnn.MathFidelity.LoFi,
    )


def _deepseek_v3_workloads() -> list[MatmulWorkload]:
    """All Deepseek V3 matmul workloads (MLA, gate, dense MLP, shared expert, routed expert)."""
    workloads: list[MatmulWorkload] = []
    for row in MLA_MATMUL_PARAMS:
        M, K, N, batch, in1_dtype, workload_id = row
        workloads.append(
            MatmulWorkload(
                workload_id=workload_id,
                M=M,
                K=K,
                N=N,
                batch=batch,
                in1_dtype=in1_dtype,
                math_fidelity=ttnn.MathFidelity.HiFi2,
            )
        )
    M, K, N, in1_dtype, workload_id = GATE_MATMUL_CONFIG
    workloads.append(
        MatmulWorkload(
            workload_id=workload_id,
            M=M,
            K=K,
            N=N,
            batch=1,
            in1_dtype=in1_dtype,
            math_fidelity=ttnn.MathFidelity.HiFi2,
        )
    )
    for row in DENSE_MLP_MATMUL_PARAMS:
        M, K, N, workload_id = row
        workloads.append(_moe_workload(M, K, N, workload_id))
    for row in SHARED_EXPERT_MATMUL_PARAMS:
        M, K, N, workload_id = row
        workloads.append(_moe_workload(M, K, N, workload_id))
    for row in ROUTED_EXPERT_MATMUL_PARAMS:
        M, K, N, workload_id = row
        workloads.append(_moe_workload(M, K, N, workload_id))
    return workloads


# ---------------------------------------------------------------------------
# Run one (workload, program_config) and return duration
# ---------------------------------------------------------------------------


def _short_error(e: BaseException, max_len: int = 120) -> str:
    """Type and first line of exception message, no backtrace."""
    msg = str(e).strip().split("\n")[0] if str(e).strip() else ""
    out = f"{type(e).__name__}: {msg}" if msg else type(e).__name__
    return out[:max_len] + ("..." if len(out) > max_len else "")


def _get_tracy_timing_and_cores(device_id: int) -> tuple[int | None, int | None]:
    """
    After ReadDeviceProfiler, get timing (ns) and core count from Tracy program perf data.
    Returns (duration_ns, core_count) or (None, None) if unavailable (profiler not enabled).
    """
    try:
        latest = ttnn.get_latest_programs_perf_data()
    except Exception:
        return (None, None)
    if not latest or device_id not in latest:
        return (None, None)
    programs = latest[device_id]
    if not programs:
        return (None, None)
    duration_ns = None
    max_cores = 0
    for p in programs:
        if p.core_count > max_cores:
            max_cores = p.core_count
        for key in ("DEVICE KERNEL DURATION [ns]", "DEVICE FW DURATION [ns]"):
            if key in p.program_analyses_results:
                d = p.program_analyses_results[key].duration
                if d is not None:
                    duration_ns = max(duration_ns, d) if duration_ns is not None else d
                break
    out_duration = int(duration_ns) if duration_ns is not None else None
    out_cores = max_cores if max_cores > 0 else None
    return (out_duration, out_cores)


@dataclass
class SweepResult:
    workload_id: str
    M: int
    K: int
    N: int
    batch: int
    in0_block_w: int
    out_subblock_h: int
    out_subblock_w: int
    duration_ns: int
    iterations: int
    utilization_pct: float = 0.0
    memory_configs: str = ""  # e.g. "in0:DRAM in1:DRAM out:DRAM"
    core_count: int = 0  # cores the matmul actually ran on (from Tracy or config grid)

    @property
    def duration_per_iter_ns(self) -> int:
        return self.duration_ns // max(1, self.iterations)

    def to_csv_row(self) -> str:
        return (
            f"{self.workload_id},{self.M},{self.K},{self.N},{self.batch},"
            f"{self.in0_block_w},{self.out_subblock_h},{self.out_subblock_w},"
            f"{self.duration_ns},{self.duration_per_iter_ns},{self.duration_per_iter_ns / 1e3:.2f},"
            f'{self.utilization_pct},{self.core_count},"{self.memory_configs}"'
        )


def _run_single_config(
    mesh_device: Any,
    wl: MatmulWorkload,
    program_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig,
    iterations: int,
) -> SweepResult | None:
    """Run matmul for one (workload, program_config) and return timing. Returns None on skip (e.g. batch > 1)."""
    if wl.batch != 1:
        return None
    dram_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=wl.math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    in0_shape = [1, wl.batch, wl.M, wl.K]
    in1_shape = [1, wl.batch, wl.K, wl.N]
    memory_configs = "in0:DRAM in1:DRAM out:DRAM"

    try:
        activation = OpParameter(in0_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT, dram_mem_config)
        arguments = [OpParameter(in1_shape, wl.in1_dtype, ttnn.TILE_LAYOUT, dram_mem_config)]
        test = OpTestBase(
            mesh_device,
            activation=activation,
            arguments=arguments,
            out_mem_config=dram_mem_config,
            out_dtype=ttnn.DataType.BFLOAT16,
            program_config=program_config,
            compute_config=compute_config,
            loop_count=iterations,
            determinism_check_enabled=False,
            determinism_check_interval=False,
        )
        test.set_seed()
        a_shape = test.activation.shape
        b_shape = test.arguments[0].shape
        A = test.generate_torch_activations(a_shape)
        B = test.generate_torch_input(b_shape)
        a_t = test.generate_tt_activations_from_torch(A)
        test.inputs = [
            test.generate_tt_input_from_torch(
                B,
                test.arguments[0].dtype,
                test.arguments[0].layout,
                test.arguments[0].mem_config,
                0,
            )
        ]
        test.activations = test.convert_activations_to_memory_config(a_t)

        start = start_measuring_time()
        for _ in range(iterations):
            out = test.run_device_operation()
            for device_idx in test.device_ids:
                ttnn.device.synchronize_device(test.get_device(device_idx))
            out.deallocate(True)
        duration_ns = stop_measuring_time(start)

        ttnn.ReadDeviceProfiler(mesh_device)

        tracy_duration_ns, tracy_core_count = _get_tracy_timing_and_cores(test.device_ids[0])
        grid = program_config.compute_with_storage_grid_size
        config_cores = grid[0] * grid[1] if isinstance(grid, (tuple, list)) else grid.x * grid.y
        if tracy_duration_ns is None:
            pytest.fail(
                "Tracy device profiler data unavailable. "
                "Make sure Tracy env vars are exported before running pytest."
            )
        duration_per_iter_ns = tracy_duration_ns
        duration_ns = tracy_duration_ns * iterations
        num_cores_used = tracy_core_count if tracy_core_count is not None else config_cores
        # Utilization vs full grid (110 cores) so configs are comparable
        grid_x, grid_y = GRID_SIZE
        num_cores_full_grid = grid_x * grid_y
        utilization_pct = compute_utilization_pct(
            wl.M,
            wl.K,
            wl.N,
            duration_per_iter_ns,
            num_cores_full_grid,
            wl.math_fidelity,
        )

        test.deallocate_activations()
        test.inputs[0].deallocate(True)

        return SweepResult(
            workload_id=wl.workload_id,
            M=wl.M,
            K=wl.K,
            N=wl.N,
            batch=wl.batch,
            in0_block_w=program_config.in0_block_w,
            out_subblock_h=program_config.out_subblock_h,
            out_subblock_w=program_config.out_subblock_w,
            duration_ns=duration_ns,
            iterations=iterations,
            utilization_pct=utilization_pct,
            memory_configs=memory_configs,
            core_count=num_cores_used,
        )
    except Exception as e:
        err_short = _short_error(e)
        logger.warning(f"Skip (OOM or unsupported): {wl.workload_id} [{memory_configs}] -> {err_short}")
        print(
            f"# SKIP {wl.workload_id} ({wl.M}x{wl.K}x{wl.N}) "
            f"in0_bw={program_config.in0_block_w} subblock={program_config.out_subblock_h}x{program_config.out_subblock_w}: {err_short}",
            flush=True,
        )
        return None


# ---------------------------------------------------------------------------
# Sweep entrypoint
# ---------------------------------------------------------------------------

SWEEP_CSV_HEADER = (
    "workload_id,M,K,N,batch,in0_block_w,out_subblock_h,out_subblock_w,"
    "duration_ns,duration_per_iter_ns,duration_per_iter_us,utilization_pct,core_count,memory_configs"
)


@skip_for_wormhole_b0("Grid 11x10 requires Blackhole")
@pytest.mark.timeout(3600)  # 1 hour default; override with pytest --timeout=SECONDS
@pytest.mark.parametrize("mesh_device", [pytest.param(1, id="1chips")], indirect=["mesh_device"])
def test_sweep_deepseek_v3_matmul_tune(
    mesh_device: Any,
    didt_workload_iterations: int,
) -> None:
    """Sweep program configs per workload; print timing and utilization. Use -s for output.

    Requires Tracy env vars to be exported before running pytest. See module docstring for usage.
    """
    _required_env = {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    }
    missing = [k for k, v in _required_env.items() if os.environ.get(k) != v]
    if missing:
        pytest.fail(
            f"Tracy env vars not set: {', '.join(missing)}. "
            "These must be exported before launching pytest:\n"
            "  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1\n"
            "  pytest <this_file> -v -s --didt-workload-iterations N"
        )

    iterations = max(1, min(didt_workload_iterations, 1000))
    workloads = [w for w in _deepseek_v3_workloads() if w.batch == 1]

    print(SWEEP_CSV_HEADER, flush=True)
    all_results: list[SweepResult] = []

    for wl in workloads:
        param_list = list(_sweep_program_config_params(wl.M, wl.K, wl.N, grid_size=GRID_SIZE))
        logger.info(f"Workload {wl.workload_id} ({wl.M}x{wl.K}x{wl.N}): {len(param_list)} configs")
        for params in param_list:
            pc = _make_program_config(wl.M, wl.K, wl.N, params)
            try:
                res = _run_single_config(mesh_device, wl, pc, iterations)
            except Exception as e:
                print(
                    f"# SKIP {wl.workload_id} in0_bw={params['in0_block_w']} "
                    f"subblock={params['out_subblock_h']}x{params['out_subblock_w']}: {_short_error(e)}",
                    flush=True,
                )
                continue
            if res is None:
                continue
            all_results.append(res)
            print(res.to_csv_row(), flush=True)

    # Summary: best config(s) per workload — includes all configs within 1% of best duration
    print("\n# Best config(s) per workload (within 1% of fastest duration):", flush=True)
    by_workload: dict[str, list[SweepResult]] = {}
    for r in all_results:
        by_workload.setdefault(r.workload_id, []).append(r)
    for wid, results in sorted(by_workload.items()):
        ranked = sorted(results, key=lambda x: x.duration_per_iter_ns)
        best_ns = ranked[0].duration_per_iter_ns
        threshold = best_ns * 1.01
        top = [r for r in ranked if r.duration_per_iter_ns <= threshold]
        shape = f"{top[0].M}x{top[0].K}x{top[0].N}"
        print(f"\n  {wid} ({shape}) — {len(top)} config(s) within 1%:", flush=True)
        for i, r in enumerate(top):
            tag = "*" if i == 0 else " "
            print(
                f"    {tag} in0_block_w={r.in0_block_w} "
                f"out_subblock_h={r.out_subblock_h} out_subblock_w={r.out_subblock_w} "
                f"cores={r.core_count} [{r.memory_configs}] "
                f"-> {r.utilization_pct:.2f}% util, {r.duration_per_iter_ns} ns/iter ({r.duration_per_iter_ns/1e3:.2f} us)",
                flush=True,
            )
