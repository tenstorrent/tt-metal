# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Sweep matmul program-config parameters for the Mistral-Medium-3.5 dense MLP and print timing.

Ported from ``deepseek_v3_d_p/tests/didt/sweep_deepseek_v3_matmul_tune.py`` for the per-chip MLP
shapes at the SP=8 x TP=4 target: w13 = M x 12288 x 14336 (fused gate/up), w2 = M x 7168 x 12288
(down), M = global_seq / 8. Sweeps in0_block_w x out_subblock_h x out_subblock_w of
MatmulMultiCoreReuseMultiCastProgramConfig on ONE chip, DRAM interleaved in/out, bf16 activations
x bfp8 weights. Timing = DEVICE KERNEL DURATION from the in-process device profiler; prints a CSV
and a best-configs summary per workload. Winners go into ``_SWEPT_PROGRAM_CONFIGS`` in tt/mlp.py.

Requires the profiler env vars to be exported BEFORE launching pytest:

  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
  pytest models/demos/mistral_medium_d_p/tests/perf/sweep_mlp_matmul_tune.py -v -s \
      --didt-workload-iterations 10 --timeout=7200
  pytest ... -k "w13 and hifi2" ...
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Iterator

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, skip_for_wormhole_b0
from tests.didt.op_test_base import OpParameter, OpTestBase

TILE_SIZE = 32

# Per-chip MLP matmul dims at TP=4 (see tests/unit/shapes.py: HIDDEN=12288, FFN=28672).
HIDDEN = 12288
FFN_LOCAL = 7168  # FFN / tp
W13_N = 2 * FFN_LOCAL  # fused gate|up

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
    """Compute utilization % vs theoretical peak at ``num_cores`` (use the full grid so
    configs that end up on fewer cores stay comparable)."""
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


def _sweep_program_config_params(M: int, K: int, N: int, grid_size: tuple[int, int]) -> Iterator[dict[str, Any]]:
    """Yield dicts of (in0_block_w, out_subblock_h, out_subblock_w) and derived per_core_* for valid configs."""
    grid_x, grid_y = grid_size
    M_tiles = math.ceil(M / TILE_SIZE)
    K_tiles = math.ceil(K / TILE_SIZE)
    N_tiles = math.ceil(N / TILE_SIZE)
    per_core_M = math.ceil(M_tiles / grid_y)
    per_core_N = math.ceil(N_tiles / grid_x)

    # in0_block_w: divisors of K_tiles; cap to avoid L1 overflow
    in0_block_w_candidates = _divisors_up_to(K_tiles, 32) or [1]

    # out_subblock_h must divide per_core_M; out_subblock_w must divide per_core_N; product <= 8
    out_subblock_h_candidates = _divisors_up_to(per_core_M, 8) or [1]
    out_subblock_w_candidates = _divisors_up_to(per_core_N, 8) or [1]

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


def _make_program_config(params: dict[str, Any]) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
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
    in1_dtype: Any
    math_fidelity: Any


# Per-chip M values to sweep: global prefill length / 8 (SP rows). Extend when new prefill
# lengths land, re-sweep, and copy the winners into tt/mlp.py.
_M_VALUES = (640,)  # 5k prefill


def _mlp_workloads() -> list[MatmulWorkload]:
    """Mistral MLP per-chip matmuls: w13 (gate/up fused) and w2 (down), bfp8 weights."""
    return [
        MatmulWorkload(
            workload_id=f"{name}_m{m}_{fid_name}",
            M=m,
            K=k,
            N=n,
            in1_dtype=ttnn.DataType.BFLOAT8_B,
            math_fidelity=fidelity,
        )
        for m in _M_VALUES
        for name, k, n in (("w13", HIDDEN, W13_N), ("w2", FFN_LOCAL, HIDDEN))
        for fid_name, fidelity in (("hifi2", ttnn.MathFidelity.HiFi2), ("lofi", ttnn.MathFidelity.LoFi))
    ]


# ---------------------------------------------------------------------------
# Run one (workload, program_config) and return duration
# ---------------------------------------------------------------------------


def _short_error(e: BaseException, max_len: int = 120) -> str:
    msg = str(e).strip().split("\n")[0] if str(e).strip() else ""
    out = f"{type(e).__name__}: {msg}" if msg else type(e).__name__
    return out[:max_len] + ("..." if len(out) > max_len else "")


def _get_tracy_timing_and_cores(device_id: int) -> tuple[int | None, int | None]:
    """After ReadDeviceProfiler, get timing (ns) and core count from the in-process profiler data."""
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
    in0_block_w: int
    out_subblock_h: int
    out_subblock_w: int
    duration_ns: int
    iterations: int
    utilization_pct: float = 0.0
    memory_configs: str = ""
    core_count: int = 0

    @property
    def duration_per_iter_ns(self) -> int:
        return self.duration_ns // max(1, self.iterations)

    def to_csv_row(self) -> str:
        return (
            f"{self.workload_id},{self.M},{self.K},{self.N},"
            f"{self.in0_block_w},{self.out_subblock_h},{self.out_subblock_w},"
            f"{self.duration_ns},{self.duration_per_iter_ns},{self.duration_per_iter_ns / 1e3:.2f},"
            f'{self.utilization_pct:.2f},{self.core_count},"{self.memory_configs}"'
        )


def _run_single_config(
    mesh_device: Any,
    wl: MatmulWorkload,
    program_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig,
    iterations: int,
    full_grid_cores: int,
) -> SweepResult | None:
    dram_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=wl.math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    in0_shape = [1, 1, wl.M, wl.K]
    in1_shape = [1, 1, wl.K, wl.N]
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
        A = test.generate_torch_activations(test.activation.shape)
        B = test.generate_torch_input(test.arguments[0].shape)
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

        for _ in range(iterations):
            out = test.run_device_operation()
            for device_idx in test.device_ids:
                ttnn.device.synchronize_device(test.get_device(device_idx))
            out.deallocate(True)

        ttnn.ReadDeviceProfiler(mesh_device)

        tracy_duration_ns, tracy_core_count = _get_tracy_timing_and_cores(test.device_ids[0])
        grid = program_config.compute_with_storage_grid_size
        config_cores = grid[0] * grid[1] if isinstance(grid, (tuple, list)) else grid.x * grid.y
        if tracy_duration_ns is None:
            pytest.fail(
                "Device profiler data unavailable. "
                "Make sure the profiler env vars are exported before running pytest."
            )
        duration_per_iter_ns = tracy_duration_ns
        num_cores_used = tracy_core_count if tracy_core_count is not None else config_cores
        utilization_pct = compute_utilization_pct(
            wl.M, wl.K, wl.N, duration_per_iter_ns, full_grid_cores, wl.math_fidelity
        )

        test.deallocate_activations()
        test.inputs[0].deallocate(True)

        return SweepResult(
            workload_id=wl.workload_id,
            M=wl.M,
            K=wl.K,
            N=wl.N,
            in0_block_w=program_config.in0_block_w,
            out_subblock_h=program_config.out_subblock_h,
            out_subblock_w=program_config.out_subblock_w,
            duration_ns=tracy_duration_ns * iterations,
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
    "workload_id,M,K,N,in0_block_w,out_subblock_h,out_subblock_w,"
    "duration_ns,duration_per_iter_ns,duration_per_iter_us,utilization_pct,core_count,memory_configs"
)

_WORKLOADS = _mlp_workloads()


@skip_for_wormhole_b0("Tuned for the Blackhole Galaxy chips")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [pytest.param(1, id="1chips")], indirect=["mesh_device"])
@pytest.mark.parametrize("workload", _WORKLOADS, ids=[w.workload_id for w in _WORKLOADS])
def test_sweep_mistral_mlp_matmul_tune(
    mesh_device: Any,
    workload: MatmulWorkload,
    didt_workload_iterations: int,
) -> None:
    """Sweep program configs for one MLP matmul workload; print CSV + best-configs summary. Use -s."""
    _required_env = {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    }
    missing = [k for k, v in _required_env.items() if os.environ.get(k) != v]
    if missing:
        pytest.fail(
            f"Profiler env vars not set: {', '.join(missing)}. "
            "These must be exported before launching pytest:\n"
            "  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1"
        )

    compute_grid = mesh_device.compute_with_storage_grid_size()
    grid_size = (compute_grid.x, compute_grid.y)
    full_grid_cores = compute_grid.x * compute_grid.y

    iterations = max(1, min(didt_workload_iterations, 1000))
    wl = workload

    param_list = list(_sweep_program_config_params(wl.M, wl.K, wl.N, grid_size=grid_size))
    logger.info(f"Workload {wl.workload_id} ({wl.M}x{wl.K}x{wl.N}) on grid {grid_size}: {len(param_list)} configs")

    print(SWEEP_CSV_HEADER, flush=True)
    results: list[SweepResult] = []
    for params in param_list:
        pc = _make_program_config(params)
        try:
            res = _run_single_config(mesh_device, wl, pc, iterations, full_grid_cores)
        except Exception as e:
            print(
                f"# SKIP {wl.workload_id} in0_bw={params['in0_block_w']} "
                f"subblock={params['out_subblock_h']}x{params['out_subblock_w']}: {_short_error(e)}",
                flush=True,
            )
            continue
        if res is None:
            continue
        results.append(res)
        print(res.to_csv_row(), flush=True)

    if not results:
        pytest.fail(f"No config of {len(param_list)} ran successfully for {wl.workload_id}")

    # Summary: all configs within 1% of the fastest
    ranked = sorted(results, key=lambda x: x.duration_per_iter_ns)
    best_ns = ranked[0].duration_per_iter_ns
    top = [r for r in ranked if r.duration_per_iter_ns <= best_ns * 1.01]
    print(f"\n# {wl.workload_id} ({wl.M}x{wl.K}x{wl.N}) — {len(top)} config(s) within 1% of fastest:", flush=True)
    for i, r in enumerate(top):
        tag = "*" if i == 0 else " "
        print(
            f"    {tag} in0_block_w={r.in0_block_w} "
            f"out_subblock_h={r.out_subblock_h} out_subblock_w={r.out_subblock_w} "
            f"cores={r.core_count} -> {r.utilization_pct:.2f}% util, "
            f"{r.duration_per_iter_ns} ns/iter ({r.duration_per_iter_ns / 1e3:.2f} us)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# DRAM-sharded variant: weights width-sharded across the DRAM banks, activations
# width-sharded in L1 (the llama decode-MLP layout, models/common/modules/mlp/mlp_1d.py).
# ---------------------------------------------------------------------------


def _dram_sharded_weight_mem_config(mesh_device, K: int, N: int) -> ttnn.MemoryConfig:
    """Width-shard a [K, N] weight across the chip's DRAM banks (8 on BH p150-class)."""
    dram_grid_size = mesh_device.dram_grid_size()
    assert dram_grid_size.y == 1, "DRAM sharding assumes a 1-row DRAM grid"
    dram_cores = dram_grid_size.x
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    padded_n = math.ceil(N / (TILE_SIZE * dram_cores)) * (TILE_SIZE * dram_cores)
    shard_spec = ttnn.ShardSpec(dram_grid, (K, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _l1_width_sharded_mem_config(grid: tuple[int, int], num_cores: int, M: int, K: int) -> ttnn.MemoryConfig:
    """Width-shard a [M, K] activation across num_cores worker cores in L1."""
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(grid[0], grid[1]), row_wise=True)
    shard_spec = ttnn.ShardSpec(core_grid, (M, K // num_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def _dram_sharded_sweep_params(K: int, grid: tuple[int, int]) -> Iterator[dict[str, Any]]:
    """(num_cores, in0_block_w) candidates: num_cores must divide K_tiles (in0 L1 width shard),
    in0_block_w must divide the per-core K tiles."""
    K_tiles = K // TILE_SIZE
    full = grid[0] * grid[1]
    # The largest core counts give the most read parallelism; sweep the top few.
    core_candidates = sorted((d for d in _divisors_up_to(K_tiles, full) if d >= full // 4), reverse=True)[:4]
    for num_cores in core_candidates:
        for in0_block_w in _divisors_up_to(K_tiles // num_cores, 32):
            yield {"num_cores": num_cores, "in0_block_w": in0_block_w}


def _run_dram_sharded_config(
    mesh_device: Any,
    wl: MatmulWorkload,
    num_cores: int,
    in0_block_w: int,
    iterations: int,
    grid: tuple[int, int],
    full_grid_cores: int,
) -> SweepResult | None:
    memory_configs = f"in0:L1-WS({num_cores}) in1:DRAM-WS out:L1-WS"
    ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    compute_config = ComputeConfigClass(
        math_fidelity=wl.math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=math.ceil(wl.M / TILE_SIZE),
        per_core_N=math.ceil(wl.N / (TILE_SIZE * num_cores)),
        fused_activation=None,
    )
    x_dram = w = x_l1 = None
    try:
        torch.manual_seed(0)
        A = torch.randn(1, 1, wl.M, wl.K, dtype=torch.bfloat16)
        B = torch.randn(1, 1, wl.K, wl.N, dtype=torch.bfloat16)
        x_dram = ttnn.from_torch(
            A,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        x_l1 = ttnn.to_memory_config(x_dram, _l1_width_sharded_mem_config(grid, num_cores, wl.M, wl.K))
        w = ttnn.from_torch(
            B,
            dtype=wl.in1_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=_dram_sharded_weight_mem_config(mesh_device, wl.K, wl.N),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        for _ in range(iterations):
            out = ttnn.linear(
                x_l1,
                w,
                dtype=ttnn.DataType.BFLOAT16,
                program_config=program_config,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=compute_config,
            )
            ttnn.synchronize_device(mesh_device)
            out.deallocate(True)

        ttnn.ReadDeviceProfiler(mesh_device)
        device_id = mesh_device.get_device_ids()[0]
        tracy_duration_ns, tracy_core_count = _get_tracy_timing_and_cores(device_id)
        if tracy_duration_ns is None:
            pytest.fail("Device profiler data unavailable — export the profiler env vars first.")
        utilization_pct = compute_utilization_pct(
            wl.M, wl.K, wl.N, tracy_duration_ns, full_grid_cores, wl.math_fidelity
        )
        return SweepResult(
            workload_id=wl.workload_id,
            M=wl.M,
            K=wl.K,
            N=wl.N,
            in0_block_w=in0_block_w,
            out_subblock_h=0,  # not a knob in the DRAM-sharded config
            out_subblock_w=0,
            duration_ns=tracy_duration_ns * iterations,
            iterations=iterations,
            utilization_pct=utilization_pct,
            memory_configs=memory_configs,
            core_count=tracy_core_count if tracy_core_count is not None else num_cores,
        )
    except Exception as e:
        print(
            f"# SKIP {wl.workload_id} ({wl.M}x{wl.K}x{wl.N}) "
            f"num_cores={num_cores} in0_bw={in0_block_w}: {_short_error(e)}",
            flush=True,
        )
        return None
    finally:
        for t in (x_l1, w, x_dram):
            if t is not None:
                try:
                    t.deallocate(True)
                except Exception:
                    pass


_DS_WORKLOADS = [w for w in _WORKLOADS if w.workload_id in ("w13_m640_hifi2", "w2_m640_hifi2")]


@skip_for_wormhole_b0("Tuned for the Blackhole Galaxy chips")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [pytest.param(1, id="1chips")], indirect=["mesh_device"])
@pytest.mark.parametrize("workload", _DS_WORKLOADS, ids=[f"{w.workload_id}_ds" for w in _DS_WORKLOADS])
def test_sweep_mistral_mlp_dram_sharded(
    mesh_device: Any,
    workload: MatmulWorkload,
    didt_workload_iterations: int,
) -> None:
    """Sweep the DRAM-sharded matmul config (num_cores x in0_block_w) for one MLP matmul. Use -s."""
    _required_env = {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    }
    missing = [k for k, v in _required_env.items() if os.environ.get(k) != v]
    if missing:
        pytest.fail(f"Profiler env vars not set: {', '.join(missing)} — export them before pytest.")

    wl = workload
    if wl.M > TILE_SIZE:
        # The op rejects multi-tile M ("currently only support in0 tensor height of tile
        # height") — decode-only; kept for when decode (M<=32) lands.
        pytest.skip(f"DRAM-sharded matmul supports only single-tile M (<=32), got M={wl.M}")

    compute_grid = mesh_device.compute_with_storage_grid_size()
    grid = (compute_grid.x, compute_grid.y)
    full_grid_cores = compute_grid.x * compute_grid.y
    iterations = max(1, min(didt_workload_iterations, 1000))

    param_list = list(_dram_sharded_sweep_params(wl.K, grid))
    logger.info(f"[dram-sharded] {wl.workload_id} ({wl.M}x{wl.K}x{wl.N}): {len(param_list)} configs")

    print(
        "workload_id,M,K,N,num_cores,in0_block_w,duration_per_iter_ns,duration_per_iter_us,utilization_pct", flush=True
    )
    results: list[tuple[int, SweepResult]] = []
    for params in param_list:
        res = _run_dram_sharded_config(
            mesh_device, wl, params["num_cores"], params["in0_block_w"], iterations, grid, full_grid_cores
        )
        if res is None:
            continue
        results.append((params["num_cores"], res))
        print(
            f"{res.workload_id},{res.M},{res.K},{res.N},{params['num_cores']},{res.in0_block_w},"
            f"{res.duration_per_iter_ns},{res.duration_per_iter_ns / 1e3:.2f},{res.utilization_pct:.2f}",
            flush=True,
        )

    if not results:
        pytest.fail(f"No DRAM-sharded config of {len(param_list)} ran for {wl.workload_id}")

    ranked = sorted(results, key=lambda x: x[1].duration_per_iter_ns)
    best_ns = ranked[0][1].duration_per_iter_ns
    top = [(nc, r) for nc, r in ranked if r.duration_per_iter_ns <= best_ns * 1.01]
    print(f"\n# {wl.workload_id} DRAM-sharded — {len(top)} config(s) within 1% of fastest:", flush=True)
    for i, (nc, r) in enumerate(top):
        tag = "*" if i == 0 else " "
        print(
            f"    {tag} num_cores={nc} in0_block_w={r.in0_block_w} "
            f"-> {r.utilization_pct:.2f}% util, {r.duration_per_iter_ns} ns/iter ({r.duration_per_iter_ns / 1e3:.2f} us)",
            flush=True,
        )
