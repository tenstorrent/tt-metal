# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
1D-decode matmul block-size sweep for HunyuanImage-3.0 (Teja's family C).

SINGLE-DEVICE, FABRIC-FREE. Sweeps the decode-path matmuls (small M=32, batch=1
padded to a tile) over the classic 1D systolic
`ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig` family with
`mcast_in0=True, fuse_batch=True` and collects the BEST config per shape.

This is a tuning-table artifact for the (parked) decode path — completeness, not
an image win. No mesh, no fabric, no collectives.

Decode shapes (per-device, TP=8, M=32) — confirmed against the model:
  qkv            (M=32, K=4096, N=768)    act bf16, w bf16   (image3_decoder_layer qkv_proj)
  o_proj         (32,   512,   4096)      act bf16, w bf16   (o_proj, K=tp_num_heads*hd=512)
  expert_gate_up (32,   4096,  12288)     act bf16, w bf4    (mo_e merged gate|up, EP-merged)
  expert_down    (32,   6144,  4096)      act bf16, w bf4    (mo_e merged down)
  shared_gate_up (32,   4096,  768)       act bf16, w bf16   (2*si/32, si/32=384 -> 768)
  shared_down    (32,   384,   4096)      act bf16, w bf16   (si/32=384)
(K,N pairs cross-checked against mo_e._MMCFG_WINNERS.)

Requires Tracy for device-level profiling. Export before running pytest:
  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
  pytest models/demos/vision/generative/hunyuanimage_3_0/tests/perf/sweep_1d_decode.py -v -s \
      --didt-workload-iterations 30 -o timeout=0

Winners CSV path: $HUNYUAN_DECODE1D_WINNERS_CSV (default $TT_METAL_HOME/generated/decode1d_winners.csv).
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Iterator

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, skip_for_wormhole_b0
from tests.didt.op_test_base import OpParameter, OpTestBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_SIZE = 32
TILE_H = TILE_W = 32
GRID_MAX = (12, 10)  # Blackhole compute-with-storage grid (x, y) = 120 cores
DECODE_M = 32  # batch=1 padded to a tile

# Cycles per tile per math fidelity (Tensix matmul on 2 tiles). Decode runs LoFi.
CYCLES_PER_TILE = {
    ttnn.MathFidelity.LoFi: 16,
    ttnn.MathFidelity.HiFi2: 32,
    ttnn.MathFidelity.HiFi3: 48,
    ttnn.MathFidelity.HiFi4: 64,
}

# Core counts to sweep the 1D grid over (num_cores = grid_x * grid_y). Each must be
# expressible as x*y with x<=12, y<=10. mcast_in0=True splits N across cores, so
# num_cores is the dominant decode lever; per_core_N = ceil(N_tiles / num_cores).
CANDIDATE_CORES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 120]


# Device frequency (MHz): Blackhole 1350, Wormhole 1000.
def _device_freq_mhz() -> float:
    return 1350.0 if is_blackhole() else 1000.0


def compute_utilization_pct(M, K, N, duration_ns, num_cores, math_fidelity) -> float:
    """Utilization % vs theoretical peak (ideal cycles / actual cycles).
    ideal_cycles = M*K*N / (tile_h*tile_w*32) * cycle_per_tile / num_cores."""
    cycle_per_tile = CYCLES_PER_TILE.get(math_fidelity, CYCLES_PER_TILE[ttnn.MathFidelity.LoFi])
    ideal_cycles = (M * K * N) / (TILE_H * TILE_W * 32) * cycle_per_tile / max(1, num_cores)
    inference_cycles = (duration_ns * 1e-9) * (_device_freq_mhz() * 1e6)
    if inference_cycles <= 0:
        return 0.0
    return (ideal_cycles / inference_cycles) * 100.0


# ---------------------------------------------------------------------------
# Divisors / grid mapping / parameter sweep generation
# ---------------------------------------------------------------------------


def _divisors_up_to(n: int, max_val: int) -> list[int]:
    """Sorted divisors of n that are <= max_val."""
    out = []
    for d in range(1, min(int(math.isqrt(n)) + 1, max_val + 1)):
        if n % d == 0:
            if d <= max_val:
                out.append(d)
            comp = n // d
            if comp != d and comp <= max_val:
                out.append(comp)
    return sorted(set(out))


def _cores_to_grid(c: int, grid_max=GRID_MAX) -> tuple[int, int] | None:
    """Map a target core count to an (x, y) grid within grid_max (widest x). None if impossible."""
    gx_max, gy_max = grid_max
    for x in range(min(c, gx_max), 0, -1):
        if c % x == 0 and (c // x) <= gy_max:
            return (x, c // x)
    return None


def _sweep_1d_params(K: int, N: int) -> Iterator[dict[str, Any]]:
    """Yield valid 1D-config param dicts for (M=DECODE_M, K, N) with mcast_in0=True.

    per_core_M = M_tiles (all cores hold the full M; =1 for decode). N is split across
    the grid: per_core_N = ceil(N_tiles / num_cores). in0_block_w sweeps divisors of
    K_tiles (K is NOT sharded under mcast_in0). out_subblock_h|per_core_M,
    out_subblock_w|per_core_N, product <= 8."""
    M_tiles = math.ceil(DECODE_M / TILE_SIZE)
    K_tiles = math.ceil(K / TILE_SIZE)
    N_tiles = math.ceil(N / TILE_SIZE)
    per_core_M = M_tiles

    in0_block_w_candidates = _divisors_up_to(K_tiles, 32) or [1]
    out_subblock_h_candidates = _divisors_up_to(per_core_M, 8) or [1]

    for c in CANDIDATE_CORES:
        if c > N_tiles:  # avoid idle cores (matches the canonical grid-shrink rule)
            continue
        grid = _cores_to_grid(c)
        if grid is None:
            continue
        num_cores = grid[0] * grid[1]
        per_core_N = math.ceil(N_tiles / num_cores)
        out_subblock_w_candidates = _divisors_up_to(per_core_N, 8) or [1]
        for in0_block_w in in0_block_w_candidates:
            for out_subblock_h in out_subblock_h_candidates:
                for out_subblock_w in out_subblock_w_candidates:
                    if out_subblock_h * out_subblock_w > 8:
                        continue
                    if per_core_M % out_subblock_h != 0 or per_core_N % out_subblock_w != 0:
                        continue
                    yield {
                        "grid_size": grid,
                        "num_cores": num_cores,
                        "in0_block_w": in0_block_w,
                        "out_subblock_h": out_subblock_h,
                        "out_subblock_w": out_subblock_w,
                        "per_core_M": per_core_M,
                        "per_core_N": per_core_N,
                    }


def _make_program_config(params: dict[str, Any]) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=params["grid_size"],
        in0_block_w=params["in0_block_w"],
        out_subblock_h=params["out_subblock_h"],
        out_subblock_w=params["out_subblock_w"],
        per_core_M=params["per_core_M"],
        per_core_N=params["per_core_N"],
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------


@dataclass
class MatmulWorkload:
    workload_id: str
    M: int
    K: int
    N: int
    in1_dtype: Any
    math_fidelity: Any


# Decode runs LoFi (see image3_decoder_layer._DECODE_MM_CFG). bf4 weights for the two
# routed-expert matmuls (DRAM-bw bound), bf16 else.
_LOFI = ttnn.MathFidelity.LoFi


def _decode_workloads() -> list[MatmulWorkload]:
    bf16 = ttnn.DataType.BFLOAT16
    bf4 = ttnn.DataType.BFLOAT4_B
    return [
        MatmulWorkload("qkv", DECODE_M, 4096, 768, bf16, _LOFI),
        MatmulWorkload("o_proj", DECODE_M, 512, 4096, bf16, _LOFI),
        MatmulWorkload("expert_gate_up", DECODE_M, 4096, 12288, bf4, _LOFI),
        MatmulWorkload("expert_down", DECODE_M, 6144, 4096, bf4, _LOFI),
        MatmulWorkload("shared_gate_up", DECODE_M, 4096, 768, bf16, _LOFI),
        MatmulWorkload("shared_down", DECODE_M, 384, 4096, bf16, _LOFI),
    ]


# ---------------------------------------------------------------------------
# Run one (workload, program_config)
# ---------------------------------------------------------------------------


def _short_error(e: BaseException, max_len: int = 120) -> str:
    msg = str(e).strip().split("\n")[0] if str(e).strip() else ""
    out = f"{type(e).__name__}: {msg}" if msg else type(e).__name__
    return out[:max_len] + ("..." if len(out) > max_len else "")


def _get_tracy_timing_and_cores(device_id: int) -> tuple[int | None, int | None]:
    """(duration_ns, core_count) from Tracy program perf data after ReadDeviceProfiler."""
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
    in1_dtype: str
    grid_x: int
    grid_y: int
    num_cores: int
    in0_block_w: int
    per_core_M: int
    per_core_N: int
    out_subblock_h: int
    out_subblock_w: int
    duration_per_iter_ns: int
    utilization_pct: float
    core_count: int

    def to_csv_row(self) -> str:
        return (
            f"{self.workload_id},{self.M},{self.K},{self.N},{self.in1_dtype},"
            f"{self.grid_x},{self.grid_y},{self.num_cores},{self.in0_block_w},"
            f"{self.per_core_M},{self.per_core_N},{self.out_subblock_h},{self.out_subblock_w},"
            f"{self.duration_per_iter_ns},{self.duration_per_iter_ns / 1e3:.3f},"
            f"{self.utilization_pct:.3f},{self.core_count}"
        )


CSV_HEADER = (
    "workload_id,M,K,N,in1_dtype,grid_x,grid_y,num_cores,in0_block_w,"
    "per_core_M,per_core_N,out_subblock_h,out_subblock_w,"
    "duration_per_iter_ns,duration_per_iter_us,utilization_pct,core_count"
)


def _run_single_config(
    mesh_device: Any,
    wl: MatmulWorkload,
    params: dict[str, Any],
    iterations: int,
) -> SweepResult | None:
    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    ComputeCfg = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    # Match the decode compute config (image3_decoder_layer._DECODE_MM_CFG): LoFi, approx.
    compute_config = ComputeCfg(
        math_fidelity=wl.math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    program_config = _make_program_config(params)
    in0_shape = [1, 1, wl.M, wl.K]
    in1_shape = [1, 1, wl.K, wl.N]

    try:
        activation = OpParameter(in0_shape, ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT, dram)
        arguments = [OpParameter(in1_shape, wl.in1_dtype, ttnn.TILE_LAYOUT, dram)]
        test = OpTestBase(
            mesh_device,
            activation=activation,
            arguments=arguments,
            out_mem_config=dram,
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
                B, test.arguments[0].dtype, test.arguments[0].layout, test.arguments[0].mem_config, 0
            )
        ]
        test.activations = test.convert_activations_to_memory_config(a_t)

        for _ in range(iterations):
            out = test.run_device_operation()
            for device_idx in test.device_ids:
                ttnn.device.synchronize_device(test.get_device(device_idx))
            out.deallocate(True)

        ttnn.ReadDeviceProfiler(mesh_device)
        tracy_ns, tracy_cores = _get_tracy_timing_and_cores(test.device_ids[0])
        if tracy_ns is None:
            pytest.fail("Tracy device profiler data unavailable. Export the Tracy env vars before pytest.")
        num_cores_used = tracy_cores if tracy_cores is not None else params["num_cores"]
        gx_full, gy_full = GRID_MAX
        util = compute_utilization_pct(wl.M, wl.K, wl.N, tracy_ns, gx_full * gy_full, wl.math_fidelity)

        test.deallocate_activations()
        test.inputs[0].deallocate(True)

        return SweepResult(
            workload_id=wl.workload_id,
            M=wl.M,
            K=wl.K,
            N=wl.N,
            in1_dtype=str(wl.in1_dtype).split(".")[-1],
            grid_x=params["grid_size"][0],
            grid_y=params["grid_size"][1],
            num_cores=params["num_cores"],
            in0_block_w=params["in0_block_w"],
            per_core_M=params["per_core_M"],
            per_core_N=params["per_core_N"],
            out_subblock_h=params["out_subblock_h"],
            out_subblock_w=params["out_subblock_w"],
            duration_per_iter_ns=int(tracy_ns),
            utilization_pct=util,
            core_count=num_cores_used,
        )
    except Exception as e:
        print(
            f"# SKIP {wl.workload_id} ({wl.M}x{wl.K}x{wl.N}) grid={params['grid_size']} "
            f"in0_bw={params['in0_block_w']} sub={params['out_subblock_h']}x{params['out_subblock_w']} "
            f"pcN={params['per_core_N']}: {_short_error(e)}",
            flush=True,
        )
        return None


# ---------------------------------------------------------------------------
# Sweep entrypoint
# ---------------------------------------------------------------------------


@skip_for_wormhole_b0("Grid 12x10 requires Blackhole")
@pytest.mark.parametrize("mesh_device", [pytest.param(1, id="1chips")], indirect=["mesh_device"])
def test_sweep_1d_decode(mesh_device: Any, didt_workload_iterations) -> None:
    """Single-device 1D decode-matmul block-size sweep. Use -s for output.

    Requires Tracy env vars exported before launching pytest (see module docstring)."""
    _required_env = {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    }
    missing = [k for k, v in _required_env.items() if os.environ.get(k) != v]
    if missing:
        pytest.fail(
            f"Tracy env vars not set: {', '.join(missing)}. Export before pytest:\n"
            "  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 "
            "TT_METAL_PROFILER_CPP_POST_PROCESS=1"
        )

    iters_opt = didt_workload_iterations if didt_workload_iterations is not None else 30
    iterations = max(1, min(int(iters_opt), 1000))
    workloads = _decode_workloads()

    print(CSV_HEADER, flush=True)
    all_results: list[SweepResult] = []

    for wl in workloads:
        param_list = list(_sweep_1d_params(wl.K, wl.N))
        logger.info(f"Workload {wl.workload_id} ({wl.M}x{wl.K}x{wl.N}): {len(param_list)} configs")
        for params in param_list:
            res = _run_single_config(mesh_device, wl, params, iterations)
            if res is None:
                continue
            all_results.append(res)
            print(res.to_csv_row(), flush=True)

    # ---- best-per-shape winners ----
    by_workload: dict[str, list[SweepResult]] = {}
    for r in all_results:
        by_workload.setdefault(r.workload_id, []).append(r)

    winners_csv = os.environ.get(
        "HUNYUAN_DECODE1D_WINNERS_CSV",
        os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated", "decode1d_winners.csv"),
    )
    print("\n# ==== BEST config per shape (min device_kernel_duration_ns) ====", flush=True)
    winner_rows: list[SweepResult] = []
    for wid in sorted(by_workload):
        ranked = sorted(by_workload[wid], key=lambda x: x.duration_per_iter_ns)
        best = ranked[0]
        winner_rows.append(best)
        # spread: best vs worst (and vs 2nd) to quantify how much the block size matters
        worst = ranked[-1]
        spread = worst.duration_per_iter_ns / max(1, best.duration_per_iter_ns)
        print(
            f"\n  {wid} ({best.M}x{best.K}x{best.N}) [{best.in1_dtype}] — {len(ranked)} valid configs, "
            f"best->worst spread {spread:.2f}x",
            flush=True,
        )
        print(
            f"    * grid={best.grid_x}x{best.grid_y} ({best.num_cores} cores) "
            f"in0_block_w={best.in0_block_w} per_core_M={best.per_core_M} per_core_N={best.per_core_N} "
            f"out_subblock={best.out_subblock_h}x{best.out_subblock_w} "
            f"-> {best.duration_per_iter_ns} ns ({best.duration_per_iter_ns/1e3:.3f} us), "
            f"{best.utilization_pct:.2f}% util, cores_used={best.core_count}",
            flush=True,
        )

    try:
        os.makedirs(os.path.dirname(winners_csv), exist_ok=True)
        with open(winners_csv, "w") as f:
            f.write(CSV_HEADER + "\n")
            for r in winner_rows:
                f.write(r.to_csv_row() + "\n")
        print(f"\n# Winners CSV written: {winners_csv} ({len(winner_rows)} shapes)", flush=True)
    except Exception as e:
        print(f"# WARN could not write winners CSV {winners_csv}: {_short_error(e)}", flush=True)

    assert all_results, "No configs measured successfully — check device/Tracy setup."
