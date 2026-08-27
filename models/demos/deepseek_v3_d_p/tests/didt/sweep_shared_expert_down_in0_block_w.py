# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sweep in0_block_w for the shared expert's down projection, reporting time against PCC.

Only the down projection is swept: gate and up sit at the HiFi2 FLOP ceiling (99.7-100.4% of
2.0 TFLOPs/core), so no tiling change moves them, and their K is the unsharded emb_dim -- a
constant 224 tiles for every model here. down's K is hidden_dim/TP, the one reduction dim TP
shards, so it is the only in0_block_w in the op with anything to choose.

The matmul is exercised standalone rather than through TtSharedExpert.forward: the surrounding
ReduceScatter is an order of magnitude larger than the effect under measurement and would bury it.

Requires Tracy. Export before launching pytest:
  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
  pytest models/demos/deepseek_v3_d_p/tests/didt/sweep_shared_expert_down_in0_block_w.py \
      -s --didt-workload-iterations 20 --timeout=3600
"""

import math
import os
import statistics
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, skip_for_wormhole_b0
from models.demos.deepseek_v3_d_p.tests.didt.sweep_deepseek_v3_matmul_tune import _get_tracy_timing_and_cores
from models.demos.deepseek_v3_d_p.tests.pcc.test_shared_expert import shared_expert_sub_device
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    MAX_IN0_BLOCK_W,
    _in0_block_w,
    _out_subblock,
)

TILE = 32
EMB_DIM = 7168
SEQ_LEN_PER_CHIP = 640
TP_FACTOR = 4

# (label, hidden_dim) for every shared-expert shape the model family ships.
DOWN_SHAPES = [
    ("base-2048", 2048),
    ("v4pro-3072", 3072),
    ("k3-6144", 6144),
]

# Rotated between rounds so a monotone drift in device state cannot be read as a config effect.
ROUNDS = 3

# The gate test_ttnn_moe's shared_output must clear. A candidate below this is unshippable
# regardless of how fast it is.
PCC_FLOOR = 0.999


@dataclass
class Candidate:
    label: str
    k_tiles: int
    in0_block_w: int
    duration_ns: int
    core_count: int
    pcc: float

    @property
    def is_shipped(self) -> bool:
        return self.in0_block_w == _in0_block_w(self.k_tiles)


def _divisors(n: int) -> list[int]:
    out: set[int] = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            out.add(i)
            out.add(n // i)
    return sorted(out)


def _down_program_config(
    grid: ttnn.CoreCoord, m_tiles: int, k_tiles: int, n_tiles: int, in0_block_w: int
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """The shipped down config with in0_block_w overridden.

    per_core_M/N and out_subblock are held at their derived values: an exhaustive (per_core_M,
    per_core_N) sweep on 11x9 already established 77 cores as the maximum for this shape, so
    varying them here would only reintroduce a settled axis and confound the one under test.
    """
    per_core_M = -(-m_tiles // grid.y)
    per_core_N = -(-n_tiles // grid.x)
    subblock_h, subblock_w = _out_subblock(per_core_M, per_core_N, deep_k=k_tiles >= 2 * n_tiles)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=None,
    )


def _local_tensor(mesh_device, torch_tensor: torch.Tensor, dtype: ttnn.DataType) -> ttnn.Tensor:
    """Push a per-device tensor, replicated: every device runs the identical local matmul."""
    return ttnn.from_torch(
        torch_tensor,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _readback(tensor: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _run_candidate(
    mesh_device,
    subdevice_id: ttnn.SubDeviceId,
    a_t: ttnn.Tensor,
    b_t: ttnn.Tensor,
    program_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig,
    iterations: int,
) -> tuple[int, int, torch.Tensor]:
    out = None
    for _ in range(iterations):
        if out is not None:
            out.deallocate(True)
        out = ttnn.matmul(
            a_t,
            b_t,
            program_config=program_config,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            sub_device_id=subdevice_id,
        )
        ttnn.synchronize_device(mesh_device, sub_device_ids=[subdevice_id])

    ttnn.ReadDeviceProfiler(mesh_device)
    duration_ns, core_count = _get_tracy_timing_and_cores(mesh_device.get_device_ids()[0])
    if duration_ns is None:
        pytest.fail(
            "Tracy device profiler data unavailable. Export TT_METAL_DEVICE_PROFILER=1 "
            "TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 before pytest."
        )
    result = _readback(out)
    out.deallocate(True)
    return duration_ns, core_count or 0, result


@skip_for_wormhole_b0("Sub-device grid and per-core extents are calibrated for Blackhole's 11x10")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [pytest.param(1, id="1chips")], indirect=["mesh_device"])
def test_sweep_shared_expert_down_in0_block_w(mesh_device, didt_workload_iterations: int) -> None:
    required_env = {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    }
    missing = [k for k, v in required_env.items() if os.environ.get(k) != v]
    if missing:
        pytest.fail(f"Tracy env vars not set: {', '.join(missing)}. See the module docstring.")

    iterations = max(1, min(didt_workload_iterations, 200))
    m_tiles = SEQ_LEN_PER_CHIP // TILE
    n_tiles = EMB_DIM // TILE

    print("label,k_tiles,in0_block_w,us,cores,pcc,shipped", flush=True)
    results: dict[str, list[Candidate]] = {}

    with shared_expert_sub_device(mesh_device) as (subdevice_id, subdevice_cores):
        grid = subdevice_cores.bounding_box().grid_size()
        logger.info(f"Sub-device grid {grid.x}x{grid.y}, m_tiles={m_tiles}, n_tiles={n_tiles}")

        for label, hidden_dim in DOWN_SHAPES:
            down_k = hidden_dim // TP_FACTOR
            k_tiles = down_k // TILE
            candidates = _divisors(k_tiles)

            torch_a = torch.randn(1, SEQ_LEN_PER_CHIP, down_k, dtype=torch.float32)
            torch_b = torch.randn(1, down_k, EMB_DIM, dtype=torch.float32)
            a_t = _local_tensor(mesh_device, torch_a, ttnn.bfloat16)
            b_t = _local_tensor(mesh_device, torch_b, ttnn.bfloat8_b)

            # Reference from the dequantized device inputs, so the reported PCC is the matmul's own
            # accumulation error and carries none of bfloat8_b's quantization error.
            reference = (_readback(a_t).to(torch.float32) @ _readback(b_t).to(torch.float32)).squeeze()

            per_candidate: dict[int, list[int]] = {w: [] for w in candidates}
            pccs: dict[int, float] = {}
            cores: dict[int, int] = {}
            # A candidate that fails once (L1 overflow at the wide end) is out for every later
            # round; retrying it would only re-raise, and dropping its earlier samples would
            # discard good data.
            failed: set[int] = set()

            for round_idx in range(ROUNDS):
                rotated = candidates[round_idx % len(candidates) :] + candidates[: round_idx % len(candidates)]
                for in0_block_w in rotated:
                    if in0_block_w in failed:
                        continue
                    pc = _down_program_config(grid, m_tiles, k_tiles, n_tiles, in0_block_w)
                    try:
                        duration_ns, core_count, out_torch = _run_candidate(
                            mesh_device, subdevice_id, a_t, b_t, pc, iterations
                        )
                    except Exception as e:
                        logger.warning(f"{label} in0_block_w={in0_block_w} skipped: {type(e).__name__}: {e}")
                        failed.add(in0_block_w)
                        continue
                    per_candidate[in0_block_w].append(duration_ns)
                    cores[in0_block_w] = core_count
                    if in0_block_w not in pccs:
                        _, pccs[in0_block_w] = comp_pcc(reference, out_torch.squeeze(), pcc=PCC_FLOOR)

            shape_results = [
                Candidate(
                    label=label,
                    k_tiles=k_tiles,
                    in0_block_w=w,
                    duration_ns=int(statistics.median(samples)),
                    core_count=cores.get(w, 0),
                    pcc=pccs.get(w, float("nan")),
                )
                for w, samples in per_candidate.items()
                if samples and w not in failed
            ]
            results[label] = sorted(shape_results, key=lambda c: c.in0_block_w)

            for c in results[label]:
                print(
                    f"{c.label},{c.k_tiles},{c.in0_block_w},{c.duration_ns / 1e3:.2f},"
                    f"{c.core_count},{c.pcc:.6f},{int(c.is_shipped)}",
                    flush=True,
                )

            ttnn.deallocate(a_t)
            ttnn.deallocate(b_t)

    print(f"\n# Verdict (PCC floor {PCC_FLOOR}, MAX_IN0_BLOCK_W currently {MAX_IN0_BLOCK_W})", flush=True)
    for label, shape_results in results.items():
        if not shape_results:
            continue
        shipped = next((c for c in shape_results if c.is_shipped), None)
        viable = [c for c in shape_results if c.pcc >= PCC_FLOOR]
        fastest = min(viable, key=lambda c: c.duration_ns) if viable else None
        line = f"# {label} K={shape_results[0].k_tiles}t: shipped w={shipped.in0_block_w} @ {shipped.duration_ns / 1e3:.2f}us pcc={shipped.pcc:.6f}"
        if fastest is None:
            line += " | NO candidate clears the PCC floor"
        elif shipped is not None and fastest.in0_block_w != shipped.in0_block_w:
            gain = 100.0 * (shipped.duration_ns - fastest.duration_ns) / shipped.duration_ns
            line += f" | best w={fastest.in0_block_w} @ {fastest.duration_ns / 1e3:.2f}us pcc={fastest.pcc:.6f} ({gain:+.1f}%)"
        else:
            line += " | shipped is optimal"
        print(line, flush=True)
