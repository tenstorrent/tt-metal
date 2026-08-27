# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact determinism checks for the optimized fused routed-expert kernel.

The expensive tensors never leave the device.  Each replay is compared with
the first output by ``ne -> max`` on device, and the per-replay markers are
folded with ``maximum``.  The green path transfers one scalar after every case
has run, rather than transferring a model-sized output after every dispatch.

The cases deliberately cover the schedules most exposed to ordering bugs:

* a short first MBLOCK, where weights are loaded but runtime M shrinks;
* a full MBLOCK followed by the smallest tail, where resident weights, deferred
  output writeback, next-X prefetch, and the full-to-short schedule transition
  meet at one boundary;
* four full MBLOCKs, which reaches steady state and wraps the depth-2/depth-3
  circular pipelines.

Only tile-aligned counts are used, and activation capacity equals count.  Thus
the complete output tensor is defined and can be compared without slicing or
copying an undefined output suffix.
"""

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs


TILE = 32
M_BLOCK_TILES = 8
EMB = 7168
HIDDEN = 2048
GRID = ttnn.CoreCoord(11, 8)
NUM_GLOBAL_EXPERTS = 256
GLOBAL_EXPERT_ID = 137
# Replays per schedule. The default finishes inside the 300 s pytest timeout so this is a gate;
# the soak it was written as (25_000_000, ~9.6 h across the three schedules) is one env var away.
ITERATIONS = int(os.environ.get("MOE_FUSED_SWIGLU_DETERMINISM_ITERS", "2000"))


# (name, token count)
_CASES = (
    ("short-first-mblock", TILE),
    ("full-plus-short-tail", (M_BLOCK_TILES + 1) * TILE),
    ("four-full-mblocks", 4 * M_BLOCK_TILES * TILE),
)


def _to_device(tensor, dtype, layout, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def _device_mismatch_marker(reference, actual):
    """Return a device scalar that is nonzero iff any element differs."""
    return ttnn.max(ttnn.ne(reference, actual, dtype=ttnn.bfloat16))


def _merge_device_markers(current, new):
    return new if current is None else ttnn.maximum(current, new)


def _marker_value(marker):
    """The only device-to-host transfer on the passing path."""
    return float(ttnn.to_torch(ttnn.from_device(marker)).max())


def _comparable(output, output_tensor):
    # Elementwise comparison does not consume BFP8 directly.  Typecasting on
    # device is a deterministic one-to-one expansion of the packed values.
    return ttnn.typecast(output, ttnn.bfloat16, output_tensor=output_tensor)


def _run_case(device, weights, counts, expert_ids, compute_config, count):
    torch.manual_seed(20260821 + count)
    activations = _to_device(
        torch.randn((1, 1, count, EMB), dtype=torch.bfloat16),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
    )

    # Two fixed output buffers avoid allocating or freeing a model-sized tensor
    # on every replay.  Queue ordering keeps each comparison ahead of the next
    # overwrite of scratch_output.
    reference_output = ttnn.empty(
        activations.shape,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scratch_output = ttnn.empty(
        activations.shape,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    reference_comparable = ttnn.empty(
        activations.shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scratch_comparable = ttnn.empty(
        activations.shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    op = ttnn.experimental.deepseek_prefill.moe_fused_swiglu
    common_args = (
        activations,
        [weights[0]],
        [weights[1]],
        [weights[2]],
        counts,
        expert_ids,
    )
    common_kwargs = {
        "input_m_tiles": count // TILE,
        "compute_kernel_config": compute_config,
        "core_grid": GRID,
    }

    op(*common_args, output=reference_output, **common_kwargs)
    reference = _comparable(reference_output, reference_comparable)
    case_marker = None
    for _ in range(ITERATIONS - 1):
        op(*common_args, output=scratch_output, **common_kwargs)
        current = _comparable(scratch_output, scratch_comparable)
        case_marker = _merge_device_markers(case_marker, _device_mismatch_marker(reference, current))

    return case_marker


@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
def test_moe_fused_swiglu_is_bit_exact_across_cached_replays(device):
    """Run every overlap-sensitive M schedule; read back one scalar on success."""
    available_grid = device.compute_with_storage_grid_size()
    if GRID.x > available_grid.x or GRID.y > available_grid.y:
        pytest.skip(
            f"requested {GRID.y}x{GRID.x} grid exceeds available " f"{available_grid.y}x{available_grid.x} grid"
        )

    torch.manual_seed(20260821)
    gate_up_memory_config, down_memory_config = weight_memory_configs(device, EMB, HIDDEN, core_grid=GRID)
    host_weights = (
        torch.randn((EMB, HIDDEN), dtype=torch.bfloat16) * 2.0e-2,
        torch.randn((EMB, HIDDEN), dtype=torch.bfloat16) * 2.0e-2,
        torch.randn((HIDDEN, EMB), dtype=torch.bfloat16) * 2.0e-2,
    )
    weights = tuple(
        _to_device(host, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device, memory_config)
        for host, memory_config in zip(
            host_weights,
            (gate_up_memory_config, gate_up_memory_config, down_memory_config),
        )
    )
    expert_ids = _to_device(
        torch.tensor([GLOBAL_EXPERT_ID], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
    )
    count_tensors = {}
    for _, count in _CASES:
        if count not in count_tensors:
            host_counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
            host_counts[GLOBAL_EXPERT_ID] = count
            count_tensors[count] = _to_device(
                host_counts,
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                device,
            )

    compute_config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )

    overall_marker = None
    case_markers = []
    for name, count in _CASES:
        marker = _run_case(
            device,
            weights,
            count_tensors[count],
            expert_ids,
            compute_config,
            count,
        )
        case_markers.append((name, marker))
        overall_marker = _merge_device_markers(overall_marker, marker)

    # One scalar crosses PCIe on the green path, after every case's kernel runs and
    # all full-output comparisons have completed on device.
    if _marker_value(overall_marker) != 0.0:
        # Failure-only readbacks identify the offending schedule without making
        # the normal determinism test pay a per-case synchronization cost.
        failed_cases = [name for name, marker in case_markers if _marker_value(marker) != 0.0]
        pytest.fail(f"moe_fused_swiglu was not bit-exact across {ITERATIONS} cached replays in: {failed_cases}")
