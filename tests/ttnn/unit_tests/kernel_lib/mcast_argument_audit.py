# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicitly collected, matched before/after argument and timing measurements.

Run through scripts/run_safe_pytest.sh, optionally with --profile, and retain
stdout. These measurements are not performance assertions. Descriptor snapshots
include complete positional/named CT values and per-core RT, not dispatch bytes.
"""

import json
import statistics
import time

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import inspect_mcast_ct


def _kernel_snapshot(kernel):
    runtime = [
        {"core": [x, y], "words": list(words)}
        for rectangle in (kernel.core_ranges.ranges() if len(kernel.runtime_args) else [])
        for x in range(rectangle.start.x, rectangle.end.x + 1)
        for y in range(rectangle.start.y, rectangle.end.y + 1)
        for words in [kernel.runtime_args[x][y]]
    ]
    return {
        "source": kernel.kernel_source,
        "placement": str(kernel.core_ranges),
        "processor": str(getattr(kernel.config, "processor", "compute")),
        "noc": str(getattr(kernel.config, "noc", "compute")),
        "defines": list(kernel.defines),
        "positional_ct": list(kernel.compile_time_args),
        "named_ct": list(kernel.named_compile_time_args),
        "runtime": runtime,
        "aggregate_rt_words": sum(len(item["words"]) for item in runtime),
    }


@pytest.mark.parametrize("kind", ["fixed_1d", "rotating_1d", "fixed_2d", "rotating_2d"])
def test_matmul_argument_measurement(device, kind):
    torch.manual_seed(7)
    rotating = kind.startswith("rotating")
    two_dimensional = kind.endswith("2d")
    m, k, n = (256, 1024, 256) if two_dimensional else (256, 1024, 1024)
    grid = (4, 4) if two_dimensional else (8, 1)
    a_host = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    b_host = torch.randn((1, 1, k, n), dtype=torch.bfloat16)
    a_memory = ttnn.DRAM_MEMORY_CONFIG
    if rotating:
        a_memory = ttnn.create_sharded_memory_config(
            (m, k),
            core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]),
            strategy=ttnn.ShardStrategy.BLOCK if two_dimensional else ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    a = ttnn.from_torch(a_host, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_memory)
    b = ttnn.from_torch(b_host, layout=ttnn.TILE_LAYOUT, device=device)
    common = dict(
        compute_with_storage_grid_size=grid,
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m // 32 // (grid[1] if two_dimensional else 1),
        per_core_N=n // 32 // grid[0],
        fused_activation=None,
        fuse_batch=True,
    )
    config = (
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**common, transpose_mcast=False)
        if two_dimensional
        else ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(**common, mcast_in0=True)
    )
    config.allowed_worker_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))]
    )
    params = ttnn.MatmulParams()
    params.program_config = config
    params.output_dtype = ttnn.bfloat16
    params.output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    attributes = ttnn.create_matmul_attributes(a, b, params, [])
    inputs = ttnn.MatmulInputs()
    inputs.input_tensors = [a, b]
    inputs.optional_input_tensors = [None]
    output_spec = ttnn.MatmulDeviceOperation.compute_output_specs(attributes, inputs)[0]
    output = ttnn.allocate_tensor_on_device(output_spec, device)
    factory = ttnn.matmul_select_program_factory(attributes, inputs)

    construction_ns = []
    for iteration in range(25):
        start = time.perf_counter_ns()
        descriptor = factory.create_descriptor(attributes, inputs, [output])
        elapsed = time.perf_counter_ns() - start
        if iteration >= 5:
            construction_ns.append(elapsed)

    for kernel in descriptor.kernels:
        named = dict(kernel.named_compile_time_args)
        for prefix in ("in0_mcast", "in1_mcast"):
            if prefix + "_ct_offset" not in named:
                continue
            control = kernel.compile_time_args[named[prefix + "_ct_offset"]]
            if control == 0:
                continue
            if control & 15 == 3:
                metadata = inspect_mcast_ct(kernel, prefix)
                # Count only multicast CT entries, including the two attachment offsets.
                expected = 11 if rotating and prefix == "in0_mcast" else 6 if metadata["roles"] == 1 else 5
                assert metadata["words"] + 2 == expected

    if kind == "fixed_1d":
        for kernel in descriptor.kernels:
            named = dict(kernel.named_compile_time_args)
            if "in0_mcast_ct_offset" not in named:
                continue
            ct_base = named["in0_mcast_ct_offset"]
            if kernel.compile_time_args[ct_base] == 1:
                continue  # The same measurement script also captures the v1 baseline.
            if kernel.compile_time_args[ct_base] == 2:
                roles = kernel.compile_time_args[ct_base + 11]  # Original compact-RT baseline.
            else:
                roles = inspect_mcast_ct(kernel, "in0_mcast")["roles"]
            assert roles in (1, 2)
            expected_helper = 4 if roles == 1 else 2
            expected_total = 8 if roles == 1 else 2
            for runtime in _kernel_snapshot(kernel)["runtime"]:
                assert len(runtime["words"]) == expected_total
                assert len(runtime["words"]) - named["in0_mcast_rt_offset"] == expected_helper

    start = time.perf_counter_ns()
    result = ttnn.matmul(a, b, program_config=config, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    first_call_ns = time.perf_counter_ns() - start
    assert_with_pcc(a_host @ b_host, ttnn.to_torch(result), 0.999)
    cache_hit_ns = []
    for iteration in range(25):
        start = time.perf_counter_ns()
        result = ttnn.matmul(a, b, program_config=config, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
        if iteration >= 5:
            cache_hit_ns.append(time.perf_counter_ns() - start)

    print(
        "MCAST_ARGUMENT_AUDIT "
        + json.dumps(
            {
                "case": kind,
                "shape_mkn": [m, k, n],
                "grid": grid,
                "input_sharded": rotating,
                "dtype": "bfloat16",
                "output": "interleaved_dram",
                "fusion": False,
                "input_accessor_ct": ttnn.TensorAccessorArgs(a).get_compile_time_args(),
                "construction_ns": construction_ns,
                "construction_median_ns": statistics.median(construction_ns),
                "first_call_including_sync_ns": first_call_ns,
                "cache_hit_including_sync_ns": cache_hit_ns,
                "cache_hit_including_sync_median_ns": statistics.median(cache_hit_ns),
                "kernels": [_kernel_snapshot(kernel) for kernel in descriptor.kernels],
            },
            sort_keys=True,
        )
    )
