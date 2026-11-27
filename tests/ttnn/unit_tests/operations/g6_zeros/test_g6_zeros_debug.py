# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_zeros debug: drive the dtypes Phase 0 refuses through the same program, past validate().

The kernel writes bytes; only the page size differs per dtype. These cases check that claim on the
two production shapes outside Phase 0 SUPPORTED (p4 bfloat16, p1 bfloat8_b) over a poisoned range,
and give the profiler a handle on p1, the performance focus.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_zeros.g6_zeros_program_descriptor import Knobs, get_program

P1 = (32831, 1, 32, 256)
P4 = (1, 4, 256, 128)


def run_kernel(shape, dtype, device, knobs=None):
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    program = get_program(output, device) if knobs is None else get_program(output, device, knobs)
    return ttnn.generic_op([output], program)


def poison_range(shape, dtype, device):
    poison = torch.full(tuple(shape), 1.0, dtype=torch.bfloat16)
    dirty = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(poison, dtype=dtype, layout=ttnn.TILE_LAYOUT), dirty)
    assert int(torch.count_nonzero(ttnn.to_torch(dirty))) == dirty.logical_volume(), "the poison did not land"
    address = dirty.buffer_address()
    ttnn.deallocate(dirty)
    return address


def assert_all_positive_zero(output):
    bits = ttnn.to_torch(output).to(torch.float32).contiguous().view(torch.int32)
    nonzero = int(torch.count_nonzero(bits))
    assert nonzero == 0, f"{nonzero} of {bits.numel()} elements are not +0.0"


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(P4, ttnn.bfloat16, id="p4-bfloat16"),
        pytest.param(P1, ttnn.bfloat8_b, id="p1-bfloat8_b"),
    ],
)
def test_dirty_destination_outside_phase0(shape, dtype, device):
    ttnn.deallocate(run_kernel(shape, dtype, device))
    address = poison_range(shape, dtype, device)
    output = run_kernel(shape, dtype, device)
    assert output.buffer_address() == address
    assert_all_positive_zero(output)


@pytest.mark.parametrize(
    "knobs",
    [
        Knobs(num_worker_cores=8, placement="dram_adjacent"),
        Knobs(num_worker_cores=8, placement="row_major"),
        Knobs(num_worker_cores=110, placement="row_major"),
    ],
    ids=lambda k: f"W{k.num_worker_cores}-{k.placement}",
)
@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(P1, ttnn.bfloat8_b, id="p1-bfloat8_b"),
        pytest.param((16, 12, 128, 128), ttnn.float32, id="p2-float32"),
        pytest.param((8, 12, 128, 128), ttnn.float32, id="p3-float32"),
        pytest.param(P4, ttnn.bfloat16, id="p4-bfloat16"),
    ],
)
def test_dirty_destination_knobs(shape, dtype, knobs, device):
    ttnn.deallocate(run_kernel(shape, dtype, device, knobs))
    address = poison_range(shape, dtype, device)
    output = run_kernel(shape, dtype, device, knobs)
    assert output.buffer_address() == address
    assert_all_positive_zero(output)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(P1, ttnn.bfloat8_b, id="p1-bfloat8_b"),
        pytest.param((16, 12, 128, 128), ttnn.float32, id="p2-float32"),
        pytest.param((8, 12, 128, 128), ttnn.float32, id="p3-float32"),
        pytest.param(P4, ttnn.bfloat16, id="p4-bfloat16"),
    ],
)
def test_perf_shape(shape, dtype, device):
    """No readback: a profiling handle, one dispatch per shape."""
    ttnn.deallocate(run_kernel(shape, dtype, device))


@pytest.mark.parametrize(
    "knobs",
    [
        Knobs(num_worker_cores=8, placement="dram_adjacent"),
        Knobs(num_worker_cores=8, placement="dram_adjacent", zero_chunk_bytes_cap=8192),
        Knobs(num_worker_cores=8, placement="row_major"),
        Knobs(num_worker_cores=16, placement="row_major"),
        Knobs(num_worker_cores=110, placement="row_major"),
    ],
    ids=lambda k: f"W{k.num_worker_cores}-{k.placement}-Z{k.zero_chunk_bytes_cap}",
)
@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(P1, ttnn.bfloat8_b, id="p1-bfloat8_b"),
        pytest.param((16, 12, 128, 128), ttnn.float32, id="p2-float32"),
        pytest.param((8, 12, 128, 128), ttnn.float32, id="p3-float32"),
        pytest.param(P4, ttnn.bfloat16, id="p4-bfloat16"),
    ],
)
def test_perf_knobs(shape, dtype, knobs, device):
    """Profiling handle for the design's host-only perf lamps (core count, placement, transfer quantum)."""
    ttnn.deallocate(run_kernel(shape, dtype, device, knobs))


HOST_TIME_CONFIGS = [(8, "row_major"), (8, "dram_adjacent"), (16, "row_major"), (110, "row_major")]


@pytest.mark.parametrize(
    "shape, dtype, repeats",
    [
        pytest.param(P1, ttnn.bfloat8_b, 20, id="p1-bfloat8_b"),
        pytest.param((16, 12, 128, 128), ttnn.float32, 200, id="p2-float32"),
        pytest.param(P4, ttnn.bfloat16, 1000, id="p4-bfloat16"),
    ],
)
def test_host_time_knobs(shape, dtype, repeats, device):
    """Unprofiled host time per call (steady repetition, one sync) per core count; prints, never asserts."""
    import time

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    results = {}
    for rnd in range(2):
        for config in HOST_TIME_CONFIGS:
            program = get_program(output, device, Knobs(num_worker_cores=config[0], placement=config[1]))
            for _ in range(3):
                ttnn.generic_op([output], program)
            ttnn.synchronize_device(device)
            start = time.perf_counter()
            for _ in range(repeats):
                ttnn.generic_op([output], program)
            ttnn.synchronize_device(device)
            results.setdefault(config, []).append((time.perf_counter() - start) / repeats * 1e6)
    for config, us in results.items():
        print(f"HOST_TIME {shape} {config}: {' / '.join(f'{u:.1f}' for u in us)} us per call")


@pytest.mark.parametrize(
    "knobs",
    [
        Knobs(num_worker_cores=8, placement="dram_adjacent"),
        Knobs(num_worker_cores=8, placement="dram_adjacent", zero_chunk_bytes_cap=8192),
        Knobs(num_worker_cores=8, placement="dram_adjacent", zero_chunk_bytes_cap=4096),
        Knobs(num_worker_cores=110, placement="row_major"),
    ],
    ids=lambda k: f"W{k.num_worker_cores}-{k.placement}-Z{k.zero_chunk_bytes_cap}",
)
@pytest.mark.parametrize(
    "shape, dtype",
    [
        pytest.param(P1, ttnn.bfloat8_b, id="p1-bfloat8_b"),
        pytest.param(P4, ttnn.bfloat16, id="p4-bfloat16"),
    ],
)
def test_perf_repeat(shape, dtype, knobs, device):
    """Three dispatches per config, for configs whose difference sits near the noise band."""
    for _ in range(3):
        ttnn.deallocate(run_kernel(shape, dtype, device, knobs))
