# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device contracts for the C/D state helpers; not end-to-end recipe qualification."""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.parametrize("device_params", [{"trace_region_size": 1048576}], indirect=True)


def _run(device, records, width, *, normalize=False, identity=False, first_column=False, bf16_output=False):
    jobs = len(records)
    host = torch.cat(records).reshape(1, 1, -1, 32)
    source = ttnn.from_torch(host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    dtype = ttnn.bfloat16 if bf16_output else ttnn.float32
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, jobs * width * 32, 32]), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(index, tiles, data_format=ttnn.float32):
        page = 2048 if data_format == ttnn.bfloat16 else 4096
        return ttnn.CBDescriptor(
            total_size=tiles * page,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page)],
        )

    def args(values):
        result = ttnn.RuntimeArgs()
        result[0][0] = values
        return result

    unary = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
    reader = ttnn.KernelDescriptor(
        kernel_source=unary + "reader_unary_interleaved_start_id.cpp",
        core_ranges=core,
        compile_time_args=ttnn.TensorAccessorArgs(source).get_compile_time_args(),
        runtime_args=args([source.buffer_address(), host.numel() // 1024, 0]),
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=unary + "writer_unary_interleaved_start_id.cpp",
        core_ranges=core,
        compile_time_args=[16] + ttnn.TensorAccessorArgs(output).get_compile_time_args(),
        runtime_args=args([output.buffer_address(), jobs * width, 0]),
        config=ttnn.WriterConfigDescriptor(),
    )
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=True
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for index in [0, 1, 2, 3]:
        modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    compute = ttnn.KernelDescriptor(
        kernel_source="tests/ttnn/unit_tests/operations/sdpa/kernels/fp32_state.cpp",
        core_ranges=core,
        compile_time_args=[int(normalize), width, int(identity), int(first_column), jobs],
        config=config,
    )
    record_tiles = width + 1 if normalize else 2 * width + 1
    program = ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],
        cbs=[cb(0, 2 * record_tiles), cb(1, 1), cb(2, width), cb(3, 1), cb(16, 2 * width, dtype)],
    )
    ttnn.generic_op([source, output], program)
    actual = ttnn.to_torch(output).float().reshape(jobs, width, 32, 32)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    ttnn.generic_op([source, output], program)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            replay = ttnn.to_torch(output).float().reshape_as(actual)
            assert torch.equal(replay, actual)
    finally:
        ttnn.release_trace(device, trace)
    assert torch.equal(ttnn.to_torch(source), host), "State helpers must not mutate their input buffers"
    return actual, (source, output)


def _record_errors(actual, expected, record_property):
    delta = actual.double() - expected.double()
    record_property("max_absolute_error", delta.abs().max().item())
    record_property("relative_l2", (delta.norm() / expected.double().norm()).item())
    record_property("exact_fraction", (actual == expected).float().mean().item())


@pytest.mark.parametrize(
    "width,identity,first_column",
    [
        (1, False, False),
        (2, False, False),
        (3, False, False),
        (2, True, False),
        (4, True, False),
        (1, False, True),
        (1, True, True),
    ],
)
def test_fp32_state_rescale_and_accumulate(device, width, identity, first_column, record_property):
    if not is_blackhole():
        pytest.skip("FP32 streaming state currently targets Blackhole")
    device.enable_program_cache()
    generator = torch.Generator().manual_seed(20260921)
    records, expected = [], []
    # More jobs than CB slots: force pointer wrap and exercise both DST halves.
    for case in range(12):
        old = torch.randn((width, 32, 32), generator=generator)
        correction = torch.rand((1, 32, 1), generator=generator).expand(1, 32, 32).clone()
        if identity or case % 4 == 0:
            correction.fill_(1)
        product = old * correction
        current = torch.randn(old.shape, generator=generator)
        if case % 4 == 1:
            current = -product + 2**-20  # expose lost low mantissa bits
        elif case % 4 == 2:
            current *= 2**-20  # tiny updates
        elif case % 4 == 3:
            old += 1024  # large common mode, still a finite FP32 state
            product = old * correction
        expected.append(product + current)
        records.append(torch.cat([old, correction, current]))
    actual, first_buffers = _run(device, records, width, identity=identity, first_column=first_column)
    cache_entries = device.num_program_cache_entries()
    # A new descriptor with new live addresses and changed data must not reuse
    # runtime addresses embedded in the cached program.
    changed_records = [r.clone() for r in records]
    for r in changed_records:
        r[:width] *= 0.5
        r[width + 1 :] *= 0.5
    changed, second_buffers = _run(device, changed_records, width, identity=identity, first_column=first_column)
    assert first_buffers[0].buffer_address() != second_buffers[0].buffer_address()
    assert device.num_program_cache_entries() == cache_entries
    expected = torch.stack(expected)
    if first_column:
        actual, expected = actual[..., :1], expected[..., :1]
        changed = changed[..., :1]
    _record_errors(actual, expected, record_property)
    # Keep the multiplication's FP32 rounding point before the L1 addition.
    # These finite, normal-range cases match that two-operation reference exactly.
    assert torch.equal(actual, expected)
    assert torch.equal(changed, expected * 0.5)


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("bf16_output", [False, True])
def test_fp32_state_normalize(device, width, bf16_output, record_property):
    if not is_blackhole():
        pytest.skip("FP32 streaming state currently targets Blackhole")
    generator = torch.Generator().manual_seed(42)
    records, expected = [], []
    for case in range(12):
        numerator = torch.randn((width, 32, 32), generator=generator) * (2.0 ** (case - 6))
        denominator = (1 + torch.rand((1, 32, 1), generator=generator)) * (2.0 ** (case - 3))
        records.append(torch.cat([numerator, denominator.expand(1, 32, 32)]))
        expected.append((numerator.double() / denominator.double()).float())
    actual, _ = _run(device, records, width, normalize=True, bf16_output=bf16_output)
    expected = torch.stack(expected)
    _record_errors(actual, expected, record_property)
    torch.testing.assert_close(actual, expected, rtol=0.004 if bf16_output else 5e-7, atol=1e-8)
