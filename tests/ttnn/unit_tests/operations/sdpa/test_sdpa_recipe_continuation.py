# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Raw L1/DRAM continuation contract; this does not qualify ring CCL."""

import pytest
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, make_inputs, metrics, prepare, reference, run


def continuation(device, inputs, variant, chunks, split, reload_q, stage):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    descriptor = ttnn._ttnn.operations.transformer._sdpa_recipe_compute_program(
        getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")), inputs[1].dtype, grid, chunks
    )
    compute = descriptor.kernels[0]
    prefix = "tests/ttnn/unit_tests/operations/sdpa/kernels/"
    compute.kernel_source = prefix + "compute_recipe_continuation.cpp"
    compute.compile_time_args = list(compute.compile_time_args) + [split, int(reload_q), int(stage)]
    output = ttnn.allocate_tensor_on_device(
        inputs[0].shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    def args(values):
        result = ttnn.RuntimeArgs()
        result[0][0] = values
        return result

    reader_cta = [chunks, split, int(reload_q), int(stage)]
    for x in inputs:
        reader_cta += ttnn.TensorAccessorArgs(x).get_compile_time_args()
    writer = ttnn.KernelDescriptor(
        kernel_source=prefix + "writer_resident_recipe.cpp",
        core_ranges=grid,
        compile_time_args=[1] + ttnn.TensorAccessorArgs(output).get_compile_time_args(),
        runtime_args=args([output.buffer_address()]),
        config=ttnn.WriterConfigDescriptor(),
    )
    buffers = [*inputs, output]
    if stage:
        fp32 = compute.config.fp32_dest_acc_en
        pages = 45 if fp32 else 77
        backing = ttnn.allocate_tensor_on_device(
            [1, 1, pages * 32, 32], ttnn.uint32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        buffers.insert(0, backing)
        descriptor.cbs = list(descriptor.cbs) + [
            ttnn.CBDescriptor(
                total_size=4096,
                core_ranges=grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=4096)
                ],
            )
            for index in [17, 18]
        ]
        writer.kernel_source = prefix + "writer_recipe_continuation.cpp"
        writer.compile_time_args = (
            [int(fp32)]
            + ttnn.TensorAccessorArgs(output).get_compile_time_args()
            + ttnn.TensorAccessorArgs(backing).get_compile_time_args()
        )
        writer.runtime_args = args([output.buffer_address(), backing.buffer_address()])
    descriptor.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=prefix + "reader_recipe_continuation.cpp",
            core_ranges=grid,
            compile_time_args=reader_cta,
            runtime_args=args([x.buffer_address() for x in inputs]),
            config=ttnn.ReaderConfigDescriptor(),
        ),
        writer,
        compute,
    ]
    return lambda: ttnn.generic_op(buffers, descriptor)


@pytest.mark.parametrize("variant", VARIANTS[1:])
@pytest.mark.parametrize("distribution", ["normal", "uniform", "changed_max"])
@pytest.mark.parametrize("chunks,split", [(3, 1), (3, 2), (5, 3)])
@pytest.mark.parametrize("reload_q,stage", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16777216}], indirect=True)
def test_recipe_continuation(device, variant, distribution, chunks, split, reload_q, stage, record_property):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    device.enable_program_cache()
    host = make_inputs(chunks * 512, distribution)
    original = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in host]
    inputs = prepare(original, variant)
    expected = ttnn.to_torch(run(inputs, variant))
    invoke = continuation(device, inputs, variant, chunks, split, reload_q, stage)
    actual = ttnn.to_torch(invoke())
    assert digest(actual) == digest(expected)
    for key, value in metrics(actual, reference(*host)).items():
        record_property(key, value)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    traced = invoke()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert digest(ttnn.to_torch(traced)) == digest(expected)
    finally:
        ttnn.release_trace(device, trace)
    assert [digest(ttnn.to_torch(x)) for x in original] == [digest(x) for x in host]
