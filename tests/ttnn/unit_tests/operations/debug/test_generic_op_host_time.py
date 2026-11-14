# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-call host time of ttnn.experimental.generic_op on a ProgramSpec, against ttnn.generic_op on a ProgramDescriptor.

Both forms run output = exp(input) with one tile per core, on 1 core, 64 cores and the full grid: the
ProgramDescriptor form with TT-NN's unary kernels, the ProgramSpec form with their Metal 2.0 forks. Only cache hits are
timed, and the device is not waited for inside the loop, so a time is the host's work for one call. Each form is timed
twice: with the same arguments on every call (the op alone), and with the arguments built again for every call, as a
caller does when the values change. The ProgramDescriptor holds its runtime arguments, so that form rebuilds the
whole descriptor; the ProgramSpec form rebuilds only the ProgramRunArgs.

The times are logged, not checked against a limit. In slow dispatch a call waits for its program to finish, so the
test runs only with fast dispatch.
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp

E = ttnn.experimental
UNARY_KERNELS = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels"
TILE_BYTES = 2 * 32 * 32  # one bfloat16 tile
EXP_DEFINES = {"SFPU_OP_EXP_INCLUDE": "1", "SFPU_OP_CHAIN_0": "exp_tile_init(); exp_tile(0);"}
WARMUP_CALLS = 3
TIMED_CALLS = 200


def first_cores(device, num_cores):
    """The first num_cores cores of the grid, row by row, as a CoreRangeSet and as a list."""
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    return cores, ttnn.corerange_to_cores(cores, None, True)


def descriptor_program(input_tensor, output_tensor, cores, core_list):
    """The ProgramDescriptor for one tile per core, with tile i on core i."""
    in_cb, out_cb = 0, 2

    def cb(index):
        format_descriptor = ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=TILE_BYTES)
        return ttnn.CBDescriptor(total_size=2 * TILE_BYTES, core_ranges=cores, format_descriptors=[format_descriptor])

    reader_args, writer_args, compute_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for tile, core in enumerate(core_list):
        reader_args[core.x][core.y] = [input_tensor.buffer_address(), 1, tile]
        writer_args[core.x][core.y] = [output_tensor.buffer_address(), 1, tile]
        compute_args[core.x][core.y] = [1]
    writer_compile_time_args = [out_cb] + ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    return ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=f"{UNARY_KERNELS}/dataflow/reader_unary_interleaved_start_id.cpp",
                core_ranges=cores,
                compile_time_args=ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args(),
                runtime_args=reader_args,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{UNARY_KERNELS}/dataflow/writer_unary_interleaved_start_id.cpp",
                core_ranges=cores,
                compile_time_args=writer_compile_time_args,
                runtime_args=writer_args,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{UNARY_KERNELS}/compute/eltwise_sfpu.cpp",
                core_ranges=cores,
                defines=list(EXP_DEFINES.items()),
                runtime_args=compute_args,
                config=ttnn.ComputeConfigDescriptor(),
            ),
        ],
        semaphores=[],
        cbs=[cb(in_cb), cb(out_cb)],
    )


def program_spec(input_tensor, output_tensor, cores):
    """The ProgramSpec of the same program."""
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_pages", "start_id"])
    return E.ProgramSpec(
        name="exp",
        kernels=[
            E.KernelSpec(
                unique_id="reader",
                source=f"{UNARY_KERNELS}/dataflow/reader_unary_interleaved_start_id_metal2.cpp",
                dfb_bindings=[E.producer_of("in", "in")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="src")],
                runtime_arg_schema=dm_schema,
                hw_config=E.create_reader_datamovement_config(),
            ),
            E.KernelSpec(
                unique_id="writer",
                source=f"{UNARY_KERNELS}/dataflow/writer_unary_interleaved_start_id_metal2.cpp",
                dfb_bindings=[E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="dst")],
                runtime_arg_schema=dm_schema,
                hw_config=E.create_writer_datamovement_config(),
            ),
            E.KernelSpec(
                unique_id="compute",
                source=f"{UNARY_KERNELS}/compute/eltwise_sfpu_metal2.cpp",
                compiler_options=E.KernelSpec.CompilerOptions(defines=EXP_DEFINES),
                dfb_bindings=[E.consumer_of("in", "in"), E.producer_of("out", "out")],
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
                hw_config=E.ComputeHardwareConfig(),
            ),
        ],
        dataflow_buffers=[
            E.DataflowBufferSpec(
                unique_id=name, entry_size=TILE_BYTES, num_entries=2, data_format_metadata=ttnn.bfloat16
            )
            for name in ("in", "out")
        ],
        tensor_parameters=[
            E.TensorParameter(unique_id="input", spec=input_tensor.spec),
            E.TensorParameter(unique_id="output", spec=output_tensor.spec),
        ],
        work_units=[E.WorkUnitSpec(name="exp", kernels=["reader", "writer", "compute"], target_nodes=cores)],
    )


def program_run_args(input_tensor, output_tensor, core_list):
    """The ProgramRunArgs of one call, with tile i on core i."""
    dm_args = {"num_pages": {core: 1 for core in core_list}, "start_id": {core: i for i, core in enumerate(core_list)}}
    return E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="compute", runtime_arg_values={"num_tiles": {core: 1 for core in core_list}}),
        ],
        tensor_args={"input": input_tensor, "output": output_tensor},
    )


def per_call_us(device, call):
    """The median host time of one cache-hit call, in microseconds."""
    for _ in range(WARMUP_CALLS):
        call()
    ttnn.synchronize_device(device)
    entries = device.num_program_cache_entries()
    samples = []
    for _ in range(TIMED_CALLS):
        start = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - start)
    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == entries, "the timed calls must be cache hits"
    return statistics.median(samples) / 1000


@pytest.mark.parametrize("cores", ["1", "64", "full_grid"])
def test_per_call_host_time_against_the_program_descriptor_form(device, cores):
    if os.environ.get("TT_METAL_SLOW_DISPATCH_MODE"):
        pytest.skip("In slow dispatch a call waits for its program to finish, so the time would be the device's")
    grid = device.compute_with_storage_grid_size()
    num_cores = {"1": 1, "64": 64, "full_grid": grid.x * grid.y}[cores]
    if num_cores > grid.x * grid.y:
        pytest.skip(f"The grid has fewer than {num_cores} cores")
    core_set, core_list = first_cores(device, num_cores)

    torch_input = torch.rand([1, num_cores, 32, 32]).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, device)
    io_tensors = [input_tensor, output_tensor]

    descriptor = descriptor_program(input_tensor, output_tensor, core_set, core_list)
    spec = program_spec(input_tensor, output_tensor, core_set)
    run_args = program_run_args(input_tensor, output_tensor, core_list)

    # Both forms compute the same output.
    assert_with_ulp(
        expected_result=torch.exp(torch_input),
        actual_result=ttnn.to_torch(ttnn.generic_op(io_tensors, descriptor)),
        ulp_threshold=2,
    )
    assert_with_ulp(
        expected_result=torch.exp(torch_input),
        actual_result=ttnn.to_torch(E.generic_op(io_tensors, spec, run_args)),
        ulp_threshold=2,
    )

    times = {
        "ProgramDescriptor": per_call_us(device, lambda: ttnn.generic_op(io_tensors, descriptor)),
        "ProgramSpec": per_call_us(device, lambda: E.generic_op(io_tensors, spec, run_args)),
        "ProgramDescriptor, rebuilt each call": per_call_us(
            device,
            lambda: ttnn.generic_op(io_tensors, descriptor_program(input_tensor, output_tensor, core_set, core_list)),
        ),
        "ProgramSpec, run args rebuilt each call": per_call_us(
            device,
            lambda: E.generic_op(io_tensors, spec, program_run_args(input_tensor, output_tensor, core_list)),
        ),
    }
    lines = [f"generic_op per-call host time on {num_cores} cores ({ttnn.get_arch_name()}):"]
    for name, us in times.items():
        reference = times["ProgramDescriptor, rebuilt each call" if "rebuilt" in name else "ProgramDescriptor"]
        lines.append(f"  {name:<40} {us:9.1f} us  ({us / reference:.2f}x the ProgramDescriptor form)")
    logger.info("\n".join(lines))
