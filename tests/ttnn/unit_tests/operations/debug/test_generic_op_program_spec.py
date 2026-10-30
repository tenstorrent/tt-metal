# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.generic_op on a Metal 2.0 ProgramSpec and ProgramRunArgs.

Most device tests run out = a * b on bfloat16 tiles, with three kernels in kernels/program_spec_mul/ and the tiles
split over the cores. The tests cover correctness, the program cache (a later call with the same spec reuses the
compiled program with new tensors and argument values), the checks every call runs on the run args and tensors,
the mesh form, preparation without dispatch, and the hash the program cache keys on.
"""

import os
import pathlib

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

E = ttnn.experimental

MUL_KERNELS = pathlib.Path(__file__).parent / "kernels" / "program_spec_mul"
UNARY_KERNELS = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels"
SEMAPHORE_KERNELS = "tests/tt_metal/tt_metal/test_kernels/dataflow"
TILE_BYTES = 2 * 32 * 32  # one bfloat16 tile


@pytest.fixture(autouse=True)
def validate_program_args():
    """Per-core coverage is checked on later calls only with validate_program_args on, as in CI."""
    with ttnn.manage_config("validate_program_args", True):
        yield


def split_tiles(device, num_tiles, tile_limit=None):
    """The cores to run on, and each core's tile count and first tile, row by row.

    The cores come from num_tiles and the counts from tile_limit, so a smaller tile_limit changes only argument values.
    """
    grid = device.compute_with_storage_grid_size()
    _, cores, *_ = ttnn.split_work_to_cores(grid, num_tiles, row_wise=True)
    work = num_tiles if tile_limit is None else min(tile_limit, num_tiles)
    _, _, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, work, row_wise=True)
    counts = {}
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core:
            for core in ttnn.corerange_to_cores(group, None, True):
                counts[core] = per_core
    num_tiles_by_core, start_id_by_core, start = {}, {}, 0
    for core in ttnn.corerange_to_cores(cores, None, True):
        num_tiles_by_core[core] = counts.get(core, 0)
        start_id_by_core[core] = start
        start += counts.get(core, 0)
    return cores, num_tiles_by_core, start_id_by_core


def mul_program(
    a, b, out, *, tile_limit=None, cores_for_tiles=None, relaxations=None, varargs=False, aliased_buffer=False
):
    """(spec, run_args) for out = a * b. Tensor parameters a, b and out are bound to a, b and out.

    The program runs on the cores for cores_for_tiles tiles (by default, the output's tile count).
    """
    num_tiles = out.volume() // (32 * 32)
    work = num_tiles if tile_limit is None else min(tile_limit, num_tiles)
    cores, num_tiles_by_core, start_id_by_core = split_tiles(a.device(), cores_for_tiles or num_tiles, work)

    dfb_names = ["in_a", "in_b", "out"]
    reader_dfbs = [E.producer_of("in_a", "in_a"), E.producer_of("in_b", "in_b")]
    compute_dfbs = [E.consumer_of("in_a", "in_a"), E.consumer_of("in_b", "in_b"), E.producer_of("out", "out")]
    buffers = {}
    if aliased_buffer:
        # A second view of in_a's memory. The kernels never use it.
        dfb_names.append("in_a_view")
        reader_dfbs.append(E.producer_of("in_a_view", "in_a_view"))
        compute_dfbs.append(E.consumer_of("in_a_view", "in_a_view"))
        buffers["in_a"] = E.DFBAdvancedOptions(alias_with=["in_a_view"])
        buffers["in_a_view"] = E.DFBAdvancedOptions(alias_with=["in_a"])

    if varargs:
        dm_options = dict(
            compiler_options=E.KernelSpec.CompilerOptions(defines={"USE_VARARGS": "1"}),
            advanced_options=E.KernelAdvancedOptions(num_runtime_varargs=2),
        )
    else:
        dm_options = dict(runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"]))

    spec = E.ProgramSpec(
        name="mul",
        kernels=[
            E.KernelSpec(
                unique_id="reader",
                source=str(MUL_KERNELS / "reader.cpp"),
                hw_config=E.create_reader_datamovement_config(),
                dfb_bindings=reader_dfbs,
                tensor_bindings=[
                    E.TensorBinding(tensor_parameter_name="a", accessor_name="a"),
                    E.TensorBinding(tensor_parameter_name="b", accessor_name="b"),
                ],
                **dm_options,
            ),
            E.KernelSpec(
                unique_id="compute",
                source=str(MUL_KERNELS / "compute.cpp"),
                hw_config=E.ComputeHardwareConfig(),
                dfb_bindings=compute_dfbs,
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
            ),
            E.KernelSpec(
                unique_id="writer",
                source=str(MUL_KERNELS / "writer.cpp"),
                hw_config=E.create_writer_datamovement_config(),
                dfb_bindings=[E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="out", accessor_name="out")],
                **dm_options,
            ),
        ],
        dataflow_buffers=[
            E.DataflowBufferSpec(
                unique_id=name,
                entry_size=TILE_BYTES,
                num_entries=2,
                data_format_metadata=ttnn.bfloat16,
                advanced_options=buffers.get(name, E.DFBAdvancedOptions()),
            )
            for name in dfb_names
        ],
        tensor_parameters=[
            E.TensorParameter(unique_id=name, spec=tensor.spec, relaxations=relaxations or E.TensorSpecRelaxations())
            for name, tensor in (("a", a), ("b", b), ("out", out))
        ],
        work_units=[E.WorkUnitSpec(name="mul", kernels=["reader", "compute", "writer"], target_nodes=cores)],
    )

    if varargs:
        varargs_by_core = {core: [num_tiles_by_core[core], start_id_by_core[core]] for core in num_tiles_by_core}
        dm_run_args = dict(advanced_options=E.AdvancedKernelRunArgs(runtime_varargs=varargs_by_core))
    else:
        dm_run_args = dict(runtime_arg_values={"num_tiles": num_tiles_by_core, "start_id": start_id_by_core})
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", **dm_run_args),
            E.KernelRunArgs(kernel="compute", runtime_arg_values={"num_tiles": num_tiles_by_core}),
            E.KernelRunArgs(kernel="writer", **dm_run_args),
        ],
        tensor_args={"a": a, "b": b, "out": out},
    )
    return spec, run_args


def inputs(device, shape=(1, 1, 64, 64), seed=0):
    torch.manual_seed(seed)
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)
    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.allocate_tensor_on_device(a.spec, device)
    return torch_a * torch_b, a, b, out


def kernel_run_args(run_args, kernel):
    return next(args for args in run_args.kernel_run_args if args.kernel == kernel)


def replace_kernel_run_args(run_args, kernel, new_args):
    run_args.kernel_run_args = [new_args if args.kernel == kernel else args for args in run_args.kernel_run_args]


# ------------------------------------------------------------------------------------------------------- correctness


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 128, 128), (2, 3, 64, 96)])
def test_mul(device, shape):
    expected, a, b, out = inputs(device, shape)
    spec, run_args = mul_program(a, b, out)
    result = E.generic_op([a, b, out], spec, run_args)
    assert_with_pcc(expected, ttnn.to_torch(result), 0.9999)


def exp_program(input_tensor, output_tensor, compute_threads=1):
    """(spec, run_args) for output = exp(input), as in the design doc's example.

    With compute_threads > 1 (Quasar only), each compute thread takes every compute_threads-th tile of its core.
    """
    num_tiles = input_tensor.volume() // (32 * 32)
    cores, num_pages, start_id = split_tiles(input_tensor.device(), num_tiles)
    assert all(count % compute_threads == 0 for count in num_pages.values())

    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_pages", "start_id"])
    spec = E.ProgramSpec(
        name="exp",
        kernels=[
            E.KernelSpec(
                unique_id="reader",
                source=f"{UNARY_KERNELS}/dataflow/reader_unary_interleaved_start_id_metal2.cpp",
                dfb_bindings=[E.producer_of("in", "in")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="src")],
                runtime_arg_schema=dm_schema,
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
            ),
            E.KernelSpec(
                unique_id="writer",
                source=f"{UNARY_KERNELS}/dataflow/writer_unary_interleaved_start_id_metal2.cpp",
                dfb_bindings=[E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="dst")],
                runtime_arg_schema=dm_schema,
                hw_config=E.create_writer_datamovement_config(disable_dfb_implicit_sync_for_all=True),
            ),
            E.KernelSpec(
                unique_id="compute",
                source=f"{UNARY_KERNELS}/compute/eltwise_sfpu_metal2.cpp",
                num_threads=compute_threads,
                compiler_options=E.KernelSpec.CompilerOptions(
                    defines={"SFPU_OP_EXP_INCLUDE": "1", "SFPU_OP_CHAIN_0": "exp_tile_init(); exp_tile(0);"},
                    opt_level=E.KernelBuildOptLevel.O3,
                ),
                dfb_bindings=[E.strided_consumer_of("in", "in"), E.producer_of("out", "out")],
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
    dm_args = {"num_pages": num_pages, "start_id": start_id}
    tiles_per_thread = {core: count // compute_threads for core, count in num_pages.items()}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="compute", runtime_arg_values={"num_tiles": tiles_per_thread}),
        ],
        tensor_args={"input": input_tensor, "output": output_tensor},
    )
    return spec, run_args


def exp_inputs(device, num_tiles):
    torch_input = torch.rand([1, num_tiles, 32, 32]).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return torch_input, input_tensor, ttnn.allocate_tensor_on_device(input_tensor.spec, device)


def test_exp_in_place(device, expect_error):
    # The guide example runs the program out of place; see test_the_guide_example.
    torch_input, input_tensor, output_tensor = exp_inputs(device, 8)
    spec, run_args = exp_program(input_tensor, output_tensor)

    # A tensor bound to two parameters must be listed twice, so io_tensors holding only the output is rejected.
    in_place_run_args = E.ProgramRunArgs(
        kernel_run_args=run_args.kernel_run_args, tensor_args={"input": input_tensor, "output": input_tensor}
    )
    with expect_error(RuntimeError, "needs its own io_tensors entry"):
        E.generic_op([input_tensor], spec, in_place_run_args)

    # To run in place with only the output in io_tensors, bind both kernels to one tensor parameter.
    spec.tensor_parameters = [E.TensorParameter(unique_id="tensor", spec=input_tensor.spec)]
    kernels = spec.kernels
    kernels[0].tensor_bindings = [E.TensorBinding(tensor_parameter_name="tensor", accessor_name="src")]
    kernels[1].tensor_bindings = [E.TensorBinding(tensor_parameter_name="tensor", accessor_name="dst")]
    spec.kernels = kernels
    in_place_run_args.tensor_args = {"tensor": input_tensor}
    result = E.generic_op([input_tensor], spec, in_place_run_args)
    assert_with_pcc(torch.exp(torch_input), ttnn.to_torch(result), 0.999)


def guide_example(device):
    """The example in docs/source/ttnn/ttnn/generic_op_program_spec.rst, which includes the lines between the markers."""
    # [guide-example-start]
    E = ttnn.experimental
    KERNELS = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels"
    TILE_BYTES = 2 * 32 * 32  # one bfloat16 tile

    num_tiles = 8
    torch_input = torch.rand([1, num_tiles, 32, 32]).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # generic_op does not create its output: preallocate it.
    output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, device)

    grid = device.compute_with_storage_grid_size()
    _, cores, group_1, group_2, tiles_1, tiles_2 = ttnn.split_work_to_cores(grid, num_tiles)

    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_pages", "start_id"])
    spec = E.ProgramSpec(
        name="exp",
        kernels=[
            E.KernelSpec(
                unique_id="reader",
                source=f"{KERNELS}/dataflow/reader_unary_interleaved_start_id_metal2.cpp",
                dfb_bindings=[E.producer_of("in", "in")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="src")],
                runtime_arg_schema=dm_schema,
                hw_config=E.create_reader_datamovement_config(),
            ),
            E.KernelSpec(
                unique_id="writer",
                source=f"{KERNELS}/dataflow/writer_unary_interleaved_start_id_metal2.cpp",
                dfb_bindings=[E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="dst")],
                runtime_arg_schema=dm_schema,
                hw_config=E.create_writer_datamovement_config(),
            ),
            E.KernelSpec(
                unique_id="compute",
                source=f"{KERNELS}/compute/eltwise_sfpu_metal2.cpp",
                compiler_options=E.KernelSpec.CompilerOptions(
                    defines={"SFPU_OP_EXP_INCLUDE": "1", "SFPU_OP_CHAIN_0": "exp_tile_init(); exp_tile(0);"}
                ),
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

    # The values of this call, by argument name and then by core. Tensors are bound by name.
    num_pages, start_id, start = {}, {}, 0
    for group, n in ((group_1, tiles_1), (group_2, tiles_2)):
        for core in ttnn.corerange_to_cores(group):
            num_pages[core], start_id[core] = n, start
            start += n
    dm_args = {"num_pages": num_pages, "start_id": start_id}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="compute", runtime_arg_values={"num_tiles": num_pages}),
        ],
        tensor_args={"input": input_tensor, "output": output_tensor},
    )

    output_tensor = E.generic_op([input_tensor, output_tensor], spec, run_args)
    # [guide-example-end]
    return torch_input, output_tensor


def test_the_guide_example(device):
    torch_input, output_tensor = guide_example(device)
    assert_with_pcc(torch.exp(torch_input), ttnn.to_torch(output_tensor), 0.999)


def test_two_compute_threads_on_quasar(device):
    if "quasar" not in ttnn.get_arch_name():
        pytest.skip("A kernel has more than one thread on Quasar only")
    num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    torch_input, input_tensor, output_tensor = exp_inputs(device, 2 * num_cores)
    spec, run_args = exp_program(input_tensor, output_tensor, compute_threads=2)
    for _ in range(2):
        result = E.generic_op([input_tensor, output_tensor], spec, run_args)
        assert_with_pcc(torch.exp(torch_input), ttnn.to_torch(result), 0.999)


def test_sharded_typecast_on_borrowed_buffers(device):
    """Both buffers sit on the tensors' own L1 shards, as in ttnn's sharded typecast."""
    if "quasar" in ttnn.get_arch_name() and os.environ.get("TT_METAL_SIMULATOR"):
        pytest.skip("ttsim's Quasar model cannot pack bfloat16 into float32 yet")
    shape = (1, 1, 64, 64)
    memory_config = ttnn.create_sharded_memory_config(
        shape, core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.HEIGHT
    )
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, device, memory_config
    )
    cores = memory_config.shard_spec.grid
    tiles_per_core = 2
    # tt::DataFormat values of bfloat16 and float32; the kernel takes them as compile-time arguments.
    float16_b, float32 = 5, 0

    spec = E.ProgramSpec(
        name="typecast_sharded",
        kernels=[
            E.KernelSpec(
                unique_id="reader",
                source=f"{UNARY_KERNELS}/dataflow/reader_unary_sharded_metal2.cpp",
                dfb_bindings=[E.producer_of("in", "in")],
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles_per_core"]),
                hw_config=E.create_reader_datamovement_config(),
            ),
            E.KernelSpec(
                unique_id="compute",
                source="ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp",
                # No writer drains the output buffer, so the compute kernel both produces and consumes it.
                dfb_bindings=[
                    E.consumer_of("in", "in"),
                    E.producer_of("out", "out"),
                    E.consumer_of("out", "out"),
                ],
                compile_time_args={
                    "per_core_block_cnt": 1,
                    "per_core_block_dim": tiles_per_core,
                    "in_data_format": float16_b,
                    "out_data_format": float32,
                },
                hw_config=E.ComputeHardwareConfig(),
            ),
        ],
        dataflow_buffers=[
            E.DataflowBufferSpec(
                unique_id="in",
                entry_size=TILE_BYTES,
                num_entries=tiles_per_core,
                data_format_metadata=ttnn.bfloat16,
                borrowed_from="input",
            ),
            E.DataflowBufferSpec(
                unique_id="out",
                entry_size=2 * TILE_BYTES,
                num_entries=tiles_per_core,
                data_format_metadata=ttnn.float32,
                borrowed_from="output",
            ),
        ],
        tensor_parameters=[
            E.TensorParameter(unique_id="input", spec=input_tensor.spec),
            E.TensorParameter(unique_id="output", spec=output_tensor.spec),
        ],
        work_units=[E.WorkUnitSpec(name="typecast", kernels=["reader", "compute"], target_nodes=cores)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel="reader",
                runtime_arg_values={
                    "num_tiles_per_core": {core: tiles_per_core for core in ttnn.corerange_to_cores(cores)}
                },
            )
        ],
        tensor_args={"input": input_tensor, "output": output_tensor},
    )
    for _ in range(2):
        result = ttnn.to_torch(E.generic_op([input_tensor, output_tensor], spec, run_args))
        assert result.dtype == torch.float32
        assert torch.equal(result, torch_input.float())


def test_varargs_and_aliased_buffers(device):
    expected, a, b, out = inputs(device, (1, 1, 96, 64))
    spec, run_args = mul_program(a, b, out, varargs=True, aliased_buffer=True)
    for _ in range(2):
        assert_with_pcc(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)), 0.9999)


def test_semaphores_start_at_zero_on_every_call(device):
    out = ttnn.allocate_tensor_on_device(ttnn.TensorSpec((32, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT), device)
    node = ttnn.CoreCoord(0, 0)
    spec = E.ProgramSpec(
        name="semaphore_loopback",
        kernels=[
            E.KernelSpec(
                unique_id="producer",
                source=f"{SEMAPHORE_KERNELS}/semaphore_accessor_loopback_producer.cpp",
                semaphore_bindings=[E.SemaphoreBinding(semaphore_spec_name="sem", accessor_name="signal")],
                hw_config=E.create_writer_datamovement_config(),
            ),
            E.KernelSpec(
                unique_id="consumer",
                source=f"{SEMAPHORE_KERNELS}/semaphore_accessor_loopback_consumer.cpp",
                semaphore_bindings=[E.SemaphoreBinding(semaphore_spec_name="sem", accessor_name="waiter")],
                hw_config=E.create_reader_datamovement_config(),
            ),
        ],
        semaphores=[E.SemaphoreSpec(unique_id="sem", target_nodes=node)],
        work_units=[E.WorkUnitSpec(kernels=["producer", "consumer"], target_nodes=node)],
    )
    # The consumer waits for the producer's signal; a program that runs at all on the second call reused the cache.
    entries = device.num_program_cache_entries()
    for _ in range(2):
        E.generic_op([out], spec, E.ProgramRunArgs())
    assert device.num_program_cache_entries() == entries + 1


# ----------------------------------------------------------------------------------------------------- program cache


def test_later_calls_reuse_the_program_with_new_tensors_and_values(device):
    expected, a, b, out = inputs(device, seed=1)
    spec, run_args = mul_program(a, b, out)
    E.generic_op([a, b, out], spec, run_args)
    entries = device.num_program_cache_entries()

    # New tensors at new addresses, and one tile fewer for the same cores: only argument values change. The output
    # starts as a known value, and the core that the first call gave the fourth tile now has none, so the fourth tile
    # keeps that value only if the new values reached the device.
    expected, a2, b2, _ = inputs(device, seed=2)
    initial = torch.full((1, 1, 64, 64), 7.0, dtype=torch.bfloat16)
    out2 = ttnn.from_torch(initial, layout=ttnn.TILE_LAYOUT, device=device)
    spec2, run_args2 = mul_program(a2, b2, out2, tile_limit=3)
    assert E.compute_program_spec_hash(spec2) == E.compute_program_spec_hash(spec)
    result = tile_order(ttnn.to_torch(E.generic_op([a2, b2, out2], spec2, run_args2)))
    assert device.num_program_cache_entries() == entries
    assert_with_pcc(tile_order(expected)[: 3 * 1024], result[: 3 * 1024], 0.9999)
    assert torch.equal(result[3 * 1024 :], tile_order(initial)[3 * 1024 :])


def tile_order(tensor):
    """The elements of a (1, 1, 64, 64) tensor in tile order, so that the first n tiles are its first n * 1024."""
    return tensor.reshape(2, 32, 2, 32).permute(0, 2, 1, 3).flatten()


def test_custom_program_hash_replaces_the_spec_hash(device):
    expected, a, b, out = inputs(device, seed=3)
    spec, run_args = mul_program(a, b, out)
    other_spec, _ = mul_program(a, b, out)
    other_spec.name = "renamed"
    kernels = other_spec.kernels
    kernels[1].compiler_options = E.KernelSpec.CompilerOptions(defines={"UNUSED": "1"})
    other_spec.kernels = kernels

    entries = device.num_program_cache_entries()
    E.generic_op([a, b, out], spec, run_args, custom_program_hash=12345)
    result = E.generic_op([a, b, out], other_spec, run_args, custom_program_hash=12345)
    assert device.num_program_cache_entries() == entries + 1
    assert_with_pcc(expected, ttnn.to_torch(result), 0.9999)
    E.generic_op([a, b, out], other_spec, run_args)
    assert device.num_program_cache_entries() == entries + 2


def test_custom_program_hash_still_keys_on_vararg_counts(device):
    expected, a, b, out = inputs(device, seed=9)
    spec, run_args = mul_program(a, b, out, varargs=True)
    entries = device.num_program_cache_entries()
    E.generic_op([a, b, out], spec, run_args, custom_program_hash=777)

    # Three varargs per core under the same custom hash. Metal writes varargs by position, so reusing the program
    # compiled for two would write past its argument buffer: this call compiles a program of its own.
    kernels = spec.kernels
    for index in (0, 2):
        kernels[index].advanced_options = E.KernelAdvancedOptions(num_runtime_varargs=3)
    spec.kernels = kernels
    for kernel in ("reader", "writer"):
        args = kernel_run_args(run_args, kernel)
        varargs = {core: values + [0] for core, values in args.advanced_options.runtime_varargs.items()}
        args.advanced_options = E.AdvancedKernelRunArgs(runtime_varargs=varargs)
        replace_kernel_run_args(run_args, kernel, args)
    result = E.generic_op([a, b, out], spec, run_args, custom_program_hash=777)
    assert device.num_program_cache_entries() == entries + 2
    assert_with_pcc(expected, ttnn.to_torch(result), 0.9999)


def test_custom_program_hash_still_keys_on_the_cores(device):
    expected, a, b, out = inputs(device, seed=12)
    spec, run_args = mul_program(a, b, out)
    entries = device.num_program_cache_entries()
    E.generic_op([a, b, out], spec, run_args, custom_program_hash=4242)

    # One tile on one core under the same custom hash. A cached program keeps the values of cores that the run args
    # leave out, so reusing the one above would run its other cores on stale values: this call compiles its own.
    expected, a, b, out = inputs(device, (1, 1, 32, 32), seed=13)
    spec, run_args = mul_program(a, b, out)
    result = E.generic_op([a, b, out], spec, run_args, custom_program_hash=4242)
    assert device.num_program_cache_entries() == entries + 2
    assert_with_pcc(expected, ttnn.to_torch(result), 0.9999)


def test_a_precomputed_spec_hash_gives_the_default_cache_entry(device):
    expected, a, b, out = inputs(device, seed=14)
    spec, run_args = mul_program(a, b, out)
    entries = device.num_program_cache_entries()
    E.generic_op([a, b, out], spec, run_args)
    result = E.generic_op([a, b, out], spec, run_args, custom_program_hash=E.compute_program_spec_hash(spec))
    assert device.num_program_cache_entries() == entries + 1
    assert_with_pcc(expected, ttnn.to_torch(result), 0.9999)


def test_dfb_run_overrides(device):
    expected, a, b, out = inputs(device, seed=10)
    spec, run_args = mul_program(a, b, out)
    run_args.dfb_run_overrides = [E.DFBRunOverrides(dfb="in_a", num_entries=4)]
    assert_with_pcc(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)), 0.9999)
    run_args.dfb_run_overrides = []
    assert_with_pcc(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)), 0.9999)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ([("missing", 4)], "names DFB 'missing', which the ProgramSpec does not declare"),
        ([("in_a", 4), ("in_a", 4)], "lists DFB 'in_a' more than once"),
        ([("in_a", 8), ("in_b", 0)], "sets a size of 0 for DFB 'in_b'"),
    ],
    ids=["unknown", "repeated", "zero"],
)
def test_dfb_run_overrides_are_checked_on_every_call(device, overrides, message, expect_error):
    # Without validate_program_args, Metal would apply the entries before a bad one and leave those buffers resized.
    ttnn.CONFIG.validate_program_args = False
    expected, a, b, out = inputs(device, seed=11)
    spec, run_args = mul_program(a, b, out)
    E.generic_op([a, b, out], spec, run_args)
    run_args.dfb_run_overrides = [E.DFBRunOverrides(dfb=dfb, num_entries=num_entries) for dfb, num_entries in overrides]
    with expect_error(RuntimeError, message):
        E.generic_op([a, b, out], spec, run_args)
    run_args.dfb_run_overrides = []
    assert_with_pcc(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)), 0.9999)


def test_prepare_fills_the_program_cache(device):
    expected, a, b, out = inputs(device, seed=4)
    spec, run_args = mul_program(a, b, out)
    entries = device.num_program_cache_entries()
    E.prepare_generic_op([a, b, out], spec, run_args)
    assert device.num_program_cache_entries() == entries + 1
    result = E.generic_op([a, b, out], spec, run_args)
    assert device.num_program_cache_entries() == entries + 1
    assert_with_pcc(expected, ttnn.to_torch(result), 0.9999)


def test_dynamic_shape_relaxation_reuses_the_program(device):
    relaxations = E.TensorSpecRelaxations(dynamic_tensor_shape=True)
    expected, a, b, out = inputs(device, (1, 1, 32, 64), seed=5)
    spec, run_args = mul_program(a, b, out, relaxations=relaxations, cores_for_tiles=4)
    E.generic_op([a, b, out], spec, run_args)
    entries = device.num_program_cache_entries()

    # Twice the tiles on the same cores: only the tensor shapes and the argument values differ. With the first call's
    # values, two cores would have no tiles and half the output would stay unwritten.
    expected, a2, b2, out2 = inputs(device, (1, 1, 64, 64), seed=6)
    spec2, run_args2 = mul_program(a2, b2, out2, relaxations=relaxations, cores_for_tiles=4)
    result = E.generic_op([a2, b2, out2], spec2, run_args2)
    assert device.num_program_cache_entries() == entries
    assert_with_pcc(expected, ttnn.to_torch(result), 0.9999)


def test_a_tensor_bound_to_two_parameters_is_listed_twice(device, expect_error):
    torch.manual_seed(7)
    torch_a = torch.randn((1, 1, 64, 64), dtype=torch.bfloat16)
    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.allocate_tensor_on_device(a.spec, device)
    spec, run_args = mul_program(a, a, out)
    for _ in range(2):  # the first call and a cache hit
        result = E.generic_op([a, a, out], spec, run_args)
        assert_with_pcc(torch_a * torch_a, ttnn.to_torch(result), 0.9999)
    with expect_error(RuntimeError, "needs its own io_tensors entry"):
        E.generic_op([a, out], spec, run_args)


# ------------------------------------------------------------------------------------------------------------ checks


@pytest.mark.parametrize("validate", [True, False], ids=["validate_on", "validate_off"])
def test_a_spec_reused_with_other_tensors_is_rejected(device, validate, expect_error):
    _, a, b, out = inputs(device, (1, 1, 64, 64))
    spec, run_args = mul_program(a, b, out)
    E.generic_op([a, b, out], spec, run_args)

    ttnn.CONFIG.validate_program_args = validate
    _, a2, b2, out2 = inputs(device, (1, 1, 128, 64))
    run_args.tensor_args = {"a": a2, "b": b2, "out": out2}
    with expect_error(RuntimeError, "does not match the parameter's spec"):
        E.generic_op([a2, b2, out2], spec, run_args)


def test_tensor_args_must_match_the_parameters(device, expect_error):
    _, a, b, out = inputs(device)
    spec, run_args = mul_program(a, b, out)
    run_args.tensor_args = {"a": a, "b": b}
    with expect_error(RuntimeError, "declares 3 tensor parameters, but run_args.tensor_args gives 2"):
        E.generic_op([a, b, out], spec, run_args)
    run_args.tensor_args = {"a": a, "b": b, "c": out}
    with expect_error(RuntimeError, "tensor parameter 'out' has no tensor"):
        E.generic_op([a, b, out], spec, run_args)
    run_args.tensor_args = {"a": a, "b": b, "out": out}
    with expect_error(RuntimeError, "tensor of tensor parameter 'b' is not in io_tensors"):
        E.generic_op([a, out], spec, run_args)
    with expect_error(RuntimeError, "io_tensors must hold at least the output tensor"):
        E.generic_op([], spec, run_args)


def test_tensors_on_another_mesh_device_are_rejected(device, expect_error):
    submesh = device.create_submesh(ttnn.MeshShape(1, 1))
    _, a, b, _ = inputs(device)
    out = ttnn.allocate_tensor_on_device(a.spec, submesh)
    spec, run_args = mul_program(a, b, out)
    try:
        with expect_error(RuntimeError, "io_tensors\\[0\\] is on a different MeshDevice than the output tensor"):
            E.generic_op([a, b, out], spec, run_args)
    finally:
        # The parent MeshDevice keeps its submeshes alive, so an open submesh would outlive the device fixture's close
        # and then close the chip again, on hardware while the next test was opening it. Close it while the parent is
        # open.
        ttnn.deallocate(out)
        ttnn.close_mesh_device(submesh)


@pytest.mark.parametrize("cache_hit", [False, True], ids=["first_call", "later_call"])
def test_run_args_must_match_the_kernels(device, cache_hit, expect_error):
    _, a, b, out = inputs(device)
    spec, good_run_args = mul_program(a, b, out)
    if cache_hit:
        E.generic_op([a, b, out], spec, good_run_args)

    def run_with(update):
        _, run_args = mul_program(a, b, out)
        update(run_args)
        E.generic_op([a, b, out], spec, run_args)

    with expect_error(RuntimeError, "kernel 'compute' takes runtime arguments, but run_args.kernel_run_args"):
        run_with(lambda run_args: setattr(run_args, "kernel_run_args", run_args.kernel_run_args[::2]))
    with expect_error(RuntimeError, "names kernel 'missing', which the ProgramSpec does not declare"):
        run_with(
            lambda run_args: setattr(
                run_args, "kernel_run_args", run_args.kernel_run_args + [E.KernelRunArgs(kernel="missing")]
            )
        )
    with expect_error(RuntimeError, "lists kernel 'reader' more than once"):
        run_with(
            lambda run_args: setattr(
                run_args, "kernel_run_args", run_args.kernel_run_args + [kernel_run_args(run_args, "reader")]
            )
        )

    def drop_start_id(run_args):
        reader = kernel_run_args(run_args, "reader")
        values = reader.runtime_arg_values
        del values["start_id"]
        reader.runtime_arg_values = values
        replace_kernel_run_args(run_args, "reader", reader)

    with expect_error(
        RuntimeError,
        r"declares the runtime arguments \[num_tiles, start_id\], but its run args give \[num_tiles\]",
    ):
        run_with(drop_start_id)

    def add_common_arg(run_args):
        reader = kernel_run_args(run_args, "reader")
        reader.common_runtime_arg_values = {"undeclared": 1}
        replace_kernel_run_args(run_args, "reader", reader)

    with expect_error(RuntimeError, "declares the common runtime arguments"):
        run_with(add_common_arg)


@pytest.mark.parametrize("cache_hit", [False, True], ids=["first_call", "later_call"])
def test_vararg_counts_are_checked_on_every_call(device, cache_hit, expect_error):
    _, a, b, out = inputs(device)
    spec, run_args = mul_program(a, b, out, varargs=True)
    if cache_hit:
        E.generic_op([a, b, out], spec, run_args)

    reader = kernel_run_args(run_args, "reader")
    varargs = reader.advanced_options.runtime_varargs
    core = next(iter(varargs))
    varargs[core] = list(varargs[core]) + [0]
    reader.advanced_options = E.AdvancedKernelRunArgs(runtime_varargs=varargs)
    replace_kernel_run_args(run_args, "reader", reader)
    with expect_error(RuntimeError, "declares 2 runtime varargs on node .*, but its run args give 3"):
        E.generic_op([a, b, out], spec, run_args)

    reader.advanced_options = E.AdvancedKernelRunArgs(runtime_varargs=varargs, common_runtime_varargs=[1])
    replace_kernel_run_args(run_args, "reader", reader)
    with expect_error(RuntimeError, "declares 0 common runtime varargs, but its run args give 1"):
        E.generic_op([a, b, out], spec, run_args)


@pytest.mark.parametrize("validate", [True, False], ids=["validate_on", "validate_off"])
def test_a_core_left_out_of_a_later_call(device, validate, expect_error):
    expected, a, b, out = inputs(device)
    spec, run_args = mul_program(a, b, out)
    E.generic_op([a, b, out], spec, run_args)

    ttnn.CONFIG.validate_program_args = validate
    compute = kernel_run_args(run_args, "compute")
    num_tiles = compute.runtime_arg_values["num_tiles"]
    num_tiles.pop(next(iter(num_tiles)))
    compute.runtime_arg_values = {"num_tiles": num_tiles}
    replace_kernel_run_args(run_args, "compute", compute)
    if validate:
        with expect_error(RuntimeError, "runtime argument 'num_tiles' does not have exactly one value"):
            E.generic_op([a, b, out], spec, run_args)
    else:
        # The left-out core keeps the value from the first call.
        assert_with_pcc(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)), 0.9999)


# ---------------------------------------------------------------------------------------------------------- mesh form


def mesh_range(device):
    return ttnn.MeshCoordinateRange(device.shape)


def test_mesh_form(device):
    expected, a, b, out = inputs(device, seed=8)
    spec, run_args = mul_program(a, b, out)
    mesh_spec = E.MeshProgramSpec(programs=[(mesh_range(device), spec)])
    mesh_run_args = E.MeshProgramRunArgs(programs=[(mesh_range(device), run_args)])
    assert mesh_run_args.programs[0][1].tensor_args["a"] is a
    for _ in range(2):
        result = E.generic_op([a, b, out], mesh_spec, mesh_run_args)
        assert_with_pcc(expected, ttnn.to_torch(result), 0.9999)

    entries = device.num_program_cache_entries()
    E.prepare_generic_op([a, b, out], mesh_spec, mesh_run_args)
    assert device.num_program_cache_entries() == entries


def test_mesh_form_checks_its_ranges(device, expect_error):
    _, a, b, out = inputs(device)
    spec, run_args = mul_program(a, b, out)
    whole_mesh = mesh_range(device)
    with expect_error(RuntimeError, "mesh ranges .* overlap"):
        E.generic_op(
            [a, b, out],
            E.MeshProgramSpec(programs=[(whole_mesh, spec), (whole_mesh, spec)]),
            E.MeshProgramRunArgs(programs=[(whole_mesh, run_args), (whole_mesh, run_args)]),
        )
    with expect_error(RuntimeError, "has 1 programs, but the MeshProgramRunArgs has run args for 2"):
        E.generic_op(
            [a, b, out],
            E.MeshProgramSpec(programs=[(whole_mesh, spec)]),
            E.MeshProgramRunArgs(programs=[(whole_mesh, run_args), (whole_mesh, run_args)]),
        )
    other_range = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(5, 5), ttnn.MeshCoordinate(5, 5))
    with expect_error(RuntimeError, "has no run args for mesh range"):
        E.generic_op(
            [a, b, out],
            E.MeshProgramSpec(programs=[(whole_mesh, spec)]),
            E.MeshProgramRunArgs(programs=[(other_range, run_args)]),
        )
    with expect_error(RuntimeError, "has no programs"):
        E.generic_op([a, b, out], E.MeshProgramSpec(), E.MeshProgramRunArgs())


# ------------------------------------------------------------------------------------------------------------- hash

SPEC = ttnn.TensorSpec((1, 1, 64, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
NODES = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])


def hash_spec(**kernel_fields):
    """A host-only spec for hash tests; kernel_fields replace fields of its first kernel."""
    reader = dict(
        unique_id="reader",
        source="reader.cpp",
        hw_config=E.create_reader_datamovement_config(),
        compile_time_args={"block": 1, "count": 2},
        compiler_options=E.KernelSpec.CompilerOptions(defines={"A": "1", "B": "2"}),
        dfb_bindings=[E.producer_of("in", "in")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="src")],
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["x", "y"]),
    )
    reader.update(kernel_fields)
    return E.ProgramSpec(
        name="hash",
        kernels=[
            E.KernelSpec(**reader),
            E.KernelSpec(
                unique_id="compute",
                source="compute.cpp",
                hw_config=E.ComputeHardwareConfig(),
                dfb_bindings=[E.consumer_of("in", "in")],
            ),
        ],
        dataflow_buffers=[
            E.DataflowBufferSpec(
                unique_id="in", entry_size=TILE_BYTES, num_entries=2, data_format_metadata=ttnn.bfloat16
            )
        ],
        tensor_parameters=[E.TensorParameter(unique_id="input", spec=SPEC)],
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute"], target_nodes=NODES)],
    )


def with_field(program_spec, field, index, /, **changes):
    items = getattr(program_spec, field)
    for name, value in changes.items():
        setattr(items[index], name, value)
    setattr(program_spec, field, items)
    return program_spec


STRUCTURAL_EDITS = {
    "source": lambda: hash_spec(source="other.cpp"),
    "inline_source": lambda: hash_spec(source=E.KernelSpec.SourceCode(code="void kernel_main() {}")),
    "num_threads": lambda: hash_spec(num_threads=2),
    "define": lambda: hash_spec(compiler_options=E.KernelSpec.CompilerOptions(defines={"A": "1", "B": "3"})),
    "opt_level": lambda: hash_spec(
        compiler_options=E.KernelSpec.CompilerOptions(defines={"A": "1", "B": "2"}, opt_level=E.KernelBuildOptLevel.O3)
    ),
    "compile_time_arg_value": lambda: hash_spec(compile_time_args={"block": 1, "count": 3}),
    "compile_time_arg_name": lambda: hash_spec(compile_time_args={"block": 1, "other": 2}),
    "runtime_arg_names": lambda: hash_spec(runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["x"])),
    "dfb_accessor": lambda: hash_spec(dfb_bindings=[E.producer_of("in", "other")]),
    "dfb_access_pattern": lambda: hash_spec(dfb_bindings=[E.blocked_producer_of("in", "in", 2)]),
    "tensor_accessor": lambda: hash_spec(
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="other")]
    ),
    "hw_config": lambda: hash_spec(hw_config=E.create_writer_datamovement_config()),
    "varargs": lambda: hash_spec(advanced_options=E.KernelAdvancedOptions(num_runtime_varargs=1)),
    "common_varargs": lambda: hash_spec(advanced_options=E.KernelAdvancedOptions(num_common_runtime_varargs=1)),
    "compute_config": lambda: with_field(
        hash_spec(), "kernels", 1, hw_config=E.ComputeHardwareConfig(fpu_math_fidelity=ttnn.MathFidelity.LoFi)
    ),
    "dfb_entry_size": lambda: with_field(hash_spec(), "dataflow_buffers", 0, entry_size=TILE_BYTES * 2),
    "dfb_num_entries": lambda: with_field(hash_spec(), "dataflow_buffers", 0, num_entries=4),
    "dfb_format": lambda: with_field(hash_spec(), "dataflow_buffers", 0, data_format_metadata=ttnn.float32),
    "dfb_tile": lambda: with_field(hash_spec(), "dataflow_buffers", 0, tile_format_metadata=ttnn.Tile([16, 32])),
    "dfb_borrowed_from": lambda: with_field(hash_spec(), "dataflow_buffers", 0, borrowed_from="input"),
    "dfb_alias": lambda: with_field(
        hash_spec(), "dataflow_buffers", 0, advanced_options=E.DFBAdvancedOptions(alias_with=["other"])
    ),
    "tensor_parameter_name": lambda: with_field(hash_spec(), "tensor_parameters", 0, unique_id="other"),
    "tensor_parameter_spec": lambda: with_field(
        hash_spec(), "tensor_parameters", 0, spec=ttnn.TensorSpec((1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    ),
    "tensor_parameter_relaxations": lambda: with_field(
        hash_spec(), "tensor_parameters", 0, relaxations=E.TensorSpecRelaxations(match_padded_shape_only=True)
    ),
    "work_unit_nodes": lambda: with_field(hash_spec(), "work_units", 0, target_nodes=ttnn.CoreCoord(0, 0)),
    "work_unit_kernels": lambda: with_field(hash_spec(), "work_units", 0, kernels=["reader"]),
    "semaphore": lambda: setattr_and_return(
        hash_spec(), "semaphores", [E.SemaphoreSpec(unique_id="sem", target_nodes=ttnn.CoreCoord(0, 0))]
    ),
    "scratchpad": lambda: setattr_and_return(
        hash_spec(), "scratchpads", [E.ScratchpadSpec(unique_id="scratch", size_per_node=64)]
    ),
}

NON_STRUCTURAL_EDITS = {
    "program_name": lambda: setattr_and_return(hash_spec(), "name", "renamed"),
    "work_unit_name": lambda: with_field(hash_spec(), "work_units", 0, name="renamed"),
    "define_order": lambda: hash_spec(compiler_options=E.KernelSpec.CompilerOptions(defines={"B": "2", "A": "1"})),
    "compile_time_arg_order": lambda: hash_spec(compile_time_args={"count": 2, "block": 1}),
}


def setattr_and_return(spec, field, value):
    setattr(spec, field, value)
    return spec


@pytest.mark.parametrize("edit", STRUCTURAL_EDITS.values(), ids=STRUCTURAL_EDITS.keys())
def test_hash_changes_with_every_structural_field(edit):
    assert E.compute_program_spec_hash(edit()) != E.compute_program_spec_hash(hash_spec())


@pytest.mark.parametrize("edit", NON_STRUCTURAL_EDITS.values(), ids=NON_STRUCTURAL_EDITS.keys())
def test_hash_ignores_names_and_dict_order(edit):
    assert E.compute_program_spec_hash(edit()) == E.compute_program_spec_hash(hash_spec())


def test_hash_ignores_relaxed_tensor_fields():
    relaxations = E.TensorSpecRelaxations(dynamic_tensor_shape=True)
    small = ttnn.TensorSpec((1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    first = with_field(hash_spec(), "tensor_parameters", 0, relaxations=relaxations)
    second = with_field(hash_spec(), "tensor_parameters", 0, relaxations=relaxations, spec=small)
    assert E.compute_program_spec_hash(first) == E.compute_program_spec_hash(second)


def test_hash_tells_an_unset_data_format_from_float32():
    # With libc++, ttsl hashes an unset data_format_metadata like DataFormat.Float32, whose value is 0.
    unset = with_field(hash_spec(), "dataflow_buffers", 0, data_format_metadata=None)
    float32 = with_field(hash_spec(), "dataflow_buffers", 0, data_format_metadata=ttnn.float32)
    assert E.compute_program_spec_hash(unset) != E.compute_program_spec_hash(float32)


def test_hash_keeps_adjacent_lists_apart():
    split = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["x"], common_runtime_arg_names=["y"])
    together = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["x", "y"])
    assert E.compute_program_spec_hash(hash_spec(runtime_arg_schema=split)) != E.compute_program_spec_hash(
        hash_spec(runtime_arg_schema=together)
    )
