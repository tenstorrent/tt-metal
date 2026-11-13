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
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp

E = ttnn.experimental

MUL_KERNELS = pathlib.Path(__file__).parent / "kernels" / "program_spec_mul"
UNARY_KERNELS = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels"
MATMUL_KERNELS = "ttnn/cpp/ttnn/operations/matmul/device/kernels"
SEMAPHORE_KERNELS = "tests/tt_metal/tt_metal/test_kernels/dataflow"
TILE_BYTES = 2 * 32 * 32  # one bfloat16 tile


def assert_close(expected, actual):
    """For the elementwise kernels: each output element must be within 2 bfloat16 ULPs of torch's."""
    assert_with_ulp(expected_result=expected, actual_result=actual, ulp_threshold=2)


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
    a, b, out, *, tile_limit=None, cores_for_tiles=None, relaxations=None, array_args=False, aliased_buffer=False
):
    """(spec, run_args) for out = a * b. Tensor parameters a, b and out are bound to a, b and out.

    The program runs on the cores for cores_for_tiles tiles (by default, the output's tile count). With array_args, the
    reader and writer take each core's tile count and first tile as the array work instead of two named arguments.
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

    if array_args:
        dm_options = dict(
            compiler_options=E.KernelSpec.CompilerOptions(defines={"USE_ARRAY_ARGS": "1"}),
            runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_array_args=[E.ArrayArgDecl(name="work", size=2)]),
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

    if array_args:
        work = {core: [num_tiles_by_core[core], start_id_by_core[core]] for core in num_tiles_by_core}
        dm_run_args = dict(runtime_array_arg_values={"work": work})
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
    assert_close(expected, ttnn.to_torch(result))


def test_add(device):
    """The binary add case of test_generic_op.cpp, with the compute kernel of test_mul adding."""
    _, a, b, out = inputs(device, (11, 9, 32, 32), seed=17)
    spec, run_args = mul_program(a, b, out)
    kernels = spec.kernels
    kernels[1].compiler_options = E.KernelSpec.CompilerOptions(defines={"BINARY_ADD": "1"})
    spec.kernels = kernels
    result = E.generic_op([a, b, out], spec, run_args)
    assert_close(ttnn.to_torch(a) + ttnn.to_torch(b), ttnn.to_torch(result))


def test_matmul(device):
    """The matmul case of test_generic_op.cpp, with the Metal 2.0 kernels of TT-NN's multi-core matmul.

    [3, 1, 320, 64] times [3, 1, 64, 128], one output tile after another, split over the grid in core groups that differ
    in tile count. The compute kernel loops over batch, Mt and Nt as one loop, so each group's compute kernel sets only
    Nt, to the group's tile count.
    """
    batch, m_tiles, k_tiles, n_tiles = 3, 10, 2, 4
    torch.manual_seed(18)
    torch_a = torch.randn((batch, 1, m_tiles * 32, k_tiles * 32), dtype=torch.bfloat16)
    torch_b = torch.randn((batch, 1, k_tiles * 32, n_tiles * 32), dtype=torch.bfloat16)
    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec((batch, 1, m_tiles * 32, n_tiles * 32), ttnn.bfloat16, ttnn.TILE_LAYOUT), device
    )
    grid = device.compute_with_storage_grid_size()
    _, all_cores, *groups_and_tiles = ttnn.split_work_to_cores(grid, batch * m_tiles * n_tiles)
    groups = [
        (f"group_{i + 1}", group, tiles)
        for i, (group, tiles) in enumerate(zip(groups_and_tiles[:2], groups_and_tiles[2:]))
        if group.num_cores() > 0
    ]

    spec = E.ProgramSpec(
        name="matmul",
        kernels=[
            E.KernelSpec(
                unique_id="reader",
                source=f"{MATMUL_KERNELS}/dataflow/reader_bmm_8bank_output_tiles_partitioned_metal2.cpp",
                dfb_bindings=[E.producer_of("in0", "in0"), E.producer_of("in1", "in1")],
                tensor_bindings=[
                    E.TensorBinding(tensor_parameter_name="in0", accessor_name="in0"),
                    E.TensorBinding(tensor_parameter_name="in1", accessor_name="in1"),
                ],
                compile_time_args={"in0_last_ktile_w": 0, "in0_last_ktile_h": 0},
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
                    runtime_arg_names=["output_tile_start_id", "num_output_tiles"],
                    common_runtime_arg_names=["Mt", "Kt", "Nt", "MtKt", "KtNt", "batch", "bcast_B", "MtNt"],
                ),
                hw_config=E.create_reader_datamovement_config(),
            ),
            E.KernelSpec(
                unique_id="writer",
                source=f"{MATMUL_KERNELS}/dataflow/writer_unary_interleaved_start_id.cpp",
                dfb_bindings=[E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_pages", "start_id"]),
                hw_config=E.create_writer_datamovement_config(),
            ),
        ]
        + [
            E.KernelSpec(
                unique_id=f"compute_{name}",
                source=f"{MATMUL_KERNELS}/compute/bmm_metal2.cpp",
                dfb_bindings=[E.consumer_of("in0", "in0"), E.consumer_of("in1", "in1"), E.producer_of("out", "out")],
                compile_time_args={"batch": 1, "Mt": 1, "Kt": k_tiles, "Nt": tiles},
                hw_config=E.ComputeHardwareConfig(
                    unpack_modes={"in0": E.UnpackMode.UnpackToSrc, "in1": E.UnpackMode.UnpackToSrc}
                ),
            )
            for name, _, tiles in groups
        ],
        dataflow_buffers=[
            E.DataflowBufferSpec(
                unique_id=name, entry_size=TILE_BYTES, num_entries=2, data_format_metadata=ttnn.bfloat16
            )
            for name in ("in0", "in1", "out")
        ],
        tensor_parameters=[
            E.TensorParameter(unique_id="in0", spec=a.spec),
            E.TensorParameter(unique_id="in1", spec=b.spec),
            E.TensorParameter(unique_id="output", spec=out.spec),
        ],
        work_units=[
            E.WorkUnitSpec(name=name, kernels=["reader", "writer", f"compute_{name}"], target_nodes=group)
            for name, group, _ in groups
        ],
    )

    start_id, num_tiles, start = {}, {}, 0
    for core in ttnn.corerange_to_cores(all_cores, None, False):
        tiles = next(tiles for _, group, tiles in groups if group.contains(core))
        start_id[core], num_tiles[core] = start, tiles
        start += tiles
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel="reader",
                runtime_arg_values={"output_tile_start_id": start_id, "num_output_tiles": num_tiles},
                common_runtime_arg_values={
                    "Mt": m_tiles,
                    "Kt": k_tiles,
                    "Nt": n_tiles,
                    "MtKt": m_tiles * k_tiles,
                    "KtNt": k_tiles * n_tiles,
                    "batch": batch,
                    "bcast_B": 0,
                    "MtNt": m_tiles * n_tiles,
                },
            ),
            E.KernelRunArgs(kernel="writer", runtime_arg_values={"num_pages": num_tiles, "start_id": start_id}),
        ],
        tensor_args={"in0": a, "in1": b, "output": out},
    )
    result = E.generic_op([a, b, out], spec, run_args)
    assert_with_pcc(torch.matmul(torch_a.float(), torch_b.float()), ttnn.to_torch(result).float(), 0.999)


def exp_program(input_tensor, output_tensor, compute_threads=1):
    """(spec, run_args) for output = exp(input), with the tiles split over the cores.

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


def test_the_exp_program(device, expect_error):
    torch_input, input_tensor, output_tensor = exp_inputs(device, 8)
    spec, run_args = exp_program(input_tensor, output_tensor)
    result = E.generic_op([input_tensor, output_tensor], spec, run_args)
    assert_close(torch.exp(torch_input), ttnn.to_torch(result))

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
    assert_close(torch.exp(torch_input), ttnn.to_torch(result))


def test_two_compute_threads_on_quasar(device):
    if "quasar" not in ttnn.get_arch_name():
        pytest.skip("A kernel has more than one thread on Quasar only")
    num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    torch_input, input_tensor, output_tensor = exp_inputs(device, 2 * num_cores)
    spec, run_args = exp_program(input_tensor, output_tensor, compute_threads=2)
    for _ in range(2):
        result = E.generic_op([input_tensor, output_tensor], spec, run_args)
        assert_close(torch.exp(torch_input), ttnn.to_torch(result))


def test_sharded_typecast_on_borrowed_buffers(device):
    """Both buffers sit on the tensors' own L1 shards, as in ttnn's sharded typecast."""
    if "quasar" in ttnn.get_arch_name() and os.environ.get("TT_METAL_SIMULATOR"):
        pytest.skip("ttsim's Quasar model cannot pack bfloat16 into float32 yet")
    shape = (1, 1, 64, 64)
    memory_config = ttnn.create_sharded_memory_config(
        shape, core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.HEIGHT
    )

    def sharded_tensors(seed):
        torch.manual_seed(seed)
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        # The output starts as a known value, so a program that wrote to another tensor's shard would leave it.
        output_tensor = ttnn.from_torch(
            torch.full(shape, 7.0),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )
        return torch_input, input_tensor, output_tensor

    torch_input, input_tensor, output_tensor = sharded_tensors(seed=0)
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
    result = ttnn.to_torch(E.generic_op([input_tensor, output_tensor], spec, run_args))
    assert result.dtype == torch.float32
    assert torch.equal(result, torch_input.float())

    # A cache hit with new tensors, made while the first ones still exist so that their shards are elsewhere: the
    # buffers move to the new tensors' shards.
    torch_input2, input_tensor2, output_tensor2 = sharded_tensors(seed=1)
    run_args.tensor_args = {"input": input_tensor2, "output": output_tensor2}
    entries = device.num_program_cache_entries()
    result = ttnn.to_torch(E.generic_op([input_tensor2, output_tensor2], spec, run_args))
    assert device.num_program_cache_entries() == entries
    assert torch.equal(result, torch_input2.float())


def test_array_args_and_aliased_buffers(device):
    shape = (1, 1, 96, 64)
    expected, a, b, out = inputs(device, shape)
    spec, run_args = mul_program(a, b, out, array_args=True, aliased_buffer=True)
    assert_close(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)))

    # A cache hit with new tensors and one tile fewer for the same cores, as in
    # test_later_calls_reuse_the_program_with_new_tensors_and_values: the last tile keeps the output's starting value
    # only if the new array values reached the device.
    expected, a2, b2, _ = inputs(device, shape, seed=1)
    initial = torch.full(shape, 7.0, dtype=torch.bfloat16)
    out2 = ttnn.from_torch(initial, layout=ttnn.TILE_LAYOUT, device=device)
    _, run_args2 = mul_program(a2, b2, out2, array_args=True, aliased_buffer=True, tile_limit=5)
    entries = device.num_program_cache_entries()
    result = tile_order(ttnn.to_torch(E.generic_op([a2, b2, out2], spec, run_args2)))
    assert device.num_program_cache_entries() == entries
    assert_close(tile_order(expected)[: 5 * 1024], result[: 5 * 1024])
    assert torch.equal(result[5 * 1024 :], tile_order(initial)[5 * 1024 :])


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
        work_units=[E.WorkUnitSpec(name="loopback", kernels=["producer", "consumer"], target_nodes=node)],
    )
    # The producer raises the semaphore by 1 and the consumer waits until it equals 1. If a call started from the value
    # the previous one left, the third call would start at 2 and never finish.
    entries = device.num_program_cache_entries()
    for _ in range(3):
        E.generic_op([out], spec, E.ProgramRunArgs())
    # The calls only enqueue the program: wait for them, so that one that never finishes fails this test.
    ttnn.synchronize_device(device)
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
    assert_close(tile_order(expected)[: 3 * 1024], result[: 3 * 1024])
    assert torch.equal(result[3 * 1024 :], tile_order(initial)[3 * 1024 :])


def tile_order(tensor):
    """The elements of a (1, 1, h, w) tensor in tile order, so that the first n tiles are its first n * 1024."""
    height, width = tensor.shape[-2:]
    return tensor.reshape(height // 32, 32, width // 32, 32).permute(0, 2, 1, 3).flatten()


def num_tiles_as_a_common_runtime_arg(spec, run_args, value):
    """Makes num_tiles, in each kernel of a mul_program spec, a common runtime argument with this value on every core."""
    kernels = spec.kernels
    for kernel in kernels:
        schema = kernel.runtime_arg_schema
        schema.runtime_arg_names = [name for name in schema.runtime_arg_names if name != "num_tiles"]
        schema.common_runtime_arg_names = ["num_tiles"]
    spec.kernels = kernels
    kernel_run_args = run_args.kernel_run_args
    for args in kernel_run_args:
        values = args.runtime_arg_values
        del values["num_tiles"]
        args.runtime_arg_values = values
        args.common_runtime_arg_values = {"num_tiles": value}
    run_args.kernel_run_args = kernel_run_args


def test_later_calls_write_new_common_runtime_args(device):
    # num_tiles as a common runtime argument, one value for every core. The 4 tiles split evenly over the cores (one
    # or two each, by the grid), so one value fits them all.
    expected, a, b, out = inputs(device, seed=16)
    spec, run_args = mul_program(a, b, out)
    counts = set(kernel_run_args(run_args, "compute").runtime_arg_values["num_tiles"].values())
    assert len(counts) == 1
    num_tiles_as_a_common_runtime_arg(spec, run_args, counts.pop())
    assert_close(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)))
    entries = device.num_program_cache_entries()

    # A cache hit with no tiles for any core: the output keeps its starting value only if the new value reached the
    # cores.
    initial = torch.full((1, 1, 64, 64), 7.0, dtype=torch.bfloat16)
    out2 = ttnn.from_torch(initial, layout=ttnn.TILE_LAYOUT, device=device)
    spec2, run_args2 = mul_program(a, b, out2)
    num_tiles_as_a_common_runtime_arg(spec2, run_args2, 0)
    result = ttnn.to_torch(E.generic_op([a, b, out2], spec2, run_args2))
    assert device.num_program_cache_entries() == entries
    assert torch.equal(result, initial)


def test_other_array_lengths_compile_a_program_of_their_own(device):
    expected, a, b, out = inputs(device, seed=9)
    spec, run_args = mul_program(a, b, out, array_args=True)
    entries = device.num_program_cache_entries()
    E.generic_op([a, b, out], spec, run_args)

    # Arrays of three values. Metal places each array by the length it compiled, so the program compiled for two
    # values would reject these: this call compiles a program of its own.
    kernels = spec.kernels
    for index in (0, 2):
        kernels[index].runtime_arg_schema.runtime_array_args = [E.ArrayArgDecl(name="work", size=3)]
    spec.kernels = kernels
    for kernel in ("reader", "writer"):
        args = kernel_run_args(run_args, kernel)
        work = {core: values + [0] for core, values in args.runtime_array_arg_values["work"].items()}
        args.runtime_array_arg_values = {"work": work}
        replace_kernel_run_args(run_args, kernel, args)
    result = E.generic_op([a, b, out], spec, run_args)
    assert device.num_program_cache_entries() == entries + 2
    assert_close(expected, ttnn.to_torch(result))


def test_dfb_run_overrides_stay_in_effect_on_later_calls(device, expect_error):
    expected, a, b, out = inputs(device, seed=10)
    spec, run_args = mul_program(a, b, out, aliased_buffer=True)
    # Buffers that share memory keep one total size, so Metal rejects a call that resizes only one of them.
    run_args.dfb_run_overrides = [E.DFBRunOverrides(dfb=name, num_entries=4) for name in ("in_a", "in_a_view")]
    assert_close(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)))
    run_args.dfb_run_overrides = []
    assert_close(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)))

    # The sizes stayed: in_a alone can take the size it already has, but not its original size, which in_a_view no
    # longer has.
    run_args.dfb_run_overrides = [E.DFBRunOverrides(dfb="in_a", num_entries=4)]
    assert_close(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)))
    run_args.dfb_run_overrides = [E.DFBRunOverrides(dfb="in_a", num_entries=2)]
    with expect_error(RuntimeError, "was not overridden in alias group"):
        E.generic_op([a, b, out], spec, run_args)


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
    assert_close(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)))


def test_prepare_fills_the_program_cache(device):
    expected, a, b, out = inputs(device, seed=4)
    spec, run_args = mul_program(a, b, out)
    entries = device.num_program_cache_entries()
    E.prepare_generic_op([a, b, out], spec, run_args)
    assert device.num_program_cache_entries() == entries + 1
    result = E.generic_op([a, b, out], spec, run_args)
    assert device.num_program_cache_entries() == entries + 1
    assert_close(expected, ttnn.to_torch(result))


def test_a_compilation_failure_leaves_the_program_cache_unchanged(device, expect_error):
    expected, a, b, out = inputs(device, seed=16)
    spec, run_args = mul_program(a, b, out)
    kernels = spec.kernels
    kernels[1].source = E.KernelSpec.SourceCode(code="this does not compile")
    spec.kernels = kernels
    entries = device.num_program_cache_entries()
    with expect_error(RuntimeError, "build failed"):
        E.prepare_generic_op([a, b, out], spec, run_args)
    assert device.num_program_cache_entries() == entries
    with expect_error(RuntimeError, "build failed"):
        E.generic_op([a, b, out], spec, run_args)
    assert device.num_program_cache_entries() == entries

    # The same spec with a kernel that compiles still runs.
    kernels[1].source = str(MUL_KERNELS / "compute.cpp")
    spec.kernels = kernels
    assert_close(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)))
    assert device.num_program_cache_entries() == entries + 1


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
    assert_close(expected, ttnn.to_torch(result))


def test_a_tensor_bound_to_two_parameters_is_listed_twice(device, expect_error):
    torch.manual_seed(7)
    torch_a = torch.randn((1, 1, 64, 64), dtype=torch.bfloat16)
    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.allocate_tensor_on_device(a.spec, device)
    spec, run_args = mul_program(a, a, out)
    for _ in range(2):  # the first call and a cache hit
        result = E.generic_op([a, a, out], spec, run_args)
        assert_close(torch_a * torch_a, ttnn.to_torch(result))
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


@pytest.mark.parametrize("cache_hit", [False, True], ids=["first_call", "later_call"])
def test_tensor_args_must_match_the_parameters(device, cache_hit, expect_error):
    _, a, b, out = inputs(device)
    spec, run_args = mul_program(a, b, out)
    if cache_hit:
        E.generic_op([a, b, out], spec, run_args)
        # These checks run on every call, so the later calls run without validate_program_args.
        ttnn.CONFIG.validate_program_args = False
    run_args.tensor_args = {"a": a, "b": b}
    with expect_error(RuntimeError, "tensor parameter 'out' has no tensor"):
        E.generic_op([a, b, out], spec, run_args)
    run_args.tensor_args = {"a": a, "b": b, "out": out, "c": out}
    with expect_error(
        RuntimeError, "run_args.tensor_args has a tensor for 'c', which the ProgramSpec does not declare"
    ):
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
        # These checks run on every call, so the later calls run without validate_program_args.
        ttnn.CONFIG.validate_program_args = False

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


@pytest.mark.parametrize("duplicate", ["kernel", "kernel_with_other_arguments", "tensor_parameter"])
def test_a_name_declared_twice_gets_metals_error(device, duplicate, expect_error):
    """When a kernel or tensor parameter is declared twice, generic_op checks it once, against its first declaration,
    so when the run args fit that declaration, the call gets past generic_op's checks to Metal's, which names the
    duplicate."""
    _, a, b, out = inputs(device)
    spec, run_args = mul_program(a, b, out)
    if duplicate == "kernel":
        spec.kernels = spec.kernels + [spec.kernels[0]]
        message = "Duplicate KernelSpec name 'reader'"
    elif duplicate == "kernel_with_other_arguments":
        # A first declaration of the reader without runtime arguments, which run args without a reader entry fit.
        reader = spec.kernels[0]
        reader.runtime_arg_schema = E.KernelSpec.RuntimeArgSchema()
        spec.kernels = [reader] + spec.kernels
        run_args.kernel_run_args = [entry for entry in run_args.kernel_run_args if entry.kernel != "reader"]
        message = "Duplicate KernelSpec name 'reader'"
    else:
        spec.tensor_parameters = spec.tensor_parameters + [spec.tensor_parameters[0]]
        message = "Duplicate TensorParameter name 'a'"
    with expect_error(RuntimeError, message):
        E.generic_op([a, b, out], spec, run_args)


@pytest.mark.parametrize("cache_hit", [False, True], ids=["first_call", "later_call"])
def test_arrays_are_checked_on_every_call(device, cache_hit, expect_error):
    _, a, b, out = inputs(device)
    spec, run_args = mul_program(a, b, out, array_args=True)
    # The reader also declares a common array, which it does not read.
    kernels = spec.kernels
    kernels[0].runtime_arg_schema.common_runtime_array_args = [E.ArrayArgDecl(name="table", size=2)]
    spec.kernels = kernels
    reader = kernel_run_args(run_args, "reader")
    reader.common_runtime_array_arg_values = {"table": [1, 2]}
    replace_kernel_run_args(run_args, "reader", reader)
    if cache_hit:
        E.generic_op([a, b, out], spec, run_args)
        # Metal checks a later call's arrays only as it writes them, so these checks matter most without
        # validate_program_args.
        ttnn.CONFIG.validate_program_args = False

    work = reader.runtime_array_arg_values["work"]
    core = next(iter(work))

    def run_with(arrays, common_arrays):
        reader.runtime_array_arg_values = arrays
        reader.common_runtime_array_arg_values = common_arrays
        replace_kernel_run_args(run_args, "reader", reader)
        E.generic_op([a, b, out], spec, run_args)

    table = {"table": [1, 2]}
    with expect_error(RuntimeError, r"declares the runtime arrays \[work\], but its run args give \[\]"):
        run_with({}, table)
    with expect_error(RuntimeError, r"declares the common runtime arrays \[table\], but its run args give \[other\]"):
        run_with({"work": work}, {"other": [1, 2]})
    with expect_error(RuntimeError, "declares common runtime array 'table' with 2 values, but its run args give 3"):
        run_with({"work": work}, {"table": [1, 2, 3]})
    with expect_error(RuntimeError, "declares runtime array 'work' with 2 values, but its run args give 3 on node"):
        run_with({"work": {**work, core: work[core] + [0]}}, table)


LEFT_OUT_CORE_ERRORS = {
    False: "runtime argument 'num_tiles' does not have exactly one value",
    True: "runtime array 'work' does not have exactly one value",
}


def leave_out_a_core(run_args, array_args):
    """Drops one core from the compute kernel's num_tiles, or with array_args, from the reader's array work."""
    if array_args:
        reader = kernel_run_args(run_args, "reader")
        work = reader.runtime_array_arg_values["work"]
        work.pop(next(iter(work)))
        reader.runtime_array_arg_values = {"work": work}
        replace_kernel_run_args(run_args, "reader", reader)
    else:
        compute = kernel_run_args(run_args, "compute")
        num_tiles = compute.runtime_arg_values["num_tiles"]
        num_tiles.pop(next(iter(num_tiles)))
        compute.runtime_arg_values = {"num_tiles": num_tiles}
        replace_kernel_run_args(run_args, "compute", compute)


@pytest.mark.parametrize("array_args", [False, True], ids=["named_args", "array_args"])
def test_a_core_left_out_of_the_first_call(device, array_args, expect_error):
    # Checked before the kernels compile, with validate_program_args off too.
    ttnn.CONFIG.validate_program_args = False
    _, a, b, out = inputs(device)
    spec, run_args = mul_program(a, b, out, array_args=array_args)
    leave_out_a_core(run_args, array_args)
    with expect_error(RuntimeError, LEFT_OUT_CORE_ERRORS[array_args]):
        E.generic_op([a, b, out], spec, run_args)


@pytest.mark.parametrize("array_args", [False, True], ids=["named_args", "array_args"])
@pytest.mark.parametrize("validate", [True, False], ids=["validate_on", "validate_off"])
def test_a_core_left_out_of_a_later_call(device, validate, array_args, expect_error):
    expected, a, b, out = inputs(device)
    spec, run_args = mul_program(a, b, out, array_args=array_args)
    E.generic_op([a, b, out], spec, run_args)

    ttnn.CONFIG.validate_program_args = validate
    leave_out_a_core(run_args, array_args)
    if validate:
        with expect_error(RuntimeError, LEFT_OUT_CORE_ERRORS[array_args]):
            E.generic_op([a, b, out], spec, run_args)
    else:
        # The left-out core keeps the values from the first call.
        assert_close(expected, ttnn.to_torch(E.generic_op([a, b, out], spec, run_args)))


# ---------------------------------------------------------------------------------------------------------- mesh form


def mesh_range(device):
    return ttnn.MeshCoordinateRange(device.shape)


def one_device(column):
    return ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, column), ttnn.MeshCoordinate(0, column))


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_a_range_over_a_device_without_a_tensor_is_rejected(mesh_device, expect_error):
    """A program runs on every device of its mesh range, so each of them must hold every tensor."""
    torch.manual_seed(19)
    first_device_only = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(0)], mesh_shape_override=ttnn.MeshShape(1, 1)
        ),
    )
    input_tensor = ttnn.from_torch(
        torch.rand([1, 1, 32, 32]).to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=first_device_only,
    )
    output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, mesh_device)
    spec, run_args = exp_program(input_tensor, output_tensor)
    whole_mesh = mesh_range(mesh_device)
    with expect_error(RuntimeError, r"a program runs on device .*, which does not hold io_tensors\[0\]"):
        E.generic_op(
            [input_tensor, output_tensor],
            E.MeshProgramSpec(programs=[(whole_mesh, spec)]),
            E.MeshProgramRunArgs(programs=[(whole_mesh, run_args)]),
        )


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_mesh_form_runs_a_program_of_its_own_on_each_device(mesh_device):
    """Each range runs its own program with its own run args, on a cache hit too, and the ranges are part of the key."""
    first, second = one_device(0), one_device(1)
    _, a, b, out = inputs(mesh_device, seed=21)
    torch_a, torch_b = (ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]) for tensor in (a, b))
    mul_spec, mul_args = mul_program(a, b, out)
    add_spec, add_args = mul_program(a, b, out)
    kernels = add_spec.kernels
    kernels[1].compiler_options = E.KernelSpec.CompilerOptions(defines={"BINARY_ADD": "1"})
    add_spec.kernels = kernels

    def run(programs, run_args, output):
        result = E.generic_op(
            [a, b, output], E.MeshProgramSpec(programs=programs), E.MeshProgramRunArgs(programs=run_args)
        )
        return [tile_order(ttnn.to_torch(shard)) for shard in ttnn.get_device_tensors(result)]

    mul, add = tile_order(torch_a * torch_b), tile_order(torch_a + torch_b)
    on_first, on_second = run([(first, mul_spec), (second, add_spec)], [(first, mul_args), (second, add_args)], out)
    assert_close(mul, on_first)
    assert_close(add, on_second)

    # A cache hit, with the run args listed in the other order and only 3 tiles for the second device's program.
    initial = torch.full((1, 1, 64, 64), 7.0, dtype=torch.bfloat16)
    out2 = ttnn.from_torch(initial, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    _, mul_args2 = mul_program(a, b, out2)
    _, add_args2 = mul_program(a, b, out2, tile_limit=3)
    entries = mesh_device.num_program_cache_entries()
    on_first, on_second = run([(first, mul_spec), (second, add_spec)], [(second, add_args2), (first, mul_args2)], out2)
    assert mesh_device.num_program_cache_entries() == entries
    assert_close(mul, on_first)
    assert_close(add[: 3 * 1024], on_second[: 3 * 1024])
    assert torch.equal(on_second[3 * 1024 :], tile_order(initial)[3 * 1024 :])

    # The same programs on swapped devices key a new entry, since each program's range is part of the key.
    on_first, on_second = run([(second, mul_spec), (first, add_spec)], [(second, mul_args), (first, add_args)], out)
    assert mesh_device.num_program_cache_entries() == entries + 1
    assert_close(add, on_first)
    assert_close(mul, on_second)


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_the_program_spec_form_runs_on_the_devices_of_its_tensors(mesh_device, expect_error):
    """With its tensors on one device of the mesh, the program runs on that device only, and a later call with the
    same key but a tensor on another device is rejected."""
    expected, a, b, _ = inputs(mesh_device, seed=22)
    initial = torch.full((1, 1, 64, 64), 7.0, dtype=torch.bfloat16)
    out = ttnn.from_torch(initial, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    (_, a1), (_, b1), (out0, out1) = (ttnn.get_device_tensors(tensor) for tensor in (a, b, out))
    spec, run_args = mul_program(a1, b1, out1)
    entries = mesh_device.num_program_cache_entries()
    assert_close(expected, ttnn.to_torch(E.generic_op([a1, b1, out1], spec, run_args)))
    assert torch.equal(ttnn.to_torch(out0), initial)

    # The MeshProgramSpec form over the tensors' device keys the same entry.
    E.generic_op(
        [a1, b1, out1],
        E.MeshProgramSpec(programs=[(one_device(1), spec)]),
        E.MeshProgramRunArgs(programs=[(one_device(1), run_args)]),
    )
    assert mesh_device.num_program_cache_entries() == entries + 1

    # a1 decides the devices, so this call keys the same entry, but its output is on the other device. Every call checks
    # this, with validate_program_args off too.
    ttnn.CONFIG.validate_program_args = False
    _, run_args0 = mul_program(a1, b1, out0)
    with expect_error(RuntimeError, r"a program runs on device .*, which does not hold io_tensors\[2\]"):
        E.generic_op([a1, b1, out0], spec, run_args0)
    assert mesh_device.num_program_cache_entries() == entries + 1


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_one_program_runs_on_every_device_of_its_tensors(mesh_device):
    """With its tensors on both devices, each holding data of its own, the program runs on both, on a cache hit too,
    and the MeshProgramSpec form over the whole mesh keys the same entry."""
    initial = torch.full((1, 1, 64, 64), 7.0, dtype=torch.bfloat16)
    # Each call's tensors stay allocated, so the next call's are at other addresses, which a cache hit must bind.
    tensors_of_each_call = []

    def call(seed, *, tile_limit=None, mesh_form=False):
        """Runs a * b, with a and b split over the devices. Returns the expected and actual output of each device, in
        tile order, and the number of entries generic_op added to the program cache (creating the tensors adds some
        of its own)."""
        torch.manual_seed(seed)
        torch_a, torch_b = (torch.randn((2, 1, 64, 64), dtype=torch.bfloat16) for _ in range(2))
        by_device = ttnn.shard_tensor_to_mesh_mapper(mesh_device, 0)
        a, b = (
            ttnn.from_torch(tensor, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=by_device)
            for tensor in (torch_a, torch_b)
        )
        out = ttnn.from_torch(initial, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        tensors_of_each_call.append((a, b, out))
        spec, run_args = mul_program(a, b, out, tile_limit=tile_limit)
        if mesh_form:
            whole_mesh = mesh_range(mesh_device)
            spec = E.MeshProgramSpec(programs=[(whole_mesh, spec)])
            run_args = E.MeshProgramRunArgs(programs=[(whole_mesh, run_args)])
        entries = mesh_device.num_program_cache_entries()
        result = E.generic_op([a, b, out], spec, run_args)
        new_entries = mesh_device.num_program_cache_entries() - entries
        expected = [tile_order(product) for product in (torch_a * torch_b).split(1)]
        return expected, [tile_order(ttnn.to_torch(shard)) for shard in ttnn.get_device_tensors(result)], new_entries

    expected, outputs, new_entries = call(seed=23)
    assert new_entries == 1
    assert len(outputs) == 2
    for expected_output, output in zip(expected, outputs):
        assert_close(expected_output, output)

    # A cache hit with new tensors and only 3 tiles.
    expected, outputs, new_entries = call(seed=24, tile_limit=3)
    assert new_entries == 0
    for expected_output, output in zip(expected, outputs):
        assert_close(expected_output[: 3 * 1024], output[: 3 * 1024])
        assert torch.equal(output[3 * 1024 :], tile_order(initial)[3 * 1024 :])

    expected, outputs, new_entries = call(seed=25, mesh_form=True)
    assert new_entries == 0
    for expected_output, output in zip(expected, outputs):
        assert_close(expected_output, output)


def test_mesh_form(device):
    expected, a, b, out = inputs(device, seed=8)
    spec, run_args = mul_program(a, b, out)
    mesh_spec = E.MeshProgramSpec(programs=[(mesh_range(device), spec)])
    mesh_run_args = E.MeshProgramRunArgs(programs=[(mesh_range(device), run_args)])
    assert mesh_run_args.programs[0][1].tensor_args["a"] is a
    assert_close(expected, ttnn.to_torch(E.generic_op([a, b, out], mesh_spec, mesh_run_args)))

    # A cache hit with new tensors and one tile fewer, as in
    # test_later_calls_reuse_the_program_with_new_tensors_and_values.
    expected, a2, b2, _ = inputs(device, seed=15)
    initial = torch.full((1, 1, 64, 64), 7.0, dtype=torch.bfloat16)
    out2 = ttnn.from_torch(initial, layout=ttnn.TILE_LAYOUT, device=device)
    _, run_args2 = mul_program(a2, b2, out2, tile_limit=3)
    mesh_run_args.programs = [(mesh_range(device), run_args2)]
    entries = device.num_program_cache_entries()
    result = tile_order(ttnn.to_torch(E.generic_op([a2, b2, out2], mesh_spec, mesh_run_args)))
    assert device.num_program_cache_entries() == entries
    assert_close(tile_order(expected)[: 3 * 1024], result[: 3 * 1024])
    assert torch.equal(result[3 * 1024 :], tile_order(initial)[3 * 1024 :])

    E.prepare_generic_op([a2, b2, out2], mesh_spec, mesh_run_args)
    assert device.num_program_cache_entries() == entries


def test_the_two_forms_share_a_cache_entry(device):
    """The ProgramSpec form runs as the MeshProgramSpec form would on the tensors' devices, so it keys the same way."""
    expected, a, b, out = inputs(device, seed=20)
    spec, run_args = mul_program(a, b, out)
    entries = device.num_program_cache_entries()
    E.generic_op([a, b, out], spec, run_args)
    result = E.generic_op(
        [a, b, out],
        E.MeshProgramSpec(programs=[(mesh_range(device), spec)]),
        E.MeshProgramRunArgs(programs=[(mesh_range(device), run_args)]),
    )
    assert device.num_program_cache_entries() == entries + 1
    assert_close(expected, ttnn.to_torch(result))


def test_a_mesh_run_args_subclass_setattr_cannot_drop_the_tensors(device):
    class MeshRunArgsIgnoringWrites(E.MeshProgramRunArgs):
        def __setattr__(self, name, value):
            pass

    class MeshRunArgsIgnoringPrivateWrites(E.MeshProgramRunArgs):
        def __setattr__(self, name, value):
            if not name.startswith("_"):
                super().__setattr__(name, value)

    _, a, b, out = inputs(device)
    _, run_args = mul_program(a, b, out)
    mesh_run_args = MeshRunArgsIgnoringWrites(programs=[(mesh_range(device), run_args)])
    assert mesh_run_args.programs[0][1].tensor_args["a"] is a

    # An assignment to programs reaches the setter, which stores the tensors' owners itself.
    mesh_run_args = MeshRunArgsIgnoringPrivateWrites()
    mesh_run_args.programs = [(mesh_range(device), run_args)]
    assert mesh_run_args.programs[0][1].tensor_args["a"] is a


def test_a_mesh_run_args_keeps_its_tensor_owners(device, expect_error):
    _, a, b, out = inputs(device)
    _, run_args = mul_program(a, b, out)
    mesh_run_args = E.MeshProgramRunArgs(programs=[(mesh_range(device), run_args)])
    with expect_error(AttributeError, "'MeshProgramRunArgs' object attribute '_tensor_args' is read-only"):
        del mesh_run_args._tensor_args
    with expect_error(AttributeError, "'MeshProgramRunArgs' object attribute '__dict__' is read-only"):
        mesh_run_args.__dict__ = {}
    with expect_error(AttributeError, "'MeshProgramRunArgs' object attribute '_tensor_args' is read-only"):
        mesh_run_args._tensor_args = []
    with expect_error(AttributeError, "'MeshProgramRunArgs' object attribute '__dict__' is read-only"):
        del mesh_run_args.__dict__
    with expect_error(AttributeError, "'MeshProgramRunArgs' object has no attribute 'program'"):
        mesh_run_args.program = []
    assert mesh_run_args.programs[0][1].tensor_args["a"] is a


def test_a_mesh_run_args_takes_only_run_args(expect_error):
    whole = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 0))
    with expect_error(TypeError, r"programs\[0\] holds a ProgramSpec, not a ttnn.experimental.ProgramRunArgs"):
        E.MeshProgramRunArgs(programs=[(whole, E.ProgramSpec(name="spec"))])
    mesh_run_args = E.MeshProgramRunArgs()
    with expect_error(TypeError, r"programs\[1\] holds a NoneType, not a ttnn.experimental.ProgramRunArgs"):
        mesh_run_args.programs = [(whole, E.ProgramRunArgs()), (whole, None)]
    assert mesh_run_args.programs == []


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
    past_the_mesh = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1))
    with expect_error(RuntimeError, "mesh range .* reaches outside the MeshDevice"):
        E.generic_op(
            [a, b, out],
            E.MeshProgramSpec(programs=[(past_the_mesh, spec)]),
            E.MeshProgramRunArgs(programs=[(past_the_mesh, run_args)]),
        )
    with expect_error(RuntimeError, r"have different numbers of programs \(1 and 2\)"):
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


def hash_schema(names=("x", "y"), arrays=(("z", 2),), common_arrays=()):
    return E.KernelSpec.RuntimeArgSchema(
        runtime_arg_names=list(names),
        runtime_array_args=[E.ArrayArgDecl(name=name, size=size) for name, size in arrays],
        common_runtime_array_args=[E.ArrayArgDecl(name=name, size=size) for name, size in common_arrays],
    )


def hash_spec(**kernel_fields):
    """A host-only spec for hash tests; kernel_fields replace fields of its first kernel."""
    reader = dict(
        unique_id="reader",
        source="reader.cpp",
        hw_config=E.create_reader_datamovement_config(),
        compile_time_args={"block": 1, "count": 2},
        compile_time_array_args={"sizes": [1, 2], "steps": [4]},
        compiler_options=E.KernelSpec.CompilerOptions(defines={"A": "1", "B": "2"}),
        dfb_bindings=[E.producer_of("in", "in")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="src")],
        runtime_arg_schema=hash_schema(),
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
    "runtime_arg_names": lambda: hash_spec(runtime_arg_schema=hash_schema(names=("x", "w"))),
    "compile_time_array_value": lambda: hash_spec(compile_time_array_args={"sizes": [1, 3], "steps": [4]}),
    "compile_time_array_name": lambda: hash_spec(compile_time_array_args={"sizes": [1, 2], "other": [4]}),
    "runtime_array_length": lambda: hash_spec(runtime_arg_schema=hash_schema(arrays=[("z", 3)])),
    "runtime_array_name": lambda: hash_spec(runtime_arg_schema=hash_schema(arrays=[("w", 2)])),
    "common_runtime_array": lambda: hash_spec(runtime_arg_schema=hash_schema(common_arrays=[("w", 2)])),
    "dfb_accessor": lambda: hash_spec(dfb_bindings=[E.producer_of("in", "other")]),
    "dfb_access_pattern": lambda: hash_spec(dfb_bindings=[E.blocked_producer_of("in", "in", 2)]),
    "tensor_accessor": lambda: hash_spec(
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="other")]
    ),
    "hw_config": lambda: hash_spec(hw_config=E.create_writer_datamovement_config()),
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
    "compile_time_array_order": lambda: hash_spec(compile_time_array_args={"steps": [4], "sizes": [1, 2]}),
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


@pytest.mark.parametrize("first, second", [([4, 32], [9, 14]), ([1, 63], [2, 0]), ([3, 12, 0], [8, 0, 0])])
def test_hash_tells_apart_compile_time_arrays_that_metal_hashes_alike(first, second):
    # Metal's std::hash for an array value gives each pair one hash; the values are compiled into the kernel.
    first_spec = hash_spec(compile_time_array_args={"sizes": first, "steps": [4]})
    second_spec = hash_spec(compile_time_array_args={"sizes": second, "steps": [4]})
    assert E.compute_program_spec_hash(first_spec) != E.compute_program_spec_hash(second_spec)


def test_hash_keeps_adjacent_lists_apart():
    split = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["x"], common_runtime_arg_names=["y"])
    together = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["x", "y"])
    assert E.compute_program_spec_hash(hash_spec(runtime_arg_schema=split)) != E.compute_program_spec_hash(
        hash_spec(runtime_arg_schema=together)
    )
