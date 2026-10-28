# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Python classes for the Metal 2.0 ProgramSpec and ProgramRunArgs types (ttnn.experimental).

Each class binds the C++ struct with the same name and fields. These tests need no device, apart from
the ones that bind tensors.
"""

import pathlib
import sys

import pytest
import torch

import ttnn

E = ttnn.experimental

TENSOR_SPEC = ttnn.TensorSpec(ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT)

# Every class and the fields Python binds. A field Metal adds to one of these types fails the ttnn build
# (program_spec.cpp counts each type's fields) until it is bound here or noted as C++ only, except for
# KernelAdvancedOptions and DFBAdvancedOptions, which Python binds only in part.
FIELDS = {
    E.TensorSpecRelaxations: {
        "match_padded_shape_only",
        "dynamic_tensor_shape",
        "relax_logical_rank",
        "match_page_size",
    },
    E.TensorParameter: {"unique_id", "spec", "relaxations"},
    E.DFBAdvancedOptions: {"alias_with"},
    E.DataflowBufferSpec: {
        "unique_id",
        "entry_size",
        "num_entries",
        "data_format_metadata",
        "tile_format_metadata",
        "borrowed_from",
        "advanced_options",
    },
    E.SemaphoreSpec: {"unique_id", "target_nodes"},
    E.ScratchpadSpec: {"unique_id", "size_per_node", "data_format_metadata", "tile_format_metadata"},
    E.DataMovementHardwareConfig.DataMovement1XXConfig: {"processor", "noc", "noc_mode", "dedicated_noc_sharing"},
    E.DataMovementHardwareConfig.DataMovement2XXConfig: {
        "disable_dfb_implicit_sync_for",
        "disable_dfb_implicit_sync_for_all",
    },
    E.DataMovementHardwareConfig: {"config_1xx", "config_2xx"},
    E.ComputeHardwareConfig.Compute1XXConfig: {"bfp_pack_precision_mode"},
    E.ComputeHardwareConfig.Compute12XConfig: {"enable_pack_rvv"},
    E.ComputeHardwareConfig.Compute2XXConfig: {"enable_unpack_rvv"},
    E.ComputeHardwareConfig: {
        "fpu_math_fidelity",
        "sfpu_precision_mode",
        "enable_32_bit_dest",
        "double_buffer_dest",
        "unpack_modes",
        "config_1xx",
        "config_12x",
        "config_2xx",
    },
    E.KernelSpec.SourceCode: {"code"},
    E.KernelSpec.CompilerOptions: {"include_paths", "defines", "opt_level"},
    E.KernelSpec.RuntimeArgSchema: {"runtime_arg_names", "common_runtime_arg_names"},
    E.DFBBinding: {"dfb_spec_name", "accessor_name", "endpoint_type", "access_pattern", "block_size"},
    E.SemaphoreBinding: {"semaphore_spec_name", "accessor_name"},
    E.ScratchpadBinding: {"scratchpad_spec_name", "accessor_name"},
    E.TensorBinding: {"tensor_parameter_name", "accessor_name"},
    E.KernelAdvancedOptions: {"num_runtime_varargs", "num_common_runtime_varargs"},
    E.KernelSpec: {
        "unique_id",
        "source",
        "num_threads",
        "compiler_options",
        "dfb_bindings",
        "semaphore_bindings",
        "scratchpad_bindings",
        "tensor_bindings",
        "compile_time_args",
        "runtime_arg_schema",
        "hw_config",
        "advanced_options",
    },
    E.WorkUnitSpec: {"name", "kernels", "target_nodes"},
    E.ProgramSpec: {
        "name",
        "kernels",
        "dataflow_buffers",
        "semaphores",
        "scratchpads",
        "tensor_parameters",
        "work_units",
    },
    E.AdvancedKernelRunArgs: {"runtime_varargs", "common_runtime_varargs"},
    E.KernelRunArgs: {"kernel", "runtime_arg_values", "common_runtime_arg_values", "advanced_options"},
    E.DFBRunOverrides: {"dfb", "entry_size", "num_entries"},
    E.ProgramRunArgs: {"kernel_run_args", "tensor_args", "dfb_run_overrides"},
}


def _properties(cls):
    return {name for name, attribute in vars(cls).items() if isinstance(attribute, property)}


def _kernel(**overrides):
    fields = dict(unique_id="reader", source="reader.cpp", hw_config=E.create_reader_datamovement_config())
    fields.update(overrides)
    return E.KernelSpec(**fields)


def test_classes_are_under_ttnn_experimental_only():
    for name in ("ProgramSpec", "KernelSpec", "ProgramRunArgs", "KernelRunArgs", "producer_of"):
        assert hasattr(E, name)
        assert not hasattr(ttnn, name)
    assert E.KernelBuildOptLevel is E.fabric_mux.KernelBuildOptLevel


@pytest.mark.parametrize("cls", FIELDS, ids=lambda cls: cls.__qualname__)
def test_each_class_binds_exactly_its_fields(cls):
    assert _properties(cls) == FIELDS[cls]


def test_metal_aliases_name_the_nested_classes():
    assert E.DFBBinding is E.KernelSpec.DFBBinding
    assert E.TensorBinding is E.KernelSpec.TensorBinding
    assert E.SemaphoreBinding is E.KernelSpec.SemaphoreBinding
    assert E.ScratchpadBinding is E.KernelSpec.ScratchpadBinding
    assert E.DFBEndpointType is E.DFBBinding.EndpointType
    assert E.DFBAccessPattern is E.DFBBinding.AccessPattern
    assert E.KernelRunArgs is E.ProgramRunArgs.KernelRunArgs
    assert E.DFBRunOverrides is E.ProgramRunArgs.DFBRunOverrides


def test_defaults_are_the_header_defaults():
    kernel = _kernel()
    assert kernel.num_threads == 1
    assert kernel.compiler_options.opt_level == E.KernelBuildOptLevel.O2
    assert kernel.dfb_bindings == [] and kernel.compile_time_args == {}
    assert kernel.advanced_options.num_runtime_varargs == 0

    compute = E.ComputeHardwareConfig()
    assert compute.fpu_math_fidelity == ttnn.MathFidelity.HiFi4
    assert compute.sfpu_precision_mode == E.Precision.Precise
    assert not compute.enable_32_bit_dest and compute.double_buffer_dest
    assert compute.unpack_modes == {} and compute.config_1xx is None
    assert E.ComputeHardwareConfig.Compute1XXConfig().bfp_pack_precision_mode == E.Precision.Approximate

    dm_1xx = E.DataMovementHardwareConfig.DataMovement1XXConfig(
        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
    )
    assert dm_1xx.noc_mode == ttnn.NOC_MODE.DM_DEDICATED_NOC
    assert dm_1xx.dedicated_noc_sharing == E.DedicatedNocSharing.EXCLUSIVE

    dfb = E.DataflowBufferSpec(unique_id="in")
    assert (dfb.entry_size, dfb.num_entries) == (0, 0)
    assert dfb.data_format_metadata is None and dfb.borrowed_from is None
    assert E.DFBBinding(
        dfb_spec_name="in", accessor_name="in", endpoint_type=E.DFBEndpointType.PRODUCER
    ).access_pattern == (E.DFBAccessPattern.STRIDED)
    relaxations = E.TensorParameter(unique_id="input", spec=TENSOR_SPEC).relaxations
    assert not (relaxations.match_padded_shape_only or relaxations.dynamic_tensor_shape)
    assert E.WorkUnitSpec(target_nodes=ttnn.CoreCoord(0, 0)).name == ""
    assert E.ProgramSpec().kernels == []


def test_constructors_take_keywords_and_need_the_required_fields(expect_error):
    with expect_error(TypeError, "incompatible function arguments"):
        E.TensorBinding("input", "src")
    with expect_error(TypeError, "incompatible function arguments"):
        E.KernelSpec(unique_id="reader", source="reader.cpp")
    with expect_error(TypeError, "incompatible function arguments"):
        E.DataflowBufferSpec(entry_size=2048)
    with expect_error(TypeError, "incompatible function arguments"):
        E.KernelSpec(unique_id=None, source="reader.cpp", hw_config=E.ComputeHardwareConfig())


def test_names_are_str_and_tables_are_dicts():
    core = ttnn.CoreCoord(1, 2)
    kernel_run_args = E.KernelRunArgs(
        kernel="reader",
        runtime_arg_values={"num_tiles": {core: 3, (0, 0): 4}},
        common_runtime_arg_values={"stride": 2**32 - 1},
    )
    assert kernel_run_args.kernel == "reader"
    assert kernel_run_args.runtime_arg_values == {"num_tiles": {core: 3, ttnn.CoreCoord(0, 0): 4}}
    assert kernel_run_args.common_runtime_arg_values == {"stride": 2**32 - 1}

    kernel = _kernel(compile_time_args={"block": 8}, compiler_options=E.KernelSpec.CompilerOptions(defines={"A": "1"}))
    assert kernel.compile_time_args == {"block": 8}
    assert kernel.compiler_options.defines == {"A": "1"}
    compute = E.ComputeHardwareConfig(unpack_modes={"in": E.UnpackMode.UnpackToDest})
    assert compute.unpack_modes == {"in": E.UnpackMode.UnpackToDest}


def test_two_keys_for_one_node_are_rejected(expect_error):
    with expect_error(TypeError, "incompatible function arguments"):
        E.KernelRunArgs(kernel="reader", runtime_arg_values={"num_tiles": {(0, 0): 1, ttnn.CoreCoord(0, 0): 2}})


@pytest.mark.parametrize("value", [-1, 2**32, 1.5])
def test_argument_values_are_uint32(value, expect_error):
    with expect_error(TypeError, "incompatible function arguments"):
        E.KernelRunArgs(kernel="reader", common_runtime_arg_values={"stride": value})


def test_variant_fields_keep_their_type():
    assert _kernel(source="reader.cpp").source == pathlib.Path("reader.cpp")
    assert _kernel(source=pathlib.Path("/tmp/reader.cpp")).source == pathlib.Path("/tmp/reader.cpp")
    inline = _kernel(source=E.KernelSpec.SourceCode(code="void kernel_main() {}")).source
    assert isinstance(inline, E.KernelSpec.SourceCode) and inline.code == "void kernel_main() {}"

    assert isinstance(_kernel(hw_config=E.ComputeHardwareConfig()).hw_config, E.ComputeHardwareConfig)
    assert _kernel(hw_config=E.ComputeHardwareConfig()).is_compute_kernel()
    assert _kernel().is_data_movement_kernel()

    for nodes in (
        ttnn.CoreCoord(1, 1),
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),
    ):
        target_nodes = E.SemaphoreSpec(unique_id="sem", target_nodes=nodes).target_nodes
        assert type(target_nodes) is type(nodes) and target_nodes == nodes


def test_data_formats_are_ttnn_data_types():
    dfb = E.DataflowBufferSpec(
        unique_id="in", data_format_metadata=ttnn.bfloat8_b, tile_format_metadata=ttnn.Tile([16, 32])
    )
    assert dfb.data_format_metadata == ttnn.bfloat8_b
    assert dfb.tile_format_metadata.tile_shape == [16, 32]
    dfb.data_format_metadata = ttnn.float32
    assert dfb.data_format_metadata == ttnn.float32
    dfb.data_format_metadata = None
    dfb.tile_format_metadata = None
    assert dfb.data_format_metadata is None and dfb.tile_format_metadata is None

    scratchpad = E.ScratchpadSpec(unique_id="scratch", size_per_node=64, data_format_metadata=ttnn.uint32)
    assert scratchpad.data_format_metadata == ttnn.uint32

    # An int8 buffer takes the UInt8 format, as in ttnn's own ops, so it reads back as uint8.
    assert E.DataflowBufferSpec(unique_id="in", data_format_metadata=ttnn.int8).data_format_metadata == ttnn.uint8


def test_optional_fields_take_none():
    config = E.DataMovementHardwareConfig(config_2xx=E.DataMovementHardwareConfig.DataMovement2XXConfig())
    config.config_2xx = None
    assert config.config_2xx is None
    overrides = E.DFBRunOverrides(dfb="in", entry_size=64)
    overrides.entry_size = None
    assert overrides.entry_size is None
    dfb = E.DataflowBufferSpec(unique_id="in", borrowed_from="input")
    assert dfb.borrowed_from == "input"
    dfb.borrowed_from = None
    assert dfb.borrowed_from is None


def test_lists_dicts_and_optional_fields_are_copies():
    spec = E.ProgramSpec(kernels=[_kernel()])
    spec.kernels.append(_kernel(unique_id="writer"))
    spec.kernels[0].unique_id = "renamed"
    assert [kernel.unique_id for kernel in spec.kernels] == ["reader"]
    spec.kernels = spec.kernels + [_kernel(unique_id="writer")]
    assert [kernel.unique_id for kernel in spec.kernels] == ["reader", "writer"]

    kernel = _kernel()
    kernel.compile_time_args["block"] = 8
    assert kernel.compile_time_args == {}
    # hw_config is a variant and config_1xx an optional, so each read returns a copy.
    kernel.hw_config.config_1xx = None
    assert kernel.hw_config.config_1xx is not None


def test_writes_reach_nested_objects():
    kernel = _kernel()
    kernel.compiler_options.opt_level = E.KernelBuildOptLevel.O0
    kernel.advanced_options.num_runtime_varargs = 2
    kernel.runtime_arg_schema.runtime_arg_names = ["a"]
    assert kernel.compiler_options.opt_level == E.KernelBuildOptLevel.O0
    assert kernel.advanced_options.num_runtime_varargs == 2
    assert kernel.runtime_arg_schema.runtime_arg_names == ["a"]

    dfb = E.DataflowBufferSpec(unique_id="in")
    dfb.advanced_options.alias_with = ["other"]
    assert dfb.advanced_options.alias_with == ["other"]

    kernel_run_args = E.KernelRunArgs(kernel="reader")
    kernel_run_args.advanced_options.runtime_varargs = {ttnn.CoreCoord(0, 0): [5, 6]}
    assert kernel_run_args.advanced_options.runtime_varargs == {ttnn.CoreCoord(0, 0): [5, 6]}

    parameter = E.TensorParameter(unique_id="input", spec=TENSOR_SPEC)
    parameter.relaxations.dynamic_tensor_shape = True
    assert parameter.relaxations.dynamic_tensor_shape

    # A nested object keeps its parent alive.
    references = sys.getrefcount(kernel)
    options = kernel.advanced_options
    assert sys.getrefcount(kernel) == references + 1
    del kernel
    assert options.num_runtime_varargs == 2


def test_advanced_options_are_the_ones_blaze_uses():
    kernel = _kernel(advanced_options=E.KernelAdvancedOptions(num_runtime_varargs=2, num_common_runtime_varargs=1))
    assert (kernel.advanced_options.num_runtime_varargs, kernel.advanced_options.num_common_runtime_varargs) == (2, 1)

    dfb = E.DataflowBufferSpec(unique_id="a", advanced_options=E.DFBAdvancedOptions(alias_with=["b"]))
    assert dfb.advanced_options.alias_with == ["b"]

    varargs = E.AdvancedKernelRunArgs(runtime_varargs={ttnn.CoreCoord(0, 0): [1, 2]}, common_runtime_varargs=[7])
    run_args = E.KernelRunArgs(kernel="reader", advanced_options=varargs)
    # Vararg values read back as plain lists, like every other list field.
    assert run_args.advanced_options.runtime_varargs == {ttnn.CoreCoord(0, 0): [1, 2]}
    assert type(run_args.advanced_options.runtime_varargs[ttnn.CoreCoord(0, 0)]) is list
    assert run_args.advanced_options.common_runtime_varargs == [7]
    assert type(run_args.advanced_options.common_runtime_varargs) is list
    varargs.common_runtime_varargs = varargs.common_runtime_varargs + [8]
    assert varargs.common_runtime_varargs == [7, 8]


PRODUCER, CONSUMER = E.DFBEndpointType.PRODUCER, E.DFBEndpointType.CONSUMER
STRIDED, BLOCKED, ALL = E.DFBAccessPattern.STRIDED, E.DFBAccessPattern.BLOCKED, E.DFBAccessPattern.ALL


@pytest.mark.parametrize(
    "binding, expected",
    [
        (E.producer_of("in", "a"), (PRODUCER, STRIDED, 0)),
        (E.blocked_producer_of("in", "a", 4), (PRODUCER, BLOCKED, 4)),
        (E.consumer_of("in", "a"), (CONSUMER, STRIDED, 0)),
        (E.strided_consumer_of("in", "a"), (CONSUMER, STRIDED, 0)),
        (E.all_consumer_of("in", "a"), (CONSUMER, ALL, 0)),
        (E.blocked_consumer_of("in", "a", 2), (CONSUMER, BLOCKED, 2)),
    ],
    ids=[
        "producer_of",
        "blocked_producer_of",
        "consumer_of",
        "strided_consumer_of",
        "all_consumer_of",
        "blocked_consumer_of",
    ],
)
def test_dfb_binding_helpers(binding, expected):
    assert (
        binding.dfb_spec_name,
        binding.accessor_name,
        binding.endpoint_type,
        binding.access_pattern,
        binding.block_size,
    ) == ("in", "a", *expected)


def test_reader_and_writer_configs():
    reader = E.create_reader_datamovement_config()
    writer = E.create_writer_datamovement_config(disable_dfb_implicit_sync_for_all=True)
    # ttnn.NOC values have aliases, so compare the values rather than the enum members.
    assert reader.config_1xx.processor == ttnn.DataMovementProcessor.RISCV_1
    assert reader.config_1xx.noc.value == ttnn.NOC.NOC_0.value
    assert writer.config_1xx.processor == ttnn.DataMovementProcessor.RISCV_0
    assert writer.config_1xx.noc.value == ttnn.NOC.NOC_1.value
    assert not reader.config_2xx.disable_dfb_implicit_sync_for_all
    assert writer.config_2xx.disable_dfb_implicit_sync_for_all


def test_program_spec_from_the_design_example():
    kernels = ["reader", "writer", "compute"]
    spec = E.ProgramSpec(
        name="exp",
        kernels=[
            _kernel(
                dfb_bindings=[E.producer_of("in", "in")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="src")],
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_pages", "start_id"]),
            ),
            _kernel(
                unique_id="writer",
                hw_config=E.create_writer_datamovement_config(),
                dfb_bindings=[E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="dst")],
            ),
            _kernel(
                unique_id="compute",
                hw_config=E.ComputeHardwareConfig(),
                compiler_options=E.KernelSpec.CompilerOptions(
                    defines={"SFPU_OP_EXP_INCLUDE": "1"}, opt_level=E.KernelBuildOptLevel.O3
                ),
                dfb_bindings=[E.consumer_of("in", "in"), E.producer_of("out", "out")],
            ),
        ],
        dataflow_buffers=[
            E.DataflowBufferSpec(unique_id=name, entry_size=2048, num_entries=2, data_format_metadata=ttnn.bfloat16)
            for name in ("in", "out")
        ],
        tensor_parameters=[
            E.TensorParameter(unique_id="input", spec=TENSOR_SPEC),
            E.TensorParameter(unique_id="output", spec=TENSOR_SPEC),
        ],
        work_units=[E.WorkUnitSpec(name="exp", kernels=kernels, target_nodes=ttnn.CoreCoord(0, 0))],
    )
    assert [kernel.unique_id for kernel in spec.kernels] == kernels
    assert spec.kernels[2].compiler_options.opt_level == E.KernelBuildOptLevel.O3
    assert spec.work_units[0].kernels == kernels
    assert spec.tensor_parameters[1].spec == TENSOR_SPEC


def test_tensor_args_reject_what_is_not_a_device_tensor(expect_error):
    host_tensor = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "tensor_args\\['input'\\] must be a tensor allocated on a device"):
        E.ProgramRunArgs(tensor_args={"input": host_tensor})
    with expect_error(TypeError, "tensor_args\\['input'\\] must be a ttnn.Tensor"):
        E.ProgramRunArgs(tensor_args={"input": None})
    with expect_error(TypeError, "tensor_args keys must be tensor parameter names"):
        E.ProgramRunArgs(tensor_args={0: host_tensor})


def test_run_args_reject_unknown_attributes(expect_error):
    run_args = E.ProgramRunArgs()
    with expect_error(AttributeError, "'ProgramRunArgs' object has no attribute 'dfb_run_override'"):
        run_args.dfb_run_override = [E.DFBRunOverrides(dfb="in", num_entries=4)]
    with expect_error(AttributeError, "'ProgramRunArgs' object has no attribute 'dfb_run_override'"):
        run_args.dfb_run_override = None
    run_args.dfb_run_overrides = [E.DFBRunOverrides(dfb="in", num_entries=4)]
    assert run_args.dfb_run_overrides[0].num_entries == 4

    # A Python subclass may add attributes of its own, as for the other classes.
    class StepRunArgs(E.ProgramRunArgs):
        pass

    subclass_run_args = StepRunArgs()
    subclass_run_args.step = 1
    assert subclass_run_args.step == 1


def test_tensor_args_hold_the_tensors(device):
    tensor = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    run_args = E.ProgramRunArgs(tensor_args={"input": tensor})
    assert run_args.tensor_args["input"] is tensor

    run_args.tensor_args["output"] = tensor
    assert list(run_args.tensor_args) == ["input"]

    run_args.tensor_args = {"output": tensor}
    assert list(run_args.tensor_args) == ["output"]
