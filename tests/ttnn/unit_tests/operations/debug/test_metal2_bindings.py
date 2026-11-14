# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the Metal 2.0 program spec Python bindings (ttnn.experimental.metal2).

No device is opened. The tests pin: the exact set of fields every struct exposes (a field renamed
or removed in tt_metal fails here, a field added fails the static_assert in
ttnn/cpp/ttnn-nanobind/program_specs.cpp); that the keyword constructors' defaults equal the C++
default member initialisers; every container conversion in both directions (Table <-> dict,
Group <-> list, Nodes <-> CoreCoord / CoreRange / CoreRangeSet, the source and hw_config
variants); and the copy semantics of every property (nothing read from a spec can dangle after
the spec changes).
"""

import copy
import pathlib

import pytest
import torch

import ttnn
from ttnn.experimental import metal2 as m2

# The Python-visible fields of every bound struct, exactly. Keep in step with the static_asserts in
# program_specs.cpp: that file pins the C++ field count, this pins what Python sees.
EXPECTED_FIELDS = {
    m2.TensorSpecRelaxations: {
        "match_padded_shape_only",
        "dynamic_tensor_shape",
        "relax_logical_rank",
        "match_page_size",
    },
    m2.TensorParameter: {"unique_id", "spec", "relaxations"},
    m2.DFBAdvancedOptions: {"alias_with", "allow_instance_multi_binding", "prefetcher_pipe_relays"},
    m2.DataflowBufferSpec: {
        "unique_id",
        "entry_size",
        "num_entries",
        "data_format",
        "data_format_raw",
        "tile_format",
        "borrowed_from",
        "advanced_options",
    },
    m2.SemaphoreAdvancedOptions: {"max_value"},
    m2.SemaphoreSpec: {"unique_id", "target_nodes", "advanced_options"},
    m2.ScratchpadSpec: {"unique_id", "size_per_node", "data_format", "data_format_raw", "tile_format"},
    m2.DataMovement1XXConfig: {"processor", "noc", "noc_mode", "dedicated_noc_sharing"},
    m2.DataMovement2XXConfig: {"disable_dfb_implicit_sync_for", "disable_dfb_implicit_sync_for_all"},
    m2.DataMovementHardwareConfig: {"config_1xx", "config_2xx"},
    m2.Compute1XXConfig: {"bfp_pack_precision_mode"},
    m2.Compute2XXConfig: set(),
    m2.ComputeHardwareConfig: {
        "fpu_math_fidelity",
        "sfpu_precision_mode",
        "enable_32_bit_dest",
        "double_buffer_dest",
        "unpack_modes",
        "config_1xx",
        "config_2xx",
    },
    m2.SourceCode: {"code"},
    m2.CompilerOptions: {"include_paths", "defines", "opt_level"},
    m2.DFBBinding: {"dfb_spec_name", "accessor_name", "endpoint_type", "access_pattern", "block_size"},
    m2.SemaphoreBinding: {"semaphore_spec_name", "accessor_name"},
    m2.ScratchpadBinding: {"scratchpad_spec_name", "accessor_name"},
    m2.TensorBinding: {"tensor_parameter_name", "accessor_name"},
    m2.RuntimeArgSchema: {"runtime_arg_names", "common_runtime_arg_names"},
    m2.TensorBindingSequence: {"sequence_name", "members"},
    m2.PrefetcherPipeBinding: {"pipe_parameter_names", "accessor_name"},
    m2.KernelAdvancedOptions: {
        "num_runtime_varargs",
        "num_common_runtime_varargs",
        "compile_time_varargs",
        "tensor_binding_sequences",
        "prefetcher_pipe_bindings",
    },
    m2.KernelSpec: {
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
    m2.WorkUnitSpec: {"name", "kernels", "target_nodes"},
    m2.PrefetcherPipeParameter: {"unique_id", "receivers", "ring_size", "entry_size"},
    m2.ProgramAdvancedOptions: {"prefetcher_pipe_parameters"},
    m2.ProgramSpec: {
        "name",
        "kernels",
        "dataflow_buffers",
        "semaphores",
        "scratchpads",
        "tensor_parameters",
        "work_units",
        "advanced_options",
    },
    m2.AdvancedKernelRunArgs: {"runtime_varargs", "common_runtime_varargs"},
    m2.KernelRunArgs: {"kernel", "runtime_arg_values", "common_runtime_arg_values", "advanced_options"},
    m2.DFBRunOverrides: {"dfb", "entry_size", "num_entries"},
    m2.ProgramRunArgs: {"kernel_run_args", "dfb_run_overrides", "prefetcher_pipe_names"},
}

# Required constructor arguments per class (everything else has a default).
REQUIRED = {
    m2.TensorParameter: lambda: {"unique_id": "a", "spec": _host_spec()},
    m2.DataflowBufferSpec: lambda: {"unique_id": "ring", "entry_size": 2048, "num_entries": 2},
    m2.SemaphoreSpec: lambda: {"unique_id": "s", "target_nodes": ttnn.CoreCoord(0, 0)},
    m2.ScratchpadSpec: lambda: {"unique_id": "tmp", "size_per_node": 64},
    m2.DataMovement1XXConfig: lambda: {"processor": ttnn.DataMovementProcessor.RISCV_1, "noc": ttnn.NOC.NOC_0},
    m2.SourceCode: lambda: {"code": ""},
    m2.DFBBinding: lambda: {"dfb_spec_name": "r", "accessor_name": "a", "endpoint_type": m2.DFBEndpointType.PRODUCER},
    m2.SemaphoreBinding: lambda: {"semaphore_spec_name": "s", "accessor_name": "a"},
    m2.ScratchpadBinding: lambda: {"scratchpad_spec_name": "s", "accessor_name": "a"},
    m2.TensorBinding: lambda: {"tensor_parameter_name": "t", "accessor_name": "a"},
    m2.TensorBindingSequence: lambda: {"sequence_name": "seq"},
    m2.PrefetcherPipeBinding: lambda: {"pipe_parameter_names": ["p"], "accessor_name": "a"},
    m2.KernelSpec: lambda: {"unique_id": "k", "source": "/tmp/k.cpp", "hw_config": m2.ComputeHardwareConfig()},
    m2.WorkUnitSpec: lambda: {"name": "main", "kernels": ["k"], "target_nodes": ttnn.CoreCoord(0, 0)},
    m2.PrefetcherPipeParameter: lambda: {"unique_id": "pipe", "receivers": ttnn.CoreCoord(0, 0)},
    m2.KernelRunArgs: lambda: {"kernel": "k"},
    m2.DFBRunOverrides: lambda: {"dfb": "ring"},
}


def _host_spec(shape=(32, 32)):
    return ttnn.from_torch(torch.zeros(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).spec


def _instance(cls):
    # TensorParameter has no default constructor (TensorSpec does not); everything else does.
    if cls is m2.TensorParameter:
        return m2.TensorParameter(**REQUIRED[cls]())
    return cls()


def _exposed_fields(instance):
    cls = type(instance)
    return {
        name
        for name in dir(instance)
        if not name.startswith("_")
        and not callable(getattr(cls, name, None))
        and isinstance(getattr(cls, name, None), property)
    }


def _plain(value):
    """A comparable Python value: bound structs become dicts of their fields, enums their names."""
    if type(value) in EXPECTED_FIELDS:
        return {f: _plain(getattr(value, f)) for f in EXPECTED_FIELDS[type(value)]}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) or type(value).__name__ == "VectorUInt32":
        return [_plain(v) for v in value]
    if hasattr(value, "name") and hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return value.name  # enum member
    if isinstance(value, (ttnn.TensorSpec, ttnn.CoreCoord, ttnn.CoreRange, ttnn.CoreRangeSet, ttnn.Tile)):
        return str(value)
    return value


def test_namespace_is_experimental_only():
    assert ttnn.experimental.metal2.ProgramSpec is ttnn._ttnn.program_spec.ProgramSpec
    bound = {n for n in dir(ttnn._ttnn.program_spec) if not n.startswith("_")}
    assert set(m2.__all__) == bound | {"KernelBuildOptLevel"}
    # The host API is experimental; nothing leaks into the public ttnn namespace.
    leaked = set(m2.__all__) & set(dir(ttnn))
    assert not leaked, f"{sorted(leaked)} must live under ttnn.experimental.metal2 only"


@pytest.mark.parametrize("cls", list(EXPECTED_FIELDS), ids=lambda c: c.__name__)
def test_struct_exposes_exactly_the_expected_fields(cls):
    assert _exposed_fields(_instance(cls)) == EXPECTED_FIELDS[cls]


@pytest.mark.parametrize("cls", list(EXPECTED_FIELDS), ids=lambda c: c.__name__)
def test_keyword_constructor_defaults_equal_cpp_defaults(cls):
    """cls(required...) equals a default-constructed instance with the same required fields set."""
    if cls is m2.TensorParameter:
        return  # no default constructor; its only defaulted field is checked in test_defaults_mirror_cpp
    required = REQUIRED.get(cls, dict)()
    via_keywords = cls(**required)
    via_default = cls()
    for name, value in required.items():
        setattr(via_default, name, value)
    assert _plain(via_keywords) == _plain(via_default)


@pytest.mark.parametrize("cls", list(EXPECTED_FIELDS), ids=lambda c: c.__name__)
def test_copy_and_deepcopy(cls):
    original = _instance(cls)
    for clone in (copy.copy(original), copy.deepcopy(original)):
        assert clone is not original
        assert _plain(clone) == _plain(original)


def test_enums():
    assert {m.name for m in m2.DFBAccessPattern} == {"STRIDED", "ALL", "BLOCKED"}
    assert {m.name for m in m2.DFBEndpointType} == {"PRODUCER", "CONSUMER"}
    assert {m.name for m in m2.UnpackMode} == {"UnpackToSrc", "UnpackToDest"}
    assert {m.name for m in m2.Precision} == {"Approximate", "Precise"}
    assert {m.name for m in m2.DedicatedNocSharing} == {"EXCLUSIVE", "SHARED_DISJOINT_TRAFFIC"}
    assert m2.KernelBuildOptLevel is ttnn.experimental.fabric_mux.KernelBuildOptLevel


def test_defaults_mirror_cpp():
    relax = m2.TensorSpecRelaxations()
    assert not any(
        (relax.match_padded_shape_only, relax.dynamic_tensor_shape, relax.relax_logical_rank, relax.match_page_size)
    )
    assert _plain(m2.TensorParameter("a", _host_spec()).relaxations) == _plain(m2.TensorSpecRelaxations())

    binding = m2.DFBBinding("ring", "in", m2.DFBEndpointType.PRODUCER)
    assert binding.access_pattern == m2.DFBAccessPattern.STRIDED
    assert binding.block_size == 0

    dm = m2.DataMovement1XXConfig(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0)
    assert dm.noc_mode == ttnn.NOC_MODE.DM_DEDICATED_NOC
    assert dm.dedicated_noc_sharing == m2.DedicatedNocSharing.EXCLUSIVE

    compute = m2.ComputeHardwareConfig()
    assert compute.fpu_math_fidelity == ttnn.MathFidelity.HiFi4
    assert compute.sfpu_precision_mode == m2.Precision.Precise
    assert compute.enable_32_bit_dest is False
    assert compute.double_buffer_dest is True
    assert compute.unpack_modes == {}
    assert compute.config_1xx is None and compute.config_2xx is None
    assert m2.Compute1XXConfig().bfp_pack_precision_mode == m2.Precision.Approximate

    assert m2.CompilerOptions().opt_level == m2.KernelBuildOptLevel.O2
    assert m2.CompilerOptions(opt_level=m2.KernelBuildOptLevel.O3).opt_level == m2.KernelBuildOptLevel.O3

    kernel = m2.KernelSpec("k", "/tmp/k.cpp", m2.create_reader_dm_config())
    assert kernel.num_threads == 1
    assert kernel.compile_time_args == {}
    assert kernel.runtime_arg_schema.runtime_arg_names == []
    assert kernel.advanced_options.num_runtime_varargs == 0

    assert m2.DataflowBufferSpec("ring", 2048, 2).data_format is None
    assert m2.DataflowBufferSpec("ring", 2048, 2).data_format_raw is None
    assert m2.DataflowBufferSpec("ring", 2048, 2).borrowed_from is None
    assert m2.SemaphoreAdvancedOptions().max_value == 0
    assert m2.DFBRunOverrides("ring").entry_size is None
    assert m2.ProgramSpec().name == ""
    assert m2.SourceCode().code == ""
    pipe = m2.PrefetcherPipeParameter("pipe", ttnn.CoreCoord(0, 0))
    assert (pipe.ring_size, pipe.entry_size) == (0, 0)


def test_optional_tails_are_keyword_only(expect_error):
    with expect_error(TypeError, "incompatible function arguments"):
        m2.DataflowBufferSpec("in", 2048, 2, ttnn.bfloat16)
    with expect_error(TypeError, "incompatible function arguments"):
        m2.PrefetcherPipeParameter("pipe", ttnn.CoreCoord(0, 0), 4096, 64)
    with expect_error(TypeError, "incompatible function arguments"):
        m2.KernelSpec("k", "/tmp/k.cpp", m2.ComputeHardwareConfig(), 4)
    with expect_error(TypeError, "incompatible function arguments"):
        m2.DFBBinding("r", "a", m2.DFBEndpointType.PRODUCER, m2.DFBAccessPattern.ALL)


def test_binding_helpers():
    assert m2.producer_of("r", "a").endpoint_type == m2.DFBEndpointType.PRODUCER
    assert m2.producer_of("r", "a").access_pattern == m2.DFBAccessPattern.STRIDED
    assert m2.consumer_of("r", "a").endpoint_type == m2.DFBEndpointType.CONSUMER
    assert m2.strided_consumer_of("r", "a").access_pattern == m2.DFBAccessPattern.STRIDED
    assert m2.all_consumer_of("r", "a").access_pattern == m2.DFBAccessPattern.ALL
    blocked = m2.blocked_producer_of("r", "a", 4)
    assert (blocked.endpoint_type, blocked.access_pattern, blocked.block_size) == (
        m2.DFBEndpointType.PRODUCER,
        m2.DFBAccessPattern.BLOCKED,
        4,
    )
    blocked = m2.blocked_consumer_of("r", "a", 8)
    assert (blocked.endpoint_type, blocked.access_pattern, blocked.block_size) == (
        m2.DFBEndpointType.CONSUMER,
        m2.DFBAccessPattern.BLOCKED,
        8,
    )
    assert m2.producer_of("r", "a").dfb_spec_name == "r"
    assert m2.producer_of("r", "a").accessor_name == "a"


def test_reader_writer_configs():
    # ttnn.NOC carries value aliases (NOC_0 is also RISCV_0_default), so compare by value.
    reader = m2.create_reader_dm_config()
    assert reader.config_1xx.processor == ttnn.DataMovementProcessor.RISCV_1
    assert reader.config_1xx.noc.value == ttnn.NOC.NOC_0.value
    assert reader.config_2xx is None
    writer = m2.create_writer_dm_config()
    assert writer.config_1xx.processor == ttnn.DataMovementProcessor.RISCV_0
    assert writer.config_1xx.noc.value == ttnn.NOC.NOC_1.value

    shared = m2.DataMovementHardwareConfig(
        config_1xx=m2.DataMovement1XXConfig(
            ttnn.DataMovementProcessor.RISCV_0,
            ttnn.NOC.NOC_1,
            noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
            dedicated_noc_sharing=m2.DedicatedNocSharing.SHARED_DISJOINT_TRAFFIC,
        )
    )
    assert shared.config_1xx.noc_mode == ttnn.NOC_MODE.DM_DYNAMIC_NOC
    assert shared.config_1xx.dedicated_noc_sharing == m2.DedicatedNocSharing.SHARED_DISJOINT_TRAFFIC

    quasar = m2.DataMovementHardwareConfig(config_2xx=m2.DataMovement2XXConfig(disable_dfb_implicit_sync_for_all=True))
    assert quasar.config_1xx is None
    assert quasar.config_2xx.disable_dfb_implicit_sync_for_all is True
    assert quasar.config_2xx.disable_dfb_implicit_sync_for == []

    compute = m2.ComputeHardwareConfig(config_1xx=m2.Compute1XXConfig(bfp_pack_precision_mode=m2.Precision.Precise))
    assert compute.config_1xx.bfp_pack_precision_mode == m2.Precision.Precise
    assert m2.ComputeHardwareConfig(config_2xx=m2.Compute2XXConfig()).config_2xx is not None


def test_tables_round_trip_as_dicts():
    kernel = m2.KernelSpec("k", "/tmp/k.cpp", m2.ComputeHardwareConfig(), compile_time_args={"wt": 4, "ht": 2})
    assert kernel.compile_time_args == {"wt": 4, "ht": 2}
    kernel.compile_time_args = {"wt": 8}
    assert kernel.compile_time_args == {"wt": 8}

    compute = m2.ComputeHardwareConfig(enable_32_bit_dest=True, unpack_modes={"in_a": m2.UnpackMode.UnpackToDest})
    assert compute.unpack_modes == {"in_a": m2.UnpackMode.UnpackToDest}

    c00, c10 = ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)
    run_args = m2.KernelRunArgs(
        kernel="k",
        runtime_arg_values={"num_tiles": {c00: 3, c10: 4}, "start_id": {c00: 0, c10: 3}},
        common_runtime_arg_values={"scaler": 0x3F800000},
        advanced_options=m2.AdvancedKernelRunArgs(runtime_varargs={c00: [1, 2, 3]}, common_runtime_varargs=[9]),
    )
    assert run_args.runtime_arg_values["num_tiles"][c10] == 4
    assert run_args.runtime_arg_values["start_id"] == {c00: 0, c10: 3}
    assert run_args.common_runtime_arg_values == {"scaler": 0x3F800000}
    assert list(run_args.advanced_options.runtime_varargs[c00]) == [1, 2, 3]
    assert list(run_args.advanced_options.common_runtime_varargs) == [9]

    defines = m2.CompilerOptions(defines={"FAST": "1"}, include_paths=["/tmp/inc"])
    assert defines.defines == {"FAST": "1"}
    assert [pathlib.Path(p) for p in defines.include_paths] == [pathlib.Path("/tmp/inc")]


def test_duplicate_keys_after_conversion_are_rejected(expect_error):
    # (0, 0) and CoreCoord(0, 0) are distinct dict keys in Python but the same node once converted.
    with expect_error(TypeError, "incompatible function arguments"):
        m2.KernelRunArgs("k", runtime_arg_values={"n": {(0, 0): 1, ttnn.CoreCoord(0, 0): 2}})


def test_vector_uint32_fields_are_the_opaque_vector_type():
    # std::vector<uint32_t> is bound opaquely (VectorUInt32) for the whole extension; lists convert on input.
    advanced = m2.KernelAdvancedOptions(compile_time_varargs=[7, 8])
    assert type(advanced.compile_time_varargs).__name__ == "VectorUInt32"
    assert list(advanced.compile_time_varargs) == [7, 8]
    assert advanced.compile_time_varargs == [7, 8]


def test_variants_round_trip():
    from_path = m2.KernelSpec("k", "/tmp/k.cpp", m2.create_reader_dm_config())
    assert pathlib.Path(from_path.source) == pathlib.Path("/tmp/k.cpp")
    assert from_path.is_data_movement_kernel() and not from_path.is_compute_kernel()
    assert isinstance(from_path.hw_config, m2.DataMovementHardwareConfig)

    inline = m2.KernelSpec("k", m2.SourceCode("void kernel_main() {}"), m2.ComputeHardwareConfig())
    assert inline.source.code == "void kernel_main() {}"
    assert inline.is_compute_kernel()
    assert isinstance(inline.hw_config, m2.ComputeHardwareConfig)

    inline.source = "/tmp/other.cpp"
    assert pathlib.Path(inline.source) == pathlib.Path("/tmp/other.cpp")
    inline.hw_config = m2.create_writer_dm_config()
    assert inline.is_data_movement_kernel()

    core = ttnn.CoreCoord(2, 3)
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))
    core_set = ttnn.CoreRangeSet([core_range])
    for nodes, cls in ((core, ttnn.CoreCoord), (core_range, ttnn.CoreRange), (core_set, ttnn.CoreRangeSet)):
        unit = m2.WorkUnitSpec("main", ["k"], nodes)
        assert isinstance(unit.target_nodes, cls)
        assert m2.SemaphoreSpec("s", nodes).target_nodes == nodes
    assert m2.PrefetcherPipeParameter("pipe", core_set, ring_size=4096, entry_size=64).receivers == core_set


def test_dataflow_buffer_spec_formats(expect_error):
    dfb = m2.DataflowBufferSpec("in", entry_size=2048, num_entries=2, data_format=ttnn.bfloat16)
    assert dfb.data_format == ttnn.bfloat16
    assert dfb.data_format_raw is not None
    dfb.data_format = ttnn.float32
    assert dfb.data_format == ttnn.float32
    dfb.data_format = None
    assert dfb.data_format is None and dfb.data_format_raw is None

    tiled = m2.DataflowBufferSpec("in", 2048, 2, data_format=ttnn.bfloat16, tile_format=ttnn.Tile((32, 32)))
    assert tiled.tile_format.tile_shape == [32, 32]

    borrowed = m2.DataflowBufferSpec(
        "shard",
        2048,
        8,
        data_format=ttnn.bfloat16,
        borrowed_from="a",
        advanced_options=m2.DFBAdvancedOptions(alias_with=["other"], prefetcher_pipe_relays=["pipe"]),
    )
    assert borrowed.borrowed_from == "a"
    assert borrowed.advanced_options.alias_with == ["other"]
    assert borrowed.advanced_options.prefetcher_pipe_relays == ["pipe"]

    scratch = m2.ScratchpadSpec("tmp", 4096, data_format=ttnn.uint32)
    assert scratch.size_per_node == 4096
    assert scratch.data_format == ttnn.uint32
    assert scratch.tile_format is None
    scratch.data_format = ttnn.bfloat16
    scratch.tile_format = ttnn.Tile((16, 32))
    assert scratch.data_format == ttnn.bfloat16
    assert scratch.tile_format.tile_shape == [16, 32]

    # A DataType with no DataFormat is refused before the object is built.
    with expect_error(RuntimeError, "Unsupported DataType"):
        m2.DataflowBufferSpec("r", 1, 2, data_format=ttnn.DataType.INVALID)


def test_advanced_options():
    advanced = m2.KernelAdvancedOptions(
        num_runtime_varargs=2,
        compile_time_varargs=[7, 8],
        tensor_binding_sequences=[m2.TensorBindingSequence("inputs", members=["a", "b"])],
        prefetcher_pipe_bindings=[m2.PrefetcherPipeBinding(["pipe"], "weights")],
    )
    assert advanced.num_runtime_varargs == 2
    assert list(advanced.compile_time_varargs) == [7, 8]
    assert advanced.tensor_binding_sequences[0].members == ["a", "b"]
    assert advanced.prefetcher_pipe_bindings[0].pipe_parameter_names == ["pipe"]
    assert advanced.prefetcher_pipe_bindings[0].accessor_name == "weights"

    kernel = m2.KernelSpec(
        "k",
        "/tmp/k.cpp",
        m2.create_reader_dm_config(),
        num_threads=2,
        compiler_options=m2.CompilerOptions(defines={"A": "1"}, opt_level=m2.KernelBuildOptLevel.O3),
        semaphore_bindings=[m2.SemaphoreBinding("sem", "ready")],
        scratchpad_bindings=[m2.ScratchpadBinding("tmp", "scratch")],
        advanced_options=advanced,
    )
    assert kernel.num_threads == 2
    assert kernel.compiler_options.defines == {"A": "1"}
    assert kernel.compiler_options.opt_level == m2.KernelBuildOptLevel.O3
    assert kernel.semaphore_bindings[0].semaphore_spec_name == "sem"
    assert kernel.semaphore_bindings[0].accessor_name == "ready"
    assert kernel.scratchpad_bindings[0].scratchpad_spec_name == "tmp"
    assert kernel.advanced_options.prefetcher_pipe_bindings[0].accessor_name == "weights"

    semaphore = m2.SemaphoreSpec("s", ttnn.CoreCoord(0, 0), advanced_options=m2.SemaphoreAdvancedOptions(max_value=4))
    assert semaphore.advanced_options.max_value == 4

    pipe = m2.PrefetcherPipeParameter("pipe", ttnn.CoreCoord(1, 1), ring_size=4096, entry_size=64)
    spec = m2.ProgramSpec(advanced_options=m2.ProgramAdvancedOptions(prefetcher_pipe_parameters=[pipe]))
    assert spec.advanced_options.prefetcher_pipe_parameters[0].unique_id == "pipe"

    run_args = m2.ProgramRunArgs()
    assert run_args.prefetcher_pipe_names == []
    assert callable(run_args.set_prefetcher_pipe)  # a PrefetcherPipe needs a device; the setter is exercised in PR 2


def test_properties_return_copies_and_never_dangle():
    spec = m2.ProgramSpec(
        kernels=[
            m2.KernelSpec("original_kernel_name_long_enough_to_heap_allocate", "/tmp/k.cpp", m2.ComputeHardwareConfig())
        ]
    )
    first = spec.kernels[0]
    spec.kernels = []  # the C++ vector is destroyed; `first` must not point into it
    spec.kernels = [m2.KernelSpec(f"k{i}", "/tmp/k.cpp", m2.ComputeHardwareConfig()) for i in range(64)]
    assert first.unique_id == "original_kernel_name_long_enough_to_heap_allocate"

    spec.kernels[0].num_threads = 7
    assert spec.kernels[0].num_threads == 1, "an element read from a property is a copy"
    spec.kernels.append(m2.KernelSpec("k", "/tmp/k.cpp", m2.ComputeHardwareConfig()))
    assert len(spec.kernels) == 64, "the list read from a property is a copy"
    kernel = spec.kernels[0]
    kernel.compile_time_args["wt"] = 8
    assert kernel.compile_time_args == {}, "a dict read from a property is a copy"

    dm = m2.create_reader_dm_config()
    config = dm.config_1xx
    dm.config_1xx = None
    assert config.processor == ttnn.DataMovementProcessor.RISCV_1

    kernel = m2.KernelSpec("k", "/tmp/k.cpp", m2.ComputeHardwareConfig(unpack_modes={"a": m2.UnpackMode.UnpackToDest}))
    hw = kernel.hw_config
    kernel.hw_config = m2.create_reader_dm_config()
    assert hw.unpack_modes == {"a": m2.UnpackMode.UnpackToDest}

    args = m2.AdvancedKernelRunArgs(runtime_varargs={ttnn.CoreCoord(0, 0): [1, 2]})
    varargs = args.runtime_varargs
    args.runtime_varargs = {}
    assert list(varargs[ttnn.CoreCoord(0, 0)]) == [1, 2]


def test_full_program_spec_assembly():
    """The toy eltwise-multiply spec (reader, compute, writer over two input rings and one output ring)."""
    tile_bytes = 2 * 32 * 32
    a_spec, b_spec, out_spec = _host_spec(), _host_spec(), _host_spec()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    rings = [
        m2.DataflowBufferSpec(name, entry_size=tile_bytes, num_entries=2, data_format=ttnn.bfloat16)
        for name in ("in_a", "in_b", "out")
    ]
    rta = m2.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"])
    reader = m2.KernelSpec(
        unique_id="reader",
        source="/tmp/reader.cpp",
        hw_config=m2.create_reader_dm_config(),
        dfb_bindings=[m2.producer_of("in_a", "in_a"), m2.producer_of("in_b", "in_b")],
        tensor_bindings=[m2.TensorBinding("a", "a"), m2.TensorBinding("b", "b")],
        runtime_arg_schema=rta,
    )
    compute = m2.KernelSpec(
        unique_id="compute",
        source="/tmp/compute.cpp",
        hw_config=m2.ComputeHardwareConfig(fpu_math_fidelity=ttnn.MathFidelity.LoFi),
        dfb_bindings=[m2.consumer_of("in_a", "in_a"), m2.consumer_of("in_b", "in_b"), m2.producer_of("out", "out")],
        runtime_arg_schema=m2.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    writer = m2.KernelSpec(
        unique_id="writer",
        source=m2.SourceCode("void kernel_main() {}"),
        hw_config=m2.create_writer_dm_config(),
        dfb_bindings=[m2.consumer_of("out", "out")],
        tensor_bindings=[m2.TensorBinding("out", "out")],
        runtime_arg_schema=rta,
    )
    spec = m2.ProgramSpec(
        name="toy_spec_mul",
        kernels=[reader, compute, writer],
        dataflow_buffers=rings,
        tensor_parameters=[
            m2.TensorParameter("a", a_spec),
            m2.TensorParameter("b", b_spec),
            m2.TensorParameter("out", out_spec, relaxations=m2.TensorSpecRelaxations(match_padded_shape_only=True)),
        ],
        work_units=[m2.WorkUnitSpec("main", ["reader", "compute", "writer"], cores)],
    )
    assert spec.name == "toy_spec_mul"
    assert [k.unique_id for k in spec.kernels] == ["reader", "compute", "writer"]
    assert [d.unique_id for d in spec.dataflow_buffers] == ["in_a", "in_b", "out"]
    assert spec.tensor_parameters[2].relaxations.match_padded_shape_only is True
    assert spec.tensor_parameters[0].spec.shape == a_spec.shape
    assert spec.work_units[0].kernels == ["reader", "compute", "writer"]
    assert spec.kernels[1].hw_config.fpu_math_fidelity == ttnn.MathFidelity.LoFi
    assert [b.accessor_name for b in spec.kernels[1].dfb_bindings] == ["in_a", "in_b", "out"]

    # A spec is a usable template: deep copies are independent.
    variant = copy.deepcopy(spec)
    variant.name = "toy_spec_mul_v2"
    assert spec.name == "toy_spec_mul"
    assert _plain(variant.kernels) == _plain(spec.kernels)

    c00, c10 = ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)
    per_core = {"num_tiles": {c00: 1, c10: 1}, "start_id": {c00: 0, c10: 1}}
    run_args = m2.ProgramRunArgs(
        kernel_run_args=[
            m2.KernelRunArgs(kernel="reader", runtime_arg_values=per_core),
            m2.KernelRunArgs(kernel="compute", runtime_arg_values={"num_tiles": per_core["num_tiles"]}),
            m2.KernelRunArgs(kernel="writer", runtime_arg_values=per_core),
        ],
        dfb_run_overrides=[m2.DFBRunOverrides("out", num_entries=4)],
    )
    assert [k.kernel for k in run_args.kernel_run_args] == ["reader", "compute", "writer"]
    assert run_args.kernel_run_args[0].runtime_arg_values["start_id"][c10] == 1
    assert run_args.dfb_run_overrides[0].num_entries == 4
    assert run_args.dfb_run_overrides[0].entry_size is None
    override = run_args.dfb_run_overrides[0]
    override.entry_size = 4096
    override.entry_size = None
    assert override.entry_size is None
