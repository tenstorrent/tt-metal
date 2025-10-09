I'll look up how these C++ configs are exposed via pybind to see what attributes/constructors are available, then propose a codegen path that reconstructs them either via keyword ctor, setters, or a custom snapshot/restore helper. Now I'll search the repo for their bindings and Python-facing APIs.

[6 tools called]

I found the C++ pybind definitions for these config types. I'll open the binding file to see what constructors and attributes are exposed so we can reliably reconstruct them.

[12 tools called]

I checked the pybinds—both configs expose keyword ctors + readable fields. I’ll give you a serializer that emits valid Python source to recreate them.

```python
import ttnn

def _enum_ref(e):
    enum_type = e.__class__.__name__
    try:
        name = e.name  # pybind11 enums usually have .name
    except AttributeError:
        # Fallback via __members__ or int value mapping
        members = getattr(e.__class__, "__members__", {})
        inv = {v: k for k, v in members.items()}
        name = inv.get(e, str(int(e)))
    return f"ttnn.{enum_type}.{name}"

def emit_wormhole_compute_kernel_config(cfg) -> str:
    return (
        "ttnn.WormholeComputeKernelConfig("
        f"math_fidelity={_enum_ref(cfg.math_fidelity)}, "
        f"math_approx_mode={bool(cfg.math_approx_mode)}, "
        f"fp32_dest_acc_en={bool(cfg.fp32_dest_acc_en)}, "
        f"packer_l1_acc={bool(cfg.packer_l1_acc)}, "
        f"dst_full_sync_en={bool(cfg.dst_full_sync_en)}, "
        f"throttle_level={_enum_ref(cfg.throttle_level)})"
    )

def _emit_core_coord(c) -> str:
    # pybind exposes 2-arg CoreCoord(x, y) for mem config module
    return f"ttnn.CoreCoord({int(c.x)}, {int(c.y)})"

def _emit_core_range(r) -> str:
    return f"ttnn.CoreRange({_emit_core_coord(r.start)}, {_emit_core_coord(r.end)})"

def _emit_core_range_set(crs) -> str:
    ranges = list(crs.ranges())
    ranges_src = ", ".join(_emit_core_range(r) for r in ranges)
    # CoreRangeSet ctor accepts list/vec
    return f"ttnn.CoreRangeSet([{ranges_src}])"

def _emit_shard_spec(spec) -> str:
    shape = spec.shape  # e.g. [h, w] or tuple
    h, w = int(shape[0]), int(shape[1])
    grid_src = _emit_core_range_set(spec.grid)
    orient_src = _enum_ref(spec.orientation)  # ShardOrientation
    return f"ttnn.ShardSpec({grid_src}, ({h}, {w}), {orient_src})"

def emit_memory_config(mc) -> str:
    # Prefer simple ctor when no shard spec
    shard_spec = mc.shard_spec
    nd_shard_spec = getattr(mc, "nd_shard_spec", None)

    if shard_spec is not None:
        ml_src = _enum_ref(mc.memory_layout)      # TensorMemoryLayout
        bt_src = _enum_ref(mc.buffer_type)        # BufferType
        ss_src = _emit_shard_spec(shard_spec)
        return f"ttnn.MemoryConfig({ml_src}, {bt_src}, {ss_src})"

    # ND sharding path (if present)
    if nd_shard_spec is not None:
        # Most tests construct NdShardSpec(shape, grid)
        # Pull out minimal fields if exposed; otherwise repr() and raise
        if hasattr(nd_shard_spec, "shape") and hasattr(nd_shard_spec, "grid"):
            shp = getattr(nd_shard_spec, "shape")
            dims = list(getattr(shp, "tolist", lambda: list(shp))())
            shape_src = f"ttnn.Shape({dims})"
            grid_src = _emit_core_range_set(nd_shard_spec.grid)
            bt_src = _enum_ref(mc.buffer_type)
            return f"ttnn.MemoryConfig({bt_src}, ttnn.NdShardSpec({shape_src}, {grid_src}))"
        raise ValueError("NdShardSpec serialization requires .shape and .grid")

    # Interleaved/simple
    ml_src = _enum_ref(mc.memory_layout)
    bt_src = _enum_ref(mc.buffer_type)
    return f"ttnn.MemoryConfig({ml_src}, {bt_src})"
```

Example with your generator:
```python
compute_kernel_config, memory_config = generate_hardware_config(hw_config)
ck_src = emit_wormhole_compute_kernel_config(compute_kernel_config)
mem_src = emit_memory_config(memory_config)
# lines you can splice into generated code:
# f"compute_kernel_config = {ck_src}"
# f"memory_config = {mem_src}"
```

Notes:
- Works because bindings expose KW ctors + public fields.
- If you later add sharding, `_emit_shard_spec` reconstructs `CoreRangeSet`/`CoreRange`/`CoreCoord`.
- If any enum lacks `.name`, the fallback maps via `__members__` or uses integer value.

Sources:
```55:70:/localdev/gwang/tmp/tt-metal-3/ttnn/cpp/ttnn-pybind/operations/core.cpp
py::class_<WormholeComputeKernelConfig>(module, "WormholeComputeKernelConfig")
    .def(
        py::init<MathFidelity, bool, bool, bool, bool, ttnn::operations::compute_throttle_utils::ThrottleLevel>(),
        py::kw_only(),
        py::arg("math_fidelity") = MathFidelity::Invalid,
        py::arg("math_approx_mode") = true,
        py::arg("fp32_dest_acc_en") = false,
        py::arg("packer_l1_acc") = false,
        py::arg("dst_full_sync_en") = false,
        py::arg("throttle_level") = compute_throttle_utils::ThrottleLevel::NO_THROTTLE)
    .def_readwrite("math_fidelity", &WormholeComputeKernelConfig::math_fidelity)
    .def_readwrite("math_approx_mode", &WormholeComputeKernelConfig::math_approx_mode)
    .def_readwrite("fp32_dest_acc_en", &WormholeComputeKernelConfig::fp32_dest_acc_en)
    .def_readwrite("packer_l1_acc", &WormholeComputeKernelConfig::packer_l1_acc)
    .def_readwrite("dst_full_sync_en", &WormholeComputeKernelConfig::dst_full_sync_en)
    .def_readwrite("throttle_level", &WormholeComputeKernelConfig::throttle_level);
```

```343:404:/localdev/gwang/tmp/tt-metal-3/ttnn/cpp/ttnn-pybind/tensor.cpp
pyMemoryConfig
    .def(py::init<>([](TensorMemoryLayout memory_layout, BufferType buffer_type, std::optional<ShardSpec> shard_spec) {
            return MemoryConfig{memory_layout, buffer_type, std::move(shard_spec)};
        }),
        py::arg("memory_layout") = TensorMemoryLayout::INTERLEAVED,
        py::arg("buffer_type") = BufferType::DRAM,
        py::arg("shard_spec") = std::nullopt)
    .def(py::init<>([](BufferType buffer_type, NdShardSpec nd_shard_spec) {
            return MemoryConfig{buffer_type, std::move(nd_shard_spec)};
        }),
        py::arg("buffer_type"),
        py::arg("nd_shard_spec"))
    .def_property_readonly("buffer_type", &MemoryConfig::buffer_type)
    .def_property_readonly("memory_layout", &MemoryConfig::memory_layout)
    .def_property_readonly("shard_spec", &MemoryConfig::shard_spec)
    .def_property_readonly("nd_shard_spec", &MemoryConfig::nd_shard_spec);
```

- Built the serializer for the pybind-backed configs you flagged. If you want, I can wire this into `function_to_source` so the emitted function includes exact `compute_kernel_config`/`memory_config` assignments.
