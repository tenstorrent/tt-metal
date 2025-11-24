# Layer 4: C++ Project Structure

File organization and CMake configuration.

## File Hierarchy

```yaml
files:
  base_path: "ttnn/cpp/ttnn/operations/<scope>/<op>/"

  components:
    - { path: "device/<op>_device_operation_types.hpp", desc: "Attributes and Args structs" }
    - { path: "device/<op>_device_operation.hpp", desc: "DeviceOperation declaration" }
    - { path: "device/<op>_device_operation.cpp", desc: "DeviceOperation implementation" }
    - { path: "device/<op>_program_factory.hpp", desc: "ProgramFactory declaration" }
    - { path: "device/<op>_program_factory.cpp", desc: "create() and override()" }
    - { path: "device/kernels/", desc: "Kernel sources" }
    - { path: "<op>.hpp", desc: "Public API declaration" }
    - { path: "<op>.cpp", desc: "Public API (calls Prim)" }
    - { path: "<op>_pybind.cpp", desc: "Python bindings" }
    - { path: "CMakeLists.txt", desc: "Build config" }
```

## CMake Template

```yaml
cmake:
  target: "ttnn_op_<scope>_<op>"
  alias: "TTNN::Ops::<Scope>::<Op>"

  config:
    - "target_precompile_headers(${target} REUSE_FROM TT::CommonPCH)"
    - "TT_ENABLE_UNITY_BUILD(${target})"

  file_sets:
    api: ["<op>.hpp"]
    kernels: ["device/kernels/*"]
    private: ["device/*.cpp", "<op>.cpp"]
```

## Naming Conventions

```yaml
naming:
  namespace: "ttnn::operations::<scope>::<op>"
  device_op: "<Op>DeviceOperation"
  factory: "<Op>ProgramFactory"
  prim: "ttnn::prim::<op>"
  public_api: "ttnn::<op>" or "ttnn::experimental::<op>"
```

## Include Order

```yaml
include_order:
  1: "Corresponding header (for .cpp)"
  2: "Project-local headers"
  3: "tt-metal/ttnn headers"
  4: "Standard library"
```
