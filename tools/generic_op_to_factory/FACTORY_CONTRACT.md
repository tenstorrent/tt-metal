# Descriptor-only factory contract

The migration flow produces a named C++ operation whose factories return
`tt::tt_metal::ProgramDescriptor`. It does not produce a regular factory with
`cached_program_t` / `create`, a `ProgramSpec`, or a `WorkloadDescriptor`.
This is a flow policy, not a restriction of the wider TTNN framework.

## Authoring

Declare these methods on each factory, using the device operation's types:

```cpp
static tt::tt_metal::ProgramDescriptor create_descriptor(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensors,
    tensor_return_value_t& outputs);

static void override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensors,
    tensor_return_value_t& outputs,
    const std::optional<ttnn::MeshCoordinate>& coordinate = std::nullopt);
```

The ordinary adapter also accepts a fourth optional-coordinate argument on
`create_descriptor`. This signature alone does not extend the mapping to a
coordinate-dependent algorithm. Do not put the refresh hook on the operation,
add a factory `apply_descriptor`, or combine it with `get_dynamic_runtime_args`.
Do not retain a legacy `cached_program_t` / `create` path: concept selection must
resolve unambiguously to `ProgramDescriptorFactoryConcept`.

Populate kernels, CBs and semaphores once, on a cache miss. Use typed `Buffer*`
runtime-argument bindings, including common arguments, and explicit tensor/CB
backing relationships. Preserve absent/conditional CBs and each actual format;
the descriptor's default format is not evidence that the source uses FP32.
Use canonical tile/format/architecture APIs for geometry and storage sizes.

Resource positions still matter with descriptors. Allocate CB indices with a
bounded allocator checked against the target's hardware limit; share the result
with kernel compile arguments. Derive kernel handles from the descriptor's
insertion order using one shared role schema. Assign semaphore IDs with
`ProgramDescriptor::find_available_semaphore_id` and check availability on all
participating cores before insertion. Never silently assume cores with different
existing allocations have the same free ID. Share host/kernel argument schemas,
derive ABI counts from them, and preserve accessor tails. A namespace of manually
numbered constants is not allocation or a shared ABI definition.

The explicit refresh hook bypasses automatic binding resolution. It therefore
owns **all** varying runtime values: per-core/common addresses, scalars, and
tensor-backed CB addresses. Use existing runtime-argument references and CB
updates; do not reconstruct the descriptor, allocate a new Program, or launch a
Python planner on hits. Prove fresh-buffer and legal distinct↔aliased transitions,
optional-presence and return-to-key hits with operation-specific cache tests.
See [MAPPING.md](MAPPING.md) for the actual adapter conditions and source links.

## Enforced gate and evidence

The required `factory_contract` config identifies the device-operation header,
qualified operation type and the registered factory `.cpp` (see [PORT_FLOW.md](PORT_FLOW.md)).
After the normal build, `validate_port` writes a C++ probe and compiles it with
`-fsyntax-only` using that translation unit's actual flags from
`build_Release/compile_commands.json`. Both direct and CMake unity compilation
entries are supported. Missing/ambiguous entries and unsupported command forms
fail explicitly; the gate neither guesses include paths nor edits CMake.
Build the target with `./build_metal.sh --export-compile-commands` (plus its
other required flags). The default build can produce a partial, UMD-only
database. This option currently disables TT unity builds, so choose it at the
first target build to avoid switching configurations late in validation.

Static assertions check **every** alternative in `program_factory_t`: the
framework descriptor concept, exact descriptor return type with a supported
signature, and the explicit adapter-compatible refresh hook. Legacy factories,
mixed descriptor/legacy variants, missing hooks and workload-only factories fail
before any source/native device validation starts. The stage retains its probe,
command, compiler log, contract receipt and compilation-database/unity hashes;
drift invalidates resume. Completion reports `factory_kind: ProgramDescriptor`.

This is not proof that a Python binding dispatches to the configured type, that
the hook updates the right argument positions, or that its body avoids expensive
work. The required independent `descriptor_factory` review must trace dispatch
and inspect the final implementation. Golden/cache tests and all existing
review topics remain required. A passing shape check is not a performance
measurement. Optional framework descriptor-patching parity instrumentation
reconstructs an oracle on hits; do not time that debug path as normal cache cost.

Historical receipts and already authored operations are unchanged. A new port
must supply this contract in a new validation workspace; there is no option to
silently fall back to regular factories.
