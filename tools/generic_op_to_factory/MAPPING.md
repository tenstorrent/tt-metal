<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Generic op → C++ program factory: mapping rules

Mapping guidance for migrating a Python `ttnn.generic_op` operation using a
`ProgramDescriptor` to a named C++ device operation with `create_descriptor`.
This document covers ordinary single-device planners, including file-backed and
inline kernels. `ProgramSpec` and coordinate-dependent/workload-scoped mesh
factories need separate mappings.

**Flow output policy:** new ports must use a C++ `ProgramDescriptor` factory
with typed buffer bindings and an explicit factory-owned per-Program
`override_runtime_arguments` hook. Regular `CachedProgram` factories are not an
alternative output. [FACTORY_CONTRACT.md](FACTORY_CONTRACT.md) defines the
compile-time validation gate and the additional dispatch/wiring review. This
policy does not retroactively change previously authored operations.

Implementation facts below are checked against `dfc0dae18e45`. Requirements
labelled **migration policy** are choices for this migration, not capabilities
or guarantees enforced by the descriptor API. Planner arithmetic is interpreted
under §5; an unlisted API needs investigation or an explicit refusal, not a
silent approximation. Source links at the end identify the evidence to re-check
when migrating to another revision.

## Python and C++ binding surfaces

The Python descriptor classes are nanobind bindings of *the same C++ structs*.
`ttnn.CBDescriptor` **is** `tt::tt_metal::CBDescriptor`. So most of this mapping
is not a translation at all — it is re-expressing the same struct from C++.

The binding does not expose every field directly. In particular, runtime-arg
buffer bindings and CB backing storage have different Python surfaces:

| C++ | Purpose | In Python? |
|---|---|---|
| `KernelDescriptor::buffer_bindings` (via `emplace_runtime_args`) | declare a runtime-arg slot containing a buffer base address | **no** |
| `KernelDescriptor::common_buffer_bindings` (via `emplace_common_runtime_args`) | the same for common args | **no** |
| `CBDescriptor::buffer` | buffer-backed CB | **yes**, through `ttnn.cb_descriptor_from_sharded_tensor` and `set_buffer_from_cb`; no direct pointer setter |
| `CBDescriptor::tensor` | `const MeshTensor*` backing storage | no direct Python field |
| `CBDescriptor::global_circular_buffer` | global circular-buffer backing storage | **yes**, through `set_global_circular_buffer` and `set_global_circular_buffer_from_cb` |

A Python descriptor planner cannot declare a per-core or common runtime-arg
`BufferBinding` through these bindings. To put a tensor base address in such a
slot, it passes an integer. For example:

```python
reader_rt = ttnn.RuntimeArgs()
reader_rt[0][0] = [input_tensor.buffer_address(), n, start]
# Pass reader_rt as KernelDescriptor's runtime_args argument.
```

**Migration policy:** use typed runtime-arg buffer bindings in C++ where
applicable. A raw integer carries no buffer identity for the resolver. A manual
override can patch it without rebuilding, but omitting both the binding and an
appropriate override can cause rebuilds or stale addresses, depending on which
adapter path is selected. Bindings alone do not guarantee a fast cache hit; see
§2. The Python CB helpers already preserve buffer identity.

Evidence: [Python bindings][bindings], [C++ descriptors][descriptors],
[sharded CB helper][tensor-utils], and [cache-hit adapter][adapter].

So constructs fall into three classes: **identity** (re-express as-is),
**upgrade** (the Python form is a workaround; C++ must use the real mechanism),
and **relocation** (wrapper responsibilities assigned to native operation layers).

---

## 1. Identity — same struct, re-express

Preserve the resulting descriptor state, including defaults and constructor
conversions. Rows listing fields below describe state, not Python constructor
signatures; not every writable property is also a constructor argument.
Translate positional Python arguments using the binding's parameter names;
their order need not match C++ aggregate field order (notably compute config).

| Python | C++ | Notes |
|---|---|---|
| `ttnn.ProgramDescriptor(kernels=, semaphores=, cbs=)` | `tt::tt_metal::ProgramDescriptor` — `.kernels`, `.semaphores`, `.cbs` | push order is load-bearing; see §4 |
| `ttnn.KernelDescriptor`: `kernel_source`, `source_type`, `core_ranges`, `compile_time_args`, `named_compile_time_args`, `defines`, `config`, `compiler_include_paths`; constructor `opt_level=` | same fields, including `.opt_level` | `opt_level` is accepted by the Python constructor but has no exposed property |
| `ttnn.CBDescriptor`: `total_size`, `core_ranges`, `format_descriptors`, `remote_format_descriptors`, `address_offset` | same fields | only the first three are accepted by the non-default Python constructor; the others are assigned afterward |
| `ttnn.CBFormatDescriptor`: `buffer_index`, `data_format`, `page_size`, `tile`, `face_geometry` | same fields after conversion | Python's TTNN `DataType` constructor overload calls `datatype_to_dataformat_converter`; C++ stores `tt::DataFormat`. Assign `face_geometry` after Python construction. Use `data_format_as_uint8` for inspection: the binding documents that `.data_format` cannot be read because its enum is unbound. The accessor adds no state |
| `ttnn.TileDescriptor(height=, width=, transpose=)` | same | |
| `ttnn.FaceGeometry(face_r_dim=, num_faces=)` | same | |
| `ttnn.SemaphoreDescriptor(id=, core_type=, core_ranges=, initial_value=)` | same fields | Python `id` is read-only after construction |
| `ttnn.ReaderConfigDescriptor()` / `WriterConfigDescriptor()` | same | |
| `ttnn.DataMovementConfigDescriptor(processor=, noc=, noc_mode=)` | same | |
| `ttnn.ComputeConfigDescriptor`: `math_fidelity`, `fp32_dest_acc_en`, `dst_full_sync_en`, `unpack_to_dest_mode`, `bfp8_pack_precise`, `math_approx_mode`, `enable_trisc2_rvv` | same fields | Python assigns `unpack_to_dest_mode` and `enable_trisc2_rvv` after construction |
| `ttnn.TensorAccessorArgs(tensor).get_compile_time_args()` | `tt::tt_metal::TensorAccessorArgs(*tensor.buffer()).get_compile_time_args()` | same values, same order |
| `ttnn.TensorAccessorArgs()` for an absent optional operand | default-constructed `tt::tt_metal::TensorAccessorArgs` | preserve its placeholder arguments |
| `TensorAccessorArgs.get_common_runtime_args()` | same method in C++ | classify returned values using §4 |
| `ttnn.CoreCoord` / `CoreRange` / `CoreRangeSet` | same types | |
| non-address runtime args (tile counts, offsets, flags) | `runtime_args.emplace_back(core, CoreRuntimeArgs{...})` | raw values only; see row U1 for anything address-derived |
| non-address `common_runtime_args` | `.common_runtime_args` | preserve values and order; patch any hash-excluded values on hits |
| `ttnn.cb_descriptor_from_sharded_tensor(...)` | `ttnn::cb_descriptor_from_sharded_tensor(...)` | preserve backing buffer, format, tile, aligned sizes, core override, and offset; cache-hit caveats in §2 |
| CB `set_buffer_from_cb`, `set_global_circular_buffer`, `set_global_circular_buffer_from_cb` | assign the corresponding backing pointer | preserve resource lifetime; global CB resources are not automatically rebound as input/output tensors |

For `SourceType::FILE_PATH`, **migration policy** is to use a stable
repository-relative path, not derive it from `__FILE__`. For
`SourceType::SOURCE_CODE`, preserve the inline source string and source type;
the [program constructor][program] calls `CreateKernelFromString`. Existing
[inline-kernel examples][inline-example] exercise this surface.

The four Python `blaze_named_*` runtime-arg properties populate
`KernelDescriptor::blaze_named_args`. They are exposed but marked experimental
and temporary by the bindings. **Migration policy:** refuse planners using
these properties until a dedicated mapping is supplied; do not drop them.

Evidence: [Python bindings][bindings], [C++ descriptors][descriptors], and
[TensorAccessorArgs bindings][accessor-bindings].

## 2. Upgrade — the Python form is a workaround

These rows are where faithfulness means preserving *behaviour*, not shape.

| Id | Python | C++ | Why |
|---|---|---|---|
| **U1** | `rt[x][y] = [t.buffer_address(), n, start]` using `ttnn.RuntimeArgs` | `kernel.emplace_runtime_args(core, {t.buffer(), n, start})` | records a `BufferBinding`; subject to the resolver and cache-hit conditions below. The smuggled-RTA guard detects raw-address patterns in descriptor sinks |
| **U2** | common args carrying an address | `kernel.emplace_common_runtime_args(...)` | same mechanism for common args |
| **U3** | a manually supplied tensor base address for a CB | use `.buffer` or `.tensor` backing storage; `.address_offset` is relative to that buffer base | applies only when the tensor backing and relative offset are established. Do not copy an absolute address into `.address_offset` or reinterpret unrelated fixed L1 storage as a tensor binding. See cache-hit conditions below |
| **U4** | `desc.custom_program_hash = ...` | omit the descriptor hash and audit the named operation's key under §4 | the generic descriptor path hashes the realized descriptor or uses its custom hash. A named operation normally hashes reflected attributes and tensor arguments |
| **U5** | `os.environ` / `_knob()` tuning reads | **migration policy:** resolve the knob into a typed `operation_attributes_t` field with a default | ensure structural tuning enters the hash; a cache-miss-only environment read is not itself hashed. Reading an environment value before launch and including it in attributes can be cache-correct |
| **U6** | kernel `#include` of a package-local helper snapshot | **migration policy:** use its verified canonical in-tree helper, often under `ttnn/cpp/ttnn/kernel_lib/` | first establish equivalent behavior and required API availability; matching a helper name or path alone does not establish equivalence |

**Cache-hit behavior at the checked revision:** for the ordinary
`ProgramDescriptor` adapter, an `override_runtime_arguments` hook on the
program factory supersedes both `resolve_bindings` and
`get_dynamic_runtime_args`. The adapter static-asserts against putting this
hook on the device operation or combining it with `get_dynamic_runtime_args`.
It must patch every varying runtime value, including per-core/common buffer
addresses and tensor-backed CB addresses. Use `GetRuntimeArgs`, `GetCommonRuntimeArgs`, and
the appropriate CB address update. **Migration policy:** do not rebuild the
full descriptor in this hook. The flow requires this explicit hook even when
the automatic resolver would suffice for the initial non-aliased case; it must
cover every supported alias transition and every dynamic field.

Without an override, the adapter takes its binding fast path only if resolved
runtime-arg bindings are nonempty, or both dynamic scalar args and resolved
bindings are nonempty. Otherwise it invokes `create_descriptor` again and
applies its runtime state. Consequences:

- CB-only bindings with no dynamic args still rebuild on cache hits.
- Repeated buffers within the input region (including some optional-output
  aliases) can make the resolver return empty bindings and force a rebuild.
  Do not enable aliasing opt-ins without proving their stronger contract.
- Each runtime-arg `Buffer*` must be reachable through the adapter's tensor
  enumeration. A declared pointer is not a stable logical operand identity.
- Omitting the override can be sufficient for correctness when only addresses
  vary, but does not establish that descriptor reconstruction has been removed.
  Verify cache hits with new allocations, supported alias patterns, and CB-only
  cases as applicable. No speedup follows from the API shape alone.

`get_dynamic_runtime_args` exists for older scalar-patching implementations;
the [migration recipe][recipe] directs new migrations toward the override.
See [adapter `apply_descriptor`][adapter] and
[resolver `resolve_bindings`][patching] for the actual conditions. Workload
descriptor factories have different rules and are outside this mapping.
The resolver and scalar-patching helpers are explicitly marked temporary in
their implementation; do not add direct callers from migrated operations.

For mixed C++ arg lists, use unsigned literals such as `0u`, explicitly narrow
wider integers after checking their range, and use `KernelDescriptor::RTArgList`
for dynamically constructed lists. A null `Buffer*` emits zero without a
binding; optional presence must remain structural unless the override handles
the transition. See [descriptor overloads][descriptors].

## 3. Relocation — wrapper responsibilities

Separate wrapper behavior according to its role. These are migration placement
choices; the API does not enforce a unique source-file layout.

| Python (in the wrapper) | C++ destination |
|---|---|
| argument checks that must hold on every call | public entry point or validation invoked on both hits and misses; preserve the source exception contract, using `TT_FATAL` only where compatible |
| registry support refusal (`SUPPORTED` / `EXCLUSIONS` / `validate`) | native validation plus explicit Python exception translation where needed; preserve the support-refusal class, and validate before allocation if allocation could fail first |
| `ttnn.allocate_tensor_on_device(shape, dtype, layout, device, memory_config)` | `compute_output_specs` returns the `TensorSpec`; `create_output_tensors` allocates it |
| scalars / config / tuning knobs the planner reads | fields of `operation_attributes_t` |
| tensor arguments, with optionality | fields of `tensor_args_t` (`std::optional<Tensor>` for optional operands) |
| `create_program_descriptor(...)` body | `create_descriptor(attributes, tensor_args, output)` |
| `ttnn.generic_op(io_tensors, descriptor)` | `ttnn::device_operation::launch<Op>(attributes, tensor_args)` from `ttnn::prim::<op>` |
| the value the wrapper returns | `tensor_return_value_t` — derive it from the *wrapper*, not from `generic_op`, which returns the last IO tensor |

Evidence: [generic-op output selection and descriptor hashing][generic-op],
and [device-operation dispatch][dispatch].

`validate_on_program_cache_miss` is also called on a hit when the operation
does not define `validate_on_program_cache_hit`; the [adapter][adapter] supplies
that fallback. If a hit validator is defined, it owns the checks required on
hits. Miss-only checks are appropriate only when their validity is guaranteed
by the cache key and other per-call checks.

The Python [support-contract module][op-contract] defines `SupportRefusal`,
`UnsupportedAxisValue`, and `ExcludedCell`, all deriving from
`NotImplementedError`. Moving a check to C++ does not automatically recreate
these Python classes. A port must supply and verify a binding-layer translator
or retain the appropriate boundary validation. The module documents typed
harness matching; the particular source planner and harness must also be
identified before claiming their exact behavior is preserved.

Recommended layout (some existing operations split types and factories into
additional headers or reuse kernels from another operation):

```
ttnn/cpp/ttnn/operations/<path>/<op>/
├── <op>.hpp  <op>.cpp  <op>_nanobind.cpp
└── device/
    ├── <op>_device_operation.hpp   (attributes, tensor args, factory, ttnn::prim decl)
    ├── <op>_device_operation.cpp   (validate / specs / outputs / prim body)
    ├── <op>_program_factory.cpp    (create_descriptor)
    └── kernels/                    (own file-backed kernels, when needed)
```

`device/` is not cosmetic: `.pre-commit-config.yaml` scopes `detect-smuggled-rta`
and `detect-override-rebuild` to `^ttnn/.*/device/.*\.(cpp|hpp)$`. A factory
elsewhere is outside both guards.

Reuse existing kernel files when available. Preserve their behavior and source
type; a verified include relocation under U6 is an intentional source change.
Add new source files to the relevant `sources.cmake`, as required by `AGENTS.md`.

## 4. Classification — structural and per-dispatch values

For every value the planner computes, decide **structural** or **per-dispatch**:

- **Structural** — affects kernels, compile-time args, layouts, work sets, CB
  sizes, core ranges. Belongs in the cache key; baked at cache-miss time.
- **Per-dispatch** — must be re-applied on every dispatch.

The default hash in [device-operation dispatch][dispatch] combines
`type_hash<Op>`, `operation_attributes_t`, and `tensor_args_t` through reflection.
The [Tensor reflection surface][tensor] includes storage and `TensorSpec`;
[DeviceStorage][storage] reflects no buffer pointers. For ordinary device
tensor arguments, specs contribute shape, dtype, layout, memory config, tile
and shard information, not allocation addresses.

The actual mesh program-cache key also includes target coordinates and a
canonical encoding used to distinguish hash collisions. With a custom
`compute_program_hash`, that canonical encoding contains only the operation
type identity; attribute-level collision resolution is disabled. See
`compute_mesh_workload_hash` and `compute_mesh_workload_canonical_key` in the
[adapter][adapter].

This is not a guarantee that every declared attribute is hashed: custom
`attribute_values()` can omit fields (the checked-in [Bernoulli attributes][bernoulli]
omit `seed`). **Migration policy:** reflect all structural fields and use an
explicit custom hash for intentional exclusions, rather than narrowing
reflection. With a complete key, spec-derived values may be baked. Audit any
other structural inputs, including device/configuration queries, and reapply
varying addresses on hits.

Prefer the default hash when it captures all structural inputs. A custom
`compute_program_hash` is appropriate to include structural state not already
represented, or to exclude/canonicalize values that do not change program
structure. Factory selection must remain unambiguous under the key; include
the selected variant when it is not already determined by hashed inputs.
Merely deriving a value from already-hashed inputs does not require hashing it
again. For each excluded dynamic value, comment which mechanism reapplies it.
A custom hash runs on every dispatch, so keep expensive descriptor construction
out of it. See [hash guidance][hash-guide] and [migration recipe][recipe]; these
are recommendations to reconcile with the implementation, not hash code.

Push order of `desc.kernels` / `desc.cbs` is part of the contract: an override
hook and `resolve_bindings` both index by position.

## 5. Interpreted — the planner's arithmetic

Everything else. Work splits, CB sizing, block factors, mask spans, compile-time
arg vectors, per-core assignment: ordinary imperative arithmetic, re-expressed
in typed C++. There is no table for it and it does not need one.

Migration policy for that interpretation:

- **Preserve semantics that change results.** Floor division and modulo sign for
  negative operands, integer width, float→bit-pattern conversions, narrowing
  points, iteration order where order is observable.
- **Preserve the established error contract.** The support-contract module
  documents class-based matching. For a harness confirmed to use that contract,
  exact `repr` and message formatting need not be reproduced. Inspect other
  callers/tests before deciding their error text is irrelevant.
- **Do not refactor.** Same branch structure, same constants, same arithmetic.
  An algebraically equivalent rewrite that happens to pass is still a departure.
- **Prefer the API over a reimplementation.** `tt::round_up` / `div_up` /
  `round_down` from [math.hpp][math], when sign and overflow preconditions match
  the source; these formulas are not general Python signed-division replacements.
  Use `Buffer::aligned_page_size()` when an aligned stride is required, rather
  than silently substituting it for an unaligned size. The capacity calculation
  `device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1)`
  is used by [Moreh group norm][group-norm]; it is not a query for currently free
  allocator space or a universal substitute for the source planner's budget.
- **Label transcribed constants.** A tuning constant carried over from the
  source is legitimate; an unlabelled one is indistinguishable from an invented
  one. Say what constrains it.
- **Keep device-affecting host work outside the factory.** Ordinary planner
  arithmetic and descriptor/vector construction are host work and belong in
  `create_descriptor`. Tensor allocation, host-generated tensor fills, and
  device buffer I/O do not; the [operation review guidance][ops-review] flags
  these as tracing hazards.
- **Clean under this repository's `.clang-tidy`.** It is a merge gate, not
  advice: [pr-gate.yaml][pr-gate] routes eligible changes through
  [code-analysis.yaml][code-analysis] to [clang-tidy-reusable.yaml][clang-tidy-ci]
  with `--warnings-as-errors=*`. Diagnostics fail that check. Inline reporting
  is conditional (including non-fork PRs and the step's `success()` guard)
  and filtered to diff context;
  machine-applicable suggestions require a suitable replacement. Registered
  `.cpp` sources can be checked against the build's compile database.
  Two relevant checks:
  bind `Shape`-like accessors by const reference — `const auto& padded =
  t.padded_shape();`, since the by-value form copies
  (`performance-unnecessary-copy-initialization`) — and return the negation
  rather than `if (cond) return false; return true;`
  (`readability-simplify-boolean-expr`). Note clang-format is a pre-commit hook
  but clang-tidy is not; formatting hooks never load `.clang-tidy`.

## 6. Refuse rather than guess

The following are outside this migration's defined scope and must fail loudly,
not be approximated. This is migration policy, not a claim that the underlying
framework cannot express these constructs:

- A construct with no mapping above and no interpretation under §5 —
  `ProgramSpec` frontend, coordinate-dependent or workload-scoped mesh dispatch,
  Blaze named runtime args, arbitrary Python classes, generators,
  external calls, mutable module state, address arithmetic.
- A helper the source needs that the target checkout does not contain. Report
  which; do not ship a private copy.
- A vendored helper with no recorded canonical path — it has no in-tree home.
- Anything requiring a Python callback at dispatch time. There is no fallback to
  `generic_op` or `GenericOpDeviceOperation`.

---

## Evidence and verification

Implementation sources take precedence over comments, recipes, and examples
when establishing current API or cache behavior. In particular, older migration
guidance does not fully describe the adapter's binding fallbacks. Existing
operations demonstrate patterns, not universal conformance to this policy.

The [smuggled-RTA guard][rta-guard] and [override-rebuild guard][rebuild-guard]
are text checks with documented suppressions; the latter also has a baseline
for existing violations. Their [pre-commit scope][pre-commit] does not make them
a proof of cache correctness. Kernel order and CB order must agree with the
indices used by the resolver and any override.

For each migrated operation, record the source planner/wrapper revision and
the applicable harness. Verify supported results, rejection classes, cache hits
with changed allocations and dynamic scalars, and supported alias/CB cases.
Check that the chosen hit path avoids full descriptor reconstruction if that
is the migration objective. Device correctness, tracing, and performance need
their respective tests or measurements; this document's static source audit
does not establish them.

[bindings]: ../../ttnn/cpp/ttnn-nanobind/program_descriptors.cpp
[descriptors]: ../../tt_metal/api/tt-metalium/program_descriptors.hpp
[tensor-utils]: ../../ttnn/core/tensor/tensor_utils.cpp
[accessor-bindings]: ../../ttnn/cpp/ttnn-nanobind/tensor_accessor_args.cpp
[program]: ../../tt_metal/impl/program/program.cpp
[inline-example]: ../../ttnn/ttnn/operations/examples/reduce_block/program_descriptor_with_inline_kernels.py
[adapter]: ../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp
[patching]: ../../tt_metal/impl/program/program_descriptor_patching.cpp
[dispatch]: ../../ttnn/api/ttnn/device_operation.hpp
[generic-op]: ../../ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
[bernoulli]: ../../ttnn/cpp/ttnn/operations/bernoulli/device/bernoulli_device_operation.hpp
[tensor]: ../../ttnn/api/ttnn/tensor/tensor.hpp
[storage]: ../../ttnn/api/ttnn/tensor/storage.hpp
[op-contract]: ../../ttnn/ttnn/operations/_op_contract.py
[recipe]: ../../.cursor/commands/ttnn/descriptor-migration-recipe.md
[hash-guide]: ../../.cursor/commands/ttnn/verify-device-operation-hash.md
[ops-review]: ../../.github/instructions/ttnn-ops.instructions.md
[math]: ../../tt_metal/api/tt-metalium/math.hpp
[group-norm]: ../../ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/moreh_group_norm_program_factory.cpp
[pr-gate]: ../../.github/workflows/pr-gate.yaml
[code-analysis]: ../../.github/workflows/code-analysis.yaml
[clang-tidy-ci]: ../../.github/workflows/clang-tidy-reusable.yaml
[rta-guard]: ../../scripts/detect_smuggled_rta.py
[rebuild-guard]: ../../scripts/detect_override_rebuild.py
[pre-commit]: ../../.pre-commit-config.yaml
