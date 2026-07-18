# Metal 2.0 Migration Guide (WH/BH)

This guide helps developers migrate from the legacy `ProgramDescriptor` / `host_api.hpp` APIs to the new  Metal 2.0 host APIs in `tt_metal/api/tt-metalium/experimental/metal2_host_api/`.

The focus of this guide is **Wormhole** and **Blackhole** (Gen1 architectures).

Things to remember:
 - The Metal 2.0 APIs are experimental and subject to changes.
 - Metal 2.0 is a work in progress; not all legacy API features fully implemented.
 - Not all planned improvements are available yet.

> **Prerequisite**: Before attempting Metal 2.0 migration, complete the [Device 2.0 Data Movement migration](../../kernel_apis/data_movement/device_api_migration_guide.md). All kernel code migration examples in this guide assume Device 2.0 migration is already done.

## Table of Contents

1. [Overview](#overview)
2. [Concept Map](#concept-map)
3. [Header Files](#header-files)
4. [Host API Migration](#host-api-migration)
   - [ProgramSpec](#programspec)
   - [KernelSpec](#kernelspec)
   - [DataflowBufferSpec](#dataflowbufferspec)
   - [SemaphoreSpec](#semaphorespec)
   - [TensorParameter](#tensorparameter)
   - [WorkUnitSpec](#workunitspec)
   - [ProgramRunArgs](#programrunargs)
5. [Device-Side Migration](#device-side-migration)
   - [Circular Buffers → Dataflow Buffers](#circular-buffers--dataflow-buffers)
   - [TensorAccessor](#tensoraccessor)
   - [Kernel Argument Retrieval Syntax](#kernel-argument-retrieval-syntax)
6. [Design Principles](#design-principles)
   - [Principle 1: Immutable spec, mutable run-args](#principle-1-immutable-spec-mutable-run-args)
   - [Principle 2: First-class named resource bindings](#principle-2-first-class-named-resource-bindings)
   - [Principle 3: Named arguments throughout](#principle-3-named-arguments-throughout)
7. [TTNN Framework Integration](#ttnn-framework-integration)
8. [Complete Migration Examples](#complete-migration-examples)
9. [Troubleshooting](#troubleshooting)
   - [Cryptic error → likely cause](#cryptic-error--likely-cause)

---

## Overview

Metal 2.0 introduces a new family of host APIs for Program specification. Metal 2.0 is designed to:
 - Enable **Quasar** (which the legacy APIs do not — and will not — support)
 - Address longstanding user pain points in the legacy APIs

Key Metal 2.0 changes, at a glance:
 - *Immutable and mutable descriptors*. Like `ProgramDescriptor`, Metal 2.0 is a descriptor-based API. But, it separates the mutable properties of a Program (`ProgramSpec`) from those properties that are updated for each execution (`ProgramRunArgs`).
 - *Dataflow Buffers (DFBs)* replace Circular Buffers (CBs). Both host and device-side syntax is improved.
 - *Kernel arguments* specification (host side) and retrieval (device side) are signficantly improved. (_Note_: Only the first of several improvements is currently available in Metal 2.0; expect further changes to this part of the API.)
 - *Resource placement* (i.e. `core_ranges`) is inferred where possible to make the API more AI-friendly and more intuitive with Quasar's multi-threaded kernels. The mapping of kernels to worker nodes in the device is communicated via a new top-level concept (`WorkUnitSpec`).
- *Quasar-specific features* like kernel threading.

Some benefits of Metal 2.0:

- **Named resource bindings**. Metal 2.0 natively supports binding resources (DFBs, semaphores, tensors, etc) to kernels. The corresponding handles are cleanly passed to the device code with user-defined accessor names.
- **Named arguments**. Compile-time, runtime, and common runtime arguments are addressed by name on both the host and device sides. Positional CTAs are gone from the API entirely; positional RTAs/CRTAs survive only as **varargs**, intended for the rare kernels that read a dynamic-count argument tail in a loop (e.g. shape of an N-D tensor where N is a CTA). For everything else, prefer names.

> **Stated goal: eliminate raw pointer arguments.** With DFBs, semaphores, and tensors all bindable as named resources, runtime args carrying a buffer or tensor address should now be the exception rather than the rule. If you're about to put `tensor.buffer()->address()` in a runtime arg, you're probably doing it wrong — bind the tensor as a `TensorParameter` instead.

> **Argument naming.** Metal 2.0 is designed around named arguments. Compile-time arguments must be named — positional CTAs are no longer part of the API. Runtime and common runtime arguments may be named (the typical case) or positional (varargs, intended for kernels with a genuinely dynamic argument count consumed in a loop — e.g., an N-dimensional shape where N is a CTA). When porting from a legacy kernel, individually-known RTAs translate naturally to named RTAs; reach for varargs only when the kernel actually loops over the arguments.

Many additional improvements are planned, but are not yet available in the experimental APIs.

---

## Concept Map

| ProgramDescriptor (legacy) | Metal 2.0 |
|---|---|
| `ProgramDescriptor` | `ProgramSpec` (immutable description) + `ProgramRunArgs` (per-execution values) |
| `KernelDescriptor` | `KernelSpec` |
| `KernelDescriptor::core_ranges` | derived from `WorkUnitSpec::target_nodes` |
| `KernelDescriptor::compile_time_args` (positional)<br>`KernelDescriptor::named_compile_time_args` (named — partial legacy support) | `KernelSpec::compile_time_args` (named *only*) |
| `KernelDescriptor::runtime_args` / `common_runtime_args` | **Schema** (names): declared on `KernelSpec::runtime_arg_schema`<br>**Values**: supplied per execution on `ProgramRunArgs::KernelRunArgs` |
| `CBDescriptor` | `DataflowBufferSpec` (placement derived from kernel bindings) |
| `SemaphoreDescriptor` | `SemaphoreSpec` |
| `TensorAccessorArgs<...>` <br> (plumbing + buffer-address RTA) | `TensorAccessor(ta::name)` in the kernel code; <br>`TensorParameter` on `ProgramSpec` (parallel to DFB / Semaphore) |
| *(no analogue)* | `WorkUnitSpec` — declares groups of kernels that operate together on a worker node, and on which nodes they run |
| `CoreCoord` / `CoreRange` / `CoreRangeSet`  | `NodeCoord` / `NodeRange` / `NodeRangeSet` |

> **Terminology Note**: Metal 2.0 renames "core" to "node" to disambiguate a previously overloaded term. The legacy "core" was used for both individual RISC-V cores within a node *and* for the larger hardware unit at a given (x,y) NoC address. In Metal 2.0, "node" means the latter.

---

## Header Files

```cpp
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
```

Sub-headers are pulled in transitively:
 - `kernel_spec.hpp`
 - `dataflow_buffer_spec.hpp`
 - `semaphore_spec.hpp`
 - `tensor_parameter.hpp`

**These header files are self-documenting, with extensive comments.** Please read them!

All Metal 2.0 host API symbols currently live in the `tt::tt_metal::experimental` namespace.

> **Note:** The Program-creation entry points in `metal2_host_api/program.hpp` are *temporary*. Migration to the final form will take place when the APIs leave `experimental` to join the main API directory. User code updates will be trivial.

---

## Host API Migration

> **Type system heads-up (before the per-spec sections below).** Two API-wide conventions to keep in mind as you read the examples:
>
> - **Spec names are `ttsl::StrongType`-wrapped, not bare strings.** Each spec category has its own name type — `KernelSpecName`, `DFBSpecName`, `SemaphoreSpecName`, `TensorParamName` (all wrapping `std::string`). Strong types forbid implicit conversion from `std::string` or `const char*`, so `.unique_id = "reader"` does not compile; the literal must be explicitly wrapped: `.unique_id = KernelSpecName{"reader"}`. The doc examples use a typed-constants pattern — declare each name once as a typed `const`, then reference the constant at each use site:
>     ```cpp
>     const KernelSpecName  READER{"reader"};
>     const DFBSpecName     MY_DFB{"my_dfb"};
>     const SemaphoreSpecName DONE{"done"};
>     const TensorParamName INPUT{"input"};
>     ```
>   (`ProgramSpec::name` and `WorkUnitSpec::name` are *not* strong-typed — they're plain `std::string` debug/messaging labels with no uniqueness invariant. Don't wrap those.)
>
> - **Collection types: `Group<T>` and `Table<K, V>`.** Metal 2.0's API uses two utility types to disambiguate intent:
>   - **`Group<T>`** — an unordered collection of `T` ("a bunch of these," order not semantic). Currently a thin alias for `std::vector<T>`; you can construct with brace-init `{a, b, c}` exactly like a vector.
>   - **`Table<K, V>`** — a key-value map with the syntax of `std::unordered_map` and hash-friendly storage. Brace-init with `{{key, value}, {key, value}, ...}` works as expected.
>
>   These appear pervasively in the spec headers (`ProgramSpec::kernels` is `Group<KernelSpec>`, `ProgramRunArgs::tensor_args` is `Table<TensorParamName, TensorArgument>`, etc.). For initializer-list construction you rarely need to type them; for explicit-type sites (e.g., a conditional ternary that needs a common type), use the `Group<T>` / `Table<K, V>` name directly rather than `std::vector` / `std::unordered_map`.

### ProgramSpec

`ProgramSpec` is the top-level descriptor, replacing `ProgramDescriptor`. It describes the immutable properties of a Program, analogous to a function's signature and body.

A `Program` is built once from its `ProgramSpec`, then executed many times by supplying fresh `ProgramRunArgs` per execution.

**Legacy:**

```cpp
ProgramDescriptor desc = {
    .kernels = {kernel_desc_1, kernel_desc_2},
    .semaphores = {sem_desc},
    .cbs = {cb_desc_1, cb_desc_2},
};
Program program = Program(desc);
```

**Metal 2.0:**

```cpp
ProgramSpec spec{
    .name = "my_program",
    .kernels = {kernel_1, kernel_2},
    .dataflow_buffers = {dfb_1, dfb_2},
    .semaphores = {sem_1},
    .work_units = {main_work_unit},
};
Program program = MakeProgramFromSpec(*mesh_device, spec);  // temporary free function
//Program program = Program(spec);            // stable API form
```

Two structural additions vs. `ProgramDescriptor`:

- `name` — a human-readable label for debug and messaging (plain `std::string`; no uniqueness invariant).
- `work_units` — the new top-level concept that declares where kernels run. (See [WorkUnitSpec](#workunitspec).)

---

### KernelSpec

`KernelSpec` replaces `KernelDescriptor`. Notes:

1. **Placement.** The kernel's effective node set is derived from the `WorkUnitSpec`(s) that include it.
2. **Runtime arguments.** The `KernelSpec` declares a runtime arguments _schema_; runtime-arg _values_ are supplied per execution through `ProgramRunArgs` or `ProgramRunArgsView`.
3. **Resource bindings.** New syntax to bind DFB endpoints, semaphores, and tensors to the kernel, and retrieve them by name in device code. (See [TensorParameter](#tensorparameter) for the tensor case.)
4. **Multiple `KernelSpec`s per source.** A single kernel source file may be represented by multiple `KernelSpec`s if structural specialization is needed (different CTA bindings, different DFB or semaphore bindings, etc.). Each `KernelSpec` is compiled and placed independently.


**Legacy** (`KernelDescriptor`):

```cpp
KernelDescriptor reader = {
    .kernel_source = "kernels/reader.cpp",
    .core_ranges = CoreRangeSet{CoreRange{{0,0}, {0,0}}},
    .compile_time_args = {src_cb_idx, dst_cb_idx, page_size},
    .runtime_args = {{{0,0}, {start_page, num_pages}}},
    .config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
    },
};
```

**Metal 2.0** (`KernelSpec`):

```cpp
// ----- KernelSpec: kernel declaration -----
const KernelSpecName READER{"reader"};

KernelSpec reader{
    .unique_id = READER,
    .source = "kernels/reader.cpp",
    // (Placement is declared on WorkUnitSpec, below.)
    .compile_time_args = {
        {"src_cb_idx", src_cb_idx},
        {"dst_cb_idx", dst_cb_idx},
        {"page_size", page_size},
    },
    .runtime_arg_schema = {
        // Schema only — argument values are set per execution, on ProgramRunArgs.
        .runtime_arg_names = {"start_page", "num_pages"},
    },
    .hw_config = DataMovementHardwareConfig{
        // RoleHint is the primary intent knob — READER/WRITER auto-infers the
        // Gen1 processor/NOC pair (and the Gen2 default config). UNSPECIFIED
        // requires you to provide gen1_config explicitly (power-user override).
        .role = DataMovementRoleHint::READER,
    },
};

// ----- WorkUnitSpec: kernel placement and worker assignments -----
WorkUnitSpec main_work_unit{
    .name = "main",
    .kernels = {READER},
    .target_nodes = NodeCoord{0, 0},
};

// ----- ProgramSpec: assemble into the program description -----
ProgramSpec spec{
    .name = "my_program",
    .kernels = {reader},
    .work_units = {main_work_unit},
};
Program program = MakeProgramFromSpec(*mesh_device, spec);

// ----- ProgramRunArgs: argument values, set per execution -----
ProgramRunArgs params;
params.kernel_run_args = {{
    .kernel = READER,
    .runtime_arg_values = {{"start_page", {{NodeCoord{0, 0}, start_page}}},
                           {"num_pages",  {{NodeCoord{0, 0}, num_pages}}}},
}};
SetProgramRunArgs(program, params);
```


---

### DataflowBufferSpec

`DataflowBufferSpec` replaces `CBDescriptor`. Some host-side changes:

1. **Kernel bindings.** A DFB's producer and consumer kernels each declares a `DFBBinding` on its `KernelSpec`, naming the DFB endpoint and a local accessor name. The kernel code references the DFB through that local accessor name (e.g. `dfb::my_dfb`); the magic-number CB index is gone.
2. **Placement is derived.** A DFB lives wherever its bound producer / consumer kernels run; you do not pass `core_ranges`.
3. **Multi-threaded DFB access patterns.** These are pertinent to multi-threaded kernels on Quasar.

**Legacy** (`CBDescriptor`):

```cpp
constexpr uint32_t cb_idx = 0;

CBDescriptor cb_desc = {
    .total_size = num_pages * page_size,
    .core_ranges = CoreRangeSet{CoreRange{{0,0}, {0,0}}},
    .format_descriptors = {{
        .buffer_index = cb_idx,
        .data_format = tt::DataFormat::Float16_b,
        .page_size = page_size,
    }},
};
// Producer and consumer kernels reference the CB by `cb_idx` (= 0) in their CTAs.
```

**Metal 2.0** (`DataflowBufferSpec`):

```cpp
const DFBSpecName MY_DFB{"my_dfb"};

DataflowBufferSpec dfb{
    .unique_id = MY_DFB,
    .entry_size = page_size,
    .num_entries = num_pages,
    .data_format_metadata = tt::DataFormat::Float16_b,
    // No node_ranges — placement is derived from kernel bindings
};

// Bound on each kernel that uses the DFB:
KernelSpec producer{ /* ... */
    .dfb_bindings = {{
        .dfb_spec_name = MY_DFB,
        .accessor_name = "out_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER
    }},
};
KernelSpec consumer{ /* ... */
    .dfb_bindings = {{
        .dfb_spec_name = MY_DFB,
        .accessor_name = "in_dfb",  // independent of the producer's name
        .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER
    }},
};
```

**Spec-validator: every DFB needs ≥1 PRODUCER and ≥1 CONSUMER binding.** The host validator rejects programs where a DFB has only producers or only consumers across the kernels that bind it. When kernels have asymmetric conditional usage of the same DFB (e.g., a reader unconditionally produces a tile but a compute kernel only conditionally needs it), satisfy the constraint by declaring the conditional-side `DFBBinding` unconditionally on the host. Do not modify the kernel's `wait_front` / `pop_front` patterns to "balance" the topology: per-execution DFB state is reinitialized at each program enqueue, so a tile produced and never consumed is harmless.

**Spec-validator: `unpack_to_dest_mode` required for Float32 DFB consumers on `fp32_dest_acc_en` compute kernels.** When a compute `KernelSpec` sets `ComputeHardwareConfig::fp32_dest_acc_en = true`, every Float32-formatted DFB it consumes must appear in `ComputeHardwareConfig::unpack_to_dest_mode` (a `Table<DFBSpecName, UnpackToDestMode>`) with an explicit entry. Legacy `ComputeConfig` defaulted silently; Metal 2.0's validator requires the choice to be made explicit. Use `{DFB_UNIQUE_ID, UnpackToDestMode::Default}` to preserve legacy semantics; pick a non-default mode only when the kernel needs it. Symptom of forgetting: the validator complains at program-spec build time with a message pointing at the FP32 DFB.

**Borrowed-memory DFBs.** A DFB can be built on top of an existing `Buffer`'s memory rather than allocating its own L1 storage — the Metal 2.0 form of the legacy "dynamic circular buffer." Set `DataflowBufferSpec::borrowed_from` to the name of a `TensorParameter` whose buffer backs the DFB; the DFB's L1 address resolves at runtime from the corresponding `TensorArgument` in `ProgramRunArgs::tensor_args`.

**Aliased DFBs.** Two or more DFBs can share backing L1 memory via `DataflowBufferSpec::advanced_options.alias_with` (the field lives on `DFBAdvancedOptions`). The aliased DFBs are logically distinct (each has its own `unique_id` and bindings) but physically occupy the same L1 region. Useful when two same-shape DFBs are produced and consumed in non-overlapping phases of the kernel — the L1 footprint collapses. All aliased DFBs must have the same total size (`num_entries * entry_size`), must be bound to the same kernels, and must mutually declare each other in `alias_with`. Aliased DFBs offer no guarantee against data clobbering between the logical buffers; correctness is the kernel author's responsibility.

Remote DFBs spanning nodes are described in `dataflow_buffer_spec.hpp` but are not yet supported.

> **`tile_format_metadata` (the legacy `format_descriptors[i].tile` field).** `DataflowBufferSpec` carries a second optional metadata field, `std::optional<tt::tt_metal::Tile> tile_format_metadata`, that mirrors the legacy `CBFormatDescriptor::tile`. It is load-bearing for non-default tile geometry — tiny tiles, non-32x32, transposed-face, narrow-tile configurations (introduced for matmul tiny tiles in PR #12908). The value threads through the JIT path into per-CB `constexpr uint8_t unpack_tile_r_dim[]` / `unpack_tile_c_dim[]` / `unpack_num_faces[]` (and pack-side equivalents) inside the compute kernel's generated `chlkc_descriptors.h`, where LLK unpacker / packer init reads it.
>
> For DFBs holding **standard 32x32 tiles**, the JIT fallback when this field is `nullopt` yields the same generated arrays as setting `Tile()` — so leaving the field unset is observably identical to setting it. Most CBs in most ops are standard 32x32, which is why the field can look vestigial at a glance.
>
> **Operational rule for porters:** copy this field from the legacy CB's `format_descriptors[i].tile`. That preserves correctness for any non-default-tile use case and is harmless when the legacy field was `nullopt` or default.

---

### SemaphoreSpec

`SemaphoreSpec` replaces `SemaphoreDescriptor`. Some notes:

1. **Kernel resource binding**: Semaphores are bound by the `KernelSpec`. The kernel code accesses the semaphore by name through the binding's `accessor_name`.
2. **Initial value**: Semaphores are default-initialized to zero. Semaphores with non-zero initial values are not support in Quasar; they are temporarily available for WH/BH, but support will be deprecated once remote DFB support is available.

**Legacy** (`SemaphoreDescriptor`):

```cpp
SemaphoreDescriptor sem_desc{
    .id = 0,
    .core_type = CoreType::WORKER,
    .core_ranges = CoreRangeSet{CoreRange{{0,0}, {3,3}}},
    .initial_value = 0,
};
// Kernels access the semaphore via an ID, typically passed as a runtime argument.
```

**Metal 2.0** (`SemaphoreSpec`):

```cpp
const SemaphoreSpecName DONE{"done"};

SemaphoreSpec done_sem{
    .unique_id = DONE,
    .target_nodes = NodeRange{{0,0}, {3,3}}, // explicit placement
};

// Bound on each kernel that uses the semaphore:
KernelSpec writer{ /* ... */
    .semaphore_bindings = {{
        .semaphore_spec_name = DONE,
        .accessor_name = "kernel_done",  // kernel code accesses as `sem::kernel_done`
    }},
};
```

---

### TensorParameter

`TensorParameter` declares a tensor as a Program-scope resource. Kernels access it via `KernelSpec::TensorBinding`; the runtime `MeshTensor` is supplied per execution via `ProgramRunArgs::TensorArgument`. The kernel-author API collapses to a single line: `TensorAccessor(ta::name)`.

Three pieces, paralleling the DFB / Semaphore pattern with one deliberate asymmetry: tensors are *user-managed* resources (you own the lifetime), so the program-scope type is named `TensorParameter` (distinguished from the "Spec" pattern used elsewhere in the API) — echoing the "ProgramSpec is a function signature; ProgramRunArgs is the call args" framing.

One thing to be aware of: `TensorSpec` is a property of a `MeshTensor`. This pre-exists Metal 2.0; it is not part of the "Spec" object pattern in the rest of the Metal 2.0 APIs.

**Spec-validator: every TensorParameter needs ≥1 TensorBinding across the program's kernels.** Symmetric sibling of the DFB producer/consumer rule. A `TensorParameter` declared on `ProgramSpec` but never bound by any kernel is rejected by the validator. If you find yourself with an unbound `TensorParameter`, either drop the declaration (the tensor isn't actually needed by the program) or add the missing `TensorBinding` on the kernel that uses it.

> **⚠ Pre-migration check.** Before migrating an op, grep its kernel sources for `ArgConfig::Runtime`. If any kernel uses **`ArgConfig::RuntimeTensorShape`** (or the closely related `RuntimeShardShape` / `RuntimeBankCoords`), Metal 2.0 supports the capability via `TensorParameter::advanced_options.dynamic_tensor_shape = true` (full relaxation: `logical_shape` and `padded_shape` may both vary at enqueue) or `match_padded_shape_only = true` (weaker: `logical_shape` may vary, `padded_shape` must match). Both opt-ins are documented **UNSAFE** in the framework header — most kernels won't function correctly if the bound tensor's spec deviates from the declared spec — and adopting them has structural implications for the factory's interaction with the framework's per-dispatch caching path. Treat their use as a deliberate design choice, not a drop-in fix. The remaining flavors — `RuntimeRank`, `RuntimeNumBanks` — do not have a clean translation today; they have ~zero user sites outside tests, so this rarely matters in practice. **Family heads-up — `eltwise`:** ops in this family (binary / unary / ternary) very likely trip this check. Their dataflow kernels are written shape-agnostically (they iterate tile-by-tile and bake in no specific dimensions), and the legacy factories typically declare `RuntimeTensorShape` on the I/O tensors — so the faithful port very likely sets `dynamic_tensor_shape = true` on those `TensorParameter`s. This is *mirroring* a relaxation the legacy op already had, not speculatively adding a new one; still confirm with the grep above rather than assuming.

#### Legacy

```cpp
// host: append TensorAccessor args to the kernel's positional CTA list,
// pass the buffer base address as a runtime arg.
std::vector<uint32_t> reader_cta = {input_page_size, /* other CTAs */};
tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_cta);

auto reader_kid = CreateKernel(program, "reader.cpp", core,
    ReaderDataMovementConfig(reader_cta));
SetRuntimeArgs(program, reader_kid, core,
    {input.buffer()->address(), /* other RTAs */});
```

```cpp
// kernel: read the buffer address from the RTA list, manually thread
// CTA offsets to construct TensorAccessorArgs, then build the accessor.
constexpr uint32_t input_page_size = get_compile_time_arg_val(0);
// ...other compile-time args...
constexpr auto input_args = TensorAccessorArgs<N>();   // N = number of preceding CTAs
uint32_t input_addr = get_arg_val<uint32_t>(0);
auto input = TensorAccessor(input_args, input_addr);
```

#### Metal 2.0

```cpp
const TensorParamName INPUT{"input"};

// In ProgramSpec — declare the tensor as a Program-scope parameter.
// Use the tensor's own TensorSpec; the binding's spec must equal the runtime tensor's spec.
ProgramSpec spec;
spec.tensor_parameters = {
    {.unique_id = INPUT, .spec = input_tensor.tensor_spec()},
};

// In KernelSpec — bind the parameter, naming the kernel-side accessor.
KernelSpec reader{ /* ... */
    .tensor_bindings = {{
        .tensor_parameter_name = INPUT,
        .accessor_name = "input",   // kernel accesses as `ta::input`
    }},
};
spec.kernels = {reader};

Program program = MakeProgramFromSpec(*mesh_device, spec);

// In ProgramRunArgs — supply the actual MeshTensor per execution.
ProgramRunArgs params;
params.tensor_args = {
    {INPUT, TensorArgument{input_tensor}},
};
SetProgramRunArgs(program, params);
```

```cpp
// kernel: one line. No CTA-offset bookkeeping, no buffer-address RTA.
auto input = TensorAccessor(ta::input);
```

The buffer-address RTA is gone — the binding mechanism auto-injects the per-enqueue base address. The `TensorAccessorArgs<N>()` line is gone too — the layout metadata is packed by the host at program creation.

#### Migration recipe

For each op:

1. **Pre-flight.** Grep the op's kernel sources for `ArgConfig::Runtime`. If `RuntimeTensorShape` (or `RuntimeShardShape` / `RuntimeBankCoords`) appears anywhere, the migration is possible but the `TensorParameter`s for the affected tensors need `advanced_options.dynamic_tensor_shape = true` (or the weaker `match_padded_shape_only = true`) — see callout above for the safety and structural caveats before adopting either.
2. **Find each `TensorAccessor`.** In each kernel, locate every `TensorAccessor(args, addr)` construction. For each, trace `addr` back through the host code to the originating `Tensor`.
3. **Declare `TensorParameter`s.** Add one entry per tensor to `ProgramSpec::tensor_parameters`, using `tensor.tensor_spec()`. Pick a stable `unique_id`; declare it as a typed constant (`const TensorParamName INPUT{"input"};`) and reuse the constant at each use site.
4. **Add `TensorBinding`s.** On each `KernelSpec` whose kernel accesses the tensor, add an entry to `tensor_bindings` referencing the parameter by name. The `accessor_name` is the kernel-side identifier — it will appear as `ta::<accessor_name>`.
5. **Update kernel code.**
   - Replace `TensorAccessor(args, addr)` with `TensorAccessor(ta::<accessor_name>)`.
   - Drop the `TensorAccessorArgs<offset>()` line and any manual `next_compile_time_args_offset()` chaining.
   - Drop the `get_arg_val<uint32_t>(N)` line that retrieved the buffer address — those bytes are no longer in your RTA list, so re-index any RTAs that came after.
6. **Wire `tensor_args`.** Add one `TensorArgument` per `TensorParameter` to `ProgramRunArgs::tensor_args`, passing the actual `MeshTensor`.

#### Validation

At enqueue, the runtime `MeshTensor`'s `TensorSpec` must equal the binding's declared spec. Mismatches error loudly with a message naming the binding. The binding declaration is the single source of truth for layout — if the supplied tensor doesn't match, the bug is at the call site (typically: a layout transformation that should have been wired into the program but was applied externally to the tensor).

#### Multi-kernel-same-tensor

The common case (matmul reader + compute reading the same input; reshard reader/writer pipelines) is the cleanest fit for the binding model. Each kernel declares its own `TensorBinding` to the same `TensorParameter` with its own kernel-local `accessor_name`; the underlying tensor identity stays singular at the program level. This is structurally different from the legacy pattern, where the same tensor's address would be passed as a separate RTA on every kernel — and divergence between those was a real (silent) failure mode.

#### What's not covered yet

- **Manual `DistributionSpec` reuse** (the "advanced reuse-bank-coords-across-accessors" path from the [TensorAccessor guide](../../../../../../../tech_reports/tensor_accessor/tensor_accessor.md)). Not commonly used in production ops.
- **Tensor bindings on compute kernels** are out of scope. `TensorAccessor` was never supported for compute kernels (TRISC builds don't compile its NoC-using includes), so migration should not encounter any.

---

### WorkUnitSpec

`WorkUnitSpec` is a new top-level concept in Metal 2.0. It declares groups of kernels that operate together on a worker node, and on which nodes they run.

**Single work unit on one node:**

```cpp
WorkUnitSpec wu{
    .name = "main",
    .kernels = {READER, WRITER},
    .target_nodes = NodeCoord{0, 0},
};
```

**One work unit spanning a range of nodes:**

```cpp
WorkUnitSpec wu{
    .name = "main",
    .kernels = {READER, COMPUTE, WRITER},
    .target_nodes = NodeRange{{0, 0}, {3, 3}},  // 4×4 grid
};
```

**A kernel belonging to multiple work units** (e.g. for a kernel that runs on both an inner block and a halo region, with different DFB partners in each):

```cpp
WorkUnitSpec wu_inner{
    .name = "inner",
    .kernels = {COMPUTE, INNER_READER},
    .target_nodes = inner_node_range_set
};
WorkUnitSpec wu_halo{
    .name = "halo",
    .kernels = {COMPUTE, HALO_READER},
    .target_nodes = halo_node_range_set
};
// COMPUTE's effective node set is the union of inner + halo.
```

---

### ProgramRunArgs

`ProgramRunArgs` describes the mutable properties of the Program. These parameters are specified anew with each Program enqueue:
 - Kernel runtime arguments
 - Kernel common runtime arguments
 - Tensor arguments (`tensor_args`) — one `MeshTensor` per declared `TensorParameter`. See [TensorParameter](#tensorparameter). Borrowed-memory DFBs draw their backing L1 address from the corresponding `tensor_args` entry automatically; they don't require a separate `dfb_run_overrides` entry.

All kernel arguments are named arguments in Metal 2.0. For kernels that accept a variable number of arguments, the API additionally provides an  "varargs" mechanism.

**Legacy** (values on the descriptor):

```cpp
KernelDescriptor reader = {
    // ...
    .runtime_args = {{{0, 0}, {start_page, num_tiles}}},
    .common_runtime_args = {bank_id},
};
```

**Metal 2.0** (schema on the spec, values on the run params):

```cpp
// Schema declared on the kernel:
KernelSpec reader{
    .unique_id = READER,
    // ...
    .runtime_arg_schema = {
        .runtime_arg_names = {"start_page", "num_tiles"},
        .common_runtime_arg_names = {"bank_id"},
    },
};

// Values supplied per execution:
ProgramRunArgs params;
params.kernel_run_args = {{
    .kernel = READER,
    .runtime_arg_values = {{"start_page", {{NodeCoord{0, 0}, start_page}}},
                           {"num_tiles",  {{NodeCoord{0, 0}, num_tiles}}}},
    .common_runtime_arg_values = {{"bank_id", bank_id}},
}};
SetProgramRunArgs(program, params); // temporary free function
```

Vararg form — for kernels with a genuinely dynamic argument tail (loop-retrieved on the device side). The canonical fit is a CTA-bound count plus a per-element payload, e.g. an N-dimensional shape. Vararg counts live on `KernelAdvancedOptions` (schema side); vararg values live on `AdvancedKernelRunArgs` (per-execution side, nested under each `KernelRunArgs` entry's `.advanced_options`):

```cpp
// Schema (on KernelSpec::advanced_options, which is KernelAdvancedOptions):
.compile_time_args = {{"rank", rank}},
.advanced_options = {
    .num_runtime_varargs = rank,  // shape: one entry per dimension
},

// Values (on the matching KernelRunArgs entry's .advanced_options, which is AdvancedKernelRunArgs):
//   .runtime_varargs is Table<NodeCoord, std::vector<uint32_t>>
params.kernel_run_args = {{
    .kernel = MY_KERNEL,
    .advanced_options = {
        .runtime_varargs = {{NodeCoord{0, 0}, shape_dims}},  // shape_dims.size() == rank
    },
}};
```

Named and vararg forms can coexist on the same kernel. Vararg indices are stable across schema changes — promoting a named RTA to a CRTA, or adding/removing named args, does not shift vararg indices. Common runtime varargs (`num_common_runtime_varargs`, retrieved on the device via `get_common_vararg(i)`) work analogously, broadcast across all nodes.

> The vararg form is intended for kernels whose device-side code retrieves arguments in a loop — i.e., `get_vararg(i)` with `i` a runtime variable. When the kernel reads each argument by a constant index (`get_vararg(0)`, `get_vararg(1)`, …), the named form reads more clearly on both sides.

---

## Device-Side Migration

### Circular Buffers → Dataflow Buffers

Metal 2.0 replaces Circular Buffers (CBs) with Dataflow Buffers (DFBs). Notes:

1. **Construction from local accessor name**. You construct your DFB from its local accessor, not a magic-number `cb_id`. The accessor is auto-generated from the host-side DFB binding and lives in the `dfb::` namespace inside `kernel_bindings_generated.h`.
2. **`experimental::CircularBuffer` compatible APIs**. The kernel-side DFB APIs are drop-in-compatible with the Device 2.0 `experimental::CircularBuffer` wrapper methods and semantics on Gen1. Gen2 permits more advanced capabilities.
3. **Direct use in LLK compute APIs (WH/BH)**. The `dfb::my_dfb` accessor constants implicitly convert to `uint32_t`, so on WH/BH you can pass them directly to LLK compute APIs (`reduce_init`, `pack_tile`, `cb_wait_front`, etc.) that take a raw CB id — no `.id` extraction or temporary `DataflowBuffer` needed. The Quasar LLK signatures are intended to take DFB handles natively; that refresh is in progress.

**Legacy (Device 2.0 `experimental::CircularBuffer`):**

```cpp
constexpr uint32_t cb_id = 0;
experimental::CircularBuffer cb(cb_id);

cb.reserve_back(num_pages);
uint32_t write_ptr = cb.get_write_ptr();
// ... write data ...
cb.push_back(num_pages);
```

**Metal 2.0 (`experimental::DataflowBuffer`):**

```cpp
// Host-side DFB binding declared accessor_name = "my_dfb".
// Auto-generated from that: constexpr DFBAccessor dfb::my_dfb;
experimental::DataflowBuffer dfb(dfb::my_dfb);

dfb.reserve_back(num_entries);
uint32_t write_ptr = dfb.get_write_ptr();
// ... write data ...
dfb.push_back(num_entries);
```


> **Implicit sync (Quasar)**: On Gen2, you can elide the explicit `reserve_back` / `push_back` (or `wait_front` / `pop_front`) pattern entirely. Pass the DFB directly to `experimental::Noc::async_read` or `async_write`, and the runtime hardware handles the FIFO sync via ISR. This is the default behavior; you can disable it per-DFB endpoint by listing the DFB's `unique_id` in `Gen2Config::disable_implicit_sync_for` (on the kernel's `DataMovementHardwareConfig::gen2_config`): e.g. `gen2_config = Gen2Config{.disable_implicit_sync_for = {"my_dfb"}}` (rarely needed). On Gen1, the option is a no-op — the explicit FIFO pattern shown above remains the way.

---

### TensorAccessor

Construction collapses to one line. `TensorAccessor` takes the codegen-emitted token directly; everything else is unchanged from today (`get_noc_addr`, `get_bank_and_offset`, `dspec()`, `noc_async_read_page`, etc.).

The token lives in the `ta::` namespace inside `kernel_bindings_generated.h` (auto-generated from the host-side `TensorBinding`), parallel to the `dfb::` and `sem::` namespaces.

**Legacy:**

```cpp
constexpr uint32_t input_page_size = get_compile_time_arg_val(0);
// ...other compile-time args...
constexpr auto input_args = TensorAccessorArgs<N>();    // N = preceding CTA count
constexpr auto index_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
uint32_t input_addr = get_arg_val<uint32_t>(0);
uint32_t index_addr = get_arg_val<uint32_t>(1);

auto input = TensorAccessor(input_args, input_addr);
auto index = TensorAccessor(index_args, index_addr);
```

**Metal 2.0:**

```cpp
auto input = TensorAccessor(ta::input);
auto index = TensorAccessor(ta::index);
```

The `TensorAccessorArgs<N>()` lines, the manual `next_compile_time_args_offset()` chaining for multi-tensor stacking, and the buffer-address RTAs are all gone. Layout metadata is packed by the host into the kernel's compile-time args at program creation, not retrieved by the kernel; the per-enqueue base address rides on a host-managed slot in the kernel's CRTA buffer that the kernel never sees directly.

See [TensorParameter](#tensorparameter) for the host-side declaration that produces the `ta::` namespace.

---

### Kernel Argument Retrieval Syntax

The Metal 2.0 host API declares kernel arguments by name; the kernel-side API retrieves them by name. This replaces the legacy positional `get_arg_val<uint32_t>(N)` style.

**The only `#include` a porter adds to a kernel is `experimental/kernel_args.h`** — that pulls in the accessor templates (`get_arg`, `args::`, `dfb::`, `sem::`, `ta::`). The generated headers `kernel_bindings_generated.h` (which carries `dfb::` / `sem::` / `ta::` declarations from the host bindings) and `kernel_args_generated.h` (which carries `args::` declarations from `compile_time_args` + `runtime_arg_schema`) are auto-included by the build system via `<kernel_includes.hpp>` before the kernel source. **Do not** `#include` either generated header from your kernel.

**Example 1 — Named arguments only.**

Suppose the host declared the following on `KernelSpec`:

```cpp
.compile_time_args = {{"bank_id", 0}, {"entry_size", 1024}},
.runtime_arg_schema = {
    .runtime_arg_names = {"start_page"},
    .common_runtime_arg_names = {"num_entries"},
},
```

**Legacy (positional):**

```cpp
void kernel_main() {
    constexpr uint32_t bank_id    = get_compile_time_arg_val(0);
    constexpr uint32_t entry_size = get_compile_time_arg_val(1);
    uint32_t start_page    = get_arg_val<uint32_t>(0);    // RTA index 0
    uint32_t num_entries = get_common_arg_val<uint32_t>(0);  // CRTA index 0
    // ...
}
```

**Metal 2.0 (named):**

```cpp
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto bank_id    = get_arg(args::bank_id);     // CTA — compile-time constant
    constexpr auto entry_size = get_arg(args::entry_size);  // CTA
    auto start_page             = get_arg(args::start_page);    // RTA
    const auto num_entries    = get_arg(args::num_entries); // CRTA
    // ...
}
```

CTAs, RTAs, and CRTAs are accessed through the same `get_arg(args::name)` mechanism in device code. The dispatch type (compile-time, runtime, common runtime) is a host-only concept — the kernel author no longer needs to know or care which kind of argument they're reading. The C++ type-deduction context (`constexpr auto`, `auto`, `const auto`) is the only place the distinction surfaces.

**Example 2 — Named arguments plus varargs (the niche case).**

Some kernels read a dynamic-count argument tail in a loop — e.g. an N-dimensional tensor's shape, where N is itself a CTA. That's the case varargs are designed for:

```cpp
.compile_time_args = {{"rank", rank}},
.runtime_arg_schema = {
    .runtime_arg_names = {"start_page"},
    .common_runtime_arg_names = {"num_entries"},
    .num_runtime_varargs = rank,  // shape: one entry per dimension
},
```

```cpp
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto rank      = get_arg(args::rank);          // CTA
    auto start_page          = get_arg(args::start_page);    // RTA
    const auto num_entries   = get_arg(args::num_entries);   // CRTA

    // Vararg RTAs read in a loop — count is bound to `rank`:
    uint32_t shape[rank];
    for (uint32_t i = 0; i < rank; ++i) {
        shape[i] = get_vararg(i);
    }
    // ...
}
```

Common runtime varargs are analogous: declare with `num_common_runtime_varargs`, retrieve on the device with `get_common_vararg(i)`. Same loop-retrieval criterion applies.

> When porting a legacy kernel with positional RTAs, the natural Metal 2.0 form is named RTAs — one per legacy positional argument. A translation into vararg slots compiles and runs (and may be useful as an interim step on a large kernel), but the named form is the recommended endpoint for new code. The distinguishing criterion is whether the kernel's device-side `get_vararg(i)` calls use `i` as a runtime variable (varargs are appropriate) or constants (named RTAs are clearer).

Vararg indices are stable across schema changes: promoting a named RTA to a CRTA, or adding or removing named arguments, does not shift vararg indices. (This stability lets you migrate incrementally — rename arguments to named form one at a time without disturbing the remaining varargs.)

> **Note**: This argument-retrieval syntax will evolve again to support custom argument types beyond `uint32_t` (including std::array and user-defined POD types). The named-accessor mechanism (`get_arg(args::name)`) will be preserved; the underlying types will become user-extensible. Migrators should plan to revisit kernel arguments after these changes.

---

## Design Principles

The sections above describe the Metal 2.0 API surface piece by piece. This section steps back and presents the design intent that ties those pieces together. Reading these principles helps you recognize when a port is *faithful* (preserves intent and reaps the ergonomic improvement Metal 2.0 promises) versus when it merely *translates syntax* (carrying forward workarounds that should evaporate).

### Principle 1: Immutable spec, mutable run-args

Metal 2.0 splits a Program's description into two parts:

- **`ProgramSpec`** — the immutable description (the "function signature and body"). Captures kernel sources, resource declarations, work-unit assignments, and argument *schemas*.
- **`ProgramRunArgs`** — the per-execution mutable values (the "call arguments"). Carries runtime argument values, tensor identities, and any other field that may legitimately differ between executions.

This split is the **only** structural divergence from legacy `ProgramDescriptor`. Every other Metal 2.0 type maps roughly 1:1 with its `ProgramDescriptor` counterpart — `KernelSpec` ↔ `KernelDescriptor`, `DataflowBufferSpec` ↔ `CBDescriptor`, `SemaphoreSpec` ↔ `SemaphoreDescriptor`, and so on.

The separation enables two performance optimizations the legacy API couldn't express:

1. **Caching**: a `Program` built once from a `ProgramSpec` can be executed many times with fresh `ProgramRunArgs` — the compile/build cost is paid once.
2. **Partial updates**: for ops whose mutable surface is small (typically just the tensor identities), the framework can apply a partial update on cache hit instead of re-issuing the full `ProgramRunArgs`.

For the op author, the practical implication is that **the schema and the values are conceptually paired** even though they live in different structures. When you add a named RTA to a `KernelSpec`'s schema, you simultaneously add its per-node value to `ProgramRunArgs::kernel_run_args`. When you declare a `TensorParameter`, you simultaneously add its `TensorArgument`. The spec/run-args separation is most visible to the framework and tooling; during construction, the two are built together.

### Principle 2: First-class named resource bindings

This is the largest design shift in Metal 2.0 — and the principle responsible for the bulk of the stylistic improvement Metal 2.0 offers.

**Resources (tensors, DFBs, semaphores) bind by name.** Each resource is declared once on the `ProgramSpec` (as a `TensorParameter`, `DataflowBufferSpec`, or `SemaphoreSpec`), then bound to each kernel that uses it via a `TensorBinding` / `DFBBinding` / `SemaphoreBinding` on the `KernelSpec`. The kernel accesses the resource through an auto-generated handle (`ta::name`, `dfb::name`, `sem::name`) without needing to know its underlying ID, address, or layout.

Legacy `ProgramDescriptor` had no equivalent abstraction. Resources were referenced indirectly:

- **Tensors** via their buffer's base address, packed into a runtime argument; layout metadata via `TensorAccessorArgs` plumbing appended to the kernel's compile-time arg list.
- **Circular buffers** via a magic-number CB index assigned to `CBDescriptor::format_descriptors[].buffer_index`; producer and consumer kernels referenced that index via a positional or named CTA.
- **Semaphores** via their integer ID, typically passed as a runtime argument.

The named-binding design produces a series of consequent changes — patterns that are *non-translations* during port. They appear in legacy code; they should **not** appear in a faithful Metal 2.0 port.

**Tensor accessor.**

```cpp
// Legacy: address as RTA, layout metadata via TensorAccessorArgs<N>() chains.
constexpr auto args = TensorAccessorArgs<N>();
uint32_t addr = get_arg_val<uint32_t>(0);
auto a = TensorAccessor(args, addr);

// Metal 2.0: one line.
auto a = TensorAccessor(ta::input);
```

Neither `tensor.buffer()->address()` (host side) nor `TensorAccessorArgs<N>()` (device side) survives a faithful port.

**Dataflow buffer.**

```cpp
// Legacy: magic-number CB index passed via CTA, used to construct CB wrapper.
constexpr uint32_t cb_id = 0;
experimental::CircularBuffer cb(cb_id);

// Metal 2.0: named handle from the binding.
experimental::DataflowBuffer cb(dfb::my_dfb);
```

The magic-number CB index doesn't appear on the host or in the kernel.

**Semaphore.**

```cpp
// Legacy: ID forwarded as RTA, kernel constructs the access from the ID.
uint32_t sem_id = get_arg_val<uint32_t>(2);

// Metal 2.0: named handle from the binding; access via sem::done.
```

The semaphore-ID RTA is gone.

**Multi-kernel access to the same resource.** The named-binding design makes resource identity singular at the program level. Legacy: each kernel reads the same tensor's address as an independent RTA; divergence between those RTAs was a silent failure mode. Metal 2.0: one `TensorParameter`, multiple `TensorBinding`s — each with its own per-kernel `accessor_name`. The accessor names are local aliases; the binding declarations are the single source of truth.

**Optional resources.** A binding may be declared conditionally on the host (omitted from `KernelSpec::dfb_bindings` / `tensor_bindings` when the path isn't taken). On the kernel side, the gate has to happen at the **preprocessor** level: Metal 2.0's `dfb::<name>` namespace is generated from the actual host bindings — `dfb::cb_scaled` exists only when the host actually binds it — and `if constexpr` in non-template `kernel_main` still performs name lookup on the discarded branch, so `if constexpr (false) { … dfb::cb_scaled … }` fails to compile at parse time. The host emits a matching `#define` via `KernelSpec::compiler_options.defines`; the kernel `#ifdef`-gates both the binding's `constexpr` alias and every expression referencing it.

```cpp
// Host side:
KernelSpec compute{
    .compile_time_args = { /* ... */ },
    .compiler_options = {.defines = do_scale
        ? Table<std::string, std::string>{{"DO_SCALE", "1"}}
        : Table<std::string, std::string>{}},
    .dfb_bindings = do_scale
        ? Group<DFBBinding>{INPUT, OUTPUT, SCALED}
        : Group<DFBBinding>{INPUT, OUTPUT},
};

// Kernel side:
#ifdef DO_SCALE
constexpr uint32_t cb_scaled_id = dfb::cb_scaled;
#endif

#ifdef DO_SCALE
experimental::DataflowBuffer cb_scaled(dfb::cb_scaled);
cb_scaled.wait_front(...);
// ... all uses of cb_scaled
#endif
```

The full discussion (file-scope ternaries, preprocessor-stage parsing) is in [Pattern: Conditional / optional DFB bindings](ai/metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings). The "always bind and gate only the uses" alternative — wrapper declared unconditionally — is wrong on two counts: it pays L1 unnecessarily for an unused buffer, *and* `if constexpr` doesn't gate name lookup in a non-template kernel even if you reached for it.

---

When a port carries any of these patterns forward verbatim — buffer-address RTAs, magic CB indices, semaphore-ID RTAs, always-bound optional DFBs — it has translated syntax without internalizing the design shift. The code compiles, may even run correctly, but it defeats the ergonomics improvement Metal 2.0 is built to deliver. The [Concept Map](#concept-map) early in this guide pairs the syntax-level translations; the named-binding principle is what explains *why* certain legacy idioms shouldn't reappear at all.

### Principle 3: Named arguments throughout

Argument naming extends the named-binding ethos to scalar values:

- **Compile-time arguments are named only.** Positional CTAs are gone from the Metal 2.0 API.
- **Runtime arguments are typically named.** A `varargs` mechanism exists as a niche escape hatch.
- **Common runtime arguments are typically named.** Same `varargs` mechanism applies.

The `args::` namespace generated from a `KernelSpec`'s schema gives the kernel one uniform retrieval API: `get_arg(args::name)`. The CTA/RTA/CRTA distinction is a host-only concern; the kernel author doesn't need to know which dispatch slot a given argument lives in.

Legacy `ProgramDescriptor` offered a hybrid model: `KernelDescriptor::compile_time_args` (positional, vector of `uint32_t`) and `KernelDescriptor::named_compile_time_args` (named, vector of `{name, value}` pairs). Many recent ops — matmul among them — use the named half. For ports of those ops:

- Named CTAs in legacy → named CTAs in Metal 2.0. 1:1 mechanical.
- Positional CTAs in legacy → named CTAs in Metal 2.0. Explicit naming required during port; pick names that reflect what the kernel actually does with the value.

> **Use varargs only when the kernel reads its arguments in a loop.** Varargs are designed for kernels whose device-side code retrieves arguments via `get_vararg(i)` where `i` is a runtime variable — the canonical case is an N-dimensional shape gated on a CTA-known `rank`. When each argument is referenced by a constant index (`get_vararg(0)`, `get_vararg(1)`, ...), the named form is clearer on both sides. A port from positional RTAs to varargs may compile and run, but it preserves the legacy positional vocabulary instead of upgrading to Metal 2.0's named one. See the [patterns catalog](ai/metal2_port_patterns.md) caution on varargs.

---

## TTNN Framework Integration

TTNN ops integrate with the Metal 2.0 host API through the `ttnn::device_operation` framework. The framework provides three factory *concepts*; an op's program factory satisfies whichever concept matches its construction model:

- **`ProgramFactoryConcept`** — the oldest concept; legacy `host_api.hpp` builder-style factories. Returns a `CachedProgram<shared_variables_t>` from `create()`; updates per-execution state via `override_runtime_arguments()`.
- **`ProgramDescriptorFactoryConcept`** — the intermediate concept; legacy `ProgramDescriptor`-based factories. Returns a `tt::tt_metal::ProgramDescriptor` from `create_descriptor()`.
- **`MetalV2FactoryConcept`** — the Metal 2.0 concept. Returns a `ttnn::device_operation::ProgramArtifacts` (the `ProgramSpec`, its `ProgramRunArgs`, and any op-owned tensors) from `create_program_artifacts()`. **This is the concept ops port to.**

> The porter-facing detail for this concept — the feasibility gate, the device-op-class edits a port forces, and the cache lifecycle in operational terms — lives in [`port_op_to_metal2_ttnn_factory.md`](ai/port_op_to_metal2_ttnn_factory.md). This section is the conceptual overview.

### Factory skeleton

A Metal 2.0 program factory has the following shape:

```cpp
// In your factory header:
struct MyProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const operation_attributes_t& attributes,
        const tensor_args_t&           tensor_args,
        tensor_return_value_t&         tensor_return_value);
};

// In your factory implementation:
ttnn::device_operation::ProgramArtifacts MyProgramFactory::create_program_artifacts(
    const operation_attributes_t& attributes,
    const tensor_args_t&           tensor_args,
    tensor_return_value_t&         tensor_return_value) {

    // ... build the spec and run args ...

    tt::tt_metal::experimental::ProgramSpec spec{
        .name = "my_program",
        // ...
    };
    tt::tt_metal::experimental::ProgramRunArgs run_params;
    // ... populate run_params ...

    return ttnn::device_operation::ProgramArtifacts{
        .spec       = std::move(spec),
        .run_params = std::move(run_params),
        // .op_owned_tensors = {},  // default-empty; populate only when the factory
        //                          // carries op-owned (config / index-table / workspace) tensors,
        //                          // whose lifetime then tracks the cached entry.
    };
}
```

`ProgramArtifacts` has three fields — `spec`, `run_params`, and `op_owned_tensors` (default-empty). Most ports leave `op_owned_tensors` defaulted; a factory that carries op-owned `MeshTensor`s (config / index-table / workspace tensors it builds beyond the op's io) returns them there, and the framework keeps them alive at a stable address for the cached `Program`'s lifetime. (Op-owned `GlobalSemaphore`s are not supported — the artifact carries only `MeshTensor`s.)

> **Tensor types in a ProgramFactory.** A factory sits between two tensor-type universes; you'll see three names but only two real types:
> - **`ttnn::Tensor`** — TTNN's PyTorch-like tensor wrapper. Can in general represent either a host-resident or device-resident tensor, but **is always device-resident in a ProgramFactory.** This is what `tensor_args` and `tensor_return_value` carry into `create_program_artifacts`.
> - **`tt::tt_metal::MeshTensor`** — Metalium's native device-tensor type. RAII handle with unique ownership over a (mesh) device allocation. Metal 2.0's `TensorArgument` (the entries you populate in `ProgramRunArgs::tensor_args`) takes a `const MeshTensor&` — this is the type the rest of the Metal 2.0 API speaks.
> - **`tt::tt_metal::Tensor`** — the same type as `ttnn::Tensor`, not a third type. Today the class is defined in the `tt::tt_metal` namespace (with its header living under `ttnn/`) and `ttnn::Tensor` is an alias to it — a refactoring artifact that an in-flight migration is unwinding, consolidating the type under the `ttnn::Tensor` name and retiring `tt_metal::Tensor`. You don't need to track which is currently the class and which the alias: **write `ttnn::Tensor`** in the code you author, and read any `tt_metal::Tensor` you encounter in legacy code as the same type.
>
> **Extract `MeshTensor` from each input at the top of the factory.** `ttnn::Tensor::mesh_tensor()` returns `const MeshTensor&` (the rvalue overload is `= delete`'d to prevent dangling references on temporaries). Recommended style — extract once at factory entry, work with `MeshTensor` references throughout:
>
> ```cpp
> ttnn::device_operation::ProgramArtifacts MyProgramFactory::create_program_artifacts(
>     const operation_attributes_t& attributes,
>     const tensor_args_t&           tensor_args,
>     tensor_return_value_t&         tensor_return_value) {
>
>     const auto& input  = tensor_args.input.mesh_tensor();
>     const auto& output = tensor_return_value.mesh_tensor();
>     // Use `input` / `output` (MeshTensor&) for the rest of the factory body —
>     // pass to helpers, build TensorParameter specs, populate TensorArguments.
> }
> ```
>
> Pass `MeshTensor` directly to helper functions as `const MeshTensor&`; do **not** wrap in `std::cref` / `std::reference_wrapper<const MeshTensor>`. Reaching back to `.mesh_tensor()` at each call site compiles and runs correctly but is not the recommended style — extract once and pass the reference.

A Metal 2.0 factory **does not** construct the `Program` itself, nor does it call `SetProgramRunArgs` directly. Those are the framework adapter's responsibilities.

### Cache-miss vs cache-hit lifecycle

The framework manages two distinct execution paths:

- **Cache miss** (first execution, or a new op-args key): the framework calls `create_program_artifacts`, then internally calls `MakeProgramFromSpec(*device, spec)` to build the `Program`, then `SetProgramRunArgs(program, run_params)` to apply the per-execution values, resolving each `TensorArgument` against the op's io tensors and any `op_owned_tensors`. The full spec is realized once; op-owned tensors are parked in the cache entry at a stable address.
- **Cache hit** (subsequent executions with a still-valid cached `Program`): the framework **does not re-run the factory**. It refreshes the tensor bindings — the io tensors plus the parked op-owned ones — via `UpdateTensorArgs`, and skips the build. Only the tensor bindings are refreshed; other run-arg values keep the values they were given at cache-miss.

The cache key is the op itself — its type, attributes, and tensor args (the framework's automatic hash), combined with the target mesh coordinates; a custom `compute_program_hash`, if present, replaces that default. A factory satisfying `MetalV2FactoryConcept` is routed through `MetalV2MeshWorkloadFactoryAdapter`, which handles both paths.

### TTNN's immutability convention

A practical consequence of TTNN's caching: **most parameters that *could* vary between calls are treated as if they were immutable, and the cache is invalidated when they change**. The actually-mutable surface across calls is small — typically just the tensor identities (`tensor_args` in `ProgramRunArgs`).

An op author specifying a Metal 2.0 factory does not directly experience the spec/run-args separation as a per-call distinction. The two are constructed together in `create_program_artifacts`; the framework's cache machinery converts that single construction call into either a full realization (cache miss) or a tensor-binding refresh (cache hit). Designing your factory to construct paired resources and values together — `TensorParameter` and its `TensorArgument`, RTA schema and its values — reflects the actual authoring flow.

---

## Complete Migration Examples

### Example 1: Single-Core Reader / Writer with One DFB

Reader kernel reads pages from an input tensor into a DFB; writer kernel pulls from the DFB and writes pages to an output tensor.

**Legacy (`ProgramDescriptor`):**

```cpp
constexpr uint32_t cb_idx     = 0;
constexpr uint32_t page_size  = 1024;
constexpr uint32_t num_pages  = 8;
const CoreCoord core{0, 0};

CBDescriptor cb_desc = {
    .total_size = num_pages * page_size,
    .core_ranges = CoreRangeSet{CoreRange{core}},
    .format_descriptors = {{
        .buffer_index = cb_idx,
        .data_format = tt::DataFormat::Float16_b,
        .page_size = page_size,
    }},
};

// Reader: TensorAccessor args appended to the positional CTA list; buffer
// address passed as an RTA.
std::vector<uint32_t> reader_cta = {cb_idx, page_size};
tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_cta);

KernelDescriptor reader = {
    .kernel_source = "kernels/reader.cpp",
    .core_ranges = CoreRangeSet{CoreRange{core}},
    .compile_time_args = reader_cta,
    .runtime_args = {{core, {input_tensor.buffer()->address(), num_pages}}},
    .config = ReaderConfigDescriptor{},
};

// Writer: same shape on the output side.
std::vector<uint32_t> writer_cta = {cb_idx, page_size};
tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_cta);

KernelDescriptor writer = {
    .kernel_source = "kernels/writer.cpp",
    .core_ranges = CoreRangeSet{CoreRange{core}},
    .compile_time_args = writer_cta,
    .runtime_args = {{core, {output_tensor.buffer()->address(), num_pages}}},
    .config = WriterConfigDescriptor{},
};

ProgramDescriptor program_desc = {
    .kernels = {reader, writer},
    .cbs = {cb_desc},
};
```

**Metal 2.0:**

```cpp
const KernelSpecName  READER{"reader"};
const KernelSpecName  WRITER{"writer"};
const DFBSpecName     DFB{"loopback_dfb"};
const TensorParamName INPUT{"input"};
const TensorParamName OUTPUT{"output"};

constexpr uint32_t page_size = 1024;
constexpr uint32_t num_pages = 8;
const NodeCoord node{0, 0};

KernelSpec reader{
    .unique_id = READER,
    .source = "kernels/reader.cpp",
    .compile_time_args = {{"page_size", page_size}},
    .runtime_arg_schema = {.runtime_arg_names = {"num_pages"}},
    .dfb_bindings = {{
        .dfb_spec_name = DFB,
        .accessor_name = "out_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER
    }},
    .tensor_bindings = {{
        .tensor_parameter_name = INPUT,
        .accessor_name = "input",   // kernel accesses as `ta::input`
    }},
    .hw_config = DataMovementHardwareConfig{
        .role = DataMovementRoleHint::READER,
    },
};

KernelSpec writer{
    .unique_id = WRITER,
    .source = "kernels/writer.cpp",
    .compile_time_args = {{"page_size", page_size}},
    .runtime_arg_schema = {.runtime_arg_names = {"num_pages"}},
    .dfb_bindings = {{
        .dfb_spec_name = DFB,
        .accessor_name = "in_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER
    }},
    .tensor_bindings = {{
        .tensor_parameter_name = OUTPUT,
        .accessor_name = "output",  // kernel accesses as `ta::output`
    }},
    .hw_config = DataMovementHardwareConfig{
        .role = DataMovementRoleHint::WRITER,
    },
};

DataflowBufferSpec dfb{
    .unique_id = DFB,
    .entry_size = page_size,
    .num_entries = num_pages,
    .data_format_metadata = tt::DataFormat::Float16_b,
};

ProgramSpec spec{
    .name = "loopback",
    .kernels = {reader, writer},
    .dataflow_buffers = {dfb},
    .tensor_parameters = {
        {.unique_id = INPUT,  .spec = input_tensor.tensor_spec()},
        {.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
    },
    .work_units = {{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = node,
    }},
};
Program program = MakeProgramFromSpec(*mesh_device, spec);

ProgramRunArgs params;
params.kernel_run_args = {
    {.kernel = READER,
     .runtime_arg_values = {{"num_pages", {{node, num_pages}}}}},
    {.kernel = WRITER,
     .runtime_arg_values = {{"num_pages", {{node, num_pages}}}}},
};
params.tensor_args = {
    {INPUT,  TensorArgument{input_tensor}},
    {OUTPUT, TensorArgument{output_tensor}},
};
SetProgramRunArgs(program, params);
```

`ProgramDescriptor` collapses all of placement, schema, and values into one nested struct, while Metal 2.0 keeps each concern visible as its own field. This example demonstrated named CTAs, DFB binding-by-name, derived DFB placement, TensorAccessor binding (replacing the manual `TensorAccessorArgs` plumbing chain on the host and the `TensorAccessorArgs<N>()` offset bookkeeping in the kernel), and mutable / immutable separation.

### Example 2: Multi-Core with Cross-Core Synchronization

Same reader/writer kernels as Example 1, but now running on a 2×2 grid with a semaphore-based handshake. Each node reads (and writes) its own slice of the tensor. Only the deltas are shown — fields that are unchanged from Example 1 are elided with `// (unchanged)`.

**Legacy (`ProgramDescriptor`):**

```cpp
const CoreRangeSet cores = CoreRangeSet{CoreRange{{0, 0}, {1, 1}}};
const uint32_t pages_per_core = num_pages / 4;  // 8 / 4 = 2 pages per core

SemaphoreDescriptor done_sem{
    .id = 0,
    .core_type = CoreType::WORKER,
    .core_ranges = cores,
    .initial_value = 0,
};

KernelDescriptor reader = {
    // (unchanged — same TensorAccessor CTAs, kernel source, etc.)
    .core_ranges = cores,
    // Per-core runtime args. Same buffer base address on every core; each
    // core reads a different slice via start_page. Semaphore ID rides as an RTA.
    .runtime_args = {
        {{0,0}, {input_tensor.buffer()->address(), 0u,                  pages_per_core, /*sem_id=*/0}},
        {{1,0}, {input_tensor.buffer()->address(), 1u*pages_per_core,   pages_per_core, 0}},
        {{0,1}, {input_tensor.buffer()->address(), 2u*pages_per_core,   pages_per_core, 0}},
        {{1,1}, {input_tensor.buffer()->address(), 3u*pages_per_core,   pages_per_core, 0}},
    },
    // ...
};

ProgramDescriptor program_desc = {
    .kernels = {reader, writer},
    .semaphores = {done_sem},
    .cbs = {cb_desc},
};
```

**Metal 2.0:**

```cpp
const SemaphoreSpecName DONE{"done"};
const NodeRange cores{{0, 0}, {1, 1}};
const uint32_t pages_per_node = num_pages / 4;  // 8 / 4 = 2 pages per node

SemaphoreSpec done_sem{
    .unique_id = DONE,
    .target_nodes = cores,
};

KernelSpec reader{
    // (unchanged — same TensorBinding to INPUT, DFB binding, source, etc.)
    // No core_ranges — placement comes from WorkUnitSpec, below.
    // No semaphore in the args — bind it instead:
    .semaphore_bindings = {{
        .semaphore_spec_name = DONE,
        .accessor_name = "done",  // kernel accesses as `sem::done`
    }},
    // RTA schema gains start_page so each node knows which slice it owns.
    // The buffer address is gone — the TensorBinding auto-injects it.
    .runtime_arg_schema = {.runtime_arg_names = {"start_page", "num_pages"}},
    // ...
};

ProgramSpec spec{
    .name = "loopback_2x2",
    .kernels = {reader, writer},
    .dataflow_buffers = {dfb},
    .semaphores = {done_sem},
    .tensor_parameters = {
        {.unique_id = INPUT,  .spec = input_tensor.tensor_spec()},
        {.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
    },
    .work_units = {{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = cores,  // 2×2 grid
    }},
};
Program program = MakeProgramFromSpec(*mesh_device, spec);

ProgramRunArgs params;
// runtime_arg_values is keyed by argument name, then by node. (Legacy factories
// usually build these node-first; the AddRuntimeArgsForNode helper in
// program_run_args.hpp produces this name-first table from a node-first loop.)
// Tensor identity is singular: one TensorParameter for the input tensor,
// regardless of how many nodes access it. Per-node access varies via slice
// indices, not addresses.
params.kernel_run_args = {{
    .kernel = READER,
    .runtime_arg_values = {
        {"start_page", {{{0,0}, 0u}, {{1,0}, 1u*pages_per_node}, {{0,1}, 2u*pages_per_node}, {{1,1}, 3u*pages_per_node}}},
        {"num_pages",  {{{0,0}, pages_per_node}, {{1,0}, pages_per_node}, {{0,1}, pages_per_node}, {{1,1}, pages_per_node}}},
    },
}, /* writer entry similarly */ };
params.tensor_args = {
    {INPUT,  TensorArgument{input_tensor}},
    {OUTPUT, TensorArgument{output_tensor}},
};
SetProgramRunArgs(program, params);
```

Key differences vs. the legacy pattern:

- Multi-core placement moves from `core_ranges` on each kernel to a single `target_nodes` on the `WorkUnitSpec`.
- The semaphore is bound at the kernel spec; the semaphore ID no longer travels as a runtime argument. Kernel code accesses it as `sem::done`.
- **Tensor identity is singular** — one `TensorParameter` per tensor, regardless of how many nodes access it. The legacy column repeats `input_tensor.buffer()->address()` on every node; Metal 2.0 makes that singular by construction. Per-node access varies through `start_page` / `num_pages` slice RTAs only.
- Per-node runtime args are keyed by argument **name**, then by `NodeCoord`.

---

## Troubleshooting

Common pitfalls when migrating from `ProgramDescriptor`:

- **Don't pass tensor addresses as runtime arguments.** Metal 2.0's `TensorBinding` auto-injects per-enqueue base addresses; that's the supported path. If your migration ports `tensor.buffer()->address()` over verbatim as an RTA, revisit — you want a `TensorBinding`.
- **Every kernel must belong to a `WorkUnitSpec`.** A kernel listed in `ProgramSpec::kernels` but not referenced by any `WorkUnitSpec::kernels` has no place to run. This will trigger an error.
- **DFB placement is derived, not specified.** Don't pass a node range to `DataflowBufferSpec` — the DFB lives wherever its bound producer / consumer kernels run. `DataflowBufferSpec` has no `target_nodes` field by design.
- **Local DFB invariant.** A local DFB's producer and consumer kernels must share *identical* `WorkUnitSpec` membership.
- **Compile-time arguments are named only.** Positional CTAs are not part of the Metal 2.0 API; use named CTAs throughout.
- **Runtime varargs are intended for dynamic-count tails.** `num_runtime_varargs` (and `num_common_runtime_varargs`) is the right fit for kernels that consume a variable number of arguments in a loop — e.g., an N-dimensional shape gated on a CTA-known `rank`. For kernels with a fixed set of individually-known arguments, named RTAs are the recommended form, even when porting from a positional legacy interface.
- **`ProgramRunArgs` requires that every named RTA must be set on every node.** Missing an entry for a node where the kernel runs causes `SetProgramRunArgs` to error. The same applies to varargs. (Note: There is also a power-user `ProgramRunArgsView` API that provides a stateful view into the dispatch buffers; it is not yet supported.)

### Cryptic error → likely cause

When a build error or spec-validator failure doesn't make the fix obvious, search this table for the message text before debugging from scratch. Entries here are cases where the symptom and the fix sit far apart — most other errors are best diagnosed by reading the message directly.

| Symptom (substring to grep for in your output) | Likely cause | Fix |
|---|---|---|
| Linker: `undefined reference to 'dfb::...'` (e.g. `dfb::cb_scaled`) inside a kernel TU | Kernel `#include`s the wrong generated header. `dfb::*` lives in `kernel_bindings_generated.h`; `args::*` lives in `kernel_args_generated.h`. | Remove any explicit include of either generated header. The only include a ported kernel adds is `experimental/kernel_args.h` — the framework injects both. |
| Spec-validator: DataflowBuffer with 0 producers / 0 consumers | Host-side topology mismatch — the DFB was declared but only one end of its pipeline binds it. The other end is bound to a different DFB, missing entirely, or doesn't appear in any kernel's `kernel_bindings`. | Fix on the host: add the missing binding. **Do not** modify the kernel's `wait_front` / `pop_front` calls to mask the imbalance — per-execution DFB state is reinitialized, so a tile produced and never consumed is harmless. |
| Spec-validator: TensorParameter with 0 TensorBindings | TensorParameter declared but no kernel binds it. | Bind the tensor on the kernel(s) that use it, or remove the parameter. |
| Spec-validator: `unpack_to_dest_mode` required (or LLK-side dest-accumulator dtype assert) | Compute kernel sets `fp32_dest_acc_en = true` and consumes a `DataType::Float32` DFB, but no `UnpackToDestMode` was set for that DFB on the compute kernel's `ComputeHardwareConfig`. | On the compute kernel's `hw_config` (a `ComputeHardwareConfig`), add an entry to the `unpack_to_dest_mode` table keyed by the DFB's `DFBSpecName`: `unpack_to_dest_mode = {{DFB_NAME, UnpackToDestMode::UnpackToDestFp32}}`. |
| Spec-validator: aliased-DFB rule failure (size / strict-clique / borrowed-memory consistency) | The DFBs declared via `advanced_options.alias_with` violate one of the three legality rules. | The error names which rule fails. Most common case: a DFB was added on one member's alias list but missed on the other(s) — `advanced_options.alias_with` must form a strict clique (every member names every other member). |
| `UpdateTensorArgs` `TensorSpec` legality check fires — typically on a fast-path program-cache hit | The op defines a custom `compute_program_hash` that doesn't fold `TensorSpec` into the hash key. The program cache reports a hit, but the cached `ProgramSpec` was built for a different `TensorSpec`. Metal 2.0's `UpdateTensorArgs` legality check (intentionally kept on) catches the resulting mismatch. | **Do not** attempt to update the custom hash to include `TensorSpec`. Revert the op to use the default TTNN hash (reflection-based, naturally folds in `TensorSpec`). Note the revert in `METAL2_PORT_REPORT.md` under "Open items" so the op author can decide whether to reintroduce a corrected custom hash later. The unnecessary cache misses incurred by reverting are the right trade-off vs. silently-incorrect cache hits. |
