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
   - [ProgramRunParams](#programrunparams)
5. [Device-Side Migration](#device-side-migration)
   - [Circular Buffers → Dataflow Buffers](#circular-buffers--dataflow-buffers)
   - [TensorAccessor](#tensoraccessor)
   - [Kernel Argument Retrieval Syntax](#kernel-argument-retrieval-syntax)
6. [Design Principles](#design-principles)
   - [Principle 1: Immutable spec, mutable run-params](#principle-1-immutable-spec-mutable-run-params)
   - [Principle 2: First-class named resource bindings](#principle-2-first-class-named-resource-bindings)
   - [Principle 3: Named arguments throughout](#principle-3-named-arguments-throughout)
7. [TTNN Framework Integration](#ttnn-framework-integration)
8. [Complete Migration Examples](#complete-migration-examples)
9. [Troubleshooting](#troubleshooting)

---

## Overview

Metal 2.0 introduces a new family of host APIs for Program specification. Metal 2.0 is designed to:
 - Enable **Quasar** (which the legacy APIs do not — and will not — support)
 - Address longstanding user pain points in the legacy APIs

Key Metal 2.0 changes, at a glance:
 - *Immutable and mutable descriptors*. Like `ProgramDescriptor`, Metal 2.0 is a descriptor-based API. But, it separates the mutable properties of a Program (`ProgramSpec`) from those properties that are updated for each execution (`ProgramRunParams`).
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
| `ProgramDescriptor` | `ProgramSpec` (immutable description) + `ProgramRunParams` (per-execution values) |
| `KernelDescriptor` | `KernelSpec` |
| `KernelDescriptor::core_ranges` | derived from `WorkUnitSpec::target_nodes` |
| `KernelDescriptor::compile_time_args` (positional)<br>`KernelDescriptor::named_compile_time_args` (named — partial legacy support) | `KernelSpec::compile_time_arg_bindings` (named *only*) |
| `KernelDescriptor::runtime_args` / `common_runtime_args` | **Schema** (names): declared on `KernelSpec::runtime_arguments_schema`<br>**Values**: supplied per execution on `ProgramRunParams::KernelRunParams` |
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
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
```

Sub-headers are pulled in transitively:
 - `kernel_spec.hpp`
 - `dataflow_buffer_spec.hpp`
 - `semaphore_spec.hpp`
 - `tensor_parameter.hpp`

**These header files are self-documenting, with extensive comments.** Please read them!

All Metal 2.0 host API symbols currently live in the `tt::tt_metal::experimental::metal2_host_api` namespace.

> **Note:** The Program-creation entry points in `metal2_host_api/program.hpp` are *temporary*. Migration to the final form will take place when the APIs leave `experimental` to join the main API directory. User code updates will be trivial.

---

## Host API Migration

### ProgramSpec

`ProgramSpec` is the top-level descriptor, replacing `ProgramDescriptor`. It describes the immutable properties of a Program, analogous to a function's signature and body.

A `Program` is built once from its `ProgramSpec`, then executed many times by supplying fresh `ProgramRunParams` per execution.

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
    .program_id = "my_program",
    .kernels = {kernel_1, kernel_2},
    .dataflow_buffers = {dfb_1, dfb_2},
    .semaphores = {sem_1},
    .work_units = {main_work_unit},
};
Program program = MakeProgramFromSpec(spec);  // temporary free function
//Program program = Program(spec);            // stable API form
```

Two structural additions vs. `ProgramDescriptor`:

- `program_id` — a string identifier for the `ProgramSpec` within a `MeshWorkload`.
- `work_units` — the new top-level concept that declares where kernels run. (See [WorkUnitSpec](#workunitspec).)

---

### KernelSpec

`KernelSpec` replaces `KernelDescriptor`. Notes:

1. **Placement.** The kernel's effective node set is derived from the `WorkUnitSpec`(s) that include it.
2. **Runtime arguments.** The `KernelSpec` declares a runtime arguments _schema_; runtime-arg _values_ are supplied per execution through `ProgramRunParams` or `ProgramRunParamsView`.
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
constexpr const char* READER = "reader";

KernelSpec reader{
    .unique_id = READER,
    .source = KernelSpec::SourceFilePath{"kernels/reader.cpp"},
    // (Placement is declared on WorkUnitSpec, below.)
    .compile_time_arg_bindings = {
        {"src_cb_idx", src_cb_idx},
        {"dst_cb_idx", dst_cb_idx},
        {"page_size", page_size},
    },
    .runtime_arguments_schema = {
        // Schema only — argument values are set per execution, on ProgramRunParams.
        .named_runtime_args = {"start_page", "num_pages"},
    },
    .config_spec = DataMovementConfiguration{
        // For WH/BH
        .gen1_data_movement_config = DataMovementConfiguration::Gen1DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
        },
        // For Quasar
        .gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{}
        // (Only one config is required; supply both for architecture portability)
    },
};

// ----- WorkUnitSpec: kernel placement and worker assignments -----
WorkUnitSpec main_work_unit{
    .unique_id = "main",
    .kernels = {READER},
    .target_nodes = NodeCoord{0, 0},
};

// ----- ProgramSpec: assemble into the program description -----
ProgramSpec spec{
    .program_id = "my_program",
    .kernels = {reader},
    .work_units = {main_work_unit},
};
Program program = MakeProgramFromSpec(spec);

// ----- ProgramRunParams: argument values, set per execution -----
ProgramRunParams params;
params.kernel_run_params = {{
    .kernel_spec_name = READER,
    .named_runtime_args = {{NodeCoord{0, 0},
        {{"start_page", start_page}, {"num_pages", num_pages}}}},
}};
SetProgramRunParameters(program, params);
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
constexpr const char* MY_DFB = "my_dfb";

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
        .local_accessor_name = "out_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER
    }},
};
KernelSpec consumer{ /* ... */
    .dfb_bindings = {{
        .dfb_spec_name = MY_DFB,
        .local_accessor_name = "in_dfb",  // independent of the producer's name
        .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER
    }},
};
```

Advanced features (aliased DFBs, borrowed-memory DFBs, remote DFBs spanning nodes) are described in `dataflow_buffer_spec.hpp`, but are not yet supported by Metal 2.0.

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
constexpr const char* DONE = "done";

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

`TensorParameter` declares a tensor as a Program-scope resource. Kernels access it via `KernelSpec::TensorBinding`; the runtime `MeshTensor` is supplied per execution via `ProgramRunParams::TensorArg`. The kernel-author API collapses to a single line: `TensorAccessor(ta::name)`.

Three pieces, paralleling the DFB / Semaphore pattern with one deliberate asymmetry: tensors are *user-managed* resources (you own the lifetime), so the program-scope type is named `TensorParameter` (distinguished from the "Spec" pattern used elsewhere in the API) — echoing the "ProgramSpec is a function signature; ProgramRunParams is the call args" framing.

One thing to be aware of: `TensorSpec` is a property of a `MeshTensor`. This pre-exists Metal 2.0; it is not part of the "Spec" object pattern in the rest of the Metal 2.0 APIs.

> **⚠ Pre-migration check.** Before migrating an op, grep its kernel sources for `ArgConfig::Runtime`. If any kernel uses **`ArgConfig::RuntimeTensorShape`**, this op cannot migrate to Metal 2.0 yet. Metal 2.0 has no positional-CTA mechanism, so the legacy `TensorAccessorArgs(buffer, ArgConfig::RuntimeTensorShape).append_to(...)` plumbing has no equivalent in the current API. Stay on the legacy `ProgramDescriptor` path until the follow-up PR adds runtime-shape support. (The other deferred flavors — `RuntimeRank`, `RuntimeNumBanks`, `RuntimeShardShape`, `RuntimeBankCoords` — have zero user sites outside tests, so this check almost always reduces to "is `RuntimeTensorShape` present?".)

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
constexpr const char* INPUT = "input";

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

// In ProgramRunParams — supply the actual MeshTensor per execution.
ProgramRunParams params;
params.tensor_args = {
    {.tensor_parameter_name = INPUT, .tensor = input_tensor},
};
SetProgramRunParameters(program, params);
```

```cpp
// kernel: one line. No CTA-offset bookkeeping, no buffer-address RTA.
auto input = TensorAccessor(ta::input);
```

The buffer-address RTA is gone — the binding mechanism auto-injects the per-enqueue base address. The `TensorAccessorArgs<N>()` line is gone too — the layout metadata is packed by the host at program creation.

#### Migration recipe

For each op:

1. **Pre-flight.** Grep the op's kernel sources for `ArgConfig::Runtime`. If `RuntimeTensorShape` appears anywhere, **do not migrate this op** (see callout above).
2. **Find each `TensorAccessor`.** In each kernel, locate every `TensorAccessor(args, addr)` construction. For each, trace `addr` back through the host code to the originating `Tensor`.
3. **Declare `TensorParameter`s.** Add one entry per tensor to `ProgramSpec::tensor_parameters`, using `tensor.tensor_spec()`. Pick a stable `unique_id`; reuse a `constexpr const char*` constant per the convention.
4. **Add `TensorBinding`s.** On each `KernelSpec` whose kernel accesses the tensor, add an entry to `tensor_bindings` referencing the parameter by name. The `accessor_name` is the kernel-side identifier — it will appear as `ta::<accessor_name>`.
5. **Update kernel code.**
   - Replace `TensorAccessor(args, addr)` with `TensorAccessor(ta::<accessor_name>)`.
   - Drop the `TensorAccessorArgs<offset>()` line and any manual `next_compile_time_args_offset()` chaining.
   - Drop the `get_arg_val<uint32_t>(N)` line that retrieved the buffer address — those bytes are no longer in your RTA list, so re-index any RTAs that came after.
6. **Wire `tensor_args`.** Add one `TensorArg` per `TensorParameter` to `ProgramRunParams::tensor_args`, passing the actual `MeshTensor`.

#### Validation

At enqueue, the runtime `MeshTensor`'s `TensorSpec` must equal the binding's declared spec. Mismatches error loudly with a message naming the binding. The binding declaration is the single source of truth for layout — if the supplied tensor doesn't match, the bug is at the call site (typically: a layout transformation that should have been wired into the program but was applied externally to the tensor).

#### Multi-kernel-same-tensor

The common case (matmul reader + compute reading the same input; reshard reader/writer pipelines) is the cleanest fit for the binding model. Each kernel declares its own `TensorBinding` to the same `TensorParameter` with its own kernel-local `accessor_name`; the underlying tensor identity stays singular at the program level. This is structurally different from the legacy pattern, where the same tensor's address would be passed as a separate RTA on every kernel — and divergence between those was a real (silent) failure mode.

#### What's not covered yet

- **`RuntimeTensorShape`** — eltwise unary/ternary/binary_ng currently rely on this. Follow-up PR ships shortly.
- **Manual `DistributionSpec` reuse** (the "advanced reuse-bank-coords-across-accessors" path from the [TensorAccessor guide](../../../../tech_reports/tensor_accessor/tensor_accessor.md)). Not commonly used in production ops.
- **Tensor bindings on compute kernels** are out of scope. `TensorAccessor` was never supported for compute kernels (TRISC builds don't compile its NoC-using includes), so migration should not encounter any.

---

### WorkUnitSpec

`WorkUnitSpec` is a new top-level concept in Metal 2.0. It declares groups of kernels that operate together on a worker node, and on which nodes they run.

**Single work unit on one node:**

```cpp
WorkUnitSpec wu{
    .unique_id = "main",
    .kernels = {READER, WRITER},
    .target_nodes = NodeCoord{0, 0},
};
```

**One work unit spanning a range of nodes:**

```cpp
WorkUnitSpec wu{
    .unique_id = "main",
    .kernels = {READER, COMPUTE, WRITER},
    .target_nodes = NodeRange{{0, 0}, {3, 3}},  // 4×4 grid
};
```

**A kernel belonging to multiple work units** (e.g. for a kernel that runs on both an inner block and a halo region, with different DFB partners in each):

```cpp
WorkUnitSpec wu_inner{
    .unique_id = "inner",
    .kernels = {COMPUTE, INNER_READER},
    .target_nodes = inner_node_range_set
};
WorkUnitSpec wu_halo{
    .unique_id = "halo",
    .kernels = {COMPUTE, HALO_READER},
    .target_nodes = halo_node_range_set
};
// COMPUTE's effective node set is the union of inner + halo.
```

---

### ProgramRunParams

`ProgramRunParams` describes the mutable properties of the Program. These parameters are specified anew with each Program enqueue:
 - Kernel runtime arguments
 - Kernel common runtime arguments
 - Tensor arguments (`tensor_args`) — one `MeshTensor` per declared `TensorParameter`. See [TensorParameter](#tensorparameter).
 - (optional) DFB size + entry size (not yet supported)
 - (optional) DFB borrowed memory (not yet supported)

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
    .runtime_arguments_schema = {
        .named_runtime_args = {"start_page", "num_tiles"},
        .named_common_runtime_args = {"bank_id"},
    },
};

// Values supplied per execution:
ProgramRunParams params;
params.kernel_run_params = {{
    .kernel_spec_name = READER,
    .named_runtime_args = {{NodeCoord{0, 0},
        {{"start_page", start_page}, {"num_tiles", num_tiles}}}},
    .named_common_runtime_args = {{"bank_id", bank_id}},
}};
SetProgramRunParameters(program, params); // temporary free function
```

Vararg form — for kernels with a genuinely dynamic argument tail (loop-retrieved on the device side). The canonical fit is a CTA-bound count plus a per-element payload, e.g. an N-dimensional shape:

```cpp
// Schema:
.compile_time_arg_bindings = {{"rank", rank}},
.runtime_arguments_schema = {
    .num_runtime_varargs = rank,  // shape: one entry per dimension
},

// Values:
.runtime_varargs = {{NodeCoord{0, 0}, shape_dims}},  // shape_dims.size() == rank
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
// Host-side DFB binding declared local_accessor_name = "my_dfb".
// Auto-generated from that: constexpr DFBAccessor dfb::my_dfb;
experimental::DataflowBuffer dfb(dfb::my_dfb);

dfb.reserve_back(num_entries);
uint32_t write_ptr = dfb.get_write_ptr();
// ... write data ...
dfb.push_back(num_entries);
```


> **Implicit sync (Quasar)**: On Gen2, you can elide the explicit `reserve_back` / `push_back` (or `wait_front` / `pop_front`) pattern entirely. Pass the DFB directly to `experimental::Noc::async_read` or `async_write`, and the runtime hardware handles the FIFO sync via ISR. This is the default behavior; you can disable it per-DFB by setting `DataflowBufferSpec::disable_implicit_sync = true` (rarely needed). On Gen1, the option is a no-op — the explicit FIFO pattern shown above remains the way.

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

Include `experimental/kernel_args.h` in any kernel that uses the named-argument API. The `args::` and `dfb::` / `sem::` namespaces are auto-generated from the host-side bindings into `kernel_bindings_generated.h`, which the kernel build system pulls in.

**Example 1 — Named arguments only.**

Suppose the host declared the following on `KernelSpec`:

```cpp
.compile_time_arg_bindings = {{"bank_id", 0}, {"entry_size", 1024}},
.runtime_arguments_schema = {
    .named_runtime_args = {"start_page"},
    .named_common_runtime_args = {"num_entries"},
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
.compile_time_arg_bindings = {{"rank", rank}},
.runtime_arguments_schema = {
    .named_runtime_args = {"start_page"},
    .named_common_runtime_args = {"num_entries"},
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

### Principle 1: Immutable spec, mutable run-params

Metal 2.0 splits a Program's description into two parts:

- **`ProgramSpec`** — the immutable description (the "function signature and body"). Captures kernel sources, resource declarations, work-unit assignments, and argument *schemas*.
- **`ProgramRunParams`** — the per-execution mutable values (the "call arguments"). Carries runtime argument values, tensor identities, and any other field that may legitimately differ between executions.

This split is the **only** structural divergence from legacy `ProgramDescriptor`. Every other Metal 2.0 type maps roughly 1:1 with its `ProgramDescriptor` counterpart — `KernelSpec` ↔ `KernelDescriptor`, `DataflowBufferSpec` ↔ `CBDescriptor`, `SemaphoreSpec` ↔ `SemaphoreDescriptor`, and so on.

The separation enables two performance optimizations the legacy API couldn't express:

1. **Caching**: a `Program` built once from a `ProgramSpec` can be executed many times with fresh `ProgramRunParams` — the compile/build cost is paid once.
2. **Partial updates**: for ops whose mutable surface is small (typically just the tensor identities), the framework can apply a partial update on cache hit instead of re-issuing the full `ProgramRunParams`.

For the op author, the practical implication is that **the schema and the values are conceptually paired** even though they live in different structures. When you add a named RTA to a `KernelSpec`'s schema, you simultaneously add its per-node value to `ProgramRunParams::kernel_run_params`. When you declare a `TensorParameter`, you simultaneously add its `TensorArg`. The spec/run-params separation is most visible to the framework and tooling; during construction, the two are built together.

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

**Optional resources.** A binding may be declared conditionally on the host (omitted from `KernelSpec::dfb_bindings` / `tensor_bindings` when the path isn't taken), and the kernel gates **both the wrapper declaration and uses** with `if constexpr` on a CTA:

```cpp
// Host side:
KernelSpec compute{
    .compile_time_arg_bindings = {{"do_scale", do_scale ? 1u : 0u}, /* ... */},
    .dfb_bindings = do_scale
        ? std::vector<DFBBinding>{INPUT, OUTPUT, SCALED}
        : std::vector<DFBBinding>{INPUT, OUTPUT},
};

// Kernel side:
constexpr auto do_scale = get_arg(args::do_scale);
if constexpr (do_scale) {
    experimental::DataflowBuffer cb_scaled(dfb::cb_scaled);  // wrapper inside the gate
    cb_scaled.wait_front(...);
    // ... all uses of cb_scaled
}
```

The "always bind and gate only the uses" pattern (wrapper declared outside the `if constexpr`) is an anti-pattern: it pays L1 unnecessarily for an unused buffer. See the [patterns catalog](metal2_port_patterns.md) for the full discussion.

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

> **Use varargs only when the kernel reads its arguments in a loop.** Varargs are designed for kernels whose device-side code retrieves arguments via `get_vararg(i)` where `i` is a runtime variable — the canonical case is an N-dimensional shape gated on a CTA-known `rank`. When each argument is referenced by a constant index (`get_vararg(0)`, `get_vararg(1)`, ...), the named form is clearer on both sides. A port from positional RTAs to varargs may compile and run, but it preserves the legacy positional vocabulary instead of upgrading to Metal 2.0's named one. See the [patterns catalog](metal2_port_patterns.md) caution on varargs.

---

## TTNN Framework Integration

TTNN ops integrate with the Metal 2.0 host API through the `ttnn::device_operation` framework. The framework provides three factory *concepts*; an op's program factory satisfies whichever concept matches its construction model:

- **`ProgramFactoryConcept`** — the oldest concept; legacy `host_api.hpp` builder-style factories. Returns a `CachedProgram<shared_variables_t>` from `create()`; updates per-execution state via `override_runtime_arguments()`.
- **`ProgramDescriptorFactoryConcept`** — the intermediate concept; legacy `ProgramDescriptor`-based factories. Returns a `tt::tt_metal::ProgramDescriptor` from `create_descriptor()`.
- **`ProgramSpecFactoryConcept`** — the Metal 2.0 concept. Returns a `ttnn::device_operation::ProgramArtifacts` (a pair of `ProgramSpec` and `ProgramRunParams`) from `create_program_spec()`.

A new Metal 2.0 program factory has the following shape:

```cpp
// In your factory header:
struct MyProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const OpParams& operation_attributes,
        const OpInputs& tensor_args,
        Tensor& tensor_return_value);
};

// In your factory implementation:
ttnn::device_operation::ProgramArtifacts MyProgramFactory::create_program_spec(
    const OpParams& operation_attributes,
    const OpInputs& tensor_args,
    Tensor& tensor_return_value) {

    // ... build the spec and run-params ...

    metal2_host_api::ProgramSpec spec{
        .program_id = "my_program",
        // ...
    };
    metal2_host_api::ProgramRunParams run_params;
    // ... populate run_params ...

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}
```

> **`OpParams` and `OpInputs` are placeholders.** Each real device-op anchors to its own `operation_attributes_t` and `tensor_args_t` aliases — use those.
>
> **Extract `MeshTensor` from each input `ttnn::Tensor` at the start of host code.** In a Metal 2.0 factory body, the `MeshTensor` (not the `ttnn::Tensor`) is the object that gets passed around. The first step inside `create_program_spec` is to extract the `MeshTensor` from each incoming tensor and work with those for the remainder. A fuller mini-recipe for MeshTensor migration is forthcoming and will be slotted into the workflow; for now, treat this extraction as a fixed prelude.

A Metal 2.0 factory **does not** construct the `Program` itself, nor does it call `SetProgramRunParameters` directly. Those are the framework adapter's responsibilities.

### Cache-miss vs cache-hit lifecycle

The framework manages two distinct execution paths:

- **Cache miss** (first execution, or stale cache): the framework calls `create_program_spec`, then internally calls `MakeProgramFromSpec(*device, spec)` to build the `Program`, then calls `SetProgramRunParameters(program, run_params)` to apply the per-execution values. The full spec is realized exactly once.
- **Cache hit** (subsequent executions with a still-valid cached `Program`): the framework calls `create_program_spec` again to get the fresh `ProgramArtifacts`, then internally calls `UpdateTensorArgs(program, run_params)` (or, where unavailable, a full re-call of `SetProgramRunParameters`) to apply just the changed pieces. The build cost is skipped.

The framework dispatches based on which concept the factory satisfies — there is no per-op opt-in. A factory satisfying `ProgramSpecFactoryConcept` is routed through `ProgramSpecMeshWorkloadFactoryAdapter`, which handles both paths.

### TTNN's immutability convention

A practical consequence of TTNN's caching: **most parameters that *could* vary between calls are treated as if they were immutable, and the cache is invalidated when they change**. The actually-mutable surface across calls is small — typically just the tensor identities (`tensor_args` in `ProgramRunParams`).

This means an op author specifying a Metal 2.0 factory does not directly experience the spec/run-params separation as a per-call distinction. The two are constructed together in `create_program_spec`; the framework's cache machinery converts that single construction call into either a full realization (cache miss) or a partial update (cache hit). Designing your factory to construct paired resources and values together — `TensorParameter` and its `TensorArg`, RTA schema and its values — reflects the actual authoring flow.

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
constexpr const char* READER = "reader";
constexpr const char* WRITER = "writer";
constexpr const char* DFB    = "loopback_dfb";
constexpr const char* INPUT  = "input";
constexpr const char* OUTPUT = "output";

constexpr uint32_t page_size = 1024;
constexpr uint32_t num_pages = 8;
const NodeCoord node{0, 0};

KernelSpec reader{
    .unique_id = READER,
    .source = KernelSpec::SourceFilePath{"kernels/reader.cpp"},
    .compile_time_arg_bindings = {{"page_size", page_size}},
    .runtime_arguments_schema = {.named_runtime_args = {"num_pages"}},
    .dfb_bindings = {{
        .dfb_spec_name = DFB,
        .local_accessor_name = "out_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER
    }},
    .tensor_bindings = {{
        .tensor_parameter_name = INPUT,
        .accessor_name = "input",   // kernel accesses as `ta::input`
    }},
    .config_spec = DataMovementConfiguration{
        .gen1_data_movement_config = {.processor = DataMovementProcessor::RISCV_0},
    },
};

KernelSpec writer{
    .unique_id = WRITER,
    .source = KernelSpec::SourceFilePath{"kernels/writer.cpp"},
    .compile_time_arg_bindings = {{"page_size", page_size}},
    .runtime_arguments_schema = {.named_runtime_args = {"num_pages"}},
    .dfb_bindings = {{
        .dfb_spec_name = DFB,
        .local_accessor_name = "in_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER
    }},
    .tensor_bindings = {{
        .tensor_parameter_name = OUTPUT,
        .accessor_name = "output",  // kernel accesses as `ta::output`
    }},
    .config_spec = DataMovementConfiguration{
        .gen1_data_movement_config = {.processor = DataMovementProcessor::RISCV_1},
    },
};

DataflowBufferSpec dfb{
    .unique_id = DFB,
    .entry_size = page_size,
    .num_entries = num_pages,
    .data_format_metadata = tt::DataFormat::Float16_b,
};

ProgramSpec spec{
    .program_id = "loopback",
    .kernels = {reader, writer},
    .dataflow_buffers = {dfb},
    .tensor_parameters = {
        {.unique_id = INPUT,  .spec = input_tensor.tensor_spec()},
        {.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
    },
    .work_units = {{
        .unique_id = "main",
        .kernels = {READER, WRITER},
        .target_nodes = node,
    }},
};
Program program = MakeProgramFromSpec(*mesh_device, spec);

ProgramRunParams params;
params.kernel_run_params = {
    {.kernel_spec_name = READER,
     .named_runtime_args = {{node, {{"num_pages", num_pages}}}}},
    {.kernel_spec_name = WRITER,
     .named_runtime_args = {{node, {{"num_pages", num_pages}}}}},
};
params.tensor_args = {
    {.tensor_parameter_name = INPUT,  .tensor = input_tensor},
    {.tensor_parameter_name = OUTPUT, .tensor = output_tensor},
};
SetProgramRunParameters(program, params);
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
constexpr const char* DONE = "done";
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
    .runtime_arguments_schema = {.named_runtime_args = {"start_page", "num_pages"}},
    // ...
};

ProgramSpec spec{
    .program_id = "loopback_2x2",
    .kernels = {reader, writer},
    .dataflow_buffers = {dfb},
    .semaphores = {done_sem},
    .tensor_parameters = {
        {.unique_id = INPUT,  .spec = input_tensor.tensor_spec()},
        {.unique_id = OUTPUT, .spec = output_tensor.tensor_spec()},
    },
    .work_units = {{
        .unique_id = "main",
        .kernels = {READER, WRITER},
        .target_nodes = cores,  // 2×2 grid
    }},
};
Program program = MakeProgramFromSpec(*mesh_device, spec);

ProgramRunParams params;
// One named_runtime_args entry per node where the kernel runs.
// Tensor identity is singular: one TensorParameter for the input tensor,
// regardless of how many nodes access it. Per-node access varies via slice
// indices, not addresses.
params.kernel_run_params = {{
    .kernel_spec_name = READER,
    .named_runtime_args = {
        {{0,0}, {{"start_page", 0u},                  {"num_pages", pages_per_node}}},
        {{1,0}, {{"start_page", 1u*pages_per_node},   {"num_pages", pages_per_node}}},
        {{0,1}, {{"start_page", 2u*pages_per_node},   {"num_pages", pages_per_node}}},
        {{1,1}, {{"start_page", 3u*pages_per_node},   {"num_pages", pages_per_node}}},
    },
}, /* writer entry similarly */ };
params.tensor_args = {
    {.tensor_parameter_name = INPUT,  .tensor = input_tensor},
    {.tensor_parameter_name = OUTPUT, .tensor = output_tensor},
};
SetProgramRunParameters(program, params);
```

Key differences vs. the legacy pattern:

- Multi-core placement moves from `core_ranges` on each kernel to a single `target_nodes` on the `WorkUnitSpec`.
- The semaphore is bound at the kernel spec; the semaphore ID no longer travels as a runtime argument. Kernel code accesses it as `sem::done`.
- **Tensor identity is singular** — one `TensorParameter` per tensor, regardless of how many nodes access it. The legacy column repeats `input_tensor.buffer()->address()` on every node; Metal 2.0 makes that singular by construction. Per-node access varies through `start_page` / `num_pages` slice RTAs only.
- Per-node runtime args are still per-node, but addressed by `NodeCoord` and named.

---

## Troubleshooting

Common pitfalls when migrating from `ProgramDescriptor`:

- **Don't pass tensor addresses as runtime arguments.** Metal 2.0's `TensorBinding` auto-injects per-enqueue base addresses; that's the supported path. If your migration ports `tensor.buffer()->address()` over verbatim as an RTA, revisit — you want a `TensorBinding`.
- **Every kernel must belong to a `WorkUnitSpec`.** A kernel listed in `ProgramSpec::kernels` but not referenced by any `WorkUnitSpec::kernels` has no place to run. This will trigger an error.
- **DFB placement is derived, not specified.** Don't pass a node range to `DataflowBufferSpec` — the DFB lives wherever its bound producer / consumer kernels run. `DataflowBufferSpec` has no `target_nodes` field by design.
- **Local DFB invariant.** A local DFB's producer and consumer kernels must share *identical* `WorkUnitSpec` membership.
- **Compile-time arguments are named only.** Positional CTAs are not part of the Metal 2.0 API; use named CTAs throughout.
- **Runtime varargs are intended for dynamic-count tails.** `num_runtime_varargs` (and `num_common_runtime_varargs`) is the right fit for kernels that consume a variable number of arguments in a loop — e.g., an N-dimensional shape gated on a CTA-known `rank`. For kernels with a fixed set of individually-known arguments, named RTAs are the recommended form, even when porting from a positional legacy interface.
- **`ProgramRunParams` requires that every named RTA must be set on every node.** Missing an entry for a node where the kernel runs causes `SetProgramRunParameters` to error. The same applies to varargs. (Note: There is also a power-user `ProgramRunParamsView` API that provides a stateful view into the dispatch buffers; it is not yet supported.)
