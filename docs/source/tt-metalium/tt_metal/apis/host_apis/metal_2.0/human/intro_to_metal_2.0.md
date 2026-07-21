# Introduction to Metal 2.0 Migration

> **This is a human-facing document!**
> (It's a WIP and a little messy right now, apologies!)
>
> If you are an AI agent, please refer to `../ai/shared/migration_guide.md` instead.
>

This guide introduces the new Metal 2.0 host APIs in `tt_metal/api/tt-metalium/experimental/metal2_host_api/`, and outlines the migration from the legacy `ProgramDescriptor` / `host_api.hpp` APIs. Its goal is to give you enough of the concepts and vocabulary to understand a Metal 2.0 program — and to read and vet a port — not to be an exhaustive API reference. The headers themselves are the reference; they are extensively commented.

The focus of this guide is **Wormhole** and **Blackhole** (Gen1 architectures).

Things to remember:
 - Metal 2.0 is still an `experimental` API.
 - Not all planned improvements are available yet.
 - A small handful of legacy API features are not yet supported.

> **Prerequisite**: [Device 2.0 Data Movement migration](../../../kernel_apis/data_movement/device_api_migration_guide.md) is a pre-requisite to Metal 2.0 migrations. All the kernel code migration examples in this document use Device 2.0 APIs.

## Table of Contents

1. [Overview](#overview)
2. [Concept Map](#concept-map)
3. [Header Files](#header-files)
4. [Host API](#host-api)
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
7. [TTNN Framework Integration](#ttnn-framework-integration)
8. [Complete Migration Example](#complete-migration-example)

---

## Overview

Metal 2.0 introduces a new family of host APIs for Program specification. Metal 2.0 is designed to:
 - Enable **Quasar** (which the legacy APIs do not — and will not — support)
 - Address longstanding user pain points in the legacy APIs

Key Metal 2.0 changes, at a glance:
 - **Immutable and mutable descriptors**. Like `ProgramDescriptor`, Metal 2.0 is a descriptor-based API. However, it separates the immutable properties of a Program (`ProgramSpec`) from those properties that are updated for each execution (`ProgramRunArgs`).
 - **Dataflow Buffers (DFBs)** replace Circular Buffers (CBs). Both host and device-side syntax is improved.
 - **Kernel arguments** specification (host side) and retrieval (device side) are significantly improved. (_Note_: Typed and array kernel arguments are planned, but not yet implemented.)
 - **Resource placement** (i.e. `core_ranges`) is inferred where possible. The mapping of kernels to worker nodes in the device is communicated via a new top-level concept (`WorkUnitSpec`).

Two benefits stand out:

- **Named arguments**. Compile-time, runtime, and common runtime arguments are addressed by name on both the host and device sides. Positional CTAs are gone from the API entirely; positional RTAs/CRTAs survive only as **varargs**, intended for the rare kernels that read a dynamic-count argument tail in a loop.
- **Named resource bindings**. Metal 2.0 natively supports binding resources (DFBs, semaphores, tensors, etc) to kernels. The corresponding handles are cleanly passed to the device code with user-defined accessor names.

> **Raw pointer argument deprecation.** Eliminating raw device pointer kernel arguments was an explicit goal of Metal 2.0. With tensor parameters, DFBs, and semaphores all bindable as named resources, runtime args carrying a buffer or tensor address should now be the exception rather than the rule. (On the default TTNN factory concept, the TTNN infra makes pointer arguments fully illegal.)

> **Argument naming.** Metal 2.0 is designed around named arguments. All argument types (compile time, runtime, common runtime) can be named. Positional arguments survive only as varargs — intended for kernels with a genuinely dynamic argument count consumed in a loop.

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
| `TensorAccessorArgs<...>` <br> (plumbing + buffer-address RTA) | `TensorAccessor(tensor::name)` in the kernel code; <br>`TensorParameter` on `ProgramSpec` (parallel to DFB / Semaphore) |
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

Each Metal 2.0 descriptor object is defined in its own sub-header (`kernel_spec.hpp`, `dataflow_buffer_spec.hpp`, `semaphore_spec.hpp`, `tensor_parameter.hpp`, etc.), pulled in transitively by the headers above.

**All Metal 2.0 header files are self-documenting, with extensive comments.** Please read them!

All Metal 2.0 host API symbols temporarily live in the `tt::tt_metal::experimental` namespace.

> **Note:** The Program-creation entry points in `metal2_host_api/program.hpp` are *temporary*; these will be replaced with `Program` and `MeshWorkload` constructors and methods once the Metal 2.0 APIs graduate from `experimental`. User code updates will be trivial.

---

## Host API

### ProgramSpec

`ProgramSpec` is the top-level descriptor, replacing `ProgramDescriptor`. It describes the immutable properties of a Program, analogous to a function's signature and body. A `Program` is built once from its `ProgramSpec`, then executed many times by supplying fresh `ProgramRunArgs` per execution.

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
//Program program = Program(*mesh_device, spec);            // stable API form
```

Two structural additions vs. `ProgramDescriptor`:

- `name` — a human-readable label for debug and messaging (plain `std::string`; no uniqueness invariant).
- `work_units` — the new top-level concept that declares where kernels run. (See [WorkUnitSpec](#workunitspec).)

---

### KernelSpec

`KernelSpec` replaces `KernelDescriptor`. The main differences:

1. **Placement.** The kernel's effective node set is derived from the `WorkUnitSpec`(s) that include it — you no longer set `core_ranges` on the kernel.
2. **Runtime arguments.** The `KernelSpec` declares a runtime arguments **_schema_** (names); the **_values_** are supplied per execution through `ProgramRunArgs`.
3. **Resource bindings.** New syntax to bind DFB endpoints, semaphores, and tensors to the kernel, and retrieve them by name in device code.

**Legacy** (`KernelDescriptor`):

```cpp
KernelDescriptor reader = {
    .kernel_source = "kernels/reader.cpp",
    .core_ranges = CoreRangeSet{CoreRange{{0,0}, {0,0}}},
    .compile_time_args = {src_cb_idx, dst_cb_idx, page_size},
    .runtime_args = {{{0,0}, {start_page, num_pages}}},
    .config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_0,
    },
};
```

**Metal 2.0** (`KernelSpec`):

```cpp
const KernelSpecName READER{"reader"};

KernelSpec reader{
    .unique_id = READER,
    .source = "kernels/reader.cpp",
    // (Placement is declared on WorkUnitSpec — see below.)
    .compile_time_args = {
        {"src_cb_idx", src_cb_idx},
        {"dst_cb_idx", dst_cb_idx},
        {"page_size", page_size},
    },
    .runtime_arg_schema = {
        // Schema only — argument values are set per execution, on ProgramRunArgs.
        .runtime_arg_names = {"start_page", "num_pages"},
    },
    .hw_config = CreateReaderGen1DataMovementConfig(),
};
```

Placement comes from a `WorkUnitSpec` that names this kernel, and the runtime-arg *values* come from `ProgramRunArgs` at execution time — both shown in the [Complete Migration Example](#complete-migration-example).

> **Multiple `KernelSpec`s per source.** A single kernel source file may be represented by multiple `KernelSpec`s if structural specialization is needed (different CTA bindings, different DFB or semaphore bindings, etc.). Each `KernelSpec` is compiled and placed independently.

---

### DataflowBufferSpec

`DataflowBufferSpec` replaces `CBDescriptor`. Some host-side changes:

1. **Kernel bindings.** A DFB's producer and consumer kernels each declare a `DFBBinding` on their `KernelSpec`, naming the DFB endpoint and a local accessor name. The kernel code references the DFB through that accessor name (e.g. `dfb::my_dfb`); the magic-number CB index is gone.
2. **Placement is derived.** A DFB lives wherever its bound producer / consumer kernels run; you do not pass `core_ranges`.

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

### SemaphoreSpec

`SemaphoreSpec` replaces `SemaphoreDescriptor`. Some notes:

1. **Kernel resource binding.** Semaphores are bound by the `KernelSpec`. The kernel code accesses the semaphore by name through the binding's `accessor_name`.
2. **Placement is explicit.** Unlike DFBs, semaphores are not node-local resources; they are accessed by kernels all over the node grid. You must explicitly specify their target nodes.
3. **Initial value.** Semaphores are default-initialized to zero. Non-zero semaphore initialization is not supported on Quasar.

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

`TensorParameter` declares a tensor as a Program-scope resource. Kernels access it via a `TensorBinding` on the `KernelSpec`; the runtime `MeshTensor` is supplied per execution via `ProgramRunArgs::tensor_args`. The kernel-author API collapses to a single line: `TensorAccessor(tensor::name)`.

Three pieces, paralleling the DFB / Semaphore pattern with one deliberate asymmetry: tensors are *user-managed* resources (you own the lifetime), so the program-scope type is named `TensorParameter` rather than following the "Spec" pattern used elsewhere.

**Legacy:**

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

**Metal 2.0:**

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
        .accessor_name = "input",   // kernel accesses as `tensor::input`
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
auto input = TensorAccessor(tensor::input);
```

The buffer-address RTA is gone — the binding mechanism auto-injects the per-enqueue base address. The `TensorAccessorArgs<N>()` line is gone too — the layout metadata is packed by the host at program creation.

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

**A kernel belonging to multiple work units** (e.g. a kernel that runs on both an inner block and a halo region, with different DFB partners in each):

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

`ProgramRunArgs` describes the mutable properties of the Program — the values specified anew with each Program enqueue:
 - Kernel runtime arguments (and common runtime arguments)
 - Tensor arguments (`tensor_args`) — one `MeshTensor` per declared `TensorParameter`.

**Legacy** (values on the descriptor):

```cpp
KernelDescriptor reader = {
    // ...
    .runtime_args = {{{0, 0}, {start_page, num_tiles}}},
    .common_runtime_args = {bank_id},
};
```

**Metal 2.0** (schema on the spec, values on the run args):

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

> **Varargs (niche).** For kernels that read a genuinely dynamic argument tail in a loop — the canonical case being an N-dimensional shape where N is a compile-time `rank` — Metal 2.0 provides a positional "varargs" mechanism alongside named arguments.

---

## Device-Side Migration

### Circular Buffers → Dataflow Buffers

Metal 2.0 replaces Circular Buffers (CBs) with Dataflow Buffers (DFBs). Notes:

1. **Construction from local accessor name.** You construct your DFB from its local accessor, not a magic-number `cb_id`. The accessor is auto-generated from the host-side DFB binding and lives in the `dfb::` namespace.
2. **`CircularBuffer`-compatible APIs.** The kernel-side DFB APIs are drop-in-compatible with the Device 2.0 `CircularBuffer` wrapper methods and semantics on Gen1.
3. **Direct use in LLK compute APIs (WH/BH).** The `dfb::my_dfb` accessor constants implicitly convert to `uint32_t`, so on WH/BH you can pass them directly to LLK compute APIs (`reduce_init`, `pack_tile`, `cb_wait_front`, etc.) that take a raw CB id — no `.id` extraction needed.

**Legacy (Device 2.0 `CircularBuffer`):**

```cpp
constexpr uint32_t cb_id = 0;
CircularBuffer cb(cb_id);

cb.reserve_back(num_pages);
uint32_t write_ptr = cb.get_write_ptr();
// ... write data ...
cb.push_back(num_pages);
```

**Metal 2.0 (`DataflowBuffer`):**

```cpp
// Host-side DFB binding declared accessor_name = "my_dfb".
// Auto-generated from that: constexpr DFBAccessor dfb::my_dfb;
DataflowBuffer dfb(dfb::my_dfb);

dfb.reserve_back(num_entries);
uint32_t write_ptr = dfb.get_write_ptr();
// ... write data ...
dfb.push_back(num_entries);
```

> **Implicit sync (Quasar).** On Gen2, you can elide the explicit `reserve_back` / `push_back` (or `wait_front` / `pop_front`) pattern entirely — pass the DFB directly to `Noc::async_read` / `async_write` and the hardware handles the FIFO sync. On Gen1 this is a no-op; the explicit FIFO pattern shown above remains the way.

---

### TensorAccessor

Construction collapses to one line. `TensorAccessor` takes the codegen-emitted token directly; everything else is unchanged from today (`get_noc_addr`, `get_bank_and_offset`, `dspec()`, `noc_async_read_page`, etc.). The token lives in the `tensor::` namespace, parallel to `dfb::` and `sem::`.

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
auto input = TensorAccessor(tensor::input);
auto index = TensorAccessor(tensor::index);
```

The `TensorAccessorArgs<N>()` lines, the manual `next_compile_time_args_offset()` chaining, and the buffer-address RTAs are all gone. Layout metadata is packed by the host into the kernel's compile-time args at program creation; the per-enqueue base address rides on a host-managed slot the kernel never sees directly.

---

### Kernel Argument Retrieval Syntax

The Metal 2.0 host API declares kernel arguments by name; the kernel-side API retrieves them by name. This replaces the legacy positional `get_arg_val<uint32_t>(N)` style.

On the kernel side, Metal 2.0 adds a single new include, `experimental/kernel_args.h`, which pulls in the accessor templates (`get_arg`, `args::`, `dfb::`, `sem::`, `tensor::`). The `args::` / `dfb::` / `sem::` / `tensor::` namespaces are generated per-kernel from the host-side declarations and auto-included by the build system before the kernel source.

Suppose the host declared:

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
    uint32_t start_page  = get_arg_val<uint32_t>(0);         // RTA index 0
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
    auto start_page           = get_arg(args::start_page);  // RTA
    const auto num_entries    = get_arg(args::num_entries); // CRTA
    // ...
}
```

CTAs, RTAs, and CRTAs are all accessed through the same `get_arg(args::name)` mechanism. The dispatch type (compile-time, runtime, common runtime) is a host-only concept — the kernel author no longer needs to know which kind of argument they're reading. The C++ type-deduction context (`constexpr auto`, `auto`, `const auto`) is the only place the distinction surfaces.

> **Note**: This argument-retrieval syntax will evolve again to support custom argument types beyond `uint32_t` (including `std::array` and user-defined POD types). The named-accessor mechanism (`get_arg(args::name)`) will be preserved; the underlying types will become user-extensible.

---

## Design Principles

The sections above describe the Metal 2.0 API surface piece by piece. This section steps back to the design intent that ties those pieces together. It helps you recognize when a port is *faithful* (reaps the ergonomic improvement Metal 2.0 promises) versus when it merely *translates syntax* (carrying forward workarounds that should evaporate).

### Principle 1: Immutable spec, mutable run-args

Metal 2.0 splits a Program's description into two parts:

- **`ProgramSpec`** — the immutable description (the "function signature and body"). Captures kernel sources, resource declarations, work-unit assignments, and argument *schemas*.
- **`ProgramRunArgs`** — the per-execution mutable values (the "call arguments"). Carries runtime argument values, tensor identities, and any other field that may legitimately differ between executions.

This split is the **only** structural divergence from legacy `ProgramDescriptor`; every other Metal 2.0 type maps roughly 1:1 with its `ProgramDescriptor` counterpart. It enables two performance optimizations the legacy API couldn't express: a `Program` built once can be executed many times (paying the build cost once), and on cache hit the framework can apply a partial update to just the mutable side (typically the tensor identities).

For the op author, the practical implication is that **the schema and the values are conceptually paired** even though they live in different structures. When you add a named RTA to a `KernelSpec`'s schema, you simultaneously add its per-node value to `ProgramRunArgs`. When you declare a `TensorParameter`, you simultaneously add its `TensorArgument`.

### Principle 2: First-class named resource bindings

This is the largest design shift in Metal 2.0 — and the principle responsible for the bulk of the stylistic improvement it offers.

**Resources (tensors, DFBs, semaphores) bind by name.** Each resource is declared once on the `ProgramSpec`, then bound to each kernel that uses it via a `TensorBinding` / `DFBBinding` / `SemaphoreBinding` on the `KernelSpec`. The kernel accesses the resource through an auto-generated handle (`tensor::name`, `dfb::name`, `sem::name`) without needing to know its underlying ID, address, or layout.

Legacy `ProgramDescriptor` had no equivalent abstraction — resources were referenced indirectly (a tensor via its buffer's base address in an RTA; a CB via a magic-number index; a semaphore via its integer ID). The named-binding design makes those indirections disappear:

```cpp
// Tensor — legacy: address as RTA, layout metadata via TensorAccessorArgs<N>() chains.
constexpr auto args = TensorAccessorArgs<N>();
uint32_t addr = get_arg_val<uint32_t>(0);
auto a = TensorAccessor(args, addr);
// Metal 2.0: one line.
auto a = TensorAccessor(tensor::input);

// DFB — legacy: magic-number CB index passed via CTA.
constexpr uint32_t cb_id = 0;
CircularBuffer cb(cb_id);
// Metal 2.0: named handle from the binding.
DataflowBuffer cb(dfb::my_dfb);

// Semaphore — legacy: ID forwarded as an RTA.
uint32_t sem_id = get_arg_val<uint32_t>(2);
// Metal 2.0: named handle from the binding; access via sem::done.
```

When a port carries any of these patterns forward verbatim — buffer-address RTAs, magic CB indices, semaphore-ID RTAs — it has translated syntax without internalizing the design shift. The code may run correctly, but it defeats the ergonomics improvement Metal 2.0 is built to deliver.

The named-binding design also makes **resource identity singular** at the program level: where a legacy op passed the same tensor's address as an independent RTA to every kernel (and divergence between those was a silent failure mode), Metal 2.0 has one `TensorParameter` with multiple `TensorBinding`s, each with its own local `accessor_name`.

Resources can also be bound *conditionally* (a binding omitted from a kernel when a code path isn't taken).

### Principle 3: Named arguments throughout

Argument naming extends the named-binding ethos to scalar values:

- **Compile-time arguments are named only.** Positional CTAs are gone from the Metal 2.0 API.
- **Runtime and common runtime arguments are typically named.** A `varargs` mechanism exists as a niche escape hatch for dynamic-count argument tails read in a loop.

The `args::` namespace generated from a `KernelSpec`'s schema gives the kernel one uniform retrieval API: `get_arg(args::name)`. The CTA/RTA/CRTA distinction is a host-only concern.

Legacy `ProgramDescriptor` offered a hybrid: positional `compile_time_args` plus named `named_compile_time_args`. Many recent ops (matmul among them) use the named half. In Metal 2.0 every CTA is named — legacy named CTAs carry over directly, and legacy positional CTAs each become a named one.

---

## TTNN Framework Integration

TTNN ops integrate with the Metal 2.0 host API through the `ttnn::device_operation` framework. The framework provides three factory *concepts*; an op's program factory satisfies whichever matches its construction model:

- **`ProgramFactoryConcept`** — the oldest; legacy `host_api.hpp` builder-style factories.
- **`ProgramDescriptorFactoryConcept`** — the intermediate; legacy `ProgramDescriptor`-based factories.
- **`MetalV2FactoryConcept`** — the Metal 2.0 concept. **This is the concept ops port to.** It returns a `ttnn::device_operation::ProgramArtifacts` (the `ProgramSpec`, its `ProgramRunArgs`, and any op-owned tensors) from `create_program_artifacts()`.

A Metal 2.0 program factory has this shape:

```cpp
ttnn::device_operation::ProgramArtifacts MyProgramFactory::create_program_artifacts(
    const operation_attributes_t& attributes,
    const tensor_args_t&           tensor_args,
    tensor_return_value_t&         tensor_return_value) {

    // Extract MeshTensor references from inputs at the top:
    const auto& input  = tensor_args.input.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();

    tt::tt_metal::experimental::ProgramSpec spec{ .name = "my_program", /* ... */ };
    tt::tt_metal::experimental::ProgramRunArgs run_params;
    // ... populate spec and run_params ...

    return ttnn::device_operation::ProgramArtifacts{
        .spec       = std::move(spec),
        .run_params = std::move(run_params),
        // .op_owned_tensors = {},  // default-empty; populate only for
        //                          // config / index-table / workspace tensors.
    };
}
```

A Metal 2.0 factory **does not** construct the `Program` itself, nor call `SetProgramRunArgs` directly — those are the framework adapter's responsibilities. On a cache miss the framework builds the `Program` from the spec and applies the run args; on a cache hit it refreshes only the tensor bindings and skips the rebuild. A practical consequence of TTNN's caching: most parameters that *could* vary between calls are treated as immutable, and the cache is invalidated when they change; the actually-mutable surface is usually just the tensor identities.

> **A note on tensor types.** A factory sits between two tensor-type universes: **`ttnn::Tensor`** (TTNN's PyTorch-like wrapper; always device-resident inside a factory) and **`tt::tt_metal::MeshTensor`** (Metalium's native device-tensor type, which the Metal 2.0 API speaks). A `ttnn::Tensor`'s underlying `MeshTensor` is obtained via `.mesh_tensor()` (as in the skeleton above); the Metal 2.0 API operates on the `MeshTensor`.

---

## Complete Migration Example

Reader kernel reads pages from an input tensor into a DFB; writer kernel pulls from the DFB and writes pages to an output tensor. Single core.

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
        .accessor_name = "input",   // kernel accesses as `tensor::input`
    }},
    .hw_config = CreateReaderGen1DataMovementConfig(),
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
        .accessor_name = "output",  // kernel accesses as `tensor::output`
    }},
    .hw_config = CreateWriterGen1DataMovementConfig(),
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

`ProgramDescriptor` collapses placement, schema, and values into one nested struct; Metal 2.0 keeps each concern visible as its own field. This example demonstrated named CTAs, DFB binding-by-name, derived DFB placement, TensorAccessor binding (replacing the manual `TensorAccessorArgs` plumbing on the host and the offset bookkeeping in the kernel), and the mutable / immutable separation.
