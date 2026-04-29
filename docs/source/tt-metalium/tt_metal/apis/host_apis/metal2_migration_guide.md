# Metal 2.0 Migration Guide (WH/BH)

This guide helps developers migrate from the legacy `ProgramDescriptor` / `host_api.hpp` APIs to the new  Metal 2.0 host APIs in `tt_metal/api/tt-metalium/experimental/metal2_host_api/`.

The focus of this guide is **Wormhole** and **Blackhole** (Gen1 architectures).

Things to remember:
 - The Metal 2.0 APIs are experimental and subject to changes.
 - Metal 2.0 is a work in progress; not all legacy API features fully implemented.
 - Not all planned improvements are available yet.


> **Prerequisite**: Before attempting Metal 2.0 migration, complete the [Device 2.0 Data Movement migration](../kernel_apis/data_movement/device_api_migration_guide.md). All kernel code migration examples in this guide assume Device 2.0 migration is already done.

## Table of Contents

1. [Overview](#overview)
2. [Concept Map](#concept-map)
3. [Header Files](#header-files)
4. [Host API Migration](#host-api-migration)
   - [ProgramSpec](#programspec)
   - [KernelSpec](#kernelspec)
   - [DataflowBufferSpec](#dataflowbufferspec)
   - [SemaphoreSpec](#semaphorespec)
   - [WorkUnitSpec](#workunitspec)
   - [ProgramRunParams](#programrunparams)
5. [Device-Side Migration](#device-side-migration)
   - [Circular Buffers → Dataflow Buffers](#circular-buffers--dataflow-buffers)
   - [Kernel Argument Retrieval Syntax](#kernel-argument-retrieval-syntax)
6. [Complete Migration Examples](#complete-migration-examples)
7. [Troubleshooting](#troubleshooting)

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
- **Named arguments**. Compile-time, runtime, and common runtime arguments are addressed by name on both the host and device sides

Many additional improvements are planned, but are not yet available in the experimental APIs.

---

## Concept Map

| ProgramDescriptor (legacy) | Metal 2.0 |
|---|---|
| `ProgramDescriptor` | `ProgramSpec` (immutable description) + `ProgramRunParams` (per-execution values) |
| `KernelDescriptor` | `KernelSpec` |
| `KernelDescriptor::core_ranges` | derived from `WorkUnitSpec::target_nodes` |
| `KernelDescriptor::compile_time_args` (positional) | `KernelSpec::compile_time_arg_bindings` (named *only*) |
| `KernelDescriptor::runtime_args` / `common_runtime_args` | **Schema** (names): declared on `KernelSpec::runtime_arguments_schema`<br>**Values**: supplied per execution on `ProgramRunParams::KernelRunParams` |
| `CBDescriptor` | `DataflowBufferSpec` (placement derived from kernel bindings) |
| `SemaphoreDescriptor` | `SemaphoreSpec` |
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

**These header files are self-documenting, with extensive comments.** Please read them.

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
3. **Resource bindings.** New syntax to bind DFB endpoints and semaphores to the kernel, and retrieve them by name in device code.
4. **Tensor bindings.** Coming soon.


**Legacy** (`KernelDescriptor`):

```cpp
KernelDescriptor reader = {
    .kernel_source = "kernels/reader.cpp",
    .core_ranges = CoreRangeSet{CoreRange{{0,0}, {0,0}}},
    .compile_time_args = {src_cb_idx, dst_cb_idx, page_size},
    .runtime_args = {{{0,0}, {src_addr, dst_addr}}},
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
        .named_runtime_args = {"src_addr", "dst_addr"},
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
        {{"src_addr", src_addr}, {"dst_addr", dst_addr}}}},
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
 - (optional) DFB size + entry size (not yet supported)
 - (optional) DFB borrowed memory (not yet supported)

All kernel arguments are named arguments in Metal 2.0. For kernels that accept a variable number of arguments, the API additionally provides an  "varargs" mechanism.

**Legacy** (values on the descriptor):

```cpp
KernelDescriptor reader = {
    // ...
    .runtime_args = {{{0, 0}, {src_addr, num_tiles}}},
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
        .named_runtime_args = {"src_addr", "num_tiles"},
        .named_common_runtime_args = {"bank_id"},
    },
};

// Values supplied per execution:
ProgramRunParams params;
params.kernel_run_params = {{
    .kernel_spec_name = READER,
    .named_runtime_args = {{NodeCoord{0, 0},
        {{"src_addr", src_addr}, {"num_tiles", num_tiles}}}},
    .named_common_runtime_args = {{"bank_id", bank_id}},
}};
SetProgramRunParameters(program, params); // temporary free function
```

Vararg form (positional, dynamic count):

```cpp
// Schema:
.runtime_arguments_schema = {
    .num_runtime_varargs = 3,         // 3 RTA varargs (per node)
    .num_common_runtime_varargs = 1,  // 1 CRTA vararg (broadcast)
},

// Values:
.runtime_varargs = {{NodeCoord{0, 0}, {dim0, dim1, dim2}}},
.common_runtime_varargs = {flag},
```

Named and vararg forms can coexist on the same kernel. Vararg indices are stable across schema changes — promoting a named RTA to a CRTA, or adding/removing named args, does not shift vararg indices.

---

## Device-Side Migration

### Circular Buffers → Dataflow Buffers

Metal 2.0 replaces Circular Buffers (CBs) with Dataflow Buffers (DFBs). Notes:

1. **Construction from local accessor name**. You construct your DFB from its local accessor, not a magic-number `cb_id`. The accessor is auto-generated from the host-side DFB binding and lives in the `dfb::` namespace inside `kernel_bindings_generated.h`.
2. **`experimental::CircularBuffer` compatible APIs**. The kernel-side DFB APIs are drop-in-compatible with the Device 2.0 `experimental::CircularBuffer` wrapper methods and semantics on Gen1. Gen2 permits more advanced capabilities.

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

### Kernel Argument Retrieval Syntax

The Metal 2.0 host API declares kernel arguments by name; the kernel-side API retrieves them by name. This replaces the legacy positional `get_arg_val<uint32_t>(N)` style.

Include `experimental/kernel_args.h` in any kernel that uses the named-argument API. The `args::` and `dfb::` / `sem::` namespaces are auto-generated from the host-side bindings into `kernel_bindings_generated.h`, which the kernel build system pulls in.

**Example 1 — Named arguments only.**

Suppose the host declared the following on `KernelSpec`:

```cpp
.compile_time_arg_bindings = {{"bank_id", 0}, {"entry_size", 1024}},
.runtime_arguments_schema = {
    .named_runtime_args = {"src_addr"},
    .named_common_runtime_args = {"num_entries"},
},
```

**Legacy (positional):**

```cpp
void kernel_main() {
    constexpr uint32_t bank_id    = get_compile_time_arg_val(0);
    constexpr uint32_t entry_size = get_compile_time_arg_val(1);
    uint32_t src_addr    = get_arg_val<uint32_t>(0);    // RTA index 0
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
    auto src_addr             = get_arg(args::src_addr);    // RTA
    const auto num_entries    = get_arg(args::num_entries); // CRTA
    // ...
}
```

CTAs, RTAs, and CRTAs are accessed through the same `get_arg(args::name)` mechanism in device code. The dispatch type (compile-time, runtime, common runtime) is a host-only concept — the kernel author no longer needs to know or care which kind of argument they're reading. The C++ type-deduction context (`constexpr auto`, `auto`, `const auto`) is the only place the distinction surfaces.

**Example 2 — Named arguments plus varargs.**

Some kernels need a variable-count argument tail (e.g. an N-dimensional tensor's shape, where N is itself a CTA). Metal 2.0 supports this as positional **varargs**, which coexist with named arguments:

```cpp
.runtime_arguments_schema = {
    .named_runtime_args = {"src_addr"},
    .named_common_runtime_args = {"num_entries"},
    .num_runtime_varargs = 3,         // 3 positional RTA varargs (per node)
    .num_common_runtime_varargs = 1,  // 1 positional CRTA vararg (broadcast)
},
```

```cpp
#include "experimental/kernel_args.h"

void kernel_main() {
    auto src_addr          = get_arg(args::src_addr);
    const auto num_entries = get_arg(args::num_entries);

    // Vararg RTAs (positional, indexed from 0):
    uint32_t dim0 = get_vararg(0);
    uint32_t dim1 = get_vararg(1);
    uint32_t dim2 = get_vararg(2);

    // Vararg CRTAs (broadcast across all nodes the kernel runs on):
    uint32_t flag = get_common_vararg(0);
    // ...
}
```

Vararg indices are stable across schema changes: if you later promote a named RTA to a CRTA, or add or remove named arguments, the existing `get_vararg(N)` / `get_common_vararg(N)` calls still resolve to the same vararg slots. Migrators porting kernels with a variable argument count can rely on this — leave existing positional access in place, layer named arguments on top.

> **Note**: This argument-retrieval syntax will evolve again to support custom argument types beyond `uint32_t` (including std::array and user-defined POD types). The named-accessor mechanism (`get_arg(args::name)`) will be preserved; the underlying types will become user-extensible. Migrators should plan to revisit kernel arguments after these changes.

---

## Complete Migration Examples

### Example 1: Single-Core Reader / Writer with One DFB

Reader kernel reads from DRAM into a DFB; writer kernel pulls from the DFB and writes to DRAM.

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

KernelDescriptor reader = {
    .kernel_source = "kernels/reader.cpp",
    .core_ranges = CoreRangeSet{CoreRange{core}},
    .compile_time_args = {cb_idx, page_size},
    .runtime_args = {{core, {src_addr, num_pages}}},
    .config = ReaderConfigDescriptor{},
};

KernelDescriptor writer = {
    .kernel_source = "kernels/writer.cpp",
    .core_ranges = CoreRangeSet{CoreRange{core}},
    .compile_time_args = {cb_idx, page_size},
    .runtime_args = {{core, {dst_addr, num_pages}}},
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

constexpr uint32_t page_size = 1024;
constexpr uint32_t num_pages = 8;
const NodeCoord node{0, 0};

KernelSpec reader{
    .unique_id = READER,
    .source = KernelSpec::SourceFilePath{"kernels/reader.cpp"},
    .compile_time_arg_bindings = {{"page_size", page_size}},
    .runtime_arguments_schema = {.named_runtime_args = {"src_addr", "num_pages"}},
    .dfb_bindings = {{
        .dfb_spec_name = DFB,
        .local_accessor_name = "out_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER
    }},
    .config_spec = DataMovementConfiguration{
        .gen1_data_movement_config = {.processor = DataMovementProcessor::RISCV_0},
    },
};

KernelSpec writer{
    .unique_id = WRITER,
    .source = KernelSpec::SourceFilePath{"kernels/writer.cpp"},
    .compile_time_arg_bindings = {{"page_size", page_size}},
    .runtime_arguments_schema = {.named_runtime_args = {"dst_addr", "num_pages"}},
    .dfb_bindings = {{
        .dfb_spec_name = DFB,
        .local_accessor_name = "in_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER
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
    .work_units = {{
        .unique_id = "main",
        .kernels = {READER, WRITER},
        .target_nodes = node,
    }},
};
Program program = MakeProgramFromSpec(spec);

ProgramRunParams params;
params.kernel_run_params = {
    {.kernel_spec_name = READER,
     .named_runtime_args = {{node, {{"src_addr", src_addr}, {"num_pages", num_pages}}}}},
    {.kernel_spec_name = WRITER,
     .named_runtime_args = {{node, {{"dst_addr", dst_addr}, {"num_pages", num_pages}}}}},
};
SetProgramRunParameters(program, params);
```

`ProgramDescriptor` collapses all of placement, schema, and values into one nested struct, while Metal 2.0 keeps each concern visible as its own field. This example demonstrated named CTAs, DFB binding-by-name, derived DFB placement, and mutable / immutable separation.

### Example 2: Multi-Core with Cross-Core Synchronization

Same reader/writer kernels as Example 1, but now running on a 2×2 grid with a semaphore-based handshake. Only the deltas are shown — fields that are unchanged from Example 1 are elided with `// (unchanged)`.

**Legacy (`ProgramDescriptor`):**

```cpp
const CoreRangeSet cores = CoreRangeSet{CoreRange{{0, 0}, {1, 1}}};

SemaphoreDescriptor done_sem{
    .id = 0,
    .core_type = CoreType::WORKER,
    .core_ranges = cores,
    .initial_value = 0,
};

KernelDescriptor reader = {
    // (unchanged)
    .core_ranges = cores,
    // Per-core runtime args; the semaphore ID is passed as a runtime arg:
    .runtime_args = {
        {{0,0}, {src_addr_0, num_pages, /*sem_id=*/0}},
        {{1,0}, {src_addr_1, num_pages, 0}},
        {{0,1}, {src_addr_2, num_pages, 0}},
        {{1,1}, {src_addr_3, num_pages, 0}},
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

SemaphoreSpec done_sem{
    .unique_id = DONE,
    .target_nodes = cores,
};

KernelSpec reader{
    // (unchanged)
    // No core_ranges — placement comes from WorkUnitSpec, below.
    // No semaphore in the args — bind it instead:
    .semaphore_bindings = {{
        .semaphore_spec_name = DONE,
        .accessor_name = "done",  // kernel accesses as `sem::done`
    }},
    // ...
};

ProgramSpec spec{
    .program_id = "loopback_2x2",
    .kernels = {reader, writer},
    .dataflow_buffers = {dfb},
    .semaphores = {done_sem},
    .work_units = {{
        .unique_id = "main",
        .kernels = {READER, WRITER},
        .target_nodes = cores,  // 2×2 grid
    }},
};
Program program = MakeProgramFromSpec(spec);

ProgramRunParams params;
// One named_runtime_args entry per node where the kernel runs:
params.kernel_run_params = {{
    .kernel_spec_name = READER,
    .named_runtime_args = {
        {{0,0}, {{"src_addr", src_addr_0}, {"num_pages", num_pages}}},
        {{1,0}, {{"src_addr", src_addr_1}, {"num_pages", num_pages}}},
        {{0,1}, {{"src_addr", src_addr_2}, {"num_pages", num_pages}}},
        {{1,1}, {{"src_addr", src_addr_3}, {"num_pages", num_pages}}},
    },
}, /* writer entry similarly */ };
SetProgramRunParameters(program, params);
```

Key differences vs. the legacy pattern:

- Multi-core placement moves from `core_ranges` on each kernel to a single `target_nodes` on the `WorkUnitSpec`.
- The semaphore is bound at the kernel spec; the semaphore ID no longer travels as a runtime argument. Kernel code accesses it as `sem::done`.
- Per-node runtime args are still per-node, but addressed by `NodeCoord` and named.

---

## Troubleshooting

Common pitfalls when migrating from `ProgramDescriptor`:

- **Every kernel must belong to a `WorkUnitSpec`.** A kernel listed in `ProgramSpec::kernels` but not referenced by any `WorkUnitSpec::kernels` has no place to run. This will trigger an error.
- **DFB placement is derived, not specified.** Don't pass a node range to `DataflowBufferSpec` — the DFB lives wherever its bound producer / consumer kernels run. `DataflowBufferSpec` has no `target_nodes` field by design.
- **Local DFB invariant.** A local DFB's producer and consumer kernels must share *identical* `WorkUnitSpec` membership.
- **Unnamed/positional compile-time arguments are not supported**. Use named compile time arguments.
- **Unnamed/position runtime arguments _are_ supported**. You can use varargs to restore the old behavior, but this is discouraged.
- **`ProgramRunParams` requires that every named RTA must be set on every node.** Missing an entry for a node where the kernel runs causes `SetProgramRunParameters` to error. The same applies to varargs. (Note: There is also a power-user `ProgramRunParamsView` API that provides a stateful view into the dispatch buffers; it is not yet supported.)
