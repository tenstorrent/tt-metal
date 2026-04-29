# Metal 2.0 Migration Guide — From the Temporary Quasar APIs

This guide is for Quasar developers migrating from the temporary placeholder APIs in `tt_metal/api/tt-metalium/experimental/host_api.hpp` and `tt_metal/api/tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp` to the new Metal 2.0 host APIs in `tt_metal/api/tt-metalium/experimental/metal2_host_api/`.

> **Audience**: Internal-only. The temporary Quasar APIs were always meant to be short-lived.

> **Note**: The Metal 2.0 APIs are experimental and subject to change.

The temporary Quasar APIs (`experimental::quasar::CreateKernel`, `experimental::dfb::CreateDataflowBuffer`) will soon be moved out of the public API surface — they will remain accessible to tests, but customer-facing code (i.e. IP customer deliverables) should use the Metal 2.0 APIs covered in this guide. Metal 2.0 has feature parity with the temporary Quasar APIs.

> **Prerequisites**: Before moving to Metal 2.0, ensure your device DM code is Device 2.0 compliant [Device 2.0 Data Movement migration](../kernel_apis/data_movement/device_api_migration_guide.md).

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
6. [Complete Migration Example](#complete-migration-example)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The temporary Quasar host APIs were meant as a placeholder to unblock early Quasar bring-up. The Metal 2.0 host APIs are the user-facing replacement.

The temporary Quasar APIs are imperative builders based on the legacy `host_api.hpp`:
 - (custom) `experimental::quasar::CreateKernel`
 - (custom) `experimental::dfb::CreateDataflowBuffer`
 - standard `host_api.hpp` calls
     - `CreateProgram`
     - `CreateSemaphore`
     - `SetRuntimeArgs`, ...)

Metal 2.0 is a descriptor-based API: you populate a `ProgramSpec` data structure describing the entire program, then call `MakeProgramFromSpec(spec)` to build it.

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

`ProgramSpec` replaces the legacy "build a `Program` imperatively, then enqueue it" pattern. The entire program — kernels, DFBs, semaphores, work units — is described as data first, then realized with `MakeProgramFromSpec`. This unlocks legality checks at construction time (no more "discover the bug at enqueue") and matches the descriptor-based caching strategy used elsewhere in the codebase.

**Legacy (imperative builder):**

```cpp
Program program = CreateProgram();

KernelHandle reader_h = experimental::quasar::CreateKernel(
    program, "kernels/reader.cpp", core_spec, dm_config);
KernelHandle writer_h = experimental::quasar::CreateKernel(
    program, "kernels/writer.cpp", core_spec, dm_config);

uint32_t dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_spec, dfb_config);
experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
    program, dfb_id, reader_h, writer_h);

CreateSemaphore(program, core_spec, /*initial_value=*/0);

SetRuntimeArgs(program, reader_h, core, {src_addr, num_pages});
SetRuntimeArgs(program, writer_h, core, {dst_addr, num_pages});

EnqueueProgram(cq, program, /*blocking=*/false);
```

**Metal 2.0 (declarative):**

```cpp
ProgramSpec spec{
    .program_id = "my_program",
    .kernels = {reader, writer},
    .dataflow_buffers = {dfb},
    .semaphores = {sem},
    .work_units = {main_work_unit},
};
Program program = MakeProgramFromSpec(spec);  // temporary free function

ProgramRunParams params;
params.kernel_run_params = { /* per-execution argument values */ };
SetProgramRunParameters(program, params);  // temporary free function

EnqueueProgram(cq, program, /*blocking=*/false);
```

`MakeProgramFromSpec` performs all legality checks and JIT compilation up front; if anything is wrong, it throws. Kernel handles are gone — kernels are referenced everywhere (work-unit membership, run params, DFB bindings) by their `unique_id` string.

---

### KernelSpec

The Quasar `experimental::quasar::CreateKernel` family is replaced by populating `KernelSpec`s and adding them to `ProgramSpec::kernels`. This is the largest shape change in Metal 2.0: kernels are no longer added to a `Program` imperatively — they are declared as data and consumed at the end by `MakeProgramFromSpec`.

**Legacy (`experimental::quasar::CreateKernel`):**

```cpp
Program program;

QuasarDataMovementConfig dm_config{
    .num_threads_per_cluster = 8,
    .compile_args = {src_cb_idx, dst_cb_idx, page_size},
};
KernelHandle reader_handle = experimental::quasar::CreateKernel(
    program,
    "kernels/reader.cpp",
    CoreCoord{0, 0},
    dm_config);

SetRuntimeArgs(program, reader_handle, CoreCoord{0, 0}, {src_addr, dst_addr});
```

**Metal 2.0 (`KernelSpec` and the surrounding `ProgramSpec` / `ProgramRunParams`):**

```cpp
// ----- KernelSpec: kernel declaration -----
constexpr const char* READER = "reader";

KernelSpec reader{
    .unique_id = READER,
    .source = KernelSpec::SourceFilePath{"kernels/reader.cpp"},
    .num_threads = 6,  // Metal 2.0 reserves DM0/DM1; max user threads = 6.
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
        .gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{},
    },
};

// ----- WorkUnitSpec: where the kernel runs -----
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
Program program = MakeProgramFromSpec(spec);  // temporary free function

// ----- ProgramRunParams: argument values, set per execution -----
ProgramRunParams params;
params.kernel_run_params = {{
    .kernel_spec_name = READER,
    .named_runtime_args = {{NodeCoord{0, 0},
        {{"src_addr", src_addr}, {"dst_addr", dst_addr}}}},
}};
SetProgramRunParameters(program, params);  // temporary free function
```

Reuse a single `constexpr const char*` for `unique_id` everywhere the kernel is referenced (`WorkUnitSpec`, `ProgramRunParams`, DFB and semaphore bindings) — this catches typos at compile time.

> **Tripwire — DM thread cap**: `QuasarDataMovementConfig::num_threads_per_cluster` allowed up to 8 DM threads. Metal 2.0 reserves DM0 and DM1 for runtime use, so `KernelSpec::num_threads` is capped at 6 for DM kernels. Specifying 7 or 8 will fail validation.

Full API surface (varargs, per-node thread overrides, compiler options): `tt_metal/api/tt-metalium/experimental/metal2_host_api/kernel_spec.hpp`.

---

### DataflowBufferSpec

`DataflowBufferSpec` replaces the two-step Quasar pattern — `experimental::dfb::CreateDataflowBuffer` followed by `BindDataflowBufferToProducerConsumerKernels`. Two shape changes:

1. **Placement is derived, not specified.** The DFB lives wherever its bound producer / consumer kernels run; you do not pass a `core_spec` to the DFB. The runtime now infers everything that the legacy bind-after-create call had to be told (which RISCs, how many threads, etc.). Local DFB invariant: producer and consumer kernels must share *identical* `WorkUnitSpec` membership.
2. **Endpoints are bound at the kernel spec.** Each producer / consumer kernel declares a `DFBBinding` on its `KernelSpec`, naming the DFB and a local accessor name. The kernel code references the DFB through that local accessor name (e.g. `dfb::my_dfb`).

**Legacy (`experimental::dfb`):**

```cpp
experimental::dfb::DataflowBufferConfig dfb_config{
    .entry_size = page_size,
    .num_entries = num_pages,
    .producer_risc_mask = 0x01,  // DM0
    .consumer_risc_mask = 0x02,  // DM1
    .pap = experimental::dfb::AccessPattern::STRIDED,
    .cap = experimental::dfb::AccessPattern::STRIDED,
    .data_format = tt::DataFormat::Float16_b,
};
uint32_t dfb_id = experimental::dfb::CreateDataflowBuffer(program, core_spec, dfb_config);
experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
    program, dfb_id, producer_handle, consumer_handle);
```

**Metal 2.0 (`DataflowBufferSpec`):**

```cpp
constexpr const char* MY_DFB = "my_dfb";

DataflowBufferSpec dfb{
    .unique_id = MY_DFB,
    .entry_size = page_size,
    .num_entries = num_pages,
    .data_format_metadata = tt::DataFormat::Float16_b,
    // No core_spec, no producer/consumer RISC masks — derived from kernel bindings.
};

// Bound on each kernel that uses the DFB:
KernelSpec producer{ /* ... */
    .dfb_bindings = {{
        .dfb_spec_name = MY_DFB,
        .local_accessor_name = "out_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER,
        .access_pattern = DFBAccessPattern::STRIDED,
    }},
};
KernelSpec consumer{ /* ... */
    .dfb_bindings = {{
        .dfb_spec_name = MY_DFB,
        .local_accessor_name = "in_dfb",  // independent of the producer's name
        .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER,
        .access_pattern = DFBAccessPattern::STRIDED,
    }},
};
```

Each `local_accessor_name` is independent per binding; the producer and consumer can — and often will — name the same DFB differently in their respective kernel sources. The producer/consumer RISC masks that the legacy `DataflowBufferConfig` carried are no longer needed: the runtime determines RISC assignment from the kernel bindings.

Advanced cases (aliased DFBs, borrowed-memory DFBs built on a `Buffer` or `MeshTensor` view, remote DFBs spanning nodes) are described in `dataflow_buffer_spec.hpp`. Note: `RemoteDataflowBufferSpec` is exposed in the API surface but the runtime does not yet support it — using one will trigger a `TT_FATAL` at `MakeProgramFromSpec`.

---

### SemaphoreSpec

`SemaphoreSpec` replaces `CreateSemaphore`. Two shape changes:

1. **Semaphores are declared as data**, like everything else in the new API — no longer minted by an imperative `CreateSemaphore` call into a Program object.
2. **Semaphore IDs no longer travel as runtime arguments.** Each kernel that uses a semaphore declares a `SemaphoreBinding` naming it and giving it a local accessor name; kernel code accesses it as `sem::accessor_name`.

Unlike DFBs, semaphore placement is *not* derived: a semaphore is a remote resource accessible from any kernel, so its node set is specified directly via `target_nodes`.

**Legacy (`CreateSemaphore`):**

```cpp
uint32_t sem_id = CreateSemaphore(program, core_spec, /*initial_value=*/0);

// Pass the ID to kernels that need it, as a runtime arg:
SetRuntimeArgs(program, reader_h, core, {/* ... */, sem_id});
```

**Metal 2.0 (`SemaphoreSpec`):**

```cpp
constexpr const char* DONE = "done";

SemaphoreSpec done_sem{
    .unique_id = DONE,
    .target_nodes = NodeRange{{0,0}, {3,3}},
};

// Bound on each kernel that uses the semaphore:
KernelSpec writer{ /* ... */
    .semaphore_bindings = {{
        .semaphore_spec_name = DONE,
        .accessor_name = "done",  // kernel code accesses as `sem::done`
    }},
};
```

Notes:

- `target_nodes` is `std::variant<NodeCoord, NodeRange, NodeRangeSet>` — pass any of the three. Any kernel in the `ProgramSpec` can bind to the semaphore regardless of where the kernel runs.
- `SemaphoreSpec::SemaphoreMemoryType::Register` is a placeholder for a Gen2 hardware feature (register-backed semaphores); it is not yet supported. Only `L1` works today. The runtime team intends to automate the L1 / Register selection eventually.
- Setting `initial_value` to a non-zero value is not supported on Gen2; the runtime team intends to deprecate non-zero initial values entirely once remote DFBs land.

---

### WorkUnitSpec

`WorkUnitSpec` is a new top-level concept that replaces the `core_spec` argument from `experimental::quasar::CreateKernel` (and from `experimental::dfb::CreateDataflowBuffer`). It declares groups of kernels that operate together on a worker node, and on which nodes they run.

A `WorkUnitSpec` is `{unique_id, kernels, target_nodes}` — that is all.

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

**A kernel belonging to multiple work units** — for example, a compute kernel that runs on both an inner block (paired with one reader/writer) and a halo region (paired with a different reader/writer):

```cpp
WorkUnitSpec wu_inner{
    .unique_id = "inner",
    .kernels = {COMPUTE, INNER_READER},
    .target_nodes = NodeRange{{1, 1}, {2, 2}},
};
WorkUnitSpec wu_halo{
    .unique_id = "halo",
    .kernels = {COMPUTE, HALO_READER},
    .target_nodes = halo_node_range_set,
};
// COMPUTE's effective node set is the union: inner ∪ halo.
```

Notes:

- `target_nodes` is `std::variant<NodeCoord, NodeRange, NodeRangeSet>` — pass any of the three.
- A kernel may belong to multiple `WorkUnitSpec`s; its effective node set is the union of those work units' `target_nodes`.
- Every kernel referenced in `ProgramSpec::kernels` must be referenced by at least one `WorkUnitSpec`. Otherwise it has no place to run.
- **Local DFB invariant**: the producer and consumer kernels of a local `DataflowBufferSpec` must share *identical* `WorkUnitSpec` membership. If they differ, the runtime cannot determine where to allocate the DFB.

---

### ProgramRunParams

`ProgramRunParams` replaces the imperative `SetRuntimeArgs(program, kernel_handle, core, args)` family. The kernel declares its runtime-arg *schema* (names, count of varargs) on `KernelSpec::runtime_arguments_schema`; values are supplied per execution via `ProgramRunParams`.

The execution flow:

1. Build the immutable `ProgramSpec` once.
2. `Program program = MakeProgramFromSpec(spec);` — JIT compile and structural setup.
3. For each execution: populate a fresh `ProgramRunParams`, call `SetProgramRunParameters(program, params)`, then `EnqueueProgram`.

**Legacy (imperative `SetRuntimeArgs`):**

```cpp
SetRuntimeArgs(program, reader_h, core, {src_addr, num_pages, sem_id});
SetCommonRuntimeArgs(program, reader_h, {bank_id});
```

**Metal 2.0 (schema on the spec, values on the run params):**

```cpp
// Schema declared on the kernel:
KernelSpec reader{
    .unique_id = READER,
    // ...
    .runtime_arguments_schema = {
        .named_runtime_args = {"src_addr", "num_pages"},
        .named_common_runtime_args = {"bank_id"},
    },
    // (sem_id no longer travels as a runtime arg — bind the semaphore instead.)
};

// Values supplied per execution:
ProgramRunParams params;
params.kernel_run_params = {{
    .kernel_spec_name = READER,
    .named_runtime_args = {{NodeCoord{0, 0},
        {{"src_addr", src_addr}, {"num_pages", num_pages}}}},
    .named_common_runtime_args = {{"bank_id", bank_id}},
}};
SetProgramRunParameters(program, params);  // temporary free function
```

Vararg form (positional, dynamic count):

```cpp
.runtime_arguments_schema = {
    .num_runtime_varargs = 3,         // 3 RTA varargs (per node)
    .num_common_runtime_varargs = 1,  // 1 CRTA vararg (broadcast)
},

// Values:
.runtime_varargs = {{NodeCoord{0, 0}, {dim0, dim1, dim2}}},
.common_runtime_varargs = {flag},
```

Named and vararg forms can coexist on the same kernel. Vararg indices are stable across schema changes.

`ProgramRunParams` must be specified for every kernel that has runtime arguments, on every node where the kernel runs. Kernel handles are gone — kernels are referenced by `unique_id` string (`kernel_spec_name`).

---

## Device-Side Migration

### Circular Buffers → Dataflow Buffers

Metal 2.0 replaces Circular Buffers (CBs) with Dataflow Buffers (DFBs) on both the host and device sides. The kernel-side DFB FIFO API is shape-compatible with the Device 2.0 `experimental::CircularBuffer` wrapper: same method names (`reserve_back` / `push_back` / `wait_front` / `pop_front` / `get_write_ptr` / `get_read_ptr`).

The change at the call site is twofold:

1. **The class is `experimental::DataflowBuffer`** (instead of `experimental::CircularBuffer`).
2. **The constructor takes a named binding accessor**, not a magic-number `cb_id`. The accessor is auto-generated from the host-side DFB binding into the `dfb::` namespace in `kernel_bindings_generated.h`.

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

> **Quasar bonus — implicit sync**: On Quasar, you can elide the explicit `reserve_back` / `push_back` (or `wait_front` / `pop_front`) pattern entirely. Pass the `DataflowBuffer` directly to `experimental::Noc::async_read` or `async_write`, and the runtime hardware handles the FIFO sync via ISR. This is the default behavior; you can disable it per-DFB by setting `DataflowBufferSpec::disable_implicit_sync = true` (rarely needed).

*[Almeet to fill in: complete method-by-method mapping including `pages_reservable_at_back` / `pages_available_at_front` / `finish` / `write_barrier`; DFB ↔ `Noc` integration examples; an end-to-end implicit-sync example.]*

---

### Kernel Argument Retrieval Syntax

The Metal 2.0 host API declares kernel arguments by name; the kernel-side API retrieves them by name. This replaces the legacy positional `get_arg_val<uint32_t>(N)` style.

> **Note**: This syntax is expected to evolve again to support custom argument types beyond `uint32_t`. The named-accessor mechanism will be preserved; the underlying argument types will become user-extensible.

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

Vararg indices are stable across schema changes: if you later promote a named RTA to a CRTA, or add or remove named arguments, the existing `get_vararg(N)` / `get_common_vararg(N)` calls still resolve to the same vararg slots.

> **Note**: This argument-retrieval syntax will evolve again to support custom argument types beyond `uint32_t` (including user-defined POD types). The named-accessor mechanism (`get_arg(args::name)`) will be preserved; the underlying types will become user-extensible. No-one will be unhappy about it — every team has been asking for this.

---

## Complete Migration Example

Reader kernel reads from DRAM into a DFB; writer kernel pulls from the DFB and writes to DRAM. Single-threaded kernels on a single Quasar cluster, no semaphores. End-to-end host code.

**Legacy (temporary Quasar APIs + standard `host_api.hpp`):**

```cpp
constexpr uint32_t page_size = 1024;
constexpr uint32_t num_pages = 8;
const CoreCoord cluster{0, 0};

Program program = CreateProgram();

// Reader kernel
experimental::quasar::QuasarDataMovementConfig reader_config{
    .num_threads_per_cluster = 1,
    .compile_args = {page_size},
};
KernelHandle reader_h = experimental::quasar::CreateKernel(
    program, "kernels/reader.cpp", cluster, reader_config);

// Writer kernel
experimental::quasar::QuasarDataMovementConfig writer_config{
    .num_threads_per_cluster = 1,
    .compile_args = {page_size},
};
KernelHandle writer_h = experimental::quasar::CreateKernel(
    program, "kernels/writer.cpp", cluster, writer_config);

// DFB
experimental::dfb::DataflowBufferConfig dfb_config{
    .entry_size = page_size,
    .num_entries = num_pages,
    .producer_risc_mask = 0x01,  // DM0
    .consumer_risc_mask = 0x02,  // DM1
    .pap = experimental::dfb::AccessPattern::STRIDED,
    .cap = experimental::dfb::AccessPattern::STRIDED,
    .data_format = tt::DataFormat::Float16_b,
};
uint32_t dfb_id = experimental::dfb::CreateDataflowBuffer(program, cluster, dfb_config);
experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
    program, dfb_id, reader_h, writer_h);

// Runtime args
SetRuntimeArgs(program, reader_h, cluster, {src_addr, num_pages});
SetRuntimeArgs(program, writer_h, cluster, {dst_addr, num_pages});

EnqueueProgram(cq, program, /*blocking=*/false);
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
    .num_threads = 1,
    .compile_time_arg_bindings = {{"page_size", page_size}},
    .runtime_arguments_schema = {.named_runtime_args = {"src_addr", "num_pages"}},
    .dfb_bindings = {{
        .dfb_spec_name = DFB,
        .local_accessor_name = "out_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER,
        .access_pattern = DFBAccessPattern::STRIDED,
    }},
    .config_spec = DataMovementConfiguration{
        .gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{},
    },
};

KernelSpec writer{
    .unique_id = WRITER,
    .source = KernelSpec::SourceFilePath{"kernels/writer.cpp"},
    .num_threads = 1,
    .compile_time_arg_bindings = {{"page_size", page_size}},
    .runtime_arguments_schema = {.named_runtime_args = {"dst_addr", "num_pages"}},
    .dfb_bindings = {{
        .dfb_spec_name = DFB,
        .local_accessor_name = "in_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER,
        .access_pattern = DFBAccessPattern::STRIDED,
    }},
    .config_spec = DataMovementConfiguration{
        .gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{},
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

EnqueueProgram(cq, program, /*blocking=*/false);
```

The Metal 2.0 form is longer, but the additional structure is information that previously had to be threaded through several imperative calls (and several arguments to those calls). DFB endpoint information, in particular, was scattered across `DataflowBufferConfig` (RISC masks, access patterns) and the subsequent `BindDataflowBufferToProducerConsumerKernels` call (kernel handles); in Metal 2.0 it lives entirely on each `KernelSpec::dfb_bindings`. RISC assignment is the runtime's job again.

---

## Troubleshooting

Common pitfalls when migrating from the temporary Quasar APIs:

- **Don't pass core specs to DFBs.** `DataflowBufferSpec` has no `target_nodes` field by design — placement is derived from kernel DFB bindings. The legacy two-step "create the DFB, then bind it" pattern collapses into one declarative form.
- **Quasar kernel handles are gone.** Kernels are referenced by `unique_id` string everywhere (work-unit membership, run params, DFB / semaphore bindings). Define each `unique_id` as a `constexpr const char*` and reuse the constant — catches typos at compile time.
- **All CTAs are named.** The Quasar `compile_args` (positional) / `named_compile_args` (named) split is gone; populate `KernelSpec::compile_time_arg_bindings` instead. If you had a positional `compile_args` vector, give every entry a name.
- **Producer / consumer RISC masks are not specified.** The legacy `DataflowBufferConfig::producer_risc_mask` and `consumer_risc_mask` are gone — the runtime determines RISC assignment from the kernel bindings. Don't try to reconstitute these in Metal 2.0.
- **DM threads per kernel are capped at 6.** DM0 and DM1 are reserved for runtime use. The temporary Quasar API allowed up to 8; Metal 2.0 caps at 6 and validates. (See the `KernelSpec` section's tripwire callout.)
- **Every kernel must belong to a `WorkUnitSpec`.** A kernel listed in `ProgramSpec::kernels` but not referenced by any `WorkUnitSpec::kernels` has no place to run; `MakeProgramFromSpec` will reject it.
- **A kernel may belong to multiple `WorkUnitSpec`s.** Its effective node set is the union of those work units' `target_nodes`. Use this for kernels that participate in multiple roles.
- **Local DFB invariant.** Producer and consumer kernels of a local DFB must share *identical* `WorkUnitSpec` membership. Not "compatible" or "overlapping" — identical.
- **Semaphore IDs are no longer runtime args.** Bind the semaphore at the kernel spec instead; access it as `sem::accessor_name` in kernel code. If your migration leaves a `sem_id` runtime arg behind, you've missed a binding.
- **Every named RTA must be set on every node.** `ProgramRunParams::KernelRunParams::named_runtime_args` is per-node; missing an entry for a node where the kernel runs causes `SetProgramRunParameters` to error. Same for varargs.
- **Remote DFBs are not yet supported by the runtime.** `RemoteDataflowBufferSpec` is exposed in the API surface as a breadcrumb, but using one triggers a `TT_FATAL` at `MakeProgramFromSpec`.
- **`MakeProgramFromSpec` and `SetProgramRunParameters` are temporary free functions.** They will become a `Program` constructor and member function respectively — call sites will need a one-line update later.

### Kernel globals

Metal 2.0 has no equivalent of the temporary Quasar API's `QuasarDataMovementConfig::is_legacy_kernel` flag.

On WH/BH, each DM core has its own private memory and gets its own copy of the kernel binary, so `.data` / `.bss` are effectively per-core for free. On Quasar, the 8 DM cores in a cluster share L1, and the natural model is one shared binary with multiple threads — globals live in one place and are *shared* across all DM threads. Setting `is_legacy_kernel = true` recovered WH/BH semantics by duplicating the kernel binary per DM core, at the cost of L1 footprint.

Metal 2.0 always uses the shared-binary model. If you depended on `is_legacy_kernel = true`, the kernel must be rewritten — use `thread_local` for variables that genuinely need per-thread state.

The flag's implicit premise was that you would want to thread an existing WH/BH kernel as a first porting step. That premise is no longer correct: the recommended quick-port path from WH/BH to Quasar is to keep kernels **single-threaded**. (A Quasar Neo Cluster is not four BH workers in a trenchcoat; and Metal 2.0's 6-DM-thread cap precludes the brute-thread approach in any case.) Single-threaded kernels are unaffected by this behavior change.
