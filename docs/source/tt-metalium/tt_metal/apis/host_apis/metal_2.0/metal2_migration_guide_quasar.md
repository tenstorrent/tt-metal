# Metal 2.0 Migration Guide — From the Temporary Quasar APIs

> ⚠️ **STALE — DO NOT READ.** This guide has not been updated for the Metal 2.0 API cleanups landed in PRs #45290 (structural), #45598 (naming), #45160 (disable_implicit_sync relocation), and several others. Field names, type names, namespace paths, and feature locations referenced below are out of date. A full rewrite is pending. If you need Quasar migration guidance in the meantime, ask Audrey directly.

This guide is for Quasar developers migrating from the temporary placeholder APIs in `tt_metal/api/tt-metalium/experimental/host_api.hpp` and `tt_metal/api/tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp` to the new Metal 2.0 host APIs in `tt_metal/api/tt-metalium/experimental/metal2_host_api/`. (For WH/BH migration to Metal 2.0, look [here instead](metal2_migration_guide.md).)

> **Audience**: Internal-only. The temporary Quasar APIs were always meant to be short-lived.

> **Note**: The Metal 2.0 APIs are experimental and subject to change.

The temporary Quasar APIs (`experimental::quasar::CreateKernel`, `experimental::dfb::CreateDataflowBuffer`) will soon be moved out of the public API surface — they will remain accessible to tests, but customer-facing code (i.e. IP customer deliverables) should use the Metal 2.0 APIs covered in this guide. Metal 2.0 has feature parity with the temporary Quasar APIs.

> **Prerequisites**: Before moving to Metal 2.0, ensure your device DM code is Device 2.0 compliant [Device 2.0 Data Movement migration](../../kernel_apis/data_movement/device_api_migration_guide.md).

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

Two style shifts to watch for during migration:

> **Stated goal: eliminate raw pointer arguments.** With DFBs, semaphores, and tensors all bindable as named resources, runtime args carrying a buffer or tensor address should now be the exception rather than the rule. If you're about to put `tensor.buffer()->address()` in a runtime arg, you're probably doing it wrong — bind the tensor as a `TensorParameter` instead.

> **Argument naming.** Metal 2.0 is designed around named arguments. Compile-time arguments must be named — positional CTAs are no longer part of the API. Runtime and common runtime arguments may be named (the typical case) or positional (varargs, intended for kernels with a genuinely dynamic argument count consumed in a loop — e.g., an N-dimensional shape where N is a CTA). When porting from a legacy kernel, individually-known RTAs translate naturally to named RTAs; reach for varargs only when the kernel actually loops over the arguments.

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
See `program_spec.hpp`.

**Legacy:**

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

SetRuntimeArgs(program, reader_h, core, {start_page, num_pages});
SetRuntimeArgs(program, writer_h, core, {start_page, num_pages});

EnqueueProgram(cq, program, /*blocking=*/false);
```

**Metal 2.0:**

```cpp
ProgramSpec spec{
    .program_id = "my_program",
    .kernels = {reader, writer},
    .dataflow_buffers = {dfb},
    .semaphores = {sem},
    .work_units = {main_work_unit},
};
Program program = MakeProgramFromSpec(spec);  // (temporary free function)

ProgramRunParams params;
params.kernel_run_params = { /* per-execution argument values */ };
SetProgramRunParameters(program, params);  // (temporary free function)

EnqueueProgram(cq, program, /*blocking=*/false);
```

---

### KernelSpec
See `kernel_spec.hpp`.

The Quasar `experimental::quasar::CreateKernel` family is replaced by populating `KernelSpec`, and adding it to `ProgramSpec::kernels`. This is the difference between a builder API and a descriptor API. You don't add kernels to the Program imperatively; they are declared as data and consumed at the end at Program creation.

**Legacy:**

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

SetRuntimeArgs(program, reader_handle, CoreCoord{0, 0}, {start_page, num_pages});
```

**Metal 2.0:**

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
        .named_runtime_args = {"start_page", "num_pages"},
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
        {{"start_page", start_page}, {"num_pages", num_pages}}}},
}};
SetProgramRunParameters(program, params);  // temporary free function
```

Reuse a single `constexpr const char*` for `unique_id` everywhere the kernel is referenced (`WorkUnitSpec`, `ProgramRunParams`, DFB and semaphore bindings) — this catches typos at compile time.

> **New DM thread cap**: `QuasarDataMovementConfig::num_threads_per_cluster` allowed up to 8 DM threads. Metal 2.0 reserves DM0 and DM1 for runtime use, so `KernelSpec::num_threads` is capped at 6 for DM kernels. Specifying 7 or 8 will fail validation.

---

### DataflowBufferSpec
See `dataflow_buffer_spec.hpp`.

`DataflowBufferSpec` replaces the two-step Quasar pattern — `experimental::dfb::CreateDataflowBuffer` followed by `BindDataflowBufferToProducerConsumerKernels`.

Changes:

1. **Placement is derived, not specified.** The DFB lives wherever its bound producer / consumer kernels run; you don't pass a `core_spec` to the DFB. The runtime now infers everything that the legacy bind-after-create call had to be told (which RISCs, how many threads, etc.). Local DFB producer and consumer kernels must have identical `WorkUnitSpec` membership.
2. **Endpoints are bound at the kernel spec.** Each producer / consumer kernel declares a `DFBBinding` on its `KernelSpec`, naming the DFB and a local accessor name. The kernel code references the DFB through that local accessor name (e.g. `dfb::my_dfb`).

**Legacy:**

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

**Metal 2.0:**

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

Each `local_accessor_name` is independent per binding; the producer and consumer can name the same DFB differently in their respective kernel sources. The producer/consumer RISC masks that the legacy `DataflowBufferConfig` carried are no longer needed: the runtime determines RISC assignment from the kernel bindings.

**Borrowed-memory DFBs.** A DFB can be built on top of an existing `Buffer`'s memory rather than allocating its own L1 storage — the Metal 2.0 form of the legacy "dynamic circular buffer." Set `DataflowBufferSpec::borrowed_from` to the name of a `TensorParameter` whose buffer backs the DFB; the DFB's L1 address resolves at runtime from the corresponding `TensorArg` in `ProgramRunParams::tensor_args`.

**Aliased DFBs.** Two or more DFBs can share backing L1 memory via `DataflowBufferSpec::alias_with`. The aliased DFBs are logically distinct (each has its own `unique_id` and bindings) but physically occupy the same L1 region — useful when same-shape DFBs are produced and consumed in non-overlapping phases. All aliased DFBs must have the same total size (`num_entries * entry_size`), must be bound to the same kernels, and must mutually declare each other in `alias_with`. Aliased DFBs offer no guarantee against data clobbering; correctness is the kernel author's responsibility.

Remote DFBs are exposed in `dataflow_buffer_spec.hpp` but aren't supported yet.

---

### SemaphoreSpec

`SemaphoreSpec` replaces `CreateSemaphore`. Changes:

1. **Descriptor API**, no longer minted by an imperative `CreateSemaphore` call into a Program object.
2. **Semaphore IDs no longer travel as runtime arguments.** Each kernel that uses a semaphore declares a `SemaphoreBinding` naming it and giving it a local accessor name; kernel code accesses it as `sem::accessor_name`.

Unlike DFBs, semaphore placement is *not* derived: a semaphore is a remote resource accessible from any kernel, so its node set is specified directly via `target_nodes`.

**Legacy:**

```cpp
uint32_t sem_id = CreateSemaphore(program, core_spec, /*initial_value=*/0);

// Pass the ID to kernels that need it, as a runtime arg:
SetRuntimeArgs(program, reader_h, core, {/* ... */, sem_id});
```

**Metal 2.0:**

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

- `SemaphoreSpec::SemaphoreMemoryType::Register` is a placeholder for a Gen2 hardware feature (register-backed semaphores); it is not yet supported. Only `L1` works today. If we ever decide to support `Register`, we'll hopefully automate the L1 / Register selection on Quasar.
- Setting `initial_value` to a non-zero value is not supported on Gen2 (for compatibility with Register-based semaphores). We hope to deprecate non-zero initial values on Gen 1 as well, with the help of remote DFB.

---

### TensorParameter

`TensorParameter` declares a tensor as a Program-scope resource. Kernels access it via `KernelSpec::TensorBinding`; the runtime `MeshTensor` is supplied per execution via `ProgramRunParams::TensorArg`. The kernel-author API collapses to a single line: `TensorAccessor(ta::name)`.

Three pieces, paralleling the DFB / Semaphore pattern with one deliberate asymmetry: tensors are *user-managed* resources (you own the lifetime), so the program-scope type is named `TensorParameter` rather than `TensorSpec`.

> **⚠ Pre-migration check.** If your kernel sources contain `ArgConfig::RuntimeTensorShape`, this op cannot migrate to Metal 2.0 yet — Metal 2.0 has no positional-CTA mechanism, so the legacy plumbing has no equivalent. (`RuntimeTensorShape` is a production-op feature; you almost certainly won't encounter it in Quasar tests.)

#### Legacy

```cpp
// host: append TensorAccessor args to the kernel's positional CTA list,
// pass the buffer base address as a runtime arg.
std::vector<uint32_t> reader_cta = {page_size};
tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_cta);

experimental::quasar::QuasarDataMovementConfig reader_config{
    .num_threads_per_cluster = 3,
    .compile_args = reader_cta,
};
KernelHandle reader_h = experimental::quasar::CreateKernel(
    program, "kernels/reader.cpp", cluster, reader_config);
SetRuntimeArgs(program, reader_h, cluster, {input_tensor.buffer()->address(), num_pages});
```

```cpp
// kernel: read the buffer address from the RTA list, manually thread
// CTA offsets to construct TensorAccessorArgs, then build the accessor.
constexpr uint32_t page_size = get_compile_time_arg_val(0);
constexpr auto input_args = TensorAccessorArgs<1>();   // 1 = number of preceding CTAs
uint32_t input_addr = get_arg_val<uint32_t>(0);
auto input = TensorAccessor(input_args, input_addr);
```

#### Metal 2.0

```cpp
constexpr const char* INPUT = "input";

// In ProgramSpec — declare the tensor as a Program-scope parameter.
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

For each test:

1. **Find each `TensorAccessor`.** In each kernel, locate every `TensorAccessor(args, addr)` construction. For each, trace `addr` back through the host code to the originating `Tensor`.
2. **Declare `TensorParameter`s.** Add one entry per tensor to `ProgramSpec::tensor_parameters`, using `tensor.tensor_spec()`. Pick a stable `unique_id` constant.
3. **Add `TensorBinding`s.** On each `KernelSpec` whose kernel accesses the tensor, add an entry to `tensor_bindings`. The `accessor_name` will appear as `ta::<accessor_name>` in the kernel.
4. **Update kernel code.**
   - Replace `TensorAccessor(args, addr)` with `TensorAccessor(ta::<accessor_name>)`.
   - Drop the `TensorAccessorArgs<offset>()` line.
   - Drop the `get_arg_val<uint32_t>(N)` line that retrieved the buffer address — re-index any RTAs that came after.
5. **Wire `tensor_args`.** Add one `TensorArg` per `TensorParameter` to `ProgramRunParams::tensor_args`, passing the actual `MeshTensor`.

#### Validation

At enqueue, the runtime `MeshTensor`'s `TensorSpec` must equal the binding's declared spec. Mismatches error loudly with a message naming the binding.

---

### WorkUnitSpec

`WorkUnitSpec` is a new top-level concept that replaces the `core_spec` argument from `experimental::quasar::CreateKernel` (and from `experimental::dfb::CreateDataflowBuffer`). It declares groups of kernels that operate together on a worker node, and on which nodes they run.

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

---

### ProgramRunParams

`ProgramRunParams` replaces `SetRuntimeArgs(program, kernel_handle, core, args)` and related APIs. The kernel declares its runtime-arg *schema*; values are supplied per execution via `ProgramRunParams`. Tensor arguments — one `MeshTensor` per declared `TensorParameter` — also live on `ProgramRunParams::tensor_args` (see [TensorParameter](#tensorparameter)).

**Legacy:**

```cpp
// All arguments are positional
SetRuntimeArgs(program, reader_h, core, {start_page, num_pages, sem_id});
SetCommonRuntimeArgs(program, reader_h, {bank_id});
```

**Metal 2.0:**

```cpp
// Schema declared on the kernel (with named arguments!):
KernelSpec reader{
    .unique_id = READER,
    // ...
    .runtime_arguments_schema = {
        .named_runtime_args = {"start_page", "num_pages"},
        .named_common_runtime_args = {"bank_id"},
    },
    // (sem_id no longer travels as a runtime arg — bind the semaphore instead.)
};

// Values supplied per execution:
ProgramRunParams params;
params.kernel_run_params = {{
    .kernel_spec_name = READER,
    .named_runtime_args = {{NodeCoord{0, 0},
        {{"start_page", start_page}, {"num_pages", num_pages}}}},
    .named_common_runtime_args = {{"bank_id", bank_id}},
}};
SetProgramRunParameters(program, params);  // temporary free function
```

Some kernels can't get away with just named arguments — they're written to consume a dynamic-count argument tail (e.g. variable tensor rank), retrieved in a loop on the device. For those, varargs:

```cpp
// Schema:
.compile_time_arg_bindings = {{"rank", rank}},
.runtime_arguments_schema = {
    .num_runtime_varargs = rank,  // shape: one entry per dimension
},

// Values:
.runtime_varargs = {{NodeCoord{0, 0}, shape_dims}},  // shape_dims.size() == rank
```

Named and vararg forms can coexist on the same kernel. Vararg indices are stable across schema changes. Common runtime varargs (`num_common_runtime_varargs`, retrieved on the device via `get_common_vararg(i)`) work analogously, broadcast across all nodes.

> The vararg form is intended for kernels whose device-side code retrieves arguments in a loop — i.e., `get_vararg(i)` with `i` a runtime variable. When the kernel reads each argument by a constant index (`get_vararg(0)`, `get_vararg(1)`, …), the named form reads more clearly on both sides.

`ProgramRunParams` must be specified for every kernel that has runtime arguments, on every node where the kernel runs. Kernel handles are gone — kernels are referenced by `unique_id` string (`kernel_spec_name`).

There's a "power user, efficient inner loop" version of the runtime args update API: `ProgramRunParamsView`. It's not implemented yet.

---

## Device-Side Migration

### TensorAccessor

Construction collapses to one line. `TensorAccessor` takes the codegen-emitted token directly; everything else is unchanged from today (`get_noc_addr`, `get_bank_and_offset`, `dspec()`, `noc_async_read_page`, etc.).

The token lives in the `ta::` namespace inside `kernel_bindings_generated.h` (auto-generated from the host-side `TensorBinding`), parallel to the `dfb::` and `sem::` namespaces.

**Legacy:**

```cpp
constexpr uint32_t page_size = get_compile_time_arg_val(0);
constexpr auto input_args = TensorAccessorArgs<1>();
uint32_t input_addr = get_arg_val<uint32_t>(0);
auto input = TensorAccessor(input_args, input_addr);
```

**Metal 2.0:**

```cpp
auto input = TensorAccessor(ta::input);
```

The `TensorAccessorArgs<N>()` line, the manual `next_compile_time_args_offset()` chaining for multi-tensor stacking, and the buffer-address RTA are all gone. Layout metadata is packed by the host into the kernel's compile-time args at program creation; the per-enqueue base address rides on a host-managed CRTA slot the kernel never sees directly.

See [TensorParameter](#tensorparameter) for the host-side declaration that produces the `ta::` namespace.

---

### Kernel Argument Retrieval Syntax

The Metal 2.0 host API declares kernel arguments by name; the kernel-side API retrieves them by name. This replaces the legacy positional `get_arg_val<uint32_t>(N)` style.

**The only `#include` a porter adds to a kernel is `experimental/kernel_args.h`** — that pulls in the accessor templates (`get_arg`, `args::`, `dfb::`, `sem::`, `ta::`). The generated headers `kernel_bindings_generated.h` (which carries `dfb::` / `sem::` / `ta::` declarations from the host bindings) and `kernel_args_generated.h` (which carries `args::` declarations from `compile_time_arg_bindings` + `runtime_arguments_schema`) are auto-included by the build system via `<kernel_includes.hpp>` before the kernel source. **Do not** `#include` either generated header from your kernel.

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

> **Note**: This argument-retrieval syntax will evolve again to support custom argument types beyond `uint32_t` (including user-defined POD types). The named-accessor mechanism (`get_arg(args::name)`) will be preserved; the underlying types will become user-extensible.

---

## Complete Migration Example

Reader kernel reads pages from an input tensor into a DFB; writer kernel pulls from the DFB and writes pages to an output tensor. Multi-threaded kernels (3 producer threads, 3 consumer threads — at Metal 2.0's 6-DM-thread cap) on a single Quasar cluster, no semaphores. End-to-end host code.

**Legacy:**

```cpp
constexpr uint32_t page_size = 1024;
constexpr uint32_t num_pages = 8;
const CoreCoord cluster{0, 0};

Program program = CreateProgram();

// Reader kernel — 3 threads. Append TensorAccessor args to the CTA list.
std::vector<uint32_t> reader_cta = {page_size};
tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_cta);

experimental::quasar::QuasarDataMovementConfig reader_config{
    .num_threads_per_cluster = 3,
    .compile_args = reader_cta,
};
KernelHandle reader_h = experimental::quasar::CreateKernel(
    program, "kernels/reader.cpp", cluster, reader_config);

// Writer kernel — 3 threads. Same shape on the output side.
std::vector<uint32_t> writer_cta = {page_size};
tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_cta);

experimental::quasar::QuasarDataMovementConfig writer_config{
    .num_threads_per_cluster = 3,
    .compile_args = writer_cta,
};
KernelHandle writer_h = experimental::quasar::CreateKernel(
    program, "kernels/writer.cpp", cluster, writer_config);

// DFB — multi-threaded producer (DM0-DM2) and consumer (DM3-DM5)
experimental::dfb::DataflowBufferConfig dfb_config{
    .entry_size = page_size,
    .num_entries = num_pages,
    .producer_risc_mask = 0x07,  // DM0 | DM1 | DM2
    .num_producers = 3,
    .pap = experimental::dfb::AccessPattern::STRIDED,
    .consumer_risc_mask = 0x38,  // DM3 | DM4 | DM5
    .num_consumers = 3,
    .cap = experimental::dfb::AccessPattern::STRIDED,
    .data_format = tt::DataFormat::Float16_b,
};
uint32_t dfb_id = experimental::dfb::CreateDataflowBuffer(program, cluster, dfb_config);
experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
    program, dfb_id, reader_h, writer_h);

// Runtime args — buffer addresses ride as RTAs.
SetRuntimeArgs(program, reader_h, cluster, {input_tensor.buffer()->address(), num_pages});
SetRuntimeArgs(program, writer_h, cluster, {output_tensor.buffer()->address(), num_pages});

EnqueueProgram(cq, program, /*blocking=*/false);
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
    .num_threads = 3,  // 3 producer threads
    .compile_time_arg_bindings = {{"page_size", page_size}},
    .runtime_arguments_schema = {.named_runtime_args = {"num_pages"}},
    .dfb_bindings = {{
        .dfb_spec_name = DFB,
        .local_accessor_name = "out_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::PRODUCER,
        .access_pattern = DFBAccessPattern::STRIDED,
    }},
    .tensor_bindings = {{
        .tensor_parameter_name = INPUT,
        .accessor_name = "input",   // kernel accesses as `ta::input`
    }},
    .config_spec = DataMovementConfiguration{
        .gen2_data_movement_config = DataMovementConfiguration::Gen2DataMovementConfig{},
    },
};

KernelSpec writer{
    .unique_id = WRITER,
    .source = KernelSpec::SourceFilePath{"kernels/writer.cpp"},
    .num_threads = 3,  // 3 consumer threads
    .compile_time_arg_bindings = {{"page_size", page_size}},
    .runtime_arguments_schema = {.named_runtime_args = {"num_pages"}},
    .dfb_bindings = {{
        .dfb_spec_name = DFB,
        .local_accessor_name = "in_dfb",
        .endpoint_type = KernelSpec::DFBEndpointType::CONSUMER,
        .access_pattern = DFBAccessPattern::STRIDED,
    }},
    .tensor_bindings = {{
        .tensor_parameter_name = OUTPUT,
        .accessor_name = "output",  // kernel accesses as `ta::output`
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

EnqueueProgram(cq, program, /*blocking=*/false);
```

The Metal 2.0 form is more verbose, but carries additional structure. DFB endpoint information, in particular, was scattered across `DataflowBufferConfig` (RISC masks, access patterns) and the subsequent `BindDataflowBufferToProducerConsumerKernels` call (kernel handles); in Metal 2.0 it lives entirely on each `KernelSpec::dfb_bindings`. Tensor identity moves out of runtime args entirely — `TensorParameter` declarations live alongside DFBs at the program level, and the kernel-side `TensorAccessorArgs<N>()` offset bookkeeping disappears.

---

## Troubleshooting

Common pitfalls when migrating from the temporary Quasar APIs:

- **Don't pass tensor addresses as runtime arguments.** Metal 2.0's `TensorBinding` auto-injects per-enqueue base addresses; that's the supported path. If your migration ports `tensor.buffer()->address()` over verbatim as an RTA, revisit — you want a `TensorBinding`.
- **Runtime varargs are intended for dynamic-count tails.** `num_runtime_varargs` is the right fit for kernels that consume a variable number of arguments in a loop — e.g., an N-dimensional shape gated on a CTA-known `rank`. For kernels with a fixed set of individually-known arguments, named RTAs are the recommended form, even when porting from a positional legacy interface.

---

## Kernel globals

Metal 2.0 has no equivalent of the temporary Quasar API's `QuasarDataMovementConfig::is_legacy_kernel` flag.

On WH/BH, each DM core has its own private memory and gets its own copy of the kernel binary, so `.data` / `.bss` are effectively per-core for free. On Quasar, the 8 DM cores in a cluster share L1, and the natural model is one shared binary with multiple threads — globals live in one place and are *shared* across all DM threads. Setting `is_legacy_kernel = true` recovered WH/BH semantics by duplicating the kernel binary per DM core, at the cost of L1 footprint.

Metal 2.0 always uses the shared-binary model. If you depended on `is_legacy_kernel = true`, the kernel must be rewritten — use `thread_local` for variables that genuinely need per-thread state.

The flag's implicit premise was that you would want to thread an existing WH/BH kernel as a first porting step. That premise is no longer correct: the recommended quick-port path from WH/BH to Quasar is to keep kernels **single-threaded**. (A Quasar Neo Cluster is not four BH workers in a trenchcoat; and Metal 2.0's 6-DM-thread cap precludes the brute-thread approach in any case.) Single-threaded kernels are unaffected by this behavior change.
