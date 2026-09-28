# kernel_spec invariant tests

Local invariants of `KernelSpec` and its nested structs, as written in `kernel_spec.hpp`. Every rule here reads one
KernelSpec and nothing else in the ProgramSpec. Rules that relate a kernel's bindings to the rest of the ProgramSpec
(for example, "the bound DFB is declared") are structural and live in [`../program_spec/`](../program_spec/).

Fixtures: `ProgramSpecTestQuasar` (Q) and `ProgramSpecTestGen1` (WH). All test names start with `CPU_`.

## Files

| File | Tests | Covers |
|---|---|---|
| `basic_kernel_info.cpp` | 7 | `num_threads` per architecture and kernel kind, `source` |
| `compiler_options.cpp` | 1 | `CompilerOptions` |
| `dfb_binding.cpp` | 8 | `DFBBinding` and `dfb_bindings` |
| `semaphore_binding.cpp` | 6 | `SemaphoreBinding` and `semaphore_bindings` |
| `scratchpad_binding.cpp` | 4 | `ScratchpadBinding` and `scratchpad_bindings` |
| `tensor_binding.cpp` | 4 | `TensorBinding` and `tensor_bindings` |
| `kernel_arguments.cpp` | 8 | `compile_time_args` and `RuntimeArgSchema` |
| `hardware_config.cpp` | 4 | `hw_config` against the kernel's own bindings |

## Coverage of the header invariants

"Accepted" lists tests that pin the legal side of a rule.

| Invariant | Tests |
|---|---|
| `source` path is non-empty, exists and is readable; `SourceCode::code` is non-empty | **Untested.** Accepted: Q `SourceCodeKernelSucceeds` |
| Gen1: `num_threads == 1` | WH `MultiThreadedDMKernelFails`, WH `MultiThreadedComputeKernelFails` |
| Gen2 data-movement: `num_threads` in [1, 6] | Q `KernelWithZeroThreadsFails` (0), Q `DMKernelExceedingMaxThreadsFails` (9). The 6/7 boundary is untested |
| Gen2 compute: `num_threads` in {1, 2, 4} | Q `ComputeKernelExceedingMaxThreadsFails` (5). Accepted: Q `MaxComputeThreadsSucceeds` (4). 3 is untested |
| `DFBBinding::accessor_name` is a C++ identifier of at most `MAX_ACCESSOR_NAME_LENGTH` characters | Q `InvalidLocalAccessorNameFails` |
| `DFBBinding::access_pattern` is not BLOCKED, and is STRIDED for a producer | **Untested** |
| At most one PRODUCER and one CONSUMER binding per DFB | Q `DuplicateProducerBindingForSameLocalAccessorNameFails`, Q `DFBBoundTwiceInSameRoleUnderDifferentNamesFails` |
| Two bindings share an `accessor_name` only as the PRODUCER and CONSUMER of one DFB | Q `SharedLocalAccessorNameForDifferentDFBsFails`. Accepted: Q `SelfLoopWithSharedLocalAccessorNameSucceeds` |
| Gen2: a data-movement kernel does not self-loop a DFB | Q `DMKernelSelfLoopOnGen2Fails`. Accepted: WH `DMKernelSelfLoopOnGen1Succeeds`, Q `DFBSelfLoopOnComputeKernelSucceeds` |
| A compute kernel that self-loops a DFB uses STRIDED on its CONSUMER binding | **Untested** |
| A CONSUMER binding with access pattern ALL requires `num_threads <= 4` | **Untested** |
| `SemaphoreBinding::accessor_name` is a C++ identifier of at most `MAX_ACCESSOR_NAME_LENGTH` characters | Q `KernelSemaphoreBindingInvalidAccessorFails` (identifier only; the length limit is untested) |
| `semaphore_spec_name` is unique across `semaphore_bindings` | **Untested** |
| `accessor_name` is unique across `semaphore_bindings` | Q `KernelSemaphoreBindingDuplicateAccessorFails` |
| Gen2 and Wormhole: a compute kernel has no `semaphore_bindings` | Q `SemaphoreBoundToComputeKernelFailsOnQuasar`, WH `SemaphoreBoundToComputeKernelFailsOnWormhole`. Accepted: WH `SemaphoreBoundToDMKernelSucceedsOnGen1`, Q `KernelSemaphoreBindingsSucceed` |
| `ScratchpadBinding::accessor_name` is a C++ identifier of at most `MAX_ACCESSOR_NAME_LENGTH` characters | Q `InvalidScratchpadAccessorNameFails` |
| `scratchpad_spec_name` is unique across `scratchpad_bindings` | Q `ScratchpadBoundTwiceInOneKernelFails` |
| `accessor_name` is unique across `scratchpad_bindings` | Q `DuplicateScratchpadAccessorNameFails`. Accepted across kernels: Q `MultipleScratchpadsEachBoundToOwnKernelSucceeds` |
| `TensorBinding::accessor_name` is a C++ identifier of at most `MAX_ACCESSOR_NAME_LENGTH` characters | WH `InvalidTensorAccessorNameFails` |
| `accessor_name` is unique across `tensor_bindings` | WH `DuplicateTensorAccessorNameWithinKernelFails` |
| `compile_time_args` keys are C++ identifiers | **Untested** |
| No name is shared between `compile_time_args` and `runtime_arg_schema` | Q `NamedRtaCtaCollisionFails` (RTA against CTA; CRTA against CTA is untested) |
| `runtime_arg_names` and `common_runtime_arg_names` are C++ identifiers | Q `InvalidNamedRtaIdentifierFails`, Q `InvalidNamedCrtaIdentifierFails` |
| No name repeats within `runtime_arg_names` or within `common_runtime_arg_names` | **Untested** |
| No name repeats across `runtime_arg_names` and `common_runtime_arg_names` | Q `NamedRtaCrtaCollisionFails` |
| Every `unpack_modes` key names a DFB this kernel binds | Q `ComputeConfigUnpackToDestModeReferencesUnboundDFBFails` |
| Gen1: UnpackToDest on a DFB this kernel consumes requires `enable_32_bit_dest` | WH `ConsumerUnpackToDestBelow32BitWithoutEnableFailsForPerf`. Accepted on Gen2: Q `ConsumerUnpackToDestBelow32BitWithoutEnableSucceeds` |
| Every `config_2xx->disable_dfb_implicit_sync_for` entry names a DFB this kernel binds | **Untested** |

Tests that document a freedom rather than a rule:

- WH `AccessorNamesAcrossCategoriesAreSeparateNamespaces`: the same string may be a DFB, semaphore and tensor
  accessor name in one kernel.
- Q `TensorBindingOnComputeKernelIsAccepted`: compute kernels may bind tensors.
- Q `DifferentKernelsMayReuseArgNames`: the argument-name rules apply per kernel.
- Q `NamedRuntimeArgsSucceeds`, Q `CompileTimeArgBindingsSucceeds`, Q `RuntimeArgsSchemaSucceeds`,
  Q `CompilerOptionsDefinesSucceeds`, Q `ComputeConfigMathFidelitySucceeds`: ordinary uses of these fields are accepted.
