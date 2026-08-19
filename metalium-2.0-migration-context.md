# Metalium 2.0 migration context

Compiled from the migration learnings in `knowledge/work` on 2026-08-19. Use
this as working context for future TTNN operation migrations, then verify API
names and examples against the target branch because Metalium 2.0 interfaces
are still evolving.

## Executive rule

A Metalium 2.0 migration is complete only when the whole host/kernel contract
has one canonical, semantic description. It is not a collection of isolated
API substitutions.

The completion boundary is:

- the factory returns `ProgramArtifacts` containing a `ProgramSpec` and
  `ProgramRunArgs` (or the target branch's current equivalent);
- every migrated kernel uses `TT_KERNEL`;
- compile-time arguments are named template parameters;
- runtime and common-runtime arguments are named function parameters, with the
  host schema owning their RTA/CRTA classification;
- tensors and dataflow buffers are bound by semantic name;
- device code uses the current object APIs for tensor access, DFB/CB queues,
  NoC, endpoints, semaphores, and local memory;
- no positional argument recovery, raw tensor-address plumbing, numeric-CB
  shadow path, or deprecated initializer remains where the semantic boundary
  owns that fact;
- every meaningful specialization compiles cold, and every migration-owned
  warning is fixed.

Partial migrations can build and even pass cached tests while retaining two
sources of truth. Treat a hybrid legacy/new boundary as unfinished.

## Mental model: one fact, one owner

| Fact | Canonical owner | Kernel expression | Migration smell |
|---|---|---|---|
| Compile-time choice | `ProgramSpec` kernel schema | named `TT_KERNEL` template parameter | positional CTA lookup |
| Per-dispatch scalar | `ProgramRunArgs`/named run schema | named function parameter | manual positional unpacking |
| Tensor identity/address | semantic tensor binding | `TensorAccessor(tensor::<name>)` | raw address passed independently |
| DFB identity | semantic DFB declaration | `DataflowBuffer(dfb::<name>)` | parallel hardcoded numeric CB ID |
| Cache identity | operation attributes/spec | none | behavior-changing value omitted from key |
| Cache-hit mutation | standard binding refresh, or a justified override | current run value | stale address/value after same-key reuse |
| Public compute option | validation + cache key + descriptor translation | hardware behavior | accepted and keyed but ignored |

If a legacy helper or raw identifier duplicates a semantic binding, remove it.
Do not retain compatibility scaffolding without a real remaining consumer.

## Migration workflow

### 1. Freeze observable behavior first

Before changing implementation, record and test:

- public inputs, outputs, defaults, modes, failure behavior, and non-goals;
- shape, dtype, layout, placement, allocation, ownership, and input
  immutability for every output;
- numerical accuracy, exact repeated-device determinism, trace replay, and
  production performance;
- boundary cases that exercise control flow, not only nominal shapes;
- same-key cache hits using newly allocated input and output tensors;
- every public compute-config field at its default, supported non-defaults,
  and unsupported non-neutral values.

Use adversarial cases for semantic invariants. For example, noncommuting
operands expose reversed affine composition, non-power-of-two group counts
exercise prefix stages, and explicit first-output assertions expose exclusive
scan boundary errors.

Keep the tests frozen during the migration. A performance-reference change is
a contract decision, not a cleanup hidden inside implementation work.

### 2. Audit the complete boundary before editing

Inventory every kernel and every meaningful specialization. For each value
crossing the boundary, record:

| Value | Semantic name | CTA/RTA/CRTA/tensor/DFB | Changes between calls? | In cache key? | Refreshed on dispatch? |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... |

Also draw the DFB topology. For every DFB, record:

- producer and consumer;
- processor kind for each endpoint (compute or data movement);
- format and capacity by mode;
- whether the queue is streamed, setup-only, or compute-local scratch;
- whether its lifetime permits producer/consumer overlap;
- whether it aliases another DFB's physical storage.

This audit should decide the design before mechanical conversion begins.

### 3. Convert the host factory as one coherent unit

- Return the branch's current `ProgramArtifacts` form.
- Put structural/kernel facts in `ProgramSpec` and per-dispatch bindings in
  `ProgramRunArgs`.
- Give tensors, DFBs, CTAs, RTAs, and CRTAs matching semantic names on host and
  device.
- Bind a kernel only to tensors and DFB endpoints it actually uses.
- Use mode-specific specs when modes genuinely consume different tensors or
  use DFB roles differently. Do not preserve unused tensor slots with sentinel
  addresses merely to make specs look identical.
- Derive DFB format and capacity from the semantic contents in that mode, not
  from a similarly named input tensor.
- Remove superseded legacy descriptor helpers and numeric/raw plumbing.

### 4. Convert every kernel completely

- Use `TT_KERNEL` everywhere in the migrated operation.
- Replace positional CTA/RTA recovery with named parameters.
- Construct each `TensorAccessor`, `DataflowBuffer`, or legitimate
  `CircularBuffer` once in the kernel scope and reuse the handle.
- Always namespace-qualify generated tokens when the local object has the same
  name:

  ```cpp
  DataflowBuffer scratch_one(dfb::scratch_one);
  ```

  `DataflowBuffer scratch_one(scratch_one)` is valid-looking C++
  self-initialization. It can compile, leave the identity undefined, and appear
  only as a hardware FIFO deadlock.
- Migrate the whole Device 2.0 surface that the kernel touches: NoC,
  source/destination endpoints, DFB/CB, semaphore, and local-memory APIs—not
  only circular-buffer calls.
- Before replacing a compute initializer, compare operand sources,
  destination state, buffer requirements, FPU/SFPU backend, and sequencing.

### 5. Validate DFB ownership and queue semantics

On Gen2, one local DFB endpoint role cannot be shared across compute and
data-movement processor kinds. A spec that assigns one consumer role to both
will be rejected before JIT.

Preferred resolutions, in increasing cost:

1. Keep a single consumer and let compute inspect producer-local state without
   consuming when that matches the algorithm.
2. Add an explicit handoff/mirror DFB so each processor kind owns a distinct
   logical endpoint.
3. Alias logical DFBs onto one physical allocation when supported and when
   duplicate packing/storage is the only problem.

Aliasing shares bytes, not queue state. Each logical alias must independently
reserve, publish, wait, and consume. One-stage tests can hide pointer errors;
exercise multi-stage and ping/pong paths.

`DataflowBuffer::scoped_write_lock` protects and flushes writable storage. It
does not reserve or publish pages. A locally produced payload still needs:

1. `reserve_back`;
2. scoped write access;
3. `push_back`.

### 6. Make cache and public configuration contracts truthful

For every public option, record one disposition: honored, neutral-only,
unsupported, or outside scope. Audit these obligations separately:

1. default resolution;
2. validation;
3. program-cache identity;
4. descriptor translation;
5. actual runtime effect.

Being present in the cache key does not prove a field is honored. Reject an
unsupported non-neutral value before program construction; otherwise the API
may create distinct cached programs with identical behavior.

Every per-call value must either change the cache key or be reapplied on
dispatch. Prefer standard semantic tensor/buffer rebinding. Add a manual
runtime-argument override only for varying non-keyed values that the descriptor
binding mechanism cannot represent.

The required cache test uses the same shape/configuration with fresh tensor
allocations, proves the cache-entry count does not increase, verifies addresses
changed, and validates the second call against its own reference.

### 7. Preserve performance deliberately

Object wrappers can regress device time if helper boundaries stop DFB IDs from
becoming compile-time constants. Persist object handles for queue operations,
but keep static LLK IDs explicit where required and inspect whether large
helpers block constant propagation. Measure before and after; force-inlining is
an evidence-driven remedy, not a default style.

Use exposed API defaults as controls. Override a default only when all three
are present:

1. a concrete intent;
2. semantics explaining why the setting applies;
3. a repeatable experiment showing benefit above noise without unacceptable
   accuracy, determinism, L1, or portability cost.

Start streamed CBs double-buffered. Classify producer, consumer, access
threads, lifetime, overlap opportunity, depth, and L1 cost before dialing a
setup-only or compute-local buffer down to one slot. Compare reduced layouts
against the all-double baseline.

Report the timing domain. Use device-profiler program time for device work and
synchronized wall time for end-to-end dispatch. Do not substitute one ratio
for the other.

### 8. Build, install, and prove cold compilation

An incremental build may update `build_Release/ttnn/_ttnncpp.so` while Python
loads `build_Release/lib/_ttnncpp.so`. After a host rebuild in a worktree,
install into the configured worktree prefix and verify the loaded library when
runtime behavior contradicts source.

Use an isolated cache rather than deleting the shared user cache:

```bash
migration_cache_dir=$(mktemp -d /tmp/metalium2-jit-XXXXXX)
TT_METAL_CACHE="$migration_cache_dir" <full safe test command> 2>&1 | tee <log>
```

Cold-validation evidence must include:

- exact build, install, and test commands;
- hardware/architecture and production shapes;
- test totals and duration;
- JIT hit count (expect zero for a true cold run);
- cache directories/kernel basenames and generated compile defines proving
  every intended specialization compiled;
- PCC, maximum absolute error, repeat determinism, and performance samples;
- full compiler transcript;
- warnings normalized by unique message, source, kernel, architecture, and
  ownership;
- formatting/hooks and `git diff --check` results.

A zero-hit aggregate is necessary but not sufficient: inspect generated
artifacts/defines to prove mode and shape specialization coverage. Separate
expected negative-test fatal diagnostics and framework/runtime notices from
compiler warnings. Fix every migration-owned warning; record external causes
with an owner and rationale.

## Failure signatures and likely causes

| Symptom | First hypothesis to test | Evidence that distinguishes it |
|---|---|---|
| Host build passes; validator says kernel is not Metalium 2.0 | Python loaded an uninstalled/stale extension | inspect loaded `_ttnncpp.so`; install worktree build |
| Compiles, then hardware hangs around scratch DFBs | same-name DFB token self-initialized | scan `DataflowBuffer x(x)`; qualify `dfb::x` |
| Spec rejected before JIT for mixed consumer masks | one DFB role bound to compute and data movement | inspect producer/consumer processor masks |
| Trace hangs after locally seeding a DFB | write lock used without queue lifecycle | verify reserve → write → push |
| One-stage correctness passes; multi-stage PCC collapses | aliased storage treated as shared FIFO state | verify independent protocol on every alias |
| One mode corrupts data despite matching tensor formats | reused DFB role has different semantic payload | derive mode-specific DFB format/capacity |
| Fresh tensors reuse cache but produce stale output | varying value is neither keyed nor rebound | run same-key/fresh-allocation address audit |
| Config changes cache entries but not behavior | descriptor adapter ignores keyed field | trace field from default/validation through hardware config |
| Correctness passes but device time regresses | wrappers/helpers prevent constant propagation | pass named static IDs to LLKs; inspect/instrument helper boundary |
| Tests pass with deprecation output | cached kernels or incomplete API conversion | isolated forced JIT plus warning ownership inventory |

## Definition of done

- [ ] Observable and performance contracts were frozen before implementation.
- [ ] Every kernel and meaningful specialization is inventoried.
- [ ] The factory uses the current semantic `ProgramArtifacts` boundary.
- [ ] Every migrated kernel uses `TT_KERNEL` and named arguments.
- [ ] Tensors and DFBs are semantic bindings end-to-end.
- [ ] No duplicate positional, raw-address, numeric-CB, or legacy descriptor
      path remains without a documented legitimate boundary.
- [ ] DFB endpoint ownership is valid for each processor kind.
- [ ] Queue lifecycle is explicit, including for locally produced and aliased
      DFBs.
- [ ] Mode-specific tensor participation, DFB formats, and capacities are
      truthful.
- [ ] Every public config field is honored or rejected; cache behavior is
      audited independently.
- [ ] Same-key cache hits with fresh allocations are correct.
- [ ] Build and worktree-local install pass.
- [ ] Full frozen suites pass on real target hardware.
- [ ] Forced JIT reports zero hits and artifact inspection proves specialization
      coverage.
- [ ] There are zero unexplained migration-owned warnings.
- [ ] Accuracy, determinism, and production performance remain within their
      frozen contracts.
- [ ] Hooks/formatting and `git diff --check` pass.
- [ ] Exact commands, results, timing domain, and anything not validated are
      recorded.

## Canonical resources

Read these on the target branch before designing the migration:

1. [Kernel Arguments as Function & Template Parameters](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/NamedKernelArgs/kernel_args_as_parameters.md)
   — `TT_KERNEL`, named CTAs, RTAs, and CRTAs. In-tree path:
   `tech_reports/NamedKernelArgs/kernel_args_as_parameters.md`.
2. [Device 2.0 Data Movement API Migration Guide](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md)
   — complete NoC, endpoint, buffer, semaphore, and local-memory conversion.
3. [Merged explicit-kernel-argument PR #46623](https://github.com/tenstorrent/tt-metal/pull/46623)
   — concrete host/kernel named-argument migration history.
4. [Descriptors and Specs: how a TTNN op describes its program](https://gist.github.com/dgomezTT/7584e4eb0dc6ddc5214f9a7e90e77181)
   — cache-key versus per-dispatch refresh and factory/spec concepts.
5. Current host API headers:
   `tt_metal/api/tt-metalium/experimental/metal2_host_api/program_spec.hpp`,
   `program_run_args.hpp`, `kernel_spec.hpp`,
   `dataflow_buffer_spec.hpp`, and `tensor_parameter.hpp`.
6. Current in-tree examples (choose one similar to the target operation and
   re-check it on the target branch):
   `ttnn/cpp/ttnn/operations/full/device/full_program_factory_interleaved.cpp`,
   `ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.cpp`,
   `ttnn/cpp/ttnn/operations/reduction/accumulation/ema/device/ema_program_factory.cpp`,
   and their paired kernels.

## Evidence boundaries

The rules above generalize repeated causal findings. Numeric PCC thresholds,
performance bands, buffer depths, compiler modes, and individual optimization
winners in the source notes belong to their original operations and hardware.
Reuse the decision method, not the measured winner.

The current local `tt-metal` checkout used to verify resource paths was 371
commits behind `origin/main` on 2026-08-19. Treat the listed source paths as
orientation and confirm the latest interfaces/examples on the actual migration
branch.
