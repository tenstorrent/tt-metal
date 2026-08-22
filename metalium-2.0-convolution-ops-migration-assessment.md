# Metalium 2.0 migration assessment: convolution-owned TTNN operations

Date: 2026-08-19

## Executive summary

This is a static source assessment against
[`metalium-2.0-migration-context.md`](metalium-2.0-migration-context.md). It does
not claim build, cold-JIT, correctness, determinism, trace, or performance
validation on hardware.

The effective operation ownership in [`.github/CODEOWNERS`](.github/CODEOWNERS)
covers nine C++ operation trees: `conv`, `data_movement/fold`, `pool`,
`sliding_window`, `experimental/adaptive_pool`, `experimental/cnn`,
`experimental/conv3d`, `experimental/padded_slice`, and
`experimental/slice_write`. It also covers the Python frontend files
`activations.py`, `conv2d.py`, and `pool.py`. Shared build files, tests, models,
`kernel_lib`, and `scripts/run_safe_pytest.sh` are owned too, but are not
independent TTNN operations and are not rated as migration units.

The owned C++ trees contain 12 independent device-program migration units and
51 unique owned kernel entry files. At this revision:

- none of those 51 files uses a `TT_KERNEL` entry point;
- Fold's two factories already return semantic `ProgramArtifacts`;
- three of Upsample's four factories already return `ProgramArtifacts`;
- Conv2D, Pool2D, and Halo use `WorkloadDescriptor`; Grid Sample, Rotate,
  Conv3D, and bilinear Upsample use `ProgramDescriptor`;
- Convert-to-CHW/HWC, Padded Slice, and Slice Write still use legacy
  `CachedProgram` factories with manual cache-hit patching.

Overall assessment:

| Category | Independent migration units | Public operations covered |
|---|---:|---|
| No independent work needed | wrappers only | `conv1d`, `conv_transpose2d`, `adaptive_avg_pool2d`, `adaptive_max_pool2d`, `global_avg_pool2d`, host preparation helpers |
| Easy | 2 | `fold`, `convert_to_chw` |
| Medium | 5 | `rotate`, `grid_sample`, `convert_to_hwc`, `padded_slice`, `slice_write` |
| Hard | 5 | `conv2d`, `max_pool2d`/`avg_pool2d`, `upsample`, `halo`, `conv3d` |

“No independent work” means the wrapper has no host/kernel boundary of its own.
It is not a claim that the end-to-end public operation is already compliant: its
compliance follows the underlying device operation.

## CODEOWNERS inventory used for the assessment

CODEOWNERS uses last-match-wins ordering. The operation-specific rules occur
after the generic TTNN and generic kernel-directory rules, so the following are
the effective convolution-team operation scopes (CMake rules additionally add
the infrastructure team):

| CODEOWNERS scope | Ownership note | Migration units or surfaces in this report |
|---|---|---|
| `ttnn/cpp/ttnn/operations/conv/` | convolution team | Conv1D, Conv2D, ConvTranspose2D, and weight/bias preparation |
| `ttnn/cpp/ttnn/operations/data_movement/fold/` | convolution team; overrides the broader data-movement owner | Fold |
| `ttnn/cpp/ttnn/operations/pool/` | convolution team | Pool2D, Grid Sample, Rotate, Upsample |
| `ttnn/cpp/ttnn/operations/sliding_window/` | convolution team | Halo plus host sliding-window/op-slicing utilities |
| `ttnn/cpp/ttnn/operations/experimental/adaptive_pool/` | convolution team | adaptive pooling wrappers |
| `ttnn/cpp/ttnn/operations/experimental/cnn/` | shared with `@esmalTT` | Convert-to-CHW and Convert-to-HWC |
| `ttnn/cpp/ttnn/operations/experimental/conv3d/` | convolution team | Conv3D |
| `ttnn/cpp/ttnn/operations/experimental/padded_slice/` | shared with the data-movement team | Padded Slice |
| `ttnn/cpp/ttnn/operations/experimental/slice_write/` | shared with the data-movement team | Slice Write |
| `ttnn/ttnn/operations/conv2d.py` | convolution team | Python conv preparation/golden glue |
| `ttnn/ttnn/operations/pool.py` | convolution team | Python pool wrappers/golden functions, including `global_avg_pool2d` |
| `ttnn/ttnn/operations/activations.py` | shared with the MM/fused-reduce team | activation lookup/golden helpers only |

The convolution team also owns or co-owns `ttnn/cpp/ttnn/kernel_lib/`, shared
operation CMake files, tests/sweeps, and model directories. These are migration
dependencies or validation assets, not additional operation factories. Shared
kernels invoked by a rated operation are called out under that operation even
when their source file has a different effective owner.

## Rating method

The rating is relative to the other operations in this ownership set and
includes the full definition of done in the migration guide, not merely making
the code compile.

- **Easy:** one bounded topology, small/fixed argument schemas, or a semantic
  host factory already in place. The main remaining work is complete kernel
  conversion plus focused cache/cold-JIT/performance proof.
- **Medium:** several modes or factories, shared kernels, manual cache-hit
  state, or a nontrivial descriptor-to-`ProgramSpec` rewrite, but no apparent
  architectural blocker.
- **Hard:** large specialization matrices, many kernels and arguments,
  multicast/semaphore/config-tensor topology, broad public configuration, or a
  high performance-regression risk.
- **Impossible without infrastructure:** the current semantic APIs cannot
  express a required behavior and no reasonable operation-local redesign is
  visible. No complete public operation conclusively meets that bar, but one
  Conv2D specialization family is conditional under a zero-performance-regression
  requirement.

Factory and kernel line counts below are orientation only. Comments, shared
helpers, and generated specialization complexity make raw LOC a poor standalone
measure.

## No independent work needed

| Public operation or surface | Assessment and dependency |
|---|---|
| `conv1d` | No device factory or kernels of its own. It reshapes/reinterprets inputs and calls `conv2d`; therefore no separate Metalium boundary needs migration, but end-to-end completion depends on Conv2D. |
| `conv_transpose2d` | No independent device factory. Its implementation prepares/transforms inputs and dispatches the Conv2D primitive. It inherits Conv2D's migration status and validation burden. |
| `adaptive_avg_pool2d` | Converts the requested adaptive output into ordinary pooling parameters and calls `avg_pool2d`. No separate kernels; completion follows Pool2D. |
| `adaptive_max_pool2d` | Same composition pattern, calling `max_pool2d`. No separate Metalium boundary; completion follows Pool2D. |
| `global_avg_pool2d` | Python wrapper around `avg_pool2d`, with memory-layout conversion and reshape. It has no device program of its own and inherits Pool2D compliance. |
| Conv/transpose weight and bias preparation helpers | `prepare_conv_weights`, `prepare_conv_bias`, `prepare_conv_transpose2d_weights`, and `prepare_conv_transpose2d_bias` are host/frontend preparation surfaces, not independent device-program factories. They need regression coverage but not a Metalium 2.0 host/kernel conversion. |
| `prepare_grid_sample_grid` | Frontend preparation wrapper with no independent device program; the actual device migration unit is Grid Sample. |
| `activations.py` ownership | The owned file contains activation lookup/golden-function helpers, not a convolution-owned device factory. There is no Metalium 2.0 boundary to migrate in this file. |

The `sliding_window` configuration/generation code and `op_slicing` code are
host orchestration utilities rather than standalone device operations. Their
behavior must remain covered when migrating Conv2D, Pool2D, Upsample, and Halo,
but they are not separate factory migrations.

## Easy

### `fold`

Evidence: 2 factory paths, about 645 factory LOC, 5 owned kernel entry files
(about 365 kernel LOC), plus the shared `untilize_metal2.cpp` compute kernel.

Why easy:

- Both sharded and interleaved/DRAM factories already return
  `ProgramArtifacts` with `ProgramSpec`, `ProgramRunArgs`, semantic tensors, and
  DFB declarations.
- All five owned kernels already use named generated tokens through
  `get_arg(args::...)`, `TensorAccessor(tensor::...)`, and
  `DataflowBuffer(dfb::...)`; the remaining entry-point conversion is localized.
- The data movement is structurally simple compared with convolution and
  pooling: no multicast topology and a small number of tensors/DFBs.

Remaining work and risks:

- Replace every `kernel_main`/`get_arg(args::...)` entry, including the shared
  untilize kernel used by tiled Fold, with the guide's `TT_KERNEL` named
  template/function parameters.
- Finish the remaining legacy CB/NoC calls in the row-major DRAM kernels and
  verify DFB queue lifecycle.
- Cold-compile all sharded, tiled-interleaved, and row-major-interleaved paths;
  add same-key/fresh-allocation cache tests and preserve the existing `-Os`
  performance choice with measurements.

### `convert_to_chw`

Evidence: 1 legacy factory (about 183 LOC) and 3 small owned kernels (about 131
LOC total).

Why easy:

- The topology is fixed: reader, transpose compute, writer, three DFBs, and one
  input/output tensor pair.
- Runtime schemas are small, and the legacy factory's borrowed input/output CBs
  map directly to semantic tensor-backed DFBs.
- There is only one factory and no public mode matrix.

Remaining work and risks:

- Replace `CachedProgram`, `CreateKernel`, `SetRuntimeArgs`, dynamic CB address
  updates, and the manual cache-hit override with `ProgramArtifacts` and normal
  tensor rebinding.
- Convert all three kernels to `TT_KERNEL`, named parameters, and persistent
  DFB objects.
- Compare the transpose compute initializer and CB formats carefully, then run
  fresh-allocation cache and performance tests; this is small but still a
  compute-kernel migration.

## Medium

### `rotate`

Evidence: 2 `ProgramDescriptor` factories (nearest and bilinear), about 718
factory LOC, and 3 owned kernel files (about 308 LOC). Bilinear also shares the
Pool2D compute kernel and Grid Sample writer.

Why medium:

- Host construction is already declarative enough to serve as a useful
  inventory, but it still uses positional descriptor schemas rather than the
  canonical semantic `ProgramSpec` boundary.
- Two interpolation modes have different kernel participation and DFB needs,
  so separate truthful specs are preferable.
- Shared kernels couple this migration to Pool2D/Grid Sample; changing a shared
  entry point must keep every consumer compiling.

Key proof: nearest and bilinear, center/default center, fill, expand, output
allocation, shared-kernel specializations, cache reuse, and device performance.

### `grid_sample`

Evidence: 2 `ProgramDescriptor` factories, about 736 factory LOC, 4 owned kernel
files (about 634 LOC), and the shared Pool2D compute kernel.

Why medium:

- It has a manageable kernel count, but bilinear/nearest, interleaved/sharded,
  precomputed/standard grids, padding, `align_corners`, and batched-channel
  options produce a meaningful specialization matrix.
- All owned kernels still recover positional CTAs/RTAs and tensor accessor
  metadata.
- The bilinear path's shared compute kernel means DFB names, formats, and
  capacities must be coordinated with Pool2D rather than converted in isolation.

Key proof: every mode/layout combination, truthful handling of each public
option and compute config, fresh tensor addresses on a same-key cache hit, and
cold-JIT evidence for both factories.

### `convert_to_hwc`

Evidence: 1 legacy factory (about 589 LOC) and 2 owned kernels (about 181 LOC).

Why medium:

- The kernel count is small, but the factory builds gather/gateway work,
  multiple writer placements, CB aliases/roles, and per-core argument tables.
- It still uses `CachedProgram`, manual runtime updates, numeric CB identities,
  and positional kernel arguments.
- The migration must preserve the gather ordering and output-core mapping; a
  nominal single-shape test would not exercise enough of the topology.

Key proof: different input/output core grids, multi-stage gather paths,
same-key/fresh-allocation reuse, output ordering, and performance.

### `padded_slice`

Evidence: 2 legacy factories (row-major and tiled), about 964 factory LOC, 4
owned dataflow kernels (about 577 LOC), plus shared Halo untilize and generic
sharded-writer kernels.

Why medium:

- Both factories require a coherent conversion from manual `CachedProgram`,
  dynamic borrowed-CB rebinding, and positional argument vectors.
- Row-major and tiled paths have genuinely different participants and should
  use mode-specific specs rather than sentinel bindings.
- Shared kernels cross ownership boundaries and must either receive compatible
  semantic entry points or be replaced with operation-local equivalents.

Key proof: tiled/row-major, sharded/interleaved output, padding boundaries,
nontrivial slice starts/ends, shared-kernel cold compilation, cache rebinding,
and output immutability.

### `slice_write`

Evidence: 3 legacy factories (interleaved row-major, sharded row-major input,
and sharded tiled input), about 1,086 factory LOC, 3 owned kernels (about 312
LOC), plus the shared unary sharded reader.

Why medium:

- Three factory variants duplicate work splitting and cache-hit patching; the
  migration should first establish one semantic argument/DFB inventory and then
  express mode-specific specs.
- Strided writes and borrowed sharded input/output CBs make address refresh and
  output preservation more subtle than a unary copy.
- The local kernels are small, so after the host schemas are designed the
  mechanical device conversion is bounded.

Key proof: contiguous and strided writes, tiled and row-major sharded inputs,
untouched output regions, fresh input/output allocations on cache hits, and all
shared-reader specializations.

## Hard

### `conv2d`

Evidence: 2 `WorkloadDescriptor` factories plus a large common factory helper,
about 3,197 factory LOC, and 11 owned kernel entry files (about 2,983 LOC).

Why hard:

- This is the largest owned boundary: width-, height-, and block-sharded modes,
  depthwise paths, 1D/2D activation multicast, weight multicast, optional bias,
  tilize/untilize, fused activation, packer/L1 accumulation, and padded cores.
- Kernels contain large positional CTA/RTA schemas, dynamic NoC-coordinate
  arrays, numeric CB identities, semaphores, multicast endpoints, and specialized
  compute initializers. Converting only the factory or only the kernels would
  leave two sources of truth.
- Activation-reuse and split-reader paths directly rewrite FIFO cursor state.
  The completed Wormhole/Blackhole migration retains that zero-copy behavior
  through the explicit Gen1 plain-CB compatibility surface; activation reuse is
  rejected on Quasar, whose DFB interface does not expose cursor repositioning.
- Conv1D and ConvTranspose2D depend on this unit, so its validation surface is
  broader than the `conv2d` API alone.
- Performance sensitivity is high: wrapper boundaries, DFB identity
  constant-propagation, buffer depth, and multicast sequencing can all change
  device time or deadlock behavior.

Recommended decomposition: inventory and freeze each major factory mode first;
define semantic tensor/DFB/semaphore/config bindings; convert one complete mode
end-to-end; then migrate every remaining specialization without retaining a
descriptor/numeric-CB shadow path. Production Conv1D, Conv2D, and
ConvTranspose2D shapes need forced cold-JIT and profiler coverage.

### `max_pool2d` / `avg_pool2d` (`Pool2D`)

Evidence: 1 large `WorkloadDescriptor` factory (about 1,289 LOC) and 4 large
owned kernels (about 1,473 LOC). Both public APIs share this primitive.

Why hard:

- One implementation covers max/average pooling, optional max indices,
  count/include-pad and divisor behavior, multiple output layouts/dtypes,
  sharding choices, halo/config tensors, and multiple compute-kernel variants.
- The reader and compute kernels have very large positional schemas and
  performance-sensitive CB topology.
- `adaptive_*pool2d` and `global_avg_pool2d` inherit this migration, enlarging
  the observable-contract and regression surface.
- Public compute-config fields, pool-specific scalars, and config-tensor
  placement all require independent validation, cache-key, translation, and
  runtime-effect audits.

The migration needs adversarial pooling boundaries, return-indices correctness,
all padding/divisor semantics, production model shapes, cache refresh, and
before/after device-profiler measurements.

### `upsample`

Evidence: 4 factories, about 1,348 factory LOC, and 7 owned kernels (about 831
LOC), plus the shared untilize kernel. Three factories already use
`ProgramArtifacts`; bilinear still uses `ProgramDescriptor`.

Why hard despite the partial host migration:

- Nearest integer, nearest float, interleaved, sharded, and bilinear paths have
  different kernels, tensor/config participation, and buffering.
- None of the owned kernels uses `TT_KERNEL`. Five use the interim named
  `get_arg(args::...)` form and two remain fully positional, so every device
  entry still needs final conversion.
- The sharded path owns a generated halo lookup tensor, while bilinear has a
  compute kernel and LUT/DFB performance concerns. The four factories must end
  with one truthful cache/config contract.
- Floating scale factors and mode selection need careful cache identity and
  boundary-case validation.

Complete the three partially migrated paths rather than treating their
`ProgramArtifacts` return type as done, then migrate bilinear and validate all
layout/mode/scale specializations cold.

### `halo`

Evidence: 1 `WorkloadDescriptor` factory (about 517 LOC), 2 owned kernels (about
424 LOC), and four generated configuration tensors used across the core grid.

Why hard:

- The kernel count hides a complex topology: pad/gather configuration streams,
  local and remote shard access, multicast orientation, tiled/row-major modes,
  and untilize compute participation.
- The dataflow kernel has a large positional schema and raw NoC/semaphore/CB
  behavior. The public `remote_read` option is currently rejected inside the
  kernel with a compile-time assertion, so the host contract must explicitly
  validate/reject unsupported values or implement them.
- Halo is a dependency of Conv2D and Pool2D. Queue, ownership, or performance
  regressions propagate into both high-traffic operations.
- Config-tensor lifetime and mesh rebinding must move from workload-descriptor
  storage to semantic tensor parameters/op-owned tensors without stale cache-hit
  addresses.

Multi-stage, ping/pong, block/height sharding, transpose multicast, tiled output,
and production consumer performance are required evidence; a one-core halo test
is insufficient.

### `conv3d`

Evidence: 1 `ProgramDescriptor` factory (about 1,328 LOC) and 3 very large owned
kernels (about 2,065 LOC).

Why hard:

- The reader alone has a very large positional schema and implements vol2col,
  gather tuning, halo/offset access, and many shape specializations.
- The writer implements weight sharing by chain or multicast with multiple
  semaphore roles and per-core NoC topology. The compute kernel has extensive
  matmul/untilize specializations and runtime work selection.
- Three kernels therefore encode much more state than their count suggests;
  naming every boundary fact and drawing the DFB/semaphore topology must precede
  editing.
- Weight-sharing choices and compute configuration are performance-critical and
  must be traced through validation, cache identity, descriptor translation,
  and actual hardware behavior.

Cold-JIT must cover gather/normal reader paths, chain/multicast/no-share weight
modes, bias/no-bias, output/layout variants, edge depths, and production shapes.

## Metalium 2.0 capability summary

The current `ProgramSpec` surface supports local DFBs,
borrowed tensor-backed DFBs, DFB aliasing, semaphores, scratchpads, multiple work
units, tensor parameters, and per-node named runtime arguments. Existing Fold
and Upsample migrations also demonstrate op-owned configuration-tensor lifetime
and mesh stamping.

The following are escalation gates, not current classifications:

1. **Mesh-varying program specs.** `ProgramArtifacts` currently represents one
   `ProgramSpec` stamped across mesh coordinate ranges; the source notes a future
   `MeshWorkloadSpecFactory`. The owned `WorkloadDescriptor` factories currently
   appear to build one descriptor and copy it across ranges. If hardware testing
   proves that Conv2D, Pool2D, or Halo requires a different immutable spec per
   mesh coordinate, framework support is required before full migration.
2. **Cross-node/global semantic resources.** `KernelSpec` marks global semaphore,
   global DFB, and mesh-buffer bindings as future work, and cross-node DFBs are
   not implemented. The current owned ops express multicast/remote traffic with
   explicit NoC coordinates and local semaphores, so this is not automatically a
   blocker. It becomes one only if the acceptance policy requires that traffic to
   be represented by unavailable cross-node semantic objects.
3. **Typed array arguments.** Host schemas currently support scalar `uint32_t`
   named arguments; typed arrays are still TODO. Conv2D's variable NoC-coordinate
   lists are the clearest pressure point. An operation-local config tensor is a
   viable redesign; if that is rejected for correctness or measured performance,
   typed-argument infrastructure would be needed.
4. **Gen2 DFB ownership.** Some legacy CB topologies may reveal a producer or
   consumer role shared across compute and data-movement processors. The first
   response should be an operation-local topology change (single consumer,
   handoff DFB, or supported alias). Only a topology that cannot be expressed
   after that audit should be escalated as infrastructure work.

## Suggested migration order

1. Fold, to finish an already semantic host boundary and establish the exact
   `TT_KERNEL`/cold-JIT test pattern on this branch.
2. Convert-to-CHW, then Rotate and Grid Sample, to exercise a small legacy
   factory and shared-kernel coordination.
3. Convert-to-HWC, Padded Slice, and Slice Write, consolidating manual cache-hit
   patching into semantic rebinding.
4. Upsample, completing its three partial factories before converting bilinear.
5. Halo, then Pool2D, so the shared sliding-window/config topology is proven
   before migrating pooling.
6. Conv3D and Conv2D last, after the smaller migrations have established current
   API patterns, validation commands, and performance baselines.

## Definition-of-done reminder for every rated unit

A classification is not reduced when a factory already returns a modern-looking
descriptor. Completion still requires all of the following:

- a single semantic `ProgramSpec`/`ProgramRunArgs` host boundary;
- `TT_KERNEL` and named parameters for every participating owned or shared
  kernel specialization;
- semantic tensors, DFBs, semaphores, endpoints, and current device APIs with no
  unjustified positional/raw/numeric shadow path;
- truthful public configuration, cache identity, and cache-hit rebinding;
- frozen correctness, output, determinism, trace, and performance contracts;
- same-key tests using freshly allocated inputs and outputs;
- isolated forced cold-JIT evidence for every meaningful specialization,
  including zero unexplained migration-owned warnings;
- real-hardware correctness and device-profiler measurements, followed by
  formatting/hooks and `git diff --check`.

## Implementation status (2026-08-22)

This document's ratings and ordering record the original static assessment.
The branch has since completed the convolution migration sequence, including
Halo, Pool2D, Conv3D, and Conv2D as separate per-operation commits. Conv2D
evolved its eleven canonical kernels in place, uses named constexpr arguments
without feature macros, and preserves Gen1 FIFO compatibility without adding
CB/DFB storage or copies. Quasar validation is intentionally outside scope.
Detailed implementation and validation evidence is recorded in
`metalium-2.0-convolution-ops-migration-learnings.md`.
