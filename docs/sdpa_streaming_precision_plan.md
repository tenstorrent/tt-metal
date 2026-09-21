# Streaming SDPA precision integration — PR 1

Status: recipe/evidence freeze, clean branch setup, and internal numerical-policy
and compatibility-resolver foundation implemented and built on the new base.
The first shared streaming-buffer helpers are extracted and device-checked.
New numerical recipes are not yet ported or qualified. No new public recipe
interface or dispatch is enabled. Public enum/helper names below are intentionally
not prescribed until the interface implementation is reviewed.

## Immutable inputs to the port

- Production base: `dfaf6dc802f0a1321bbb2578ba7c4a0fb9b71ab8` (main fetched
  2026-09-21).
- Research snapshot: `e13f445161ad598de9700edf915e5ed7faa6dc34`.
- Annotated local tag: `sdpa-recipes-20260921-v1`.
- Research snapshot branch: `cglagovich/sdpa-recipes-frozen-20260921`.
- Snapshot manifest and verifier:
  `experiments/sdpa-l2/recipe-freeze-v1/{manifest.json,freeze.py}` at that tag.

The snapshot and production branch have different purposes and histories.
Do not merge/cherry-pick the entire snapshot into this branch. It contains
experimental implementations, historical results and model instrumentation.
The production patch series should contain only selected implementation,
maintainable tests and documentation. Research media/results stay on the
snapshot branch. The refs are local until explicitly published.

## Frozen selection

| Research ID | QK / PV | Destination / recurrent state | Input contract |
| --- | --- | --- | --- |
| D | HiFi4 / HiFi4 | FP32; full-FP32 subtraction and accurate exp path | Ordinary BF16 QKV |
| C | HiFi4 / HiFi2 | FP32; selected cheaper subtraction and matched cubic exp | Ordinary BF16 QKV |
| B | HiFi2 / HiFi2 | BF16; compensated numerator/denominator state | Ordinary BF16 QKV |
| A | HiFi2 / HiFi2 | Original uncompensated BF16 streaming baseline | Ordinary BF16 QKV |
| E_bf16 | LoFi / LoFi | BF16; selected compensated E algorithm | Q RNE7 and KV RNE5, stored BF16 |
| E_bfp8 | LoFi / LoFi | Same E algorithm | Q RNE7 BF16; KV RNE5 then BFP8 pack |
| E_bfp4 | LoFi / LoFi | Same E algorithm | Q RNE7 BF16; KV final-grid BFP4 RNE with saturation |

Significand bit counts include the leading bit. B requires no external
rounding. Former E/G map to E_bfp8/E_bfp4; F is dropped. Plain-input E tests
are diagnostic controls, not new recommended recipes. Frozen A is a measured
research reference, not a promise that every caller/default on current main
is numerically identical to it.

Authoritative definitions live in the snapshot, not this abbreviated table:

- `experiments/sdpa-l2/flux2-frontier-v1/device_attention.py` defines base
  recipes and preparation.
- `experiments/sdpa-l2/compute-sprint-v3/pareto/collect.py` selects the latest
  qualified D/C/B/A implementations and their exact flags.
- `experiments/sdpa-l2/compute-sprint-v3/kv-precision-v1/bench.py` defines the
  three E storage/preparation variants using the same grouped-state kernel.
- `experiments/sdpa-l2/recipe-freeze-v1/manifest.json` pins selected sources,
  evidence and plots; the parent Git tree preserves committed dependencies.

The latest optimized geometry is Blackhole, noncausal/unmasked Q256/K512/D128,
BF16 output. Do not infer arbitrary head counts, tails, shapes, GQA, ring,
causal or architecture support from that specialization's tests. Preserve the
actual per-family input-buffer depths and state-bank capacities. Expand
eligibility only with explicit implementation and qualification.

## Interface and implementation constraints

1. Add optional named numerical recipe selection; preserve all old signatures,
   omitted-config defaults and explicit compute/program configurations when
   it is absent. Existing configurations need not map to these seven points.
2. Keep numerical policy separate from scheduling, chunk/grid selection and
   memory placement. Reject conflicting explicit numerical controls instead
   of assigning silent precedence.
3. Use one maintained streaming framework with focused numerical/state
   policies. Do not introduce seven copied compute headers or expose the
   research preprocessor-flag matrix as the public interface.
4. Input storage and preparation must be explicit. BF16/BFP8/BFP4 dtype alone
   cannot prove that E's required rounding was performed. Provide a deliberate
   preparation path and a documented prepared-input contract; avoid hidden
   re-quantization of caches and repeat preprocessing on trace replay.
5. Explicit recipes must reject unsupported configurations on the host, before
   device launch. No fallback to a different recipe or silent fidelity change.
   Unmodified callers retain their existing compatibility path while coverage
   is migrated; PR 1 does not delete required legacy coverage.
6. Cache/program identity must include every effective numerical policy and
   format distinction. Runtime addresses and trace-varying offsets must remain
   runtime values, not stale captured constants.
7. Preserve output dtype/layout/placement contracts. E's BF16 Q preparation
   must not silently change an existing all-BFP8 caller's output contract.

## Internally staged commits, one implementation PR

This document is the first branch-setup commit. The remaining stages are not
complete and should remain separate reviewable commits within PR 1:

- [ ] Refresh the affected main-side dispatch/default tests and establish a
  current-main device baseline. The September 16 usage audit is historical.
- [x] Introduce numerical-policy types, validation and compatibility resolver;
  test omitted versus explicitly constructed configs and conflicting knobs.
- [ ] Extract shared streaming primitives and FP32 state support; integrate
  D/C with their exact exp/subtraction/normalization decisions.
- [ ] Integrate B compensation and E's shared compute path, with explicit
  state ownership, changed-max handling, final flush and input preparation.
- [ ] Transfer qualified scheduling optimizations only behind their supported
  geometry/lifetime guards. Rejected experimental branches, debug counters,
  global monkeypatches and research-only include overrides are not retained.
- [ ] Add production-path accuracy, trace/cache-hit, boundary and performance
  regression tests, plus concise user documentation and model opt-in examples.
- [ ] Build on the actual new base and qualify the extracted implementations
  against both the frozen reference and current main.

Current-main checks at branch creation:

- `sdpa.cpp` still calls `init_device_compute_kernel_config` with HiFi2 as
  the omitted-config fidelity for ordinary dense SDPA.
- `sdpa_program_factory.cpp::can_use_streaming_compute` is still
  `!fp32_dest_acc_en`; FP32 streaming is substantive new integration work.
- Main contains newer dynamic logical-length and profiler changes. Preserve
  these rather than overwriting the current kernel with an older research copy.

### Foundation implementation and validation (2026-09-21)

`sdpa_precision_policy.hpp` records the seven frozen numerical identities.
`sdpa_numerics.cpp` resolves either the existing compute configuration or an
internal explicit recipe. The ordinary dense entry point uses only its legacy
branch. Other entry points, program configuration, public signatures and device
dispatch are unchanged. Explicit recipe resolution is not proof of feature or
geometry support and is not reachable from the public operation yet.

The compatibility branch preserves the difference between omitted config
(HiFi2) and an explicitly empty `ComputeKernelConfig` (LoFi), every existing
compute-config field, and the independent exponential-approximation setting.
The Python binding is a separate compatibility case: an empty
`WormholeComputeKernelConfig()` currently sets `MathFidelity.Invalid`, not LoFi.
Do not silently reinterpret it as the C++ default. Its constructor contract is
tested without launching an invalid-fidelity kernel; device default-equivalence
tests explicitly request LoFi or omit the entire config.
Explicit recipes reject a simultaneous compute config or `exp_approx_mode=false`
instead of silently overriding it. C retains separate QK/PV fidelity intent;
D's accurate softmax cannot be represented by simply flipping the generic
approximation booleans. Device policy integration remains required for both.

Initial partial validation (superseded by the full build below):

- All six policy GoogleTests passed on macOS with pinned GoogleTest v1.13.0.
- New resolver and resolver tests passed a C++20 syntax check with warnings
  treated as errors.
- All twelve policy/resolver GoogleTests passed in the reserved Blackhole
  Linux container, compiling the new sources with clang 20 and warnings as
  errors. This standalone executable linked the existing `_ttnncpp.so` for
  `init_device_compute_kernel_config`; it is partial host validation, not a
  complete build or device validation of this branch.
- `git diff --check` passed.

The first full configuration attempt used this branch's exact base and pinned submodules
in an independent remote worktree. It failed downloading Boost because of
container DNS/connectivity. Transferring the SHA256-verified pinned Boost archive
locally advanced configuration, which then failed fetching protobuf. Those
initial attempts did not establish a full build or device result.

### Infrastructure recovery and first shared helpers (2026-09-21)

A fresh reservation on `yyzo-bh-04` restored dependency access. The full network
home filesystem required both ccache and the firmware/device JIT cache to move
to task-specific `/localdev` directories. No existing user data was deleted.

The production branch then built successfully with its exact main base and
pinned submodules, including TTNN bindings and tests. All 13 registered
policy/resolver tests passed against the newly built libraries, replacing the
earlier standalone/old-library validation limitation.

`compute/streaming/circular_buffer.hpp` now owns the shared out-of-order pack,
outlined CB publication/consumption, and retained-write-origin publication
helpers. Their bodies, linkage, and inlining attributes are unchanged. The
shared include replaces the definitions in `compute_streaming.hpp`; dataflow,
formats, arithmetic, scheduling, and dispatch are unchanged.

The compatibility suite passed before and after extraction (11 cases each),
with all 10 device output hashes identical across independent JIT caches. It
also passed with Watcher/device assertions enabled. Existing prefill tests
passed 8 cases with 2 pre-existing skips. These are Blackhole P100 checks of the
legacy paths, not qualification or performance measurements of the new recipes.
See [validation details and commands](sdpa_streaming_precision_validation.md).

## Merge gates

### Numerical and state correctness

- Test the same original BF16 inputs against an independent FP64 reference;
  report relative L2 and PCC, row-error tails and absolute errors near zero.
- Reproduce the frozen broad suite: normal, clipped, scaled Q/K, outliers and
  uniform attention at KV4K/32K/256K with 256 query rows and D128.
- Keep common-mode, cancellation, constant/zero V, tiny-update and changed-max
  stress separate and visible. They are diagnostic failures where documented,
  not evidence that a recipe meets an absolute threshold on every input.
- Cover K1/K2/K3, odd final group, multiple query jobs, state-bank reset,
  identity-to-changed-max transitions, exact CB capacities and pack-mode reset.
- Preserve exact external preparation semantics with independent quantizer
  checks. Ordinary casts are not equivalent to E's preparation.
- Preserve bitwise behavior where promised. Where accepted v3 reassociation
  is retained, the recorded regression gate is per-case L2 <= 1.05 × frozen
  baseline L2 + 0.0001 percentage points, plus the frozen zero-reference gate.
  This is not a universal absolute-accuracy guarantee.

### Execution and performance

- Eager, repeated actual trace replay and program-cache-hit tests must agree
  as specified; original/prepared input tensors must remain unchanged.
- Invalid recipe/feature/dtype combinations fail before device launch.
- Measure resident useful QK+PV TFLOP/s/core separately from distinct-input
  device time and end-to-end time including input preparation.
- Frozen resident optimizations favor unchanged maxima. Check changing-max
  and shorter-context regressions before selecting dispatch conditions; no
  universally optimal context threshold has been established.
- Existing FLUX/Wan results predate the final v3 implementation and do not
  qualify this newly extracted source. Rerun representative model tests.
- All C++/binding/build-system changes must build; device kernels must JIT
  compile and execute. Old libraries are not validation of new host code.

### Compatibility and scope

- Preserve mixed storage, independent math/exp approximation controls and
  existing non-preset fidelity/destination combinations through legacy config
  resolution. Do not force callers into the nearest recommended recipe.
- Preserve helpers shared by sparse attention, decode and CCL. Decode remains
  outside this migration but must continue to compile.
- Wormhole/Quasar and remaining joint/ring/paged/chunked/MLA feature migrations
  are not established by Blackhole dense qualification. Unsupported explicit
  recipes must be clear; supported old calls retain their prior behavior.

## Follow-on PRs

PR 2 expands feature/platform coverage and migrates actual callers where
justified by model evidence. PR 3 removes non-streaming loops only after no
in-scope supported configuration selects them. Extract reusable utilities
rather than deleting `compute_common.hpp` wholesale. Independent training
attention and decode are not part of that deletion.
