---
description: 'PR review for tt-train — training loop correctness, gradient flow, optimizer safety, kernel quality, and config stability'
applyTo: 'tt-train/**'
excludeAgent: "cloud-agent"
---

# TT-Train Review

## 🔴 CRITICAL

- **Gradient accumulation correctness**: if a change touches backward pass or gradient accumulation, verify that gradients are not silently zeroed, double-counted, or accumulated with wrong scaling.
- **Optimizer state mutation**: optimizer state tensors (momentum, variance buffers) must be updated in-place consistently. A change that allocates a new tensor instead of updating in-place will silently ignore accumulated statistics.
- **Config backward compatibility**: renaming or removing fields in YAML training configs breaks downstream consumers (tt-training-service, console). Config name changes require a deprecated-alias mapping that keeps old names working during a transition period (30–60 days minimum).
- **Training script exit codes**: training scripts and model example scripts (`examples/`, `models/`) must propagate non-zero exit codes on failure. A `main()` that catches all exceptions and exits 0 makes CI appear green on a broken training run — silent success is a footgun in automated pipelines.
- **`ttnn.all_reduce` tensor placement**: after `ttnn.all_reduce` in multi-device training code (DDP, GRPO, RL loops), verify that the output tensor's `.placement[cluster_axis]` is set to `Replicate`. Missing this causes incorrect gradient synchronization across devices with no error message.

## 🟡 IMPORTANT

- **Op additions** (`tt-train/sources/ttml/ops/`): verify the op has a corresponding backward pass if it will appear in a differentiable context.
- **Device op struct naming**: device operation parameter/input structs must have unique, descriptive names (e.g., `MoeGroupAttributes`, `MoeGroupTensorArgs`) — not generic names like `Attributes` or `TensorArgs`.
- **Test coverage** (`tt-train/tests/`): new training utilities must have at least one test that verifies numerical correctness, not just that code runs without exception. Tests with meaningless tolerance checks (e.g., `abs_max` on MoE routing where bf16 diffs flip expert selection) should be flagged.
- **Test hygiene**: no debug prints, no oversized comments, no temporary tests pushed to main. Tests must be independently runnable and not assume execution order.
- **`ttnn` API stability**: tt-train must track stable ttnn APIs. Flag any use of an `experimental` ttnn op without a comment acknowledging instability risk.
- **Performance evidence**: changes to training kernels or ops that claim perf improvement must include step-time or loss evidence (even a brief summary) in the PR description.
- **Stale comments in kernels**: flag comments that describe old behavior after refactoring. Kernel comments about buffer usage, broadcast modes, or dataflow patterns must match the current implementation.
- **Inf/NaN vs tolerance failures**: SDPA, softmax, and cross-entropy tests that fail with inf/NaN values cannot be fixed by loosening `xt::allclose` tolerances — those indicate overflow (e.g., softmax accumulation past the bf16 range), not precision drift. Do not suggest loosening tolerances on inf/NaN failures; flag them as a separate correctness bug.
- **CI tier placement for new tests**: when a PR adds a new test to `tt-train/tests/`, confirm it targets the right tier — smoke (merge-gate, N150 only, fast C++ subset), unit (L2 nightly, all hardware: N150/N300/P100/P150), or perf (L2 nightly Python models). New tests must not be added to `sanity-tests.yaml`; tt-train tests were deliberately removed from sanity.
- **C++ test device sharing**: C++ tests should use a shared device fixture (one `tt::tt_metal::IDevice*` open for the whole test suite) rather than open/close per test — the per-test pattern was measured at ~2.5× slower and wastes hundreds of hours of N300 machine time per week. Python tests that exercise different mesh shapes/topologies are exempt.

## Kernel Review (`tt-train/sources/ttml/metal/`)

- **Naming clarity**: kernel variables like "block", "chunk", "step" must be used consistently. If `total_blocks` actually means `active_steps * num_chunks`, rename or add a clarifying comment.
- **Overlap opportunities**: when a kernel reads then computes sequentially, ask whether double-buffering or pipelining (read chunk N+1 while computing chunk N) is feasible.
- **Dest reuse**: if a compute kernel packs an intermediate to a CB only to immediately unpack it for the next op, flag the opportunity to use `binary_dest_reuse_tiles` to keep the result in DST registers.
- **Shared dataflow utilities**: before writing new tile-read loops, check if `read_tiles_by_row()` or similar utilities in `dataflow_utils.hpp` already cover the pattern. If the kernel has special needs (strided reads, mcast, transpose-on-read), document why the fork is necessary.
- **Unnecessary init/uninit pairs**: flag redundant `_init` / `_uninit` calls in compute kernels. Reconfigs must match the actual CB data formats being used — wrong source/dest CB in a `reconfigure_*` call is a silent correctness bug.
- **Scalar constants**: prefer `add_unary_tile(idst, param)` over generating an entire constant tile in a reader CB when only a scalar value is needed.
- **Validate device tensors early**: ops must check that input/output tensors have allocated DEVICE storage, live on the same device, and have compatible shape/dtype/layout before launching kernels. Don't rely on kernel-level crashes for validation.
- **Config and magic number comments**: explain why specific block sizes, subblock dimensions, or mcast grid orientations are chosen. Link to perf rationale (e.g., "2x2 subblock to fill all 4 FP32 DEST regs").

## 🟢 SUGGESTION

- Training ops with numerically sensitive behavior (softmax, layernorm, loss functions): include a comment citing the reference implementation or paper equation.
- Prefer `ttnn.bfloat16` explicitly over implicit dtype promotion in training kernels.
- Extract repeated logic into shared helpers — if the same pattern appears in multiple trainers/examples, factor it out.
- Extract repeated compute sequences in kernels into local helper functions (DRY).
- Use shared test utilities (`test_utils/random_data.hpp`, `make_uniform_xarray`) for test data generation.
- For ops claiming program-cache reuse, add a cache-hit test variant that exercises `override_runtime_arguments` with changing parameters.
- APIs must not silently ignore parameters — if a function accepts `mask` but never uses it, either remove the parameter or document the limitation.
- For numerical computation in training ops, prefer the two-pass variance form `E[(x − mean)²]` over the algebraically equivalent `E[x²] − E[x]²` — the single-pass form suffers catastrophic cancellation in bfloat16. Note the accuracy/performance tradeoff explicitly in the PR description if the slower form is chosen.
- When including perf evidence, use the standard L2 nightly metrics: `last_loss`, `mfu`, `average_iteration_time_ms`, `step_time_p50`. Compare to the latest scheduled main run. MFU differences of <1–2% between runs may be host-CPU noise rather than a real regression.

## Review Checklist

- [ ] Gradient accumulation logic correct (no silent zero or double-count)
- [ ] Optimizer state updated in-place, not reallocated
- [ ] New ops have backward pass if used in differentiable context
- [ ] New utilities have numerical correctness test (not just smoke test)
- [ ] Config field renames include deprecated-alias backward compatibility
- [ ] Device op structs have unique descriptive names
- [ ] Kernel comments match current implementation (no stale descriptions)
- [ ] No experimental ttnn ops without stability acknowledgment
- [ ] Training/model scripts propagate non-zero exit code on failure
- [ ] `ttnn.all_reduce` output tensor placement set to `Replicate` on all cluster axes
- [ ] New tests assigned to correct CI tier (smoke/unit/perf); not added to sanity-tests.yaml
- [ ] C++ tests use shared device fixture, not per-test open/close
