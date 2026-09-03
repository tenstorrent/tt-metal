---
description: 'PR review rules for eltwise operations (unary, binary, ternary, backward, complex, quantization) and their tests'
applyTo: 'ttnn/cpp/ttnn/operations/eltwise/**,tests/ttnn/unit_tests/operations/eltwise/**,tests/ttnn/nightly/unit_tests/operations/eltwise/**,tests/ttnn/unit_tests/gtests/udm/eltwise/**,tests/sweep_framework/sweeps/eltwise/**,tt_metal/hw/inc/api/compute/eltwise_unary/**'
excludeAgent: "cloud-agent"
---

# Eltwise Operations Review

These rules are additive to `ttnn-ops.instructions.md` (op implementations) and
`python.instructions.md` (test style). Apply those first, then the eltwise-specific
rules below.

## 🔴 CRITICAL

- **Flag the required CI runs.** Any PR that touches eltwise host code, eltwise
  device kernels, or eltwise tests must state in its description that the eltwise
  pipelines were run, with links to the runs. Post a comment asking for them if the
  description does not have them. The two pipelines are:
  - **Sanity** — `Sanity tests nightly debug run`
    (`.github/workflows/sanity-tests-debug.yaml`), which runs the whole
    `tests/pipeline_reorg/ttnn_sanity_tests.yaml` matrix including the four
    `ttnn eltwise group 1..4` entries over
    `tests/ttnn/unit_tests/operations/eltwise`.
  - **L2 nightly** — `Nightly tt-metal L2 tests`
    (`.github/workflows/tt-metal-l2-nightly.yaml`), dispatched with
    `additional_test_categories: eltwise`, which runs the four
    `ttnn nightly eltwise tests group 1..4` entries
    (`category: eltwise`) in `tests/pipeline_reorg/ops_unit_tests.yaml` over
    `tests/ttnn/nightly/unit_tests/operations/eltwise`.

  Neither pipeline runs automatically on a PR — the PR gate and merge gate only run
  smoke-level tests, and `PR - Sanity tests` is gated to forks. Both must be
  dispatched by hand, so a PR with no linked runs has not been covered. This applies
  to draft PRs as well: ask for the runs, do not wait for the PR to be marked ready.
  A PR that changes only comments or documentation is exempt.

- **Scalar-variant dtype coverage.** Every dtype a unary or binary op newly accepts
  must have its case wired up in `pack_scalar_runtime_arg`. A dtype added to the
  nanobind/validation list but missing from the scalar packing path silently
  produces garbage on the tensor-scalar call. Each conversion needs a short inline
  example comment showing how a Python scalar maps to the packed bytes for that
  dtype, so the packing can be checked by eye.

- **Scalar precision for integer dtypes.** Do not route an integer scalar through
  `float`. `static_cast<float>` loses bits above 2^24, so integer binary ops must use
  integer arithmetic or an int64 intermediate. Any fast, native, or short-circuit
  scalar path must still honor an explicitly-requested `output_dtype` and
  `memory_config` rather than falling back to defaults.

## 🟡 IMPORTANT

- **Exercise the tensor-scalar (TS) variant.** New or modified tests in the binary
  and ternary infrastructure must cover TS alongside TT — TS is the variant most
  often skipped, and scalar-path bugs slip through when it is. Where the surrounding
  file already parametrizes over variants (`TT`/`TS`/`ST`, or `TTT`/`TTS`/`TST`), a
  new test that covers only one variant should say why in the PR description.

- **Program-cache tests must assert the cache actually changed.** A test that just
  calls an eltwise op twice and compares against a golden will pass whether or not
  the program cache misbehaved. Assert on `device.num_program_cache_entries()` so
  the test distinguishes "two entries, both correct" from "one entry reused
  correctly" from "one entry reused incorrectly".

- **Broadcast strides vs. the cache key.** Eltwise broadcast strides depend on
  individual leading dimensions, while `compute_program_hash` frequently coarsens
  shapes to a volume. Two shapes with the same volume and different leading dims
  (e.g. `[4,1,32,32]` and `[1,4,32,32]`) can collide. When a PR changes shape
  handling, broadcast handling, or `compute_program_hash`, check that everything a
  runtime arg is derived from is in the key — including the output shape.

- **Binary dtype guards must inspect both operands.** A guard keyed on dtype in a
  binary op has to evaluate both inputs. Reuse `is_floating_point(DataType)` from
  `tensor_types.hpp` rather than enumerating dtypes; enumerated lists go stale as
  dtypes are added.

- **State dtype omissions.** When an op gains a subset of the integer or
  low-precision dtypes, the PR description should say which are covered and which
  are deferred, and why. Silent omissions read as oversights.

- **Template over data format rather than duplicating bodies.** New dtype variants
  in SFPU kernels should be handled by templating on the data format with
  compile-time dispatch, not by copying the function body per format.

- **Gate dtype-specific circular buffers.** A staging CB that exists only for one
  dtype or one mode must be allocated only in that case; unconditional allocation
  charges L1 to every other dtype.

## 🟢 SUGGESTION

- **New accuracy tests: prefer ULP over PCC.** A newly added test that compares an
  eltwise result against a golden reference should use a ULP (unit in the last
  place) bound rather than a Pearson correlation coefficient. PCC on a narrow input
  range hides real numerical error. This is a suggestion, not a merge blocker, and
  does not apply to existing PCC tests a PR happens to touch.

- Parametrize accuracy thresholds (e.g. a `ulp_threshold` argument defaulting to 3)
  in shared assertion helpers instead of hard-coding them, so a shared helper cannot
  silently loosen a test that had a stricter bound.

- Test ranges should cover the full domain of the dtype under test, especially
  around signed/unsigned boundaries, and should be asymmetric between the two inputs
  of a binary op so operand-order bugs are visible.

- When a PR removes shapes or dtypes from a parametrization, ask where the removed
  coverage now lives, and have the PR note it.

- Prefer `@pytest.mark.parametrize` over a manual for-loop. The exception is a test
  that deliberately shares one program cache across several configurations — there
  the loop body is the thing under test, and that intent should be stated in a
  comment.

## Review Checklist

- [ ] PR description links a Sanity run and an L2-nightly (`eltwise` category) run
- [ ] New dtypes are wired into `pack_scalar_runtime_arg` with an inline example comment
- [ ] Integer scalars are not routed through `float`; fast paths honor `output_dtype`/`memory_config`
- [ ] TS variant exercised alongside TT for binary/ternary test changes
- [ ] Program-cache tests assert `num_program_cache_entries()`, not just numerics
- [ ] Everything a runtime arg derives from — including output shape — is in `compute_program_hash`
- [ ] Dtype guards on binary ops inspect both operands via `is_floating_point`
- [ ] Covered vs. deferred dtypes stated in the description
- [ ] New SFPU dtype variants templated on data format, not duplicated
- [ ] Dtype-specific CBs gated on that dtype
- [ ] New accuracy tests use ULP rather than PCC (suggestion only)
