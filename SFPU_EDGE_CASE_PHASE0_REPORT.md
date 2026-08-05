# Phase 0 — scope and status (complete)

Part of [#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739).
Plan: [SFPU_EDGE_CASE_EXPANSION_PLAN.md](SFPU_EDGE_CASE_EXPANSION_PLAN.md) §3 · Audit: [SFPU_EDGE_CASE_COVERAGE.md](SFPU_EDGE_CASE_COVERAGE.md) §0

## Why Phase 0 exists

The audit's finding #1: `test_eltwise_unary_sfpu_float` passed no `spec_A`, so `generate_stimuli`
fell back to `default_spec_for_format` = `uniform(0.1, 1.1)`. All 31 ops on `ALL_MATHOPS` were
only ever fed small positive inputs — no `x<0` branch, no piecewise knee, no saturation tail, no
argument reduction. The per-op signed domains in `_OP_DOMAIN_REGISTRY` already existed but were
reachable only from `test_eltwise_unary_sfpu_domain`, whose op list is **disjoint** from
`ALL_MATHOPS`, so none of those 31 registry entries had ever executed.

The original plan scoped this as "≈0 new lines: reroute the default". The reroute is 5 lines.
Making it pass took four more commits, because inputs that benign cannot distinguish a correct
golden from a wrong one, a correct comparison from a wrong one, or an accurate kernel from an
inaccurate one. **Phase 0 is therefore: reroute the unary sweep onto the registry, and fix
everything the wider domain exposes.**

## What Phase 0 covers, and its status — all nine items done

| | Item | Status |
|---|---|---|
| 0a | Default `eltwise_unary_sfpu`'s `spec_A` to the op's registered signed domain minus its undefined ranges | ✅ `f8590d5` |
| 0b | Register the 5 sweep ops with no registry entry (`Tanhshrink`, `Floor`, `Ceil`, `Trunc`, `Frac`); make a missing entry a hard `KeyError`, not a silent fallback to the default being removed | ✅ `f8590d5` |
| 0c | Bound the domain by `narrowest_range_format(input, output)` — the sweep pairs every input format with every output, and the **output** is often the binding constraint | ✅ `f8590d5` |
| 0d | **Golden:** `UnarySFPUGolden` skipped Bfp8_b when quantizing inputs to the unpack format, so for Bfp8_b the golden ran on values the hardware never saw | ✅ `14f2133` |
| 0e | **Comparison:** `passed_test` treated Bfp8_b as a scalar format, not a block one; now accepts either the lattice or the tolerance criterion | ✅ `14f2133`, `a7c0312` |
| 0f | **Recalibrate** the three registry domains that had never executed and were bounded for range rather than accuracy: `exp` 80→16, `exp2` 100→23, `reciprocal` format-sensitive | ✅ `8841e51` |
| 0g | **Record** the residual genuine accuracy limit as an xfail instead of loosening a tolerance | ✅ `507e229` |
| **0h** | **Verify on Blackhole.** 0e/0f are calibrated against *measured Wormhole error*, so neither was known to be arch-independent; had BH wanted different bounds, the registry entries would have needed an arch axis. | ✅ **green, no arch-sensitivity needed** |
| **0i** | **Merge `ALL_MATHOPS` into `DOMAIN_MATHOPS`**, collapsing the two unary drivers into one (`a5bd408`). Both now read the same registry, so two lists (31 + 63) and two nightly tests are duplication — and an op added to the wrong list silently got the wrong format coverage. Not a concatenation: the lists differ on formats (`_domain` was Float16_b + Float32 only) and on the fast/approx-mode product. | ✅ **merged, provably behaviour-preserving** |

Out of scope by definition: binary/ternary/scalar (Phase 1) and any deliberate edge-value
injection (Phases 2–4) — Phase 0 only widens the *random domain*, so it lands *near* knees,
never *on* them.

## Detail on 0d–0g — three defects finding #1 had been masking

- **`UnarySFPUGolden` did not quantize Bfp8_b inputs.** It inlined a partial copy of
  `quantize_input_to_unpack_format` handling Bfp2_b/Bfp4_b/MX but not Bfp8_b. Smooth ops absorbed
  the discrepancy inside tolerance; discontinuous ops turn a sub-ULP quantization step across an
  integer into a full 1.0 error. `floor`/`ceil`/`trunc`/`frac`: 16 failures each → 320/320.
- **`passed_test` let Bfp8_b fall through to a flat `isclose`** while its Bfp4_b/Bfp2_b siblings
  were block-aware, even though Bfp8_b is equally a block format (7 magnitude bits against an
  exponent shared across 16 elements). Routing it to the lattice check *alone* was wrong in the
  other direction: 7 mantissa bits make a lattice step *tighter* than the SFPU's own approximation
  error, so it fixed 64 `Square` + 48 `Rsqrt` variants and broke 20 `Gelu` + 40 `Silu`. The two
  criteria describe different regimes — quantization dominates far below the block max,
  approximation error dominates near it — so a result satisfying **either** is accepted. That is a
  superset of the pre-change behaviour, so it cannot regress a test that passed before.
  108 failures → 38.
- **Three registry domains were range-correct but accuracy-wrong.** `exp`: relative condition
  number equals the argument, so the shared approximation's error grows linearly with x; at the
  old high of 80 the largest outputs land 11–13% off against a 5% rtol (4.449e26 vs 3.989e26),
  visible only at `dest_acc=Yes` where no 16-bit dst rounds both sides back together. `exp2`: the
  same argument-16 ceiling lands at 16/ln2 ≈ 23. `reciprocal`: 1/x passes input relative error
  straight through, and `[0.1, 100]` puts a 1000:1 ratio inside a 16-element block — on Bfp8_b
  that left 13/1024 elements outside tolerance with a max difference of `inf`, because the
  smallest inputs quantize to zero. Holding block-float inputs to 10:1 leaves none.

**One genuine kernel limit survived and is recorded rather than tolerated.** Approximate `exp`
overshoots the golden by a systematic ~5.7% (peak 6.75%) once its argument passes ~8; the smallest
output breaching the default 5% rtol is exactly `exp(8.00) = 2976`. Nothing had measured this
before — the sweep never fed `exp` an argument above 1.1. Rather than loosen a tolerance or shrink
the domain away from the region where an approximation is most worth testing, the three affected
`(input, output, dest_acc)` combinations are listed exhaustively and marked via
`request.node.add_marker` so the case still **executes** and reports XPASS if the approximation
tightens. The one-directional bias is a suspicious shape for approximation noise and may be worth
a look on the kernel side.

## Detail on 0h — Blackhole

`test_sfpu_unary.py` on Blackhole (p300a): **4270 passed, 1174 skipped, 4 xfailed, 0 failed.**
The recalibrated `exp`/`exp2`/`reciprocal` domains and the Bfp8_b either-criterion comparison
hold on BH unchanged, so **0f's constants stay arch-independent** and no registry entry needs
an arch axis — the outcome that was not knowable without running it.

BH skips far more than WH (1174 vs 334) and reports 4 xfails against WH's 6. Both differences
are the two existing architecture guards, not lost coverage: on BH at `dest_acc=No`,
`_skip_bh_unsupported_float_combo` excludes a Float16 input or `Float32->Float16` and
`_skip_bh_unless_fp32` allows only `Float32->Float32`. Two of the three
`_APPROX_EXP_ACCURACY_XFAIL` combinations have a Float16 input at `dest_acc=No`, so they are
skipped before the xfail applies; the approximate-`exp` limit reproduces on BH wherever it is
reachable.

## Detail on 0i — how the merge was verified

Behaviour-preserving by measurement, not by inspection:

- **Same case set** — collection is 5448 tests before and after, 5368 in the merged sweep:
  4864 broad + 504 standard, matching the two old tests exactly.
- **Same stimuli** — the old domain test passed `spec_A` keyed on the *input* format; the
  driver default resolves through `narrowest_range_format(input, output)`, which for the
  standard profile's four format pairs ties and resolves to the input (Float16_b and Float32
  share bfloat16's exponent range). Verified for all four. The broad profile already used the
  driver default.
- **Same hardware result** — Blackhole, before and after: 4270 passed, 1174 skipped, 4 xfailed.
  A per-`(op, outcome)` comparison across all 94 ops and 5368 cases is **identical**.

The merge also closes a silent failure mode: the lists are now coverage profiles
(`BROAD_SWEEP_OPS` / `STANDARD_SWEEP_OPS`), and `_assert_sweep_profiles_disjoint()` fails at
**collection** if an op sits in both or has no registered domain. Previously an op inherited
whichever envelope its author happened to pick a list from, unchecked.

## Result

Wormhole: **5108 passed, 334 skipped, 6 xfailed.**
Blackhole: **4270 passed, 1174 skipped, 4 xfailed.**
31 ops gained their negative branch, piecewise knees and saturation tails; 31 dead registry
entries became live; one nightly test and its duplicated skip paths deleted.

## Blast radius of 0d / 0e, and what is verified

0e changed `passed_test`'s Bfp8_b path, which **26 test files** can reach. Two things bound the
risk:

- **It cannot regress.** Before, Bfp8_b had no dedicated branch and fell through to
  `is_valid = is_close | is_nan`; it is now `is_close | is_nan | lattice`. A strict superset, so
  any element that passed before still passes. The separate PCC gate is unchanged, so gross
  errors are still caught.
- **The residual risk is masking, not breaking** — a lattice-adjacent Bfp8_b error is now
  accepted where it previously was not. That is the intent of the change, and re-running tests
  cannot probe it (they pass either way). It is a judgement call to review, not a test gap.

Sanity-checked anyway on the four sibling suites most likely to surface an interaction —
`test_sfpu_binary.py`, `test_sfpu_ternary.py`, `test_eltwise_unary_datacopy.py`,
`test_sfpu_binop_scalar.py` on Blackhole: **1551 passed, 466 skipped, 3 xfailed, 0 failed.**
0d is narrower — it touches `UnarySFPUGolden` only, and unary is fully verified on both arches.

## What is NOT verified

- **Wormhole was not re-run for this work.** The WH figures above come from the branch's own
  earlier commit messages; this host has only a Blackhole p300a. Blackhole is first-hand,
  Wormhole is carried forward.
- **One Blackhole board variant** (p300a). Other BH boards unexercised.
- **`WITH_COVERAGE` builds.** 0i restructured the coverage skips — merging the two per-op unroll
  lists and replacing the `@skip_for_coverage` decorator with an in-body profile skip. The
  semantics are preserved by construction (see the commit), but no coverage build was run.
- **CI.** These are `@pytest.mark.nightly` and were run directly, not through the repo's
  `--compile-producer` / `--compile-consumer` two-phase flow that CI uses.
- **The other 22 Bfp8_b-capable suites**, beyond the four above.

## Notes for reviewers

- 0d and 0e are **shared infrastructure**: they affect every suite producing a Bfp8_b output, not
  just unary. See the blast-radius section above.
- 0f moves bounds that `accuracy/accuracy_harness.py` also consumes via `for_op()`, so the
  accuracy/ULP sweep shifts with them.
- Two thirds of Phase 0's cost was that shared golden/comparison correctness. Phases 1–5 widen
  stimuli further but no longer have to pay for it.
- **A test-infra bug this work surfaced, worth filing separately:** `TestConfig` does
  `shutil.rmtree(ARTEFACTS_DIR)` at session setup against the fixed path `/tmp/tt-llk-build`,
  so a second pytest session on the same host — even a one-op `-k` run — deletes the build
  tree out from under a running sweep. The victim reports `ld: cannot open output file`
  attributed to whichever variant was linking, which reads exactly like a real kernel bug.
  This produced 2 phantom failures in the first BH run here. Not a Phase 0 item, but a live
  trap for the triage re-runs Phases 1–4 will need.
- The plan doc was rewritten against the post-#50602 tree in the same commit — the phase
  structure after 0 changed materially (no `spec_B` in the unified binary driver; the
  `compare_special` flag is already satisfied upstream; `StimuliSpec.custom` silently clamps
  `INT32_MIN`). See [SFPU_EDGE_CASE_EXPANSION_PLAN.md](SFPU_EDGE_CASE_EXPANSION_PLAN.md) §2.
