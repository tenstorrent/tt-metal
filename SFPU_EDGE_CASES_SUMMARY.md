# SFPU edge-case coverage — what this branch does

Part of [#49739 — [LLK] SFPU testing edge cases](https://github.com/tenstorrent/tt-metal/issues/49739)
· branch `ldjurovic/sfpu_edge_cases_2` ·
[PR #52416](https://github.com/tenstorrent/tt-metal/pull/52416), rebased onto post-#52172 main

Sequencing for what comes next is in
[SFPU_EDGE_CASE_PR3_PLAN.md](SFPU_EDGE_CASE_PR3_PLAN.md).

The audit behind this work found that the SFPU test suite fed almost every op a positive-only
`uniform(0.1, 1.1)`. Four phases follow from that: point the sweeps at the per-op domains that
already existed (0 and 1), then build shared metadata so the interesting *individual* values can
be hit on purpose (2 and 3), then drive them (4).

Each phase's mechanism was small. Each phase's cost was the golden and comparison defects that
benign stimuli had been hiding — the same lesson three times, and worth expecting a fourth.

---

## Phase 0 — the unary sweep, and three defects it exposed

Phase 0 was meant to be one change: stop `test_eltwise_unary_sfpu_float` feeding every op a
positive-only `uniform(0.1, 1.1)` and point it at the per-op signed domains already sitting in
`_OP_DOMAIN_REGISTRY`, so 31 ops would finally exercise their `x<0` branch, piecewise knees and
saturation tails. The reroute is 5 lines. Making it pass took four more commits, because stimuli
that benign cannot distinguish a correct golden from a wrong one, a correct comparison from a
wrong one, or an accurate kernel from an inaccurate one:

- **`UnarySFPUGolden` never quantized Bfp8_b inputs**, so the golden ran on values the hardware
  never saw. Smooth ops absorbed it inside tolerance; a sub-ULP quantization step across an
  integer becomes a full 1.0 error for `floor`/`ceil`/`trunc`/`frac` — 16 failures each.
- **`passed_test` judged Bfp8_b as a scalar format**, not a block one, even though it has 7
  magnitude bits against an exponent shared across 16 elements. Routing it to the lattice check
  *alone* was wrong the other way. The two criteria describe different regimes — quantization
  dominates far below the block max, approximation error near it — so a result satisfying
  **either** is accepted. That is a strict superset of the old behaviour, so it cannot regress a
  passing test.
- **Registry domains were range-correct but accuracy-wrong**: `exp` 80→16, `exp2` 100→23, and
  `exp_with_base` 160→32, which its own docstring derives as double `exp`'s. `reciprocal` was
  recalibrated here too — a 1000:1 ratio inside a 16-element block quantizes the smallest elements
  to zero and sends the golden to `inf` — but #52172 landed a strictly better version of that one
  (a Bfp4_b tier at 25:1 alongside the 10:1), so this branch defers to it.

One genuine kernel limit survived and is recorded rather than tolerated: approximate `exp`
overshoots by a systematic ~5.7% (peak 6.75%) once its argument passes ~8. The three affected
combinations are listed exhaustively and marked so the case still **executes** and reports XPASS
if the approximation tightens.

Phase 0 also merged the two unary sweeps into one test, provably behaviour-preserving by
measurement (same case set, same stimuli, identical per-`(op, outcome)` results), which deleted a
nightly test and its duplicated skip paths and made an op in the wrong list a **collection-time**
error instead of a silent coverage change.

**Net:** 31 ops gained their negative branch, 31 dead registry entries became live, two shared
golden/comparison bugs were fixed for every suite that outputs Bfp8_b, one accuracy limit was
documented instead of hidden, one duplicated test removed.

---

## Phase 1 — the other three families

Phase 1 applied the same reroute to binary, ternary and scalar, none of which had ever imported
`sfpu_domains.py`. Seven float elementwise binary ops now take their registered domain —
add/sub/mul/rsub/div go from **0% to 50% negative operands**, and div's divisor spans both sides
of the pole it is registered to avoid. The scalar binops sweep `{0, 1, 2, −2, 8, 0.25}` rather
than a hard-coded `2.0`; `SfpuElwLt/Gt/Le/Ge` got their **first LLK-level correctness test**,
driving the exact tie where lt/gt and le/ge disagree; and recording op arity made the unary
sweep's exhaustiveness a collection-time error.

As in Phase 0 the fallout was most of the work:

- **Per-operand pairing had only ever filled the first tile pair.** `face_specs` is applied
  positionally and not cycled, so `mask`, `isclose` and `eq`/`ne` were each testing one
  sixteenth of what they appeared to.
- Fixing that exposed **`calculate_mask`'s one-pair-per-block limit** — the kernel ignores the
  forwarded dst indices, so 12 of 16 pairs failed until the test moved to a `[64, 32]` buffer.
- **`pow` and `xlogy` needed accuracy-bounded domains**, neither having ever executed.
- The suite had never seeded its stimuli, so a flaky variant was indistinguishable from a real
  finding.

**Net:** 7 binary ops gained negative-operand and pole-spanning coverage, 4 comparison ops their
first test, pairing went from 1-in-16 tile pairs to all of them.

---

## Bfp4_b input coverage (not a phase, and no longer on this branch)

Two commits enabled the seven Bfp4_b unary ops that had been commented out since the list was
created, and extended the Bfp4_b axis to the whole broad op list. **Both dropped out in the rebase
onto post-#52172 main**, which reached the same place by a better route — Bfp4_b is a second
*format axis* over the existing op list there, not a second op set. The findings are kept here
because they are what the work established, not because the commits survive.

Six of the seven passed as-is; `Reciprocal` needed a Bfp4_b-specific domain tier, because 3
mantissa bits cannot hold the 10:1 window that suffices for Bfp8_b's 7 — ~6% of it quantizes to
exactly zero, and the golden then computes `1/±0 = ±inf` (sign-preserving) while the SFPU returns
`+inf` for both signed zeros. That tier is the version main now carries.

**A third commit was dropped from this branch.** "Fold Bfp4_b into the broad format matrix" made
Bfp4_b an *output* format, creating `Float16 -> Bfp4_b`, which fails **100%** at `dest_acc=No` on
Wormhole — 126 variants, 0 passing, across 30 of the 31 broad ops, with garbage output (`Neg` on
a `[-10, 10]` input returning values around `4.8e24`) rather than an out-of-tolerance result.
Every neighbouring cell is clean, which acquits each half of the pair independently:
`Float16 -> Bfp8_b` passes, so a Float16 input is fine, and `Float16_b -> Bfp4_b` passes, so a
Bfp4_b output is fine. It is specifically exponent-A input into exponent-B output with no 32-bit
dest intermediate.

That commit was verified green on Blackhole, where `_skip_bh_unsupported_float_combo` skips every
Float16-input `dest_acc=No` variant — so the failing cell was never executed there. Re-adding
Bfp4_b as an output format needs either a guard for that pair or a fix to the Wormhole pack path;
the garbage magnitudes point at the shared-exponent write.

---

## Phases 2–4 — hitting the edges on purpose

Phases 0 and 1 widen the *random* domain, so the sweeps land **near** knees and poles but never
**on** them. Phases 2–4 add the shared metadata to land on them and one thin test per family that
consumes it.

**Phase 2 — metadata.** `format_ulp()` for format-relative probe spacing, `_OP_SINGULARITIES`
(exact poles, each carrying which side the op is *defined* on), `FLOAT_SPECIALS` and a
width-derived `integer_specials()`, and `_OP_EDGE_POINTS` — 43 entries of knees, thresholds and
exact rounding ties, every constant verified against the golden that owns it.

The plan expected cat A to need no new data, since the finite edge of each hole in
`_SFPU_UNDEFINED_RANGES` looks like the boundary. It isn't: a hole is a *guard band*.
Reciprocal's is `(-1e-6, 1e-6)`, so deriving probes from it yields ±1e-6 and never `0`. Adding
the missing holes is not an option either — `exclude_intervals()` always rewrites into the
`intervals` form, and that sampler draws twice per element where the plain path draws once, so
declaring a hole **re-rolls that op's entire stimulus set at the same seed**. Hence a separate
table that never touches the draw path.

**Phase 3 — `edge_spec()`.** Composes the three categories, clipped against the narrowest format
in the *pipeline* rather than the input format, because passing `spec_A` bypasses the driver's own
resolution. Plus `edge_pair_values()` for binary, which crosses both operands' edges — a divisor
of 0 against a positive, a negative and a zero numerator are three different cases.

**Phase 4 — the sweeps.** A unary edge sweep over all 94 swept ops (752 cases) and a binary one
over the five ops with a registered pole, both landing green with the divergences below recorded
as non-strict xfails. A scalar wrapper (20 cases) exists for the same shape but skips entirely
today: all five scalar ops are `x ⊕ c` for a compile-time `c`, so they are smooth in `x` and have
no cat-A or cat-D edge at all. It is there to make the driver's new `spec_A` knob reachable, and
goes live on the same trigger as cat B.

Cat B (IEEE specials) is measured, and gated **per op**. The special-safe `(format, dest_acc)`
matrix was established first, by driving the isinf/isnan predicates over the full 5×5 matrix with
no skips: a `Float16` anywhere in the pipeline never preserves specials, and a 16-bit input with
`dest_acc=Yes` keeps `+inf` but loses `-inf` and `NaN`. Even on the 7 triples that do carry them,
injecting specials fails **272 of 564** variants — because the goldens do not model non-finite
*inputs*. That is golden work, not a stimulus change.

The gate is therefore two-sided and per op rather than one global bool: `specials_safe()` says the
*pipeline* delivers specials intact, `SPECIALS_READY_OPS` says the *golden* has an answer for them,
and both must pass. The mapping starts empty — identical behaviour to the old global `False` — and
an op joins it carrying its reason once its golden is extended. That makes the remaining cat-B work
a series of one-op commits rather than one commit that cannot land.

---

## What Phases 2–4 found

Ten ops across 42 (op, format, dest_acc) cells — 5 unary ops over 20 cells and 5 binary over 22 —
none of it previously measured, because the random sweep lands near these points and
never on them. All of it is cross-checked against
[tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation), which splits the
results cleanly into "documented" and "still open". Everything stays xfailed either way — the
test's job is to notice a divergence, not to judge it — but only the second group is worth a
kernel-side look.

### Documented hardware behaviour

**The sign of a zero *result* is lost on Wormhole, and this is specified.** `div(0, -x)`,
`fmod`/`remainder` with a negative divisor, and `xlogy(0, tiny)` all return `+0.0` where IEEE
gives `-0.0`. Every one of those ops is built on `SFPMAD`, and:

> Wormhole — "If the output (before rounding) is denormal or negative zero, it'll be flushed to
> **positive** zero." — `WormholeB0/TensixTile/TensixCoprocessor/SFPMAD.md`
>
> Blackhole — "If the output (after rounding) is denormal, it'll be flushed to **sign-preserved**
> zero." — `BlackholeA0/TensixTile/TensixCoprocessor/SFPMAD.md`

Blackhole's page lists "improved edge-case handling of NaNs and of negative zero" among its
upgrades over Wormhole. So this is a documented Wormhole limitation that Blackhole is documented
to fix — which is why these are **non-strict** xfails: they should report XPASS on Blackhole
rather than failing the suite.

**`sign(-0.0)` and `heaviside(-0.0)` are outside the primitive's documented contract.**
`SFPSETCC`, which those kernels compare with, is specified only:

> "Provided that `VC` is neither negative zero nor any kind of NaN: set per-lane flags based on
> `VC < 0` or `VC != 0` or `VC >= 0` or `VC == 0`" — `VectorUnit.md`, identically on both arches

The golden follows torch and IEEE-1985 and is right about the mathematics; the hardware was
never promised to agree at `-0.0`. Worth knowing that this caveat persists on Blackhole, so
unlike the `SFPMAD` group it is not an arch-generation issue.

### Still open — not explained by the ISA

| Finding | Ops |
|---|---|
| **`signbit(-0.0)` returns 0** where the kernel's own docstring promises 1 ("logical-shift the fp32 bit pattern right by 31 … incl. `-0.0`"). Unlike `sign`/`heaviside` this op claims to read the sign bit *directly*, so either the claim or the implementation is wrong — a kernel-contract bug rather than a hardware one. `-0.0` is delivered correctly, verified host-side. | `signbit` |
| **`0/0` and `x%0` return `inf`, not `nan`.** `SFPMAD` states "if any input is NaN or ±Infinity, then the result will be NaN or ±Infinity, following the usual IEEE754 rules", which makes `0 × inf` a NaN — so this is the kernels' own reciprocal composition, not the multiply. Specifically the indeterminate form: the finite poles agree exactly and every ±inf lines up. | `div`, `fmod`, `remainder`, `xlogy` |
| **`0**0` returns 0** where C, torch and the golden give 1. `pow` evaluates `exp(b·ln a)`, so a composition artifact. | `pow` |
| **`RsqrtCompat(0)` saturates to `1.7014118e38`** (`0x7F000000`) instead of returning `inf`, on all 8 combinations — while plain `Rsqrt` over the same probe does not diverge. Two implementations of one function disagreeing at their shared pole, with nothing in the ISA prescribing either answer. | `RsqrtCompat` |
| **`Erfinv(±1)` saturates** rather than returning ±inf, on the fp32-dest combinations only. | `erfinv` |

Two smaller results worth keeping:

- **The bitwise kernels need the two's-complement pack path** for negative operands.
  `(INT32_MIN+1) & -1` returned `-1`. Nothing had established this because
  `test_sfpu_binary_bitwise` draws from the positive-only default and has never fed them a
  negative.
- Both Blackhole guards are now non-strict **xfails** rather than skips, so a kernel fix reports
  XPASS instead of leaving the case green by omission indefinitely.

---

## Verification, and what is not verified

Wormhole n150, full suites:

| Suite | Result |
|---|---|
| `test_sfpu_unary.py` | 5896 passed, 694 skipped, 26 xfailed, 0 failed |
| `test_sfpu_binary.py` | 13 passed / 22 xfailed for the new edge tests; 5 passed for int extremes |

Phases 2–4 were proven **inert** with respect to the existing suites before anything was built on
them: registry behaviour is byte-identical to the pre-change tree across 1265 `(op, format)`
entries — covering `for_op`, `exclude_undefined_pair`, `for_op_pipeline` over all five output
formats, and a hash of the actually-drawn tensor — and a per-test junitxml A/B of the binary suite
shows the same 1081 tests with **0 outcome differences**.

### After the rebase onto post-#52172 main

The same two checks were re-run across the rebase, since eight of the branch's twenty commits
dropped out as already-upstream and the rest replayed onto a moved base:

- **No test lost**, per-test pre-rebase against post-, and every branch-unique test kept its exact
  cardinality (unary edges 752, binary edges 40, int extremes 5, the scalar split 20 + 100).
- **Registry stimuli byte-identical** across every `(op, format)` entry — resolved spec and
  drawn-tensor hash — once the `exp_with_base` ceiling was carried across (see below).
- **Collection is 9496, not the 9436 of the pre-rebase branch.** The +60 is main's own widening of
  two tests this branch also touches (`eq_ne` 16 → 36, `float_comparison` 32 → 72), not a change
  the rebase made. With the scalar wrapper it is 9516.

One collision was not textual and so did not conflict: `_exp_with_base_spec` derives its bound as
double `_exp_spec`'s *in prose*, #52172 recomputed it against the old ceiling of 80 while this
branch moved that ceiling to 16, and nothing failed — the two simply stopped agreeing, leaving
`exp_with_base` running at arguments this branch had measured at 11–13% off golden. Derived
constants need to be derived in code or asserted, not described.

**Not verified:**

- **Blackhole — partial.** The reduce xfail's tightening and the scalar presubmit/nightly split
  *were* measured there (p100a: 84 passed, 28 xfailed, 0 xpassed; scalar 120 collected, all
  passing). Everything else above is Wormhole, and two parts are arch-sensitive by construction and
  must be re-measured: the special-safe matrix (the unpack paths differ, and it is a measurement
  rather than a derivation, so it may be wrong there in either direction) and the shift xfail,
  whose whole purpose is the Blackhole path — on Wormhole it is a deliberate no-op. The `SFPMAD`
  signed-zero xfails are a **testable prediction** there: Blackhole's ISA documents sign-preserved
  zero, so they should XPASS, and if they do not, the documentation and the hardware disagree.
- **CI.** The broad unary profile runs in **no automated job on any arch**: every LLK pytest job
  either excludes `nightly` (pr-gate smoke, bit-exact) or runs `--coverage`, under which the
  broad profile is skipped wholesale. That predates this branch, but it means these coverage gains
  are currently unguarded. Either `llk-e2e` needs a non-coverage companion group or the broad
  profile must stop being coverage-gated. Note the attribution to tt-llk#1435 — carried in the
  suite's own skip reason — looks wrong: that issue is open but is about test *ordering*
  (`test_eltwise_unary_sfpu.py` failing after `test_eltwise_binary`), not coverage. Worth resolving
  before anything cites it further.
- **`WITH_COVERAGE` builds**, and Bfp4_b output formats, which the dropped commit covered.
- **Everything above the rebase line was measured pre-rebase.** The post-rebase checks were
  static — collection, registry A/B and module-level assertions — because the host available for
  the rebase had no Blackhole and a venv whose `tt-exalens` did not match `tests/requirements.txt`.
  The Wormhole suite numbers have not been re-measured against the rebased tree.
