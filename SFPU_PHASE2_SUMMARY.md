# SFPU Phases 2–4 — in short

Part of [#49739](https://github.com/tenstorrent/tt-metal/issues/49739) ·
[PR #52416](https://github.com/tenstorrent/tt-metal/pull/52416) · branch
`ldjurovic/sfpu_edge_cases_2` · full write-up in
[SFPU_EDGE_CASES_SUMMARY.md](SFPU_EDGE_CASES_SUMMARY.md)

Phases 0–1 ([#52172](https://github.com/tenstorrent/tt-metal/pull/52172)) widened the *random*
domains so the sweeps land **near** knees, poles and negative branches. Phases 2–4 make them land
**on** them, with one mechanism per gap-category rather than one test per op: edge metadata
(`_OP_SINGULARITIES` — 19 exact poles carrying which side the op is defined on; `_OP_EDGE_POINTS` —
43 knees and ties, every constant read out of the golden that owns it; `FLOAT_SPECIALS`; a
width-derived `integer_specials()`; and `format_ulp()` so a probe steps by the *format's* ULP), one
builder (`edge_spec()` / `edge_pair_values()`, clipped against the narrowest format in the
*pipeline* because passing `spec_A` bypasses the driver's own resolution), and three thin sweeps
over the existing drivers — 752 unary cases, 40 binary poles, 5 integer-extreme cases. Because
`edge_spec` is keyed off the op, **adding a new op auto-enrols it in edge testing**. 50 of the 97
unary ops now have at least one deliberate edge value.

Two of the plan's premises did not survive contact. Cat A was supposed to need no new data, since
the finite edge of each hole in `_SFPU_UNDEFINED_RANGES` looks like the boundary — but **a hole is
a guard band**: `Reciprocal`'s is `(-1e-6, 1e-6)`, so deriving from it yields `±1e-6` and never
`0`. Adding the missing holes was not available either, because `exclude_intervals()` always
rewrites into the `intervals` form and that sampler draws twice per element where the plain path
draws once, so **declaring a hole re-rolls that op's entire stimulus set at the same seed**. Hence
a separate table off the draw path. And cat B was supposed to be "inject, xfail the handful the
golden cannot express" — measured, the handful is **272 of 564 variants**, so it is per-op golden
work and ships measured but switched off rather than as 270 xfails.

**Net:** 10 ops disagree with their golden at their edges across 42 (op, format, dest_acc) cells,
none of it previously measurable, all recorded as non-strict xfails and each cross-checked against
tt-isa-documentation into "documented hardware behaviour" (the `SFPMAD` flush-to-positive-zero
group, which Blackhole is documented to fix, and `SFPSETCC`'s negative-zero carve-out) or "still
open" — of which `signbit(-0.0)` contradicting its own kernel docstring and `RsqrtCompat(0)`
saturating where plain `Rsqrt` does not are the two worth a kernel-side look. Two shared defects
were fixed on the way: `_FORMAT_MAX_MAGNITUDE` had the MX fp8 ceilings transposed and no
`Fp8_e4m3` entry, and `_assert_domain_sets_consistent()` claimed a partition of the binary suite's
ops that it did not have. Also found: the bitwise kernels need the two's-complement pack path for
negative operands, which nothing had established because they had never been fed one. Both
Blackhole guards became non-strict xfails so a kernel fix reports XPASS instead of staying green by
omission. Wormhole n150, the two edge sweeps: **385 passed, 370 skipped, 42 xfailed, 0 failed**;
collection across the five SFPU suites is 9436 before and after.

**Not closed by this PR:** cat B goldens (the largest remaining item — and the *entire* edge story
for the 47 ops that are smooth everywhere), the ternary and scalar edge wrappers, Blackhole
verification of the edge sweeps and the special-safe matrix, category E's unary shift amount, and
per-op tolerances for `xlogy` and `pow`. Next steps and sequencing are in
[SFPU_EDGE_CASE_PR3_PLAN.md](SFPU_EDGE_CASE_PR3_PLAN.md).
