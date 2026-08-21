# SFPU corpus scoreboard — pin 18 (cc1plus 664bbf81b2ca), 2026-08-21

BODY-ZONE (diagnostic) verdicts: 43 WIN / 16 PARITY / 36 LOSS (+1 ALGO, 52 sem-only causal rows).
METRIC CHANGE IN FLIGHT: owner ratified END-TO-END KERNEL time as the verdict metric (2026-08-21);
the e2e re-measure weekly is running — this file rebooks from its sealed report (kernel-decided
verdicts; body zones demote to diagnostics). Interim kernel-decided tally at 147/179 rows:
44W/11P/34L with 3 verdict flips (welford P->W, recip W->L, binary-bcast W->P).

Highlights (body-zone, device-golden gated): divint32floor -45% WIN (certified recip rewrite),
gcd -19%, log -9%, sdpa -8%, minmax -5.1%, welford WIN; cast/absint32/eqz/unaryshift parity via
knob configs; 13+3 fitted/LUT rows = owner accuracy-contract decision pending; every loss carries
a named mechanism or named refusal. Full live board: the claude.ai artifact (session-published);
provenance per row in each evidence root's ROW-VERDICT.json.

History: pins 14-18 in sweep_2x2.conf PIN HISTORY; review_records/ per pin.
