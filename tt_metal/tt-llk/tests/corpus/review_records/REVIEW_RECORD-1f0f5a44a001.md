# REVIEW_RECORD — pin 33

cc1plus sha256: 1f0f5a44a00104522b978f56bf7372ea76a225255dfc040c7f18c55a76027926
driver (g++) sha256: 581597a7ec7a77752cb9596dea25256c764848a95d50ae086fbc44b441d48bbd
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi 624e8448977 = pin-32 union tip 8012f513916
+ lane HR merge (agent/crossloop-cc-peel, fast-forward). Companions:
tt-metal chain through ab40b8d04b (HQ ON-36 promotion b76108e6c04 +
HR knob de3858dbbe union-resolved + the stale on-plus KNOB_MODES
duplicate drop for the promoted pair — HR's branch predated the HQ
ceremony; the pin-30 shadowing class, caught by the post-merge dup
check before any sweep ran). No sfpi include/ changes. Built in
gcc-build-laneFR (build-pin33.log rc=0); the new flag smoke-accepted
with the pin-32 knob set (OPTCHECK, mcpu=tt-bh); installed via
pin-install-fast with loud --expect-cc1plus; no live sweeps at
install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

One lane (closing report, evidence dir + SHA256SUMS 966, memory
file): HR crossloop-cc-peel (-mtt-tensix-optimize-crossloop-cc-peel
Init(0)). The CC-canonical peel existed solely to manufacture an
in-loop all-lanes programming point, and the crossloop region scan's
statement discipline blanket-refused every sets_cc statement — so a
loop needing a peel guaranteed crossloop-cc-unproven on its enclosing
walks and the peel+programming re-executed per enclosing iteration
(atan2: 29 words/face). The fix extends the laneCD scan (no parallel
logic) with a cc-immaterial mode licensing PROGRAMMING-ONLY lifts:
fail-closed typed structured-CC-atom whitelist, fn-entry all-lanes
ambient proof at the lifted preheader, a pressure-park consumer audit
including the pre-CC prefix (per-candidate peel fallback), and a
single-trip gate — each failure a named refusal that keeps the peel
byte-identically. No CC statement is inserted, deleted, or reordered
vs source, so no ES/FJ exec-state shape can form. CENSUS: 47 rows /
593 instances carried crossloop-cc-unproven at pin-32 ON-34 — the
only placement-walk stop reason corpus-wide; lcm-fresh is the only
LOSS member and closed HONESTLY INERT (the TU-wide PRGM freedom proof
refuses all residency placements in every flag state:
opaque-region-undeclared, raw opcode in run_kernel — named
successor). Six dg twins 36/36.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin33; env = the pinned
  install ~/sfpi-uplift/sfpi/build/sfpi): 6512 PASS = pin-32's 6476 +
  HR's 36 exactly; FAIL set 16 rows LINE-IDENTICAL to the pin-32
  frozen baseline (diff empty). New flag smoke-accepted together with
  the pin-32 knobs (OPTCHECK).
- Corpus byte-gates (corpus-legs-laneHR = the first pin-32 base
  store): OFF/TD/ON-34 fix-vs-base 3300/3300 x3 byte-identical; knob
  delta = exactly 2 TUs (the atan2 corr impl 1+2 TUs,
  dump-attributed), paired CRAQ on the pinned sim 32489dda + device
  corr on those exact nodes.
- Silicon (BH p150, 3-rep, corr-first; off legs reproduce the booked
  cells exactly): atan2 -15.27 -> -16.20 SAME-LEG (the hand LLK TU
  fires too — a compiled C++ face loop; HN same-leg convention);
  atan2-fitted -8.06 -> -16.20 (impl2 .text == impl1 at ON-34 — the
  old cell was a stale pre-promotion measurement, refreshed);
  divint32floor-fresh -45.68 -> -47.87 (hand byte-identical inertness
  control).
- Evidence: laneHR-evidence-20260826 (+SHA256SUMS). Board edited via
  GE mechanics with snapshot; BOARD 76W/35P/24L (shape unchanged —
  three WIN rows deepened, lcm-fresh stays an honest loss with the
  successor named).
- Install: sha-verified 4943ca7fe176... -> 1f0f5a44a001...; driver
  read from the fresh manifest entry; no sfpi include/ staging owed.
- ON set UNCHANGED at 36 (crossloop-cc-peel registers as an on-plus
  booking knob).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin33-ceremony/).
