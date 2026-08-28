# REVIEW_RECORD — pin 41

cc1plus sha256: d8046e29c2b382d39a4b2f154cd9e26a4aad5649c5a6efd81aeabd077fd23b33
driver (g++) sha256: e5908e9a530afda3d8f24a477f947704db5630a9f6f1ab5a32285e4057d551f1
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi ccfc4ccabee = pin-40 union tip b779810620f
+ lane II merge (agent/tanh-shared-reload, fast-forward). Companions:
tt-metal chain through c89b472799 (II knob registration 1f663297c9 +
measured note; IH's knob 18317d700893 merged at pin 40). KNOB_MODES
dup grep clean (only the known benign lut-select-fp16 pair). No sfpi
include/ changes. Built in gcc-build-laneFR (build-pin41.log rc=0);
flags smoke-accepted (OPTCHECK, mcpu=tt-bh); installed via
pin-install-fast with loud --expect-cc1plus; no live sweeps at
install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

One lane (closing report, evidence dir + SHA256SUMS 836, memory
file): II CROSS-ROW SHARED-RELOAD DEDUPE
(-mtt-tensix-optimize-crossrow-shared-reload Init(0)) — the tanh
endgame. Autopsy at pin-40: the 2datum+stochrnd tanh record carried
4 duplicated coefficient loadi words per pair (reg 84 / L4: row B
re-materializes row A's C3 and C1 verbatim because the renamer
starves at the 8/8 LREG wall, dump-named
round-interleave-rename-no-free-lreg). Lane IC's wrong-code fact
restated in uids: a naive dedupe binds row B's consumer to row A's
group BEFORE any scheduling — ls_dependence derives value flow from
position alone, so no edge can express "read the earlier def"; and
the post-schedule peephole layer is REFUTED by the same dump (A's C1
group intervenes in the committed order, so deletion requires
reordering, which re-opens lane HM's positional delay-contract
discharge). Shipped layer: the EPOCH MERGE inside the pairing
transaction — the copy half's definition groups are deleted and the
pairing's original order is re-sequenced epoch by epoch, so position
is value-correct again and the existing name-based vocabulary
derives the sharing constraints itself (RAW def_e -> consumers_e,
WAR consumers_e -> def_e+1). Byte-identity of the halves' groups is
re-verified on single_set dest/src (rtx_equal_p is false on
scratch-clobber pairs — gotcha banked). 12 named refusals
(copy-shape, web-mutated, crossrow-interference, live-in/out,
atom-*, seeded-row, ii-regression, ...) plus the committed-order
value-oracle belt (crossrow-pairing-shared-reload-final-order)
carrying the dominance+liveness proof as a belt. 7 dg twins incl the
naive-dedupe wrong-code near-miss (refuses by name).

RESULT: the paired record drops 30 -> 26 words (ttreplay 0,26,1,1 +
15 launches, zero SFPNOP) = 28 slots/pair, BELOW hand's 30 — the
body is the hand kernel's sequential shared-reload discipline
verbatim. COMPOSITION MATRIX (BH p150, 3-rep cycle-identical,
corr-first 4/4 x15 sessions; anchors and booked controls reproduced
EXACT): shared-reload alone byte- and cycle-inert (pairing refuses
at the swap without 2datum); 2datum+SR 67642; 2datum+stochrnd+SR
63544 vs hand 67378 = tanh-fresh +6.47 -> WIN -5.69 (causal -24.03;
the same-leg stochrnd-moved hand arm 65326 is also beaten, -2.73 —
that hand movement is lane HZ's store-fold fact, byte-proven
SR-inert). The row stood at +35.91 at pin 30; the flip chain is
IB's bar refusal -> IC's 2-datum window -> IG's stochrnd composition
-> II's dedupe.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin41; pinned-install
  env): 6834 PASS = pin-40's 6803 + II's 31 exactly; FAIL set 16
  rows LINE-IDENTICAL to the pin-40 frozen baseline (diff empty).
  Flags smoke-accepted (OPTCHECK).
- Preservation: 71-leg loss+WIN screen 0 CHANGED; 9-leg
  seed-composed screen 0 CHANGED (HB roundingops/ceil seed cells
  byte-preserved); IC's booked crossrow-2datum 12-TU delta
  reproduced EXACT on the icknob preservation legs.
- Corpus: OFF/TD/ON-36 base-vs-fix 3300/3300 x3 .text-identical;
  knob delta = ZERO corpus TUs (corpus-inert knob; fresh-harness
  fires adjudicated by paired CRAQ 4/4 across three arms on the
  pinned sims + device corr).
- Board: 449914d7b8e4 -> 20b1ac492cb2, tanh-fresh only (pre-state
  asserted); tally 84W/35P/15L; tanh-fitted stays SUPERSEDED per the
  owner-ratified fold.
- Evidence: laneII-evidence-20260828 (+SHA256SUMS 836).
- Install: sha-verified 48e381718f5c... -> d8046e29c2b3...; driver
  read from the fresh manifest entry.
- ON set UNCHANGED at 36 (crossrow-shared-reload registers as an
  on-plus booking knob).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin41-ceremony/).
