# REVIEW_RECORD — pin 40

cc1plus sha256: 48e381718f5cc888e8acb37feb0853adef275c411a2c0232b72d2b0e81965e92
driver (g++) sha256: 43a97fc673ea2069161f8e56e1de10e8b4203ba69a7942ccaaa3feb34a7f2885
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi b779810620f = pin-39 union tip 07835727b3d
+ lane IH merge (agent/post-autoincr-window, fast-forward).
Companions: tt-metal chain through 18317d700893 (IH knob registration;
IG's six-flip rows 984ae9ac4d merged earlier). KNOB_MODES dup grep
clean (only the known benign lut-select-fp16 pair). No sfpi include/
changes. Built in gcc-build-laneFR (build-pin40.log rc=0); flags
smoke-accepted (OPTCHECK, mcpu=tt-bh); installed via
pin-install-fast with loud --expect-cc1plus; no live sweeps at
install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

One lane (closing report, evidence dir + SHA256SUMS 529, memory
file): IH POST-AUTOINCR WINDOW RE-FORMATION
(-mtt-tensix-optimize-post-autoincr-window Init(0)). A second
blocker was adjudicated beyond lane IF's: a naive second former run
starves — the pre-fold former's own tail-shuffle windows plus the
envelope record leave 2 free slots against the 45-word carried-body
candidate. The fix is DEFERRAL: the flag gates pass_rvtt_replay off
and runs the same transform once as pass_rvtt_replay_reform (after
dst-autoincr, before mop_form); deferral provably loses nothing
(the fold only removes barrier words). Soundness rests on the
stream-identity theorem: word-exact replacement is insertion-only
over the folded stream, so the window's launch count times the walk
equals the intended traversal by construction (IF's cumulative-RWC
model), and lane HM's delay shadows survive verbatim (counted-row
runs once here under its own HM/ID audits; its vocabulary provably
never moves an RWC word). Carried payloads carry a structural
deliveries==sites audit (named refusal
post-autoincr-window-launch-arithmetic-skew with an affirmative dump
line) plus named refusals for the two word-inexact mechanisms
(carried-peel-launch-arithmetic-unproven — the near-miss twin
refuses by name and falls back to sound in-block formation — and
carried-isomorphic-conversion-unproven as a belt). The FS/ES/FJ
belts (un-hoist rules 1-3, raw census, epoch scoping, disjoint slot
spans) all re-run over the post-fold layout; FR's checker runs last
as always. 5 twins, 32 dg checks.

DELIVERY CLASS ACHIEVED: the useq blaze bodies reach hand's class
exactly — one hoisted no-exec record + 8 launches/tile, zero raw
sync words, tile words sum 97->51 / max 99->53 (a testing-only
prefer-longest flag builds the literal 16-word/4-launch hand shape).
SILICON (BH p150, 37 sessions rc=0, corr-first, 3-rep
cycle-identical; anchors 1495/1472/1775/1758 and IF's cells
1596/1823 reproduced EXACT): reform sum 1606-1608 / max 1822 —
THE ENVELOPE LAW IS CONFIRMED A THIRD TIME. Removing 46 issue
words/tile recovered ~1 cy/tile of IF's <=23/8 bound: the family is
execution/delivery-bound, and every expressible delivery shape of
the carried body is now silicon-measured (straight-push IF, pre-fold
capture IE, post-fold capture in both window sizes IH). The blaze
A9/A10 flippable pathway is CLOSED (G3 all-shapes-measured
certificates); the cells stay lift-booked; the hand-exact 4-launch
shape measures WORSE (the slot budget binds, not the launch count).
copydest-fresh's booked -13.07 cell is byte- and cycle-exact
preserved; the carrier+reform composition is envelope-refused
(+0.75%) — never compose.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin40; pinned-install
  env): 6803 PASS = pin-39's 6771 + IH's 32 exactly; FAIL set 16
  rows LINE-IDENTICAL to the pin-39 frozen baseline (diff empty).
  Flags smoke-accepted (OPTCHECK).
- IH gates: blaze compile screen 174/174; paired CRAQ 8/8 across
  three arms on the pinned sims (bh 1d162f0adf67) incl the t32
  hazard on device; corpus base-vs-fix OFF/TD/ON-36 3300/3300 x3
  .text-identical (knob-off byte-inert); knob delta = 76 TUs, all
  attributed via a kept-build corpus rerun and all paired-CRAQ clean
  (the only failures are the pre-existing reciprocal-class and
  binary_int_uniform sim failures, identical on both arms).
- Board: notes-only 86d138f941f5 -> 449914d7b8e4; tally 83W/35P/16L
  unchanged.
- Evidence: laneIH-evidence-20260828 (+SHA256SUMS 529). Incident
  banked with recovery: the hybrid cc1plus was swapped mid
  corpus-batch — poisoned legs killed, store key deleted, all 7 legs
  rerun at the final binary.
- Install: sha-verified 287a307f4836... -> 48e381718f5c...; driver
  read from the fresh manifest entry.
- ON set UNCHANGED at 36 (post-autoincr-window registers as an
  on-plus booking knob).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin40-ceremony/).
