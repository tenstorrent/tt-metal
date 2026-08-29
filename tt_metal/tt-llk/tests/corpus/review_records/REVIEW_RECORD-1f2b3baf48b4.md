# REVIEW_RECORD — pin 45 (the zero-loss-or-cert pin)

cc1plus sha256: 1f2b3baf48b4505fec876fd3d317979e14f45c3e8062ebde2c9207b916ff5e8f
driver (g++) sha256: cc2123de744a5e99113b4807e3254994ac4e536f76892452429d57834210d91d
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi 18318a7b5e4 = pin-44 union tip c95ee644414
+ lane IO merge (agent/addrsqrt-reattack, fast-forward). Companions:
tt-metal chain through 29669d90f8 (IO cert notes + knob
registration). KNOB_MODES dup grep clean (only the known benign
lut-select-fp16 pair). No sfpi include/ changes. Built in
gcc-build-laneFR (build-pin45.log rc=0); flags smoke-accepted
(OPTCHECK, mcpu=tt-bh); installed via pin-install-fast with loud
--expect-cc1plus; no live sweeps at install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

One lane (closing report, evidence dir + SHA256SUMS 6428, memory
file): IO addrsqrt re-attack — the close that brings the board to
the owner-ratified ZERO-LOSS-OR-CERT accounting state.

DRIFT AUTOPSY (the headline): the pins-39->43 hand-tree drift DOES
NOT EXIST. The hand production TU (ckernel_sfpu_add_rsqrt.h) is
instruction-identical pins 39->44 at plain ON-36 (the only diffs are
profiler zone-id LUIs, the known farm-path artifact), and the
tt-metal tree diff across those pins touches only
tests/conf/review-records. The 123937 -> 119837 "hand improvement"
was a SAME-LEG VIOLATION in lane IG's scan booking: the hand cell
was booked at plain ON-36 while the row's booked composition is
ON-36+stochrnd-store-fold — and the HZ license fires on the hand TU
too (its row drops 26 -> 25 words, the sfpstochrnd folds into the
store, worth ~4100cy). IG's -2.26 "WIN" was sem@stoch vs hand@plain
and never a real same-leg win; lane IN's +0.98 re-book was the first
honest same-leg cell and stands. LESSON BANKED: same-leg violations
masquerade as tree drift; scan bookings bind to the HE same-leg
convention like every other booking.

ATTACK: (1) knob re-scan at pin-44 — 35 sem legs (all 22 IG-era
knobs as stoch-pairs + all 6 post-IG knobs both ways): all
byte-inert except prera (silicon 121134 x3, worse) and rlu==dshape
(125103, worse); the composition route closed. (2) Mechanism:
-mtt-tensix-optimize-counted-capture-peel (Init(0)) — the lane-GQ
exec-while-record first-trip peel extended to the counted-loop
capture class (the plain hoist priced -833 because the no-exec
record re-delivers its payload; the peel prices +2037). It fires
hand's delivery class exactly (0,24,1,1 capture + 30 straight
launches; the launch-loop unroll composes) — and silicon measures it
WORSE: 121006 -> 121513 x3 (+0.42), hand byte-inert. THE ENVELOPE
LAW'S FIFTH CONFIRMATION. With the delivery pathway
all-shapes-measured (rolled 121006 best / peel+unroll 121513 /
dshape-rlu 125103 / prera 121134), addrsqrt-fresh is CERTIFIED at
LOSS +0.98 (the HK/HG standard): the residual is row-execution
content at the 8-LREG wall (sem 24w/row delivery-paced at 29.46
cy/row vs hand 25w/row execution-paced at 29.19; walk pressure 7/8
with PRGM L12-L14 exhausted; the one in-loop loadi is a CC-merge
lv). Named successors: a licensed/fresh-source CC-dance restructure
below 24 words, or an 8-LREG rename mechanism. 8 dg twins incl the
trips near-miss refusal.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin45; pinned-install
  env): 7096 PASS = pin-44's 7070 + IO's 26 exactly; FAIL set 16
  rows LINE-IDENTICAL to the pin-44 frozen baseline (diff empty).
  Flags smoke-accepted (OPTCHECK).
- Corpus: base-vs-fix OFF/TD/ON-36 3300/3300 x3 .text-identical;
  the pin43->44 ON delta verified = exactly the 2 lane-IN TUs (lane
  IM corpus-inert re-verified); knob delta = 7 corr TUs, all
  dump-attributed to the one class (signbit x3, relu-hand, unary
  max/min, gcd); paired CRAQ 16/16 on the pinned sims (bh
  1d162f0adf67); the 84-node headline screen (877 ELFs/leg) all
  byte-identical.
- Silicon (BH p150, 3-rep, corr-first): sem 121006 (lane-IN cell
  reproduced), hand 119837 x3 exact; the peel form 121513 x3.
- Board: cert note appended only (71c2fa3c27c3 -> 518f0d0112e5,
  columns 1-8 diff-verified identical); tally 83W/36P/15L.
  STATE: every loss on the board is certified or floor-named.
- Evidence: laneIO-evidence-20260829 (+SHA256SUMS 6428).
- Install: sha-verified a68c361f5cd5... -> 1f2b3baf48b4...; driver
  read from the fresh manifest entry.
- ON set UNCHANGED at 36 (counted-capture-peel registers as an
  on-plus booking knob).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin45-ceremony/).
