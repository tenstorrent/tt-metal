# REVIEW_RECORD — pin 44

cc1plus sha256: a68c361f5cd5e622f1a6a956dde38539d85c0c174e81060f498c190ffc7ce1e5
driver (g++) sha256: 1fed40d880f1a7f935688dc8e900f3a4db08ffb490598b71865c56451c0f9abf
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi c95ee644414 = pin-43 union tip d1e0ae6565a
+ two lane merges (IN agent/laneIN-hand-arm 8de41b1c2e8, IM
agent/lcm-window-sizing 92803f18513). Companions: tt-metal chain
through 0961da593a8b (IN notes cbb70c26e753 + IM knob registration
f2d0c53696 + measured/closure note). KNOB_MODES dup grep clean (only
the known benign lut-select-fp16 pair). No sfpi include/ changes.
Built in gcc-build-laneFR (build-pin44.log rc=0); flags
smoke-accepted (OPTCHECK, mcpu=tt-bh); installed via
pin-install-fast with loud --expect-cc1plus; no live sweeps at
install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

Two lanes — the flip-or-certify endgame pair — each with its own
closing report, evidence dir (+SHA256SUMS) and memory file:

IN (park-ordering PRE-PEEL PLACEMENT, same-flag refinement under the
HY precedent): the HN hand-arm +0.65 was 8 words per face-loop entry
of double materialization — park-ordering deferred all 9 in-region
constants, the residency walk peeled iteration one, and
pressure-park's LREG tier hoisted 4 candidates to the POST-peel
programming point while the peel had already duplicated their loads
inline. Fix: peel-class candidates place at the peel-block head and
the peel's duplicate is erased, gated by a new kill-aware pre-peel
ambient-all-lanes proof (the canonical tail's word-exact all-lanes
SFPENCC is the kill; the pre-existing cc_write_reaches_point_p has
no kill modeling and self-refuses every canonical loop around the
enclosing backedge); fail-closed refusal
park-prepeel-ambient-unproven; 4 dg twins incl the outer-kill fire
(the softplus anatomy) and the ambient near-miss. RESULT: hand
139060 -> 138289 x3 (+0.65 -> +0.09 vs the EL-era form; the last
word is walk candidate-ranking, named polish), sem 134841 -> 134204
x3; the softplus WIN -2.95 stands with BOTH arms faster; logsigmoid
-1.23 bonus (166057 -> 164010, causal -15.11). ON-36 delta =
exactly 2 corr TUs (softplus fp32 + erfinv production), paired CRAQ
10/10; fix-attributed silicon account -4351cy with +512cy on the
erfinv hand arm only (a deep-WIN row, class unaffected); zero
fix-caused class regressions; 84-node WIN screen run. HONEST DRIFT
CATCH booked: addrsqrt-fresh's WIN -2.26 was already dead at pin-43
— hand-tree drift between pins 39 and 43 (the hand TU is
byte-identical across the lane's binaries; the sem base reproduced
exactly) — re-booked LOSS +0.98 with the re-attack successor named;
lgamma-fitted +66.18 and erfinv-fresh -42.03 drift re-books
alongside.

IM (-mtt-tensix-optimize-replay-window-sizing Init(0)): the lcm
window-sizing bound REFUTED by construction plus measurement. The
autopsy found both suspected pieces real but the trim decisive:
pick_replay's (clones-1)x(length-1) key is measured-right in-block
(the IH result), and the hand 28x4 shape was INEXPRESSIBLE — its
4th delivery is a partial launch (REPLAY Count < recorded length)
that the former never emitted; the 4-word inline trim is a
word-exact window prefix per the ISA launch semantics (hand's
REPLAY(0,13) is the silicon witness). The knob gates hoisted-window
re-pricing + the partial-launch trim behind a full re-proof of the
hoist admission (7 named refusals, 8 twins) and REACHES HAND'S
EXACT 4-LAUNCH ROW (entry record widened 14->28 words; per-trip
deliveries 7 -> 4, dump-proven). SILICON (corr-first, paired CRAQ
2/2, device corr both legs, 3 reps cycle-identical): sem off
678213.0 x3 exact; knob 682580.0 x3 = +0.64 WORSE; hand 649518.0 x3
exact, knob byte-inert. lcm's cell UNCHANGED at +4.42 — no board
edit (snapshot asserted). THE WINDOW-DENSITY DELIVERY PATHWAY IS
CERTIFIED CLOSED (HS -> HH -> IL -> IM): every delivery shape of
the Stein body is silicon-measured — per-row re-record 680400,
7x14+inline 678213 (the minimum, booked), hand-exact 4x28+partial
682580 — the envelope law's fourth confirmation; the ~<=1.5pp
window-sizing bound refuted; the +4.42 residual is the round-shape
execution class. Preservation: topk (launch-flatten leg), copydest
(load-carrier leg), blaze sum+max t8 both arms, gcd — all
byte-identical under the knob; knob census 11 TUs all attributed
(one widening class) with paired CRAQ 12/12; the 84-node screen
delta = the unarymaxmin perf pair only, dump-verified sound.

With both lanes closed, the flippable tail is EMPTY except the
IN-named addrsqrt drift re-attack: every remaining loss on the board
is certified or floor-named — the owner-ratified flip-or-certify
accounting state.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin44; pinned-install
  env): 7070 PASS = pin-43's 6987 + IN's 23 + IM's 60 exactly; FAIL
  set 16 rows LINE-IDENTICAL to the pin-43 frozen baseline (diff
  empty). Flags smoke-accepted (OPTCHECK).
- Corpus: IN — first pin-43 base store (corpus-legs-laneIN),
  OFF/TD 3300/3300 x2 identical, ON-36 delta = exactly 2 corr TUs;
  IM — store corpus-legs-laneIL reused with pin42->43 zero-drift
  verified, OFF/TD/ON-36 3300/3300 x3 identical, knob census 11 TUs.
- Paired CRAQ green in both lanes on the pinned sims (bh
  1d162f0adf67); device corr corr-first every session; the lcm perf
  node stays corr-gated per the HU long-perf precedent.
- Board: 564b0693ea -> 71c2fa3c27c3 (IN's 7 rows incl the honest
  addrsqrt drift re-book); IM no board edit (asserted). Tally
  83W/36P/15L.
- Evidence: laneIN-evidence-20260828 (+SHA256SUMS 837),
  laneIM-evidence-20260828 (+SHA256SUMS 168).
- Install: sha-verified 7651df7fac78... -> a68c361f5cd5...; driver
  read from the fresh manifest entry.
- ON set UNCHANGED at 36 (replay-window-sizing registers as an
  on-plus booking knob; IN's fix is same-flag inside the reviewed
  park-ordering).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin44-ceremony/).
