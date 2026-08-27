# REVIEW_RECORD — pin 37

cc1plus sha256: ec1ceaa776dd9195e680806fc1320b902587eeddb9d3af42bbe4a564f91d2579
driver (g++) sha256: 731587dfb881b617743408259c747b56a8b57a8ece5fc1f9de2351b6b70917c4
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi 39617ab327c = pin-36 union tip ff2199b67d5
+ lane IC merge (agent/tanh-2datum-pairing, fast-forward). Companions:
tt-metal chain through 5f3de5d5c4de (IC knob registration + measured
rows; the IB refusal notes d73d27af518d and the tanh-license
ratification 115208abce merged earlier). KNOB_MODES dup grep clean
(only the known benign lut-select-fp16 pair). No sfpi include/
changes. Built in gcc-build-laneFR (build-pin37.log rc=0); flags
smoke-accepted (OPTCHECK, mcpu=tt-bh); installed via
pin-install-fast with loud --expect-cc1plus; no live sweeps at
install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

One lane (closing report, evidence dir + SHA256SUMS 1041, memory
file), downstream of lane IB's REFUSE-BY-BAR adjudication (the
tanh accuracy-license premise was refuted by measurement: the hand
perf arm is the Sollya p6 at 0.810541 pure-ulp normal-range and the
sem body is numerically BIT-IDENTICAL — the +24.14 pair gap was
window density, never accuracy; IB's certificates live in
laneIB-evidence-20260827):

IC tanh 2-DATUM WINDOW FORMATION. Autopsy named two composed
blockers at pin-36 ON-36: (1) 372r crossrow pairing categorically
refused every next-slot acceptance-stall word
(crossrow-pairing-effect-unproven at the SFPSWAP; the word itself is
fully audited — result latency 0, replay-safe — only the stall
excluded it); (2) 296t const-residency left the function at 8/8
LREG pressure because the C6-C4 preheader hoists could not see the
TU-claimed bit-identical PRGM slots (only in-loop materializations
were walk candidates), so the pair renames starved. Fix = two
Init(0) flags (knob crossrow-2datum on-plus):
-mtt-tensix-optimize-crossrow-pairing-stall-words (stall words at
audited result latency, priced 2 issue slots in the II model per the
rvtt-cost.md consumer rule with recorded count 1; copies-first
rename priority; critical-path item selection — the earliest-ready
order left the mul->swap shadow bare, inserting an SFPNOP that
pushed the body to 33 words and LOST the capture to a rolled loop,
the adjudicated round-cc-modulo defect observed live mid-lane;
capture-overflow belt; audited_latency() untouched and the
downstream nop inserter re-discharges the lane-HM erratum contract
over the final order) and -mtt-tensix-optimize-hoisted-prgm-reuse
(the HOISTED-REUSE residency class: free slot, TU value-identical
reuse for the fresh path, and the DEAD-claim reclaim for the fitted
path via a TU-wide sfpreadlreg no-reader census + call-free
programming window; fail-closed named refusals). Delivered form on
both tanh rows: ONE 32-word 2-datum record + 15 launches, trips
32->16, zero SFPNOP, coefficients on CRegs (hand's register
discipline); modeled II 42->34. Residual named: 4 duplicated
coefficient loadi words per pair — hand's sequential shared-reload
register is inexpressible in the position-blind web vocabulary
(naive dedupe analyzed as wrong-code); modeled 34 vs hand's 30
slots/pair; successor named. dg focused 33/33.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin37; pinned-install
  env): 6728 PASS = pin-36's 6695 + IC's 33 exactly; FAIL set 16
  rows LINE-IDENTICAL to the pin-36 frozen baseline (diff empty).
  Flags smoke-accepted (OPTCHECK).
- Corpus (IC's own store): base-vs-fix OFF/TD/ON-36 3300/3300 x3
  .text-identical; knob delta = exactly 12 build.h-attributed TUs,
  all adjudicated (paired CRAQ on the pinned sims bh 1d162f0adf67 +
  device corr PASS/PASS; hardmish/hardtanh/sigmoid-appx via
  fresh-harness nodes; one sdpa scale TU = symmetric harness skip,
  named).
- Preservation: 71-leg loss+WIN screen 0 CHANGED; 9-leg
  seed-composed screen (ceil/rops/hardsigmoid/sigmoidappx/relu at
  ON+pairing-seed) 0 CHANGED — the HB/HY booked seed cells and every
  WIN row byte-preserved.
- Silicon (BH p150, 3-rep cycle-identical, corr-first 4/4 every
  session): anchors EXACT (sem 83640 / hand 67378, both rows); knob
  sem 75834 both rows -> vs-hand +24.14 -> +12.55, causal -36.19;
  hand arm byte- and cycle-inert.
- Board: FINAL-BOARD 542ab2c29f91 -> 0702d3fa7f4d (the two tanh rows
  only, still LOSS; tally 78W/35P/22L unchanged; the HA-Q2 twin fold
  remains an open owner question).
- Evidence: laneIC-evidence-20260827 (+SHA256SUMS 1041), with IB's
  refusal certificates in laneIB-evidence-20260827.
- Install: sha-verified 86fcd08e1bab... -> ec1ceaa776dd...; driver
  read from the fresh manifest entry.
- ON set UNCHANGED at 36 (both new flags Init(0); crossrow-2datum
  registers as an on-plus booking knob).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin37-ceremony/).
