# REVIEW_RECORD — pin 35

cc1plus sha256: 133619cf6b77005e275a6a551ed4e39ddb0e174746dab37210f051a5f91f8f0c
driver (g++) sha256: bd4695a7812f40768d001ec4873b088af3b629732720429852d34a96de40c9a3
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi 0befdec8214 = pin-34 union tip 386c65166c7
+ lane HY merge (agent/park-seed-composition: 4367cf26c15 +
0befdec8214 keep gates). Companions: tt-metal chain through
1431f3226e40 (+ the HW/HV/HX note branches merged earlier tonight:
9461e1130fa0, 90a5f76cd30d, a73f8366932a). KNOB_MODES dup grep clean
(only the known benign lut-select-fp16 pair). No sfpi include/
changes. Built in gcc-build-laneFR (build-pin35.log rc=0); flags
smoke-accepted (OPTCHECK, mcpu=tt-bh); installed via
pin-install-fast with loud --expect-cc1plus; no live sweeps at
install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

One lane (closing report, evidence dir + SHA256SUMS 2342, memory
file), completing a three-lane finding chain (HV found the
regression via the loss-set rescan; HX delivered the dump-cited
mechanism; HY completed it per-row and shipped the fix):

HY park-ordering KEEP GATES (same flag, no new flag): the promoted
114t deferral handed CC-restore loops' entire invariant-immediate
candidate set to the 296t const-residency walk, which re-places
candidates only behind its manufactured CC-canonical first-iteration
peel — charging pure prologue words (rdiv +2 / softsign +2 / i0 +4 /
sqrt +7 words = 253-895cy) and, on even-trip paired loops, flipping
trip parity 32->31 so 372r refuses crossrow-pairing-trips-odd and
the paired 0,28/15-launch capture dies for Rule-A and the Rule-B
seed (ceil/roundingops, -4736cy forgone; HW-F1 was the downstream
symptom). Fix at the deferral admission: in-region (CC-depth>0)
candidates defer (both HN defect classes live in-region —
softplus's 9-candidate class dump-verified); depth-zero candidates
keep the early mask-exact hoist (refusal-named
depth-zero-hoist-dominant); wholesale keep-gates
lut-coefficient-authority (SFPLUT/SFPLUTFP32 bodies) and
in-region-demand (>=3 in-region constants — sigmoid-appx-tree
counterfactual archived: three unconditional keeps drove
lut-coefficient-pressure LREG 9>8, 29861->43447 on silicon).
Discoveries banked: 114t sees unrotated loops (CC-depth, not
position, is the sound boundary); i0's depth-zero lv-carriers refute
carrier-based deferral. dg focused 137/137.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin35; pinned-install
  env): 6633 PASS; FAIL set 16 rows LINE-IDENTICAL to the pin-34
  frozen baseline (diff empty). Flags smoke-accepted (OPTCHECK).
- Corpus byte-gates: OFF/TD 3252/3252 identical; ON-36 delta =
  exactly 2 TUs (tan/cosh), CRAQ-adjudicated; paired CRAQ 34/34 on
  the pinned sim 32489dda.
- Silicon (BH p150, 3-rep cycle-identical, corr-first 4/4 every
  session): ceil plain 66360 / roundingops restored; seed-knob
  62264/61490 = the booked cells with seed bytes byte-exact to the
  pin-31 forms (Rule-B II 30->28 restored); rdiv 50101, sqrt 112185,
  softsign 70714, i0 166457 — every registered drop-one leg
  reproduced BYTE-EXACT at plain ON-36 (recovery compositions
  retired); hardsigmoid-fresh WIN RESTORED 46263/53809 x3 (the
  promotion had regressed it unmeasured to 55097 — caught by HY's
  25-row WIN screen).
- Preservation: softplus ON-36 byte-preserved + 134841/139060
  cycle-exact x3 (R9 witness line survives); fill+delivery-shape
  byte-identical + 12474 exact x3; gelu/sigmoidlut/tanhlut/
  sigmoid-appx-tree byte-identical; all 23 loss-row hand arms
  byte-identical.
- Evidence: laneHY-evidence-20260827 (+SHA256SUMS 2342), with the
  HV/HX chains in laneHV-evidence-20260826 and
  laneHX-evidence-20260827. Board 9 rows re-attributed to plain
  cells + hardsigmoid restored; tally 77W/35P/23L.
- Install: sha-verified e6cf49e9bd83... -> 133619cf6b77...; driver
  read from the fresh manifest entry.
- ON set UNCHANGED at 36. PROCESS RULE banked (HV): ON-set
  promotions must gate on the loss-set rescan.

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin35-ceremony/).
