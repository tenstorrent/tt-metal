# REVIEW_RECORD — pin 49 (FABLE_GOES_BURR wave 1)

cc1plus sha256: da957c5793b72c0266d968934412a4f77a0b2c8c71688f19d4eb0591536c4c55
driver (g++) sha256: 17f9a895ffdd926560582570b64fb9ad6417d9a516c4f865ae6d1f4b880e423a
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver REBUILT — JR/JT
added .opt options, drivers ship with cc1plus)
source: sfpi-gcc nkapre/sfpi 1bb9b654b53 = pin-48 6af4fb42f9b + the plan
commit 1f0fd915674 (gcc/config/riscv/tt/FABLE_GOES_BURR.md) + four wave-1
lane merges: JQ agent/laneJQ-refusal-registry ffcbfd476c85 (#1), JS
agent/laneJS-generated-tables 085a529863a (#3), JR
agent/laneJR-trips-facade 79b7c4e6f1a (#2 stage A), JT
agent/laneJT-briggs-coalescing e70c1a4610e (#6). Companions: tt-metal
canon chain through d585ec046d (JN sweep tool + JU coeff repairs + JO
prover merges) + JT witness merge (agent/laneJT-coalesce-witness
0f7f06ba11). KNOB_MODES 48 entries, dup grep NONE. Union conflict
resolutions reviewed: t-riscv-tt both-appended blocks kept;
gimple-rvtt-store-fold.cc composes JS table dispatch with JQ
licensed-variant emission in the STOREFOLD_REFUSE arm (out-of-scope
licensed gate preserved); orphaned pre-table ladder deleted. Built in
gcc-build-laneFR (make all-gcc, sfpi binutils on PATH, pinned auto-host.h
per the laneJS gotcha; build-pin49.log rc=0); OPTCHECK + installed-driver
smoke incl the two new flags; installed via pin-install-fast with loud
--expect-cc1plus.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

FABLE_GOES_BURR WAVE 1 — the audited-roadmap execution opens. Four items,
all CLASS-I byte-identity held except JT's Init(0) knob (ON-delta EMPTY):

JQ #1 REFUSAL REGISTRY: all named-refusal emission through one registry +
helper (245 direct points, ~230 helper sites); catalogue generated at
build with duplicate-name and unregistered-literal build-breaks
(red/green proven); -fopt-info-missed surfaces refusals (all 55 tt
passes OPTGROUP_NONE->OTHER); per-name counters; 85 UNNAMED prose
refusals enumerated in laneJQ RESIDUE.tsv — OWNER NAMING REVIEW OWED
(name-minting is frozen API).

JS #3 GENERATED VERDICT TABLES: rvtt-storefold-verdicts.def GENERATED
from tt/proofs RESULT censuses (fire class DERIVED: EQUAL->FIRE,
all-denormal NOT-EQUAL->LICENSED, else REFUSE); byte-compare .chk on
every build + reviewed-regen target (red/green proven); madpair
discovery/combine mirrors deleted for one generated query; crf
plan-order single-interpreter; zero source-of-truth discrepancies; bonus
-fchecking corpus leg ran deleted mirrors as recompute-asserts, zero
disagreements. Build-env gotcha banked: never reconfigure the pinned
build dirs (auto-host.h feature drift emits spurious .cfi_*).

JR #2 TRIPS FACADE STAGE A: rvtt-trips dual-oracle (legacy bounded
simulation DECIDES both faces; loop-iv + SCEV number_of_iterations_exit
cross-check; trip-oracle-divergence named dump; testing knob
-mtt-tensix-trips-oracle-skew Init(0)); census ZERO disagreements over
23,100 dumps / 4 corpus legs (gimple: agree 1,194, classical-only 49,008
= the stage-B widening surface, legacy-only 0). STAGE-B FINDINGS BANKED:
RTL loop-iv is hard-register-blind (simple_reg_p) and replay runs
post-reload — stage B must carry the GIMPLE proof across expand;
number_of_latch_executions unusable pre-expand.

JT #6 BRIGGS/GEORGE COALESCING: Init(0)
-mtt-tensix-optimize-lreg-coalesce in lp-alloc before spill selection;
ERFINV IP-2 EXPERIMENT REFUTED STRUCTURALLY (rvtt_dst_ownership precedes
rvtt_lp_alloc — coalescing cannot relieve the 9>8 refusal; refusal
re-certified with coalescing in the census; relief owned by plan item
#13); corpus never crosses the 8-LREG wall at ON-36 (DSATUR engages
0/5451; ON-delta EMPTY, 113/113 byte-identical; AB-TD identical); 57
twins incl the wasted-victim fire anatomy (2 spills -> 1) and George
precolored-pair; machinery banked for wave 3 pressure creators.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin49; pinned-install env):
  7294 PASS = pin-48's 7222 + exactly JQ's 4 + JR's 11 + JT's 57; FAIL
  set 16 rows LINE-IDENTICAL to the pin-48 frozen baseline (diff empty);
  dg ERROR count == 0. OPTCHECK green incl lreg-coalesce and
  trips-oracle-skew re-smoked on the INSTALLED driver.
- Corpus: per-lane byte gates all green pre-merge (JQ 3300x3 identical;
  JS 3300x3 + -fchecking leg; JR 13,200 rows x4 legs 0 changed; JT
  AB-ON/TD 113/113 identical, mine-vs-pinned 20/20).
- Board: UNCHANGED 84W/35P/15L by this pin (lane JU's coeff-repair
  re-books are a separate tt-metal booking, board b561204323 — margins
  moved, classes unchanged, no compiler change).
- Install: sha-verified 93f973d9dd94... -> da957c5793b7...; driver
  rebuilt (new .opt options), read from the fresh manifest entry.
- ON set UNCHANGED at 36; new knobs: lreg-coalesce (KNOBS), trips-oracle-skew
  (testing-only, Init(0)).
- Push state: pending at record time; pushed both hops + tt-metal in the
  ceremony commit (verified before pin close).

## Gates

conf_lint GREEN; witness_preflight at ON-36 on the installed binary in
~/sfpi-uplift/sweep-2x2/pin49-ceremony/.

## Witness re-seat (the pin-11 rule, exercised)

The crosscall-hoist R9 witness went RED at first preflight: its expected
line 'hoisted 6 contract materializations' rode the sigmoid-tree vehicle
whose semantic body lane JU repaired (dispatch scalars -> template
constants), shrinking the hoisted contract set to 4. Re-seated to
'hoisted 4 contract materializations' after TWO-SIDED verification at the
pin-49 union (positive: single-row A/B compile shows the line; negative:
union-minus-crosscall-hoist compiles rc=0 with zero rvtt_crosscall dumps
— non-vacuous). conf_lint GREEN and witness_preflight ALL GREEN after the
re-seat (pin49-ceremony/wp3).
