# REVIEW_RECORD — pin 34

cc1plus sha256: e6cf49e9bd832fab780236d681bcbf60a51f87740a71c1e68b230f20f65521ca
driver (g++) sha256: 85dc3d86dc507ecec24ff1fb7f532b023267af7fb2ab70961f1042ef961d5732
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi 386c65166c7 = pin-33 union tip 624e8448977
+ two lane merges (HT agent/tanhderivlut-4word 81be41cecde, HS
agent/lcm-opaque-region ec1329697ec). Companions: tt-metal chain
through c181f61908d (HS opaque-replay-record knob registration;
KNOB_MODES dup grep clean — only the known benign lut-select-fp16
pair). No sfpi include/ changes. Built in gcc-build-laneFR
(build-pin34.log rc=0); new flags smoke-accepted together (OPTCHECK,
mcpu=tt-bh); installed via pin-install-fast with loud
--expect-cc1plus; no live sweeps at install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

Two lanes, each with its own closing report, evidence dir
(+SHA256SUMS) and memory file:

HT (lut-select 4-word unlock, riding the reviewed
lut-select-leaf-ext flag — byte-identical-off proven): the licensed
tanhderivlut residual 5th word was LRA reloading the const-leaf into
the LUT's implicit slot hard register in-loop (SFPMOV L6,L10 at the
use, a RAW hazard before SFPLUTFP32) — not the mad addend. Fix:
slot_coeff_operand gives FLOATB-exact creg slot coefficients their
own preheader FLOATB materialization; the mad reads L10 directly.
ISA-verified against tt-isa-documentation: SFPMAD's VC is a plain u4
(LReg10 legal direct addend); SFPLOADI FLOATB imm16<<16 is bit-exact
for 1.0 and 0.0 — NO value change, the CY/GU accuracy license
untouched; non-FLOATB cregs refuse by name. Booked pair leg = pure
4-word replay rows vs hand's 5. tanhderivlut-fresh +6.62 -> WIN
-0.86 (3-rep cycle-identical; anchors reproduced exactly; +0.02 vs
the strongest hand arm noted on-row). dg focused 270/270 incl the
non-FLOATB refuse-and-keep near-miss. Named: the pinned sim cannot
run pin-33-ceremony PERF nodes (riscv_debug_regs_wr32 offset 0x80) —
the owed craq-sim ec15220f pin-bump now also blocks perf-node CRAQ.

HS (-mtt-tensix-optimize-opaque-replay-record Init(0)): HR's lcm
blocker opaque-region-undeclared adjudicated = TTI_REPLAY(0,28,0,1),
a REPLAY RECORD in the inlined gcd init plus its 28 recorded SFPU
words. The BH tt-isa-documentation tree lacks REPLAY.md; the WH
functional model and the pinned sim's replay-expander arm agree that
load=1/exec=0 window words are architecturally swallowed. The TU
PRGM freedom proof now derives through record windows (the laneBL
derivation discipline — derived from decoded fields, never trusted):
a no-playback theorem (raw executes refuse by name; the typed replay
passes bound record->launch extents against any asm/call) plus a
fail-closed region walk (straight-line + one structurally-counted
loop with exact trips; interleave/count/shape/trip refusals; belt
refusals for recorded SFPCONFIG, SFPLOADI-high, nested expander
words; the Exec=1 neutered-nested hole found and closed by
adversarial self-review, twinned). Unlock: the cc-peel lift parks
sfploadi 19200 in L14 once at the lifted entry; the inner Stein loop
drops 14 -> 11 words. lcm-fresh +6.61 -> +4.75 — an honest
improvement, still a LOSS: the residual is the algorithmic
replay-window-density gap vs the replay-buffered hand body, named
for successors. Hand arm byte-identical (raw REPLAY executes keep it
refused — the inertness control); knob-only arm byte-identical
(honest pricing refusal, break-even 46 trips); divint32floor
reproduces HR's cells exactly; gcd byte-identical. dg 58/58.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin34; env = the pinned
  install): 6598 PASS = pin-33's 6512 + HT's 28 + HS's 58 exactly;
  FAIL set 16 rows LINE-IDENTICAL to the pin-33 frozen baseline.
  New flags smoke-accepted together (OPTCHECK).
- Corpus byte-gates: HT OFF/TD/ON-36/LIC36/LICRLU36 all 3300/3300
  byte-identical (fire contained to the 2 SFPU marker variants of
  the licensed sem TU); HS OFF/TD/ON-36 3300/3300 x3 + knob 0-delta
  + composition delta = exactly the 2 atan2 cc-peel TUs
  (build.h-attributed). corpus-legs-laneHS = the first pin-33 base
  store; corpus-legs-laneHT banked.
- CRAQ on the pinned sim (32489dda) green both lanes (HT 10/10; HS
  paired on both corr and perf nodes). Device corr corr-first
  everywhere.
- Silicon (BH p150, 3-rep, cycle-identical): tanhderivlut-fresh WIN
  -0.86 same-leg (HT); lcm-fresh +4.75 (HS). Anchors reproduced the
  booked cells exactly in both lanes.
- Evidence: laneHT-evidence-20260826, laneHS-evidence-20260826
  (SHA256SUMS in each). Board edited via GE mechanics with
  snapshots; BOARD 77W/35P/23L.
- Install: sha-verified 1f0f5a44a001... -> e6cf49e9bd83...; driver
  read from the fresh manifest entry; no sfpi include/ staging owed.
- ON set UNCHANGED at 36 (opaque-replay-record registers as an
  on-plus booking knob; the 4-word unlock rides leaf-ext).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin34-ceremony/).
