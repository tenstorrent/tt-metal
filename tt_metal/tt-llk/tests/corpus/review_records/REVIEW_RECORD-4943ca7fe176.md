# REVIEW_RECORD — pin 32

cc1plus sha256: 4943ca7fe17642e935f32aee81c9496f5a9a754e86e4497d5f3b4dc93d39f701
driver (g++) sha256: 89fdc2f03b4274900d83cfa6d4703ea8ad5cb862b2d816c410006c1227595a7e
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi 8012f513916 = pin-31 union tip 657413d694b
+ two lane merges (HN agent/el-park-ordering dcae8a2069e, HO
agent/store-source-tier 587f59ea055). Companions: tt-metal chain
through 32c6540fa3 (HN knob 15da120795 + HO knob b55ee81fef; one
sweep_2x2.py both-append conflict union-resolved, py-parse +
conf-lint verified; the long-standing lut-select-fp16 KNOB_MODES
duplicate remains the known benign pair). No sfpi include/ changes.
Built in gcc-build-laneFR (build-pin32.log rc=0); both new flags
smoke-accepted together with the pin-31 knobs (OPTCHECK,
mcpu=tt-bh); installed via pin-install-fast with loud
--expect-cc1plus; no live sweeps at install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

Two lanes, each with its own closing report, evidence dir
(+SHA256SUMS) and memory file:

HN (EL-vs-residency ORDERING, -mtt-tensix-optimize-park-ordering
Init(0): a CC-restore loop with hoist candidates defers them all to
the 295t residency walk, refusal-named residency-walk-ordering,
iff const-residency and pressure-park are both enabled — an
authority transfer, not a coverage change. Root cause: ONE EL hoist
(the polynomial lv-carrier) carried TWO composition defects — +2
phantom function pressure at 295t that starved pressure-park's
budget from 3 parks to 1, and lv-carrier disabled-lane forging that
blocked rvtt_live's break_liveness elision so RA emitted a
per-iteration all-lanes sfpmov. The knob-on softplus kernel .text
equals the noel 26-word hand-parity bytes EXACTLY; flag-off equals
the pin-31 base exactly. softplus-fresh +5.93 -> WIN -3.03, 3-rep
cycle-identical, corr-first 4/4, anchors reproduced the booked
cells exactly. 19 dg twins; knob delta 19 TUs all varmap-mapped
(the deferral class: 13 unary CC-loop bodies incl
softplus-production + 6 logsigmoid), CRAQ 34/34. Named successors:
the hand arm's own +0.65 residual; ON promotion owes the full R9 +
ON-vs-ON attribution ceremony given the 19-TU delta.)

HO (HL-F1 generalization, -mtt-tensix-optimize-store-source-tier
Init(0): store-consumed loop-class prgm-const candidates take the
pressure-park LREG tier first, with a strictly-never-worse
fallthrough that keeps the park byte-identically; priced in
rvtt-cost.md STORE-SOURCE TIER; HL's license refusal in place()
kept verbatim and license precedence proven at scale. CENSUS
VERDICT: the compiler-attributable unlicensed copy tax at pin-31
ON-34 is ZERO corpus TUs — of 264 sfpmov-from-L12+ sites, 88
store-consumed, 80 are moe_gate L15 lane-index reads (not a PRGM
register, out of class) and 8 are hand-written moe_gate
TTI_SFPCONFIG L14 source parks (LLK author choice, not a compiler
defect). The one perf-vehicle fire (HO-F2 probe over 54 fresh
causal-lift nodes) is fill-fresh: 1 SFPMOV word/row; WIN deepened
-21.21 -> -25.05 (sem-ds+knob 12474, 3-rep cycle-exact, hand and
sem-ds anchors reproduced the booked lane-HD cells exactly, corr
12/12). 8 dg twins 53/53. Named: HO-F1 (non-store operand-position
PRGM copies, per-insn ceiling question), HO-F2 rule (corpus-inert
verdicts require a perf-vehicle probe).)

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin32; env = the pinned
  install ~/sfpi-uplift/sfpi/build/sfpi): 6476 PASS = pin-31's 6404
  + HN's 19 + HO's 53 exactly; FAIL set 16 rows LINE-IDENTICAL to
  the pin-31 frozen baseline (diff empty). Both new flags
  smoke-accepted together on the union (OPTCHECK).
- Per-lane corpus byte-gates vs pin-31: OFF/TD/ON-34 all 3252/3252
  byte-identical in both lanes (Init(0) proven); HN knob delta 19
  TUs fully adjudicated; HO knob deltas 0/3252 on ON-34,
  ON-34+store-fold, and the LICENSED store-sink leg (fire set = the
  fill perf vehicles exactly). corpus-legs-laneHO = the published
  pin-31 base store for future lanes.
- CRAQ on the pinned sim (bh 32489dda) green both lanes (HN 34/34;
  HO fill 4-leg + moe control 89/89). Device corr corr-first
  everywhere (HN 4/4; HO 12/12).
- Silicon (BH p150, 3-rep, cycle-identical): softplus-fresh WIN
  -3.03 same-leg (HN); fill-fresh WIN -25.05 (HO); all anchor pairs
  reproduced booked cells exactly.
- Evidence: laneHN-evidence-20260826, laneHO-evidence-20260826
  (SHA256SUMS in each).
- Install: sha-verified 4182e7a23ab7... -> 4943ca7fe176...; driver
  read from the fresh manifest entry; no sfpi include/ staging owed.
- ON set UNCHANGED at 34 (park-ordering and store-source-tier
  register as on-plus booking knobs). BOARD at cut: 76W/35P/24L.

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-34 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin32-ceremony/).
