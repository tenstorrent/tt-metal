# REVIEW_RECORD — pin 51 (FABLE_GOES_BURR wave 3 + scaffolding deletion)

cc1plus sha256: 77ce6392c080fd45925990fa68daa82ce62bbc104d18ec854d39be0910ccf6b5
driver (g++) sha256: 456b4a2720b79f3127e457062bf9a7280efa22b856479fdaf464957e7c21523b
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver REBUILT — four
new wave-3 .opt options)
source: sfpi-gcc nkapre/sfpi 1b531ee842f = pin-50 b329ce192e6 + five
wave-3 merges (union 164829f31db) + the scaffolding-deletion commit:
KD agent/laneKD-rau-modulo-sched d1c697409de (#5), KE
agent/laneKE-duchain-regrename 45087ab7082 (#7), KF
agent/laneKF-placement-arbiter 4376fa38bd2 (#13), KG
agent/laneKG-accum-splitting c574e396419 (#8), KH
agent/laneKH-ipa-summaries b49ec40c5ed (#15).  tt-metal companion:
agent/laneKF-placement-witness fdb5ade960 merged (KNOBS priced-placement,
49 entries, dup grep NONE).  Union conflicts: t-riscv-tt object list
only (all three new objects kept); KF∩KH crosscall seam auto-merged
disjoint exactly per KH's pre-stated plumbing/decision split; KE's
pass-order requirement verified line-identical to its branch.  Built in
gcc-build-laneFR (make all-gcc, pinned auto-host.h, rc=0 twice: union +
deletion); OPTCHECK + installed-driver smokes for all four new knobs;
installed via pin-install-fast with loud --expect-cc1plus.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

FABLE_GOES_BURR WAVE 3 — the machinery wave.  Full lane records in each
lane's evidence dir (laneK{D,E,F,G,H}-evidence-20260831/RESULTS.md);
headline facts:

KD #5 RAU IMS+MVE: modulo scheduling in the ONE timing engine (shared
marshaller with the acceptance simulator; 43/43 unit tests); acceptance
authority unchanged; knob leg = exactly 9 changed TUs, 9/9
dump-attributed strict II decreases, paired-CRAQ outcome-identical;
trig's laneIJ +0.63 constrained-floor cert RE-CERTIFIED by byte-identical
independent re-derivation; the wall named: mve-rename-exhausted kmin=2
demand=36/44 capacity=8 (the R1 doorway fact).

KE #7 DU-CHAIN REGRENAME: general engine beside untouched v1; 2,446
chains renameable (v1: 13, not in ON); per-refusal wall profile; service
export rvtt_lreg_rename_chain for R1/#5; post-commit lockstep belt
red/green proven; a dev-time wrong-code near-miss was belt-caught and
fixed pre-ship; v1 NOT-YET-SUBSUMED (3 residual sites) -> W4-C.

KF #13 PLACEMENT ARBITER: one priced placement authority; MONOTONE
FAIL-CLOSED (its own census caught a trig keep->defer regression of the
pin-34/35 shape pre-close).  ERFINV FLIPPED: 1 reload folded, zero
lreg-pressure-exceeded, .text -296B, device-golden 6/6, exactly one
moved TU corpus-wide.  Words/bytes evidence; NOT booked; promotion-time
obligations named.

KG #8 LICENSED ACCUM SPLITTING: both-keys license wall; census = ZERO
fires, ONE candidate corpus-wide (polygamma k=1, refuses by name) —
VACUOUS ON THIS CORPUS, stated plainly; machinery banked.

KH #15 IPA SUMMARIES: per-cgraph digests, three conversions
shadow-proven zero disagreements; cc-region ambient carry dump-only; asm
tightening census clean.

## Scaffolding deletion (1b531ee842f, -1,212 lines)

The pin-50 one-pin list, served: item #3 five assert phases (OVERDUE
from pin 49 — the pin-50 ceremony missed them), #4 effects phase + both
legacy tables, #10 pressure LEGACY MIRRORS + mirror asserts, #12 twenty
recompute-asserts + the orphaned rvtt-schedule.h checking field.
ADJUDICATED DIVERGENCE from the pin-50 note: item #14-A (KA) asserts
carry no one-pin markers and are documented permanent
FINDING/fail-closed channels — KEPT BY DESIGN.  One dg test deleted with
its scaffolding (pressure-engine-reassoc-checking-bh.C, -3 checks,
counted in the gate arithmetic below).  PIN-52 DELETION LIST: KH's #15
shadows + the legacy asm parser.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin51): 7488 PASS = 7321 +
  KD 21 + KE 42 + KF 48 + KG 30 + KH 29 - 3; FAIL set 16 rows
  LINE-IDENTICAL; dg ERROR 0.
- UNION CORPUS GATES (sweep-2x2/pin51-ceremony/corpus/): pinned-50 vs
  union ON 3300/3300 byte-identical (diff-on.txt empty); union chkon
  rc=0, zero assert ICEs, .text == union-ON (diff-chkon.txt empty).
  Per-lane byte gates were green pre-merge in each lane's evidence.
- OPTCHECK + installed-driver smokes: ims, lreg-rename-chains,
  priced-placement, reassoc-loop-carried, -fchecking.
- Board: UNCHANGED 84W/35P/15L (nothing booked this pin).
- ON set UNCHANGED at 36; KNOB_MODES 48 -> 49 (+priced-placement).
- Install: sha-verified 4a85f9e26d62... -> 77ce6392c080...; driver
  rebuilt, read from the fresh manifest entry.
- Push state: pending at record time; pushed tt-metal + sfpi-gcc both
  hops + the private craq mirror per the four-target policy (verified
  before pin close).

## Gates

conf_lint GREEN; witness_preflight at ON-36 on the installed binary in
~/sfpi-uplift/sweep-2x2/pin51-ceremony/ (re-seats noted below if any).
