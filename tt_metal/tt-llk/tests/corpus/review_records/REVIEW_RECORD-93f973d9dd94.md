# REVIEW_RECORD — pin 48 (the measurability-closures pin)

cc1plus sha256: 93f973d9dd94e80c13b26bfab28a0b5c157c6c479c95da44652342f234f00dab
driver (g++) sha256: 38af15bd3390404351149ab60dd971dd46ec91bbe8943556507af55e9ac956ca
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with
cc1plus and REPRODUCED IDENTICAL to pin-47's driver — no .opt changes)
source: sfpi-gcc nkapre/sfpi 6af4fb42f9b = pin-47 union tip 7190c2460a3
+ lane IZ merge (agent/lp-schedule-debug-transparency, fast-forward).
Companion: tt-metal ff41285609 (IZ witness rows + KNOBS measurability
comments; conf +27 / sweep_2x2.py +14 comment-only). KNOB_MODES dup
grep: NONE (AST-scoped, 47 entries). No sfpi include/ changes. Built
in gcc-build-laneFR (build-pin48.log rc=0, make all-gcc with the sfpi
toolchain binutils on PATH); OPTCHECK smoke incl the milp flag pair;
installed via pin-install-fast with loud --expect-cc1plus (first
attempt rolled back cleanly on a --flags spelling error; second
attempt GREEN); no live sweeps at install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

LANE IZ MEASURABILITY CLOSURES — the adversarial audit campaign's two
open dispositions (IP-6, IP-2) closed at the compiler layer:

IP-6 MILP MEASURABILITY. The pressure-schedule/milp pass gated on
debug_info_level == DINFO_LEVEL_NONE while the corpus harness
compiles with -g (test_config.py debug_flag): the pass had NEVER run
in a harness compile — every milp knob cell ever booked was an A/A
leg — and the gate was compare-debug-dirty when active (the gate
verdict: doubly wrong). Fix: the pass is debug-TRANSPARENT (debug
stmts transparent to region formation; pressure model debug-blind;
span binds re-emitted post-region; gate condition dropped), with 4
new twins incl -fcompare-debug identity twins the old gate fails by
construction (16 new dg checks total); family suite 108/108.
FIRST-EVER measurement (sharded compiler-A/B census, harness -g kept,
plus a pin-47 control census): the pass RUNS in all 5,451 harness TUs
(control: 0 dumps — corpus-wide A/A proven); applied=yes ZERO across
524,864 regions (524,767 rejected=cfg; the 64 fully-analyzed regions
all <=8 LREGs); ON-36 mine-vs-pin47 5,564/5,564 byte-identical —
Init(0) PROVEN, zero fires, no silicon owed. Pass-matrix answer:
"milp: measured, zero fires at ON-36" (its corpus domain is
straight-line single-BB SFPU functions; loop-shaped LLK kernels
honestly refuse by cfg). One 3-TU knob-flag ripple adjudicated: the
rvtt_lp_alloc audit hook's df_note_add_problem — pre-existing at
pin-47, deterministic, paired CRAQ 414/414 on both legs on the pinned
sim — filed as a named P3 hygiene successor.

IP-2 DST-OWNERSHIP DISPOSITION. Adjudicated NOT-extinct: erfinv's
fold fires solo (1 reload folded); at ON-36 the same proven
candidates refuse lreg-pressure-exceeded (9 > 8) — composition-
pressure growth ON-28 -> ON-36 with the refusal belt working
correctly; addcmul died of body evolution (dst-store-may-alias /
lossy-join). Gotcha banked: the flag is Init(1), attribution needs
-mno- (probed both directions). Transform-stage R9 witness seeded on
erfinv (two-sided at the pinned toolchain). Successor named: price
the fold through the pressure-park tier.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin48; pinned-install
  env): 7222 PASS = pin-47's 7206 + exactly the 16 new IZ checks;
  FAIL set 16 rows LINE-IDENTICAL to the pin-47 frozen baseline
  (diff empty); dg ERROR count == 0. Flags smoke-accepted (OPTCHECK,
  mcpu=tt-bh; milp pair re-smoked on the INSTALLED driver post-install).
- Census evidence: laneIZ-evidence-20260830 (MILP-MEASUREMENT.md +
  IP2-ADJUDICATION.md read-first; SHA256SUMS). ON-36 byte-identity
  5,564/5,564 mine-vs-pin47 makes the pin corpus-inert by
  construction; no board cells move.
- Board: UNCHANGED 84W/35P/15L (ec2119b5942f tally intact) —
  measurement + twins only.
- Install: sha-verified dc1287020336... -> 93f973d9dd94...; driver
  read from the fresh manifest entry, reproduced identical.
- ON set UNCHANGED at 36; no new knobs (milp already registered).
- Push state: sfpi-gcc nkapre/sfpi 6af4fb42f9b on BOTH hops (hub +
  github ls-remote verified); tt-metal nkapre/sfpi ff41285609.

## Gates

conf_lint and witness_preflight run at ON-36 on the installed binary
(outputs in ~/sfpi-uplift/sweep-2x2/pin48-ceremony/).
