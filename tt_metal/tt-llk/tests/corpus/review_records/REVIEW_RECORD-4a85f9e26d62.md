# REVIEW_RECORD — pin 50 (FABLE_GOES_BURR wave 2)

cc1plus sha256: 4a85f9e26d62a1945f98a2b4e979bfddc1e705f673bfb63fe93adeb79cced1db
driver (g++) sha256: ed8b531e3ead33821f246a00094a019c4a30dddc4d701b64dcc6c7b7db43fb73
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver REBUILT — KB
added -mtt-tensix-replay-shadow-discovery to riscv.opt; cc1/lto1 also
moved, expected: they link the new rvtt engine objects)
source: sfpi-gcc nkapre/sfpi b329ce192e6 = pin-49 1bb9b654b53 + six
wave-2 lane merges: JZ agent/laneJZ-attr-migration eb21e900a46 (#4), JW
agent/laneJW-pressure-engine 17d604c0815 (#10), KA
agent/laneKA-cc-region-tree 4f3b4d46575 (#14-A), KB
agent/laneKB-suffix-discovery 6640ae8ec87 (#9 stage A), JY
agent/laneJY-delivery-cost 73e469f8e46 (#12), JX
agent/laneJX-timing-engine 444cd1bf485 (#11).  Built in gcc-build-laneFR
(make all-gcc, sfpi binutils on PATH, pinned auto-host.h; two-stage:
5-branch union then the recovered KB merge; both builds rc=0); OPTCHECK
+ installed-driver smokes; installed via pin-install-fast with loud
--expect-cc1plus.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

FABLE_GOES_BURR WAVE 2 — the one-engine consolidation wave.  Six items,
every one CLASS-I (byte-identical corpus at every flag state, with a
live-assert -fchecking verdict-identity leg before any legacy engine
retired):

JZ #4 ATTR MIGRATION: 24 effect-table override rows + the 97-entry
allowlist retired into xtt_lane_local/xtt_cc_write/xtt_lane_gated
attribute declarations; re-freeze NO ORACLE ROW CHANGED (WP8 11/11
bit-identical, cc-enable identity 5/5).  One-pin assert phase in
rvtt-effects.cc — DELETE AT PIN 51.  Deliverable B (word-fact table)
still owed by the lane.

JW #10 PRESSURE ENGINE: tt/rvtt-pressure.{h,cc} unifies three
pressure/liveness mirrors plus the replay-unroll capacity literal;
incremental selector profile; verdict-identity census over 3300 chkon
rows ZERO disagreements; rvtt +6 checks.  Legacy in-module mirrors —
DELETE AT PIN 51.

JY #12 DELIVERY-COST API: tt/rvtt-delivery-cost{-core.h,.h,.cc} + 50/50
self-test (pin-13 -383 anchor, WP13 one-sidedness invariant); 17 sites
migrated (planner/crosscall/replay pricings, 3 comparators -> 1,
mop-form, prgm-const, invariant, bnb mirrors, delivery-shape,
dst-autoincr); model seams closed (mirror = the same replay_pricing as
the gate; autoincr setup = XTT_AUTOINCR_SETUP_COST_X100=0); loadi
spellings proven equivalent (no latent pricing bug).  One-pin
recompute-asserts — DELETE AT PIN 51.

JX #11 TIMING ENGINE: tt/rvtt-timing.{h,cc} (audited-latency
discipline, dependence classification, adjacent_stall,
simulate/cyclic_ii, 16-reg interlock_sim); FIVE simulators DELETED
(fill adjacency, ls_simulate, ls_cyclic_ii, prera makespan mirror,
replay scoreboard + bnb ds_* mirrors) only after a stage-A shadow chkon
leg with ZERO timing disagreements.

KA #14-A CC-REGION TREE: tt/rvtt-cc-region.{h,cc} pushc/popc frame tree
(refinement chains, ENCC/opaque poison, all-lanes fact, v_endif
drain-diamond stitching, fail-closed on unstructured CC); four consumer
passes keyed off the tree restricted to their old shapes with
flag_checking recompute-asserts; ZERO disagreements (dg-chk + 6,600 chk
TUs).  Recompute-asserts — DELETE AT PIN 51.

KB #9 SUFFIX-AUTOMATON DISCOVERY (stage A, shadow): the automaton runs
beside the legacy grow-by-one picker and never decides.  HEADLINE: the
legacy discovery is CANDIDATE-COMPLETE within the replay-buffer bound
(54007 == 54007 candidates, automaton-only 0) — stage-B value is the
1123 over-buffer maximal repeats (longest 104) and the maximality
structure, not new in-bound candidates.  New TESTING knob
-mtt-tensix-replay-shadow-discovery Init(0), NOT added to KNOB_MODES
(the trips-oracle-skew precedent: measurement-only knobs stay out of
the booking table); +21 rvtt checks.

## Union seams (reviewed compositions)

- JX∩JY hoist pricing (rtl-rvtt-replay.cc + rvtt-bnb.cc x2): resolved
  per the plan's joint-ownership rule — words->centislot ECONOMICS
  through JY's rvtt_delivery_cost::replay_pricing (the replay gate and
  both bnb downstream mirror sites price through the ONE spelling);
  execution-side simulation through JX's rvtt-timing engine (the
  clean-merged interlock_sim walker).  JX's counted_hoist_price /
  rerecord_hoist_price call sites retired at the seam.  Discharged by
  the union chkon leg (below), per JX's own merge note.
- KB∩JY comparator: both lanes had factored the SAME extend_sequence
  word-equality predicate (bodies line-for-line identical).  The union
  keeps one spelling — rvtt_dcost_replay_word_equal_p — at the legacy
  picker AND KB's automaton re-check; KB's duplicate static
  replay_ignore_rtx_p deleted (its own comment demanded one spelling).
- t-riscv-tt: all three new engine objects kept (cc-region,
  delivery-cost, timing); #include unions kept in
  gimple-rvtt-invariant.cc / gimple-rvtt-prgm-const.cc.
- KB BRANCH RECOVERY: agent/laneKB-suffix-discovery 6640ae8ec87 existed
  on NO remote and NO local ref (dangling commit in the lane clone; the
  lane's PUSHED record was false).  Caught by the union dg run coming
  up exactly KB's 21 checks short (7300 vs 7321).  Recovered by
  fetching the lane clone's HEAD, merged, and the branch re-anchored +
  pushed to both hops at this ceremony.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin50, pinned-install env):
  7321 PASS = pin-49's 7294 + exactly JW's 6 + KB's 21; FAIL set 16
  rows LINE-IDENTICAL to the frozen baseline (diff empty); dg ERROR
  count == 0.  (A pre-KB union run scored 7300 with the same FAIL-16 —
  that shortfall is what exposed the missing KB merge.)
- UNION CORPUS GATES (sweep-2x2/pin50-ceremony/corpus/): pinned-pin-49
  vs union ON leg 3300/3300 .text BYTE-IDENTICAL (diff-on.txt empty);
  union chkon leg (ON + -fchecking, every wave-2 one-pin
  recompute-assert live) rc=0, ZERO assert ICEs, .text == union-ON
  3300/3300 (diff-chkon.txt empty).  Per-lane byte gates were all green
  pre-merge in each lane's own evidence dir.
- OPTCHECK smoke on union xgcc (lreg-coalesce, window-pairing,
  pressure-park, store-sink, -fchecking) + installed-driver smoke
  (replay-shadow-discovery, lreg-coalesce).
- Board: UNCHANGED 84W/35P/15L (CLASS-I everywhere; no booking moves).
- ON set UNCHANGED at 36; KNOB_MODES UNCHANGED at 48.
- Install: sha-verified da957c5793b7... -> 4a85f9e26d62...; driver
  rebuilt (KB .opt), read from the fresh manifest entry.
- PIN-51 DELETION LIST (one-pin scaffolding to remove): JZ
  rvtt-effects.cc assert phase; JW legacy in-module mirrors; JY +
  bnb recompute-asserts; KA flag_checking recompute-asserts.
- Push state: pending at record time; pushed both hops + tt-metal in
  the ceremony commit (verified before pin close).

## Gates

conf_lint + witness_preflight at ON-36 on the installed binary in
~/sfpi-uplift/sweep-2x2/pin50-ceremony/ (results appended below if
re-seats were needed).
