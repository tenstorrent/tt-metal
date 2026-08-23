# REVIEW_RECORD — pin 26

cc1plus sha256: 6781b2063277d87b0898d79bc42bcfe2202bb0bca2244819eb1e38195fc0fe79
driver (g++) sha256: 774e83d7a3d53d2e000730c47080b60a96cc8993bbabd2bbe62aa7fbd110e31e (unchanged)
source: sfpi-gcc nkapre/sfpi 97f38861a94 (merge of lane FW
agent/record-hoist-loop 9b61bf16a95, single commit off pin-24
92629b12c64, onto the pin-25 tip 452167b53e7; only rvtt-cost.md
overlapped with FV and git auto-merged the disjoint regions cleanly).
Companion: tt-metal nkapre/sfpi 0775c3e1fc (row-note bookings).
Built in gcc-build-laneFR at the merged tip (build-pin26.log rc=0);
installed via pin-install-fast.sh with loud --expect-cc1plus
verification; no live sweeps at install; no sfpi header changes.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

- Lane FW (record-hoist tile-loop generalization, closing report
  2026-08-23). Autopsy-first attribution (dump-proven before design):
  the blaze sdpa_reduce_row lift's runtime-counted tile loop gets two
  invariant windows exec-recorded IN-BODY (18 issued words/tile,
  re-delivered every tile) while the hand original records once at
  init — the marginal-words/tile gap that widens with tile count.
  Blocking refusals named in order: record-hoist-loop-opaque (blanket
  call/asm/typed-owner scan refuses every real LLK tile loop),
  record-hoist-trip-count-unproven (runtime bound),
  replay-reissue-latency-unproved, record-hoist-loop-shape (multi-bb
  profiler vehicles). Compiler layer; no kernel restructure.
- Mechanism (all riding the reviewed default-off
  -mtt-tensix-optimize-replay-record-hoist knob): (a) loop
  replay-preservation audit rvtt_macro_epoch_loop_replay_preserved_p
  reusing the WP11 interval resolver — opcode-interval REPLAY
  exclusion for raw words and volatile-stored FIFO words,
  address-benign store classes, pass-own launches admitted
  (multi-record calendars), user records/launches refused per the FS
  persistence model; (b) MopCfg template census per MOPExpander.md
  (9/9 dominating coverage, write-only-aperture discharge); (c)
  runtime-trip admission: structural trips>=1 (dedicated preheader) +
  2-trip break-even vs existing MIN_BENEFIT, named refusal
  record-hoist-runtime-trips-break-even, 493 cs single-trip exposure
  documented; (d) multi-bb loops via capture-bb-dominates-latch; (e)
  reissue gate discharged BH/WH only (QSR + default keep it); (f)
  doomed-hoist mirror — a Dst-store payload hoist the FJ sweep would
  un-hoist now refuses by the sweep's own name instead of forming a
  pessimization; (g) FW-F1 finding FIXED FAIL-CLOSED in all flag
  states: loop_preserves_replay_p was blind to plain volatile stores
  that could FIFO-push a REPLAY record word (ON-25 corpus zero-delta
  proven). All derivations audited in rvtt-cost.md; no new constants,
  no op-name keys.

## Gates

- Union full rvtt.exp (dejagnu-pin26, srcdir at merged tip): 5795
  PASS (= pin-25's 5734 + FW's 61 exactly); FAIL set 16 rows
  LINE-IDENTICAL to frozen (diff vs dejagnu-pin25/fail-set.txt empty).
- Lane FW gates (its build 3a332901a6d8, closing report): dg
  record-hoist 119/119 (13 new/updated twins incl. renamed-varied +
  WH); full rvtt.exp 5716 PASS FAIL-16 line-identical to pin-24
  frozen; corpus base(pin-24)-vs-fix TRUE-DEFAULT/OFF/ON-25 each
  3249/3249 byte-identical; knob leg exactly 38 changed math TUs
  (identity + replay-shape census archived; doomed shapes reverted);
  CRAQ 67 pass / 18 BH-profile skips / 1 pre-existing xfail / ZERO
  failures on the pinned sim 32489dda.
- SILICON (headline-laneFW-rh-20260823b, corr device-golden + paired
  CRAQ PASS, no hangs), record-hoist on-plus KERNEL: blaze
  sdpa_reduce max-t8 -0.39%, sum-t8 -1.21%, max-t32 -0.52%,
  sum-t32 -0.84% — the only loss class that widened with tile count
  now narrows at every point (vs-hand max +1.61 -> ~+1.08 at t32;
  sum +2.81 -> ~+1.95). FT composition proven twice: mulint32-fresh
  ON25+wp and ON25+wp+record-hoist both .text a49e94f21001d1b8 (FT's
  booked hash); wp leg re-measured 63388 -> 56214.7 (-11.32%) on
  device. The sweep's single RED = the known mulint32 stale-anchor
  line (v2 baseline anchored at FT's knob booking, not a lane-FW
  regression); sum-row reassoc lreg-pressure notes pre-existing.
- Install: sha-verified 1540d13a4054... -> 6781b2063277...; driver
  unchanged; no include/ changes.
- Open (owner steps): record-hoist knob ON-set promotion (silicon A/B
  now exists), FI walk-row composition, mulint32 v2-anchor re-book.
- Evidence: ~/sfpi-uplift/laneFW-evidence-20260822/ (SHA256SUMS, 533
  files), ~/sfpi-uplift/dejagnu-pin26/.
