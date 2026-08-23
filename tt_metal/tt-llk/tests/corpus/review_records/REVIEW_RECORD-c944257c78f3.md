# REVIEW_RECORD — pin 24

cc1plus sha256: c944257c78f3b2c27d954f7e9378776215bfe5bd3827e2023f0741483e9b58d3
driver (g++) sha256: 774e83d7a3d53d2e000730c47080b60a96cc8993bbabd2bbe62aa7fbd110e31e
source: sfpi-gcc nkapre/sfpi 92629b12c64 (union merge of lane FT
agent/window-pairing 595ef1cf89d + lane FU agent/ira-dualbank cc4c82f8604,
off pin-23 tip dfd9121124a; only rvtt-cost.md overlapped — both-append
audit sections, union-resolved keeping both). Built in gcc-build-laneFR
at the union tip (build-pin24-union.log rc=0); .opt changed (FT's new
flag) — union OPTCHECK discharged by the build regenerating options and
both new-generation flags (-mtt-tensix-optimize-window-pairing +
-mtt-tensix-optimize-lreg-alloc) smoke-accepted together on the union
driver. Installed via pin-install-fast.sh with loud --expect-cc1plus
verification; no live sweeps at install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

- Lane FT (window-pairing, closing report 2026-08-22):
  rvtt_macro_interrow_drain_tuned in rtl-rvtt-schedule.cc under new
  Init(0) flag -mtt-tensix-optimize-window-pairing — replaces lane EV's
  register-blind FULL inter-row drain with the minimal spacing the
  exact pending-event model proves (SequenceBits-decoded horizon
  require-exact; per-event realized footprints under SFPLOADMACRO
  override rules; LREG/CC/config intersection + sub-unit occupancy +
  retire-before-issue equality; Dst disjointedness via the audited
  physical-row model plus a new column-parity clause from
  SFPLOAD.md/SFPSTORE.md under the DP-precedent LaneConfig discipline;
  row-stride validity from the absorber-last-word invariant; per-row
  footprint-identity guard). 13 named refusals, every unproven piece
  fail-closed to EV's bytes. Constants audited in rvtt-cost.md
  (WINDOW-PAIRING INTER-ROW DRAIN MODEL, F1-F6, ISA citations).
  Discrimination dg-twinned: mulint32-class 2->0 + renamed-varied;
  signbit frozen 3->2 bound=lreg-overlap (BH+WH); EV's mul24-commuted
  race shape REFUSES; laneconfig-poison 2->1; INT32_ALL 2->1;
  off-identity twin. ON-set promotion deliberately NOT claimed — the
  flag rides as an on-plus knob with its silicon booking.
- Lane FU (IRA dual-bank, closing report 2026-08-22): root cause =
  IRA's per-operand alternative cost scan prices L0..L3 identically
  for companion=value+4 twelve-alternative patterns, coloring
  combinations no single alternative admits; LRA repair reloads then
  spill at peak pressure 8 (spill-diag fatal on compilable kernels).
  Fix = dual-bank pinned-chain binding, layer 3 of rtl-rvtt-lp-alloc.cc,
  riding the existing reviewed default-off lreg-alloc flag: generic
  constraint detection (no op-name keys), DFS with forced-color-
  consistency pruning + union-find matching equalities + DSATUR
  completion certificate, pin-derived webs committed as explicit hard
  LREGs pre-IRA; independent point-wise DF disjointedness + call-crossing
  soundness belt before any mutation; nine named refusals = stand-down
  to today's allocation; caps audited in rvtt-cost.md (MAX_OPS 24 /
  MAX_ALTS 16 / BUDGET 4096 with consumption evidence). Two refuted
  mechanisms banked with measurements.

## Gates

- Union build gate: merge conflict only in rvtt-cost.md (both-append),
  resolved as union; build rc=0; both flags smoke-accepted together.
- Union full rvtt.exp (dejagnu-pin24, SFPI env laneFP-sfpi-env, srcdir
  at union tip so BOTH lanes' twins run): 5655 PASS (= pin-23's 5609 +
  FT's 27 + FU's 19 exactly); FAIL set 16 rows LINE-IDENTICAL to the
  frozen baseline (diff vs dejagnu-pin23/fail-set.txt empty). 46
  window-pairing/dualbank twin PASS lines present.
- Lane FT gates (its build cf5ab965d544, closing report): dg
  window-pairing 27/27, macro-planner 1200/1200, drain 140/140,
  replay 399/399; full rvtt.exp 5636 PASS frozen-16 identical; corpus
  base-vs-fix TRUE-DEFAULT/OFF/ON-25 each 3249/3249 .text-identical
  (Init(0) proven corpus-wide); KNOB leg (ON-25+flag): exactly ONE
  changed TU corpus-wide = mulint32-fresh. SILICON (BH p150, sweep
  headline-laneFT-wp-20260822, device-golden + paired CRAQ green):
  mulint32-fresh KERNEL 63388 -> 56213.3 = -11.32%, vs-hand +72.31% ->
  +52.81% (the pin-19 soundness payment recovered on model-proven
  bytes, within 1.3 cycles of the old racy-byte anchor); blaze
  sdpa_reduce {max,sum}x{t8,t32} honest no-fire (byte-identical under
  the knob — record-hoist class, as FI/FE attributed).
- Lane FU gates (closing report): dg 5 new tests + lreg-alloc family
  72/0; full rvtt.exp 5628 PASS frozen-16 identical; DS arsenal
  --mode future 25/25; corpus (bh, 3213 ELFs/leg, 8 legs) OFF/
  TRUE-DEFAULT/ON-25+crosslane/knob ALL ZERO-delta (not a
  no-engagement artifact: topk TU dump-probed — binding engages,
  124+360 webs, reproduces IRA's bytes exactly); crosslane arsenal
  56/56 + sortnet sim gate 64/64 with the flag ON; KV bitonic stage-10
  binds 136/136 webs, bit-exact on the pinned sim. Silicon: hand top16
  corr PASS 1096/1096/1096; lift body 1091/1091/1091 diagnostic-only
  (FU-F1 kernel ordering bug — winners 4x4-transposed, sim==silicon,
  compiler exonerated by KV-sortnet control; moegatetop16 blocker
  narrowed SKIP_BLOCKED_ON_COMPILER -> SKIP_BLOCKED_ON_KERNEL).
- FU-F2: sfpi::transp8 ADL ambiguity with the vendored blaze bridge
  fixed on tt-metal (qualification); tt-blaze source fix owed.
- tt-metal merges: FU 100744b75e (add/add on test_crosslane_sortnet.py
  resolved to canon's newer FQ-fixed version — FU had vendored a stale
  pre-FQ copy only to run its gate), FT knob registration 0f55b3684b.
- Evidence: ~/sfpi-uplift/laneFT-evidence-20260822/ (SHA256SUMS),
  ~/sfpi-uplift/laneFU-evidence-20260822/ (SHA256SUMS),
  ~/sfpi-uplift/dejagnu-pin24/.
