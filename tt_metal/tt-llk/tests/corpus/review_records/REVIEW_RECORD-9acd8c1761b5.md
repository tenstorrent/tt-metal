# REVIEW_RECORD — pin 23

cc1plus sha256: 9acd8c1761b5712287c6d45cf49fc5804368b7e41d21351a0ee8fed52bdb7671
driver (g++) sha256: 836ddcd6e9801950b9737a23879adced06c18958331fe63cff775df3038558cf
source: sfpi-gcc nkapre/sfpi dfd9121124a (union merge of lane FR
agent/window-replay-vision 01e436bffae + lane FS agent/replay-state-model
dd3bf7ab829, off pin-22 tip e2034101ef9; zero overlapping files between
the two lanes' diffs; no .opt changes — no new flags, ON set unchanged
at 25). Built in gcc-build-laneFR at the union tip (build-pin23-union.log,
27s ccache-hot); installed via pin-install-fast.sh with loud
--expect-cc1plus verification (PIN-INSTALL-MANIFEST.txt appended).

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

- Lane FR (FP-2 fix, closing report 2026-08-22): rtl-rvtt-crosslane-window.cc
  gains exact Replay Expander vision per tt-isa-documentation REPLAY.md —
  no-exec captures (Load=1/Exec=0) exempt from diagnosis AND marker state
  transfer at the record site (kills two witnessed false classes);
  playback launches expand the resolved record payload at the launch
  position (delivered non-exempt LReg4-7 writes under proven-OPEN =
  hard dest-index-window-violation; delivered SFPCONFIG markers re-scope
  window state at the launch); fail-closed resolution (constant sane
  span, Count 0 = 64-wrap refused, exactly one covering record, hardened
  payload shadow walk, record must dominate the launch per FP-3/FS,
  raw REPLAY-opcode census poisons all); unresolved launch in
  proven-OPEN window = crosslane-window-replay-unproven; TTMOP in
  proven-OPEN window = crosslane-window-mop-unproven (state degrades
  UNKNOWN). Diagnostics-only: zero codegen edits. Fixpoint and
  diagnostic walk share one replay-aware transfer.
- Lane FS (FP-3 model + obligation, closing report 2026-08-22):
  silicon-proven persistence model audited into rvtt-cost.md
  (REPLAY-STATE PERSISTENCE MODEL): replay expander slots persist
  across kernel invocations (EXP-1 cross-ELF delivery after TRISC
  soft-reset + reload); sibling-function record/launch reassembles
  within a launch (EXP-2, the pfj1 admit shape); post-full-board-reset
  zero-word replay WEDGES (BASE negative control) — only eraser is a
  full board reset. Obligations: rtl-rvtt-dst-autoincr.cc refuses
  same-function no-exec captures of replay-delivered groups that are
  forward-reachable OR do not dominate the group (record-hoist
  preheader class preserved); rtl-rvtt-replay.cc unhoist rule 3
  (noexec-record-dststore-nondominating-launch-persist-unaudited) for
  pass-formed records; user-authored records remain the user's contract.
- Retirement statement adopted: with replay vision landed, the
  sfpu_bridge.hpp disassembly gate's MANDATORY status (FP-2 interim)
  is retired for replay-/MOP-delivered shapes; it remains an optional
  belt for hand-authored raw-TTI kernels compiled without
  -mtt-tensix-optimize-crosslane.

## Gates

- Union build gate (EN discipline): merge clean, zero overlapping
  files; no .opt records touched (both lanes flagless) — no
  riscv.opt parser hazard; union compiled from scratch objects via
  ccache (build-pin23-union.log rc=0).
- Union full rvtt.exp (dejagnu-pin23, SFPI env laneFP-sfpi-env,
  srcdir at union tip so BOTH lanes' twins run): 5609 PASS; FAIL set
  16 rows LINE-IDENTICAL to the frozen baseline (diff vs
  dejagnu-laneFR/fail-set.txt empty). Twin spot-check: FS persistence
  family + FR replay-vision/window family PASS lines present (6 + 26).
- Lane FR gates (its build 892b4d611c10237d, closing report): dg
  family 33/33; full rvtt.exp 5600 PASS frozen-16 line-identical;
  corpus farm tt-metal-laneFR @ 157cfa5a10, REAL-copied include
  hybrids: FIVE flag legs (TRUE-DEFAULT, OFF, ON-25,
  TRUE-DEFAULT+crosslane, ON-25+crosslane) x base/fix ALL per-TU
  .text BYTE-IDENTICAL 3249/3249; FP's pw1 blindness pair flips to
  2 hard errors; FK kv-window witnessed-good class silent +
  byte-identical.
- Lane FS gates (its build ab987118fd44c45b, closing report): dg
  376/0 dst-autoincr + 488/0 replay/record-hoist; full rvtt.exp 5587
  PASS frozen-16 line-identical; corpus ON-25 base-vs-fix ZERO DELTA
  3249/3249 .text-identical (CRAQ vacuous, ES/FJ witnesses
  byte-preserved); OFF structurally identical (changes behind pass
  gates); 4 twins (sibling-arm FIRE x3 + dominating-deliverer KEEP);
  4 silicon experiments + health checks on BH p150 (dual flocks,
  flush protocol; the one hang = BASE expected control,
  flush-verified).
- Install: pin-install-fast.sh sha-verified cc1plus
  1505b01f7b6f... -> 9acd8c1761b5...; no live sweeps at install
  (pgrep sweep_2x2 clear of real runs; flocks clear); sfpi repo
  untouched by both lanes so the pin-22 include/ staging remains
  current (sfpi_crosslane.h/sfpi_sortnet.h present).
- Evidence: ~/sfpi-uplift/laneFR-evidence-20260822/ (SHA256SUMS 47
  files), ~/sfpi-uplift/laneFS-evidence-20260822/ (SHA256SUMS 35
  files), ~/sfpi-uplift/dejagnu-pin23/.
