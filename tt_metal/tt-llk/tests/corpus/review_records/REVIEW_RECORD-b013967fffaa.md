# REVIEW_RECORD — pin 59 (the dead-flag retirement pin)

cc1plus sha256: b013967fffaa8285f452afd7f6c18bda36e9d35bc6d7a94f5a718a2dad912821
driver (g++) sha256: bf439d4a8c914edeaa3ba060ece7840b1e50ddd03a1330c68c8e7ef75923bf35 (REBUILT — lp-schedule alias removed)
source: sfpi-gcc nkapre/sfpi ebeac6bb71b = pin-58 f048362a2d5 + laneMA
dead-flag retirement (fast-forward/merge).  OWNER-RATIFIED.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed
Two audit owner-decision flags retired on the hll/loadmacro precedent
(the option errors on use), each dead-verified by grep before removal:
- -mtt-tensix-optimize-combine: riscv_tt_opt_combine had ZERO functional
  consumers across gcc/; -mno- silently did nothing.  Var + option
  removed; errors on use; invoke.texi entry deleted.
- -mtt-tensix-optimize-lp-schedule: an Undocumented alias, 0 in-tree
  consumers (corpus/test/canon); retired, errors on use.
DEPLOYMENT: combine errors at cc1plus; lp-schedule is DRIVER-resolved,
so this pin installs the rebuilt driver (bf439d4a8c91) — the installed
driver is verified to error both flags, both -m/-mno- forms.

## Gates checked
- Union rvtt.exp (dejagnu-pin59): 7787 PASS = 7783 + 4 (two error-twins
  x 2 PASS lines) EXACT; FAIL-16 LINE-IDENTICAL; ERROR 0.
- Corpus (parallel legs): pinned-58 vs union ON-39 3300/3300
  BYTE-IDENTICAL (unused-flag retirement changes no codegen); chkon
  rc=0, ZERO ICEs, .text == ON.
- Installed-driver smoke: both retired flags error; control compiles.
- make rvtt-unit-tests green.  conf_lint GREEN; witness_preflight (below).
- Board UNCHANGED 88W/30P/16L @ FINAL-BOARD sha 4274cdc3.
- Install: cc1plus 1e8c93459dac -> b013967fffaa; driver f285b235fcee ->
  bf439d4a8c91 (rebuilt, alias removed).
- OPEN: HotCRP submission + FSF/DCO (owner); llk.rst -mno-combine doc
  line refreshed this ceremony.
