# REVIEW_RECORD — pin 20 (cc1plus 31611bc75cb2)

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com) — independent of lane FJ (author); gates executed by lane FJ, verified by the orchestrator against laneFJ-evidence-20260822 (FJ-F1.md, GATES.md, WITNESS-HASHES, SHA256SUMS 45 files).

Candidate: sfpi-gcc nkapre/sfpi @ 06b39306f00 = pin-19 + the lane FJ hang-guard trip-32 escape fix.
Installed: cc1plus 31611bc75cb201f153c33e03c30254513f0a7b966ae54f2b9927ff62ee36cd14 (pin-install-fast, manifest appended).

## Reviewed
- FJ-F1 (three defects): (1) the W_drain window unit counted frontend words only — replay-launch-delivered payload words escape the bound (a launch issues 1 frontend word, the expander delivers 11); (2) HANG-3 new device datum: the composition wedges WITHOUT the mod-write — the lethal trio is no-exec-re-record x outstanding-launches x Dst-store payload; (3) a position-insensitive asm refusal blocked the witnessed-good exec-while-record conversion.
- Rule extensions, no kernel special-casing: replay-delivered-row groups refuse reachable no-exec captures at any distance (W_drain retained for explicit-row groups); position-aware asm refusal; new fail-closed unhoist sweep (noexec-rerecord-dststore-composition-unaudited) with the storeless/loop-free/user-authored classes untouched; delivery-boundary audited fact in rvtt-cost.md.

## Gates
- dg focused families 819/819; full rvtt.exp 5440 PASS, frozen FAIL set 16 lines LINE-IDENTICAL.
- ES witnesses byte-identical (celu/xielu/lcm/lcm-knob hashes pinned).
- Corpus: OFF 3222/3222 identical; ON delta = exactly 21 TUs (19 noexec->exec conversion flips to witnessed delivery + 2 un-hoists), CRAQ green on every BH-runnable changed TU.
- Device (flush protocol): sparse_k_filter ON-25 t32 — the FE-F1 deterministic hang — PASSES 2/2 on the fixed binary; t8 PASS; verify legs green.
- Ceremony note: blaze-sparsekfilter-t32 row UNSKIPPED (semantic kind restored, t32 nodes rebuilt from the t8 sibling); moegatetop16 stays skip-blocked (IRA gap, unrelated).
- Next-sweep bytes-change watch: exp, relu_max, skf-t8 + the 19 conversion-flip TUs.
