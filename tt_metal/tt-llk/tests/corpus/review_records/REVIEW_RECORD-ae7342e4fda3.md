# REVIEW_RECORD — pin 17 (cc1plus ae7342e4fda3)

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com) — independent of lane EQ (author); gates executed by lane EQ and verified by the orchestrator against laneEQ-evidence-20260821 (EVIDENCE.md, SHA256SUMS 473 files).

Candidate: sfpi-gcc nkapre/sfpi @ 75ace9d643e = pin-16 union 927bce7e526 + the lane EQ W_drain crossing-charge correction (EP-F1).
Installed: cc1plus ae7342e4fda3d039e3c59a0237dae947c84cfb2bf31dbed2b59f35380bbbedd8, driver xg++ 8996f902c104fa792f9adfe5ab6e5d4bd6821acf778a6b41425d8a10fe43d021 (pin-install-fast, manifest appended).

## Reviewed
- rvtt-cost.md W_drain=7 audited entry (fit 6.5; five silicon witnesses cited, both regimes bracketed).
- Covering walk counts the iteration's own issue words; re-anchor rule and mod-write-dominates-rolled-body refusal retained; entry residual moved to config-cost side (domloop-refuse preserved).
- EP TU proof: threshold MATH_ISOLATE .text restored BIT-EXACT to the pin-15 winning form.

## Gates
- dg dst-autoincr family 330/330 (EB 300 + 30 new incl. W_drain-edge boundary twins).
- Full rvtt.exp: FAIL set 16 lines line-identical to the pin-16 frozen reference (5339 PASS).
- Corpus: ON-25 delta vs pin-16 = exactly 51 rows, twice-enumerated set-identical, strict subset of the adjudicated EB-58; all 51 dominates->fire; CRAQ 49/49 ops on the pinned sim (oracle-bit-identical rebuild). OFF/TRUE-DEFAULT: structural proof (Init(0) pass gate) + 5-TU x 2-flagset spot compiles .text-identical (full legs proven at pin-16; owner speedrun order 2026-08-21).
- Witness preservation: absint32/unaryshift/bitwisenot TUs refuse byte-identically; stay-reverted corpus rows verified.
- sigmoidappx-tree re-fires at the W_drain edge (covered 7 >= 7) — the owner mid-run witness closed.
