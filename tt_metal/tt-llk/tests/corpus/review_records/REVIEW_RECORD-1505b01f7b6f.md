# REVIEW_RECORD — pin 22 (cc1plus 1505b01f7b6f)

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com) — independent of lane FP (author); gates executed by FP, verified against laneFP-evidence-20260822 (DELTA-AUDIT.md, GATE-VERDICTS.txt, SHA256SUMS).

Candidate: sfpi-gcc nkapre/sfpi @ e2034101ef9 = pin-21 + the FP-1 fix.
Installed: cc1plus 1505b01f7b6fde1f2c3d00fc42e9cc1018f7321bb0650324992ce1edd6ffd646.

## Reviewed
- FP-1 (P1 wrong-code, probe-witnessed): the crosslane R5 zip-chain collapse deleted lhs-less CC statements unconditionally while an external tap kept a frame swap alive — the surviving mod-0 swap ran all-lanes. Fix: use-exclusivity guard, refusal crosslane-frame-value-escape, 3 twins closing the uncovered window refusal names. In practice the defect was masked by an accidental pressure refusal — no corpus or measured cell was affected (corpus byte-identity proves it).
- FP-2 (P1, FILED not fixed): the TEN-2932 window checker is positionally blind to replay-DELIVERED writes — the default-ON replay former can convert a 3-error program into a silently-accepted binary. Interim: the sfpu_bridge disasm gate stays mandatory where replay/MOP fire. Owner: FG/EX successor lane.
- FP-3 (P1-boundary, FILED + architectural): replay slot state is device-persistent across invocations — caller tile loops can reassemble the wedge trio across kernel invocations, outside any intra-function walk. Needs an architectural model (BH REPLAY doc gap dependency).

## Gates
- Full rvtt.exp 5578 PASS, FAIL set 16 line-identical to the frozen baseline.
- Corpus tip-vs-fix: TRUE-DEFAULT/OFF/ON-25 all .text byte-identical 3222/3222 (probe-shape-only fix; CRAQ vacuous).
