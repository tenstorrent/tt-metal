# FP32 v2 hypotheses

Baseline: frozen v1 winners (`state_lazy_scan` for D,
`c_refine_hoist_state_lazy_scan` for C). V1 files remain untouched.
The v2 benchmark adds those flags to every case and selects the v1 header
for the baseline; its numeric authority remains canonical `recipe()`.

V1 cycle profiles show 30–31% of time with neither FPU nor SFPU active.
Existing prior fixed-half QK/exp and independent-MATH pack schedules did not
improve throughput. Avoid repeating those without a new ordering advantage.

1. **State block-pack:** numerator unpacks already batch four FP32 tiles,
   but state update still issues independent per-tile packs. A width-four
   (or width-two nonidentity) pack uses the same destination tile order and
   independent L1 additions. It must restore width-one before the sum pack.
   Format/capacity, state math and Blackhole ZEROACC handling stay unchanged.
2. **Whole-chunk exact identity:** scan all eight BF16 maximum tiles once,
   with one mailbox decision. If every first-column word matches, the
   existing per-row exact identity branches can skip duplicate row scans
   and mailboxes. Correction CB reserve/push/pop remains unchanged; as in
   v1 the identity path does not consume the undefined correction payload.
   Any changed maximum falls back to the original row decisions and math.
   Resident inputs favor this path; distinct-KV timing/correctness is needed.
3. **Further state/PV overlap:** investigate in-place L1 old-state addition
   on identity rows only after proving CB ownership. Preserve the existing
   final addition after the complete PV reduction. The split-drain first
   row cannot simply initialize its partial PV accumulation with old state:
   that would change rounding order and is outside scope. Implemented as
   `identitybatch`; exact but rejected on distinct-data tradeoff evidence.

Completed follow-ups: source-A-only unary reconfiguration and coalesced
whole-half ZEROACC flag clearing are exact but neutral. Sparse timestamped
phase markers establish that the QK/overlapped-exp region remains largest.
An internal QK2x2/PV1x4 probe preserves the four-K-tile split-drain grouping;
D needs rowwise L1 subtraction/two-score exp, C needs rowwise1x4 drain to
avoid overflowing the FP32 destination half. Its first C adaptation failed
the bitwise gate and was corrected before the bounded timing screen.
See REPORT.md and raw logs for final disposition.

Initial candidates are compile-time restricted to noncausal, unmasked,
non-ring Q256/K512/D128, with original CB formats/capacities and one input
KV slot. General production dispatch is not changed.

Evidence: fresh v1-winner paired timing, uint16 equality, final trace equality,
distinct KV with max changes, several Q jobs/core, short/odd K loops and
stress. Profiles are diagnostic only. No unmeasured speedup is a result.
