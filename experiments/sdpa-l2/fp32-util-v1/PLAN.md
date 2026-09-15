# 80% useful FLOP-utilization effort

Start from accepted accuracy-investigation mode 4: HiFi4 QK/PV and denominator,
FP32 score subtraction/state, unbiased approximate exp, original BF16 Q/K/V.
Keep B1 H10 D128 noncausal, Q/K chunks128/1024 and existing buffering fixed.
User's estimated 34% current utilization implies an 80% target near131 TFLOP/s,
2.69s for full256K. This denominator must be calibrated against architecture,
available cores, and clock; activity counters are not useful FLOP utilization.

Initial work:
1. Rebuild/verify mode4 baseline and freeze full-output fingerprints.
2. Two-score/shared-max DST batching, preserving arithmetic and requiring exact
   baseline output equality before timing. No denominator precision changes yet.
3. Profile phase budgets and isolated matching-format matmuls/SFPU processing.
4. Fuse subtraction/grid/refinement and explicitly pipeline independent subblocks
   if the first batching control is correct. Each change is independently checked.

Use full-operation timing with40warmups/10replays for retained candidates;
shorter32K/128K screens are not substituted for sustained256K measurements.
Full-output finiteness/trace equality and FP64 spread references accompany tests.
Keep raw failures. Changes are experimental; save patch/source provenance and
preserve the earlier retained implementation. Do not claim80% until measured.
