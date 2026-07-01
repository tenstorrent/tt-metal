# Resolves #48285

This PR addresses the -NaN edge case in Wormhole (and potential +inf edge case in Blackhole) and improves the execution time (perf regression) in Wormhole.

### 1. Fix -NaN Edge Case (WH) and Upper Clamp Bug (BH)
In both the Wormhole and Blackhole `_sfpu_exp_21f_bf16_tti_` implementations, the dual-output `TTI_SFPSWAP` instruction with mode `VEC_MIN_MAX` was writing its clamped bounds into the wrong target registers. 
- In **Wormhole**, the upper and lower clamp logic had its target registers swapped (e.g., `LREG0` vs `LREG1`). This caused the un-clamped values to be passed forward instead of the clamped values, leading to a `-NaN` edge case for out-of-bound inputs when polynomial refinement failed.
- In **Blackhole**, a similar issue was present for the upper clamp `SFPSWAP` (register `LREG3` vs `LREG1`), which was corrected.

### 2. Fix Perf Regression (WH)
In Wormhole, the `SFPLOADI` instruction used to refresh `255.0f` for the upper clamp was originally located at the end of the loop, which meant `SFPNOP` was required between the two `SFPSWAP` instructions. 
By moving the `SFPLOADI` instruction to fill the delay slot after the lower clamp `SFPSWAP`, we successfully removed the `TTI_SFPNOP` instruction, perfectly hiding the latency and reducing `BODY_LEN` by 1. This reclaims the 1 instruction footprint that was previously added by the round-to-nearest `STOCH_RND` correctness fix, restoring performance in the TTI kernel while retaining `bf16` accuracy.

### Payment Info
PayPal: https://paypal.me/sureshc26
