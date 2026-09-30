# WIP handoff (temporary — delete once the plan below is done or moved)

Picked up 2026-09-29 on bh-lb-120-a07u24: every P150b opens at 12×10 (checked chips 0 / 3 / 7), the Release build is
current, bs1 baseline matches the old box (cold 15.6 / sustained 15.8 ms).

## Plan (agreed order)

From the e2e-vs-roofline analysis on the device-profile artifact (https://claude.ai/artifact/EyeLiogdyYu3soMry6akYn):

1. ~~bs1 fused SwiGLU with K_block 40.~~ Done, negative (NEGATIVE_RESULTS §61): at M=512 the SwiGLU is only 3% exposed
   and the fused kernel is data-movement-bound. bs1 stays unfused.
2. ~~Cheaper SwiGLU in FF1+FF3.~~ Landed (§62): Schraudolph exp + bare SFPARECIP, 71% less SFPU work, STS-B within
   noise; sustained −1.8 / −0.3 / −0.5% at bs8 / 16 / 32 against a no-SFPU ceiling of −3.3 / −3.8% (bs16 / 32).
   The tail exposure was minimal_matmul's output writer (§63, landed: cold bs16 / 32 −1.8 / −2.1%, sustained
   −0.7 / −0.6%, bs8 sustained +0.4%). Left: the partial-sum add (66-69 µs, 4% at bs16; only K_block 80 avoids it and
   that fits only small blocks), and the power cap eating most cold gains (sustained gets ~⅓).
3. SDPA DRAM traffic. Landed at bs8 / 16 (§64): only the K / V reads matter; K / V in L1 (`QWEN_HEADS_KV_L1`), SDPA
   −15 / −13%, cold −1.7 / −0.8%, sustained 0.0 / −0.6%, bit-identical. bs1 already has Q/K/V in L1. **Open at bs32:**
   K / V are 190 KB per core short beside the first QKV chunk; 4 QKV chunks would fit and the matmuls are ~neutral
   since §63 (needs a 4-output `fused_add_rmsnorm_split`). The head-major QKV write (#57722) is untried.
4. SDPA compute (§65). Compute-only / DM-only floors measured (bs32: 519 / 483 µs against 597); the pack thread paced
   the Q·Kᵀ / exp loop. Landed: row sums on the math thread (SDPA −6 / −7 / −6%). **Next:** a cheaper exp (~510 of
   ~1,300 pack cycles per column block; the no-exp probe is −19% of a compute-only unit).

`sustained_run.sh` reports AICLK / power over the sustained window and J/inference, but tt-smi samples swing 30-155 W
within a window (host gaps), so J/inference is too noisy to rank variants yet.
