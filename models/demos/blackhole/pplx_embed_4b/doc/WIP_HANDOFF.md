# WIP handoff (temporary — delete once the plan below is done or moved)

Picked up 2026-09-29 on bh-lb-120-a07u24: every P150b opens at 12×10 (checked chips 0 / 3 / 7), the Release build is
current, bs1 baseline matches the old box (cold 15.6 / sustained 15.8 ms).

## Plan (agreed order)

From the e2e-vs-roofline analysis on the device-profile artifact (https://claude.ai/artifact/EyeLiogdyYu3soMry6akYn):

1. ~~bs1 fused SwiGLU with K_block 40.~~ Done, negative (NEGATIVE_RESULTS §61): at M=512 the SwiGLU is only 3% exposed
   and the fused kernel is data-movement-bound. bs1 stays unfused.
2. ~~Cheaper SwiGLU in FF1+FF3.~~ Landed (§62): Schraudolph exp + bare SFPARECIP, 71% less SFPU work, STS-B within
   noise; sustained −1.8 / −0.3 / −0.5% at bs8 / 16 / 32 against a no-SFPU ceiling of −3.3 / −3.8% (bs16 / 32).
   **Open:** the exposed time did not shrink with the pass (bs16 58 → ~45 µs; not the per-call SFPU stall). About half
   of it (44-65%) is each output block's last subblock (`SW_SKIP=last` vs `first`, §62); next, profiler zones on the
   math / pack threads at the block boundary to see what the math thread waits on. The partial-sum add (70 / 153 µs at
   bs16 / 32) is a separate cost of the same size.
3. **(next) SDPA DRAM traffic** (41–60% of its DRAM roof; Q/K/V round-trip through DRAM from the heads op). Keep the
   heads output in L1 for SDPA, or the head-major QKV write (#57722).

`sustained_run.sh` reports AICLK / power over the sustained window and J/inference, but tt-smi samples swing 30-155 W
within a window (host gaps), so J/inference is too noisy to rank variants yet.
