# WIP handoff (temporary — delete once the plan below is done or moved)

Picked up 2026-09-29 on bh-lb-120-a07u24: every P150b opens at 12×10 (checked chips 0 / 3 / 7), the Release build is
current, bs1 baseline matches the old box (cold 15.6 / sustained 15.8 ms).

## Plan (agreed order)

From the e2e-vs-roofline analysis on the device-profile artifact (https://claude.ai/artifact/EyeLiogdyYu3soMry6akYn):

1. ~~bs1 fused SwiGLU with K_block 40.~~ Done, negative (NEGATIVE_RESULTS §61): at M=512 the SwiGLU is only 3% exposed
   and the fused kernel is data-movement-bound; best 214.6 µs standalone vs ~192 for FF1 + FF3 + mul, and 16.1 / 16.2
   vs 15.6 / 15.8 ms e2e. bs1 stays unfused.
2. **(next) Cheaper SwiGLU in FF1+FF3** (40% of the batched replay, biggest gap at every batch: 9.3 / 16.6 / 28.4 ms
   to roofline at bs8 / 16 / 32). The SFPU pass is ~1400 cycles/output tile (§60) and costs power even when hidden;
   under the power cap sustained only gets ~half of a cold gain (c61c7e3: cold −4.8/−5.3/−6.1%, sustained
   −2.5/−2.8/−2.8%). Try a cheaper SiLU; check STS-B. Rank by sustained time and J/inference.
3. **(later) SDPA DRAM traffic** (41–60% of its DRAM roof; Q/K/V round-trip through DRAM from the heads op). Keep the
   heads output in L1 for SDPA, or the head-major QKV write (#57722).

`sustained_run.sh` now reports AICLK / power over the sustained window (last half of iterations) and J/inference
(sustained power × sustained time / bs); bs1 runs are too short to catch a tt-smi sample there (n/a).
