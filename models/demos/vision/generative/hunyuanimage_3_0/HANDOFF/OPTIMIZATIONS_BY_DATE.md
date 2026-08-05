# HunyuanImage-3.0 — optimizations done (summary + by date)

Native-TTNN bring-up of `tencent/HunyuanImage-3.0` (80B MoE / ~13B active) on BH Galaxy, from first render to **~84 s/image warm** (1024²/50-step). All numbers source-tagged; sourcing caveats at the bottom.

## Headline arcs
- **E2E render s/image:** hybrid 453.7 → 351.4 → on-device cold 216.5 → **warm ~84**
- **Steady per-step (hybrid 32L/12-step/trace):** 7770 → **5512 ms/step (−29.1%)**
- **On-device head-glue step:** 5548 → **947 ms/step** (removed ~4600 ms/step host round-trip)
- **Per-op device_ms (LOCAL-Mac track):** 137.4 → 95.4

## By date

| date | commit | optimization | perf impact (metric labeled) |
|---|---|---|---|
| 2026-07-09 | `85c6a45a24` | GalaxyBH bring-up + e2e (single-device stubs) | correctness — PCC 0.9997 @1 layer |
| 2026-07-10 | `de260da73b` | **TP=8 shard** + sharded e2e pipeline | correctness/infra — no isolated delta |
| 2026-07-13 | `cc2935e851` | decode: incremental-KV loop + trace | decode t/s/u 1.37 → 8.29 (~6×, trace) |
| 2026-07-19 | `2b97c850c7` | **MoE device_ms wins** (2 merged matmuls, host pre-cast, bf8/bf4_b experts, full grid) | prefill 40.93 → 19.09 ms/token (2.14×); MoE device_ms 36.97 → 4.88 (~7.6×) |
| 2026-07-19 | `a06d4a8a2d` | attention host pre-cast infra | NEUTRAL (below wall-clock floor) |
| 2026-07-19 | `942b6dafc7` | **Build A** (drop router all_gather + skip inference l_aux + fuse SwiGLU silu) | prefill 19.09 → 17.81 ms/token (−6.7%) |
| 2026-07-21 | `2930f748e8` | dtype fix + traced diffusion loop | **hybrid RENDER baseline 453.7 s/image**; trace ~no win (compute-bound) |
| 2026-07-22 | `bb2b8592fc` | host-glue stage 1/3: TTNN final_layer (velocity head) | correctness — PCC 0.99986 |
| 2026-07-22 | `8e4b975bd9` | host-glue stage 2/3: TTNN patch_embed | correctness — PCC 0.99973 |
| 2026-07-22 | `ae1e571f6b` | **host-glue stage 3/3: fully-on-device head-glue** | RENDER 406.8 → 282.2 s warm; steady 7401 → 2782 ms/step (−62%); PCC 0.99989 |
| 2026-07-23 | `1c4585ccb6` | sparse-MoE scatter fix + finding | **DEAD lever** — sparse ~47× slower (132.3 vs 2.78 s/step); opt-in only |
| 2026-07-31 | `ee61b0ce3c` | **EP=32** full-mesh expert-parallel (gated) | infra at commit; mechanism's decode win +70% t/s/u |
| 2026-07-31 | `abe7a0ba5a` | **shard-shared** expert + 2-axis all_reduce (gated) | infra at commit; decode +9.1% t/s/u |
| 2026-07-31 | `09f00b6f00` | single-chip fabric-free run | correctness/infra |
| 2026-08-02 | `f61879f725` | **flip EP=32 + shard-shared default ON** | steady 7770 → 6368 ms/step (−18.0%); E2E 157.8 → 143.1 |
| 2026-08-02 | `20461561a7` | **CCL_LINKS 1 → 2** | steady 6368 → 6093 (−4.3%); E2E 143.1 → 135.8 |
| 2026-08-02 | `37c9f2908a` | **MM_FULLGRID OFF → ON** | steady 6093 → 5736 (−5.9%); E2E 135.8 → 131.6 |
| 2026-08-02 | `1338c17fa3` | MM_FIDELITY=lofi | NO-OP (default already lofi-equiv) |
| 2026-08-02 | `c283bf1765` | attention **fused all_reduce** (+ CFG-parallel gated) | steady 5736 → 5512 (−3.9%); **cumulative 7770→5512 = −29.1%** |
| 2026-08-03 | `7254f6fb1c` | **on-device host-glue render + conv-head caching** | host-glue 5548 → 947 ms/step; **E2E 351.4 → 216.5 cold → ~84 warm** |
| 2026-08-04 | `c5506dece4` | doc: VAE dtype clarify | VAE fp32 56.4s → bf16 36.4s (~20s of the E2E gain) |

## Phases
1. **Bring-up & correctness** (07-09→07-13): native GalaxyBH bring-up (PCC 0.9997), TP=8, incremental-KV decode.
2. **MoE device_ms wins** (07-19): merged matmuls + bf4_b experts + full grid + Build A → prefill 40.93→17.81 ms/token.
3. **Hybrid T2I + host-glue on-device** (07-20→07-23): render 453.7 baseline → 3-stage host-glue → 282.2 s warm; sparse-MoE landed but DEAD (opt-in).
4. **EP=32 + shard-shared** (07-31 gated → 08-02 default): full-mesh expert-parallel → steady −18.0%.
5. **Per-step CCL + matmul levers** (08-02): CCL_LINKS=2, MM_FULLGRID, fused all_reduce → cumulative −29.1%.
6. **On-device host-glue render + warm caching** (08-03): kill the per-step host round-trip → 5548→947 ms/step, ~84 s warm.
7. **VAE dtype** (fp32→bf16): ~20s of the E2E gain.

## Sourcing caveats
1. **device_ms 137→95.4 is a separate LOCAL Mac checkout track** — not on the box branch; box-branch equivalent is `2b97c850c7` (MoE device_ms 36.97→4.88). Sub-steps undated.
2. **VAE-bf16 landing has no model-dir commit** on this branch (only its scaffold 07-22 + doc 08-04).
3. **1-day offset**: commits `f61879…/20461…/37c9f…` are git-dated 08-02; their UNIFIED_STACK writeups say 08-01.
4. **Two different step-ms harnesses**: the 7770→5512 ladder is 32L/12-step/trace; the 5548→947 host-glue win is 50-step/eager — do NOT chain them (5512≈5548 is coincidental).
5. **TP=8 (07-10) has no isolated perf delta** in any source — infra only.
6. **Decode t/s/u figures are pedigree, not T2I render deltas** (decode is a separate gen_text path).
