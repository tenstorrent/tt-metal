# HunyuanImage-3.0 — optimizations attempted (lever log for the optimization tool)

> **How to read this (for a Claude Code agent / the optimization tool):** every entry below is one discrete optimization *attempt* — **what was tried**, **what changed** (code / gated flag), and the **measured result + verdict**. Verdicts: **WIN** (kept, default-on) · **WIN-gated** (kept, opt-in flag) · **WASH** (≈no delta, parked) · **DEAD** (correct but big regression / deadlock) · **PARKED** (blocked/incomplete). **Read the verdict before re-attempting a lever.** Model: `tencent/HunyuanImage-3.0` (~80B total / ~13B active MoE, text→image) on a Blackhole 6U Galaxy `MeshShape(8,4)` (32 chips, FABRIC_1D), TP=8 + EP=32 + shard-shared, 1024²/50-step. Metric discipline: ship = **traced wall-clock** (STEADY ms/step steps 2..N, warm decode_forward_s); per-op `device_ms` is a *separate* profiling track (does not track render wall-clock — useful only as a regression filter).

## Headline arcs
- **E2E render s/image (warm):** hybrid 453.7 → 351.4 → on-device cold 216.5 → **~84 (default)** → **~67 (+SP)** → **~58 (+on-device VAE)**
- **Steady per-step (hybrid 32L/12-step/trace):** 7770 → 5512 ms/step (−29.1%)
- **On-device head-glue step:** 5548 → 947 ms/step (removed the ~4600 ms/step host round-trip)
- **Diffusion loop (50-step, warm):** 947 → **623 ms/step (SP, −31%)**
- **VAE decode:** 56.4s fp32 → 36.4s bf16 host → **25.2s on-device mesh** (in-render) / 16.2s isolated
- **Per-op device_ms (separate LOCAL-Mac track):** 137.4 → 95.4

## By date (box wall-clock timeline; metric labeled per row)

| date | commit | optimization — what changed | result + verdict |
|---|---|---|---|
| 2026-07-09 | `85c6a45a24` | GalaxyBH native-TTNN bring-up + e2e (single-device stubs) | PCC 0.9997 @1 layer — **correctness** |
| 2026-07-10 | `de260da73b` | **TP=8** attn/dense shard + sharded e2e pipeline | infra, no isolated delta |
| 2026-07-13 | `cc2935e851` | decode: incremental-KV loop + trace | decode 1.37→8.29 t/s/u (~6×, trace) |
| 2026-07-19 | `2b97c850c7` | **MoE device_ms wins**: 2 merged matmuls + host pre-cast + **bf8/bf4_b experts** + full grid | prefill 40.93→19.09 ms/tok (2.14×); MoE device_ms 36.97→4.88 — **WIN** |
| 2026-07-19 | `a06d4a8a2d` | attention host pre-cast infra | NEUTRAL (below wall-clock floor) |
| 2026-07-19 | `942b6dafc7` | **Build A**: drop router all_gather + skip infer l_aux + fuse SwiGLU silu | prefill 19.09→17.81 ms/tok (−6.7%) — **WIN** |
| 2026-07-21 | `2930f748e8` | dtype fix + traced diffusion loop | hybrid render baseline **453.7 s**; trace ≈no win (compute-bound) |
| 2026-07-22 | `bb2b8592fc` | host-glue 1/3: TTNN final_layer (velocity head) | PCC 0.99986 — correctness |
| 2026-07-22 | `8e4b975bd9` | host-glue 2/3: TTNN patch_embed | PCC 0.99973 — correctness |
| 2026-07-22 | `ae1e571f6b` | **host-glue 3/3: fully-on-device head-glue** | render 406.8→282.2 s; steady 7401→2782 ms/step (−62%) — **WIN** |
| 2026-07-23 | `1c4585ccb6` | sparse top-8 MoE (scatter-index fix) | **DEAD** — ~47× slower (132.3 vs 2.78 s/step), datamove-bound; opt-in only |
| 2026-07-31 | `ee61b0ce3c` | **EP=32** full-mesh expert-parallel (gated) | mechanism decode +70% t/s/u — **WIN-gated→default** |
| 2026-07-31 | `abe7a0ba5a` | **shard-shared** expert + 2-axis all_reduce (gated) | decode +9.1% t/s/u — **WIN-gated→default** |
| 2026-08-02 | `f61879f725` | flip EP=32 + shard-shared **default ON** | steady 7770→6368 ms/step (−18.0%) — **WIN** |
| 2026-08-02 | `20461561a7` | **CCL_LINKS 1→2** | steady 6368→6093 (−4.3%) — **WIN** |
| 2026-08-02 | `37c9f2908a` | **MM_FULLGRID** off→on | steady 6093→5736 (−5.9%) — **WIN** |
| 2026-08-02 | `1338c17fa3` | MM_FIDELITY=lofi | NO-OP (default already lofi-equiv) |
| 2026-08-02 | `c283bf1765` | attention **fused all_reduce** (+ CFG-parallel gated) | steady 5736→5512 (−3.9%); cumulative −29.1% — **WIN** |
| 2026-08-03 | `7254f6fb1c` | **on-device host-glue render** + conv-head caching | host-glue 5548→947 ms/step; E2E 351.4→216.5 cold→**~84 warm** — **WIN** |
| 2026-08-04 | `c5506dece4` | VAE dtype fp32→bf16 (host) | VAE 56.4→36.4 s (~20s of E2E gain) — **WIN** |
| 2026-08-05 | `a4bf416605` | argmax → ROW_MAJOR last-dim (multi-core vocab reduce) | device_ms 137.4→121.5, PCC 0.999 — **WIN (device_ms track)** |
| 2026-08-05 | `a5a0fadaaf` | MoE SwiGLU gate/up + down weights bf16→**bf8_b** | device_ms 121.5→103.5, PCC 0.999 — **WIN (device_ms track)** |
| 2026-08-05 | `b6bb2277b3` | MoE: batch 64 dense experts into 2 bmms (repeat+batched gate/up+SwiGLU+down) | device_ms 103.5→100.0, PCC 0.999 — **WIN (device_ms track)** |
| 2026-08-05 | `3bf0e74b59` | lm_head weight bf16→**bf8_b** (memory-bound 4096×133120 vocab proj) | device_ms 100.0→95.4, logits PCC 0.999 — **WIN (device_ms track)** |
| 2026-08-06 | (gated) | **matmul block-size sweep** (LEVER 1): tuned `minimal_matmul` per flux2 method | **WASH** +2.4%, PCC 0.9999; matmul only ~23% device-kernel → small ceiling |
| 2026-08-06 | `addd63c003` | wire flux2 `minimal_matmul` (`HUNYUAN_MINMM`, OFF) | gated-off scaffold (see wash above) |
| 2026-08-08 | `1be484e7` | **Sequence parallelism (SP-only)**: token-shard the diffusion loop, EP=32→8, KV all-gather in SDPA, pad-S-to-128 | **623.3 vs 905.7 ms/step (−31%)**, PCC 0.99999, ~67 s warm — **WIN-gated `HUNYUAN_SP`** |
| 2026-08-08 | (gated) | SP_FUSED-Linear (H-shard + AG+MM + fused Linear) | correct (PCC 0.99999) but **REGRESS** 850 ms/step — gated-off `HUNYUAN_SP_FUSED` |
| 2026-08-08 | (gated) | SP-RING (ring fabric + fused Ring RS+MM + distributed RMSNorm) | correct but **REGRESS** 996 ms/step — gated-off `HUNYUAN_SP_RING` |
| 2026-08-08 | — | **AG+MM block sweep** | winners collected; no net gain on our layout (activations replicated) |
| 2026-08-08 | — | **1D-decode matmul sweep** | winners collected; **no perf gain** |
| 2026-08-08 | — | **RS+MM block sweep** | **PARKED** — fused Ring op deadlocks certain configs → wedges fabric; resume coarse-only |
| 2026-08-08 | — | **FSDP** (weight-shard + per-layer all_gather) | **DEAD** at scale — PCC-correct (0.9999972) but deadlocks the 32-layer render; gated OFF |
| 2026-08-08 | `148085fb05` | **on-device VAE — single-chip decoder** (`AutoencoderKLConv3D.decode`, factor=1) | PCC ladder ≥0.9993 vs HF — **correctness oracle** (single-chip is perf-dead) |
| 2026-08-08 | `4215de39d3` | **on-device VAE — mesh HW spatial-shard** (H÷8/W÷4, conv halo + gather-GroupNorm-partition) | mesh PCC 0.99954 (256²) / 0.99962 (1024²) — **WIN-gated (correctness)** |
| 2026-08-09 | `7e09747e01` | on-device VAE — full-res 1024² perf test | **16.2 s warm** isolated decode (57 s cold), no OOM |
| 2026-08-09 | `a40fc8c057` | **on-device VAE — WIRED** behind `HUNYUAN_ONDEVICE_VAE` (host = oracle/fallback) | in-render decode **25.2 s** vs 36 s host; real-weight PCC **0.99718**; **e2e ~58 s** (SP+VAE) — **WIN-gated** |
| 2026-08-09 | `25f555f20f` | VAE conv3d block-size sweep (`HUNYUAN_VAE_CINBLK`), C_in_block 32→64 | **WASH** 16.2→16.05 s; decode not input-channel-blocking-bound |

## Lever catalog (grouped by verdict — the tool-ingestible view)

### WIN — default-on (keep, structural)
- **EP=32 + shard-shared expert** (`f61879f725`): 64 experts 2/chip across all 32 + shared expert with a 2nd all_reduce on the 4-axis. −18.0% steady.
- **CCL_LINKS=2** (`20461561a7`): 2 fabric links per collective. −4.3%.
- **MM_FULLGRID** (`37c9f2908a`): matmuls use the full compute grid. −5.9%.
- **fused attention all_reduce** (`c283bf1765`): fuse the attn output all_reduce. −3.9%.
- **on-device head-glue render** (`ae1e571f6b`/`7254f6fb1c`): patch_embed + velocity-head convs on-device, hidden never leaves the device across the loop → removes the ~4600 ms/step host round-trip. 5548→947 ms/step. **Largest single win.**
- **bf8/bf4_b experts + merged MoE matmuls** (`2b97c850c7`) and the device_ms track (`a4bf…`→`3bf0…`): argmax ROW_MAJOR, MoE SwiGLU bf8_b, batched-64-expert bmm, lm_head bf8_b.

### WIN — gated opt-in (keep, flag)
- **`HUNYUAN_SP`** — sequence-parallel diffusion loop, token-shard EP=32→8 (`1be484e7`). 623 vs 906 ms/step (−31%), PCC 0.99999. Default off (byte-identical).
- **`HUNYUAN_ONDEVICE_VAE`** — mesh VAE decode (`148085fb05`→`a40fc8c057`). 25.2 s in-render vs 36 s host, real-weight PCC 0.99718 → e2e ~58 s. Default off (host = oracle). Mechanism: shard H/W (`ShardTensor2dMesh`) → conv3d local shard + cross-chip H/W halo (`vae_neighbor_pad`, **zeros** boundary = HF) + on-chip T pad → GroupNorm gathers full H/W → `GroupNorm3D` → re-partition → local pixel-shuffle upsample → gather-for-SDPA attn. Reuses tt_dit **Mochi** CCL. Gotcha: `num_links=1` at T=1 levels; `ttnn.pad` can't pad rank-5 T (use conv internal T pad).

### WASH — tried, ≈no delta (don't rechase)
- **matmul block-size sweep** (LEVER 1, `minimal_matmul`/`HUNYUAN_MINMM`): +2.4%. Matmul is only ~23% of device-kernel → small ceiling. The parallelism restructure (SP) was the real win, not block tuning.
- **AG+MM sweep** / **1D-decode matmul sweep**: winners collected but no net gain — they gather/shard activations, but our TP+EP keeps activations replicated (only "other matmuls" fit → those are the sweep above).
- **VAE conv3d C_in_block 32→64** (`HUNYUAN_VAE_CINBLK`): 16.2→16.05 s. Decode is not input-channel-blocking-bound. (Structural lever = reduce-moments GroupNorm — not yet done.)
- **MM_FIDELITY=lofi**: no-op (default already lofi-equiv).

### DEAD — correct but big regression / deadlock (do not ship)
- **Sparse top-8 MoE** (`1c4585ccb6`): ~47× slower than dense (132.3 vs 2.78 s/step) — gather/scatter datamove-bound. Correct (PCC 1.0 after uint16 scatter-index fix). Opt-in only.
- **FSDP** (weight-shard + per-layer all_gather): PCC-correct (0.9999972) but **deadlocks the 32-layer render** (survives SIGKILL = device-stuck). Gated OFF.
- **SP_FUSED / SP-RING** elaborations: PCC-correct but regress (850 / 996 ms/step) — on this Linear/ring fabric the added collective traffic + Ring latency outweigh the overlap. Simple token-sharding wins.

### PARKED — blocked / incomplete (resume with care)
- **RS+MM block sweep**: the fused Ring op deadlocks certain configs and a hang **wedges the fabric** (one `tt-smi -glx_reset` per hang). Full sweep impractical → resume as a **coarse safe-config sweep**. Scripts: `~/nkira/run_rsmm_chunked.sh`, `extract_rsmm_winners.sh`.
- **Ring-SDPA-chunk sweep**: not started.
- **VAE reduce-moments GroupNorm**: the top open VAE lever (replace the full-spatial gather with an all_reduce of per-group moments → cut CCL + the 25.2-vs-16.2 memory-pressure gap). Not started.
- **VAE build-to-setup**: move the one-time 19.4 s decoder build out of the per-image path.

## Sourcing caveats
1. **device_ms 137→95.4 is a separate LOCAL-Mac profiling track** (per-op tracy), not box wall-clock; box-branch equivalent is `2b97c850c7`. The 08-05 argmax/MoE-bf8/lm_head rows are that track.
2. **VAE-bf16 host** had no model-dir commit (scaffold 07-22 + doc 08-04); the on-device VAE (08-08→09) supersedes it as the perf path.
3. **Two step-ms harnesses**: the 7770→5512 ladder is 32L/12-step/trace; the 5548→947 host-glue win is 50-step/eager — do NOT chain them.
4. **Decode t/s/u figures are the gen_text pedigree, not T2I render deltas** (decode is a separate path).
5. **E2E ~58/67/84 s are STEADY warm** (steps 2..N + build-amortized VAE); the harness all-step mean (e.g. 3106 ms/step) is compile-diluted — ignore it. Single cold image with VAE build = ~200 s + ~24 min one-time model tilize.
