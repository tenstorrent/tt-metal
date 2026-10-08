# t-diffvae: LTX-2.5 NA DiffVAE decode under 1 s at 1080p 145f on 4x8 — plan (#210, re-planned #257)

Date 2026-10-07. Base: origin/ttp/t48-ltx25-integrated @ 5e4e0cd643a. CPU-only planning task: no device jobs ran.

Goal (user #143): the real DiffVAE (`LTX25_DIFFVAE=1`, `diffvae_ltx*.py`) decodes 1088x1920, 145 frames
in < 1 s on a 4x8 BH galaxy. Today: 11.77 s in the pipeline (t20, blx03 job 874; first gen 17.8 s).
Gate timer: the pipeline's `VAE decode (forward)` (`pipeline_ltx_distilled.py:2321`). It wraps
`decode_latents`, so it **includes the pull of the pixels to host**.

## 0. Roofline re-plan (#257, 2026-10-08) — read this first

Base: t48 @ 946a37952bd (+ be5d1bc045a hash fix). CPU/notes only, no device jobs. Calculator:
`tt-project/t-diffvae/roofline_calc.py` (assumptions at its top; rerun after changing them).

**Where we are.** 16.5 s (#214 unoptimized) -> 3.378 s (#246). The last six levers (S5_2D, gather rephase,
packed lanes, approx exp, ...) each bought 4-12%. Rejected, do not redo: query-chunk sizes, tracing and
the overlapped uint8 pull (#241: device-bound), NA LoFi (#246), GNA stride (#216/#224), MLP LoFi (t11).

**Measured split** (#253, blx01 job 912, deep tree 3820 ms, sync-inflated; real 3.379 s, so x0.885):
| part | tree ms | notes |
|---|---|---|
| det stage 1 (dim 2048, 4 blocks, replicated on all 32 chips) | 607 | linear-order attention 288, qkv-to-volume 93, MLP 102, qkv-proj 40 |
| det stage 2 (dim 1024, 6 blocks, W/8) | 235 | NA 41, halo+brick 51, head-unflatten 18, head-allgather 11 |
| det stage 3 (dim 512, 4 blocks, W/8) | 147 | |
| det stage 4 (dim 512, 2 blocks, W/8) + upsample 4 | 374 | head-unflatten 45, head-allgather 23, MLP 66, upsample 57 |
| upload, ghost crop | 42 | |
| stage 5 pre (context reshard, randn+embed, brick) | 64 | |
| stage 5, 8 blocks x 277 | 2219 | **NA op 152.7**/block; halo+brick 42.8 (CCL 2x10.4, rebrick 2x3.2, 15.6 other); qkv-lanes 27.7; MLP 25.7; norm+mod 5.3; qkv-proj 5.7; context 4.9; out-proj 2.8; residual 2.8 |
| stage 5 tail (pixel pull) | 99 | |

### 0.1 Lower bounds per chip (BH: 130 cores x 1.35 GHz, HiFi2 ~360 TF/s; DRAM 512 GB/s spec)
- **NA op.** Each chip reads **34.2 GB** of K/V per block for **0.81 GB** of unique K/V (halo included):
  every brick is re-read **42x**, once per covering query chunk (the reader fetches each chunk's 112 key
  bricks from DRAM; `neighborhood_sdpa_program_factory.cpp` gives each core a contiguous run of work items
  but keeps nothing between them). 34.2 GB in 152.7 ms = 224 GB/s. That is not "Wormhole bandwidth" (#253
  note is wrong); it is 44% of BH spec and near the **~270-290 GB/s chip-wide ceiling that tt-metal SDPA's
  own K/V streaming hits on BH** (issue #56691, M. Vlahovic). So the op is DRAM-stream bound and read
  tuning will not move it; only fewer reads will.
  Compute: 2.19e12 FLOP performed (2.7x the exact 11^3 window, brick granularity; t213 found tile-level
  narrowing removes only 1.03-1.19x of it) -> 6 ms of HiFi2 matmul, **~15 ms** with softmax at d64.
  | NA K/V reuse scheme | bricks fetched / chunk | DRAM ms | NA ms/block (floor) |
  |---|---|---|---|
  | today | 112 | 122 at 280 GB/s (152.7 measured) | 152.7 |
  | BF8 K/V only (#253 lever, A/B pending) | 112 | 61 | ~61-80 |
  | 1-D ring: core walks a W row (15 chunks), keeps the 7x4x4-brick union in L1, adds one 7x4 column per step | 33.6 | 37 | ~37-45 |
  | 1-D ring + BF8 | 33.6 | 18 | ~18-25 |
  | 2-D group (4x15 chunks per core group, K/V multicast) | 14.7 | 16 | ~16-25 (compute) |
- **Stage-5 rest.** Linears 1.24e12 FLOP = 3.5 ms HiFi2. One 256-ch tensor is 303 MB/chip, ~2 ms per
  read+write pass at 280 GB/s. Unfused SwiGLU round-trips a 4.8 GB intermediate (~17 ms of the 25.7).
  Halo is ~0.2 GB/chip/block. Floor ~10-15 ms/block; **124 ms today**.
- **Det stages.** Full FLOPs at HiFi2 peak: stage 1 64 ms *replicated* (2 ms at 32-way), stage 2 12 ms at
  8-way, stage 3 4, stage 4 15. Activations are smaller than stage 5. Floor ~30-60 ms total;
  **1409 ms today**. They are bound by op count, glue (qkv-to-volume, head-unflatten, head-allgather,
  linear-order gather) and replication, not by math or DRAM.
- **Pixel pull.** 0.91 GB of uint8 RGB in 99 ms (~9 GB/s). Overlap gave nothing (#241); ~50-100 ms floor.

**Floor:** det ~0.05 + stage 5 8 x (15-25 + 10-15) = 0.2-0.3 + pre/pull ~0.1 = **~0.35-0.45 s**.
**< 1 s is reachable on paper, with about 2.5x slack over the floor.** It is not reachable by more
4-7% knobs: it needs the three structural items below, each landing near its target.

### 0.2 Ranked structural levers (real-time savings = tree x 0.885)
| # | lever | target | saving (real) | risk | effort |
|---|---|---|---|---|---|
| R1 | **NA K/V reuse in `neighborhood_sdpa`** (reader keeps a sliding K/V ring in L1 across a core's W-adjacent chunks; same per-chunk key order, so bit-identical) | 152.7 -> ~45 ms/block (~25 with BF8) | **-0.76 s** (-0.9 s with BF8) | none (exact) | C++ kernel, 3-5 d |
| R2 | **Det stages on 32 chips, glue removed**: stage 1 off replication (2-D split with NA halos like stage 5, or TP heads 8 x T 4); stages 2-4 on the stage-5 2-D split with all heads per chip (removes head-allgather, head-unflatten, the stage-4 -> 5 reshard and, if the head-allgather means only heads split on the 4-axis, 4x replicated MLP/norm work: verify first) | 1409 -> ~400-500 ms tree | **-0.8 to -0.9 s** | none (exact) | Python/ttnn, 1 wk |
| R3 | **Fused stage-5 block**: K and V halo as one CCL started before the NA op and overlapped with interior chunks; SwiGLU with the intermediate in L1; qkv-proj epilogue does norm+rope; fold context/norm-mod/residual | 124 -> ~50 ms/block | **-0.5 s** | low | 1 wk |
| R4 | BF8 K/V (#253) | NA -45% today; ~-20 ms/block after R1 | -0.5 s alone, ~-0.15 s after R1 | medium (t11 saw PCC 0.15; recheck) | A/B queued |
| R5 | pre/tail: device randn + brick fused, pixel pull split per tray | 163 -> ~90 | -0.06 s | none | small |

Projection: 3.38 -> R1 2.62 -> R2 ~1.8 -> R3 ~1.3 -> R4 ~1.15 -> R5 ~1.1 s. **Best realistic: ~1.0-1.2 s.**
**< 1 s needs R1 as the 2-D/multicast form (NA at its ~20 ms compute floor) or R2/R3 beating their
targets**: aggressive case 8 x (20 + 35) + 0.3 + 0.1 = **~0.85 s**. Decide after R1 + R2 are measured.

**Stop lever-per-task iteration** for anything predicted under 5% of decode (~0.17 s): tile-level NA
narrowing (t213: <=1.19x of NA, ~2% after R1), further chunk/brick shape sweeps, single elementwise
folds (context/residual/norm-mod 1-2% each: bundle into R3), fidelity flips, tracing. Each costs
$4-6 to A/B; they go into R1-R3 as sub-steps, not as tasks.

Prior work to reuse: tt-metal SDPA K/V store-and-forward chains and multicast (PR #57395, issue #56691:
chains are disabled for windowed/sliding-window SDPA because cores with different K ranges deadlock on
lock-step forwarding; the issue proposes forwarding the union of K ranges and letting each core pop
what it does not use, which is R1's 2-D form); tt-blaze Gemma-4 sliding-window SDPA keeps "an L1 ring
buffer that holds exactly one window" (`gemma4_sliding_window_decoder_layer_stage/op.py`), the 1-D form;
Confluence "SDPA and TopK on Blackhole: the performance counter investigation" (TA space, page 2887909469)
for BH DRAM-read limits.

### 0.3 Next tasks (run side by side; device A/Bs one at a time per box)
1. **R1 NA K/V L1 ring** (code, C++). Spec in the hand-off of #257. Accept: output md5-identical to
   default on host-noise seeds 0-1 (else PCC >= 0.9999 and PSNR within 0.1 dB of 55), NA op <= 60 ms/block
   in the stage tree, decode <= 2.75 s. If L1 does not fit the 896 KB bf16 ring next to the CBs, do the ring
   for K only or along H (4x7 column), or take it with BF8 if #253 passes.
2. **R2 det stages 32-way** (code, Python). Accept: PCC >= 0.9999 vs #214 refs, PSNR within 0.5 dB,
   det tree <= 600 ms, decode -0.7 s or better; 5-seed visuals if PSNR drops > 0.5 dB.
3. **R3 fused stage-5 block** (code). Accept: non-NA stage-5 <= 70 ms/block, decode -0.4 s or better,
   same quality gates. Can start now; rebase on R1 before its A/B if R1 lands first.

## 1. What we know

### 1.1 Model (checkpoint metadata)
| stage | grid at 1080p 145f (T,H,W) | dim | depth | NA window | executor today |
|---|---|---|---|---|---|
| 1 (det) | (21,34,60) incl. 2 ghost frames | 2048 | 4 | (3,7,7) | linear-order gather + dense masked SDPA, **replicated on all 32 chips** (W=60 not /8) |
| 2 (det) | upsample (1,2,2) | 1024 | 6 | (3,7,7) | bricked_sp_w_sharded, per-block to_bricked/to_natural |
| 3 (det) | upsample (2,1,1) | 512 | 4 | (3,5,5) | same |
| 4 (det) | (T,136,240) | 512 | 2 | (3,5,5) | same |
| 5 (1-step diffusion, x0) | patch 4: (145,272,480) = 18.9 M sites | 256, 4 heads x 64 | 8 | (11,11,11) | keep-bricked, W over 8 (sp_axis 1), heads over 4 (tp_axis 0), bands of 78 frames |

417 M params; stages 1-4 hold 97.7% of weights, stage 5 almost none but nearly all FLOPs.
The DiffVAE has no conv3d blocks: conv_in/conv_out are patch linears, upsamples are linear + depth-to-space.

### 1.2 Time breakdown (all numbers measured earlier; sources in brackets)
Pipeline, t48 code (t20 NOTES, blx03 job 874): DiffVAE **11.77 s** (gen#1), 17.8 s (gen#0, compile).

Standalone decode bench (`run_ltx25_diffvae.sh` history, pre-t11 code, same as t48's DiffVAE):
| item | ms |
|---|---|
| decode, original | 10753 |
| + ring topology, 2 links | -1711 |
| + det fused_qkv | -499 |
| **decode** | **8543** (8818 with TT_DIT_BLOCK_PROF) |
| det stages 1-4 | 1782 (20%) — stage 1 alone 533 (was 947) |
| stage 5 | 7033 (80%) |
| stage-5 kv all-gather (inside stage 5) | 840 (was 2163) |
| stage-5 head all-gather | 130 (was 409) |

t11/t23/t25 branches (NOT in t48 — confirmed by log grep), standalone bench: **11.66 s -> 5.48 s**
(6.79 s was the bit-exact `2,1,1` chunk reference). Commits: 115286885c8 (stride-1 NA chunks two
bricks along T), c6ad9bfc447 (Q/output tiled, one head per chip), 6ff194f9b13 (stage-5 AdaLN folded
into fused RMSNorm, PCC 0.999922 / 53.9 dB), 55f146f4891 (halo tiles / widened bricked tiles),
39b5ca67409 (fused stage-5 qkv default, bit-identical, 5607 -> 5481 ms), f5bb6546350
(`DIFFVAE_LATENT` saved-latent decode). In that run stage 5 was 78% of decode,
`neighborhood_sdpa` ~1.8 s, pixel pull 424 ms. Rejected there: KV bf8 (PCC 0.15), stage-5 MLP LoFi
(no gain). Branches: ttp/t11-diffvae-na-decoder-profile-optimization- e72eef929d0, ttp/t23-* 53a44f4b268,
ttp/t25-* 21ee6a13483.

### 1.3 Unmeasured (must be measured before the big code items commit to numbers)
1. Per-op device profile (Tracy) of a full decode on 4x8. No op-level split exists for stage 5:
   linears vs NA vs layout/permute vs CCL vs norms/AdaLN/RoPE. t11's profile logs were deleted.
2. The pipeline vs standalone gap on the same code: 11.8 s (pipeline) vs 8.5 s (bench). Candidates:
   co-resident DiT forcing smaller bands, eager dispatch with a cold program cache, the pixel pull, ttnn
   allocator fragmentation. And why t11's own baseline was 11.66 s, not 8.5 s.
3. Traced decode time (`test_decode_trace_timing` exists; never run at 145f on 4x8).
4. SDPA math fidelity actually used by `neighborhood_sdpa`, and the real per-core matmul utilization.
5. Tile-level sparsity of the gather for each brick/chunk/stride choice (CPU, cheap — task C2).
6. How much of the 424 ms pull overlaps with compute today (`defer_yuv` exists on t48).

### 1.4 Compute floor (back-of-envelope, stage 5 at 145f)
- Linears per site per block: qkv 256x768 + out 256x256 + SwiGLU 256x2048 + 1024x256 (+ context proj)
  ~ 2.2 MFLOP -> x 18.9 M sites x 8 blocks = **3.3e14 FLOP**.
- Exact NA: 4 heads x 1331 x 64 x 2 (QK, PV) x 2 = 1.36 MFLOP/site/block -> **2.1e14 FLOP**.
- Tile granularity: a 32-site query brick (2,4,4) has a union window of 12x14x14 = 2352 sites
  = 1.77x the exact 1331. Today's gather is 147-192 key bricks per query brick vs 41.6 exact: **3.5-4.6x**.
  So: today ~1e15 FLOP; with tile-level narrowing ~7e14; with GNA stride = brick ~5.4e14.
- Per chip (32 chips) at 7e14: 2.2e13 FLOP. BH bf16 matmul peak is roughly 190 TF/s at LoFi,
  ~half at HiFi2 (approximate, to be checked). At 40-50% utilization that is **0.3-0.5 s for stage 5 alone**.
- Activation traffic: a 256-ch tensor is ~300 MB per chip; the 2048-wide SwiGLU intermediate ~2.4 GB per
  chip per block if it round-trips DRAM (~6 ms at ~400 GB/s), so the MLP is DRAM-bound unless fused.
  This explains why stage-5 MLP LoFi gave nothing on t11.

**Verdict:** < 1 s is at the edge of the hardware with exact math. It needs (a) all 32 chips doing
distinct stage-5 work (today 4-way TP replicates part of it), (b) tile-level narrowing, (c) fused,
L1-resident blocks at >= 30-40% matmul utilization, (d) the det stages and the pull below ~0.3 s together,
(e) tracing. GNA stride (approximate, quality risk) is the reserve lever if exact math stalls at ~1.2-1.5 s.

## 2. Prior art

- Our own: `models/tt_dit/layers/NEIGHBORHOOD_ATTENTION.md` (design, terminology, 27 boundary regimes,
  keep-bricked; the det stages' per-block to_bricked/to_natural is listed as open), `neighborhood_sdpa` op
  (`ttnn/.../sdpa/device/neighborhood_*`), `neighborhood_reference.py` (torch reference of the window rules).
- t11/t23/t25 (above): 2.1x on the standalone bench, not yet in t48.
- James Lee, PR #54624 "Neighborhood attention helper refactor": merged into `na-integration` on
  2026-08-27, **not on main, not in t48**. origin/na-integration last updated 2026-09-17. Slack
  #dit-project 2026-09-11: LTX-2.5 is the only TT model using NA. Read it before T6 to avoid duplicate work.
- Others' DiffVAE branches: origin/ltx25-diffvae-stage5 (Noble Woodall), origin/nwoodall/ltx/diffvae-6s-memory-2026-08-14
  (Jonathan Su), origin/rsalman/diffvae/mesh-sharding-2026-08-15 (Rahmy Salman). Check the last for 2-D sharding ideas (T7).
- TT sliding-window SDPA: ssinghal/ring_sdpa_sliding_window, gchoudhary/quasar-sdpa-windowed-zero-unattended-rows
  (skip fully-masked rows — the 1-D version of narrowing), 57684-sdpa-decode-sliding-window-barrier,
  ring_joint_sdpa multi-hop sliding-window halo (#57453, b0000e31941), kevinmi/neighbor-pad-conv3d-fused.
- tt-metal issue #50438 (losullivanTT, "Aang"): a sparse-attention kernel and neighbor-pad buffering fix
  (8.3 -> 6.0 s from single-buffer + barrier). Different model; read only for ideas.
- GNA (Hassani et al. 2025, https://arxiv.org/abs/2504.16922, https://research.nvidia.com/labs/cosmos-lab/gna/):
  stride groups queries so they share one window; stride = window -> blocked attention. Their Blackwell
  kernel reaches the theoretical block-sparse speedup; training-free use on Cosmos-7B/HunyuanVideo/FLUX
  gave 28-46% end-to-end with small quality loss. The upstream LTX-2.5 code ships NATTEN, Triton, eager
  and CuTe-DSL NA paths (Kevin Mi's diff summary in #dit-project, 2026-08-11).
- Fused NA (Hassani et al., https://arxiv.org/abs/2403.04690): fuse the gather into the attention kernel,
  tile-level masking only on boundary tiles. That is what our bricked executor does; narrowing is the missing part.

## 3. Ranked plan

Savings are against the pipeline gate (11.8 s today). "Exact" = same math up to bf16 rounding.

| # | item | expected saving | quality risk | effort |
|---|---|---|---|---|
| P1 | **Port t11/t23/t25 onto t48** (rebase 6 commits; resolve against t48's NA changes) | 11.8 -> ~5.5-6.5 s | none/low (bit-exact or PCC 0.9999) | 1-2 d |
| P2 | **Profile** the ported decode (Tracy, 1 job) and close the pipeline/bench gap (1.3 #1-2) | enables P3-P8 numbers | none | 1 d |
| P3 | **Stage 5 on all 32 chips: 2-D spatial split H/4 x W/8, all 4 heads per chip, K/V halo exchange (5 sites) on both mesh axes**; drops the TP-heads all-gathers and the 4x replication of the MLP/norm work; with 1/4 the per-chip memory, **one band** (no slab T-halo recompute). 272/4 = 68 rows = 17 bricks, 480/8 = 60 = 15 bricks. Halo traffic ~0.2 GB/chip/block ~ 2 ms over 2 eth links. | -1.5 to -2.5 s | none (exact) | 1-2 wk |
| P4 | **Narrowing in `neighborhood_sdpa`** (block-sparse over tiles): per query chunk, iterate only key bricks whose tile is not fully masked; host-side plan tables per regime (27). Attention FLOPs 3.5-4.6x -> ~1.8x exact. | NA sdpa ~1.8 s -> ~0.8 s (before P3; ~-0.25 s after P3) | none (exact) | 1 wk |
| P5 | **Fused stage-5 block kernels**: RMSNorm+AdaLN+qkv in one op (exists), out-proj+residual, and a SwiGLU MLP that keeps the 2048-wide intermediate in L1 per row block (no DRAM round trip). Then retry LoFi/bf8 for linears and bf8 for the MLP intermediate. | -30-50% of stage-5 linear time | low (fidelity part needs PCC) | 1-2 wk |
| P6 | **Det stages**: (a) keep-bricked across det blocks like stage 5; (b) stage 1 off replication: TP heads 8-way (32 heads -> 4/chip, colpar qkv/rowpar out) x T split 4 with 1-frame halos; (c) det stages 2-4 use the same 2-D split as P3 so the handoff to stage 5 is local (the old 2.2 s handoff that killed `stages_sp_axis` disappears). | 1.8 s -> ~0.3-0.5 s | none (exact) | 1-2 wk |
| P7 | **Pixel path**: emit uint8 YUV on device (`defer_yuv` path), pull per band/per chip on a second CQ while the next band computes, all 32 chips in parallel. | -0.3 to -0.4 s | none | 3-5 d |
| P8 | **Brick / chunk / tile shape sweep** after P4: Q chunk (2,4,4) vs (4,4,4)/64 sites (K reuse), K chunk, subblocks; CPU table from C2 first. | 5-15% of NA | none | 2-3 d |
| P9 | **Trace the whole decode** (fixed shapes at 1080p 145f; noise and timestep already device-resident) | -50-200 ms once device time < 2 s | none | 3-5 d |
| P10 | **GNA stride = brick (2,4,4)** (`gna_stride` already in `DiffVAEStage5Config`): every query in a brick shares one 11^3 window -> dense blocked attention, no mask, 1.0x FLOPs. Optional (5,5,5)-scale strides only if (2,4,4) is clean. | NA -40-50% beyond P4 | **medium-high**: model trained at stride 1. CPU study first (C3), then 5-seed visuals + VBench. | 2 d + quality |
| P11 | **dtype/fidelity**: SDPA QK at LoFi vs HiFi2; bf8 Q (not KV: bf8 KV broke, PCC 0.15). RoPE must stay HiFi (pair-swap matmul). | 5-15% of NA | low-medium | 2 d |
| P12 | **Stage-5 window 11^3 smaller** (e.g. (7,11,11)) | large | **high** (changes the trained model). Only as a CPU curiosity, never default without user sign-off. | — |
| P13 | **Reuse of the 2.3 conv-VAE work**: no conv3d in the DiffVAE, and stages cannot be mixed between the two decoders (different weights/feature spaces). Reusable: neighbor_pad halo machinery (P3/P6), the exact-shard output assembly idea of `LTX_VAE_EXACT_SHARD` (P7), the t93 tracing approach (P9). `LTX_CONV3D_BLOCKING_MESH` does not apply. | via P3/P7/P9 | — | — |

Expected path: P1 ~5.5-6.5 s -> P3+P4 ~3-3.5 s -> P5+P6 ~1.5-2 s -> P7+P8+P9 ~1.0-1.3 s -> P10 (if quality
holds) < 1 s. Every step after P2 must be re-estimated from the profile.

Constraints to respect: `neighbor_pad_async` deadlocks on Ring (halo pinned to Linear); 2 eth links per pair
is the max on BH galaxy (num_links=4 impossible, run 745); exact NA at 6 s 1080p does not fit co-resident with the
DiT at today's banding (P3 should fix this; verify DRAM: 19.7 of 31.4 GiB was allocated before decode on blx03);
the weight cache is keyed by `parameter_layout()`, so new forms write a new cache dir (watch /home free space).

## 4. Benchmark and quality protocol

1. **Latents (once)**: a t48 pipeline run (current defaults, DEFAULT prompt, seeds 0-4, 1080p 145f) saves the S2
   output latent per seed (~10 MB each) to `/var/tmp/fasth3/diffvae/latents/seed{0..4}.pt` on the box and to
   `tt-project/baselines/ltx25_1080p_6s/diffvae_latents/`. If t48 lacks a dump hook, add `LTX_DUMP_LATENT=<dir>`.
2. **Fixed stage-5 noise**: with `device_boundaries=True` stage 5 draws noise on device, so a resharded decode
   gets different noise and the comparison floors at ~43.7 dB PSNR (the DiffVAE's own seed-to-seed floor).
   The bench passes host noise `torch.randn(..., Generator().manual_seed(seed))` explicitly to both arms.
3. **Reference**: the unoptimized DiffVAE (t48 @ 5e4e0cd643a, production options, fixed noise) decoding those
   latents -> `ref_dvx_<seed>` pixels (uint8 YUV) stored next to the latents. ref_dv145 (older 5-seed set) stays a
   cross-check only.
4. **Decode-only job** (`test_diffvae_ltx.py::test_decode_wsp_timing` + `DIFFVAE_LATENT` from f5bb6546350, or a new
   `test_decode_saved_latents`): open full 4x8 mesh, load cached weights, 1 warm-up decode, then decode the
   5 latents, save pixels, print per-seed time and the timing tree. One config per job. First job: `-t 600`;
   after that measured +50% (expect ~150-250 s warm). Cold JIT on a fresh box needs a cache-warming job first.
5. **Gates per change**: (a) PCC/PSNR of each seed vs reference: exact items must stay >= 0.9999 / >= 45 dB;
   below that, or for any approximate item (P10-P12), (b) VBench subset (subject/background consistency,
   motion smoothness, imaging/aesthetic quality) on the 5 seeds vs reference, and (c) visuals: mp4 + a still
   frame per seed shown to the user. (d) Final: one full pipeline run, `VAE decode (forward)` < 1 s on 5 seeds.

## 5. Tasks (can run side by side)

Boxes: blx01 (broker; tray 3 dropped twice on the t166 config), blx03 (serial runner
`blx03-enqueue.sh`, watch tray 2 / chip 12), exabox dit nodes idle 2 h+ (self-ending sbatch, Mac tunnel
needed). g15blx02-device is **paused**: builds and CPU work only. One project device job per box at a time.

| id | kind | what | box | depends on |
|---|---|---|---|---|
| C1 | code | Port t11/t23/t25 DiffVAE commits onto t48 (P1); build on g15blx02 | g15blx02 build, blx01 smoke | — |
| C2 | CPU | Tile-sparsity calculator from `neighborhood_reference.py`: key tiles touched vs gathered vs exact per brick/chunk/stride/window, all 27 regimes, every stage | g15blx02 CPU | — |
| M1 | measure | Save 5-seed latents; build fixed-noise reference decodes (protocol 1-3) | blx03 runner (pipeline) then blx01 | — |
| C3 | CPU | GNA stride / window quality study on the torch reference: crop of 25 frames from M1 latents, stride (2,4,4) etc. vs stride 1, PSNR/visuals | g15blx02 CPU | M1 |
| M2 | measure | Decode-only bench of C1 + Tracy per-op profile + pipeline/bench gap (1.3 #1-3) | blx01 | C1, M1 |
| C4 | code | Narrowing in `neighborhood_sdpa` (P4), then shape sweep (P8) | blx03 runner / exabox | C1, C2; read na-integration first |
| C5 | code | Stage-5 2-D spatial SP over 32 chips, single band (P3) | blx01 | M2 |
| C6 | code | Det stages: keep-bricked, stage-1 TP8 x T4, shared 2-D split (P6) | exabox or blx03 | M2; coordinate with C5 |
| C7 | code | Pixel path: uint8 on device, overlapped per-band pull (P7) | blx03 runner | C1 |
| C8 | code | Fused stage-5 MLP / out-proj kernels + fidelity/dtype sweep (P5, P11) | exabox | M2 |
| C9 | code | Trace the full decode (P9) | blx01 | C5, C6, C7 |
| R1 | review | Independent review + 5-seed quality gate for each landed item; GNA decision memo if C3 is clean | any free box | each code task |

Landing: code tasks push to their own ttp/tNNN branches and land on ttp/t48-ltx25-integrated (DiffVAE stays
opt-in, so the default pipeline is untouched). No PRs.

Skills to use: tenstorrent/skills tt-model-bringup (optimizing, tracing, multichip, graph fusing),
tt-debug-tools (profiler, triage); tt-buddy `tt:profiler` for M2.
