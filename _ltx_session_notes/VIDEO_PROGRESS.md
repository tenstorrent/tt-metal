# LTX-2.3 Distilled AV — Video/DiT Optimization Progress

Resumable log for executing /home/smarton/LTX_PERF_PLAN.md. Branch:
`smarton/optimizer/ltx-video` off `origin/ltx-perf`. Target: speed up DiT
diffusion (Stage1+2 ~6s) on BH 4x8; audio already optimized.

Timestamps are UTC (host clock). PT = UTC-7.

## Status legend
- [ ] pending  [~] in progress  [x] done  [!] negative/failed result (still valid)

---

## Phase 1 — Baseline + per-op profile

### 1a. Clean E2E gen#1 per-stage timings
- [~] Running `run_video_baseline.sh` (test_pipeline_distilled -k bh_4x8sp1tp0_ring,
  LTX_TRACED=1 LTX_VOC_TRACE=0 LTX_TIME_STAGES=1). gen#0 captures, gen#1 = steady-state replay.
- 2026-06-14 ~08:04Z: prior run 075325-37 was killed (-9) at 162s while still in gen#0
  warmup (reached "warmup upsample" then VAE reload) — external kill, NOT broker timeout
  (timeout was 7200s); no gen#1 stage numbers captured. Re-submitted as job 080445-39
  (queued behind sibling audio job 080305-38).
- NOTE: audio-decode warmup floods ~30k benign "TT_FATAL: Index N larger than runtime args
  size N" log lines (audio vocoder op runtime-arg logging on the full-mesh T-shard=8 path,
  LTX_AUDIO_SUBMESH unset) — warmup continues past them; not a crash. Out of video scope.
- [x] DONE 2026-06-14 08:24Z (job 080445-39, clean, WALL_SEC=1034). 4x8 BH ring,
  1088x1920, 145f, 10 steps (s1=7, s2=3), LTX_TRACED=1, bf16/HiFi2 baseline.
  gen#0 = capture pass (Stage1 30.3s, Stage2 30.0s, Total 62.5s — capture overhead).
  **gen#1 STEADY-STATE (pure trace replay) = the baseline:**

  | Stage            | gen#1   |
  |------------------|---------|
  | Encoder (cached) | 0.00 s  |
  | Stage 1 denoise  | 2.71 s  |
  | Latent upsample  | 0.25 s  |
  | Stage 2 denoise  | 3.26 s  |
  | VAE decode       | 1.15 s  |
  | Audio decode     | 0.89 s  |
  | **Total**        | **8.26 s** |

  **DiT denoise = 2.71+3.26 = 5.97s = 72% of E2E.** Confirmed dominant cost / opt target.
  (Encoder cached at 0.00s since prompt reused; audio 0.89s already optimized; VAE 1.15s = Tier-3.)

### 1b. Per-op DiT profile (perf counters + tt-npe NoC/DRAM BW)
- [!] First attempt (job 083344-42, --collect-noc-traces) FAILED: enabling PROFILE_NOC_EVENTS=1
  breaks the fabric writer kernel build on THIS checkout —
  `fabric_event_profiler.hpp:134/141: error: 'tt::tt_fabric::MeshRoutingFields' has not been
  declared` (the type exists in fabric_edm_packet_header.hpp but isn't visible in the kernel
  compile ctx) -> TT_THROW "Failed to generate binaries for broadcast_rm_writer". Pre-existing
  tt-metal NoC-event-profiler/fabric incompat, not introduced here. So tt-npe DRAM-BW
  (the secondary bfp8-vs-bfp4 question) can't run without patching that header.
- [~] Corrected: re-run WITHOUT --collect-noc-traces (compute perf-counters only —
  fpu/sfpu/pack/unpack/instrn + per-op device time). That resolves the §5 GEMM-vs-attention
  fork directly; bfp8 was already locked by the plan so the missing DRAM-BW number is non-blocking.
- NOTE: baseline is actually **8+3 = 11 steps** (Stage 1 logs "8 steps"; DISTILLED_SIGMA_VALUES
  = 9 values = 8 intervals), not the 10 the plan text assumed. Matters for L2 (10->7 becomes 11->7).
- [!] Full-E2E untraced profile (job 095136-53, no NoC) RAN the gen but FAILED post-processing:
  the untraced full pipeline — esp. the 814s profiler-instrumented AUDIO DECODE — overflowed the
  profiler DRAM buffers ("markers were dropped, bufferEndIndex=768000" across all 32 devices), so
  no valid profile_log_device.csv was produced -> tracy -r AssertionError, no ops CSV. Lesson:
  must SCOPE the profile to just the DiT (not full E2E with audio/VAE).
- [x] Scoped profile DONE (job 105824-56, 2026-06-14 11:00Z). ONE DiT transformer block
  (stage_2 shape 38760 tok, 4x8 ring, video), tracy per-op device-kernel time. CSV:
  generated/profiler/reports/2026_06_14_11_00_41/. (perf-counters flag dropped — it forces
  tracy's legacy parser which breaks -k; per-op device TIME share answers §5 directly.)

  **Per-op device-kernel time, one block (total 518 ms):**

  | class                                            | ops  | ms    | share |
  |--------------------------------------------------|------|-------|-------|
  | matmul/linear (incl CCL-FUSED AG+mm, RS+mm)      | 224  | 182.8 | 35.3% |
  | SDPA/attention (RingJointSDPA 29.0% + SDPA 0.4%) | 64   | 152.2 | 29.4% |
  | other (tilize 11.7 / typecast 8.2 / ternary 2.2 / AG / slice) | 2432 | 162.7 | 31.4% |
  | norm (RMSNorm)                                   | 448  | 20.3  | 3.9%  |

  Top ops: RingJointSDPA **29.0%**, AllGatherMinimalMatmulAsync 21.5%, MinimalMatmulStridedReduceScatter
  13.3%, Tilize 11.7%, Typecast 8.2%.

  **DiT BOUND CLASS = attention + CCL/data-movement, NOT FPU/GEMM-compute.** Evidence: (1) the
  "matmul" GEMMs are fused with TP all-gather / reduce-scatter CCL (AllGatherMinimalMatmul,
  MinimalMatmulStridedReduceScatter) -> gated by data movement, not pure matrix math -> why LoFi
  quant gave only ~5%; (2) SDPA (RingJointSDPA) is the single biggest op at 29%, dense;
  (3) tilize+typecast+ternary ~26% is layout/format overhead. **§5 FORK RESOLVED -> L4 (sparse
  attention) is the higher-ceiling lever** (dense SDPA is the largest single addressable chunk),
  matching plan §1. bfp4 would NOT help (compute isn't the bottleneck) — confirms bfp8 was right.

---

## Phase 1 SUMMARY (deliverable)
- Baseline E2E (gen#1 steady-state, bf16/HiFi2, 8+3=11 steps): **8.26 s**; DiT denoise **5.97 s = 72%**.
- DiT bound class: **attention/CCL/data-movement-bound, NOT GEMM/FPU-bound** (per-op profile + L1 result).
- L1 quant (all_bf8_lofi): quality PASS (block PCC 99.89-99.93 vs 0.988 gate); E2E **8.26 -> 7.90 s (-4.4%)**,
  DiT -5.5%. Modest because matmul is CCL-gated. SHIPPED behind LTX_QUANT (off by default).

---

## L1 — DiT linear quant (bfp8_b weights+acts + LoFi)
- [x] PORTED + WIRED. `pipelines/ltx/quant_config.py` (commit 6adcc26c67e): QuantConfig
  dataclasses + all_bf8_lofi(); apply_quant_config / set_quant_config; pipeline hook
  `_maybe_apply_quant_config` gated on env `LTX_QUANT=<preset>` (default OFF = bf16/HiFi2,
  byte-identical baseline). Carve-outs: attn1.to_out + video attn2.to_out keep bf16 weights
  (fused dit_minimal_matmul_addcmul ternary inputs must match weight tile fmt); ffn.ff2 (RS
  addcmul) is bf8; SDPA stays bf16/HiFi2. Verified hook attrs exist (mm/sdpa_compute_kernel_config,
  ff_compute_kernel_config, to_qkv/to_out/to_q/to_kv, audio_* mirrors) + Parameter._data/.dtype path.
- [x] QUALITY GATE wired (commit c81a72dab93): factored apply_quant_config_to_block; block PCC
  test (test_ltx_transformer_block, ring_bh_4x8, video) honors LTX_QUANT to apply the exact quant
  path vs the diffusers bf16 oracle. Threshold 0.988 (>8 devices). Runner run_l1_quant_gate.sh.
- [x] QUALITY GATE PASSED (job 084033-43, 2026-06-14 08:57Z). Block PCC vs diffusers bf16 oracle,
  ring_bh_4x8, video, threshold 0.988:

  | shape (tokens)    | baseline bf16     | all_bf8_lofi          | verdict |
  |-------------------|-------------------|-----------------------|---------|
  | stage_1 (9690)    | PCC 99.9658% (RMSE 2.8%) | PCC 99.9316% (RMSE 4.3%) | PASS |
  | stage_2 (38760)   | PCC 99.9658% (RMSE 2.8%) | PCC 99.8890% (RMSE 5.6%) | PASS |

  bf8_lofi degrades PCC only ~0.03-0.08% — far above the 0.988 gate. **L1 quant quality CLEARED.**
- DEVICE-HEALTH DETOUR (resolved): at 09:07Z a sibling audio job (44) was killed (-9) and left
  the board in a bad FW state — every full-mesh job 09:07-09:25 died at ttnn device init
  ("failed to initialize FW! Try resetting the board"). NOT my code (crash before LTX runs).
  I did NOT reset (per constraints); operator ran `tt-smi -glx_reset` (job 51, 09:25Z) -> board
  recovered. Retry (job 092733-52) ran clean.
- [x] E2E DONE (job 092733-52, 2026-06-14 09:45Z, clean, WALL_SEC=1076). LTX_QUANT=all_bf8_lofi
  applied to all 48 blocks (confirmed in log; re-applies after dynamic_load reload too).
  gen#1 STEADY-STATE before->after:

  | Stage           | baseline bf16 | L1 all_bf8_lofi |
  |-----------------|---------------|-----------------|
  | Stage 1 denoise | 2.71 s        | 2.57 s          |
  | Latent upsample | 0.25 s        | 0.26 s          |
  | Stage 2 denoise | 3.26 s        | 3.07 s          |
  | VAE decode      | 1.15 s        | 1.15 s          |
  | Audio decode    | 0.89 s        | 0.85 s          |
  | **Total**       | **8.26 s**    | **7.90 s**      |

  **DiT denoise 5.97 -> 5.64 s (-0.33s, -5.5%); E2E 8.26 -> 7.90 s (-0.36s, -4.4%).** Quality PASS
  (block PCC 99.93/99.89). Test PASSED. Commit shas: 6adcc26c67e (quant_config) + c81a72dab93 (gate).
- **KEY INFERENCE for the §5 fork:** a LoFi quant that should ~halve GEMM math yields only ~5%
  on the DiT -> the traced DiT is **NOT FPU/GEMM-bound** (matmul math is not the bottleneck).
  Points to attention/CCL/dispatch dominating -> **L4 (sparse attn) is the higher-ceiling lever**,
  matching plan §1 "attention-dominated at 145f@1080p". Profile (1b) to confirm with per-op counters.
  LTX hook differs from WAN: linears take `mm_compute_kernel_config` (LTXAttention) /
  `ff_compute_kernel_config` (block) explicitly — they do NOT read per-Linear `.compute_config`.
  So quant = typecast weights + set those two compute-config attrs per block.
  Carve-out: self_attn_out fused addcmul needs bf16 weights; cross_attn_out (attn2 video) and
  ffn ff2 also run forward_fused_addcmul / dit_minimal_matmul_addcmul_fused -> verify whether
  they need the same bf16 carve-out.

## L2 — denoise 11 -> 7 (stage1 8->5 + stage2 3->2)
- [x] SCAFFOLDING (commit c98760b308d): sigma schedules now env-overridable
  (LTX_S1_SIGMAS / LTX_S2_SIGMAS, comma-separated, strictly decreasing, must end at 0.0).
  Unset = shipped 8+3 defaults, byte-identical baseline. No default change.
- L2 schedule chosen: S1 5-step `1.0,0.975,0.909375,0.725,0.421875,0.0` (keep 1.0 + the
  structure-defining large-σ-jump tail, thin the 5 near-σ=1 micro-steps), S2 2-step
  `0.909375,0.725,0.0` (drop one interior). = 7 total, matching FastVideo's 5+2.
- QUALITY GATE approach (honest): the distilled checkpoint IS the model -> NO torch oracle for
  the multi-step trajectory. Fewer steps -> a genuinely different (not just lossy) output, so
  the test is "is 7-step acceptable vs 11-step", measured by decoded-frame PCC/PSNR/SSIM between
  11-step and 7-step at the SAME seed/prompt/quant (bf16). NOTE: the L1 run overwrote the
  default-named baseline mp4s, so regenerating a fixed-name 11-step bf16 reference.
- [x] 11-step bf16 ref (job 57): gen#1 Stage1 2.7s + Stage2 3.2s = 5.9s, Total 8.1s (= baseline).
  mp4 ltx_av_ref_11step.mp4.
- [!] 7-step L2 (job 58, S1=1.0,0.975,0.909375,0.725,0.421875,0.0 + S2=0.909375,0.725,0.0):
  PERF WIN big — gen#1 Stage1 **1.7s** + Stage2 **2.2s = 3.9s**, Total **6.1s** (DiT -2.0s/-34%,
  E2E 8.1->6.1 -25%). BUT **QUALITY GATE FAILED HARD**: frame-compare vs 11-step ref
  (compare_videos.py) PSNR **14.5 dB**, PCC **0.31**, SSIM **0.61** — and visual inspection of
  frame 72 confirms it's a COMPLETELY DIFFERENT scene (different woman/clothing/background/
  framing), not a degraded version. The aggressive cut (1.0 -> 0.975, dropping the whole near-σ=1
  micro-step cluster) sends the trajectory to a different mode. Confirms plan §2 warning: the
  near-σ=1 steps do real STRUCTURE-setting refinement. The 7-step image itself is sharp/coherent,
  but the output is not the same generation -> for fixed-seed reproduction this is a fail.
- [x] L2 v2 (job 59, drop ONE mid near-σ=1 step 0.98125 => S1 7-step + S2 3-step = 10 total):
  gen#1 Total 7.9s (DiT 5.7s, only -0.2s — 1/11 steps). Quality HOLDS: vs 11-step ref PSNR
  **28.3 dB, PCC 0.972, SSIM 0.92**, frame 72 visually = same woman/scene/composition. So a
  single near-σ=1 drop is on-trajectory.
- SENSITIVITY MAP: dropping 1 near-σ=1 step -> PCC 0.97 (safe, tiny win); dropping the whole
  4-step near-σ=1 cluster (1.0->0.975) -> PCC 0.31 (different scene, big win). The near-σ=1
  cluster sets GLOBAL STRUCTURE; cutting it reroutes the generation. Win and trajectory-fidelity
  trade off directly.
- [x] L2 v3 (job 60, S1=1.0,0.99375,0.975,0.909375,0.725,0.421875,0.0 + S2=0.909375,0.725,0.0
  => S1 6-step + S2 2-step = 8 total): gen#1 Stage1 2.1s + Stage2 2.2s = 4.3s, Total **6.5s**.
  DiT 5.9->4.3 (-27%), E2E 8.1->6.5 (**-20%**). Quality HOLDS: vs 11-step ref PSNR 22.8 dB,
  PCC **0.90**, SSIM 0.84; frame 72 = SAME woman/scene/composition (minor pose/detail diff only).
  On-trajectory, high-quality. **VIABLE.**

### L2 RESULT — quality/perf curve (gen#1, vs 11-step bf16 ref, fixed seed=10/prompt)
| variant | steps (s1+s2) | E2E    | DiT    | PCC  | SSIM | scene vs ref      |
|---------|---------------|--------|--------|------|------|-------------------|
| ref     | 8+3 = 11      | 8.1 s  | 5.9 s  | 1.00 | 1.00 | —                 |
| v2      | 7+3 = 10      | 7.9 s  | 5.7 s  | 0.97 | 0.92 | identical         |
| **v3**  | **6+2 = 8**   | **6.5 s** | **4.3 s** | **0.90** | 0.84 | **same, minor diff** |
| 7-step  | 5+2 = 7       | 6.1 s  | 3.9 s  | 0.31 | 0.61 | DIFFERENT scene   |

**VERDICT:** the near-σ=1 cluster sets global structure — there is a cliff between 8 steps
(holds, PCC 0.90) and 7 steps (different generation, PCC 0.31). **8-step (v3) is the safe floor:
-20% E2E with the trajectory preserved.** FastVideo's 5+2=7 almost certainly uses a sigma schedule
tuned/distilled for 7 steps (we reuse the 11-step schedule's subset) — so 7 is NOT viable on our
schedule without retuning. Shipped opt-in via LTX_S1_SIGMAS/LTX_S2_SIGMAS (commit c98760b308d);
recommended L2 config = v3. Default stays 11-step (byte-identical baseline).

## L4 — sparse/sliding-tile attention (moonshot) — SCOPED PLAN (not landed)
Status: L1 + L2 banked; L4 documented as a grounded plan (per instruction: build only if banked +
time; the kernel+mask-search+finetude effort exceeds remaining time, and it needs its own quality
loop). The profile makes L4 the highest-ceiling lever: **RingJointSDPA is the single biggest op at
29.0% of a DiT block** and is fully DENSE.

**Key discovery — a windowed-SDPA primitive ALREADY EXISTS in this checkout:**
`ttnn.transformer.windowed_scaled_dot_product_attention` (ttnn/cpp/.../sdpa_windowed/, Python-exposed,
verified). It builds BLOCK-DIAGONAL attention internally from `cu_window_seqlens` (cumulative window
lengths) — written for Qwen2.5-VL windowed attention. This is the STA/VSA block-local primitive, so
L4 may NOT need a from-scratch C++ kernel — but it does NOT drop in directly. Gaps to close:

1. **Ring + joint-prompt mismatch.** LTX self-attn uses `ring_joint_scaled_dot_product_attention`
   (SP across 8 chips, `joint_strategy="rear"` appends the text prompt to K/V, `logical_n` padding
   mask — attention_ltx.py:490/547). `windowed_sdpa` is the single-device non-ring op with no joint
   tensor. Options: (a) run windowed-SDPA per-chip if each 3D window fits inside one SP shard (avoids
   ring entirely for local windows) + a separate global/prompt attention pass; (b) extend the ring
   joint kernel to accept `cu_window_seqlens` (the real C++ work).
2. **3D-local window -> 1D cu_window_seqlens mapping.** Tokens are F*H*W frame-major flattened, so a
   spatial-local 3D window is NOT 1D-contiguous. Needs either a token reorder (tile/window-major
   permutation, then inverse) so windows become contiguous blocks, or strided window descriptors.
   Block-diagonal (non-overlapping) is the floor; true sliding windows w/ halo overlap are the ceiling.
3. **Quality gate = mask search or finetune.** Training-free local-window masking is the floor
   (expect ~1.4x attn); finetune is the ceiling (~2x, plan §L4). Must validate with the SAME frame
   PCC/SSIM gate built for L2 (compare_videos.py) AND likely a perceptual check — sparsity changes
   long-range coherence, the riskiest quality axis.

**Effort/sequencing:** medium-high. Recommended first step (cheap, no kernel): a feasibility probe —
take ONE block's Q/K/V at stage_2 shape, run a windowed-SDPA with a candidate 3D window (e.g.
frame-local or 8x8x8 spatial tile) on a single chip, PCC vs the dense ring output. If a block-diagonal
windowed pass holds reasonable PCC, proceed to the reorder + per-chip integration; if not, it needs
finetune (out of scope here). **Expected ceiling: 1.4-2.0x on SDPA = ~12-20% of the DiT block (29%
* (1 - 1/1.4..1/2)) -> ~0.7-1.2s E2E** — the largest remaining single lever, bigger than L1.

---

## Landed commits (on smarton/optimizer/ltx-video — pushed to origin; NEVER pushed to ltx-perf)
- 6adcc26c67e  ltx video: DiT-linear quant config (LTX_QUANT, off by default)
- c81a72dab93  ltx video: block-level quant helper + LTX_QUANT PCC gate in block test
- c98760b308d  ltx video: env-overridable denoise sigma schedules (LTX_S1_SIGMAS/LTX_S2_SIGMAS)
All three are OFF by default -> the default pipeline path is byte-identical to the ltx-perf baseline.
The user pulls the good parts to ltx-perf themselves.

## FINAL SUMMARY (observed numbers, 4x8 BH ring, 1088x1920, 145f, gen#1 steady-state)
- **Baseline E2E 8.26 s; DiT denoise 5.97 s = 72%** (the optimization target).
- **DiT bound class: attention/CCL/data-movement, NOT GEMM/FPU** (per-op profile: RingJointSDPA
  29% top op, matmuls CCL-fused; + the L1 result below). bfp8 was the right call (bfp4 wouldn't help).
- **L1 quant (all_bf8_lofi):** PASS quality (block PCC 99.89-99.93 vs 0.988 gate); **E2E 8.26->7.90 s
  (-4.4%)**, DiT -5.5%. Modest because matmul is CCL-gated, not FPU-bound. Shipped behind LTX_QUANT.
- **L2 step cut:** 8-step (s1=6,s2=2) HOLDS quality (PCC 0.90/SSIM 0.84 vs 11-step, same scene) for
  **E2E 8.1->6.5 s (-20%)**, DiT -27%. 7-step (5+2) does NOT hold (PCC 0.31, different generation) —
  the near-σ=1 cluster sets global structure; cliff between 8 and 7 steps. Shipped via LTX_S*_SIGMAS;
  recommended config = 8-step.
- **Combined L1+L2 (8-step) would stack** (independent): ~quant -5% on top of step-cut -20%. Not run
  together yet (each measured solo); a combined run is the natural next confirmation.
- **L4 sparse attn:** documented a grounded plan (above). Top remaining lever (~0.7-1.2s);
  `ttnn.transformer.windowed_scaled_dot_product_attention` exists but needs ring/joint-prompt +
  3D-window-mapping work + a quality loop. Not landed (effort > remaining time).

## Helper scripts (in /home/smarton, not in repo)
run_video_baseline.sh (11-step bf16 baseline) · run_video_l1_e2e.sh (LTX_QUANT) ·
run_l1_quant_gate.sh (block PCC gate) · run_dit_block_prof.sh (per-op profile) ·
run_video_ref_11step.sh / run_video_l2_7step.sh / run_video_l2_v2.sh / run_video_l2_v3.sh (L2 A/B) ·
compare_videos.py (frame PSNR/PCC/SSIM gate).
