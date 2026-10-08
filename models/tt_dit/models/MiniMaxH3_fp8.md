# MiniMax-H3 denoiser: 8-bit matmuls on Blackhole — approaches, design, measurement

Design note for the opt-in 8-bit mode of the MiniMax-H3 DiT denoiser's matmuls (`FAST_H3_FP8`, see
[MiniMaxH3.md](MiniMaxH3.md)), written for the HyperFlow / Turbo branch and the 4x8 Blackhole Galaxy (TP 4 x SP 8).
The default stays bf16. Section 8 holds the measurements as they land.

## 1. What "FP8" means on this stack

- There is no fp8 e4m3 / e5m2 matmul path in this tree. `ttnn.fp8_e4m3` exists as a dtype, but it is Blackhole-only,
  row-major-only, unpacks to the FP16 (5-bit exponent) register family, and every matmul the DiT linears use —
  `minimal_matmul`, `minimal_matmul_split`, `all_gather_minimal_matmul_async`, `strided_all_gather_minimal_matmul_async`,
  `minimal_matmul_strided_reduce_scatter_async`, `dit_minimal_matmul_addcmul_fused` — whitelists only
  `BFLOAT16 | BFLOAT8_B | BFLOAT4_B | FLOAT32` for activations, weights and bias
  (`minimal_matmul_device_operation.cpp`, `all_gather_minimal_matmul_async_device_operation.cpp`,
  `minimal_matmul_strided_reduce_scatter_async_op.cpp`). The hardware encodes e4m3 as "Lf8 plus a flag"
  (`tensix_types.h`), and a change to stream 8-bit tiles through the matmul LLK is in review elsewhere (#58817); true
  e4m3 is a later step, not this one.
- The 8-bit format that runs today is **bfloat8_b**: 16 consecutive values (one 16-wide face row) share one 8-bit
  exponent set by the block's largest value; each value keeps a sign and 7 magnitude bits with the leading 1 explicit;
  a 32x32 tile is 1088 B instead of 2048 B (`blockfloat_common.cpp`, `tt_backend_api_types.hpp`). Structurally this is
  MXINT8 with block 16, not E4M3: a value 2^k below its block maximum keeps 7-k significant bits, and anything more
  than 128x below the block maximum rounds to zero. For a Gaussian block that is about 44 dB SQNR, better than E4M3's
  32 dB; for a block holding one large outlier it is much worse. Weights (one input channel, 16 output columns per
  block) are the benign case; activations (16 input channels of one token) are where outliers land.
- Two independent levers come with the format:
  1. **Bytes.** Weight reads, activations crossing the TP all-gather, K/V crossing the SP ring: all halve.
  2. **FPU passes (MathFidelity).** The matrix unit multiplies 5 bits of SrcA by 7 bits of SrcB per pass; in a matmul
     the weight (in1) is SrcA and the activation (in0) is SrcB (`llk_math_matmul_api.h`, `tech_reports/matrix_engine`).
     `HiFi2` (two passes, today's default for every H3 matmul) consumes the weight's full 7-bit mantissa and the top 6
     bits of the activation, so it is **exact for bfloat8_b x bfloat8_b**. `LoFi` (one pass, 16 instead of 32 cycles
     per tile) keeps the activation's 7 bits but rounds the **weight** to 5 significant bits. `HiFi3/4` only add the
     activation's last bit and buy nothing for 8-bit operands.
  So bfloat8_b at HiFi2 is a pure bandwidth win with no extra rounding beyond the format itself, and the compute win
  needs LoFi, which costs weight precision. The earlier H3 experiment (#58677) only pulled the first lever and saw
  about 2.5 % per step.

## 2. Where a 5 s forward's work goes (1344x768, HyperFlow 8 forwards, bucket rung 38 912 rows, 4 864 per device)

FLOPs = 2·M·K·N over the per-device shard, per forward (50 blocks):

| matmul (per block) | op path on 4x8 | M x K x N per device | TFLOP / forward | share |
|---|---|---|---|---|
| `ff.ff1` (SwiGLU, packed gate+up) | all-gather-matmul with fused SwiGLU | 4864 x 5376 x 7168 | 18.8 | 16 % |
| `attn.to_qkv` (3 chunks) | all-gather-matmul | 4864 x 5376 x 5376 | 14.1 | 12 % |
| `ff.ff2` | matmul + strided reduce-scatter with fused `residual + gate·x` | 4864 x 3584 x 5376 | 9.4 | 8 % |
| `attn.to_out` | all-gather-matmul with fused `residual + gate·x` epilogue | 4864 x 7168 x 1344 | 4.7 | 4 % |
| `adaln_proj` | plain matmul on 9 rows | 32 x 2688 x 24192 | 0.2 | 0.2 % (but 130 MB of weights per block) |
| ring joint SDPA, 14 local heads | QKᵀ and PV | Q 4864 x K 38912 x 128 x 14 | 67.8 | 59 % |

- The four block linears are **41 %** of the FLOPs at 5 s and the ring attention **59 %**; at 15 s attention grows
  quadratically and the linears linearly, so the linears matter most at the short clips the Turbo adapter targets.
- Everything else (`proj_in`, `audio_proj_in`, `context_embedder`, the two refiner blocks, `proj_out`,
  `audio_proj_out`, `norm_out.linear`, the float32 time embedders) runs once per request or is under 1 % of a forward.
- Per device per forward the blocks read 16 GB of bf16 weights (323 MB per block, 130 MB of it the adaLN projection).

## 3. Constraints found in this tree

1. **Fused addcmul epilogue needs weight format == residual format.** `to_out` folds `residual + gate·out` into the
   all-gather-matmul's epilogue, and every program factory asserts `ternary_a_data_format == in1_data_format`
   (`all_gather_minimal_matmul_async_program_factory.cpp`, `minimal_matmul_program_factory.cpp`). The residual stream
   is bf16 (the norms accept only bf16/fp32), so **`to_out`'s weight stays bf16 while its epilogue is fused**; its
   activation can still be cast. `FAST_H3_FP8_OUT_WEIGHT=1` un-fuses the epilogue (one extra `addcmul` pass) to
   quantize the weight too. `ff2`'s reduce-scatter applies the gate at the ring write, after the matmul, and has no
   such rule.
2. **Activation casts belong before the TP all-gather** (`ColParallelLinear.activation_dtype`): the gather's page size
   follows the gathered dtype, so a bfloat8_b cast halves the fabric bytes of `to_qkv`, `to_out` and `ff1`.
   `RowParallelLinear` (`ff2`) has no cast; its input is *produced* in bfloat8_b by `ff1`'s matmul instead
   (`ParallelFeedForward.ff1_output_dtype`), which costs no extra pass.
3. **Outputs that feed a norm or the residual stream are pinned back to bf16** (`pin_output_bf16`), and a
   `RowParallelLinear` fed a block-float input pins its matmul output to bf16 so the cross-device partial sums of the
   reduce-scatter are never accumulated in block float (the reduce-scatter reuses the matmul output's dtype).
4. **Ring SDPA wants one dtype across Q, K, V and the joint dummies** on H3's non-causal path
   (`ring_joint_sdpa_device_operation.cpp`), so an 8-bit SDPA input means all three cast after the fused QK-norm and
   RoPE (which keep full precision), as LTX does. It is a separate knob (`FAST_H3_FP8_SDPA`), off in every preset: the
   SDPA math itself is covered by the precision recipes of #59522 / #59523, and K/V in bfloat8_b cost 2 dB per
   forward on the 50-step base model (#58512).
5. **The adapter is fused in place in bf16** (`LoRAMixin._apply_delta`: `ttnn.add(weight, delta, output_tensor=weight)`)
   and touches exactly `to_qkv`, `to_out`, `ff1`, `ff2` of all 52 blocks. Merging a rank-256 delta into an already
   block-float weight is a no-op (the delta is below the block's quantization step; measured in #59443), so the
   8-bit conversion runs **after** the adapter is bound: an in-place typecast of the live device tensor. The bf16
   weight cache is unchanged and adapter-independent, the declared `Parameter.dtype` stays bf16 so a reload after
   eviction still passes the cache's dtype check, and the conversion is re-applied (idempotently) after every load.
6. **Fidelity is set at the call site.** The block and the attention pass their own compute config into every linear
   call, overriding the linear's own; per-role configs (`qkv_compute_kernel_config`, `out_compute_kernel_config`,
   `ff_compute_kernel_config`) are therefore set on the attention and the block, and `adaln_proj` keeps the original.
7. **The 4x8 preset runs the denoise eagerly** (buckets on, no denoise trace, audio traced), so the knob can change
   between generations in one process; programs recompile once per new dtype/fidelity combination. On the quad the
   denoise is traced per rung, so a weight typecast must precede capture (it does: the conversion runs in
   `_prepare_transformer`, before the warm-up captures).
8. `fp32_dest_acc_en` halves the DST tile budget (subblocks capped at 2x2) and every swept H3 blocking assumes it on;
   turning it off is a separate knob rather than part of a preset, and a measured one.

## 4. Prior 8-bit art in this repo

| where | what | result |
|---|---|---|
| LTX `models/tt_dit/models/transformers/ltx/quant_config.py` | role profile: all projections bfloat8_b weights + bfloat8_b activations + LoFi, `to_out` weight carved out to bf16, gates pinned, SDPA inputs bfloat8_b at HiFi2 math, one knob for the activation casts | shipped 1080p tier, VBench-gated; bfloat4_b activations "visibly destroy quality" |
| Wan `models/tt_dit/pipelines/wan/quant_config.py` | per-linear config applied in place after load (weight typecast + compute configs); `all_bf8_lofi` keeps `self_attn_out` bf16 for the fused addcmul | shipped; its `activation_dtype` field is never applied to the linears (only the SDPA inputs) |
| H3 #58677 (open) | weight-only bfloat8_b typecast after load on `qkv,ff1`, HiFi2 unchanged | 15 s, 50 steps: 4.175 -> 4.071 s/step (-2.5 %); per forward vs the fp32 CPU reference 45.52 -> 45.34 dB video, 36.76 -> 36.26 dB audio; K/V bfloat8_b (43.6 dB) and `ff2` bfloat8_b (44.87 dB, no speed gain) rejected |
| H3 #59443 (draft, single p150) | LTX-style profile built at construction with a quant-tagged cache | one block: no fp32 dest acc -2.5 %, LoFi -12.5 %; full-depth LoFi video PCC 0.9892; LoRA merge into a bfloat8_b weight found to be a no-op |
| SDPA #59209 / #59523 (Colman) | precision recipes for every DiT attention; FAST = LoFi with bfloat8_b K/V | 5 s t2va 840 -> 825 ms/step; 15 s 4131 -> 3310 ms/step |

## 5. External approaches (video / image DiT denoisers)

Survey of what ships elsewhere, read for the layer policy and the failure modes rather than the format (GPU FP8 is
per-tensor- or per-channel-scaled E4M3; the closest analogue of bfloat8_b with published numbers is MXINT8 /
block-INT8, which the MX paper finds lossless for direct-cast inference where MXFP8-E4M3 loses 0.5-2 points).

| implementation | 8-bit where | kept in high precision | W-only / W+A | scale | quality evidence | speed |
|---|---|---|---|---|---|---|
| HunyuanVideo official (`fp8_optimization.py`) | linears inside the blocks (incl. in-block modulation) | img/txt/time/vector/guidance embedders, final layer, norms | W-only, dequant to bf16 | per-tensor absmax | none stated | ~10 GB saved |
| Kijai wrappers (Hunyuan / Wan / CogVideoX / Mochi) | block linears | norm, bias, time_in, patch/text embedding, modulation (+ MLP in Hunyuan "fast" mode; CogVideoX-1.5 fast mode must keep `ff` or it NaNs) | W-only; optional "fast" W+A with no activation scale (clamp ±448) | per-tensor | "fp8_fast seems to cause huge quality degradation" (workflow note) | — |
| Wan 2.1 report | all GEMMs in the DiT block | — | W+A | per-tensor W, per-token A | "minimal performance loss"; FA3 native FP8 attention "significant quality degradation" | 2x GEMM, 1.13x DiT |
| LightX2V (Wan, HV1.5, Qwen, H3 profile) | all block linears | non-linear layers (fp32) | W+A | per-channel W, per-token dynamic A (SmoothQuant recommended) | "minimal degradation", no numbers; FP16-accumulate profile overflow warning | ~50 % VRAM |
| NVIDIA ModelOpt diffusion recipes (SDXL, SD3, FLUX, LTX, Wan2.2, Qwen) | linears (+ optional MHA) | per-model filters: time/text/patch embedders, `proj_out`, `norm_out`, adaLN / modulation, and for LTX / Wan / Qwen the first and last 2-3 blocks | W+A static | per-tensor (percentile calibration exists because early-step activations differ) | SDXL FP8 1.95x vs INT8 1.72x (images only) | 1.95x |
| torchao fp8dq / fp8dqrow (FLUX, CogVideoX) | all linears | — | W+A dynamic | per-tensor / per-row | "very close to bf16"; FP8 weight-only is *slower* than bf16 | FLUX -30 %, CogVideoX -14 % |
| FLUX.2-Klein-4B fp8 card | 100 of 109 linears (attn + FF) | embedders, time/guidance, adaLN, `norm_out`, `proj_out`, norms | W-only | — | PSNR 25.8 dB, SSIM 0.92, LPIPS 0.06 vs bf16; CLIP / PickScore ±0.02 | 12.1 -> 9.2 s |
| SGLang / vLLM-Omni online FP8 (Qwen-Image, HV1.5, H3) | DiT linears | H3: fp32 patch, timestep and output projections; HV1.5 attention off ("degrades under FP8"); Qwen keeps `img_mlp` | W+A dynamic | per-tensor | Qwen-Image-2.1 ≈ 33 dB vs bf16 | 7.34 -> 6.06 s |
| SGLang ConvRot INT8 for H3 | 209 linears | adaLN projections (group size), heads | W+A | per-row + Hadamard group 256 | video PSNR 23.4 vs 23.8 dB, SSIM 0.80 vs 0.81, audio 0.96 vs 0.97 (two backends vs bf16) | 1.07x (1 GPU) / 1.15x (4 GPU) |
| Community H3 fp8 cards (SabiQG, abhishekchohan, endman100, unsloth, ModelOpt Mixed9) | block attention + FF linears (200-400 tensors) | time embedders, `proj_in` / `audio_proj_in`, `context_embedder`, `token_refiner`, `norm_out`, `proj_out` / `audio_proj_out`, norms; Mixed9 also keeps `to_out` in blocks 29-48 and some FFNs | mostly W-only | per-tensor / per-channel | unsloth SSIM 0.88 vs bf16 at 20 steps; Mixed9 PSNR 24.8 dB at 10 steps, 20.3 dB at 50 | 66 -> 32-34 GB; 4 % faster |
| SVDQuant / Nunchaku | attention + MLP projections | modulation activations 16-bit | W4A4 + rank-32 branch | per-group | FLUX W8A8: PSNR 27 dB, LPIPS 0.09 | 3x vs NF4 |
| SageAttention 2 / 2++ | QKᵀ INT8 (K mean-subtracted), PV FP8 | softmax, norms | attention only | per-block Q/K, per-channel V | HunyuanVideo VQA-t 75.9 -> 75.4 (8-bit); 4-bit QK 65.4 | 2.7-3.9x attention |
| TT LTX / Wan `all_bf8_lofi` | qkv / ff1 / ff2 weights + activations, SDPA inputs | `to_out` weight (kernel rule), SDPA math HiFi2, norms | W+A | 16-element block | LTX VBench-gated | LoFi 2x HiFi2 |

What everyone agrees on:

- **Always kept high precision:** time embedders and `time_proj`; patch / `x_embedder` / `proj_in`; the context /
  caption embedder; the output head(s) and `norm_out`; every norm and bias. The H3 checkpoint declares `proj_in`,
  `audio_proj_in`, the time embedder, `proj_out` and `audio_proj_out` float32; this port runs the time embedder in
  float32 and the others in bf16, all outside the 8-bit mode.
- **Usually kept:** adaLN / modulation projections (ModelOpt FLUX / LTX / Qwen filters, Kijai Wan and Hunyuan-fast,
  Nunchaku's 16-bit modulation activations, every H3 community card). musubi-tuner saw rendered text and digits break
  when modulation was per-channel e4m3; OrbitQuant and SemanticDialect call the modulation linear the most sensitive
  layer because its input is a single compressed vector that every downstream layer reads. H3's adaLN projection is
  0.2 % of the FLOPs, so there is nothing to gain from quantizing it anyway.
- **Quantized everywhere:** the block's q/k/v (or fused qkv), out projection, and both FFN linears. The exceptions are
  kernel- or NaN-driven, not quality-driven (Kijai's fast mode keeps MLPs; the TT `to_out` carve-out is the fused
  epilogue's tile-format rule).
- **Attention core:** QKᵀ is the sensitive matmul (TensorRT-LLM: "video quality is generally more sensitive to BMM1
  accuracy than BMM2"); 8-bit Q/K needs smoothing (SageAttention) and native FP8 FlashAttention-3 degrades video
  (Wan report). Kept out of the presets here.
- **First / last 2-3 blocks** are excluded by ModelOpt's LTX / Wan / Qwen filters and some community builds, and by
  none of the "all layers" FP8 recipes; no published DiT ablation justifies it. #59443 found pinning H3's first and
  last block to bf16 made its LoFi PCC *worse*. Decided by measurement here (`FAST_H3_FP8_LINEARS` can restrict
  roles; a block range knob is not provided until a measurement asks for it).
- **Activations are the fragile operand.** Weights tolerate 8 bits and even 4 (LTX: "bf4 weights alone hold up");
  activation outliers are channel-wise, prompt-independent and timestep-varying (Q-DiT, PTQ4DiT, ViDiT-Q), the FFN's
  post-activation input is the worst (TaQ-DiT), and static per-tensor calibration fails at early steps (NVIDIA
  percentile quant). bfloat8_b's per-16-value exponent is effectively dynamic scaling, so the failure mode maps to
  outlier flushing inside a 16-channel block rather than calibration drift; it must be measured per layer.
- **Measurement:** full-clip PSNR between a quantized and a reference run only tells identical from different once
  the trajectories diverge (22-24 dB after 50 steps for any bf16-level perturbation; community H3 FP8 cards report
  20-25 dB); the usable per-change metric is the per-matmul and per-forward error, and few-step schedules diverge
  less (H3 GPU FP8: 24.8 dB at 10 steps vs 20.3 dB at 50).

## 6. Chosen approach

- **Format:** bfloat8_b weights, optionally bfloat8_b activations, selectable fidelity; no change to the matmul
  kernels. True e4m3 is deferred until the matmul LLK streams 8-bit tiles.
- **Scope:** the four block linears `to_qkv`, `to_out`, `ff1`, `ff2` in all 50 blocks (41 % of a 5 s forward's
  FLOPs, essentially all of its weight bytes). Excluded on purpose, matching the consensus above and the model's own
  float32 choices: the adaLN projections and `norm_out.linear`, the float32 time embedders (and HyperFlow's endpoint
  embedder), `proj_in` / `audio_proj_in` / `context_embedder`, the token refiner, `proj_out` / `audio_proj_out`,
  norms, the residual stream and the SDPA math.
- **Presets** (`FAST_H3_FP8`): `w8` (weights, HiFi2: bytes only), `w8a8` (weights + activations, HiFi2: bytes
  including the all-gathers, still exact 8-bit math), `w8_lofi` (weights at LoFi, bf16 activations: the compute
  lever without quantizing activations), `w8a8_lofi` (the Wan / LTX tier). `1` means `w8a8` until the measurements
  below pick the shipped default. Knobs refine a preset: `FAST_H3_FP8_LINEARS` (subset of roles),
  `FAST_H3_FP8_ACTIVATIONS`, `FAST_H3_FP8_FIDELITY`, `FAST_H3_FP8_FP32_ACC`, `FAST_H3_FP8_SDPA` (cast Q/K/V),
  `FAST_H3_FP8_OUT_WEIGHT` (un-fuse `to_out`'s epilogue and quantize its weight). A typecast of `ff1`'s bf16
  output for `ff2` (instead of the matmul writing block float directly, which packs without the precise rounding
  path `ttnn.typecast` uses) was tried and dropped: the fused SwiGLU returns non-finite values when a block-float
  input is paired with a bf16 output override.
- **Mechanics:** `MiniMaxH3QuantConfig` / `apply_quant_config` in
  `models/tt_dit/models/transformers/minimax_h3/quant_config.py`; the pipeline reads the environment (or a
  `quant_config=` argument) at construction and applies it after every transformer load, and the Turbo pipeline after
  the adapter is bound. Weights are typecast in place (bf16 cache kept); `ColParallelLinear` inputs are cast before
  their all-gather; `ff1` writes bfloat8_b for `ff2`; outputs feeding norms or the residual are pinned to bf16; the
  attention and block carry per-role compute configs.
- **What the knob does not touch:** the quad's traced denoise is unchanged in structure (the typecast precedes
  capture); the weight cache name is unchanged; nothing changes when the flag is unset (every added attribute defaults
  to the previous behaviour).

## 7. Measurement plan

All on the 4x8 Galaxy at the HyperFlow working point (1344x768, 5 s, 8 forwards, seed 0, one prompt), the same
`default` run of the unmodified pipeline as the reference in every process.

1. **Noise floor.** Two default generations in separate processes: PSNR between them (expected infinite, bit-identical).
2. **Per-matmul error, teacher-forced, real weights and activations.** A default run captures the inputs of blocks
   0, 12, 25, 37 and 49 at the first and a middle forward, plus each block's device-layout (adapter-fused) weights. A
   standalone block harness reloads the block, runs it in bf16 while recording every linear call, then for each preset
   re-runs each linear on the *same* recorded input and the whole block on the block input, reporting relative L2
   error, SQNR, PCC and max-abs/std against the bf16 result, plus the per-call time. This separates the weight cast,
   the activation cast and the fidelity, per role, per depth, per timestep.
3. **Per-forward error.** Every forward's predicted video and audio velocity is dumped for the default and each
   preset; the first forward sees identical inputs, so its PSNR is the per-forward error of the whole model, and later
   forwards show how the trajectory drifts over the 8-step grid.
4. **Clip-level.** Final video and audio PSNR against the default run, plus the decoded mp4s for viewing.
5. **Timing.** Denoise seconds per forward and total pipeline time from the warm, measured generation (programs
   compiled in a quiet pass first, zero program-cache misses asserted), with the pipeline's own
   `denoise breakdown` line; three repeats where the delta is small.

## 8. Results

Filled in by the runs recorded in the PR that added the knob.
