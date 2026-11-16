# Wan2.2-Animate-14B (proof of concept)

## Introduction

[Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) does character animation
and character replacement. The inputs are a driving video (for example, a dance) and one image of a
character (for example, a cat). It has two modes:

- **animation**: the character from the image performs the driving video's body motion and facial
  expression. The output takes the image's background.
- **replacement**: the character replaces the person in the driving video. The video's background
  is kept, and an optional relighting LoRA adapts the character to the scene's lighting.

This POC runs the model in TT-DiT on a Galaxy (4x8). It reuses the Wan 2.2 transformer, VAE, text
encoder, parallel configs and solver, and adds only what Animate needs. The bring-up steps and the
results log are in [Wan2_2_Animate_RUNBOOK.md](Wan2_2_Animate_RUNBOOK.md).

Status: the code is written and the CPU checks pass. **Nothing has run on hardware yet.**

## What Animate adds over Wan 2.2

The list below was checked against three sources:

- the upstream model code: `wan/modules/animate/{model_animate,face_blocks,motion_encoder}.py`
  and `wan/animate.py` in [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2);
- the Diffusers port that tt_dit loads: `diffusers/models/transformers/transformer_wan_animate.py`
  and `pipelines/wan/pipeline_wan_animate.py` in diffusers 0.38.0, the version tt-metal pins;
- the weight index of `Wan-AI/Wan2.2-Animate-14B-Diffusers`.

The checkpoint holds 17.27B parameters, stored in fp32 (69 GB).

| Addition | Checkpoint keys | Params | Runs on | Notes |
|---|---|---|---|---|
| CLIP image context, cross-attention K/V in every block | `blocks.*.attn2.{add_k_proj,add_v_proj,norm_added_k}` | 2.10 B | device | Missing from the brief's list. Same mechanism as Wan **2.1** I2V; Wan 2.2 A14B dropped CLIP. The image tokens get their own SDPA with the block's Q, and that output is added to the text attention output before `to_out`. |
| CLIP image embedder | `condition_embedder.image_embedder.*` | 8 M | host | Turns the ViT-H/14 penultimate hidden states (257 x 1280) into a 257 x 5120 context. Runs once per call. |
| Face adapter (cross-attention) | `face_adapter.{0..7}.*` | 0.84 B | device | Runs after blocks 0, 5, …, 35. Latent frame *f* attends only to the 5 motion tokens of frame *f*. RMSNorm on Q and K is per head. Zero-initialized motion is used for the reference frame. |
| Face encoder | `face_encoder.*` | 17.8 M | device | Three causal Conv1d (kernel 3, replicate pad; strides 1, 2, 2) take 77 face frames down to 20 latent frames. That gives 4 heads plus 1 learned padding token, so 5 tokens per frame. |
| Face motion encoder (LIA) | `motion_encoder.*` | 22.7 M | host | A StyleGAN-style CNN on 512x512 face crops. Its 20-d latent passes through a QR "direction" basis and comes out 512-d per frame. |
| Pose latent patch embedding | `pose_patch_embedding.*` | 0.3 M | device | A Conv3d from 16 channels to 5120 on the VAE latents of the pose skeleton video. It is **added** after the patch embedding, to latent frames 1..T only (not the reference frame). |
| Relighting LoRA (replacement mode) | not in the Diffusers repo: `Wan2.2-Animate-14B/relighting_lora/` | 2.9 GB fp32 | host merge | PEFT, rank 128, alpha 128, 960 tensors. Targets q/k/v/o of self- and cross-attention, `k_img`, `v_img`, `ffn.0` and `ffn.2` in all 40 blocks. Merged into the weights on the host (W += BA). |
| Preprocessing | `process_checkpoint/` | | host | YOLOv10m detection plus **ViTPose-H wholebody** (ONNX; DWPose is not used) give the pose skeleton video and the 512x512 face crops. **SAM2** hiera-large gives the character mask (replacement mode only). Optional pose retargeting, and FLUX.1-Kontext for the enhanced variant. |

### Corrections to the starting hypothesis

- **No extra input channels.** `in_channels` is 36: 16 noise, 4 mask and 16 conditioning latent
  channels, exactly as in Wan 2.2 I2V. What changes is the time axis:
  - The latent gets one extra leading frame (T+1). It holds the reference-image latent, with mask 1.
  - The other T frames hold one of:
    - the previous segment's last 1 or 5 decoded frames (mask 1), followed by zeros;
    - in replacement mode, the background video, with mask = 1 − character mask.
  - Pose is not an input channel. It is added after the patch embedding.
- **The "face motion encoder" is two modules.** A per-frame CNN (`motion_encoder`) feeds a temporal
  conv stack (`face_encoder`).
- **The largest addition is missing from the list:** the CLIP image branch adds 2.1B parameters of
  image K/V across the 40 blocks.
- **Single transformer.** There is no MoE `transformer_2` and no boundary ratio.
- **Pose preprocessing uses ViTPose, not DWPose.**

### Inference recipe (upstream defaults)

- Segments of 77 frames, overlapping by 1 frame (or 5). The driving video is reflect-padded to a
  whole number of segments.
- 20 steps, flow shift 5. Diffusers uses UniPC; upstream uses DPM++.
- Guidance 1.0, so no CFG. If CFG is used, the unconditional pass uses face pixels of −1.
- Default prompt `视频中的人在做动作` ("the person in the video is moving").
- Per segment, the latent has 21 frames: 20 plus the reference. At 720p that is 75,600 tokens,
  the same as an 81-frame Wan 2.2 I2V clip.

## What was reused from tt-local-generator

The `patches/tt_dit/pipelines/wan/pipeline_wan_animate.py` file in
[tenstorrent/tt-local-generator](https://github.com/tenstorrent/tt-local-generator) is a stub:

- It loads the Animate checkpoint into the I2V transformer with `strict=False`.
- It drops every Animate-only key and ignores the driving video.
- Its own comments say the result has no motion conditioning. `docs/PERFORMANCE.md` lists it as
  BLOCKED, and `bin/start_animate.sh` now falls back to plain I2V.

What we reused, with credit:

- its inventory of Animate-only keys, which matches the checkpoint index;
- its findings that Animate has no `transformer_2`, so `boundary_ratio` must be None and
  `guidance_scale_2` must be None;
- its warning that the old I2V warmup path fails on this checkpoint.

We did not reuse the code, because it drops exactly the modules this POC implements. The animate
design spec under `docs/superpowers/specs/` is about gallery and UI integration, not the model.

## Implementation

```
models/tt_dit/
├── models/transformers/wan2_2/
│   ├── attention_wan.py          # + image_kv (CLIP K/V branch), per_head_qk_norm, masked-matmul fallback
│   └── transformer_wan.py        # + image_kv / added_kv_proj_dim pass-through (defaults unchanged)
└── experimental/
    ├── wan_animate/
    │   ├── reference_math.py      # torch-only reformulations, shared by device code, pipeline and CPU tests
    │   ├── transformer_wan_animate.py  # WanAnimateTransformer3DModel, WanAnimateFaceAdapter, WanAnimateFaceEncoder
    │   ├── video_io.py            # preprocessed-input loading, mp4 writing, official preprocess launcher
    │   └── run_animate.py         # CLI: image + driving video -> mp4
    ├── pipelines/pipeline_wan_animate.py
    └── tests/
        ├── test_wan_animate_cpu.py      # 21 CPU checks vs Diffusers (pass)
        ├── test_wan_animate_modules.py  # device-vs-reference PCC tests
        └── test_pipeline_wan_animate.py # end-to-end
```

### Reused unchanged

- `WanTransformerBlock` and the self-attention and text cross-attention paths of `WanAttention`
  (ring SDPA, fused norm and RoPE, fused AG-matmul).
- RoPE (for T+1 frames), `WanTimeTextImageEmbedding` for time and text, `norm_out`, `proj_out`.
- `cache_tail`, which serves as the output head.
- `TextEncoder` (UMT5 on device), `WanVAEDecoderAdapter` and `WanEncoder` (VAE on device).
- The `WanPipelineConfig` presets (mesh axes, topology, links, FSDP), the UniPC solver, and the
  `cache.load_model` weight cache.

### Extended (backward compatible; T2V and I2V behaviour unchanged)

`WanAttention(image_kv=True)` adds three things:

- a fused `to_added_kv` with the same per-device head interleaving as `to_kv`, and `norm_added_k`;
- an image SDPA that reuses the block's Q, whose output is added before `to_out`;
- `per_head_qk_norm` for the face adapter, which tiles the [head_dim] weight across heads.

`WanTransformerBlock` and `WanTransformer3DModel` pass these through.

### New on device

Each reformulation below is checked on CPU against Diffusers (`test_wan_animate_cpu.py`).

1. **Pose embedding, fused into the patch embedding.**
   - Append the pose latents (zero on frame 0) and a 0/1 indicator channel to the conditioning,
     giving 36 + 17 channels.
   - Concatenate the Conv3d weights. The pose bias goes on the indicator channel, so it applies to
     frames 1..T only.
   - The result is one `WanPatchEmbed` matmul with K = 212 instead of 144, at no extra launch cost.
2. **Image cross-attention.**
   - 257 CLIP tokens are zero-padded to 512, with an additive `-inf` mask on the padding.
   - Padding to 512 keeps the existing `k_chunk_size=256` program config.
   - The image context is replicated, like the text prompt. The mask is fractured on the
     sequence-parallel (SP) axis along query rows.
3. **Face adapter**, the temporally aligned cross-attention.
   - The per-frame grouping becomes a block-diagonal additive mask over the flattened motion
     tokens: 21 x 5 = 105 tokens, padded to 256.
   - The queries keep the transformer's SP layout, so frames that straddle device boundaries need
     no regrouping. The attention itself is the existing `WanAttention` cross path.
   - Pre-norms: a `DistributedLayerNorm` on the spatial tokens (TP-fractured D) and a plain
     `LayerNorm` on the replicated motion tokens.
4. **Face encoder.**
   - Each causal, strided Conv1d is `Linear(concat_k(P_k @ X))`, where P_k are 0/1 selection
     matrices that do the replicate padding, the shift and the stride.
   - Only matmul, layer_norm and silu are used; tile padding is inert.
   - It runs once per segment, with all weights replicated (36 MB).

### On the host (CPU is fine for a POC)

Each of these runs once per call or once per segment, never per step:

- CLIP ViT-H/14 image encoder (632 M);
- the image embedder MLP;
- the LIA face motion encoder on 77 face crops per segment;
- letterboxing, frame resizing and reflect padding;
- segment bookkeeping: masks and the conditioning layout (`reference_math.segment_conditioning`,
  checked against Diffusers' `prepare_*` methods);
- relighting-LoRA key mapping and merge;
- the official preprocessing (pose, face crops, SAM2 mask).

### Pipeline

`WanAnimatePipeline` builds its components from the reused pieces and runs Diffusers' segment
loop. Its denoising loop is **untraced and has no DBCache**:

- `WanPipeline`'s traced loop assumes two experts and the I2V input set.
- Animate needs 5 more per-segment inputs: image context, motion tokens, 2 masks and the
  17-channel pose conditioning.
- Moving the loop onto `Tracer` follows once correctness is established. Every per-step input is
  already a persistent device tensor, except the timestep.

The VAE encoder uses forward chunk 4, the reference schedule: 1 frame, then groups of 4. Chunk 4
divides 76 = 77 − 1, whereas the I2V table's chunk of 16 would drop 12 frames.

## Target mesh, memory and sharding

The target is a Blackhole Galaxy (4x8) with the Wan 2.2 preset: SP=8 on axis 1, TP=4 on axis 0,
no FSDP. A Wormhole Galaxy uses the same axes with FSDP on SP, which leaves about 1.1 GB of weights
per chip.

| Module | Sharding | bf16 per chip (BH, TP=4) |
|---|---|---|
| 40 Wan blocks (unchanged) | as Wan 2.2 | 7.0 GB |
| Image K/V (`to_added_kv`, 40x) | column-parallel on TP (heads), like `to_kv`; image context replicated | +1.05 GB |
| Face adapters (8x) | Q and O like `attn2`; K/V column-parallel from replicated motion tokens; SDPA local per device | +0.42 GB |
| Face encoder | replicated | +0.04 GB |
| Pose embedding (fused) | output TP-fractured, as before | ~0 |
| UMT5 text encoder, VAE | as Wan 2.2 | about 3 GB |
| Masks at 720p (face 256 + image 512 key columns) | SP-fractured rows, 9,450 per device | 14.5 MB |

That totals about 12 GB of 32 GB per chip. The activations match Wan 2.2 I2V at the same
resolution, because 21 latent frames is the same sequence.

**Compute per step**, relative to one Wan 2.2 expert step:

- about +3% for the face adapters: 8 x (Q and O projections), against 40 blocks of attention plus
  FFN;
- about +2% for the image branch: one 512-key SDPA per block, with Q reused.

**Per segment**, the default recipe runs 20 steps and no CFG: 20 transformer passes, against 80 for
Wan 2.2 I2V (40 steps x CFG). So the denoise should cost roughly a quarter of a Wan 2.2 I2V clip at
the same size, before the slowdown from running untraced. This is an estimate; the runbook's RESULTS
section records the measured numbers.

## Open risks

1. **SDPA with an additive mask where key length ≠ query chunk.** This has not been exercised in the
   Wan path. If it fails or gives a low PCC, set `WanAttention.cross_attn_impl = "matmul"` (QK^T +
   mask, softmax, @V). The face-adapter test runs both modes.
2. **Per-head `dit_fused_distributed_rmsnorm` without RoPE** in a cross-attention: used by
   MiniMax-H3 and Flux2, but new for Wan shapes. The face-adapter test isolates it.
3. **Tile padding in the face encoder.** Logical T=77 is padded to 96. Selection matrices that are
   zero outside the logical range are proven inert on CPU, but not yet on device.
4. **VAE encoder.** It encodes 77-frame pose, background and temporal videos with fallback conv3d
   blockings at forward chunk 4. Expect slowness, not wrong numbers; `WAN_ANIMATE_VAE_ENC_CHUNK`
   must divide 76.
5. **Untraced loop.** Host dispatch overhead is likely large at 720p. Run 480p first.
6. **Weights.** 69 GB fp32 transformer plus text encoder, CLIP and VAE: about 100 GB of download.
   About 70 GB of host RAM is needed for the first conversion; the tt_dit cache removes that on
   later runs.
7. **Upstream preprocessing hard-codes CUDA** for SAM2 (`sam_utils.py`, replacement mode). On a
   CPU-only Galaxy host it needs a small patch, given in the runbook, or a GPU box.
8. **Portrait inputs** (dance videos are often 9:16) give sizes that are not in the tuned VAE
   blocking tables. They should work through the fallback, but are untested.
9. **Reference image preprocessing** uses letterboxing (official `padding_resize`). Diffusers'
   `resize_mode="fill"` may differ by a resampling filter. This affects only the reference latent.
10. **Not checked by execution:** the official-repo modules against the Diffusers port. They were
    compared by reading the code; the Diffusers port is the reference for every test here.

## Follow-ups after a correct run

- Trace the denoising loop.
- Add DBCache. Animate's no-CFG, 20-step schedule leaves less to skip.
- Run the face motion encoder on device: conv2d stacks at 512².
- Overlap the host preprocessing of segment k+1 with denoising of segment k.
- Use bf8 weights for the image K/V and the adapters.
