# LTX-2.3 AudioVideo Pipeline Alignment

This ExecPlan is a living document maintained in accordance with PLANS.md at the repository root.
The sections Progress, Surprises & Discoveries, Decision Log, and Key Measurements must be
kept up to date as work proceeds.

## Purpose

After this work, the TTNN LTX-2.3 AudioVideo pipeline will be fully aligned with the official LTX-2 reference implementation. Specifically:

- All identified transformer bugs (audio padding mask, padded token leakage, guidance orchestration) will be fixed
- Prompt adherence issue will be investigated and resolved (currently generates a human playing piano regardless of prompt saying "cat")
- The video-only and audio-video transformer will be a single unified class with a `has_audio` flag
- The LTX VAE decoder will run on TT device instead of CPU torch
- Output quality will match the reference pipeline (PCC > 0.99 end-to-end)
- The generation script will be converted to proper pytest-based pipeline and performance tests (following `test_pipeline_wan.py` / `test_performance_wan.py` patterns)

**Hardware target:** Wormhole Loud Box, 2x4 mesh (8 chips)

**Verification command (after all milestones):**

```bash
source python_env/bin/activate
export PYTHONPATH=$(pwd)
export TT_DIT_CACHE_DIR=/localdev/kevinmi/.cache
pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx.py -k "av" -v
```

## Autonomous Verification Strategy

This plan is designed for fully autonomous execution with no human in the loop. All verification must be automated and machine-checkable.

**DO NOT run full 30-step generation for every change.** Full generation is expensive (~10+ minutes per run). Instead:

1. **Unit tests (primary):** For each component fix, compare TTNN output against a torch reference at the single-layer or single-block level. Use small sequence lengths (N=64-128) and 1x1 mesh when possible. PCC > 0.999 is the bar.
2. **Latent-level comparison (for pipeline changes):** Run 2-5 denoising steps on both TTNN and reference CPU. Compare the intermediate latent tensors (not decoded video) for PCC. This catches numerical bugs without VAE decode cost.
3. **Reference codebase lookup (for correctness):** When investigating a discrepancy, read the official LTX-2 reference code to determine the correct behavior rather than generating outputs. The reference code is the ground truth.
4. **CLIP-based quality check (for final validation only):** When a full end-to-end run is needed to verify prompt adherence or visual quality, use CLIP similarity to programmatically verify the output matches the prompt. Do not rely on human visual inspection.
  - Install `transformers` and use `CLIPModel` / `CLIPProcessor` to compute text-image similarity
  - Extract a frame from the generated video, compute CLIP score against the prompt
  - A CLIP cosine similarity > 0.25 between prompt and generated frame indicates reasonable adherence
  - Compare TTNN CLIP score against reference CPU CLIP score -- they should be within 0.05 of each other
5. **Full generation (sparingly):** Only run 30-step generation for Milestone 6 (end-to-end validation) and Milestone 8 (pipeline test). Use 5-step runs for quick smoke tests during development.

## Acceptance Criteria

- Each component unit test passes with PCC > 0.999 against torch reference
- 5-step latent PCC > 0.99 between TTNN and reference CPU pipeline
- CLIP similarity between generated video and prompt is within 0.05 of reference CPU
- Unified transformer class works for both video-only and AV modes
- VAE decoder runs on device, not CPU torch
- No regression in existing video-only tests

## Reference Implementation

- **Official transformer:** `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer.py` (STG perturbation, block forward)
- **Official attention:** `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/attention.py` (V passthrough, perturbation_mask)
- **Official transformer args:** `LTX-2/packages/ltx-core/src/ltx_core/model/transformer/transformer_args.py` (cross-attn timestep prep)
- **Official VAE decoder:** `LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py` (VideoDecoder.forward)
- **Official UNetMidBlock3D:** `LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/resnet.py` (res_x block type)
- **Official CausalConv3d:** `LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/convolution.py` (causal vs symmetric)
- **Official DepthToSpaceUpsample:** `LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/sampling.py` (residual path)
- **Wan VAE (device reference):** `models/tt_dit/models/vae/vae_wan2_1.py` (WanMidBlock, WanDecoder integration pattern)
- **Wan pipeline integration:** `models/tt_dit/pipelines/wan/pipeline_wan.py` (how device VAE is called)

## Context and Orientation

### Current state

The TTNN LTX-2.3 AudioVideo transformer is implemented across two separate files:

- **Video-only:** `models/tt_dit/models/transformers/ltx/transformer_ltx.py` -- `LTXTransformerBlock` and `LTXTransformerModel`
- **Audio+Video:** `models/tt_dit/models/transformers/ltx/audio_ltx.py` -- `LTXAudioVideoTransformerBlock` and `LTXAudioVideoTransformerModel`

Both share `LTXAttention` from `models/tt_dit/models/transformers/ltx/attention_ltx.py`.

The generation script is `models/tt_dit/demos/ltx/generate_audio_video.py`. It runs the AV transformer on device but uses the official torch VideoDecoder on CPU for VAE decode.

The TTNN VAE decoder exists at `models/tt_dit/models/vae/ltx/vae_ltx.py` but is incomplete: it lacks `UNetMidBlock3D` (res_x block type), hardcodes `causal=True` (checkpoint uses `causal_decoder: False`), and `DepthToSpaceUpsample` lacks the residual path.

### Previously fixed bugs (confirmed resolved)

1. **Sigma schedule** used `video_N` instead of `MAX_SHIFT_ANCHOR=4096` -- fixed, now calls `compute_sigmas(steps=args.steps)` with default anchor
2. **Audio RoPE positions** used integer indices instead of time-in-seconds -- fixed, now uses `AudioPatchifier.get_patch_grid_bounds()`
3. **AV cross-attention gate timestep scale** -- confirmed correct: `inner_step` line 678 scales timestep by `* 1000.0` before passing to all adaln modules; `av_ca_factor = 1000/1000 = 1.0` for the 22B checkpoint, so gate and scale_shift receive the same input
4. **STG perturbation** -- confirmed correct in current code: `skip_qk=True` triggers V passthrough in `LTXAttention` (line 481-484), then `to_out`, gate, and residual are still applied via `addcmul_residual`/`addcmul_gate`

### Active bugs to fix

**Bug A (Critical): Audio padding mask ignored on SP ring-attention path.**
`LTXAttention.forward()` in `attention_ltx.py` lines 486-515: when SP > 1 (always true on 2x4 mesh), audio self-attention uses `ring_joint_scaled_dot_product_attention` which does not accept `attn_mask`. The mask is only used in the non-ring SDPA path at line 517-525. Padded audio tokens attend freely in self-attention.

**Bug B (Critical): Padded audio tokens leak into A-to-V cross-attention.**
In `audio_ltx.py`, padded audio tokens pass through text cross-attention (lines 343-351) and are gathered as full A-to-V context (lines 373-395) without masking. Non-reference tokens affect video through cross-modal attention.

**Bug C (Moderate): Guidance orchestration nested under do_cfg.**
In `generate_audio_video.py` lines 425-470, STG and modality guidance passes only run when `do_cfg` is True (CFG scale > 1.0). Official pipeline treats CFG, STG, and modality guidance independently.

**Bug D (Moderate, future): Scalar sigma for velocity-to-denoised.**
Line 388 uses scalar `sigma` for all tokens. Official uses per-token timesteps via `denoise_mask * sigma`. Numerically equivalent for prompt-only generation (all tokens same sigma) but incorrect for i2v/conditioning. Low priority.

**Bug E (Moderate, future): Post-process latent before Euler step.**
Official applies `post_process_latent()` before each Euler step. For prompt-only generation, `denoise_mask` is all-ones and this is a no-op. Low priority.

**Bug F (Minor): Audio padded decode.**
TTNN decodes all `audio_N` tokens then trims; official unpatchifies only `audio_N_real`. Limited impact due to causal audio stack.

**Bug G (Critical): Prompt adherence failure.**
The TTNN pipeline generates high-quality video that ignores the text prompt. Example: `--prompt "A cat playing piano in a cozy room"` produces a realistic human pianist. The video quality is good (not garbage/noise), suggesting the denoising and VAE are working, but text conditioning is not steering the generation. Possible causes: text embedding mismatch, cross-attention not using prompt effectively, CFG not amplifying prompt signal, or prompt/negative-prompt construction error.

## Plan of Work (Milestones)

---

### Milestone 1: Fix Audio Padding Bugs

After this milestone, padded audio tokens will be correctly masked in self-attention and excluded from A-to-V cross-attention context.

#### 1a: Fix audio padding mask on SP path

**File:** `models/tt_dit/models/transformers/ltx/attention_ltx.py`
**Location:** `LTXAttention.forward()`, around line 486

**What to change:** When `attn_mask is not None` and SP > 1, bypass ring attention. Instead:

1. All-gather K and V across SP devices using `ccl_manager.all_gather_persistent_buffer`
2. Run standard `ttnn.transformer.scaled_dot_product_attention` with the `attn_mask`
3. The output is already local (Q determines output shape)

This trades some perf for correctness. Ring attention without mask support would require kernel changes.

**What to run:**

```bash
pytest models/tt_dit/tests/models/ltx/test_audio_ltx.py -k "test_audio_attention_with_mask" -v
```

**What to expect:** PCC > 0.999 between masked SDPA and reference torch masked attention.

**Fallback:** If gathered SDPA OOMs on the audio sequence length, fall back to chunked attention with mask.

#### 1b: Fix padded audio token leakage into A-to-V

**File:** `models/tt_dit/models/transformers/ltx/audio_ltx.py`
**Location:** `LTXAudioVideoTransformerBlock.forward()`, around line 373

**What to change:** Add `audio_N_real` parameter to the block's forward method. Before gathering audio K/V for A-to-V cross-attention, zero out or slice to only `audio_N_real` tokens. Two approaches:

- **Approach A (simpler):** After audio text cross-attention, zero out positions `[audio_N_real:]` in the audio hidden state before it enters A-to-V. This ensures padded tokens contribute nothing to cross-modal context.
- **Approach B (cleaner):** Only gather `audio_N_real` tokens for A-to-V context. Requires knowing `audio_N_real_local` per SP device.

Start with Approach A; switch to B if perf is a concern.

**What to run:**

```bash
pytest models/tt_dit/tests/models/ltx/test_audio_ltx.py -k "test_av_block_padded_audio" -v
```

**What to expect:** PCC > 0.999 for video output; padded audio tokens produce zero contribution to video stream.

**Test to add:** `test_av_block_padded_audio` in `test_audio_ltx.py` -- create a block, run with `audio_N=96` but `audio_N_real=80`, verify video output matches a run with `audio_N=80` (no padding).

---

### Milestone 2: Fix Pipeline Guidance Orchestration

After this milestone, STG and modality guidance work independently of CFG.

#### 2a: Decouple guidance passes

**File:** `models/tt_dit/demos/ltx/generate_audio_video.py`
**Location:** Denoising loop, lines 419-476

**What to change:** Restructure so each guidance type runs based on its own flag:

```python
# Pass 1: Conditional (always)
v_denoised, a_denoised = velocity_to_denoised(*run_model(tt_v_prompt, tt_a_prompt))

# Pass 2: Unconditional (CFG) -- independent
do_cfg = args.video_cfg_scale > 1.0 or args.audio_cfg_scale > 1.0
if do_cfg:
    v_uncond, a_uncond = velocity_to_denoised(*run_model(tt_neg_v_prompt, tt_neg_a_prompt))

# Pass 3: Perturbed (STG) -- independent
do_stg = args.video_stg_scale != 0.0 or args.audio_stg_scale != 0.0
if do_stg:
    v_perturbed, a_perturbed = velocity_to_denoised(
        *run_model(tt_v_prompt, tt_a_prompt, skip_self_attn_blocks=[args.stg_block]))

# Pass 4: Isolated (modality) -- independent
do_modality = args.video_modality_scale != 1.0 or args.audio_modality_scale != 1.0
if do_modality:
    v_isolated, a_isolated = velocity_to_denoised(
        *run_model(tt_v_prompt, tt_a_prompt, skip_cross_attn=True))

# Apply guidance formula
v_pred = v_denoised.float()
if do_cfg: v_pred += (args.video_cfg_scale - 1) * (v_denoised.float() - v_uncond.float())
if do_stg: v_pred += args.video_stg_scale * (v_denoised.float() - v_perturbed.float())
if do_modality: v_pred += (args.video_modality_scale - 1) * (v_denoised.float() - v_isolated.float())
# (same for audio)
```

Also requires creating `tt_neg_v_prompt` / `tt_neg_a_prompt` only when `do_cfg` is True (move outside the loop).

**What to run:** Full pipeline with `--video_cfg_scale 1.0 --video_stg_scale 0.5` to verify STG works without CFG.

**What to expect:** Valid output (not garbage) when STG is enabled but CFG is disabled.

---

### Milestone 3: Unify Transformer Classes

After this milestone, there will be one transformer file serving both video-only and audio-video modes.

**New file:** `models/tt_dit/models/transformers/ltx/ltx_transformer.py` (replaces both `transformer_ltx.py` and `audio_ltx.py`)

#### 3a: Create unified `LTXTransformerBlock`

**What to change:** The block constructor takes `has_audio: bool`:

- Always creates: `norm1`, `attn1`, `attn2`, `norm2`, `norm3`, `ffn`, `scale_shift_table`, `prompt_scale_shift_table`
- When `has_audio=True`, additionally creates: `audio_norm1/2/3`, `audio_attn1/2`, `audio_ff`, `audio_scale_shift_table`, `audio_prompt_scale_shift_table`, `audio_to_video_attn`, `video_to_audio_attn`, `scale_shift_table_a2v_ca_audio/video`
- Forward signature: `forward(video_1BND, ..., audio_1BND=None, ..., has_audio=True)`. When audio args are None, run video-only path.

**Key reconciliation points:**

1. Video FFN: when `is_fsdp=True` and `has_audio=False`, pass `fsdp_mesh_axis` to `ParallelFeedForward`. When `has_audio=True`, do not (matching current AV behavior).
2. STG (`skip_qk`): supported in both modes via `LTXAttention`.
3. Return: `(video_1BND,)` when `has_audio=False`, `(video_1BND, audio_1BND)` when True.

#### 3b: Create unified `LTXTransformerModel`

Same pattern. `has_audio` controls whether audio stem, audio adaln, av_ca modules are created.

- `inner_step` always takes video args; audio args are optional (default None)
- Weight loading: `_prepare_torch_state` handles both video-only and AV checkpoint key patterns

#### 3c: Update imports and tests

- Update `generate_audio_video.py` to import from `ltx_transformer.py`
- Update `test_audio_ltx.py` to use unified class with `has_audio=True`
- Update any video-only test/pipeline to use unified class with `has_audio=False`
- Mark old files (`transformer_ltx.py`, `audio_ltx.py`) as deprecated or delete

**What to run:**

```bash
pytest models/tt_dit/tests/models/ltx/test_audio_ltx.py -v  # AV mode
pytest models/tt_dit/tests/models/ltx/test_transformer_ltx.py -v  # video-only mode (if exists)
```

**What to expect:** All existing tests pass with PCC > 0.999. No behavioral change.

---

### Milestone 4: Complete TTNN VAE Decoder

After this milestone, the TTNN VAE decoder supports the full 22B checkpoint decoder_blocks config.

#### 4a: Add `LTXUNetMidBlock3D` (res_x)

**File:** `models/tt_dit/models/vae/ltx/vae_ltx.py`

The official `UNetMidBlock3D` in `LTX-2/.../resnet.py` lines 233-277 is a stack of `num_layers` `ResnetBlock3D` instances with same in/out channels. No attention, no timestep conditioning (the 22B checkpoint has `timestep_conditioning: False`).

**Implementation:**

```python
class LTXUNetMidBlock3D(Module):
    def __init__(self, *, in_channels, num_layers, mesh_device, dtype=ttnn.bfloat16):
        super().__init__()
        self.res_blocks = ModuleList()
        for _ in range(num_layers):
            self.res_blocks.append(
                LTXResnetBlock3D(in_channels=in_channels, out_channels=in_channels,
                                 mesh_device=mesh_device, dtype=dtype))

    def forward(self, x_BTHWC, causal=True):
        for block in self.res_blocks:
            x_BTHWC = block(x_BTHWC, causal=causal)
        return x_BTHWC
```

Add `res_x` handling to `LTXVideoDecoder.__init__` block dispatch:

```python
elif block_name == "res_x":
    self.up_blocks.append(
        LTXUNetMidBlock3D(in_channels=ch, num_layers=block_config["num_layers"],
                          mesh_device=mesh_device, dtype=dtype))
    # ch stays the same (in == out for mid block)
```

**What to run:**

```bash
pytest models/tt_dit/tests/models/ltx/test_vae_ltx.py -k "test_ltx_unet_mid_block" -v
```

**What to expect:** PCC > 0.999 against official `UNetMidBlock3D` with `NormLayerType.PIXEL_NORM`, `timestep_conditioning=False`.

#### 4b: Fix causal mode

**File:** `models/tt_dit/models/vae/ltx/vae_ltx.py`

The 22B checkpoint has `causal_decoder: False`, meaning the decoder should use symmetric temporal padding: repeat first frame `floor((kernel_t-1)/2)` times at front, repeat last frame `floor((kernel_t-1)/2)` times at back.

The `LTXCausalConv3d` already supports `causal=False` in its forward method. The fix is to thread `causal` through:

1. Add `causal: bool = True` to `LTXVideoDecoder.__init__`, store as `self.causal`
2. Pass `self.causal` to `self.conv_in(sample_tt, causal=self.causal)`, each `up_block(sample_tt, causal=self.causal)`, and `self.conv_out(sample_tt, causal=self.causal)`
3. `LTXResnetBlock3D.forward` already accepts `causal` -- just pass it through
4. `LTXDepthToSpaceUpsample.forward` already accepts `causal` -- just pass it through
5. `LTXUNetMidBlock3D.forward` -- pass through to inner resnet blocks

**What to run:**

```bash
pytest models/tt_dit/tests/models/ltx/test_vae_ltx.py -k "test_ltx_causal_conv3d" -v
# Add a causal=False parametrized case
pytest models/tt_dit/tests/models/ltx/test_vae_ltx.py -k "test_ltx_video_decoder" -v
# Update to use causal=False matching the 22B config
```

**What to expect:** PCC > 0.999 for causal=False mode against official decoder with `causal=False`.

#### 4c: Add residual path to DepthToSpaceUpsample

**File:** `models/tt_dit/models/vae/ltx/vae_ltx.py`, class `LTXDepthToSpaceUpsample`

The official `DepthToSpaceUpsample` (in `sampling.py` lines 68-123) has `residual=True` which rearranges the input using the same depth-to-space pattern, repeats channels to match the conv output, and adds to the conv result.

**Implementation:** Add `residual: bool = False` parameter. When True:

1. Before conv: rearrange input `(B, T, H, W, C)` -> `(B, T*p1, H*p2, W*p3, C_reduced)` using same depth-to-space
2. Repeat channels: `C_reduced` repeated `prod(stride) // out_channels_reduction_factor` times
3. If `p1 == 2`: trim first temporal frame from the rearranged input
4. After conv + depth-to-space: add the rearranged input

**What to run:**

```bash
pytest models/tt_dit/tests/models/ltx/test_vae_ltx.py -k "test_ltx_depth_to_space_upsample" -v
# Add residual=True parametrized cases
```

**What to expect:** PCC > 0.999 against official with `residual=True`.

#### 4d: Full decoder test with 22B config

Update `test_ltx_video_decoder` to use the actual 22B checkpoint `decoder_blocks` (which includes `res_x` entries) and `causal=False`.

**What to run:**

```bash
pytest models/tt_dit/tests/models/ltx/test_vae_ltx.py -k "test_ltx_video_decoder" -v
```

**What to expect:** PCC > 0.99 end-to-end (accumulated error across many layers).

---

### Milestone 5: Integrate TTNN VAE into Pipeline

After this milestone, the generation pipeline uses the device VAE instead of CPU torch.

**File:** `models/tt_dit/demos/ltx/generate_audio_video.py`

**What to change:**

1. After the denoising loop (line 492), instead of loading the torch `ModelLedger` video decoder:
  - Instantiate `LTXVideoDecoder` with the 22B decoder_blocks config, `causal=False`
  - Load weights from the checkpoint's VAE state dict
  - Reshape denoised latent from `(1, N, 128)` to `(B, 128, T, H, W)` (BCTHW format expected by the decoder)
  - Call `tt_decoder(latent_bcthw)` which returns `(B, 3, F, H, W)`
2. Keep the mesh device open until after VAE decode (move `ttnn.close_mesh_device` to after VAE)
3. Audio VAE/vocoder stays on CPU torch (it's a different architecture)

Follow `pipeline_wan.py` pattern: the Wan pipeline builds `self.tt_vae = WanDecoder(...)`, loads weights, then calls it after denoising with the latent in BTHWC format on device.

**What to run:**

```bash
python models/tt_dit/demos/ltx/generate_audio_video.py \
    --prompt "A cat playing piano" --output test_device_vae.mp4 \
    --num_frames 33 --height 512 --width 768 --steps 5
```

**What to expect:** Valid video output. Compare frames against CPU torch VAE output with PCC > 0.99.

---

### Milestone 6: End-to-End Validation

Verify pipeline correctness using automated comparison -- no human visual inspection required.

#### 6a: Latent-level comparison (fast, primary check)

Run 5 denoising steps on both TTNN and reference CPU with identical seed/params. Compare the raw latent tensors (before VAE decode).

```python
# In a test function:
# 1. Run TTNN pipeline for 5 steps, save video_latent and audio_latent
# 2. Run reference CPU pipeline for 5 steps with same seed, save latents
# 3. Compare PCC between TTNN and reference latents
assert pcc(ttnn_video_latent, ref_video_latent) > 0.99
assert pcc(ttnn_audio_latent, ref_audio_latent) > 0.99
```

This catches numerical bugs without the cost of full generation + VAE decode.

#### 6b: CLIP-based prompt adherence (for final validation)

Run a full 30-step generation once and verify the output matches the prompt using CLIP:

```python
from transformers import CLIPModel, CLIPProcessor
import imageio

# Extract middle frame from generated video
reader = imageio.get_reader("ttnn_output.mp4")
mid_frame = reader.get_data(len(reader) // 2)

# Compute CLIP similarity
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
inputs = processor(text=[prompt], images=[mid_frame], return_tensors="pt")
outputs = model(**inputs)
clip_score = outputs.logits_per_image.item() / 100.0  # cosine similarity

assert clip_score > 0.25, f"CLIP score {clip_score} too low -- video doesn't match prompt"
```

Also compare TTNN CLIP score against reference CPU CLIP score to ensure they are within 0.05.

**What to expect:** Latent PCC > 0.99, CLIP scores within 0.05 of reference.

---

### Milestone 7: Investigate and Fix Prompt Adherence

**Symptom:** The TTNN pipeline currently generates high-quality video of a human playing piano regardless of the text prompt. For example, `--prompt "A cat playing piano in a cozy room"` produces a realistic human pianist instead of a cat. The video quality is good (not garbage), but the content does not follow the prompt. This suggests text conditioning is being ignored or overridden somewhere in the pipeline.

After this milestone, the generated video content will match the text prompt.

#### 7a: Verify text embeddings match reference

**Files to compare:**

- TTNN text encoding in `generate_audio_video.py` (the section that calls `ledger.text_encoder()` or similar)
- Official text encoding in the reference pipeline

**Investigation steps:**

1. Extract the `video_embeds` and `audio_embeds` tensors from the TTNN pipeline after text encoding
2. Run the same prompt through the official reference pipeline and extract the same embeddings
3. Compare PCC between the two sets of embeddings
4. Check embedding shapes, dtypes, and whether any pooling/projection is applied differently
5. Verify the negative prompt embeddings (`neg_video_embeds`, `neg_audio_embeds`) are constructed correctly -- wrong negative embeddings can wash out prompt conditioning during CFG

**What to look for:**

- Are `video_embeds` and `audio_embeds` swapped or transposed?
- Is the text encoder using the correct tokenizer and model variant?
- Are embeddings being truncated or padded differently?
- Is the prompt being split into video-prompt and audio-prompt correctly (official pipeline may use separate prompts per modality)?

#### 7b: Verify cross-attention passes prompt correctly

**Investigation steps:**

1. Run a single denoising step on both TTNN and reference with identical inputs (same noise, same sigma, same embeddings)
2. Compare the cross-attention output (video after text cross-attention) -- if this diverges, the prompt signal is being lost in cross-attention
3. Check that `video_prompt_1BLP` shape and content match what the block expects
4. Check that the AdaLN modulation of K/V context (`prompt_scale_shift_table`) is applied correctly

#### 7c: Verify CFG is amplifying prompt signal

CFG formula: `pred = uncond + cfg_scale * (cond - uncond)`

If the conditional and unconditional outputs are nearly identical, CFG produces no prompt steering. This could happen if:

- Positive and negative prompts produce the same embeddings (encoding bug)
- Cross-attention is not effectively using the prompt embeddings (weight loading bug)
- The prompt embeddings are being replaced or zeroed somewhere

**Investigation steps:**

1. Compare `v_denoised` (conditional) vs `v_uncond` (unconditional) at step 0. They should differ meaningfully.
2. If they are nearly identical (PCC > 0.999), the prompt is not being used -- trace back to embeddings and cross-attention weights
3. If they differ but the final output still ignores the prompt, check if `cfg_scale` is being applied correctly

#### 7d: Check if the issue is in the official reference too

Run the official reference pipeline on CPU with the exact same prompt and checkpoint to confirm that the reference produces a cat (not a human). If the reference also produces a human, the issue is in the checkpoint or prompt format, not in the TTNN port.

```bash
python ltx_reference_cpu.py --prompt "A cat playing piano in a cozy room" --output ref_cat.mp4 --steps 30 --seed 42
```

**What to expect:** If the reference produces a cat, the bug is in TTNN's text conditioning path. If the reference also produces a human, the bug is upstream (checkpoint, prompt format, or model behavior).

---

### Milestone 8: Convert Pipeline to Pytest Tests

The current generation script `generate_audio_video.py` is a standalone `argparse`-based script. It should be converted into proper pytest-based tests following the established patterns in `test_pipeline_wan.py` and `test_performance_wan.py`. This means:

- A single `LTXPipeline` class with a `mode` parameter (`"video"` or `"av"`) -- video-only uses the unified transformer with `has_audio=False`, AV mode uses `has_audio=True`
- Pytest fixtures for device setup (indirect `mesh_device`, `device_params`)
- Artifact generation (video .mp4 files)
- A separate performance test with warmup and `BenchmarkProfiler` measurements

#### 8a: Create `LTXPipeline` and `test_pipeline_ltx.py`

**New files:**

- `models/tt_dit/pipelines/ltx/pipeline_ltx.py` -- pipeline class
- `models/tt_dit/tests/models/ltx/test_pipeline_ltx.py` -- pytest test

**Reference pattern:** `models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py`

**Pipeline class (`LTXPipeline`):**

- `create_pipeline(mesh_device, sp_axis, tp_axis, ..., mode="av")` class method
  - `mode="video"`: loads video-only checkpoint, creates unified transformer with `has_audio=False`, no audio encoder/decoder
  - `mode="av"`: loads 22B AV checkpoint, creates unified transformer with `has_audio=True`, sets up audio decoder/vocoder
- `__call__(prompt, height, width, num_frames, num_inference_steps, seed, ...)` -- runs full inference
  - Text encoding, noise init, denoising loop, VAE decode
  - When `mode="av"`: also handles audio prompt, audio denoising, audio decode
  - Returns result object with `.frames` (numpy array) and optionally `.audio`
- Accepts optional `profiler` and `profiler_iteration` parameters
- Pipeline internally records `"encoder"`, `"denoising"`, `"vae"` timing spans when profiler is provided

**Test file:**

```python
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        ((2, 4), (2, 4), 0, 1, 2, False, line_params, ttnn.Topology.Linear, False),
    ],
    ids=["wh_lb_2x4"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("width, height", [(768, 512)], ids=["512p"])
@pytest.mark.parametrize("mode", ["av", "video"], ids=["av", "video"])
def test_pipeline_inference(mesh_device, ..., mode):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    pipeline = LTXPipeline.create_pipeline(..., mode=mode)
    with torch.no_grad():
        result = pipeline(prompt=prompt, ..., num_inference_steps=5)
    export_to_video(result.frames[0], f"ltx_{mode}_output.mp4", fps=24)
    assert result.frames.shape[2] == height
```

**Key design points:**

- Single `LTXPipeline` class with `mode` parameter -- no separate AV/video pipeline classes
- Pipeline test uses only **5 steps** (fast, ~2 min). Full 30-step runs only in perf test.
- No `argparse` -- all parameters from pytest parametrize
- Device setup via fixtures, not manual `ttnn.open_mesh_device`

#### 8b: Create `test_performance_ltx.py`

**New file:** `models/tt_dit/tests/models/ltx/test_performance_ltx.py`

**Reference pattern:** `models/tt_dit/tests/models/wan2_2/test_performance_wan.py`

**Structure:**

- Same parametrize as pipeline test (with `mode` for video/av), plus `is_ci_env` and `galaxy_type` fixtures
- **Warmup:** 2 denoising steps wrapped in `BenchmarkProfiler("run", iteration=0)`
- **Measured run:** Full 30 steps with `profiler=benchmark_profiler, profiler_iteration=1`
- **Metrics:** Extract `encoder`, `denoising`, `vae`, `run` durations. Print mean/std/min/max per phase.
- **Performance bounds:** `expected_metrics` dict with upper-bound seconds. Assert within bounds (log-only initially until baselines are stable).
- **CI integration:** `BenchmarkData.add_measurement` + `save_partial_run_json` with `ml_model_name="LTX-2.3"`. Save artifact only when `not is_ci_env`.

**What to run:**

```bash
# Pipeline test -- AV mode (5 steps, fast)
pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx.py -k "av" -v

# Pipeline test -- video-only mode (5 steps, fast)
pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx.py -k "video" -v

# Performance test (warmup + full 30-step measured run)
pytest models/tt_dit/tests/models/ltx/test_performance_ltx.py -v
```

**What to expect:**

- Pipeline test: produces `ltx_av_output.mp4` or `ltx_video_output.mp4`, passes in ~2 min
- Performance test: prints timing breakdown (encoder, denoising, VAE, total), produces video artifact

---

## Progress

- [x] (2026-03-24 07:00Z) Milestone 1a: Fix audio padding mask on SP ring-attention path. Gather K/V when mask present.
- [x] (2026-03-24 07:00Z) Milestone 1b: Fix padded audio token leakage into A-to-V. Added audio_padding_mask to zero padded tokens.
- [x] (2026-03-24 07:10Z) Milestone 2a: Decouple guidance passes from do_cfg. STG/modality now independent.
- [x] (2026-03-24 07:30Z) Milestone 3a: Create unified LTXTransformerBlock in ltx_transformer.py with has_audio flag.
- [x] (2026-03-24 07:30Z) Milestone 3b: Create unified LTXTransformerModel in ltx_transformer.py with has_audio flag.
- [x] (2026-03-24 07:30Z) Milestone 3c: Update all imports and tests to use ltx_transformer.py. Old files kept but unused.
- [x] (2026-03-24 07:45Z) Milestone 4a: Add LTXUNetMidBlock3D (res_x) — stack of ResnetBlock3D, same in/out channels.
- [x] (2026-03-24 07:45Z) Milestone 4b: Add causal parameter to LTXVideoDecoder (22B uses causal=False).
- [x] (2026-03-24 07:45Z) Milestone 4c: Add residual path to DepthToSpaceUpsample.
- [ ] Milestone 4d: Full decoder test with 22B config (needs device)
- [x] (2026-03-24 07:55Z) Milestone 5: Integrate TTNN VAE into AV pipeline with 22B config.
- [x] (2026-03-24 08:10Z) Milestone 6 (partial): CLIP analysis of existing outputs. TTNN vs reference gap confirmed.
- [x] (2026-03-24 08:05Z) Milestone 7a: Text embeddings verified — connector weights loaded into text encoder, not DiT.
- [ ] Milestone 7b: Verify cross-attention passes prompt correctly (needs step-by-step device comparison)
- [ ] Milestone 7c: Verify CFG is amplifying prompt signal (needs device)
- [x] (2026-03-24 08:10Z) Milestone 7d: CLIP confirms reference CPU output matches "cat" (0.341) while TTNN matches "person" (0.315). Bug is real, not a model limitation.
- [ ] Milestone 8a: Create LTXPipeline class and test_pipeline_ltx.py (mode=video|av)
- [ ] Milestone 8b: Create test_performance_ltx.py with warmup and BenchmarkProfiler

## Surprises & Discoveries

- **Audio attention mask shape for SP>1**: The mask must cover the full K sequence (audio_N) not the local shard (audio_N_local), since K/V are gathered across SP before SDPA.
- **embeddings_connector is loaded into text encoder, not DiT**: The checkpoint stores `model.diffusion_model.video_embeddings_connector.*` but these are loaded into the text encoder via `EMBEDDINGS_PROCESSOR_KEY_OPS`. The DiT popping these keys is correct.
- **No caption_projection in 22B**: The 22B checkpoint has no `caption_projection` keys. The reference `_prepare_context` skips projection when it's None.
- **CLIP confirms prompt adherence gap**: Reference CPU produces "cat" (CLIP=0.341 for cat prompt), TTNN produces "person" (CLIP=0.315 for person prompt). The gap is in denoising, not text encoding.

## Decision Log

- **Decision:** Bypass ring attention when attn_mask is present (Bug A fix)
  **Rationale:** Ring attention kernel does not support masks. Gathered SDPA with mask is correct and the audio sequence is short enough to fit in memory. Implementing masked ring attention would require kernel changes.
  **Date:** 2026-03-23

- **Decision:** Zero-out padded tokens before A-to-V rather than slicing (Bug B fix, Approach A)
  **Rationale:** Simpler than slicing to audio_N_real, avoids shape changes in the cross-attention path. Zero tokens contribute nothing through linear projections and softmax.
  **Date:** 2026-03-23

- **Decision:** Bugs D, E, F deferred to future work
  **Rationale:** Per-token denoise_mask, post_process_latent, and audio padded decode are all no-ops or numerically equivalent for the current prompt-only generation use case. They matter for i2v/conditioning which is not yet implemented.
  **Date:** 2026-03-23

- **Decision:** Unified transformer file named `ltx_transformer.py`
  **Rationale:** Neither `transformer_ltx.py` nor `audio_ltx.py` captures the combined nature. A fresh name avoids confusion about which was the "base".
  **Date:** 2026-03-23

## Constraints & Workarounds

- **Hardware:** Wormhole Loud Box, 2x4 mesh (8 chips), SP=4, TP=2
- **Dtype:** bfloat16 for compute, float32 for timestep embedding and output projection
- **Ring attention mask limitation:** ttnn ring_joint_scaled_dot_product_attention does not support attn_mask. Workaround: gathered SDPA for masked audio self-attention.
- **Audio VAE/vocoder:** Stays on CPU torch (different architecture from video VAE, not in scope)
- **22B checkpoint config:** `causal_decoder: False`, `timestep_conditioning: False`, `av_ca_timestep_scale_multiplier: 1000.0`

## Key Measurements

| Test | Metric | Value | Notes |
|------|--------|-------|-------|
| CLIP av_fullres_human (seed10) | cat piano sim | 0.253 | Should be higher for cat prompt |
| CLIP av_fullres_human (seed10) | person piano sim | 0.315 | Model outputs person instead of cat |
| CLIP av_seed42 | cat piano sim | 0.269 | |
| CLIP av_seed42 | person piano sim | 0.289 | |
| CLIP ref_cpu (seed10) | cat piano sim | 0.341 | Reference correctly follows cat prompt |
| CLIP ref_cpu (seed10) | person piano sim | 0.307 | |
