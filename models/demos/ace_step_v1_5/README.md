# ACE-Step v1.5 (Torch ref + TTNN)

This folder provides:

- `torch_ref/`: PyTorch reference implementation
- `ttnn_impl/`: TTNN implementation with one-to-one module mapping
- `tests/`: per-module PCC validation (Torch vs TTNN)
- `run_prompt_to_wav.py`: end-to-end text-to-music demo (preprocessing + TTNN DiT + VAE decode)

## Quick start — prompt-to-WAV demo

Generate a WAV audio file from a text prompt using TTNN-accelerated inference:

```bash
cd /home/ubuntu/proj_sdk/tt-metal

python3 models/demos/ace_step_v1_5/run_prompt_to_wav.py \
  --prompt "Electronic dance track with deep bass, punchy kick drum, bright synth lead, energetic rhythm" \
  --variant acestep-v15-base \
  --lm_variant acestep-5Hz-lm-1.7B \
  --duration_sec 15 \
  --infer_steps 4 \
  --out /tmp/ttnn_wav.wav
```

On first run, any missing model checkpoints are automatically downloaded from HuggingFace
into `~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints/`.

### Prerequisites

Install the extra dependency (from the `python_env` virtualenv):

```bash
pip install torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

### CLI options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--prompt` | `str` | **(required)** | Text description of the music to generate. |
| `--ckpt_dir` | `str` | `~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints` | Root directory for model checkpoints. Missing variants are auto-downloaded from HuggingFace. |
| `--variant` | `str` | `acestep-v15-base` | DiT model variant. Choices: `acestep-v15-base`, `acestep-v15-sft`, `acestep-v15-turbo`. |
| `--lm_variant` | `str` | `acestep-5Hz-lm-1.7B` | 5 Hz Language Model variant. Choices: `acestep-5Hz-lm-0.6B`, `acestep-5Hz-lm-1.7B`, `acestep-5Hz-lm-4B`. |
| `--device_id` | `int` | `0` | TT device index. |
| `--duration_sec` | `float` | `10.0` | Duration of the generated audio in seconds. |
| `--infer_steps` | `int` | auto | Number of diffusion inference steps. Defaults to 8 for turbo, 50 for base/sft. |
| `--seed` | `int` | `0` | Random seed for reproducibility. |
| `--guidance_scale` | `float` | auto | Classifier-free guidance (CFG) strength. Defaults to 7.0 for base/sft, 1.0 for turbo. Set to 1 to disable CFG. |
| `--shift` | `float` | `1.0` | Timestep shift factor for the diffusion schedule. |
| `--timesteps` | `str` | `None` | Comma-separated custom timestep schedule (overrides `--infer_steps`). |
| `--cfg_interval_start` | `float` | `0.0` | Timestep fraction where CFG begins (0.0 = start). |
| `--cfg_interval_end` | `float` | `1.0` | Timestep fraction where CFG ends (1.0 = end). |
| `--use_adg` | flag | auto | Use ADG guidance on host after TTNN forward. Defaults to on for base, off for turbo. Use `--no-use_adg` to force off. |
| `--out` | `str` | `ttnn_out.wav` | Output WAV file path. |
| `--fast-preprocess` | flag | off | Skip 5 Hz LM + AceStepHandler; use lightweight Qwen-only preprocessing. Auto-selected if torchaudio is not installed. |
| `--use-official-lm` | flag | off | Run the full official `generate_music` path (LLM + handlers, CPU only). Does not use TTNN; useful for A/B comparison. |
| `--ace-step-repo-root` | `str` | auto | Path to the ACE-Step-1.5 repo (contains `acestep/`). Auto-detected from well-known locations or `ACE_STEP_REPO_ROOT` env var. |
| `--no-ttnn-strict` | flag | off | Do not set `throw_exception_on_fallback` (may hide silent TTNN fallbacks to PyTorch). |

### DiT model variants

| Variant | CFG | Steps | Quality | Diversity | Fine-Tunability |
|---------|-----|-------|---------|-----------|-----------------|
| `acestep-v15-base` | Yes | 50 | Medium | High | Easy |
| `acestep-v15-sft` | Yes | 50 | High | Medium | Easy |
| `acestep-v15-turbo` | No | 8 | Very High | Medium | Medium |

### LM variants

| Variant | Parameters | Composition | Melody Copy |
|---------|-----------|-------------|-------------|
| `acestep-5Hz-lm-0.6B` | 0.6B | Medium | Weak |
| `acestep-5Hz-lm-1.7B` | 1.7B | Medium | Medium |
| `acestep-5Hz-lm-4B` | 4B | Strong | Strong |

### Examples

**Base model with 1.7B LM (balanced quality/speed):**

```bash
python3 models/demos/ace_step_v1_5/run_prompt_to_wav.py \
  --prompt "Acoustic guitar ballad with soft vocals and ambient strings" \
  --variant acestep-v15-base \
  --lm_variant acestep-5Hz-lm-1.7B \
  --duration_sec 30 \
  --seed 42 \
  --out /tmp/ballad.wav
```

**Turbo model for fast generation (fewer steps, no CFG):**

```bash
python3 models/demos/ace_step_v1_5/run_prompt_to_wav.py \
  --prompt "Lo-fi hip hop beat with vinyl crackle and mellow piano" \
  --variant acestep-v15-turbo \
  --lm_variant acestep-5Hz-lm-0.6B \
  --duration_sec 15 \
  --out /tmp/lofi.wav
```

**Base model with 4B LM (highest quality, slower):**

```bash
python3 models/demos/ace_step_v1_5/run_prompt_to_wav.py \
  --prompt "Orchestral film score with dramatic brass and timpani" \
  --variant acestep-v15-base \
  --lm_variant acestep-5Hz-lm-4B \
  --duration_sec 20 \
  --guidance_scale 10.0 \
  --infer_steps 50 \
  --out /tmp/orchestral.wav
```

**Using a custom checkpoint directory:**

```bash
python3 models/demos/ace_step_v1_5/run_prompt_to_wav.py \
  --prompt "Jazz piano trio with upright bass and brushed drums" \
  --ckpt_dir /path/to/my/checkpoints \
  --variant acestep-v15-base \
  --lm_variant acestep-5Hz-lm-1.7B \
  --out /tmp/jazz.wav
```

---

## Mandatory constraints (enforced by design)

- **TTNN device purity**: TTNN modules must not call PyTorch ops inside their `forward()`; the only allowed transfers are:
  - Host → device at the start of the run (inputs + weights)
  - Device → host at the end (final outputs for PCC comparison)
- **One-to-one mapping**: every Torch module has a TTNN equivalent.

## Attention modes (`attention_impl`)

`AceConfig.attention_impl` / `AceConfigTTNN.attention_impl`:

| Value | Torch module | TTNN module |
|-------|----------------|-------------|
| `"explicit"` | `MultiHeadSelfAttention` | `MultiHeadSelfAttentionTTNN` (matmul + softmax + causal mask) |
| `"sdpa"` | `MultiHeadSelfAttentionSDPA` (`F.scaled_dot_product_attention`) | `MultiHeadSelfAttentionSDPATTNN` (`ttnn.transformer.scaled_dot_product_attention`) |

Use the same string on both configs so `TransformerBlock` / `TransformerBlockTTNN` stay aligned.

TTNN SDPA rejects TILE tensors whose **logical** head dimension is smaller than the tile padding on that axis (e.g. head_dim 16 inside 32-wide tiles). The TTNN module zero-pads Q/K/V along head_dim to the next multiple of **32** before SDPA and slices the output back to the real `d_head`, with **`scale = 1/sqrt(d_head)`** so scaling still matches PyTorch.

## Layout

```
ace_step_v1_5/
  torch_ref/
  ttnn_impl/
  tests/
  run_prompt_to_wav.py
```

## Running tests

From repo/workspace root:

```bash
python -m pytest models/demos/ace_step_v1_5/tests \
  --confcutdir=models/demos/ace_step_v1_5/tests -q
```

If you have TT hardware/runtime, set:

```bash
export MESH_DEVICE=N150   # or N300 / T3K
```

## Notes

- PCC threshold in tests is set to `>= -0.9` per request (very lenient). You can tighten it later.
- The `mesh_device` pytest fixture is **session-scoped** (one `open_mesh_device` per process). Opening and closing a mesh around every test exhausts Metal context IDs and can trigger invalid `context_id` / teardown crashes on distributed meshes.

## TTNN demo (`ttnn_impl/full_pipeline.py`) — weights default to **Base**

The TTNN entrypoint defaults to Hugging Face **`ACE-Step/acestep-v15-base`** when you omit a local checkpoint:

```bash
cd /home/ubuntu/proj_sdk/tt-metal
python3 models/demos/ace_step_v1_5/ttnn_impl/full_pipeline.py \
  --out-npy /tmp/ace_features.npy
```

Override weights explicitly:

```bash
python3 models/demos/ace_step_v1_5/ttnn_impl/full_pipeline.py \
  --checkpoint-safetensors /path/to/model.safetensors \
  --out-npy /tmp/ace_features.npy
```

Turbo lives under the umbrella repo ``ACE-Step/Ace-Step1.5``; pick it with ``--hf-subfolder``:

```bash
python3 models/demos/ace_step_v1_5/ttnn_impl/full_pipeline.py \
  --hf-repo-id ACE-Step/Ace-Step1.5 \
  --hf-subfolder acestep-v15-turbo \
  --out-npy /tmp/out_turbo.npy
```

### HF parity note (`ttnn_impl/full_pipeline.py` / `ttnn_impl/dit_decoder_core.py`)

This TTNN path implements the **DiT decoder stack** (patch embed → conditioned transformer layers → output head) but is **not** a guaranteed byte-for-byte match to HF ``modeling_acestep_v15_*.py``. Documented gaps include **RoPE**, **per-layer sliding-window attention masks**, and **runtime vs lookup timestep embeddings**. See the docstring on ``AceStepV15TTNNPipeline``.

## Torch demo (HF weights + deterministic output signature)

This demo downloads an ACE-Step 1.5 checkpoint from Hugging Face, extracts the **DiT output head**
weights (`norm_out`, `scale_shift_table`, `proj_out`), runs a small forward pass, and prints:

- the **full snapshot path** on disk
- the inferred **state_dict prefix** for the output head inside the checkpoint
- output **shape** and a small numeric signature (`mean/std/first8`)

```bash
cd /home/ubuntu/proj_sdk/tt-metal
python3 -m models.demos.ace_step_v1_5.torch_ref.hf_output_head_demo \
  --repo-id "ACE-Step/Ace-Step1.5" \
  --subfolder "acestep-v15-turbo" \
  --seed 0 --batch 1 --original-seq-len 257 --noise-std 1.0
```
