# ACE-Step v1.5

## Platforms

Blackhole (BH_QB — 2×2 mesh)

## About ACE-Step

[ACE-Step](https://github.com/ace-step/ACE-Step-1.5) is an open-source music foundation model that generates stereo audio from text (and optional lyrics). Version 1.5 targets commercial-grade quality with fast inference: variable-length output at 48 kHz, multilingual prompts, and optional editing workflows (cover, repaint, vocal-to-BGM). Checkpoints are published on Hugging Face as [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5).

The pipeline has three main stages:

1. **5 Hz language model** — Turns the user prompt into structured conditioning (audio codes, captions, metadata) that guides generation.
2. **Diffusion Transformer (DiT)** — Denoises latents in the VAE space using flow matching.
3. **VAE decode** — Maps latents to a waveform (`AutoencoderOobleck`).

This TT-Metal demo accelerates the DiT sampler and much of the preprocessing stack on Tenstorrent devices via TTNN, while keeping host-side orchestration aligned with the official ACE-Step inference path.

### Currently supported

| Component | Supported today | Notes |
|-----------|-----------------|-------|
| DiT (`--variant`) | **`acestep-v15-base`**, **`acestep-v15-turbo`** | `acestep-v15-sft` is not validated on the TTNN e2e path yet. |
| 5 Hz LM (`--lm_variant`) | **`acestep-5Hz-lm-1.7B`** (default) | Used for official-style preprocessing and conditioning. |

### Qwen3 embedding (TTNN)

Caption / text conditioning uses a **local copy** of the TTNN Qwen3-Embedding stack, vendored under this demo until upstream lands in `tt-metal` main:

- `ttnn_impl/qwen3_embedding_encoder.py` — Qwen3 transformer blocks + `TtQwen3EmbeddingEncoder`
- `ttnn_impl/qwen3_embedding_ace_step.py` — ACE-Step wrapper (`AceStepQwen3Encoder`) over the HF `Qwen3-Embedding-0.6B` checkpoint

This code is replicated from the work in [tt-metal PR #42463](https://github.com/tenstorrent/tt-metal/pull/42463) ([`ign/qwen3-embedding-4b-optimizations`](https://github.com/tenstorrent/tt-metal/tree/ign/qwen3-embedding-4b-optimizations)). Once that PR is merged, this demo can switch to importing the shared implementation from `tt-metal` instead of maintaining duplicate modules here.

## This repo folder

This folder provides:

- `torch_ref/`: PyTorch reference implementation
- `ttnn_impl/`: TTNN implementation with one-to-one module mapping
- `tests/`: per-module PCC validation (Torch vs TTNN)
- `run_prompt_to_wav.py`: end-to-end text-to-music demo (preprocessing + TTNN DiT + VAE decode)

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- Installed [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- HuggingFace token: `huggingface-cli login` or `export HF_TOKEN=<token>` (checkpoints auto-download on first run to `~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints/`)

Install demo-specific Python packages from the `tt-metal` repo root:

```bash
pip install -r models/demos/ace_step_v1_5/requirements.txt
```


## Quick start — prompt-to-WAV demo

Generate a WAV audio file from a text prompt using TTNN-accelerated inference.

```bash
python models/demos/ace_step_v1_5/run_prompt_to_wav.py \
  --variant acestep-v15-turbo \
  --duration_sec 15 \
  --prompt "epic orchestral cinematic music for a movie trailer" \
  --infer_steps 8 \
  --guidance_scale 7 \
  --mesh-device BH_QB \
  --out /tmp/15_turbo_1.7B.wav
```

On first run, any missing model checkpoints are automatically downloaded from HuggingFace
into `~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints/`.

### Trace (default: on)

TTNN trace + 2CQ is **enabled by default**. To run in fully-eager single-CQ mode (e.g. for debugging or Tracy profiling):

```bash
python models/demos/ace_step_v1_5/run_prompt_to_wav.py ... --no-use-trace
```

### More examples

**Turbo model — fast generation (8 steps, CFG=7):**

```bash
python models/demos/ace_step_v1_5/run_prompt_to_wav.py \
  --variant acestep-v15-turbo \
  --lm_variant acestep-5Hz-lm-1.7B \
  --duration_sec 15 \
  --infer_steps 8 \
  --guidance_scale 7 \
  --mesh-device BH_QB \
  --out /tmp/lofi.wav
```

**Base model with 1.7B LM (balanced quality/speed):**

```bash
python models/demos/ace_step_v1_5/run_prompt_to_wav.py \
  --prompt "Acoustic guitar ballad with soft vocals and ambient strings" \
  --variant acestep-v15-base \
  --lm_variant acestep-5Hz-lm-1.7B \
  --duration_sec 30 \
  --seed 42 \
  --mesh-device BH_QB \
  --out /tmp/ballad.wav
```

**Base model with 4B LM (highest quality, slower):**

```bash
python models/demos/ace_step_v1_5/run_prompt_to_wav.py \
  --prompt "Orchestral film score with dramatic brass and timpani" \
  --variant acestep-v15-base \
  --lm_variant acestep-5Hz-lm-4B \
  --duration_sec 20 \
  --guidance_scale 10.0 \
  --infer_steps 50 \
  --mesh-device BH_QB \
  --out /tmp/orchestral.wav
```

### Weight caching (avoid reloading from disk)

The demo caches host-side safetensors in two places:

| Tier | Location | When it applies |
|------|----------|-----------------|
| **Disk** | `~/.cache/ace_step_v1_5/host_weights/` | Every CLI run after the first load of a checkpoint file |
| **RAM + TT device** | In-process registry | Default `run_prompt_to_wav.py` path or `serve_prompt_to_wav.py` |

Look for log lines:

```
[ace_step_v1_5] ⏳ LOAD   DiT-pipeline          path=.../model.safetensors
[ace_step_v1_5] ♻  REUSE  condition-encoder-weights [already loaded in memory]
```

**HTTP service (weights load once at startup, multiple HTTP requests):**

```bash
python3 models/demos/ace_step_v1_5/serve_prompt_to_wav.py --port 8765
```

Disable in-memory weight caching with `ACE_STEP_DISABLE_WEIGHT_CACHE=1`.

Logging defaults to **INFO** (hides TTNN per-tensor flatbuffer cache DEBUG spam).

### CLI options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--prompt` | `str` | **(required)** | Text description of the music to generate. |
| `--variant` | `str` | `acestep-v15-base` | DiT model variant. Choices: `acestep-v15-base`, `acestep-v15-sft`, `acestep-v15-turbo`. |
| `--lm_variant` | `str` | `acestep-5Hz-lm-1.7B` | 5 Hz Language Model variant. Choices: `acestep-5Hz-lm-0.6B`, `acestep-5Hz-lm-1.7B`, `acestep-5Hz-lm-4B`. |
| `--device_id` | `int` | `0` | TT device index. |
| `--mesh-device` | `str` | env | DiT/VAE mesh SKU (`P150`, `BH_QB`, `BH_LB`, …). Also `ACE_STEP_MESH_DEVICE` or `MESH_DEVICE`. |
| `--duration_sec` | `float` | `10.0` | Duration of the generated audio in seconds. |
| `--infer_steps` | `int` | auto | Number of diffusion inference steps. Defaults to 8 for turbo, 50 for base/sft. |
| `--seed` | `int` | `0` | Random seed for reproducibility. |
| `--guidance_scale` | `float` | auto | Classifier-free guidance (CFG) strength. Defaults to 7.0 for base/sft, 1.0 for turbo. Set to 1 to disable CFG. |
| `--out` | `str` | `ttnn_out.wav` | Output WAV file path. |
| `--use-official-lm` | flag | off | Run the full official `generate_music` path (LLM + handlers, CPU only). Does not use TTNN; useful for A/B comparison. |
| `--use-trace` | flag | on | TTNN trace + 2CQ for preprocess + DiT (see table below). Use `--no-use-trace` for fully eager. |

### Trace + 2 command queues (default `--use-trace`)

With trace on, the device opens with **`num_command_queues=2`** and a **128 MiB** trace region. The usual pattern is **CQ 1** for host→device input refresh and **CQ 0** for `begin_trace_capture` / `execute_trace`. The only switch that turns trace off everywhere is **`--no-use-trace`**.

| Stage | Trace + 2CQ? | Notes |
|-------|----------------|-------|
| **5 Hz LM init** (`handler_init`) | **No** | Weight upload / `create_tt_model` (one-time) |
| **5 Hz LM prefill** | **Yes** | ``QwenModelTtTransformers._prefill_traced`` (default with ``--use-trace``); trace cache keyed by ``(padded_len, seq_len)`` because ``get_last_token`` is baked into capture |
| **5 Hz LM decode** | **Yes** | Per-token ``_decode_traced`` / ``execute_trace`` on CQ0 (default with ``--use-trace``) |
| **5 Hz LM sampling / FSM** | **Partial** | Fused on-device penalties + top-k/top-p/sample (``apply_penalty_filter_sample``); constrained FSM on host — no CQ trace replay |
| **LM CFG logit combine** | **Yes** (when used) | ``cfg_linear_combination_bf16`` trace per ``(device, K, cfg_scale)`` when ``_ttnn_lm_use_trace``; demo often sets ``lm_cfg_scale=1`` |
| Qwen3 caption (`forward_traced`) | **Yes** | Handler / fast-preprocess text prefill |
| Lyric token embed (`embed_tokens_traced`) | **Yes** | During handler `preprocess_batch` |
| Audio-code detokenizer (`forward_traced`) | **Yes** | When LM emits audio codes |
| **Lyric transformer (8L)** | **Yes** | Inside ``forward_payload_traced`` capture (shape-keyed) |
| **Timbre transformer (4L)** | **Yes** | Same |
| Condition payload (`forward_payload_traced`) | **Yes** | Lyric + timbre + text + concat on CQ0; CQ1 input refresh |
| Context latents + chunk mask (`ctx_concat_traced`) | **Yes** | ``concat([src_latents, chunk_mask], dim=-1)`` |
| Cover / LM hints for `src_latents` | **Host only** | Host branch; chosen array staged into ctx trace inputs |
| Condition `enc_mask` | **Host + device** | NumPy for handler; ``upload_enc_mask_dev`` mirrors mask on device |
| DiT CFG batch concat (pre-loop) | **No (eager)** | One-shot ``enc``/``ctx``/null setup; traced replay caused audible noise |
| DiT ``compute_temb_tp`` | **No (eager)** | Precomputed per step; **streamed** into body trace via ``ttnn.copy`` on ``temb_buf``/``tp_buf`` |
| DiT cross-attention SDPA mask | **In DiT trace** | ``_E2EDenoiseTrace.mask_buf`` when built from ``enc_mask`` |
| DiT denoise body (`_E2EDenoiseTrace`) | **Yes** | ``patch_embed`` + DiT core + ``output_head`` on CQ0 |
| DiT pre-step row cast + CFG dup | **No (eager)** | Per-step before body trace replay |
| DiT CFG / ADG / APG + Euler | **No (eager)** | After ``release_trace_only`` each step (allocator + ADG ``sigma`` on host) |
| DiT warmup (first 2 Euler steps) | **No (eager)** | Prime program cache before first capture |
| VAE `decode_tiled` | **No (eager)** | All tiles eager; traced decode was not bit-accurate → noise (``decode_chunk_traced`` kept for tests only) |

#### Stages that stay eager or host-only (by design)

| Stage | Why |
|-------|-----|
| **5 Hz LM weight init** | Not a replay graph |
| **LM constrained FSM** | Python control flow + dynamic token masks |
| **LM CFG combine** | Skipped when ``lm_cfg_scale=1`` |
| **DiT CFG enc/ctx concat** | Traced prep not bit-accurate vs eager |
| **DiT full-step trace** | Opt-in only (``use_full_step``); default is body trace + eager post |
| **VAE decode** | Traced replay not bit-accurate vs eager |
| **DiT warmup (2 steps)** | Capture priming |
| **Cover hint selection** | Host branch |
| **VAE multi-tile overlap-add** | Trace replay PCC / audible noise |

---

## Mandatory constraints (enforced by design)

- **TTNN device purity**: TTNN modules must not call PyTorch ops inside their `forward()`; the only allowed transfers are:
  - Host → device at the start of the run (inputs + weights)
  - Device → host at the end (final outputs for PCC comparison)
- **One-to-one mapping**: every Torch module has a TTNN equivalent.

## DiT decoder PCC tests (demo path)

Production DiT code lives in `ttnn_impl/dit_decoder_core.py` (`TtAceStepDiTLayer`, `TtAceStepAttentionSDPA`, …). Layer PCC tests target that stack:

| Test file | Component |
|-----------|-----------|
| `test_pcc_adaln.py` | Scale-shift AdaLN modulation inside `TtAceStepDiTLayer` |
| `test_pcc_attention.py` | `TtAceStepAttentionSDPA` (self + cross, GQA) |
| `test_pcc_block.py` | `TtQwen3MLP` + full `TtAceStepDiTLayer` |
| `test_pcc_dit_decoder_core.py` | Full `TtAceStepDiTCore` |

`ttnn_impl/modules.py` is a legacy block library for small experiments; it is **not** used by `run_prompt_to_wav.py`.

TTNN SDPA in `dit_decoder_core` uses **tile-aligned `head_dim`** (often 32). Self-attention pads sequence length to a tile multiple and masks padded keys so softmax matches the torch reference.

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
export MESH_DEVICE=N150   # or N300 / T3K / BH_QB
```

### Blackhole QB (2×2 mesh)

Same algorithm as P150 (batch=2 CFG, trace, TTNN VAE). Preprocess (5 Hz LM + handler) runs on **host CPU**; DiT and VAE use the full **2×2** mesh.

```bash
python models/demos/ace_step_v1_5/run_prompt_to_wav.py \
  --mesh-device BH_QB \
  --variant acestep-v15-turbo \
  --lm_variant acestep-5Hz-lm-1.7B \
  --duration_sec 15 \
  --infer_steps 8 \
  --guidance_scale 7 \
  --out /tmp/ttnn_wav.wav
```

DiT init on 2×2 can take several minutes on first run (24 decoder layers). Progress lines (`DiT core: layer N/24`) indicate forward motion, not a hang.

## Notes

- PCC tests print the score and assert at **0.99** via ``assert_pcc_print`` in ``tests/_dit_decoder_pcc_common.py``.
- The `mesh_device` pytest fixture is **session-scoped** (one `open_mesh_device` per process). Opening and closing a mesh around every test exhausts Metal context IDs and can trigger invalid `context_id` / teardown crashes on distributed meshes.

## TTNN demo (`ttnn_impl/full_pipeline.py`) — weights default to **Base**

The TTNN entrypoint defaults to Hugging Face **`ACE-Step/acestep-v15-base`** when you omit a local checkpoint:

```bash
python models/demos/ace_step_v1_5/ttnn_impl/full_pipeline.py \
  --out-npy /tmp/ace_features.npy
```

Override weights explicitly:

```bash
python models/demos/ace_step_v1_5/ttnn_impl/full_pipeline.py \
  --checkpoint-safetensors /path/to/model.safetensors \
  --out-npy /tmp/ace_features.npy
```

Turbo lives under the umbrella repo ``ACE-Step/Ace-Step1.5``; pick it with ``--hf-subfolder``:

```bash
python models/demos/ace_step_v1_5/ttnn_impl/full_pipeline.py \
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
python -m models.demos.ace_step_v1_5.torch_ref.hf_output_head_demo \
  --repo-id "ACE-Step/Ace-Step1.5" \
  --subfolder "acestep-v15-turbo" \
  --seed 0 --batch 1 --original-seq-len 257 --noise-std 1.0
```

Use **`--no-use-trace`** for a fully eager single-CQ pipeline (debugging or Tracy device profiling).
