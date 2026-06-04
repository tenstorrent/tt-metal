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

### Validated on BH_QB (2×2 mesh)

End-to-end prompt-to-WAV (`demo/run_prompt_to_wav.py`) has been tested on **Blackhole QB** for the full matrix below. Module-level PCC tests (see [PCC and module tests](#pcc-and-module-tests)) cover DiT, VAE, condition encoder, and 5 Hz LM stacks independently.

**DiT variants (`--variant`) — all tested:**

| Variant | CLI value | Default steps | Default CFG | Status |
|---------|-----------|---------------|-------------|--------|
| Base | `acestep-v15-base` | 50 | 7.0 | Tested e2e |
| SFT | `acestep-v15-sft` | 50 | 7.0 | Tested e2e |
| Turbo | `acestep-v15-turbo` | 8 | 1.0 | Tested e2e |

**5 Hz LM sizes (`--lm_variant`) — all tested:**

| LM size | CLI value | Status |
|---------|-----------|--------|
| 0.6B | `acestep-5Hz-lm-0.6B` | Tested e2e |
| 1.7B | `acestep-5Hz-lm-1.7B` | Tested e2e (default) |
| 4B | `acestep-5Hz-lm-4B` | Tested e2e |

Default demo pairing: **`acestep-v15-base`** + **`acestep-5Hz-lm-1.7B`**. Turbo runs use **`acestep-v15-turbo`** with any of the three LM sizes. SFT uses the same CFG/step defaults as base (`infer_steps=50`, `guidance_scale=7`).

**BH_QB benchmark data** (prompt *Guitar*, branch `ign/ACE_demo_modified`): full timing + audio-quality notes for **15s / 30s / 60s** × **base / turbo / sft** × **LM 0.6B / 1.7B / 4B** are in [`perf/BENCHMARK_RESULTS.md`](perf/BENCHMARK_RESULTS.md) (exported from [`perf/Testing_ACE.xlsx`](perf/Testing_ACE.xlsx)). At **30s+**, base/sft subjective quality is often **Partially good** or **Noise** vs **GOOD** for turbo — see the note below.

**Why base / SFT can sound noisier than turbo at 30s+**

Turbo is **CFG-distilled** for **8 steps @ `guidance_scale=1`**: a short, stable TTNN path with no classifier-free guidance loop. Base and SFT use the **full sampler** (**50 steps**, **CFG=7**, ADG/APG on host between steps), so small TTNN numeric differences (LoFi matmul, `bfloat8_b` / `bfloat4_b` weights, host↔device Euler updates) **accumulate over many more denoise steps**. At **≥30s** on mesh, the stack also switches to **long-clip mode**: DiT trace off, DRAM activations, eager multi-tile VAE decode, and wider overlap-add — each adds boundary/rounding error that turbo’s shorter path mostly avoids. That does **not** mean base/sft are broken on hardware; it means **turbo is the recommended default for long clips on BH_QB today**. For cleaner base/sft output at 30–60s, try `--clarity`, `--lm_variant acestep-5Hz-lm-4B`, `ACE_STEP_MAX_AUDIO_CODES=350`, and `--torch-vae` if hiss persists; see [Quality presets](#quality-presets-examples) below.

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
- `demo/`: end-to-end demo entry points (`run_prompt_to_wav.py`, `serve_prompt_to_wav.py`, `demo.md`)
- `perf/`: benchmark Excel log + export to `BENCHMARK_RESULTS.md`

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- Installed [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- HuggingFace token: `huggingface-cli login` or `export HF_TOKEN=<token>` (checkpoints auto-download on first run to `~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints/`)

Install demo-specific Python packages from the `tt-metal` repo root:

```bash
pip install -r models/experimental/ace_step_v1_5/requirements.txt
pip install openpyxl   # benchmark log xlsx (perf/create_benchmark_log_xlsx.py)
```

### Environment (every run)

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export MESH_DEVICE=BH_QB          # or P150 / BH_LB — DiT/VAE mesh SKU
# optional: export ACE_STEP_MESH_DEVICE=BH_QB
```

---

Generate a WAV audio file from a text prompt using TTNN-accelerated inference.

Entry point: `demo/run_prompt_to_wav.py` (see `demo/demo.md` for session passes and perf tuning).

```bash
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py \
  --variant acestep-v15-turbo \
  --lm_variant acestep-5Hz-lm-1.7B \
  --duration_sec 15 \
  --prompt "epic orchestral cinematic music for a movie trailer" \
  --infer_steps 8 \
  --guidance_scale 1 \
  --mesh-device BH_QB \
  --out /tmp/turbo_15s.wav
```

On first run, any missing model checkpoints are automatically downloaded from HuggingFace
into `~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints/`.

### Trace (default: on)

TTNN trace + 2CQ is **enabled by default**. To run in fully-eager single-CQ mode (e.g. for debugging or Tracy profiling):

```bash
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py ... --no-use-trace
```

### More examples

**Turbo model — fast generation (8 steps, CFG off):**

```bash
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py \
  --variant acestep-v15-turbo \
  --lm_variant acestep-5Hz-lm-1.7B \
  --duration_sec 15 \
  --infer_steps 8 \
  --guidance_scale 1 \
  --mesh-device BH_QB \
  --prompt "EDM: deep bass, punchy kick, bright synth lead" \
  --out /tmp/turbo_15s.wav
```

**Base model with 1.7B LM (balanced quality/speed):**

```bash
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py \
  --prompt "Acoustic guitar ballad with soft vocals and ambient strings" \
  --variant acestep-v15-base \
  --lm_variant acestep-5Hz-lm-1.7B \
  --duration_sec 30 \
  --seed 42 \
  --mesh-device BH_QB \
  --out /tmp/ballad.wav
```

**SFT model (same defaults as base — 50 steps, CFG=7):**

```bash
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py \
  --variant acestep-v15-sft \
  --lm_variant acestep-5Hz-lm-1.7B \
  --duration_sec 30 \
  --mesh-device BH_QB \
  --prompt "Lo-fi hip hop beat with warm vinyl crackle and mellow Rhodes piano" \
  --out /tmp/sft_30s.wav
```

**Base model with 4B LM (highest quality, slower):**

```bash
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py \
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
python3 models/experimental/ace_step_v1_5/demo/serve_prompt_to_wav.py --port 8765
```

Disable in-memory weight caching with `ACE_STEP_DISABLE_WEIGHT_CACHE=1`.

Logging defaults to **INFO** (hides TTNN per-tensor flatbuffer cache DEBUG spam).

### Demo CLI reference (`demo/run_prompt_to_wav.py`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--prompt` | `str` | **(required)** | Text description of the music to generate. |
| `--variant` | `str` | `acestep-v15-base` | DiT checkpoint. Choices: `acestep-v15-base`, `acestep-v15-sft`, `acestep-v15-turbo`. |
| `--lm_variant` | `str` | `acestep-5Hz-lm-1.7B` | 5 Hz LM size. Choices: `acestep-5Hz-lm-0.6B`, `acestep-5Hz-lm-1.7B`, `acestep-5Hz-lm-4B`. |
| `--device_id` | `int` | `0` | TT device index (single-chip / preprocess device). |
| `--mesh-device` | `str` | env | DiT/VAE mesh SKU (`P150`, `BH_QB`, `BH_LB`, …). Also `ACE_STEP_MESH_DEVICE` or `MESH_DEVICE`. |
| `--duration_sec` | `float` | `10.0` | Target audio length in seconds. |
| `--infer_steps` | `int` | auto | Diffusion steps. Default **8** turbo, **50** base/sft. |
| `--seed` | `int` | `0` | RNG seed (latents + sampling). |
| `--guidance_scale` | `float` | auto | DiT CFG strength. Default **7** base/sft, **1** turbo (turbo ignores CFG >1). |
| `--out` | `str` | `ttnn_out.wav` | Output WAV path. |
| `--use-official-lm` | flag | off | Full official `generate_music` on CPU (PyTorch DiT). A/B reference; no TTNN. |
| `--pytorch-lm` / `--no-pytorch-lm` | bool | off | Host PyTorch 5 Hz LM instead of TTNN causal LM. |
| `--torch-vae` | flag | off | PyTorch Oobleck decode instead of TTNN tiled VAE. |
| `--use-trace` / `--no-use-trace` | bool | **on** | TTNN trace + 2 command queues for preprocess + DiT body. Auto-off for ≥30s mesh long clips. |
| `--clarity` / `--no-clarity` | bool | off | Mesh quality preset: ADG guidance, wider VAE overlap, BF16 VAE at ≥30s. |
| `--use-adg` / `--no-use-adg` | bool | auto | DiT CFG: **ADG** (default base/sft) vs **APG** (default mesh without clarity). |
| `--llm-debug` | flag | off | Extra constrained-decoding debug logs. Tokens/sec always in KEY METRICS when LM runs. |

**Variant defaults (when flags omitted):**

| Variant | `infer_steps` | `guidance_scale` | Notes |
|---------|---------------|------------------|-------|
| `acestep-v15-turbo` | 8 | 1.0 | CFG-distilled; fastest path |
| `acestep-v15-base` | 50 | 7.0 | Full CFG + ADG; use `--clarity` for ≥30s mesh |
| `acestep-v15-sft` | 50 | 7.0 | Same defaults as base |

**Common environment overrides:**

| Variable | Purpose |
|----------|---------|
| `ACE_STEP_MESH_DEVICE` / `MESH_DEVICE` | Default mesh SKU if `--mesh-device` omitted |
| `ACE_STEP_MAX_AUDIO_CODES` | Max LM audio codes for detokenizer (use **350** for 60s) |
| `ACE_STEP_VAE_QUALITY=1` | Force BF16 VAE decode on mesh |
| `ACE_STEP_DEMO_PERF_LOG=1` | Enable `[ace_step_v1_5][perf]` timing tables |
| `ACE_STEP_DISABLE_WEIGHT_CACHE=1` | Disable in-memory weight reuse |
| `HF_TOKEN` | HuggingFace download token |

**Quality presets (examples):**

```bash
# Base 60s — long-clip quality (slower)
export ACE_STEP_MAX_AUDIO_CODES=350
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py \
  --mesh-device BH_QB --variant acestep-v15-base --lm_variant acestep-5Hz-lm-4B \
  --duration_sec 60 --infer_steps 50 --guidance_scale 7 --clarity \
  --prompt "..." --out /tmp/base_60s.wav

# Isolate VAE hiss vs DiT noise
  ... --torch-vae

# Softer ambient mix (less ADG smear)
  ... --no-use-adg --guidance_scale 5
```

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

## Layout

```
ace_step_v1_5/
  demo/           run_prompt_to_wav.py, serve_prompt_to_wav.py, demo.md
  perf/           benchmark xlsx + export scripts
  torch_ref/
  ttnn_impl/
  tests/
```

## PCC and module tests

All tests live under `tests/`. PCC tests compare TTNN outputs to PyTorch references and assert **PCC ≥ 0.99** via `assert_pcc_print` in `tests/_dit_decoder_pcc_common.py`.

### Prerequisites

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export MESH_DEVICE=BH_QB    # or P150 — required for device/mesh tests
```

Run from repo root with `--confcutdir` so pytest loads `ace_step_v1_5/conftest.py`:

```bash
pytest models/experimental/ace_step_v1_5/tests \
  --confcutdir=models/experimental/ace_step_v1_5/tests -q
```

### Per-module PCC commands

| Subsystem | Test file | What it validates | Command |
|-----------|-----------|-------------------|---------|
| **DiT AdaLN** | `test_pcc_adaln.py` | Scale-shift modulation in `TtAceStepDiTLayer` | `pytest models/experimental/ace_step_v1_5/tests/test_pcc_adaln.py -q` |
| **DiT attention** | `test_pcc_attention.py` | Self + cross SDPA, GQA | `pytest models/experimental/ace_step_v1_5/tests/test_pcc_attention.py -q` |
| **DiT block** | `test_pcc_block.py` | `TtQwen3MLP` + full DiT layer | `pytest models/experimental/ace_step_v1_5/tests/test_pcc_block.py -q` |
| **DiT core** | `test_pcc_dit_decoder_core.py` | Full `TtAceStepDiTCore`, timestep embed | `pytest models/experimental/ace_step_v1_5/tests/test_pcc_dit_decoder_core.py -q` |
| **DiT linear** | `test_pcc_dit_linear.py` | LoFi BFP8 matmul projections | `pytest models/experimental/ace_step_v1_5/tests/test_pcc_dit_linear.py -q` |
| **DiT patchify** | `test_patchify_pcc.py` | Patch embed + depatchify | `pytest models/experimental/ace_step_v1_5/tests/test_patchify_pcc.py -q` |
| **DiT output head** | `test_output_head.py` | Output head vs torch | `pytest models/experimental/ace_step_v1_5/tests/test_output_head.py -q` |
| **DiT HF parity** | `test_hf_parity_patch_output_head.py` | HF base patch embed + output head | `pytest models/experimental/ace_step_v1_5/tests/test_hf_parity_patch_output_head.py -q` |
| **DiT denoise loop** | `test_dit_denoise_loop_pcc.py` | Full Euler loop vs torch (750/1500 frames) | `pytest models/experimental/ace_step_v1_5/tests/test_dit_denoise_loop_pcc.py -q` |
| **DiT denoise + CFG** | `test_dit_denoise_loop_pcc_cfg.py` | APG + DCW vs torch with CFG | `pytest models/experimental/ace_step_v1_5/tests/test_dit_denoise_loop_pcc_cfg.py -q` |
| **VAE Snake1D** | `test_vae_decoder_pcc.py` | Snake, residual, decoder block, tiny decoder | `pytest models/experimental/ace_step_v1_5/tests/test_vae_decoder_pcc.py -q` |
| **VAE trace equiv** | `test_vae_chunk_trace_equiv.py` | Tiled/trace vs eager VAE decode | `pytest models/experimental/ace_step_v1_5/tests/test_vae_chunk_trace_equiv.py -q` |
| **Condition encoder** | `test_condition_embedding_ttnn.py` | Text embed, Qwen3 TTT prefill, ctx latents | `pytest models/experimental/ace_step_v1_5/tests/test_condition_embedding_ttnn.py -q` |
| **5 Hz LM logits** | `test_llm_handler_logits_pcc.py` | LM postprocess, CFG combine, prefill/decode | `pytest models/experimental/ace_step_v1_5/tests/test_llm_handler_logits_pcc.py -q` |
| **LM memory patches** | `test_qwen_lm_mem_patches.py` | Prefill L1 / decode shard patches (unit) | `pytest models/experimental/ace_step_v1_5/tests/test_qwen_lm_mem_patches.py -q` |

### Run all DiT PCC tests

```bash
pytest models/experimental/ace_step_v1_5/tests/test_pcc_adaln.py \
  models/experimental/ace_step_v1_5/tests/test_pcc_attention.py \
  models/experimental/ace_step_v1_5/tests/test_pcc_block.py \
  models/experimental/ace_step_v1_5/tests/test_pcc_dit_decoder_core.py \
  models/experimental/ace_step_v1_5/tests/test_pcc_dit_linear.py \
  models/experimental/ace_step_v1_5/tests/test_patchify_pcc.py \
  models/experimental/ace_step_v1_5/tests/test_output_head.py \
  models/experimental/ace_step_v1_5/tests/test_hf_parity_patch_output_head.py \
  --confcutdir=models/experimental/ace_step_v1_5/tests -q
```

### Run all VAE PCC tests

```bash
pytest models/experimental/ace_step_v1_5/tests/test_vae_decoder_pcc.py \
  models/experimental/ace_step_v1_5/tests/test_vae_chunk_trace_equiv.py \
  models/experimental/ace_step_v1_5/tests/test_vae_matmul_program_config.py \
  models/experimental/ace_step_v1_5/tests/test_vae_tile_passthrough.py \
  --confcutdir=models/experimental/ace_step_v1_5/tests -q
```

### CPU-only / host tests (no device)

```bash
pytest models/experimental/ace_step_v1_5/tests/test_apg_guidance_host.py \
  models/experimental/ace_step_v1_5/tests/test_safetensors_loader.py \
  models/experimental/ace_step_v1_5/tests/test_dcw_sampler.py \
  models/experimental/ace_step_v1_5/tests/test_tt_device_mesh.py -q
```

### Single test by name

```bash
pytest models/experimental/ace_step_v1_5/tests/test_pcc_dit_decoder_core.py::test_dit_decoder_core_matches_torch -q
```

Production DiT code: `ttnn_impl/dit_decoder_core.py`. The `mesh_device` pytest fixture is **session-scoped**.

---

## Benchmark run log (Excel)

Measured on **BH_QB** with prompt *Guitar* (see [`perf/Testing_ACE.xlsx`](perf/Testing_ACE.xlsx)). Markdown export (same block layout as the spreadsheet):

| File | Purpose |
|------|---------|
| [`perf/Testing_ACE.xlsx`](perf/Testing_ACE.xlsx) | **Current BH_QB results** — Configuration + 15s / 30s / 60s sheets |
| [`perf/BENCHMARK_RESULTS.md`](perf/BENCHMARK_RESULTS.md) | Markdown mirror of `Testing_ACE.xlsx` (full tables) |
| `perf/export_testing_ace_md.py` | Regenerate `BENCHMARK_RESULTS.md` after editing the xlsx |
| `perf/create_benchmark_log_xlsx.py` | Optional flat-row template (`ace_step_turbo_benchmark_log.xlsx`) |
| `perf/export_benchmark_summary_md.py` | Export flat template → md (legacy workflow) |

**Workflow (Testing_ACE format):**

1. Edit [`perf/Testing_ACE.xlsx`](perf/Testing_ACE.xlsx) after each run (Metric / Value / Description / OUTPUT AUDIO GOOD/BAD blocks per variant × LM).
2. Refresh markdown: `python models/experimental/ace_step_v1_5/perf/export_testing_ace_md.py`
3. Commit xlsx + `BENCHMARK_RESULTS.md` when updating benchmark data.

**Latest results:** [perf/BENCHMARK_RESULTS.md](perf/BENCHMARK_RESULTS.md) — includes Configuration (mesh **BH QB**, steps **50/8/50** for base/turbo/sft) and all 27 variant×LM×duration runs.

---

## TTNN demo (`ttnn_impl/full_pipeline.py`) — weights default to **Base**

The TTNN entrypoint defaults to Hugging Face **`ACE-Step/acestep-v15-base`** when you omit a local checkpoint:

```bash
python models/experimental/ace_step_v1_5/ttnn_impl/full_pipeline.py \
  --out-npy /tmp/ace_features.npy
```

Override weights explicitly:

```bash
python models/experimental/ace_step_v1_5/ttnn_impl/full_pipeline.py \
  --checkpoint-safetensors /path/to/model.safetensors \
  --out-npy /tmp/ace_features.npy
```

Turbo lives under the umbrella repo ``ACE-Step/Ace-Step1.5``; pick it with ``--hf-subfolder``:

```bash
python models/experimental/ace_step_v1_5/ttnn_impl/full_pipeline.py \
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
python -m models.experimental.ace_step_v1_5.torch_ref.hf_output_head_demo \
  --repo-id "ACE-Step/Ace-Step1.5" \
  --subfolder "acestep-v15-turbo" \
  --seed 0 --batch 1 --original-seq-len 257 --noise-std 1.0
```

Use **`--no-use-trace`** for a fully eager single-CQ pipeline (debugging or Tracy device profiling).
