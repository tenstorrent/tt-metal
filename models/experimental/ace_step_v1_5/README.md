# ACE-Step v1.5

## Platforms

Blackhole (BH_QB)

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
| Turbo | `acestep-v15-turbo` | 8 | 1.0 | Tested e2e (default) |

**5 Hz LM sizes (`--lm_variant`) — all tested:**

| LM size | CLI value | Status |
|---------|-----------|--------|
| 0.6B | `acestep-5Hz-lm-0.6B` | Tested e2e |
| 1.7B | `acestep-5Hz-lm-1.7B` | Tested e2e (default) |
| 4B | `acestep-5Hz-lm-4B` | Tested e2e |

Default demo pairing: **`acestep-v15-turbo`** + **`acestep-5Hz-lm-1.7B`**. Turbo runs use **`acestep-v15-turbo`** with any of the three LM sizes. SFT uses the same CFG/step defaults as base (`infer_steps=50`, `guidance_scale=7`).

**BH_QB benchmark data** (prompt *Guitar*, branch `ign/ACE_demo_modified`): full timing + audio-quality notes for **15s / 30s / 60s** × **base / turbo / sft** × **LM 0.6B / 1.7B / 4B** are in [`perf/BENCHMARK_RESULTS.md`](perf/BENCHMARK_RESULTS.md)

**Why base / SFT can sound noisier than turbo at 60s+**

Turbo is **CFG-distilled** for **8 steps @ `guidance_scale=1`**: a short, stable TTNN path with no classifier-free guidance loop. Base and SFT use the **full sampler** (**50 steps**, **CFG=7**, ADG/APG on host between steps), so small TTNN numeric differences (LoFi matmul, `bfloat8_b` / `bfloat4_b` weights, host↔device TTNN updates) **accumulate over many more denoise steps**. At **≥30s** on mesh, the stack also switches to **long-clip mode**: DiT trace off, DRAM activations, eager multi-tile VAE decode, and wider overlap-add — each adds boundary/rounding error that turbo’s shorter path mostly avoids. That does **not** mean base/sft are broken on hardware; it means **turbo is the recommended default for long clips on BH_QB today**. For cleaner base/sft output at >30s try `--clarity`, `--lm_variant acestep-5Hz-lm-4B`, `ACE_STEP_MAX_AUDIO_CODES=350`, and `--torch-vae` if hiss persists; see [Quality presets](#quality-presets-examples) below.

### Qwen3 embedding (TTNN)

Caption / text conditioning uses a **local copy** of the TTNN Qwen3-Embedding stack, vendored under this demo until upstream lands in `tt-metal` main:

- `ttnn_impl/qwen3_embedding_encoder.py` — Qwen3 transformer blocks + `TtQwen3EmbeddingEncoder`
- `ttnn_impl/qwen3_embedding_ace_step.py` — ACE-Step wrapper (`AceStepQwen3Encoder`) over the HF `Qwen3-Embedding-0.6B` checkpoint

This code is replicated from the work in [tt-metal PR #42463](https://github.com/tenstorrent/tt-metal/pull/42463) ([`ign/qwen3-embedding-4b-optimizations`](https://github.com/tenstorrent/tt-metal/tree/ign/qwen3-embedding-4b-optimizations)). Once that PR is merged, this demo can switch to importing the shared implementation from `tt-metal` instead of maintaining duplicate modules here.

## Tensor parallelism & mesh parallelism (optional)

Beyond the stock replicate pipeline, the DiT and VAE can shard work across the BH_QB 2×2 mesh. **Every feature here is flag-gated and byte-identical to the replicate path when off.** The library default is OFF; the demo (`demo/run_prompt_to_wav.py`) additionally defaults `ACE_STEP_TP=2` on a BH_QB mesh when `--use-trace` is active (set `ACE_STEP_TP=off` to opt out). The full implementation matrix, PCC tables, per-duration benchmark matrix (P150 / repl / TP2 / TP4), and key findings are in [`perf/ACE_STEP_TP_FINAL_REPORT.md`](perf/ACE_STEP_TP_FINAL_REPORT.md).

| Feature | Flag | Status | Headline |
|---------|------|--------|----------|
| **DiT tensor-parallel** (MLP col/row + head-parallel attention, all-reduce) | `ACE_STEP_TP=on` (2-way) / `=4` (4-way) | Done, PCC-validated (full block PCC 0.999) | Latency-neutral with `--use-trace`; ~27% faster DiT at 60s (TP2); halves (TP2) / quarters (TP4) DiT weights per chip |
| **VAE time-parallel decode** (batch tiles, shard batch across mesh) | `ACE_STEP_VAE_TIME_PARALLEL=1` | Done, PCC 1.0, bit-identical audio | **4.3× VAE decode, −32% wall at 60s** — the biggest throughput win |
| **Encoder tensor-parallel** (shared Qwen3 encoder layer: detok + caption + condition) | `ACE_STEP_TP` | Code done, isolated PCC 1.0 | Inert until Phase-A-on-mesh lands |
| **Phase-A-on-mesh** (run preprocess + DiT on one mesh, skip re-exec) | `ACE_STEP_PREPROCESS_ON_MESH=1` | Wiring done | Blocked — module tuned configs assume non-TP dims |

**Enable (compose freely, always with `--use-trace`):**

```bash
# DiT tensor-parallel — recommended: 2-way + trace = parity + half DiT weights/chip
ACE_STEP_TP=on ... --mesh-device BH_QB --use-trace ...
ACE_STEP_TP=4  ... --mesh-device BH_QB --use-trace ...     # 4-way: quarter weights, slower

# VAE time-parallel decode (the 4.3× win) — composes with the above
ACE_STEP_VAE_TIME_PARALLEL=1 ...
```

**Notes and trade-offs:**

- **Always run DiT TP with `--use-trace`.** Eager TP is ~2.5× slower (un-amortised collective launch); trace replay recovers it.
- **DiT TP has a duration crossover** (base/sft, trace): 15s slower → 30s ~break-even → **60s TP2 ~27% faster DiT, ~4–5% faster wall**. Longer sequences amortise the collectives, so TP's main value on short clips is **memory** (½ or ¼ weights/chip), not latency.
- **4-way < 2-way on latency** (more collective traffic) but gives ¼ weights/chip.
- The mesh (even replicate) already beats P150 by ~2.5–3× at 30/60s because P150's single-chip DiT/VAE scale badly with sequence length.

**Deferred / blocked (documented):** 5 Hz LM on mesh (stock `tt_transformers` shard-grid `TT_FATAL` on BH 2×2), Phase-A modules under TP (tuned sharded configs assume non-TP dims), and VAE tensor-parallel conv (`prepare_conv_weights` has no cross-mesh shard — time-parallel decode replaces it).

## This repo folder

This folder provides:

- `host_preprocess/`: demo-only ACE-Step host stack (`acestep.handler`, HF modeling for preprocess init)
- `torch_ref/`: PyTorch reference modules for PCC tests (not a runnable demo)
- `ttnn_impl/`: TTNN implementation with one-to-one module mapping
- `tests/`: per-module PCC validation (Torch vs TTNN)
- `demo/`: end-to-end demo entry points (`run_prompt_to_wav.py`, `serve_prompt_to_wav.py`, `demo.md`)
- `perf/`: benchmark Excel log + export to `BENCHMARK_RESULTS.md`

## Prerequisites
- HuggingFace token: `huggingface-cli login` or `export HF_TOKEN=<token>` (checkpoints auto-download on first run to `~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints/`)

Install demo-specific Python packages from the `tt-metal` repo root:

```bash
python_env/bin/python -m ensurepip --upgrade
python_env/bin/python -m  pip install -r models/experimental/ace_step_v1_5/requirements.txt
```

### Environment (every run)

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export MESH_DEVICE=BH_QB
```

---

## How to use the demo

### 1. Minimal command

```bash
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py \
  --mesh-device BH_QB \
  --variant acestep-v15-turbo \
  --lm_variant acestep-5Hz-lm-1.7B \
  --duration_sec 15 \
  --prompt "guitar, saxophone and prominent drums with clear kick and snare" \
  --infer_steps 8 \
  --guidance_scale 1 \
  --no-use-cot-caption \
  --out /tmp/turbo_15s.wav
```

Required: `--prompt`. Everything else has sensible defaults (`--variant acestep-v15-turbo`, `--lm_variant acestep-5Hz-lm-1.7B`, `--duration_sec 10`, `--out ttnn_out.wav`).

### 2. BH_QB run flow (what happens)

On **BH_QB** the demo uses **split preprocess**:

| Phase | Device | Work |
|-------|--------|------|
| **A — preprocess** | 1×1 TT chip | 5 Hz LM, Qwen3 caption, audio-code detokenizer, (optional) condition encode |
| **Handoff** | process re-exec | Closes 1×1, opens fresh 2×2 mesh (required on Blackhole) |
| **B — generate** | 2×2 mesh | Condition encode (long clips), DiT denoise, TTNN VAE decode → WAV |

Healthy long-clip logs include:

```
[ace_step_v1_5] mesh SKU=BH_QB split_preprocess=True
[ace_step_v1_5] defer TTNN condition encoder to DiT mesh ...   # ≥30 s (≥750 latent frames)
[ace_step_v1_5] DiT: re-exec for full mesh (BH_QB), handoff=...
[ace_step_v1_5] opened DiT mesh for SKU=BH_QB
[condition] backend=ttnn on DiT mesh (...)
```

If a prior run left the device busy: `kill <pid>` then `tt-smi -r 0` if needed.

### 3. Recommended recipes

**Turbo — fast, best for long clips (60–120 s):** `--variant acestep-v15-turbo` with `--infer_steps 8` and `--guidance_scale 1` (turbo defaults).

**Multi-instrument — when drums or listed instruments drop out**, add `--no-use-cot-caption` so DiT gets your exact prompt (see [Caption / text conditioning](#caption--text-conditioning-default-lm-cot-rewrite)).

**Base / SFT — higher quality, slower:** `--variant acestep-v15-base` or `acestep-v15-sft` with `--infer_steps 50` and `--guidance_scale 7`; add `--clarity` on long mesh clips (see [Quality presets](#quality-presets-examples)).


### 4. TTNN vs host PyTorch (BH_QB default path)

On **BH_QB**, the default is **split TTNN preprocess** (Phase A on a 1×1 chip → re-exec → Phase B on 2×2 mesh). PyTorch on CPU/GPU handles orchestration, tokenization, and stages where TTNN either cannot match HF numerics yet or would regress audio quality.

The denoise loop runs DiT forwards on TTNN; on BH_QB the per-step CFG (APG/ADG) and TTNN update run on host PyTorch between TTNN steps.

| Stage | Default backend (BH_QB) | Notes |
|-------|-------------------------|-------|
| Handler / `preprocess_batch` | **Host** | Upstream `AceStepHandler` orchestration on CPU |
| 5 Hz LM forward | **TTNN** | 1×1 preprocess chip; `--pytorch-lm` or `ACE_STEP_MESH_HOST_PREPROCESS=1` → host PyTorch |
| LM tokenize / constrained FSM | **Host** | HF `AutoTokenizer` + metadata FSM (not the causal LM matmuls) |
| Qwen3 caption + lyric embed | **TTNN** | 1×1 preprocess chip (host PyTorch if `ACE_STEP_MESH_HOST_PREPROCESS=1`) |
| Audio-code → 25 Hz hints | **HF PyTorch** (>200 codes) / **TTNN** (≤200) | See [Host PyTorch fallbacks](#host-pytorch-fallbacks-why-not-full-ttnn) below |
| Condition encoder | **TTNN** | 1×1 preprocess for **<30 s** (<750 latent frames); **deferred to 2×2 DiT mesh** at **≥30 s** (avoids 1×1 readback drift) |
| DiT denoise forward | **TTNN** | Opt-in HF DiT: `ACE_STEP_PYTORCH_DIT=1` |
| Latent noise init | **On mesh** (short) / **Host CPU** (long) | With default `--use-trace`, latents stay on device for **≤15 s**; trace is forced off at **≥30 s** → host latent init |
| Euler / APG / ADG between steps | **Host CPU** | Always on multi-device mesh (short and long clips); DiT forwards stay TTNN |
| VAE decode | **TTNN** (eager tiled) | Opt-in: `--torch-vae` |
| WAV write | **Host** | Peak normalize + file I/O |

#### Host PyTorch fallbacks — why not full TTNN

These are **production defaults**, not temporary debug flags. Each row is something that still runs (fully or partially) on host PyTorch, why a pure-TTNN path is not enabled, and how it shows up in WAV output when the wrong backend is forced.

| Stage | Default | Why TTNN-only is not used | What you hear if forced wrong |
|-------|---------|---------------------------|-------------------------------|
| **Audio-code detokenizer** (>200 codes, e.g. **>40 s**) | **HF PyTorch** (global attention over all codes) | TTNN is L1-limited to **~200 codes/forward** on 1×1 BH. A multi-forward TTNN path **concatenates chunks with no cross-chunk attention**, so latent hints around the ~40 s seam and the whole second half are wrong vs HF. Demo **auto-sets** `ACE_STEP_PYTORCH_DETOK=1` when expect >200 codes. | Muddy / noisy / lost instruments — **most obvious with 4B LM @ 60 s**; turbo + 1.7B can also degrade. **Fix:** leave auto HF detok on (default) or do not set `ACE_STEP_PYTORCH_DETOK=0`. |
| **Audio-code detokenizer** (≤200 codes, e.g. **≤40 s**) | **TTNN** | Single forward fits in L1; PCC ~0.986 vs HF on 15–30 s shapes. Hints stay on device as `precomputed_lm_hints_25Hz_tt` until the condition encoder consumes them. | Good on default path. `ACE_STEP_TORCH_DETOK_HINTS=1` adds a host round-trip (debug/A-B only). |
| **DiT Euler + APG/ADG** (mesh) | **Host CPU** between TTNN forwards | Per-step guidance math (`sigma`, APG/ADG) uses host control flow and FP32 staging; not fused into the DiT body trace. | Stable default. Forcing full-step DiT trace without eager post-step CFG was tried — **audible noise**. |
| **DiT CFG batch setup** (enc/ctx concat before loop) | **Eager TTNN** (not traced) | Traced prep for CFG duplicate batches was **not bit-accurate** vs eager setup. | Noise / smear if traced CFG prep is re-enabled. |
| **VAE decode** | **TTNN eager** `decode_tiled` | Traced VAE replay failed PCC / produced **hiss** on long multi-tile decode. | Tile-boundary hiss on long clips → try `--torch-vae` or `--clarity` (wider overlap). |
| **Condition encode** (≥30 s mesh) | **TTNN on DiT mesh** (not 1×1 preprocess) | Encoding on 1×1 then readback for long clips added **numeric drift** at ≥750 latent frames. After mesh encode, enc/ctx are **restaged** (f32 readback → BF16 TILE L1) then fed to DiT on-device — raw encoder tensors are not used directly. | Long-clip drift / wrong timbre if condition is forced back to 1×1 preprocess only. |
| **Latent noise init** (≥30 s mesh) | **Host CPU** | DiT body trace is forced off on long clips; host init matches the eager denoise path. | Minor vs detok/VAE issues; trace-on long clips was unstable. |
| **LM constrained FSM + tokenize** | **Host** | Python metadata FSM and HF tokenizer — not LM matmuls. | N/A (orchestration only). |

**Not landed (reverted or debug-only) — do not expect these on the default demo path:**

| Idea | Status | Why dropped / not default | Demo impact if re-enabled |
|------|--------|---------------------------|---------------------------|
| TTNN **chunked** detok as default for >200 codes | **Off** (auto HF detok instead) | Chunks break bidirectional detok attention | Bad 60 s audio (especially 4B); use `ACE_STEP_PYTORCH_DETOK=0` only to reproduce the bug |
| Skip condition **f32 readback** / all-device condition handoff (P4 v1) | **Fixed (P4 v2)** | Raw encoder tensors → DiT caused **60 s drift**; v2 **restages** through validated f32→BF16 TILE L1 upload, then keeps device tensors for denoise | Default ≥750 frames; opt-out host cache: `ACE_STEP_TORCH_CONDITION_HANDOFF=1` |
| Device-only Qwen caption / lyric (no host token path) | **Reverted** | Did not improve 60 s quality | No benefit on default path |
| Device Euler / CFG on mesh as default | **Not default** | Host Euler + TTNN DiT body is the validated long-clip path | Unvalidated for 60 s golden audio |

**Opt-in host PyTorch (A/B isolation — not default):**

| Flag / env | Effect |
|------------|--------|
| `--pytorch-lm` | 5 Hz LM on host PyTorch instead of TTNN (also forced when `ACE_STEP_MESH_HOST_PREPROCESS=1`) |
| `--torch-vae` | PyTorch Oobleck decode instead of TTNN tiled VAE |
| `ACE_STEP_PYTORCH_DIT=1` | HF PyTorch DiT denoise instead of TTNN |
| `ACE_STEP_PYTORCH_CONDITION=1` | HF `prepare_condition` instead of TTNN condition encoder |
| `ACE_STEP_MESH_HOST_PREPROCESS=1` | Skip TTNN Phase A — host PyTorch preprocess, Qwen, and 5 Hz LM |
| `ACE_STEP_PYTORCH_DETOK=0` | Force TTNN chunked detok even when >200 codes (**debug only** — expect bad long-clip audio) |
| `ACE_STEP_TORCH_DETOK_HINTS=1` | After TTNN detok (≤200 codes), round-trip hints via `ttnn.to_torch` instead of device-native `precomputed_lm_hints_25Hz_tt` |
| `ACE_STEP_TORCH_CONDITION_HANDOFF=1` | After mesh condition encode, keep host torch enc/ctx cache + host denoise staging (legacy path; skips P4 device handoff) |

### 5. Long clips (≥30 s on mesh)

Automatic behavior (no extra flags):

- **DiT trace off** — `--use-trace` is forced off at ≥30 s (long-clip quality mode).
- **Condition encode on DiT mesh** — avoids 1×1 readback drift at ≥750 latent frames; enc/ctx are restaged on device for DiT (P4).
- **Host latent sampler** — latent init on CPU when trace is off (forced at ≥30 s on mesh); Euler/APG/ADG always on CPU on multi-device mesh.
- **Wider VAE overlap** — fewer tile-boundary artifacts on decode.
- **HF detokenizer** — demo auto-enables `ACE_STEP_PYTORCH_DETOK=1` when expect **>200 audio codes** (>40 s @ 5 Hz). See [Host PyTorch fallbacks](#host-pytorch-fallbacks-why-not-full-ttnn).

### Weight caching (avoid reloading from disk)

Weights are cached at two levels:

| Tier | Location | When it applies |
|------|----------|-----------------|
| **HuggingFace disk** | `~/.cache/huggingface/hub/ACE-Step-1.5-checkpoints/` | First download of each checkpoint; reused across all runs |
| **Process RAM** | In-memory registry (`utils/weight_cache.py`) | Within one Python process — avoids re-reading the same `.safetensors` for condition encoder, detokenizer, VAE, etc. |

Each **`run_prompt_to_wav.py` invocation is a new process**: you still pay **handler init** and **TT device weight upload** every time, even though safetensors are not re-read from disk after the first load in that process. For multiple generations without reload, use the HTTP service below.

Look for log lines (loguru, no `[ace_step_v1_5]` prefix):

```
⏳ LOAD   condition-encoder-weights    path=.../model.safetensors
♻  REUSE  condition-encoder-weights    [already loaded in memory]
```

**HTTP service (models loaded once at startup, reused across requests):**

```bash
python3 models/experimental/ace_step_v1_5/demo/serve_prompt_to_wav.py --port 8765
```

Disable the in-process safetensors cache with `ACE_STEP_DISABLE_WEIGHT_CACHE=1` (HF files on disk are unchanged).

Logging defaults to **INFO** (hides TTNN per-tensor flatbuffer cache DEBUG spam).

### Quality presets (examples)

For cleaner **base/sft** output at **>30 s** on mesh (when turbo-quality speed is not the goal):

```bash
# Base 60s — long-clip quality (slower; demo auto-sets ACE_STEP_MAX_AUDIO_CODES from --duration_sec)
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py \
  --mesh-device BH_QB \
  --variant acestep-v15-base \
  --lm_variant acestep-5Hz-lm-4B \
  --duration_sec 60 \
  --infer_steps 50 \
  --guidance_scale 7 \
  --no-use-cot-caption \
  --clarity \
  --prompt "guitar, saxophone and prominent drums with clear kick and snare" \
  --out /tmp/base_60s.wav
```

### Demo CLI reference (`demo/run_prompt_to_wav.py`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--prompt` | `str` | **(required)** | Text description of the music to generate. |
| `--variant` | `str` | `acestep-v15-turbo` | DiT checkpoint. Choices: `acestep-v15-base`, `acestep-v15-sft`, `acestep-v15-turbo`. |
| `--lm_variant` | `str` | `acestep-5Hz-lm-1.7B` | 5 Hz LM size. Choices: `acestep-5Hz-lm-0.6B`, `acestep-5Hz-lm-1.7B`, `acestep-5Hz-lm-4B`. |
| `--device_id` | `int` | `0` | TT device index (single-chip / preprocess device). |
| `--mesh-device` | `str` | env | DiT/VAE mesh SKU (`P150`, `BH_QB`, `BH_LB`, …). Also `ACE_STEP_MESH_DEVICE` or `MESH_DEVICE`. |
| `--duration_sec` | `float` | `10.0` | Target audio length in seconds. |
| `--infer_steps` | `int` | auto | Diffusion steps. Default **8** turbo, **50** base/sft. |
| `--seed` | `int` | `0` | RNG seed (latents + sampling). |
| `--guidance_scale` | `float` | auto | DiT CFG strength. Default **7** base/sft, **1** turbo (turbo ignores CFG >1). |
| `--out` | `str` | `ttnn_out.wav` | Output WAV path. |
| `--pytorch-lm` / `--no-pytorch-lm` | bool | off | Host PyTorch 5 Hz LM instead of TTNN causal LM. |
| `--torch-vae` | flag | off | PyTorch Oobleck decode instead of TTNN tiled VAE. |
| `--use-trace` / `--no-use-trace` | bool | **on** | TTNN trace + 2 command queues for preprocess + DiT body. Auto-off for ≥30s mesh long clips. |
| `--clarity` / `--no-clarity` | bool | off | Mesh quality preset: ADG guidance, wider VAE overlap, BF16 VAE at ≥30s. |
| `--use-adg` / `--no-use-adg` | bool | auto | DiT CFG: **ADG** (default base/sft) vs **APG** (default mesh without clarity). |
| `--no-use-cot-caption` | flag | off | Skip LM caption rewrite; use exact `--prompt` for DiT (multi-instrument lists). **Default: LM CoT caption.** |
| `--llm-debug` | flag | off | Extra constrained-decoding debug logs. Tokens/sec always in KEY METRICS when LM runs. |


**Common environment overrides:**

| Variable | Purpose |
|----------|---------|
| `ACE_STEP_MESH_DEVICE` / `MESH_DEVICE` | Default mesh SKU if `--mesh-device` omitted |
| `ACE_STEP_MAX_AUDIO_CODES` | LM audio-code planning cap (auto **350** for 60s via demo) |
| `ACE_STEP_DETOK_CHUNK_CODES` | Max codes per TTNN detokenizer forward (default **200**, L1 limit) |
| `ACE_STEP_PYTORCH_DETOK` | **Auto `=1`** when expect >200 codes. HF global detok (default for long clips). **`=0`** forces TTNN chunked detok — debug only, bad long-clip audio. **`=1`** always HF |
| `ACE_STEP_TORCH_DETOK_HINTS=1` | Round-trip TTNN detok hints through host torch (≤200-code path only; irrelevant when HF detok is active) |
| `ACE_STEP_VAE_QUALITY=1` | Force BF16 VAE decode on mesh |
| `ACE_STEP_DEMO_PERF_LOG=0` | Disable `[ace_step_v1_5][perf]` timing tables (on by default; set `=1` to force on) |
| `ACE_STEP_DISABLE_WEIGHT_CACHE=1` | Disable in-memory weight reuse |
| `HF_TOKEN` | HuggingFace download token |

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
| Audio-code detokenizer (`forward_traced`) | **Yes** (≤200 codes) | TTNN single-chunk path; **>200 codes use HF PyTorch** (no trace) |
| **Lyric transformer (8L)** | **Yes** | Inside ``forward_payload_traced`` capture (shape-keyed) |
| **Timbre transformer (4L)** | **Yes** | Same |
| Condition payload (`forward_payload_traced`) | **Yes** | Lyric + timbre + text + concat on CQ0; CQ1 input refresh |
| Context latents + chunk mask (`ctx_concat_traced`) | **Yes** | ``concat([src_latents, chunk_mask], dim=-1)`` |
| Cover / LM hints for `src_latents` | **TTNN on device** (≤200 codes) | `precomputed_lm_hints_25Hz_tt`; long streams use HF detok → host hints. Debug: `ACE_STEP_TORCH_DETOK_HINTS=1` |
| Condition `enc_mask` | **Host + device** | NumPy for handler; ``upload_enc_mask_dev`` mirrors mask on device |
| DiT CFG batch concat (pre-loop) | **No (eager)** | One-shot ``enc``/``ctx``/null setup; traced replay caused audible noise |
| DiT ``compute_temb_tp`` | **No (eager)** | Precomputed per step; **streamed** into body trace via ``ttnn.copy`` on ``temb_buf``/``tp_buf`` |
| DiT cross-attention SDPA mask | **In DiT trace** | ``_E2EDenoiseTrace.mask_buf`` when built from ``enc_mask`` |
| DiT denoise body (`_E2EDenoiseTrace`) | **Yes** | ``patch_embed`` + DiT core + ``output_head`` on CQ0 |
| DiT pre-step row cast + CFG dup | **No (eager)** | Per-step before body trace replay |
| DiT CFG / ADG / APG + TTNN | **No (eager)** | After ``release_trace_only`` each step (allocator + ADG ``sigma`` on host) |
| DiT warmup (first 2 TTNN steps) | **No (eager)** | Prime program cache before first capture |
| VAE `decode_tiled` | **No (eager)** | All tiles eager; traced decode was not bit-accurate → noise (``decode_chunk_traced`` kept for tests only) |

#### Stages that stay eager or host-only (by design)

| Stage | Why | Demo impact if traced / forced TTNN-only |
|-------|-----|------------------------------------------|
| **5 Hz LM weight init** | Not a replay graph | N/A |
| **LM constrained FSM** | Python control flow + dynamic token masks | N/A |
| **LM CFG combine** | Skipped when ``lm_cfg_scale=1`` | N/A |
| **Audio-code detok** (>200 codes) | TTNN chunked path lacks global attention | **Bad long-clip audio** — use default HF detok |
| **DiT CFG enc/ctx concat** | Traced prep not bit-accurate vs eager | Audible noise |
| **DiT full-step trace** | Opt-in only (``use_full_step``); default is body trace + eager post | Noise / smear |
| **VAE decode** | Traced replay not bit-accurate vs eager | Hiss → ``--torch-vae`` |
| **DiT warmup (2 steps)** | Capture priming | N/A |
| **Cover hint selection** | Host branch | N/A |
| **VAE multi-tile overlap-add** | Trace replay PCC / audible noise | Hiss on long decode |
| **DiT Euler + APG/ADG** | Host staging between TTNN body replays | Noise if fused incorrectly |

---


## PCC and module tests

All tests live under `tests/`. PCC tests compare TTNN outputs to PyTorch references and assert **PCC ≥ 0.99** via `assert_pcc_print` in `tests/_dit_decoder_pcc_common.py`.


Run from repo root with `--confcutdir` so pytest loads `ace_step_v1_5/conftest.py`:

```bash
pytest models/experimental/ace_step_v1_5/tests \
  --confcutdir=models/experimental/ace_step_v1_5/tests -q
```


#### Recorded PCC baselines (Blackhole QB, 2×2 mesh)

Measured on local **BH_QB** (4× Blackhole, ``acestep-v15-base`` HF checkpoints, May 2026).
Pearson PCC vs PyTorch/HF reference; thresholds are the test assert floors.

| Module | Test / case | Production shape | BH_QB PCC | p150 PCC | Threshold | Device |
|--------|-------------|------------------|-----------|----------|-----------|--------|
| Condition encoder | ``condition_enc_15s`` | ``enc [1,187,2048]`` | **0.999882** | **0.999882** | 0.97 | 1×1 |
| Condition encoder | ``condition_ctx_15s`` | ``ctx [1,375,128]`` | **0.999999** | **0.999999** | 0.99 | 1×1 |
| Condition encoder | ``condition_enc_30s`` | ``enc [1,172,2048]`` | **0.999883** | **0.999882** | 0.97 | 1×1 |
| Condition encoder | ``condition_ctx_30s`` | ``ctx [1,750,128]`` | **0.999999** | **0.999999** | 0.99 | 1×1 |
| Audio-code detokenizer | ``audio_detokenizer_15s_75codes`` | ``[1,375,64]`` @ 25 Hz | **0.986272** | **0.986272** | 0.97 | 1×1 |
| Audio-code detokenizer | ``audio_detokenizer_30s_150codes`` | ``[1,750,64]`` @ 25 Hz | **0.986632** | **0.986632** | 0.97 | 1×1 |
| VAE ``decode_tiled`` | ``vae_decode_tiled_15s`` | 375 latent frames → 720k audio samples | **0.994937** | **0.994761** | 0.98 | 1×1 |
| VAE ``decode_tiled`` | ``vae_decode_tiled_30s`` | 750 latent frames (overlap=14) | **0.995019** | **0.994956** | 0.98 | 1×1 |
| DiT denoise loop (no CFG) | ``dit_denoise_loop_15s_no_cfg`` | 375 frames, 20 TTNN steps | *run test* | _mesh-only_ | 0.92 | BH_QB mesh |
| DiT denoise loop (no CFG) | ``dit_denoise_loop_30s_no_cfg`` | 750 frames, 5 TTNN steps† | **0.995483** | _mesh-only_ | 0.90 | BH_QB mesh |
| DiT denoise + CFG + APG | ``dit_denoise_loop_30s_cfg_apg`` | 750 frames, 5 steps, gs=7 | **0.992419** | _mesh-only_ | 0.85 | BH_QB mesh |

``test_dit_denoise_loop_pcc.py`` defaults to 20 steps; the 30 s row above used ``ACE_STEP_DIT_DENOISE_PCC_STEPS=5``. ``test_dit_denoise_loop_pcc_cfg.py`` also defaults to 5 steps.
