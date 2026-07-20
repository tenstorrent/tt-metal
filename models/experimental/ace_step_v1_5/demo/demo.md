# ACE-Step v1.5 demo guide

How `run_prompt_to_wav.py` runs on **BH_QB** (and other SKUs): device lifecycle, perf buckets, and tuning.

**Entry point:** `models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py`

For quick start, recipes, and the full CLI table see [`../README.md`](../README.md).

| File | Role |
|------|------|
| `demo_session.py` | In-process state (devices, handlers, weight/pipeline cache for one run) |
| `ace_step_perf_log.py` | RUN SUMMARY + optional SESSION SUMMARY timing tables |
| `tt_device.py` | Mesh SKU, split-device lifecycle, re-exec handoff, readback |
| `serve_prompt_to_wav.py` | HTTP service (weights loaded once; multiple requests) |
| `runner/performant_runner.py` | Performant runner for perf tests (not wired into CLI) |

---

## Quick start (BH_QB)

Fast everyday path (turbo):

```bash
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py \
  --mesh-device BH_QB \
  --variant acestep-v15-turbo \
  --lm_variant acestep-5Hz-lm-1.7B \
  --duration_sec 15 \
  --infer_steps 8 \
  --guidance_scale 1 \
  --prompt "Electronic dance track with deep bass and bright synth lead" \
  --seed 0 \
  --out /tmp/turbo_15s.wav
```

Upstream RTF comparison (170.64 s / 60 steps / guidance 15 / Euler / base):

```bash
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py \
  --mesh-device BH_QB \
  --upstream-benchmark \
  --lm_variant acestep-5Hz-lm-1.7B \
  --out /tmp/upstream_rtf_compare.wav
```

Each CLI invocation is **one generation** → one WAV → process exit (devices released in `finally`).

Perf logging is **on by default** on mesh SKUs. On single-chip SKUs set `ACE_STEP_DEMO_PERF_LOG=1` to enable `[ace_step_v1_5][perf]` tables.

---

## One run lifecycle

```
  [Host]  handler_init (AceStepHandler + 5 Hz LM tokenizer/FSM; TTNN weights upload)
     │
     ▼
  Phase A — 1×1 preprocess device (BH_QB)
    five_hz_lm_generate → Qwen3 caption → audio-code detokenizer
    (condition encode here OR deferred — see below)
     │ close 1×1 + os.execv handoff (split mesh SKUs)
     ▼
  Phase B — 2×2 DiT mesh
    open_dit_device → condition encode (if deferred) → dit_denoise_loop → vae_decode → WAV
     │
     ▼
  demo_session.release() — close mesh, free TTNN objects
```

**Long clips (≥750 latent frames, ~30 s+):** TTNN condition encode is **deferred** to Phase B so enc/ctx tensors are built on the full mesh (avoids 1×1 readback drift). The demo pickles a preprocess payload and **re-execs** itself with `--ace-step-dit-handoff` so Blackhole opens 2×2 in a fresh process (cannot transition 1×1 → 2×2 in-process).

Healthy long-clip logs:

```
[ace_step_v1_5] defer TTNN condition encoder to DiT mesh ...
[ace_step_v1_5] DiT: re-exec for full mesh (BH_QB), handoff=/tmp/ace_step_dit_handoff_....pkl
[ace_step_v1_5] DiT handoff: deferred condition payload loaded ...
[condition] backend=ttnn on DiT mesh (...)
```

**Single-chip (`P150`):** no split; one device for the whole pipeline.

**Legacy host preprocess on mesh:** `ACE_STEP_MESH_HOST_PREPROCESS=1` skips TTNN Phase A.

---

## BH_QB device lifecycle (detail)

On multi-chip SKUs, preprocess and inference use **sequential device sessions** — not one continuous handle across 1×1 and 2×2.

1. **Topology** — LM/Qwen/detokenizer fit 1×1; DiT/VAE use the full mesh (`tt_device.py`).
2. **Command queues** — preprocess opens with **1 CQ** (or 2 with trace on short clips); traced DiT on short clips uses **2 CQs** on the mesh after re-exec.
3. **Handoff** — for standard short clips, condition tensors are read back to host before re-exec; for deferred long clips, a **payload pickle** is handed off instead of final enc tensors.

Implementation: `ace_step_reexec_for_dit_mesh()` in `utils/tt_device.py`; child entry via hidden `--ace-step-dit-handoff`.

---

## In-process reuse (within one run)

During a **single** process (after handoff, on Phase B):

| Cached object | When reused |
|---------------|-------------|
| `AceStepV15TTNNPipeline` | Same `frames` + long-clip quality key |
| `TtOobleckVaeDecoder` | Same `frames` + VAE quality key |
| DiT trace buffers | Recapture on shape match when `--use-trace` (auto-off ≥30 s on mesh) |

There is **no multi-pass CLI** (`--warmup`, `--repeat`, etc. were removed). To run twice, invoke the script twice. For weights loaded once across many requests, use `serve_prompt_to_wav.py`.

---

## Perf modules

### Preprocessing (text → DiT conditioning)

| Module | Notes |
|--------|-------|
| `handler_init` | Once per process; ~30 s (LM tokenizer + handler setup) |
| `five_hz_lm_generate` | TTNN 5 Hz LM + CoT + audio codes (default) |
| `qwen_encoder_init` | First TTNN Qwen + detokenizer build |
| `handler_preprocess` | Handler payload / LM hints assembly |
| `condition_encoder` | TTNN (1×1 or DiT mesh depending on clip length) |
| `preprocess_readback` | Host tensor copy when condition ran on 1×1 |

### Inference (diffusion + decode)

| Module | Notes |
|--------|-------|
| `dit_pipeline_init` | Build `AceStepV15TTNNPipeline` |
| `dit_mask_prep` | Encoder SDPA masks |
| `dit_denoise_loop` | TTNN × `--infer_steps`; dominant cost |
| `vae_init` | TTNN VAE first init in this run |
| `vae_decode` | Tiled latent → waveform |
| `(other/overhead)` | temb upload, trace capture, sync, staging |

At ≥30 s on mesh, trace is forced off: latents + TTNN run on **host CPU** while DiT **forwards** stay on TTNN.

---

## Perf output

### RUN SUMMARY

Printed at end of each run when perf logging is enabled (`demo_total` label):

- Per-module milliseconds
- **KEY METRICS** table: Wall / LM / DiT / VAE / **RTF** / Tokens/sec
- **RTF** = `audio_duration / wall` (same as [upstream ACE-Step](https://github.com/ace-step/ACE-Step#%EF%B8%8F-hardware-performance); higher = faster)
- **RTF COMPARISON** table: this-run RTF vs upstream **A100** / **RTX 3090** (27- and 60-step reference rows)
- Parameters: `frames`, `infer_steps`, `duration_sec`, backends, etc.

### SESSION SUMMARY

Rollup for the single process: `handler_init` + `demo_total` wall time.

---

## Performance tuning

### Cold vs steady state

| Phase | Typical cost | Notes |
|-------|--------------|-------|
| `handler_init` | ~30 s | Every new process (includes tokenizer load) |
| First `dit_pipeline_init` + compile | varies | Program cache fills on first denoise |
| **`demo_total` (turbo 15 s)** | ~few s denoise + VAE | After init |
| **`demo_total` (turbo 60 s)** | longer LM + DiT + VAE | Trace off; host TTNN |

Second invocation in a **new process** repeats `handler_init` unless you use `serve_prompt_to_wav.py` or disk/RAM weight cache (see README).

### Reduce `dit_denoise_loop`

Scales with **`frames`** (`duration_sec × 25`) and **`infer_steps`**. Turbo at `guidance_scale=1` skips CFG (one forward per step).

| Lever | Effect |
|-------|--------|
| `--infer_steps 8` + turbo | Default fast path |
| `--variant acestep-v15-base --infer_steps 50` | Higher quality, ~6× more steps |
| Shorter `--duration_sec` | Fewer latent frames |

### Reduce `vae_decode`

Scales with duration (more latent tiles). Long mesh clips use wider overlap automatically. Try `--torch-vae` to A/B hiss vs TTNN.

### Quality vs speed

| Goal | Suggestion |
|------|------------|
| Long clips on BH_QB | `acestep-v15-turbo`, `--guidance_scale 1`, `--infer_steps 8` |
| Cleaner base/sft at 30–60 s | `--clarity`, `acestep-5Hz-lm-4B`, `--torch-vae` if hiss persists |
| Multi-instrument prompt | `--no-use-cot-caption` if instruments drop from LM caption |
| Debug TTNN vs trace | `--no-use-trace` (short clips only; long clips auto-off) |

---

## CLI flags (current)

Full table in [`../README.md`](../README.md#demo-cli-reference-demorun_prompt_to_wavpy).

| Flag | Description |
|------|-------------|
| `--prompt` | **Required** text caption |
| `--variant` / `--lm_variant` | DiT + 5 Hz LM checkpoints |
| `--mesh-device` | SKU (`BH_QB`, `P150`, …) |
| `--duration_sec` / `--infer_steps` / `--guidance_scale` / `--seed` / `--out` | Generation controls |
| `--pytorch-lm` / `--no-pytorch-lm` | Host PyTorch LM vs TTNN (default TTNN) |
| `--torch-vae` | PyTorch VAE decode vs TTNN |
| `--use-trace` / `--no-use-trace` | TTNN trace + 2 CQ (default on; auto-off long mesh clips) |
| `--clarity` / `--no-clarity` | Mesh quality preset |
| `--use-adg` / `--no-use-adg` | DiT CFG: ADG vs APG |
| `--no-use-cot-caption` | Exact `--prompt` for DiT (skip LM caption rewrite) |
| `--llm-debug` | Extra LM constrained-decoding logs |

**Removed (no longer in CLI):** `--warmup`, `--repeat`, `--warmup-prompt`, `--warmup-perf`, `--serve`, `--fast-preprocess`, `--close-dit-between-passes`, `--perf-log`, `--perf-log-steps`.

**Opt-in env overrides:** `ACE_STEP_PYTORCH_DIT=1`, `ACE_STEP_PYTORCH_CONDITION=1`, `ACE_STEP_MESH_HOST_PREPROCESS=1`, `ACE_STEP_DEMO_PERF_LOG=1`, `ACE_STEP_MAX_AUDIO_CODES`, etc. — see README.

---

## HTTP service (multiple generations)

```bash
python models/experimental/ace_step_v1_5/demo/serve_prompt_to_wav.py --port 8765
```

Loads weights once at startup; each POST reuses in-memory caches. Different entry point from the CLI script above.

---

## Comparison with other model demos

Most `models/demos/*` scripts are single-shot. ACE-Step adds **four subsystems** (LM → condition → DiT → VAE), **split 1×1 / 2×2 devices** on BH_QB with **process re-exec handoff**, and custom `ace_step_perf_log.py`. Internal DiT **warmup** (two eager TTNN steps before trace capture) is not a CLI flag — it happens inside the denoise loop when trace is enabled.

---

## See also

- [`../README.md`](../README.md) — how to use the demo, TTNN vs PyTorch table, variants
- `utils/tt_device.py` — mesh open, preprocess env, re-exec handoff
- `tests/test_tt_device_mesh.py` — mesh SKU helpers, session perf unit tests
