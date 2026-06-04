# ACE-Step v1.5 demo guide

How `run_prompt_to_wav.py` runs on **BH_QB** (and other SKUs): session passes, device lifecycle, pipeline reuse, perf buckets, and tuning.

**Entry point:** `models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py`

| File | Role |
|------|------|
| `demo_session.py` | Session state, preprocess cache, pipeline/mesh reuse policy |
| `ace_step_perf_log.py` | Per-pass RUN SUMMARY + end-of-session SESSION SUMMARY |
| `tt_device.py` | Mesh SKU, split-device lifecycle, readback, CFG helpers |
| `runner/performant_runner.py` | Performant runner for perf tests (not wired into CLI) |

---

## Quick start (BH_QB, recommended)

```bash
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py \
  --mesh-device BH_QB \
  --variant acestep-v15-base \
  --duration_sec 30 --infer_steps 30 \
  --guidance_scale 7 --use-trace \
  --warmup \
  --prompt "Electronic dance track with deep bass and bright synth lead" \
  --out /tmp/ttnn_wav.wav
```

- **`--warmup`** — one cold compile pass, then timed generation in the same process.
- Pass 1+ skips LM/preprocess when **prompt + duration + seed** match a prior pass.
- Pass 1+ **reuses the DiT mesh and pipeline** when preprocess cache hits (see [Pipeline reuse](#pipeline-reuse)).

Perf logging is **on by default** on mesh SKUs. Use `--perf-log` or `ACE_STEP_DEMO_PERF_LOG=1` elsewhere.

---

## Session passes

**Session mode** activates with `--warmup`, `--repeat N` (N > 1), or `--serve` (stub).

Each pass: preprocess (maybe) → DiT denoise → VAE decode → (maybe) write WAV.

| CLI | Passes | Notes |
|-----|--------|-------|
| Default | 1 | Single `demo_total`; writes WAV |
| `--warmup` | 2 | Pass 0: `warmup_total` (compile, no WAV). Pass 1: `demo_total` (timed, WAV) |
| `--warmup --repeat 3` | 4 | Warmup + `demo_total`, `demo_total_2`, `demo_total_3` |
| `--repeat 2` | 2 | Two timed passes, no warmup |
| `--fast-preprocess` | 1 | Disables `--warmup` / `--repeat` (requires full handler path) |

**One-time per process:** `handler_init` (~32 s) — loads `AceStepHandler` + 5 Hz LM; shown in SESSION SUMMARY, not in each pass total.

**Preprocess cache:** If pass N+1 has the same **prompt, duration_sec, seed**, the demo logs `reusing cached preprocess tensors` and skips `five_hz_lm_generate` + condition encode.

---

## Pipeline reuse

**Pipeline reuse** means the demo keeps the **DiT mesh**, `AceStepV15TTNNPipeline`, TTNN VAE, and trace buffers alive across session passes instead of closing and rebuilding them.

### Default behavior (since recent session work)

When the **next pass can reuse cached preprocess** (same prompt / duration / seed), the demo:

1. **Keeps** the 2×2 DiT mesh open (`demo_session.dit_dev`)
2. **Reuses** `AceStepV15TTNNPipeline` → skips **`dit_pipeline_init`** (~2 s)
3. **Reuses** TTNN VAE → skips **`vae_init`** (~0.3 s)
4. **Retains** DiT trace persistent buffers → **`recapture`** instead of full capture on denoise

Log lines to expect:

```
[ace_step_v1_5] keeping DiT mesh open for next pass (reuse pipe/VAE/trace; cached preprocess on next pass)
[ace_step_v1_5] reusing DiT mesh device from session
[ace_step_v1_5] reusing DiT pipeline from session
```

Implementation: `AceStepDemoSession.should_keep_dit_mesh_open()` in `demo_session.py`; trace uses `release_trace_only()` (not full `release()`) when keeping the mesh.

### When reuse does *not* happen

| Condition | Action |
|-----------|--------|
| Next pass needs **new preprocess** (different prompt / duration / seed) | Must **close** DiT mesh (1×1 preprocess cannot coexist with 2×2 DiT on BH_QB) |
| **`--close-dit-between-passes`** | Force close/reopen every pass (legacy / debug) |
| **`--warmup-prompt`** differs from `--prompt` | Pass 1 cache miss → mesh closes before preprocess |
| Last pass in session | `demo_session.release()` closes everything |

### What reuse saves vs what still runs

| Reused (setup) | Still runs every generation |
|----------------|----------------------------|
| DiT mesh, pipeline weights, programs | `dit_denoise_loop` (full diffusion) |
| TTNN VAE instance | `vae_decode` |
| Trace buffers (recapture) | Latent noise from `--seed` |

Reuse is a **latency** optimization only — same model math.

### Measured impact (BH_QB, 30 s, 30 steps, base, trace, CFG 7)

| Metric | Without reuse (pass 1 after warmup) | With pipeline reuse |
|--------|-------------------------------------|---------------------|
| `dit_pipeline_init` | ~2124 ms | **0 ms** |
| `vae_init` | ~345 ms | **0 ms** |
| **`demo_total` wall** | ~12.2 s | **~9.0 s** |

Typical steady-state breakdown at ~9 s:

| Module | ~Time | Share |
|--------|-------|-------|
| `dit_denoise_loop` | 4.6 s | 51% |
| `vae_decode` | 3.7 s | 41% |
| `(other/overhead)` | 0.8 s | 8% |

---

## BH_QB device lifecycle (one pass)

On multi-chip SKUs, preprocess and inference use **sequential device sessions** on the same four cards — not one continuous device handle.

<a id="bh-qb-two-phase-device-lifecycle"></a>

```
  [Host]  handler_init (once per process)
     │
     ▼
  Phase A — 1×1 device, 1 CQ
    five_hz_lm_generate → qwen/condition → preprocess_readback → host tensors
     │ close 1×1
     ▼
  Phase B — 2×2 mesh, 2 CQ (+ trace)
    dit_pipeline_init (once; reused on later passes)
    dit_mask_prep → dit_denoise_loop → vae_init / vae_decode
     │ keep mesh open OR close (see Pipeline reuse)
     ▼
  Next pass: cached preprocess → Phase B only (if reuse)
```

**Why two phases?**

1. **Topology** — LM/Qwen/condition fit 1×1; DiT/VAE use the full mesh (`tt_device.py`: preprocess on 1×1, DiT on mesh).
2. **Command queues** — preprocess opens with **1 CQ**; traced DiT needs **2 CQs**. CQ count is fixed at device open → preprocess must close before DiT mesh opens with trace layout (`transition_preprocess_to_dit_device()`).
3. **Handoff** — conditioning tensors copied to **host** (`preprocess_readback`) before preprocess device closes; DiT mesh re-stages them for denoise.

**Legacy CPU preprocess on mesh:** `ACE_STEP_MESH_HOST_PREPROCESS=1` skips TTNN Phase A on mesh.

**Single-chip (`P150`):** no split; one device for the whole pipeline.

More detail: sections in `tt_device.py` and comments in `run_prompt_to_wav.py` around mesh open/transition.

---

## Perf modules

### Preprocessing (text → DiT conditioning)

Runs before DiT mesh open on BH_QB (or on single device before denoise on P150).

| Module | Notes |
|--------|-------|
| `handler_init` | Once per process; amortized in SESSION SUMMARY |
| `five_hz_lm_generate` | 5 Hz LM + CoT + audio codes (default path) |
| `qwen_encoder_init` | First TTNN Qwen + detokenizer build |
| `handler_preprocess` | Official handler payload prep |
| `condition_encoder` | TTNN condition tensors |
| `preprocess_readback` | BH_QB only: 1×1 → host before mesh open |

**Skipped** on cache hit (pass 1+ with same prompt/duration/seed).

### Inference (diffusion + decode)

| Module | Notes |
|--------|-------|
| `dit_pipeline_init` | Build `AceStepV15TTNNPipeline`; **0 ms when pipeline reused** |
| `dit_mask_prep` | Encoder SDPA masks |
| `dit_denoise_loop` | Euler loop × `--infer_steps`; dominant cost |
| `vae_init` | TTNN VAE first init; **0 ms when VAE reused** |
| `vae_decode` | Tiled latent → waveform |
| `(other/overhead)` | temb upload, trace capture, sync, staging |

---

## Perf output

### RUN SUMMARY (each pass)

Per-pass module breakdown + parameters (`frames`, `infer_steps`, `session_pass`, etc.).

### SESSION SUMMARY (after last pass)

Rollup across init + all passes:

- `handler_init` subtotal
- Per-pass wall times (`warmup_total`, `demo_total`, …)
- Module rollup **summed across passes** (warmup modules omitted unless `--warmup-perf`)
- `steady-state` — last timed pass wall time
- `SESSION (init + passes)` vs `SESSION (process wall)`

---

## Performance tuning

### Cold vs steady state

| Phase | Typical cost | Mitigation |
|-------|--------------|------------|
| `handler_init` | ~32 s | Long-lived process / `--warmup` session |
| `warmup_total` | ~70–85 s | One-time compile; use `--warmup` |
| **`demo_total` (steady)** | **~9 s @ 30 s / 30 steps** | Pipeline reuse + preprocess cache |

### Reduce `dit_denoise_loop`

Scales with **`frames`** (= `duration_sec × 25`) and **`infer_steps`**. On BH_QB with CFG, each step often runs **two** B=1 forwards (sequential CFG).

| Lever | Effect |
|-------|--------|
| `--infer_steps 25` | ~linear cut |
| `--cfg_interval_end 0.85` | Fewer CFG steps late in chain |
| `--variant acestep-v15-turbo --infer_steps 8` | Much faster; different model |
| `--perf-log-steps` | Per-step ms for profiling |

### Reduce `vae_decode`

Scales with duration (more latent tiles). Future: VAE trace. Tune `--vae-chunk-latents` / overlap if L1 allows.

### Quality vs speed (audio accuracy)

If WAV quality regressed vs earlier runs:

- `--no-experimental-5hz-ttnn-causal-lm` (official LM CFG)
- Confirm `--guidance_scale 7` on base
- Try `ACE_STEP_MESH_USE_ADG=1 --use-adg` on mesh
- Drop `--clarity` for A/B
- Avoid `--warmup-prompt` different from `--prompt` (stale cache confusion)

Trace and pipeline reuse do **not** improve quality; they reduce setup time only.

---

## Session CLI flags

| Flag | Description |
|------|-------------|
| `--warmup` | Cold compile pass then timed pass in one process |
| `--warmup-prompt` | Warmup caption (default: same as `--prompt`) |
| `--warmup-perf` | Log modules for warmup pass |
| `--repeat N` | N timed generations after warmup |
| `--close-dit-between-passes` | Force close DiT mesh every pass (disable pipeline reuse) |
| `--serve` | Interactive stdin loop (stub; not fully wired) |
| `--perf-log` / `--perf-log-steps` | Perf logging / per-Euler-step lines |

---

## Comparison with other model demos

Most `models/demos/*` scripts are single-shot or use `BenchmarkProfiler` / `PerformantRunner` (warmup at construct, then `run()`).

ACE-Step is unique: **four subsystems** (LM → condition → DiT → VAE), **split 1×1 / 2×2 devices** on BH_QB, **session + preprocess cache + pipeline reuse**, and custom `ace_step_perf_log.py`. The warmup *idea* matches SwinV2 / `AceStepPerformantRunner`; the CLI adds mesh orchestration. The performant runner is **not** wired into `run_prompt_to_wav.py`.

---

## See also

- `README.md` — quick start, variants, full CLI table
- `tests/test_tt_device_mesh.py` — warmup specs, session summary, `should_keep_dit_mesh_open`
