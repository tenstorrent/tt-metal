# VibeVoice-1.5B (TT-Metal experimental)

Reference PyTorch setup for porting [VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) to TTNN. The backbone is **Qwen2.5-1.5B** (28 layers, hidden 1536, GQA); plan to reuse or wrap [`models/tt_transformers/`](../../tt_transformers/) for `language_model`.

Weights and demo assets are **not** vendored in this tree. On first run, demos and tests download:

- **Model weights:** [`microsoft/VibeVoice-1.5B`](https://huggingface.co/microsoft/VibeVoice-1.5B) into
  `models/experimental/vibevoice/weights/VibeVoice-1.5B` (requires `huggingface_hub`).
- **Demo text + voices:** [vibevoice-community/VibeVoice](https://github.com/vibevoice-community/VibeVoice/tree/main/demo)
  (`demo/text_examples` and `demo/voices`) into `models/experimental/vibevoice/resources/` via
  `common/resource_utils.py`.

Override the checkpoint location with:

```bash
export VIBEVOICE_MODEL_PATH=/path/to/VibeVoice-1.5B
```

## Supported Devices

Validated on Tenstorrent Blackhole:

| Device | Configuration |
|--------|---------------|
| **Blackhole P150** | 1 × Blackhole ASIC (`ARCH_NAME=blackhole`) |

## Layout

```
vibevoice/
├── README.md
├── common/
│   ├── config.py            # paths, HF repo id, transformers pin
│   ├── model_utils.py       # resolve path + auto-download weights
│   └── resource_utils.py    # download demo text/voices from upstream GitHub
├── demo/
│   ├── demo.py                  # TT-only inference entry point (writes wav + meta)
│   └── perf_metrics.py          # generate() timing summary + ISL cropping
├── reference/               # vendored 1.5B-only torch model (from VibeVoice repo)
│   ├── modular/             # config + modeling
│   ├── processor/           # tokenizer/audio processor
│   ├── schedule/            # DPM solver
│   └── lm_runner.py         # NOT vendored (ours): CPU fp32 LM swap-in for PCC tests
├── resources/               # auto-downloaded demo assets (gitignored content)
│   ├── voices/              # from github .../demo/voices
│   └── text/                # from github .../demo/text_examples
├── weights/                 # auto-downloaded HF checkpoint (gitignored content)
├── tests/
│   ├── conftest.py              # shared fixtures (weights/resources download, config, LM state)
│   ├── pcc/                     # per-component + end-to-end correctness
│   └── perf/                    # Tracy device-perf + single-step prefill/decode dumps
└── tt/                          # TTNN port
```

Everything is imported by its full path from the tt-metal root, e.g.
`from models.experimental.vibevoice.reference.processor.vibevoice_processor import VibeVoiceProcessor`.

## Dependencies

The reference processor also pulls **Qwen/Qwen2.5-1.5B** tokenizer assets from the Hugging Face
cache; they are not bundled in the VibeVoice-1.5B checkpoint. Reference parity requires
**transformers 4.51.3** — 4.57 changed the `generate()` KV-cache API.

## Quick start (from tt-metal root)

```bash
export PYTHONPATH=$(pwd)

# PCC tests (auto-download weights; skipped if download fails)
pytest models/experimental/vibevoice/tests/pcc/ -v
```

## TTNN demo (on device)

`demo/demo.py` runs on-device TTNN inference (no HuggingFace reference model) and writes
`{output_dir}/{demo_id}/{demo_id}_tt.wav`. It is text-driven: `--text <path>` for a custom script,
or `--demo <id>` as a shortcut for `resources/text/<id>.txt`. Multi-speaker demos auto-enable
voice cloning from `resources/voices/`.

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
export ARCH_NAME=wormhole_b0 WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml   # or blackhole

# Default demo (default script, eager — no trace)
python models/experimental/vibevoice/demo/demo.py

# Multi-speaker demo, cap the AR loop at 32 tokens, verbose stage/timing logs
python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_45min --max_new_tokens 32 --debug
```

### Trace (default)

Fused-frame trace is **on by default** (`--trace` / `VV_TRACE_SEGMENT=1`): ttnn-captures the whole
steady-state speech-diffusion frame (neg-LM + diffusion + post-diffusion + pos-LM) as one fully
device-driven graph — the "llama shape": positions self-advance on device, RoPE is gathered on
device, and the pos hidden is loop-carried — and replays it per frame. It gives **≈11–12 tok/s**
steady-state decode vs ≈2.4 tok/s eager on the 45-min climate demo, and opens the device with a
~1.4 GB trace region + 2 command queues. Pass `--no-trace` for eager decode.

```bash
python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_45min --max_new_tokens 32
python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_45min --max_new_tokens 32 --no-trace
```

| Flag | Env var | Scope | Notes |
|------|---------|-------|-------|
| `--trace` (default) | `VV_TRACE_SEGMENT=1` | whole segment, device-driven (llama shape) | fused frame replayed per frame; ~1.4 GB trace region + 2 CQs |
| `--no-trace` | `VV_TRACE_SEGMENT=0` | eager AR loop | for debugging / A/B |

### Environment variables

Every entry is an A/B switch: the default is the validated path, and flipping it falls back to the
older behaviour (or opts into an unvalidated one). Values are `1`/`0` unless stated.

| Variable                  | Default        | Purpose |
| ------------------------- | -------------- | ------- |
| `VV_TRACE_SEGMENT`        | `0` (`1` via `--trace`) | Whole-segment fused decode trace. The demo sets this from `--trace`/`--no-trace`. |
| `VV_CFG_BATCH2`           | `1` (on)       | Run the CFG negative and positive LM rows as one batch-2 decode (weights read once). Requires `VV_CAP_SPLIT=1`. |
| `VV_CAP_SPLIT`            | `1` (on)       | Capture the frame as separate diffusion / LM graphs instead of one graph. |
| `VV_FUSE_FRAME_OUT`       | `1` (on)       | Append the constrained-argmax token to the frame's audio so one D2H returns both. `0` reads them separately. |
| `VV_DEV_NOISE`            | `1` (on)       | Gather the frame's diffusion init noise on device from a pre-uploaded table. `0` restores the per-frame host write. |
| `VV_DEV_ENC_AUDIO`        | `1` (on)       | Upload each voice row once and let the encode replay gather its own chunk on device. `0` uploads per chunk. |
| `VV_DEV_ENC_LAT`          | `0` (off)      | Accumulate voice-clone encode latents in-capture on device, so the host reads one row instead of one per chunk (663 → 4 transfers). Verified bit-identical; off by default because the win (~199 ms of a ~17.6 s voice-clone prefill) is inside run-to-run noise. |
| `VV_FUSED_ROPE`           | `0` (off)      | Fused bf16 `rotary_embedding_llama` decode RoPE + on-device cos/sin tables. Off by default: a 100-min acceptance run showed the speaking rate accelerating (median 208 wpm vs 153) even though every energy/spectral gate passed. |
| `VV_FP32_ROPE`            | `1` (on)       | Keep the traced decode on fp32 RoPE rows written per position. |
| `VV_TTNN_RANDN`           | `0` (off)      | Draw diffusion init noise and the acoustic fix-std jitter with `ttnn.randn` on device instead of torch. Off by default: it is a different generator, so renders stop matching the torch reference and PCC comparison against it is no longer meaningful. |
| `VV_STREAM_AUDIO`         | unset (off)    | Path for a raw fp32 stream of each frame's audio. Bounds host RAM on long renders (a 100-min render otherwise peaks ~1.15 GB holding the chunks plus the concatenated copy) and preserves partial audio if the process dies. |
| `VV_LOG_TRAJ`             | unset (off)    | Path for a per-frame CSV of hidden/audio RMS, peak, and CFG cosine. Long-form diagnostics. |
| `VV_DEBUG`                | `0` (off)      | Verbose per-stage logs. `--debug` sets this together with `VV_PROFILE`. |
| `VV_PROFILE`              | `0` (off)      | Device-synced timing breakdown per stage. |
| `VV_PROFILE_PREFILL`, `VV_PROFILE_SPEECH_FRAME`, `VV_PROFILE_DIFFUSION` | `0` (off) | Wrap the n-th prefill chunk / speech frame / diffusion in Tracy signposts. Each has a matching `*_EXIT=1` to stop right after the profiled unit. |

### Host operations (trace-accelerated run)


**1. Steady-state frame — runs every AR step.**

At the defaults this is **one D2H per frame** plus host bookkeeping: the diffusion noise is gathered
on device, and the audio and token come back in a single transfer.

| Op | Type | Used for |
|----|------|----------|
| fused audio+token readback | D2H ×1 | one `to_torch` returns `[audio …, token_idx]` (`VV_FUSE_FRAME_OUT=1`); the token half is what the AR loop blocks on, since a trace cannot branch |
| `_emit_audio` (append, or disk write) | hostCPU | accumulate frame audio into the waveform, or stream it out under `VV_STREAM_AUDIO` |
| `_gen_tokens.append` / `valid_ids[idx]` | hostCPU | token record; map the constrained-argmax **local** index to the global id (kept local so it survives the bf16 cast into the fused tensor) |
| host pos/neg mirror `+=1`, RoPE write ×4 | hostCPU + H2D ×4 | **only when `VV_FUSED_ROPE=0`** (the default): the mirrors exist solely to index the fp32 cos/sin tables. On the fused path the rows are gathered on device and none of this runs |

Removed since the earlier revision of this table: the per-frame diffusion-noise H2D (now an
in-capture gather from a pre-uploaded table, `VV_DEV_NOISE=1`) and the separate audio and token
D2Hs (now fused into one).

**2. Segment boundary — runs only when a new speaker segment starts.**

| Op | Type | Used for |
|----|------|----------|
| inter-segment token readbacks (`speech_end`/text/`speech_start` argmax) | D2H | advance AR control across boundary/text tokens |
| frame-0 `write_int` pos/neg + seed-hidden copy | H2D+D2D | rewind device positions; seed loop-carried hidden from segment-start hidden |
| `_sf_zero_conv` | H2D | reset acoustic/semantic conv streaming caches for the new segment |
| `_boot` writes (embed + RoPE for neg-prefill) | D2D (+ H2D ×2 when `VV_FUSED_ROPE=0`) | seed the negative-CFG condition for the segment's first frame. The embed is a device-to-device copy; only the fp32 RoPE rows touch the host |

**3. One-time — runs once per `generate()` call.**

| Op | Type | Used for |
|----|------|----------|
| voice-clone encode: audio up | H2D ×1 per speaker | one upload of the whole voice row; the traced chunk graph gathers its own chunk on device (`VV_DEV_ENC_AUDIO=1`). `0` restores one H2D per chunk |
| voice-clone encode: latents down | D2H per chunk, or ×1 per speaker | the chunk graph is ttnn-traced (`_ensure_encode_trace`) and replayed per chunk; the latent row comes back per chunk by default, or once per speaker under `VV_DEV_ENC_LAT=1` (663 → 4 transfers for the 4-speaker climate prompt). The trace is released before the LM prefill |
| scale/bias + `feats=(lat+bias)*scale` | hostCPU | normalize latents before the acoustic connector. `scale`/`bias` come from the checkpoint, so no reduction is needed |
| `reset_*_cache` | H2D | reset conv streaming caches for a fresh generation |
| `torch.randn(max_steps,…)` | hostCPU | pre-draw all diffusion init noise, then upload once as a gather table (RNG-aligned; `ttnn.randn` on device under `VV_TTNN_RANDN=1`) |
| first `_greedy_argmax` | D2H | first token after prefill |
| `cat(audio_chunks)` / build `sequences` / output | hostCPU | assemble final waveform + token sequence. Peaks at ~2× the audio (~1.15 GB for a 100-min render) — set `VV_STREAM_AUDIO` to halve it |

The prefill embed scatter is **no longer a host op**: splicing the voice embeds into the text
embeds' speech slots runs entirely on device as slice/concat over the mask's contiguous runs
(`_scatter_speech_embeds`), replacing a ~141 MB D2H + boolean-mask assign + ~141 MB H2D.

## Speaker similarity (SIM) test

`tests/pcc/test_e2e_sim.py` checks that on-device TTNN generation preserves the *cloned speaker's
identity*: it voice-clones a target speaker on TT, embeds the generated audio and the
reference/impostor voices with a speaker-verification (SV) model, and asserts the generated speech
is closer to the intended target than to any impostor (standard SIM-O verification), including a
4-speaker self-identification confusion matrix.

```bash
pytest models/experimental/vibevoice/tests/pcc/test_e2e_sim.py -v -s
```

**SV backend — why `microsoft/wavlm-base-plus-sv`, not the model from the paper.** The
[VibeVoice technical report](https://arxiv.org/abs/2508.19205) computes SIM with a **WavLM-large
fine-tuned** SV model — the UniSpeech `wavlm_large_finetune.pth` (WavLM-large backbone + an
ECAPA-TDNN x-vector head). We deliberately do **not** ship that model. Its code and checkpoint
([microsoft/UniSpeech](https://github.com/microsoft/UniSpeech), which in turn borrows from the
unlicensed [lawlict/ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN)) are licensed
**CC BY-SA 3.0**, whose *ShareAlike* clause requires derivative works to carry the same license —
incompatible with this repo's **Apache-2.0**. Instead the test uses
**`microsoft/wavlm-base-plus-sv`**, a WavLM x-vector head that ships with 🤗 transformers
(Apache-2.0), needs no extra dependency or separately-licensed checkpoint, and keeps the whole path
license-clean.

Trade-off: base_plus produces a *compressed* cosine scale (different speakers still score ~0.5-0.7,
vs the fine-tuned model's ~0.9 same-speaker / ~0 impostor separation). So the test asserts a
**relative** target-vs-impostor margin — the SV model must still rank the correct speaker first, by
a margin — which is robust to the compressed scale, rather than the paper's absolute same-speaker
threshold.

## Language model / chain PCC tests

The LM prefill and decode paths are validated as part of the **full prefill / decode chain** PCC
tests (vs a bf16 HuggingFace Qwen2 reference), plus a standalone decoder-layer regression. Shared
helpers live in `tests/pcc/pcc_helpers.py`; fixtures (`vv_config`, `lm_state`) are in
`tests/conftest.py`.

- **Decoder layer (regression):** `test_decoder_layer_pcc.py::test_decoder_layer_decode_pcc` —
  Devstral-style layer-0 decode; random hidden states `[1, 1, H]`, empty KV cache, positions 0–9,
  no prefill. Isolates decode SDPA at low cache depth (measured min PCC ≈ 0.99819; gate ≥ 0.9975).
- **Full prefill chain:** `test_prefill.py::test_full_prefill_chain_pcc` — the integrated
  prefill path (acoustic tokenizer → connector → scatter into embeddings → LM prefill →
  `last_hidden_state`) plus per-layer KV cache, vs the bf16 HF Qwen2 reference; synthetic-input ISL
  sweep. Speech embeds / KV are gated at `PCC >= 0.99`; LM hidden is gated on per-position
  **median** `>= 0.96` (flattened PCC can be pulled down by a few text-token outliers).
- **Full decode chain:** `test_decode.py::test_decode_ref_cond_frame_pcc` — **open-loop,
  per-stage parity** of the whole decode vs the fp32 reference over a teacher-forced stream. Each
  frame compares all three decode stages, each fed the *reference* input for that stage (open loop
  → per-stage error is isolated and cannot accumulate → PCC-gate-able):
  - **diffusion** — TT DPM sampler on the reference condition + shared noise vs the reference latent,
  - **chain** — TT acoustic decode → semantic encode → connectors on the reference latent vs the
    reference fused embed,
  - **LM** — TT LM vs the reference hidden.

  Chain and LM are strict per-frame `min PCC >= 0.99` (essentially exact on identical inputs). The
  diffusion latent is *distribution*-gated (no frame below `DIFF_LATENT_FLOOR`, at most
  `DIFF_LATENT_OUTLIER_FRAC` of frames below threshold): the DPM sampler is **separatrix-sensitive**
  for a rare, perceptually-inert subset of conditions — a benign input can push the discrete bf16
  trajectory across a contractive/expansive boundary — so a per-frame `min` would false-fail while
  the distribution gate still catches a real regression. The *closed* decode loop is intentionally
  not PCC-gated here (it's chaotic — a single separatrix frame cascades under feedback, latent PCC
  0.999 → 0.16 over ~24 frames); whole-loop fidelity lives in the e2e/WER tests, and the diffusion
  head/scheduler have their own PCC tests.

```bash
# Decoder-layer regression (fast)
pytest models/experimental/vibevoice/tests/pcc/test_decoder_layer_pcc.py -v -s

# Full prefill / decode chain
pytest models/experimental/vibevoice/tests/pcc/test_prefill.py \
       models/experimental/vibevoice/tests/pcc/test_decode.py -v -s
```

Individual component PCC tests (acoustic/semantic tokenizers, connector, diffusion head, DPM
scheduler, LM head) live alongside these in `tests/pcc/`.

## PCC Results

Measured on **Blackhole P150** against the PyTorch / HuggingFace reference paths
(values from `tests/logs/` suite runs + component re-measure).

| File Name | Test Case | PCC / metric |
|-----------|-----------|-------------:|
| `test_connector_pcc.py` | Acoustic connector | 0.99999803 |
| | Semantic connector | 0.99999753 |
| `test_dpm_scheduler_pcc.py` | DPM scheduler (10 steps) | 0.99993365 |
| `test_lm_head_pcc.py` | LM head last-token logits | 0.99995073 |
| `test_diffusion_head_pcc.py` | Diffusion head | 0.99985695 |
| `test_semantic_tokenizer_pcc.py` | Semantic tokenizer encode | 0.99943061 |
| `test_acoustic_tokenizer_pcc.py` | Acoustic encode | 0.99985584 |
| | Acoustic decode (scaled random latents) | 0.99974055 |
| | Acoustic decode (real encoder latents) | 0.99980000 |
| `test_decoder_layer_pcc.py` | Layer-0 decode min / mean (steps 0–9) | 0.99819 / 0.99858 |
| `test_decode.py` | Open-loop diffusion latent min / mean | 0.9996 / 0.9999 |
| | Open-loop chain (fused) min / mean | 0.9999 / 1.0000 |
| | Open-loop LM hidden min / mean | 0.9991 / 0.9995 |
| `test_e2e_wer.py` | Teacher-forced WER (`4p_climate_45min`, cap=512) | 0.0000 |
| `test_e2e_sim.py` | Voice-clone SIM (Carter vs best impostor) | 0.9914 / margin +0.3854 |
| | 4-speaker self-ID (Alice / Carter / Frank / Maya) | 0.9805 / 0.9914 / 0.9957 / 0.9761 |

### Prefill chain ISL sweep (`test_prefill.py`)

Speech embeds + KV gated at ≥ 0.99; LM hidden gated on per-position **median** ≥ 0.96.

| ISL | speech_PCC | hidden_med | kv_K_med | kv_V_med |
|----:|-----------:|-----------:|---------:|---------:|
| 32 | 0.999976 | 0.99643 | 0.99859 | 0.99718 |
| 64 | 0.999932 | 0.99705 | 0.99888 | 0.99769 |
| 128 | 0.999932 | 0.99814 | 0.99917 | 0.99800 |
| 256 | 0.999930 | 0.99408 | 0.99810 | 0.99701 |
| 512 | 0.999929 | 0.96330 | 0.99620 | 0.99709 |
| 1024 | 0.999924 | 0.98275 | 0.98640 | 0.99708 |
| 2048 | 0.999936 | 0.98755 | 0.95016 | 0.99691 |
| 4096 | 0.999939 | 0.99080 | 0.86048 | 0.99626 |
| 8192 | 0.999940 | 0.99232 | 0.86293 | 0.99581 |
| 16384 | 0.999915 | 0.99433 | 0.75053 | 0.99521 |
| 24000 | 0.999942 | 0.99529 | 0.69571 | 0.99499 |

> Post fused-RoPE K layout remapping. Speech / hidden_med / V stay high; **K median drops with
> ISL** (below the 0.99 KV gate from ISL≥1024 — worst layer often 13). Gate / remapping follow-up
> tracked separately from these measured values.

## Performance Summary

### Prefill / decode ISL sweep (`4p_climate_100min`)

Wall-clock `tests/perf/test_e2e_isl_sweep_perf.py` on Blackhole P150 with fused-frame trace
enabled (`VV_TRACE_SEGMENT=1`). Prompt cropped to each ISL after tokenization; warmup then timed
`max_new_tokens=None` (until EOS / `max_length_times×ISL`). AR toks may stop early on EOS before
the 2× ISL cap.

| ISL | Prefill tok/s | Decode tok/s | ms/tok | E2E | AR toks |
|----:|-------------:|-------------:|-------:|----:|--------:|
| 32 | 5.3 | 14.03 | 71 | 11s | 64 |
| 64 | 10.5 | 13.78 | 73 | 15s | 128 |
| 128 | 21.1 | 15.39 | 65 | 23s | 256 |
| 256 | 42.2 | 9.84 | 102 | 58s | 512 |
| 512 | 82.2 | 11.71 | 85 | 96s | 1024 |
| 1024 | 160.5 | 18.14 | 55 | 121s | 2048 |
| 2048 | 300.2 | 19.94 | 50 | 200s | 3802 (EOS before 2×) |
| 4096 | 508.8 | 20.14 | 50 | 417s | 8192 |
| 8192 | 809.1 | 19.18 | 52 | 730s | 13770 (EOS before 2×) |
| 16384 | 850.9 | 18.86 | 53 | 1717s | 31971 |
| 23038 | 730.0 | 17.96 | 56 | 2088s | 36895 |

Steady-state decode peaks around **~20 tok/s** (≈50 ms/tok) for mid/long ISLs; prefill scales
roughly linearly up to ~16k tokens (~850 tok/s), then drops at the full ~23k prompt.

## Performance tests (Tracy)

`tests/perf/` follows the Voxtral / Seamless pattern: outer drivers spawn Tracy; inner workloads
warm outside the window, call `ttnn.ReadDeviceProfiler` to clear load markers, then bracket the
measured region with `start` / `stop` signposts. Run **one at a time** (single device). From
tt-metal root:

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
```

### 1. Device perf (LM prefill 256 + 2 decode steps)

Eager LM only (no metal trace). Aggregates signposted kernel time via `has_signposts=True`.

```bash
pytest models/experimental/vibevoice/tests/perf/test_vibevoice_device_perf.py \
  -v -m models_device_performance_bare_metal

CSV=$(ls -td generated/profiler/ttnn_vibevoice_lm/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report "$CSV" --start-signpost start --end-signpost stop
```

### 2. Single-chunk prefill dump

One warm `forward` chunk (default length **256**). Optional:
`VV_PREFILL_PERF_SEQ_LEN`, `VV_PREFILL_PERF_START_POS`.

```bash
python models/experimental/vibevoice/tests/perf/test_device_perf_single_step_prefill.py

CSV=$(ls -td generated/profiler/vibevoice_lm_single_step_prefill/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report "$CSV" --start-signpost start --end-signpost stop
# optional: redirect to a file to keep the report
```

### 3. Single-step decode dump

Untimed prefill, then one `decode_step` inside signposts. Optional:
`VV_DECODE_PERF_PREFILL_LEN` (default 256).

```bash
python models/experimental/vibevoice/tests/perf/test_device_perf_single_step_decode.py

CSV=$(ls -td generated/profiler/vibevoice_lm_single_step_decode/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report "$CSV" --start-signpost start --end-signpost stop
# optional: redirect to a file to keep the report
```

| Test | Inner workload | Profiler subdir |
|------|----------------|-----------------|
| Device perf | `test_device_perf_forwards.py::test_lm` | `generated/profiler/ttnn_vibevoice_lm/` |
| Prefill dump | `test_profile_single_step_prefill.py` | `generated/profiler/vibevoice_lm_single_step_prefill/` |
| Decode dump | `test_profile_single_step_decode.py` | `generated/profiler/vibevoice_lm_single_step_decode/` |

Wall-clock demo timings (`VV_PROFILE=1` / `--debug`) are separate from these Tracy op dumps.

### 4. E2E ISL sweep (`4p_climate_100min`)

Wall-clock sweep (not Tracy): crop the demo prompt to each ISL after tokenization, warmup
generate, then timed `max_new_tokens=None`. **Fused-frame trace is on by default** (same as
demo `--trace`). Prints prefill time / tok/s, TTFT, decode tok/s, ms/tok, E2E, AR tokens.

```bash
# Default ISLs: 32,64,128,…,16384, then full tokenized length (~23k for 4p_climate_100min)
# Trace on by default — set VV_TRACE_SEGMENT=0 for eager
pytest models/experimental/vibevoice/tests/perf/test_e2e_isl_sweep_perf.py -q -s

# Cap / override
VV_ISL_SWEEP_MAX_ISL=1024 pytest models/experimental/vibevoice/tests/perf/test_e2e_isl_sweep_perf.py -q -s
VV_ISL_SWEEP=32,64,128 VV_ISL_WARMUP_TOKENS=4 \
  pytest models/experimental/vibevoice/tests/perf/test_e2e_isl_sweep_perf.py -q -s
```

Same knobs via demo CLI (trace on by default):

```bash
python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_100min \
  --isl 1024 --warmup
# omit --max_new_tokens for until-EOS / max_length_times×ISL (same as the sweep)
```

## CI

VibeVoice is wired into the **Blackhole demo tests** pipeline
([`.github/workflows/blackhole-demo-tests.yaml`](../../../.github/workflows/blackhole-demo-tests.yaml)
→ entries in [`tests/pipeline_reorg/blackhole_demo_tests.yaml`](../../../tests/pipeline_reorg/blackhole_demo_tests.yaml)),
which runs nightly (04:00 UTC) and on manual dispatch. Only single-P150 (`bh_p150b_civ2`) is
targeted. The three jobs run as **independent parallel matrix jobs** (`fail-fast: false`, one
`test-group` per entry) — the same fan-out the seamless-m4t-v2 model uses; actual concurrency
depends on the P150b runner-pool size.

| Job | Command | Gate | Timeout |
|-----|---------|------|---------|
| demo `4p_climate_100min` | `demo.py --demo 4p_climate_100min --trace` | full long-form render completes | 80 min |
| e2e WER | `pytest tests/pcc/test_e2e_wer.py` (`VV_WER_MAX_NEW_TOKENS=256`) | TT-vs-reference WER ≤ 0.05 | 25 min |
| speaker similarity | `pytest tests/pcc/test_e2e_sim.py` | SIM target floor 0.5 / margin 0.05 | 25 min |

> **Timeout budget caveat.** The `models → demo → bh_p150b_civ2` pipeline has a **130-minute**
> total budget (`.github/time_budget.yaml`), enforced as the *sum* of the per-job timeouts at
> matrix-load time. The three jobs are split to fit exactly (80 + 25 + 25 = 130), so the
> long-form `4p_climate_100min` render gets only **80 min**. A full render is ~60–75 min of
> device time, so under load / measurement variance **the demo job may hit its 80-min timeout**.
> If it does, either raise the demo budget (ping `#tt-metal-infra`) and bump the demo timeout, or
> cap the render (`--max_new_tokens` / `--isl`). WER (~20 min incl. downloads) and sim comfortably
> fit their 25-min slices.

Weights (`microsoft/VibeVoice-1.5B`) + demo text/voices auto-download and cache under `HF_HOME`;
WER/sim additionally pull Whisper (`openai/whisper-medium`) and the SV model
(`microsoft/wavlm-base-plus-sv`). Trigger manually with **Actions → (Blackhole) Demo tests →
Run workflow → model: `vibevoice-1.5b`** (optionally system-type `bh_p150b_civ2`).

## Porting notes

| Submodule | Reference | TT target |
|-----------|-----------|-----------|
| Language model | Qwen2 in `modeling_vibevoice` | `tt_transformers` Qwen2.5-1.5B |
| Acoustic / semantic tokenizers | `modular_vibevoice_tokenizer.py` | `tt/` (later) |
| Diffusion head | `modular_vibevoice_diffusion_head.py` | `tt/` (later) |
| Pipeline | `modeling_vibevoice_inference.py` | `tt/` generate loop |

Closest template: [`models/experimental/speecht5_tts/`](../speecht5_tts/) (`reference/` = PyTorch gold, `tt/` = TTNN, `tests/pcc/` = PCC).
