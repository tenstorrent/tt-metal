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

## Layout

```
vibevoice/
├── README.md
├── common/
│   ├── config.py            # paths, HF repo id, transformers pin
│   ├── model_utils.py       # resolve path + auto-download weights
│   └── resource_utils.py    # download demo text/voices from upstream GitHub
├── reference/               # vendored 1.5B-only torch model (from VibeVoice repo)
│   ├── modular/             # config + modeling (imported as `modular.*`)
│   ├── processor/           # tokenizer/audio processor (`processor.*`)
│   └── schedule/            # DPM solver (`schedule.*`)
├── resources/               # auto-downloaded demo assets (gitignored content)
│   ├── voices/              # from github .../demo/voices
│   └── text/                # from github .../demo/text_examples
├── weights/                 # auto-downloaded HF checkpoint (gitignored content)
├── tests/
│   ├── conftest.py              # pytest: reference/ on PYTHONPATH + shared fixtures
│   ├── pcc/
│   │   ├── pcc_helpers.py       # shared LM PCC helpers (HF ref, TT LM builder, PCC utils)
│   │   ├── test_decoder_layer_pcc.py
│   │   ├── test_prefill.py
│   │   └── test_decode.py
│   └── perf/                    # Tracy device-perf + single-step prefill/decode dumps
└── tt/                          # TTNN port
```

## Dependencies

Pin `transformers` — **4.57+** breaks `generate()` KV-cache behavior for this model.

```bash
pip install 'transformers==4.51.3' torch accelerate diffusers tqdm librosa scipy huggingface_hub
```

The processor also pulls **Qwen/Qwen2.5-1.5B** tokenizer assets from the Hugging Face cache (`QWEN_TOKENIZER` in `common/config.py`).

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
  no prefill. Isolates decode SDPA at low cache depth (min PCC ~0.99997).
- **Full prefill chain:** `test_prefill.py::test_full_prefill_chain_pcc` — the integrated
  prefill path (acoustic tokenizer → connector → scatter into embeddings → LM prefill →
  `last_hidden_state`) plus per-layer KV cache, vs the bf16 HF Qwen2 reference; synthetic-input ISL
  sweep 2k … 64k, gated at `PCC >= 0.99`.
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
# optional: > models/experimental/vibevoice/lm/prefill_expN.txt
```

### 3. Single-step decode dump

Untimed prefill, then one `decode_step` inside signposts. Optional:
`VV_DECODE_PERF_PREFILL_LEN` (default 256).

```bash
python models/experimental/vibevoice/tests/perf/test_device_perf_single_step_decode.py

CSV=$(ls -td generated/profiler/vibevoice_lm_single_step_decode/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report "$CSV" --start-signpost start --end-signpost stop
# optional: > models/experimental/vibevoice/lm/decode_expN.txt
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

## Porting notes

| Submodule | Reference | TT target |
|-----------|-----------|-----------|
| Language model | Qwen2 in `modeling_vibevoice` | `tt_transformers` Qwen2.5-1.5B |
| Acoustic / semantic tokenizers | `modular_vibevoice_tokenizer.py` | `tt/` (later) |
| Diffusion head | `modular_vibevoice_diffusion_head.py` | `tt/` (later) |
| Pipeline | `modeling_vibevoice_inference.py` | `tt/` generate loop |

Closest template: [`models/experimental/speecht5_tts/`](../speecht5_tts/) (`reference/` = PyTorch gold, `tt/` = TTNN, `tests/pcc/` = PCC).
