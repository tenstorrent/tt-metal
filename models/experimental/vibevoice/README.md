# VibeVoice-1.5B

TTNN implementation of [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B),
a long-form, multi-speaker text-to-speech model. Runs fully on Tenstorrent hardware
(prefill + autoregressive decode + audio rendering), with a vendored PyTorch reference kept
alongside for PCC/WER/SIM comparison.

## Model description

VibeVoice is **not** a plain TTS decoder: it is an autoregressive LLM whose "audio token" is a
continuous latent produced by a **diffusion head**, decoded to a waveform by a **VAE-style acoustic
tokenizer**. Each AR step emits one 3200-sample frame of 24 kHz audio (**7.5 frames/s**), which is
what makes ~90-minute single-pass renders feasible.

| Property | Value |
|----------|-------|
| Task | Text → speech (long-form, multi-speaker, voice cloning) |
| Language-model backbone | Qwen2.5-1.5B — 28 layers, hidden 1536, 12 Q heads / 2 KV heads (GQA), head_dim 128, FFN 8960, `rope_theta` 1e6, `rms_norm_eps` 1e-6, vocab 151936 |
| Max context | **65536 positions (64K)** — `decoder.max_position_embeddings` in the checkpoint's `config.json`. Trained with a curriculum from 4,096 to 65,536 tokens. **Caution:** the `DecoderConfig` dataclass in [tt/vibevoice_config.py](tt/vibevoice_config.py) carries a stale `32768` **fallback**; it only applies if `config.json` is missing, so real runs get 65536 |
| Max speakers | 4 distinct speakers |
| Languages | English and Chinese only (upstream constraint) |
| Speech:text token ratio | ≈ 2:1 — two speech tokens ≈ one BPE text token |
| Audio codec | Causal streaming conv VAE, 6 downsampling layers `[8,5,5,4,2,2]` → **3200×** compression, 7 stages of depthwise-causal-conv blocks (`3-3-3-3-3-3-8`); acoustic `vae_dim=64` (`fix_std=0.5`), semantic `vae_dim=128` (encoder-only, no VAE sampling — `fix_std=0`). The acoustic side is a **sigma-VAE**: the latent variance is a fixed constant from the checkpoint (`fix_std`) rather than a learned output, which is what makes the latents stable enough to model autoregressively |
| Audio | 24 kHz mono |
| Frame rate | 7.5 Hz (one AR step = 3200 samples = 133.3 ms of audio) |
| Diffusion | DPM-Solver++ multistep, **10 steps**, CFG scale **1.3** — both are the paper's stated inference settings. Note `config.json`'s `ddpm_num_inference_steps: 20` is *not* what inference uses |
| Diffusion head | 4 × (adaLN + SwiGLU FFN), ffn_ratio 3.0, latent_size 64, `v_prediction`, cosine beta schedule (~123M params) |
| Batch | 1 script per `generate()` call |
| Precision | bf16 weights/activations on the LM & tokenizers; fp32 for RoPE rows, latent inverse-normalization and scheduler coefficients |

Speakers are addressed in the text itself (`Speaker 1:` / `Speaker 2:` …); each speaker can be
bound to a reference WAV for voice cloning, which is encoded on device during prefill.

**Upstream model scope** (properties of VibeVoice-1.5B itself, which this port inherits unchanged —
per the [technical report](https://arxiv.org/abs/2508.19205) Section 4 ("Conclusion, Limitations, and
Risks") and the
[model card](https://huggingface.co/microsoft/VibeVoice-1.5B)): English and Chinese only; at most
4 speakers; no explicitly modelled overlapping speech; speech only — no background noise, music, or
sound effects; and upstream intends it for research and development, not commercial deployment,
flagging deepfake / disinformation misuse risk. Upstream also reports the 1.5B checkpoint as the
weaker variant (SIM 0.548 / WER 1.11 vs the 7B's 0.692 / 1.29), which bounds the best achievable
render quality regardless of backend.

## Supported devices

**Blackhole P150 is the only supported device.**

| Device | Configuration | Status |
|--------|---------------|--------|
| **Blackhole P150** | 1 × Blackhole ASIC, `ARCH_NAME=blackhole`, `MESH_DEVICE=P150` | **Supported** — every PCC / WER / SIM / perf number below was measured here, and it is the only target in nightly CI |

No other device is supported: Wormhole and multi-chip Blackhole boards (P300, Galaxy, QuietBox) are
untested, have no measured numbers, and have untuned trace-region / memory budgets. Single-device only
(`mesh_device` shape `[1]`, `device_id=0`) — there is no tensor-parallel or multi-chip path.

## Architecture

```
        text script ("Speaker 1: …")                    voice prompt WAV(s) @ 24 kHz
                    │                                             │
      VibeVoiceProcessor (Qwen2 tokenizer                 Acoustic Tokenizer ENCODE
      + speech slot mask)                                [1,1,T] → [1,64,T/3200]
                    │                                             │
                    │                                    (lat + bias) × scale  (ckpt consts)
                    │                                             │
                    │                                    Acoustic Connector  64 → 1536
                    │                                             │
        text embeds [1,S,1536] ◄──── scatter into speech slots ────┘   (on device, slice/concat)
                    │
 ┌──────────────────▼─────────────────────────────────────────────────────────────────┐
 │ PREFILL — Qwen2.5-1.5B LM: 28 × (RMSNorm → GQA+RoPE → RMSNorm → SwiGLU FFN)        │
 │ chunked prefill → last_hidden_state + per-layer KV cache                           │
 └──────────────────┬─────────────────────────────────────────────────────────────────┘
                    │  hidden [1,1,1536]
 ┌──────────────────▼──────────────────── AR loop, one frame per step ────────────────┐
 │  LM head → constrained greedy argmax → token id                                    │
 │      │                                                                             │
 │      ├── token == speech_diffusion_id ?                                            │
 │      │        │  CFG condition = (pos hidden, neg hidden)   ← 2 LM rows            │
 │      │        ▼                                                                    │
 │      │   DPM-Solver++ multistep: 10 x [ Diffusion Head (4 x adaLN + SwiGLU) ]      │
 │      │        │  latent [1,64]                                                     │
 │      │        │  latent * (1/scale) - bias  (inverse of the prefill norm)          │
 │      │        ├─► Acoustic Tokenizer DECODE (streaming) ─► 3200 samples ─► waveform│
 │      │        │                                    │                               │
 │      │        │            Semantic Tokenizer ENCODE (streaming) ─► [1,128]        │
 │      │        │                                    │                               │
 │      │        │                         Semantic Connector 128 → 1536              │
 │      │        │                                    │                               │
 │      │        └─► Acoustic Connector 64 → 1536 ───(+)──► next input embed          │
 │      │                                              │                              │
 │      └──────────────► LM decode step (KV cache) ◄────┘                             │
 │                                                                                    │
 │  else: plain text/control token → LM decode step (segment / speaker boundary)      │
 └────────────────────────────────────────────────────────────────────────────────────┘
                    │
        concatenated 24 kHz mono waveform → {demo_id}_tt.wav + {demo_id}_meta.json
```

Under `--trace` (default) the entire AR-loop box above — neg-LM + diffusion + post-diffusion + pos-LM
— is captured as **one device-driven ttnn trace** and replayed per frame. Note the acoustic connector
is applied to the *latent*, not to the decoded audio; only the semantic branch re-encodes the
rendered waveform.

## Model modules

Each TTNN module has a 1:1 vendored PyTorch counterpart under [reference/](reference/) that the PCC
tests compare against.

| Module | TTNN implementation | Reference | What it does |
|--------|--------------------|-----------|--------------|
| Public API | [tt/ttnn_vibevoice_model.py](tt/ttnn_vibevoice_model.py) | — | `TTVibeVoiceModel.from_checkpoint()` assembles every submodule; `.generate()` is the single entry point |
| Generator / pipeline | [tt/ttnn_vibevoice_generator.py](tt/ttnn_vibevoice_generator.py) | [reference/modular/modeling_vibevoice_inference.py](reference/modular/modeling_vibevoice_inference.py) | Prefill + constrained-greedy AR loop, CFG batch-2 decode, fused-frame trace capture/replay, segment boundaries, audio accumulation |
| Language model | [tt/ttnn_vibevoice_lm.py](tt/ttnn_vibevoice_lm.py) | Qwen2 in [reference/modular/modeling_vibevoice.py](reference/modular/modeling_vibevoice.py) | Qwen2.5-1.5B backbone: chunked `prefill()` on `inputs_embeds`, single-token `decode()`, KV cache, RoPE, LM head |
| Acoustic tokenizer | [tt/ttnn_acoustic_tokenizer.py](tt/ttnn_acoustic_tokenizer.py) | [reference/modular/modular_vibevoice_tokenizer.py](reference/modular/modular_vibevoice_tokenizer.py) | Causal streaming conv sigma-VAE (~340M params per encoder/decoder). **Encode** (voice prompt → latents, prefill) and **decode** (latent → 3200 audio samples, every frame) |
| Semantic tokenizer | [tt/ttnn_semantic_tokenizer.py](tt/ttnn_semantic_tokenizer.py) | same | Encoder-only twin (`vae_dim=128`, no VAE sampling; upstream trained it with an ASR proxy task) — re-encodes each rendered frame to give the LM its semantic feedback |
| Speech connectors | [tt/ttnn_speech_connector.py](tt/ttnn_speech_connector.py) | `SpeechConnector` in [reference/modular/modeling_vibevoice.py](reference/modular/modeling_vibevoice.py) | `fc1 → RMSNorm(eps=1e-6) → fc2`, twice: acoustic 64→1536 and semantic 128→1536; summed into the next input embed |
| Diffusion head | [tt/ttnn_diffusion_head.py](tt/ttnn_diffusion_head.py) | [reference/modular/modular_vibevoice_diffusion_head.py](reference/modular/modular_vibevoice_diffusion_head.py) | Timestep embedder (sin-cos + 2-layer SiLU MLP) + 4 × adaLN/SwiGLU `HeadLayer` + adaLN final projection → latent 64. Called once per DPM step |
| DPM scheduler | [tt/ttnn_dpm_scheduler.py](tt/ttnn_dpm_scheduler.py) | [reference/schedule/dpm_solver.py](reference/schedule/dpm_solver.py) | DPM-Solver++ multistep. Noise schedule precomputed on host; `step()` / `convert_model_output()` are pure TTNN |
| Config | [tt/vibevoice_config.py](tt/vibevoice_config.py) | [reference/modular/configuration_vibevoice.py](reference/modular/configuration_vibevoice.py) | Parses `config.json` into dataclasses (decoder / diffusion head / acoustic / semantic) |
| Weight loading | [tt/load_weights.py](tt/load_weights.py) | — | safetensors read, submodule split, Qwen key remap, weight-norm folding, `speech_scaling_factor`/`bias` — host-only |
| Processor | — | [reference/processor/](reference/processor/) | Text → `input_ids` + `speech_input_mask`, voice WAV loading/normalization. Used as-is (host) |
| Reference LM swap | — | [reference/lm_runner.py](reference/lm_runner.py) | *Not vendored* — swaps the TT LM for a CPU fp32 reference LM so PCC tests can isolate the diffusion path from LM drift |

## File paths

```
models/experimental/vibevoice/
├── README.md
├── common/
│   ├── config.py                 # paths, HF repo id, upstream demo-asset repo
│   ├── model_utils.py            # resolve checkpoint path + auto-download weights
│   ├── resource_utils.py         # download demo text/voices from upstream GitHub
│   └── safe_paths.py             # path-containment helpers for all model I/O
├── demo/
│   ├── demo.py                   # TT-only inference entry point (writes wav + script + meta)
│   └── perf_metrics.py           # generate() timing summary + ISL cropping
├── reference/                    # vendored 1.5B-only torch model (from the VibeVoice repo)
│   ├── modular/                  # config + modeling (+ diffusion head, tokenizers)
│   ├── processor/                # text/audio processor
│   ├── schedule/dpm_solver.py    # DPM-Solver++ multistep
│   └── lm_runner.py              # ours: CPU fp32 LM swap-in for PCC tests
├── resources/                    # auto-downloaded demo assets (content gitignored)
│   ├── text/                     # 1p/2p/3p/4p demo scripts
│   └── voices/                   # 9 reference speaker WAVs
├── weights/VibeVoice-1.5B/       # auto-downloaded HF checkpoint (content gitignored)
├── output/{demo_id}/             # demo artifacts
├── tests/
│   ├── conftest.py               # shared fixtures (weights/resources download, config, LM state)
│   ├── pcc/                      # per-component + end-to-end correctness
│   └── perf/                     # Tracy device-perf + wall-clock ISL sweep
└── tt/                           # TTNN implementation
```

Everything is imported by its full path from the tt-metal root, e.g.
`from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel`.

Weights and demo assets are **not** vendored. On first run, demos and tests download:

- **Model weights** — [`microsoft/VibeVoice-1.5B`](https://huggingface.co/microsoft/VibeVoice-1.5B)
  (~5.4 GB across 3 safetensors shards) → `weights/VibeVoice-1.5B/`, via `huggingface_hub`. Override
  the location with `export VIBEVOICE_MODEL_PATH=/path/to/VibeVoice-1.5B`.
- **Demo text + voices** → `resources/` from
  [vibevoice-community/VibeVoice](https://github.com/vibevoice-community/VibeVoice/tree/main/demo).
  Branch-tracked at `main`, **not commit-pinned** — upstream edits change the demo inputs.
- The WER / SIM tests additionally pull `openai/whisper-medium` and `microsoft/wavlm-base-plus-sv`.

Two environment notes (these apply to the reference too, not just the TT path):

- The reference processor loads **`Qwen/Qwen2.5-1.5B` tokenizer assets from the HF cache** — they are
  not bundled in the VibeVoice-1.5B checkpoint, so a fully offline first run fails.
- **`transformers` API drift.** The checkpoint was saved with 4.51.3 while the repo pins a 5.x release;
  the vendored reference carries shims for both (KV-cache API, `use_model_defaults`,
  `_prepare_cache_for_generation` signature, `tie_word_embeddings`). Use the repo's `python_env` — an
  arbitrary version outside that range may break `generate()` parity.

## Quick start

```bash
cd /path/to/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole

# 1. Correctness (auto-downloads weights + demo assets on first run)
pytest models/experimental/vibevoice/tests/pcc/ -v

# 2. Demo — writes output/1p_vibevoice/1p_vibevoice_tt.wav
python models/experimental/vibevoice/demo/demo.py
```

## Example inputs

Inputs are plain-text scripts using `Speaker N:` line prefixes, one turn per line. Un-prefixed free
text is also accepted and is read as a single speaker (the default script is written this way).
Bundled scripts live in `resources/text/` and are addressable by id (`--demo <id>`):

| `--demo` id | Speaker ids | Turns | Size | Content |
|-------------|-------------|------:|-----:|---------|
| `1p_vibevoice` (default) | *none — free text* | — | 625 B | Short English intro to VibeVoice |
| `1p_abs` | 1 | 2 | 1.1 KB | The VibeVoice paper abstract, read aloud (English) |
| `1p_Ch2EN` | 1 | 10 | 1.8 KB | English host explaining Chinese expressions — **mixed English + Chinese** text (8 lines contain CJK) |
| `2p_short` | 1–2 | 2 | 238 B | 2-turn dialogue — fastest multi-speaker smoke test |
| `2p_yayi` | 1–2 | 3 | 238 B | **Chinese** dialogue (colloquial / dialect-heavy) |
| `2p_music` | 1–2 | 14 | 1.0 KB | Dialogue that includes a **sung** excerpt |
| `2p_goat` | 1–2 | 22 | 3.5 KB | Sports-debate podcast (English) |
| `3p_gpt5` | 1–3 | 47 | 13 KB | 3-way tech panel discussion |
| `4p_climate_45min` | 1–4 | 211 | 60 KB | ~45-min 4-speaker podcast |
| `4p_climate_100min` | 1–4 | 363 | 107 KB | ~100-min 4-speaker podcast (~23k prefill tokens) — the perf/CI workload |

Only English and Chinese are supported upstream, which is why the bundled set covers just those two.

`resources/text/2p_short.txt`:

```
Speaker 1: I heard there's big news in TTS lately?
Speaker 2: Yes! Microsoft Research just open-sourced VibeVoice. The model can generate speech up to 90 minutes long, with smooth delivery and rich emotion — it's absolutely amazing.
```

**Voice prompts** — 9 reference WAVs in `resources/voices/`: `en-Alice_woman`, `en-Carter_man`,
`en-Frank_man`, `en-Maya_woman`, `en-Mary_woman_bgm`, `in-Samuel_man`, `zh-Anchen_man_bgm`,
`zh-Bowen_man`, `zh-Xinran_woman`. Three demo ids carry an **auto-bound cast**
(`DEMO_VOICE_CLONES` in [common/resource_utils.py](common/resource_utils.py)):

| Demo id | Speaker → voice |
|---------|-----------------|
| `4p_climate_45min`, `4p_climate_100min` | 1→Alice, 2→Carter, 3→Frank, 4→Maya |
| `3p_gpt5` | 1→Alice, 2→Carter (as "Andrew"), 3→Frank |

Any other script takes explicit `--voice A.wav B.wav …` (positional: Speaker 1, 2, …), or runs
text-only. `--no-voice-cloning` forces text-only even for the three ids above.

Custom script:

```bash
cat > /tmp/my_script.txt <<'EOF'
Speaker 1: Welcome to the show.
Speaker 2: Glad to be here.
EOF

python models/experimental/vibevoice/demo/demo.py --text /tmp/my_script.txt \
  --voice models/experimental/vibevoice/resources/voices/en-Alice_woman.wav \
          models/experimental/vibevoice/resources/voices/en-Carter_man.wav
```

## Expected outputs

`demo.py` writes three artifacts under `{output_dir}/{demo_id}/`:

| File | Content |
|------|---------|
| `{demo_id}_tt.wav` | Generated audio — 24 kHz mono, PCM. Duration ≈ `ar_tokens × 133.3 ms` |
| `{demo_id}_script.txt` | Verbatim copy of the script that was rendered |
| `{demo_id}_meta.json` | Run manifest + perf: `demo_id`, `text_file`, `voice_cloning`, `voice_mapping`, `isl`, `full_prefill_tokens`, `warmup`, `max_length_times`, `max_new_tokens`, `tt_wav`, `script_copy`, and `prefill_s` / `prefill_tok_s` / `ttft_s` / `decode_tok_s` / `ms_per_tok_steady` / `e2e_s` / `ar_tokens_generated` |

The run also prints a one-line perf summary (`prefill` tok/s, TTFT, `decode` tok/s, ms/tok, e2e,
`ar_tokens`, `isl`) — the same fields written to `meta.json`. For representative timings at each ISL
see [Performance](#performance). Each AR frame is a fixed 3200 samples, so the rendered duration is
`ar_tokens / 7.5` seconds, and `ar_tokens` may come in under `--max_new_tokens` if EOS fires first.

Sanity checks on a good render: audio is finite and non-silent, duration matches
`ar_tokens / 7.5` seconds, speech is intelligible under Whisper, and the cloned speaker is
identifiable (this is exactly what `test_e2e_wer.py` and `test_e2e_sim.py` assert).

## Running the demo

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole

# Default script, trace on
python models/experimental/vibevoice/demo/demo.py

# 4-speaker demo, cap the AR loop at 32 frames, verbose stage/timing logs
python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_45min --max_new_tokens 32 --debug

# Full long-form render (the CI workload, ~60-75 min of device time)
python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_100min --trace
```

### Demo parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--demo <id>` | — | Script id; shortcut for `resources/text/<id>.txt` |
| `--text <path>` | — | Custom script file. Overrides `--demo` |
| *(neither)* | `1p_vibevoice.txt` | Default bundled script |
| `--voice WAV [WAV …]` | auto-preset | Voice-clone WAV(s) for Speaker 1, 2, 3, … in order |
| `--no-voice-cloning` | off | Text-only prompt even when the demo has a speaker preset |
| `--model_path <dir>` | auto-download | VibeVoice checkpoint directory |
| `--output_dir <dir>` | `vibevoice/output` | Root output dir; artifacts go to `{output_dir}/{demo_id}/` |
| `--cfg_scale <float>` | `1.3` | Classifier-free guidance scale (2 LM rows per frame) |
| `--num_steps <int>` | `10` | DPM-Solver++ diffusion steps per frame |
| `--max_new_tokens <int>` | `None` | AR frame cap. Default: run until EOS, bounded by `max_length_times` |
| `--max_length_times <float>` | `2.0` | Max AR steps ≈ this × prefill length (HF default) |
| `--isl <int>` | `None` | Crop the processor batch to the first N tokens *after* tokenization — for ISL-controlled perf runs |
| `--warmup` | off | Untimed short `generate()` before the measured pass (warms the program cache) |
| `--warmup_tokens <int>` | `4` | AR steps for the `--warmup` pass |
| `--chunks <int>` | `1` | Render the script as N independently-prefilled parts and concatenate. Only needed beyond one pass's context (~93 min); each boundary costs ~1 garbled minute. Splits only at speaker turns |
| `--seed <int>` | `0` | `torch.manual_seed` for diffusion init noise |
| `--trace` / `--no-trace` | `--trace` | Fused-frame device trace (see below) |
| `--debug` | off | Verbose stage logs (`VV_DEBUG=1`) + device-synced timing breakdown (`VV_PROFILE=1`) |

### Trace (on by default)

`--trace` / `VV_TRACE_SEGMENT=1` ttnn-captures the whole steady-state speech-diffusion frame
(neg-LM + diffusion + post-diffusion + pos-LM) as one fully device-driven graph — the "llama shape":
positions self-advance on device, RoPE is gathered on device, and the pos hidden is loop-carried —
then replays it per frame. It gives **≈21–22 tok/s** steady-state decode vs ≈2.4 tok/s eager
(~9× on the AR loop), and opens the device with a ~1.4 GB trace region + 2 command queues.

| Flag | Env var | Scope | Notes |
|------|---------|-------|-------|
| `--trace` (default) | `VV_TRACE_SEGMENT=1` | whole segment, device-driven | fused frame replayed per frame; ~1.4 GB trace region + 2 CQs |
| `--no-trace` | `VV_TRACE_SEGMENT=0` | eager AR loop | for debugging / A-B |

### Environment variables

The validated decode path — split-capture, CFG batch-2, fused frame output, device-side noise, and
device-side voice-clone encode audio/latents — is unconditional and has no switch. What remains
below either selects a path still under evaluation or configures a test. Values are `1`/`0` unless
stated.

| Variable | Default | Purpose |
|----------|---------|---------|
| `VIBEVOICE_MODEL_PATH` | unset | Checkpoint directory override (skips auto-download) |
| `VV_TRACE_SEGMENT` | `0` in the generator; `1` in the ISL-sweep test | Whole-segment fused decode trace. The demo always sets it explicitly from `--trace` (default on) / `--no-trace`, so the generator's `0` default only applies to callers that set neither |
| `VV_FUSED_ROPE` | `0` (off) | Fused bf16 `rotary_embedding_llama` decode RoPE + on-device cos/sin tables, replacing the default per-position fp32 RoPE rows. Off by default: a 100-min acceptance run showed the speaking rate accelerating (median 208 wpm vs 153) even though every energy/spectral gate passed |
| `VV_TTNN_RANDN` | `0` (off) | Draw diffusion init noise and the acoustic fix-std jitter with `ttnn.randn` on device instead of torch. Off by default: it is a different generator, so renders stop matching the torch reference and PCC comparison against it is no longer meaningful |
| `VV_PREFILL_ISL_SWEEP` | full sweep | Comma list to shorten `test_prefill.py`'s ISL sweep |
| `VV_ISL_SWEEP`, `VV_ISL_SWEEP_MAX_ISL`, `VV_ISL_WARMUP_TOKENS` | see perf section | ISL-sweep perf overrides |
| `VV_WER_MAX_NEW_TOKENS`, `VV_WER_THRESHOLD` | 512 / 0.05 | WER test cap and gate |
| `VV_SIM_MAX_NEW_TOKENS`, `VV_SIM_TARGET_FLOOR`, `VV_SIM_MARGIN` | 200 / 0.5 / 0.05 | SIM test AR cap (≈20–25 s of audio) and the two gates |
| `VV_SIM_TEXT_ID`, `VV_SIM_TARGET_VOICE`, `VV_SIM_REUSE_TT` | `1p_abs` / `en-Carter_man.wav` / off | SIM test script, cloned target voice, and reuse of previously saved TT wavs for a rescore-only run |
| `VV_PREFILL_PERF_SEQ_LEN`, `VV_PREFILL_PERF_START_POS`, `VV_DECODE_PERF_PREFILL_LEN` | 256 / 0 / 256 | Single-step perf-dump shapes |

## Test cases

One line per test. All PCC tests take `mesh_device` shape `[1]`; the device-free ones are marked.

### PCC / correctness — `tests/pcc/`

| File | Test | What it checks |
|------|------|----------------|
| `test_connector_pcc.py` | `test_connector_pcc[acoustic_connector]` | Acoustic connector (64→1536) vs torch on real weights |
| | `test_connector_pcc[semantic_connector]` | Semantic connector (128→1536) vs torch on real weights |
| `test_dpm_scheduler_pcc.py` | `test_dpm_scheduler_pcc` | DPM-Solver++ scheduler math alone on synthetic eps tensors, 10 steps |
| `test_diffusion_head_pcc.py` | `test_diffusion_head_pcc` | Diffusion head forward (timestep embed + 4 adaLN/SwiGLU layers + final proj) vs torch |
| `test_lm_head_pcc.py` | `test_lm_head_logits_pcc` | LM-head logits for the last token vs the HF Qwen2 reference |
| `test_semantic_tokenizer_pcc.py` | `test_semantic_tokenizer_pcc` | Semantic encoder on a fixed 24000-sample (1 s) audio segment |
| `test_acoustic_tokenizer_pcc.py` | `test_acoustic_tokenizer_encode_pcc` | Acoustic VAE encode (audio → 64-dim latents) |
| | `test_acoustic_tokenizer_decode_pcc` | Acoustic VAE decode from scaled-random latents |
| | `test_acoustic_tokenizer_decode_real_latents_pcc` | Acoustic VAE decode from real encoder latents (in-distribution) |
| `test_decoder_layer_pcc.py` | `test_decoder_layer_decode_pcc` | Layer-0 decode regression: random hidden `[1,1,H]`, empty KV cache, positions 0–9 — isolates decode SDPA at low cache depth |
| `test_prefill.py` | `test_full_prefill_chain_pcc` | Whole prefill chain (acoustic encode → connector → scatter → LM prefill → `last_hidden_state` + per-layer KV) vs a bf16 HF Qwen2 reference, swept over 11 ISLs |
| `test_decode.py` | `test_decode_ref_cond_frame_pcc` | Open-loop per-stage decode parity vs the fp32 reference over a teacher-forced stream: diffusion latent, post-diffusion chain, and LM hidden, each fed the *reference* input for its stage |
| `test_e2e_wer.py` | `test_e2e_wer_teacher_forced` | End-to-end audio fidelity: TT replays the reference token stream with reference embeds fed back, Whisper transcribes both waveforms, WER(TT vs reference) ≤ 0.05 |
| `test_e2e_sim.py` | `test_e2e_sim_tt_voice_clone` | Speaker similarity: TT voice-clones one target, an SV model must rank the target above every impostor by a margin (SIM-O) |
| | `test_e2e_sim_4speaker` | 4-speaker self-identification confusion matrix (Alice / Carter / Frank / Maya) |
| `test_safe_paths.py` | 7 parametrized tests (`test_safe_join_*`, `test_safe_output_path_*`, `test_load_weights_rejects_traversal_shard_name`) | **Host-only, no device** — path-containment guards for checkpoint / asset / artifact I/O, incl. a traversal shard name in `model.safetensors.index.json` |

### Performance — `tests/perf/`

| File | Test / entry point | What it measures |
|------|--------------------|------------------|
| `test_vibevoice_device_perf.py` | `test_perf_device_bare_metal_vibevoice_lm` | Outer Tracy driver: aggregated device kernel time for one LM prefill chunk (256) + 2 decode steps, eager |
| `test_device_perf_forwards.py` | `test_lm` | The inner workload the driver above spawns (warm outside the window, signpost the measured region) |
| `test_device_perf_single_step_prefill.py` | `main()` | Outer driver for a single-prefill-chunk Tracy op dump |
| `test_profile_single_step_prefill.py` | `test_profile_single_step_prefill` | Inner workload: one warm `prefill` chunk (default seq 256) inside signposts |
| `test_device_perf_single_step_decode.py` | `main()` | Outer driver for a single-decode-step Tracy op dump |
| `test_profile_single_step_decode.py` | `test_profile_single_step_decode` | Inner workload: untimed prefill (default 256) then one `decode_step` inside signposts |
| `test_e2e_isl_sweep_perf.py` | `test_e2e_isl_sweep_4p_climate_100min` | Wall-clock end-to-end sweep over 11 ISLs on the 100-min script: prefill tok/s, TTFT, decode tok/s, ms/tok, E2E, AR tokens |

## Commands — PCC checks

```bash
cd $TT_METAL_HOME
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole

# Everything (auto-downloads weights, demo assets, and the WER/SIM models on first run)
pytest models/experimental/vibevoice/tests/pcc/ -v

# Fast component sweep (seconds-to-minutes each)
pytest models/experimental/vibevoice/tests/pcc/test_connector_pcc.py \
       models/experimental/vibevoice/tests/pcc/test_dpm_scheduler_pcc.py \
       models/experimental/vibevoice/tests/pcc/test_diffusion_head_pcc.py \
       models/experimental/vibevoice/tests/pcc/test_lm_head_pcc.py \
       models/experimental/vibevoice/tests/pcc/test_semantic_tokenizer_pcc.py \
       models/experimental/vibevoice/tests/pcc/test_acoustic_tokenizer_pcc.py \
       models/experimental/vibevoice/tests/pcc/test_decoder_layer_pcc.py -v -s

# Full prefill / decode chain
pytest models/experimental/vibevoice/tests/pcc/test_prefill.py \
       models/experimental/vibevoice/tests/pcc/test_decode.py -v -s

# Shorten the prefill ISL sweep for a smoke run
VV_PREFILL_ISL_SWEEP=32,64,128 \
  pytest models/experimental/vibevoice/tests/pcc/test_prefill.py -v -s

# Perceptual end-to-end (downloads whisper-medium / wavlm-base-plus-sv on first run)
VV_WER_MAX_NEW_TOKENS=256 pytest models/experimental/vibevoice/tests/pcc/test_e2e_wer.py -v -s
pytest models/experimental/vibevoice/tests/pcc/test_e2e_sim.py -v -s

# Host-only guards (no device needed)
pytest models/experimental/vibevoice/tests/pcc/test_safe_paths.py -v
```

| Parameter | Applies to | Meaning |
|-----------|-----------|---------|
| `VV_PREFILL_ISL_SWEEP=32,64,…` | `test_prefill.py` | Replace the 11-point ISL sweep with this list |
| `VV_WER_MAX_NEW_TOKENS=<int>` | `test_e2e_wer.py` | AR frame cap for the teacher-forced stream (CI uses 256; the table below is 512) |
| `VV_WER_THRESHOLD=<float>` | `test_e2e_wer.py` | WER gate (default 0.05) |
| `-s` | all | Required to see the per-test PCC / metric tables on stdout |
| `mesh_device` / `device_params` | all | Fixed at `[1]` and `l1_small_size=32768` — do not override |

### Test gates

| Quantity | Gate |
|----------|------|
| Component PCC (connectors, scheduler, diffusion head, LM head, tokenizers) | ≥ 0.99 |
| Diffusion head (secondary) | relative Frobenius error ≤ 0.10, plus an allclose check |
| Decoder-layer decode min PCC | ≥ 0.9975 (secondary: relative Frobenius ≤ 0.07) |
| Prefill: speech embeds, KV cache | ≥ 0.99 |
| Prefill: LM hidden | per-position **median** ≥ 0.96 (flattened PCC is pulled down by a few text-token outliers) |
| Decode: chain (fused embed), LM hidden | per-frame **min** ≥ 0.99 |
| Decode: diffusion latent | *distribution* gate — no frame < 0.5, and ≤ 20 % of frames below 0.99 |
| Teacher-forced WER | ≤ 0.05 |
| Speaker similarity | target SIM ≥ 0.5 **and** target-vs-best-impostor margin ≥ 0.05. (The test also *reports* `microsoft/wavlm-base-plus-sv`'s suggested 0.86 same-speaker threshold for context, but does not gate on it) |

Why the decode gates differ: the *closed* decode loop is chaotic — one separatrix frame cascades
under feedback (measured latent PCC 0.999 → 0.16 over ~24 frames) — so it is deliberately not
PCC-gated. Each stage is instead fed the *reference* input for that stage (open loop), which
localizes any regression and cannot accumulate. Whole-loop fidelity is covered perceptually by the
WER and SIM tests. The diffusion latent gets a distribution gate rather than a per-frame min because
the DPM sampler is separatrix-sensitive for a rare, perceptually-inert subset of conditions: a benign
input can push the discrete bf16 trajectory across a contractive/expansive boundary, so a `min` gate
would false-fail while the distribution gate still catches a real regression.

**SV backend — why `microsoft/wavlm-base-plus-sv`, not the model from the paper.** The
[VibeVoice technical report](https://arxiv.org/abs/2508.19205) states only that SIM(-O) is computed
"by extracting speaker embeddings with **WavLM-large**" (Sections 3.1 and 3.2) — it never names a
checkpoint.
The conventional SIM-O setup behind that phrasing is the UniSpeech `wavlm_large_finetune.pth`
(WavLM-large backbone + an ECAPA-TDNN x-vector head), and we deliberately do **not** ship it: its code
and checkpoint
([microsoft/UniSpeech](https://github.com/microsoft/UniSpeech), which in turn borrows from the
unlicensed [lawlict/ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN)) are licensed
**CC BY-SA 3.0**, whose *ShareAlike* clause requires derivative works to carry the same license —
incompatible with this repo's **Apache-2.0**. Instead the test uses `microsoft/wavlm-base-plus-sv`,
a WavLM x-vector head that ships with HuggingFace `transformers` (Apache-2.0), needs no extra
dependency or
separately-licensed checkpoint, and keeps the whole path license-clean. Trade-off: base_plus
produces a *compressed* cosine scale (different speakers still score ~0.5–0.7, vs the fine-tuned
model's ~0.9 same-speaker / ~0 impostor separation), so the test asserts a **relative**
target-vs-impostor margin rather than the paper's absolute same-speaker threshold.

## PCC results

Measured on **Blackhole P150** against the PyTorch / HuggingFace reference paths.

| File | Test case | PCC / metric |
|------|-----------|-------------:|
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
| `test_e2e_wer.py` | Teacher-forced WER (`4p_climate_45min`, cap = 512) | 0.0000 |
| `test_e2e_sim.py` | Voice-clone SIM (Carter vs best impostor) | 0.9914 / margin +0.3854 |
| | 4-speaker self-ID (Alice / Carter / Frank / Maya) | 0.9805 / 0.9914 / 0.9957 / 0.9761 |

There is a single column because **Blackhole P150 is the only supported device** — see
[Supported devices](#supported-devices).

### Prefill chain ISL sweep (`test_prefill.py`, Blackhole P150)

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
> ISL** (below the 0.99 KV gate from ISL ≥ 1024 — worst layer often 13). See
> [Known limitations](#known-limitations).

## Performance

### End-to-end ISL sweep (`4p_climate_100min`, Blackhole P150)

`tests/perf/test_e2e_isl_sweep_perf.py` on Blackhole P150 with fused-frame trace
(`VV_TRACE_SEGMENT=1`, fp32 RoPE / `VV_FUSED_ROPE=0`). Prompt cropped to each ISL after
tokenization; untimed warmup (`VV_ISL_WARMUP_TOKENS=32`) then timed `max_new_tokens=None`
(EOS / `max_length_times×ISL`). **Decode tok/s** is steady fused-frame *replay* only
(`decode_mode=steady_trace`) — capture / first-time compile frames are excluded.

| ISL | Prefill tok/s | TTFT (s) | Decode tok/s | ms/tok | E2E | AR toks |
|----:|-------------:|---------:|-------------:|-------:|----:|--------:|
| 32 | 5.3 | 6.073 | 21.52 | 46 | 11s | 64 |
| 64 | 10.5 | 6.088 | 21.50 | 47 | 16s | 128 |
| 128 | 20.7 | 6.173 | 21.50 | 47 | 24s | 256 |
| 256 | 41.8 | 6.128 | 21.49 | 47 | 62s | 512 |
| 512 | 82.0 | 6.247 | 21.52 | 46 | 12s | 66 (EOS early) |
| 1024 | 158.7 | 6.452 | 21.45 | 47 | 135s | 2048 |
| 2048 | 292.8 | 6.995 | 21.35 | 47 | 220s | 4096 |
| 4096 | 506.6 | 8.085 | 21.12 | 47 | 412s | 7795 (EOS before 2×) |
| 8192 | 780.8 | 10.492 | 20.79 | 48 | 848s | 15122 (EOS before 2×) |
| 16384 | 813.2 | 20.147 | 20.21 | 49 | 1858s | 32768 |
| 23038 | 724.5 | 31.799 | 19.83 | 50 | 2617s | 42498 (EOS before 2×) |

Steady decode is **~21.5 tok/s** (≈46–47 ms/tok) through mid ISLs, easing slightly to
**~20 tok/s** at full prompt (longer KV). TTFT stays ~6s through ISL≈2k, then rises with
prefill cost to ~32s at full length. Log: `tests/logs/test_e2e_isl_sweep_perf_full_steady.txt`.

### Performance tests (Tracy)

`tests/perf/` follows the Voxtral / Seamless pattern: outer drivers spawn Tracy; inner workloads warm
outside the window, call `ttnn.ReadDeviceProfiler` to clear load markers, then bracket the measured
region with `start` / `stop` signposts. Run **one at a time** (single device). From tt-metal root:

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
```

**1. Device perf (LM prefill 256 + 2 decode steps)** — eager LM only, no metal trace; aggregates
signposted kernel time with `has_signposts=True`.

```bash
pytest models/experimental/vibevoice/tests/perf/test_vibevoice_device_perf.py \
  -v -m models_device_performance_bare_metal

CSV=$(ls -td generated/profiler/ttnn_vibevoice_lm/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report "$CSV" --start-signpost start --end-signpost stop
```

**2. Single-chunk prefill dump** — one warm `forward` chunk (default length 256). Optional:
`VV_PREFILL_PERF_SEQ_LEN`, `VV_PREFILL_PERF_START_POS`.

```bash
python models/experimental/vibevoice/tests/perf/test_device_perf_single_step_prefill.py

CSV=$(ls -td generated/profiler/vibevoice_lm_single_step_prefill/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report "$CSV" --start-signpost start --end-signpost stop
```

**3. Single-step decode dump** — untimed prefill, then one `decode_step` inside signposts. Optional:
`VV_DECODE_PERF_PREFILL_LEN` (default 256).

```bash
python models/experimental/vibevoice/tests/perf/test_device_perf_single_step_decode.py

CSV=$(ls -td generated/profiler/vibevoice_lm_single_step_decode/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report "$CSV" --start-signpost start --end-signpost stop
```

| Test | Inner workload | Profiler subdir |
|------|----------------|-----------------|
| Device perf | `test_device_perf_forwards.py::test_lm` | `generated/profiler/ttnn_vibevoice_lm/` |
| Prefill dump | `test_profile_single_step_prefill.py` | `generated/profiler/vibevoice_lm_single_step_prefill/` |
| Decode dump | `test_profile_single_step_decode.py` | `generated/profiler/vibevoice_lm_single_step_decode/` |

Wall-clock demo timings (`VV_PROFILE=1` / `--debug`) are separate from these Tracy op dumps.

**4. E2E ISL sweep** — wall-clock (not Tracy). Trace on by default, same as the demo.

```bash
# Default ISLs: 32,64,128,…,16384, then the full tokenized length (~23k for 4p_climate_100min)
pytest models/experimental/vibevoice/tests/perf/test_e2e_isl_sweep_perf.py -q -s

# Cap / override
VV_ISL_SWEEP_MAX_ISL=1024 pytest models/experimental/vibevoice/tests/perf/test_e2e_isl_sweep_perf.py -q -s
VV_ISL_SWEEP=32,64,128 VV_ISL_WARMUP_TOKENS=32 \
  pytest models/experimental/vibevoice/tests/perf/test_e2e_isl_sweep_perf.py -q -s
VV_TRACE_SEGMENT=0 pytest models/experimental/vibevoice/tests/perf/test_e2e_isl_sweep_perf.py -q -s  # eager
```

Same knobs via the demo CLI:

```bash
python models/experimental/vibevoice/demo/demo.py --demo 4p_climate_100min --isl 1024 --warmup
# omit --max_new_tokens for until-EOS / max_length_times × ISL (same as the sweep)
```

### Host operations (trace-accelerated run)

**1. Steady-state frame — runs every AR step.** At the defaults this is **one D2H per frame** plus
host bookkeeping: the diffusion noise is gathered on device, and the audio and token come back in a
single transfer.

| Op | Type | Used for |
|----|------|----------|
| fused audio+token readback | D2H ×1 | one `to_torch` returns `[audio …, token_idx]`; the token half is what the AR loop blocks on, since a trace cannot branch |
| `_emit_audio` (append) | hostCPU | accumulate frame audio into the waveform |
| `_gen_tokens.append` / `valid_ids[idx]` | hostCPU | token record; map the constrained-argmax **local** index to the global id (kept local so it survives the bf16 cast into the fused tensor) |
| host pos/neg mirror `+=1`, RoPE write ×4 | hostCPU + H2D ×4 | **only when `VV_FUSED_ROPE=0`** (the default): the mirrors exist solely to index the fp32 cos/sin tables. On the fused path the rows are gathered on device and none of this runs |

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
| voice-clone encode: audio up | H2D ×1 per speaker | one upload of the whole voice row; the traced chunk graph gathers its own chunk on device |
| voice-clone encode: latents down | D2H ×1 per speaker | the chunk graph is ttnn-traced (`_ensure_encode_trace`) and replayed per chunk, scattering each latent row into a device accumulator; the host reads the accumulator once per speaker (663 → 4 transfers for the 4-speaker climate prompt). The trace is released before the LM prefill |
| scale/bias + `feats=(lat+bias)*scale` | hostCPU | normalize latents before the acoustic connector. `scale`/`bias` come from the checkpoint, so no reduction is needed |
| `reset_*_cache` | H2D | reset conv streaming caches for a fresh generation |
| `torch.randn(max_steps,…)` | hostCPU | pre-draw all diffusion init noise, then upload once as a gather table (RNG-aligned; `ttnn.randn` on device under `VV_TTNN_RANDN=1`) |
| first `_greedy_argmax` | D2H | first token after prefill |

## Known limitations

Limitations of **this TTNN implementation relative to the vendored PyTorch reference**. Constraints of
the VibeVoice-1.5B model itself (language coverage, speaker count, upstream usage scope) are not
repeated here — see [Model description](#model-description) and [Upstream references](#upstream-references).

**Functional gaps vs the reference `generate()`**

- **Greedy decoding only.** The reference honours `generation_config.do_sample` and samples via
  `torch.multinomial` (temperature / top-p / top-k). The TT generator implements **constrained greedy
  argmax only** — there is no sampling path, so TT cannot reproduce a sampled reference run.
- **No HuggingFace generation surface.** The reference accepts `logits_processor`,
  `stopping_criteria`, `prefix_allowed_tokens_fn`, `assistant_model` (assisted decoding),
  `stop_check_fn`, `return_speech` and `is_prefill`. `TTVibeVoiceModel.generate()` accepts none of
  these; its token constraint is built in rather than injectable.
- **No custom CFG negative prompt.** The reference takes `negative_prompt_ids` /
  `negative_prompt_attention_mask`; the TT port constructs the negative condition internally, so the
  negative branch is not user-controllable.
- **Batch 1 only.** The reference generates batched (per-row `finished_tags`, `correct_cnt` and
  `audio_chunks`). The TT generator hard-wires row 0 throughout, so one script per `generate()` call.

**Device / deployment constraints of the port**

- **Blackhole P150 only.** Wormhole and multi-chip Blackhole boards are untested; no measured numbers
  exist for them and the trace-region / memory budgets are untuned. The reference runs anywhere torch
  does.
- **Single device.** No tensor-parallel or multi-chip path (`mesh_device` shape `[1]`).
- **Trace costs a ~1.4 GB trace region + 2 command queues.** Enabled by default; anything else
  sharing the device must fit around it.

**Long-form rendering**

- **`--chunks` boundaries are lossy.** The 64K context bound (≈94 min for the ~23k-token 100-min
  script) is the model's, not the port's — but the port's workaround for exceeding it is
  `--chunks N`, which re-prefills each part with fresh KV caches and streaming conv state, so **each
  boundary costs roughly one garbled minute** of audio. The reference has no equivalent stitching, so
  this artifact is specific to this implementation.
- **Host RAM peaks at ~1.15 GB for a 100-min render** (frame chunks + the concatenated copy).
- **CI's long-form demo job can time out.** See the budget caveat under [CI](#ci).

**Unvalidated optimizations (off by default)**

- `VV_FUSED_ROPE=1` — faster bf16 fused decode RoPE, but a 100-min acceptance run showed the speaking
  rate accelerating (median 208 wpm vs 153) despite every energy/spectral gate passing.
- `VV_TTNN_RANDN=1` — on-device noise generation is a *different* RNG, so renders stop matching the
  torch reference and PCC comparison against it becomes meaningless.

## CI

VibeVoice is wired into the **Blackhole demo tests** pipeline
([`.github/workflows/blackhole-demo-tests.yaml`](../../../.github/workflows/blackhole-demo-tests.yaml)
→ entries in [`tests/pipeline_reorg/blackhole_demo_tests.yaml`](../../../tests/pipeline_reorg/blackhole_demo_tests.yaml)),
which runs nightly (`cron: "0 4 * * *"`, 04:00 UTC) and on manual dispatch. Only single-P150
(`bh_p150b_civ2`) is targeted, with `MESH_DEVICE=P150`. The three entries fan out as **independent
parallel matrix jobs** in `blackhole-demo-tests-impl.yaml` (`fail-fast: false`, one `test-group` per
entry); actual concurrency depends on the P150b runner-pool size.

| Job | Command | Gate | Timeout |
|-----|---------|------|---------|
| demo `4p_climate_100min` | `demo.py --demo 4p_climate_100min --trace` | full long-form render completes | 80 min |
| e2e WER | `pytest tests/pcc/test_e2e_wer.py` (`VV_WER_MAX_NEW_TOKENS=256`) | TT-vs-reference WER ≤ 0.05 | 25 min |
| speaker similarity | `pytest tests/pcc/test_e2e_sim.py` | SIM target floor 0.5 / margin 0.05 | 25 min |

> **Timeout budget caveat.** The `models → demo → bh_p150b_civ2` pipeline has a **130-minute** total
> budget (`.github/time_budget.yaml`), enforced as the *sum* of the per-job timeouts at matrix-load
> time. The three jobs are split to fit exactly (80 + 25 + 25 = 130), so the long-form
> `4p_climate_100min` render gets only **80 min**. A full render is ~60–75 min of device time, so
> under load / measurement variance **the demo job may hit its 80-min timeout**. If it does, either
> raise the demo budget (ping `#tt-metal-infra`) and bump the demo timeout, or cap the render
> (`--max_new_tokens` / `--isl`).

Weights + demo text/voices auto-download and cache under `HF_HOME`; WER/SIM additionally pull
Whisper and the SV model. Trigger manually with **Actions → (Blackhole) Demo tests → Run workflow →
model: `vibevoice-1.5b`** (optionally system-type `bh_p150b_civ2`).

## Upstream references

- [VibeVoice Technical Report (arXiv:2508.19205)](https://arxiv.org/abs/2508.19205)
- [microsoft/VibeVoice-1.5B model card](https://huggingface.co/microsoft/VibeVoice-1.5B)
- [github.com/microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)
- [microsoft.github.io/VibeVoice](https://microsoft.github.io/VibeVoice/)
