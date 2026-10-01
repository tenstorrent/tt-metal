# Voxtral-4B-TTS-2603 on Tenstorrent (TTNN)

[`mistralai/Voxtral-4B-TTS-2603`](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) text-to-speech on
**one Blackhole chip**: English text plus one of the model's 20 preset voices in, 24 kHz speech out.
The pipeline serves **32 requests at once** (batch 32) and runs every stage on device.

## How it works

Speech is produced in **frames**. One frame is **80 ms of audio** (12.5 frames per second) and is described
by 37 codes: 1 *semantic* code (what is said) and 36 *acoustic* codes (how it sounds).

| Stage | What it does | When it runs |
|---|---|---|
| **prefill** | The 26-layer Mistral text stack reads the request -- the voice sample (147 slots), the text and control tokens -- in one pass and fills the KV cache | once per request |
| **decode** | One step of the text stack produces the hidden state for the next frame | every frame |
| **acoustic** | A 3-layer flow-matching model (7 Euler steps, classifier-free guidance 3.0) turns that hidden state into the frame's 37 codes, which are fed back into decode | every frame |
| **vocode** | The codec decoder turns all codes into a 24 kHz waveform | once per utterance |

Code layout:

```
models/demos/voxtral_4b_tts_2603/
  tt/            the pipeline: pipeline.py (run_text_to_speech, build_pipeline, trace hooks),
                 text_stack.py (26 KV-cached decoder layers), acoustic_stage.py (flow-matching sampler),
                 vocode_stage.py (codec decoder), common.py (tokenizer, prompts, reference loading)
  tt/modules/    the 7 TTNN modules the stages are built from: layer.py (one fused text decoder layer),
                 token_embed.py, mistral_rotary_embedding.py, mistral_r_m_s_norm.py, multi_vocab_embeddings.py,
                 flow_matching_audio_transformer.py (the acoustic velocity field),
                 voxtral_t_t_s_audio_tokenizer.py (the codec decoder)
  reference/     reference_loader.py (the PyTorch reference model, built from the native checkpoint),
                 golden.py (CPU reference chains), quality.py (speech scoring: Whisper WER, UTMOS MOS)
  demo/          demo_text_to_speech.py
  tests/e2e/     test_text_to_speech_ci.py (short accuracy check), test_e2e_text_to_speech.py (full accuracy),
                 test_text_to_speech_perf.py (performance)
```

## Setup

Build tt-metal and its Python environment as usual (`./build_metal.sh`, `./create_venv.sh`), then:

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1   # show test output live when piping through tee
```

The checkpoint (~8 GB) downloads from Hugging Face on first use, pinned to snapshot
`b81be46c3777f88621676791b512bb01dc1cb970` (override with `VOXTRAL_HF_REVISION`); `HF_HUB_OFFLINE=1` runs from
the local cache. The full accuracy test and the demo's `--score` also use Whisper large-v3-turbo (pinned to
revision `41f01f3f`) and UTMOS22. UTMOS22 runs from the local torch hub cache with its checkpoint checked
against a pinned SHA-256; to populate that cache once, run any scoring command with
`VOXTRAL_ALLOW_REMOTE_MOS=1` (this fetches `tarepan/SpeechMOS` v1.2.0 and executes its hub code).

Pick the chip with `VOXTRAL_DEVICE_ID` (accuracy tests) or `TT_PERF_DEVICE_ID` (perf test); both default to 0.
If opening the device fails with `failed to initialize FW! Try resetting the board`, run `tt-smi -r` (resets
every chip on the host). The CPU reference results are cached in `~/.cache/voxtral_4b_tts_2603_golden`
(`VOXTRAL_GOLDEN_CACHE` overrides it).

## Tests to run

| # | Test | What it checks | Time | Pass criteria |
|---|---|---|---|---|
| 0 | `pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_text_to_speech_ci.py -s` | **Short accuracy check (CI-sized).** The full pipeline on device for 8 frames, in three cases (`casual_male`; `ar_male`, whose shared prompt prefix runs past a tile edge; and one 203-token text repeated in all 32 rows, as the demo does with a single `--text`), each stage compared with the teacher-forced PyTorch reference; no WER/MOS | ~15 min first run, a few minutes after (cached reference) | all three pass: every stage PCC >= 0.99, >= 98% acoustic and >= 99% semantic codes equal |
| 1 | `pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_e2e_text_to_speech.py -s` | **Full accuracy.** 32 sentences generated on device to their own stop, compared with the PyTorch reference on CPU: per-stage PCC, discrete codes, waveform PCC, and the audio scored for intelligibility (Whisper WER) and naturalness (UTMOS MOS) | ~50 min on a first run (CPU reference, ~16 GB RAM), ~15-20 min after | all 10 tests pass |
| 2 | `pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_text_to_speech_perf.py -s` | **Performance.** Each stage traced and replayed on one command queue; prints per-stage ms, ms per audio frame, frames/s | ~2-3 min | every stage and the per-frame time within 10% of *Expected performance* (`TT_PERF_MARGIN` adjusts) |
| 3 | `python -m models.demos.voxtral_4b_tts_2603.demo.demo_text_to_speech --device-id 0 --out-dir <dir> --score` | **Listen.** Writes one WAV per sentence (cut at its own end) and prints each clip's WER and MOS | ~3 min | corpus WER / mean MOS as in *Expected accuracy* |

`demo_text_to_speech.py --text "Your sentence."` speaks your own text; `--voice` picks any of the 20 presets
(an unknown name prints the list).

**Guidance strength.** The acoustic stage uses classifier-free guidance alpha 3.0 by default
(`VOXTRAL_CFG_ALPHA` overrides it). The checkpoint ships no sampling alpha -- the reference takes it as an
argument -- so the value is this package's choice. vLLM-Omni is reported to serve this model at 1.2; every
number in this README, and the accuracy tests' fixed tolerances, were measured at 3.0.

### What the accuracy test checks

The device side runs free, exactly as the demo does. The reference is run twice on the same inputs,
voice and noise: once free-running (the baseline for the WER/MOS scores) and once **teacher-forced** onto the
device's own trajectory (fed the device's codes and hidden states at every frame), so every per-stage
comparison measures the device's arithmetic rather than the drift between two independent runs.

| Test | Checks |
|---|---|
| `test_golden_is_the_references_own_arithmetic` | the reference chain equals the reference model's own code |
| `test_run_ended_on_the_models_stop_rule` | every sentence ended on the model's end-of-audio code, not the 256-frame cap |
| `test_shapes_and_real_task_output` | the output is real 24 kHz audio (finite, in range, not constant) |
| `test_batch_is_32_independent_samples` | 32 inputs give 32 distinct waveforms |
| `test_per_stage_pcc` | worst per-row PCC >= 0.99 for prefill hidden, decode hidden, semantic logits and the acoustic output `x_final`, over every frame |
| `test_discretization_is_the_references_own_rule` | the device turns `x_final` / logits into codes with the reference's exact rule |
| `test_discrete_codes_equal_the_teacher_forced_reference` | codes equal the reference's wherever the reference's decision is clear of a FIXED noise band (measured on this build; it does not widen when the device is less accurate), >= 98% of live acoustic codes equal, and the acoustic stage's error on a held-out input at most 2x this build's |
| `test_signal_quality_wer_and_mos` | device WER <= reference WER + 0.05 and device MOS >= reference MOS - 0.2 |
| `test_free_running_divergence_is_reported` | the first frame's semantic code matches (diagnostic otherwise) |
| `test_gate3_e2e_pcc` | the final waveform PCC >= 0.99 against the teacher-forced reference |

## Expected performance

Measured with test 2 on **one Blackhole chip** (a p150 card; the previous build measured within 1% of a p300c
chip of a QuietBox 2 on every stage), batch 32, trace + 1 command queue:

| Stage | Time per step |
|---|---|
| prefill (once per request) | 85.6 ms |
| decode (every frame) | 37.2 ms |
| acoustic (every frame) | 25.8 ms |
| vocode (32-frame chunk, 2.56 s of audio per row) | 75.3 ms |

| Throughput | Value |
|---|---|
| ms per audio frame (decode + acoustic), all 32 users | **63.0 ms** |
| frames/s per user (real time = 12.5) | **15.9** (1.27x real time) |
| frames/s total | **508** (~41 s of audio per second) |

These are traced, steady-state stage times. A request additionally pays prefill before its first frame and
vocode after its last. Chips differ slightly (two p300c chips of one QuietBox 2 measured ~2.5% apart), which
the perf test's 10% margin covers.

## Expected accuracy

From test 1 (32 sentences, `casual_male` voice) and test 0 (8 frames, `casual_male` / `ar_male`):

| Metric | Device (test 1) | Test 0: casual_male / ar_male | Reference (PyTorch, CPU) | Limit |
|---|---|---|---|---|
| Naturalness, mean UTMOS MOS (1-5) | 3.668 | -- | 3.739 | >= reference - 0.2 |
| Intelligibility, corpus WER | 0.017 (up to ~0.05 on some runs) | -- | 0.023 | <= reference + 0.05 |
| Prefill hidden worst per-row PCC | 0.997 | 0.997 / 0.991 | -- | >= 0.99 |
| Decode hidden worst per-row PCC | 0.9985 | 0.999 / 0.999 | -- | >= 0.99 |
| Acoustic `x_final` worst per-row PCC | 0.997 | 0.9995 / 0.99995 | -- | >= 0.99 |
| Waveform worst per-row PCC (teacher-forced codec) | 0.9991 | 0.9996 / 0.9988 | -- | >= 0.99 |
| Acoustic codes equal to the teacher-forced reference | 98.9% of live codes | 99.1% / 99.0% | -- | every decidable code, and >= 98% |
| Semantic codes equal to the teacher-forced reference | 99.97% | 100% / 100% | -- | every decidable code |

Whisper occasionally collapses one whole clip to a single word (e.g. "Yeah.") although the audio is correct,
which lifts the corpus WER toward 0.05 on some runs; transcribing that clip in halves recovers the sentence.

## Accuracy fix (why the acoustic stage runs at full precision)

The acoustic sampler is numerically sensitive: classifier-free guidance at 3.0 over 7 Euler steps amplifies
small errors at some elements, and its output is rounded onto a 21-level grid. An automated optimization run
had lowered the acoustic stage's precision step by step, and together those steps failed the accuracy test
(MOS 3.13, acoustic worst PCC 0.71, 30% of acoustic codes wrong). Bisected on device, the acoustic stage now
undoes all of them:

| Precision cut (optimize run) | Now |
|---|---|
| attention output and FFN down weights in bfloat4_b; other weights in bfloat8_b | all acoustic weights bfloat16 |
| HiFi2 / LoFi math fidelity | HiFi4 |
| block-sharded `ttnn.rms_norm` (bfloat16 reduction scaler) | the exact spelled-out RMSNorm |
| bfloat16 norm, q/k/v, attention-context, gate/up, SiLU-product and o_proj outputs | float32 |
| 8-tile K blocks on the qkv projection (more bfloat16 partial-sum roundings) | widest K blocks |
| C++ SwiGLU kernel (bfloat8_b weight, bfloat16 output) | removed; stock ttnn matmuls |

The exact-math optimizations (batch folded into the matmul rows, compact layouts, fused ops, program configs,
L1 placement) are kept. Cost: acoustic 21.9 -> 41.8 ms per frame; running all 32 rows (64 with guidance)
through one body instead of two 16-row halves later brought it to 25.8 ms at the same precision.

One change outside the acoustic stage: the prefill's fused gate/up weight is bfloat8_b instead of
bfloat4_b. At 4 bits the prefill hidden state fell to 0.980 PCC for some voices (e.g. `ar_male`); at 8 bits
it is 0.991-0.997, and the decode hidden that reads the prefill's KV cache improves from 0.993 to 0.999.
Cost: prefill +1.9 ms, once per request.

A cheaper middle point restores only the two bfloat4_b weights to bfloat8_b and the (since removed) C++ SwiGLU
kernel to HiFi2: 60.7 ms per frame, MOS 3.62 and WER pass, but it fails `test_per_stage_pcc` (acoustic worst PCC ~0.85).

## Limitations

- **Batch 32 only.** The pipeline and its program configs are built and validated for 32 rows.
- **Equal-length texts.** The prefill has no per-row padding mask, so every row's text must tokenize to the
  same length; the package's 32 speech texts are tuned to exactly 18 tokens and `build_voice_prompt` refuses a
  ragged batch.
- **No audio input.** The open checkpoint ships no codec encoder, so there is no voice cloning from audio;
  voices come from the 20 presets.
- **Prefill hidden state at individual text positions.** The test's prefill check pools each sentence's
  whole prompt into one PCC, which passes (>= 0.99). Looked at one position at a time, a few text positions
  deviate much more (worst position PCC ~0.4-0.6 across 32 sentences). This is not weight precision -- it
  persists with every text-stack weight in bfloat16 -- but the bfloat16 accumulation in the tall prefill
  matmuls (fp32 accumulation does not fit their current block configuration). Decode reads every prompt
  position's keys and values from the KV cache, so these positions do feed the output; the checks downstream of
  them (decode hidden states, codes, waveform, WER and MOS) all pass.
- **Single segment.** Long-form multi-segment TTS (as in vLLM-Omni) is not implemented; the safety cap is 256
  frames (20.5 s).
