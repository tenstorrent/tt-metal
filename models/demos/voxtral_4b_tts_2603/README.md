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
                 text_stack.py, acoustic_stage.py, vocode_stage.py, common.py
  reference/     CPU reference chains (golden.py) and speech scoring (quality.py: Whisper WER, UTMOS MOS)
  demo/          demo_text_to_speech.py
  tests/e2e/     test_e2e_text_to_speech.py (accuracy), test_text_to_speech_perf.py (performance)
models/tt_transformers/demo/voxtral_4b_tts_2603/
  _stubs/        the 31 TTNN model parts the pipeline is composed from
  tests/pcc/_reference_loader.py   builds the PyTorch reference model from the native checkpoint
```

## Setup

Build tt-metal and its Python environment as usual (`./build_metal.sh`, `./create_venv.sh`), then:

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1   # show test output live when piping through tee
```

The checkpoint (~8 GB) downloads from Hugging Face on first use. The accuracy test and the demo's `--score`
also download Whisper large-v3-turbo and UTMOS22 for scoring. Pick the chip with `VOXTRAL_DEVICE_ID` (accuracy
test) or `TT_PERF_DEVICE_ID` (perf test); both default to 0. If opening the device fails with
`failed to initialize FW! Try resetting the board`, run `tt-smi -r` (resets every chip on the host).

## Tests to run

| # | Test | What it checks | Time | Pass criteria |
|---|---|---|---|---|
| 1 | `pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_e2e_text_to_speech.py -s` | **Accuracy.** 32 sentences generated on device to their own stop, compared with the PyTorch reference on CPU: per-stage PCC, discrete codes, waveform PCC, and the audio scored for intelligibility (Whisper WER) and naturalness (UTMOS MOS) | ~20 min first run (the CPU reference is cached in `/tmp/voxtral_4b_tts_2603_golden` afterwards) | all 11 tests pass |
| 2 | `pytest models/demos/voxtral_4b_tts_2603/tests/e2e/test_text_to_speech_perf.py -s` | **Performance.** Each stage traced and replayed on one command queue; prints per-stage ms, ms per audio frame, frames/s | ~1-2 min | completes; numbers as in *Expected performance* |
| 3 | `python -m models.demos.voxtral_4b_tts_2603.demo.demo_text_to_speech --device-id 0 --out-dir <dir> --score` | **Listen.** Writes one WAV per sentence (cut at its own end) and prints each clip's WER and MOS | ~3 min | corpus WER / mean MOS as in *Expected accuracy* |

`demo_text_to_speech.py --text "Your sentence."` speaks your own text; `--voice` picks another preset.

### What the accuracy test checks

The device side runs free, exactly as the demo does. The reference is run twice on the same inputs,
voice and noise: once free-running (the baseline for the WER/MOS scores) and once **teacher-forced** onto the
device's own trajectory (fed the device's codes and hidden states at every frame), so every per-stage
comparison measures the device's arithmetic rather than the drift between two independent runs.

| Test | Checks |
|---|---|
| `test_golden_is_the_references_own_arithmetic` | the reference chain equals the reference model's own code |
| `test_gate2_every_call_1_stub_was_invoked` | all 28 model parts on the TTS path ran on device |
| `test_run_ended_on_the_models_stop_rule` | every sentence ended on the model's end-of-audio code, not the 256-frame cap |
| `test_shapes_and_real_task_output` | the output is real 24 kHz audio (finite, in range, not constant) |
| `test_batch_is_32_independent_samples` | 32 inputs give 32 distinct waveforms |
| `test_per_stage_pcc` | worst per-row PCC >= 0.99 for prefill hidden, decode hidden, semantic logits and the acoustic output `x_final`, over every frame |
| `test_discretization_is_the_references_own_rule` | the device turns `x_final` / logits into codes with the reference's exact rule |
| `test_discrete_codes_equal_the_teacher_forced_reference` | codes equal the reference's wherever the reference's own decision is clear of the device's measured noise |
| `test_signal_quality_wer_and_mos` | device WER <= reference WER + 0.05 and device MOS >= reference MOS - 0.2 |
| `test_free_running_divergence_is_reported` | the first frame's semantic code matches (diagnostic otherwise) |
| `test_gate3_e2e_pcc` | the final waveform PCC >= 0.99 against the teacher-forced reference |

## Expected performance

Measured with test 2 on **one Blackhole p300c chip** (QuietBox 2), batch 32, trace + 1 command queue:

| Stage | Time per step |
|---|---|
| prefill (once per request) | 86.8 ms |
| decode (every frame) | 37.1 ms |
| acoustic (every frame) | 41.8 ms |
| vocode (32-frame chunk, 2.56 s of audio per row) | 80.1 ms |

| Throughput | Value |
|---|---|
| ms per audio frame (decode + acoustic), all 32 users | **78.9 ms** |
| frames/s per user (real time = 12.5) | **12.7** (1.01x real time) |
| frames/s total | **406** (~32 s of audio per second) |

These are traced, steady-state stage times. A request additionally pays prefill before its first frame and
vocode after its last.

## Expected accuracy

From test 1 (32 sentences, `casual_male` voice):

| Metric | Device | Reference (PyTorch, CPU) | Limit |
|---|---|---|---|
| Naturalness, mean UTMOS MOS (1-5) | 3.695 | 3.739 | >= reference - 0.2 |
| Intelligibility, corpus WER | 0.017-0.05 | 0.023 | <= reference + 0.05 |
| Acoustic `x_final` worst per-row PCC | 0.994 | -- | >= 0.99 |
| Decode hidden worst per-row PCC | 0.994 | -- | >= 0.99 |
| Acoustic codes equal to the teacher-forced reference | 99.0% | -- | every decidable code |
| Semantic codes equal to the teacher-forced reference | ~100% | -- | every decidable code |

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
| C++ SwiGLU kernel (bfloat8_b weight, bfloat16 output) | off unless `VOXTRAL_CPP_SWIGLU=1` |

The exact-math optimizations (batch folded into the matmul rows, compact layouts, fused ops, program configs,
L1 placement) are kept, and the text stack and codec are unchanged. Cost: acoustic 21.9 -> 41.8 ms per frame.

A cheaper middle point restores only the two bfloat4_b weights to bfloat8_b and the SwiGLU kernel to HiFi2:
60.7 ms per frame, MOS 3.62 and WER pass, but it fails `test_per_stage_pcc` (acoustic worst PCC ~0.85).

## Limitations

- **Batch 32 only.** The pipeline and its program configs are built and validated for 32 rows.
- **Equal-length texts.** The prefill has no per-row padding mask, so every row's text must tokenize to the
  same length; the package's 32 speech texts are tuned to exactly 18 tokens and `build_voice_prompt` refuses a
  ragged batch.
- **No audio input.** The open checkpoint ships no codec encoder, so there is no voice cloning from audio;
  voices come from the 20 presets.
- **Single segment.** Long-form multi-segment TTS (as in vLLM-Omni) is not implemented; the safety cap is 256
  frames (20.5 s).
