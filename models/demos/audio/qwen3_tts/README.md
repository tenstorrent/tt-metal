<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3-TTS

Text-to-speech with voice cloning ([Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base))
on Tenstorrent hardware. A 28-layer talker generates codebook 0 of each 12.5 Hz frame, a
5-layer code predictor fills codebooks 1 to 15, and a 0.2B neural codec decodes the codes to a
24 kHz waveform. Apache 2.0.

| Component | Where | Accuracy |
|---|---|---|
| Checkpoint access (`weights.py`) | host | |
| Mel front-end, 128 bins at 24 kHz (`audio.py`) | host | matches upstream exactly |
| Tokenizer and prompt assembly (`frontend.py`, `tt/ttnn_qwen3_pipeline.py`) | host | prompts bit-exact against upstream |
| Sampling (`sampling.py`) | host | matches `transformers` |
| Speaker encoder (ECAPA-TDNN) | device | PCC 0.999996 |
| Talker (28 layers, MRoPE), KV cache + trace | device | PCC 0.995 |
| Code predictor (5 layers, 15 steps per frame), KV cache + trace | device | logits PCC 0.995 |
| Codec decoder, codes to waveform (1920 samples a frame) | device | waveform PCC 0.995 |
| Codec encoder, waveform to codes | device | latents PCC 0.9999 |

## Releases and sizes

Five releases, all pinned in `weights.RELEASES`. They share one architecture and differ in how
the voice is chosen; `tts_model_type` in `config.json` says which, and the pipeline refuses the
wrong input.

| release | 1.7B | 0.6B | voice from |
|---|---|---|---|
| Base | yes | yes | a reference clip (speaker encoder, empty `spk_id`) |
| CustomVoice | yes | yes | nine named speakers, no encoder |
| VoiceDesign | yes | **no** | a sentence of English |

0.6B is the 1.7B architecture at half the talker's width (`hidden_size` 1024, `intermediate_size`
3072, speaker `enc_dim` 1024). Layers, heads (16 query over 8 KV), `head_dim` 128, the codec and
the whole code predictor are unchanged, so at 0.6B the attention's head space (2048) is wider
than the hidden size; code uses `heads * head_dim`, never `hidden`. 0.6B has no
`small_to_mtp_projection`, which becomes an identity, as upstream does. The pipeline refuses an
instruction on a 0.6B checkpoint, where upstream silently drops it.

## Checkpoint

Fetched from the HF hub on first use (3.6 GB at 1.7B, 1.8 GB at 0.6B), or point
`$QWEN3_TTS_CKPT` at a local directory holding `config.json` and `model.safetensors`:

```bash
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir qwen3_tts_ref
export QWEN3_TTS_CKPT=$(pwd)/qwen3_tts_ref
```

`weights.py` resolves `$QWEN3_TTS_CKPT`, then `$HF_MODEL` (a hub id or a path, the tiered-CI
convention), then the default repo. `$QWEN3_TTS_REVISION` overrides the pinned revision of the
ambient checkpoint. For 0.6B, set `HF_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base` or pass `--ckpt` to
the demos. The tests run on Base and switch to the CustomVoice or VoiceDesign sibling at the
same size where they need it (`tests/checkpoints.py`).

No dependencies beyond the tt-metal environment: `safetensors`, `huggingface_hub`, `librosa`
and `soundfile` all ship in `python_env`, so there is no `requirements.txt`.

## Demo

One utterance:

```bash
python -m models.demos.audio.qwen3_tts.demo.demo "Text to speak." \
    --ref my_voice.wav --ref-text "exactly what my_voice.wav says"
```

Interactively, loading the weights once and speaking every line you type:

```bash
python -m models.demos.audio.qwen3_tts.demo.demo_server \
    --ref my_voice.wav --ref-text "exactly what my_voice.wav says"
```

```
text [1]> One.  This is the first line the server speaks today.
  END-TO-END: 14.20 s  |  5.60 s audio (0.39x faster than real time)  |  outputs/out_1.wav
    prefill 1.6 s, capture 3.3 s, decode 2.4 s (70 frames at 34 ms), codec 6.7 s
text [2]> Two.  And here is a second one, in the same voice.
  END-TO-END: 2.55 s  |  5.60 s audio (2.19x faster than real time)  |  outputs/out_2.wav
    prefill 0.02 s, capture 0.04 s, decode 2.2 s (70 frames at 32 ms), codec 0.27 s
```

The first utterance compiles its kernels; later ones run from captured traces. Server commands:
`\ref PATH | TRANSCRIPT`, `\instruct DESCRIPTION`, `\speaker NAME`, `\language NAME`, `\seed N`,
`\similarity`, `\quit`.

`--ref-text` must be the clip's transcript, word for word: in-context cloning puts it in the
prompt beside the clip's codes. Use three to ten seconds of clean speech. Other ways to pick a
voice, on either demo:

- `--x-vector` with `--ref`: clone from the voice alone, no transcript needed.
- `--speaker ryan`: a CustomVoice speaker (needs a CustomVoice `--ckpt`).
- `--instruct "A calm older man speaking slowly, with a slight rasp."`: design a voice
  (needs a VoiceDesign `--ckpt`). With `--speaker` it directs that speaker instead; it also
  combines with `--x-vector`.
- `--streaming`: stream the text in a token per frame (see below).

Other flags: `--language`, `--seed`, `--max-frames` (default 400), and `--out` / `--no-similarity`
(`demo.py`) or `--output-dir` (`demo_server.py`).

## Python API

```python
from models.demos.audio.qwen3_tts import audio
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import Qwen3TTSPipeline, build_clone_reference

# Base: clone a clip, in context or from the voice alone (no transcript)
clip = audio.read_clip("reference.wav")                       # any rate, resampled to 24 kHz
reference = build_clone_reference(device, clip, "what the clip says")
voice = build_clone_reference(device, clip, x_vector_only=True)
pipeline = Qwen3TTSPipeline(device, max_frames=400, seed=0)
waveform, codes = pipeline.generate_clone("New words in that voice.", reference)
waveform, codes = pipeline.generate_clone("New words in that voice.", voice, x_vector_only=True)

# CustomVoice: a named speaker, optionally directed
waveform, codes = pipeline.generate("The kettle is on.", speaker="ryan", instruct="Whisper it.")

# VoiceDesign: a voice described in a sentence
waveform, codes = pipeline.generate_design("Never recorded.", "A calm older man with a warm low pitch.")
```

| how the voice is chosen | prompt length | release | entry point |
|---|---|---|---|
| a named speaker | `n_text + 11` | CustomVoice | `generate(text, speaker=...)` |
| a named speaker, directed | plus the instruction | CustomVoice | `generate(..., instruct=...)` |
| described in words | plus the instruction, no speaker position | VoiceDesign | `generate_design(text, instruction)` |
| a clip's voice alone | `n_text + 11` | Base | `generate_clone(..., x_vector_only=True)` |
| a clip, in context | plus one position per reference frame | Base | `generate_clone(text, reference)` |

An instruction combines with every mode except in-context cloning, where upstream's prompt has
no room for one.

`build_clone_reference` runs the codec encoder and the speaker encoder once; keep the
`CloneReference` to reuse a voice. Call it before the pipeline's first generation, or call
`pipeline.release()` first: eager work beside a live trace hangs the card. For in-context
cloning the codec decodes the reference frames together with the generated ones, because the
decoder is causal and the first generated frames need the reference as context, then cuts them
off the waveform. On a 7.28 s clip, speaker similarity (the speaker encoder's cosine) is 0.9940
in context and 0.9925 from the voice alone, against about 0.82 for an unrelated voice; the
voice-alone number is biased upward, since that mode is fed the vector the metric measures. The
voice-alone prompt is 23 positions, the in-context one 132.

**Streaming text** (`streaming=True` on all three entry points, upstream's default regime): the
text reaches the model one token per frame instead of all in the prompt, so the CustomVoice
prompt is 10 positions whatever the text length. `text` may also be an iterable of pieces that
arrive during generation (`StreamingText` carries the schedule). Each piece is tokenised on its
own, so feed whole words or clauses. A streaming clone sums the text and codec tracks position by
position, so it still waits for as much text as the clip has frames before the prompt closes.

**Languages**: all ten, plus two Chinese dialects via the speakers `eric` (Sichuanese) and
`dylan` (Beijing). Whisper-small transcriptions scored CER 0.000 on eight languages; Japanese
(0.050) and Chinese (0.385) differ only in homophone spelling and Traditional against Simplified
characters.

## Performance

Throughput is audio over wall clock, so above 1 is faster than real time. Warm, CustomVoice,
`ryan`:

| device | size | talker step | predictor, 15 steps | per frame | faster than real time |
|---|---|---|---|---|---|
| Blackhole, one P300 chip | 1.7B | 9.8 ms | 19.8 ms | 31.9 ms | **2.35x** (12.6 s of audio in 5.35 s) |
| Wormhole N150 | 1.7B | 15.6 ms | 31.2 ms | 49.3 ms | **1.51x** (13.3 s in 8.8 s) |
| Wormhole N150 | 0.6B | 9.7 ms | 31.6 ms | 43.8 ms | **1.70x** (12.6 s in 7.4 s) |

On Blackhole the per-utterance costs are small: prefill 0.03 s, both trace captures 0.05 s,
codec 0.27 s. A one-second utterance still comes out at 1.7x because they are a third of its
wall clock. 0.6B is not twice as fast: only the talker's projections and MLP halve, and the
predictor is the same model at both sizes.

`tests/perf/test_perf.py` prints this table block by block. It syncs the device at every split,
which costs a few percent, so `decode_s` is the honest total.

What made it fast:

- **Fused rotation**, `ttnn.experimental.rotary_embedding_hf`: 7.8 us against 26.7 for eight
  heads, same error.
- **Swept matmul program configs** (`decode_matmul_config`): every winner uses 11 to 22 cores
  rather than 64; `down_proj` went 112.8 to 69.6 us.
- **MLP weights in `bfloat8_b`** (`MLP_WEIGHT_DTYPE`). Norm weights stay bf16 (bfp8 norms: PCC
  0.9751 against 0.9954); attention weights stay bf16 too (below).
- **The predictor's codebook lookups and the talker's hidden state stay on device**: 1.1 ms a
  frame, identical values.
- **Shape buckets.** Prompts round up to 32 positions (`PROMPT_BUCKET`) and codec lengths to 32
  frames (`LENGTH_BUCKET`), because each new shape compiles (a new prompt length costs 1.41 s
  once). Each codec length also holds 8 to 19 KB of L1_SMALL until the program cache is cleared,
  so the pipeline clears it before an uncompiled length (0.12 s) rather than run out of the 64 KB
  region. Padding changes nothing kept: identical frames for prompts, PCC 0.9999 for the codec.

Measured and rejected; don't retry without a new reason:

- **Sampling on device.** `ttnn.sampling` costs 0.298 ms a call against 0.19 ms of host work,
  and the predictor would call it 15 times a frame. That also rules out one trace over all 15
  predictor steps.
- **Attention matmuls in `bfloat8_b`.** 9.2 ms against 9.6 for the talker step, but the sampling
  distance went to 0.1455 against a 0.098 noise band. It would need a listening test.
- **Chunked codec decoding.** The codec's transformer attends over the whole prefix, so a 32-frame
  chunk with 16 frames of context scores PCC 0.51. Streaming the codec needs carried conv state
  and a KV cache.

Open: fewer ops per predictor step (at 5.4 us per traced op, its 21 ops a layer are half its
time), a streaming codec for latency, and batch above 1.

## Numerics

- **Sample; don't decode greedily.** The checkpoint ships `do_sample: true`. Greedy decoding runs
  away: upstream on CPU used 699 of 700 frames on a four-sentence prompt, against 414 and a clean
  stop sampled. `sampling.py` follows `transformers`' processor order (repetition penalty,
  temperature, top_k, top_p) with the settings from `generation_config.json`, and the top token
  always survives top_p, so `top_p=0` is greedy. Pass `seed` to `Qwen3TTSPipeline` for a
  reproducible run. Two upstream rules not in the config also apply: control ids above the 2048
  codebook entries are suppressed except end-of-speech, and end-of-speech waits for two frames.
- **Judge by sampling distribution, not top-1.** On a real prompt the reference's own top-1
  probability is under 0.1 at several positions, and perturbing the prompt by a quarter of a bf16 rounding step scatters top-1
  agreement over 22 to 24 of 26. The tests gate total variation distance at the shipped
  temperature (0.076 to 0.098 for the talker, about 1.0 for a wiring error), per-step PCC for
  greedy chains, and generation stopping over eight seeds.
- **The talker's `codec_head` runs HiFi4 with fp32 accumulation and fp32 logits.** With ttnn's
  defaults its worst sampling distance was 0.035 at 1.7B and 0.081 at 0.6B; now 0.003 and 0.0008,
  for about 4 us a call.
- **The codec encoder runs in fp32**, alone among the blocks: in bf16 the latents drop to PCC
  0.9984 and a rounding error becomes a different code. It runs once per reference clip.
- **The 0.6B code predictor is less exact.** Without the projection its bf16 residual stream runs
  near 2665, where bf16 steps by 16. Worst greedy per-step logits PCC 0.9755 on Wormhole and
  0.9633 on Blackhole against 1.7B's 0.9914, sampling distance 0.184 against 0.085, so the two
  predictor test files have per-size gates. fp32 activations cut the distance to 0.118 but were
  not taken, for speed. Utterances still stop over eight seeds.
- **Padding modes matter.** The speaker encoder's reflect padding is built by hand, since
  `ttnn.conv1d` lacks it; zero padding still scores 0.9961, so that test gates at 0.999. The codec
  encoder's `downsample` pads by replication; zero padding costs 0.009 PCC on the tensor the codes
  come from.
- **Prepared conv weights are cached per input length**, since `ttnn.conv1d` prepares them for a
  length-dependent parallelisation. Caching by name alone dropped a second decode from 0.995 to
  0.104.

## Hardware

Single chip, batch 1. Bring-up ran on Blackhole (P300); the 0.6B work and the N150 numbers are
from a Wormhole N150. Wormhole needed three things Blackhole did not:

- **Codec decoder convs keep their config tensors in DRAM** (`config_tensors_in_dram=True`). In
  L1_SMALL, Wormhole hung once a decode reached 64 frames, and a hang left running took the host
  down with a fatal hardware error.
- **64 KB of L1_SMALL** for the codec tests, the pipeline's own figure; at 32 KB two decode
  lengths in one process do not fit.
- **A codec stage gate of 0.985** (`WORMHOLE_STAGE_PCC`) instead of 0.99; the waveform still
  clears 0.99. HiFi3 measured no better than HiFi4, so the model stays on HiFi4.

## Tests

References are computed live from the checkpoint, so the suite needs only the checkpoints and,
for device tests, a card. The same tests run at either size; at 0.6B the VoiceDesign and
instruction tests skip.

CI runs the module PCC files and `tests/e2e/test_e2e.py` (see [CI](#ci)). Everything else is
run by hand:

```bash
pytest models/demos/audio/qwen3_tts/tests/                             # everything, at 1.7B
HF_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base pytest models/demos/audio/qwen3_tts/tests/   # at 0.6B
```

| file | covers | in CI |
|---|---|---|
| `tests/pcc/test_speaker_pcc.py` | speaker encoder | both sizes |
| `tests/pcc/test_talker_pcc.py` | talker, per layer and end to end | both sizes |
| `tests/pcc/test_code_predictor_pcc.py` | code predictor | both sizes |
| `tests/pcc/test_decode_pcc.py` | cached, traced talker and predictor against the uncached graphs | both sizes |
| `tests/pcc/test_codec_pcc.py` | codec decoder, including bucketing and the per-length weight cache | 1.7B |
| `tests/pcc/test_codec_encoder_pcc.py` | codec encoder, latents and codes | 1.7B |
| `tests/e2e/test_e2e.py` | accuracy, determinism and perf of one utterance | both sizes |
| `tests/test_checkpoint_loading.py` | checkpoint layout against `config.json` (host) | no |
| `tests/test_tokenizer.py` | token ids in ten languages, prompt scaffolding (host) | no |
| `tests/test_sampling.py` | sampler against `transformers` (host) | no |
| `tests/pcc/test_pipeline.py` | CustomVoice prompts, the decode loop, every language, prompt buckets | no |
| `tests/pcc/test_clone_pcc.py` | voice cloning, both modes | no |
| `tests/pcc/test_voice_design_pcc.py` | VoiceDesign | no |
| `tests/pcc/test_streaming_pcc.py` | streaming prompts and text feed | no |
| `tests/pcc/test_generation_stops.py` | utterances stop, over eight seeds | no |
| `tests/perf/test_perf.py` | per-block perf table | no |

Upstream's prompt assembly needs transformers 4.57.3, which cannot share an environment with
this repository's version. It was diffed offline at max absolute difference 0.0 for every voice
mode in both regimes, and the tests pin the composition instead. The e2e test's teacher-forcing
references in `tests/e2e/reference_outputs/` are regenerated on CPU with
`python -m models.demos.audio.qwen3_tts.tests.e2e.generate_reference`, once per `HF_MODEL`.

## CI

Tier 2 on WH N150 and BH P150, one leg per size: model identifiers `qwen3-tts-1.7b` and
`qwen3-tts-0.6b`, with `HF_MODEL` set to the canonical HF name. Dispatch a single run from
[`all-model-tests`](https://github.com/tenstorrent/tt-metal/actions/workflows/all-model-tests.yaml)
with tier 2 and that identifier.

The unit legs (`tests/pipeline_reorg/models_unit_tests.yaml`) run only the module PCC files: the
speaker encoder, talker, code predictor and cached decoders at both sizes, and both codec halves
at 1.7B, since the codec is identical at both sizes.

The e2e legs (`tests/pipeline_reorg/models_e2e_tests.yaml`) run `tests/e2e/test_e2e.py` on the
CustomVoice release with the watcher off, and report benchmark payloads checked against
`models/model_targets.yaml`:

- **accuracy**, top-1 and top-5 by teacher forcing: the device is fed the frames the CPU
  reference sampled and is scored at every codebook of every frame on whether its argmax, or its
  top five, holds the reference's argmax;
- **determinism**, the same seed twice giving the same frames and audio, asserted in the test;
- **perf**, warm, batch 1: time to the first frame as time-to-token, frames per second as
  tokens/s/user.

On one N150:

| size | top-1 | top-5 | first frame | frames/s |
|---|---|---|---|---|
| 1.7B | 86.20% | 99.86% | 0.10 s | 20.3 |
| 0.6B | 86.99% | 99.92% | 0.10 s | 22.9 |

The targets stay TODO until a CI run on each SKU gives runner numbers.

Timeouts are the cold-kernel-cache time on the CI runners plus 20%, the unit legs with the
watcher on. The runners take about 2.35x a local N150's time on Wormhole and 1.85x on Blackhole,
measured file by file on the first CI runs, so the timeouts scale the local cold times by those:

| leg | local N150 cold | `wh_n150` | `bh_p150` |
|---|---|---|---|
| 1.7B unit | 8.6 min | 25 | 20 |
| 0.6B unit | 3.2 min | 10 | 8 |
| 1.7B e2e | 2.1 min | 6 | 5 |
| 0.6B e2e | 1.7 min | 5 | 4 |

The weights come from the runners' read-only `/mnt/MLPerf/huggingface` cache, which has to hold
the Base and CustomVoice releases of both sizes at the revisions `weights.py` pins.

## Directory layout

| Path | Role |
|---|---|
| `weights.py` | checkpoint resolution, pinned releases, weight readers |
| `frontend.py` | host text path: tokenizer, prompt wrappers, language resolution |
| `sampling.py` | host sampler, matching `transformers`' processor order |
| `audio.py` | host audio path: file to 24 kHz waveform, waveform to log-mel |
| `demo/` | one-shot CLI (`demo.py`) and interactive server (`demo_server.py`) |
| `tt/` | TTNN blocks; `ttnn_qwen3_pipeline.py` builds prompts and runs the frame loop |
| `reference/` | CPU references (PCC oracles); `reference/qwen/` is vendored upstream, Apache-2.0 |
| `tests/` | host tests; `pcc/` device correctness, `e2e/` the CI e2e test, `perf/` the perf table |

`reference/qwen/` is vendored because the `qwen-tts` package pins transformers 4.57.3, which
conflicts with the version this repository runs. Its license is `reference/qwen/LICENSE-qwen-tts`.
`reference/qwen/speaker_encoder.py` is a byte-for-byte copy of the upstream encoder apart from two
mechanical deviations recorded in its header. Treat it as an oracle: any edit that is not a
faithful copy makes it useless.
