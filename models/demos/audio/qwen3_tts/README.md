<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3-TTS

Text-to-speech with voice cloning ([Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base))
on Tenstorrent hardware. The model generates discrete audio tokens with a 1.7B decoder at a
12.5 Hz frame rate, then decodes them to a 24 kHz waveform with a 0.2B neural codec. Apache 2.0.

Both sizes run on the same code: the 0.6B releases
([Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base),
[CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)) are the 1.7B
architecture at half the talker's width. See [Two sizes](#two-sizes).

## Status

Bring-up in progress. This directory holds what is finished, and nothing that is not.

| Block | Component | Where | State |
|---|---|---|---|
| 0 | Checkpoint access (`weights.py`) | host | **done** |
| 1 | Mel front-end, 128 bins at 24 kHz (`audio.py`) | host | **done**, matches upstream exactly |
| 2 | Speaker encoder (ECAPA-TDNN) → `[1, hidden]` | device | **done**, PCC 0.999996 |
| 3 | BPE tokenizer and prompt assembly (`frontend.py`) | host | **done** |
| 4 | Talker (28 layers, hidden 2048 or 1024, MRoPE) | device | **done**, PCC 0.995, KV cache + trace |
| 5 | Code Predictor (5 layers, 15 steps per frame) | device | **done**, logits PCC 0.995, KV cache + trace |
| 6 | Codec decoder → waveform (1920x) | device | **done**, waveform PCC 0.995 |
| 7 | Dual-track prompt + decode loop | host + device | **done**, end to end |
| 8 | Sampling (`sampling.py`) | host | **done**, matches `transformers` |
| 9 | Codec encoder, waveform → codes | device | **done**, latents PCC 0.9999 |
| 10 | Voice clone: reference clip → prompt → speech | host + device | **done**, prompt bit-exact |
| 11 | VoiceDesign: a voice described in a sentence | host + device | **done**, prompt bit-exact |
| 12 | Streaming text input, all voice modes | host + device | **done**, prompt bit-exact |
| 13 | Instruction with a named speaker | host + device | **done**, prompt bit-exact |
| 14 | Cloning from the voice alone (`x_vector_only`) | host + device | **done**, prompt bit-exact |
| 15 | The 0.6B releases, on the same code | host + device | **done**, same suite |

The speaker encoder reads a reference clip and emits one vector as wide as the talker (2048
at 1.7B, 1024 at 0.6B), which occupies a single position of the talker's prompt. Its width
matches the talker's hidden size, so nothing projects between them.

## Demo

Two entry points under `demo/`, mirroring the XTTS-v2 demo next door.

One utterance:

```bash
python -m models.demos.audio.qwen3_tts.demo.demo "Text to speak." \
    --ref my_voice.wav --ref-text "exactly what my_voice.wav says"
```

Or interactively, which loads the weights once and then speaks every line you type:

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

The first utterance compiles its kernels. Every one after it runs from the captured traces
and warm buckets, at **2.2x faster than real time**. `\ref PATH | TRANSCRIPT` switches
voice, `\similarity` reports how close each utterance is to the reference clip, `\seed N`
changes the sampler, `\quit` leaves.

`--ref-text` is required, and it is the easiest thing to get wrong. This model clones in
context: the prompt carries the clip's transcript beside its codes, so the model reads what
the clip said as well as hearing how it sounded. Give it three to ten seconds of clean
speech, and make the transcript cover the whole clip.

Either demo takes `--instruct "A calm older man speaking slowly, with a slight rasp."`
instead of `--ref`, which designs a voice from the description rather than a recording, or
`--speaker ryan` to use one of the nine CustomVoice voices. `--instruct` also combines with
`--speaker`, where it directs that speaker instead of inventing a voice, and with
`--x-vector`. `--x-vector` clones from the clip's voice alone and needs no `--ref-text`,
which is the flag to reach for when the transcript is unknown. `--ckpt` points at a checkpoint
directory, which you need for the first two, since each way of choosing a voice lives in a
different release.

## Voice cloning

Cloning joins all ten blocks. Hand it a reference clip and that clip's transcript, and it
speaks new text in the same voice:

```python
from models.demos.audio.qwen3_tts import audio
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import Qwen3TTSPipeline, build_clone_reference

clip = audio.read_clip("reference.wav")                       # any rate, resampled to 24 kHz
reference = build_clone_reference(device, clip, "what the clip says")
pipeline = Qwen3TTSPipeline(device, max_frames=400, seed=0)
waveform, codes = pipeline.generate_clone("New words in that voice.", reference)
```

`build_clone_reference` runs the codec encoder and the speaker encoder once and drops both,
so a server that clones one voice repeatedly should hold on to the `CloneReference` rather
than rebuild it. This is a **Base checkpoint** path: Base carries the speaker encoder and an
empty `spk_id`, CustomVoice the nine speakers and no encoder, so cloning and named speakers
are mutually exclusive releases.

The prompt is `generate_icl_prompt` with `non_streaming_mode=True`. Capturing upstream's
own assembly at the talker's door under transformers 4.57.3 and diffing the two gives **max
absolute difference 0.0** across all 72 positions, and 71 positions with the language tag
off. Two details govern how it behaves:

  * The text track carries the reference transcript **before** the text to speak, so the
    model reads what the clip said as well as hearing how it sounded. The codec track then
    carries the clip itself, one summed 16-codebook embedding per frame.
  * The codec decodes the reference frames alongside the generated ones, and the pipeline
    cuts them off the front of the waveform afterwards. That decoder is causal, so the first
    generated frames read the reference as context; decode them alone and the onset comes
    out different and worse. Each frame is 1920 samples, so the cut lands exactly.

Measured on one P300 chip, cloning a 7.28 s reference clip that CustomVoice had just
spoken, then saying 5.60 s of new text. Second utterance in that voice, so the only cold
cost left is the pair of encoders:

| stage | time |
|---|---|
| both encoders, once per clip | 7.9 s |
| prompt, 137 positions, and 70 frames | 2.3 s |
| codec decoder, 161 frames (the reference rides along) | 0.27 s |
| **total** | **2.55 s, 2.19x faster than real time** |

A long reference lengthens the codec stage, since the decoder sees its frames too. Both
encoders run once: keep the `CloneReference` and you skip them on the next utterance in that
voice. The first utterance in a process pays its compiles instead, 14.2 s for this one.

The speaker encoder answers whether the voice carried over. Cosine between the reference
clip's vector and the clone's: **0.9832**, against about 0.82 for an unrelated voice.

A clone can also run in the streaming regime, where the two tracks are summed instead of
placed one after the other. See **Streaming text input**.

## Voice design

The third way to pick a voice: describe it. No clip, no speaker id, one sentence of English.

```python
waveform, codes = pipeline.generate_design(
    "This voice was never recorded.",
    "A calm older man speaking slowly, with a slight rasp and a warm low pitch.",
)
```

This needs the **VoiceDesign** release, `tts_model_type: voice_design`. Its `talker_config`
is Base's field for field, so every ported block runs on it untouched and only the prompt
differs. `generate_design` checks `tts_model_type` first and says which checkpoint to fetch
rather than producing a voice the instruction never shaped.

Two things separate the prompt from CustomVoice's. It has **no speaker position**, which
shortens the head from nine positions to eight. And the instruction goes in **whole**: the
text to speak has its role tokens sliced off, the instruction keeps its `<|im_start|>user`
opener and `<|im_end|>` closer, and it sits on the text track with nothing added from the
codec track. Capturing upstream's own assembly under transformers 4.57.3 and diffing gives
**max absolute difference 0.0** at 37 positions with a language tag, 33 with `Auto`, and 20
with an empty instruction, which upstream treats as no instruction at all.

Whether the instruction reaches the voice is measurable two ways. At a fixed seed, changing
it moves 382 of 384 codes. And the Base checkpoint's speaker encoder puts two designs of the
same sentence at cosine **0.8923**, against 0.99 for one speaker and 0.82 for an unrelated
pair, so the description carries most of the way.

## The five ways to choose a voice

Upstream builds five shapes of prompt, and each works in either text regime. All five are
here:

| how the voice is chosen | prompt | release | entry point |
|---|---|---|---|
| a named speaker | `n_text + 11` positions | CustomVoice | `generate(text, speaker=...)` |
| a named speaker, directed | plus the instruction's tokens | CustomVoice | `generate(..., instruct=...)` |
| described in words | plus the instruction's tokens, no speaker position | VoiceDesign | `generate_design(text, instruction)` |
| a clip's voice alone | `n_text + 11`, the clip's length irrelevant | Base | `generate_clone(..., x_vector_only=True)` |
| a clip, in context | plus one position per reference frame | Base | `generate_clone(text, reference)` |

**Directing a named speaker** is upstream's `generate_custom_voice(..., instruct=...)`, which
the CustomVoice card documents with an example. It differs from VoiceDesign in what stays
fixed: the voice is the speaker's and the instruction shapes the delivery, rather than the
instruction inventing a voice. At one seed, "Say it in a very angry tone." took a line from
53 frames to 42 and "Whisper it, slowly and gently." took it to 111.

**Cloning from the voice alone** is upstream's `x_vector_only_mode`. Only the speaker
encoder runs, so there is no transcript to supply and no codes to carry, and the prompt is
the named-speaker one with a measured voice in the speaker position. Measured against
in-context cloning on a 7.28 s clip:

| | in context | voice alone |
|---|---|---|
| prompt | 132 positions | 23 |
| reference, once per clip | 4.39 s, both encoders | 3.38 s, speaker encoder only |
| codec decode | the clip's frames ride along | only what was generated |
| speaker similarity | 0.9940 | 0.9925 |

That last row flatters the voice-alone mode by construction: the metric is the speaker
encoder's cosine and this mode hands the model the very vector it is computed from. In
context carries the clip itself, which is detail the encoder does not necessarily measure.
Choose by ear. What is not in doubt is that a transcript is usually unavailable, which is
the mode's real argument, and that its prompt does not grow with the clip.

An instruction combines with every mode except in-context cloning, where upstream's prompt
leaves no room for one.

## Streaming text input

The second regime this model was trained in, and upstream's default: `non_streaming_mode=False`.
The only difference is **when** each text token reaches the model.

| | non-streaming | streaming |
|---|---|---|
| prompt, CustomVoice | `n_text + 11` positions | **10**, whatever the text |
| text during decode | a constant `tts_pad` | the next text token, one per frame |
| text needed to start | all of it | the first token |

```python
waveform, codes = pipeline.generate("The kettle is on.", speaker="ryan", streaming=True)

# or hand the text over as it arrives
waveform, codes = pipeline.generate(sentences_from_somewhere(), speaker="ryan", streaming=True)
```

`generate`, `generate_design` and `generate_clone` all take `streaming=True`, and the demos
take `--streaming`.

**What it buys.** The prompt stops growing with the text, so the prefill is a fixed ten
positions instead of one per token: a paragraph prefills what a sentence does. And the text
does not have to exist when generation starts, which is the part upstream leaves out. Its
own docstring is careful about that: the flag "only simulates streaming text input", since
it knows the whole string up front and changes nothing but the schedule. Here `text` may be
an iterable, and any piece that lands before the frame that needs it is indistinguishable
from having had the whole string. `StreamingText` carries the schedule.

The caveat with pieces is tokenisation: each piece is tokenised on its own, so a merge that
would have spanned a boundary does not happen. `"kettle"` is `[74, 47626]` whole and
`[25475, 11239]` split after three letters. Feed whole words, and prefer whole clauses.

**The clone prompt is a different shape again.** Upstream's ICL streaming does not put the
tracks one after the other; it sums them position by position and cuts to the shorter. The
reference transcript and the text to speak share the positions the clip's codes occupy, and
a text longer than the clip leaves the surplus to stream. A clip longer than the text, which
is the usual way round, pads the text track out and leaves nothing to stream, so decode adds
pads exactly as non-streaming does over a shorter prompt. Streaming a clone therefore still
waits for as much text as the clip has frames before the prompt can close.

**Verified against upstream, position by position.** Prompts and text tracks captured at the
talker's own door under transformers 4.57.3 and diffed: **max absolute difference 0.0**
across twelve comparisons, four CustomVoice cases (short, longer, `Auto`, a dialect speaker)
and two clone cases (a text shorter than the clip and one longer), each in both regimes. Two
things that comparison caught, both of which PCC would have shrugged at and a listener might
not: a second `tts_eos` after a prompt that had already closed the text track, 0.068 away
from upstream; and projecting the reference transcript separately from the text, 7.5e-08
away. `tests/pcc/test_streaming_pcc.py` pins both.

**Does it speak better?** Upstream's claim is that non-streaming is where the sampler
wanders, and the frame counts agree. Six seeds, frames per word:

| text | non-streaming | streaming |
|---|---|---|
| 4 words | 3.2, 9.8, 3.2, 5.8, 4.0, 3.8 | 6.2, 7.5, 4.2, 5.2, 5.8, 3.8 |
| 14 words | 9.1, 5.1, 3.3, 5.8, 4.6, 3.5 | 6.0, 5.1, 3.9, 4.3, 3.8, 3.9 |

Streaming's worst case is tighter on both (7.5 against 9.8, and 6.0 against 9.1) and its
spread is narrower, which is what less wandering looks like. Neither regime is uniformly
faster per word, and whether the speech is *better* is not something frame counts settle.

## Languages

Ten, and both Chinese dialects. The language reaches the model as a single
codec-vocabulary id in the think block and changes nothing else about the prompt, which is
why one mechanism covers all of them; `eric` and `dylan` override it to their dialect, as
upstream does.

All ten were generated and then transcribed by Whisper-small, character error rate against
the input text:

| language | CER | what the difference was |
|---|---|---|
| English, Korean, German, French, Spanish, Italian, Portuguese, Russian | **0.000** | punctuation only |
| Japanese | 0.050 | `湧いて` for `沸いて`: homophones, so the speech was right and the transcriber chose the other spelling |
| Chinese | 0.385 | transcribed in Traditional characters against a Simplified input, same words and same sounds |

So neither nonzero score is this model mispronouncing anything. `test_every_language_decodes_and_stops`
holds the nine non-English cases to what a test can judge without a second model in the
leg: each stops on its own, every code lands inside the codebook, and the length is speech
rather than a spent budget. The transcription is a measurement, not a gate, because putting
Whisper in CI would cost a 970 MB download and add its own failure modes.

## Two sizes

Five releases exist: 1.7B Base, CustomVoice and VoiceDesign, and 0.6B Base and CustomVoice.
**There is no 0.6B VoiceDesign.** Both sizes share the tokenizer, the codec (the identical
`speech_tokenizer/` weights) and `generation_config.json`, and their configs differ in four
fields:

| field | 1.7B | 0.6B |
|---|---|---|
| `talker_config.hidden_size` | 2048 | 1024 |
| `talker_config.intermediate_size` | 6144 | 3072 |
| `speaker_encoder_config.enc_dim` | 2048 | 1024 |
| `tts_model_size` | `1b7` | `0b6` |

Layers (28), heads (16 query over 8 key/value), `head_dim` (128) and the whole code predictor
are the same. So at 0.6B the attention's head space is **2048 wide against a hidden size of
1024**. `head_dim` is a config field and never `hidden // heads`, and the head-merge reshapes
use `heads * head_dim`. The one missing tensor is `talker.code_predictor.small_to_mtp_projection`,
which exists at 1.7B to take the talker's 2048 down to the predictor's 1024; at 0.6B the
widths already match and upstream builds an identity, as the device predictors do.

The pipeline refuses an instruction on a 0.6B checkpoint, where upstream silently drops it.

## Speed

**2.35x faster than real time, warm, on one P300 chip**, against 1.19x when this directory
first produced a waveform. A CustomVoice utterance of 12.6 s of speech,
157 frames, second run at that length:

| stage | time | per second of audio |
|---|---|---|
| prefill, 33 positions | 0.03 s | once per utterance |
| both trace captures | 0.05 s | once per utterance |
| talker + code predictor, 157 frames | 5.0 s | 0.40 s |
| codec decoder | 0.27 s | 0.02 s |
| **total** | **5.35 s** | **0.43 s, or 2.35x faster than real time** |

31.9 ms per frame, of which the talker's 28-layer step is 9.8 and the predictor's 15 steps
are 19.8. Both run from captured traces over a KV cache; the uncached talker step cost
2471 ms.

Every figure here is audio over wall clock, so above 1 is faster than real time. An earlier
version of this file quoted the reciprocal, 0.84 s of compute per second of audio, which is
the same thing said the other way round and easy to mistake for a speedup.

A one-second utterance comes out at 1.7x rather than 2.35x. The prefill, the two captures
and the codec are per utterance rather than per frame, and on 13 frames they are a third of
the wall clock.

### On Wormhole N150, and at 0.6B

`tests/perf/test_perf.py` on one N150, warm, CustomVoice, `ryan`. The block columns come from
the profiled run, whose device syncs cost a few percent:

| | talker step | predictor, 15 steps | per frame | long utterance | faster than real time |
|---|---|---|---|---|---|
| 1.7B | 15.6 ms | 31.2 ms | 49.3 ms | 13.3 s of audio in 8.8 s | **1.51x** |
| 0.6B | 9.7 ms | 31.6 ms | 43.8 ms | 12.6 s of audio in 7.4 s | **1.70x** |

Against Blackhole's 31.9 ms frame, Wormhole spends 1.6x on both decoders, and the predictor
is an even larger share of the frame: 63% at 1.7B, 72% at 0.6B. **0.6B is not twice as fast.**
Only the talker's projections and MLP halve. The predictor is the same model at both sizes,
and so is the talker's attention (head_dim 128 over the same KV cache). The decode matmul
configs were swept at 1.7B shapes on Blackhole; they are valid at 0.6B and on Wormhole but
have not been re-swept for either.

`tests/perf/test_perf.py` prints this table and charges the frame loop block by block, so a
regression says which block. It syncs the device at every split, which costs a few percent
of the frame and is why `decode_s` rather than the columns is the honest total. Where the
time went, against the first version that generated a waveform:

| block | before | after |
|---|---|---|
| talker step | 13.32 | 9.63 ms |
| code predictor, 15 steps | 25.65 | 19.81 |
| host sampling, all 16 codebooks | 1.82 | 1.93 |
| everything else in the frame | 0.4 | 0.4 |
| **per frame** | **41.1** | **31.9 ms** |
| prefill, warm | 1.5 s | 0.03 s |
| codec decoder, warm | 5-8 s | 0.27 s |

Four changes did it, and the order matters because the first two are what the third and
fourth were chosen by.

**The rotation in one kernel.** `ttnn.experimental.rotary_embedding_hf` replaces a slice per
half, a negate, a concatenation and three elementwise ops: 7.8 us against 26.7 for eight
heads, and the same error to seven digits under HiFi4 with fp32 dest accumulation. Both
decoders and the uncached graphs use it, which is what keeps them comparable.

**Matmul program configs, swept rather than reasoned about.** Every rectangle of Blackhole's
11 x 10 grid against all eight decode shapes, in a trace. Every winner spends 11 to 22
cores, not the 64 the old search picked by insisting the output tiles divide evenly across
them: at one position each core does almost no arithmetic, so the multicast dominates and a
wider spread costs more than it buys. `down_proj` went 112.8 to 69.6 us, `o_proj` 39.7 to
26.3, the predictor's `down_proj` 47.0 to 22.0. `decode_matmul_config` carries the table.

**The MLP's weights in `bfloat8_b`.** A single-position matmul is bandwidth bound on
weights, and the MLP is 60% of the step's weight bytes. Norm weights stay bf16: their values
cluster around 1 and a shared-exponent block of 16 quantises that to almost nothing, PCC
0.9751 against 0.9954. Attention's two matmuls stay bf16 as well, for a reason PCC does not
give; `MLP_WEIGHT_DTYPE` has all of it.

**The predictor's codebook lookups on the device.** Its 15 steps each fed the next position
a 2048-value row, and `ttnn.from_torch` of one costs 76 us against 4.5 for a copy of a
tensor already there. Now the step writes one index and the device reads its own table, and
the talker's hidden state reaches the predictor as the device tensor the talker produced
rather than a round trip through the host. 1.1 ms a frame, with identical values.

### Two things that are compiled per shape, and what that costs

**Prompt lengths.** Each one compiles its own prefill: 1.41 s the first time, 0.015 s after.
A server sees a new length per sentence, so prompts round up to 32 positions and four
lengths cost 1.11 s of prefill instead of 3.72 s. Padding is invisible because attention is
causal: the filler sits after the last real position, the hidden state decode starts from is
sliced at the true last one, and the filler's cache slots are the ones decode overwrites
before reading. `test_the_prompt_bucket_changes_nothing_it_keeps` holds it to identical
frames.

**Codec frame counts.** Same story with a harder edge: each length also holds 8 to 19 KB of
L1_SMALL convolution scratch until the device's program cache is dropped. Four lengths fill
the 64 KB region, measured at 16, 32, 50 and 58 KB, and the fifth fails to allocate, which
is what killed the interactive demo on its third line. Buckets of 32 frames keep repeats
free, and a length that has not been compiled since the last drop gets the cache dropped
first: that reclaims the whole region and costs 0.12 s, because the kernels stay built on
the host. The decoder throws the padding frames away and
`test_bucketing_does_not_change_the_samples_it_keeps` measures what that costs: PCC 0.9999
against an exact decode, inside what bf16 costs already.

Chunked decoding would have removed the per-length programs altogether and it does not work
here. The decoder's transformer attends over the whole prefix, so a 32-frame chunk carrying
16 frames of context scores 0.51 against a one-shot decode. Real streaming needs the
convolution state and a KV cache carried between chunks, not a recomputed window.

## Sampling

The checkpoint ships `do_sample: true`, and the pipeline follows it. **Greedy decoding is
not a safe simplification of this model.** Upstream's own package, on CPU, greedy, on a four
sentence prompt: 699 frames of a 700 frame budget, speaking about half the text and filling
the rest with a silence code. The same prompt sampled: 414 frames and a clean stop. The
device behaves the same way for the same reason, so `sampling.py` reproduces the library's
processor order (repetition penalty, then temperature, then top_k and top_p) and the
pipeline reads the settings out of `generation_config.json`.

Pass `seed` to `Qwen3TTSPipeline` for a reproducible run.

Two rules constrain the draw, both upstream's and neither in `generation_config.json`.
**Control ids never leave the talker**: its vocabulary is 3072 and the codec's codebooks
hold 2048, and upstream suppresses that gap except for end-of-speech, so a code reaching
the codec is always a real code. Measured before this port had it: 1.4e-09 of the
probability mass on average and never once inside the top 50 over 84 positions, so it
closes a case nobody had hit rather than changing any draw. **And end-of-speech waits for
two frames**, upstream's `min_new_tokens=2`, since one frame is 80 ms and not speech.

## Hardware

Bring-up ran on Blackhole; the 0.6B work and everything below was measured on a Wormhole
N150. CI covers both single-chip SKUs, Wormhole N150 and Blackhole P150, so an
architecture-specific regression shows up on whichever side it breaks. P150 stands in for
the dev machine's P300s: the model is single-device at batch 1, so a two-card SKU buys no
coverage, and P150 uses the standard shared weight cache while P300 runs in LFC mode and
would need to pull its own weights.

**Wormhole needed three things Blackhole did not.**

- **The codec decoder's convolutions keep their config tensors in DRAM.** In L1_SMALL,
  Wormhole hung once a decode reached 64 frames (about 5 s of audio): cold in the final conv,
  warm in the last transposed conv. Each op passes on its own at that length; it fails only
  beside the decode's other resident programs. A hang left running took the whole host down
  with a fatal hardware error. `config_tensors_in_dram=True` fixes it at 64 and 160 frames,
  cold and warm, with identical PCC.
- **The codec tests open the device with 64 KB of L1_SMALL**, the pipeline's own figure. At
  32 KB, two decode lengths in one process no longer fit.
- **A stage gate of 0.985 in `test_codec_pcc.py`** rather than 0.99. Wormhole's codec
  intermediates land lower on random codes (`decoder.3` at 0.9895 on one seed, 0.9963 on
  another) while the waveform clears 0.99. HiFi3, which tt-metal recommends on Wormhole with
  fp32 accumulation, measured no better than HiFi4, so the model stays on HiFi4.

## Dependencies

Nothing beyond the tt-metal environment. The reader uses `safetensors` and `huggingface_hub`,
and the mel front-end will use `librosa` and `soundfile`, all of which ship in `python_env`.
This directory carries no `requirements.txt` on purpose: adding one that installs nothing
would put the file under a codeowner for no gain. Add it when a real dependency appears.

## Checkpoint

Fetched from the HF hub on first use and cached (3.6 GB at 1.7B, 1.8 GB at 0.6B), or point
`$QWEN3_TTS_CKPT` at a local directory holding `config.json` and `model.safetensors`:

```bash
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir qwen3_tts_ref
export QWEN3_TTS_CKPT=$(pwd)/qwen3_tts_ref
```

`weights.py` resolves `$QWEN3_TTS_CKPT`, then `$HF_MODEL` (a hub id or a path, matching the
tiered-CI convention), then the default repo. Every release in `weights.RELEASES` is pinned
to a revision; override the ambient one with `$QWEN3_TTS_REVISION`. For 0.6B, set
`HF_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base`, or pass `--ckpt` to the demos.

Three kinds of release share one architecture and differ in which voices they answer to:
**Base** carries `speaker_encoder` and an empty `spk_id` and clones from a clip,
**CustomVoice** the nine named speakers and no encoder, **VoiceDesign** neither, taking a
sentence of English instead. `tts_model_type` in `config.json` is how each says which it is,
and the pipeline reads it before refusing the wrong input. The suite runs on a Base
checkpoint and switches to its CustomVoice and VoiceDesign siblings **at the same size** for
the files that need them (`tests/checkpoints.py`).

The Base checkpoint holds two top-level prefixes, `speaker_encoder.` (76 tensors, 12.0M
parameters at 1.7B, 8.9M at 0.6B) and `talker.` (the rest). Readers open the file lazily
and name their keys, so speaker-encoder work never materialises the talker.

## Tests

The suite is self-contained: references are computed live in-process from the checkpoint, so
it needs only the checkpoints and, for the device tests, a card. 148 tests, the same ones at
either size. On one N150, warm: **1.7B 147 passed and 1 skipped** in 10.6 min, **0.6B 136
passed and 12 skipped** in 7.0 min. The 1.7B skip is the test that 0.6B refuses an
instruction; the 0.6B skips are the ten VoiceDesign tests (there is no such release) and the
two that need an instruction to work.

```bash
HF_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base pytest models/demos/audio/qwen3_tts/tests/   # at 0.6B
```

**The 0.6B code predictor is less exact than 1.7B's, by a measured amount.** Its residual
stream is not scaled down by a projection and runs near 2665, where bf16 steps by 16. Per-step
greedy logits reach PCC 0.9755 against 1.7B's 0.9914, and the distance between the device's
and the reference's sampling distributions reaches 0.184 against 0.085. So
`test_code_predictor_pcc.py` and `test_decode_pcc.py` carry per-size gates with the
measurements beside them. fp32 activations cut that distance to 0.118 and were not taken,
since they cost time in the block that is already 72% of the 0.6B frame; utterances still
stop over eight seeds at 0.6B.

```bash
pytest models/demos/audio/qwen3_tts/tests/                             # everything
pytest models/demos/audio/qwen3_tts/tests/test_checkpoint_loading.py   # host only
pytest models/demos/audio/qwen3_tts/tests/test_tokenizer.py            # host only
pytest models/demos/audio/qwen3_tts/tests/pcc/test_speaker_pcc.py      # speaker encoder
pytest models/demos/audio/qwen3_tts/tests/pcc/test_talker_pcc.py       # talker
pytest models/demos/audio/qwen3_tts/tests/pcc/test_code_predictor_pcc.py  # code predictor
pytest models/demos/audio/qwen3_tts/tests/pcc/test_codec_pcc.py        # codec decoder
pytest models/demos/audio/qwen3_tts/tests/pcc/test_codec_encoder_pcc.py  # codec encoder
pytest models/demos/audio/qwen3_tts/tests/pcc/test_pipeline.py         # end to end, CustomVoice
pytest models/demos/audio/qwen3_tts/tests/pcc/test_clone_pcc.py       # end to end, voice clone
pytest models/demos/audio/qwen3_tts/tests/pcc/test_voice_design_pcc.py  # end to end, VoiceDesign
```

`test_checkpoint_loading.py` derives every speaker-encoder tensor name and shape from
`config.json` and checks them against the file, so a checkpoint that stops matching its own
config fails there rather than surfacing later as a PCC miss. Nothing is skipped when the
checkpoint is missing: a skip would turn an unreachable checkpoint into a green run.

`test_tokenizer.py` pins the ids for a phrase in each of the ten languages and checks the
prompt scaffolding has the shape the model was trained on: the text prompt leaves a turn
open for the model to continue, a reference transcript closes its turn, and a VoiceDesign
instruction speaks as the user. It also pins the seam that is easiest to get wrong later:
language never enters the text stream, and every language id falls inside the talker's
3072-entry codec vocabulary rather than the 151k text one.

`pcc/test_talker_pcc.py` runs a real prompt, built from real token ids through the model's
own embedding and projection path, and reports three things:

| measurement | value |
|---|---|
| per layer, each fed the reference's fp32 input | 0.9998 to 0.99999 |
| end to end, 28 layers of bf16 | **0.9954** |
| sampling distribution distance, mean | 0.098 |
| codec top-1 token agreement | 22/26, diagnostic only |

The per-layer number measures the implementation, since feeding each layer the reference's
own input removes accumulated drift: a wiring error shows as one bad layer, rounding shows
as nothing. The end-to-end number carries 28 layers of bf16 rounding on top.

**The distance is the verdict and the agreement is not**, which took a wrong turn to learn.
Agreement was the verdict until a fused rotation and a matmul config each flipped a pick
while leaving PCC where it was or better. Perturbing the prompt by a quarter of one bf16
rounding step settles it: agreement scatters over 22 to 24 of 26 and the device lands as
far down as the reference's 8th choice, because the reference's own top-1 probability is
under 0.1 at several positions here. Pick equality on this prompt is not a property any
implementation has. Total variation distance between the two sampling distributions, at the
temperature the checkpoint ships, moves with the whole vector instead: 0.076 to 0.098 over
those perturbations and across both spellings of the rotation, against roughly 1.0 for a
wiring error. `test_decode_pcc.py` judges the cached graph against the uncached one the same
way, for the same reason.

Neither of them sees an utterance that never stops, which is the failure this model actually
has. `pcc/test_generation_stops.py` is that test, and it takes eight seeds because at one
seed the measurement is noise: the same four-word sentence ran 3.5 and 38.0 frames per word
at two different seeds of one build.

The input choice is load-bearing. Random embeddings sit far outside the activation
distribution the weights were trained on, and the same graph scores 0.936 with 71% token
agreement on them. Raising device tensors to fp32 recovers almost nothing (0.9396), and fp32
weights change nothing at all, because the compute is bf16-class whatever the tensors say.

`pcc/test_code_predictor_pcc.py` runs a frame the model produced itself: the talker on a real
prompt, its last hidden state, codebook 0 from `codec_head`, then codebooks 1 to 15 decoded
greedily by the reference. Teacher-forced blocks hold 0.9965 to 0.99999 and the 15 output
heads reach 0.9946.

Greedy decode is scored per step, not as a sequence. One flipped token changes the input to
every later step, so comparing whole sequences measures the cascade rather than the port: the
device matches 12 of 15 steps but only 8 of 15 codes. Each disagreement is judged by how much
the reference prefers its own pick over the device's, which is the question worth asking.
Every one measured is a near-tie, widest gap 0.10 against logits spanning several units, and
at the single step where the device took the reference's third choice its top three sat within
0.042 of each other.

`pcc/test_speaker_pcc.py` gates every block and the embedding at **0.999**, not the usual
0.99. Upstream pads each convolution in reflect mode, which `ttnn.conv1d` cannot do, so this
port builds the mirrored columns by hand; substituting plain zero padding still scores 0.9961,
which a 0.99 gate would wave through. Two further tests keep the first one honest: the padding
is compared against `torch.nn.functional.pad` for an exact match, and the angle between a low
voice and a high one on device is checked against the same angle on the reference (0.9496 vs
0.9497), which a graph that ignored its input could not reproduce.

`pcc/test_codec_pcc.py` decodes codes to a waveform: stages 0.995 to 0.999997 and the
waveform 0.995. Test input is random *valid* codes, which is legitimate here and was not for
the talker: codes index learned codebooks, so any valid code gives an in-distribution latent
by construction, and what random codes lack is temporal coherence rather than validity. A
separate test uses real frames from the reference models, which score lower (0.954) because
they are quiet and the fixture is short; a real 210-frame utterance reaches 0.9956 and is
indistinguishable from the CPU decode by ear.

One test there is a regression guard worth knowing about. `ttnn.conv1d` prepares weights for
the parallelisation it picks, and that depends on input length, so caching a prepared weight
by name alone silently corrupts the next clip of a different length. Measured: 0.995 falling
to 0.104 on a second decode through the same instance. The fix keys the cache by length, and
the test decodes 4, 8 then 4 frames through one object.

`pcc/test_codec_encoder_pcc.py` measures two different things. The latents are floating
point and gated like any other block: every convolution stage, every transformer layer and
the final `downsample` hold 0.9997 or better on a 2 s clip. The codes are the output of a
nearest-neighbour search over 2048 entries, repeated 16 times down a residual chain, so they
either match or they do not, and they are scored per step with the reference's codes forced
into the chain, for the same reason the code predictor is:

| measurement | value |
|---|---|
| stages, conv stack through latents | 0.9997 to 0.9999998 |
| codes, per step with the prefix forced | 369/400 |
| codes, free running | 248/400 |
| round trip against the clip, reference vs device | 0.96135 vs 0.96144 |

Every one of the 31 disagreements is a near-tie: the device picked the reference's second or
third nearest entry, at most 0.6% further away. The round-trip row is the one that says
whether it matters. Re-decoding the device's codes lands as close to the original clip as
re-decoding the reference's, which puts the difference inside what the codec itself loses at
12.5 Hz.

**The encoder runs in fp32**, alone among these blocks. bf16 puts the latents at 0.9984 and
per-step agreement at 75%, because a rounding error at the end of a residual chain becomes a
different code rather than a slightly different number. It costs 1.5 s on a 3 s clip and the
block runs once per reference clip.

One test there is a regression guard. `downsample` is the only convolution in the codec that
replicates its padding rather than zeroing it, and `MimiModel` passes `pad_mode="replicate"`
for that one call alone. Zero padding cost 0.009 of PCC on exactly the tensor the codes come
from, and nothing else in the graph noticed.

`pcc/test_pipeline.py` covers the dual-track prefill and the decode loop, and is the one file
that needs the **CustomVoice** checkpoint. The two releases are complementary: Base carries
`speaker_encoder` and an empty `spk_id`, CustomVoice carries the nine speakers and no speaker
encoder. It resolves CustomVoice by repo id rather than from the ambient `$QWEN3_TTS_CKPT`,
so a first run downloads 4.3 GB.

The prefill is checked position by position against the tables it is built from. Its
bit-exactness against upstream's own assembled prefill was verified separately, by capturing
that prefill at the talker's door under transformers 4.57.3: max absolute difference 0.0 for
`ryan` tagged and untagged, and for the dialect speaker `dylan` under Chinese, Auto and
English. Two transformers versions cannot share an environment, so the suite pins the
composition instead.

Two prompt rules were wrong before that comparison and are now tested. `Auto` means no
language tag, and upstream marks the absence with `codec_nothink_id` rather than
`codec_think_id`. And two of the nine speakers are dialect speakers, `eric` (Sichuanese) and
`dylan` (Beijing): for them Chinese or `Auto` tags the dialect, so `Auto` stops meaning "no
tag" at all.

Free-running greedy decode diverges from a CPU greedy run: one near-tie flip changes the
input to every later step, so 13 of 14 steps match with a forced prefix while only a handful
of codes match when running free. The loop is therefore gated per step, not on sequence
equality.

## Performance work still open

The frame loop is 62% code predictor: 19.8 ms of a 31.9 ms frame, of which 16.2 is its
five layers run fifteen times. That is the number to move, and the two obvious ways of
moving it were measured and are not worth it:

- **Sampling on the device**, to keep the host out of the inner loop. `ttnn.sampling` costs
  0.298 ms a call, traced or eager, against the 0.19 ms of host work per step it would
  replace. Fifteen of them a frame turns a 2.9 ms saving into a 4.5 ms bill. It does work
  and it seeds reproducibly (`ttnn.manual_seed` with a seed tensor), so this is a cost
  verdict, not a capability one.
- **One trace over all 15 steps.** Dies with the above: each step's codebook id has to
  reach the host for the next step's lookup, so the graph cannot close over the chain.

What is left, in order of what it would buy:

- **Fewer ops per predictor step.** At 5.4 us of launch cost per op in a trace, its 21 ops
  a layer are half its time and its matmuls the other half. Keeping the residual stream
  sharded across a layer would remove the four conversions a layer spends around its two
  sharded norms. Attention's own layout dance is another four.
- **Attention's matmuls in `bfloat8_b`.** 9.2 ms against 9.6 for the talker's step, but the
  sampling distribution moved to 0.1455 against a 0.098 noise band, so it needs a reason
  better than PCC to take. A listening test would settle it.
- **Streaming the codec**, for the latency rather than the throughput: audio could start
  after the first frames instead of after the last. Needs carried state, as above.
- **Batch above 1**, and the second P300 chip, which sits idle at batch 1.

`tests/perf/test_perf.py` is a table, not a CI gate. A device-perf leg wants its own budget
file and a threshold per SKU; nothing in CI catches a speed regression yet.

## CI

Registered in the Tier 3 unit pipeline on WH N150 and BH P150
(`tests/pipeline_reorg/models_unit_tests.yaml`, model identifier `qwen3-tts-1.7b-base`). The
identifier drops the frame rate that the HF name carries; `HF_MODEL` keeps the canonical
`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, and the target resolver matches on that through its aliases.
Dispatch a single run from
[`all-model-tests`](https://github.com/tenstorrent/tt-metal/actions/workflows/all-model-tests.yaml)
with tier 3, type unit, and that identifier.

A second leg, `qwen3-tts-0.6b-base`, runs the same files on
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` on the same two SKUs. Timeouts are from one N150: the 1.7B
leg took 10.6 min warm and 14.8 with a cold kernel cache (its N150 timeout is 20, where the
original 10 was sized on Blackhole), the 0.6B leg 7.0 warm and 8 cold (timeout 12).

The end-to-end leg is deliberately absent. It lands with the first change that produces a
waveform, together with its own `e2e_tier3` budget; registering one before then would either
duplicate these tests or claim coverage that does not exist.

## Directory layout

| Path | Role |
|---|---|
| `weights.py` | checkpoint resolution and the speaker-encoder weight reader |
| `frontend.py` | host text path: tokenizer, prompt wrappers, language resolution |
| `sampling.py` | host sampler, matching `transformers`' processor order |
| `audio.py` | host audio path: file to 24 kHz waveform, waveform to log-mel |
| `demo/` | one-shot CLI and an interactive server |
| `tt/` | TTNN blocks |
| `reference/` | CPU references (PCC oracles); `reference/qwen/` is vendored upstream, Apache-2.0 |
| `tests/` | host tests, `tests/pcc/` for device correctness |

The vendored reference exists because the `qwen-tts` package pins transformers 4.57.3, which
conflicts with the version this repository runs. `reference/qwen/speaker_encoder.py` is a
byte-for-byte copy of the upstream encoder apart from two mechanical deviations recorded in
its header. Treat it as an oracle: any edit that is not a faithful copy makes it useless.
