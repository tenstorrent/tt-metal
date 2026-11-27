# CosyVoice2 on Tenstorrent (TTNN)

TTNN bring-up of [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B), FunAudioLLM's streaming
text-to-speech model, for [tenstorrent/tt-metal#54104](https://github.com/tenstorrent/tt-metal/issues/54104).

## Platforms

- Wormhole (`n150`). Every figure in this package comes from one N150.

## Status

The whole model runs on device, non-streaming: text → speech tokens (LLM) → mel (flow matching) → 24 kHz waveform
(HiFT vocoder). **The four Stage 1 targets are met.** Streaming runs too: chunks of audio are produced while the LLM
generates, on upstream's chunk schedule. That schedule, run over fixed tokens, matches upstream's own streaming run
(`docs/VALIDATION.md`). It misses both Stage 3 targets: the first audio arrives after 1.31–1.50 s, and the worst
streaming RTF is 1.10–1.12.

| target (#54104) | stage | measured | status |
|---|---|---|---|
| RTF < 1.0, non-streaming | 1 | worst **0.654**, aggregate 0.483, over six distinct utterances | met |
| token accuracy > 95 % vs the PyTorch reference | 1 | **95.94 %**, teacher-forced over 5,003 positions (27 sequences, 4 speakers) | met |
| WER < 5 % | 1 | **0.68 %** (1 error in 147 words) in each of five noise draws; the PyTorch reference also 0.68 % in each | met |
| speaker similarity > 0.60 (cosine) | 1 | **0.959** (0.958–0.959 over five draws); the PyTorch reference 0.952 | met |
| time to first packet < 500 ms; streaming RTF < 0.4 | 3 | first audio **1.31–1.50 s**; streaming RTF worst **1.10–1.12**, aggregate 0.84–0.85 | missed |

- Each verdict is recorded in [`tests/perf/gates.py`](tests/perf/gates.py). Tests enforce the non-streaming RTF, the
  token accuracy and both streaming figures; the missed streaming targets are each held inside a recorded band.
- A second N150 (2026-09-29) reproduced the 09-28 figures, with the same tokens and scores. The figures above are
  its re-run of 2026-09-30, after HiFT's padded calls were masked to match upstream's endings.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) has the per-utterance tables and how each figure was produced.
- [`PERF.md`](PERF.md) has the timing, including start-up.

How the figures were taken:
- **RTF:** measured after the start-up warm-up. Each utterance is a different sentence, synthesized once.
- **WER and similarity:**
  - the corpus is small: six LibriSpeech test-clean utterances from two speakers;
  - ASR is Whisper large-v3;
  - similarity is the `microsoft/wavlm-base-plus-sv` x-vector cosine;
  - both are scored by the same script for this port and for the PyTorch reference;
  - each is the mean over five draws of the vocoder's noise, with the tokens fixed, so no figure rests on one draw
    (`scripts/noise_draws.py`, the reference scripts' `--noise-seed`, `scripts/eval_draws.py`).

## What runs where

| step | where | notes |
|---|---|---|
| text normalization, splitting, tokenization | host | [`tt/text.py`](tt/text.py): upstream's English path, with parity tests |
| prompt features: speech tokens, speaker embedding, prompt mel | host, once per prompt | the upstream frontend (ONNX speech tokenizer, CAM++). It runs in the reference venv, and `scripts/prepare_inputs.py` writes one `.npz` per case |
| LLM: Qwen2-0.5B backbone, speech-token head | device | prefill, then a traced decode loop; sampling (RAS) on the host |
| flow: Conformer encoder + conditional flow matching (10 Euler steps) | device | bucketed lengths with padding masks |
| HiFT vocoder: F0 predictor, NSF source, upsampling stack, iSTFT | device, fp32 | a mel of 512 frames or more runs in 512-frame chunks with upstream's streaming cache; a shorter one, and streaming's final call, run at a bucket, masked past the real length so they compute upstream's call exactly ([`tt/hifigan/valid_length.py`](tt/hifigan/valid_length.py)) |

- Between stages, only the sampled token ids, the mel and the waveform return to the host.
- The configuration behind every reported figure is `CosyVoice2Config.reported()` in
  [`tt/pipeline.py`](tt/pipeline.py). It covers the fp32-logit LLM head, the decode trace, HiFT in fp32 and
  bucketing.
- The module docstrings give each choice's reasons.

## Two environments

| environment | runs | adds |
|---|---|---|
| tt-metal's `python_env` | the model, the demo, every test under `tests/` | [`requirements.txt`](requirements.txt): `inflect` |
| a host-only reference venv | upstream's frontend, upstream's PyTorch CosyVoice2, the WER/similarity scorer | [`requirements-reference-torch.txt`](requirements-reference-torch.txt), then [`requirements-reference.txt`](requirements-reference.txt) |

- The device side never imports `onnxruntime`, `whisper` or upstream's `cosyvoice` package. The two sides exchange
  files only.
- [`docs/security.md`](docs/security.md) documents:
  - the boundary between the two environments;
  - the pins, and why the reference venv runs transformers 5.12.1 with two exact shims;
  - the two advisories open against the reference venv's pins.

## Running

The commands below use these variables:

```bash
export COSYVOICE2_REPO=/path/to/CosyVoice               # upstream checkout, commit 074ca6dc9e80
export COSYVOICE2_REF_ENV=/path/to/cosyvoice2_ref_env   # the reference venv
export LIBRISPEECH_ROOT=/path/to/data                   # holds LibriSpeech/test-clean
export COSYVOICE2_INPUTS=/path/to/cosyvoice2_inputs     # prepare_inputs.py's output
REF=$COSYVOICE2_REF_ENV/bin/python
S=models/experimental/cosyvoice2/scripts
```

**1. The reference side (host only, once).**

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $COSYVOICE2_REPO
git -C $COSYVOICE2_REPO checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git -C $COSYVOICE2_REPO submodule update --init --recursive

uv venv --python 3.10 $COSYVOICE2_REF_ENV
VIRTUAL_ENV=$COSYVOICE2_REF_ENV uv pip install -r models/experimental/cosyvoice2/requirements-reference-torch.txt
VIRTUAL_ENV=$COSYVOICE2_REF_ENV uv pip install -r models/experimental/cosyvoice2/requirements-reference.txt \
    -c models/experimental/cosyvoice2/requirements-reference-lock.txt

# the checkpoint, at the revision every figure here was measured with
$REF -c "from huggingface_hub import snapshot_download as s; s('FunAudioLLM/CosyVoice2-0.5B', revision='eec1ae6c79877dbd9379285cf8789c9e0879293d')"

# LibriSpeech test-clean (md5 32fa31d27d2e1cad72775fee3f4849a9)
curl -L https://www.openslr.org/resources/12/test-clean.tar.gz | tar xz -C $LIBRISPEECH_ROOT
```

**2. Prompt inputs (reference venv).** The fixed corpus is [`scripts/corpus.py`](scripts/corpus.py). The primary set
is six LibriSpeech targets plus a CosyVoice1-parity case. The extension adds 20 teacher-forced sequences for token
accuracy.

```bash
$REF $S/prepare_inputs.py --parity --out-dir $COSYVOICE2_INPUTS
$REF $S/prepare_inputs.py --extension --out-dir $COSYVOICE2_INPUTS
```

**3. The demo (device).** It warms every bucket, then synthesizes the six targets. It writes wavs, `results.json` and
a timing table. `--stream` streams them instead, after warming the streaming set too.

```bash
pip install -r models/experimental/cosyvoice2/requirements.txt   # once, into python_env
python models/experimental/cosyvoice2/demo/demo.py --inputs $COSYVOICE2_INPUTS --out <run dir>
```

**4. The PyTorch reference and scoring (reference venv).**

```bash
$REF $S/run_reference.py --parity --out-dir <reference dir>
$REF $S/eval_wer_sim.py --run-dir <reference dir>
$REF $S/eval_wer_sim.py --run-dir <run dir> --baseline <reference dir>
```

WER and similarity over five noise draws, as `docs/VALIDATION.md` reports them. The draws keep the tokens fixed and
vary only the vocoder's noise:

```bash
python models/experimental/cosyvoice2/scripts/noise_draws.py --inputs $COSYVOICE2_INPUTS --out <draws dir>  # device
for n in 1 2 3 4 5; do
  $REF $S/run_reference.py --noise-seed $n --out-dir <draws dir>/ref_stage1_seed$n
  $REF $S/streaming_reference.py --noise-seed $n --inputs $COSYVOICE2_INPUTS --tokens-from <Stage 1 run dir> \
      --out-dir <draws dir>/ref_stream_seed$n
done
$REF $S/eval_draws.py --out draws.json --group "TT Stage 1" <draws dir>/tt_stage1_seed{1..5} \
    --group "reference Stage 1" <draws dir>/ref_stage1_seed{1..5}   # ... and the two streaming groups
```

## Tests

```bash
# host tier: no device, 134 tests
pytest models/experimental/cosyvoice2/tests -k "not test_device"

# the whole suite: the host tier and 98 device tests (the two perf tests deselected). Some need reference-side
# files, and skip without them.
COSYVOICE2_INPUTS=$COSYVOICE2_INPUTS \
COSYVOICE2_TOKEN_REF=<token_accuracy_reference.py out dir> \
COSYVOICE2_HIFT_STREAM_REF=<hift_streaming_reference.py out dir> \
COSYVOICE2_STREAM_REF=<streaming_reference.py out dir> \
pytest models/experimental/cosyvoice2/tests --deselect models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py

# the Stage 1 RTF gate, then Stage 3's two figures, each in its own process
COSYVOICE2_INPUTS=$COSYVOICE2_INPUTS pytest "models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py::test_device_nonstreaming_rtf_distinct_utterances"
COSYVOICE2_INPUTS=$COSYVOICE2_INPUTS pytest "models/experimental/cosyvoice2/tests/perf/test_pipeline_perf.py::test_device_streaming_first_audio_and_rtf_distinct_utterances"
```

- **The token-accuracy test's references** come from the reference venv:
  1. `run_reference.py --parity` and `run_reference.py --extension`, one output directory each;
  2. `token_accuracy_reference.py --inputs $COSYVOICE2_INPUTS --run-dir <each of the two> --out-dir <one token dir>`.
- **The chunked-HiFT seam gate's reference** is `scripts/hift_streaming_reference.py --out-dir <dir>`.
- **The streaming gates' reference** is upstream streaming the demo's tokens:
  `scripts/streaming_reference.py --inputs $COSYVOICE2_INPUTS --tokens-from <Stage 1 run dir> --out-dir <dir>`.
- `tests/reference/` runs in the reference venv as plain scripts and skips under `python_env`:
  - `test_reference_env.py` checks the reference venv's two transformers shims;
  - `test_you_clip.py` transcribes the "you" clip's 11 noise draws, which the device suite writes when
    `COSYVOICE2_YOU_OUT` is set, and fails on a trailing "you" (`docs/VALIDATION.md`, "Masked end padding").
- An allocation-tracker test for the CFM traces is opt-in: it must run alone, with
  `COSYVOICE2_RUN_TRACE_ALLOC_TRACKER=1`.
- The suite took 73 minutes on an N150 from an empty kernel cache (2026-09-29). On 2026-09-30, with most kernels
  on disk, it took 27 minutes and compiled 1,103.

## Known issues

- **[#36487](https://github.com/tenstorrent/tt-metal/issues/36487): `ttnn.prepare_conv_weights` gives wrong weights
  when `conv1d` slices its input through DRAM.**
  - Some HiFT and flow convolutions hit it at particular lengths.
  - Every new conv geometry is checked once against a raw-weight, safe-config reference, and a float64 host conv
    arbitrates any disagreement. A wrong fast path never reaches the output
    ([`tt/hifigan/conv.py`](tt/hifigan/conv.py)).
  - Where the weight prepared for the activation's TILE layout is wrong, one prepared declaring a ROW_MAJOR input
    is usually right. The checks then keep that one, so the conv stays on a prepared (traceable) weight
    (`docs/VALIDATION.md`, "Prepared conv weights").
  - The checks rerun in every process: 35 s of the warm start (23.5 s for the buckets, 11.5 s for the streaming
    set).
- **Cached kernels are reused only when a process allocates identically.**
  - With conv config tensors in DRAM, the conv and halo reader kernels take those tensors' DRAM addresses as
    compile-time arguments.
  - So start-up runs a fixed warm-up sequence. It takes 3.1 minutes when the kernels are on disk and 31.4 minutes on
    an empty cache, and the streaming set adds 2.5 and 13.0 minutes ([`PERF.md`](PERF.md)).
  - Any change to the code, the configuration or the checkpoint costs one cold start.
- **Streaming needs its own warm-up.** A chunk runs between decode steps while the LLM's decode trace is alive, so
  `warmup_streaming()` compiles and verifies every streaming geometry first (2.5 minutes with the kernels on disk).
  `synthesize_stream` refuses to run without it. On a pipeline without it, the first chunk allocated 1,259 buffers
  under the live trace, where its next replay could overwrite them (`docs/VALIDATION.md`).
- **Blackhole is untested.**
  `tests/pcc/test_flow_decoder.py::test_device_decoder_fused_sdpa_ignores_tile_padding_at_t_1_mod_32` guards the
  fused-SDPA tile-padding bug reported for Blackhole ([#57608](https://github.com/tenstorrent/tt-metal/issues/57608)).
  It passes on Wormhole, where the bug doesn't reproduce.

## Architecture

Three stages, as in CosyVoice1, but not identical to it:

1. **LLM:** a Qwen2-0.5B backbone predicts speech tokens (6,561 codes, 25 per second) from the text, conditioned on
   the prompt's text and speech tokens.
2. **Flow:** `CausalMaskedDiffWithXvec`. An `UpsampleConformerEncoder` feeds a `CausalConditionalCFM` that
   integrates 10 Euler steps into an 80-bin mel at 50 frames per second. Both are chunk-aware (causal, 3 look-ahead
   tokens); that is where streaming lives.
3. **HiFT:** `HiFTGenerator`, an NSF source (sine harmonics from a predicted F0) plus a 3-stage upsampling stack
   (8 × 5 × 3), ending in an inverse STFT (`n_fft` 16, hop 4) at 24 kHz.

### Why the vocoder's iSTFT is not an FFT problem

TTNN has no FFT. It doesn't need one here, because `n_fft = 16`.
- The inverse DFT of 9 one-sided bins is a fixed 16 × 9 real matrix pair: a matmul.
- Windowing and overlap-add are one transposed convolution with a diagonal kernel: `out[t*hop+j] += frame[j,t]*w[j]`
  is `conv_transpose1d`.
- The window-sum normalization depends only on the frame count, so it is a precomputed multiply.

[`tt/hifigan/istft.py`](tt/hifigan/istft.py) has the derivation, and [`tt/hifigan/stft.py`](tt/hifigan/stft.py) the
forward twin for the NSF source. The identity is the one the CosyVoice1 port
([#52540](https://github.com/tenstorrent/tt-metal/pull/52540)) validated.

## Layout

| path | what |
|---|---|
| `tt/pipeline.py` | the whole model, wired: `CosyVoice2TTNN.synthesize` and `synthesize_stream`, the configuration, bucketing, the warm-ups |
| `tt/streaming.py` | streaming: upstream's chunk schedule, `StreamSession`, HiFT over a stream with upstream's cache |
| `tt/text.py`, `tt/prompt.py` | the text frontend; what a call is conditioned on and what it draws at random |
| `tt/llm/` | the Qwen2 backbone (adapted from tt_transformers) and RAS sampling |
| `tt/flow/` | the Conformer encoder, the CFM estimator and solver, and the module that ties them together |
| `tt/hifigan/` | the vocoder: convs with per-geometry checks, resblocks, NSF source, F0 predictor, STFT/iSTFT, chunking |
| `tt/checkpoint.py`, `tt/geometry_cache.py` | checkpoint loading; the per-geometry weight caches |
| `demo/demo.py` | the Stage 1 demo; `--stream` for streaming |
| `scripts/` | reference-venv scripts: the corpus, inputs, the PyTorch reference and upstream's streaming, the scorer and its noise-draw report, the tests' references; `noise_draws.py` runs on the device |
| `tests/` | `e2e/` (pipeline, text, prompt, token accuracy, streaming), `pcc/` (per module, against torch or upstream), `perf/` (the gates), `reference/` (reference-venv checks) |
