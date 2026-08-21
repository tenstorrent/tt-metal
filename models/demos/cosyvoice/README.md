# CosyVoice-300M on Tenstorrent (TTNN)

TTNN bring-up of [CosyVoice-300M](https://github.com/FunAudioLLM/CosyVoice), Alibaba
FunAudioLLM's multilingual TTS model, for
[tenstorrent/tt-metal#32178](https://github.com/tenstorrent/tt-metal/issues/32178).

CosyVoice generates speech in three stages: an **LLM** predicts supervised semantic
tokens from text, a **flow-matching decoder** turns those tokens into a mel
spectrogram, and a **HiFTNet vocoder** turns the mel into a waveform. All three run
through TTNN here — including the vocoder, which is the part that is normally left
on the host.

| | |
|---|---|
| Parameters | ~300 M (llm 1.24 GB + flow 0.42 GB + hift 0.08 GB, fp32) |
| Sample rate | 22 050 Hz |
| Languages | Chinese, English, Japanese, Cantonese, Korean |
| Modes | SFT, zero-shot, cross-lingual, instruct |
| Status | **All three stages on device.** flow tokens->mel PCC 0.99920 · tokens->waveform 0.99514 · vocoder 0.99964 · LLM prefill 0.99974 |

---

## Why the vocoder is the interesting part

TTNN has **no FFT of any kind**. A case-insensitive search for `fft|rfft|irfft|stft|istft`
across `ttnn/` and `tt_metal/` returns nothing. HiFTNet ends in an inverse STFT, so at
first glance the vocoder cannot run on device at all — and the previous attempt at this
bring-up left it on the host, which is why it was rejected.

It does not actually block anything, because CosyVoice uses **`n_fft = 16`**. At that
size the inverse DFT of 9 one-sided bins is a fixed 16×9 real matrix pair — smaller than
a single 32×32 tile — so it is a **matmul**, and a matmul maps onto the FPU, the widest
unit on a Tensix core. Windowing and overlap-add then fuse into a single
**transposed convolution** with a diagonal kernel, because OLA
(`out[t·h + j] += frames[j,t]·w[j]`) and `conv_transpose1d`
(`out[o, t·s + k] += in[i,t]·W[i,o,k]`) are the same operation. The NOLA normalisation
depends only on the frame count, so it is a precomputed constant and one multiply.

Net: `matmul + conv_transpose2d + multiply`. All of which TTNN already has.

Verified against the real vocoder's captured magnitude/phase — spanning
1.06e-13 to 1.21e+01, fourteen decades:

```
PCC fp32      1.0000000000   max|Δ| 2.98e-07
PCC bf16-in   0.9999765688   max|Δ| 5.25e-03
```

See `tt/hifigan/istft.py` for the derivation and `tests/pcc/test_istft.py` for the checks.

---

## Setup

Two environments, deliberately separate. They do not share packages and should not.

| environment | what it is for |
|---|---|
| tt-metal `python_env` | running the model on device, PCC/perf tests |
| `cosyvoice_env` | the PyTorch reference: goldens, baseline audio, WER/SIM scoring |

The reference pins its own `torch` and `transformers` — see `requirements-reference.txt`,
which records why each pin is where it is. Forcing those into the tt-metal environment
would risk the tt-metal build for no benefit, since the reference never runs on device.

### 1. Reference environment (host only, no device)

```bash
cd $TT_METAL_HOME/models/demos/cosyvoice

uv venv --python 3.10 /mnt/cosyvoice_env
VIRTUAL_ENV=/mnt/cosyvoice_env uv pip install -r requirements-reference.txt

git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /mnt/CosyVoice
git -C /mnt/CosyVoice checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git -C /mnt/CosyVoice submodule update --init --recursive

/mnt/cosyvoice_env/bin/python scripts/download_model.py --skip-onnx-trt
```

`--skip-onnx-trt` omits `flow.decoder.estimator.fp32.onnx` (329 MB × 2), which only the
TensorRT export path reads. The three checkpoints then total ~6.9 GB.

> **First inference downloads more.** The text frontend fetches ~31 MB of `wetext` FSTs
> from ModelScope the first time it normalises text — not at import. A network-isolated
> run will fail there unless `/root/.cache/modelscope` is already populated.

### 2. Goldens

```bash
export PYTHONPATH=/mnt/CosyVoice:/mnt/CosyVoice/third_party/Matcha-TTS
/mnt/cosyvoice_env/bin/python scripts/gen_golden.py --mode zero_shot
```

Writes `tests/golden/*.npz` — one per module boundary, plus `manifest.json`.

---

## Running

### PCC tests

```bash
# host tier: no device, no silicon, ~40 s. 111 tests.
pytest models/demos/cosyvoice/tests/ -k "not device"

# device tier: needs /dev/tenstorrent. 41 tests.
pytest models/demos/cosyvoice/tests/pcc/ models/demos/cosyvoice/tests/e2e/ -v

# performance -- see PERF.md for what the numbers mean
pytest models/demos/cosyvoice/tests/perf/ -v -s
```

### Demo

```bash
export PYTHONPATH=$TT_METAL_HOME
python models/demos/cosyvoice/demo/demo.py --out out.wav
```

Runs all three stages on device and writes a 22.05 kHz wav. With no arguments it
synthesises the captured golden utterance, which needs no front-end.

The front-end — text normalisation, the Whisper-family tokenizer, and the two **ONNX**
encoders (`speech_tokenizer_v1.onnx`, `campplus.onnx`) — is not ported: three of those
four are ONNX blobs and none is on the bounty's critical path. It runs once in the
CosyVoice venv via `scripts/prepare_inputs.py`, which writes a flat `.npz` the device
side loads without importing cosyvoice or onnxruntime — the same boundary
`export_weights.py` draws.

### Weight export

```bash
export PYTHONPATH=/mnt/CosyVoice:/mnt/CosyVoice/third_party/Matcha-TTS
for m in hift flow llm; do
  /mnt/cosyvoice_env/bin/python scripts/export_weights.py --module $m --fp16
done
```

Flattens each submodule's `state_dict` into `tests/golden/<module>_weights.npz` with a
JSON `__meta__` blob carrying the architectural constants that cannot be read off a
tensor shape. `weight_norm` is folded at export (verified bit-exact) so the device never
recomputes a constant normalisation.

### Reference baseline and scoring

```bash
export PYTHONPATH=/mnt/CosyVoice:/mnt/CosyVoice/third_party/Matcha-TTS
/mnt/cosyvoice_env/bin/python scripts/run_reference.py --out /tmp/ref
/mnt/cosyvoice_env/bin/python scripts/eval_wer_sim.py --run-dir /tmp/ref
```

> **Never run scoring and synthesis at the same time.** Whisper large-v3 is ~9 GB
> resident on CPU and synthesis is ~4.6 GB; together they OOM an 11 GB host. The harness
> preflights available memory and refuses rather than getting killed mid-run. Pass
> `--asr-model medium` on a small box.

---

## Validation strategy

Designed first rather than last, because the previous attempt at this bounty had working
code and was still rejected on validation.

**Token accuracy is exact agreement, not top-k overlap.** Two gates: teacher-forced argmax
match per position, and free-running greedy (`top_k=1`) full-sequence comparison. RAS
sampling (`top_p 0.8`, `top_k 25`) is stochastic and is reported for audio quality only —
it is never the accuracy gate.

**Per-module PCC ≥ 0.99** against goldens captured from the reference.

**Streaming is compared on content**, not chunk count: concatenated streamed audio versus
non-streamed audio for the same text and seed.

**Perf targets are ordinary passing tests.** If a target is missed the number is reported
and the gap explained; `xfail` reads as concealment. **Every measured figure lives in
[`PERF.md`](PERF.md) and nowhere else** — this file deliberately quotes none of them, so
there is no second copy to drift. That includes the end-to-end RTF, the per-stage
breakdown, the Blackhole/Wormhole comparison, and which targets are met on which part.

**Two things cannot be gated on exact agreement, and both say so explicitly.** RAS
sampling is a multinomial draw, so the LLM is gated on its *logits* and the audio chain
on the reference's *captured tokens*. And NSF excitation phase is chaotically sensitive
to f0 — a 0.03 Hz error accumulates a tenth of a cycle over an utterance, finer than
Tensix arithmetic delivers — so waveform *samples* are gated with the reference
excitation injected (PCC 0.9951) and the *envelope* without it (0.9975).

---

## What runs where

| stage | module | on device |
|---|---|---|
| text -> semantic tokens | `tt/llm/` — 6-block causal Conformer text encoder, 14-block AR decoder with a fixed-width KV cache, 4097-way head | yes |
| semantic tokens -> mel | `tt/flow/` — Conformer encoder, length regulator, 10-step CFM solver over a 16-resnet/64-transformer UNet | yes |
| mel -> waveform | `tt/hifigan/` — f0 predictor, NSF excitation, 40-odd convolutions, inverse STFT | yes |
| RAS sampling | `tt/llm/sampling.py` | host (Stage 1) |
| front-end (tokenizer, 2 ONNX encoders) | `scripts/prepare_inputs.py` | host, by design |

`tt/flow/reference.py` and `tt/llm/reference.py` reimplement their stages in plain torch
from the flat weight export alone — no cosyvoice, no diffusers, no device. Both reproduce
the captured goldens to PCC 0.9999999, which is what lets a device bring-up start from
"the graph is right" instead of bisecting eighty blocks on rented silicon.

### The trap: three modules draw from the RNG mid-forward

`ConditionalCFM.forward` draws `z = randn_like(mu)`; `SineGen.forward` draws a per-harmonic
phase offset from `U(−π, π)` plus Gaussian noise; `SourceModuleHnNSF.forward` draws noise
again.

Seeding does **not** make TTNN and PyTorch comparable — it only makes PyTorch reproducible
against itself, because TTNN cannot consume the torch RNG stream in the same order. So
`gen_golden.py` **captures every draw as a named array**, and the TTNN modules take them as
explicit inputs during PCC tests. Get this wrong and a perfectly correct vocoder port scores
PCC ≈ 0.3.

---

## Layout

```
models/demos/cosyvoice/
├── README.md                    this file
├── requirements-reference.txt   reference-environment pins (CPU)
├── scripts/
│   ├── download_model.py        stdlib-only, resumable checkpoint fetch
│   ├── gen_golden.py            capture per-module goldens from the reference
│   ├── run_reference.py         4 modes x 5 languages, with tok/s + RTF
│   └── eval_wer_sim.py          WER/CER + speaker similarity (R9)
├── tt/
│   ├── model_config.py          every dtype, memcfg and shape constant
│   ├── common.py                golden loading, PCC, weight_norm folding
│   ├── llm/                     text encoder, AR decoder, rel-pos attention, RAS
│   ├── flow/                    conformer encoder, length regulator, CFM, estimator
│   └── hifigan/
│       └── istft.py             the iSTFT identity  <- the enabling result
└── tests/
    ├── golden/                  captured .npz + manifest.json
    ├── pcc/                     per-module PCC >= 0.99
    ├── e2e/                     4 modes x 5 languages, exact-token, WER, SIM
    └── perf/                    tok/s and RTF gates (passing, never xfail)
```

---

## Known constraints

- **`weight_norm` must be folded at load time.** Every conv in HiFT is wrapped in it;
  folding `w = g·v/‖v‖` on host removes a per-inference normalisation for free.
- **The CFM solver's classifier-free guidance is already batched upstream** into one
  2-row call. Preserve it; unrolling it into two calls doubles the cost of the hottest
  module in the model.
- **`att_cache` is `[n_layers, n_heads, T, 2·head_dim]`** with K and V concatenated on
  the last axis. `ttnn.experimental.paged_cache` does not take this shape directly.
- **`SineGen` integrates phase with `cumsum` over the audio-rate signal.** In bfloat16 an
  accumulator reaching ~1e3 loses the ~1e-2 increments entirely; use
  `ttnn.cumsum(dtype=ttnn.float32)`.
- **RAS's retry path is a plain multinomial over the full 4097-token vocabulary**, not a
  re-draw from the truncated distribution. `ttnn.sampling` covers the primary path only.
