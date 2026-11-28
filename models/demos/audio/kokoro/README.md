# Kokoro-82M

## Platforms:
    Blackhole (p150)

## Introduction

[hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is a non-autoregressive
StyleTTS2 / ISTFTNet text-to-speech model. Its only attention transformer is a
weight-tied ALBERT encoder (`plbert`, 12 layers, hidden 768); the prosody predictor,
text encoder, and ISTFTNet vocoder are convolutional / recurrent.

This bring-up runs the **entire pipeline on Tenstorrent** (TTNN) — plbert, the prosody
predictor, the text encoder, and the ISTFTNet decoder + iSTFT vocoder — via
`KokoroDevicePipeline.synthesize_device()` in `tt/device_pipeline.py`. The only
host-side steps left are indexing, not compute: the duration→alignment scatter and the
embedding lookup. On p150 it matches the torch reference at STFT log-magnitude PCC ≈ 0.98
(waveform-domain PCC is not a valid metric here — the on-device harmonic source uses a
deterministic phase model, so the audio is spectrally equivalent but phase-decorrelated).

A simpler hybrid entrypoint, `synthesize()`, keeps the prosody/vocoder stages on the host
(torch) and runs only the plbert encoder on device — matching the SpeechT5 TT-transformer +
CPU-vocoder pattern used by the tt-inference-server media-server runner. It backs the demo.

**Status: EXPERIMENTAL.**

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Model-specific host dependencies: `pip install -r models/demos/audio/kokoro/requirements.txt`
- System package for grapheme-to-phoneme: `apt-get install espeak-ng`

> Note: `misaki`/`kokoro` pull in `spacy` transitively, which wants a numpy-2 ABI that
> conflicts with the numpy 1.26 TTNN is built against. The G2P path uses
> `misaki.espeak.EspeakFallback` (libespeak-ng) to avoid loading spaCy. See the demo.

## How to Run

### Text-to-speech demo (single-chip p150)

Synthesize speech from text and write a WAV file:

```sh
pytest --disable-warnings models/demos/audio/kokoro/demo/demo.py::test_demo
```

### plbert encoder correctness (PCC vs HuggingFace)

```sh
pytest --disable-warnings models/demos/audio/kokoro/tests/test_optimized_decoder.py
```

### Functional encoder test

```sh
pytest --disable-warnings models/demos/audio/kokoro/tests/test_functional_decoder.py
```

## Model precision

The default policy is **BFP8 weights / bf16 activations / HiFi2**, baked into
`OptimizedDecoder`'s `PrecisionPolicy` — chosen from a 10-config datatype sweep as the
fastest configuration that clears the accuracy gate.

## Performance

The model is launch/dispatch-bound (small ops, ~5% DRAM utilization), so throughput is
dominated by op-to-op dispatch rather than compute.

Measured on P150 (single Blackhole chip):

| Path | Metric | Value |
|---|---|---|
| plbert encoder | last_hidden_state PCC vs HF (worst, T≤512) | ≈ 0.997 |
| plbert encoder | eager prefill latency (T=128 / T=512) | ≈ 2.5 ms / 3.0 ms |
| **Fully on-device** TTS (`synthesize_device`) | latency (≈2.4 s clip) | ≈ 0.88 s |
| **Fully on-device** TTS (`synthesize_device`) | real-time factor | ≈ 2.7× |
| Fully on-device audio | STFT log-magnitude PCC vs torch | ≈ 0.98 |

Reproduce the perf numbers:

```sh
pytest -m models_performance_bare_metal models/demos/audio/kokoro/tests/test_perf_optimized.py         # encoder
pytest -m models_performance_bare_metal models/demos/audio/kokoro/tests/test_perf_device_pipeline.py   # full TTS
```

## Details

- `tt/device_pipeline.py` — the full text→audio pipeline on device (`synthesize_device`), plus the hybrid `synthesize` (device plbert + host vocoder).
- `tt/optimized_decoder.py` — single-chip plbert encoder (packed QKV, FlashAttention SDPA, BFP8/HiFi2). Used by the demo and the device pipeline.
- `tt/functional_decoder.py` — reference functional plbert encoder.
