# Qwen3-TTS

Text-to-speech model ([Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)) on Tenstorrent Blackhole and Wormhole hardware. License: Apache-2.0 (model weights, per the Hugging Face model card).

The pipeline consists of two components running fully on-device with Metal trace:

| Component | Description |
|---|---|
| **Talker** | 28-layer autoregressive transformer that generates Mimi codec codes from text + reference audio |
| **CodePredictor (CP)** | 5-layer transformer that predicts residual codec codes in parallel with the Talker |

Audio is encoded by the Qwen3-TTS 12 Hz speech tokenizer (reference PyTorch, on CPU). The speech **decoder** runs on
device for serving (`tt/speech_tokenizer.py`, `TtSpeechTokenizerDecoder`): the exact CPU front-end (codebooks and
pre-transformer) runs from a cached per-voice reference state, and the conv back-end runs on device over 12 frames of
context plus the generated frames (`decode_icl_audio(..., device_decoder=...)`). The demo keeps the CPU decoder.

**Device decoder rule:** build and warm it (`prepare_device_decoder`) **before** `init_server_context` captures any
trace. Device buffers allocated while a trace exists can overlap the trace's freed intermediates and are overwritten
every time the trace replays (silently wrong audio, not an error).

## Hardware

- **Board:** Blackhole P150 (single chip)
- **Board:** Wormhole N150 (single chip; also run as one chip of a Wormhole Galaxy in the n150 configuration)

## Performance

Measured on Blackhole P150 with Metal trace + KV cache + 2CQ:

| Metric | Value |
|---|---|
| Prefill latency | < 22 ms |
| Steady-state AR step | ~43.3 ms/frame |
| Audio sample rate | 24 kHz |
| Codec frame rate | 12.5 Hz |

Wormhole N150 configuration (one Galaxy chip, 2026-10-06, branch `nkira/qwen3-tts-rtr-exp`, 1.7B-Base):

| Metric | Value | How |
|---|---|---|
| Steady-state AR step | 49.3-50.3 ms/frame | demo, 1 and 2 command queues |
| Prefill latency | 15.0-15.7 ms | demo |
| Server TTFT / RTR | 0.99-1.04 s / 1.085-1.129 | tt-inference-server benchmark, 3 passes |
| Device decoder vs CPU decoder | 27.7-29.8 dB SNR | 22 server requests |

`TT_MM_THROTTLE_PERF=5` costs about 10 % (55.2 ms/frame); leave it unset.

## Precision

Talker and CodePredictor: as in PR #56212 (bf16 / bfp8 weights, swept program configs). Speech decoder back-end:
HiFi4 with fp32 accumulation, exact GELU, SnakeBeta parameters folded on the host in fp32 (+0.5 dB over the default
fidelity; HiFi3 measured equal). The native `ttnn.conv1d` decoder path is more accurate (about 33 dB vs 29 dB against the CPU decoder) but hangs the chip
once the talker's traces have executed, so it is blocked unless `TT_QWEN3_ALLOW_UNSAFE_CONV1D=1`.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `QWEN3_TTS_REVISION` | pinned per model (`tt/server.py` `HF_REVISIONS`) | Hugging Face revision to download |
| `QWEN3_TTS_MAX_PREFILL_BUCKET` | 512 | largest Talker prefill (tokens); longer prompts raise. 1024 overflows L1 with the current prefill configs |
| `TT_QWEN3_DECODE_MIN_BUCKET` | 32 | smallest speech-decoder bucket (frames) |
| `QWEN3_TTS_CP_FUSED` | 0 | 1 = fuse the whole CodePredictor frame into one trace (demo path only; slower on WH N150: 55.6 vs 49.3 ms/frame) |
| `TT_QWEN3_DECODE_NUMERICS` | `hifi` | `legacy` restores the default-fidelity decoder numerics |
| `TT_QWEN3_DECODE_FIDELITY` | `HiFi4` | decoder matmul fidelity |
| `TT_QWEN3_DECODE_CONV` | `matmul` | `conv1d` is blocked (see Precision) |

## Prerequisites

- Cloned [tt-metal](https://github.com/tenstorrent/tt-metal) repository.
- TT-Metalium / TT-NN installed: see [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).
- HuggingFace cache populated (model is auto-downloaded on first run via `snapshot_download`, revision pinned;
  about 4.3 GB for 1.7B-Base).

## Quick Start

```bash
python models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py \
    --text "Hello, how are you today?" \
    --ref-audio models/demos/qwen3_tts/demo/jim_reference.wav \
    --ref-text "Jason, can we take a look at the review slides" \
    --output /tmp/tts_output.wav \
    --seed 42
```

The generated audio is written to `--output` as a 24 kHz WAV file.

A reference audio file and transcript are included at `demo/jim_reference.wav` and `demo/jim_reference.txt`. If `--ref-audio` is omitted, these defaults are used automatically.

## Tests

```bash
# Numerical accuracy (PCC vs PyTorch reference)
pytest models/demos/qwen3_tts/tests/test_qwen3_tts_pcc.py -s -v

# End-to-end performance gate (prefill + steady-state timing)
pytest models/demos/qwen3_tts/tests/test_qwen3_tts_perf_device.py -s -v

# Device decoder sharing the chip with the traced talker (serving path; includes a negative control)
pytest models/demos/qwen3_tts/tests/test_device_decoder_coexistence.py -s -v

# End-to-end demo (CI smoke test)
pytest models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py::test_demo -s -v
```

## Architecture Notes

- **Metal trace:** Talker prefill, Talker decode, and CP decode are all captured as Metal traces after compilation. Inference replays traces with no host dispatch overhead.
- **2CQ mode:** Host-to-device tensor copies run on CQ1 while trace replay runs on CQ0, overlapping H2D with compute.
- **KV cache:** Statically allocated; decode steps write in-place without reallocation.
- **Prefill bucketing:** Input sequences are padded to the nearest bucket in `[32, 64, 128]` tokens; a separate trace is captured per bucket.
