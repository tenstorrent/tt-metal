# Qwen3-TTS

Text-to-speech model ([Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)) on Tenstorrent Blackhole hardware.

The pipeline consists of two components running fully on-device with Metal trace:

| Component | Description |
|---|---|
| **Talker** | 28-layer autoregressive transformer that generates Mimi codec codes from text + reference audio |
| **CodePredictor (CP)** | 5-layer transformer that predicts residual codec codes in parallel with the Talker |

Audio is encoded and decoded by [Kyutai Mimi](https://huggingface.co/kyutai/mimi) (reference PyTorch, on CPU).

## Hardware

- **Board:** Blackhole P150 (single chip)

## Performance

Measured on Blackhole P150 with Metal trace + KV cache + 2CQ:

| Metric | Value |
|---|---|
| Prefill latency | < 22 ms |
| Steady-state AR step | ~43.3 ms/frame |
| Audio sample rate | 24 kHz |
| Codec frame rate | 12.5 Hz |

## Prerequisites

- Cloned [tt-metal](https://github.com/tenstorrent/tt-metal) repository.
- TT-Metalium / TT-NN installed: see [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).
- HuggingFace cache populated (model is auto-downloaded on first run via `snapshot_download`).

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

# End-to-end demo (CI smoke test)
pytest models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py::test_demo -s -v
```

## Architecture Notes

- **Metal trace:** Talker prefill, Talker decode, and CP decode are all captured as Metal traces after compilation. Inference replays traces with no host dispatch overhead.
- **2CQ mode:** Host-to-device tensor copies run on CQ1 while trace replay runs on CQ0, overlapping H2D with compute.
- **KV cache:** Statically allocated; decode steps write in-place without reallocation.
- **Prefill bucketing:** Input sequences are padded to the nearest bucket in `[32, 64, 128]` tokens; a separate trace is captured per bucket.
