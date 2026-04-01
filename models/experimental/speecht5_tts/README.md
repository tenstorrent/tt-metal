# SpeechT5 TTS (Text-to-Speech) on Tenstorrent Hardware

## Platforms

| Device | Status | Notes |
|--------|--------|-------|
| N150 (Wormhole) | ✅ Supported | Single chip, fully tested |
| N300 (Wormhole) | ⚠️ Single-chip only | Runs on one chip only; tensor parallelism across both chips not yet implemented |

---

## Introduction

SpeechT5 is a unified-modal encoder-decoder framework for text-to-speech synthesis, introduced in the paper [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205). The model generates 16 kHz mono audio from arbitrary text input using speaker embeddings to control voice characteristics.

This implementation runs SpeechT5 fully on Tenstorrent hardware using TTNN, with a hybrid FP32/BF16 precision strategy for optimal quality and performance, and trace capture for maximum throughput.

---

## Model Architecture

The SpeechT5 TTS pipeline processes text through four sequential stages:

```
Input Text
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Text Encoder  (12 layers, 768 hidden, 12 heads) │
│  • T5-relative position bias                     │
│  • Pads input to fixed shapes: 64 / 128 / 256    │
└────────────────────┬────────────────────────────┘
                     │  encoder_hidden_states
                     ▼
┌─────────────────────────────────────────────────┐
│  Speech Decoder  (6 layers, 768 hidden, 12 heads)│
│  • Autoregressive: generates one mel frame/step  │
│  • Prenet (2-layer MLP, 256 units)               │
│  • Self-attention (KV cached)                    │
│  • Cross-attention (cached after step 0)         │
│  • FFN (3072 dim)                                │
│  • T5-relative position bias                     │
│  • Traced from step 1 onward                     │
└────────────────────┬────────────────────────────┘
                     │  raw mel-spectrogram (80 bins)
                     ▼
┌─────────────────────────────────────────────────┐
│  Postnet  (5-layer Conv1d, 256 channels)         │
│  • Residual refinement of mel-spectrogram        │
│  • FP32 activations for precision                │
└────────────────────┬────────────────────────────┘
                     │  refined mel-spectrogram
                     ▼
┌─────────────────────────────────────────────────┐
│  HiFi-GAN Vocoder  (runs on CPU)                 │
│  • Converts mel-spectrogram → 16 kHz audio       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
              speech_fp32.wav
```

### Key Model Parameters

| Parameter | Value |
|-----------|-------|
| Encoder layers | 12 |
| Decoder layers | 6 |
| Hidden size | 768 |
| Attention heads | 12 (encoder), 12 (decoder) |
| FFN dimension | 3072 |
| Mel bins | 80 |
| Reduction factor | 2 |
| Speaker embedding dim | 512 |
| Postnet layers | 5 |
| Postnet channels | 256 |
| Postnet kernel | 5 |
| Audio sample rate | 16 kHz |

### Precision Strategy

| Component | Precision | Reason |
|-----------|-----------|--------|
| Encoder weights | BF16 | Speed; encoder runs once per chunk |
| Decoder weights | FP32 | Prevents autoregressive drift over long sequences |
| Decoder SDPA | BF16 + HiFi4 | Trace-compatible; high fidelity matmuls |
| Layer norms | FP32 (HiFi4) | Accumulation precision |
| Postnet activations | FP32 | Final mel quality |

---

## TTNN Optimizations

### Input Shape Normalization (key to consistent performance)
All encoder inputs are zero-padded to one of three fixed canonical sizes before running the encoder. This mirrors the LLM `get_padded_prefill_len` approach:

| Token count | Padded to |
|-------------|-----------|
| 1 – 64 | 64 |
| 65 – 128 | 128 |
| 129 – 256 | 256 |

Because the encoder always runs at a fixed shape, compiled kernels are **always reused** for any input text — including new texts never seen during warm-up.

### Trace Capture
The decoder runs on a trace from step 1 onward. Traces are pre-captured during warm-up for all three encoder sizes, so inference never triggers on-the-fly compilation.

### KV Cache
Self-attention KV cache is pre-allocated per encoder size. Cross-attention K/V are computed once at step 0 and reused for all subsequent steps.

### Text Chunking
Long texts are automatically split at sentence boundaries into chunks of at most `--max_chunk_size` characters (default 256). Each chunk is processed independently and the audio is concatenated.

---

## Prerequisites

1. Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
2. Installed [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
3. HuggingFace login (for model weights):
   ```bash
   huggingface-cli login
   # or
   export HF_TOKEN=<your_token>
   ```
   Token: https://huggingface.co/docs/hub/security-tokens

---

## Environment Setup

Run from the `tt-metal` root directory:

```bash
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate
```

---

## How to Run

### Basic Usage

```bash
# N150
MESH_DEVICE=N150 python models/experimental/speecht5_tts/demo_ttnn.py \
    --text "Hello world, this is text to speech synthesis on Tenstorrent hardware."

# N300
MESH_DEVICE=N300 python models/experimental/speecht5_tts/demo_ttnn.py \
    --text "Hello world, this is text to speech synthesis on Tenstorrent hardware."
```

Output is saved to `speech_fp32.wav` by default.

### Long Text (automatic chunking)

```bash
MESH_DEVICE=N150 python models/experimental/speecht5_tts/demo_ttnn.py \
    --text "They knelt on the rough floorboards before the large wooden chest at the far end of the room. \
It was the only object free of dust, likely because Elias had been subconsciously polishing it \
with his sleeve every time he came up here. Julianne reached in and pulled out a heavy brass object \
wrapped in velvet. She unwrapped it slowly. It was a sextant." \
    --max_chunk_size 256
```

### Custom Output Path

```bash
MESH_DEVICE=N150 python models/experimental/speecht5_tts/demo_ttnn.py \
    --text "Your text here." \
    --output /path/to/output.wav
```

### Multiple Texts (batch)

```bash
MESH_DEVICE=N150 python models/experimental/speecht5_tts/demo_ttnn.py \
    --texts "First sentence." "Second sentence." "Third sentence." \
    --output_dir ./outputs/
```

### All CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--text` | — | Single input text string |
| `--texts` | — | Multiple input texts (space-separated) |
| `--output` | `speech_fp32.wav` | Output WAV file path |
| `--output_dir` | `./` | Output directory for multi-text mode |
| `--max_steps` | `300` | Max mel frames per chunk (longer = more audio) |
| `--max_chunk_size` | `256` | Max characters per chunk for long text |

---

## Warm-up Behavior

On first run (cold kernel cache), the demo spends ~2–3 minutes compiling kernels. This is a one-time cost per machine. Subsequent runs load kernels from disk cache in ~7 seconds.

```
First run  (cold cache):  warm-up ~160s  →  inference ~5–6s
Second run (warm cache):  warm-up ~7s    →  inference ~5–6s
```

The warm-up compiles kernels for all three encoder shapes (64, 128, 256) using synthetic inputs, so **any input text gets consistent inference performance** regardless of its token count.

---

## Performance (N150, Wormhole)

Measured on a 454-character input split into 2 chunks (`--max_chunk_size 256`):

| Metric | Value |
|--------|-------|
| TTFT (Time to First Token) | ~70 ms |
| Decoder throughput | ~160–165 mel frames/sec |
| Total inference time | ~5.4 s |
| Generated audio duration | ~19 s |
| Warm-up (cold cache) | ~160 s |
| Warm-up (warm cache) | ~7 s |

> **Note**: TTFT and Token/s are consistent across runs (cold or warm) because all kernel compilation is absorbed into the warm-up phase.

---

## Running Tests

### End-to-end PCC test

```bash
MESH_DEVICE=N150 pytest models/experimental/speecht5_tts/tests/test_end_to_end_pcc.py -v
```

### Autoregressive PCC tracking (numerical fidelity vs PyTorch)

```bash
MESH_DEVICE=N150 python models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py \
    "Your test sentence here." \
    --true-autoregressive 100
```

For long texts (automatic chunking):

```bash
MESH_DEVICE=N150 python models/experimental/speecht5_tts/tests/test_autoregressive_pcc_tracking.py \
    "Long text spanning multiple chunks..." \
    --true-autoregressive 200 \
    --max-chunk-size 256
```

---

## References

- [SpeechT5 Paper (Ao et al., 2021)](https://arxiv.org/abs/2110.07205)
- [microsoft/speecht5_tts on HuggingFace](https://huggingface.co/microsoft/speecht5_tts)
- [microsoft/speecht5_hifigan (vocoder)](https://huggingface.co/microsoft/speecht5_hifigan)
- [CMU ARCTIC speaker embeddings](https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors)
