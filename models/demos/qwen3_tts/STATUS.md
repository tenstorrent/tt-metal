# Qwen3-TTS Status

**Model:** [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
**Target Device:** N150 (Wormhole B0)
**Status:** ✅ Working

## Model Architecture

```
Reference Audio (.wav)
        │
        ▼
┌─────────────────────────┐
│ Speech Tokenizer Encoder│  ◄── CPU (MimiModel)
│ (Mimi codec)            │
└───────────┬─────────────┘
            │
            ▼
    RVQ Codes [1, 16, seq]
            │
            │         ┌─────────────────────┐
            │         │   Speaker Encoder   │  ◄── TTNN (ECAPA-TDNN)
            │         │   (ECAPA-TDNN)      │
            │         └──────────┬──────────┘
            │                    │
            ▼                    ▼
┌─────────────────────────────────────────┐
│         ICL Embedding Layer             │  ◄── TTNN
│  (text + codec + speaker embedding)     │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│              Talker                     │  ◄── TTNN (KV-cached, traced)
│   28-layer GQA Transformer              │
│   (16 heads, 8 KV heads, 2048 hidden)   │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Code Predictor                │  ◄── TTNN (KV-cached, traced)
│   5-layer Transformer + 15 LM heads     │
│   (8 heads, 4 KV heads, 1024 hidden)    │
└───────────────────┬─────────────────────┘
                    │
                    ▼
        Codec Tokens [seq, 16]
                    │
                    ▼
┌─────────────────────────────────────────┐
│       Speech Tokenizer Decoder          │  ◄── CPU (ConvNext)
│       (ConvNext decoder)                │
└───────────────────┬─────────────────────┘
                    │
                    ▼
          Audio Waveform @ 24kHz
```

## Component Status

| Component | Implementation | Status | Notes |
|-----------|---------------|--------|-------|
| **Speech Tokenizer Encoder** | CPU (PyTorch) | ✅ Done | MimiModel, conv-heavy |
| **Speaker Encoder** | TTNN | ✅ Done | ECAPA-TDNN, 76 weight tensors |
| **Text Embedding** | TTNN | ✅ Done | Embedding + projection |
| **Codec Embedding** | TTNN | ✅ Done | 16 codebook embeddings |
| **Talker (28 layers)** | TTNN | ✅ Done | GQA attention, KV cache, traced |
| **Code Predictor (5 layers)** | TTNN | ✅ Done | Autoregressive, 15 LM heads, traced |
| **Speech Tokenizer Decoder** | CPU (PyTorch) | ✅ Done | ConvNext (too large for L1) |

### TTNN Optimizations
- ✅ KV Cache (O(n²) → O(n))
- ✅ Trace capture (eliminates kernel dispatch overhead)
- ✅ Program cache enabled
- ✅ Pre-computed RoPE tables

## Benchmark: CPU vs TTNN

**Test Configuration:**
- Text: "Hello, this is a test of the Qwen3 TTS speech system running on Tenstorrent hardware."
- ~100 frames generated
- Audio duration: ~7.7s

| Metric | CPU (PyTorch) | TTNN (N150) | Speedup |
|--------|---------------|-------------|---------|
| **Generation time** | 173.86s | 12.12s | **14.3×** |
| **Throughput** | 0.56 fps | 9.47 fps | **16.9×** |
| **Time per frame** | 1,792 ms | 105.8 ms | **16.9×** |
| **Total wall time** | 187s | 29s | **6.4×** |

### TTNN Performance Breakdown
| Stage | Time |
|-------|------|
| Prefill (25 tokens) | 164 ms |
| TTFT (prefill + 1 decode) | 285 ms |
| Talker decode | 37.9 ms/frame |
| CodePredictor | 67.2 ms/frame |
| **Total decode** | **105.8 ms/frame** |

## Demo Commands

### Setup
```bash
cd /home/ubuntu/qwen3_tts/tt-metal
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate
```

### Basic Usage (uses included Jim reference voice)
```bash
python models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py \
    --text "Hello, this is a test of text to speech."
```

### Full Options
```bash
python models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py \
    --text "Your text here" \
    --ref-audio /path/to/reference.wav \
    --ref-text "Transcript of reference audio" \
    --output /tmp/output.wav \
    --max-tokens 256 \
    --language english \
    --trim-frames 30 \
    --auto-trim-bleed
```

### CPU Reference (for comparison)
```bash
python models/demos/qwen3_tts/demo/demo_pure_reference_tts.py \
    --text "Hello, this is a test." \
    --output /tmp/cpu_output.wav
```

## Supported Languages

| Language | Code |
|----------|------|
| English | `english` |
| Chinese | `chinese` |
| French | `french` |
| German | `german` |
| Italian | `italian` |
| Japanese | `japanese` |
| Korean | `korean` |
| Portuguese | `portuguese` |
| Russian | `russian` |
| Spanish | `spanish` |

Cross-lingual voice cloning is supported (e.g., English reference → French output).

## Files

```
models/demos/qwen3_tts/
├── demo/
│   ├── demo_full_ttnn_tts.py      # Main TTNN demo
│   ├── demo_pure_reference_tts.py # CPU reference demo
│   ├── bleed_detector.py          # Reference audio bleed detection
│   ├── jim_reference.wav          # Default reference audio (4s)
│   └── REFERENCE_AUDIO.md         # Reference audio documentation
├── reference/
│   └── functional.py              # PyTorch reference implementations
├── tt/
│   ├── qwen3_tts.py              # Main TTNN model
│   ├── talker.py                 # Talker transformer
│   ├── code_predictor.py         # Code predictor
│   ├── attention.py              # GQA attention with KV cache
│   ├── decoder_layer.py          # Transformer decoder layer
│   ├── speaker_encoder.py        # ECAPA-TDNN speaker encoder
│   └── rope.py                   # Rotary position embeddings
└── tests/
    ├── test_voice_clone_tts.py   # E2E voice cloning tests
    ├── test_ttnn_blocks.py       # Component PCC tests
    └── ...
```

## Known Limitations

1. **Speech Tokenizer Encoder/Decoder on CPU** - Conv layers too large for L1 SRAM
2. **Single device only** - N300 tensor parallel support not yet implemented
3. **Reference audio bleeding** - ICL models briefly echo reference; use `--trim-frames` or `--auto-trim-bleed`
