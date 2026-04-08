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

| Optimization | Status | Notes |
|--------------|--------|-------|
| **KV Cache** | ✅ Done | Talker (28 layers) + CodePredictor (5 layers), O(n²) → O(n) |
| **Trace Capture** | ✅ Done | Talker decode + CP prefill + 14 CP decode traces |
| **Program Cache** | ✅ Done | `device.enable_program_cache()` |
| **Pre-computed RoPE** | ✅ Done | Tables computed once at init, O(1) slice per step |
| **Prefill Bucketing** | ✅ Done | 9 buckets for variable-length ICL sequences |
| **2CQ Streaming** | ✅ Done | Async token transfer + parallel CPU decode (`generator_2cq.py`) |
| **Tokenizer** | ✅ Done | HuggingFace `AutoTokenizer` integration |
| **Weight Caching** | ✅ Done | Weights cached on first run for faster startup |
| **Paged Attention** | ❌ Not implemented | Standard KV cache only |
| **Continuous Batching** | ❌ Not implemented | Batch size = 1 only |
| **Tensor Parallel (N300)** | 🔧 WIP | See `tt/qwen3_tts_mesh.py` |

### Trace Architecture
The demo pre-captures traces on startup for zero compile/dispatch overhead during inference:
- **Talker decode trace** (1): Single token decode through 28 layers
- **CP prefill trace** (1): 2-token prefill (past_hidden + code0)
- **CP decode traces** (14): One trace per code position (each with baked-in LM head)

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
    --text "Hello, this is a test of text to speech." \
    --ref-audio models/demos/qwen3_tts/demo/jim_reference.wav \
    --ref-text "Jason, can we take a look at the review slides"
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

### Web Demo (Gradio UI)
```bash
python models/demos/qwen3_tts/demo/web_demo.py
# Opens on http://localhost:7777
# Pre-warms all prefill buckets and traces on startup
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
│   ├── web_demo.py                # Gradio web UI (port 7777)
│   ├── bleed_detector.py          # Reference audio bleed detection
│   ├── jim_reference.wav          # Default reference audio (4s)
│   └── REFERENCE_AUDIO.md         # Reference audio documentation
├── reference/
│   └── functional.py              # PyTorch reference implementations
├── tt/
│   ├── qwen3_tts.py              # Main TTNN model (N150)
│   ├── qwen3_tts_mesh.py         # N300 tensor parallel (WIP)
│   ├── talker.py                 # Talker transformer
│   ├── code_predictor.py         # Code predictor
│   ├── attention.py              # GQA attention with KV cache
│   ├── decoder_layer.py          # Transformer decoder layer
│   ├── speaker_encoder.py        # ECAPA-TDNN speaker encoder
│   ├── rope.py                   # Rotary position embeddings
│   ├── kv_cache.py               # KV cache allocation/management
│   └── generator_2cq.py          # 2CQ streaming generator
└── tests/
    ├── test_voice_clone_tts.py   # E2E voice cloning tests
    ├── test_ttnn_blocks.py       # Component PCC tests
    └── ...
```

## Tests

```bash
# Component PCC tests
pytest models/demos/qwen3_tts/tests/test_ttnn_blocks.py -v

# Layer-by-layer PCC tests
pytest models/demos/qwen3_tts/tests/test_layer_pcc.py -v

# End-to-end voice cloning tests
pytest models/demos/qwen3_tts/tests/test_voice_clone_tts.py -v

# Audio quality tests (trace vs no-trace vs CPU)
pytest models/demos/qwen3_tts/tests/test_ttnn_audio_quality.py -v
```

### PCC Results (TTNN vs PyTorch Reference)

| Component | PCC | Notes |
|-----------|-----|-------|
| RMSNorm | 0.999985 | Exact match |
| MLP (SwiGLU) | 0.999976 | Exact match |
| Attention (single layer) | 0.996 | QK-norm + fused QKV |
| DecoderLayer | 0.973 | Pre-norm transformer block |
| Talker (28 layers) | 0.978 | ~0.001 PCC drop per layer |

## Known Limitations

1. **Speech Tokenizer Encoder/Decoder on CPU** - Conv layers too large for L1 SRAM (would need 2.3MB, max L1 is 1.5MB)
2. **Single device only** - N300 tensor parallel support is WIP (see `tt/qwen3_tts_mesh.py`)
3. **Reference audio bleeding** - ICL models briefly echo reference; use `--trim-frames` or `--auto-trim-bleed`
4. **Batch size = 1 only** - No continuous batching or multi-user support
5. **Greedy decoding degrades** - Long text with `--greedy` causes repetition loops; use sampling (default)
