# Qwen3-TTS 1.7B on Tenstorrent P150a

Japanese text-to-speech pipeline running Qwen3-TTS-12Hz-1.7B-Base on a single Tenstorrent P150a (Blackhole) device.

## Pipeline Overview

```
Text ("こんにちは")
  │
  ▼ Tokenize + embed (CPU)
  │
  ▼ Talker prefill (TT device) ─── 28-layer Qwen3, 1.7B params
  │
  ├──────────── per-frame decode loop ────────────┐
  │                                                │
  │  ▼ Talker decode (TT device)                   │
  │  │  → CB0 token + post-norm hidden state       │
  │  │                                              │
  │  ▼ Code Predictor (TT device) ─── 5-layer Qwen3│
  │  │  prefill(2 tok) + 14 decode steps w/ KV cache│
  │  │  → CB1-CB15 tokens                           │
  │  │                                              │
  │  ▼ Build next input embedding (CPU)             │
  │  │  sum(16 codec embeddings) + text_hidden      │
  │  │  → H2D transfer → next Talker input          │
  │  │                                              │
  │  └──── repeat until EOS ────────────────────────┘
  │
  ▼ Vocoder (CPU) ─── 114M params, Conv + Transformer
  │  16 codebooks → 24kHz waveform
  │
  ▼ WAV output
```

## Execution Location: TT Device vs CPU

| Component | Location | Parameters | Notes |
|-----------|----------|------------|-------|
| **Tokenizer** | CPU | - | HuggingFace AutoTokenizer |
| **Text embedding + projection** | CPU → TT → CPU | ~300M (shared w/ Talker) | `embed_text_tokens()` on CPU, `text_projection()` on TT, result back to CPU for concat |
| **Talker (prefill)** | **TT device** | 1.7B | 28-layer Qwen3, paged attention, bfloat16, full prefill sequence |
| **Talker (decode)** | **TT device** | 1.7B | Single-token decode with KV cache, returns logits + post-norm hidden |
| **Code Predictor** | **TT device** | ~30M | 5-layer Qwen3 (dim=1024), prefill(2 tok) + 14 decode steps with KV cache per frame |
| **Codec embedding lookup** | CPU | 15 tables, 2048-dim each | `F.embedding()` + `F.linear()` projection, then H2D transfer to TT |
| **Decode input construction** | CPU | - | Sum of 16 codec embeddings + trailing text hidden |
| **Speaker Encoder** | CPU | 12M | ECAPA-TDNN (Conv1d), runs once per utterance for voice cloning |
| **Vocoder** | CPU | 114M | Code2Wav: SplitResidualVQ + Transformer + ConvNeXt upsampler |

### Why some components run on CPU

- **Speaker Encoder (12M)**: Small CNN with dilated Conv1d and attentive pooling. Runs once per utterance. The complex Conv1d patterns would require significant ttnn tuning for minimal benefit.
- **Vocoder (114M)**: Non-autoregressive (runs once after all frames generated). Primarily ConvNet upsampler which is hard to port efficiently. The 8-layer Transformer portion could move to TT.
- **Codec embedding + projection**: Each decode step looks up 15 embedding tables and applies a linear projection. This is a lightweight CPU operation (~0.1ms) followed by a single H2D transfer.

### Latency Breakdown (per decode step)

Based on profiling with "こんにちは" (Japanese, ~10 frames):

| Stage | Time (ms) | % of step |
|-------|-----------|-----------|
| Code Predictor (TT) | ~169 | 72.6% |
| Talker decode (TT) | ~3.5 | 1.5% |
| Embedding build (CPU) | ~0.3 | 0.1% |
| H2D transfer | ~1.0 | 0.4% |
| D2H + postprocess | ~1.5 | 0.6% |
| Decode prep (CPU) | ~57 | 24.7% |

The Code Predictor dominates step time. Its 5-layer Transformer runs 15 forward passes per frame (1 prefill + 14 decode) on the TT device with KV cache.

## Evaluation Results

100-sentence Japanese dataset evaluation (7 categories):

| Metric | Value |
|--------|-------|
| **CER** (Character Error Rate) | **6.3%** (2.8% excluding numbers) |
| **UTMOS** (Naturalness MOS) | **3.98** / 5.0 |
| **RTF** (Real-Time Factor) | **3.16** (target: < 1.0) |

### Per-Category Breakdown

| Category | Samples | CER | UTMOS | RTF |
|----------|---------|-----|-------|-----|
| short | 20 | 3.7% | 3.84 | 3.08 |
| medium | 30 | 2.6% | 4.09 | 3.18 |
| long | 10 | 2.4% | 4.13 | 3.24 |
| numbers | 10 | 36.5% | 4.13 | 3.21 |
| mixed_language | 10 | 4.9% | 3.92 | 3.17 |
| emotion | 10 | 3.2% | 3.86 | 3.16 |
| keigo | 10 | 0.9% | 3.84 | 3.17 |

- CER measured via Whisper Large v3 ASR (generate audio → transcribe → compare to original text)
- UTMOS measured via UTMOS22 neural MOS predictor
- Numbers category has high CER due to TTS/ASR difficulty with digit sequences (known issue)
- bfloat16 precision on TT device yields ~71% token match vs CPU float32, but audio quality (UTMOS ~4.0) is not significantly degraded

## Quick Start

```bash
# Docker container with TT device
sudo docker exec -w /path/to/tt-metal qwen3_tts_test bash -c "
  source /opt/venv/bin/activate &&
  export PYTHONPATH=/path/to/tt-metal:/tt-metal &&
  python models/demos/qwen3_tts/demo/demo_ja.py
"
```

## Running Evaluation

```bash
# 5-sample smoke test
python models/demos/qwen3_tts/evaluation/run_benchmark.py \
  --config models/demos/qwen3_tts/evaluation/benchmark_config.yaml \
  --backend tt --dataset custom_ja --max_samples 5 --save_audio \
  --output_dir models/demos/qwen3_tts/evaluation/results/tt_smoke

# Full 100-sample evaluation
python models/demos/qwen3_tts/evaluation/run_benchmark.py \
  --config models/demos/qwen3_tts/evaluation/benchmark_config.yaml \
  --backend tt --dataset custom_ja --save_audio \
  --output_dir models/demos/qwen3_tts/evaluation/results/tt_full
```

Dependencies for evaluation: `openai-whisper`, `librosa`, `soundfile`, `torchaudio` (CPU), `pyyaml`

## File Structure

```
models/demos/qwen3_tts/
├── tt/                          # TT device implementations
│   ├── generator.py             # End-to-end pipeline orchestrator
│   ├── talker.py                # 28-layer Qwen3 on TT (prefill + decode)
│   ├── code_predictor.py        # 5-layer Qwen3 on TT with KV cache
│   ├── code_predictor_cpu.py    # CPU float32 reference (fallback)
│   ├── speaker_encoder.py       # ECAPA-TDNN on CPU
│   ├── vocoder.py               # Code2Wav on CPU
│   ├── model_config.py          # TalkerModelArgs, CodePredictorModelArgs
│   └── configs.py               # CodePredictorConfig, etc.
├── reference/                   # PyTorch reference implementations
├── evaluation/                  # Benchmark framework
│   ├── run_benchmark.py         # Main benchmark runner
│   ├── benchmark_config.yaml    # Evaluation config
│   ├── datasets/custom_ja.json  # 100-sentence Japanese dataset
│   └── metrics/                 # CER, UTMOS, RTF, PESQ evaluators
├── demo/                        # Demo scripts
└── tests/                       # Unit and integration tests
```

## Architecture Details

### Talker (TT device)
- Qwen3-1.7B: 28 layers, hidden=2048, 16 Q-heads / 8 KV-heads, head_dim=128
- SwiGLU FFN: intermediate=11264
- RMSNorm, bfloat16 weights
- Paged attention with chunked prefill
- Outputs: CB0 logits (vocab=3072) + post-norm hidden state (dim=2048)

### Code Predictor (TT device)
- 5-layer Qwen3: hidden=1024, 16 Q-heads / 8 KV-heads, head_dim=128
- FFN intermediate=3072
- Input: Linear(2048→1024) projection from Talker space
- 15 separate LM heads (1024→2048), one per codebook
- 15 codec embedding tables (vocab=2048, dim=2048) on host
- Per-frame: prefill 2 tokens, then 14 decode steps with KV cache
- bfloat16 on device, float32 embedding/projection on host

### Vocoder (CPU)
- 114M params, non-autoregressive
- SplitResidualVQ dequant + 8-layer Transformer + ConvNeXt upsampler
- Total upsample factor: 1920x (12.5 Hz codebook rate → 24 kHz audio)
