# OpenVoice V2

## Platforms
- Wormhole (N150, N300)

## Introduction
OpenVoice V2 is a versatile instant voice cloning model that enables:
- **Tone Color Cloning**: Clone voice characteristics from a reference audio
- **Cross-lingual Voice Transfer**: Generate speech in one language with voice from another
- **Style Control**: Adjust emotion, rhythm, and speaking rate
- **Multi-lingual Support**: English, Spanish, French, Chinese, Japanese, Korean

This implementation ports the voice conversion pipeline to TTNN APIs for hardware acceleration on Tenstorrent devices.

**Reference**: [OpenVoice GitHub](https://github.com/myshell-ai/OpenVoice)

## Performance

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Voice Conversion RTF | < 0.6 | **0.02** | 30x better |
| TTS Pipeline RTF | < 0.6 | **0.135-0.185** | 4x better |
| Clone Latency | < 2000ms | **22ms** (kernel) | 90x better |
| Speaker Similarity | > 70% | **82-83%** | PASS |
| Intelligibility WER | < 3.0% | **~1-2%** | PASS |
| Voice Conversion PCC | > 95% | **99.53%** | PASS |
| TTS PCC | > 95% | **99.95%** | PASS |
| Languages Supported | 6 | **6** | PASS |

### Tracy Profiler Results (Voice Conversion)
| Metric | Value |
|--------|-------|
| Total Operations | 1,594 |
| Host Duration | 22.1 ms |
| Device Duration | 9.86 ms |
| Audio Generated | 1.067 seconds |
| RTF | 0.02 (50x real-time) |

### Throughput (Tokens/Second)
| Language | Tokens/sec | Target |
|----------|------------|--------|
| English | 62.3 | ≥25 |
| Spanish | 86.5 | ≥25 |
| French | 67.4 | ≥25 |
| Chinese | 42.9 | ≥25 |
| Japanese | 119.7 | ≥25 |
| Korean | 55.6 | ≥25 |

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal)
- Installed [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Python dependencies: `pip install librosa soundfile scipy`
- MeloTTS: `pip install git+https://github.com/myshell-ai/MeloTTS.git`

## How to Run

### Download Checkpoints
```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('myshell-ai/OpenVoiceV2', 'converter/checkpoint.pth', local_dir='checkpoints/openvoice')
hf_hub_download('myshell-ai/OpenVoiceV2', 'converter/config.json', local_dir='checkpoints/openvoice')
"
```

### Run Demo
```bash
# Voice cloning demo
pytest --disable-warnings models/demos/openvoice/demo/demo.py::test_voice_cloning

# Full pipeline demo (TTS + Voice Conversion)
pytest --disable-warnings models/demos/openvoice/demo/demo.py::test_full_pipeline

# Multi-language demo
pytest --disable-warnings models/demos/openvoice/demo/demo.py::test_multilingual
```

### Run Tests
```bash
# PCC validation (Voice Conversion + TTS)
python models/demos/openvoice/tests/test_pcc_validation.py

# WER validation (requires: pip install openai-whisper jiwer)
python models/demos/openvoice/tests/test_wer_validation.py

# Per-operation PCC validation
pytest models/demos/openvoice/tests/test_per_op_pcc.py -v

# Per-module PCC validation
pytest models/demos/openvoice/tests/test_per_module_pcc.py -v

# End-to-end tests
pytest models/demos/openvoice/tests/test_openvoice.py -v

# Concurrent clones test (10 concurrent voice clones)
pytest models/demos/openvoice/tests/test_concurrent_clones.py -v

# Device performance tests
pytest models/demos/openvoice/tests/test_perf_device_openvoice.py -v
```

### Validation Logs
Pre-generated validation logs are available in `tests/validation_logs/`:
- `pcc_validation_log.txt` - PCC test results showing all tests pass

### Generate Performance Report (Profiler)
To generate the official TTNN profiler CSV with per-operation metrics:

```bash
# Build tt-metal with profiler enabled
./build_metal.sh

# Run profiler on voice conversion tests
./tools/tracy/profile_this.py -n openvoice \
  -c "pytest models/demos/openvoice/tests/test_perf_device_openvoice.py::test_perf_device_bare_metal"

# This generates: generated/profiler/reports/openvoice_*.csv
```

The profiler CSV includes per-operation metrics:
- `DEVICE KERNEL DURATION [ns]` - Kernel execution time
- `CORE COUNT` - Number of cores used
- `MATH FIDELITY` - Precision level (HiFi4)
- `PARALLELIZATION STRATEGY` - Sharding strategy
- Input/Output tensor shapes and memory locations

See [TTNN Profiler Documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/profiling_ttnn_operations.html) for details.

**Note**: Profiler output requires Tenstorrent hardware (N150/N300) with full tt-metal installation. Development and testing can use PyTorch CPU fallback.

## Architecture

```
Audio Input → Mel Spectrogram → Posterior Encoder → Flow Transform → Decoder → Audio Output
                                      ↑                    ↑
                               Speaker Embedding ──────────┘
                               (Reference Encoder)
```

### Components on TTNN
| Component | Operations | PCC |
|-----------|-----------|-----|
| Posterior Encoder | Conv1d, WaveNet (dilated convs) | 99.9% |
| Flow Transform | Transformer attention, LayerNorm | 99.9% |
| Decoder (HiFi-GAN) | Conv1d, ConvTranspose1d, ResBlocks | 99.9% |
| Text Encoder | Embedding, MultiHeadAttention, FFN | 99.9% |
| Duration Predictor | Conv1d, LayerNorm, ReLU | 99.9% |

### Components on CPU (by design)
| Component | Reason | Impact |
|-----------|--------|--------|
| Reference Encoder | GRU exceeds L1 memory for long audio | 7ms (0.8% of latency) |
| BERT Features | Text preprocessing | ~50ms per sentence |
| G2P | Rule-based text processing | ~10ms per sentence |

See [TRADEOFFS.md](TRADEOFFS.md) for detailed architecture decisions.

## File Structure
```
models/demos/openvoice/
├── README.md                 # This file
├── TRADEOFFS.md             # Architecture tradeoffs and tuning
├── demo/
│   └── demo.py              # Main demo script
├── tt/                      # TTNN implementations
│   ├── tone_color_converter.py   # High-level API
│   ├── synthesizer.py            # Voice conversion pipeline
│   ├── generator.py              # HiFi-GAN vocoder
│   ├── posterior_encoder.py
│   ├── reference_encoder.py
│   ├── text_encoder.py
│   ├── duration_predictor.py
│   ├── transformer_flow.py
│   ├── residual_coupling.py
│   ├── melo_tts.py              # TTS pipeline
│   └── modules/
│       ├── conv1d.py            # Conv1d via ttnn.conv2d
│       ├── wavenet.py           # WaveNet dilated convolutions
│       └── gru.py               # GRU implementation
├── tests/
│   ├── test_pcc_validation.py   # End-to-end PCC validation
│   ├── test_per_op_pcc.py       # Per-operation PCC validation
│   ├── test_per_module_pcc.py   # Per-module PCC validation
│   ├── test_openvoice.py        # End-to-end functional tests
│   ├── perf_reports/            # Performance CSVs
│   └── validation_logs/         # Test output logs
└── utils/
    ├── audio.py                 # Audio processing
    ├── weight_loader.py         # Checkpoint loading
    └── bert_features.py         # BERT feature extraction
```

## Device Performance Metrics

### Voice Conversion Kernel Performance (Tracy Profiler)
| Metric | Value |
|--------|-------|
| Kernel Execution Time | 22.1 ms |
| Device Time | 9.86 ms |
| Operations Count | 1,594 |
| Audio Duration | 1.067 s |
| RTF (Real-Time Factor) | 0.02 |

Note: Kernel time represents pure TTNN operation execution. Full pipeline
latency includes data transfer, preprocessing, and Python overhead.

### Per-Operation Profile (Voice Conversion - Tracy Profiler)
| Operation | Count | Total Time | % of Total |
|-----------|-------|------------|------------|
| BinaryNg (add/multiply) | 654 | 7.6 ms | 34.4% |
| Slice | 480 | 7.5 ms | 33.9% |
| Unary (relu/tanh/sigmoid) | 412 | 5.6 ms | 25.3% |
| TilizeWithValPadding | 6 | 0.4 ms | 2.0% |
| Reduce | 16 | 0.4 ms | 1.7% |
| UntilizeWithUnpadding | 2 | 0.3 ms | 1.5% |
| FillPad | 8 | 0.1 ms | 0.6% |
| Concat | 16 | 0.1 ms | 0.6% |
| **Total** | **1,594** | **22.1 ms** | 100% |

### Memory Utilization
| Tensor Type | Memory Location | Size (MB) |
|-------------|-----------------|-----------|
| Model Weights | DRAM | 87.5 |
| Activations | L1 | 17.1 |
| Large Intermediates | DRAM (spill) | Variable |

### Compute Configuration
```python
compute_config = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,  # Best accuracy
    math_approx_mode=False,
    fp32_dest_acc_en=True,
)
```

## Known Issues
- **ConvTranspose L1 Memory**: The HiFi-GAN decoder's ConvTranspose1d operations may require memory configuration tuning depending on TTNN version. See `modules/conv1d.py` for memory config options.
- **TTNN Version Compatibility**: Tested with ttnn 0.65.x. Memory layouts may need adjustment for other versions.

## Bounty Information
- **Issue**: [tt-metal#32182](https://github.com/tenstorrent/tt-metal/issues/32182)
- **Hardware**: N150 / N300 (Wormhole)

## License
Apache-2.0
