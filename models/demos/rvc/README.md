# RVC (Retrieval-based Voice Conversion) — TTNN Implementation

Voice conversion pipeline on Tenstorrent N300 (Wormhole B0) using TTNN APIs.
Converts source speech to a target speaker's voice while preserving content.

## Architecture

The pipeline uses a hybrid Torch/TTNN design. Preprocessing runs on the host (CPU),
while the compute-intensive decoder and vocoder run on the Tenstorrent accelerator.

```
Source WAV (any sample rate)
  ↓  resample to 16kHz
HuBERT Feature Extraction (torch, CPU)
  ↓  [1, T, 768] speech features
Feature Retrieval (FAISS, optional, CPU)
  ↓  [1, T, 768] retrieved + blended features
TextEncoder / Posterior (torch, CPU)
  ↓  z_p [1, 192, T] latent prior + logs_p
F0 Pitch Extraction — RMVPE (torch, CPU)
  ↓  f0 [T] fundamental frequency contour
SineGen (torch, CPU)
  ↓  har_source [1, 1, T×480] harmonic excitation signal
Flow Decoder (TTNN, N300)
  ↓  z [1, 192, T] decoded latent
HiFi-GAN Generator (TTNN, N300)
  ↓  audio [1, 1, T×480] raw waveform
WAV Output (48kHz)
```

### Torch ↔ TTNN Boundary

| Component | Runtime | Notes |
|---|---|---|
| HuBERT | Torch (CPU) | Transformer with relative attention — not ported |
| TextEncoder | Torch (CPU) | WaveNet-style multi-layer conv — not ported |
| RMVPE | Torch (CPU) | Conv2d + BiGRU pitch model — not ported |
| FAISS retrieval | CPU | Index search is inherently CPU-native |
| SineGen | Torch (CPU) | Small sinusoidal generation — trivial compute |
| **Flow Decoder** | **TTNN (N300)** | 4-flow ResidualCouplingBlock with conditioned WaveNet |
| **HiFi-GAN Generator** | **TTNN (N300)** | 4 upsample stages + 12 ResBlocks (72 conv1d ops) |

## Setup

### Hardware

- Tenstorrent N300 (Wormhole B0)
- tt-metal SDK with `ttnn` Python package

### Model Assets

Download and place in `models/demos/rvc/data/`:

| File | Source | Notes |
|---|---|---|
| `f0G48k.safetensors` | [RVC-Project HuggingFace](https://huggingface.co/lj1995/VoiceConversionWebUI) | RVC v2 48kHz checkpoint |
| `f0G48k.json` | Same repository | Model config |
| `hubert.safetensors` | [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960) | HuBERT feature extractor |
| `hubert.json` | Same repository | HuBERT config |
| `rmvpe.safetensors` | Converted from [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt) | Pitch extraction model |
| `sample.wav` | Any speech recording | Input audio for conversion |

A download helper is provided: `bash assets-download.sh`

### Python Dependencies

```bash
pip install pyworld scipy soundfile safetensors librosa faiss-cpu
```

For evaluation (optional):
```bash
pip install resemblyzer openai-whisper
```

## Usage

### Inference Demo

```bash
# Default: 5s clip, RMVPE pitch extraction
python -m models.demos.rvc.demo

# With options
python -m models.demos.rvc.demo \
  --max_secs 10.0 \
  --f0_method rmvpe \
  --key 0 \
  --speaker_id 0

# Using DIO pitch (no RMVPE model needed)
python -m models.demos.rvc.demo --f0_method dio

# With FAISS feature retrieval
python -m models.demos.rvc.demo \
  --index_path models/demos/rvc/data/speaker.index \
  --index_rate 0.5
```

**Output:**
- `data/output/ttnn_output.wav` — TTNN-generated audio (48kHz)
- `data/output/torch_reference.wav` — PyTorch reference for comparison
- Timing summary with RTF and PCC printed to stdout

### Profiling

```bash
python -m models.demos.rvc.profile --max_secs 3.0
```

### Evaluation

```bash
python -m models.demos.rvc.evaluate \
  --ttnn_audio data/output/ttnn_output.wav \
  --torch_audio data/output/torch_reference.wav \
  --source_audio data/sample.wav
```

### Tests

```bash
pytest models/demos/rvc/tests/ -v
```

## Stage 1 Results

### Correctness

| Metric | Value | Target | Status |
|---|---|---|---|
| Audio PCC (TTNN vs Torch) | 0.998 | > 0.95 | ✅ |
| Flow PCC | 0.9999 | — | ✅ |
| Speaker similarity (TTNN vs Torch) | 0.999 | > 0.75 | ✅ |
| WER | 0.000 | < 2.5 | ✅ |
| Flow throughput | ~1973 frames/s | 30 tokens/s | ✅ |
| Tests | 5/5 | — | ✅ |

### Performance

| Metric | Value | Target | Status |
|---|---|---|---|
| **RTF (TTNN only)** | **3.14** | **< 0.5** | **❌** |
| Generator time (5s audio) | 15.4s | — | — |
| Flow time (5s audio) | 0.26s | — | — |

### Runtime Breakdown (5s input, 10 chunks × 50 frames)

| Stage | Time | % of TTNN |
|---|---:|---:|
| TTNN Flow Decoder | 0.26s | 1.7% |
| **TTNN Generator** | **15.38s** | **98.3%** |
| TTNN Total | 15.65s | — |

### RTF Bottleneck Analysis

The RTF gap is caused by **host↔device dispatch overhead**, not insufficient device compute:

- Generator executes ~79 `ttnn.conv1d` dispatches per chunk
- Each dispatch: ~45ms total, of which only ~5ms is device kernel time
- The remaining ~40ms is host-side tensor conversion and command dispatch
- For 10 chunks: 790 round-trips × ~40ms overhead ≈ 31.6s of host overhead

This is a known framework-level bottleneck documented in the
[Advanced Performance Optimizations](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md)
tech report. Resolution requires Stage 2 techniques (Metal Trace, device-resident activations).

## File Structure

```
models/demos/rvc/
├── demo.py                  # End-to-end inference with timing
├── evaluate.py              # PCC, speaker similarity, WER evaluation
├── profile.py               # Detailed per-component runtime profiling
├── run_torch_inference.py   # Standalone torch reference pipeline
├── README.md
├── assets-download.sh       # Model weight download helper
├── .gitignore
│
├── data/                    # Model weights and audio (not committed)
│   └── output/              # Generated WAV files
│
├── torch_impl/              # PyTorch reference implementations
│   ├── rmvpe.py             # RMVPE pitch extraction model
│   ├── crepe.py             # CREPE pitch extraction (alternative)
│   └── vc/
│       ├── hubert.py        # HuBERT feature extractor
│       ├── pipeline.py      # Reference RVC inference pipeline
│       └── synthesizer.py   # Full VITS/RVC model architecture
│
├── ttnn/                    # TTNN implementations
│   ├── runtime.py           # Persistent modules: TTNNFlowDecoder, TTNNGeneratorNSF
│   ├── utils.py             # Device transfer and weight preprocessing
│   ├── ops/
│   │   ├── conv_transpose1d.py  # ConvTranspose1d via conv_transpose2d
│   │   ├── conv1d.py            # Conv1d wrapper (used by tests)
│   │   ├── linear.py            # Linear wrapper (used by tests)
│   │   └── layer_norm.py        # LayerNorm wrapper (used by tests)
│   └── modules/             # Early module implementations (reference only)
│       ├── flow.py           # Flow decoder functional implementation
│       ├── wavenet.py        # WaveNet functional implementation
│       ├── hubert_encoder.py # Encoder layer (not used in production)
│       └── hubert_ffn.py     # FFN block (not used in production)
│
├── tests/
│   ├── conftest.py          # Device fixture
│   ├── pcc_utils.py         # PCC assertion utilities
│   ├── test_runtime.py      # Runtime lifecycle + correctness (5 tests)
│   └── test_ttnn_ops.py     # Per-operator PCC validation (10 tests)
│
└── utils/
    ├── audio.py             # Audio loading/resampling
    ├── config.py            # Model config loading
    └── f0.py                # F0 method enum
```

## Design Decisions

1. **Persistent weight architecture** — Weights preprocessed and uploaded to device once during `from_checkpoint()`, reused across forward calls. Solves L1 OOM from per-forward weight recreation.

2. **Chunked inference with overlap-add** — Audio processed in 50-frame (~0.5s) chunks with 5-frame overlap. Required because the HiFi-GAN upsampling chain (480× total) would exceed L1 for longer sequences.

3. **Uniform chunk padding** — Last chunk zero-padded to match standard chunk shape. Prevents ttnn.conv1d JIT cache from compiling new kernels (which fills L1).

4. **Native conv1d bias** — Bias preprocessed as ttnn tensor and passed to `conv1d(bias_tensor=...)`. Eliminates host-side bias addition after every conv dispatch. Measured 25% generator speedup.

5. **RMVPE from official source** — Pitch model from the official [RVC-Project repository](https://huggingface.co/lj1995/VoiceConversionWebUI), ensuring checkpoint compatibility.

## Stage 2 Optimization Path

Profiling identifies these optimization opportunities for the RTF target:

| Optimization | Expected Impact | Description |
|---|---|---|
| Device-resident activations | 2-3× | Keep tensors on device between consecutive ops |
| Metal Trace | 3-5× (stacked) | Record dispatch sequence, replay from DRAM |
| Op fusion | 1.5-2× | Fuse conv + activation + bias into single kernel |
| Sharding | 1.3-1.5× | Height/block sharding per TTNN bringup guide |
| LoFi math fidelity | 1.1× | Config-level change, minimal code impact |

Combined theoretical improvement: 6-15× → RTF 0.21-0.52.
