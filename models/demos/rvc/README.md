# RVC (Retrieval-based Voice Conversion) — TTNN Implementation

Voice conversion pipeline on Tenstorrent N300 (Wormhole B0) using TTNN APIs.
Converts source speech into a target speaker voice while preserving linguistic content.

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
  --ttnn data/output/ttnn_output.wav \
  --ref data/output/torch_reference.wav \
  --source data/speech/sample-speech-0.wav
```

### Tests

Run each file as a separate pytest invocation — co-executing them in one
session accumulates TTNN device state and segfaults (this does not
reproduce in real demo runs, where the device is opened once per process):

```bash
pytest models/demos/rvc/tests/test_runtime.py -v
pytest models/demos/rvc/tests/test_production_shapes.py -v
pytest models/demos/rvc/tests/test_ttnn_ops.py -v
```

### End-to-end benchmark

```bash
python -m models.demos.rvc.benchmark --max_secs 10.0
```

Runs the exact `demo.py` graph with cold/warm separation and a strict
no-fallback guard. This is the authoritative RTF measurement path —
microbenchmarks are not used for headline numbers.

## Stage 1 Results

### Correctness (all targets met)

| Metric | Value | Target | Status |
|---|---|---|---|
| Audio PCC (TTNN vs Torch) | 0.998 | > 0.95 | ✅ |
| Flow PCC | 0.9999 | — | ✅ |
| Speaker similarity¹ | 0.999 | > 0.75 | ✅ |
| WER | 0.000 | < 2.5 | ✅ |
| Flow throughput | ~1973 frames/s | 30 tokens/s | ✅ |
| Tests | 53/53 | — | ✅ |

¹ Speaker similarity measured between TTNN and Torch outputs (cosine similarity
of speaker embeddings). This validates that TTNN faithfully reproduces the
torch model's voice characteristics. A value of 0.999 indicates near-identical
output, confirming TTNN correctness.

### Performance — validated end-to-end on `benchmark.py` (real demo path)

All numbers below come from `python -m models.demos.rvc.benchmark`, the
authoritative measurement harness. It runs the exact `demo.py` inference
graph (same modules, same chunked overlap-add) with one-time setup,
cold-start separated from warm steady-state, and a strict no-fallback
guard that raises on any silent torch routing. No microbenchmarks.

**TTNN-only RTF** = wall-clock time spent in `flow.forward + gen.forward`
divided by output audio duration. **Full-pipeline RTF** additionally
includes torch preprocessing (Hubert + RMVPE + TextEncoder + SineGen).

| Clip | Cold RTF (TTNN only) | Warm RTF (TTNN only) | Cold RTF (full) | Warm RTF (full) | Audio PCC | Warm cv |
|---|---:|---:|---:|---:|---:|---:|
| 3.00s   | 0.3537 | **0.2763** | 0.6547 | 0.5276 | 0.998 | 1.3% |
| 10.00s  | 0.2948 | **0.2742** | 0.4432 | **0.4067** | 0.998 | 1.2% |

Numbers are from a clean N300 environment rebuild (`warmup=1, runs=3`).
Cold-start is a single first run that pays `ttnn` JIT compilation, so it
varies run-to-run (3s cold has been observed between ~0.35 and ~0.60);
the warm steady-state means (the bolded headline) are stable to ~1–2%.

| Target | 3s | 10s |
|---|---|---|
| TTNN-only RTF < 0.5 | ✅ 0.28 | ✅ 0.28 |
| Full-pipeline RTF < 0.5 | ❌ 0.53 (preprocessing-dominated) | ✅ 0.41 |

**Preprocessing dominates at short clips.** On 3s audio the torch
preprocessing (Hubert + RMVPE) is roughly 0.75s of fixed cost — about
half the full-pipeline wall time. On 10s audio that same cost amortizes
and the full pipeline meets RTF < 0.5 comfortably. RMVPE pitch
extraction is the largest single preprocessing line item.

### Runtime Breakdown (warm steady-state)

| Stage | 3s (warm) | 10s (warm) | % of TTNN at 10s |
|---|---:|---:|---:|
| Preprocessing (Hubert + RMVPE + TextEncoder + SineGen) | 0.802s | 1.395s | — |
| TTNN Flow Decoder | 0.105s | 0.361s | ~13% |
| **TTNN Generator** | **0.709s** | **2.391s** | **~87%** |
| TTNN Total | 0.814s | 2.752s | 100% |

At 10s, RMVPE pitch extraction (~0.75s) is the largest single preprocessing
line item, and the Generator is ~87% of TTNN time — which is why the
device-resident ResBlock loop targets exactly that path.

### How the optimization works

The Generator's inner ResBlock loop runs `leaky_relu → conv1d →
leaky_relu → conv1d → add` for three dilation iterations per ResBlock,
12 ResBlocks per generator forward. Pre-optimization each conv1d went
host → device → host, paying a host roundtrip per call. The optimized
path keeps activations device-resident across the whole inner loop;
the only host roundtrips are one `from_torch` at ResBlock entry and
one `to_torch` at exit.

One subtle dependency was found and documented: `ttnn.leaky_relu`
requires `Layout.TILE`, but `ttnn.conv1d` at the HiFi-GAN ResBlock
config `(k=11, d=5, ch=128, in_width=7200)` rejects `Layout.TILE`
inputs with a program-compilation failure (storage and memory_config
are not the cause; only layout matters). The fix is one
`ttnn.to_layout(x, ROW_MAJOR_LAYOUT)` between each leaky_relu and the
conv1d that follows it. See `_resblock1_device` in `ttnn/runtime.py`.

### Validation

- `tests/test_runtime.py` — 5/5 (existing T=10 smoke + correctness)
- `tests/test_production_shapes.py` — 9/9 (k=11 at seq_lens 720, 7200,
  14400, 28800; gen at demo chunk sizes 53, 55, 60; determinism;
  PCC vs torch reference)
- `tests/test_ttnn_ops.py` — 39/39 (per-op PCC coverage)
- `benchmark.py` — strict no-fallback at both 3s and 10s, all chunks
  active on TTNN, audio PCC ≥ 0.997 across all runs

The three test files must currently be run as **separate pytest
invocations**; co-execution in one session causes TTNN device-state
accumulation that does not reproduce in real demo runs.

## File Structure

```
models/demos/rvc/
├── demo.py                  # End-to-end inference with timing
├── benchmark.py             # Authoritative RTF benchmark (cold/warm, no-fallback)
├── evaluate.py              # PCC, speaker similarity, WER evaluation
├── profile.py               # Detailed per-component runtime profiling
├── README.md
├── assets-download.sh       # Model weight download helper
├── .gitignore
│
├── data/                    # Model weights and audio (not committed)
│   └── output/              # Generated WAV files
│
├── torch_impl/              # PyTorch reference implementations
│   ├── reference.py         # Torch flow/generator for PCC comparison
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
│   └── ops/
│       └── conv_transpose1d.py  # ConvTranspose1d via conv_transpose2d
│
├── tests/
│   ├── conftest.py          # Device fixture
│   ├── pcc_utils.py         # PCC assertion utilities
│   ├── test_runtime.py              # Runtime lifecycle + correctness (5 tests)
│   ├── test_production_shapes.py    # k=11 + demo chunk regression (9 tests)
│   └── test_ttnn_ops.py             # Per-operator PCC validation (39 tests)
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

## Stage 2 Optimization — outcomes

Stage 2 investigated several optimization directions against the generator
(the ~87% of TTNN time). Each was either landed and measured, or
investigated and deferred with a concrete reason — not left as a
speculative estimate.

| Optimization | Status | Outcome |
|---|---|---|
| Device-resident ResBlock inner loop | **Landed** | Keeps activations on device across the 3-iteration ResBlock loop; one `from_torch`/`to_torch` per call instead of a host roundtrip per conv. Validated at all production seq_lens; PCC preserved (~0.998). See "How the optimization works" above. |
| Metal Trace | **Deferred** | `ttnn.conv1d` does an internal weight upload that fails during trace capture ("Writes are not supported during trace capture"). `prepare_conv_weights` works, but `prepare_conv_bias` fails with "bad optional access". Not worth the complexity until those APIs stabilize. |
| ConvTranspose1d optimization | **Deferred** | Per-stage profiling shows the upsample/ConvTranspose path is ≤3% of real-pipeline time; `prepare_conv_transpose2d_weights` has the same weights-dtype config gap. Effort not justified by the measured share. |
| Sharding / op fusion / LoFi fidelity | **Not pursued** | Standard TT bringup levers; not required to meet the RTF < 0.5 target, which the device-resident path already achieves on TTNN-only at both 3s and 10s. |

The remaining full-pipeline gap at short clips is preprocessing-bound
(torch HuBERT + RMVPE), not dispatch-bound on the device — see the
performance tables above. At 10s the full pipeline already meets
RTF < 0.5.
