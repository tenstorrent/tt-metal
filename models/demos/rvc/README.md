# RVC (Retrieval-based Voice Conversion) — TTNN Implementation

Voice conversion pipeline on Tenstorrent N300 (Wormhole B0). Converts source speech into a target speaker voice while preserving linguistic content. Hybrid design: CPU preprocessing, TTNN flow decoder and HiFi-GAN vocoder.

## Architecture

```
Source WAV  →  HuBERT (CPU)  →  [optional FAISS retrieval]  →  TextEncoder (CPU)  ─┐
                                                                                    │
                                            RMVPE / DIO pitch (CPU)  →  SineGen ───┤
                                                                                    │
                                                  Flow Decoder (TTNN, N300) ────────┤
                                                                                    │
                                            HiFi-GAN Generator (TTNN, N300)  →  48kHz WAV
```

| Component | Runtime | Notes |
|---|---|---|
| HuBERT | Torch CPU | Transformer with relative attention; not ported |
| TextEncoder | Torch CPU | WaveNet-style multi-layer conv; not ported |
| RMVPE / DIO | Torch CPU / pyworld CPU | Pitch extraction; selectable |
| FAISS retrieval | CPU | Index search is inherently CPU-native |
| SineGen | Torch CPU | Trivial sinusoidal compute |
| Flow Decoder | TTNN (N300) | 4-flow ResidualCouplingBlock with conditioned WaveNet |
| HiFi-GAN Generator | TTNN (N300) | 4 upsample stages, 12 ResBlocks, 72 conv1d ops |

## Setup

### Hardware
- Tenstorrent N300 (Wormhole B0)
- tt-metal SDK with the `ttnn` Python package

### Assets
Place in `models/demos/rvc/data/`:

| File | Source |
|---|---|
| `f0G48k.safetensors` + `f0G48k.json` | [RVC-Project HuggingFace](https://huggingface.co/lj1995/VoiceConversionWebUI) |
| `hubert.safetensors` + `hubert.json` | [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960) |
| `rmvpe.safetensors` | Converted from [rmvpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt) |
| `sample.wav` | Any speech recording (input audio) |

Helper: `bash assets-download.sh`.

### Python dependencies
```bash
pip install pyworld scipy soundfile safetensors librosa faiss-cpu
pip install resemblyzer openai-whisper       # optional, for evaluation
```

## Usage

### Inference demo
```bash
python -m models.demos.rvc.demo                                  # default 5s, RMVPE
python -m models.demos.rvc.demo --f0_method dio                  # DIO pitch
python -m models.demos.rvc.demo --speaker_id 50 --key -6         # different target + pitch shift
python -m models.demos.rvc.demo --index_path data/speaker.index --index_rate 0.5
python -m models.demos.rvc.demo --protect 0.33 --rms_mix_rate 0.25   # artifact prevention (on by default)
```

Artifact-prevention flags:
- `--protect` (default 0.33): consonant protection. On unvoiced frames (pitchf==0), blends pre-retrieval features back so consonants are not smeared by speaker conversion. Set to 0.5 to disable.
- `--rms_mix_rate` (default 0.25): volume envelope. Blends source-audio RMS into the converted output. 1.0 keeps converted dynamics; 0.0 fully transfers source dynamics.

Output: `data/output/ttnn_output.wav` (TTNN) and `data/output/torch_reference.wav`; timing and PCC printed to stdout.

### Benchmark
```bash
# Single-stream, RMVPE
python -m models.demos.rvc.benchmark --max_secs 3.0 --warmup 1 --runs 3

# Batched, 5 concurrent, with CPU/TTNN pipeline overlap (meets RTF<0.2)
python -m models.demos.rvc.benchmark --batch 5 --max_secs 3.0 --warmup 1 --runs 2 \
    --f0_method dio --overlap
```

### Evaluation
```bash
python -m models.demos.rvc.evaluate \
    --ttnn data/output/ttnn_output.wav \
    --ref data/output/torch_reference.wav \
    --source data/speech/sample-speech-0.wav
```

### Tests
```bash
pytest models/demos/rvc/tests/ -v
```

## Performance

Measured on N300 (Wormhole B0), warm JIT cache, 3 s input audio. Numbers are means across warm runs.

### Single-stream (`--batch 1`)

| Metric | Value | Bounty target |
|---|---:|---|
| TTNN-only RTF | 0.177 | < 0.5 |
| Full-pipeline RTF | 0.300 | < 0.5 |
| Audio PCC vs torch reference | 0.998 | > 0.95 |

### Batched (`--batch 5 --overlap --f0_method dio`)

| Metric | Value | Bounty target |
|---|---:|---|
| TTNN-only RTF / sample | 0.140 | < 0.5 (stretched: < 0.2) |
| Full-pipeline RTF / sample | 0.187 | stretched: < 0.2 |
| Per-row PCC vs B=1 reference | > 0.995 | — |

### Where time goes (single-stream, 3 s warm)

| Stage | Per-sample time | Share | Where |
|---|---:|---:|---|
| HuBERT | 0.15 s | 16% | torch CPU |
| RMVPE pitch | 0.19 s | 21% | torch CPU |
| TextEncoder + SineGen | 0.03 s | 4% | torch CPU |
| Flow (TTNN) | 0.04 s | 4% | N300 |
| Generator (TTNN) | 0.49 s | 54% | N300 |

CPU preprocess is ~40% of single-stream wall and is correctly off TTNN by architecture (HuBERT/RMVPE not ported — see Limitations).

### Pitch method comparison (B=5, ProcessPool spawn)

| Pitch | Preprocess /sample | Full RTF /sample | Speaker sim vs RMVPE | WER vs RMVPE |
|---|---:|---:|---:|---:|
| RMVPE | 0.295 s | 0.239 | 1.000 (reference) | 0.000 |
| DIO | 0.196 s | 0.209 | 0.969 | 0.000 |
| DIO + `--overlap` | 0.102 s (observable) | 0.187 | 0.969 | 0.000 |

DIO and RMVPE are perceptually equivalent (cosine sim 0.969, Whisper produces identical transcriptions). RMVPE remains the default.

## Diverse use case validation

| Case | Result |
|---|---|
| Speaker diversity (speaker_id 0 / 50 / 100) | Resemblyzer cosine similarity between pairs: 0.576 / 0.636 / 0.771. Same-speaker reference is 0.969, so 0.58–0.77 confirms meaningfully distinct target voices. |
| Pitch transposition (-12, -6, +6, +12 semitones) | All four shifts produce valid audio (NaN=0, Inf=0, amplitude within [-0.95, +0.95]) and all audibly different from the k=0 baseline (cosine sim 0.76–0.86, all below 0.90). |
| Real-time conversion | RTF 0.187 at B=5 with `--overlap --f0_method dio` (~5× real-time per sample). |

## Bounty status

Stage 1 (bring-up) and Stage 2 (optimization) targets are met with margin. Stage 3 status against the bounty's exact wording:

| Bullet | Status |
|---|---|
| Maximize core counts | Done (HEIGHT_SHARDED conv1d, 8×7 grid where shape permits) |
| Efficient flow-based decoder | Done (device-resident WN inner loop; B=8 at 5.4× per-sample) |
| Flash Attention or equivalent | N/A (RVC has no attention layers) |
| Minimize voice conversion latency | Done (TTNN-only RTF 0.535 → 0.140, 74% reduction) |
| Batch processing | Done (native B=2..8, per-row correctness tested at B=2/3/5/8) |
| Optimize feature index search | N/A by design (FAISS < 1% of wall time) |
| Pipeline encoder/decoder/vocoder stages | Done for batched path (`--overlap`); single-stream cross-chunk overlap not implemented |
| Efficient pitch extraction and transposition | Done (RMVPE persisted, `--key` transposition supported, DIO alternative measured) |
| Minimize memory and TM overheads | Done (WN and ResBlock inner loops eliminate host roundtrips; `prepare_conv_weights` cache) |
| Caching feature indices | N/A by design |
| Document tuning and trade-offs | Done (this README + commit history) |
| **Stretched:** 60+ tokens/second | Done (Flow at B=8: ~6900 tokens/s) |
| **Stretched:** RTF < 0.2 | Met. TTNN-only 0.140; full pipeline 0.187 with `--batch 5 --overlap --f0_method dio` |
| **Stretched:** 5+ concurrent conversions | Done (verified end-to-end at B=5 and B=8) |

## Limitations

**Trace + 2CQ not shipped.** `ttnn.conv1d` is trace-capturable in isolation (verified, 2.63× speedup, PCC 1.0). Integration across the full forward path is blocked architecturally: `Flow.forward` and `Generator.forward` are deeply hybrid — torch ops (residual adds, tanh, leaky_relu, torch conv1d for noise injection) interleave with TTNN sub-graphs, producing ~20 host↔device round-trips per inference. Trace requires a pure-device op sequence; the round-trips break capture. Closing this requires rewriting both modules as fully device-resident graphs (no torch ops, no `from_torch`/`to_torch` inside forward) — multi-day refactor outside this stage's scope.

**Single-stream cross-chunk pipeline overlap not implemented.** Preprocess is monolithic per inference (produces all frames at once), so within-stream overlap would require re-architecting preprocess as streaming. Batched path closes this for B>1 via `--overlap`.

**HuBERT and RMVPE remain on CPU.** Both are out of TTNN-port scope. RMVPE port was investigated on a sibling branch (`rvc-clean-implementation`, commit `bad66aa1a2e`) with parity tests passing (voiced f0 PCC ≥ 0.95); the DeepUnet cold-JIT cost was measured net-negative for the projected ~5% RTF gain.

**8 ResBlock conv1d shapes use DEFAULT interleaved DRAM** instead of HEIGHT_SHARDED. These are k=11 at channel≥128 shapes; they hit the conv1d kernel's shape-bound 1.92 MB per-core circular-buffer requirement (halo + multi-buffer arithmetic). A real TTNN platform constraint, documented as a finding.

## File structure

```
models/demos/rvc/
├── demo.py                    End-to-end inference with timing
├── benchmark.py               RTF + audio PCC harness (no-fallback, --overlap, --batch)
├── evaluate.py                Audio PCC, speaker similarity, WER
├── assets-download.sh         Model weight download helper
│
├── torch_impl/                PyTorch reference implementations
│   ├── reference.py
│   ├── rmvpe.py
│   └── vc/{hubert,synthesizer}.py
│
├── tt/                        TTNN implementations
│   ├── runtime.py             TTNNFlowDecoder, TTNNGeneratorNSF
│   ├── utils.py
│   └── ops/conv_transpose1d.py
│
├── tests/                     Device-running PCC tests
│   ├── test_runtime.py
│   ├── test_production_shapes.py
│   └── test_ttnn_ops.py
│
└── utils/                     Audio loading, config, F0 enum
```

## Design notes

1. **Persistent weight architecture** — Weights preprocessed and uploaded to device once at `from_checkpoint`, reused across forward calls.
2. **Chunked inference with overlap-add** — 75-frame chunks (L1-safe max), 3-frame overlap.
3. **Uniform chunk padding** — Last chunk zero-padded to standard shape so `ttnn.conv1d` JIT does not compile new kernels mid-run.
4. **Native conv1d bias** — Bias passed via `conv1d(bias_tensor=)`, eliminating host-side post-add.
5. **`prepare_conv_weights` cache** — Per-shape weight tilization hoisted out of forward.
6. **Device-resident ResBlock inner loop** — `from_torch` once at entry, `to_torch` once at exit; intermediate activations stay on device.
7. **ProcessPool (spawn ctx) for batched preprocess** — Bypasses RMVPE's GIL-bound NMS; one worker init per process (HuBERT + RMVPE + TextEncoder + SineGen + emb_g); pool forked before TTNN device open so worker procs do not inherit device file descriptors.
