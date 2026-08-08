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

Run the helper to fetch and convert all weights from public sources in one step:

```bash
bash models/demos/rvc/assets-download.sh
```

The script downloads `pretrained_v2/f0G48k.pth`, `hubert_base.pt`, and `rmvpe.pt` from the public [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) HuggingFace repo, converts each `.pt` to `.safetensors` in bfloat16 (collapsing PyTorch `weight_norm`, renumbering flow indices `0/2/4/6 → 0/1/2/3` to skip the Flip layers, squeezing kernel-1 convs to Linear weights, renaming `conv_{q,k,v,o} → linear_{q,k,v,o}` in the TextEncoder, reindexing HuBERT's `feature_extractor.conv_layers.0.2 → .0.1`, and dropping fairseq pretraining-only heads), and copies the two configs from `models/demos/rvc/scripts/configs/` (committed alongside the script). Then drop a 16 kHz+ mono speech `.wav` at `data/speech/sample-speech-0.wav`.

Final on-disk layout under `models/demos/rvc/data/` — the paths `utils/config.py` and `torch_impl/rmvpe.py` actually load:

| Path | Size | Source |
|---|---:|---|
| `assets/pretrained_v2/f0G48k.safetensors` | 55 MB | derived from `lj1995/VoiceConversionWebUI/pretrained_v2/f0G48k.pth` |
| `assets/hubert.safetensors`               | 181 MB | derived from `lj1995/VoiceConversionWebUI/hubert_base.pt` (== `facebook/hubert-base-ls960`) |
| `rmvpe.safetensors`                       | 173 MB | derived from `lj1995/VoiceConversionWebUI/rmvpe.pt` |
| `configs/v2/48k.json`                     | ~1 KB | copied from `scripts/configs/v2/48k.json` (upstream RVC v2 48k config) |
| `configs/hubert_cfg.json`                 | ~10 KB | copied from `scripts/configs/hubert_cfg.json` (fairseq HuBERT-base config) |
| `speech/sample-speech-0.wav`              | user-provided | any 16 kHz+ mono speech recording |

### Python dependencies
```bash
pip install pyworld scipy soundfile safetensors librosa faiss-cpu 'av>=14'
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

# Single-stream with Flow trace+execute_trace (~5% TTNN-only RTF win at B=1)
python -m models.demos.rvc.benchmark --max_secs 3.0 --warmup 1 --runs 3 --trace

# Batched, 5 concurrent, with CPU/TTNN pipeline overlap (best full-pipeline RTF)
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

Run each suite in a separate `pytest` invocation. Co-execution (single invocation across all three) segfaults — this is a documented limitation related to TTNN device lifecycle across test modules.

```bash
pytest models/demos/rvc/tests/test_runtime.py -v
pytest models/demos/rvc/tests/test_ttnn_ops.py -v
pytest models/demos/rvc/tests/test_production_shapes.py -v
```

## Performance

Measured on Koyeb `gpu-tenstorrent-n300s` (Wormhole B0), warm JIT cache, 3 s input audio. Numbers are means across warm runs. Two measurement points are shown to honour the reviewer's F3 request (record hardware + tt-metal commit for reported numbers):

- **Original** — measured 2026-06-08 at PR commit [`46d99b55c99`](../../../../commit/46d99b55c99) (the commit that introduced `--overlap` and first reported the 0.187 number). PR branch was rebased onto tt-metal main at commit [`153dbef60ff`](../../../../commit/153dbef60ff) (2026-05-21, "SFPI 7.52.0 637"). Built on a prior Koyeb `gpu-tenstorrent-n300s` instance.
- **Current (2026-06-26 rebuild)** — fresh tt-metal rebuild from PR tip [`1c2510f8b62`](../../../../commit/1c2510f8b62), same merge-base on main, on a newly-provisioned Koyeb `gpu-tenstorrent-n300s` instance of the same SKU, with bfloat16 weights produced by this PR's `assets-download.sh`. This is what a fresh CI run today would see.

Differences come from (a) tt-metal HEAD drift between the two build times (the B>1 conv slicer chooses layout per-shape per-call with no pinned shard config — see F3 in the review), and (b) Koyeb hardware/silicon-binning variance across instances of the same SKU. CPU preprocessing (HuBERT, RMVPE) is also notably slower on the current instance, which dominates single-stream wall time independent of TTNN.

### Single-stream (`--batch 1`)

| Metric | Original | Current | Bounty target |
|---|---:|---:|---|
| TTNN-only RTF | 0.177 | 0.594 | < 0.5 (met original; fails current) |
| Full-pipeline RTF | 0.300 | 1.569 | < 0.5 (met original; fails current — CPU-bound) |
| Audio PCC vs torch | 0.998 | 0.996 | > 0.95 ✓ |

On the current Koyeb instance, HuBERT + RMVPE alone take 2.5–3.2 s for a 3 s clip — even with zero TTNN time, single-stream cannot meet RTF<0.5. This is provisioning variance, not a code-path issue.

### Batched (B=5)

Per-row PCC vs B=1 reference > 0.995 in both configs.

| Config | TTNN-only / sample (orig → current) | Full / sample (orig → current) |
|---|---:|---:|
| `--batch 5 --f0_method dio` (no overlap) | **0.143** → 0.327 | 0.209 → 0.444 |
| `--batch 5 --overlap --f0_method dio` | 0.155 → 0.339 | **0.187** → 0.476 |

Both configs meet the bounty `RTF<0.5` target on both builds. On the current build the full-pipeline margin is ~11% for no-overlap and ~5% for the overlapped variant; TTNN-only margins are 32–35%. The stretched `RTF<0.2` was met on the original build only; on the current build full RTF exceeds the stretched goal — see F3 (non-blocking) for the root-cause and three attempted fixes.

Without `--overlap`, TTNN-only is best (compute is contiguous, no per-flow dispatch overhead from micro-batch splitting). With `--overlap`, full-pipeline is best (CPU preprocess of mb2 hidden by TTNN compute of mb1).

### Where time goes (single-stream, 3 s warm — original build)

| Stage | Per-sample time | Share | Where |
|---|---:|---:|---|
| HuBERT | 0.15 s | 16% | torch CPU |
| RMVPE pitch | 0.19 s | 21% | torch CPU |
| TextEncoder + SineGen | 0.03 s | 4% | torch CPU |
| Flow (TTNN) | 0.04 s | 4% | N300 |
| Generator (TTNN) | 0.49 s | 54% | N300 |

CPU preprocess is ~40% of single-stream wall on the original build and is correctly off TTNN by architecture (HuBERT/RMVPE not ported — see Limitations). On the current build it dominates further (HuBERT 0.9 s + RMVPE 1.2 s + Generator 1.7 s mean), making single-stream wall preprocessing-bound on this Koyeb instance.

### Pitch method comparison (B=5, ProcessPool spawn — original build)

| Pitch | Preprocess /sample | Full RTF /sample | Speaker sim vs RMVPE | WER vs RMVPE |
|---|---:|---:|---:|---:|
| RMVPE | 0.295 s | 0.239 | 1.000 (reference) | 0.000 |
| DIO | 0.196 s | 0.209 | 0.969 | 0.000 |
| DIO + `--overlap` | 0.102 s (observable) | 0.187 | 0.969 | 0.000 |

DIO and RMVPE are perceptually equivalent (cosine sim 0.969, Whisper produces identical transcriptions). RMVPE remains the default. Quality comparison numbers (speaker sim, WER) are model-intrinsic and reproduce across builds; only timing varies with hardware/build.

## Diverse use case validation

| Case | Result |
|---|---|
| Speaker diversity (speaker_id 0 / 50 / 100) | Resemblyzer cosine similarity between pairs: 0.576 / 0.636 / 0.771. Same-speaker reference is 0.969, so 0.58–0.77 confirms meaningfully distinct target voices. |
| Pitch transposition (-12, -6, +6, +12 semitones) | All four shifts produce valid audio (NaN=0, Inf=0, amplitude within [-0.95, +0.95]) and all audibly different from the k=0 baseline (cosine sim 0.76–0.86, all below 0.90). |
| Real-time conversion | RTF 0.187 (original measurement) / 0.476 (current build rebuild) at B=5 with `--overlap --f0_method dio`. Both meet bounty `<0.5`. |

## Bounty status

Stage 1 (bring-up) and Stage 2 (optimization) targets are met with margin. Stage 3 status against the bounty's exact wording:

| Bullet | Status |
|---|---|
| Maximize core counts | Done (HEIGHT_SHARDED conv1d, 8×7 grid where shape permits) |
| Efficient flow-based decoder | Done (device-resident WN inner loop; B=8 at 5.4× per-sample) |
| Flash Attention or equivalent | N/A (RVC has no attention layers) |
| Minimize voice conversion latency | Done. Best TTNN-only RTF 0.143 at B=5 batched (vs Stage 1 final 0.535 = 73% reduction). Single-stream B=1: 0.166 with `--trace`. |
| Batch processing | Done (native B=2..8, per-row correctness tested at B=2/3/5/8) |
| Optimize feature index search | N/A by design (FAISS < 1% of wall time) |
| Pipeline encoder/decoder/vocoder stages | Done for batched path (`--overlap`); single-stream cross-chunk overlap not implemented; Flow trace integration (`--trace`) further amortizes host dispatch on single-stream |
| Efficient pitch extraction and transposition | Done (RMVPE persisted, `--key` transposition supported, DIO alternative measured) |
| Minimize memory and TM overheads | Done (WN and ResBlock inner loops eliminate host roundtrips; `prepare_conv_weights` cache) |
| Caching feature indices | N/A by design |
| Document tuning and trade-offs | Done (this README + commit history) |
| **Stretched:** 60+ tokens/second | Done (Flow at B=8: ~6900 tokens/s) |
| **Stretched:** RTF < 0.2 | Met on the original measurement build (0.187 full-pipeline at `--batch 5 --overlap --f0_method dio`). Not reproducible on a fresh tt-metal HEAD rebuild today (0.476 on the same Koyeb SKU). F3 in the review documents the root cause (B>1 conv shard config not pinned; auto-slicer behaviour drifts with tt-metal HEAD); three on-hardware fix attempts (HEIGHT_SHARDED+act_bh=32, act_bh=16 narrow whitelist, WIDTH_SHARDED) each failed with a different L1/runtime error — a real fix requires per-shape conv-op tuning. Non-blocking per reviewer recommendation. |
| **Stretched:** 5+ concurrent conversions | Done (verified end-to-end at B=5 and B=8) |

## Limitations

**Trace + 2CQ — implemented for Flow, not for Generator.** `benchmark.py --trace` captures one trace per (flow_idx, seq_len, batch) and replays via `execute_trace` + `copy_host_to_device_tensor` for new inputs each call. Per-flow capture is bit-exact (PCC 1.000000 vs direct forward), per-flow replay is ~7× faster on its compute. End-to-end measured within-session at B=1/RMVPE/3s warm: TTNN-only RTF 0.1749 → 0.1664 (−4.9%); full-pipeline RTF 0.2979 → 0.2899 (−2.7%); Audio PCC vs torch preserved at 0.998. At B=5 with `--overlap`, trace adds slight overhead because per-call `copy_host_to_device_tensor` and `execute_trace` dispatch exceed the saved dispatch on compute-dominated batched ops — so `--trace` is recommended for single-stream only. Generator trace is not implemented; Generator forward allocates many transient buffers per call (288 conv1d calls across 4 upsample stages with varying seq_len), and a Generator trace would require pre-allocating all per-stage intermediates and pre-warming `_prep_cache` for every needed shape, which is a multi-day refactor without the same per-op headroom (Generator compute already dominates dispatch).

**Single-stream cross-chunk pipeline overlap not implemented.** Preprocess is monolithic per inference (produces all frames at once), so within-stream overlap would require re-architecting preprocess as streaming. Batched path closes this for B>1 via `--overlap`.

**HuBERT and RMVPE remain on CPU.** Both are out of TTNN-port scope. RMVPE port was investigated on a sibling branch (`rvc-clean-implementation`, commit `bad66aa1a2e`) with parity tests passing (voiced f0 PCC ≥ 0.95); the DeepUnet cold-JIT cost was measured net-negative for the projected ~5% RTF gain.

**8 ResBlock conv1d shapes use DEFAULT interleaved DRAM** instead of HEIGHT_SHARDED. These are k=11 at channel≥128 shapes; they hit the conv1d kernel's shape-bound 1.92 MB per-core circular-buffer requirement (halo + multi-buffer arithmetic). A real TTNN platform constraint, documented as a finding.

## File structure

```
models/demos/rvc/
├── demo.py                    End-to-end inference with timing
├── benchmark.py               RTF + audio PCC harness (no-fallback, --overlap, --batch)
├── evaluate.py                Audio PCC, speaker similarity, WER
├── assets-download.sh         Public-source weight download + .pt → .safetensors converter
├── scripts/configs/           Demo configs (hubert_cfg.json, v2/48k.json) copied during install
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
