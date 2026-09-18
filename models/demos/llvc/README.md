# LLVC (Low-Latency Low-Resource Voice Conversion) on TT-NN

TTNN bring-up of KoeAI's [LLVC](https://github.com/KoeAI/LLVC) real-time
any-to-one voice-conversion generator
([paper, arXiv:2311.00873](https://arxiv.org/abs/2311.00873)) for Tenstorrent
Wormhole / Blackhole (N150 / N300).

LLVC is a **waveform-to-waveform** converter: unlike RVC/QuickVC it has no
separate neural vocoder. A dilated-causal-conv encoder produces latent frames, a
tiny causal transformer decoder predicts a multiplicative mask, and a strided
`ConvTranspose1d` + `tanh` synthesises the 16 kHz output directly. All
convolutions are causal and keep per-layer ring buffers, so the model streams
chunk-by-chunk with sub-100 ms latency.

## Layout

```
models/demos/llvc/
├── reference/llvc_reference.py   # self-contained PyTorch reference (KoeAI-weight compatible)
├── tt/
│   ├── config.py                 # LLVCConfig + TTNN dtype/memory helpers
│   ├── ops.py                    # TTNN op helpers (causal conv, SDPA, layernorm, transpose-conv)
│   ├── model.py                  # TTNN LLVCModel + streaming state + create_llvc()
│   └── state_io.py               # KoeAI checkpoint / config.json loading
├── demo/demo.py                  # streaming + non-streaming demo, RTF/latency report
├── tests/pcc/test_llvc.py        # TTNN-vs-reference PCC + streaming equivalence
├── tests/perf/test_perf.py       # RTF / chunk-latency targets
└── conftest.py                   # device fixture
```

## Architecture mapping (reference → TTNN)

| LLVC block | Reference (`[B, C, T]`) | TTNN (`[B, T, C]`) |
|---|---|---|
| Cached conv prenet | 12 gated (`tanh·sigmoid`) residual `Conv1d(1,1,k=3)` with ring buffers | `ops.causal_window` + `ops.apply_taps` (shifted matmul-accumulate) per block |
| Input conv | `Conv1d(1, enc_dim, k=3L, stride=L)` + ReLU | `ttnn.conv1d` with fused ReLU |
| Dilated causal encoder | 8× depthwise-separable conv (`groups=enc_dim`, dilation 2ⁱ) + LN + ReLU, residual | depthwise `ttnn.conv1d(dilation=2ⁱ)` + `ttnn.layer_norm`; pointwise 1×1 as `ttnn.linear` |
| e2d / d2e projections | grouped `Conv1d(k=1, groups=dec_dim)` + ReLU | `ttnn.conv1d(kernel=1, groups=…)` |
| Causal transformer decoder | `nn.TransformerDecoderLayer` over unfolded chunks | `ttnn.transformer.scaled_dot_product_attention` + `ttnn.linear` FFN, windowed via slices |
| Output synthesis ("vocoder") | `ConvTranspose1d(enc_dim,1,k=(out_buf_len+1)L,stride=L)` + `tanh` | `ttnn.conv_transpose2d` (singleton height) + `ttnn.tanh` |

The label embedding takes a constant zero label, so its output is precomputed on
host once and uploaded as a constant — no MLP runs on device.

## Setup

The tt-metal environment provides `torch` and `ttnn`. Install the small extra
audio deps:

```bash
pip install -r models/demos/llvc/requirements.txt
```

To use the official pretrained weights, fetch them with KoeAI's downloader
(`python download_models.py` in the LLVC repo) and point the demo at
`experiments/llvc/config.json` + `llvc_models/models/checkpoints/llvc/G_500000.pth`.

## Running

Smoke run on a synthetic tone (random weights, no checkpoint):

```bash
python models/demos/llvc/demo/demo.py --synthetic --stream
```

Real conversion (streaming), file or folder in / out:

```bash
python models/demos/llvc/demo/demo.py \
  --config experiments/llvc/config.json \
  --checkpoint llvc_models/models/checkpoints/llvc/G_500000.pth \
  --input test_wavs --out-dir converted_out --stream --chunk-factor 1
```

Non-streaming (full-context) conversion: drop `--stream`.

`--chunk-factor 2` on the full-size checkpoint is the recommended real-time
setting (RTF 0.217, ~34 ms latency on N300); `--chunk-factor 1` gives the lowest
latency (~20 ms) at RTF 0.404.

## Tests

```bash
# Correctness: TTNN vs PyTorch reference (shared weights) + streaming equivalence
pytest models/demos/llvc/tests/pcc/test_llvc.py -v -s

# Performance: streaming RTF / per-chunk latency
pytest models/demos/llvc/tests/perf/test_perf.py -v -s
```

`create_llvc(config, device=..., checkpoint_path=...)` is the entry point;
`LLVCModel.stream(waveform, chunk_factor=1)` returns `(audio, StreamMetrics)`
and `LLVCModel(waveform)` does non-streaming conversion. Both accept `[T]`,
`[B, T]`, or `[B, 1, T]`: `B > 1` is concurrent streams (independent ring
buffers per row). PCC tests assert streaming vs offline equivalence and
batched vs sequential streams.

`StreamMetrics.rtf` / `.latency_ms` are **end-to-end**: each timed chunk includes
host→device input upload and device→host output download, which a real streaming
deployment pays every chunk. `.device_rtf` / `.device_latency_ms` isolate device
execution only (after upload, before download) for comparison. Perf tests assert
the Stage-1 targets against the e2e figures.

`stream()` captures `forward_chunk` as a device **trace** and replays it per
chunk (this is what removes the per-chunk host-dispatch overhead — see below).
Set `LLVCConfig(use_trace=False)` to fall back to the eager per-chunk path.

## Profiling (perf sheet)

Follow the [TT-NN model bring-up report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md#41-performance-sheet):

```bash
./tools/tracy/profile_this.py -n llvc \
  -c "pytest models/demos/llvc/tests/perf/test_perf.py::TestLLVCPerformance::test_summary"
```

## Evaluation (quality + performance metrics)

Bounty Stage-1 gates that need a device box plus the official KoeAI checkpoint:

| Metric | Target | How to measure |
|---|---|---|
| Content preservation (WER) | **< 3.0%** | Whisper `small.en` on source vs converted (normalized text) |
| Speaker similarity | **> 0.70** cosine | Resemblyzer vs **target speaker 8312** (`--target-ref` required) |
| Token-level accuracy | **> 95%** voiced frames PCC > 0.9 | convert stage vs the PyTorch reference (same `nhead`) |

The earlier **10.9% WER** used Whisper `base.en` on the 10 smoke `test_wavs`
clips and is **not** the bounty number. The measured bounty number below uses
LibriSpeech `test-clean` (the paper's eval set): short isolated clips inflate
Whisper's WER (proper nouns, single-word slips) even when conversion is clean —
the same pipeline scores 6.98% on the 10 `test_wavs` clips vs 1.23% on 50
`test-clean` utterances.

**Agreed eval set (KoeAI paper / issue #32187):**

1. Source audio: KoeAI `test_wavs`, or LibriSpeech `test-clean` (paper eval).
2. Target-speaker reference: LibriSpeech **speaker 8312** (KoeAI `f_8312`).
   After unzipping `train-clean-100.tar.gz`: `LibriSpeech/train-clean-100/8312`
   (speaker 8312, Jaimie Noy, is in the train-clean-100 subset).
3. Config / weights: `experiments/llvc/config.json` +
   `llvc_models/models/checkpoints/llvc/G_500000.pth`.

Device stage (tt-metal env):

```bash
python models/demos/llvc/eval/evaluate.py --stage convert \
  --config experiments/llvc/config.json \
  --checkpoint llvc_models/models/checkpoints/llvc/G_500000.pth \
  --input test_wavs \
  --out-dir llvc_eval_out --chunk-factor 2
```

Offline metrics (separate venv; Whisper must not share the ttnn torch):

```bash
pip install openai-whisper jiwer resemblyzer librosa onnxruntime requests soundfile 'torchmetrics[audio]'
python models/demos/llvc/eval/evaluate.py --stage metrics \
  --out-dir llvc_eval_out \
  --target-ref LibriSpeech/train-clean-100/8312 \
  --whisper-model small.en
```

Attach `llvc_eval_out/eval_report.md` and `eval_report.json` to the PR.

| Metric | How it's computed | Needs |
|---|---|---|
| Decoder throughput (latent frames/s) | `sample_rate / (L · RTF)` vs the `sample_rate / L` real-time rate | always (pure torch) |
| Token-level accuracy vs reference | per-frame PCC (hop = `L`) of non-streaming TTNN vs the PyTorch reference, plus % of frames > 0.9 | always (pure torch) |
| Content preservation (WER) | Whisper transcribes source vs converted; `jiwer` WER between them | `openai-whisper`, `jiwer` |
| Speaker similarity to target | cosine of converted vs `--target-ref` (speaker 8312) | `resemblyzer`, `--target-ref` |
| Audio quality (DNSMOS) | non-intrusive MOS of the converted speech | `torchmetrics[audio]`, `onnxruntime`, `librosa` |

The three external-model metrics are imported lazily; if a dependency is missing
the harness logs a skip line and still reports the pure-torch metrics.

## Targets and measured results (N150, `wormhole_b0`)

Measured on a Tenstorrent cloud N150 with the official KoeAI checkpoint
(`G_500000.pth`, `chunk_factor=2`, trace enabled). WER / speaker similarity /
DNSMOS from 50 LibriSpeech `test-clean` utterances (`--limit 50`), scored with
Whisper `small.en` and Resemblyzer vs speaker 8312 (106 reference files).

| Metric | Target | Measured | Where checked |
|---|---|---|---|
| Streaming e2e RTF | < 0.3 | **0.220** | `eval/evaluate.py` (test_wavs: 0.220) |
| Per-chunk e2e latency | < 100 ms | **33.7 ms** | `eval/evaluate.py` |
| Accuracy vs PyTorch | PCC > 0.90 | **0.9993** global; 99.55% voiced frames > 0.9 | `eval/evaluate.py`, `tests/pcc/test_llvc.py` |
| Content preservation | WER < 3.0% | **1.23%** (test-clean, 50 files) | `eval/evaluate.py --stage metrics` |
| Speaker similarity to target | cosine > 0.70 | **0.904 mean / 0.812 min** | `eval/evaluate.py --stage metrics` |
| Audio quality (DNSMOS OVRL) | — | 3.33 / 5 | `eval/evaluate.py --stage metrics` |

Small-config perf gates (`tests/perf/test_perf.py`, e2e incl. H2D/D2H):
`chunk_factor=1` RTF 0.156 / 17.0 ms, `chunk_factor=2` RTF 0.086 / 30.3 ms,
`chunk_factor=4` RTF 0.049 / 56.6 ms — all under the RTF < 0.3 and < 100 ms
targets. Eager (no trace) full-size RTF was 2.77 — trace gives ~13× by removing
per-chunk host dispatch, with identical numerics.

## Notes, limitations, and optimization roadmap

- **Streaming path** assumes the per-chunk encoder-frame count is a multiple of
  `dec_chunk_size` (guaranteed by `LLVCModel.stream`). The decoder unfold is done
  with slices; for `chunk_factor=1` there is exactly one attention window.
- **RTF transfer-cost caveat**: published Stage-1 numbers and the perf-test gate
  use e2e RTF/latency (upload + execute + download per chunk). Device-only RTF
  will look better because it omits H2D/D2H; both are reported by `stream()` /
  the demo so the gap is visible. Overlapping the next chunk's upload with
  compute (2-CQ) would shrink that gap further.
- **Device trace (implemented)**: the per-chunk cost was host dispatch, not
  device math. `LLVCState` holds persistent ring buffers updated in place with
  `ttnn.copy`, so `forward_chunk` (a fixed shape across the streaming loop) is
  captured once and replayed via `ttnn.execute_trace`. Conv weights *and* biases
  are cached on device after the warmup chunk so capture does no host→device
  weight writes. This is the change that meets the RTF target.
- **Trace region**: `LLVC_TRACE_REGION_SIZE` (`tt/config.py`) is sized from the
  full-model `forward_chunk` capture footprint on N300 (~22.8 MiB); demo, eval,
  and tests share that constant.
- **Concurrent streams (Stage 3, implemented):** `stream()` / `__call__` accept
  batch `B > 1` (`[B, T]` or `[B, 1, T]`). Each row is an independent stream
  with its own ring buffers. `tests/pcc/test_llvc.py::test_batched_streams_match_sequential`
  checks batched output against sequential `B=1`. This is **not** a claim of
  10+ simultaneous real-time sessions or a pipelined encoder/decoder/vocoder
  schedule; those stretched Stage-3 throughput numbers are out of scope.
- **Further opportunities** (not required to hit target; would bring
  `chunk_factor=1` under 0.3 too): fuse the encoder LN + ReLU, keep encoder
  activations sharded in L1 across layers, fold the output transpose-conv, and
  2-CQ double-buffering to overlap the input upload with compute.
- The cached-conv prenet uses per-tap matmul-accumulate (exact vs the reference
  ring buffers); for `enc_dim`-wide depthwise convs a shifted per-channel MAC is
  used instead (`ttnn.conv1d` cannot shard these depthwise layers).
