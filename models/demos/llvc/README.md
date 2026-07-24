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
| Cached conv prenet | 12 gated (`tanh·sigmoid`) residual `Conv1d(1,1,k=3)` with ring buffers | `ops.mac_causal_conv1d` (shifted matmul-accumulate) per block |
| Input conv | `Conv1d(1, enc_dim, k=3L, stride=L)` + ReLU | `ttnn.conv1d` with fused ReLU |
| Dilated causal encoder | 8× depthwise-separable conv (`groups=enc_dim`, dilation 2ⁱ) + LN + ReLU, residual | depthwise `ttnn.conv1d(dilation=2ⁱ)` + `ttnn.layer_norm`; pointwise 1×1 as `ttnn.linear` |
| e2d / d2e projections | grouped `Conv1d(k=1, groups=dec_dim)` + ReLU | `ttnn.conv1d(kernel=1, groups=…)` |
| Causal transformer decoder | `nn.TransformerDecoderLayer` over unfolded chunks | `ttnn.matmul`/`ttnn.softmax` SDPA + `ttnn.linear` FFN, windowed via slices |
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
`LLVCModel.stream(waveform, chunk_factor=1)` returns `(audio, rtf, latency_ms)`
and `LLVCModel(waveform)` does non-streaming conversion.

`stream()` captures `forward_chunk` as a device **trace** and replays it per
chunk (this is what removes the per-chunk host-dispatch overhead — see below).
Set `LLVCConfig(use_trace=False)` to fall back to the eager per-chunk path.

## Profiling (perf sheet)

Follow the [TT-NN model bring-up report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md#41-performance-sheet):

```bash
./tools/tracy/profile_this.py -n llvc \
  -c "pytest models/demos/llvc/tests/perf/test_perf.py::TestLLVCPerformance::test_summary"
```

## Targets and measured results (N300, `wormhole_b0`)

| Metric | Target | Measured (full-size, `chunk_factor=2`) | Where checked |
|---|---|---|---|
| Streaming RTF | < 0.3 | 0.217 | `tests/perf/test_perf.py` |
| Per-chunk latency | < 100 ms | 33.6 ms | `tests/perf/test_perf.py` |
| Accuracy vs PyTorch | PCC > 0.90 | 0.9997 | `tests/pcc/test_llvc.py` |

Full-size streaming RTF (real KoeAI weights, trace): 0.404 at `chunk_factor=1`,
**0.217** at `chunk_factor=2`. Eager (no trace) was 2.77 — trace gives ~7× by
removing per-chunk host dispatch, with identical numerics.

## Notes, limitations, and optimization roadmap

- **Streaming path** assumes the per-chunk encoder-frame count is a multiple of
  `dec_chunk_size` (guaranteed by `LLVCModel.stream`). The decoder unfold is done
  with slices; for `chunk_factor=1` there is exactly one attention window.
- **Device trace (implemented)**: the per-chunk cost was host dispatch, not
  device math. `LLVCState` holds persistent ring buffers updated in place with
  `ttnn.copy`, so `forward_chunk` (a fixed shape across the streaming loop) is
  captured once and replayed via `ttnn.execute_trace`. Conv weights *and* biases
  are cached on device after the warmup chunk so capture does no host→device
  writes. This is the change that meets the RTF target.
- **Further opportunities** (not required to hit target; would bring
  `chunk_factor=1` under 0.3 too): fuse the encoder LN + ReLU, keep encoder
  activations sharded in L1 across layers, fold the output transpose-conv, and
  2-CQ double-buffering to overlap the input upload with compute.
- The cached-conv prenet uses per-tap matmul-accumulate (exact vs the reference
  ring buffers); for `enc_dim`-wide depthwise convs `ttnn.conv1d(groups=…)` is
  used instead.
