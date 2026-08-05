# Model performance and accuracy

Performance and accuracy numbers for CosyVoice-300M, collected from direct pytest runs in
`models/demos/cosyvoice/tests/`. Full derivations and the reasoning behind each figure are in
[`docs/perf.md`](docs/perf.md).

## Environment
- Device: Blackhole `p150a`
- Host: 16 cores, 62 GB
- tt-metal: `b5e9cba196`
- Date: `2026-08-05`

## Benchmark commands
```bash
pytest models/demos/cosyvoice/tests/perf/test_pipeline_perf.py -v -s   # end-to-end RTF
pytest models/demos/cosyvoice/tests/perf/test_trace.py          -v -s   # trace speedup
pytest models/demos/cosyvoice/tests/perf/test_llm_perf.py       -v -s   # decode throughput
```

## Summary metrics

Measured on the captured utterance: 164 generated tokens producing 3.27 s of audio at 22 050 Hz.

| Metric | Value | Target |
|---|---:|---:|
| End-to-end RTF | `1.096` | `< 0.5` ❌ |
| LLM throughput (traced) | `65.8 tok/s` | `>= 60` ✅ |
| LLM decode latency (traced) | `15.19 ms` | — |
| Token agreement, teacher-forced | `98.56 %` | `> 95 %` ✅ |
| Token agreement, through the KV cache | `95.83 %` | `> 95 %` ✅ |
| WER (English) | `0.00 %` | `< 3.0` ✅ |
| Speaker similarity (mean, 10 utterances) | `83–96` | `> 60` ✅ |
| Streaming vs non-streamed, mel-space PCC | `0.9019` | content-equal ✅ |
| tokens → waveform PCC | `0.9951` | `>= 0.99` ✅ |

### RTF breakdown

| Stage | Cost | RTF | Share |
|---|---:|---:|---:|
| LLM (14-block AR decoder, traced) | `15.19 ms/token × 164` | `0.761` | 69 % |
| Flow decoder (10 Euler steps, traced) | `1.049 s` | `0.320` | 29 % |
| HiFT vocoder | `0.048 s` | `0.015` | 1 % |
| **Total** | `3.588 s` | **`1.096`** | |

**RTF misses its target, and both traced stages show why.** Trace capture is worth **2.22×** on the
AR decoder (34.92 → 15.71 ms/token) but only **1.09×** on the flow decoder (1.151 → 1.053 s). That
gap is the finding: tracing buys back *dispatch* overhead, so it pays in proportion to how
dispatch-bound a stage already is. The AR decoder issues ~14 small ops per token at batch 1 and is
almost pure overhead; the flow decoder runs 16 resnet and 64 transformer blocks over 608 frames at
batch 2 and is close to compute-bound. End-to-end that took RTF from 2.120 to 1.123, and the op-count work below to 1.096.

Reaching 0.5 therefore needs a shorter critical path per token, and two measurements narrow what
that means.

**`bfloat8_b` weights give 1.00×** (13.09 → 13.12 ms/step) at PCC `0.9997040033`. The decoder's
176 M linear parameters move 352 MB per token, which at 13.09 ms is **27 GB/s effective** — roughly
15× short of the bandwidth floor. The step is nowhere near bandwidth-bound, so halving the traffic
is invisible. Kept as a memory option (352 MB → 176 MB), not a speed one.

**What it is bound by is per-op cost on one-row tensors**, and counting them
(`scripts/count_decode_ops.py`) says which:

| op | count | share | | op | count | share |
|---|---:|---:|---|---|---:|---:|
| `linear` | 99 | 9.6 % | | `concat` | 42 | 4.1 % |
| `reshape` | 98 | 9.5 % | | `matmul` | 42 | 4.1 % |
| `permute` | 98 | 9.5 % | | `layer_norm` | 30 | 2.9 % |
| `add` | 84 | 8.1 % | | `multiply` | 29 | 2.8 % |
| `slice` | 70 | 6.8 % | | `softmax` | 14 | 1.4 % |

**`reshape` and `permute` together are 31 % — more than every `linear` and `matmul` combined**, and
they are pure data movement. Reading the code, the projections look like the bulk; they are 9.6 %.
Acting on it: at `T = 1` the head-split permute is a relabelling, so it is skipped, taking `permute`
from 98 to 42 and the step from 13.09 to 12.52 ms with bit-identical output.

The same list points at what is left — the five-op `rel_shift` skew behind `slice` and `concat`, and
flash attention to collapse the score chain.

Tracing the flow decoder took removing a host→device write that **every convolution** was issuing.
`ttnn.conv1d` and `ttnn.conv_transpose2d` prepare their weights — tilize, pad to the sharding
scheme, move to device — on *every call*, which a trace cannot contain; a host-resident weight fails
capture at `fd_mesh_command_queue.cpp:762` and a device-resident one at `:809`, on the read back.
`ttnn.prepare_conv_weights` hoists the transform out and both wrappers cache the result per input
geometry. Output is bit-identical. **It was a software limit, not a silicon one** — worth stating
plainly because the first reading of `:762` was that convolutions cannot be traced on this stack,
and that reading would have written off both remaining stages.

## Accuracy

| Module | PCC |
|---|---:|
| tokens → waveform (reference excitation) | `0.9951367159` |
| flow: tokens → mel | `0.9992029011` |
| whole HiFT vocoder | `0.9996373743` |
| LLM AR prefill, 209 tokens | `0.9997355989` |
| LLM AR decode step | `0.9986645835` |
| traced vs untraced decode | `1.0000000000` (bit-exact) |
| iSTFT vs captured golden | `0.9999298811` |

## Speech quality — 5 languages, 2 modes

Scored with whisper `large-v3`; CER for CJK, WER for English.

| Mode | zh | en | ja | ko | yue |
|---|---:|---:|---:|---:|---:|
| zero-shot | `3.03` | `0.00` | `5.56` | `3.12` | `64.52` |
| cross-lingual | `6.06` | `0.00` | `2.78` | `0.00` | `100.00` |

Cantonese is a **model** limitation, not a port defect: the PyTorch reference scores *worse* on the
same text through the same ASR (`83.87 %` zero-shot vs this port's `64.52 %`).

## Perf coverage
Source suites: `tests/perf/`, `tests/e2e/`, `tests/pcc/`
- End-to-end RTF with a per-stage breakdown
- Trace capture speedup and bit-exactness
- Decode throughput, cold vs warm, growing vs fixed-shape KV cache
- Streaming content equivalence and seam continuity
- Per-module PCC against captured PyTorch goldens

## Test counts

| Tier | Count | Hardware |
|---|---:|---|
| host | 85 | none |
| device | 41 | Blackhole `p150a` |
