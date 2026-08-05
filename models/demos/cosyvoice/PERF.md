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
| End-to-end RTF | `1.163` | `< 0.5` ❌ |
| LLM throughput (traced) | `63.6 tok/s` | `>= 60` ✅ |
| LLM decode latency (traced) | `15.71 ms` | — |
| Token agreement, teacher-forced | `98.56 %` | `> 95 %` ✅ |
| Token agreement, through the KV cache | `95.83 %` | `> 95 %` ✅ |
| WER (English) | `0.00 %` | `< 3.0` ✅ |
| Speaker similarity (mean, 10 utterances) | `83–96` | `> 60` ✅ |
| Streaming vs non-streamed, mel-space PCC | `0.9019` | content-equal ✅ |
| tokens → waveform PCC | `0.9951` | `>= 0.99` ✅ |

### RTF breakdown

| Stage | Cost | RTF | Share |
|---|---:|---:|---:|
| LLM (14-block AR decoder, traced) | `15.71 ms/token × 164` | `0.787` | 68 % |
| Flow decoder (10 Euler steps) | `1.151 s` | `0.352` | 30 % |
| HiFT vocoder | `0.079 s` | `0.024` | 2 % |
| **Total** | `3.807 s` | **`1.163`** | |

**RTF misses its target and the reason is specific.** Trace capture took the LLM from 34.92 to
15.71 ms/token (2.22×) and end-to-end RTF from 2.120 to 1.163. The same lever cannot be applied to
the flow decoder or the vocoder: **`ttnn.conv1d` is not trace-compatible in this build** — a bare
`conv1d` fails capture with a host-resident weight (`fd_mesh_command_queue.cpp:762`) *and* with a
device-resident one (`:809`). The AR decoder traced cleanly precisely because it contains no
convolutions; the estimator has ~37 and the vocoder ~40.

## Accuracy

| Module | PCC |
|---|---:|
| tokens → waveform (reference excitation) | `0.9951367159` |
| flow: tokens → mel | `0.9992029011` |
| whole HiFT vocoder | `0.9996373743` |
| LLM AR prefill, 209 tokens | `0.9997355989` |
| LLM AR decode step | `0.9994433945` |
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
| device | 37 | Blackhole `p150a` |
