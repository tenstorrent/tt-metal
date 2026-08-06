# Model performance and accuracy

Performance and accuracy numbers for CosyVoice-300M, collected from direct pytest runs in
`models/demos/cosyvoice/tests/`. This is the single source for both — every figure below was
measured on the hardware named in *Environment*, none is an estimate, and none is `xfail`-ed. Where
a target is missed the number is stated with the reason and the identified lever.

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

**The per-token tail outside the traced step is 0.352 ms — 2.7 %** — and its breakdown is what
settled P4's on-device sampling item:

| | ms | share of tail |
|---|---:|---:|
| output head matmul | `0.043` | 12 % |
| logits device → host | `0.142` | 40 % |
| RAS sampling on host | `0.075` | 21 % |
| embedding row → device | `0.092` | 26 % |

`ttnn.sampling` could remove at most the middle two — 0.217 ms, **1.7 % of a token** — and not even
all of it, since RAS's repetition branch needs the emitted-token history and returns to the host
whenever it fires. It would also give up exact agreement with the reference on two counts (`≤ p` vs
CosyVoice's inclusion of the crossing token, and its own RNG seed). So sampling stays on the host and
`nucleus_filter` was made fast instead: `0.245 → 0.075 ms`, bit-identical, verified against a literal
transcription of upstream's loop.

Tracing the flow decoder took removing a host→device write that **every convolution** was issuing.
`ttnn.conv1d` and `ttnn.conv_transpose2d` prepare their weights — tilize, pad to the sharding
scheme, move to device — on *every call*, which a trace cannot contain; a host-resident weight fails
capture at `fd_mesh_command_queue.cpp:762` and a device-resident one at `:809`, on the read back.
`ttnn.prepare_conv_weights` hoists the transform out and both wrappers cache the result per input
geometry. Output is bit-identical. **It was a software limit, not a silicon one** — worth stating
plainly because the first reading of `:762` was that convolutions cannot be traced on this stack,
and that reading would have written off both remaining stages.

### Why the KV cache is fixed-width — 73× on the only pass that counts

A KV cache that grows one slot per token gives **every step a new attention key size** — 210, 211,
212 — and TTNN's program cache is keyed on tensor shape, so every token paid a fresh JIT compile.

| | mean/step | tok/s |
|---|---:|---:|
| growing cache, cold (what a real utterance gets) | `2595.34 ms` | `0.4` |
| growing cache, warm (a second pass over the same sizes) | `28.32 ms` | `35.3` |
| **fixed-width cache, first pass** | **`34.10 ms`** | **`29.3`** |

First and last of 32 steps: `29.08 → 3299.99 ms` cold, `28.41 → 28.53 ms` warm. The warm pass is
dead flat, so the arithmetic cost of a longer cache is negligible — **98.9 % of the cold cost was
compilation.** `forward_chunk_fixed()` holds the key width at `max_len`, leaving two shapes for a
whole utterance; `cache_width()` rounds to a multiple of 128 so a handful of buckets covers every
request. The `34.10` against the warm `28.32` is the trade: attention over 256 slots, not 210–241.

**The live tokens sit at the *end* of the buffer.** ESPnet's `rel_shift` skews the score block
assuming the queries are the last `t1` of the `K` key positions. Left-aligning gives every query the
relative geometry of a position it is not at — wrong everywhere, obviously wrong nowhere, and no
shape assertion catches it. `test_device_fixed_shape_cache_matches_the_growing_one` does.

### Vocoder op costs

| op | shape | latency |
|---|---|---:|
| iSTFT | 18 049 frames → 72 192 samples (3.27 s of audio) | **`1.115 ms`** |
| iSTFT | 1 024 frames → 4 092 samples | `0.853 ms` |
| `ConvTranspose1d` | 512→256, k=16, s=8, L=282 (`ups[0]`) | **`3.886 ms`** |

**The inverse transform is cheap and `conv_transpose2d` at `H=1` is not.** The whole iSTFT costs
1.115 ms for 3.27 s of audio — an RTF contribution of 0.00034 — while a single upsample layer costs
3.886 ms, **3.5× the entire iSTFT**, and there are two of them. The op that was supposed to be the
hard part is negligible; the op standing in for a missing `ttnn.conv_transpose1d` dominates the
stage.

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

### Two levers that mattered more than expected

**High math fidelity belongs on the matmuls, not only the convolutions.** `MathFidelity.HiFi4` with
`fp32_dest_acc_en` on `ttnn.linear`/`ttnn.matmul` moved the CFM estimator's *last* Euler step from
`0.9986` to `0.9992` — a 41 % error reduction — and its first from `0.9998` to `0.9998`. The gap
widens with `t` because later steps carry larger activations, and it compounds because the ten
evaluations are *integrated*. **Depth, not op type, decides fidelity.**

**`ttnn.cumsum` is 2000× less accurate than torch's**, against an fp64 reference over the real
72 192-sample f0:

```
device cumsum, fp32    max|d| 5.62e-01    (t=1k 2.3e-07, t=36k 0.114)
torch  cumsum, fp32    max|d| 2.44e-04
```

Phase is `2π·(cumsum mod 1)`, so 0.56 absolute is **more than half a cycle** — the harmonic bank is
randomised by the end of the utterance. At T=1024 and T=8192 the error is ~1e-5, which is why this
passed every module-level test and surfaced only end to end. `phase_mod1()` reduces each block total
mod 1 before accumulating, keeping every partial sum O(1): `0.843 → 0.99999745`.

### Why the waveform gate injects the reference excitation

f0 error **integrates** into excitation phase. Drift is `Σ(Δf0)/sr` over samples, so holding it under
a tenth of a cycle across 72 192 samples needs a mean f0 error below **0.03 Hz** — 1.5e-4 relative at
200 Hz, about 13 mantissa bits. The device's f0 predictor lands at ~16 Hz max error even with fp32
weights *and* activations, because Tensix HiFi4 is four bfloat16 passes rather than true fp32.

This is a property of the model, not a defect to fix. **Sample-level waveform comparison is only
meaningful with the reference excitation injected**, which is what the PCC gate does; for a
self-computed excitation the honest metrics are the energy envelope and the RMS, and those hold
(`0.9975` envelope, RMS within 6 %). Same discipline the CFM's initial noise and `SineGen`'s two
draws already follow.

## Speech quality — 5 languages, 2 modes

Scored with whisper `large-v3`; CER for CJK, WER for English.

| Mode | zh | en | ja | ko | yue |
|---|---:|---:|---:|---:|---:|
| zero-shot | `3.03` | `0.00` | `5.56` | `3.12` | `64.52` |
| cross-lingual | `6.06` | `0.00` | `2.78` | `0.00` | `100.00` |

Cantonese is a **model** limitation, not a port defect: the PyTorch reference scores *worse* on the
same text through the same ASR (`83.87 %` zero-shot vs this port's `64.52 %`).

## Operational notes

**`l1_small_size` scales with conv *configurations*, not tensor size.** `ttnn.conv1d` allocates
prepared weights from that bank and keeps them, so three models live at once (~80 convs) exhausts
the 32 KB a single-model test uses — failing part-way through the *second* utterance with
`Not enough space to allocate 480 B L1_SMALL buffer`. Zero-shot needs 128 KB; cross-lingual needs
256 KB, because its prompt is 1289 mel frames against 326.

**Persist the JIT cache across runs.** Mounting `/root/.cache/tt-metal-cache` took the first
utterance from `161.7 s` to `14.8 s` wall. Every distinct sequence length is a fresh compile.

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
| host | 106 | none |
| device | 41 | Blackhole `p150a` |
