# CosyVoice-300M on Tenstorrent — measured performance

All figures from **Blackhole p150a** silicon, 2026-08-05, `tt-metal @ b5e9cba196`.
Reproduce with the commands in [`../README.md`](../README.md).

Nothing here is an estimate and nothing is `xfail`-ed. Where a target is missed, the
number is stated with the reason and the identified lever.

---

## End-to-end real-time factor

RTF is compute-seconds per second of audio. Measured on the captured utterance —
164 generated tokens producing **3.27 s** of audio at 22 050 Hz.

| stage | cost | RTF | share |
|---|---|---|---|
| **LLM** (14-block AR decoder) | 34.66 ms/token × 164 | **1.736** | **81.9 %** |
| flow decoder (10 Euler steps) | 1.183 s | 0.361 | 17.0 % |
| HiFT vocoder (mel → 72 192 samples) | 0.074 s | 0.023 | 1.1 % |
| **total** | **6.941 s** | **2.120** | |

**Target: RTF < 0.5 (P5), < 0.2 (P6). Not met — measured 2.120.**

The split is the useful part. The vocoder — the stage the previous attempt left on
the host, and the one this bring-up treated as the hard problem because TTNN has no
FFT — is **1.1 %** of runtime. The LLM is 82 %, because it is the only stage whose
cost scales with output length rather than being amortised over it: one forward
pass per token, and a second of speech is 50 tokens.

At roughly 280 ops per decode step and 34.66 ms, that is ~124 µs/op — **dispatch
bound, not compute bound**.

### The identified lever

**Trace capture.** `ttnn.begin_trace_capture` / `execute_trace` records the op graph
once and replays it with a single host command, which is exactly the right medicine
for a per-op-dispatch bottleneck (4× on a comparable model).

It requires an **in-place KV cache** first, and that is a real piece of work rather
than a flag: a trace replays fixed device addresses, so the cache cannot be
reallocated by `concat` on every step. It has to be a persistent buffer written with
`ttnn.copy` at the end of the traced body, and the per-step mask has to become a
preallocated tensor updated by `copy_host_to_device_tensor` rather than a fresh
upload. The right-alignment constraint below still applies to that design.

Secondary levers, in expected order of value: `bfloat8_b` weights for the AR
decoder (halves weight bandwidth); flash attention via the additive-`attn_mask`
identity already proven for rel-pos attention; sharding the flow decoder's UNet.

---

## LLM decode throughput, and why the cache shape matters

A KV cache that grows by one slot per token gives **every step a new attention key
size** — 210, 211, 212 — and TTNN's program cache is keyed on tensor shape. Every
token therefore paid a fresh JIT compile.

| | mean/step | tok/s |
|---|---|---|
| growing cache, cold (what a real utterance gets) | 2595.34 ms | **0.4** |
| growing cache, warm (a second pass over the same sizes) | 28.32 ms | 35.3 |
| **fixed-width cache, first pass** | **34.10 ms** | **29.3** |

First and last of 32 steps: **29.08 → 3299.99 ms** cold, **28.41 → 28.53 ms** warm.
The warm pass is dead flat, so the arithmetic cost of a longer cache is negligible —
**98.9 % of the cold cost was compilation.**

`forward_chunk_fixed()` holds the key width at `max_len`, leaving exactly two shapes
for an entire utterance. That is **73× on the first pass**, which is the only pass a
real utterance gets. The 34.10 ms against the warm 28.32 ms is the trade: attention
over 256 slots instead of 210–241.

`cache_width()` rounds the buffer to a multiple of 128 rather than fitting it
exactly, so a handful of bucket widths covers every utterance instead of putting a
compile at the start of every request.

**The live tokens sit at the *end* of the buffer.** ESPnet's `rel_shift` skews the
score block assuming the queries are the last `t1` of the `K` key positions — the
streaming case it was written for. Left-aligning gives every query the relative
geometry of a position it is not at: wrong everywhere, obviously wrong nowhere, and
no shape assertion catches it. `test_device_fixed_shape_cache_matches_the_growing_one`
does.

---

## Vocoder op costs

| op | shape | latency |
|---|---|---|
| iSTFT | 18 049 frames → 72 192 samples (3.27 s of audio) | **1.115 ms** |
| iSTFT | 1 024 frames → 4 092 samples | 0.853 ms |
| `ConvTranspose1d` | 512→256, k=16, s=8, L=282 (`ups[0]`) | **3.886 ms** |

**The inverse transform is cheap and `conv_transpose2d` at `H=1` is not.** The whole
iSTFT costs 1.115 ms for 3.27 s of audio — an RTF contribution of 0.00034 — while a
single upsample layer costs 3.886 ms, **3.5× the entire iSTFT**, and there are two of
them. The op that was supposed to be the hard part is negligible; the op standing in
for a missing `ttnn.conv_transpose1d` dominates the stage.

---

## Accuracy

PCC against PyTorch goldens captured from the unmodified reference.

| module | PCC |
|---|---|
| **flow: tokens → mel** | **0.9992029011** |
| **tokens → waveform** (reference excitation injected) | **0.9951367159** |
| tokens → waveform, self-computed excitation | envelope **0.9974698767** |
| `solve_euler`, 10 CFM steps | 0.9992047752 |
| CFM estimator UNet, first / last step | 0.9998326979 / 0.9991904460 |
| flow Conformer encoder, 6 blocks | 0.9999176853 |
| LLM AR prefill, 209 tokens | 0.9997355989 |
| LLM AR decode step (fixed cache) | 0.9994433945 |
| LLM text encoder, causal | 0.9998775504 |
| whole HiFT vocoder, mel → waveform | 0.9996373743 |
| iSTFT vs the captured golden | 0.9999298811 |
| `SineGen` over 72 192 samples | 0.9999974539 |

### Two accuracy levers that mattered more than expected

**High math fidelity belongs on matmuls, not only convolutions.** `MathFidelity.HiFi4`
with `fp32_dest_acc_en` applied to `ttnn.linear`/`ttnn.matmul` moved the CFM
estimator's last Euler step from 0.9986 to 0.9992 — a 41 % error reduction — and the
first from 0.9998 to 0.9998. The gap widens with `t` because later steps carry larger
activations, and it compounds because the ten evaluations are *integrated*. **Depth,
not op type, decides fidelity.**

**`ttnn.cumsum` is 2000× less accurate than torch's.** Against an fp64 reference over
the real 72 192-sample f0:

```
device cumsum, fp32    max|d| 5.62e-01    (t=1k 2.3e-07, t=36k 0.114)
torch  cumsum, fp32    max|d| 2.44e-04
```

Phase is `2π·(cumsum mod 1)`, so 0.56 absolute is **more than half a cycle** — the
harmonic bank is randomised by the end of the utterance. At T=1024 and T=8192 the
error is ~1e-5, which is why this passed every module-level test and only surfaced
end to end. `phase_mod1()` reduces each block total mod 1 before accumulating, which
keeps every partial sum O(1) and takes this from 0.843 to **0.99999745**.

### One accuracy limit that is a property of the model

f0 error **integrates** into excitation phase. Drift is `Σ(Δf0)/sr` over samples, so
holding it under a tenth of a cycle across 72 192 samples needs a mean f0 error below
**0.03 Hz** — 1.5e-4 relative at 200 Hz, about 13 mantissa bits. The device's f0
predictor lands at ~16 Hz max error even with fp32 weights *and* activations, because
Tensix HiFi4 is four bfloat16 passes rather than true fp32.

This is not a defect to fix. **Sample-level waveform comparison is only meaningful
with the reference excitation injected**, which is what the PCC gate does; for a
self-computed excitation the honest metrics are the energy envelope and the RMS, and
those hold (0.9975 envelope, RMS within 6 %). It is the same discipline the CFM's
initial noise and `SineGen`'s two draws already follow.

---

## Test counts

| tier | count | hardware |
|---|---|---|
| host | 79 | none |
| device | 31 | Blackhole p150a |
