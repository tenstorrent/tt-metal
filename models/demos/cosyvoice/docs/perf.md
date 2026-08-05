# CosyVoice-300M on Tenstorrent — measured performance

All figures from **Blackhole p150a** silicon, 2026-08-05, `tt-metal @ b5e9cba196`.
Reproduce with the commands in [`../README.md`](../README.md).

Nothing here is an estimate and nothing is `xfail`-ed. Where a target is missed, the
number is stated with the reason and the identified lever.

---

## End-to-end real-time factor

RTF is compute-seconds per second of audio. Measured on the captured utterance —
164 generated tokens producing **3.27 s** of audio at 22 050 Hz.

| stage | cost | RTF | share | traced? |
|---|---|---|---|---|
| **LLM** (14-block AR decoder) | 15.71 ms/token × 164 | **0.787** | **70.1 %** | yes, 2.22× |
| flow decoder (10 Euler steps) | 1.053 s | 0.322 | 28.6 % | yes, 1.09× |
| HiFT vocoder (mel → 72 192 samples) | 0.048 s | 0.015 | 1.3 % | no |
| **total** | **3.677 s** | **1.123** | | |

**Target: RTF < 0.5 (P5), < 0.2 (P6). Not met — measured 1.123**, down from 2.120
before either stage was traced.

The split is the useful part. The vocoder — the stage the previous attempt left on
the host, and the one this bring-up treated as the hard problem because TTNN has no
FFT — is **1.3 %** of runtime. The LLM is 70 %, because it is the only stage whose
cost scales with output length rather than being amortised over it: one forward
pass per token, and a second of speech is 50 tokens.

At roughly 280 ops per decode step and 34.66 ms untraced, that was ~124 µs/op —
**dispatch bound, not compute bound**, which is what made tracing the right lever.

### What trace capture actually bought

`ttnn.begin_trace_capture` / `execute_trace` records the op graph once and replays it
with a single host command. Applied to both stages, it produced very different
numbers, and the difference is the transferable lesson:

| stage | untraced | traced | speedup |
|---|---|---|---|
| AR decode step | 34.92 ms | 15.71 ms | **2.22×** |
| flow decoder, 10 steps | 1.151 s | 1.053 s | **1.09×** |

**Tracing repays dispatch overhead, so it pays in proportion to how dispatch-bound a
stage already is.** The AR decoder issues ~280 small ops per token at batch 1 and is
almost pure overhead. The flow decoder runs 16 resnet and 64 transformer blocks over
608 frames at batch 2 — enough arithmetic per op that removing the dispatch cost
moves 9 %. Both are bit-exact against their untraced selves.

Two prerequisites, one per stage. The AR decoder needed an **in-place KV cache**: a
trace replays fixed device addresses, so the cache cannot be reallocated by `concat`
every step. The flow decoder needed its **convolution weights prepared once** —
`ttnn.conv1d` and `conv_transpose2d` otherwise tilize and upload their weights on
every call, which is host traffic a trace cannot contain. Neither is a flag; both are
described in `tt/llm/decoder.py` and `tt/hifigan/conv.py`.

### `bfloat8_b` weights: no speedup, and the reason is the useful part

The obvious next lever was narrower weights for the AR decoder. At batch 1 an
autoregressive step reads every matrix from DRAM to produce one token, with no reuse
to amortise it against, so halving the weight width should halve the traffic.

Measured, it does nothing:

| weights | ms/step | tok/s | speedup |
|---|---:|---:|---:|
| `bfloat16` | 13.09 | 76.4 | — |
| `bfloat8_b` | 13.12 | 76.2 | **1.00×** |

Accuracy is fine — hidden-state PCC `0.9997040033` after 14 blocks — so the option is
kept. It is a **memory** option, not a speed one: 352 MB of weights become 176 MB.

The reason it buys no time is worth stating, because it redirects everything after
it. The AR decoder's linears are 176 M parameters, so a `bfloat16` step moves 352 MB.
At 13.09 ms that is **27 GB/s effective**, against a device that does several hundred.
The bandwidth floor for this step is ~0.88 ms; it takes 13.09. **The step is ~15×
away from being bandwidth-bound**, so halving the bandwidth requirement is invisible.

What it is bound by is per-op cost on tensors one row tall. A decode step issues
roughly 500 ops, which puts it near 26 µs/op — and a trace has already removed the
*host* side of that. So the remaining lever is **fewer, larger ops**, not narrower
ones: fusing the per-block q/k/v projections into one matmul, and collapsing the
five-op `rel_shift` skew. Flash attention via the additive-`attn_mask` identity is
the same kind of change — it replaces a chain with one kernel.

This is also the honest reading of the 2.22× the trace bought: it removed host
dispatch and left device-side per-op cost, which is now the binding constraint.

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

## Speech quality — 5 languages, 2 modes

Synthesised on Blackhole, scored with **whisper large-v3 on Tavern** (62 GB; the RAM preflight
declines large-v3 on an 11 GB host, and `medium` has no Cantonese token, so the scoring host is part
of the measurement). CER for CJK, WER for English.

| mode | zh | en | ja | ko | yue | SIM | tok/s |
|---|---|---|---|---|---|---|---|
| zero-shot | 3.03 | **0.00** | 5.56 | 3.12 | 64.52 | 83–96 | 16–27 |
| cross-lingual | 6.06 | **0.00** | 2.78 | **0.00** | 100.00 | 86–94 | 27–29 |

**R9_wer_lt_3.0 PASS · R9_sim_gt_60 PASS · R8_rtf_lt_0.5 FAIL · R8_tok_per_s_ge_30 FAIL.**

Excluding Cantonese the CJK mean is 3.90 % zero-shot and 2.95 % cross-lingual; English is perfect in
both modes.

### Cantonese is a model limitation

The PyTorch reference, same texts and same ASR, scores **worse**:

| case | TTNN | PyTorch reference |
|---|---|---|
| zero-shot yue | 64.52 % | **83.87 %** |
| cross-lingual yue | 100.00 % | 67.74 % |
| zero-shot zh | 3.03 % | **3.03 %** |

CosyVoice-300M does not do Cantonese well when prompted with a Mandarin reference voice. Chinese
matches the reference exactly. Without this baseline a 64 % CER would have read as a broken port.

One difference is real and open: `cross_lingual yue` emitted 122 tokens for 2.44 s against the
reference's 387 for 7.73 s — the LLM terminated early. RAS is stochastic and the model's Cantonese
confidence is low, so this is plausible without a bug; a greedy run would settle it.

### Two operational notes from the sweep

**`l1_small_size` scales with conv *configurations*, not tensor size.** `ttnn.conv1d` allocates
prepared weights from that bank and keeps them, so three models live at once (~80 convs) exhausts
the 32 KB a single-model test uses — failing part-way through the *second* utterance with
`Not enough space to allocate 480 B L1_SMALL buffer`. Zero-shot needs 128 KB; cross-lingual needs
256 KB because its prompt is 1289 mel frames against 326.

**Persist the JIT cache across runs.** Mounting `/root/.cache/tt-metal-cache` took the first
utterance from 161.7 s to 14.8 s wall. Every distinct sequence length is a fresh compile.

---

## Test counts

| tier | count | hardware |
|---|---|---|
| host | 79 | none |
| device | 31 | Blackhole p150a |
