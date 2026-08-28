# Model performance and accuracy

Performance and accuracy numbers for CosyVoice-300M, collected from direct pytest runs in
`models/demos/cosyvoice/tests/`. This is the single source for both, restricted to Blackhole
(`p150a`, `p150b`) and Wormhole (`n300`) — every figure below was measured on the hardware in
*Environment*, none is an estimate, and none is `xfail`-ed. Where a target is missed the number is
stated with the reason and the identified lever.

## Environment

| property | Blackhole `p150a` | Blackhole `p150b` | Wormhole n300 |
|---|---|---|---|
| Form factor | single card | Quietbox, 4 cards | T3000, 4 cards |
| Devices | 1 × 32 GB | 4 × 32 GB, one used | 4 × 12 GB, one used |
| Compute grid | **13 × 10 = 130** | **13 × 10 = 130** | **8 × 8 = 64** |
| Host | 16 cores, 62 GB | 32 cores, 512 GB | 32 cores, 512 GB |
| tt-metal | `b5e9cba196` | `b5e9cba196` | `b5e9cba196` |
| Date | `2026-08-06`; RTF rows `2026-08-18` | `2026-08-06` | `2026-08-06`; RTF + KV-default rows `2026-08-18` |

`p150a` and `p150b` are the **same silicon, differing only in cooling** — `p150a` actively, `p150b`
passively — and the passive board measures **~5 % slower per token** on identical work (`7.07` vs
`6.73 ms` explicit chain; `5.87` vs `5.58 ms` fused attention; PCC matches to ten digits across both).
That 5 % is the same order as several optimisations here, so the two stay separate columns; `p150a`
is the headline part.

## Summary metrics

Measured on the captured utterance: 164 generated tokens producing 3.27 s of audio at 22 050 Hz.

| Metric | Value | Target |
|---|---:|---:|
| **End-to-end RTF, best measured** (`p150a` + `COSYVOICE_FF2_GRID=8x2`) | **`0.354`** | `< 0.5` ✅ |
| End-to-end RTF, `p150b`, everything on | `0.365` | `< 0.5` ✅ |
| End-to-end RTF, `p150a` default | `0.377` | `< 0.5` ✅ |
| End-to-end RTF, `p150a` + `COSYVOICE_KV_INPLACE=1` | `0.449` | `< 0.5` ✅ |
| End-to-end RTF, Wormhole n300, default (in-place KV is now default there) | `0.559` | `< 0.5` ❌ |
| End-to-end RTF, Wormhole n300, default + `COSYVOICE_FF2_GRID=8x2` | `0.557` | `< 0.5` ❌ |
| **LLM throughput, best measured** (`p150a` + `COSYVOICE_KV_INPLACE=1`) | **`200.8 tok/s`** | `>= 60` ✅ |
| LLM throughput, `p150b` + in-place KV | `190.8 tok/s` | `>= 60` ✅ |
| LLM throughput, `p150a` + `COSYVOICE_FF2_GRID=8x2` | `190.0 tok/s` | `>= 60` ✅ |
| LLM throughput, `p150a` default | `175.1 tok/s` | `>= 60` ✅ |
| LLM throughput, n300 default (in-place KV) | `127.6 tok/s` | `>= 60` ✅ |
| LLM decode latency, best measured (`p150a` + `COSYVOICE_KV_INPLACE=1`) | `4.98 ms` | — |
| Token agreement, teacher-forced | `99.04 %` | `> 95 %` ✅ |
| Token agreement, through the KV cache | `100.00 %` | `> 95 %` ✅ |
| WER (English) | `0.00 %` | `< 3.0` ✅ |
| Speaker similarity (mean, 10 utterances) | `83–96` | `> 60` ✅ |
| Streaming vs non-streamed, mel-space PCC (`p150b`) | `0.9019` | content-equal ✅ |
| Streaming vs non-streamed, mel-space PCC (n300) | `0.9024` | content-equal ✅ |
| tokens → waveform PCC | `0.9951` | `>= 0.99` ✅ |

`p150a`'s RTF and throughput rows are re-measured as of `2026-08-18`; every other row is unchanged
since `2026-08-06`. All device tests pass on both architectures — the one outstanding n300 failure
(streaming, mel-space PCC `0.218` against a `0.85` gate) was a `ttnn.conv1d` defect, not a streaming
one; see *A Wormhole conv1d defect (fixed)*.

### RTF breakdown

Blackhole `p150a`, default settings.

| Stage | Cost | RTF | Share |
|---|---:|---:|---:|
| LLM (14-block AR decoder, traced, fused attention) | `5.71 ms/token × 164` | `0.286` | 76 % |
| Flow decoder (10 Euler steps, traced, SDPA) | `0.253 s` | `0.077` | 21 % |
| HiFT vocoder | `0.045 s` | `0.014` | 4 % |
| **Total** | `1.234 s` | **`0.377`** | |

The LLM is 76 % of a `p150a` utterance at default settings, up from 59 % before the flow stage was
optimised (`0.589 → 0.253 s` — see *The flow decoder*). Two flags move it further:

- **`COSYVOICE_KV_INPLACE=1`** — in-place `ttnn.update_cache` instead of rebuilding the KV cache —
  takes the decode step to `4.98 ms` (`200.8 tok/s`), total `1.470 s`, **RTF `0.449`**. Opt-in on
  Blackhole: costs a 384 MB trace region and bit-exactness (worst PCC `0.9986` over 72 steps vs `1.0`).
- **`COSYVOICE_FF2_GRID=8x2`** — explicit 16-core grid for the FFN's second linear at decode
  (`T == 1` only) — takes the step to `5.26 ms` (`190.0 tok/s`), total `1.160 s`, **RTF `0.354`**.
  That linear is bound by its `K = 4096` reduction, not weight traffic, so *fewer* cores wins: `8x2`
  measures `1.98×` the default on `p150b`, `1.50×` on n300. Free to ten digits in the one A/B run, but
  `tests/pcc/` has never run with it on. Not portable as a default — Wormhole's gain is much smaller
  (*Wormhole re-verified*) — so it stays a flag on both parts.

RTF has come down **1.096 → 0.354** (**2.120 → 0.354** since before either stage was traced); `RTF <
0.5` is met on both Blackhole boards. `RTF < 0.2` needs under 1.5 ms/token for the LLM alone, and now
rests entirely on the decode step — not reachable by further op-level work; see *The LLM decode step*.

## Reproducing these numbers

```bash
pytest models/demos/cosyvoice/tests/perf/test_pipeline_perf.py -v -s   # end-to-end RTF
pytest models/demos/cosyvoice/tests/perf/test_trace.py          -v -s   # trace speedup
pytest models/demos/cosyvoice/tests/perf/test_llm_perf.py       -v -s   # decode throughput
```

Those three suites are the reproducible set — every headline figure in this document comes from one
of them. `scripts/` also holds 34 exploratory probes with no gates; the ones behind specific figures
below are named as they come up (e.g. `probe_op_floor.py`, `probe_kv_alignment.py`,
`repro_conv1d_wormhole.py`).

## Tuning flags

Everything ships at a default that was measured, not assumed. Defaults are read from the code, not
from this document; each row names the section that carries the measurement.

| flag | default | what it does | what it is worth |
|---|---|---|---|
| `COSYVOICE_KV_INPLACE` | follows `device.arch()` — on for Wormhole, off for Blackhole | writes the KV cache with `ttnn.update_cache` instead of rebuilding it | `1.42×` on the n300 step, `1.12–1.15×` on Blackhole; costs a 384 MB trace region and bit-exactness (worst PCC `0.9986` over 72 steps). *Decode step, and what each change is worth* |
| `COSYVOICE_FF2_GRID` | unset | explicit core grid for the FFN's second linear during decode, `T == 1` only | `8x2`: RTF `0.377 → 0.354` on `p150a`, `0.559 → 0.557` on n300. *RTF breakdown* |
| `COSYVOICE_SDPA_DECODE` | `1` | fused `sdpa_decode` for the AR decoder's relative-position attention | `−17.1 %` on the Blackhole step, `−11.0 %` on Wormhole. *Fused decode attention* |
| `COSYVOICE_SDPA` | `1` | fused SDPA in the flow estimator | flow `0.707 → 0.600 s`, and more accurate on every gate. *Flash attention* |
| `COSYVOICE_CFM_TRACE_CACHE` | `1` | keeps the CFM estimator trace across utterances of the same mel length | `1.81×` on the flow stage (`p150b`), `1.50×` (n300). *Trace cache reuse* |
| `COSYVOICE_GN_PERMUTE` | unset (matmul form) | restores the permute-based GroupNorm | the matmul form is `1.41×` / `1.34×` on the stage. *GroupNorm as a matmul* |
| `COSYVOICE_FLOW_STEPS` | `10` | Euler solver depth | 5 steps buys `1.43×` at PCC `0.9825` — below every gate here. *Trace cache reuse* |
| `COSYVOICE_FIDELITY` | `HiFi4` | math fidelity for the matmuls | free in time; HiFi2 is worse on 9 of 11 modules. *Accuracy* |
| `COSYVOICE_HIFT_TRACE` | unset (per-stream heuristic) | forces vocoder trace capture on or off | effect never isolated. *Wormhole re-verified* |
| `COSYVOICE_WEIGHT_BF8` | `0` | `bfloat8_b` decoder linear weights | `1.00×`, measured twice: a memory option (352 → 176 MB), not a speed one. *What the decode step is bound by* |
| `COSYVOICE_FLOW_BF8` | `0` | `bfloat8_b` flow-estimator weights | carries its own measurement rather than inheriting the decoder's verdict |
| `COSYVOICE_FP32_ACC` | `1` | fp32 accumulation in the vocoder convolutions | off *moves* the Wormhole `conv1d` bad-length band rather than closing it. *A Wormhole conv1d defect (fixed)* |
| `COSYVOICE_CONV_PREPARE` | unset (per-geometry verification) | overrides the prepared-weight verdict either way | disabling preparation outright cost the flow stage `0.683 → 1.723 s` on n300. *A Wormhole conv1d defect (fixed)* |

Two flags are opt-in because the best setting is not portable rather than because they are risky —
`COSYVOICE_FF2_GRID` and, on Blackhole, `COSYVOICE_KV_INPLACE`.

## The LLM decode step

76 % of an utterance on `p150a`, and the stage where whatever is left to win now sits.

### Fused decode attention

The AR decoder's relative-position attention runs as
`ttnn.transformer.scaled_dot_product_attention_decode` — the positional bias term has exactly the
shape its `attn_mask` accepts at `T = 1`. Per attention block (key width 384/448): `1.563`/`1.817 ms`
explicit → `0.460`/`0.557 ms` fused (3.3–3.4×). Per decode step, default path: `6.73 → 5.58 ms`
(`148.5 → 179.2 tok/s`); RTF `0.533 → 0.477`. Free on accuracy: traced still matches untraced
bit-for-bit (PCC `1.0000000000`), fused matches the explicit chain at `0.9988`–`0.9999`, and
exact-token agreement through the KV cache holds at `100.00 %` (was `95.83 %`).
`COSYVOICE_SDPA_DECODE=0` restores the explicit chain. Trace capture alone is worth **2.54× on the
AR decoder** (20.96 → 8.26 ms/token) and **1.09× on the flow decoder**.

### Removing token-independent recomputation

`linear_pos(pos_emb)` and three related ops were being recomputed identically on every one of 164
decode steps per utterance, inside the trace, despite depending only on `max_len`. Hoisting the
head-split transpose out (the bulk of the gain), collapsing `rel_shift` to one slice at `T = 1`,
and fusing `transpose_b` + `scale_mask_softmax` into their matmuls: step **15.71 → 8.25 ms**,
throughput **63.6 → 121.3 tok/s**.

**Fusing QKV into one matmul is stage-dependent**: it helped the flow decoder (`1.075 → 0.719 s`,
T ≈ 600, batch 2) but was a wash on the AR decode step (`8.29 → 8.31 ms`, T = 1, where splitting back
into heads costs about what the fused matmul saved) — op count is a proxy for cost, not the cost.

### What the decode step is bound by

Not weight bandwidth: `bfloat8_b` weights measure `1.00×` at two different effective bandwidths (27
and 42 GB/s), so `COSYVOICE_WEIGHT_BF8` is kept as a memory option (352 → 176 MB), not a speed one.
Not the four linears either — 34 % of the step (2.82 ms/14 layers) and already near TTNN's default
grid optimum. It is a **per-op dispatch floor of ~6.3 µs, flat in tensor size**, across the ~280
non-linear ops that make up the rest: ~2.1 ms of the 8.25 ms step is irreducible there.

### KV-cache layout: tile alignment, not bandwidth

`slice` + `concat` on the `[1, 16, 256, 64]` cache cost ~228 µs against 19–64 µs for every other
non-linear op — **0.5 MB moved in 134 µs, ~3.7 GB/s**, two orders below what the byte count implies.
Slicing/concatenating at a tile-aligned row is **11–16× cheaper** than at row 1 (`78.3→7.0 µs`,
`207.4→13.1 µs`), and `bfloat8_b` (half the bytes) is identical to the last decimal — a layout cost,
not a bandwidth one. `TILE_LAYOUT` tiles the *last two* dimensions, so a `[1, h, T, d_k]` cache puts
time on a tiled axis; time-major `[1, T, h, d_k]` puts it on a free one: slice+concat `207.2 → 19.7
µs`, plus a `13.9 µs` permute back for the matmuls. Net: **traced decode step `8.26 → 6.73 ms`**
(121.4 → 148.5 tok/s), trace speedup `2.54× → 3.10×`, end-to-end RTF `0.610 → 0.533`, bit-exact vs
untraced — 6.2× of what `ttnn.update_cache`'s in-place write offers (3.7 µs, 56×) without its 32
pre-captured sub-step traces.

The per-token tail outside the traced step is **0.352 ms (2.7 %)**: output head matmul `0.043 ms`,
logits device→host `0.142 ms`, RAS sampling on host `0.075 ms`, embedding row→device `0.092 ms`.
`ttnn.sampling` could remove at most 1.7 % of a token and would give up exact agreement with the
reference, so sampling stays host-side; `nucleus_filter` itself was optimised instead (`0.245 →
0.075 ms`, bit-identical).

### Fixed-width KV cache — 73× on the first real pass

A growing cache gives every decode step a new attention shape, so TTNN's program cache pays a fresh
JIT compile per token:

| | mean/step | tok/s |
|---|---:|---:|
| growing cache, cold (what a real utterance gets) | `2595.34 ms` | `0.4` |
| growing cache, warm (second pass, same sizes) | `28.32 ms` | `35.3` |
| **fixed-width cache, first pass** | **`34.10 ms`** | **`29.3`** |

98.9 % of the growing-cache cold cost was compilation. `forward_chunk_fixed()` holds the key width at
`max_len` (rounded to a multiple of 128), leaving two shapes for a whole utterance, with live tokens
at the *end* of the buffer per ESPnet's `rel_shift` geometry — guarded by
`test_device_fixed_shape_cache_matches_the_growing_one`.

## The flow decoder

21 % of an utterance on `p150a`, down from 48 % — the stage that more than halved
(`0.589 → 0.253 s`).

### Flash attention

Plain SDPA, no mask or relative-position term, so `ttnn.transformer.scaled_dot_product_attention` is a
drop-in for the estimator's self-attention:

| | explicit chain | fused SDPA |
|---|---:|---:|
| flow decoder | `0.707 s` | **`0.600 s`** |
| end-to-end RTF | `0.647` | **`0.611`** |
| `solve_euler` PCC | `0.9992047752` | **`0.9993701398`** |
| flow tokens → mel PCC | `0.9992029011` | **`0.9993962895`** |
| CFM estimator, first / last step | `0.9998326979` / `0.9991904460` | **`0.9998480374` / `0.9994887951`** |

Faster and more accurate on every gate. `scale=1.0` because `1/sqrt(d_head)` is folded into the fused
QKV weight's q half. `COSYVOICE_SDPA=0` restores the explicit chain.

### Trace cache reuse

The flow stage is not linear in solver depth: `T(n) ≈ 0.350 s + 35.8 ms/step`. Halving the 10-step
solver buys 1.43×, not 2×, at PCC `0.9825` — below every gate here, so `COSYVOICE_FLOW_STEPS` is
available but unused by default. The fixed `0.350 s` was trace capture repeated on every call (46.6 %
of the solve; replay 52.9 %). Keeping the trace across utterances of the same mel length is worth
**1.67× on the solver** (`0.601 → 0.359 s` steady state), taking Wormhole end-to-end from `0.736` to
`0.628` — verified safe across utterances with different conditioning (the trace bakes a buffer
*address*, refilled in place each time) at PCC `1.0000000000` on three consecutive solves.
`COSYVOICE_CFM_TRACE_CACHE=0` restores the old behaviour.

### GroupNorm as a matmul

A traced, per-block-class profile (untraced timings are host-dispatch-bound and misleading — see
*Blackhole and Wormhole side by side*) found GroupNorm costing **~7× the convolution beside it**
(`0.2197`/`0.3809 ms` vs conv1d's `0.0320`/`0.0556 ms` on `p150b`/n300, at one resnet block, T=141).
33 GroupNorms run per Euler step, ~36 % of the whole estimator. The cost was the permute-based reshape
used to reduce over channel groups, which re-tiles under `TILE_LAYOUT`. Recasting the channel sum as a
matmul against a `[C, G]` indicator avoids the re-tiling:

| | `p150b` | n300 | PCC vs torch |
|---|---:|---:|---:|
| `[2, 141, 256]` permute → matmul | `0.2202` → `0.1012` (**2.18×**) | `0.3820` → `0.1874` (**2.04×**) | `0.999988854` |
| `[2, 282, 256]` permute → matmul | `0.3993` → `0.1056` (**3.78×**) | `0.6691` → `0.2045` (**3.27×**) | `0.999992251` |

On the whole stage: `1.41×` on `p150b`, `1.34×` on n300. `COSYVOICE_GN_PERMUTE=1` restores the old
form; native `ttnn.group_norm` rejects these shapes at `G=8` on both parts. **Needs a variance clamp**
(`ttnn.relu` on variance before `eps`): the matmul form's `E[x²] − E[x]²` can go slightly negative
under bfloat16 rounding and produce an unraised `Inf` through `rsqrt` on real (non-golden) utterances
— fixed at a ~2–8 % timing cost with no PCC change.

## The vocoder

4 % of an utterance.

| op | shape | latency |
|---|---|---:|
| iSTFT | 18 049 frames → 72 192 samples (3.27 s of audio) | **`1.115 ms`** |
| iSTFT | 1 024 frames → 4 092 samples | `0.853 ms` |
| `ConvTranspose1d` | 512→256, k=16, s=8, L=282 (`ups[0]`) | **`3.886 ms`** |

The inverse transform is cheap (RTF contribution `0.00034`); the `conv2d`-at-`H=1` op standing in for
a missing `ttnn.conv_transpose1d` dominates the stage, at 3.5× the entire iSTFT per upsample layer (of
two).

### A Wormhole conv1d defect (fixed)

`ttnn.conv1d` returned wrong values (up to `7e37` against a correct `9.42`) for input lengths
**8193–8704** on Wormhole, but only when its weight went through `ttnn.prepare_conv_weights` first —
0 of 21 lengths affected on Blackhole (two boards). This caused the one n300 test failure: streaming
vocoded a 130-frame chunk (prompt-extended) that fell in the bad band, producing a Snake-activation
`inf` and 12.7×-too-loud audio (mel-space PCC `0.218` against a `0.85` gate).

Root cause narrowed to weight preparation itself. Fix: verify each `(length, batch)` geometry once —
run both prepared and unprepared, keep the prepared weight only where they agree — rather than
disabling preparation outright (which cost the *flow* stage `0.683 → 1.723 s`, since `TtConv1d` also
backs the estimator's trace-captured convolutions). `COSYVOICE_CONV_PREPARE` overrides the verdict
either way. Result: vocoder `0.084 → 0.077 s` (verification is free at the utterance level);
streamed-vs-non-streamed mel PCC `0.218 → 0.9024`, matching Blackhole's `0.9019`.

## Blackhole and Wormhole side by side

Same commit, same tests, same utterance — 164 tokens, 3.27 s of audio, on all three parts. `p150a` was
unreachable for part of this sweep (13 retries over ~25 min), so the CFM-trace-cache row was measured
on `p150b` instead; `p150a`'s own cells are kept as measured, not backfilled.

### End-to-end RTF

Each row adds one change to the row above it.

| | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|
| explicit chain, no CFM cache | `0.533` | `0.584` | `0.950` |
| **+ fused decode attention** | **`0.477`** ✅ | `0.523` | `0.891` |
| **+ cached CFM trace** | *`0.367` projected | **`0.436`** ✅ | — |
| **+ in-place KV** (`COSYVOICE_KV_INPLACE=1`) | `0.449`* ✅ | `0.398` ✅ | `0.628` |
| **+ permute-free GroupNorm** (and, on n300, the conv fix) | — | **`0.365`** ✅ | **`0.575`** |

Best in this table: `0.365` (`p150b`), `0.575` (n300) — both superseded by *Summary metrics*'s
`0.354` (`p150a` + `COSYVOICE_FF2_GRID=8x2`) and *Wormhole re-verified*'s `0.559`. `RTF < 0.5` is met
on both Blackhole boards, missed on Wormhole. The last row is a median over 4 runs on `p150b`
(`0.362`–`0.368`) and 6 on n300 (`0.557`–`0.583`); every row above is a single run. *`p150a`'s
in-place-KV figure predates the CFM trace cache; `0.367` above it is projected, not measured.

### Where each change is worth what

| change | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|
| fused decode attention, on the step | `−17.1 %` | `−17.0 %` | `−11.0 %` |
| cached CFM trace, on the flow stage | — | **`1.81×`** | `1.50×` |
| cached CFM trace, on the solver alone | — | **`2.37×`** | `1.67×` |
| in-place KV cache, on the step | `1.12×` | `1.15×` | **`1.42×`** |
| trace capture, on the decode step | `3.72×` | — | `1.72×` |

The fused-attention gain is nearly identical across the two Blackhole boards (`−17.1 %`, `−17.0 %`)
and smaller on Wormhole. The CFM trace cache pays more on the faster part (`2.37×` vs `1.67×`):
capture cost is largely fixed, so it's a bigger share of a shorter replay. The in-place KV cache runs
the other way (`1.42×` on Wormhole vs `1.12–1.15×` on Blackhole) — why it is the default on Wormhole
and opt-in on `p150a`.

### Where the time goes, at the best setting

Fully loaded — fused attention, cached CFM trace, in-place KV, permute-free GroupNorm:

| stage | `p150b` | n300 | n300 : `p150b` |
|---|---:|---:|---:|
| LLM (164 tokens) | `0.845 s` | `1.290 s` | 1.53× |
| Flow decoder (10 Euler steps) | `0.277 s` | `0.493 s` | 1.78× |
| HiFT vocoder | `0.068 s` | `0.080 s` | 1.18× |
| **Total** | **`1.196 s`** | **`1.884 s`** | **1.58×** |

The `1.58×` overall ratio sits inside the **2.03× ratio in core count** (130 vs 64): the vocoder is
nearly architecture-neutral (`1.18×`, two large `conv_transpose2d` calls), the flow is the most
core-hungry stage (`1.78×`). Fully loaded, the LLM is 68 % of a Wormhole utterance and 71 % of a
Blackhole one — up from ~63 % before the flow-stage work, so it's where further RTF work has to go.

### Decode step, and what each change is worth

| | Blackhole | Wormhole |
|---|---:|---:|
| untraced | `20.83 ms` (48.0 tok/s) | `20.10 ms` (49.7 tok/s) |
| traced, moving cache @ 384 | `5.60 ms` (178.7) | `11.68 ms` (85.6) |
| traced, moving cache @ 448 | `5.88 ms` | `10.12 ms` |
| traced, in-place @ 448 | **`4.99 ms`** (200.4) | **`8.20 ms`** (122.0) |
| trace speedup | **3.72×** | **1.72×** |
| fused attention is worth | `−1.15 ms/token` (−17.1 %) | `−1.37 ms/token` (−11.0 %) |
| in-place KV is worth | `+0.61 ms` (**1.12×**) | `+3.48 ms` (**1.42×**) |
| cost of widening 384 → 448 | `+0.28 ms` | **`−1.55 ms`** — *faster* |

Untraced decode is nearly identical on both parts (`20.83` vs `20.10 ms`) — host-dispatch-bound, with
the architecture gap appearing only once tracing removes that overhead. The KV-width tile-parity
effect **flips sign** between architectures (384→448 costs Blackhole `+0.28 ms`, saves Wormhole
`1.55 ms`): the scheduling optimum is architecture-specific.

### Wormhole re-verified

`COSYVOICE_KV_INPLACE` is now the Wormhole default (`kv_inplace_default(device)` in
`tt/llm/decoder.py`, keyed on `device.arch()`; still overridable). At the corrected flag baseline:
median RTF `0.559` (KV in-place alone), `0.557` with `COSYVOICE_FF2_GRID=8x2` added (real but small —
`7.84 → 7.59 ms/token` median — against `p150a`'s much larger 6.1 % gain from the same flag, because
Wormhole's `8×8 = 64`-core default grid has less of the "too many cores for one row" problem the flag
fixes). `RTF < 0.5` remains missed on Wormhole.

## Accuracy

| Module | PCC |
|---|---:|
| tokens → waveform (reference excitation) | `0.9951367159` |
| flow: tokens → mel | `0.9993962895` |
| whole HiFT vocoder | `0.9996373743` |
| LLM AR prefill, 209 tokens | `0.9997530373` |
| LLM AR decode step | `0.9989617190` |
| traced vs untraced decode | `1.0000000000` (bit-exact) |
| iSTFT vs captured golden | `0.9999298811` |

Fidelity depth matters more than op type: `HiFi4` with `fp32_dest_acc_en` on `ttnn.linear`/`ttnn.matmul`
moved the CFM estimator's last Euler step from PCC `0.9986` to `0.9992` (41 % error reduction), so
`COSYVOICE_FIDELITY` stays `HiFi4` everywhere despite costing nothing to lower.

**`ttnn.cumsum` was 2000× less accurate than torch's** (`max|d| 5.62e-01` vs `2.44e-04` against an
fp64 reference over the real f0 signal) — phase is `2π·(cumsum mod 1)`, so 0.56 absolute is over half
a cycle, randomising the harmonic bank by the end of an utterance. Fixed by `phase_mod1()`, reducing
each block mod 1 before accumulating: `0.843 → 0.99999745` PCC, and **6.9× faster** as a side effect
(single-core serial scanning was the cause of both):

| `cumsum` + `mod 1` | `p150b` | n300 |
|---|---:|---:|
| plain, one core | `40.4 ms` | `73.3 ms` |
| `phase_mod1` | `5.9 ms` | `12.5 ms` |

The waveform-PCC gate injects the reference excitation deliberately: f0 error integrates into phase
drift, and holding drift under a tenth of a cycle across 72 192 samples needs mean f0 error below
0.03 Hz — tighter than Tensix HiFi4 delivers (~16 Hz max error, four bfloat16 passes rather than true
fp32). A model property, not a defect; with a self-computed excitation the honest metrics are energy
envelope (`0.9975`) and RMS (within 6 %).

### Per-architecture accuracy and test results

| | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|
| traced vs untraced | `1.0000000000` | `1.0000000000` | `1.0000000000` |
| in-place, worst PCC over 72 steps | `0.9987379437` | — | `0.9991855486` |
| CFM trace cache, 3 solves, new conditioning each | — | `1.0000000000` | `1.0000000000` |
| test suite | `155 passed` * | **157 passed** | **157 passed** |

*`p150a`'s count is from an older tree (two vocoder trace tests short); that board has been
unavailable since. Every device test now passes on both architectures.

## Generation modes

All four modes run on device across five languages — 20 cases, all synthesising:

| mode | prompt | on device |
|---|---|---|
| zero-shot | reference audio | ✅ 5/5 |
| cross-lingual | reference audio, different language | ✅ 5/5 |
| SFT | speaker id, no prompt audio | ✅ 5/5 |
| instruct | speaker id + description, no prompt audio | ✅ 5/5 |

The two prompt-free modes (`sft`, `instruct`) needed three flow-stage fixes for the zero-length prompt
case: two zero-length `ttnn::concat` calls (segfault rather than raise) and a full-extent `ttnn.slice`
aliasing bug. Per-case RTFs from this sweep are cold-cache (every sequence length is a fresh JIT
compile) and are not comparable to the traced end-to-end RTF numbers above.

## Speech quality — 5 languages, 2 modes

Scored with whisper `large-v3`; CER for CJK, WER for English.

| Mode | zh | en | ja | ko | yue |
|---|---:|---:|---:|---:|---:|
| zero-shot | `3.03` | `0.00` | `5.56` | `3.12` | `64.52` |
| cross-lingual | `6.06` | `0.00` | `2.78` | `0.00` | `100.00` |

Cantonese is a model limitation, not a port defect: the PyTorch reference scores worse on the same
text through the same ASR (`83.87 %` zero-shot vs this port's `64.52 %`).

## Coverage and test counts

Source suites: `tests/perf/`, `tests/e2e/`, `tests/pcc/` — 157 tests: 111 host, 46 device (a device
run executes both tiers).

- End-to-end RTF with a per-stage breakdown
- Trace capture speedup and bit-exactness
- Decode throughput, cold vs warm, growing vs fixed-shape KV cache
- Streaming content equivalence and seam continuity
- Per-module PCC against captured PyTorch goldens

| Tier | Count | Hardware | Result |
|---|---:|---|---|
| host | 111 | none | — |
| device | 46 | Blackhole `p150b` | **157 pass** |
| device | 46 | Wormhole n300 | **157 pass** |
| device | 44 | Blackhole `p150a` | **155 pass** — older tree, two vocoder trace tests short |

## Operational notes

**`l1_small_size` scales with conv *configurations*, not tensor size.** `ttnn.conv1d` allocates
prepared weights from that bank and keeps them: zero-shot needs 128 KB, cross-lingual needs 256 KB
(1289 mel-frame prompt vs 326).

**Persist the JIT cache across runs.** Mounting `~/.cache/tt-metal-cache` took the first utterance
from `161.7 s` to `14.8 s` wall. Every distinct sequence length is a fresh compile.
