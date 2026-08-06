# Model performance and accuracy

Performance and accuracy numbers for CosyVoice-300M, collected from direct pytest runs in
`models/demos/cosyvoice/tests/`. This is the single source for both — every figure below was
measured on the hardware named in *Environment*, none is an estimate, and none is `xfail`-ed. Where
a target is missed the number is stated with the reason and the identified lever.

## Environment
- Device: Blackhole `p150a`
- Host: 16 cores, 62 GB
- tt-metal: `b5e9cba196`
- Date: `2026-08-06`

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
| End-to-end RTF | `0.477` | `< 0.5` ✅ |
| End-to-end RTF, `COSYVOICE_KV_INPLACE=1` | `0.449` | `< 0.5` ✅ |
| LLM throughput (traced) | `179.2 tok/s` | `>= 60` ✅ |
| LLM throughput, `COSYVOICE_KV_INPLACE=1` | `200.8 tok/s` | `>= 60` ✅ |
| LLM decode latency (traced) | `5.58 ms` | — |
| Token agreement, teacher-forced | `99.04 %` | `> 95 %` ✅ |
| Token agreement, through the KV cache | `100.00 %` | `> 95 %` ✅ |
| WER (English) | `0.00 %` | `< 3.0` ✅ |
| Speaker similarity (mean, 10 utterances) | `83–96` | `> 60` ✅ |
| Streaming vs non-streamed, mel-space PCC | `0.9019` | content-equal ✅ |
| tokens → waveform PCC | `0.9951` | `>= 0.99` ✅ |

### RTF breakdown

| Stage | Cost | RTF | Share |
|---|---:|---:|---:|
| LLM (14-block AR decoder, traced, fused attention) | `5.58 ms/token × 164` | `0.280` | 59 % |
| Flow decoder (10 Euler steps, traced, SDPA) | `0.589 s` | `0.180` | 38 % |
| HiFT vocoder | `0.056 s` | `0.017` | 4 % |
| **Total** | `1.561 s` | **`0.477`** | |

**`COSYVOICE_KV_INPLACE=1`** writes the KV cache in place with `ttnn.update_cache` instead of
rebuilding it, taking the decode step to `4.98 ms` (`200.8 tok/s`) and the total to `1.470 s`,
**RTF `0.449`**. It is opt-in because it costs two things the default does not: a 384 MB trace
region for the 65 traces it captures, and bit-exactness — worst PCC `0.9986` over 72 steps against
the moving cache's exact `1.0`, non-accumulating. The width it needs has to keep the key axis on an
**even tile count**; a one-tile scratch zone made it *slower*. Findings F45 and F46 in the notes
carry that account.

RTF has come down **1.096 → 0.477** (and **2.120 → 0.477** since before either stage was traced),
and **the `< 0.5` target is met**. The rest of this section is the account of where the time went,
because the last step of it was the one that had been ruled out on a false premise.

### The decode attention is expressible as flash attention

The AR decoder composed its attention by hand — score matmul, positional bias add, masked softmax,
context matmul — because ESPnet relative-position attention was taken to be outside what a fused
kernel expresses. It is not. `ttnn.transformer.scaled_dot_product_attention_decode` accepts an
`attn_mask` whenever `is_causal=False` (`sdpa_decode_device_operation.cpp:111`), shaped
`[B, 1, heads, k_len]` and added to the scores before the softmax. **At `T = 1` the positional term
`(q+v)P^T` has exactly that shape** — `rel_shift` is only a two-dimensional skew when there is more
than one query — so it *is* an additive bias, and the padding mask folds into the same tensor.

| | explicit chain | `sdpa_decode` | |
|---|---:|---:|---:|
| attention block, key width 384 | `1.563 ms` | **`0.460 ms`** | 3.39× |
| attention block, key width 448 | `1.817 ms` | **`0.557 ms`** | 3.26× |
| decode step, default path | `6.73 ms` | **`5.58 ms`** | `148.5 → 179.2 tok/s` |
| decode step, `COSYVOICE_KV_INPLACE=1` | `6.23 ms` | **`4.98 ms`** | `160.5 → 200.8 tok/s` |
| **end-to-end RTF** | `0.533` | **`0.477`** | `0.449` with both |

The block figures charge the fused arm for the two `ttnn.permute`s it needs to move heads off dim 1
into the decode layout; without them it is 4.75× and 4.35×.

**It costs nothing on accuracy, which is what separates it from every other remaining lever.**
Traced still matches untraced bit-for-bit (PCC `1.0000000000`), fused matches the explicit chain at
`0.9988`–`0.9999` over six steps, and exact-token agreement through the KV cache went **`95.83 %` →
`100.00 %`** (23/24 → 24/24 — one token, so read it as "did not regress" rather than as a gain).
`COSYVOICE_SDPA_DECODE=0` restores the chain.

Two things about the op are worth carrying:

**`k_chunk_size = 32` is accepted and computed wrongly below width 512.** Swept over every value
the op admits, against a torch golden:

| key width | `32` | `64` | `128` | non-power-of-2 |
|---|---:|---:|---:|---|
| 256 | `0.396` ❌ | `0.9999` | `0.9999` | raises |
| 384 | `0.293` ❌ | `0.9999` | `0.9999` | raises |
| 448 | `0.700` ❌ | `0.9999` | — | raises |
| 512 | `0.9999` | `0.9999` | `0.9999` | raises |

Non-powers-of-two `TT_FATAL` correctly, so the op *does* validate this parameter — it validates the
wrong property. Restricting `max_cores_per_head_batch` to 1 or 2 makes `32` correct at width 384
(`0.9999`), while 4 gives `0.502` and 8+ gives `0.293`: **the fault is the multi-core split of the
key axis when chunks are one tile deep**, not the chunk size itself. Anything `>= 64` is correct
everywhere tested, and that is what the model picks. Worth reporting upstream — a configuration that
passes validation and silently corrupts is worse than one that refuses.

**The mask must be built per head on the host, not with a device `ttnn.repeat` inside the traced
body.** The op is correct on its own and traces perfectly — four replays bit-identical to untraced —
but a `repeat` as the per-step input to a replayed trace took traced-vs-untraced from `1.0` to
`0.918`. The mask is rebuilt on the host every step anyway, so emitting it already expanded removes
the op rather than working around it.

**No `1/scale` pre-division, deliberately.** The kernel computes `softmax((QK^T + M) * scale)` —
`sdpa_flash_decode.cpp:378` fuses `QK += MASK` into the matmul and `:435` scales after — so a term
meant to land after the scale would need pre-dividing. This mask is binary, `0` or `NEG_INF`, and
both survive scaling unchanged in effect. A *soft* bias would need the division.

**Trace capture is worth 2.54× on the AR decoder** (20.96 → 8.26 ms/token) and **1.09× on the flow
decoder** (1.151 → 1.053 s at the time it was measured). That gap was the first finding: tracing
buys back *dispatch* overhead, so it pays in proportion to how dispatch-bound a stage already is.

### The largest matmul in the decoder did not depend on the token

`linear_pos(pos_emb)` projects `2·max_len − 1 = 511` rows through `[1024, 1024]`, where q, k and v
each project **one** row — about 536 MFLOP against roughly 1 MFLOP apiece. And `positional()` hands
back the same cached tensor on every decode step, because the window is a function of `max_len`,
which is fixed for an utterance. The decoder was recomputing an identical result 164 times per
utterance, inside the trace.

Hoisting it, plus three op removals on the same path, took the step **15.71 → 8.25 ms** and
throughput **63.6 → 121.3 tok/s**:

| change | ops removed / layer | effect |
|---|---:|---|
| cache `linear_pos` head-split transpose | 3 | the bulk of it |
| `rel_shift` → one slice at `T = 1` | 6 | seven ops become one; identity only at `t1 = 1` |
| `transpose_b` on the score matmul | 1 | bit-exact vs permute + matmul |
| `scale_mask_softmax` on the decode mask | 2 | decode mask only — see below |

`scale_mask_softmax` accepts a `[B, 1, 1, W]` padding mask and raises `TT_FATAL` on a square causal
one with or without `is_causal_mask`. That is exactly the split between the decode path (mask
`[1, 1, 1, 256]`) and the prefill/text-encoder path (causal `[1, 1, T, T]`), so the fusion lands on
the path that runs per token and skips the one that runs per utterance.

### Fusing QKV pays in one stage and not the other

q, k and v project the same activation, so they can be one matmul over a concatenated weight plus
`split_query_key_value_and_split_heads`. Applied to both stages, it measured:

| stage | before | after |
|---|---:|---:|
| flow decoder (T ≈ 600, batch 2, 64 blocks × 10 steps) | `1.075 s` | **`0.719 s`** |
| AR decode step (T = 1) | `8.29 ms` | `8.31 ms` |

Same change, opposite outcomes. The split op physically rearranges the fused row into three
head-major tensors, and at `T = 1` that costs about what the two matmuls it removed did. **Op count
is a proxy for cost, not the cost** — the flow's numbers agree with it and the decoder's do not.

The flow also folds `1/sqrt(d_head)` into the q half of the fused weight on the host, deleting a
device `multiply` per block, and uses `concatenate_heads` for the merge.

### What the decode step is actually bound by

Three measurements, each of which closed off a line of attack:

**A per-op floor of ~6.3 µs, flat in tensor size** (`scripts/probe_op_floor.py`, a traced chain of
elementwise adds):

| shape | µs/op |
|---|---:|
| `[1, 1, 1024]` | `6.3` |
| `[1, 16, 1, 64]` | `6.3` |
| `[1, 1, 4096]` | `6.4` |
| `[2, 608, 512]` | `12.4` |

A 622 K-element tensor costs twice a 1 K-element one. There is a fixed per-program cost that trace
replay does not remove, so a ~330-op decode step carries **~2.1 ms of irreducible overhead** inside
its 8.25 ms.

**`bfloat8_b` weights measure 1.00×, twice.** First at 27 GB/s effective, then again after the work
above at 42 GB/s — `8.77` vs `8.77 ms`, PCC `0.9997040033`. Halving weight traffic changed nothing at
either operating point, which rules out DRAM bandwidth as the constraint. Kept as a memory option
(352 MB → 176 MB), not a speed one.

**Explicit core grids are worse than the default** (`scripts/probe_matmul_config.py`). Per-matmul,
traced:

| linear | default | best explicit grid |
|---|---:|---:|
| `1024 × 3072` | `43.2 µs` | `46.0 µs` (4×8) |
| `1024 × 1024` | `32.4 µs` | `44.3 µs` (8×8) |
| `1024 × 4096` | `50.0 µs` | `47.9 µs` (4×8) |
| `4096 × 1024` | `75.5 µs` | `49.3 µs` (4×8) |

TTNN's default choice already reaches 65–168 GB/s on individual matmuls. More importantly the four
linears total **201 µs per layer — 2.82 ms across 14 layers, only 34 % of the step**. The weights
were never the bottleneck; the remaining ~280 non-linear ops are, at ~19 µs each against the 6.3 µs
floor.

### Flash attention, where the model allows it

The estimator's self-attention is **plain SDPA** — no mask, no relative-position term —
so unlike the AR decoder it takes `ttnn.transformer.scaled_dot_product_attention` as a
drop-in. The score matrix it stops materialising is `[2, 8, 282, 282]`: ~2.5 MB written
and read back per block, 64 blocks × 10 Euler steps.

| | explicit chain | fused SDPA |
|---|---:|---:|
| flow decoder | `0.707 s` | **`0.600 s`** |
| end-to-end RTF | `0.647` | **`0.611`** |
| `solve_euler` PCC | `0.9992047752` | **`0.9993701398`** |
| flow tokens → mel PCC | `0.9992029011` | **`0.9993962895`** |
| CFM estimator, first / last step | `0.9998326979` / `0.9991904460` | **`0.9998480374` / `0.9994887951`** |

Faster *and* more accurate on every gate, components included — which is the reason it
ships on by default while the HiFi2 experiment below did not. `scale=1.0` because
`1/sqrt(d_head)` is folded into the q half of the fused weight; SDPA's own default would
scale twice. `COSYVOICE_SDPA=0` restores the explicit chain.

### The KV-cache shift costs what it does because of tile layout, not bytes

Pricing the decode layer op by op (`scripts/probe_decode_profile.py`) puts KV maintenance well ahead
of everything else: `slice` + `concat` on the `[1, 16, 256, 64]` buffer at ~228 µs against 19–64 µs
for every other non-linear op. **0.5 MB moved in 134 µs is about 3.7 GB/s** — two orders below what
a copy that size should cost, so the bytes are not the explanation.

They are not. `scripts/probe_kv_alignment.py` isolates it:

| operation on the `[1, 16, 256, 64]` cache | bfloat16 | bfloat8_b |
|---|---:|---:|
| slice from row 1 *(the current shift)* | `78.3 µs` | `78.3 µs` |
| slice from row 32 *(tile-aligned)* | **`7.0 µs`** | `6.9 µs` |
| concat 255 + 1 row *(the current append)* | `207.4 µs` | `207.2 µs` |
| concat 224 + 32 rows *(tile-aligned)* | **`13.1 µs`** | `13.0 µs` |

**11× and 16× cheaper at tile granularity, and `bfloat8_b` — half the bytes — is identical to the
last decimal.** In `TILE_LAYOUT` rows live in 32-row tiles, so slicing from row 1 or appending a
single row re-tiles the entire buffer. This is a layout cost wearing a bandwidth cost's clothes.

### The fix: put time on a free axis

`TILE_LAYOUT` tiles the **last two** dimensions. A `[1, h, T, d_k]` cache therefore puts time on a
*tiled* axis, and that is the whole story — appending one token re-tiles the buffer. Moving the
buffers to time-major `[1, T, h, d_k]` puts time on a free axis:

| | µs |
|---|---:|
| slice + concat on the tiled time axis *(was)* | `207.2` |
| slice + concat on a free time axis | **`19.7`** |
| permute back to `[B, h, T, d_k]` for the matmuls | `13.9` |

**Paying a 13.9 µs permute to append on a free axis is 6.2× cheaper than appending on the tiled
one.** Nothing else changes: same shapes into the matmuls, same relative-position geometry, same
single trace.

    traced decode step   8.26 → 6.73 ms   (121.4 → 148.5 tok/s)
    trace speedup        2.54× → 3.10×
    end-to-end RTF       0.610 → 0.533

Traced vs untraced stays bit-exact (PCC `1.0000000000`, max|Δ| `0.000e+00` on all eight steps) —
the test that would catch a cache written back one row out.

This is the cheap route to what `ttnn.update_cache` offers. That op writes a slot in place at
**3.7 µs against 207.2** — 56× — but needs 32 pre-captured traces, one per sub-step, because the
write index and the positional slice offset both bake at capture. A change of axis order gets 6.2×
with none of that.

### Why it stops here, and what would move it

The obvious fix — shift a tile at a time instead of a row — collides with the relative-position
attention. A 256-wide window with the newest token at the end forces a one-row move per step: any
scheme that keeps the query at a fixed physical row must move the whole window with it, and any
scheme that lets it drift needs a per-step slice offset, which a trace bakes at capture.

Padding the window to 288 with a 32-row staging block *does* keep every op tile-aligned and every
shape fixed, and the validity mask can suppress the duplicates — but it breaks the geometry. A key
at physical row `j` is assigned relative distance `287 − j`, while its true distance is `256 + i − j`
at sub-step `i`; the older context is then encoded at a distance up to 31 positions too far. That is
an approximation, not a refactor, and this model's token-agreement gate already sits at 95.83 %
against a 95 % bar.

The exact version needs **32 pre-captured traces** (one per sub-step, each baking its own offsets)
and **32 pre-rotated positional windows per layer** — about 264 MB. That is a real path and the
measurements above price it, but it is a different size of change from everything else here.

**Math fidelity is free in time and is already at its best setting.** `COSYVOICE_FIDELITY` sweeps it:
HiFi4, HiFi2 and LoFi all measure RTF `0.646`/`0.646`/`0.649` — so neither stage is MAC-throughput
bound. Accuracy is not flat: HiFi2 looked better on the two end-to-end flow numbers but is worse on
**9 of 11** modules, including AR prefill (`0.9997530373` → `0.9987304709`); the flow's end-to-end
gain was error cancellation over components that individually got worse. HiFi4 stays.

This paragraph used to read: *"neither is reachable by further op-level fusion on this decomposition.
What would move it is a fused attention kernel (new C++, outside this bring-up's scope)."* The first
half was right and the second was wrong in a way worth leaving on the record — **the fused attention
kernel existed already**, and the section above is what it was worth. `RTF < 0.5` is met at `0.477`,
or `0.449` with the in-place cache.

`RTF < 0.2` needs `0.654 s`, which at 164 tokens is under `1.5 ms` per token for the LLM with the
flow's `0.589 s` already consuming `0.180` of the budget on its own. That one is **not** reachable by
op-level work: it needs the flow decoder to cost a fraction of what it does, or batching across
utterances, which single-utterance TTS does not offer.

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
| flow: tokens → mel | `0.9993962895` |
| whole HiFT vocoder | `0.9996373743` |
| LLM AR prefill, 209 tokens | `0.9997530373` |
| LLM AR decode step | `0.9989617190` |
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

## Generation modes

All four modes run on device across five languages — 20 cases, all synthesising:

| mode | prompt | on device |
|---|---|---|
| zero-shot | reference audio | ✅ 5/5 |
| cross-lingual | reference audio, different language | ✅ 5/5 |
| SFT | speaker id, no prompt audio | ✅ 5/5 |
| instruct | speaker id + description, no prompt audio | ✅ 5/5 |

The two modes with no prompt audio needed three fixes in the flow stage, all guarded by
a length that is only ever zero without a prompt: two zero-length `ttnn::concat` calls
(which **segfault rather than raise**) and a full-extent `ttnn.slice` that aliases its
input, so the `deallocate` after it freed the tensor being returned.

Per-case RTFs from those sweeps are **cold-cache** figures — every distinct sequence
length is a fresh JIT compile and the sweep path is not traced — and are not comparable
to the `0.611` benchmark above.

## Speech quality — 5 languages, 2 modes

Scored with whisper `large-v3`; CER for CJK, WER for English. The two prompt-based modes
are the ones with a reference recording to score speaker similarity against.

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
| host | 111 | none |
| device | 44 | Blackhole `p150a` |
