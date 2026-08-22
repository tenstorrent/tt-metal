# Model performance and accuracy

Performance and accuracy numbers for CosyVoice-300M, collected from direct pytest runs in
`models/demos/cosyvoice/tests/`. This is the single source for both — every figure below was
measured on the hardware named in *Environment*, none is an estimate, and none is `xfail`-ed. Where
a target is missed the number is stated with the reason and the identified lever.

## Environment

Both architectures, same tt-metal commit. Headline figures below are Blackhole; the Wormhole
side-by-side has its own section.

| property | Blackhole `p150a` | Blackhole `p150b` | Wormhole n300 |
|---|---|---|---|
| Form factor | single card | Quietbox, 4 cards | T3000, 4 cards |
| Devices | 1 × 32 GB | 4 × 32 GB, one used | 4 × 12 GB, one used |
| Compute grid | **13 × 10 = 130** | **13 × 10 = 130** | **8 × 8 = 64** |
| Host | 16 cores, 62 GB | 32 cores, 512 GB | 32 cores, 512 GB |
| tt-metal | `b5e9cba196` | `b5e9cba196` | `b5e9cba196` |
| Date | `2026-08-06` | `2026-08-06` | `2026-08-06` |

The two Blackhole boards are the **same silicon and differ in cooling** — `p150a` is actively
cooled, `p150b` passively. Both report `Arch.BLACKHOLE` with the same 13×10 grid, and the passive
board measures **~5 % slower per token** on identical work (`7.07` vs `6.73 ms` explicit; `5.87` vs
`5.58` fused): the active cooler sustains a higher clock. Accuracy is unaffected — PCC matches to
ten digits across both.

That 5 % is the same order as several real optimisations in this document, so **`p150a` and `p150b`
are separate columns and neither backfills the other's missing cells.** `p150a` is the headline
part.

**Three sets of measurements postdate that commit** and are marked where they appear, because the
table's claim is that one commit produced everything under it and quietly widening that is how an
environment table stops meaning anything:

- the `ttnn.cumsum` occupancy measurement below (`2026-08-13`, branch head `23d1e63aa85`);
- **the `p150a` RTF rows (`2026-08-18`, branch head `384c7c6504f`)**. That board was unreachable from
  `2026-08-06` until `2026-08-18`, so its figures had gone stale by a GroupNorm rewrite, a Wormhole
  convolution fix and the vocoder trace work. They are re-measured on the current tree; every other
  row is unchanged.
- **the Wormhole re-verification and `COSYVOICE_KV_INPLACE` default switch (`2026-08-18`, same
  branch head)**. n300 was also unreachable for part of that window; see *Wormhole re-verified*
  below for what was re-measured and why.

## Summary metrics

Measured on the captured utterance: 164 generated tokens producing 3.27 s of audio at 22 050 Hz.

| Metric | Value | Target |
|---|---:|---:|
| **End-to-end RTF, best measured** (`p150a` + `COSYVOICE_FF2_GRID=8x2`) | **`0.354`** | `< 0.5` ✅ |
| End-to-end RTF, `p150b`, everything on | `0.365` | `< 0.5` ✅ |
| End-to-end RTF, `p150a` default | `0.377` | `< 0.5` ✅ |
| End-to-end RTF, `p150a` + `COSYVOICE_KV_INPLACE=1` † | `0.449` | `< 0.5` ✅ |
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

† The `COSYVOICE_KV_INPLACE` row alone still predates the GroupNorm rewrite and the Wormhole
convolution fix; it has not been re-run since that board came back. The two rows above it were
re-measured on `2026-08-18` at branch head `384c7c6504f` — **`p150a` default moved `0.477 -> 0.377`**,
a 21 % gain that had been sitting unrecorded while the board was unreachable. That is what the
footnote was for: it kept a stale figure from reading as current and named what was missing, so the
gap closed the day the hardware returned.

**The single Wormhole test failure is gone.** `test_device_streamed_matches_non_streamed` scored
mel-space PCC `0.218` on n300 against a `0.85` gate until now; it was a `ttnn.conv1d`
defect, not a streaming one — see *A Wormhole convolution defect* below. Both architectures now
pass every device test.

### RTF breakdown

Blackhole `p150a`, default settings — the configuration the rest of this document's narrative is
built on. The best measured configuration is `p150a` with `COSYVOICE_FF2_GRID=8x2`; see *Blackhole
and Wormhole side by side* for the cross-architecture comparison.

| Stage | Cost | RTF | Share |
|---|---:|---:|---:|
| LLM (14-block AR decoder, traced, fused attention) | `5.71 ms/token × 164` | `0.286` | 76 % |
| Flow decoder (10 Euler steps, traced, SDPA) | `0.253 s` | `0.077` | 21 % |
| HiFT vocoder | `0.045 s` | `0.014` | 4 % |
| **Total** | `1.234 s` | **`0.377`** | |

**The LLM is now 76 % of a `p150a` utterance at default settings, up from 59 %.** The flow stage more than halved
(`0.589 -> 0.253 s`) between the CFM trace cache and the permute-free GroupNorm, and the vocoder came
down with it, while the decode step did not move. Whatever is left to win is in the decode step, and
it is a larger share of the total than when this document started.

**`COSYVOICE_KV_INPLACE=1`** writes the KV cache in place with `ttnn.update_cache` instead of
rebuilding it, taking the decode step to `4.98 ms` (`200.8 tok/s`) and the total to `1.470 s`,
**RTF `0.449`**. This is the Blackhole account; *Decode step* carries the per-architecture figures,
and *Where each change is worth what* explains why the trade runs the other way on Wormhole. It is opt-in because it costs two things the default does not: a 384 MB trace
region for the 65 traces it captures, and bit-exactness — worst PCC `0.9986` over 72 steps against
the moving cache's exact `1.0`, non-accumulating. The width it needs has to keep the key axis on an
**even tile count**; a one-tile scratch zone made it *slower*. *Decode step, and what each change
is worth* carries that account.

**`COSYVOICE_FF2_GRID=8x2`** hands the FFN's second linear an explicit 16-core grid during decode,
taking the step to `5.26 ms` (`190.0 tok/s`) and the total to `1.160 s`, **RTF `0.354`** — measured
A/B on one board in one session, one environment variable apart.

That op is the largest in a decode step, and it is bound by its `K = d_ff` reduction rather than by
weight traffic: `w_1` holds an *identical* number of weight bytes and responds to `bfloat8_b` weights
by `−37 %` where `w_2` responds by `−2 %`. So the lever is parallelism over the reduction, and it
runs the counter-intuitive way — at one output row, spreading a 4096-deep reduction across the whole
grid leaves each core a sliver and the gather dominates, so **fewer cores is faster**. Standalone,
`[1,1,4096] x [4096,1024]` at bf16: `8x2` is `1.98×` the default on `p150b` and `1.50×` on n300,
while `4x8` — the same 32 cores, transposed — manages only `1.15×` on n300. Grid *shape* matters
independently of grid *area*.

Three things about it are worth stating plainly:

- **It applies at `T == 1` only.** The optimum is a property of `M = 1`; prefill runs the same linear
  at `M = 209`, where a 16-core grid would be a pessimisation. The flow and vocoder stage timings are
  bit-identical across the A/B, which is the evidence the guard holds.
- **It is free on accuracy, as far as it has been checked.** `bf8-alone` and `bf8-plus-grid` measure
  identically to ten digits, so the grid changes scheduling and not arithmetic. That A/B is the whole
  of the accuracy evidence for this flag: `tests/pcc/` has never been run with it on, and
  `test_device_end_to_end_rtf` does not close the gap — its only assertion is `total_s > 0`, which
  makes it a timing harness, not a correctness gate. **That gap is open.**
- **It is opt-in because the best shape is not portable**, not because it is risky. `8x2` is a good
  choice on both parts measured, but a default tuned on one architecture and mediocre on another is
  worse than a flag that says so. The Wormhole figure is smaller and has its own account in
  *Wormhole re-verified*; *Tensor parallelism* records why it does not stack with TP.

RTF has come down **1.096 → 0.354** (and **2.120 → 0.354** since before either stage was traced),
and **the `< 0.5` target is met**. The rest of this section is the account of where the time went,
because the last step of it was the one that had been ruled out on a false premise.

## Reproducing these numbers

```bash
pytest models/demos/cosyvoice/tests/perf/test_pipeline_perf.py -v -s   # end-to-end RTF
pytest models/demos/cosyvoice/tests/perf/test_trace.py          -v -s   # trace speedup
pytest models/demos/cosyvoice/tests/perf/test_llm_perf.py       -v -s   # decode throughput
```

Those three suites are the reproducible set — every headline figure in this document comes
from one of them. `scripts/` also holds 34 exploratory probes; the nine cited below name
themselves where their measurement appears, and are run directly:

```bash
python3 models/demos/cosyvoice/scripts/probe_op_floor.py        # per-op dispatch floor
python3 models/demos/cosyvoice/scripts/probe_matmul_config.py   # explicit core grids
python3 models/demos/cosyvoice/scripts/probe_kv_alignment.py    # KV slice/concat by tile alignment
python3 models/demos/cosyvoice/scripts/probe_decode_profile.py  # decode layer, op by op
python3 models/demos/cosyvoice/scripts/probe_flow_steps.py      # solver depth vs cost and PCC
python3 models/demos/cosyvoice/scripts/probe_flow_ops.py        # flow block classes, untraced
python3 models/demos/cosyvoice/scripts/probe_flow_ops_traced.py # the same, traced -- see below
python3 models/demos/cosyvoice/scripts/probe_cfm_reuse.py       # CFM trace reuse across utterances
python3 models/demos/cosyvoice/scripts/repro_conv1d_wormhole.py # the Wormhole conv1d defect
```

They are exploratory and carry no gates. A probe that contradicts a suite result means the
probe is wrong or measuring something else — *The estimator's largest cost was a GroupNorm*
is the worked example of how that goes wrong.

## Tuning flags

Everything ships at a default that was measured, not assumed. Defaults are read from the
code, not from this document; each row names the section that carries the measurement.

| flag | default | what it does | what it is worth |
|---|---|---|---|
| `COSYVOICE_KV_INPLACE` | follows `device.arch()` — on for Wormhole, off for Blackhole | writes the KV cache with `ttnn.update_cache` instead of rebuilding it | `1.42×` on the n300 step, `1.12–1.15×` on Blackhole; costs a 384 MB trace region and bit-exactness (worst PCC `0.9986` over 72 steps). *Decode step* |
| `COSYVOICE_FF2_GRID` | unset | explicit core grid for the FFN's second linear during decode, `T == 1` only | `8x2`: RTF `0.377 → 0.354` on `p150a`, `0.559 → 0.557` on n300. *RTF breakdown* |
| `COSYVOICE_SDPA_DECODE` | `1` | fused `sdpa_decode` for the AR decoder's relative-position attention | `−17.1 %` on the Blackhole step, `−11.0 %` on Wormhole. *The decode attention is expressible as flash attention* |
| `COSYVOICE_SDPA` | `1` | fused SDPA in the flow estimator | flow `0.707 → 0.600 s`, and more accurate on every gate. *Flash attention, where the model allows it* |
| `COSYVOICE_CFM_TRACE_CACHE` | `1` | keeps the CFM estimator trace across utterances of the same mel length | `1.81×` on the flow stage (`p150b`), `1.50×` (n300). *Half the flow solve was trace capture* |
| `COSYVOICE_GN_PERMUTE` | unset (matmul form) | restores the permute-based GroupNorm | the matmul form is `1.41×` / `1.34×` on the stage. *The estimator's largest cost was a GroupNorm* |
| `COSYVOICE_FLOW_STEPS` | `10` | Euler solver depth | 5 steps buys `1.43×` at PCC `0.9825` — below every gate here. *Half the flow solve was trace capture* |
| `COSYVOICE_FIDELITY` | `HiFi4` | math fidelity for the matmuls | free in time; HiFi2 is worse on 9 of 11 modules. *Why it stops here* |
| `COSYVOICE_HIFT_TRACE` | unset (per-stream heuristic) | forces vocoder trace capture on or off | effect never isolated. *Wormhole re-verified* |
| `COSYVOICE_WEIGHT_BF8` | `0` | `bfloat8_b` decoder linear weights | `1.00×`, measured twice: a memory option (352 → 176 MB), not a speed one. *What the decode step is actually bound by* |
| `COSYVOICE_FLOW_BF8` | `0` | `bfloat8_b` flow-estimator weights | carries its own measurement rather than inheriting the decoder's verdict |
| `COSYVOICE_FP32_ACC` | `1` | fp32 accumulation in the vocoder convolutions | off *moves* the Wormhole `conv1d` bad-length band rather than closing it. *A Wormhole convolution defect* |
| `COSYVOICE_CONV_PREPARE` | unset (per-geometry verification) | overrides the prepared-weight verdict either way | disabling preparation outright cost the flow stage `0.683 → 1.723 s` on n300. *A Wormhole convolution defect* |

Two of these are opt-in because the best setting is not portable rather than because they
are risky — `COSYVOICE_FF2_GRID` and, on Blackhole, `COSYVOICE_KV_INPLACE`. A default
tuned on one architecture and mediocre on another is worse than a flag that says so.

## The LLM decode step

76 % of an utterance on `p150a`, and the stage where whatever is left to win now sits. The
sections below are in the order the work landed.

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
(352 MB → 176 MB), not a speed one: `COSYVOICE_WEIGHT_BF8=1` on the decoder's linears,
`COSYVOICE_FLOW_BF8=1` on the flow estimator's, which carries its own measurement rather than
inheriting this verdict.

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
kernel existed already**, and the section above is what it was worth. `RTF < 0.5` was met at `0.477`
when this was written; it now stands at `0.377`, or `0.354` with `COSYVOICE_FF2_GRID=8x2`.

`RTF < 0.2` needs `0.654 s`, which at 164 tokens is under `1.5 ms` per token for the LLM. The flow
stage has since more than halved — `0.589 -> 0.253 s`, `0.077` of RTF — so it no longer consumes the
budget on its own the way it did, and the whole of `< 0.2` now rests on the decode step. That one is **not** reachable by
op-level work: it needs the flow decoder to cost a fraction of what it does, or batching across
utterances, which single-utterance TTS does not offer.

**The per-token tail outside the traced step is 0.352 ms — 2.7 %** — and its breakdown is what
settled the on-device sampling question:

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

## The flow decoder

21 % of an utterance on `p150a`, down from 48 % — the stage that more than halved
(`0.589 -> 0.253 s`) between the CFM trace cache and the permute-free GroupNorm.

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

### Half the flow solve was trace capture

Sweeping the solver depth (`probe_flow_steps.py`) shows the flow stage is **not linear in
`n_timesteps`**, which the obvious arithmetic assumes it is:

| steps | s | ms/step | vs 10 | PCC vs 10 steps |
|---:|---:|---:|---:|---:|
| 10 | `0.7081` | `70.8` | 1.00× | `1.00000000` |
| 8 | `0.6297` | `78.7` | 1.12× | `0.98979415` |
| 5 | `0.4942` | `98.8` | 1.43× | `0.98253800` |
| 4 | `0.4557` | `113.9` | 1.55× | `0.96111237` |
| 3 | `0.4181` | `139.4` | 1.69× | `0.73646301` |
| 1 | `0.3861` | `386.1` | 1.83× | `0.22612770` |

Fitting it gives **`T(n) ≈ 0.350 s + 35.8 ms per step`**. Halving the solver buys 1.43×, not 2×,
and *deleting* it buys only 1.83× — so **solver depth is not the lever**. It is also not an
acceptable trade: 5 steps costs PCC `0.9825` against the shipped 10-step result, below every gate
this port holds, with a cliff to `0.7365` at 3 steps. `COSYVOICE_FLOW_STEPS` exists for anyone who
wants to take that trade knowingly; nothing here does.

The fixed `0.350 s` is where the finding is. `solve_euler` called `_capture()` and `_release()` on
**every** call, so the estimator trace was recorded and thrown away each time. Timing the phases
directly agrees with the fit to within 10 ms:

| phase | s | share |
|---|---:|---:|
| trace capture | `0.3144` | **46.6 %** |
| replay, 10 Euler steps | `0.3570` | 52.9 % |
| release | `0.0032` | 0.5 % |

**Keeping the trace across utterances of the same mel length is worth 1.67× on the solver**
(`0.601 → 0.359 s` steady state) and takes Wormhole end-to-end from `0.736` to `0.628`.

Reuse turns on one detail: the trace bakes `_packed_const`'s **address**, and that buffer holds the
utterance's conditioning. Refilling it in place is correct; reassigning it would leave the replay
reading the *previous* utterance's conditioning — fluent audio in the wrong voice, with no exception,
no shape mismatch, and nothing a per-module PCC against a single golden would catch.
`probe_cfm_reuse.py` is built to catch exactly that: three consecutive solves with different
conditioning, cached against uncached, compared solve by solve. **PCC `1.0000000000` on all three.**
`COSYVOICE_CFM_TRACE_CACHE=0` restores the old behaviour.

### The estimator's largest cost was a GroupNorm, and only a traced profile could see it

`probe_flow_ops.py` times each block class of the flow estimator at the shapes the real forward
uses, with a `synchronize_device` around each. It answers the wrong question, and usefully so:

| | `p150b` | n300 |
|---|---:|---:|
| whole estimator, untraced | `85.80 ms` | **`70.80 ms`** |
| transformer block @ T=141 | `0.707` | `0.601` |
| resnet 256→256 @ T=141 | `2.331` | `1.925` |

**Wormhole is faster untraced, on every block class**, while the traced stage is `1.82×` slower.
Both are true: untraced this model is host-dispatch-bound and the accelerator barely participates,
exactly as *Decode step* above records for the AR loop. A per-call wall time measures the host. The
same trap invalidates the earlier evidence that the estimator's CFG batch-2 costs `0.90×` batch 1 — that
was also an untraced measurement, so it says the two batches dispatch the same number of ops, not
that the second row is free. (That conclusion survives for its other reason: the fabric would not
initialise.)

`probe_flow_ops_traced.py` captures each block class in its own trace and divides. That profile
names something no matmul or convolution explains:

| inside one resnet block, traced | `p150b` | n300 |
|---|---:|---:|
| conv1d k3 256→256 @141 | `0.0320 ms` | `0.0556 ms` |
| **groupnorm(8) @141** | **`0.2197 ms`** | **`0.3809 ms`** |
| mish @141 | `0.0076 ms` | `0.0079 ms` |

**GroupNorm costs ~7× the convolution it follows.** 33 of them run per Euler step — two per ResNet
block, one in the final block — putting them at ~36 % of the whole estimator. Untraced it looks
ordinary (`0.4465` against the conv's `0.3383` on n300), which is why it went unexamined for the
whole bring-up.

The cost is not the statistic, it is the route to it. `TtGroupNorm` reshaped `[B, T, C]` to
`[B, G, T·C/G]` through two `permute`s so `layer_norm` could reduce over the last axis; under
`TILE_LAYOUT` those permutes swap the tiled row axis, which is a re-tiling shuffle rather than a
view, and the intermediate's tiled face is `8 × 32` — one tile carrying 8 useful rows out of 32.

The same statistic without changing shape: each group's channel sum is a **matmul against a
`[C, G]` indicator**, what remains is a reduction over `T` (an axis that needs no re-tiling), and
normalise-plus-affine folds into one multiply and one add.

| | `p150b` | n300 | PCC vs torch |
|---|---:|---:|---:|
| `[2, 141, 256]` permute → matmul | `0.2190` → `0.0940` (**2.33×**) | `0.3805` → `0.1839` (**2.07×**) | `0.99998885` |
| `[2, 282, 256]` permute → matmul | `0.3975` → `0.0978` (**4.06×**) | `0.6656` → `0.2000` (**3.33×**) | `0.99999225` |

The matmul form is nearly **independent of T** where the permute form doubles with it, which is the
re-tiling cost showing itself directly. On the stage, A/B-ed: flow `1.41×` on `p150b` and `1.34×` on
n300 — see *Where the time goes* for the runs. `COSYVOICE_GN_PERMUTE=1` restores the old form.

TTNN's native `ttnn.group_norm` is not an alternative here and the reason has changed: `estimator.py`
recorded it as less accurate, but at `G=8` it **rejects these shapes outright** on both parts.

**The matmul form needed a variance clamp, and the PCCs above did not catch why.** Computing the
statistic this way leaves variance as `E[x^2] - E[x]^2`, which is catastrophic-cancellation-prone:
where the true variance is small against the mean's square, bfloat16 rounding drives it negative,
`rsqrt` returns `Inf` without raising, and the mel comes out full-spectrum clipped noise — 22 795
`Inf` values in a 50 560-element tensor on the first real zero-shot utterance. `ttnn.relu` on the
variance before `eps` closes it, the guard every framework uses for this formula. **The two PCCs in
the table are measured on one fixed golden geometry, which never lands in the cancellation regime**,
so they are evidence for the transform's arithmetic and not for its robustness across shapes — the
distinction that let this ship. The timings above predate the added `relu` by one op per GroupNorm,
33 per Euler step; they have not been re-measured since.

## The vocoder

4 % of an utterance, and the part that was supposed to be the hard one. The inverse
transform is negligible; the op standing in for a missing `ttnn.conv_transpose1d` is not.

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

### A Wormhole convolution defect, and the streaming failure it was causing

An earlier run recorded `test_device_streamed_matches_non_streamed` failing on n300 at mel-space PCC `0.218`
against a `0.85` gate, passing on both Blackhole boards, and filed it as *arch-specific, in the
streaming cache path*. It is neither streaming nor cache.

The test's own diagnostic was already shouting: `RMS streamed 0.63250 / non-streamed 0.04970` —
**12.7× too loud**, against a `±0.99` clamp, so the waveform was saturated rather than merely wrong.
Bisecting inward: chunk 0 is fine and chunk 1 is not → of the three carried caches only `hift_mel`
changes it → but a **known-good mel** vocoded at chunk 1's length fails identically, so the cache's
contents are innocent and its *length* is the variable (prepending 20 frames takes the chunk from
110 to 130 mel frames) → the NSF source branch → a Snake activation returning `inf` → the
convolution feeding it, whose input maxes at `1.46` and whose output reaches `1.58e38`.

`ttnn.conv1d` gives two different answers there depending on whether its weight went through
`ttnn.prepare_conv_weights` first. `scripts/repro_conv1d_wormhole.py` reproduces it with a random
weight and a random input, no model involved — `Conv1d(128 → 128, k=11, pad=5)`, bfloat16, HiFi4:

| input length | prepared | op's own preparation | torch |
|---|---:|---:|---:|
| ≤ 8192 | `9.438` | `9.438` | `9.422` |
| **8193 – 8704** | **`60` … `7e37`** | `9.438` | `9.422` |
| ≥ 8705 | `9.438` | `9.438` | `9.422` |

Blackhole: **0 of 21 lengths disagree**, on two boards. Five explanations were measured and
discarded — the input's tile padding (zeroing it changes nothing), the HiFi4 + fp32-accumulate
combination tt-metal warns about on Wormhole (`HiFi3` fails identically), fp32 accumulation itself
(`COSYVOICE_FP32_ACC=0` turns it off, which *moves* the band rather than closing it), the input data
(a different activation returns the identical wrong value), and the weight values (the prepared
weight reads back at
`max|w| = 0.1816`, exactly the stored weight, with no non-finite elements). At `HiFi2` one length
came back with **two** bad elements in a million, which is what settled it: overflow does not
produce two bad elements.

The model-side fix is not to disable preparation — the first attempt did, and cost the **flow**
stage `0.683 → 1.723 s` on n300, because `TtConv1d` is also the estimator's convolution and those
run inside a captured trace, which unprepared weights make impossible. Instead the vocoder
**verifies each geometry once**: run the convolution both ways the first time a
`(length, batch)` is seen and keep the prepared weight only where the two agree. One extra call per
geometry, amortised to nothing; the affected geometries fall back and the rest keep the fast path.
`COSYVOICE_CONV_PREPARE` overrides the verdict in either direction, so the A/B stays runnable.

That choice aged well. The vocoder can now be trace-captured too
(`TtHiFTGenerator.enable_trace()`, off by default), and this verification is a host read that a
trace cannot contain — but capture waits for a geometry's *second* sighting, so the read has always
happened before the recording starts.

Measured on n300: vocoder `0.084 → 0.077 s` (the check is free at the utterance level), and the
streamed test goes from mel-space PCC `0.218` to **`0.9024`** — matching Blackhole's `0.9019` — with
`RMS streamed 0.05173 / non-streamed 0.04934`, a ratio of `1.05×` where it was `12.73×`. The check
fires 12 times, all at `stft_frames = 8321`, all in the source ResBlocks.

## Blackhole and Wormhole side by side

Same commit, same tests, same utterance — 164 tokens producing 3.27 s of audio, on all three parts.
Wormhole is the architecture the issue names (*"N150 or N300"*), so it gets a column rather than a
footnote; the second Blackhole board is there because `p150a` became unreachable partway through and
the substitution is worth being explicit about rather than quietly making.

**Why `p150a` has gaps.** The `p150a` host stayed unavailable through 13 retries over ~25 minutes —
longer than its daily reboot window — so the CFM trace cache was measured on the `p150b` box instead.
Before any timing was taken, `p150b` was checked to report `Arch.BLACKHOLE` and the same 13×10 grid;
it does, and it runs ~5 % slower per token on identical work. **`p150a`'s numbers are kept as
measured, not replaced.** A missing cell is more honest than a substituted one.

### End-to-end RTF

164 tokens producing 3.274 s of audio. Each row adds one change to the row above it, so the column
reads top to bottom as the order the work landed.

| | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|
| explicit chain, no CFM cache | `0.533` | `0.584` | `0.950` |
| **+ fused decode attention** | **`0.477`** ✅ | `0.523` | `0.891` |
| **+ cached CFM trace** | *`0.367` projected* | **`0.436`** ✅ | — |
| **+ in-place KV** (`COSYVOICE_KV_INPLACE=1`) | `0.449`* ✅ | `0.398` ✅ | `0.628` |
| **+ permute-free GroupNorm** (and, on n300, the conv fix) | — | **`0.365`** ✅ | **`0.575`** |

**Best in this table: `0.365` on Blackhole `p150b`, `0.575` on Wormhole.** Both are superseded
elsewhere in this document — `p150a` reaches `0.354` with `COSYVOICE_FF2_GRID=8x2` (*Summary
metrics*), and the Wormhole figure was re-measured at `0.559` under the flag set that actually
produced it (*Wormhole re-verified*). `RTF < 0.5` is met on both
Blackhole boards and missed on Wormhole.

The last row is a **median over four runs on `p150b` (`0.362`–`0.368`) and six on n300
(`0.557`–`0.583`)**, not a single result. Every row above it is a single run, which was fine while
changes were worth 10–20 % but is not fine for the flow stage: it varies by ~5 % run to run on n300
(`0.480`–`0.513 s`), and the first number this row was written with — `0.557` — was the best of the
set rather than the middle of it. Where a change and the noise are the same order, one run is an
anecdote.

*The `p150a` in-place figure is measured **without** the CFM trace cache, which did not exist when
that box was last reachable — it is `0.449`, not a fully-loaded number. The `0.367` above it is the
only projection in this document: `p150a`'s flow stage scaled by the `1.81×` the cache is measured to
give on `p150b`. Everything else here was run.

### Where each change is worth what

| change | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|
| fused decode attention, on the step | `−17.1 %` | `−17.0 %` | `−11.0 %` |
| cached CFM trace, on the flow stage | — | **`1.81×`** | `1.50×` |
| cached CFM trace, on the solver alone | — | **`2.37×`** | `1.67×` |
| in-place KV cache, on the step | `1.12×` | `1.15×` | **`1.42×`** |
| trace capture, on the decode step | `3.72×` | — | `1.72×` |

Two of these are worth reading as architecture facts rather than numbers.

**The fused attention is identical across the two Blackhole boards** — `−17.1 %` and `−17.0 %` — and
smaller on Wormhole. Two independent boards agreeing to a tenth of a percent is the strongest
evidence in this document that the change does what it is claimed to do.

**The CFM trace cache pays more on the faster part**, `2.37×` against `1.67×`. Trace capture is
largely fixed host and programming work, so as the replay it is amortised against gets shorter, the
capture is a larger share of what is left. The corollary is uncomfortable and worth stating: **the
faster the silicon, the more of this stage was setup.**

**The in-place KV cache runs the other way**, `1.42×` on Wormhole against `1.12–1.15×` on Blackhole,
which is why it stays opt-in on `p150a` and — as of `2026-08-18` — **is the default on Wormhole**:
`kv_inplace_default(device)` in `tt/llm/decoder.py` follows the architecture, and
`COSYVOICE_KV_INPLACE` still overrides either way. See *Wormhole re-verified* below.

### Where the time goes, at the best setting

Fully loaded — fused attention, cached CFM trace, in-place KV and the permute-free GroupNorm:

| stage | `p150b` | n300 | n300 : `p150b` |
|---|---:|---:|---:|
| LLM (164 tokens) | `0.845 s` | `1.290 s` | 1.53× |
| Flow decoder (10 Euler steps) | `0.277 s` | `0.493 s` | 1.78× |
| HiFT vocoder | `0.068 s` | `0.080 s` | 1.18× |
| **Total** | **`1.196 s`** | **`1.884 s`** | **1.58×** |

The GroupNorm change is what moved the flow, and it was A/B-ed on both parts in one sitting with
everything else fixed rather than compared against the older rows above:

| | flow, permute form | flow, matmul form | gain | RTF |
|---|---:|---:|---:|---|
| `p150b` | `0.384`–`0.396 s` | `0.270`–`0.283 s` | **`1.41×`** | `0.398` → `0.365` |
| n300 | `0.649`–`0.672 s` | `0.480`–`0.501 s` | **`1.34×`** | `0.616` → `0.575` |

Both `permute` baselines reproduce their previously recorded stage figures (`0.375` / `0.683`),
which is what makes the comparison a measurement of the change rather than of the week.

The flow is no longer the stage carrying the worst architecture ratio — and, fully loaded, the LLM
is now **68 %** of a Wormhole utterance and **71 %** of a Blackhole one, up from ~63 %, so it is
where any further RTF work has to go. (*RTF breakdown* quotes 76 % for the same stage: that is
`p150a` at default settings, which does not carry the in-place KV cache these figures do.)

The overall ratio, `1.58×`, sits well inside the **2.03× ratio in core count** (130 vs 64), so
neither part is disproportionately hurt. The spread between stages is wider than it looks — the
vocoder is nearly architecture-neutral at `1.18×` because it is dominated by two large
`conv_transpose2d` calls, while the flow at `1.78×` is the most core-hungry stage in the model.

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

Four things in that table are worth stating rather than leaving to be read off.

**Untraced decode is the same speed on both — `20.83` vs `20.10 ms`.** Untraced, this model is
host-dispatch-bound, and the two hosts are the same class of machine; the accelerator barely
participates. The architecture gap appears only once tracing removes the dispatch overhead, which is
also why trace capture is worth 3.72× on Blackhole and 1.72× on Wormhole. **A per-op cost measured
untraced says almost nothing about the silicon.**

**The in-place KV cache is worth 1.42× on Wormhole against 1.12× on Blackhole.** It ships opt-in
because on Blackhole it buys 12 % for a 384 MB trace region and the loss of bit-exactness — a thin
trade. On Wormhole the same trade buys 42 %, which is not thin. **Recommend it on n300; keep it
optional on p150a.**

**The tile-parity effect changes sign between architectures.** An earlier Blackhole sweep had
widening the key axis 384 → 448 cost `+0.28 ms`, and it does. On Wormhole the *wider* buffer is
`1.55 ms` **faster**. Whatever the scheduling heuristic behind that is, its optimum is
architecture-specific, and a width tuned on one part is not tuned on the other.

**The fused attention pays more in absolute terms on Wormhole and less in relative terms.**
`1.37 ms` against `1.15 ms`, but 11.0 % of a slower step against 17.1 % of a faster one. The
prediction going in was that it would favour n300, because narrower memory is the one place
`bfloat8_b` ever showed a gain, and this kernel stops writing the score matrix. Half right: the
milliseconds went the predicted way, the percentage did not.

### Wormhole re-verified — the recipe held; a benchmarking mistake in between did not

`0.575` above was measured before `COSYVOICE_HIFT_TRACE` existed as a flag — the GroupNorm/conv1d fix
that produced it (`f1f27cf2322`) predates the vocoder-trace work (`63fd351d6de`) by several commits.
Re-checking it after Titan came back from a 26 h outage, a run that also set `HIFT_TRACE=1` — assuming
it was part of "everything on" by name rather than by commit history — gave `0.611`–`0.629`, ~10 %
worse, and looked at first like the hardware itself had degraded.

**It had not.** `tt-smi` telemetry showed the chip idle, cool, unthrottled and sole-tenant; the
degradation tracked one specific flag, not the board. Re-run with the flag that actually produced
`0.575` — `COSYVOICE_KV_INPLACE=1` alone, nothing else — over four runs (three explicit, one from
verifying the new default below):

| | ms/token | RTF |
|---|---:|---:|
| `0.566, 0.562, 0.555, 0.550` | `7.97, 7.94, 7.74, 7.63` | median **`0.559`** |

Within the historical `0.557`–`0.583` band. **The number was never wrong; the flag set used to check it
was.** `HIFT_TRACE` itself is not implicated by this — its own effect was not isolated here — but it is
no longer assumed to be part of the Wormhole recipe until measured as such on its own.

**`COSYVOICE_KV_INPLACE` is now the default on Wormhole**, matching what this document already
recommended (*"is worth turning on by default on n300"*, above). `kv_inplace_default(device)` in
`tt/llm/decoder.py` checks `device.arch()`; `model.py` and `test_pipeline_perf.py` both call it rather
than each re-implementing the check, and `COSYVOICE_KV_INPLACE` still overrides either direction.
Verified on both architectures with no flags set: n300 moved `0.731`(old default, moving cache)
`→ 0.550`(new default, matching the table above); Blackhole `p150a` stayed at `0.379`, unchanged from
its own already-default in-place-free measurement — confirmed by `device.arch()` correctly resolving
`Arch.WORMHOLE_B0` and `Arch.BLACKHOLE` on the two parts. `test_device_end_to_end_rtf` passes on both
with the new default active, so the PCC gates hold.

**`COSYVOICE_FF2_GRID=8x2`, at the corrected Wormhole baseline**, three runs — the Blackhole
account of the same flag is in *RTF breakdown*, and this is the second of its two measurements,
not a restatement of the first:

| | ms/token | RTF |
|---|---:|---:|
| `0.564, 0.546, 0.557` | `7.68, 7.59, 7.53` | median **`0.557`** |

`7.84 → 7.59 ms/token` median, **`0.559 → 0.557`** — real (every run favoured the grid) but much
smaller than `p150a`'s `6.1 %` end-to-end gain from the same flag. Both measurements are single-chip
at the same `K = 4096`, so this is not the TP/grid interaction below — the standalone sweep already
showed the same direction: `8x2` measured `1.50×` on n300's own `[1,1,4096]x[4096,1024]` linear in
isolation against `1.98×` on `p150b`'s. Wormhole's default grid is `8×8 = 64` cores against
Blackhole's `13×10 = 130` — half the "too many cores for one row" problem the small grid fixes is
smaller to begin with on the part with fewer cores. `RTF < 0.5` remains missed on Wormhole, now
honestly at `0.557` rather than at a figure built on the wrong flag set.

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

**It is also 6.9× faster** (measured `2026-08-13`; see *Environment*), which is not a coincidence: `ttnn.cumsum` parallelises only over the axes
it is *not* scanning (`num_rows_total = tiles / tiles_per_row`), so `[1, 72192, 9]` on `dim=1`
permutes to `[72192, 1, 1, 9]` and lands on **one core** with 72 192 serial tile-steps. Blocking gives
it 282 rows to spread across the grid. Precision and occupancy wanted the same restructuring.

| `cumsum` + `mod 1` | `p150b` | n300 |
|---|---:|---:|
| plain, one core | `40.4 ms` | `73.3 ms` |
| `phase_mod1` | `5.9 ms` | `12.5 ms` |

`BLOCK = 512` is 5 % faster again but 3.6× less accurate; below 256 the serial carry scan dominates.

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

### Per-architecture accuracy and test results

| | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|
| traced vs untraced | `1.0000000000` | `1.0000000000` | `1.0000000000` |
| in-place, worst PCC over 72 steps | `0.9987379437` | — | `0.9991855486` |
| CFM trace cache, 3 solves, new conditioning each | — | `1.0000000000` | `1.0000000000` |
| test suite | `155 passed` ‡ | **157 passed** | **157 passed** |

Traced-vs-untraced is bit-exact everywhere, and so is the cached CFM trace across three consecutive
solves with different conditioning — the test that would catch a stale `_packed_const`.

‡ `p150b` and n300 were re-run on `2026-08-13` at branch head `23d1e63aa85`, which adds two vocoder
trace tests to `tests/perf/test_trace.py` — hence `157` where the earlier run counted `155`. The
`p150a` cell is the older count on the older tree; that board has been unavailable since, so it is left
as measured rather than assumed to have gained the same two.

**Every device test now passes on both architectures.** n300 stood at 154 passed / 1 failed
until the convolution defect above was found: the failure was
`test_device_streamed_matches_non_streamed` at mel-space PCC `0.218` against a `0.85` gate, and it
now scores `0.9024` against Blackhole's `0.9019`. **Streaming is no longer Blackhole-only.**

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

## Coverage and test counts

Source suites: `tests/perf/`, `tests/e2e/`, `tests/pcc/`
- End-to-end RTF with a per-stage breakdown
- Trace capture speedup and bit-exactness
- Decode throughput, cold vs warm, growing vs fixed-shape KV cache
- Streaming content equivalence and seam continuity
- Per-module PCC against captured PyTorch goldens

157 tests in all: 111 host plus 46 device. A device run executes both tiers, so the pass
counts below are totals, not the device column.

| Tier | Count | Hardware | Result |
|---|---:|---|---|
| host | 111 | none | — |
| device | 46 | Blackhole `p150b` | **157 pass** |
| device | 46 | Wormhole n300 | **157 pass**; the streaming failure was a `ttnn.conv1d` defect, now fixed |
| device | 44 | Blackhole `p150a` | **155 pass** — the older tree, two vocoder trace tests short; see the ‡ note under *Per-architecture accuracy and test results* |

## Operational notes

**`l1_small_size` scales with conv *configurations*, not tensor size.** `ttnn.conv1d` allocates
prepared weights from that bank and keeps them, so three models live at once (~80 convs) exhausts
the 32 KB a single-model test uses — failing part-way through the *second* utterance with
`Not enough space to allocate 480 B L1_SMALL buffer`. Zero-shot needs 128 KB; cross-lingual needs
256 KB, because its prompt is 1289 mel frames against 326.

**Persist the JIT cache across runs.** Mounting `~/.cache/tt-metal-cache` took the first
utterance from `161.7 s` to `14.8 s` wall. Every distinct sequence length is a fresh compile.

## Tensor parallelism — measured, and it does not move the needle

Tried on `2026-08-18`: combining the `COSYVOICE_FF2_GRID` finding above with the tensor-parallel
decoder prototype (2-chip Megatron sharding of the AR decoder, run from a scratch probe that is
not carried in this tree). TP's own value stands on its own from an earlier session (`1.18×` on
the decode step, `PCC 0.99994`) and is unaffected by this result. The question was narrower: does the
core-grid win from earlier in this document *also* apply once TP has already sharded the same linear.

**It does not, and the reason is mechanical, not architectural.** TP shards the FFN's second linear
row-parallel: each chip's reduction is `K = d_ff / 2 = 2048` instead of the un-sharded `4096`. Sweeping
the same grid candidates against that halved `K` on an n300 T3000 (2 chips), median of 30 traced
replays per point:

| grid | ms/token | vs default |
|---|---:|---:|
| default | `5.536` | 1.00× |
| **`8x2`** | **`5.388`** | **1.03×** |
| `8x1` | `5.440` | 1.02× |
| `4x4` | `5.539` | 1.00× |
| `8x4` | `5.546` | 1.00× |
| `4x2` | `5.589` | 0.99× (slightly worse) |

At `K = 4096` (single chip, this document's opening section) the same op and the same grid candidates
gave `1.50×`–`2.11×`. At `K = 2048` the best case is `1.03×` — three percent, not a fraction of the
original effect but essentially none of it. `4x2` landing *below* the default shows the effect has a
floor: past some point there is no further "too many cores for too little work" left to fix, and
pushing the grid smaller than that just adds scheduling overhead for nothing back.

**TP and the core grid are not two independent levers — they are the same lever, applied at different
granularities.** Sharding the reduction across chips already does most of what a small grid does on
one chip: both exist to stop a one-row matmul from being spread across more parallelism than the work
justifies. Having sharded once, there is almost nothing left for the grid to shard again.

**Practically:** there is nothing to integrate. Building a shipped, mesh-aware TP path would still be
worth roughly the `1.18×` measured on the decode step in isolation — real, and, transferred to the
shipped decoder's current Wormhole baseline (`0.559` above), not enough on its own to reach `0.5`. It
would not additionally benefit from `COSYVOICE_FF2_GRID`, so the two should not be quoted as if their
gains stack. The per-grid medians above are the record; the raw sweep output is not carried here.
