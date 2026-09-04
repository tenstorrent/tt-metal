# Qwen3-TTS — Wormhole performance notes

Working notes on device-performance optimisation of the Qwen3-TTS decode path on Wormhole
(N150 = 1 chip, N300 = 2 chips at TP=2). Records what was changed, what it measured, what
was tried and rejected, and what is still on the table.

The [README](README.md) covers the Blackhole P150 path; this file is Wormhole-specific.

---

## 1. Where the time actually goes

The obvious assumption is that the 28-layer Talker (hidden 2048) dominates and the 5-layer
CodePredictor (hidden 1024) is a rounding error. It is the other way round.

One autoregressive audio frame runs the CodePredictor **15 times** — a 2-token prefill plus
13 residual decodes, one per codec group — against a single Talker decode step. That is
**75 CP layer evaluations per frame versus 28 Talker layers**.

Traced device time for one AR frame (single wormhole chip, Metal trace, `test_qwen3_tts_trace_perf.py`):

| window | device time | share |
|---|---|---|
| CodePredictor residual decodes (13x) | 42,044 us | 56 % |
| Talker decode | 30,497 us | 40 % |
| CodePredictor prefill (seq=2) | 2,987 us | 4 % |
| **full AR frame** | **75,528 us** | |

**Rule of thumb: compare per-*frame* cost (layer time x invocations per frame), never
per-layer time.** The CodePredictor is the first thing to optimise.

A second observation: in the CP decode layer, roughly 60 % of the time was *not* matmul. It
was layout churn and collectives — ops that move or reshape data rather than compute with it.

---

## 2. What was changed

### 2.1 CodePredictor N300 fast path — gated

`models/demos/qwen3_tts/tt/code_predictor.py`, gated on `mesh_utils.is_n300(device)`
(a 2-chip wormhole mesh; N150, T3K and Blackhole keep the generic path). Set
`QWEN3_TTS_CP_N300_OPT=0` to A/B at runtime.

The Talker already had sharded equivalents of all of this in `attention.py` /
`decoder_layer.py`; the CodePredictor had been written with generic ops. This ports them.

| change | before | after |
|---|---|---|
| Input + post-attention RMSNorm | 25 us each, **1 core** | 10 / 9 us, 32 / 16 cores |
| `nlp_create_qkv_heads` | 31 us, **1 core** | 2 us, 8 cores |
| `nlp_concat_heads` | 12 us, **1 core** | 0.5 us |
| TP=2 all-reduce (x2 per layer) | 107 us (2 CCL ops each) | 77 us (1 CCL op each) |

**Why the norms landed on one core:** a decode token is `[1, 1, 32, 1024]` — a single tile
row. The default RMSNorm parallelises over *rows*, so one row-block means one core while the
other 63 idle. Width-sharding splits along the hidden dim instead.

The post-attention norm emits directly in the gate/up matmul's `in0` layout, so the MLP's own
`to_memory_config` disappears.

**Sharded `nlp_create_qkv_heads` needs a permuted weight.** The sharded kernel reads a
KV-group-interleaved fused QKV (`[q..q, k, v]` per KV group), so a second copy of the QKV
weight with that row permutation is built at init (`lw["wqkv_kvgi"]`); the plain `[Q|K|V]`
copy is freed on this path.

**The 2-chip all-reduce** (`mesh_utils.tp_all_reduce_2chip`): `ttnn.all_reduce` lowers to
reduce_scatter + all_gather. On N300 both are dominated by fixed fabric setup rather than
payload — a 1-tile activation pays ~51 us to reduce 64 KB. With only two chips you can
all-gather the two partial sums and add the halves locally: one CCL op instead of two.
Same all-reduce, same maths, **same tensor parallelism** — only the lowering changes.

### 2.2 Decode-mode RoPE — NOT gated

`models/demos/qwen3_tts/tt/rope.py::apply_rope_qk`, used by both `attention.py` (Talker) and
`code_predictor.py` (CodePredictor).

`rotary_embedding_llama` with `is_decode_mode=False` **loops once per head**. Measured at
head_dim=128, seq padded to one tile:

| n_heads | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| prefill mode | 12.9 | 18.4 | 26.1 | **41.3 us** |

That is ~9 us fixed + **~2 us per head**, and it is unaffected by memory config (DRAM vs L1
cos/sin/trans) or math fidelity — both were tested. `is_decode_mode=True` rotates every head
inside a single tile: **3.4 us**.

Getting there costs a `ttnn.transpose(x, 1, 2)` — `[1, n_heads, 1, hd]` -> `[1, 1, n_heads, hd]`
— at 2 us, and one back. Transpose reads and writes the sharded layout directly, so no extra
reshard is needed on either side.

**Gate is `seq == 1`, not the device and not the `mode` argument.** Shape is the honest test:
the CP's "prefill" is only 2 tokens, so it *is* a prefill call, but it has two distinct
positions and must keep the prefill kernel. There is also a fallback for `n_heads > 32`, which
cannot pack into one tile (nothing in this model hits it; the helper is shared).

Constraints worth knowing:

- Decode mode requires **all** of Q/K, cos, sin and trans_mat to be `HEIGHT_SHARDED`.
  Interleaved and width-sharded are rejected outright.
- `get_rot_transformation_mat` **ignores its `dhead` argument** and always returns one 32x32
  tile, so prefill and decode share the same matrix — only the memory config differs.
  `get_decode_transformation_mat` builds it at module init so it predates any trace capture.
- cos/sin are reshared once and reused for both Q and K.

---

### 2.3 ECAPA SpeakerEncoder host fusion — NOT gated

`models/demos/qwen3_tts/tt/speaker_encoder.py`, env kill-switch `QWEN3_TTS_SE_HOST_FUSE=0`.

The Res2Net branch convs are k=3 **dilated with reflect pad**, which TTNN cannot express, so
they run in torch on the host. The device path put the cheap glue *between* them, so each of
the 7 branches paid a full round-trip:

```
slice(dev) -> D2H -> conv(host) -> H2D -> relu(dev) -> add(dev) -> D2H -> conv(host) -> ...
```

The block profile is unambiguous about what that costs: **30 device ops, 121 us of device
time, 24,565 us of op-to-op gap.** The 2 us ReLU carried a 3,365 us gap in front of it — that
gap *is* the D2H/conv/H2D. Device work was 0.5 % of the block.

Three fusions, all of the same shape — move the glue to the side of the fence the conv is
already on:

| | before | after |
|---|---|---|
| Res2Net glue / block | 22 ops (8 slice, 7 relu, 6 add, 1 concat) | 0 |
| Res2Net round-trips / block | 7 | 1 |
| entry TDNN (k=5) ReLU | 1 device Unary | folded into the host conv |
| ASP relu+tanh | 2 ops + 1 round-trip | folded |

The SE 1x1 convs, both TDNN matmuls, the softmax and the pooling statistics all stay on
device — that is real device work, not glue.

**Not gated by device.** This removes host<->device ping-pong, which costs the same on N150
and N300. It is gated by *shape*: the host cascade is taken only when every branch conv is
non-pointwise, so a k=1 variant would keep the device path.

Numerics: the ReLU fold is **bit-exact** (bf16 round-to-nearest preserves sign and zero, so
`relu(bf16(y)) == bf16(relu(y))` — asserted over 65k values plus +/-0 and subnormals). The
cascade is not bit-exact but is strictly *more* accurate: it removes a bf16 rounding step per
branch, so the fp32 path now runs end to end.

### 2.4 Talker decode attention — fused SDPA + decode SDPA config + layout round-trips

The Talker decode layer carried 54 device ops. Eighteen of them did no arithmetic that the
model needs, and the one real attention op ran a config built for prefill. All measured on
the **deployed** graph — `cur_pos_tensor` + a full `[1, heads, 1, kv_max]` mask, K/V written
by `paged_fused_update_cache`, attention over the whole 352-deep cache — via
`test_qwen3_tts_profile_single_layer.py -k talker_layer_decode_traced`. The pre-existing
`-k talker_layer_decode` case does **not** exercise that graph: with `cur_pos_tensor=None` it
takes the eager fallback, which slices the cache to `start_pos+1` and so runs attention over
one position instead of 352.

| # | change | ops | device time |
|---|---|---|---|
| 1 | fused SDPA on every SKU (was N150-only) | -9 | -44 us |
| 2 | padded-N QKV slice writes the `nlp_create_qkv_heads` shard spec (was N150-only) | -2 | -6 us |
| 3 | decode RoPE keeps K in the cache layout (`k_keep_decode_layout`) | -2 | ~-4 us |
| 4 | decode-shaped SDPA program config | 0 | **-55 us** |
| 5 | `scale` folded into the q_norm gain, so SDPA stops rescaling the mask | -1 | -6 us |
| 6 | SDPA writes the concat-heads shard spec directly | -1 | -1 us |
| 7 | padded-N output slice reads the width-sharded matmul output | -2 | -2 us |
| 8 | V skips the S->I it never needed | -1 | -1 us |
| | **total (median of 3 captures)** | **54 -> 36** | **552 -> 437 us** |

The four CCL rows measured 20-25 us in every one of those six captures, so none of this is
CCL luck. Per token that is 28 x 115 us = **3.2 ms** of Talker device time.

**(1) Fused SDPA everywhere.** `9b76da7abce` moved N150 off the manual fp32 BMM chain
(typecast Q/K/V, GQA `repeat_interleave`, QK / scale / mask / softmax / PV, typecast back) and
left every other card on it. On N300 that chain was 11 ops / 134 us of the layer. Fused SDPA is
GQA-native and bf16 throughout, so the expansion and the dtype hops both go.
`QWEN3_TTS_TALKER_MANUAL_SDPA=1` restores the old graph for A/B.

**(4) The decode SDPA program config — the single largest win, and free.** Decode was reusing
`sdpa_prefill_program_config` (`q_chunk_size=64`, `k_chunk_size=64`). Decode has **one** query
row, so `q_chunk_size=64` pads it to two tiles and doubles every chunk. Swept in isolation at
the model's shape (Sq=1, Sk=352, dh=128, 8 local heads), device time from the profiler CSV:

| q_chunk | k_chunk=32 | 64 | 128 | 352 |
|---|---|---|---|---|
| 64 (shipped) | 99.9 us | 81.9 us | 71.0 us | 68.2 us |
| **32** | 51.8 us | 38.6 us | 30.6 us | **26.8 us** |

`q_chunk=64, k_chunk=64` = 81.9 us reproduced the in-model 82 us exactly. Cost tracks the
*chunk-padded* KV length `ceil(S/k) * k` plus ~1.2 us per chunk — which is why k=320 (2 chunks,
640 padded rows, 60.9 us) is worse than k=128 (3 chunks, 384 rows, 30.6 us) and an exact
divisor always wins. `decode_sdpa_k_chunk()` encodes that; `k_chunk=1312` exceeds the program
size limit, so it caps at 672. Grid size is irrelevant here (8x8 38.6, 8x4 37.5, 4x4 37.9) and
so is fidelity (HiFi4 38.5, HiFi2 37.0, LoFi 36.9) — do not trade PCC for 1.5 us.

**(5) The mask rescale.** `ttnn::scaled_dot_product_attention` folds `scale` into the softmax
exponent, so its wrapper pre-multiplies the additive mask by `1/scale` on **every call**
(`sdpa.cpp`). Our mask is pure `{0, -inf}` and therefore scale-invariant, so that 6 us DRAM
pass computes nothing. Passing `scale=1.0` skips it; the softmax scale rides in the q_norm gain
instead, a load-time constant (RoPE is a rotation, so scaling Q before it is the same as
scaling the scores after). The folded gain gets its own weight-cache key so a stale cache file
cannot supply the unscaled one.

**(3, 6, 7, 8) Layout round-trips.** Four places converted a layout that the next op already
accepted:
- decode RoPE transposed K back to `[1, n_kv, 1, dh]` and attention immediately transposed it
  to `[1, 1, n_kv, dh]` — the same tensor, and `_rope_decode_memcfg(dh)` is byte-identical to
  `paged_k_input_mem_config`. Both transposes go (bit-exact; gated in
  `test_qwen3_tts_rope_decode.py::test_k_keep_decode_layout`).
- SDPA accepts a height-sharded output and `nlp_concat_heads`' input spec is exactly that.
- `ttnn.slice` reads a width-sharded input and writes any layout (verified bit-exact), so the
  padded-N trim after a DRAM-sharded matmul needs no `ShardedToInterleaved` first —
  `unpad_dram_sharded_out`.
- V's only consumer is the transpose into the paged-cache layout, and `transpose` does take
  HEIGHT_SHARDED (bit-exact). `rms_norm` genuinely does not, so Q/K keep theirs.

Not gated by device: (2)-(8) are layout/config changes with no SKU dependency, and (1) has the
env fallback. Prefill seq=32 also picks up (2): 32 -> 29 ops (its window is CCL-dominated, so
read the op count, not the time).

**Default ON, and (1) shares a risk with the CP arm that is deliberately default OFF — read
this before trusting the frame count.** A paired demo run (N300, "Hello, how are you today?",
seed 42, `--use-2cq`, same tree, only these changes differing) generated **17 -> 14 frames**;
the first three code-0 tokens are identical and it diverges at the fourth. `dccf18b66c4` saw
the same direction on the CP arm (3/3 seeds, fewer frames) and held that arm behind a flag
because n=3 could not separate a real regression from sampler chaos — 3.4 is explicit that PCC
cannot predict generation length here. Two reasons this arm still ships on:
the Talker's fused-SDPA path is **not new code** — `9b76da7abce` made it the N150 default and
listened to the result, so this extends an already-shipped path to N300 rather than enabling an
unlistened one; and the numerics move the *right* way on every gate available
(`attention_decode` 0.999785 -> 0.999786, `talker_chain` 0.974803 -> 0.975272, and the manual
chain vs fused layer A/B on N300 is PCC 0.999988). Still: `QWEN3_TTS_TALKER_MANUAL_SDPA=1`
restores the pre-fusion graph, and the honest gate is the same one 6.4 asks of the CP — a
frame-count sweep over >=8 seeds plus a listen/WER check. Items (2)-(4) and (6)-(8) are
bit-exact and carry none of this risk.

---

### 2.5 CodePredictor decode — the remaining layout round-trips, audited

A per-op audit of the traced CP frame (`ops_list/perf_report_ag_sharded/decode_cp`, 3469
device ops / 27.67 ms over one frame = 15 CP calls x 5 layers) listed eight
producer/consumer pairs that move or reshape data the next op could have read in place.
Three were removed; the other five are blocked, and the reason each is blocked is worth
more than another attempt at it.

Measured on the traced `decode_cp` window (`tests/qwen3_tts_perf_report.sh -w decode_cp`),
before -> after: **3469 -> 3249 device ops, 27.668 -> 27.252 ms** device kernel time.
Wall clock 26.88 -> 26.69 ms, but read the device column: wall swung 26.57-26.91 ms across
four captures of two builds, so it cannot resolve a 0.4 ms move.

| # | pair (count, device ms) | verdict |
|---|---|---|
| 1 | Transposes wrapping RoPE (280, 0.891) | **partly removed**: -70 |
| 2 | S2I Matmul -> AllGather (150, 0.344) | already gone in `94eeed8` |
| 3 | S2I NlpCreateHeads -> LayerNorm (150, 0.312) | blocked |
| 4 | Reshard before o_proj / before MLP down (150, 0.301) | **o_proj half removed**: -75 |
| 5 | I2S Matmul -> NlpCreateHeads (75, 0.186) | blocked |
| 6 | I2S SDPA -> NLPConcatHeads (75, 0.175) | **removed**: -75 |
| 7 | FillPad before TopK (15, 0.291) | blocked |
| 8 | S2I LayerNorm -> Matmul, QKV in0 (75, 0.095) | open, low value |

**(6) and (4a) fall together, and the lever is `nlp_concat_heads`' output spec.** The op
takes only the *layout* and buffer type from the `memory_config` you hand it; the output
**shard spec is derived from the input's**
(`nlp_concat_heads_device_operation.cpp::compute_output_specs`):

```
heads_per_shard = in_shard_h / padded_seq
out_shard       = (padded_seq, in_shard_w * heads_per_shard)   on the input's grid
```

So the concat's output grid is chosen by how many heads you pack per core on the way *in*.
The DRAM-sharded o_proj's in0 is `find_grid_k_n(K=32 tiles, N=36 tiles)` = **4 cores x 256**;
the concat was fed one head per core (8 cores x 128) and therefore emitted 8 x 128, which
needed a Reshard. Feeding it **2 heads per core on 4 cores** makes the concat emit the
o_proj in0 spec exactly — the Reshard disappears. And since SDPA writes its
`output_mem_config` verbatim, asking SDPA for that same 2-heads-per-core height-sharded
spec removes the I2S in front of the concat as well. Two ops per layer, one config change,
`-150 ops / -0.36 ms`. The Talker already did the SDPA half (`_sdpa_out_memcfg`); the
heads-per-shard half is new and applies there too (not ported — the Talker's o_proj grid
comes from a different builder and it is not measured here).

**(1) K skips its transpose back.** `apply_rope_qk(k_keep_decode_layout=True)` returns K as
the decode kernel's own `[1, 1, kv_heads, head_dim]` HEIGHT_SHARDED-on-one-core output, which
is byte-identical to what `paged_update_cache` wants — so the transpose back to
`[1, kv, 1, hd]`, which existed only to feed `ttnn.update_cache`, goes. The Talker has done
this since 2.4; the CP had not. **This is close to free, not a win:** Transpose
0.890 -> 0.669 ms (-0.221), but `paged_update_cache` costs 7.9 us against `update_cache`'s
5.3, so the cache ops go 0.751 -> 0.935 ms (+0.184). Net ~-0.04 ms and -70 ops. Kept for the
op count and because it is bit-exact (output and both K/V caches, verified against
`QWEN3_TTS_CP_K_CACHE_LAYOUT=0`); do not expect time from it.

Q's two transposes and K's inbound one — the other 210 — need the **decode-native head ops**
(`nlp_create_qkv_heads_decode` -> `sdpa_decode` -> `nlp_concat_heads_decode`), i.e. a second
attention path for `mode == "decode"` alongside the seq=2 prefill one. Not attempted here.

#### The five that are blocked, and why

**(3) QK-norm cannot take the height-sharded Q/K.** `layernorm_device_operation.cpp:166`
`TT_FATAL`s on a HEIGHT_SHARDED input outright, so the S2I is not optional. Two sharded
layouts it *does* accept were tested at the model's shape (`[1, 8, 32, 128]`, bf16):

| layout | result |
|---|---|
| BLOCK_SHARDED, 1 col x 8 rows, shard (32, 128) | runs, **max diff vs interleaved = 0** |
| WIDTH_SHARDED, 4 cores over head_dim | runs, diff 0.0625 — reduces over the wrong axis |

Block-sharded is exact (each core is its own core-row, so the reduction is over head_dim,
i.e. per head) — but it buys nothing. **This op has a ~9-11 us floor that no core count
moves:** in the same capture, RMSNorm measures 24.3 us on 1 core, 9.4 us on 8, 8.9 us on 4,
10.9 us on 32 and 11.0 us on 16. It is fixed overhead, not compute, so swapping the S2I for
a Reshard into block-sharded leaves the 0.312 ms roughly where it is. 315 LayerNorms x
~10.7 us = 3.36 ms = 12 % of the frame is essentially 315 kernel launches; the win here is
fewer norm *calls*, not a better layout for them.

**(5) The QKV matmul cannot write the split's input spec.**
`nlp_create_qkv_heads` pins the shard width exactly:
`shard_w == (num_q_heads / num_kv_heads + 2) * head_dim` = 512
(`nlp_create_qkv_heads_device_operation.cpp:133`), which for the CP's per-chip
`8q/4kv/hd=128` means **exactly 4 cores, no other grid is legal**. A matmul that wrote a
4-core width shard would be a 32x1024x2048 GEMM on 4 cores against the current 64 — far more
than the 2.5 us I2S is worth. A wider grid is not available to trade into.

**(4b) The MLP-down reshard is blocked by DRAM-shard grid arithmetic.** gate/up lands on
`find_grid_k_n(K=32, N=48)` = 16 cores, down wants `find_grid_k_n(K=48, N=36)` = 12. Forcing
gate/up to 12 fails `dram_sharded_program_config`'s `K % (TILE*cores) == 0` (32 % 12);
forcing down to 16 fails the output width shard (36 % 16), and padding down's N to 1536 to
fix that adds 33 % to a weight-bandwidth-bound matmul (~+6 us) to save a 1.7 us reshard.
0.126 ms, left alone.

**(7) The TopK fill cannot be skipped through the public API.** `ttnn.topk` calls
`fill_implicit_tile_padding` unconditionally (`topk.cpp:650`), and its early-out fires only
when the last two **logical** dims are tile-aligned. The CP runs at logical M=1 padded to a
tile, so it never fires. Nothing at the call site changes that: `ttnn.reshape`/`view` are
volume-preserving and cannot promote logical 1 row to 32, and an explicit `ttnn.pad` to 32
rows does the same write the fill does. **Placement is not the problem either** — moving the
logits from DRAM into L1 (`perf_report_cp_logits_l1`) moved FillPad 19.4 -> 18.9 us and TopK
218.1 -> 216.8 us, i.e. nothing; both are latency-bound. Reverted, comment kept at the
`lm_head` call site. The real target in this neighbourhood is TopK itself (15 calls,
217 us each, **12 % of the frame**) — see 6.x, not this fill.

**(8) Open.** The width-sharded input RMSNorm (32 cores) unshards for the QKV matmul.
Removing it needs a `MatmulMultiCoreReuseMultiCast1DProgramConfig(mcast_in0=True)` on the
LN's own 32-core grid — legal (K=32 tiles / 32 cores = 1 tile each), but it halves the
matmul's core count from 64 and that matmul is already at 56.4 % of DRAM peak. 0.128 ms at
stake against a 26 us matmul; not attempted.

#### Not on the list, found while auditing

The Gumbel noise row is sliced out of a `[1, 1, 32, 64]` tile with
`ttnn.slice(noise, [0,0,slot,0], ...)`, which is a non-tile-aligned row slice and lowers to
**Untilize -> Slice -> Tilize, 10.8 us x 14 = 0.15 ms/frame**. Storing the noise as
`[1, 32, 1, 64]` and slicing dim 1 makes every slice tile-aligned. Not done.

---

---

## 3. Measured results

### 3.1 Accuracy — no change

The two RoPE kernels are **bit-identical**, not merely close: `max|prefill - decode| == 0`
at every head count either model uses.

`tests/test_qwen3_tts_pcc.py` (real HF weights), before and after all changes:

| block | baseline | after | |
|---|---|---|---|
| `mlp_decode` | 0.999692 | 0.999692 | unchanged |
| `attention_decode` | 0.999790 | 0.999790 | unchanged |
| `cp_step` | 0.999835 | 0.999835 | unchanged |
| `talker_chain` | 0.972521 | 0.972521 | unchanged |

Identical to six decimals. The CP N300 path is bit-exact in decode and within 1.9 bf16 ULP in
prefill (the sharded RMSNorm reduces in a different order).

> **Coverage caveat.** Only `attention_decode` exercises the decode RoPE path. `cp_step` is CP
> prefill at seq=2, and `talker_chain` — despite its "seq_len=1" docstring — pads to 32 rows
> and runs `mode="prefill"`, so both use the prefill kernel. This was confirmed by
> instrumenting `rotary_embedding_llama` and counting which branch each test took. That gap is
> why `test_qwen3_tts_rope_decode.py` exists.

### 3.2 Op-level — the reliable numbers

Per-op device times were stable across all captures.

| RoPE per layer | before | after |
|---|---|---|
| Talker prefill seq=32/64/128 | 26 + 19 us | 26 + 19 us (unchanged, as intended) |
| CP prefill seq=2 | 27 + 19 us | 27 + 19 us (unchanged, 2 positions) |
| **Talker decode, N300** (8 heads / 4 KV) | 26 + 18 us | **3 + 3 us** |
| **CP decode, N300** (8 / 4) | 27 + 19 us | **4 + 4 us** |
| **Talker decode, N150** (16 / 8) | 41 + 26 us | **3 + 3 us** |
| **CP decode, N150** (16 / 8) | 43 + 27 us | **4 + 4 us** |

Counting the whole block (two cos/sin reshards + a transpose either side of each rotary op):
Talker decode 44 -> 16 us and CP decode 46 -> 18 us per layer on N300; ~67 -> ~17 us on N150.

Note the core counts in the profile: the prefill kernel spreads one token across **64 cores**
and still takes 26 us, because it walks heads serially. The decode kernel uses **1 core** and
takes 3 us.

### 3.3 Block windows

N150 (no CCL, so windows are clean — single captures):

| decode layer | before | after | |
|---|---|---|---|
| Talker | 762 us | **711 us** | -6.7 % |
| CodePredictor | 530 us | **477 us** | -10.0 % |

N300 (medians of 3 captures):

| block | baseline | after | |
|---|---|---|---|
| CP decode | 567 us | **385 us** | **-32 %** |
| CP prefill seq=2 | 517 us | **424 us** | **-18 %** |
| Talker decode | 659 us | 654 us | within noise — see below |

### 3.4 ECAPA SpeakerEncoder

> ## Generation runaways here are a `--ref-text` bug, not an encoder bug
>
> Symptom: 20.5 s of garbled audio, `Generated 256 code frames` — i.e. the
> `max_new_tokens` cap, so EOS never fired. It reproduces with **every** encoder
> config including the unmodified one, and the trigger is a reference transcript that
> does not match the reference audio:
>
> ```
> --ref-text "Jason, can we take a look at the review slides"   # WRONG
> jim_reference.txt: "So basically you put up the high level overview slides."
> ```
>
> Omit `--ref-text` and the demo reads `jim_reference.txt`, which is correct. With the
> mismatch the in-context alignment is broken, EOS becomes unreliable, and the result is
> a coin flip that a sub-1 % embedding change is enough to flip:
>
> | prompt | ref-text | fuse=0 asp=0 | fuse=1 asp=1 |
> |---|---|---|---|
> | en_long | **mismatched** | 86 ok | **256 CAP** |
> | en_short | **mismatched** | **256 CAP** | - |
> | en_long | correct | 75 / 92 ok | 73 / 73 ok |
> | en_short | correct | 16 / 17 ok | 19 / 19 ok |
> | en_mid | correct | 48 ok | 43 ok |
> | en_long / en_short | correct, `+conv` | - | 108 / 18 ok |
> | en_long / en_short | correct, traced | - | 108 / 18 ok |
>
> Note row 2: the **unmodified** encoder runs away on a short prompt under the mismatched
> ref-text, at seed 42 and seed 7. So the fusion never caused this; it only changed which
> prompts land on which side of an already-broken conditioning.
>
> Two lessons kept from getting this wrong once:
>
> * **Never gate an ECAPA numerics change on synthetic weights.** Against
>   `_synthetic_sd()` the fusion measures a 1e-6 PCC delta; with the real 1.7B weights it
>   is 0.74 % relRMS. Real-weight numbers are in the table below.
> * **PCC cannot predict generation.** `asp=1` perturbs the embedding by 0.77 % and
>   `fuse=1` by 0.74 % — indistinguishable — and under a broken ref-text one ran away and
>   the other did not. Frame count over several prompts and seeds is the only gate that
>   sees this, and a runaway must be checked against the *unmodified* baseline before it
>   is attributed to a change.
>
> Real 1.7B weights, jim_reference.wav, embedding vs the fp32 reference (the baseline is
> itself 2.72 % off, so these are all reshuffles of a similar-sized error):
>
> | fuse | asp | conv | PCC vs ref | relRMS vs ref | relRMS vs baseline |
> |---|---|---|---|---|---|
> | 0 | 0 | 0 | 0.999634 | 2.72 % | - baseline |
> | 0 | 1 | 0 | 0.999628 | 2.74 % | 0.77 % |
> | 1 | 0 | 0 | 0.999610 | 2.81 % | 0.74 % |
> | 1 | 1 | 0 | 0.999611 | 2.80 % | 0.83 % |
> | 1 | 1 | 1 | 0.998370 | 5.72 % | 3.74 % |
Runs **once per request** off the reference audio, so this is time-to-first-audio, not RTF.

`SpeakerEncoder.forward`, warm, N300, mel T=384 — interleaved A/B over 24 samples each so
drift cannot fake a winner:

| config | median | min | max | vs off |
|---|---|---|---|---|
| `off` — original device glue | 65.6 ms | 58.1 | 91.1 | |
| `fuse` — host-fused glue | 45.4 ms | 42.9 | 60.0 | **−20.2 ms** |
| `+asp` — ASP convs on device | **27.7 ms** | 26.3 | 32.4 | **−37.8 ms, −58 %** |

Op counts and host round-trips, from a `ttnn` call spy plus counters on the D2H/H2D helpers
(not a config assertion — the spy proves the calls actually stop being issued):

| config | slice | relu | tanh | add | concat | multiply | glue ops | D2H/H2D |
|---|---|---|---|---|---|---|---|---|
| off | 24 | 23 | 1 | 21 | 6 | 7 | 82 | 24 / 24 |
| fuse | 0 | 0 | 0 | 3 | 3 | 7 | 13 | 5 / 5 |
| +asp | 0 | 0 | 1 | 3 | 3 | 7 | 14 | **4 / 4** |

The survivors are the 3 residual adds, the MFA + 2 ASP concats, the 7 SE scale multiplies and
ASP's tanh — all real device work. Round-trips start at one per host conv (entry TDNN + 3×7
Res2Net branches + 2 ASP = 24); fusion collapses each block's 7 into 1; device ASP drops its own.

> **These are warm numbers, and the demo does not see them.** The demo calls
> `extract_speaker_embedding` exactly once per process, so it is always the *first* call and
> pays device program-cache population: 866–1019 ms measured across runs, statistically
> unchanged by any of this. The change is worth having for a server handling more than one
> request; the fix for the one-shot case is separate — warm the encoder at startup next to
> `capture_se_block_traces`, moving that ~900 ms off the first request. The very first run
> after this change also pays a one-time on-disk JIT cost for the two new ASP matmul shapes
> (2958 ms, then 866 ms).

Block windows under Tracy:

| block | before | after |
|---|---|---|
| SERes2Net block_idx=1 | 30 ops, 121 us device, 24,565 us gap | **9 ops, 82 us device, 10,484 us gap** |
| entry TDNN 128->512 | 1 op (6 us Unary) | **0 ops** — empty report |

Accuracy, full encoder vs the fp32 reference `speaker_encoder_forward`:

| | PCC vs reference |
|---|---|
| device glue | 0.999570 |
| host-fused | **0.999571** |
| device ASP (`+asp`) | 0.999570 |

Fusion moves *toward* the reference, and moving ASP onto the device costs nothing back — the
bf16 matmul lands on the same sixth decimal as the fp32 host conv it replaced. A/B
between the two paths is 0.999930 — so the speaker embedding does shift at the 1e-4 level,
which reseeds AR sampling and changes the generated audio length. Both demo runs
(`QWEN3_TTS_SE_HOST_FUSE=0` and `=1`) complete and produce healthy audio (no NaN, rms 0.15 /
0.11, peak 0.93 / 0.61); byte-identical wavs are **not** an available gate for this change.

**Everything convertible IS on device, and traced it is a 5x win — but only traced.**
`QWEN3_TTS_SE_DEVICE_CONV=1` expresses the k>1 reflect-pad convs as im2col + one matmul, so
nothing leaves the device: **0 host round-trips**, and the whole forward captures as a single
Metal trace. The reflect shift is exact — the row map comes from `F.pad(arange, "reflect")`
itself — and is materialised two ways (`QWEN3_TTS_SE_CONV_SHIFT`):

* `slice` (default): decompose the tap's row order into maximal ascending runs, one
  `ttnn.slice` each, then concat. ~3.3 us/slice. More ops, far less device time.
* `gather`: one `ttnn.gather` per tap. Fewer ops, but ~500 us each.

Measured, mel T=384, PCC **0.999561 in every cell** (vs 0.999570 for the host path):

| | 1 chip eager | 1 chip **traced** | 2 chip eager | 2 chip **traced** |
|---|---|---|---|---|
| `+asp` (host convs) | 23.0 ms | n/a — host in the loop | 28.2 ms | n/a |
| `+conv` / gather | 24.6 | 24.5 | 93.7 | 24.6 |
| `+conv` / slice | 43.5 | **4.89** | 208.1 | **5.15** |

Three things to take from this:

1. **Traced + slice is the answer: 23.0 -> 4.89 ms on one chip, 28.2 -> 5.15 ms on two**
   (-79 % / -82 %). Traced replay is essentially SKU-independent because it is pure device
   time with no per-op host dispatch.
2. **Eager, slice is a disaster** (208 ms on two chips). It trades 8 transfers for ~310 small
   ops, and eager dispatch cost scales with op count and with device count. So
   `QWEN3_TTS_SE_DEVICE_CONV` must never be enabled without capturing the trace — which is
   why it defaults to **0**.
3. **`gather` is the wrong primitive.** Traced it stalls at 24.5 ms: from Tracy,
   `GatherDeviceOperation` is 481 us/op (61.8 % of device time) and lowers to a
   transpose-based composite (a further 34.5 %), while the matmuls that do the actual conv are
   **1.6 %**. Same 481 us/op on one chip and on two, so this is the op, not the mesh.

**Shipped as a trace cache keyed by mel length (option A).** `forward` replays a captured
trace when one exists for this mel length and otherwise runs the host path, so nothing has to
be bucketed and no accuracy is traded. `capture_forward_trace(L)` does the capture;
`QWEN3_TTS_SE_AUTO_TRACE=1` captures lazily on first sight of a length.

| | host path | **traced replay** | capture cost (warm) |
|---|---|---|---|
| 1 chip | 23.5 ms | **5.44 ms** | 313 ms |
| 2 chips | 30.2 ms | **6.75 ms** | 384 ms |

A length with no trace falls back cleanly (L=377 measured at 26.2 / 30.3 ms, PCC intact), and
capturing it afterwards gives 6.97 / 7.73 ms. Non-tile-aligned lengths capture fine.
PCC vs the fp32 reference is 0.999561 traced against 0.999570 on the host path.

Why a cache and not buckets: mel frames are about `audio_samples / 256`, so L moves with every
~10.7 ms of reference audio, and **there is no masking anywhere in this encoder** (the
reference builds an all-ones mask; ASP's mean/std/softmax pool the whole sequence). Padding a
mel to a bucket would fold pad frames straight into the embedding, and the per-conv reflect pad
would reflect against the pad region instead of the real signal end. In a service the reference
audio per voice is a fixed asset, so the set of distinct L is small — capture one per voice.

Two constraints worth keeping:

* **The fallback must be the host path, never eager device-conv** (43 ms on one chip, 208 ms on
  two). `capture_forward_trace` forces the device-conv flags on for the capture and restores
  them, so the eager path is never left enabled.
* **A cold capture is expensive.** In the demo — one call per process — the speaker-embedding
  stage goes 994 ms -> 11,987 ms with `QWEN3_TTS_SE_AUTO_TRACE=1`, because capture JIT-compiles
  ~310 new op shapes. Amortises over a server's requests; a straight loss for one-shot use.
  Leave auto-trace off for the demo.

**End to end in the demo** (`QWEN3_TTS_SE_TRACE=1`, which captures SE-block + FC + forward
traces once before the first speaker-embedding call, in the early phase `init_server_context`
requires):

| | speaker-embedding stage | capture (reported separately) |
|---|---|---|
| untraced | 824 - 832 ms | - |
| traced | **14.3 / 13.9 / 14.0 ms** | 6016 ms cold, then 1275 / 1755 ms |

**59x on that stage**, and the 14 ms covers the host mel, the H2D, the ~6.8 ms replay and the
D2H. Three consecutive runs give byte-identical audio (2.32 s, 29 code frames, rms 0.1104,
peak 0.7968, crest 7.22, no NaN, no clipping), so the traced path is deterministic. It differs
in amplitude from the untraced path only because the embedding shifts at the 1e-4 level and
reseeds sampling.

For a one-shot demo the capture still outweighs the 816 ms saved, which is why
`QWEN3_TTS_SE_TRACE` and `QWEN3_TTS_SE_AUTO_TRACE` both default to off — the capture belongs at
server startup.

> **Note this run was also the first time the pre-existing SE-block and FC traces were ever
> exercised.** `init_server_context` is the only caller of `capture_se_block_traces` /
> `capture_fc_trace`, nothing in the repo calls `init_server_context`, and
> `activate_traced_extract()` had no callers at all, so `_se_traces_active` was never True and
> every `use_traces` guard in the encoder was dead. All three traces (3 SE blocks, fc, forward)
> capture and replay correctly.

**Reflect vs replicate padding — probed, not free.** TTNN conv offers only `Zeros` and
`Replicate`, so the tempting shortcut to getting the k>1 convs on device is to accept
`Replicate`. Measured in fp32 with real 1.7B weights and the demo reference audio, patching only
the pad mode in the reference implementation (22 convs have k>1):

| pad mode | PCC vs reflect | cosine | rel RMS diff |
|---|---|---|---|
| replicate | 0.999844 | 0.999844 | **1.80 %** |
| zeros | 0.998929 | 0.998928 | 4.83 % |

PCC flatters it. 1.8 % relative RMS on the speaker embedding is ~250× the numerical noise of
every other change here, and this codebase already has a case where the embedding drifted at
PCC 0.96 / 30 % RMS and produced audibly wrong output. Do not treat replicate as free — it needs
a listening test. The exact route is `ttnn.gather` with a constant index tensor, which builds a
reflect pad in one op.

### 3.5 End-to-end

Traced AR decode frame, single chip: **75,528 -> 70,784 us (-6.3 %)** from the RoPE change
alone (the CP N300 sharding does not apply on one chip). CP decodes -8.3 %, Talker decode
-4.1 %, CP prefill unchanged.

---

## 4. Measurement methodology — read before comparing reports

**N300 CCL timings swing ~2x run to run.** The same `ttnn.all_gather` of a 64 KB payload
measured 34 us in one Tracy capture and 65-71 us in the next three. Baseline windows for an
identical CP decode layer ranged 484-582 us.

Consequences:

- A **single capture per block is not comparable** to another single capture. In this work an
  *untouched* code path (`talker_layer_prefill_64`, seq > 1, unaffected by either change)
  moved **+10.7 %** between two single captures. Always take medians of 3+.
- The Talker decode window on N300 is dominated by its two (still unmodified) all-reduces, so
  a real 28 us/layer RoPE saving is invisible there. **Judge that change at op level, or on
  N150 where there is no CCL.**
- N150 windows are clean; `-n 1` is acceptable there.

Regenerate the block report with:

```bash
source python_env/bin/activate
./qwen3_tts_block_report.sh                                # N300, 3 runs/block, ~16 min
./qwen3_tts_block_report.sh -m N150 -n 1 -o n150_blocks.txt
./qwen3_tts_block_report.sh -b qwen3_tts_n300_blocks.txt   # add a vs-baseline column
./qwen3_tts_block_report.sh -A                             # re-assemble, no device time
```

Manually, per block:

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH="$(pwd)" ARCH_NAME=wormhole_b0 MESH_DEVICE=N300
python -m tracy -p -v -r -m pytest -s -q \
  models/demos/qwen3_tts/tests/test_qwen3_tts_profile_single_layer.py -k test_cp_layer_decode
CSV=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report --start-signpost start --end-signpost stop "$CSV"
```

Use the **full** test name — `-k talker_layer_prefill` matches all three buckets and puts
three windows in one capture — and one `-k` per Tracy run, since the CSV is picked by
newest timestamp.

### The prefill / decode perf report

`tests/test_qwen3_tts_perf_report.py` + `tests/qwen3_tts_perf_report.sh` are the report to
use when the target is "prefill ms down, decode throughput up" in the demo. Everything else
in this section profiles a *block*; this profiles the two windows the demo actually spends
its time in, and it profiles them the way the demo runs them — as **Metal-trace replays**.

```bash
source python_env/bin/activate
./models/demos/qwen3_tts/tests/qwen3_tts_perf_report.sh                       # all windows
./models/demos/qwen3_tts/tests/qwen3_tts_perf_report.sh -w prefill_demo,decode_frame
./models/demos/qwen3_tts/tests/qwen3_tts_perf_report.sh -m N150
```

Per window it writes `ops_list/perf_report/<window>/`: `ops.csv` (raw), `tt-perf-report.txt`,
`ops_list.md` (the full per-op list plus rollups by class / op code / block, and the ranked
adjacent data-movement pairs), and `totals.json`; plus a `summary.md` tying them together.

Windows: `prefill_demo`, `prefill_32`, `prefill_64`, `prefill_128`, `decode_talker`,
`decode_cp`, `decode_frame`. `decode_frame` is the throughput unit — one CP frame plus one
Talker decode — and carries inner signposts so both halves come out of that one capture.

Three things this gets right that the older `test_qwen3_tts_profile_*` tests do not:

**It replays the trace instead of running the body eagerly.** Those tests profile untraced
passes: the same kernel graph, but every op waits on the host. On the AR-step-0 capture that
showed **80 s of op-to-op gap against 46 ms of device time** — the gap column was measuring
python, not anything on the chip. Traced, prefill comes out 16.13 ms device + 0.58 ms gap
against 17.00 ms of wall clock. The profiler handles ops inside a replay (`METAL TRACE ID` /
`METAL TRACE REPLAY SESSION ID` in the CSV); `-r` on `python -m tracy` turns that on. Do not
reach for `--device-trace-profiler`: that collapses the whole replay into one `TRACE-KERNEL`
marker, which is a total, not a breakdown.

**It keeps the window inside the profiler's buffer.** The profiler's DRAM buffer holds
**1000 programs** by default; the AR frame is ~4,200 device ops. Past the budget the device
drops markers, logs `Profiler DRAM buffers were full`, and the CSV comes back partial *with
no error* — which is what a report full of `TilizeDeviceOperation` and `No signposts found`
means. The driver passes `--op-support-count 20000` and fails the run if that warning
appears; the test itself fails early if the budget is below the window's op count.

**It measures wall clock in a separate unprofiled pass.** Under the device profiler each
replay writes markers for every op on every core on every RISC; ten timing replays of the AR
frame push `profile_log_device.csv` past a gigabyte and post-processing past 9 GB of RSS. The
profiled pass replays exactly once; a second, plain pytest pass takes the median of ten.

Under TP the CSV holds one row per chip per op. `ops_list.md` merges them positionally and
keeps the max per op — an op is done when the slowest chip is done — so its op count is the
per-chip count, which is why it is lower than `tt-perf-report`'s.

### Profiling the CP layer the *demo* actually runs

`test_qwen3_tts_profile_single_layer.py -k cp_layer_*` is a faithful shape/config replica of
one CP layer (same `CodePredictor._layer_forward`, same seq 2 / 1, same KV max 32, same
`_n150` / `_n300_cp_opt` / `fast` flags), but it is **not** the demo's path:

- it runs **eager**; the demo runs the whole CP frame inside one `ttnn.execute_trace`
  (`capture_fused_cp_trace` in `tt/server.py`, replayed in `tt/utils.py`),
- it builds `Qwen3TTSCodePredictorConfig(num_hidden_layers=1)` and always feeds a fresh
  DRAM-interleaved input, so it always pays the input `to_memory_config`. In the demo only
  layer 0 does — layers 1-4 receive the previous layer's output already in `_ln_attn_memcfg`
  and skip that I2S,
- it omits everything outside the layer: `small_to_mtp_projection`, the S2I + final RMSNorm,
  the 15 `lm_head` matmuls, the in-trace concat / embeddings / sampler, the KV-restore
  `assign`s.

So `single-layer × 5` is an upper bound on the layer stack, not the frame. The profiler
handles traced ops (`METAL TRACE ID` / `METAL TRACE REPLAY SESSION ID` in the CSV), so the
real frame can be captured directly. `QWEN3_TTS_PROFILE_CP_FRAME=<step>` signposts exactly
one decode step's CP frame:

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH="$(pwd)" ARCH_NAME=wormhole_b0 MESH_DEVICE=N300
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_TRACE_TRACKING=1 \
QWEN3_TTS_PROFILE_CP_FRAME=3 \
python -m tracy -p -v -r -m models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py \
  --text "..." --max-tokens 8
CSV=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)
tt-perf-report --start-signpost cp_frame_start --end-signpost cp_frame_stop "$CSV"
```

Pick a step past the first frame so one-time warmup is excluded, and keep `--max-tokens`
small — the device profiler buffer fills fast when every frame replays two traces. The
signpost is host-side and the frame is one `execute_trace`, so the window cannot be
subdivided further; inside it the CSV is in trace order, which is enough to read off the
5-layer prefill block and each of the 14 decode steps. The same env var also works on the
per-step (non-fused, `--greedy`) path.

---

**Use `python_env/bin/python3`, not the bare `python3`.** On this host the bare interpreter
resolves `ttnn` to a stale April build in a *different* checkout (`/home/user/ign_fs/tt-metal`)
whose `physical_system_discovery.cpp` asserts `Bus ID 0 not found` on an N300 — the remote
chip has no PCI bus. It looks exactly like wedged hardware, but the board is fine; check
`python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))"` before blaming the card.

To take a single N300 while someone else is using the machine:

```bash
export TT_VISIBLE_DEVICES=0 \
       TT_METAL_CACHE=$HOME/.cache/tt_metal_n300_0 \
       TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto \
       MESH_DEVICE=N300
```

---

## 5. Tried and rejected — do not re-propose without new evidence

| idea | measured | verdict |
|---|---|---|
| `all_gather` on dim 0 for the 2-chip all-reduce | 78 us | Tile padding / a size-1 outer dim pushes it onto the composite all-broadcast fallback. Gather on the **last** dim: 34 us. |
| `num_links=2` on that all_gather | 69 us vs 34 us on auto | Payload far too small to amortise a second link's setup. |
| Moving RoPE cos/sin/trans_mat from DRAM to L1 | 40.9 vs 41.4 us | Irrelevant — the cost is the per-head loop, not memory. |
| Lowering RoPE math fidelity | 41.4 (LoFi) vs 43.4 (HiFi4) | Same. Not a fidelity problem. |
| `nlp_create_qkv_heads_decode` instead of the sharded prefill-style split | 13.3 us vs 2 us | Much worse here. Its value is feeding a full decode-layout attention pipeline, which this model does not use. |
| Running the CodePredictor at TP=1 (replicated) on N300 to delete all CCLs | est. matmul growth +103 us vs CCL saving -107 us | Net ~zero for a large, risky change. |
| DRAM-sharding the CP QKV matmul | ~2 us | N is padded 2048 -> 2304, so it needs an S2I + slice that eats the gain. |
| `ttnn.transformer.scaled_dot_product_attention_decode` for Talker decode | PCC **0.50** vs 1.00 | Its cross-chunk flash-decode reduction is wrong at dh=128 as soon as the cache spans more than one `k_chunk` — single chunk 0.999995, two full chunks 0.702, 11 chunks 0.504 against an fp32 reference in isolation. And we cannot stay inside one chunk: the op requires `k_chunk_size` to be a **power of two** (`sdpa_decode.cpp:67`) *and* to divide `k_shape[2]`, so kv=352 (=32x11) admits only `k_chunk=32`, i.e. 11 chunks. `models/tt_transformers` documents the same cliff for Gemma-2 at dh=256. Revisit only if kv_max is padded to a power of two (then `k_chunk=kv_max` is single-chunk and correct) — it would also need `nlp_concat_heads_decode` and a new wo in0 spec. The prefill-form SDPA at `q_chunk=32` is 26.8 us and correct. |
| SDPA fidelity HiFi4 -> HiFi2 for decode | 38.5 -> 37.0 us | 4 % of one op for a real PCC loss (0.99998 -> 0.99988). The decode SDPA is not math-bound. |
| Bigger `k_chunk` without checking divisibility | k=320: 60.9 us vs k=128: 30.6 us | Cost is the chunk-*padded* KV length, so a chunk that does not divide the cache is worse than a smaller one that does. |

---

## 6. What to do next — ranked

### 6.1 Port the 2-chip all-reduce to the Talker  *(largest remaining N300 win)*

The Talker still calls `tp_all_reduce` (i.e. `ttnn.all_reduce`). In its decode layer that is
**~197 us — about 30 % of the window**, and it is the reason Talker decode numbers are so
noisy. `mesh_utils.tp_all_reduce_2chip` already exists and is trace-safe; the CP measured
107 -> 77 us per layer from it.

Call sites — note they span two files, and `mlp.py` has four:

- `tt/attention.py:1063` (after `o_proj`)
- `tt/mlp.py:336`, `:351`, `:391` (after `down_proj`, one per path)

Expect roughly -60 us/layer x 28 layers. Gate on `is_n300(device)` as the CP does — the 2-chip
form is only correct for exactly two chips — and verify with medians of 3+ captures.

`mlp.py` is shared, so check whether any non-Talker caller reaches those lines before
switching them wholesale.

> **Recheck the size of the prize first — the ~197 us above no longer reproduces.** Across the
> six `talker_layer_decode_traced` captures behind 2.4, the four CCL rows measured 20-25 us
> each, i.e. **~85 us per layer (19 % of 437 us)**, not 197. Against `tp_all_reduce_2chip`'s
> measured 34 us of CCL + ~4 us of slice/add per site, the headroom is ~4 us/site — about
> -8 us/layer, close to the noise floor, and it trades two ops (reduce_scatter + all_gather)
> for three (all_gather + slice + add). The 197 us figure came from captures where a single
> `all_gather` swung to 66-70 us; that swing is real (see 4) but it is not the steady state.
> Measure both forms in the same session before spending the change.

### 6.2 Reduce the remaining CCL cost

Even in the 1-CCL form the all-gather is 34-70 us for 64 KB, on **1 core**. This is pure
fabric latency, not bandwidth. Worth trying:

- `ttnn.experimental.all_gather_async` with persistent output buffers and pre-allocated
  semaphores (avoids per-call setup).
- `use_l1_small_for_semaphores=True`.
- Investigating the run-to-run variance itself — if it is fabric arbitration, pinning
  `sub_core_grids` may stabilise it.

### 6.3 CP matmul fidelity: HiFi4 -> HiFi2

The CodePredictor hardcodes `MathFidelity.HiFi4` for all matmuls (`self.kcfg`) while the
Talker uses LoFi, and `tt-perf-report` explicitly advises HiFi2. That is ~119 us of matmul per
CP decode layer, likely 15-25 % recoverable.

**Not attempted** because it is a genuine accuracy change, unlike everything above. Gate it,
then run `test_qwen3_tts_pcc.py` and listen to the demo output. Note `sdpa_kcfg` is separate
and should stay high-fidelity: the code documents that QK-norm amplifies K by ~68x and q.k
dot products can overflow bf16, which is why the SDPA chain runs in fp32.

### 6.4 The CP's SDPA switch — flag PROMOTED, mask rescale still open

`QWEN3_TTS_CP_FUSED_SDPA` is now **default ON** (set it to `0` to restore the manual
chain for an A/B). Measured with `tests/test_qwen3_tts_perf_report.py -k test_decode_cp`
on N300 TP=2 — the demo's own fused CP frame, traced, one replay per capture:

| | ops | device | wall |
|---|---|---|---|
| manual fp32 chain | 4339 | 31.23 ms | 30.32 ms |
| fused SDPA | 3619 | 28.27 ms | 27.32 ms |
| | **-720** | **-2.97 ms** | **-3.00 ms (-9.9 %)** |

Per op code, the 75 hand-built chains (5 layers x 15 CP invocations per frame) become 75
SDPA calls: -285 typecast (0.97 ms), -150 matmul — the explicit QK^T and PV (0.92 ms),
-75 softmax (0.56 ms), -75 binary (0.49 ms), -150 repeat_interleave (0.44 ms), -75
transpose (0.29 ms), +75 SDPA (0.66 ms). Device delta and wall-clock delta agree to
0.03 ms, so this is mechanism rather than run-to-run noise.

**What gated it, and what did not.** `dccf18b66c4` named three gates and only the first is
clean:

1. `tests/test_qwen3_tts_cp_sdpa_parity.py` — **clean.** It had never actually run: its
   `state_dict` fixture passed `load_weights()`'s whole `(main, decoder)` tuple into
   `CodePredictor`, so every attempt died in the constructor. Fixed in `2b8ed3db073`. On
   the real masked shapes with real weights: prefill seq=2 PCC 0.99969828; decode
   start_pos 2/3/5 PCC 0.9998 with the **sampled token identical** (376/1741/1364) in all
   three. So the fused path is not wrong on the explicit decode/prefill masks — which is
   the only question this test was written to answer.
2. Frame-count sweep over >=8 seeds — **NOT RUN.**
3. Listen / WER check — **NOT RUN.**

The flag was promoted anyway, at the owner's direction, with 2 and 3 outstanding. Record
of what is known against it: paired demo runs generate consistently FEWER frames with it
on — `dccf18b66c4` saw seed 7 72->65, seed 42 87->68, seed 123 91->85, and a fresh seed-42
pair on this tree gave 88->81 (7.04 s -> 6.48 s of audio). Direction is 4/4. But nothing
has ever run away to the 256-frame cap, and the within-arm spread swamps the shift — the
same seed-42 ON arm measured 68 on one tree and 81 on another — so n=1 per seed cannot
separate a real regression from sampler chaos. Shorter audio is either slightly faster
speech or a dropped word, and duration alone cannot tell them apart. 3.4 is explicit that
PCC cannot predict generation length here.

**If generation quality regresses, `QWEN3_TTS_CP_FUSED_SDPA=0` is the first thing to try.**

Still open: **the mask rescale, which `dccf18b66c4` did not take.** `scale=1.0` with the
softmax scale folded into the CP's q_norm gain stops ttnn's wrapper pre-multiplying the CP
mask by `1/scale` on every call — 2.4 measured 6 us/layer on the Talker, and the CP runs 75
layer evaluations per frame. Same recipe, same `q_norm` fold, its own weight-cache key.

Do **not** reach for `scaled_dot_product_attention_decode` on either model — it is
numerically broken at these shapes (see 5).

### 6.5 The Talker's *masked prefill* SDPA config

2.4 fixed the decode SDPA config and left the masked-prefill one (`prefill_attn_mask`, Sq =
bucket, Sk = kv_max) on `q_chunk=64 / k_chunk=64`. `q_chunk=64` is right there — Sq is a real
sequence — but `k_chunk=64` leaves kv=352 as 6 chunks (384 padded rows) where 352 is 1 chunk.
Prefill runs once per utterance, so this is time-to-first-audio only; TTFT already went
141.9 -> 134.1 ms from the fused-SDPA switch.

### 6.6 Talker prefill buckets 64 and 128

`attention.py` has `use_dram_shard_qkv = seq_len <= 32`, with a TODO: buckets 64 and 128 need
their own per-`m` shard configs to engage the DRAM-sharded QKV and the sharded
`nlp_create_qkv_heads`. At seq=64 the profile shows `nlp_create_qkv_heads` at 25 us on 2 cores
— the same single-core-ish problem already fixed elsewhere.

Prefill runs once per utterance, so this matters for time-to-first-audio, not steady state.

### 6.7 Lower value

- **CP `o_proj` DRAM-sharded** — ~5 us/layer net after the S2I + slice for the 1024 -> 1152 pad.
- **Hoist the cos/sin reshard out of the layer.** `apply_rope_qk` reshards cos/sin per layer
  (2 us); they are identical across all layers of a forward pass, so they could be resharded
  once by the caller. ~2 us x 5 CP layers x 15 CP passes = ~150 us/frame. Needs care to stay
  trace-safe.

---

## 7. Files

| file | change |
|---|---|
| `tt/rope.py` | `apply_rope_qk`, `get_decode_transformation_mat`, `_rope_decode_memcfg` |
| `tt/attention.py` | Talker uses `apply_rope_qk`; builds `_decode_trans_mat` at init |
| `tt/code_predictor.py` | N300 fast path (`_n300_cp_opt`) + `apply_rope_qk` |
| `tt/mesh_utils.py` | `is_n300`, `tp_all_reduce_2chip` |
| `tt/speaker_encoder.py` | ECAPA host fusion (`_se_host_fuse`, `_res2net_cascade_torch`, `_conv1d_same_padding_torch_ncl`) |
| `tests/test_qwen3_tts_rope_decode.py` | RoPE bit-exactness + routing guard + `k_keep_decode_layout` |
| `tt/mlp.py`, `tt/dram_sharded_matmul.py` | `unpad_dram_sharded_out` (padded-N trim off the sharded output) |
| `tests/test_qwen3_tts_profile_single_layer.py` | `-k talker_layer_decode_traced` — the deployed decode window |
| `tests/test_qwen3_tts_cp_n300_opt.py` | CP fast path A/B + Metal-trace replay guard |
| `tests/test_qwen3_tts_speaker_encoder_host_fuse.py` | ECAPA op-count spy + cascade equality vs reference |
| `qwen3_tts_block_report.sh` (repo root) | regenerates the block report |
| `tests/test_qwen3_tts_perf_report.py` | traced prefill / decode windows — the report for this optimisation |
| `tests/qwen3_tts_perf_report.sh` | runs every window, one Tracy capture each, and assembles `summary.md` |
| `tests/qwen3_tts_perf_report_opslist.py` | one CSV window -> full per-op list + rollups |

### Validation run before merging

```bash
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH="$(pwd)" ARCH_NAME=wormhole_b0

pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_pcc.py            # accuracy, real weights
pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_rope_decode.py    # RoPE bit-exact + routing
pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_cp_n300_opt.py    # CP A/B + trace (opens its own 1x2 mesh)
pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_speaker_encoder_host_fuse.py  # ECAPA ops + PCC
pytest    models/demos/qwen3_tts/tests/test_qwen3_tts_trace_perf.py     # full model under Metal trace
MESH_DEVICE=N150 pytest models/demos/qwen3_tts/tests/test_qwen3_tts_profile_single_layer.py
MESH_DEVICE=N300 pytest models/demos/qwen3_tts/tests/test_qwen3_tts_profile_single_layer.py
```

Every one of these passed with the changes described above.
