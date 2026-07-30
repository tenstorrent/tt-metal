# AttnRes distribution memo — Phase 8

Written before any distributed code, per the phase ladder. Ends at a judgment gate.

Target shapes: LoudBox `(2, 4)` now, Galaxy `(8, 4)` next, a pipeline of Galaxies after
that. Production prefill `T = 5120`, `d = 7168`, `L = 93`, `Bk = 12`, `S` ramping 0→8,
**186 reads** per forward.

---

## 1. What has to be communicated, and what does not

The read is

```
score_i = rsqrt(mean_d(v_i²) + eps) · Σ_d v_i[d]·q[d]        # two reductions over d
α       = softmax_i(score)                                    # over candidates
out     = Σ_i α_i · v_i                                       # elementwise over d
```

Only the **first line** reduces over `d`. The softmax reduces over candidates, and the
mixture is elementwise over `d` with a per-(token, candidate) scalar weight. So with `d`
sharded, every rank can compute the mixture **locally and exactly** the moment it agrees
with the other ranks on `α` — and `α` is `2(S+1)` scalars per token.

The stream itself never crosses a rank boundary. Neither does `block_residual`, and
neither does `seal`/`accumulate` — both are elementwise.

Phase 7 established the other half of this by measurement: nothing in the op reduces over
`T`, and the shared token slice is **bit-identical** across `T ∈ {64, 1000, 5120}` (see
§Learnings Phase 7 in `bringup_log.md`). A sequence-parallel axis therefore needs **zero**
communication — not "little", none.

Two consequences worth stating plainly:

- `mean_d` must divide by the **global** `d`, not by the local shard width. This is the
  one place where a sharded AttnRes silently computes the wrong number instead of failing.
- `q` is folded from two `[d]` weights (D5), so it shards on `d` exactly like the stream.

---

## 2. The layout is not ours to choose

The residual stream's distributed layout is already fixed by the analog this op has to
drop into. In `deepseek_v3_d_p` prefill:

| fact | citation |
|---|---|
| `sp_axis = 0`, `tp_axis = 1` — sequence parallel on mesh rows, tensor parallel on mesh columns | `models/demos/deepseek_v3_d_p/tt/tt_prefill_transformer.py:121-122` |
| the residual stream is TP-sharded on hidden | `models/demos/deepseek_v3_d_p/tt/tt_prefill_block.py:552-553` |
| sequence is sharded on dim 2 across `sp_axis` via `ShardTensor2dMesh` | `models/demos/deepseek_v3_d_p/tt/mla/rope.py:160-161` |
| a `d`-sharded norm consumes `[batch, seq_len, emb_dim / num_devices]` | `models/demos/deepseek_v3_d_p/tt/tt_distributed_rms_norm.py:241-242` |
| statistics pattern: local stat → `all_gather(dim=3, cluster_axis)` → post op | `models/demos/deepseek_v3_d_p/tt/tt_distributed_rms_norm.py:255-286` |
| the block already all-gathers to full `emb_dim` for the dense FFN path | `models/demos/deepseek_v3_d_p/tt/tt_prefill_block.py:621-623` |
| `topology` is already a **per-axis tuple** in the analog's own 4x4 sub-torus config, with the comment that a scalar `Ring` "would deadlock dispatch/combine on a non-existent row wrap link" | `models/demos/deepseek_v3_d_p/tests/test_prefill_block.py:666-673` |
| a collective needs `device_params={"fabric_config": FABRIC_1D}`; without it `all_reduce` dies on an uninitialized fabric context rather than returning wrong numbers | `models/demos/deepseek_v3_d_p/tests/test_prefill_block.py:513-517`, `tt_metal/fabric/control_plane.cpp:2186` |

So on a `(R, C)` mesh the per-device residual stream is `[1, 1, T/R, d/C]`, and AttnRes's
snapshots are `[1, S, T/R, d/C]` — candidates on dim 1 per D10, untouched by either axis.

---

## 3. Three mappings, two rejected on arithmetic

Per-device traffic below is for one read at `S = 8` on `(2, 4)`: `T/R = 2560`,
`d/C = 1792`. The single-device baseline is the Phase-7 measurement — 7.33 GB of DRAM
traffic per read at full `T` and `d`, so 916 MB per device once split 8 ways.

**M1 — SP on axis 0, TP on axis 1. The analog's layout.**
Communication: one reduction of `2(S+1)` scalars per token per read, on `tp_axis` only.
Payload 92 KB of useful data per read; the mixture, the softmax, the seal and the
accumulate are all local. CCL traffic per forward ≈ 274 MB against ≈ 170 GB of DRAM
traffic → **0.16 %**.

**M2 — replicate `d`, shard only the sequence.** Zero reductions, which is seductive on a
bandwidth-bound op. But each rank then holds the full `d`: snapshots go 70 → 280 MiB per
device and the mixture's own traffic rises **4×** — the op is bandwidth-bound, so that is
a 4× slowdown paid on every read. Worse, it does not match §2, so every read site needs
the stream all-gathered in (`[1,1,T/R,d]` = 35 MiB) and reduce-scattered out: 186 × 35 MiB
≈ **6.5 GB** of CCL per forward, ~24× M1, on the critical path. Rejected twice over.

**M3 — shard the candidate axis `S` across `tp_axis`.** The softmax then needs a
cross-rank online-softmax merge (max, then rescaled sum) and the mixture needs a cross-rank
sum of **full-width** `[1,1,T/R,d]` partials — the same ~6.5 GB per forward as M2, except
now the stream really does cross rank boundaries. It also shards raggedly: `S` ramps 1→8,
so `C = 4` divides it evenly for exactly two of the eight blocks. Rejected.

**M1.** The interesting part was never the choice; it is what M1 costs and where it breaks.

---

## 4. What M1 actually needs

One reduction per read, of a stats tensor carrying both statistics for every candidate:

```
local  = [1, 2(S+1), T/R, 1]  # first half: Σ_d v²,  second half: Σ_d v·q
global = ttnn.all_reduce(local, cluster_axis=tp_axis, topology=topology[tp_axis])
```

`all_reduce`, not the analog's `all_gather` + post-op, for two reasons. It returns the
same shape it took, so there is no strided "sum every `C`-th column" to express in
composed ops. And it avoids a documented trap:

> "If the `input_tensor` has unaligned row-major pages or **padded tiles on the gather
> `dim`**, a slower composite all-gather implementation is used."
> — `ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather_nanobind.cpp:39`

A stats tensor's last dim is 1 or 2, which tile-pads to 32. Gathering *on that dim* is
exactly the padded case, so the analog's pattern would silently take the slow path here.
The analog does not hit it because `ttnn.rms_norm_pre_all_gather` hands it a tile-aligned
32-wide stat; we have no such op for a dot product over a dim-1 candidate axis.

**The two statistics stack on dim 1, not on the last dim.** Both forms tile-pad the last
dim to 32 regardless, so the only difference is how the halves come back apart afterwards:
slicing dim 1 lands on a tile-plane boundary, slicing a 2-wide last dim is a sub-tile read.
Dim 1 doubles a payload that is already noise (below) and buys a slice that cannot land
mid-tile.

Payload sizes per read (`S = 8`, so `2(S+1) = 18` planes), fp32 stats:

| mesh | `T/R` | useful | on the wire (last dim 1 → 32) |
|---|---|---|---|
| `(2, 4)` | 2560 | 184 KB | 5.90 MB |
| `(8, 4)` | 640 | 46 KB | 1.47 MB |

The 32× is the tile tax on a 1-wide last dim. Per forward that is 186 × 5.90 MB ≈ 1.1 GB
against ≈ 170 GB of DRAM traffic — **0.65 %**, so it does not pay for a fix. (Both figures
here hold `S` at its maximum for every read; `ROOFLINE.md` §4 sums the real schedule and
gets 657 MB against 110.3 GB, **0.595 %** — same verdict, exact numbers.) The fix, when
someone wants it: fold the candidate axis into the last dim (`[1, 1, T/R, 2(S+1)]`, 18 of
32 columns useful) for a 18× smaller payload, at the price of two `ttnn.permute` calls per
read. Trading two launches for bytes that are not the bottleneck is the wrong direction
until Phase 9 says otherwise, and a fused kernel packs statistics tightly anyway.

> **Amended 2026-07-30 (Phase 9, P4 — measured, traced).** Phase 9 says otherwise. The
> percentage above is correct and irrelevant: the collective does not charge per useful
> byte, it charges per **padded** byte, at exactly the same rate. Measured on `(2, 4)`,
> `[1, 18, 2560, 1]` (184 KiB useful in a 5 760 KiB envelope) and `[1, 18, 2560, 32]`
> (5 760 KiB all useful) both cost **348 µs** at `num_links = 1`. The folded layout costs
> **46.8 µs** — 7.4×, ~300 µs per read, and it stops scaling with `S` altogether because
> `2(S+1) ≤ 32` fits one tile column for every `S` this model uses.
>
> The two `ttnn.permute` calls cost ~40–120 µs of *device* time traced, not two launches
> of host time. So "trading two launches for bytes that are not the bottleneck" was the
> right analysis in the untraced regime — where the two extra launches at 152 µs cancel the
> saving almost exactly — and the wrong one in the traced regime the model ships in. The
> fix is on the backlog with that price on it; the fused-kernel argument in the last clause
> still stands, and this is the cheaper of the two.

**Statistics reduce in fp32.** `ttnn.all_reduce` reduces in bf16 unless the input is fp32
(`all_reduce_nanobind.cpp:48`). Measured over 186 chained reads at `d = 7168` on `(2,4)`:
fp32 stats 0.9999500, bf16 stats 0.9999401. The bf16 number lands on the single-device
baseline (0.9999408), which says fp32 is buying back rounding the single-device path also
takes rather than repairing damage sharding did. It costs 1.5 MB per read on a 900 MB
budget, so take it.

**Call count is the exposure, not bytes.** 186 all-reduces per forward.

> **Amended 2026-07-30 (`ROOFLINE.md` §3–§4, §6).** This paragraph used to price the
> collective at "a nominal 50 µs each ≈ 9 ms, about 2 %" against a 500 ms forward. The
> total was luck: modelled from fabric bandwidth the 186 reductions cost **9.85 ms** on
> `(2,4)`, but the denominator was wrong — the DRAM floor is **215.5 ms** per forward
> (1 935 µs per read at `S = 8`), not 500 ms, so the collective is **4.6 %**, and on `(8,4)`
> 44.2 µs against a 484 µs floor is **9.1 %**. Both double to 9.2 % / 18 % at the op's
> current `num_links=1`. And none of it binds in the regime we develop in: measured, one
> `ttnn.all_reduce` enqueue costs ~481 µs of *host* time against ~130 µs for a local
> `ttnn.mul`, so untraced a reduction is ~4 launches' worth of Python, and fabric bytes are
> invisible. Call count is still the exposure — but for two different reasons in the two
> regimes, and only the traced one is about the fabric.

**The split form is worse on collectives, not better.** For 24 read sites the direct form
issues 24 — one paired reduction per read, sealed set and live stream together. The split
form issues **49**: 1 for the sealed RMS, 24 for the per-site sealed dots, 24 paired in
`merge`. `inter_block` amortizes the RMS and de-amortizes nothing else, because each read
site still needs its own dot against the sealed set. Phase 7 measured the split form 1.50×
faster on one device; whether that survives 2× the collectives is a Phase-9 measurement,
not a claim to make here. If it needs saving, `inter_block` can batch its 24 dot tensors
into one `[1, 24(S+1), T/R, 1]` reduction and come down to 26.

> **Amended 2026-07-30 (Phase 9, P5 — measured, traced, `(2, 4)`).** It survives:
> **1.47×**, 3 274.6 → 2 228.3 µs per read site over a full 24-site block, against 1.50× on
> one device. Amortizing the sealed half's RMS pass is worth ~1 047 µs per site and the
> extra collective costs ~350 µs, so this section's framing — the split form is *worse* on
> collectives — is true and does not decide anything. It needs no saving; batching the 24
> dot tensors into one reduction (49 → 26) is now an optimization on top of a win, and P4's
> fold would cut the same cost by more.
>
> **Re-measured 2026-07-30 (P6).** With the fold shipped on by default the pair is
> 3 127.8 → 2 186.6, so **1.43×**. The fold is worth 3.5× more to the direct form
> (146.8 µs per site vs 41.7), because the direct form's one collective carries all 18
> stats planes while the split form's two carry ~10 between them — this section's "worse on
> collectives" is also *why* the fold has less to give it.
>
> **Retired 2026-07-30 (P8 — the batching landed).** `inter_block` now takes all 24 sites'
> dots in one matmul and one reduction, so the split form issues **26** collectives per block
> against the direct form's 24. "Worse on collectives" is down to 8% more of them, and the
> paragraph's framing no longer decides anything either way: 1 741.4 → 1 386.5 µs per site at
> `S = 8`, **1.61× the direct form**, 191.2 ms per forward against 265.0.
>
> The batched reduction's shape is *not* the `[1, 24(S+1), T/R, 1]` this section proposed. The
> site axis goes in the **last** dim — `[1, S, T/R, 24]` — because the dots come out of the
> matmul that way and because a 1-wide last dim tile-pads to 32 regardless: up to 32 sites
> ride inside the padding one site already paid for, which is `fold_stats`' argument applied
> to a second axis. Stacking on dim 1 instead would have multiplied the payload by 24.
>
> One thing the sweep found that a per-site cost model would not: **the split form is 9%
> slower at `S = 1`** (710.9 µs against direct's 649.6), because its second collective and
> `merge`'s own statistics pass are fixed costs and there is not yet enough sealed work to
> amortize them. Crossover at `S+1 = 2.30`. On a mesh the read form is therefore an `S`
> decision, not a global one — 24 of the schedule's 186 reads sit on the direct side of it.

**Per-axis topology.** `topology` is one `ttnn.Topology` per mesh axis, not a scalar —
Galaxy prefill is `[LINE, RING]`. This is not a precaution: the analog already carries the
tuple and the comment explaining that a scalar `Ring` "would deadlock dispatch/combine on
a non-existent row wrap link" (`test_prefill_block.py:666-673`).

---

## 5. Judgment gate — decided, implemented, and green on `(2,4)`

**Decided:** M1 — sequence-parallel on `sp_axis` (free), tensor-parallel on `tp_axis` with
one `all_reduce` of `[1, S+1, T/R, 2]` per read. The mapping is dictated by the analog's
stream layout, and M2/M3 lose by more than an order of magnitude on CCL traffic, so this
is not a close call.

**Implemented, in this order:**
1. `hidden_size` is the **global** `d` with an explicit `tp_factor`; `mean_d` divides by the
   global `d`; `_reject_sharded` became `_assert_shard_width` and now *accepts* `d/C`.
2. `to_query` shards the folded query on `d` across `tp_axis`. The op also owns
   `stream_mapper` / `vector_mapper` / `stream_composer`, so the layout has one definition
   and `forward` checks its input against it.
3. `_reduce_stats` / `_reduce_stats_pair`, both exact identities at `tp_factor == 1` — the
   single-device trace is unchanged, so every earlier measurement stays comparable.
4. `tests/test_tt_attn_res_distributed.py` on a real `(2,4)`.

**Measured (`(2,4)`, `T = 64`, 10/10 green; module total 92 passed):**

| gate | result |
|---|---|
| forward vs torch, `d = 7168`, `S = 8` | PCC 0.9999778, rel err 1.28e-2 (single device: 0.9999804) |
| forward vs torch, `S = 0` (identity, no collective) | PCC 0.9999986 — the placement control |
| split form, 24 read sites | every site ≥ 0.9999 |
| 93-layer walk, 186 reads / 186 collectives | device 0.9999500 vs torch-bf16 0.9999741, norm ratio 1.000183 |
| sequence-sharded vs sequence-replicated | **max\|Δ\| = 0**, and the two SP rows agree to 0 |
| statistics reduction deleted | PCC **0.5757407** — the gate has teeth |

That last pair is the point of doing this phase on hardware. The exact one holds because the
SP axis carries no traffic, so it distinguishes "reduced on the TP axis" from "reduced on
*an* axis" — a collective aimed at the SP axis mixes tokens in one placement and doubles the
statistics in the other, and neither shows up in a PCC test that stays self-consistent
within one placement. The mutation one exists because `tp_factor == 1` makes the whole
reduction an identity: on a single device, nothing in this module can tell a correct
sharded op from one that skips the collective entirely.

**Closed, this phase:** `ROOFLINE.md` — Blackhole constants each cited `file:line`, the
Gbps-vs-GB/s unit check, the DRAM floor, the fabric term, and two measurements that change
what Phase 9 should do first: `ttnn.all_reduce` picks between reduce-scatter+all-gather and
composite all-gather+local-sum by tile-unit divisibility (§5), and untraced launch overhead
(~130 µs) is above the 88 µs-per-launch break-even at production shape (§6).

**Left to Phase 9:** whether the split form's 1.50× survives 2× the collectives, and whether
batching `inter_block`'s dots (49 → 26) rescues it — `ROOFLINE.md` §6 prices those 23 saved
collectives at ~9 ms per forward untraced, which is the largest single lever measured so
far; whether the 31-of-32 wasted stats columns ever show up (§4 says 0.595 %, so probably
not); `num_links=1` → 2. Left to Galaxy: `(8,4)` with `[LINE, RING]`, where the payload
halves per device and the call count does not.

> **Answered 2026-07-30 (Phase 9).** All three, and I got two of the three predictions
> backwards. The split form survives at **1.47×** without any rescue (1.43× after the fold
> landed), so batching the dots is an optimization rather than the largest lever. The
> 31-of-32 wasted columns **do** show up — 7.4× on the bare collective, **147.6 µs per read
> once the fold's two permutes are charged for** (P6) — because padding is billed at the same
> rate as payload; "so probably not" was reasoning from a byte ratio about a cost that is not
> proportional to useful bytes. And `num_links = 1 → 2` is worth 1.48× only while the padding
> is there, so it and the fold are alternatives: with the fold in, the second link buys
> 4.7 µs, and **`num_links` stays at 1** — the opposite of what this section expected. Full
> record: `bringup_log.md` §Phase 9 perf loop.

> **A distribution fact this memo did not anticipate (P7).** Sharding the hidden dim is not
> only a memory and bandwidth argument — it decides which *primitives are available*. The
> one-pass sum of squares (`rms_norm_pre_all_gather`, 3.4× faster than `mul` + `sum`) keeps a
> whole row in one core's L1 and throws at program build past `W ≈ 5 664`. At `tp_factor = 4`
> the row is 1 792 wide and it fits; at `tp_factor = 1` the row is the full 7 168 and it does
> not, so single-device falls back. TP therefore *unlocks* a kernel rather than merely
> dividing work — and the production mesh is on the right side of the line, while the
> single-device path this bring-up validated against is on the wrong one. Worth remembering
> when a future op is benchmarked on one device and declared slow.

> **And the dot-batching question, closed (P8).** The paragraph above answered "does the split
> form need rescuing" — no — and left the batching as an unmeasured optimization. It is worth
> **1.26×** on the read and 49 collectives per block down to **26**, so the prediction this
> section got right was that the batching exists; what it got wrong is the shape (the site
> axis belongs in the last dim, not dim 1) and the sign at small `S`. The mixture, batched the
> same way, is worth **1.09×** and is rejected — see the amendment in §"The split form is
> worse on collectives".
