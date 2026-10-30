# DSpark: round-cost reduction

Status: plan. Follows `DSPARK.md` (what DSpark is and how it was built) and
`DSPARK_PARALLEL_VERIFY.md` (one step over all T = k+1 rows). Those two documents attack the model
forward. This one attacks the other 31%: **what a round does besides the model forward** — the
bookkeeping that tells the device which rows a reject invalidated. It is the cheaper half to fix,
because almost none of it is arithmetic.

Prior art: `/home/smanoj/DeepInfer` (`deepinfer/model_executor/models/mtp/dflash.py`,
`model_executor/models/v4_modules/attn.py` `DFlashAttention`, `model_executor/layers/sampler.py`,
`csrc/engine/`). Same drafter, same checkpoint, opposite deployment: CUDA, EP/DP, many users per
rank. Its *algorithms* transfer where they are about the drafter; its *scheduling* does not (section
5).

## 1. Where the round time goes

Measured (`DSPARK.md` 3.4), with the pipelined verify, k = 4, T = 5:

| phase | ms/round | what it is |
|---|---|---|
| verify | 54.0 | (8 + k) x s, s ~ 6.1 ms the slowest stage |
| draft | 17.0 | trace 13.0, `set_block` uploads 4.3, readback 0.2 |
| write | 5.0 | `spec_write_main`: trace 1.0, uploads 0.5, the rest waiting for the verify replays |
| fixup | 2.5 | `spec_fixup`: 4.8 isolated, uploads to all 8 stages |
| **total** | **78.5** | at 2.50 tokens/round = 31.8 tok/s/user |

The parallel verify (`DSPARK_PARALLEL_VERIFY.md`) cuts the 54 to ~31 and is the subject of that
plan. What is left is 24.5 ms of round that is **not** the model: **31% of the round**, and unlike
the verify it is almost all removable, because none of it is arithmetic. It is bookkeeping that
exists only to tell the device which rows a reject invalidated — and on a batch-1 pipeline, where
there is no one else to give the freed time to, that bookkeeping is pure loss.

DeepInfer's round has none of it. Four mechanisms account for the difference:

**(a) The drafter's `main_kv` write is never masked by acceptance.** It runs inside the main model
forward, before the accept is known, and writes *every* verified row. The reject is undone by the
next round, not by a mask:

```python
# deepinfer/model_executor/models/mtp/dflash.py
"""...Decode 时随主模型 CUDA graph 一起 replay，此时接受结果尚未产生，全部 token 都按
slot_mapping_store_swa 写入：将被拒绝的 draft 位置正是 forward_drafts 随后写 draft key 的位置，
在任何读取之前就会被覆盖，因此无需按接受结果屏蔽。"""
```

**(b) The sliding ring is longer than the window, so a reject cannot alias a live slot.**
`swa_block_size = window + MAX_BACKTRACKING_TOKENS` (`MAX_BACKTRACKING_TOKENS = nextn + 1`), and the
compressor state ring is `ratio + MAX_BACKTRACKING_TOKENS - 1`. There is no restore anywhere in
their spec path: no shadow, no fixup, no history.

**(c) Every per-round input is device-built.** One fused kernel produces the block's `positions`,
`input_ids`, `slot_mapping` and full `kv_index` from `[golden_token, last_valid_offset]`
(`get_fused_dflash_get_input_ref`, `model_executor/layers/tilelang_ops/get_kv_index.py`). We upload
one-hots, RoPE tables and a mask row from the host every round.

**(d) Accept, sample and finalize are device kernels over a slot-indexed proposal pool**
(`_verification` in `layers/sampler.py`, `fused_sample`, `get_accept_result`,
`finalize_verification`). Our accept is a host Python loop over k rows of logits that were just
copied to the CPU.

Sections 2-4 turn (a)-(c) into changes to this package. Section 5 is about what to leave alone.

## 2. The claim that makes (a) and (b) work

Both rest on the same inequality: **writing a row at the ring slot of a position that the next round
will overwrite is harmless, provided T is much smaller than the ring.**

Round N verifies `[p, p+T)`, accepts `[p, p+m]`, and round N+1 verifies `[p+m+1, p+m+1+T)`, which
*covers* every rejected position `[p+m+1, p+T)`. So a rejected row's slot is rewritten next round.
The only thing that can go wrong in between is a read of that slot by someone else. A reader of the
window ending at `p+m` touches positions `[p+m-W+1, p+m]`; a stale row sits at the slot of a position
in `[p+m+1, p+m+T-1]`, and two positions collide mod `W` only if they are `W` apart. With
`T - 1 < W` no live position is at an offset of `W` from a stale one, so nothing reads a stale row.

That is the whole proof, and it is why DeepInfer needs no rollback. The crucial clause is the last
one — *provided no live position is `W` apart from a stale one* — and it is the clause our rings fail.
It has three consequences here:

* **The drafter's ring is exactly `window` rows, so it is *not* safe as it stands.** Ring row `r`
  holds position `p+m+i` for a stale `i >= 1`, and that row is read as logical position
  `p+m+i-128`, which is inside the window ending at `p+m` whenever `p+m >= 128`. So dropping
  `written` is *unsound* unless the ring grows first — and it can grow, because the drafter's
  attention is a plain matmul with an additive `bias_row`, not a `cur_pos`-bounded SDPA over a
  contiguous ring. See 3.1.
* **The main model's sliding ring is also exactly `W`, and it *cannot* grow.** Same aliasing, but our
  SDPA reads it as a contiguous `[1, 1, W, Dh]` bounded by `cur_pos`
  (`flash_multi_latent_attention_decode`), and a longer ring is no longer position-contiguous. So
  there the fix is a different one: stop writing during the step and commit `m+1` rows afterwards
  (section 3.2), which removes the fixup by a different route.
* **The compressor window ring has the same aliasing and the same cheap fix as the drafter's**, and
  there the fix is literally DeepInfer's (section 3.3).

## 3. The changes

### 3.1 `write_main`: grow the drafter's ring, drop the mask, move it out of the host loop

**Step one, without which the rest is unsound: the ring must grow.** Section 2 says an unmasked write
is only safe if no live position is `window` apart from a stale one. The drafter's ring is exactly
`window` rows (`self.ring = ... zeros(1, 1, self.window, self.head_dim)`), so right now it *is*
aliased and `written` is load-bearing. Grow it to `R = window + T - 1` and index row
`pos % R` instead of `pos % window`.

The drafter can take this fix where the main model cannot, because its attention is not an
SDPA over a contiguous ring: it concatenates the ring with the block's keys and applies an additive
`bias_row`:

```python
# tt/decode/dspark.py  DeepSeekV41DSparkStage.attention
        keys = ttnn.concat([self.ring, kv], dim=2)
        scores = ttnn.matmul(q, keys, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=_HIFI4)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), ttnn.add(self.sink_bias, bias_row))
```

`bias_row` is free-form, so the extra `T-1` ring rows are simply masked by it along with the empty
slots — no `cur_pos`, no contiguity requirement.

**Budget.** The key axis is `[ring | kv padded to _KV_ROWS - ring_rows]`, so the *total* stays
`_KV_ROWS = 160` whatever the ring's depth; only the pad and the sink's column move:

```python
        kv = ttnn.pad(kv, [(0, 0), (0, 0), (0, _KV_ROWS - self.window - t), (0, 0)], 0.0)
```

`_KV_ROWS - R - t` = 160 - 133 - 5 = 22 rows of pad at `R = 133`, and the axis is
`155 = R + pad` either way, so nothing grows in DRAM and the score matmul keeps its width; the 5 block
keys still sit at the front of the pad. What moves is the sink's column: `window + block` = 133 today,
`R + block` = 138 tomorrow. Three places spell it — `sink_bias()`, the `bias_row` template in
`set_block`, and the `pad` above — and all three follow. The bound to assert is
`_KV_ROWS - R - t > t + 1` (22 > 6 at T = 6): the pad must still hold the block keys and the sink row.

Then the mask disappears. Today every round uploads, per round and per stage:

```python
# tt/decode/dspark.py  set_main_rows
        gather = torch.zeros(1, 1, w, self.stash_rows)
        scatter = torch.zeros(1, 1, self.window, w)
        for j, (p, keep_row) in enumerate(zip(positions, written)):
            gather[0, 0, j, p % self.stash_rows] = 1
            if keep_row:
                scatter[0, 0, p % self.window, j] = 1
```

and the ring update is a matmul against a 128-row one-hot, per stage:

```python
# tt/decode/dspark.py  DeepSeekV41DSparkStage.write_main
        kv = self._kv(main_x, cos, sin)
        written = ttnn.matmul(scatter, kv, compute_kernel_config=_HIFI4)
        ttnn.copy(ttnn.add(ttnn.multiply(self.ring, keep), written), self.ring)
```

Both go away. With the ring at `R = window + T - 1` rows, `written` is always all-True, so `gather`
becomes a one-hot that no longer depends on `m` (it already doesn't — it selects rows out of the stash
by `pos % stash_rows`), the ring write becomes `pos % R`, and `scatter` / `keep` are deleted. The ring
update becomes the same operation the decode path already uses for T rows:

```python
# tt/decode/attention.py  DeepSeekV41Attention.decode  (the S > 1 branch, sliding layers)
                ttnn.experimental.paged_update_cache(scache.kv, kv, update_idxs_tensor=sliding_pos)
```

`paged_update_cache` takes `[1, B, 1, F]` ROW_MAJOR height-sharded one row per core
(`_check_update_row`, `models/experimental/deepseek_v4_flash/tt/decode/attention.py`), so the
drafter's `kv` `[1, 1, T, Dh]` needs the `_one_row_per_user` reshape the decode path applies.

**Why this is worth more than the 5 ms it removes.** Because `write_main` no longer reads `m`, it can
be enqueued *before* the accept is known, so:

* it becomes the tail of stage 7's verify trace — right after the taps, at
  `_verify_submesh_static`'s `if li in sm.get("taps", {})` block (`tt/model.py` 2392), which already
  writes the stash rows the write gathers from. `spec_write_main` (`tt/model.py` 2091) and its
  two-trace handoff
  (`send_main` on the output stage, `write_main` on the drafter row) stop being a host step: the
  socket pair still carries the rows, but nothing waits on the host in between.
* the drafter's `draft` trace can be enqueued immediately after, so the drafter starts while the host
  is still reading verify logits and computing `m`. Today `spec.py` `step()` calls
  `model.spec_write_main(...)` at line 230, then `self._draft(...)` at 234; the second cannot start
  before the first returns. With the write in the trace, the ordering is device-side.

One host round trip per round disappears from the critical path, and the 5.0 ms `write` phase with
it (the trace's own 1.0 ms is now inside the verify step, overlapped with the next stage).

Constraint to assert: `R >= window + T - 1` and `_KV_ROWS - R - t > t + 1` (133 and 22 against 154 and
6 at T = 6), and `T <= stash_rows` (6 against 256). Also `T <= main_rows`; with the mask gone,
`main_rows` should just be `verify_rows`, and `set_main_rows`' padding of the tail with
`positions[-1]` and `written=False` is deleted.

### 3.2 The sliding ring: no write during the step, commit `m+1` rows after it

The step already builds each row's ring from the **pristine** ring, before touching the persistent
one:

```python
# tt/decode/attention.py  decode, sliding layers
            if s > 1:
                rings = ttnn.repeat(scache.kv, ttnn.Shape([s, 1, 1, 1]))
                self._write_causal_rows(kv, rings, ring_rows)
                ttnn.experimental.paged_update_cache(scache.kv, kv, update_idxs_tensor=sliding_pos)
```

Rows `0..j` are overlaid onto a copy of the ring, so the attention of every row is already correct
without the third line. The third line is only there so the *next* round sees the accepted prefix —
and it writes all T rows, which is why the fixup has to undo the rejects. So:

* **Delete the in-step `paged_update_cache`** in both the sliding branch and the compressed branch
  (`tt/decode/attention.py` 971 and 1055). The step becomes read-only on the ring.
* **Commit after the accept.** `DSparkSpeculator.step()` already knows `m` when it computes
  `self._rejected` (`tt/spec.py` 229, the `_rejected` assignment); add `model.spec_commit_ring(p, m + 1)`
  there, which cuts each stage's staged rows to `[0, m+1]` and issues one `paged_update_cache` per layer. The staged rows are the step's
  `kv` — they must be kept (one `[1, 1, T, Dh]` per layer, DRAM, ~6 KB) instead of being freed.
* **Delete `ring_shadow`, `fix_mask`, `fix_keep` and the ring half of `_spec_fixup`**
  (`tt/model.py` 1996-2013, allocated at 1896-1898): 5 layers x 128 x 512 bf16 = 640 KB per stage of
  DRAM, a full-ring copy every round, two 128-row multiplies and an add per layer, and two
  `copy_host_to_device_tensor` broadcasts to all 8 chips.

Ordering: the commit must precede the next round's verify. `_wait_replays_dispatched()` in
`spec_fixup` already establishes the pattern, so the commit takes that slot in the round. Whether it
costs anything depends on when it is dispatched — the drafter is on its own device and, after 3.1,
already running the draft, so the commit's dispatch overlaps 13 ms of draft trace. **Phase 0 must
measure this**: if the commit does not hide, the fallback is to trace it with a fixed `T`-row write
and reorder the copy so the rejected rows land in slots the next round rewrites anyway — which,
per section 2, they do at the *drafter*'s ring but not at the sliding ring, hence the eager commit.

A cheap optimisation to keep in reserve: `m+1` rows is one `paged_update_cache`, but the slice changes
shape with `m`. If that defeats tracing, cut with a device-side `view` over a fixed row count and a
per-row index list whose entries past `m` repeat the last accepted slot (a no-op write).

### 3.3 The compressor window ring: give it slack, delete the history

The r=2 window ring is exactly `r` rows:

```python
# tt/decode/attention.py  build_decode_caches
                    win_kv=zeros((1, 1, r, dh), ttnn.TILE_LAYOUT) if r > 1 else None,
                    win_gate=zeros((1, 1, r, dh), ttnn.TILE_LAYOUT) if r > 1 else None,
```

with `win_slot = pos % r`. With `r = 2` and `T = 5`, positions `p+m` and `p+m+2` share a slot, so a
rejected row at `p+m+2` overwrites the entry the next round's pool at `p+m+1` needs — the hazard
`DSPARK.md` 3.2 describes. `win_hist_kv` / `win_hist_gate` (`[1, 1, 32, Dh]` each, per r=2 source
per stage) plus `fix_hist` / `fix_slot` / `fix_slot_keep` exist to repair it.

DeepInfer's ring depth is exactly the fix:

```python
# deepinfer/worker/cache_engine.py
def get_inference_compress_block_size(compress_ratio: int) -> int:
    """Compression ring plus rollback history."""
    return compress_ratio + max(0, envs.DEEPINFER_MAX_BACKTRACKING_TOKENS - 1)
```

So: allocate `win_kv` / `win_gate` with `r + T - 1` rows, make `win_slot = pos % (r + T - 1)`, read
the pool's `r` entries as the `r` positions ending at the closing row, and delete `win_hist_kv`,
`win_hist_gate`, `fix_hist`, `fix_slot`, `fix_slot_keep` and the history half of `_spec_fixup`. With
`r + T - 1 = 6` on a 2-entry read, the ring is 3x bigger for a rounding error of DRAM (6 x 512 bf16 =
6 KB) and removes 32 x 512 x 2 bf16 = 32 KB plus the machinery per source per stage.

Note this also makes `hist_slot` in the verify packet (`_verify_packet`, `values["hist_slot"]`)
unnecessary, shrinking the packet.

### 3.4 Build the drafter block's inputs on device

`set_block` is 4.3 ms of pure upload — a quarter of the draft phase — and every tensor it writes is a
function of two integers, `token` and `frontier`:

```python
# tt/decode/dspark.py  set_block
        ids = torch.tensor([token] + [self.noise_id] * (t - 1))
        host_to_device(self.host_embed[ids].float().reshape(1, 1, t, self.hidden), self.inputs["x_embed"])
        ...
        cos, sin = self._rope_rows(range(frontier + 1, frontier + 1 + t))
        ...
        bias = torch.full((1, 1, 1, _KV_ROWS), _MASK_NEG)
        bias[..., : min(self.window, frontier + 1)] = 0
        bias[..., self.window : self.window + t + 1] = 0
```

Take the pattern the verify step already uses. `_verify_context` (`tt/model.py` 2235) builds exactly
these tensors from an integer packet, and `_prepare_verify_traces` (`tt/model.py` 2154) already
allocates per-stage RoPE tables read with `ttnn.embedding(idx, table)` (`rope_rows`). Apply it to the
drafter:

* block embedding: one int32 `[T]` row in DRAM (`[token, noise x (T-1)]`), gathered with
  `ttnn.embedding` from the backbone table already on the drafter's device;
* `cos` / `sin` / `q_cos` / `q_sin` / `q_neg_sin` / `kv_cos` / `kv_sin`: `ttnn.embedding` of
  `frontier + 1 + arange(T)` into the two half-tables, the negated copy a `multiply` on device;
* `bias_row`: constant except for one frontier-dependent column, so build it as
  `const_neg + scatter(is_live, dim=window+t)`, with `const_neg` a persistent row for each of the two
  cases (`frontier + 1 < window` or not).

The per-round H2D then drops from ~50 KB fp32 to one int32 token and one int32 frontier — which is
what `get_fused_dflash_get_input` reduces its own host traffic to. This can be done in the traced
`draft`, since the packet write is the same mechanism as `_build_packet`.

### 3.5 The drafter's mHC: T calls to one

`DeepSeekV41DSparkHyperConnection.forward` loops per token because the multi-token program does not
fit in the output stage's L1:

```python
        for i in range(t):
            streams = ttnn.to_memory_config(...)
            outputs = ttnn.experimental.deepseek.fused_hyperconnection(...)
```

that comment is about the *output stage*. The drafter now lives on a row of its own
(`prepare_spec_decode` picks `drafter_row = self.num_submeshes`, the first row the pipeline leaves
idle) with its own prefetch rings, so the L1 budget is different. 3 stages x 2 mHC = 6 calls at T=5
becomes 30; making it 6 requires the multi-token `fused_hyperconnection` program to fit, which is
plumbing and a budget, not an algorithm. Measure before committing to it (Phase 0), because it is the
largest single item left in the 13 ms draft trace.

### 3.6 Drafter expert precision: A/B on acceptance, not PCC

The drafter's MoE is bfp4 (`expert_dtype=ttnn.bfloat4_b`, `test_dspark.py`), and `DSPARK.md` 3.4
records what that costs on random taps: end-to-end logits PCC 0.91 mean with 5/8 first proposals
matching, against bf8's 0.987 (first lap) / 0.946 (wrapped) with 7/8 and 4/8. The drift is routing
near-ties in the top-3-of-128 flipping under bf16 scores, not arithmetic error. Our measured
acceptance histogram is bimodal in a way that looks like that:

```
rounds with 0..4 accepted: 8, 8, 3, 3, 4      (2.50 tokens/round)
```

Deriving the conditional profile from that histogram, `P(accept draft j | drafts < j accepted)` is
**0.69, 0.56, 0.70, 0.57** — essentially flat at `p ~ 0.63`, and a flat `p = 0.63` predicts
`1 + p + p^2 + p^3 + p^4 = 2.43` tokens/round against the measured 2.50. So nothing here is a
first-token problem; every position accepts at roughly the same rate, which means the value of
raising it is multiplicative at *every* row:

| flat p | expected tokens/round |
|---|---|
| 0.63 (measured) | 2.43 (measured 2.50) |
| 0.75 | 3.05 |
| 0.80 | 3.36 |

That is why this item is worth more than the milliseconds items: it is the only one that moves the
numerator. 8 of 26 rounds (31%) accept nothing at all, and each of those pays a full T-row verify to
emit one token — that is what a flat, mediocre `p` costs.

The test harness already parametrizes it (`test_dspark.py` `expert_dtype`), and the drafter row has
the DRAM (`~2.3 GB/chip` used of the drafter's row, `~9.6 GB/chip` of main experts on other chips).
Gate on **accepted drafts per round and the per-position profile `P(m >= j)`**, over the same prompt
set at fixed k — not on PCC, which measures the wrong thing.

## 4. Order, and what it is worth

The arithmetic below is *projected* — to be replaced by `SpecStats.phases` — but it is done
honestly, with each row's saving taken from the phase it actually belongs to and the new commit
cost added back rather than assumed away. Baseline is the parallel verify (~31 ms, from
`DSPARK_PARALLEL_VERIFY.md` §1) on top of section 1's other phases: **55.5 ms per round at 2.50
tokens = 45 tok/s/user**.

| # | change | effort | removes | adds | round |
|---|---|---|---|---|---|
| 1 | 3.1 unmasked `write_main`, folded into the verify trace | S | 5.0 ms write phase + one host round trip | 0.5 (in-trace) | 55.5 -> 51.0 |
| 2 | 3.3 window-ring slack | S | 2.5 ms fixup, 32 KB x 2 per source per stage | 0 | 51.0 -> 48.5 |
| 3 | 3.2 deferred ring commit | M | 640 KB/stage shadow, `fix_mask`/`fix_keep` broadcasts | 1.5 ms commit | 48.5 -> 50.0 |
| 4 | 3.4 device-built drafter inputs | M | 4.0 of the 4.3 ms upload, 50 KB/round H2D | 0 | 50.0 -> 46.0 |
| 5 | 3.5 drafter mHC, T rows in one call | M | up to ~5 ms of the 13 ms trace | 0 | 46.0 -> 41.0 |
| 6 | 3.6 bf8 drafter experts | S | nothing | nothing | 41.0, at 3.05 tokens |

Note 3 is **not** a speedup on its own: the commit is a new serialised step and the fixup it
replaces was already mostly hidden. It buys DRAM, removes the last piece of rollback bookkeeping, and
is the prerequisite for a later change (fold the commit into the next round's packet, which removes
it too). Anyone reading this table as "1-3 is a win" should read it as "1 and 2 are a win, 3 is
cleanup that must be measured".

So: 1 and 2 give **~48.5 ms = 51.5 tok/s/user** with acceptance unchanged, from two small, provably
safe changes. Add 4 and 5 and it is **41 ms = 61 tok/s/user**.

6 is the only item that moves the numerator, and the profile in 3.6 says how much: a flat `p` of
0.63 -> 0.75 is 2.43 -> 3.05 tokens/round, which in a 41 ms round is **~74 tok/s/user**; 0.80 would be
3.36 tokens and ~82. That is more than 4 and 5 combined, from an `expert_dtype` argument the test
already parametrizes — which is why it is Phase 5 by cost but should be *measured* in Phase 0, before
anything else is built, in case the answer redirects the work.

For comparison, `DSPARK_PARALLEL_VERIFY.md` §1 expects the wide verify at T = 6 to cost ~31-40 ms
against the pipelined ~61 ms, which for our measured split is the 54 -> 31 ms already in the baseline
above. The `DSPARK.md` 3.1 table's 73 tok/s/user at k = 5 should **not** be read as a target for
either plan: it assumes metal_2's acceptance profile (4.2 tokens/round) rather than our measured 2.50.

**Phases, each with an exit criterion:**

* **Phase 0 — measure, no code.** Four measurements, all of them cheap and all of them able to
  redirect the plan:
  1. `expert_dtype` bfp4 vs bf8 acceptance at fixed prompts (item 6). If it moves `p` materially, do
     it first — it is one argument.
  2. `SpecStats.phases` already splits fixup/verify/write/draft; add a `commit` phase and time the
     commit (i) eager, (ii) traced.
  3. `set_block` vs. a device-built equivalent for one stage.
  4. `fused_hyperconnection` at T rows on the drafter's own device (the L1 budget differs from
     stage 7's).
  Exit: the 24.5 ms is accounted for, item 6's value is known, and 3.2 is either traced or eager.
* **Phase 1 — 3.1 alone.** The ring growth and the mask removal are one change, and the ring growth is
  the part that must be right. Exit: `test_dspark.py` (stage-0 attention PCC, end-to-end logits PCC vs.
  the reference) plus `test_dspark_spec.py` exactness with the corrupted oracle, and a new case that
  seeds 140+ positions so the *wrapped* lap is exercised with `T > 1` (the aliasing only bites once
  `p + m >= window`), and `write` gone from the phase split.
* **Phase 2 — 3.3.** Exit: same exactness, and a targeted test that a block ending on an even
  position with all-but-last rejected produces the same compressed pool as the pipelined path
  (`tests/decode/test_attention.py` r=2 cases extended to a verify block).
* **Phase 3 — 3.2.** Exit: exactness across the ring wrap with `m` forced to every value 0..k-1 in
  turn, since the commit is the only thing the reject now affects.
* **Phase 4 — 3.4, 3.5.** Exit: `test_dspark.py`'s traced-vs-eager check (`TRACED_PCC`) and the
  draft trace time.
* **Phase 5 — 3.6.** Exit: accepted-drafts/round at fixed prompts, recorded before and after.

## 5. What not to do: adaptive k

DeepInfer's headline optimisation is variable-length verification: the confidence head predicts
`P(draft j accepted | drafts < j accepted)`, the engine cumprods it into per-request accept rates,
and a greedy marginal-return walk over a profiled `(graph_size -> forward_ms)` curve chooses each
request's verify length (`MTPVarlen::schedule`, `csrc/engine/mtp_varlen.hpp`). Measured: verified
length 5.00 -> 3.24-4.44 at accepted length 3.29 -> 3.12, for +4-7% at ~60 sequences/rank.

**It is worth approximately zero here, and the reason is structural.** Their gain is *allocation*:
with 60 users per rank, one row of verify costs shared compute, so trimming rows from requests
unlikely to accept frees it for requests that will. On a batch-1 pipeline there is no one to give
the freed row to, and the cost of a row is not its FLOPs:

* pipelined verify costs `(S + T - 1) * s` — one slowest-stage time per row, independent of which
  request it belongs to. Trimming `T` by two saves `2s` and costs the expected tokens those two rows
  would have produced. Net zero at best;
* parallel verify costs `S * s_T` with `s_T` nearly flat in `T`. That flatness is the *premise* of
  `DSPARK_PARALLEL_VERIFY.md`; if it holds, trimming `T` saves nothing at all.

`DSparkSpeculator` already computes the confidence head on device
(`DeepSeekV41DSpark.draft` returns `conf`) and `spec_draft` already returns it — then the test
ignores it. That is the right state: keep the number (it is nearly free and it is the metric that
says whether 3.6 worked), but **do not build the scheduler**. The lever for a batch-1 pipeline is
acceptance, not row allocation.

The one condition that would revive it: if Phase 0 of `DSPARK_PARALLEL_VERIFY.md` finds a stage where
`s_T` rises steeply — the multi-query select at long context is the named risk — then the marginal
row is expensive again and a confidence-driven `T` is worth revisiting. Revisit it then, not now.

Two smaller things from DeepInfer that are also not worth porting as they stand:

* **Device-side accept and finalize** (`get_accept_result`, `finalize_verification`). Ours is a host
  loop over k rows, but with fixed k and a batch of one it is microseconds of Python, and the 1.5 MB
  of logits it reads is the only reason it exists. Fold an in-trace `argmax` into the verify trace
  (already `DSPARK.md` Phase 5 item 1) and the accept becomes k integer comparisons; then the device
  kernels have nothing left to do.
* **`accept_margin`.** `DSPARK.md` and the test both default it to 3 logits, i.e. the DSpark run's
  output is not the greedy stream. That is a test convenience that should not survive into a
  shipped path, and it makes every exactness claim conditional. Either delete it (greedy, exact) or
  implement DeepInfer's rule (`min(1, q/p)` then a residual draw, `fused_sample`) which is exact for
  `temperature > 0` and which `tt/spec.py` already approximates on the host in `_accept`.

## 6. Correctness gates

All of section 3's changes are about *when* a row lands in a cache, so the gate is unchanged from
`DSPARK.md` Phase 1 and must be run at every phase: the corrupted-oracle stream equals plain greedy
token for token over >= 256 generated tokens, crossing the ring wrap, both r=2 parities and the
two-level boundary. One case must be added, because it is what 3.1-3.3 rest on:

* **forced worst-case rejects.** With `propose` returning drafts the target will reject at a chosen
  offset, run `m = 0, 1, ..., k-1` in a fixed pattern (e.g. round robin) so the "next round covers
  every stale slot" argument is exercised at each offset, not just wherever the model happens to
  diverge.

Plus, per phase: `tests/decode/test_dspark.py` (stage-0 attention PCC, traced vs. eager draft,
`TRACED_PCC`) for 3.1's ring change and for 3.4-3.5, and a new `tests/decode/test_verify_commit.py`
comparing one verify step + commit against T single-row steps at the layer level, in the style of
`DSPARK_PARALLEL_VERIFY.md`'s `test_verify_rows.py`.

## 7. Files

Follows `PORTING.md`: V4.1 files hold only what changed.

| file | change |
|---|---|
| `tt/decode/dspark.py` | ring depth `R = window + T - 1`, `pos % R`, sink column and `bias_row` template follow; `write_main`: `paged_update_cache`, no `scatter` / `keep`; `set_main_rows` without `written`; `set_block` inputs built on device (3.4) |
| `tt/model.py` | `spec_write_main` folded into `_verify_submesh_static`'s tap block; `spec_fixup` loses the ring half (3.2) and the history half (3.3); new `spec_commit_ring`; `_verify_packet` drops `hist_slot`; `ring_shadow` / `fix_*` allocation removed from `prepare_spec_decode` |
| `tt/decode/attention.py` | `decode`: no in-step ring write for `s > 1`; window ring `r + T - 1` and `win_slot` modulo it; `build_decode_caches` sizing; `decode_inputs` / `_latent_rows` follow |
| `tt/spec.py` | `step()`: commit after the accept, `write` phase replaced by `commit`; `accept_margin` removed or made exact |
| `tt/decode/dspark.py` (mHC) | `DeepSeekV41DSparkHyperConnection.forward` at T rows, if Phase 0 clears L1 (3.5) |
| `tests/decode/test_verify_commit.py` (new) | one verify step + commit vs. T single steps, per layer kind |
| `tests/decode/test_dspark_spec.py` | the forced-worst-case-reject round pattern (section 6) |

## 8. Risks

* **The commit is a new host-serialised step.** If the draft does not overlap it (a socket read that
  blocks, or the drafter's trace not being enqueued yet), Phase 0's measurement will show it as
  ~2-3 ms and 3.2 buys little. The mitigation is the traced `T`-row commit with a no-op write for
  rows past `m`, kept in reserve.
* **Losing the shadow removes an exactness oracle.** `ring_shadow` is also what makes `_spec_fixup`
  idempotent and cheap to reason about. With deferred commit, correctness depends on the *ordering*
  argument in section 2 rather than on a restore. Hence the forced-reject gate.
* **Window-ring slack changes the r=2 read.** The pool must read the `r` positions ending at the
  closing row out of a ring of `r + T - 1`; the pairing code (`_latent_rows`, `pool_rows`, the
  `closing` computation in `decode_inputs`) all key on `pos % r` today. Every one of them has to move
  together, or a variant of the trace silently reads the wrong pair.
* **`paged_update_cache` row layout.** It requires `[1, B, 1, F]` ROW_MAJOR height-sharded one row per
  core; the drafter's `kv` comes out of `matmul_decode` in a different shape. The reshape is the
  `_one_row_per_user` path the decode layers already use, but it must be inside the trace and it must
  not become a per-row loop.
* **The ring growth moves the sink column.** Both `sink_bias()` and the `bias_row` template hard-code
  `self.window + self.block` as the sink's key column, and `attention` pads to
  `_KV_ROWS - self.window - t`. All three move to `R`. Getting one of them wrong does not crash — it
  shifts every score by a masked or unmasked column — so it will show up as a PCC drop, not an assert.
  This is the most likely way to get 3.1 subtly wrong, and the wrapped-lap test in Phase 1 is what
  catches it.
* **Trace count.** Folding the write into the verify trace means stage 7's verify trace now does a
  socket send and the drafter's `write_main` a socket receive; both are currently separate traces with
  their own ordering. Check `TRACE_SIZE_SCALING.md` before adding, not after.
