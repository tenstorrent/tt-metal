# DSA indexer: proper cache ops (kill the O(n²) concat)

Plan to replace the indexer's grow-by-`concat` key cache with a persistent, slot-indexed cache
read directly by `indexer_score_dsa` — mirroring how dense MLA uses its `kvpe_cache`.

Status: design / not started. Scope: `tt/mla/indexer.py` (+ optional threading through
`tt_prefill_runtime` → `tt_prefill_transformer` → `tt_prefill_block` → `mla.py`).

---

## 1. Current state (the problem)

The indexer maintains its index-key cache as a single device tensor `self._index_kbuf`, grown by
`ttnn.concat` once per prefill chunk:

- `indexer.py:238` — `self._index_kbuf = None` (no pre-allocation).
- `indexer.py:405-413` — per chunk: `start_pos == 0` ⇒ assign; else
  `self._index_kbuf = ttnn.concat([old, k], dim=2)` then `deallocate(old)`.
- `indexer.py:483` — scoring reads the raw buffer: `indexer_score_dsa(q_dev, self._index_kbuf, weights, chunk_start_idx=start_pos, ...)`.

The author already flagged this as a scoped follow-up (`indexer.py:397-401`):

> concat reallocates the whole key-cache each chunk (O(n²) copies over a long prefill). Reusing
> the MLA block-cyclic KVPE cache machinery … would avoid that, but it changes the indexer
> key-cache layout … so it is tracked as its own task rather than done here.

### Why it matters

1. **O(n²) DRAM copies.** Each chunk `concat`s the *entire* prior prefix into a fresh buffer. Over a
   long chunked prefill the total copy volume is quadratic in sequence length. (Single-shot prefill is
   unaffected — `start_pos == 0` just assigns — so this is strictly a *chunked-prefill* cost.)
2. **No persistent allocation / not trace-stable.** The buffer's shape changes every chunk, so the
   recorded op graph differs per chunk — counter to the design goal that "a single recorded op trace
   replays identically for any input" (`mla.py:992-997`). Dense MLA avoids this with a fixed-size
   persistent cache.
3. **No multi-user / multi-layer slotting.** `kvpe_cache` is one shared buffer of
   `num_users * num_layers` slots, user-major (`kv_cache_utils.py:282-284`,
   `tt_prefill_runtime.py:167-179`). The indexer has nothing equivalent — one ad-hoc buffer per
   `TtIndexer` instance, single user.
4. **Blocks sparse chunked + migration.** The sparse chunked path already raises `NotImplementedError`
   for `on_layer_complete` (`mla.py:1140-1146`); a proper persistent indexer cache is a prerequisite
   for ever wiring that up.

---

## 2. Key finding: `indexer_score_dsa` already supports a proper cache contract

The grow-by-concat buffer is **not** required by the scoring op. `indexer_score_dsa` already has a
fully-tested persistent-cache contract that the indexer simply isn't using:

- **Slot select** — `cache_batch_idx` picks a slot of a shared `[B, 1, T, D]` cache; re-applied each
  dispatch, *not* in the program hash → switching slots does not recompile
  (`indexer_score_nanobind.cpp:52-56`, `indexer_score_device_operation.cpp:68-77`).
- **Growing prefix** — `kv_len` is the valid prefix of a `k` allocated at its full `T`; a serving loop
  growing `kv_len (≤ T)` reuses one program (hash-excluded); only columns `[0, kv_len)` are written
  (`indexer_score_nanobind.cpp:57-60`, `device_operation.cpp:78-87`, `:348-353`).
- **ND-sharded DRAM cache** — the `[B,1,T,D]` cache "may then be ND-sharded across DRAM banks"; tested
  in `test_indexer_score.py::test_indexer_score_indexed_cache_nd_sharded_k` and
  `::test_indexer_score_runtime_kv_len`.
- **TILE bf16** — the op consumes the cache in `TILE_LAYOUT` bf16 (op tests `to_device(..., layout=TILE)`),
  which is exactly what the indexer's `_device_rope_pe` already produces.

This is the same shape of contract dense MLA's sparse path uses for `kvpe_cache` — so the indexer can
adopt "proper cache ops" by *using the op as designed*, not by porting MLA's block-cyclic machinery.

---

## 3. Layout decision: replicated-natural, **not** block-cyclic

The follow-up comment proposed reusing MLA's block-cyclic SP-sharded KVPE machinery
(`update_padded_kv_cache` + `_gather_kvpe_prefix` un-rotate). **That is the wrong target for the
indexer**, because the two caches have different read patterns:

- **MLA sparse attention** reads back the prefix, un-rotates block-cyclic→natural, and each chip
  attends only the top-k *selected* latents — so SP-sharding the storage (1/sp per chip) + a gather on
  read is a net memory win (`mla.py:1315-1337`, `utils.py:148-178`).
- **The indexer scores every query against ALL keys.** `cluster_axis` only sets the per-device *query*
  causal offset (`device_operation.cpp:23-35`: `chunk_start + rank*Sq`); it does **not** SP-shard `k`.
  In every mode `k` must be the full prefix on each device. The current buffer is already
  full/replicated/natural (`indexer.py:394` SP all-gather → `:483` scores `cluster_axis=None`).

**⇒ Keep the cache replicated-natural, full-`T`, on every chip.** No block-cyclic reorder, no
`_gather_kvpe_prefix` un-rotate, no `block_cyclic_reorder`'d RoPE tables. This is *simpler* than the
MLA machinery and preserves the exact scoring semantics. Crucially, **memory footprint is unchanged**:
today's concat buffer is *also* replicated-full — we only make it persistent and stop reallocating it.

RoPE stays where it is: `write_k` already applies RoPE to the new chunk's keys at their global
positions `[start_pos, start_pos+glob)` (`indexer.py:395`), so once written the keys are
position-correct forever — no re-roping on read.

---

## 4. The write op: `ttnn.fill_cache(..., update_idx=start_pos)`

The keys are already full/replicated/natural `[1, 1, glob, D_idx]` after the SP all-gather + RoPE
(`indexer.py:394-395`). Writing them at sequence offset `start_pos` is exactly:

```python
ttnn.fill_cache(self._index_kcache, k, batch_idx=slot, update_idx=start_pos)
```

`fill_cache` "fills the cache in place … optionally offset along the sequence dimension by `update_idx`
(tile-aligned)" — `update_idx` must be a multiple of `TILE_HEIGHT (32)`. This drops in for the concat
block (`indexer.py:405-413`). Alignment holds: `glob = seq_len * sp_factor` and the chunked
`start_pos = kv_actual_isl` are chunk-size multiples (the code already asserts `sq % TILE == 0` and
`end_pos % 16 == 0`, `indexer.py:438,492`); add an explicit `start_pos % 32 == 0` assert.

---

## 5. Scoring change

```python
logits = ttnn.experimental.indexer_score_dsa(
    q_dev,
    self._index_kcache,          # [B, 1, T_full, D_idx], persistent
    weights,
    chunk_start_idx=start_pos,
    program_config=cfg,
    cluster_axis=None,
    cache_batch_idx=slot,        # NEW: user*num_layers + layer slot (omit only if B==1)
    kv_len=end_pos,              # NEW: valid prefix this chunk (tile-aligned, ≤ T_full)
)
```

`k_chunk_size = min(64, end_pos)` (`indexer.py:478`) stays — it caps to the live prefix, which `kv_len`
now expresses to the op. With `kv_len` set, the op writes only `[0, end_pos)` columns; the topk over
`logits` is unchanged.

---

## 6. Implementation — phased commits

**Commit 1 — swap concat → persistent fill_cache (indexer-owned, B=1).** De-risk the op contract
before any threading.
- Allocate one persistent `[1, 1, T_full, D_idx]` cache in `TtIndexer.__init__`
  (`T_full = config.max_seq_len`, replicated, TILE bf16), zeroed.
- `write_k`: replace the concat block with `fill_cache(update_idx=start_pos)`; drop the
  reset/dealloc dance (a new request at `start_pos==0` overwrites from offset 0; `kv_len` hides the
  stale tail, so no zeroing needed).
- `forward`: pass `kv_len=end_pos` (and `cache_batch_idx` is unnecessary at `B==1`).
- Validate: single-shot **and** chunked prefill PCC unchanged vs the concat path; confirm no recompile
  as `kv_len` grows (`device.num_program_cache_entries()` stable across chunks).
- ⚠️ **Memory caveat:** sized to `T_full` per layer (today's buffer is sized to the live prefix). For
  large `max_seq_len × num_layers` this can regress peak DRAM — measure before landing. This is the
  reason Commit 2 (one shared right-sized buffer) is the real target, not the end state.

**Commit 2 — runtime-owned shared cache (full parity with `kvpe_cache`).**
- Add an `init_index_kcache` helper (mirror `init_kvpe_cache`, `kv_cache_utils.py:261`) with
  `head_dim = index_head_dim` and a `[num_users*num_layers, 1, T_full, D_idx]` shape, **replicated**
  (not the SP-block-cyclic write path — plain replicated/interleaved DRAM first; ND-shard later only
  if capacity demands).
- Allocate once in `tt_prefill_runtime._allocate_kv_cache` (`:167`) next to `kvpe_cache`, gated on the
  model being sparse.
- Thread `index_kcache` through `transformer.forward` (`tt_prefill_transformer.py:288,360`) →
  `block.forward` (`tt_prefill_block.py:378,418`) → `mla.forward` → `indexer.forward`, exactly like
  `kvpe_cache` + `cache_user_id` + `cache_layer_idx`.
- `write_k`/`forward` use `slot = cache_user_id * num_layers + cache_layer_idx` for `batch_idx` /
  `cache_batch_idx` (user-major, matching `kv_cache_utils.py:282-284`).
- `NullIndexer` ignores the new arg (dense v3.1 unaffected).

Keep Commit 1 self-contained and validated before Commit 2 (per "commit in small, validated units").

---

## 7. Open questions / validation items

1. **`fill_cache` on an ND-sharded DRAM cache.** The op test builds the ND-sharded `k` via `from_torch`,
   not `fill_cache`. Start with **interleaved DRAM** (definitely `fill_cache`-compatible); only move to
   `NdShardSpec` if capacity requires it, and validate `fill_cache` writes it correctly first.
2. **Cache dtype.** Keep **bf16** (top-k selection is precision-sensitive; current buffer is bf16).
   bf8 would halve DRAM but needs a PCC check on top-k index agreement — defer.
3. **Memory budget.** `replicated × T_full × 128 × 2B × num_users × num_layers` per chip. Compute
   against the target board's DRAM before Commit 1; this drives whether `T_full` persistence is
   acceptable or Commit 2 must land together.
4. **Tile alignment of `start_pos`.** Assert `start_pos % 32 == 0` in `write_k`; document the
   `kv_len % 32 == 0` op requirement (`device_operation.cpp:83`) — `end_pos` already tile-aligned.
5. **Reset across requests.** With a persistent cache, a new request overwrites from offset 0 and
   `kv_len` bounds the read — confirm no cross-request bleed when `num_users > 1` shares the buffer.

---

## 8. Out of scope

- **Sparse chunked-prefill migration** (`on_layer_complete`, pad-zeroing) — still
  `NotImplementedError` (`mla.py:1140-1146`); this plan unblocks it but does not implement it.
- **Decode path** for the indexer.
- **SP-block-cyclic indexer storage** (an `update_padded_kv_cache` + SP-ring `cluster_axis` redesign to
  cut the replication factor) — a separate, larger memory optimization; only worth it if §7.3 shows the
  replicated cache is too large.

## 9. Test plan

- `tests/.../test_indexer_score.py` already covers the op contract (`cache_batch_idx`, `kv_len`,
  ND-shard, no-recompile, OOB reject) — reuse as the op-level guarantee.
- Model-level: the existing sparse MLA device test, swept over SP×TP, single-shot **and** chunked, with
  ≥2 chunks so the offset-write + growing-`kv_len` path is exercised; assert top-k indices / final PCC
  match the pre-change concat path bit-for-bit (RoPE and scoring are unchanged — only storage moves).
