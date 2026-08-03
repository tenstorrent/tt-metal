# Design: per-rank dense indexer-layer compaction for pipelined GLM-5.2

Status: implementation plan, decided (per-rank dense). Scope: **GLM-5.2 sparse (DSA) indexer key cache**
under **pipeline (layer) parallelism**. Touches the indexer, the GLM-5.2 adapter, and the layer-idx
threading through the transformer/block/MLA. Orthogonal to (and stackable with) the KV TP-sharding work
in [`glm52_kv_cache_tp_sharding.md`](./glm52_kv_cache_tp_sharding.md).

All file:line citations re-verified against the working tree on 2026-07-24 (this code churns; re-check
before editing).

---

## 1. Problem & payoff

GLM-5.2's DSA has **cross-layer indexer reuse**: `config.indexer_types` marks each layer `full` or
`shared`. Only `full` layers own a lightning-indexer and **write** the indexer key cache; `shared` layers
reuse the most recent full layer's top-k and never write. So the index key cache is *compacted* to the
full-layer count, not `num_layers`.

For the production 78-layer config the full layers are `{0,1,2} ∪ range(6,78,4)` = **21** layers:

```
{0,1,2, 6,10,14, 18,22,26,30,34, 38,42,46,50,54, 58,62,66,70,74}
```

Under pipeline parallelism the layers are split across ranks. For the `[18,20,20,20]` split the full
layers land **non-uniformly**:

| rank | global layers | full layers                | count |
|------|---------------|----------------------------|-------|
| 0    | 0–17          | {0,1,2,6,10,14}            | **6** |
| 1    | 18–37         | {18,22,26,30,34}           | **5** |
| 2    | 38–57         | {38,42,46,50,54}           | **5** |
| 3    | 58–77         | {58,62,66,70,74}           | **5** |

So **rank 0 owns 6 full-layer indexer caches to communicate/migrate; ranks 1–3 own 5 each.** The per-rank
index cache and its disaggregation KV chunk address table therefore **differ per rank** (6 vs 5 layers).

Each pipeline rank starts on a `full` layer (0/18/38/58 are all full) — required so the rank's
indexer-reuse chain has a seed. This is enforced by `GLM52Adapter.layer_split_boundaries`
(`tt/runners/adapters/glm_5_2.py:100-105`), which returns the set of full-layer indices as the valid
rank-start boundaries.

## 2. Current behavior (global compaction) and why it's wrong for pipeline

Today the compaction is computed against the **whole** config, so it is *global*, not per-rank:

- Layers are constructed with the **global** index: `layer_idx = first_layer_idx + local_idx`
  (`tt/tt_prefill_transformer.py:195`).
- `config.indexer_types` is the **global** 78-entry map and is never sliced per rank
  (`reference/glm_5_2_config.py:117`; the transformer offsets into it as
  `indexer_types[first_layer_idx + i]`, `tt/tt_prefill_transformer.py:395`).
- `TtIndexer.__init__` (`tt/mla/indexer.py:278-281`) therefore derives, off the global config:
  - `_num_index_layers = num_full_indexer_layers(config)` = **21**
  - `_index_layer_idx = full_indexer_rank(config, layer_idx)` = a **global** rank in `0..20`
  - `_index_cache_layers = _num_index_layers` = **21**
- `GLM52Adapter.allocate_kv_cache` sizes the index cache to `num_full_indexer_layers(hf_config)` = **21**
  (`tt/runners/adapters/glm_5_2.py:87`).
- The merged migration table sizes its index config from the cache: `cfg.num_layers =
  index_cache.shape[0] // num_users` (`tt/runners/kv_chunk_table.py:151`).

**Consequence:** every rank allocates a **21-wide** index cache and writes its layers into their *global*
rank slots (rank 0 → slots {0..5}, rank 1 → {6..10}, …). The per-rank table is a uniform 21 entries and
points **all 21** layers at *this rank's* cache — but a rank only ever wrote its own 6/5 slots, so the
other ~15 slots are zeros. In a real pipelined migration a consumer that reads a global layer from the
wrong rank's table gets zeros. GLM-5.2 serving is not wired yet (`tests/conftest.py:37`), so this latent
bug has not been exercised.

## 3. The design: rank-local dense compaction

Each rank's index cache holds **only its own full layers** (6 or 5), addressed by **rank-local** ranks
`0..local_full-1`. The migration table then naturally has `num_layers = 6` on rank 0 and `5` on ranks
1–3 (it already sizes itself from the cache's batch dim — see §4.4).

**The arithmetic reuses the existing global prefix-sum `full_indexer_rank`**, so the helper logic barely
changes — it is applied over the rank's window `[first_layer_idx, first_layer_idx + num_layers)`:

```
rank_local_full_count = full_indexer_rank(config, first_layer_idx + num_layers)
                      - full_indexer_rank(config, first_layer_idx)

rank_local_slot(layer_idx) = full_indexer_rank(config, layer_idx)
                           - full_indexer_rank(config, first_layer_idx)
```

`full_indexer_rank(config, p)` = count of `"full"` in `indexer_types[:p]` (`tt/mla/indexer.py:740-747`),
so both quantities are just prefix-count differences over the rank's layer range.

**Backward compatible.** Whole-model / non-pipelined (`first_layer_idx=0`, `num_layers=NUM_LAYERS`) reduces
exactly to today's global values. GLM-5.1 / DeepSeek-V3.2 carry no `indexer_types` → the helpers return
`None` → the code falls back to `num_layers` as today. So only pipelined GLM-5.2 changes.

## 4. Change-site map

### 4.1 Indexer compaction (`tt/mla/indexer.py`)
`TtIndexer.__init__` (`:278-281`) currently computes global `_num_index_layers` / `_index_layer_idx`. Make
them **rank-local** using the two prefix-count differences above. This requires the indexer to know the
rank window — thread in `first_layer_idx` and the rank's `num_layers` (see §4.2). The runtime slot formula
`cache_batch_idx = cache_user_id * _index_cache_layers + cache_layer_idx` (`:533`, with
`cache_layer_idx = _index_layer_idx` at `:520-521`) is unchanged — it just uses the smaller rank-local
stride and rank-local slot.

Optionally add thin rank-window helpers next to `num_full_indexer_layers` / `full_indexer_rank`
(`:731-747`) so callers don't open-code the subtraction, e.g.
`num_full_indexer_layers_in_range(config, start, count)` and
`full_indexer_rank_local(config, layer_idx, start)`.

### 4.2 Thread `first_layer_idx` down to the indexer
The transformer already has `first_layer_idx` (`tt/tt_prefill_transformer.py:156,195`) but does not pass it
into the block/MLA/indexer construction (the block only gets the global `layer_idx`,
`tt/tt_prefill_block.py:142,206`). Thread `first_layer_idx` (and the rank's layer count) through
`TtPrefillBlock` → `ttMLA` → `TtIndexer` so §4.1 can compute the rank window. `ttMLA` already receives a
`layer_num` (rank layer count) in some paths — reuse it rather than adding a duplicate.

### 4.3 Adapter allocation (`tt/runners/adapters/glm_5_2.py`)
`allocate_kv_cache` (`:87`) sizes the index cache to the **global** `num_full_indexer_layers(hf_config)`.
Change it to the **rank-local** full count over `[params.first_layer_idx, params.first_layer_idx +
params.num_layers)` (params already carries `first_layer_idx` and the per-rank `num_layers` — confirm on
the `PrefillModelAdapter`/params object). Falls back to `params.num_layers` when there is no
`indexer_types` map (GLM-5.1 / v3.2), unchanged.

### 4.4 Migration table — no change
`_build_and_serialize_merged_kv_chunk_table` sets the index config's `num_layers =
index_cache.shape[0] // num_users` (`tt/runners/kv_chunk_table.py:151`) and
`populate_kv_chunk_address_table_kimi` iterates `range(config.num_layers)`. So a per-rank-dense index cache
(6 or 5 slots) automatically produces a per-rank-dense table (6 or 5 layers) with rank-local layer indices.
**The table builder is already correct once the cache is dense.** (This includes the SP×TP TP-sharded
`tp_axis` branch — layer count and TP col-loop are orthogonal.)

### 4.5 Migration producer/consumer — global↔local mapping
With rank-local dense tables, the migration index-cache entries are numbered `0..local_full-1` **per rank**
and refer to *different* global full-layers on each rank. The producer/consumer that stitches ranks
together must map `(rank, local_rank) → global full-layer` (i.e. add the rank's prefix
`full_indexer_rank(config, first_layer_idx)`). Audit `common/prefill/runners/prefill_producer.py`
(`_read_kv_slice`, `:449,491,500,520`) for any assumption that a config's `layer` index is global.

## 5. Validation & tests

1. **Host-only unit test** for the rank-window compaction: for the `[18,20,20,20]` split assert the
   rank-local full counts are `[6,5,5,5]`, each rank starts on a `full` layer, and
   `rank_local_slot(first_layer_idx) == 0` for every rank. Pure Python (no device) — pin the arithmetic
   independent of the model.
2. **Adapter test**: `allocate_kv_cache` with `first_layer_idx ∈ {0,18,38,58}` yields index caches with
   batch dims `{6,5,5,5} * num_users`.
3. **Model-driven per-rank table test** (device): build a GLM-5.2 stack with `first_layer_idx` set (start
   on a full layer), fill both caches, build the merged table, and assert the index config's `num_layers`
   is the rank-local full count and every `full_indexer_rank` slot round-trips. Extend to an SP×TP
   (`tp_axis`) case per [`glm52_kv_cache_tp_sharding.md`](./glm52_kv_cache_tp_sharding.md) §8. Prefer a
   **multi-full-layer** rank (e.g. rank 0's 6) so the abs→local rank mapping is actually exercised — the
   current `test_glm52_kv_cache_table` baseline builds a single layer (rank 0 only) and does not.
4. Regression: whole-model (`first_layer_idx=0`, all layers) must reproduce today's global values; GLM-5.1
   / v3.2 (no `indexer_types`) must be byte-identical.

## 6. Open questions
1. Confirm `params` (the adapter's per-rank config object) carries both `first_layer_idx` and the per-rank
   `num_layers` at `allocate_kv_cache` time (`tt/runners/adapters/mla.py:138` threads `first_layer_idx`
   into MLA construction — verify the same is available to the allocator).
2. Confirm the migration producer/consumer (§4.5) treats index-config layer indices as **rank-local** (or
   update it to map to global). This is the one place a global↔local mismatch would silently read the wrong
   layer.
3. KVPE cache is unaffected (all layers write it), so only config 1 (index) of the merged table becomes
   per-rank dense — confirm no consumer assumes config 0 and config 1 share a layer count.
