# What scales with the traced-prefill tensor sizes

Every place in V4.1's traced prefill whose per-chunk cost or memory follows the size the trace was
captured with, not the chunk's real position. Two sizes are baked into the traces:

* **`cap`**: the compressed-entry rows a position tier reads, per layer type. It is fixed per tier, and
  the last tier's `cap` is `--prefill-max-len / rate`.
* **`capacity`**: the full entry-buffer size, `round_up(max_len, C) / rate`. Persistent buffers are
  allocated at this size once, before any trace exists.

`C` is the chunk size (`--prefill-trace-chunk`, default 1024) and `T = C` query rows per chunk. V4.1 has
compressed-entry rates 1 (layers 20 to 39) and 2 (layers 2 to 19). The index sources are 2, 8 and 14
(rate 2) and 20, 24, 28, 32 and 36 (rate 1). The kv sources are 2, 8, 14 and 20. Layer 20 is the
candidate source. `index_head_dim` is 128, `head_dim` is 512 and `candidate_block_size` is 8.

The sizes below are for the worst case: a 1M-token `--prefill-max-len`, a chunk in the last tier, with
`cap = 1,048,576` for rate 1 and `524,288` for rate 2. They are per device, in bf16 unless noted.

Attention itself does **not** scale. `deepseek_v4_prefill_sdpa` reads the start position on the device
and gathers only each query's `min(index_topk, cap)` selected rows. The projections, MoE, Engram and
sliding window are fixed per chunk.

---

## 1. How `cap` is chosen: the position tiers

`models/experimental/deepseek_v4_flash/tt/model.py:4126`

```python
def _plan_tiers(num_chunks: int, factor: int) -> list[int]:
    tiers = [1]
    if factor >= 2:
        bound = factor
        while bound < num_chunks:
            tiers.append(bound)
            bound *= factor
    if num_chunks > 1:
        tiers.append(num_chunks)
    return tiers
```

`models/experimental/deepseek_v41_flash/tt/prefill/model.py:669`

```python
def _caps_at(self, chunks: int, chunk_size: int) -> dict:
    return {
        key: min(full, _round_up(max(chunks * chunk_size // self._key_rate[key], 1), ALIGNMENT))
        for key, full in self._entry_capacity.items()
    }
```

The defaults are factor 4 (`DEEPSEEK_V4_PREFILL_TIER_FACTOR`) and C = 1024, so the tier boundaries fall
at 1, 4, 16, 64, 256 and then `max_len / C` chunks. Each tier is sized for its last chunk. So every chunk
from 256K tokens onward runs the trace sized for `max_len`. Everything in sections 2 to 4 is then that
large.

---

## 2. Per-chunk compute that scales with `cap`

### 2.1 Causal cut built every step (each stage, each indexed layer type)

`models/experimental/deepseek_v41_flash/tt/prefill/model.py:1047`

```python
for key, (entry, query) in io.cut_ramps.items():
    cap = caps[key]
    if cap < entry.shape[3]:
        entry = keep(ttnn.slice(entry, [0, 0, 0, 0], [1, 1, 1, cap]))
    threshold = ttnn.add(query, start_f)  # [1, 1, T, 1]
    unseen = ttnn.gt(entry, threshold, dtype=ttnn.bfloat16)  # [1, 1, T, cap]
    ttnn.deallocate(threshold)
    index_cuts[key] = keep(ttnn.multiply(unseen, _MASK_NEG))
    ttnn.deallocate(unseen)
```

Two `[T, cap]` elementwise ops (`gt`, `multiply`). The result is 2 GiB at rate 1 and lives for the whole
step.

### 2.2 Indexer scores (each index source: 8 layers)

`models/experimental/deepseek_v4_flash/tt/prefill/attention.py:1258`
(`_index_scores_static`, inherited by V4.1)

```python
pad = step.index_pads[self.layer_type]
keys_rm, owned = self._entry_prefix(bufs.idx_keys, cap)
self._write_rows(keys_rm, pad, 0)
if owned:
    ttnn.deallocate(keys_rm)
kv_len = pad.shape[2]
keys = ttnn.to_layout(pad, ttnn.TILE_LAYOUT)
logits = ttnn.experimental.indexer_score_dsa(
    q,
    keys,
    weights,
    chunk_start_idx=cap,
    kv_len=kv_len,
    program_config=ttnn.IndexerScoreProgramConfig(**_INDEXER_PROGRAM_CONFIG),
    seq_shard_axes=[0] if self.tp_size > 1 else [],
)
scores = ttnn.to_layout(ttnn.slice(logits, [0, 0, 0, 0], [1, 1, num_tokens, cap]), ttnn.TILE_LAYOUT)
cut = step.index_cuts[self.layer_type]
return ttnn.add(scores, cut), cut
```

* `_entry_prefix` copies the first `cap` key rows when `cap` is below the buffer size (256 MiB at rate 1).
* `_write_rows` copies them again, into the zero pad, and `to_layout` tiles the `(cap + T)`-row pad.
* `indexer_score_dsa` costs `T x cap x index_heads x index_head_dim`, with `kv_len` and
  `chunk_start_idx=cap` fixed when the trace is captured.
* `slice`, `to_layout` and `add(scores, cut)` each move a `[T, cap]` tensor (2 GiB at rate 1).

`models/experimental/deepseek_v4_flash/tt/prefill/attention.py:1176`

```python
def _entry_prefix(buf: ttnn.Tensor, cap: int) -> tuple[ttnn.Tensor, bool]:
    if cap >= buf.shape[2]:
        return buf, False
    return ttnn.slice(buf, [0, 0, 0, 0], [1, 1, cap, buf.shape[3]]), True
```

### 2.3 Top-k selection (each index source: 8 layers)

`models/experimental/deepseek_v41_flash/tt/prefill/attention.py:321`

```python
def _select_rows(self, scores: ttnn.Tensor) -> ttnn.Tensor:
    flat = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)
    rows = ttnn.experimental.topk_large_indices(flat, k=min(self.index_topk, scores.shape[3]))
    ttnn.deallocate(flat)
    return rows
```

An untilize of `[T, cap]` (a transient 2 GiB copy at rate 1), then a top-k that reads all `cap` columns
of each of the `T` rows. `topk_large_indices` also takes a `valid_length_tensor` (read on the device),
which could bound this to the entries the chunk has seen; it is not used yet.

### 2.4 Two-level candidates: the candidate source (layer 20)

`models/experimental/deepseek_v41_flash/tt/prefill/attention.py:342`
(`_candidate_blocks`)

```python
t, e = scores.shape[2], scores.shape[3]
size = self.candidate_block_size
maxima = []
for lo in range(0, e, _BLOCK_MAX_COLUMNS):
    hi = min(e, lo + _BLOCK_MAX_COLUMNS)
    segment = scores if hi - lo == e else ttnn.slice(scores, [0, 0, 0, lo], [1, 1, t, hi])
    rows = ttnn.to_layout(segment, ttnn.ROW_MAJOR_LAYOUT)
    ...
    best = None
    for j in range(size):
        part = ttnn.slice(rows, [0, 0, 0, j], [1, 1, t, hi - lo], [1, 1, 1, size])
        tiled = ttnn.to_layout(part, ttnn.TILE_LAYOUT)
        ...
        merged = ttnn.maximum(best, tiled)
        ...
pinned = ttnn.add(best, pins)
k = min(self.candidate_topk_blocks, e // size)
theta = ttnn.min(ttnn.topk(pinned, k=k, dim=-1, largest=True, sorted=True)[0], dim=-1, keepdim=True)
blocks = ttnn.ge(pinned, theta)
```

The column loop runs `cap / 65536` times (16 at rate 1). Each pass slices and untilizes a 65536-column
segment, then does 8 strided slices, 8 tilizes and 7 maxima. In total the `[T, cap]` scores are read
several times. The block top-k (`k = 2048`) runs over `cap / 8` columns.

### 2.5 Two-level candidates: expansion to an entry mask

`models/experimental/deepseek_v41_flash/tt/prefill/attention.py:384`

```python
def _expand_candidates(self, blocks: ttnn.Tensor, cut: ttnn.Tensor) -> ttnn.Tensor:
    entries = ttnn.repeat_interleave(blocks, self.candidate_block_size, dim=3)
    if entries.layout != ttnn.TILE_LAYOUT:
        tiled = ttnn.to_layout(entries, ttnn.TILE_LAYOUT)
        ttnn.deallocate(entries)
        entries = tiled
    additive = ttnn.log(entries)
    ttnn.deallocate(entries)
    mask = ttnn.add(additive, cut)
    ttnn.deallocate(additive)
    return mask
```

This builds a full `[T, cap]` mask: `repeat_interleave`, possibly a tilize, `log` and `add`, about four
2 GiB passes. It runs on the candidate source's stage and again on each consumer stage that receives the
block mask.

`models/experimental/deepseek_v41_flash/tt/prefill/model.py:1136` (each consumer stage):

```python
if blocks_in is not None and stage.candidate_consumer:
    key = self._candidate_attn.layer_type
    expander = self._representative(stage, key)
    shared.candidates = expander._expand_candidates(blocks_in, step.index_cuts[key])
```

### 2.6 Restricting a consumer's scores (index sources 24, 28, 32 and 36)

`models/experimental/deepseek_v41_flash/tt/prefill/attention.py:574`

```python
elif self.uses_candidates and shared.candidates is not None:
    restricted = ttnn.add(scores, shared.candidates)
    ttnn.deallocate(scores)
    scores = restricted
```

One more `[T, cap]` add per consumer.

### 2.7 Writing the chunk's new entries (kv sources 2, 8, 14 and 20, two buffers each)

`models/experimental/deepseek_v4_flash/tt/prefill/attention.py:1148`

```python
def _write_entries(self, buf: ttnn.Tensor, new: ttnn.Tensor, rows: ttnn.Tensor, cap: int) -> None:
    total = buf.shape[2]
    if cap == total:
        written = ttnn.indexed_fill(rows, buf, new, dim=2)
        ttnn.deallocate(new)
        ttnn.copy(written, buf)
        ttnn.deallocate(written)
        return
    head = ttnn.slice(buf, [0, 0, 0, 0], [1, 1, cap, buf.shape[3]])  # a copy: cap < total
    written = ttnn.indexed_fill(rows, head, new, dim=2)
    ttnn.deallocate(head)
    ttnn.deallocate(new)
    self._write_rows(written, buf, 0)
    ttnn.deallocate(written)
```

A chunk writes only `C / rate` rows, but `indexed_fill` rebuilds the whole `cap`-row prefix and it is
copied back. That is about three passes over `cap x width` per buffer: 1 GiB of entries plus 256 MiB of
index keys for layer 20, and half that for each rate-2 source.

`models/experimental/deepseek_v41_flash/tt/prefill/attention.py:555` (each kv source):

```python
rows, cap = step.entry_rows[self.rate], step.caps[self.layer_type]
self._write_entries(bufs.entries, entries, rows, cap)
self._write_entries(bufs.idx_keys, keys, rows, cap)
```

### 2.8 The same write into every mirror on a later stage

`models/experimental/deepseek_v41_flash/tt/prefill/model.py:1120`

```python
for src, mirror in io.mirrors.items():
    rate = self.config.compress_ratios[src]
    key = ratio_layer_type(rate)
    writer = self._representative(stage, key)
    for buf, new in zip(mirror, io.rows_in[src]):
        writer._write_entries(buf, ttnn.clone(new), step.entry_rows[rate], step.caps[key])
```

This is section 2.7 repeated on every stage that reads a group whose kv source lives on an earlier stage.
At five layers per stage, layer 20's group (layers 20 to 39) is mirrored on three further stages, so its
1.25 GiB `cap` prefix is rewritten four times per chunk.

### 2.9 Candidate pins (the candidate source's stage)

`models/experimental/deepseek_v41_flash/tt/prefill/model.py:1058`

```python
if stage.candidate_source and tier in self._candidate_tiers:
    blocks = self._candidate_blocks_at(tier)
    keys = io.pin_keys
    if blocks < keys.shape[3]:
        keys = keep(ttnn.slice(keys, [0, 0, 0, 0], [1, 1, t, blocks]))
    hit = ttnn.eq(keys, start_f)
    candidate_pins[self._candidate_attn.layer_type] = keep(
        ttnn.typecast(ttnn.multiply(hit, -_MASK_NEG), ttnn.bfloat16)
    )
```

Three `[T, cap / 8]` ops, which are small next to sections 2.1 to 2.8.

---

## 3. Stage-to-stage traffic that scales with `cap`

`models/experimental/deepseek_v41_flash/tt/prefill/model.py:1187`

```python
if candidates_on and stage.candidates_out:
    blocks = shared.candidate_blocks if stage.candidate_source else blocks_in
    ttnn.experimental.send_direct_async(blocks, stage.send)
```

The candidate block mask `[T, cap / 8]` (256 MiB at rate 1) is sent down the pipeline to the last
consumer's stage. The receive buffers are one per distinct block count
(`_allocate_io`, `candidates_in`). Everything else sent between stages is sized by `C`: the streams,
`pre_mix`, the chunk's new entry rows, the selected rows and the packet.

---

## 4. Persistent memory that scales with `max_len` (`capacity`)

These are allocated once, at the size of the longest prompt, and are not per-chunk work.

`models/experimental/deepseek_v41_flash/tt/prefill/attention.py:515` (each kv source):

```python
bufs.entries = self._zeros_rm(entry_capacity)
bufs.idx_keys = self._zeros_rm(entry_capacity, self.config.index_head_dim)
```

`models/experimental/deepseek_v41_flash/tt/prefill/model.py:857` (each stage with an index source):

```python
for tier_cap in sorted({caps.get(key, 0) for caps in self._tier_caps}):
    index_pads[(key, tier_cap)] = attn._zeros_rm(_round_up(tier_cap + t, 64), config.index_head_dim)
rate, capacity = attn.rate, self._entry_capacity[key]
entry = (rate * torch.arange(capacity, dtype=torch.float32) + 1).reshape(1, 1, 1, capacity)
query = (rate * ((torch.arange(t) + 1) // rate)).to(torch.float32).reshape(1, 1, t, 1)
cut_ramps[key] = tuple(zeros(x.shape, ttnn.TILE_LAYOUT, ttnn.float32, x) for x in (entry, query))
```

`models/experimental/deepseek_v41_flash/tt/prefill/model.py:878` (each stage reading a remote group):

```python
mirrors = {src: group_rows(src, self._entry_capacity[ratio_layer_type(rate_of[src])]) for src in stage.mirrored}
```

`models/experimental/deepseek_v41_flash/tt/prefill/model.py:883` (the candidate source's stage):

```python
if stage.candidate_source:
    blocks = self._entry_capacity[attn.layer_type] // attn.candidate_block_size
    keys = candidate_pin_keys(t, blocks, attn.rate, attn.candidate_block_size)
    pin_keys = zeros((1, 1, t, blocks), ttnn.TILE_LAYOUT, ttnn.float32, keys.reshape(1, 1, t, blocks))
```

| Buffer | Shape | At max_len = 1M |
| --- | --- | --- |
| entries, per kv source | `[capacity, 512]` | 1 GiB (layer 20), 512 MiB (rate 2) |
| idx_keys, per kv source | `[capacity, 128]` | 256 MiB / 128 MiB |
| mirrors, per remote group per stage | entries + idx_keys | 1.25 GiB for layer 20's group |
| index_pads, per stage, layer type and tier | `[cap + C, 128]` | up to 256 MiB for the last tier |
| pin_keys (fp32) | `[C, capacity / 8]` | 512 MiB |
| cut_ramps entry (fp32) | `[1, capacity]`, tile-padded to 32 rows | 128 MiB |

The traces' own intermediate tensors (the 2 GiB `[T, cap]` scores, cut, untilized copy and candidate
mask) take DRAM only while a chunk runs. The largest tier still sets the peak, and the peak has to fit
next to everything above and decode's caches.

---

## 5. Once per prompt, proportional to the prompt (not `cap`)

`models/experimental/deepseek_v41_flash/tt/prefill/model.py:1460` (`_export_states`):

```python
if attn.is_kv_source:
    entries = n // attn.rate
    state.compressed_kv = rows(bufs.entries, 0, entries)
    state.idx_keys = rows(bufs.idx_keys, 0, entries)
```

These copies are the prompt's own `n / rate` rows, made once after the last chunk for the commit into
decode.

---

## Summary

For one chunk in the last tier at a 1M `--prefill-max-len`, the `[T, cap]` work is roughly:

* **Causal cuts:** 2 elementwise ops per indexed layer type per stage (section 2.1).
* **Each of the 8 index sources:** about 4 slice, layout and add passes, the score itself, an untilize
  and a top-k (sections 2.2 and 2.3).
* **The candidate source:** about 3 more passes for the blocks (section 2.4).
* **Candidate masks:** about 4 passes per expansion, on several stages (section 2.5).
* **Each consumer:** 1 add (section 2.6).
* **Entry writes:** about 3 passes over `cap x (512 + 128)` per kv source and per mirror
  (sections 2.7 and 2.8).

The rate-1 group dominates: it has the widest `cap`, five index sources, the candidate path and three
mirrors.

Section 6 is the plan that removes the tier width: one trace shape for every chunk, working tensors sized
to the chunk, and a position index read on device in place of every `[T, cap]` activation.

---

## 6. Plan: chunk-sized tensors, with a position index

A chunk at any position runs the same programs. Every tensor allocated inside the trace has a sequence
axis of `T`, of `n = T / rate`, or of a fixed select width (`index_topk` = 512, `candidate_topk_blocks` =
2048). Nothing in the trace has an axis of `cap` or `max_len`.

The persistent entry and index-key caches stay sized to `max_len`. They are the stored context. The
position index stops each chunk from copying or scanning them as a dense activation. The indexer's score
FLOPs stay proportional to the entries written so far; that pass is the search. Its activations do not.

### 6.1 The position index

The packet already carries the chunk's token positions, uint32 `[T]`, as `positions` in `_step_inputs`
(`model.py`). Call that tensor `pos`. Entry `w` was produced at token position `w * rate`, and query `i`
may read it iff

```text
w < (pos[i] + 1) / rate
```

Three index tensors, all chunk-sized or scalar, replace the ramps, cuts, pads and pins:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `pos` | uint32 `[T]` | absolute token position of each query. Already in the packet. |
| `entry_rows[rate]` | uint32 `[n]` | cache row each of this chunk's new entries is written to. Already in the packet. Contiguous: `entry_rows[0] = pos[0] / rate`, length `n`. |
| `band_row` | uint32 `[1]` | cache row of column 0 of the history band being scored. Column `c` is cache row `band_row + c`. |

`start` (1 element, already on device) is `pos[0]`. A single valid-length scalar cannot replace `pos`:
inside the chunk, query 0 and query `T - 1` do not see the same entry. `pos` can say so per row.
`topk_large_indices(valid_length_tensor=...)` and `indexer_score_dsa(..., kv_len=)` only shorten a tensor
that was still allocated at `cap`, so they leave sections 2.1 to 2.6 in place.

### 6.2 Writes of the chunk's own rows (sections 2.7 and 2.8)

A chunk writes `n = T / rate` rows, and they are contiguous. `_write_rows` / `slice_write` already writes
`n` rows, but the destination row is a host integer captured into the trace, which is why `_write_entries`
rebuilds the `cap` prefix with `indexed_fill`.

Replace that with a write of the `n` rows whose destination row is read from `entry_rows[0]` on device.
The source is the chunk's new entries, `[1, 1, n, 512]` or `[1, 1, n, 128]`. The persistent buffer is
otherwise untouched. The same write, of the same small tensor, updates each mirror.

Cost per chunk: one pass over `n × (512 + 128)` per kv source and per mirror. At rate 1 that is 1.25 MiB
per copy, against the 1.25 GiB prefix copied today. The bytes sent between stages are already these rows.

### 6.3 Scoring history in bands of one chunk (sections 2.1 to 2.3)

`indexer_score_dsa` returns a score column per key row, and its built-in causality is token causality
(`key t` visible to `query s` iff `t <= chunk_start + s`). V4.1's entry rule is the one in section 6.1,
which is why the code pads the keys to `cap + T`, passes `chunk_start_idx=cap`, slices the scores back to
`cap` and adds `index_cuts`. That pad and that cut are the `[T, cap]` tensors.

Score the cache in bands of width `W = n = T / rate` (1024 at rate 1, 512 at rate 2). One band is one
earlier chunk of that layer type, or the chunk being written. Band `b` reads

```text
idx_keys[band_row : band_row + W],    band_row = b * W
```

and produces scores `[1, 1, T, W]`. At rate 1 that is 2 MiB. The reader takes the persistent cache plus
the `band_row` tensor, so the trace does not capture a host offset.

Causality splits in two, both decided from `pos`:

* A band that ends at or before `pos[0] / rate` is visible to every query. No cut. The op's own mask can
  be satisfied with a chunk-sized pad: keys `[W | T]` zeros, `chunk_start_idx = W`, `kv_len = W + T`,
  then a slice to `W` columns. The pad is `T × 128` (256 KiB at rate 1), and `W` is a capture-time
  constant. This is the same padding the code uses today, at the chunk width.
* The band of the chunk just written overlaps the visibility boundary. Column `c` is hidden from query
  `i` when `band_row + c >= (pos[i] + 1) / rate`. Build that as a `[T, W]` add from `pos` and a length-`W`
  ramp plus `band_row`. The ramp is `arange(W)`, persistent and tiny. `cut_ramps` (length `capacity`) and
  `index_cuts` go away.

Each band keeps its top `index_topk` scores and turns the local column into a cache row with
`band_row + column`. Merge that list into a running `[T, 512]` of scores and uint32 cache rows: concatenate
to `[T, 1024]`, top-k, gather the rows. The running state is about 4 MiB and is the only thing the next
band needs. After the last band, the uint32 rows are `cache_idx` for `deepseek_v4_prefill_sdpa`, which
already gathers and already reads `start` on device.

The number of bands is `(pos[T - 1] + 1) / rate / W`, the number of chunks of this layer type written up
to and including this one. A kernel called once from the stage trace reads that count from `pos` and stops.
The captured program is the same at 4K and at 1M; a chunk at position `P` runs `P / T` band iterations and
no further. Putting the loop on the host instead would recapture, or split the indexer out of the stage
trace, because a trace's trip count is fixed at capture.

`indexer_score_dsa` on a `[T, W]` band, plus a device-side merge, is the whole band body. A later fused
kernel can keep the running top-512 in L1 and never store the `[T, W]` scores; the tensors that cross the
op boundary stay the ones in the table below.

### 6.4 Candidates as block ids (sections 2.4 to 2.6, 2.9, section 3)

Block `b` is cache rows `[8b, 8b + 8)`. During the same band loop the candidate source keeps, per query,
the best 2048 block ids. A band contributes `W / 8` block maxima (128 at rate 1). Merge them into the
running `[T, 2048]` the way section 6.3 merges entry ids.

The pinned block is the one holding the query's newest entry,

```text
((pos[i] + 1) / rate - 1) / candidate_block_size
```

taken from `pos` and forced into the 2048. `pin_keys` (fp32 `[T, capacity / 8]`, 512 MiB at 1M) goes
away, and so does the `[T, cap / 8]` mask, the `repeat_interleave` back to `[T, cap]`, and the consumer's
add of that mask.

The pipeline carries uint32 `[T, 2048]` block ids: 8 MiB, fixed for every chunk. A consumer scores only
the rows in those blocks, 2048 × 8 = 16384 rows, gathered by a uint32 row list. It walks them in bands of
`W` too, so its score tensor is also `[T, W]`. The band's position index is that row list (length `W`)
rather than a contiguous `band_row`, because the selected blocks are scattered. Sixteen such bands at rate
1, for every chunk past the two-level threshold, independent of how long the prompt is. Entries outside
the list are not scored, so there is no full-width mask to add.

### 6.5 One trace, and what is deleted

V4.1 drops `_plan_tiers` / `_caps_at` for the indexed path. Compile two programs per rate, used at every
position:

* **Band program.** Inputs: persistent `idx_keys`, `pos`, `band_row` (or a uint32 `[W]` row list for a
  consumer). Outputs folded into the running `[T, 512]` and, on the candidate source, `[T, 2048]`.
* **Chunk program.** The stage body as it is now, with `_write_entries` replaced by the section 6.2
  write, the indexer replaced by one call that runs the bands up to `pos`, and `cache_idx` fed straight
  into `deepseek_v4_prefill_sdpa`.

Attention, the window, the projections, MoE and Engram are already chunk-sized.

Deleted with the tier width:

| Gone | Replaced by |
| --- | --- |
| position tiers and per-tier traces | one band program and one chunk program per rate |
| `cut_ramps`, `index_cuts` (section 2.1) | `pos`, and a `[T, W]` cut on the current band only |
| `index_pads` and the `cap`-row key copy (section 2.2) | a band read of `W` rows, pad `W + T` |
| `[T, cap]` scores, untilize, top-k (sections 2.2, 2.3) | `[T, W]` scores and a running `[T, 512]` |
| block scan, expansion, consumer add (sections 2.4 to 2.6) | running `[T, 2048]` block ids |
| `pin_keys` (section 2.9) | the pinned block id from `pos` |
| `indexed_fill` of the `cap` prefix (sections 2.7, 2.8) | a write of `n` rows at `entry_rows[0]` |
| candidate mask send, `[T, cap / 8]` (section 3) | send of uint32 `[T, 2048]` |

`index_pads`, `pin_keys` and the entry `cut_ramps` come out of the persistent table in section 4 (up to
256 MiB, 512 MiB and 128 MiB at a 1M context). The running select state is a few MiB.

### 6.6 What a chunk costs after this

For a chunk whose last token is at position `P` (`E = P / rate` entries):

* **Writes:** `O(T / rate)` rows per buffer copy.
* **Each index source:** `E / W` bands, each scoring `[T, W, index_heads, index_head_dim]`, merged into
  `[T, 512]`. Total FLOPs are the same order as today's indexer. The activation is 2 MiB a band at rate
  1, against a 2 GiB `[T, cap]` score, cut and untilize.
* **Candidate source:** the same bands, keeping `[T, 2048]` block ids.
* **Candidate consumers:** 16384 gathered rows, in bands of `W`, at every position past 16384 entries.
* **Attention:** 512 gathered rows plus the window, as today.

A chunk at 300K and a chunk at 1M then differ by how many bands the kernel runs, which is how many chunks
have actually been written. They allocate the same tensors and run the same programs. The tier boundary
at 256K no longer changes the shape of anything.
