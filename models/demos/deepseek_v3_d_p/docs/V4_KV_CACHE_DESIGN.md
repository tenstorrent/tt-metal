# KV cache design: from GLM-5.2 to DeepSeek-V4

A deep dive on the one subsystem V4 cannot inherit. Part 1 walks the GLM-5.2 cache end to end as
the worked reference — it is the most complex cache we serve today and it already solves two of
V4's problems. Part 2 states exactly what V4 has to cache, where our current HCA implementation
diverges from the reference, and proposes a concrete layout.

Companions: [`PREFILL_BRINGUP_TUTORIAL.md`](./PREFILL_BRINGUP_TUTORIAL.md) for the generic
contract, [`DEEPSEEK_V4_BRINGUP.md`](./DEEPSEEK_V4_BRINGUP.md) for the whole-model gap list.

This document incorporates the prefill/decode Slack thread on compressor state migration
(Naif Tarafdar, Ilia Taraban, Kartik Paigwar, Brian Liu, Aleks Knezevic). Points that are still
open with the decode team are collected in [§10](#10-open-questions-for-the-decode-team) rather
than being silently resolved here.

---

## Contents

1. [What "the cache" has to satisfy](#1-what-the-cache-has-to-satisfy)
2. [GLM-5.2: what is cached, semantically](#2-glm-52-what-is-cached-semantically)
3. [GLM-5.2: the physical substrate](#3-glm-52-the-physical-substrate)
4. [GLM-5.2: allocation, write, read](#4-glm-52-allocation-write-read)
5. [GLM-5.2: migration and read-back](#5-glm-52-migration-and-read-back)
6. [GLM-5.2: sizing, and the five invariants](#6-glm-52-sizing-and-the-five-invariants)
7. [V4: what actually has to be cached](#7-v4-what-actually-has-to-be-cached)
8. [Where our HCA implementation diverges](#8-where-our-hca-implementation-diverges)
9. [Proposals](#9-proposals)
10. [Open questions for the decode team](#10-open-questions-for-the-decode-team)
11. [Sizing comparison](#11-sizing-comparison)
12. [Suggested sequencing](#12-suggested-sequencing)

---

## 1. What "the cache" has to satisfy

Before comparing models, fix the requirements. A prefill KV cache in this stack is not just
"somewhere to put K and V". It has to satisfy five properties simultaneously, and every design
decision below is a consequence of one of them.

| # | Invariant | Why |
|---|---|---|
| **I1** | **Engine-owned, opaque.** Allocated by `adapter.allocate_kv_cache`, returned as a `KvCaches` subclass, passed into every runtime call, freed with the mesh. | The runner must not know the layout. See `common/prefill/adapter.py`. |
| **I2** | **Multi-slot.** `num_users` independent, interleavable slots. Chunks for different slots arrive interleaved. | The producer drives N concurrent users. |
| **I3** | **Addressable from the host without a device.** Every byte reachable as `(layer, position, slot[, config])` → `(fabric node, NOC address, size)`, in 32-token granules. | The migration worker and the producer's PCC read-back both use `read_dram_umd` with no device lock. |
| **I4** | **Resumable.** Everything decode needs to continue generating at token N must be either in the cache or derivable from N. | Disaggregation is the entire point. |
| **I5** | **Fixed shape across chunks.** Shapes and program offsets must not vary with chunk index or real length. | Otherwise every chunk recompiles, and trace capture is impossible. |

GLM-5.2 satisfies all five. V4, as currently implemented, satisfies none of them — which is the
honest summary of the work ahead.

---

## 2. GLM-5.2: what is cached, semantically

GLM-5.2 is MLA + DSA (lightning-indexer sparse attention) with cross-layer indexer reuse. Its
dimensions:

```
EMB_SIZE = 6144        NUM_LAYERS = 78        NUM_DENSE_LAYERS = 3
NUM_ATTENTION_HEADS = 64                      Q_LORA_RANK = 2048
KV_LORA_RANK = 512     QK_NOPE_HEAD_DIM = 192  QK_ROPE_HEAD_DIM = 64  V_HEAD_DIM = 256
INDEX_TOPK = 2048      INDEX_HEAD_DIM = 128    INDEX_N_HEADS = 32
INDEX_TOPK_FREQ = 4    INDEX_SKIP_TOPK_OFFSET = 3
```

Two things get cached, per token, per layer:

**1. The MLA latent row (`kvpe`).** MLA compresses K and V into one low-rank latent plus a shared
rope stem. One row per token: `[latent(512) || rope(64)]`, width **576**. There is no separate K
and V — that is the whole point of MLA, and it is why the cache is ~1/8 the size of an equivalent
GQA cache. Every one of the 78 layers writes it.

**2. The lightning-indexer key (`index`).** The indexer scores every cached position and keeps the
top 2048. Its keys are a separate, narrower projection: **128** wide, `bfloat8_b`.

The 5.2 twist is that **only some layers own an indexer**:

```python
@classmethod
def indexer_types(cls, num_layers=None):
    """full if (max(i - offset + 1, 0) % freq) == 0 else shared"""
    return ["full" if (max(i - off + 1, 0) % freq) == 0 else "shared" for i in range(n)]
```

With `freq=4, offset=3` that gives `full` at `{0, 1, 2, 6, 10, …, 74}` — **21 full layers out of
78**. A `shared` layer runs sparse attention but reuses the most recent full layer's top-k
selection, and the checkpoint carries indexer weights only for full layers, so reuse is required,
not an optimisation.

Consequence: **the index cache is sized to 21 layers, not 78.** This is the single most
transferable idea in the whole document, and V4 needs it four times over.

---

## 3. GLM-5.2: the physical substrate

Both caches sit on the same substrate, allocated by `init_kvpe_cache()`
(`utils/kv_cache_utils.py`):

```python
kvpe_cache = ttnn.allocate_tensor_on_device(
    ttnn.Shape([num_users * num_layers, 1, seq_len_local, kvpe_cache_head_dim]),
    dtype, layout, mesh_device, kv_mem_config,
)
DRAMZeroFill.op(kvpe_cache)
kvpe_cache.update_tensor_topology(ttnn.TensorTopology(dist_shape, placements, coords))
```

```python
kv_nd_shard_spec = ttnn.NdShardSpec(
    shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, kvpe_cache_head_dim],  # 32 tokens
    grid=grid,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
)
```

Four properties, each satisfying one invariant:

**Dim 0 is a user-major fold** — `batch_idx = slot_id * num_layers + layer_idx`. All users and all
of this rank's layers in one flat tensor. Satisfies **I2**, and gives migration a uniform layer
stride.

**Dim 2 is SP-sharded** — `seq_len_local = max_seq_len // sp`. The sequence is split across the SP
rows so no chip holds the whole context.

**32 tokens per ND DRAM shard, round-robin across banks** — `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK =
32`. This is the granule of one migration address-table entry. Satisfies **I3**.

**The sequence is block-cyclic, not contiguous.** Global position `p` does *not* live at local row
`p // sp`. The mapping is the inverse of what `update_padded_kv_cache` writes:

```python
def blockcyclic_positions(sp, chunk_size_global, seq_len_cache):
    """Shard row r -> chip c = r // seq_len_local, local row lr = r % seq_len_local, and that row
    holds global position (lr // chunk_local) * chunk_size_global + c * chunk_local + (lr % chunk_local)"""
    seq_len_local = seq_len_cache // sp
    chunk_local = chunk_size_global // sp
    c  = torch.arange(sp).repeat_interleave(seq_len_local)
    lr = torch.arange(seq_len_local).repeat(sp)
    slab, off = lr // chunk_local, lr % chunk_local
    return slab * chunk_size_global + c * chunk_local + off
```

The period is `PREFILL_CHUNK_OUTPUT_TOKENS = 5120`, the prefill chunk size. Each chunk's tokens are
split SP ways and each chip appends its share contiguously; the next chunk appends after it. So the
cache is a stack of per-chunk slabs, each internally SP-strided.

This is the layout every downstream consumer has to reproduce, and getting it wrong is a
particularly nasty bug class: a contiguous (wrong) address table still works for a blanket copy of
the *whole* cache, and only fails when you migrate a prefix or a prompt shorter than
`max_seq_len`.

---

## 4. GLM-5.2: allocation, write, read

### Allocation

`GLM52Adapter.allocate_kv_cache` — the reference implementation of a two-cache model:

```python
def allocate_kv_cache(self, *, mesh_device, hf_config, params) -> MlaKvCaches:
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,          # sparse_sdpa needs uncompressed row-major
        hf_config=hf_config, mesh_device=mesh_device,
        seq_len=params.max_seq_len, mesh_shape=list(params.mesh_shape), sp_axis=params.sp_axis,
        num_kvpe_cache_layers=params.num_layers,        # ALL of this rank's layers
        num_users=params.num_users,
    )
    first_full        = full_indexer_rank(hf_config, params.first_layer_idx)
    num_index_layers  = full_indexer_rank(hf_config, params.first_layer_idx + params.num_layers) - first_full
    index_cache = init_kvpe_cache(
        kvpe_cache_head_dim=hf_config.index_head_dim,   # 128
        mesh_device=mesh_device, seq_len=params.max_seq_len,
        mesh_shape=list(params.mesh_shape), sp_axis=params.sp_axis,
        num_kvpe_cache_layers=num_index_layers,         # FULL layers only, compacted
        num_users=params.num_users,
        dtype=ttnn.bfloat8_b,
    )
    return MlaKvCaches(kvpe=kvpe_cache, index=index_cache)
```

Note the asymmetry: `kvpe` is numbered in the model's **global** layer space, `index` in
**compacted full-indexer** space via `full_indexer_rank` — a prefix rank over `indexer_types`:

```python
def full_indexer_rank(config, layer_idx: int) -> int:
    """Prefix rank over config.indexer_types: count how many entries equal "full" before
    position layer_idx (exclusive), renumbering the matching positions into dense 0-based ranks."""
    types = getattr(config, "indexer_types", None)
    if not types:
        return layer_idx
    return sum(1 for t in types[:layer_idx] if t == "full")
```

Two knock-on effects, both of which V4 will inherit:

* Each stage's cache is sized to *that stage's* full-layer count, so its layers start at slot 0
  and the migration table needs no extra stride.
* A pipeline rank must **start on a `full` layer**, or its first layer has no prior top-k to
  reuse. Hence the adapter override:

```python
def layer_split_boundaries(self, num_layers):
    types = getattr(self.load_hf_config(), "indexer_types", None)
    return None if not types else {i for i in range(num_layers) if types[i] == "full"}
```

### Physical encoding

GLM-5.2 hardcodes `BF16_RM`; GLM-5.1 defaults to `SCALED_FP8`. The format lives next to the tensor:

```python
@dataclass(frozen=True)
class MlaKvCache:
    format: MlaKvCacheFormat
    storage: ttnn.Tensor
    geometry: MlaKvCacheGeometry
```

| Format | dtype / layout | Row |
|---|---|---|
| `BFP8_TILE` | `bfloat8_b` / TILE | 576 logical |
| `BF16_RM` | `bfloat16` / ROW_MAJOR | 576 logical |
| `SCALED_FP8` | `fp8_e4m3` / ROW_MAJOR | 512 E4M3 + 4 FP32 scales + 64 BF16 rope = **592 bytes** |

`MlaKvCache.pack(latent, rope)` produces the physical row and `unpack_host()` recovers logical
BF16. Pairing the encoding with the tensor is what lets one `sparse_sdpa` call read three different
formats — and it is the pattern V4 should copy for its compressed entries.

### Write

```python
ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
    cache.storage, values,
    slot_idx=cache_user_id, layer_idx=cache_layer_idx, num_layers=self.layer_num,
    kv_actual_global=kv_actual_isl,        # == actual_start
    cluster_axis=self.sp_axis,
)
```

Or, under trace, the overload where `slot_idx` and `kv_actual_global` are **device tensors**
(`metadata[0]`, `metadata[1]`) rather than Python ints — that is what satisfies **I5** and makes
one compiled program serve every chunk.

The op derives each chip's local write offset on-device from `kv_actual_global`, which is what
makes the block-cyclic layout work with no host index math.

The indexer writes its own cache the same way, into its compacted slot:

```python
ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
    index_kbuf, k,
    slot_idx=cache_user_id, layer_idx=cache_layer_idx,
    num_layers=self._index_cache_layers, kv_actual_global=start_pos, cluster_axis=self.sp_axis,
)
```

`TtIndexer.write_k` runs on **every** chunk even for layers whose top-k is reused downstream —
otherwise later chunks would score against missing keys for the early prefix.

After the write, the block zeroes the pad window (`zero_padded_kv_cache`) so a migrated prefix
contains no garbage, then fires the layer ack.

### Read

Sparse chunked attention gathers the prefix and runs `sparse_sdpa` against the selected indices.
The important structural point for V4: **the cache is SP-sharded, and the read path gathers what it
needs** (`_gather_kvpe_prefix`, `tt_ccl.mla_sparse_kv_gather_buffers`). Sharded storage plus a
gather at read is an established pattern here, not a new risk.

---

## 5. GLM-5.2: migration and read-back

### Stages

```python
def kv_migration_stages(self, kv_caches, first_layer_idx=None, num_my_layers=None):
    stages = [KvCacheStage(self.kv_migration_base_address(kv_caches), first_layer_idx, num_my_layers)]
    index_cache = kv_caches.index
    if index_cache is not None:
        first_full = full_indexer_rank(self.hf_config, first_layer_idx)
        count_full = full_indexer_rank(self.hf_config, first_layer_idx + num_my_layers) - first_full
        slots_per_user = index_cache.shape[0] // self.config.num_users
        if slots_per_user != count_full:
            raise RuntimeError(...)
        stages.append(KvCacheStage(int(index_cache.buffer_address()), first_full, count_full))
    return stages
```

```python
class KvCacheStage(NamedTuple):
    base_addr: int
    first_layer: int
    count: int
```

One stage per migratable cache per rank. The engine all-gathers a layout for each, across all
ranks, and hands the merged result to `build_kv_chunk_table` on rank 0. That runtime check on
`slots_per_user` is worth copying — it catches the exact class of bug where the allocator and the
table disagree about how many layers a stage owns.

### The merged table

GLM publishes **one** table with two configs: config 0 = KVPE, config 1 = index. Both share one
device-group side table. Because config 1 is compacted, the table also needs a dense-row → global-layer map so the worker can address it by model layer:

```python
index_layer_ids = [layer for layer in range(total_layers)
                   if not indexer_layer_is_reused(self.hf_config, layer)]
```

Chunk size in bytes is derived per config from the dtype:

```python
_TILE_DIM = 32
_BFP8_TILE_BYTES = 1088   # one 32x32 bfp8 tile: 1024 data + 64 exponent bytes
# bfp8 / TILE : (head_dim // 32) * 1088
# bf16 or fp8 / ROW_MAJOR : 32 * buffer_aligned_page_size()
```

Each entry is `noc_addr = (bank_id << 32) | (dram_base + bank_offset)` plus `size_bytes` and a
`device_group_index`, walked in block-cyclic order with bank round-robin.

### Read-back for PCC

The producer reads the cache device-lessly via `read_dram_umd`, decodes by physical byte size, and
PCCs against the golden trace — nope columns and rope columns separately, with the rope golden
re-interleaved from HF to Meta layout. For GLM it validates both configs and returns
`{"kvpe": min, "index": min}`.

This path is **not** adapter-dispatched: `_read_slot_kv_and_check_pcc` in
`common/prefill/runners/prefill_producer.py` branches on `ADAPTER.name`. A new cache layout needs a
new branch there.

---

## 6. GLM-5.2: sizing, and the five invariants

Per user, per layer, per chip, at 56320-token context on SP=8 (`seq_len_local = 7040`):

| Cache | Layers | Row | Bytes/row | Per chip |
|---|---|---|---|---|
| `kvpe` (BF16_RM) | all 78 | 576 | 1152 | **8.10 MB** |
| `index` (bfp8 TILE) | 21 full only | 128 | 136 | **0.96 MB** |

Aggregate across the 8 SP chips equals the logical size — nothing is replicated. That is what
sharding buys, and it is the property V4's current implementation gives up.

Scorecard:

| | GLM-5.2 |
|---|---|
| **I1** engine-owned, opaque | ✅ `MlaKvCaches(kvpe, index)` from `allocate_kv_cache` |
| **I2** multi-slot | ✅ user-major fold, `slot_id * num_layers + layer_idx` |
| **I3** host-addressable | ✅ 32-token granules, merged 2-config table |
| **I4** resumable | ✅ every token's latent and indexer key is in the cache |
| **I5** fixed shape | ✅ metadata-tensor write overload |

---

## 7. V4: what actually has to be cached

Now the part that is genuinely different. V4 replaces "cache every token's latent" with "cache a
sliding window plus compressed summaries", and compression is **stateful**.

### 7.1 The reference cache classes

From `reference/deepseek_v4/modeling_deepseek_v4.py`. HCA first:

```python
class DeepseekV4HCACache(DynamicSlidingWindowLayer):
    layer_type = "heavily_compressed_attention"

    def __init__(self, config):
        super().__init__(config)
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]   # 128
        self.buffer_kv:     dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_gate:   dict[str, torch.Tensor | None] = {"compressor": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.entry_count:   dict[str, int]                 = {"compressor": 0}
```

with the docstring spelling out the contract:

> * `compressed_kv[name]` — the running list of compressed KV entries emitted so far (one every
>   `compress_rate` source tokens; the long-range KVs the attention concatenates onto its
>   sliding-window keys / values).
> * `buffer_kv[name]` / `buffer_gate[name]` — source tokens that arrived between two full windows;
>   once the buffer hits `compress_rate` tokens the compressor closes a window, emits one entry, and
>   drains the buffer.
> * `entry_count[name]` — number of compressed entries emitted so far, so
>   `entry_count[name] * compress_rate` is the absolute position of the *next* window's first source
>   token. Tracked separately from `position_ids` so prefill -> decode -> prefill stays consistent.

The buffer mechanics:

```python
def store_compression_weights(self, name, kv, gate):
    first_window_position = self.entry_count[name] * self.compress_rate
    buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
    if buffered_kv is not None and buffered_kv.shape[1]:
        kv   = torch.cat([buffered_kv, kv], dim=1)
        gate = torch.cat([buffered_gate, gate], dim=1)
    # only return the longest prefix that's a multiple of compress_rate; the rest stays in the buffer
    usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
    self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]
    return kv[:, :usable], gate[:, :usable], first_window_position
```

That last comment is the crux of the Slack thread. The buffer holds **post-projection** `kv` and
`gate`, both at `head_dim` width — not raw hidden states.

The sliding branch keeps 127 rows and returns carry+new for attention:

```python
def update(self, key_states, value_states, *args, **kwargs):
    """V4 uses shared-KV MQA, so `keys` and `values` point to the same storage on every layer."""
    self.cumulative_length += key_states.shape[-2]
    full = torch.cat([self.keys, key_states], dim=-2)
    self.keys = full[:, :, -self.sliding_window + 1 :, :]
    self.values = self.keys
    return full, full
```

CSA extends it with a second named entry and per-name overlap state:

```python
class DeepseekV4CSACache(DeepseekV4HCACache):
    layer_type = "compressed_sparse_attention"

    def __init__(self, config):
        super().__init__(config)
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]   # 4
        self.buffer_kv["indexer"] = None
        self.buffer_gate["indexer"] = None
        self.compressed_kv["indexer"] = None
        self.entry_count["indexer"] = 0
        self.overlap_kv:   dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.overlap_gate: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
```

The overlap exists because CSA pools two interleaved series:

> the CSA `kv_proj` / `gate_proj` produce `2 * head_dim` features per source token — two independent
> compressed series Ca and Cb stored in one tensor. […] Pooled entry `w` is the softmax-gated convex
> combination of window `w-1`'s Ca slice with window `w`'s Cb slice — effective width
> `2 * compress_rate_csa`, stride `compress_rate_csa`. […] the only thing we need to carry across a
> forward boundary is `chunk[:, -1, :, :head_dim]` (Ca) of the last full window — Cb is never read
> again.

```python
def update_overlap_state(self, name, chunk_kv, chunk_gate, head_dim):
    prior_kv, prior_gate = self.overlap_kv[name], self.overlap_gate[name]
    self.overlap_kv[name]   = chunk_kv[:, -1, :, :head_dim].clone()
    self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
    return prior_kv, prior_gate
```

### 7.2 The complete state inventory

Per layer, per user. This reconciles the reference with the Slack thread — Ilia's "one compressor
state for HCA, two for CSA" is the `"compressor"` / `"indexer"` keying above, and Kartik's
"+ indexer accumulator and indexer cache" is the `"indexer"` entry's `compressed_kv` and buffer.

| # | State | HCA | CSA | sliding | Shape | Migratable as |
|---|---|---|---|---|---|---|
| 1 | Sliding-window K=V | ✅ | ✅ | ✅ | `[sliding_window, head_dim]` = 128 × 512 | fixed blob, or last 128 rows of a token-indexed cache |
| 2 | Compressed entries (compressor) | ✅ | ✅ | — | `[N // m, head_dim]`, m = 128 / 4 | **entry-indexed array** |
| 3 | Compressor residual buffer (kv + gate) | ✅ | ✅ | — | `2 × [N % m, head_dim]`, ≤ `2 × [m, head_dim]` | fixed blob |
| 4 | Compressor overlap (Ca slice) | — | ✅ | — | `2 × [head_dim]` | fixed blob |
| 5 | Compressed entries (indexer) | — | ✅ | — | `[N // m, index_head_dim]` = `[N//4, 128]` | **entry-indexed array** |
| 6 | Indexer residual buffer (kv + gate) | — | ✅ | — | `2 × [N % 4, 128]` | fixed blob |
| 7 | Indexer overlap (Ca slice) | — | ✅ | — | `2 × [128]` | fixed blob |
| 8 | `entry_count`, `cumulative_length` | ✅ | ✅ | ✅ | scalars | **derivable — do not migrate** |

**On #8 — this simplifies the migration contract considerably.** Every counter is a pure function of
the prompt length N and the layer's `m`:

```
entry_count   = N // m
residual_rows = N %  m
sliding rows  = min(N, sliding_window)
```

So the handoff payload is the four arrays plus the small blobs, and decode reconstructs all indices
from N. Nothing stateful has to cross the wire beyond bytes that are already in DRAM.

### 7.3 Answering the thread's central question

> *Naif:* Is each entry for compressor similar to that of kv cache? Is it indexable with position
> id, similar to kv cache? I ask because that was what required re-thinking for the recurrent state
> in K3. […] I'm guessing it might not be because m tokens are collapsed into 1.

**Compressed entries (#2, #5) are indexable — they are strictly easier than K3's recurrent state.**
Entry `e` covers source tokens `[e·m, (e+1)·m)`, so the mapping is dense, deterministic and
invertible: `e = pos // m`. That is a linear reindexing of the position axis, not a loss of
addressability. A K3-style recurrent state is genuinely non-indexable because it is a *fold* over
all history with no per-position decomposition; a compressed entry array is not.

The practical consequence is only that the table's position axis for those configs is measured in
**entries**, not tokens. Two ways to express it, both compatible with
`KvChunkAddressTableConfig`:

* **(a) Entry-space axis** — `chunk_n_tokens` counts 32 *entries*, and the worker converts
  `pos → pos // m` before lookup. Cleanest, but the worker must know `m` per layer.
* **(b) Token-space axis with stride** — `chunk_n_tokens = 32 * m` in token space, so one address
  entry covers `32 * m` source tokens. The table stays token-indexed and the worker needs no
  per-layer knowledge. Costs nothing for HCA (one entry per 4096 tokens) and is natural for CSA
  (one per 128 tokens).

**(b) is the recommendation** — it keeps the worker's addressing uniform with GLM's.

**The residual buffer (#3, #6) is the genuinely non-indexable part**, and it is also tiny: at most
`m` rows per layer per user. Migrate it as a single fixed-size blob per layer, always full size,
with the live prefix determined by `N % m`. Because the size is constant, it needs no addressing
scheme at all — one more config in the table with `num_slots` entries of fixed size.

---

## 8. Where our HCA implementation diverges

This is the most important finding in this document, and it is directly what the thread is asking
about.

**`TtHCACompressor` does not implement `store_compression_weights` at all. There is no residual
buffer, and the tail tokens of a ragged final chunk are silently dropped.**

```python
n_new = real_len // compress_rate
...
assert state.kv_actual % compress_rate == 0, (
    f"cannot append after a chunk with {state.kv_actual % compress_rate} leftover tokens; only "
    f"the final chunk may be ragged, non-final chunks must be a multiple of {compress_rate}"
)
```

with the comment above it:

> A non-final chunk ending mid-window would strand those leftover tokens — they never join a
> compression window — and push the next chunk off the 128-token grid the compressor assumes.

For prefill-only PCC this is **correct**: the reference also emits no entry for a partial window, so
outputs match. For handoff to decode it is **not sufficient**: decode needs those `N % m` projected
rows to close the window at token `N`. Today they do not exist anywhere on device after the chunk
returns.

Three related points that are easy to get wrong:

**`TtHCAState.tail` is not the compressor residual.** It is the partially-filled last 32-row *tile*
of the compressed **entry** cache, needed because `fill_cache_for_user_` requires a tile-aligned
write offset:

```python
self.tail = tail  # [B, 1, TILE_SIZE, head_dim]
# The cache's last tile is usually only partly filled. Its entries sit here, right-aligned, so the
# next write can place them and the new entries with a single shift.
```

Two different partial-state mechanisms at two different granularities — entries vs source tokens.
Both are needed; neither substitutes for the other.

**`sliding_carry` already exists and is correct-ish.** `TtHCA` keeps
`[B, 1, sliding_window, head_dim]` and slices it from a device-tensor start index so the offset
stays out of the program:

```python
start, end = self._carry_index[self._carry_key(real_len)]
next_carry = ttnn.slice(sliding_kv, start, end, slice_dim=2, num_devices=seq_len // self.sliding_window)
```

Note it keeps **128** rows where the reference keeps **127** (`full[:, :, -sliding_window + 1:, :]`).
The attended set is identical because the mask enforces `i - j < sw`, but a decode handoff must
agree on how many rows are transferred and which absolute positions they correspond to.

**The compressed entry cache is SP-replicated, not SP-sharded.** The compressor all-gathers on the
SP axis so every chip holds every entry:

```python
if self.sp_factor > 1:
    compressed_kv = ttnn.experimental.all_gather_async(compressed_kv, dim=2, ..., cluster_axis=self.sp_axis)
```

For HCA at 56K that is 440 entries — replication is free. For CSA it is 14 080 entries, and
replication costs **8× the DRAM** and gives the migration worker eight identical copies to choose
between. See [§9.3](#93-shard-the-csa-compressed-cache).

Finally, `entry_count` and `kv_actual` are **Python ints** advanced in `forward`, and
`_write_tail_tile` selects a one-hot matrix by `divmod(state.entry_count, TILE_SIZE)`. That design
is deliberate and effective — it keeps offsets out of the compiled program — but host-resident
state blocks trace (**I5**) and makes multi-user interleaving (**I2**) awkward, because state is
per-layer-object rather than per-slot.

---

## 9. Proposals

### 9.1 `V4KvCaches`: five tensors with per-layer-class compaction

Generalise GLM-5.2's `full_indexer_rank` from "indexer layers" to "layers of class C", then give
each layer class its own tensor sized to that class's count on this stage.

```python
def layer_class_rank(config, layer_idx: int, cls: str) -> int:
    """Prefix rank over config.layer_types: how many layers of class `cls` occur before layer_idx."""
    return sum(1 for t in config.layer_types[:layer_idx] if t == cls)
```

```python
@dataclass
class V4KvCaches(KvCaches):
    sliding:        ttnn.Tensor   # ALL layers        | sliding_window rows      | head_dim
    hca_entries:    ttnn.Tensor   # HCA layers only   | max_seq // 128 rows      | head_dim
    csa_entries:    ttnn.Tensor   # CSA layers only   | max_seq // 4   rows      | head_dim
    csa_index:      ttnn.Tensor   # CSA layers only   | max_seq // 4   rows      | index_head_dim
    residual:       ttnn.Tensor   # HCA+CSA layers    | fixed m rows (kv||gate)  | see 9.2
```

Each tensor keeps the existing substrate — `[num_users * num_layers_of_class, 1, rows_local, width]`,
ND-sharded in 32-row granules, user-major fold — so **I1, I2, I3 and I5 come for free** and every
existing helper applies unchanged.

Consequences to accept up front:

* `layer_split_boundaries` will need to constrain rank starts so each class's compacted numbering
  stays contiguous per stage, exactly as GLM-5.2 does today.
* `kv_migration_stages` returns **five** `KvCacheStage`s instead of two, each in its own layer-index
  space, with a dense-row → global-layer map per class (the `index_layer_ids` pattern).
* Pure `sliding_attention` layers own a slot only in `sliding`.

**Why this over the alternatives:** it is the only option that preserves the uniform layer stride
the migration table depends on, and the only one with a working precedent in the tree.

### 9.2 Where to put the residual buffer

Three options, in increasing order of intrusiveness:

**(a) A dedicated `residual` tensor, always full size.** `[num_users * num_compressor_layers, 1, m,
2 * width]` holding `kv` and `gate` side by side, with the live prefix implied by `N % m`. Fixed
shape (**I5** ✓), trivially addressable as one config with fixed-size chunks (**I3** ✓), and
interoperates with decode's ring-buffer reading (see below). HCA's `m = 128` and CSA's `m = 4`
differ, so this is really two tensors — or one sized to `max(m)` with CSA wasting 124 rows per
layer, which at 512 wide is ~127 KB per layer per user and probably not worth the complexity.

**(b) Reserve the tail of the entry cache.** Append `m` rows past the entry capacity and write the
residual there. Fewer tensors, but it conflates two index spaces in one config and makes the table's
position axis non-uniform. Not recommended.

**(c) Recompute on demand at migration time.** Do not store it; re-project the last `N % m` hidden
states when the handoff happens. Removes the state entirely, but requires retaining the hidden
states (which we do not) or re-running the layer. Not viable.

**Recommendation: (a).** Cost is small — see [§11](#11-sizing-comparison) — and it is the only
option that keeps the table uniform.

**Interop with decode's ring buffer.** Ilia noted decode keeps `m` slots and fills `pos % m`:

> right now stored only m values and compressor fills pos%m value

Prefill's reference instead keeps a compacted `[0, N % m)` prefix. These agree **iff** prefill
writes its residual at the head of a full `m`-row buffer, because every compression window starts at
a multiple of `m`, so source token `p` in the open window has `p % m == p - (N // m) * m`, which is
exactly its offset in the compacted prefix. So: **prefill allocates `m` rows, writes the residual at
rows `[0, N % m)`, leaves the tail stale, and decode reads `pos % m` unchanged.** No conversion at
the boundary. This is worth confirming explicitly with the decode team as it is the cheapest
possible interop.

### 9.3 Shard the CSA compressed cache

`TtHCA` currently all-gathers compressed entries so every SP chip holds every entry. Keep that for
HCA — 440 entries at 56K is nothing — but **SP-shard `csa_entries` and `csa_index` block-cyclically,
like GLM's `kvpe`**, and gather at read.

Rationale:

* At 56K, CSA holds 14 080 entries per layer per user. Replicated across SP=8 that is 8× the DRAM
  for no information gain.
* Replication also means the migration table has eight equivalent sources per chunk, so the worker
  has to pick — extra logic for no benefit.
* The read path already does this: sparse MLA gathers its prefix through
  `tt_ccl.mla_sparse_kv_gather_buffers` / `_gather_kvpe_prefix`. Sharded storage plus gather-at-read
  is the established pattern, not a new risk.
* Entry counts are `max_seq // m`, which for CSA at 56K and SP=8 gives 1760 rows per chip — a clean
  multiple of the 32-row granule.

The block-cyclic period in entry space is `chunk_size // m` (5120 / 4 = 1280 entries per chunk),
which keeps `update_padded_kv_cache`'s on-device offset derivation applicable with `m`-scaled
arguments.

### 9.4 Move the counters onto the device

`entry_count` and `kv_actual` should follow the same path `TtPrefillRuntime` already uses for
`(slot_id, actual_start, actual_end)`: persistent 1-element uint32 device tensors updated in place,
with the write op taking them as tensors rather than ints. That is what unlocks trace (**I5**) and
makes state per-slot rather than per-layer-object (**I2**).

The one-hot `_shift_cache` / `_take_cache` trick in `_write_tail_tile` already keeps the *offset* out
of the program by carrying it as data; the remaining problem is only that the *selection* of which
one-hot to use is a host-side dict lookup. Selecting it on device (a gather over the stacked
one-hots, indexed by a device scalar) would close the gap.

### 9.5 A V4 write op

Whichever of the above lands, V4 needs a cache write with:

* a `slot_idx` dimension (**I2**),
* a metadata-tensor overload (**I5**),
* entry-space offsets (`kv_actual_global // m`),
* and no tile-alignment requirement on the write offset, or the tail-tile dance stays.

Extending `update_padded_kv_cache` with an entry-stride parameter is probably cheaper than a new op,
and it keeps one chunk-table builder. Worth costing both ways early — it is on the critical path for
Proposals 9.1 and 9.3 alike.

### 9.6 Fix the dropped residual first

Independent of everything else, and cheap: `TtHCACompressor` should retain the ragged tail instead
of dropping it. Under SP sharding the residual lives on whichever chip got the tail, so use the same
device-tensor-indexed slice `_attention` already uses for `sliding_carry`
(`self._carry_index`). This is a small, self-contained change that can land and be PCC-tested against
`DeepseekV4HCACache.store_compression_weights` well before the cache redesign, and it unblocks any
real handoff experiment.

---

## 10. Open questions for the decode team

Collected from the thread; each needs an answer before the layout is frozen.

1. **Residual buffer contents: projected `(kv, gate)` or raw hidden states?** The reference buffers
   post-projection `kv` and `gate`, both at `head_dim`. Ilia described `[m, hidden_state]`
   width-sharded, which reads as pre-projection. Post-projection is smaller for both variants
   (HCA: 128 × 512 × 2 vs 128 × 4096; CSA: 4 × 512 × 2 vs 4 × 4096) and avoids re-projection at
   resume. **Proposal: post-projection `[m, 2 * head_dim]`.** Needs confirmation, and the reference
   `tests/blaze/fused_ops/deepseek_v4/test_compressor.py` per-iteration state checks are the place
   to settle it.
2. **Residual layout: head-aligned prefix or `pos % m` ring?** See §9.2 — these coincide if prefill
   writes at the head of a full `m`-row buffer. Confirm decode reads `pos % m` and tolerates a stale
   tail.
3. **Sliding window: 127 or 128 rows, and which absolute positions?** The reference keeps
   `-sliding_window + 1`; `TtHCA` keeps `sliding_window`. Functionally equivalent under the mask, but
   the handoff must be unambiguous.
4. **dtypes and layouts per cache** (Brian's item, still open): compressed entries, indexer entries,
   sliding window, residual buffer. The prefill side would prefer bf16 row-major for anything
   `sparse_sdpa` reads and bfp8 tile for indexer keys, mirroring GLM — but decode's read path should
   drive this.
5. **Does decode want CSA entries sharded or replicated?** §9.3 argues sharded on the prefill side.
   If decode's layout differs, the migration worker absorbs the transform, and someone should own
   that cost explicitly.
6. **Table position axis: entry-space or token-space-with-stride?** §7.3 recommends token-space with
   `chunk_n_tokens = 32 * m` so the worker needs no per-layer `m`.
7. **Does anything need the `entry_count` / `cumulative_length` counters shipped**, or is deriving
   them from N acceptable? §7.2 argues derivation is sufficient and strictly simpler.
8. **CSA overlap state (`overlap_kv` / `overlap_gate`)** — one `head_dim` row per name per layer.
   Confirm decode maintains the same Ca-only convention, and fold it into the residual config rather
   than giving it its own.

---

## 11. Sizing comparison

Per user, per layer, at 56 320-token context. "Logical" is the total across the mesh; "per chip"
assumes SP = 8.

| Model / cache | Rows | Width | Bytes/row | Logical | Per chip (sharded) | Per chip (replicated) |
|---|---|---|---|---|---|---|
| GLM-5.2 `kvpe` (bf16 RM) | 56 320 | 576 | 1152 | 64.9 MB | **8.10 MB** | — |
| GLM-5.2 `index` (bfp8 tile, 21/78 layers) | 56 320 | 128 | 136 | 7.7 MB | **0.96 MB** | — |
| V4 HCA entries (m=128) | 440 | 512 | 1024 | 0.45 MB | 0.06 MB | **0.45 MB** ← today |
| V4 CSA entries (m=4) | 14 080 | 512 | 1024 | 14.4 MB | **1.80 MB** | **14.4 MB** ← today |
| V4 CSA indexer entries | 14 080 | 128 | 256 | 3.6 MB | **0.45 MB** | 3.6 MB |
| V4 sliding window (all layers) | 128 | 512 | 1024 | 0.13 MB | 0.13 MB | 0.13 MB |
| V4 HCA residual (kv+gate) | 128 | 1024 | 2048 | 0.26 MB | 0.26 MB | 0.26 MB |
| V4 CSA residual (kv+gate) | 4 | 1024 | 2048 | 8 KB | 8 KB | 8 KB |
| V4 CSA indexer residual | 4 | 256 | 512 | 2 KB | 2 KB | 2 KB |
| V4 CSA overlap (kv+gate, ×2 names) | 1 | — | — | ~2.5 KB | 2.5 KB | 2.5 KB |

Two conclusions:

**V4's cache is far smaller than GLM's.** A CSA layer needs ~18 MB logical against GLM's ~65 MB, and
an HCA layer needs under half a megabyte. That is the architecture doing what it was designed to do,
and it means DRAM pressure is not the constraint it is for MLA.

**But replication throws most of that away.** Today's SP-replicated CSA entry cache costs 14.4 MB
per chip where sharding would cost 1.80 MB — an 8× multiplier on the single largest V4 cache.
Sharding CSA is the highest-value single change in this document after fixing the dropped residual.

The small blobs (residual, overlap, sliding) total well under 0.5 MB per layer per user. They are a
correctness problem, not a capacity problem — which is worth saying plainly, because the thread's
concern is about migrating them, not about their size.

---

## 12. Suggested sequencing

Ordered so each step is independently testable and the cheap correctness fixes land first.

| Step | Work | Validated by |
|---|---|---|
| 1 | Retain the compressor residual in `TtHCACompressor` (§9.6) | PCC vs `DeepseekV4HCACache.store_compression_weights`, ragged chunk sequences |
| 2 | Settle the open questions in §10 with the decode team; freeze the wire format | a short written spec both sides sign off |
| 3 | `layer_class_rank` helper + shape/stride unit tests, no device (§9.1) | unit test, mirroring `full_indexer_rank` coverage |
| 4 | Cache write op: slot dim, metadata tensors, entry stride (§9.5) | op unit test beside `test_deepseek_prefill_update_padded_kv_cache.py` |
| 5 | `V4KvCaches` + allocators; port `TtHCA` off `TtHCAState` onto the engine-owned cache | existing `test_ttnn_hca.py` chunked scenarios, unchanged PCC floors |
| 6 | SP-shard `csa_entries` / `csa_index`; gather at read (§9.3) | CSA PCC once CSA exists; DRAM accounting |
| 7 | Device-resident counters (§9.4) | trace capture on an HCA-only layer stack |
| 8 | `kv_migration_stages` (5 stages) + V4 chunk-table builder | `PREFILL_MOCK_MIGRATION=1` |
| 9 | Producer read-back branch for the V4 layout | `PREFILL_PRODUCER_CHECK_PCC=1` |
| 10 | End-to-end handoff test against decode | a real prefill → migrate → decode run at N not divisible by m |

Steps 1–4 are independent of the CSA and mHC work in
[`DEEPSEEK_V4_BRINGUP.md`](./DEEPSEEK_V4_BRINGUP.md) and can proceed in parallel. Step 5 needs a
decision on whether to keep `TtHCAState` for unit tests — recommended, as a thin adapter over
`V4KvCaches` so the existing test suite keeps working.

Step 10 is the one that actually closes the thread, and it should be scheduled deliberately: with
`N % m != 0` on both an HCA and a CSA layer, because that is precisely the case every layer of the
current implementation gets wrong.
