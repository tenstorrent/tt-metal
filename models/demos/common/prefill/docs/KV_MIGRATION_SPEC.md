# Prefill KV migration, from first principles

This document explains why KV-cache migration exists, what the **prefill source table** is, and how a
tt-metal prefill deployment publishes source addresses so a mover — the legacy shmem worker or
tt-d-gen `kv_manager` — can copy those bytes into a tt-blaze decode deployment.

It assumes no prior knowledge of the prefill runner. The decode-side counterpart — what
`KvMigrationSpec` is, how tt-blaze builds the *destination* table, and how decode resumes — is
`docs/kv_migration_first_principles.md` in the tt-blaze checkout. This file does not repeat that
material.

Shorter operational docs, once the concepts here are clear:

- wiring a model into the runner: [`ADDING_A_PREFILL_MODEL.md`](ADDING_A_PREFILL_MODEL.md)
- running the prefill-only gates: [`PREFILL_MIGRATION_TESTING.md`](PREFILL_MIGRATION_TESTING.md)

Worked source tables in this tree: GPT-OSS (`gpt_oss_d_p/tt/runners/kv_chunk_table.py`), MiniMax-M3
(`minimax_m3/tt/runners/kv_chunk_table.py`), Gemma 4 (`gemma4/tt/runners/kv_chunk_table.py`), DeepSeek
MLA (`deepseek_v3_d_p/tt/runners/kv_chunk_table.py`).

---

## 1. Why a KV cache must move

Transformer attention needs information from all earlier tokens. Recomputing that information for
every generated token would be wasteful, so each attention layer stores a **KV cache**: the keys and
values, or a model-specific compressed equivalent, produced for earlier positions.

Inference has two phases:

1. **Prefill** processes the prompt and creates KV state for positions `0..prompt_len-1`.
2. **Decode** generates one new token at a time and reads that prompt KV state on every step.

In a disaggregated deployment, prefill and decode run on different accelerator groups. Prefill has
already performed the expensive prompt computation, but the resulting bytes are in prefill's DRAM.
Decode cannot use them until they have been copied into decode's own KV-cache allocation.

That copy is KV migration.

Without migration, the decode deployment would have to prefill the prompt again. With migration,
decode can begin generation using the state prefill already computed.

---

## 2. Minimal prefill vocabulary

- A **chip** is one accelerator device.
- A **mesh** is a rectangular group of chips. Prefill usually names the two axes **SP** (sequence
  parallel: the prompt is split across rows) and **TP** (tensor parallel: heads or experts are split
  across columns). A typical shape is 8 SP rows by 4 TP columns.
- A **rank** is one host process. In a pipelined prefill run, each rank owns a contiguous slice of
  transformer layers and the KV for those layers across the *full* mesh.
- The **engine** is the model-agnostic prefill runner
  (`models/demos/common/prefill/runners/prefill_runner.py`). It owns sockets, the request loop, cache
  lifetime, and migration *comms*. It never imports a model class.
- An **adapter** is the model's factory (`PrefillModelAdapter`). It says where config and weights
  live, allocates the KV cache, and builds a runtime.
- A **runtime** is the model's operational object. It compiles, prefills a chunk, and — when
  migration is on — builds the source address table. It does not own the cache; the engine passes
  the cache in on every call.
- **DRAM** is device memory. Prefill's packed KV cache is split across DRAM **banks** (typically 8
  on a full Blackhole part; harvested parts expose fewer).
- A **NoC address** identifies a byte location in a chip's memory system. Migration encodes a DRAM
  bank and a per-bank byte offset into `noc_addr`: `(bank_id << 32) | offset`.
- A **slot** is one user's KV-cache allocation. Different requests occupy different slot IDs.
- A **chunk** is the smallest unit migration moves. It normally covers 32 token positions — one
  tile-row of the packed cache.
- A **fabric node id** (`FabricNodeId`) is the chip's identity on the TT fabric: `(mesh_id, chip_id)`.
  The worker cannot open a device from that pair alone; each rank also publishes a **device map**
  from fabric node to the chip's hardware-stable ASIC unique id (UMD id).
- A **migration endpoint** is the process that owns the outward shared-memory queues. It copies no
  KV itself. It relays commands to **migration workers**, which are the processes that read and
  write DRAM. That is the *legacy* path (tt-llm-engine, also vendored in tt-d-gen's
  `engine/legacy/migration_layer`).
- **`kv_manager`** (tt-d-gen) is the standalone migrator that *does not* sit on those queues. It
  loads the source and destination protobuf tables from disk, accepts a logical `migrate` command
  (HTTP or ZMQ), and copies bytes in three hops: device DRAM → host staging → RDMA/TCP → host
  staging → device DRAM. Prefill reaches it with `PREFILL_MIGRATION_EXPORT_TO_FILE=1`.
- A **staging slab** is a page-pinned window of host memory that `kv_manager` uses as the middle
  hop. Device DMA and the network engine never share a device address; they share a host buffer.
- **LayerAck** is a per-layer completion signal the runner emits after it has written that layer's
  KV for a chunk. A production scheduler can start migrating a layer before the rest of the prompt
  is done. The Python `migration_driver` instead drains every ack, then issues `migrate()`.

---

## 3. The key idea: logical identity is separate from physical location

Humans describe a piece of KV state logically:

```text
(tensor group, layer, token position, user slot)
```

The migration API represents the tensor group by an integer `config_idx`. A transfer has a source
key and a destination key:

```text
K_src = (config_idx, layer, src_position, src_slot)
K_dst = (config_idx, layer, dst_position, dst_slot)
```

Hardware needs a different description:

```text
(host worker, fabric chip or replica group, DRAM bank, byte offset, byte count)
```

The config and layer identify corresponding model state. Migration may deliberately change the slot
(prefill slot 7 → decode slot 2). Current prefill workloads use identical position ranges; the
worker can in principle relocate positions if both tables have matching chunk counts.

```text
source_table[K_src]      = where prefill stored this chunk
destination_table[K_dst] = where decode allocated space for this chunk

copy source_table[K_src] -> destination_table[K_dst]
```

This separation is what permits prefill and decode to use different chips, tensor base addresses,
mesh shapes, and physical address layouts. The corresponding chunks must still have the **same byte
representation and size**: migration does not convert dtypes, retile, reshape, or repack data.

If prefill stores a 32-token by 576-channel `bfloat8_b` blob, decode must have reserved a 19,584-byte
hole of the same encoding. The worker will copy those 19,584 bytes bit-for-bit onto decode's DRAM,
at whatever bank and offset decode's table named.

---

## 4. An important correction: prefill does not discover decode's DRAM addresses

The prefill table builder does **not** inspect decode hardware. Prefill Python never calls
`buffer_address()` on a decode tensor, never imports a `KvMigrationSpec`, and cannot write decode
DRAM directly.

There are two independently constructed address tables:

1. The **prefill / source table** maps source logical coordinates to prefill hardware. This
   repository builds it.
2. The **decode / destination table** maps destination logical coordinates to decode hardware.
   tt-blaze builds it from `KvMigrationSpec`.

There are two ways those tables meet a mover:

1. **Shmem worker (legacy).** Each side `SET_TABLE`s its protobuf into a co-located
   `migration_endpoint`. A `migrate()` command names logical ranges; the worker looks up both
   tables and copies. This is what `PREFILL_ENABLE_MIGRATION=1` plus `migration_driver` drive.
2. **`kv_manager` (tt-d-gen).** Each KV-manager process *poll-loads* table files from disk
   (`PREFILL_TABLE` / `DECODE_TABLE` plus a device map). A `POST /migrate` (or ZMQ equivalent)
   names the same logical ranges. The control plane plans host-pair work from *both* tables; the
   data plane performs the three hops. No `SET_TABLE`, no pairing of Python clients. This is what
   `PREFILL_MIGRATION_EXPORT_TO_FILE=1` is for.

Either mover is layout-opaque in the same sense: it never calls Python `locate()`. It only consumes
the protobuf the prefill builder already filled.

Prefill's job is:

- allocate and write the source KV;
- publish a source table whose **logical** organization (config order, chunk size, layer ids)
  matches decode;
- request `migrate(...)` for a logical range;
- tell decode, via a small JSON handoff, which destination slot and prompt length to resume from.

Prefill's job is **not** to know where decode put the bytes. That is the destination table.

---

## 5. What the source table is

Decode authors a `KvMigrationSpec`: immutable callbacks that *later* become a table once live
addresses are gathered. Prefill authors the table more directly. The model supplies a builder that,
given live tensors, fills a `KvChunkAddressTable` and serializes it to protobuf. The engine then
either `SET_TABLE`s that file into the shmem worker and waits for `WORKER_READY`, or writes it to
shared storage for `kv_manager` to poll-load (`PREFILL_MIGRATION_EXPORT_TO_FILE=1`).

The builder is not:

- the KV tensor;
- a transport protocol;
- the component that copies bytes;
- a Python object decode will import.

It is a recipe that, combined with live `buffer_address()` values and live fabric node ids, produces
the source routing table.

The types live in `ttnn.experimental.disaggregation` (`KvChunkAddressTable`,
`KvChunkAddressTableConfig`, `KvCacheLocation`). Model code that fills them lives next to the model,
typically `models/demos/<model>/tt/runners/kv_chunk_table.py`.

### 5.1 Per-config header (`KvChunkAddressTableConfig`)

Each config carries:

```text
num_layers              how many semantic transformer layers the table has rows for
max_sequence_length     how far this config's position axis goes (may be a window, not full S)
num_slots               user slots (must be uniform across configs)
chunk_n_tokens          positions per migration entry (almost always 32)
chunk_size_bytes        exact bytes the worker copies for one entry of this config
```

`chunk_size_bytes` is a function of dtype and row width, not of sequence length:

```text
tile_bytes(bfloat8_b)  = 1088     # 32x32 tile: 1024 mantissa + 64 shared exponent
tile_bytes(bfloat16)   = 2048

chunk_size_bytes = (chunk_n_tokens / 32) * (row_dim / 32) * tile_bytes
```

A 32-token by 64-channel bf8 chunk is `2 * 1088 = 2176` bytes. A 32-token by 576-channel bf8 chunk
is `18 * 1088 = 19584` bytes. K and V (or a local family and a merged-global family) may differ, as
long as *this* `config_idx` agrees with decode.

`max_sequence_length` may differ per config. A sliding-window K head only needs 1024; a global
merged row needs the full context. That is how a window becomes an **extent** (§6.4).

### 5.2 Per-chunk location (`KvCacheLocation`)

```text
noc_addr             = (bank_id << 32) | (base_addr + per_bank_offset)
size_bytes           = this config's chunk_size_bytes
device_group_index   = which replica group holds the bytes
```

`table.set(layer, position, slot, location, config_id)` writes one cell. Unset cells stay at the
canonical absent marker (zeros). The protobuf exporter skips those.

tt-d-gen treats absence as `size_bytes == 0`: `KvChunkAddressTableAdapter::lookup` returns
`nullopt`, and the chunk-index builder never enqueues that cell. A windowed config whose
`max_sequence_length` is 1024 therefore contributes no source reads past position 1023, which is
exactly the extent rule. The migrator does not wrap `p % sw`.

### 5.3 Device groups and device maps

`add_device_group(fabric_node_ids)` records the chips that hold **replicas of the same bytes**. The
worker may read any member and must write the same payload to every destination member.

That definition decides how you split configs:

- **TP-head-sharded K/V.** Column `c` holds a *different* head, not a copy. One config per
  `(tensor, head)` with a **single-member** group: that head's TP chip (and, with SP, the SP row that
  owns this position).
- **Replicated tensor.** An MLA latent or MiniMax `index_k` is the same bytes on every TP column of
  the SP row. One config with a **multi-member** group.

Heads that *share* a chip (Gemma 4 local: 16 heads / TP=4 → 4 per device) still need one config per
logical head. They share a device group and differ in the tensor's head dimension. Collapsing them
into one config would tell the worker the four heads are replicas.

The table's fabric node ids are not enough to open a device. Each rank also publishes a **device
map**: fabric node → hardware-stable ASIC unique id (`mesh_id chip_id umd_id` per line).

- **Shmem path:** the engine pushes that map to the co-located A/B workers over their *internal*
  queues (`/ep_*_{a,b}_*`), not over the outward `SET_TABLE` queues.
- **`kv_manager` path:** the same map is a file (`DEVICE_MAP` / `KV_MANAGER_DEVICE_MAP_PATH`).
  `kv_manager` poll-loads it on startup.

`set_fabric_node_host` records which *process* owns a fabric node. For `kv_manager`, that string
must equal `KVM_ID` (for example `prefill-0`). Using `socket.gethostname()` without making
`KVM_ID` the same string makes planning report "this host has no chunks" while DRAM is full.

### 5.4 Config names versus config indices

Workers route by integer `config_idx`, not by Python name. Tuple / insertion order at table
construction is the contract with decode.

Protobuf import rebuilds configs via `std::map<std::string, …>` (lexicographic name order). The list
constructor auto-names configs `"0".."N-1"`, which **reorders for N>10** (`"10"` sorts before `"2"`).
Semantic names (`k_h10`) sort equally badly. After import, integer id 2 would name a different
tensor than decode's id 2, and bytes would land in the wrong cache even though both tables look
valid in isolation.

Build with the **dict constructor** and **zero-padded decimal names** so map order equals intended
id:

```python
width = max(2, len(str(max(num_configs - 1, 0))))
name  = f"{config_id:0{width}d}"   # 36 configs → "00" .. "35"
```

Keep `k_h0` / `kv_h2` as labels in logs and in a `config_names()` tuple you can cite next to the
decode spec. After construction, assert `table.config_name(i) == padded(i)` for every `i`.

tt-d-gen's chunk-index builder then places each adapter by `configId()`, not by walking the
name-keyed `std::map`. Zero-padding is still required: import assigns those integer ids from
lexicographic name order *before* the builder sees them. The test
`ConfigIdPlacementIndependentOfNameOrder` only proves the *second* step does not re-scramble.

---

## 6. Address math is the model-specific heart of the source table

The byte mover does not understand tensor layouts. It does not know about:

- tiled versus row-major storage;
- NdShard versus interleaved DRAM;
- sequence parallelism;
- per-head sharding;
- bank round-robin;
- the user-major batch fold;
- replicas.

The model's builder translates those rules into one concrete `(noc_addr, device_group)` per logical
key.

The **KV-write path is the source of truth** — usually
`ttnn.experimental.deepseek_prefill.update_padded_kv_cache`, or the model's own fill. The table
must name the address that op actually wrote. A self-consistent but incorrect formula builds a
valid-looking table that Gate 1 will fail (or, worse, that Gate 1 never runs and decode reads
garbage).

Do not copy decode's `locate`. Decode HEIGHT_SHARDED FlashGQA banks (`bph`, `st_pb`, CYCLIC vs
BLOCK, `OPTIMAL_DRAM_BANK_ORDER`) are a different physical layout. Prefill's packed cache uses
DRAM `NdShardSpec` `ROUND_ROBIN_1D`.

### 6.1 The packed prefill cache

`adapter.allocate_kv_cache` is the single place the layout is defined. The table must describe
*those* tensors, not a standalone demo cache and not decode's allocation.

The common packed pattern (GPT-OSS, MiniMax, Gemma 4, DeepSeek kvpe):

```text
shape per chip:  [num_users * num_packed_layers, n_heads_local, seq_local, row_dim]
layout:          TILE
memory:          DRAM NdShard, shard [1, 1, 32, row_dim], ROUND_ROBIN_1D over the DRAM banks
mesh mapper:     ReplicateTensorToMesh at allocate time (empty buffer everywhere;
                 content diverges on the first write)
batch fold:      user-major, slot * num_packed_layers + packed_layer_index
seq_local:       seq_extent / sp
```

`n_heads_local` is 1 when `num_kv_heads == TP` (one head per chip). It is greater than 1 when
several logical heads share a chip; they occupy dimension 1 of the same tensor, still sharded as
`[1, 1, 32, row_dim]`.

`seq_extent` is `max_seq_len` for a full-context family and `sliding_window` for a windowed family.

When two families do not share a shape, allocate two (or three) tensors. Unused slots on a *cheap*
family (a 1024-token ring) can sit in a full `num_layers` batch dim so `layer_idx` stays semantic.
Unused slots on an *expensive* family (full context × a wide merged row) must not: pack only the
layers that own that family, and map `semantic layer → dense index` inside the address formula.

`num_dram_banks` comes from the live device (`mesh_device.dram_grid_size().x`,
`get_num_dram_banks`). Harvested parts expose 7, not 8. Hardcoding 8 stripes the table onto a bank
the buffer does not use.

### 6.2 NdShard walk order

`ROUND_ROBIN_1D` assigns shard 0 to bank 0, shard 1 to bank 1, …, then wraps, advancing the
per-bank offset each full sweep (`buffer_distribution_spec.cpp`, `iterate_over_shards`). Shards are
emitted in dimension order 0 → 1 → 2: **batch, then local head, then 32-token sequence blocks**.

Closed form:

```text
n_seq_blocks = seq_local / 32
shard_id     = batch_idx * (n_heads * n_seq_blocks)
             + local_head * n_seq_blocks
             + (local_pos / 32)

bank         = shard_id % num_dram_banks
offset       = (shard_id / num_dram_banks) * chunk_size_bytes
noc_addr     = (bank << 32) | (base_addr + offset)
```

`base_addr` is that tensor's `buffer_address()` — one base per family, not one per head. Heads of
the same tensor share the base and differ in `local_head` (and therefore in `shard_id`).

For `n_heads == 1` this is the same sequential bank walk GPT-OSS and MiniMax replay with nested
`for slot / layer / seq` loops. For `n_heads > 1`, walking slot and sequence **without** a head loop
is wrong: it pretends every head occupies the same shards. Use the closed form, or walk heads as
dimension 1.

### 6.3 Sequence-parallel placement (block-cyclic)

Prefill does not store position `p` at local index `p` when SP > 1. Each `prefill_chunk` of
`chunk_size` tokens is split across the SP rows: row `r` holds a contiguous
`chunk_size / sp`-token slice of that period, and periods concatenate along the local sequence
axis.

```text
tokens_per_chunk_local = chunk_size / sp
seq_chunk, offset_in_chunk = divmod(position, chunk_size)
sp_row     = offset_in_chunk / tokens_per_chunk_local
local_pos  = seq_chunk * tokens_per_chunk_local
           + (offset_in_chunk % tokens_per_chunk_local)
```

`sp == 1` is the identity: `local_pos = position`, `sp_row = 0`. That is the case that matches a
TP-only decode spec.

Constraints, or the formula is not tile-aligned:

```text
seq_len % chunk_size == 0
(chunk_size / sp) % 32 == 0
seq_extent % (32 * sp) == 0     # per family
```

`chunk_size` is the runtime's per-`prefill_chunk` token count, not a constant inside the table
builder. DeepSeek's older kimi builder currently hardcodes one period; GPT-OSS / MiniMax / Gemma 4
take it as an argument.

### 6.4 A sliding window is an extent, not a folded address

Some layers only keep a short ring (Gemma 4 local: 1024 tokens; GPT-OSS sliding: 128). The physical
cache for that layer is `sw` tokens long. Decode still *reads* with `p % sw`, but that wrap lives
in the attention kernel, not in the migration table.

Two ways you could author the source table:

**Extent (required).** For a windowed `(config, layer)`, only `set` positions `0, 32, …, sw-32`.
Addressing is linear: position 0 is tile-row 0 of the ring, position 32 is tile-row 1, and so on.
Set that config's `max_sequence_length = sw`. Cells past the window stay absent. Decode does the
same with `per_layer_seq_len(layer) = sw`.

**Fold (forbidden).** Put `p % sw` into the address formula and fill the whole `0..S` axis. Then
`p`, `p+sw`, and `p+2*sw` all name the same bytes. The table is “complete,” but you send `S/sw`
copies of every ring slot. At production shape that is hundreds of times redundant on that layer,
and it roughly doubles the protobuf — the exporter already drops default (absent) entries, so a
short extent is the cheap representation.

`migrate()` takes one position range shared across the layer range. Layers with different filled
extents coexist in one *logical* call: the table itself encodes which cells exist.

How a mover treats an absent cell is **not** the same on both paths:

- The **index builder** (`kv_manager`) skips `size_bytes == 0`. It never enqueues that cell, and it
  does not wrap `p % sw`. That is the extent rule implemented as data.
- The **planner** (`MigrationStrategyBuilder`) does the opposite: a missing source or destination
  lookup sets `plan.complete = false`, and `planExecutions` then fails the whole migrate with
  `migration_plan_incomplete`. A range that walks past a windowed config's `max_sequence_length`,
  or across a layer that config does not own, is currently a failed plan, not a skipped cell.
- The **shmem worker** is closer to “skip absent pairs.” Do not assume `kv_manager` will.

So: a window still changes **how many** `(layer, position)` cells exist, and it still does not
change **how** a filled cell's `position` maps to a bank and offset. What it does change, today, is
whether a single `[0, prompt_len)` over all layers can even be *accepted* by `kv_manager` when the
table the planner consults is the short-extent family. See §10.4.

### 6.5 Layer ownership

The table row is the **semantic** transformer index (layer 5 is transformer layer 5), even if a
compact tensor stores that layer at packed index 1.

If a config has no cache on that layer — local K on a full-attention layer, `index_k` on a dense
layer — skip `table.set`. Do not write zeros onto the other family's tensor. Decode's matching
config returns `None` from `device_tensor` on that stage and raises in `locate`. Both sides must
agree on which cells are absent.

---

## 7. The prefill/decode compatibility contract

The source and destination do not need identical physical addresses. They do need compatible
logical tables. Decode's write-up of this contract is tt-blaze
`docs/kv_migration_first_principles.md` §7. The prefill-visible half:

### 7.1 Config order

Insertion order defines integer `config_idx`. If decode lists:

```text
k_h0, k_h1, k_h2, k_h3, v_h0, v_h1, v_h2, v_h3, index_k
  0     1     2     3     4     5     6     7        8
```

prefill must emit the same groups in the same order. Reordering otherwise-valid configs copies K
into V. Appending a family (a drafter's `dflash_*` configs after the model's) is safe; inserting in
the middle is not.

Lock the order with a test both trees can cite (`config_names()` on each side).

### 7.2 Chunk agreement

For each `config_idx`, both sides must agree on `chunk_n_tokens`, `chunk_size_bytes`, dtype, and
row width. The worker copies `size_bytes` with no conversion.

Migration requests are chunk-aligned. For a prompt of `S` tokens and 32-token chunks, the driver
migrates:

```text
[0, S)
```

in the Python client (`pos_end_exclusive=real_len`). The shmem worker still transfers whole
32-token source chunks; positions between `S` and the next chunk boundary are padding. Decode begins
generation at the true logical position `S`, which is why the handoff carries `prompt_len`, not the
rounded endpoint.

Against current `kv_manager`, that same `[0, S)` is also checked against the *first* config's
`max_sequence_length` (§10.4). If that config is a 1024-token window, `S > 1024` is rejected
before any copy.

Current integrated paths use 32-token configs. A larger granularity would need the driver to align
requests to that size.

### 7.3 Layers

The layer coordinate is the same semantic model layer on both sides. Prefill's table has a row per
layer in `0..num_layers-1`. A pipelined prefill run still publishes **one** table: rank 0 merges
every rank's layer slice (`stage_layout`) so row 17 is layer 17 even if rank 2 owns it.

A reduced decode that only hosts layers `{0, 3}` still indexes those rows as 0 and 3, not as 0 and
1. The Python driver then issues one `migrate()` per listed layer (`PREFILL_MIGRATION_LAYERS=0,3`)
because the migrate range is symmetric (`src row == dst row`).

### 7.4 Slots

Source slot and destination slot need not match:

```text
prefill slot 0 -> decode slot 2
```

The source table is addressed with slot 0; the destination table with slot 2. Both sides must have
allocated the referenced slots. Prefill `PREFILL_NUM_USERS` must cover every *source* slot; the
destination's slot count is a decode concern. Loopback on the prefill endpoint needs
`PREFILL_NUM_USERS` large enough for *both* src and dst, because they share one table.

### 7.5 What must *not* match

| | Prefill source (typical packed path) | Decode destination (typical) |
|---|---|---|
| Layout | NdShard `ROUND_ROBIN_1D`, shard `[1,1,32,row]` | HEIGHT_SHARDED FlashGQA, or MLA kernel banks |
| Packing | User-major `[U*L, heads_local, seq_local, dim]` | Per-layer tensor; slots often folded into seq |
| Sequence | SP block-cyclic | Often TP-only; spec may flatten to `1 × TP` |
| Bank | `shard_id % num_dram_banks` | Kernel bank order (`bph`, CYCLIC / BLOCK, …) |
| Mesh coords | Real `MeshCoordinate(sp_row, tp_chip)` | Logical grid; may be flattened `(0, chip)` |
| Device groups | Fabric node ids on the prefill mesh | Fabric node ids on the decode mesh |

The worker never notices that these differ. It only notices if `config_idx` or `size_bytes` differ.

---

## 8. How the prefill runner builds and publishes the source table

The integration begins after the runtime has been built and `compile(kv_cache)` has run, **before**
the request loop opens. The worker gates on `SET_TABLE` plus the device map; if the runner entered
the loop first, the scheduler could issue `migrate()` against an empty table.

The engine owns steps 1, 2, 4, and 5. The model owns step 3.

### Step 1: gate on runtime hooks

`PREFILL_ENABLE_MIGRATION=1` requires:

```python
def kv_migration_base_address(self, kv_cache) -> int: ...
def build_kv_chunk_table(self, kv_cache, path: str, **pp_kwargs) -> str: ...
```

`kv_migration_base_address` returns *a* live DRAM base (typically K or kvpe). The engine all-gathers
it as an anchor for the stage-layout merge. It is **not** the per-config base the table uses; the
builder reads each tensor's own `buffer_address()`.

`build_kv_chunk_table` must issue **no comms**. Extra kwargs (`first_layer_idx`, `num_my_layers`,
`stage_layout`) exist for DeepSeek's pipeline-parallel merge. A single-rank GQA path ignores them.

### Step 2: every rank delivers its device map and joins the stage-layout gather

Each rank owns layers `[first_layer_idx, first_layer_idx + num_my_layers)`. It contributes:

- that layer range;
- its KV-base anchor;
- the fabric identity of every chip in its mesh;
- a host tag (crc32 of hostname) so the worker knows which host owns those chips.

Real migration also pushes the local fabric-node → ASIC map to the co-located A and B workers
(`deliver_device_map_and_gather_stage_layout`). This is a collective: every rank must reach it or
the communicator deadlocks.

Rank 0, which may own only the first layer slice, now knows where every layer's source cache lives.

### Step 3: rank 0 asks the model to build the merged table

```python
path = runtime.build_kv_chunk_table(
    kv_caches,
    table_path,
    first_layer_idx=...,
    num_my_layers=...,
    stage_layout=stage_layout,   # gathered in step 2
)
```

The model loops, conceptually:

```text
for config in ordered_specs:          # the decode contract
  for slot in num_users:
    for layer in num_layers:
      if this config has no cache on this layer: skip
      for position in 0, 32, 64, ... extent(config, layer):
        local_pos, sp_row = block_cyclic(position)
        batch = packed_batch(slot, layer, family)
        shard = ndshard_shard_id(batch, local_head, local_pos, ...)
        noc   = pack(bank=shard % banks, offset=...)
        group = device_group(sp_row, tp_chip_for_head)
        table.set(layer, position, slot, location, config_id)
```

Then `export_to_protobuf_file(table, path)`.

With `num_ranks > 1` the path must be **shared storage**. Rank 0 writes it; every host's driver
reads it. `/tmp` is rejected. The device-map JSON is the opposite: **host-local**, because each
rank publishes only its own chips and that is what scopes a driver rank to its own layers.

### Step 4: rank 0 publishes the table and waits for WORKER_READY

```text
MigrationLayerClient.send_kv_chunk_table(path)   # SET_TABLE on the outward table queue
MigrationLayerClient.wait_ready(timeout)         # blocks until workers hold table + maps + A↔B link
```

Rank 0 holds the client for the process lifetime. Dropping it destroys the client and the worker
loses the table it gated on.

### Step 5: the request loop opens

Only now may H2D chunks flow. LayerAck (if enabled) starts ticking as layers complete. The runner
still has not copied any KV. Table construction only prepared the translation from logical keys to
hardware.

### Modes that skip the shmem worker

| Mode | What it does | What it does not |
|------|----------------|------------------|
| `PREFILL_ENABLE_MIGRATION=1` | Device maps to shmem workers, merged table, `SET_TABLE`, `WORKER_READY` | Call `migrate()` |
| `PREFILL_MOCK_MIGRATION=1` only | Write table protobuf + JSON device map to disk | Talk to a worker. Single-rank only. |
| `PREFILL_MIGRATION_EXPORT_TO_FILE=1` | Write table protobuf + text device map for **tt-d-gen `kv_manager`** | `SET_TABLE` / `WORKER_READY`. `kv_manager` poll-loads the files. |

Mock exists so Gate 1 can prove the table without tt-llm-engine binaries. File-export is the
bring-up path into `kv_manager` (see §10).

---

## 9. How migration executes after both tables exist

In cross-endpoint queue mode the two sides have independently registered:

```text
prefill endpoint -> source KvChunkAddressTable + source device maps
decode endpoint  -> destination KvChunkAddressTable + destination device maps
```

No KV bytes have moved yet.

The other modes differ:

- **Prefill loopback** uses one local table and endpoint for both source and destination (Gate 2).
  `dest_endpoint_id` equals the prefill endpoint's own id.
- **File-export** writes metadata for an external KV manager and performs no endpoint registration.

### 9.1 Pair the endpoints (cross-endpoint only)

The driver calls `connect_to(peer_endpoint_id, service_name)` on the source client. Decode
establishes the matching side. The current convention:

```text
lower endpoint ID  -> PUBLISHER
higher endpoint ID -> CONNECTOR
service name       -> pd-migration-ep<low>-ep<high>
```

`PUBLISHER` / `CONNECTOR` are connection-establishment roles, not model roles. Decode may be the
publisher even though prefill is the source of the bytes.

Pairing associates the two tables. It does not send a Python spec across the boundary. Without
pairing, `migrate()` aborts with `No remote table found for destination`.

Loopback does not pair with a peer; source and destination lookups hit the same table.

### 9.2 Prefill finishes writing the requested source range

Before `migrate()` starts, every requested `(layer, position)` of the source slot must be stable.
The runner writes KV inside `prefill_chunk`. Completion is visible as LayerAck: one ack per layer
per chunk.

Two consumers of those acks:

- The **C++ PrefillScheduler** (production) can issue a per-layer `migrate()` as soon as that
  layer's ack lands, overlapping remaining prefill.
- The **Python `migration_driver`** drains every ack, then issues one (or a handful of) `migrate()`
  calls. The Python client binds no burst API.

Either way, the source range must not be overwritten while the worker is reading it.

### 9.3 Issue a logical migration request

The **runner never calls `migrate()`**. Rank 0's `MigrationLayerClient` is a setup client
(`SET_TABLE` / `wait_ready`). After `WORKER_READY` it sits idle so the command queue is free. The
driver (or scheduler) attaches a *second* client to the same outward queues and sends migrate
commands.

A request carries:

```text
migrate(
    remote_endpoint_id,          # peer, or self for loopback
    src_slot, dst_slot,
    layer_start, layer_end_exclusive,
    pos_start, pos_end_exclusive,
)
```

then `wait_complete(token)`.

Default: one call over `[0, num_layers)` and `[0, prompt_len)` per `(src, dst)` pair. A reduced
decode uses `PREFILL_MIGRATION_LAYERS` to emit one call per listed layer.

Source and destination slots may differ. The Python driver currently uses identical position
ranges. Cross-position relocation is a worker capability, not something this tree validates.

### 9.4 Expand the request into table lookups

The command is still logical. It names slots, a layer range, and a position range — not chips,
banks, or `config_idx`. Both movers expand that rectangle into concrete locations using the
already-built tables:

```text
K_src = (config_idx, layer, src_position, src_slot)
K_dst = (config_idx, layer, dst_position, dst_slot)
src_location = source_table[K_src]
dst_location = destination_table[K_dst]
```

How that expansion turns into DMA is mover-specific. The shmem worker copies
`(device_group, noc_addr, size_bytes)` pairs more or less directly. **tt-d-gen `kv_manager` does
not**: it stages through host memory and a network engine. That path is §10. The next two
subsections only apply to the shmem worker; skip to §9.7 for completion/handoff either way.

### 9.5 Resolve table locations to physical devices (shmem worker)

Each entry plus the device map answers:

```text
which host worker?
which physical chip or replica group?
which DRAM bank?
which byte offset?
how many bytes?
```

Owning-host tags in the table route work to the correct worker. The local map turns a fabric node
into an ASIC the UMD driver can open. This is why a stale or missing device-map JSON makes
device-less read-back log `device map ... not found` even when `SET_TABLE` succeeded: the *worker*
got the map over shmem, but the Python reader did not.

### 9.6 Copy the bytes (shmem worker)

```text
for each paired (K_src, K_dst):
    src = source_table[K_src]
    dst = destination_table[K_dst]
    transfer src.size_bytes
        from (src.device_group, src.noc_addr)
        to   (dst.device_group, dst.noc_addr)
```

No dtype conversion, reshape, or model computation. Replica groups are resolved independently on
each side; absolute chip identities need not match.

In prefill loopback, source and destination groups are chips on the same mesh, possibly the same
chip at two slot offsets. In cross-endpoint mode they are different galaxies.

### 9.7 Confirm completion, then publish handoff

```text
prefill KV writes finished
    -> LayerAck drain (Python) or per-layer overlap (scheduler)
    -> migrate() + wait_complete()
    -> write handoff JSON          # cross-endpoint only
    -> write DONE sentinel
    -> decode reads destination KV
```

The handoff is application state, not hardware addresses:

```json
{
  "slots": [
    {
      "dst_slot": 2,
      "prompt_len": 56320,
      "last_prompt_token": 1234
    }
  ]
}
```

`dst_slot` tells decode which migrated cache to use. `prompt_len` is the next decode position.
`last_prompt_token` lets the current decode-only path derive the first generated token by replaying
the last prompt position.

Handoff is written **before** DONE so a consumer that wakes on DONE never sees a partial file.
Loopback omits the handoff: the driver verifies the destination itself.

DONE means “copied,” not “verified.” Destination byte-compare / golden PCC runs after it.

A safe integration must not let decode read or overwrite the destination range before
`wait_complete` returns. tt-blaze cannot enforce the producer's ordering; the DONE sentinel is the
optional application-level barrier.

On the `kv_manager` path there is no `MigrationLayerClient` and no DONE from this Python driver.
Completion is the HTTP/ZMQ ack from `POST /migrate`. The handoff JSON is still the application
signal decode uses to resume.

---

## 10. How tt-d-gen `kv_manager` copies the bytes

The rest of this document is still about *authoring* the source table. This section is what that
table actually drives once `PREFILL_MIGRATION_EXPORT_TO_FILE=1` has dropped it on disk. The mover
lives in the tt-d-gen checkout under `kv_manager/`. It is a standalone process, not a Python import.

### 10.1 Two movers, one protobuf

| | Legacy shmem worker | `kv_manager` |
|---|---|---|
| How it gets the table | `SET_TABLE` over outward queues | Poll-load `PREFILL_TABLE` / `DECODE_TABLE` from disk |
| How it gets the device map | Internal `/ep_*_{a,b}_*` queues | `DEVICE_MAP` text file (`mesh_id chip_id umd_id`) |
| Command | `MigrationLayerClient.migrate(...)` | `POST /migrate` or ZMQ; JSON body, no `config_idx` |
| Copy path | Device DRAM → device DRAM (worker-internal) | Three hops through host staging (§10.5) |
| Identity | Endpoint id + service name | `KVM_ID` must match `set_fabric_node_host` |

The protobuf schema is the same object `ttnn.experimental.disaggregation` already serializes. Prefill
does not grow a second table format for `kv_manager`.

`kv_manager` also vendors the old shmem stack at `engine/legacy/migration_layer/` and wraps it as
`MigrationKvManagerClient`. That is the *other* binary. File-export talks to the standalone
`kv_manager` process, not that wrapper.

### 10.2 Control plane and data plane

Split the process in two, the way the code is split:

**Control plane** loads tables (retry loop, 5 s, up to `KV_MANAGER_TABLE_LOAD_MAX_RETRIES`),
discovers peers in etcd (`kv_control/<kvm_id>`), plans host-pair work, books staging slabs, and
fans the migrate out:

```text
parent migrate (logical rectangle)
    -> plan: one booking per (layer, prefillHost, decodeHost, position run)
    -> local child  -> data-plane TRANSFER
    -> remote child -> ZMQ COMMAND to a prefill subordinate
    -> each sealed batch -> decode DRAIN (local or GET_REMOTE_SLAB + remote DRAIN)
```

The prefill **leader** (`ROLE=leader`, typically `prefill-0`) is the one that partitions. Prefill
subordinates execute TRANSFER into slabs the leader already assigned. Decode leaders/subordinates
execute DRAIN into their own DRAM.

**Data plane** never parses protobuf. It consumes two immutable indexes the control plane installed
after load (`setChunkIndexesForDataPlane`): a **read index** on the prefill host and a **write
index** on the decode host. Operations:

- `TRANSFER` — prefill host: DRAM → local slab, then Mooncake WRITE to the peer's slab.
- `DRAIN` — decode host: walk the self-describing batch in the local slab, DRAM write.
- `COPY` — `POST /slot_copy`, decode→decode. `doCopy` is currently a stub (`(void)request`).

### 10.3 How the table is consumed

`KvChunkAddressTableAdapter::allFromProtobufFile` builds **one adapter per config**, keyed by
protobuf name, each holding the integer `configId` from import index `i`. Lookup:

```text
if size_bytes == 0: return nullopt     # absent / unauthored / past the window
else:               return (noc_addr, size_bytes, device_group_index)
```

`noc_addr` is already packed the way DeviceIO wants it: `(dram_channel << 32) | local_offset`.
Prefill's NdShard formula must emit that packing; `kv_manager` does not re-derive a bank.

The **chunk-index builder** then scans every config, every slot, every layer, every `pos +=
chunk_n_tokens`:

- **Read (TRANSFER):** each chunk is pulled **once**, from the **designated reader** — strictly
  `device_group.front()`. This host enqueues the chunk only if `hostOf(front())` equals this
  process. A later replica on this host does **not** become the reader. Replica members on other
  hosts do not appear in this host's read queues.
- **Write (DRAIN):** the same payload is fanned out to **every local replica** in the group. A
  decode host that holds no copy of that chunk skips the record (`writeLocFor` returns null); that
  skip is normal in multi-host.

Adjacent cells **coalesce** into a run only when all of these hold: contiguous token positions,
contiguous DRAM (`addr + size` matches the next `noc_addr`), same byte size, and (on the write
side) the same local replica set. Prefill NdShard `ROUND_ROBIN_1D` almost never coalesces — each
32-token tile-row is a different bank — so a run is one chunk. Decode HEIGHT_SHARDED can coalesce
up to a bank's contiguous tokens (the comment in `kv_chunk_index.hpp` says 128). Do not assume
prefill runs are long.

**Device-group order is load-bearing.** `add_device_group([A, B])` makes `A` the designated reader.
Swap the list and `B` is the only chip TRANSFER will DMA from. Single-member groups (TP-sharded
heads) make this a non-issue; replica groups (MLA, MiniMax `index_k`) must put a reachable local
chip first on each host, or that host's read index is empty for those chunks.

### 10.4 Planning: host pairs, and fail-closed on holes

`POST /migrate` JSON is the same logical rectangle the shmem `migrate()` uses. There is no
`config_idx` field:

```json
{
  "migration_id": 1,
  "src_slot": 0, "dst_slot": 2,
  "layer_begin": 0, "layer_end": 60,
  "src_position_begin": 0, "src_position_end": 4096,
  "dst_position_begin": 0, "dst_position_end": 4096
}
```

`MigrationStrategyBuilder` looks up **one** source table and **one** destination table —
`tablesFor()` returns `firstLoadedTable`, which is `std::map.begin()`, i.e. the lexicographically
first **name**. With zero-padded names that is `"00"` = `config_idx` 0.

It then walks every `(layer, position)` in the command against **that pair only**. Bookings group
by `(layer, prefillHost, decodeHost)`. Hosts come from `hostOf` on the chunk's device group.

A lookup miss is fatal to the plan:

```text
if (!srcLoc || !dstLoc)           plan.complete = false;  # keep walking, but the plan is tainted
if (hosts empty)                  plan.complete = false;
if (positionEnd > max_seq_len)    plan.complete = false; return empty   # geometry, not a hole
```

`planExecutions` refuses anything that is incomplete or has no bookings (`migration_plan_incomplete`).

Two table-author consequences, both live today:

1. **Window vs prompt length.** On Gemma 4, config 0 is local K: `max_sequence_length = 1024`. A
   migrate of `[0, 4096)` fails the geometry check before any DMA. `withinTableGeometry` uses the
   same first table, so the HTTP command is rejected even earlier. Any model whose first config is
   the short-extent family has the same trap.
2. **Wrong-family layers.** Config 0 has no cells on global layers. A migrate over `[0, n_layers)`
   hits those rows, `lookup` returns `nullopt`, `complete = false`. A dual-family table cannot be
   moved with one all-layers call through this planner until the planner walks every config and
   treats absence as skip.

The index builder already skips holes. The planner does not yet use that skip. Author the table
as if skip-absent were true — that is the contract with decode — but when you issue `POST /migrate`
against current `kv_manager`, size the rectangle to cells the *first* config actually owns, or
expect `migration_plan_incomplete`.

### 10.5 Three hops, self-describing batches

Device DMA and the network engine never share a device address. They share a **staging slab**: a
page-pinned window of the host buffer `kv_manager` registered with Mooncake once at start.

```text
prefill DRAM  --D2H-->  prefill slab  --RDMA/TCP WRITE-->  decode slab  --H2D-->  decode DRAM
                 TRANSFER                         Mooncake                      DRAIN
```

`TRANSFER` round-robins the designated-reader queues for `request.src` (one layer, one slot, a
position span), packs each chunk into the slab as:

```text
64-byte BatchKvChunkHeader (magic 0x4b564348 = "KVCH")
    configId, slot, layer, posBegin, posEnd, sizeBytes
payload, padded to 64 bytes for DMA alignment
```

Header coordinates are the **receiver's**. Slots are host-local (`dst.slot`). Token positions are
absolute and identical on both sides — the chunk's own `posStart`/`posEnd` pass through. That is
why a prefill/decode position relocation would have to be applied when the header is written, not
by decode inventing a different position axis.

Each sealed batch is one Mooncake op (one contiguous range inside one slab). The control plane
forwards the batch locator in a DRAIN message. Decode `validateBatch` checks magic, size, and
agreement with the write index *before* any H2D; then `writeLocFor` fans the payload out.

Protocol is `KV_MANAGER_TRANSFER_ENGINE_PROTOCOL` (`rdma` needs RoCE/IB; `tcp` otherwise). Device
IO is `dmk` (UMD + DRISC data mover) in a tt-metal build, or `mock` on a device-less host.

### 10.6 Config 0 on the TRANSFER path

The indexes are built for **every** config. `DataPlaneRequest.src.configId` nevertheless defaults
to `0`, and `DataPlaneRequestBuilder` never writes it. `doTransfer` therefore calls
`readIndex->queuesFor(0, slot, layer)`. Headers stamp `dst.configId`, also 0. Drain *could* write
another config if a header carried it; current TRANSFER never produces one.

So even a planner that walked all 36 Gemma 4 configs would still only *copy* config 0 until that
request field is filled. Treat multi-config P→D through `kv_manager` as **not yet wired**. The
source table must still enumerate every config: decode's destination table, the shmem worker, and
the index builder all expect the full contract. The gap is the migrate *command*, not the protobuf.

### 10.7 Host names and `KVM_ID`

`KVM_ID` is one string used for etcd (`kv_control/<kvm_id>`), Mooncake segments
(`mooncake/ram/<kvm_id>`), and table host tags. Convention: `prefill-0`, `prefill-1`, `decode-0`.
`PEERS` is the cross-role list (a prefill lists decodes).

`resolveLocalHost` / `localHost()` succeed only when some fabric node's `hostOf` equals this
process's `KVM_ID`. GPT-OSS and Gemma 4 currently stamp `socket.gethostname()`. File-export into
`kv_manager` therefore requires either:

- launching with `KVM_ID=$(hostname)`, or
- stamping `prefill-0` (etc.) from the table builder so it matches the fleet's `KVM_ID`.

A mismatch looks like “this host has no chunks” / empty indexes / `migration_plan_incomplete`, not
like a DRAM-address bug. Gate 1 will still pass, because Gate 1 never consults `hostOf`.

The device map is **host-local** (each rank's chips). The protobuf path is **shared storage**. That
split is the same as the shmem path; only the delivery vehicle changes.

### 10.8 What a prefill author should do with this

Keep authoring the table for the logical contract (§7). `kv_manager` is a consumer of that
protobuf, not a second layout.

Additionally, if the bring-up target is file-export:

1. Stamp fabric-node hosts to the fleet's `KVM_ID`, not an accidental hostname.
2. Put a local chip first in every replica `add_device_group` list.
3. Zero-pad config names so import ids match decode.
4. Know that a single `[0, prompt_len)` × all-layers migrate currently plans and copies **config
   0 only**, and fails closed on holes in that config. Dual-family / SWA models will need either a
   `kv_manager` change (walk every config, skip absent) or a driver that issues one migrate per
   `(config-family, extent)` until that change lands.
5. Do not wait on `SLOT_COPY`; it is advertised on HTTP and unimplemented in the data plane.

---

## 11. What happens on the decode side

Decode independently:

1. allocates destination KV;
2. builds a `KvMigrationSpec` and binds it to live decode `buffer_address()` values;
3. registers the destination table;
4. pairs with the prefill endpoint;
5. waits for handoff / DONE;
6. decodes from `prompt_len` on `dst_slot`.

Prefill Python does not participate in those steps. The decode first-principles doc covers them.

In tt-blaze *loopback*, tt-blaze itself fills a source slot and uses one endpoint for both tables.
That validates decode `locate` and transport **without** this prefill deployment. It does not
validate the packed NdShard table this document describes.

---

## 12. End-to-end disaggregated flow

```text
PREFILL DEPLOYMENT                         DECODE DEPLOYMENT

allocate packed source KV                 allocate destination KV
        |                                         |
compile                                build KvMigrationSpec
        |                                         |
  shmem: SET_TABLE + WORKER_READY         shmem: register dest table + pair
  file:  export protobuf + device map     file:  decode table on disk
        |                                         |
serve H2D chunks, write source DRAM
        |                                         |
LayerAck as each layer completes
        \                                         /
         \---- migrate(src slot, dst slot, layers, positions) ----/
                              |
         shmem worker: DRAM -> DRAM using both tables
         kv_manager:   DRAM -> host slab -> RDMA -> host slab -> DRAM
                              |
             prefill writes handoff, then DONE (shmem path)
             or the HTTP/ZMQ ack returns (kv_manager path)
                              |
                       decode resumes
```

---

## 13. Worked examples

These are source-side. Decode's chips, banks, and `locate` math are independent; only `config_idx`,
chunk bytes, and layer semantics must match.

### 13.1 GPT-OSS GQA

Eight KV heads, separate K and V, `head_dim=64`, `bfloat8_b`, TP columns = 8 (one head per chip).

```text
config  0 ..  7  ->  k_h0 .. k_h7     single-member group: TP column h
config  8 .. 15  ->  v_h0 .. v_h7     same
chunk_n_tokens   = 32
chunk_size_bytes = (64 / 32) * 1088 = 2176
protobuf names   = "00" .. "15"
```

Packed tensors, one each for K and V:

```text
[num_users * num_layers, 1, max_seq_len / sp, 64]
```

Batch index `slot * num_layers + layer`. With `sp=1` the closed-form `shard_id` walks that batch
axis then the 32-token sequence blocks, round-robin across the DRAM banks — the nested loop in
`gpt_oss_d_p/tt/runners/kv_chunk_table.py`.

Sliding-window layers keep a short extent on decode; the prefill table must author the same short
range on those layers and leave addressing linear.

### 13.2 MiniMax-M3 GQA + indexer

Four GQA groups plus a replicated indexer:

```text
config 0..3  ->  k_h0 .. k_h3
config 4..7  ->  v_h0 .. v_h3
config 8     ->  index_k          replica group: all TP columns of the SP row
```

`index_k` is absent on dense layers (skip `set`). Prefill and decode must both skip those cells.
MiniMax's list-constructor table historically used unpadded names; N=9 happens to sort correctly,
but N>10 does not — zero-pad new tables.

### 13.3 Gemma 4 dual family

Local / SWA and global layers do not share a shape, so one table carries two families. Order is the
prefill↔decode contract:

```text
config  0 .. 15  ->  k_h0  .. k_h15     local K, head_dim 256, extent 1024
config 16 .. 31  ->  v_h0  .. v_h15     local V, same
config 32 .. 35  ->  kv_h0 .. kv_h3     merged [K_roped_rotary | V], row 640, full context
```

Chunk sizes:

```text
local  = (256 / 32) * 1088 =  8 * 1088 =  8704 bytes
global = (640 / 32) * 1088 = 20 * 1088 = 21760 bytes
```

Packed tensors:

```text
local_k, local_v : [U * 60, 4, 1024 / sp, 256]     # 4 local heads per TP=4 chip
global_kv        : [U * 10, 1, S / sp, 640]        # 10 full-attention layers, compact
```

Local batch `slot * 60 + layer` (unused full-attention slots are cheap). Global batch
`slot * 10 + global_index(layer)`. Head `h` of the local family lives on chip `h // 4` at
`local_head = h % 4`. Global head `h` lives on chip `h`.

On a sliding layer, only configs 0–31 are authored, and only at positions `0..1023`. On a full
layer, only configs 32–35 are authored, over the full sequence. Default mesh `(1, 4)` / `sp_axis=0`
gives coords `(0, chip)`, matching decode's flattened 1×4 TP view at the *logical* level; the
physical `noc_addr` math is still NdShard, not FlashGQA.

That dual-family / dual-extent shape is exactly what current `kv_manager` planning cannot ingest as
one `[0, prompt_len)` × all-layers migrate: config 0 is local K, extent 1024, absent on global
layers (§10.4, §10.6). The table is still the right contract for decode and for the shmem worker.

### 13.4 DeepSeek-V3 MLA (source side of the decode example)

Decode publishes one config `"0"`, `head_dim = 512 + 64 = 576`, `chunk_size_bytes = 19584`. Prefill
must identify the merged latent as the same `config_idx=0` and expose 19,584-byte chunks.

Prefill stores that latent packed and SP-sharded:

```text
[num_users * num_layers, 1, max_seq_len / sp, 576]
```

with the kimi/DeepSeek block-cyclic table (`create_kv_chunk_address_table_kimi`). Replica group is
the full TP row for that SP shard — the same *logical* replica story as decode's
`(sp_row, col 0)` and `(sp_row, col 1)`, on a different mesh.

A sparse/DSA variant adds the index cache as `config_idx=1` in the same merged table. A DFlash
drafter appends `dflash_k_h00..` / `dflash_v_h00..` after the model's configs so they cannot
renumber `"0"`.

Pipeline-parallel DeepSeek: only the single-config KVPE path merges `stage_layout` today. A
KVPE+index merged table asserts `stage_layout is None`.

---

## 14. What crosses the prefill/decode boundary

At the migration-service level:

- source and destination chunk tables (protobuf);
- device maps (fabric node → UMD id);
- a logical migrate command: slots, layer range, position range — never `config_idx`, never chips.

How those three meet:

- **Shmem:** each side `SET_TABLE`s, endpoints pair by id + service name, `migrate()` +
  `wait_complete`.
- **`kv_manager`:** each process poll-loads its table file and device map; `POST /migrate` (or ZMQ)
  carries the same logical rectangle; peers find each other by `KVM_ID` in etcd. Completion is the
  command ack, not a Python `wait_complete`.

At the application handoff level (shmem Python driver; still useful beside `kv_manager`):

- destination slot;
- prompt length;
- last prompt token;
- completion sentinel, when configured.

What prefill Python does **not** send:

- a `ttnn.Tensor`;
- a `buffer_address()` object decode could dereference;
- a Python `KvMigrationSpec`;
- direct access to decode DRAM.

What prefill Python does **not** receive from decode: any of the symmetric things. The two tables
meet only inside the mover.

---

## 15. Invariants and common failure modes

### Wrong address math

The table points at valid but incorrect DRAM. Gate 1 (device-less `read_dram_umd`) is the cheapest
proof that `noc_addr` matches the live packed cache. Cross-check the formula against
`update_padded_kv_cache` / the NdShard walk, not against decode `locate`.

### Mismatched config order

Prefill's `config_idx=N` names a different tensor group from decode's `config_idx=N`. Bytes are
routed to the wrong cache even though both tables are individually valid. Zero-padded protobuf
names that silently reordered on import present the same way.

### Wrong chunk size

The worker reads or writes too few or too many bytes. `chunk_n_tokens` and `chunk_size_bytes` are a
cross-endpoint contract.

### Folded sliding window

`p % sw` inside the source formula sends `S/sw` copies of each ring slot and disagrees with
decode's short `per_layer_seq_len`. Author `0..sw-1` only.

### Semantic layer vs packed index

A compact global tensor packed as `slot * n_global + gi` must still `set(..., layer=semantic_id,
...)`. Decode looks up layer 11, not packed index 1.

### Hardcoded 8 DRAM banks

Harvested parts expose 7. `get_num_dram_banks(mesh_device)` is the same count the NdShard grid used
at allocate time.

### Device groups treated unique heads as replicas

The worker reads one member and writes that payload to all destinations. TP-sharded heads each
need their own config.

### Describing the wrong cache

The standalone Gemma 4 demo cache is INTERLEAVED `[batch, heads, seq, dim]`, not packed NdShard.
A table that describes it will not match `allocate_kv_cache` for the runner. Spec what this process
allocates and writes.

### Stale table or device map on disk

Rank 0 removes a leftover protobuf before rebuild. A leftover JSON map from a mock run can make a
misconfigured real run look like it reads chips. Device maps are host-local; the table is shared.

### `PREFILL_MOCK_MIGRATION` on multiple ranks

Each rank would publish a table covering only its layer slice. The runner rejects this. Real
migration (`PREFILL_ENABLE_MIGRATION`) merges stages.

### Runner entered the request loop before WORKER_READY

The scheduler can `migrate()` against an empty table. Setup (maps + `SET_TABLE` + `wait_ready`)
runs before H2D.

### Unpaired endpoints

Both tables exist, but `migrate()` cannot find the remote table. Cross-endpoint mode must
`connect_to` before the first migrate.

### Handoff after DONE, or DONE without `wait_complete`

Decode may start on a half-copied slot, or on a slot whose migrate never ran. Write handoff, then
DONE, both after completion.

### Missing producer PCC branch

Gate 1's golden check is not adapter-dispatched. A new layout needs a branch in
`prefill_producer.py` (`_read_slot_kv_and_check_pcc_*`). That path keys configs by integer id; a
protobuf reorder shows up as swapped heads, not as an import error.

### `KVM_ID` does not match `set_fabric_node_host`

`kv_manager` binds chunks to a process by the host string in the table. GPT-OSS / Gemma 4 stamp
`socket.gethostname()`. If the fleet launches as `prefill-0`, indexes install empty and planning
returns `migration_plan_incomplete`. Gate 1 still passes. Align the two strings (§10.7).

### Replica group order hid the designated reader

`kv_manager` reads only `group.front()`. A replica list that puts a remote chip first leaves this
host with nothing to TRANSFER even though it holds a copy. Single-member TP groups are safe.

### Dual-family / windowed migrate against current `kv_manager`

The planner consults config 0 only and fails closed on absent cells. A Gemma 4-style table with
local extent 1024 and global-only layers will not complete a single all-layers `[0, prompt_len)`
through `POST /migrate` until the planner walks every config and skips holes. TRANSFER also copies
only `configId == 0` (§10.4, §10.6). This is a mover limitation, not a reason to collapse families
in the source table.

### Slot-copy advertised, unimplemented

`POST /slot_copy` is on the HTTP surface. `DataPlane::doCopy` is a no-op. Do not build a resume
path that depends on it.

---

## 16. Validation layers

Different checks answer different questions. Operational recipes:
[`PREFILL_MIGRATION_TESTING.md`](PREFILL_MIGRATION_TESTING.md).

1. **Device-free addressing tests** compare the closed-form `shard_id` / `noc_addr` against an
   independent dim-0→1→2 walk, lock `config_names()` to the decode list, and prove windows do not
   wrap. No device, no ttnn required if math is import-light.
2. **Device table tests** (`gpt_oss_d_p/tests/test_kv_cache_table.py`) write a known chunk, then
   `table.lookup` + `read_device_chunk` (or `to_torch` of the live cache) must match.
3. **Gate 1 — mock migration + producer PCC.** The runner writes the protobuf and JSON map. The
   producer reads each chunk with `read_dram_umd` — the same UMD path the worker uses — and PCCs
   against golden. This isolates “does the source table match live prefill DRAM?” No endpoint, no
   worker. Single-rank.
4. **Gate 2 — prefill loopback (`dst-bytes`).** Real DRAM → transport → DRAM on the prefill
   endpoint. Source and destination slots share one table. Proves the worker copy is consistent
   with *this* table. Does not prove decode `locate`.
5. **Gate 2 `dst-golden`.** Same copy, then decode the destination the way Gate 1 decodes the
   source. Proves the copy preserved values, still on the prefill layout.
6. **Decode loopback** (tt-blaze `DECODE_MIGRATION.md`). Proves decode `locate` is self-consistent.
   Does not prove the packed source table.
7. **Cross-endpoint P→D.** The integer contract plus both address formulas. Prefill-source PCC
   (did transport preserve prefill's values?) is separate from golden PCC (did prefill compute the
   right values?) and from decode-through-migrated-KV (does generation look right?).

Transport and model correctness are separate. A migration can faithfully copy KV that prefill
computed incorrectly; conversely, prefill may be correct while a bad table corrupts transport.

---

## 17. Code map

Engine (model-agnostic):

- [`adapter.py`](../adapter.py): `PrefillModelAdapter`, `KvCaches`, `ADAPTER_PATHS`.
- [`runners/prefill_runner.py`](../runners/prefill_runner.py): cache lifetime, stage gather,
  `SET_TABLE` / `WORKER_READY`, LayerAck, request loop. Never calls `migrate()`.
- [`runners/migration.py`](../runners/migration.py): device-map delivery, stage-layout all-gather,
  protobuf serialize helpers, `publish_serialized_table_and_wait_ready`.
- [`runners/prefill_producer.py`](../runners/prefill_producer.py): H2D push, ack drain, Gate 1
  `read_dram_umd` PCC.
- [`runners/migration_driver.py`](../runners/migration_driver.py): pairing, `migrate()`,
  `wait_complete`, handoff, DONE, destination verification.

Per model:

- `models/demos/<model>/tt/attention/kv_cache.py` (or `prefill_kv_cache.py`): packed allocate.
- `models/demos/<model>/tt/runners/kv_chunk_table.py`: address math + table fill + serialize.
- `models/demos/<model>/tt/tt_prefill_runtime.py`: thin `build_kv_chunk_table` /
  `kv_migration_base_address` forwarders.

Decode (other checkout):

- `docs/kv_migration_first_principles.md`, `docs/kv_migration_spec.md`
- `blaze/models/<model>/kv_migration.py`, `entry.kv_migration_spec()`

Mover (tt-d-gen `kv_manager/`):

- `src/control_plane/maps/kv_chunk_address_table_adapter.cpp` — protobuf → per-config `IKvTable`;
  `size_bytes == 0` is absence.
- `src/control_plane/maps/kv_chunk_index_builder.cpp` — designated reader, write fan-out,
  coalescing; indexes placed by `configId()`.
- `src/control_plane/maps/kv_table_manager.cpp` — `firstLoadedTable` / `tablesFor` / geometry
  check.
- `src/control_plane/services/migration_strategy_builder.cpp` — host-pair plan; fail-closed on
  missing lookups.
- `src/control_plane/services/data_plane_request_builder.cpp` — builds a `KvRectangle` whose
  `configId` stays 0.
- `src/data_plane/data_plane.cpp` — `doTransfer` / `doDrain` / `rdmaTransfer`; `doCopy` stub.
- `include/data_plane/data_plane.hpp` — `BatchKvChunkHeader` (`KVCH`), three-hop batch layout.
- `include/control_plane/command/types.hpp` — `MigrationCommand` (no `config_idx`).

---

## 18. Model-author recipe

Once the concepts above are clear, the mechanical list:

1. Read the decode spec. Copy ordered `config_names`, per-config row/dtype/bytes, layer ownership,
   and window extents. If decode has no spec yet, invent the decomposition once and land it in both
   trees.
2. Allocate packed NdShard tensors that the table will describe. Two families ⇒ two tensors. Cheap
   unused slots may stay in a full layer axis; expensive ones must not.
3. Implement closed-form `shard_id` (batch → head → seq block) and SP block-cyclic `local_pos`.
   `sp=1` is identity. Do not fold `p % sw`.
4. Build with zero-padded names. Assert `table.config_name(i)`.
5. `set` only cells this config owns; semantic layer ids; windowed configs stop at `sw`.
6. Single-member device groups for TP-sharded heads; full-row groups for replicas, with a local
   chip first in the list. Real mesh coordinates, not decode's flattened tuple. Stamp
   `set_fabric_node_host` to the fleet `KVM_ID` if file-exporting to `kv_manager`.
7. Device-free contract tests. Then Gate 1. Then Gate 2. Then cross-endpoint. File-export is a
   separate bring-up: it does not replace Gate 1, and current `kv_manager` will not move a
   dual-family / windowed table in one migrate (§10).
8. Runtime hooks when serving is wired. Do not register `ADAPTER_PATHS` until `prefill_chunk`
   exists. A table + tests may land first.

---

## 19. One-sentence summary

Prefill turns a logical source KV coordinate into a real packed-DRAM location, publishes that
source table, and asks a mover — the shmem worker, or tt-d-gen `kv_manager` via three host hops —
to copy each chunk to the independently authored decode destination table; the two sides share
`config_idx`, chunk bytes, and layer semantics, not chips, banks, or address formulas.
