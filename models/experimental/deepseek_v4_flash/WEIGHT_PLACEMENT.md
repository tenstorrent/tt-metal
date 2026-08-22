# L1-Resident Weight Placement: Two Layers Per Chip

Plan for keeping **two decoder layers' non-expert weights permanently resident in
L1** on a single 120-core chip, replacing the DRISC prefetcher / GlobalCircularBuffer
path for those weights. The placement is implemented as data in
[`tt/l1_placement.py`](tt/l1_placement.py); this document explains where the numbers
come from and what constraints shaped them.

## Assumptions

| | |
| --- | --- |
| Chip | 120 Tensix cores, 12 x 10 grid, 1.5 MB (1536 KB) L1 per core, 180 MB total |
| Weight dtype | `bfloat4_b`: 32x32 tiles of **576 B** (all dims here are tile-aligned, so no padding loss) |
| Resident set | Everything in `DECODE_LAYOUTS` plus the router gate, mHC `fn` matrices and norms — for 2 layers |
| NOT resident | Routed experts (3.62 GB/layer, stay DRAM ND-sharded for `fused_experts`); Lightning Indexer (never loaded by `tt/`) |
| Small tensors | Norms / position bias / sinks stay `bfloat16`; router gate and mHC `fn` are **cast to bf4** (bf16 today — keeping them bf16 adds ~2.1 MB/layer and still fits) |
| Prefetcher | **Dropped entirely** (see below) |

Per-layer totals at bf4: **75.25 MB** (HCA layer), **77.50 MB** (CSA layer);
two layers = **152.75 MB** of the 180 MB.

## Why the GCB must go

The shared decode GCB is 16 pages x 18,432 B = **288 KB on each of its 64 receiver
cores**. Under this plan those cores already carry ~1.3 MB of resident weight, so ring
+ weights = ~1.6 MB > 1.5 MB. Residency and streaming cannot coexist on the same
cores. Dropping the GCB also removes:

* the unchecked FIFO ordering contract across all ten weights (`DECODE_GCB_GROUP`),
  whose violation is silent wrong results;
* the whole `prefetcher_session()` machinery in `tt/model.py` (start/stop/fence,
  force-stop unwinding) — nothing else queues prefetch requests;
* the `prefetch_weights()` hoisting chain through model / decoder_layer / attention / moe;
* the DRISC senders' state-zone budget (~6 GCBs/device cap).

With no ring, `fuse_qa_kv_proj` becomes legal again (it was rejected only because the
1536-wide fused weight shares no page size with the ring). The fused weight costs the
same L1 as the two it replaces (54 KB/core) and saves one dispatch per layer.

## Constraints that shape the placement

1. **Core counts are powers of two.** Every K and N is a power of two and
   `k_blocks`/`n_blocks` must divide them, so a weight occupies 8/16/32/64 cores —
   never 120. No single weight can cover more than 64 cores (128 > 120).
2. **Contiguous core ranges.** Each weight lives on one contiguous run of row-major
   core indices. Any two 64-core windows in 120 cores overlap by >= 8 cores, so there
   is effectively **one** 64-wide zone; the rest of the chip splits into 32/16/8-wide
   zones: `120 = 64 + 32 + 16 + 8`.
3. **Symmetry across layers.** Both layers' copies of a weight use the *same zone and
   shard shape*. Every layer then has identical program configs, memory configs and
   trace shape — preserving the layer-independence that `DECODE_LAYOUTS` provides
   today — and no layer is systematically slower than the other.

Consequence of (1)+(2): at most 96 MB of weight can sit at 64-wide; since the three
big projections (`q_b`, `o_a`, `o_b`) are 2 x 18 MB per pair, only two pairs fit the
64-wide zone. `o_a_proj` is the pair demoted to 32 cores.

## Zone map

Core index `c` maps to grid coordinate `(c % 12, c // 12)`. Zones are contiguous in
row-major index and decompose into <= 3 rectangular `CoreRange`s each.

```
        x0  x1  x2  x3  x4  x5  x6  x7  x8  x9 x10 x11
  y0   |------------------------ Z0 ------------------|
  y1   |------------------------ Z0 ------------------|
  y2   |------------------------ Z0 ------------------|
  y3   |------------------------ Z0 ------------------|
  y4   |------------------------ Z0 ------------------|
  y5   |---- Z0 ----|-------------- Z1 ---------------|
  y6   |------------------------ Z1 ------------------|
  y7   |------------------------ Z1 ------------------|
  y8   |------------------------ Z2 ------------------|
  y9   |---- Z2 ----|-------------- Z3 ---------------|
```

| Zone | Cores (row-major) | Count | Purpose |
| --- | --- | --- | --- |
| Z0 | 0-63 = (0,0)-(3,5) | 64 | The only 64-wide zone: `q_b`, `o_b`, small attention projections |
| Z1 | 64-95 = (4,5)-(11,7) | 32 | `o_a` (batched), compressor kv, mHC `fn` |
| Z2 | 96-111 = (0,8)-(3,9) | 16 | Shared expert gate/up, attention `kv_proj` |
| Z3 | 112-119 = (4,9)-(11,9) | 8 | Shared expert down, router gate |

## Placement (identical for both layers)

Shard shapes are per-core `[rows, cols]`; "blocks" uses `decode_weight_layout`
terminology (`n=` full width-shard, `k=/n=` partial, `b=/n=` batched).

### Z0 — 64 cores — 1310 KB/core with both layers

| Weight | [K, N] | Blocks | Shard | KB/core/layer |
| --- | --- | --- | --- | --- |
| `q_b_proj` | [1024, 32768] | n=64 | [1024, 512] | 288 |
| `o_b_proj` | [8192, 4096] | n=64 | [8192, 64] | 288 |
| `q_a_proj` | [4096, 1024] | k=2, n=32 | [2048, 32] | 36 |
| `compressor.gate_proj` (CSA) | [4096, 1024] | k=2, n=32 | [2048, 32] | 36 |
| `compressor.gate_proj` (HCA) | [4096, 512] | k=4, n=16 | [1024, 32] | 18 |
| norms / biases (bf16) | — | — | — | 16 |

### Z1 — 32 cores — 1296 KB/core

| Weight | [K, N] | Blocks | Shard | KB/core/layer |
| --- | --- | --- | --- | --- |
| `o_a_proj` | [4096, 1024] x8 groups | b=8, n=4 | [4096, 256] | 576 |
| `compressor.kv_proj` (CSA) | [4096, 1024] | n=32 | [4096, 32] | 72 |
| `compressor.kv_proj` (HCA) | [4096, 512] | k=2, n=16 | [2048, 32] | 36 |
| `attn_hc.fn` | [16384, 32] | k=32, n=1 | [512, 32] | 9 |
| `ffn_hc.fn` | [16384, 32] | k=32, n=1 | [512, 32] | 9 |

### Z2 — 16 cores — 1296 KB/core

| Weight | [K, N] | Blocks | Shard | KB/core/layer |
| --- | --- | --- | --- | --- |
| `shared_gate_proj` | [4096, 2048] | n=16 | [4096, 128] | 288 |
| `shared_up_proj` | [4096, 2048] | n=16 | [4096, 128] | 288 |
| `kv_proj` | [4096, 512] | n=16 | [4096, 32] | 72 |

### Z3 — 8 cores — 1296 KB/core

| Weight | [K, N] | Blocks | Shard | KB/core/layer |
| --- | --- | --- | --- | --- |
| `shared_down_proj` | [2048, 4096] | n=8 | [2048, 512] | 576 |
| `router_gate` | [4096, 256] | n=8 | [4096, 32] | 72 |

## Budget

| Zone | Cores | Weights (2 layers) | KB/core | Free/core |
| --- | --- | --- | --- | --- |
| Z0 | 64 | 81.9 MB | 1310 | 226 KB |
| Z1 | 32 | 40.5 MB | 1296 | 240 KB |
| Z2 | 16 | 20.3 MB | 1296 | 240 KB |
| Z3 | 8 | 10.1 MB | 1296 | 240 KB |
| **Total** | **120** | **152.75 MB / 180 MB** | | **>= 226 KB/core** |

The HCA+CSA pairing above is the common case (layers alternate). The worst pairing —
two CSA layers — adds 18 KB/core on Z0 and 36 KB/core on Z1 and still fits.
`tt/l1_placement.py::budget_report()` checks any pairing.

The ~226 KB/core of headroom must hold every op's circular buffers and activations,
including the `fused_experts` activation block for the (DRAM-streamed) routed
experts. Budget for `experts_block_size: 1` (~68 KB/core) and keep
`sdpa_max_cores_per_head_batch: 2`.

## One fused, height-sharded tensor

The L1 allocator reserves the same address range on every core, so per-weight
tensors on zone sub-grids would waste the reserved range on every core outside the
zone. All resident bf4 weights are therefore packed into **one tensor**, HEIGHT
sharded across **all 120 cores** with one equal shard per core
(implemented in [`tt/l1_weights.py`](tt/l1_weights.py)):

* A core's shard is the concatenation of the **tile streams** of every slab the
  placement assigns to that core — a slab's 32x32 tiles in row-major order, which
  is exactly the order the width-sharded and prefetched paths deliver today — in
  `WEIGHT_ORDER` (the decode consumption order), layer 0 before layer 1.
* Shards are **zero-padded to the largest zone's size** so every core carries the
  same allocation. For the typical HCA+CSA pair that is **2304 tiles = 1296 KB per
  core** (only Z0 pads, by 32 tiles); the CSA+CSA worst case is 2368 tiles
  (1332 KB).
* The shard is one tile wide: `[shard_tiles * 32, 32]`, full tensor
  `[120 * shard_tiles * 32, 32]`, `ShardOrientation.ROW_MAJOR` over the
  `(0,0)-(11,9)` grid so host shard `i` lands on grid core `i`.
* `l1_weights.shard_layout()` gives every `(layer, weight)` its tile offset inside
  the shard — identical on every core of the weight's zone. A consumer's in1 is the
  region `[tile_offset * 32, (tile_offset + num_tiles) * 32)` of its core's shard.
* The bf16 small tensors (norms, biases, position bias) are **not** in the fused
  tensor (one tensor, one dtype); they remain small separate allocations inside the
  16 KB/core/layer reserve.

Shard composition per zone (typical HCA+CSA pair; offsets in tiles from
`shard_layout()`):

| Zone | Layer 0 regions | Layer 1 regions | Used / padded |
| --- | --- | --- | --- |
| Z0 | `q_a` @0+64, `q_b` @64+512, `comp.gate` @576+32, `o_b` @608+512 | same sequence @1120 (comp.gate 64 tiles for CSA) | 2272 / 32 |
| Z1 | `comp.kv` @0+64, `o_a` @64+1024, `attn_hc.fn` @1088+16, `ffn_hc.fn` @1104+16 | same @1120 (comp.kv 128 for CSA) | 2304 / 0 |
| Z2 | `kv` @0+128, `shared_gate` @128+512, `shared_up` @640+512 | same @1152 | 2304 / 0 |
| Z3 | `shared_down` @0+1024, `router_gate` @1024+128 | same @1152 | 2304 / 0 |

## Code changes required

1. Load the fused tensor once per chip with
   `l1_weights.build_l1_weight_tensor(weights_by_layer, device)`; it returns the
   height-sharded bf4 tensor and the `ShardLayout` with per-weight tile offsets.
2. `matmul_decode` (and the batched variant) must accept its in1 as a *region of
   the resident shard* — core-local base address plus the layout's tile offset —
   instead of a per-weight width-sharded tensor or a GCB pop. The tile order inside
   a region is unchanged from today's slab order, so only the addressing changes.
3. `use_prefetcher=False` everywhere; delete the `prefetcher_session()` wrapping and
   `prefetch_weights()` calls.
4. The tile cache must be regenerated: one fused cache file per chip
   (`cache_file_name` on `build_l1_weight_tensor`) replaces the per-weight
   `*_prefetch` files.
5. Activation resharding follows the weight's zone: `matmul_decode` puts its output
   on the weight's cores (first `n_blocks` receivers in partial mode), so the
   layer's dataflow hops Z0 -> Z1 -> Z2/Z3 within a step. The input gather configs
   (`get_input_memory_config`) must be built against each zone's cores.

## Known costs and risks

* `o_a_proj` runs at 32-wide (was 64), the shared expert at 16/8-wide for **both**
  layers — that is the price of symmetry on one 64-wide zone. If
  `shared_down_proj` (8 cores) shows up in profiles, moving its pair to Z1 raises Z1
  to 1440 KB/core, leaving only 96 KB for CBs — likely a collision with
  `fused_experts`; not recommended as a starting point.
* This is a **2-layers-per-chip** design. On p150x8 (~5.4 layers/chip at PGS=1) the
  remaining ~3 layers per chip would fall back to per-call DRAM->L1 copies with no
  GCB to hide them. The layout is natural for a 32-chip Galaxy (43 layers / 32 chips
  < 2 layers/chip), where the DRAM weight path disappears from decode entirely.
* Zones are contiguous in row-major core index, not rectangles (64 cores cannot form
  a rectangle on a 12x10 grid); each zone is a `CoreRangeSet` of <= 3 ranges.
