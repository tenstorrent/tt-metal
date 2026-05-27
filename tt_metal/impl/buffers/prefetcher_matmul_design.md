# Matmul + DRAM Prefetcher Design

This document describes the contract between the gather-in0 matmul receiver
and the two DRAM prefetcher implementations that feed it: the worker-core
prefetcher (`ttnn.dram_prefetcher`) and the DRAM-core prefetcher
(`ttnn.experimental.start_dram_core_prefetcher`). It exists so that a future maintainer
touching either side can read one file and learn what is invariant across the
two paths, without having to reverse-engineer the kernels.

If you change the kernel structure or the per-tensor RTA layout in either
prefetcher — or the receiver matmul's wait/pop contract — update this doc in
the same commit. The implementation includes a sync-check pass at the end of
the DRAM-core prefetcher refactor specifically to catch drift.

## 1. Overview

The prefetcher streams weight tensors (and similarly-shaped activation
helpers) from DRAM into a ring of worker cores running a gather-in0 matmul.
The matmul kernel reads its in0 (activations) from L1 and its in1 (weights)
from a global circular buffer (GCB); the prefetcher's job is to keep that GCB
full so the matmul never stalls on a DRAM read.

Two prefetcher variants exist:

- **Worker-core** (`ttnn.dram_prefetcher`, `ttnn/cpp/ttnn/operations/prefetcher/`):
  runs a pair of kernels (BRISC reader + NCRISC writer) on a worker tile next
  to each DRAM bank, with a triple-buffered local CB sitting in worker L1.
  Most flexible (handles heterogeneous shapes/dtypes per launch), and has
  ~1.5 MB of worker L1 to play with for the local CB.
- **DRAM-core** (`ttnn.experimental.start_dram_core_prefetcher`, `tt_metal/distributed/`):
  runs a single kernel on a DRISC core (i.e. on the DRAM core itself), doing
  GDDR DMA and NoC push from the same RISC. Closer to the GDDR controller and
  can run at higher throughput on production shapes, but its DRISC L1 working
  region is only ~70 KB total, which constrains the staging strategy.

Both feed the same receiver: worker cores running
`bmm_large_block_zm_fused_bias_activation_gathered.cpp` (compute) +
`reader_bmm_tile_layout_in1_ring_all_gather.cpp` (in1 reader). Receivers see
the same byte layout regardless of which prefetcher produced it.

Production usage on Llama-3.1-8B (single Blackhole, ring=64): the DRAM-core
path covers FF1, QKV, and O matmuls; FF2 (K=14336) uses a separate
DRAM-sharded matmul without a prefetcher.

## 2. The ring

Each tensor is *width-sharded* across `num_senders` DRAM banks, then each bank
fans out to `num_receivers_per_sender` worker-core matmul receivers in a
ring. The ring size `ring_size = num_senders * num_receivers_per_sender` is
fixed for the launch.

```
senders (DRAM banks, num_senders=S):
  +---+---+---+---+---+---+---+---+
  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     each holds N/S cols of every K-row
  +---+---+---+---+---+---+---+---+
     |       |       |       |
     +--->+--->+--->+--->+--->...     (round-robin gather)
     v       v       v       v
receivers (worker cores, ring_size = S * recv_per_sender):
  [r0][r1][r2]...[r_{ring-1}]         each gets one block per ring position
                                       per layer per tensor
```

Each sender has its own GCB sender interface; each receiver has its own GCB
receiver interface. The GCB topology (which senders feed which receivers) is
fixed at GCB creation time and visible to the prefetcher via
`GlobalCircularBuffer::sender_receiver_core_mapping()`.

## 3. What is a "block"

A **block** is the per-ring-position quantum of one tensor's data. There are
`num_blocks = ring_size` blocks per layer per tensor, and one block lands on
one receiver per layer per tensor.

```
Tensor (K rows x N cols), tiled (each cell = 32x32 tiles):

        N-cols, sharded across S DRAM banks
        <------ n_per_bank tiles per bank ------>
        +-----------+-----------+-----------+...
k_tiles |  bank 0   |  bank 1   |  bank 2   |   <- K-row 0
rows    |           |           |           |
        +-----------+-----------+-----------+
        |           |           |           |   <- K-row 1
        +-----------+-----------+-----------+
        ...        (k_tiles K-rows total)

Split K into ring_size strips:
        K-block 0:  rows 0  .. kw-1
        K-block 1:  rows kw .. 2kw-1
        ...        (kw = k_tiles / ring_size = "k_block_w_tiles")

One block (e.g. ring position p, bank b):
        +-----------+
        | kw x n/S  |   <- what sender b pushes to receiver p
        +-----------+      once per layer per tensor
```

Definitions:

- `k_tiles = K / TILE_HEIGHT` (32).
- `n_per_bank_tiles = (N / num_senders) / TILE_WIDTH`.
- `k_block_w_tiles = k_tiles / num_blocks` (K-rows per block).
- `block_height_in_tiles = k_block_w_tiles` (synonym used by the worker-core
  factory and the receiver matmul).
- `n_per_recv_tiles = n_per_bank_tiles / num_receivers_per_sender`.
- `tile_bytes = tt::tile_size(dtype)` — e.g. 2048 for bf16, 1088 for bf8_b.
- `coalesced_page_size = n_per_recv_tiles * tile_bytes` (bytes per K-row per
  receiver). With multi-tile receivers, `coalesced_num_pages` splits this
  width further; for ring=64 production shapes it's 1.

The on-receiver layout of one block is `block_height_in_tiles` contiguous
K-rows, each `coalesced_num_pages * coalesced_page_size` bytes wide.

### Per-block source tiles

For block index `blk` (in `[0, num_blocks)`) pushed by sender bank `b` to its
receiver with local index `r` (in `[0, num_receivers_per_sender)`), the page
bytes correspond to the global-tensor tile range

- K-rows: `[blk * k_block_w_tiles, (blk + 1) * k_block_w_tiles)`
- N-cols: `[b * n_per_bank_tiles + r * n_per_recv_tiles, b * n_per_bank_tiles + (r + 1) * n_per_recv_tiles)`

laid out K-row-major: row `h` of the receiver page contains tiles
`(blk * k_block_w_tiles + h, b * n_per_bank_tiles + r * n_per_recv_tiles + n)`
for `n` in `[0, n_per_recv_tiles)`, contiguous in K-row order.

This holds verbatim for both prefetcher variants and across the fit-ladder
sub-band/M-chunk paths — the receiver's per-block byte layout is
sub-band/chunk-order invariant (see §6 for the DRAM-core proof). The
DRAM-prefetcher validator
(`tests/tt_metal/tt_metal/test_kernels/misc/gcb_validator_receiver.cpp`)
relies on this mapping to derive expected bytes from the source tensor via
`TensorAccessor`.

## 4. The receiver matmul contract

The receiver matmul reader is
`ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp`.
Its in1 input arrives via the GCB:

```c++
constexpr uint32_t num_blocks = get_compile_time_arg_val(5);   // = ring_size

for (uint32_t b = 0; b < batch; ++b) {
  experimental::remote_cb_wait_front(remote_cb_id, num_blocks);   // wait for ring_size pages
  /* iterate ring positions, multiply, accumulate */
  experimental::remote_cb_pop_front(remote_cb_id, num_blocks);
}
```

So **the prefetcher must push exactly `num_blocks = ring_size` pages per
receiver per layer per tensor**, each page being the on-receiver block layout
described in §3. The receiver doesn't care how those bytes were assembled —
multiple NoC writes from the prefetcher are fine, as long as:

- the page bytes are contiguous in the GCB slot before the receiver wakes up;
- `pages_sent` is bumped exactly once per page per receiver, by
  `fifo_page_size / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE`.

`fifo_page_size` (the receiver's notion of one page) is set by the sender via
`experimental::resize_remote_sender_cb_interface(...)` and equals
`block_height_in_tiles * coalesced_num_pages * coalesced_page_size`.

## 5. Worker-core prefetcher

### Architecture

```
For each DRAM bank b in [0, num_senders):
  one worker core, two kernels:
    BRISC  -> reader_dram.cpp        (DRAM -> reader_cb local CB)
    NCRISC -> writer_l1.cpp          (reader_cb -> GCB remote write)
  reader_cb sits in worker L1, triple-buffered (3 x max_block_size).
```

The reader does page-sized NoC reads from DRAM and pushes to the local
reader_cb. The writer waits on reader_cb, reserves one remote-CB page on every
receiver, then issues `remote_cb_push_back_and_write_pages` which does the
multi-receiver multi-row contiguous write + the `pages_sent` increment in one
call.

### Per-tensor derivations (program factory)

`ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp`:

- `num_blocks = num_readers * num_receivers_per_reader` (= ring_size).
- Per tensor, `t`:
  - `height_in_tiles = round_up(shard[0] / TILE_H, num_blocks)` — round-up
    handles tensors whose K doesn't divide ring_size; no FATAL.
  - `tensor_block_num_tiles[t] = height_in_tiles * width_in_tiles / num_blocks`.
  - `tensor_tile_size[t] = tile_size(dtype[t])`.
  - `(page_size[t], block_num_pages[t]) = get_max_page_size_and_num_pages(8192,
    block_num_tiles[t], tile_size[t])` — picks the largest `page_size <= 8 KB`
    that's a multiple of `tile_size` and divides `block_num_tiles * tile_size`.
  - `(coalesced_page_size[t], coalesced_num_pages[t]) =
    get_max_page_size_and_num_pages(8192, width_in_tiles / num_receivers,
    tile_size[t])`.
  - `block_height_in_tiles[t] = height_in_tiles / num_blocks`.

These per-tensor values flow through RTAs to the reader and writer kernels;
the compile-time CB sizing uses worst-case across all tensors
(`max_block_tiles`, `max_tile_size`). The reader CB is sized
`max_block_size_per_reader_core * 3` for triple-buffering.

### Why it handles arbitrary shapes / dtypes

Everything that varies per tensor is per-tensor in the RTAs: page size, num
pages, coalesced page size, num coalesced pages per row, num rows, tile size.
Heterogeneous tensors in one launch each get their own values. The only
cross-tensor invariant is `num_blocks` (= ring_size), enforced because it's a
single compile-time arg.

### Why staging budget isn't a problem

Worker L1 has ~1.5 MB of CB space, so even a `3 * max_block_size` reader CB
fits comfortably for production shapes.

## 6. DRAM-core prefetcher

### Architecture

```
For each DRAM bank b in [0, num_senders):
  one DRISC core, one kernel: dram_core_prefetcher.cpp
    do GDDR DMA into a ping-pong stage in DRISC L1
    NoC push from stage to GCB remote
  no local reader_cb; stage IS the working buffer.
```

The kernel uses `experimental::dma_async_read` (defined in
`tt_metal/hw/inc/experimental/gddr_dma.h`) for GDDR reads and the GCB sender
interface for NoC pushes. Both share the DRISC's working L1 region (see §6.5).

### Two purpose-built helpers (kernel-private)

The DRISC kernel does **not** call
`experimental::remote_cb_push_back_and_write_pages`. That API conflates data
movement and bookkeeping (it bumps `pages_sent` and the receiver semaphore as
a side effect of every call), which makes "write a partial page and finalize
later" impossible without abusing internal state. The DRAM-core kernel
instead defines two helpers, each doing one thing:

```c++
// Writes one sub-band chunk from `src_l1_addr` to `dest_l1_base` on the given
// receiver subset. No pages_sent, no semaphore, no iface.fifo_wr_ptr advance.
// Layout matches the receiver-side block layout described in section 4.
FORCE_INLINE void prefetcher_write_chunk(
    uint32_t src_l1_addr,
    uint32_t dest_l1_base,                       // start of this page on the receiver
    const volatile uint32_t* recv_xy,            // noc_x/noc_y for chunk's receivers
    uint32_t num_receivers_in_chunk,
    uint32_t num_rows,
    uint32_t coalesced_num_pages_per_row,
    uint32_t coalesced_page_size,
    uint8_t noc);

// Bumps local pages_sent + remote NoC semaphore for all receivers by one
// page (with wrap-gap adjustment), advances iface.fifo_wr_ptr by one page.
// Called once per block, after all chunks of that block have been written
// via prefetcher_write_chunk.
template <bool skip_ptr_update>
FORCE_INLINE void prefetcher_finalize_block(
    RemoteSenderCBInterface& iface,
    uint32_t page_bytes_per_recv,                // bytes per receiver per block
    uint32_t num_receivers,
    uint8_t noc);
```

### L1 budget

DRISC L1 total: 128 KB. The relevant slice for the prefetcher kernel:

```
[UNRESERVED, UNRESERVED + kGcbZoneSize)   GCB pages_sent zone (1 KB)
[UNRESERVED + kGcbZoneSize, END)          kernel_working_region    (~92 KB)

  kernel_working_region:
    +--- noc_xy table   (2 * 4 * num_receivers bytes)
    +--- config struct  (16 B)
    +--- alignment slack
    +--- stage ring     (the rest; split into two halves for ping-pong)
```

See `tt_metal/impl/buffers/drisc_l1_arena.hpp` (specifically
`kernel_working_region_base()` / `kernel_working_region_size()`). On
Blackhole the kernel-region-minus-overhead works out to ~70 KB of stage
budget; `ring_half = stage_budget / 2 ≈ 35 KB`.

This is why the DRAM-core path can't pre-allocate a full block (let alone 3
for triple-buffering): on production Llama shapes a single block is 100+ KB.
The stage is sized to fit a *sub-band* of a block, not the whole block.

### Fitting algorithm (per tensor)

Per `compute_tensor_geom`, run this ladder to pick `(rows_per_sub, M)`:

1. If `k_block_w_tiles * n_per_bank * tile_bytes <= ring_half`
   -> `rows_per_sub = k_block_w_tiles`, `M = 1`. Single push per block
   (matches the kernel's behavior pre-refactor for shapes that fit).
2. Else if `n_per_bank * tile_bytes <= ring_half` (one K-row fits)
   -> pick the largest `rows_per_sub <= k_block_w_tiles` such that the
   sub-band fits `ring_half` and is a clean factor; `M = 1`. K-sub only.
3. Else `rows_per_sub = 1`; pick the smallest `M | num_receivers_per_sender`
   such that `(n_per_bank / M) * tile_bytes <= ring_half`. K-sub plus
   N-chunking. M-chunking with `rows_per_sub = 1` reads a contiguous N-stripe
   of a single K-row, which is bit-exact today's M>1 behavior.
4. Else TT_FATAL with shape, budget, and a hint to grow the ring.

The combination `(rows_per_sub > 1) and (M > 1)` is not supported and never
chosen: it would need a row-strided DMA (multiple K-rows interleaved with
multiple N-stripes), which the DMA engine doesn't expose. The ladder always
collapses to `rows_per_sub = 1` before allowing `M > 1`.

### Per-tensor derivations

Computed by `compute_tensor_geom` in `dram_core_prefetcher_manager.cpp` (the
DRAM-core analogue of the worker-core's per-tensor derivations in §5):

- `(coalesced_page_size[t], coalesced_num_pages[t])` — same `pick_page_size`
  algorithm as worker-core's `get_max_page_size_and_num_pages` (file-local
  copy in the manager; algorithm identical but kept duplicated by design —
  no shared header).
- `rows_per_sub[t]`, `num_sub[t]`, `M[t]`, `sub_chunk_bytes[t]` —
  DRAM-core-specific. Derived from the fit ladder above.
- `sub_stride_bytes[t] = rows_per_sub × n_per_bank × tile_bytes` — GDDR
  byte stride between sub-bands within a block.
- `block_stride_bytes[t] = k_block_w_tiles × n_per_bank × tile_bytes` —
  GDDR byte stride between ring-blocks.
- `page_bytes_per_recv[t] = k_block_w_tiles × coalesced_num_pages ×
  coalesced_page_size` — total bytes pushed per receiver per block (the
  `resize_remote_sender_cb_interface` page size, and the argument to
  `prefetcher_finalize_block`).

### Compile-time args (13)

```
0  num_layers
1  num_tensors
2  num_blocks                          (= ring_size)
3  num_receivers                       (= num_receivers_per_sender)
4  stage_ring_base                     (L1 addr; split into two halves by kernel)
5  stage_ring_size                     (bytes; 2 x ring_half)
6  remote_cb_id                        (= 31)
7  pages_sent_l1_addr                  (DRISC-local pages_sent slots)
8  noc_xy_l1_addr                      (per-receiver noc xy table)
9  config_l1_addr                      (4-uint32 iface config block)
10 fifo_size_per_receiver              (= gcb.size())
11 receiver_buffer_address             (GCB receiver-side fifo base)
12 remote_pages_sent_worker_l1_addr    (per-receiver remote semaphore base)
```

Ring depth is fixed at 2 in the kernel (ping-pong slots derived from
`stage_ring_base` and `stage_ring_base + stage_ring_size / 2`).

Notably absent vs the pre-refactor kernel: `max_chunk_size`, `stage_a_addr`,
`stage_b_addr`, `num_dma_chunks_per_block (M)`, `k_block_w_tiles`. `M` and
`rows_per_sub` are per-tensor RTAs now; the kernel no longer needs a single
`k_block_w_tiles` constant.

### Runtime args

```
[0]                        bank_id
[1 .. 1+num_tensors)       bank_local_base[t]       (GDDR bank-local offset)
[+num_tensors]             num_sub[t]
[+num_tensors]             M[t]
[+num_tensors]             rows_per_sub[t]
[+num_tensors]             coalesced_page_size[t]
[+num_tensors]             coalesced_num_pages[t]
[+num_tensors]             sub_chunk_bytes[t]       (rows_per_sub*n_per_bank/M*tile_bytes)
[+num_tensors]             sub_stride_bytes[t]
[+num_tensors]             block_stride_bytes[t]
[+num_tensors]             page_bytes_per_recv[t]
[2*num_receivers]          noc_x/noc_y per receiver
```

### Kernel main loop

```
setup_iface_once();                                       // noc_xy table, config, iface
issue first DMA into stage_ring[0];                       // prologue for tensor 0
for layer in [0, num_layers):
  for t in [0, num_tensors):
    pull per-tensor RTAs into locals;

    for c in [0, num_blocks * num_sub[t] * M[t]):
      blk = c / (num_sub[t] * M[t]);
      sb  = (c % (num_sub[t] * M[t])) / M[t];
      ch  = c % M[t];

      // On block boundary: reserve a fifo page on every receiver, snapshot wr_ptr.
      if (sb == 0 and ch == 0):
        experimental::remote_cb_reserve_back(remote_cb_id, 1);
        fifo_snapshot   = iface.fifo_wr_ptr;
        cum_offset_in_page = 0;

      // Issue NEXT DMA (depth=2 ping-pong) before waiting on the current chunk.
      if (c + 1 < total_chunks):
        compute (next_blk, next_sb, next_ch) and next_src;
        experimental::dma_async_read(0, next_src, stage_ring[(c+1)&1], sub_chunk_bytes[t]);
      experimental::dma_async_read_wait_n(0, has_next ? 1 : 0);

      // Write this chunk to its receiver subset at (fifo_snapshot + cum_offset_in_page).
      prefetcher_write_chunk(
        /*src=*/   stage_ring[c & 1],
        /*dest=*/  fifo_snapshot + cum_offset_in_page,
        /*recv=*/  noc_xy_ptr + ch * recv_per_chunk * 2,
        recv_per_chunk,
        rows_per_sub[t],
        coalesced_num_pages[t],
        coalesced_page_size[t],
        noc_index);

      // After the last chunk of a sub-band, advance dest offset by the sub-band's
      // per-receiver bytes (each receiver in the subset got rows_per_sub K-rows of
      // coalesced_num_pages * coalesced_page_size bytes).
      if (ch + 1 == M[t]):
        cum_offset_in_page += rows_per_sub[t] * coalesced_num_pages[t]
                            * coalesced_page_size[t];

      // On block boundary: flush posted writes and finalize (one pages_sent inc per receiver).
      if (sb + 1 == num_sub[t] and ch + 1 == M[t]):
        noc_async_posted_writes_flushed();
        prefetcher_finalize_block<skip_ptr_update=true>(
          iface, page_bytes_per_recv[t], num_receivers, noc_index);

    if (t == num_tensors - 1):
      experimental::remote_cb_sender_barrier(remote_cb_id);

experimental::update_remote_cb_config_in_l1(remote_cb_id);
noc_async_atomic_barrier();
experimental::drisc_set_noc2axi_mode();
```

The finalize uses `skip_ptr_update=true` (posted NoC semaphore writes); the
end-of-stream `update_remote_cb_config_in_l1` + `noc_async_atomic_barrier`
drain them. `enable_performance_mode` in `DramCorePrefetcherConfig` is
currently a no-op (this fast-path is always selected); the flag is kept as
an API hook for future tuning, e.g. wiring a `<false>` instantiation for
sub-band-grained ptr updates when measured to matter.

When `(rows_per_sub, M) = (k_block_w_tiles, 1)` (fast path), the inner loop
collapses to a single `prefetcher_write_chunk` followed by
`prefetcher_finalize_block` — same number of NoC writes and the same one
`pages_sent` increment per block per receiver as the pre-refactor kernel.

## 7. Llama-3.1-8B fit table

Production ring=64 (8 banks x 8 recv/sender), bf8_b (tile_bytes=1088),
stage_budget ≈ 70 KB so `ring_half ≈ 35 KB`:

| Op  | K     | N     | k_tiles | k_block_w | n_per_bank | full block | one K-row | (rows_per_sub, M) |
|-----|-------|-------|---------|-----------|------------|-----------:|----------:|-------------------|
| FF1 | 4096  | 14336 | 128     | 2         | 56         | 121 856    | 60 928    | (1, 2)            |
| QKV | 4096  | 12288 | 128     | 2         | 48         | 104 448    | 52 224    | (1, 2)            |
| O   | 4096  | 4096  | 128     | 2         | 16         |  34 816    | 17 408    | (2, 1) — fast path|
| FF2 | 14336 | 4096  | 448     | 7         | 16         | 121 856    | 17 408    | not used (DRAM-sharded variant) |

Notes:

- For FF1 / QKV even one K-row exceeds `ring_half`, so the fit ladder lands at
  step 3 (`rows_per_sub = 1`, `M = 2`). 60 928 / 2 = 30 464 ≤ 35 008.
- For O the full block just fits (34 816 ≤ 35 008), so we use the fast path.
- FF2 isn't fed by this prefetcher in production, but if it were, the ladder
  would pick `(rows_per_sub = 2, M = 1)`.

## 8. Cross-component invariants

Whoever changes prefetcher or receiver code must preserve these:

1. **Per-tensor page bytes match.** The prefetcher's per-page push bytes equal
   `block_height_in_tiles[t] * coalesced_num_pages[t] * coalesced_page_size[t]`,
   and the receiver sees this many bytes per `wait_front(1)`-equivalent slot.
2. **Receiver `num_blocks` compile arg equals ring_size.** Enforced because
   both prefetchers derive `num_blocks = num_senders * num_receivers_per_sender`
   from the GCB.
3. **One `pages_sent` increment per page per receiver per layer per tensor.**
   The DRAM-core kernel's `prefetcher_finalize_block` is the single
   `pages_sent`-bumping primitive; `prefetcher_write_chunk` never touches it.
   The worker-core writer gets the same property by issuing exactly one
   `remote_cb_push_back_and_write_pages` per block.
4. **All tensors in a launch share `num_blocks`** (= ring_size). The kernel
   takes a single `num_blocks` compile-time arg; all other geometry is
   per-tensor.
5. **The pair `(rows_per_sub > 1, M > 1)` never occurs** in the DRAM-core
   fit ladder. The kernel uses this invariant to assume DMAs are linear
   reads (no row-strided DMA support).

## 9. Where to look in the code

- Receiver matmul reader:
  `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp`
- Receiver matmul compute:
  `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`
- Worker-core prefetcher:
  `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp`,
  `kernels/reader_dram.cpp`, `kernels/writer_l1.cpp`.
- DRAM-core prefetcher:
  `tt_metal/impl/buffers/dram_core_prefetcher_manager.cpp`,
  `tt_metal/impl/buffers/kernels/dram_core_prefetcher.cpp`.
- GCB / remote CB API:
  `tt_metal/hw/inc/api/remote_circular_buffer.h`,
  `tt_metal/impl/buffers/global_circular_buffer.cpp`.
- DRISC L1 arena: `tt_metal/impl/buffers/drisc_l1_arena.hpp`.
- DRISC GDDR DMA: `tt_metal/hw/inc/experimental/gddr_dma.h`.
