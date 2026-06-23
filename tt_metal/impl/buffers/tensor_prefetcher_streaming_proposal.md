# Streaming in1 into the ring matmul — background, problem, options

This note explores whether the gather-in0 ring matmul can begin computing
*before* the DRAM prefetcher has delivered all of a tensor's weight blocks,
and what it would take to get there. It builds on
[`prefetcher_matmul_design.md`](./prefetcher_matmul_design.md), which is the
authoritative description of the prefetcher↔receiver contract; read that first.

Status: **implemented for receiver-contiguous mode** (opt-in per-tensor `rotation` on
`QueueTensorPrefetcherRequest` + `MatmulMultiCoreReuseMultiCast1DProgramConfig.stream_in1`).
recv-contig already shards per receiver (§4.5's prerequisite), so the only change
needed was moving the per-receiver rotation out of the matmul (which random-accessed
the resident block set) and into the prefetcher's circular DRAM read: at push step `p`
receiver `r` sources block `(rotation[r] + p) mod N`, the matmul consumes the GCB FIFO
front-to-back, and the GCB shrinks to a small live window. The fan-out-mode obstruction
in §3 below does not apply to recv-contig. The original exploration (for the fan-out
layout) follows.

> **Addendum (host-owned rotation, two-DMA-read only).** The shipped API now exposes the
> rotation table directly rather than a `streaming` bool:
>
> - `TensorPrefetcherInput.rotation` (ttnn: the optional 3rd tuple element `(weight,
>   block_count, rotation)`) is a per-tensor `vector<uint32_t>` indexed by global ring
>   position, length `total_receivers` (== ring size == `block_count`), each entry in
>   `[0, block_count)`. Empty == batched. `rotation[r] = r` reproduces the natural
>   topology order described above; the host can instead choose any per-receiver lead
>   block, so it owns the full delivery order rather than being fixed to the topology
>   ring index `g_r`.
> - The rotation is **host-owned and carried in the request page**, not stamped into the
>   GCB sender state block. Because request pages broadcast identically to every sender
>   would have to carry the full `total_receivers` table (256 B at ring=64, > the 128 B
>   page), pages are now **serialized per sender**: each sender's page carries only its
>   own receivers' slice (≤ `recv_per_bank` entries), appended right after each layout
>   slot's geometry (uniform `layout_stride` sized by `max_num_receivers`). See
>   [`tensor_prefetcher_request.hpp`](./tensor_prefetcher_request.hpp).
> - Only the **two-DMA-read** source split survives: a receiver whose ring-rotated run
>   crosses the physical slab end reads it as two contiguous DMAs (tail then head) into
>   one stage slot. The earlier legacy-clamp and ragged-tail/head modes (and the
>   `TT_TENSOR_PREFETCHER_STREAMING_SPLIT` env var / compile arg) have been removed.

---

## 1. Background

### 1.1 The ring matmul (`gather_in0`)

`out[M,N] = in0[M,K] @ in1[K,N]` runs across a ring of `ring_size` worker
cores (`MatmulMultiCoreReuseMultiCast1DProgramConfig`, `gather_in0=True`).

- **in0 (activations `[M,K]`)** — width-sharded along K across the ring. Core
  `i` starts with the `[M, K/ring_size]` slice for ring position `i`.
- **in1 (weights `[K,N]`)** — each core owns one output N-block of width
  `N/ring_size`. In the prefetcher path it arrives via a Global CB (GCB).
- **out `[M,N]`** — width-sharded the same way; core `i` produces `[M, N/ring_size]`.

No core ever holds all of in0. The **in0 all-gather** rotates K-slices around
the ring: at ring step `s`, core `i` holds slice `(i+s) % ring_size`,
multiplies it by the matching in1 K-block, accumulates, and forwards the slice
to its neighbor (`reader_bmm_tile_layout_in0_ring_all_gather.cpp:67`, push at
`:96`). After `ring_size` steps each core has streamed through all of K.
**Hop cores** relay slices when logical ring neighbors aren't physically
adjacent.

Code: in0 ring `reader_bmm_tile_layout_in0_ring_all_gather.cpp`, in1 reader
`reader_bmm_tile_layout_in1_ring_all_gather.cpp`, compute
`bmm_large_block_zm_fused_bias_activation_gathered.cpp`, host factory
`matmul_multicore_reuse_mcast_1d_program_factory.cpp`.

### 1.2 The prefetcher feeds in1

The DRAM-core prefetcher (`ttnn.experimental.start_tensor_prefetcher`)
streams weight blocks from DRAM into the GCB so the matmul never stalls on a
weight read. Per tensor it pushes exactly `num_blocks = ring_size` pages to
each receiver, one per K-block (`tensor_prefetcher.cpp`, block loop at
`:317`/`:329`, finalize/`pages_sent` per block).

### 1.3 Why `num_blocks = ring_size`

`ring_size` shows up for **two unrelated reasons** that happen to coincide:

- **Width (N):** the output is width-sharded across
  `num_senders × num_receivers_per_sender = ring_size` receivers.
- **Height (K):** in0 is K-sharded into `ring_size` slices (one per ring core),
  the all-gather delivers them one per ring step, and in1 must be split into
  `ring_size` matching K-blocks so block `(ring_idx+s)` contracts against in0
  slice `(ring_idx+s)` at step `s`.

Both trace back to "there are `ring_size` cores." The K-split is dictated by the
in0 all-gather granularity, not by the receiver fan-out.

---

## 2. The problem: the receiver waits for the whole tensor

### 2.1 What gates

In the `ENABLE_GLOBAL_CB` path the compute kernel gates its *entire* block loop
on one signal:

```c++
// bmm_large_block_zm_fused_bias_activation_gathered.cpp:293
sync2_buf.wait_front(1);   // posted by the in1 reader only after ALL blocks land
sync2_buf.pop_front(1);
for (uint32_t block = 0; block < num_blocks; block++) { ... }
```

`sync2` is posted by the in1 reader *after*
`remote_cb_wait_front(remote_cb_id, num_blocks)`
(`reader_bmm_tile_layout_in1_ring_all_gather.cpp:142`). Compute then
**random-accesses** the full block set: `update_rd_ptr_to_ring_index(...)`
jumps the read pointer to `ring_idx` and walks it around the whole region
(`bmm_large_block_..._gathered.cpp:271`, `:323-334`). So per tensor it is
all-or-nothing — the matmul cannot touch block 0 until all `ring_size` in1
pages are present.

### 2.2 in0 vs. in1 asymmetry

in0 is **not** waited-for all-at-once: the ring all-gather pushes one slice at a
time and compute consumes per block (`input0_cb.wait_front`,
`...gathered.cpp:321`). So the gather half already streams; it is specifically
the prefetcher-fed in1 half that is batched. If in1 streamed too, the block
loop would pipeline against both inputs — the structure is already there for
in0.

### 2.3 Why in1 is batched today

1. **Per-receiver rotation vs. single push order.** Receiver `i` consumes
   K-blocks in order `i, i+1, …` (wrapping), forced to match the rotated in0
   slice order. The prefetcher reads each K-block once and fans its N-slice to
   all its receivers, so it pushes in *global* index order `0,1,…,N-1`. That
   order is receiver 0's ring order but nobody else's, so the design pushes in
   index order and lets each receiver random-access its own rotation out of the
   full resident set.
2. **Coarse sync.** One `wait_front(num_blocks)` + one `pop_front(num_blocks)`
   per tensor instead of `num_blocks` credit round-trips.
3. It is the documented cross-component contract
   ([`prefetcher_matmul_design.md`](./prefetcher_matmul_design.md) §4/§8).

### 2.4 Does it actually stall?

In **steady state**, mostly no: `remote_cb_reserve_back` is credit-based, so the
prefetcher runs ahead and fills tensor `t+1` while compute drains tensor `t`.
The `wait_front(num_blocks)` then resolves immediately. The all-at-once wait
costs at **cold start / pipeline fill**, and it forces the GCB to be sized for a
full tensor (`num_blocks` pages/receiver) so the run-ahead can hide it.

So there are two distinct prizes:
- **Fill latency** — only matters cold / first tensor.
- **GCB depth** — `num_blocks` pages/receiver of L1 reserved per tensor, whether
  or not the latency is hidden. Streaming would let this shrink to a small live
  window.

---

## 3. The core obstruction

A single fanned DRAM read of K-stripe `blk` serves `R = num_receivers_per_sender`
receivers, and those receivers consume that stripe at **different ring steps**
(they have distinct ring indices). No reordering of a *single* page-index→block
mapping can satisfy `R` different rotations at once. Formally, perfect streaming
would require block `(i_r + s)` available by push step `s` for every receiver
`r`; at `s = 0` that forces all `i_r` equal — impossible for `R > 1`.

The escape is to break the "one read, one mapping, R receivers" coupling. The
solutions below differ in *how*.

A topology fact makes this tractable: the factory assigns `ring_idx = i`
(row-major worker order) with
`dram_read_offset = i % num_receiver_cores_per_dram`
(`matmul_multicore_reuse_mcast_1d_program_factory.cpp:2620`), so **a bank's
receivers have contiguous ring indices** `[b·R, b·R+R-1]`. Hence at ring step
`s` a bank's receivers collectively need a **sliding window of `R` consecutive
blocks** `[b·R+s, b·R+R-1+s]`, not arbitrary ones.

---

## 4. Potential solutions

Ordered roughly from cheapest/weakest to the recommended direction.

### 4.1 Naive per-block streaming (does not work as-is)

Switch the receiver to `remote_cb_wait_front(1)/pop_front(1)` per ring step.
Blocked directly by §3: with a single index-order push, only receiver 0
streams; receiver `i` cannot start until block `i` has been pushed.

### 4.2 Staggered start

Keep index-order push, but let receiver `i` begin once its *first* needed block
(block `i`) lands instead of waiting for all `N`. Receiver 0 starts immediately;
receiver `i` after `i+1` pushes. **Bounded and uneven** — the highest-ring-index
receiver still waits ≈ all blocks, and you still need full GCB depth (early
blocks stay resident for the tail of the rotation). Helps cold-start latency
only; no GCB shrink.

### 4.3 Per-bank read-offset rotation (fill → `R-1`)

Exploit the contiguous-index fact: have bank `b` read its K-stripes starting at
`b·R` and wrapping (`b·R, b·R+1, …`). A single linear sweep then feeds the
sliding window. No byte movement — just change the starting `blk` and wrap
(`block_stride_bytes` already strides; `tensor_prefetcher.cpp:359`).

- Worst-case fill drops from `ring_size-1` to **`R-1 = num_receivers_per_sender-1`**
  (ring=64 → 63 → 7).
- But all `R` receivers of a bank still share one push sequence, so receiver at
  local offset `k` starts at page `k` — a **within-bank residue of up to `R-1`**.
- Cheap, but doesn't reach zero fill and doesn't fully free the GCB.

### 4.4 Per-receiver rotation via a skewed DRAM layout (fill → pipeline depth)

Physically store a **diagonal**: at push step `p`, co-locate each receiver's
ring-step-`p` block.

```
DRAM offset p  ==  concat over k=0..R-1 of  { block (b·R + k + p) mod N , receiver-k columns }
```

A single contiguous read at offset `p` delivers exactly what all `R` receivers
need at ring step `p`; the normal fan-out write hands each its slice. No
within-bank residue.

- **Bandwidth-neutral** — a pure permutation; each `(block, receiver)` pair
  appears once, same per-step DMA volume, and consecutive push steps become
  contiguous (simpler than today's strided read).
- Receiver becomes a pure FIFO streamer (drop `wait_front(num_blocks)` and the
  `update_rd_ptr_to_ring_index` random access); fill ≈ pipeline depth; GCB
  shrinks to the live window.
- **Cost:** a one-time weight reshuffle into the skewed layout (fine for static
  inference weights), regenerated per `(ring_size, topology)`. The intra-block
  byte layout (§3 of the design doc) is unchanged — only *which (receiver,
  block) pair lands at which push step* changes.

### 4.5 Per-receiver sharding + circular read (no reshuffle) — recommended

Achieve the same per-receiver rotation with **no byte reshuffle** by sharding
finer and reading each shard as a ring buffer.

- Shard the tensor **per receiver** (`n_per_recv` wide) instead of per bank
  (`n_per_bank` wide); place a bank's `R` shards on that bank at successive
  offsets. (A tensor sharding choice, not a data-modification pass.)
- For receiver `k` (ring index `g_k`), read its shard **circularly starting at
  block `g_k`**:

  ```
  src(blk) = shard_base_k + ((g_k + blk) mod N) * block_stride_within_shard
             // block_stride_within_shard = k_block_w_tiles * n_per_recv * tile_bytes
  ```

- The rotation lives entirely in the per-shard start offset. The wrap is two
  linear segments per shard per layer: `[g_k, N)` then `[0, g_k)`.

Correct against the natural layout for free: a K-row-major shard stores each
block as a contiguous byte range, so "block-major within the shard" holds, the
circular read is well-defined, and the receiver's expected intra-block layout is
unchanged. Page `p` to receiver `k` is block `(g_k+p)`, matching the in0 slice
arriving at ring step `p`.

Payoff identical to §4.4 — FIFO-streaming receiver, fill ≈ pipeline depth,
shallow GCB — **without** the reshuffle. The apparent cost (one read per
receiver instead of one wide read across receivers) is addressed by §4.6.

### 4.6 Tall (K-direction) reads — recover coalescing, dissolve the fit ladder

Within a per-receiver shard, **consecutive blocks are physically adjacent**
(K-row-major tile storage: tile `(kr, nc)` at `kr*n_per_recv + nc`, so a block =
`k_block_w` consecutive K-rows is contiguous). The receiver consumes blocks in
consecutive order, so read several at once:

- Read `m` consecutive blocks in one DMA — `m*k_block_w` rows tall ×
  `n_per_recv` wide, fully contiguous. `m` bounded by `ring_half`.
- The coalescing you "lose" across receivers (N-axis) is recovered along K, and
  K is the natural axis because consumption is consecutive.
- **Read granularity (tall, stage-sized) decouples from page granularity (one
  block):** read `m` blocks into a stage half, then push/finalize `m` pages to
  that receiver — the same "read chunk → write several pages" shape the kernel
  already has, with the chunk now `m` consecutive blocks of one shard and the
  write one receiver × `m` pages.

This **dissolves the fit ladder** (design doc §6): the read unit becomes "as
many consecutive blocks of a per-receiver shard as fit `ring_half`," always
satisfiable and always contiguous — no more "a 121 KB bank-wide block must be
sub-banded / M-chunked."

Production sanity check (ring=64, 8 banks × 8 recv, bf8_b `tile_bytes=1088`,
`ring_half ≈ 35 KB`):

| Op  | K    | N     | k_block_w | n_per_bank | n_per_recv | per-recv block | full bank block (today) |
|-----|------|-------|-----------|------------|-----------:|---------------:|------------------------:|
| FF1 | 4096 | 14336 | 2         | 56         | 7          | ~15 KB         | 121 856 B (subbanded `M=2`) |
| QKV | 4096 | 12288 | 2         | 48         | 6          | ~13 KB         | 104 448 B (`M=2`)        |
| O   | 4096 | 4096  | 2         | 16         | 2          | ~4.4 KB        |  34 816 B (fast path)    |

A per-receiver block (~15 KB for FF1) fits `ring_half` with room for `m ≈ 2`
blocks per read — so the path lands on a single contiguous read and skips
sub-banding entirely.

---

## 5. Recommendation and open questions

**Recommended direction:** §4.5 (per-receiver sharding + circular read) combined
with §4.6 (tall K-direction reads). Together they give:

- per-receiver rotation with **no DRAM reshuffle**;
- a **FIFO-streaming receiver** — drop `wait_front(num_blocks)` and the
  `update_rd_ptr_to_ring_index` random access;
- **fill ≈ pipeline depth** instead of `ring_size`;
- a **shallow GCB** (a small live window per receiver instead of `num_blocks`
  pages) — likely the biggest L1 win;
- **large contiguous GDDR reads** and **removal of the fit-ladder** sub-banding.

The only structural trade vs. today: a single DMA no longer feeds multiple
receivers, so each per-receiver shard stream is independent and the slowest
receiver in a bank paces *its own* stream. The reserve-on-all-receivers
back-pressure coupling is unchanged.

Open questions to settle before implementing:

1. **GDDR burst efficiency** of tall, `n_per_recv`-wide reads (~15 KB for FF1)
   vs. today's wider sub-band reads — benchmark. The current path already issues
   multiple sub-band reads per block, so this may be a wash or a win.
2. **DRAM shard→bank placement** for `ring_size` shards over `num_senders` banks
   (R shards per bank at known offsets) — confirm the buffer/allocator supports
   the per-shard base addresses the rotation needs.
3. **GCB depth / credit tuning** for the streaming window: how many blocks of
   lead the prefetcher should keep to absorb receiver-rate jitter.
4. **Wrap handling** in the kernel: two linear segments per shard per layer;
   ensure a tall read never straddles the `N-1 → 0` boundary.

---

## 6. Cross-component impact

Any of §4.3–4.6 is a **cross-component change** governed by
[`prefetcher_matmul_design.md`](./prefetcher_matmul_design.md) §4/§8:

- **Receiver matmul** (`reader_bmm_tile_layout_in1_ring_all_gather.cpp`,
  `bmm_large_block_zm_fused_bias_activation_gathered.cpp`): replace the
  `wait_front(num_blocks)` gate + random-access addressing with incremental
  `wait_front(1)`/FIFO consumption.
- **Prefetcher** (`tensor_prefetcher.cpp`,
  `tensor_prefetcher_manager.cpp`): per-shard circular read offset + tall-read
  chunking; per-block push/`pages_sent` bookkeeping stays the same.
- **Tensor sharding** (host op / memory config): per-receiver width shards.
- **Validators** (`dram_prefetcher_validator.cpp`,
  `gcb_validator_receiver.cpp`): update to the new push order / page-arrival
  contract.
- The **intra-block byte layout** (design doc §3) is preserved throughout — only
  the *inter-block ordering and read pattern* change.

Update the design doc in the same commit as any of these.
