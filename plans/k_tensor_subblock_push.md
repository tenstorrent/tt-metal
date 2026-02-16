# Plan: K Tensor Column-Subblock Push for SDPA

## 0. Build, Test & Recovery

```bash
# Build (only host .cpp changes need this; kernel .cpp files are JIT-compiled)
./build_metal.sh --release

# Run tests (always use --timeout to avoid indefinite chip hangs)
pytest tests/tt_eager/python_api_testing/unit_testing/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy --timeout=30 -svv

# On chip hang — triage then reset
./tools/tt-triage.py        # collect watcher dump & diagnostics
tt-smi -r                   # reset the chip

# Watcher: set TT_METAL_WATCHER=1 to enable watcher logging
TT_METAL_WATCHER=1 pytest tests/tt_eager/python_api_testing/unit_testing/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy --timeout=30 -svv
# Watcher status codes: NWBW = noc_async_write_barrier, CRBW = cb_reserve_back wait,
# NSW = noc_semaphore_wait, CWFW = cb_wait_front
```

**All changes must pass `test_sdpa_accuracy`** before being considered complete.

## 1. Background — Q Subblock Push Recap

Commit `ec09e4c` optimized Q fetch by pushing Q tiles at `qk_subblock_h` granularity
instead of atomically. This works because `matmul_blocks` consumes Q (in0) with an
**accumulating** `cb_wait_front`:

```cpp
// compute_common.hpp — matmul_blocks (current)
cb_wait_front(in1_cb, K * N);              // K: wait for ALL tiles at once

for (in0_subblock = 0; ...) {
    cb_wait_front(in0_cb, in0_wait_tiles); // Q: accumulating wait per subblock
    ...
    in0_wait_tiles += subblock_h * in0_block_w;
}
```

Each Q subblock push (`subblock_h * DHt` tiles) immediately unblocks the next
QK matmul row computation, overlapping remaining Q DRAM reads with compute.

**Goal**: apply the same pattern to K (in1) — push K in column subblocks so
compute can start before the full K chunk is available. All cores must benefit:
injectors, receivers, and non-participants alike.

## 2. Approach: Column-Subblock K Push

### 2.1 Core Idea

When `qk_in1_num_subblocks > 1` (i.e., `subblock_w < Sk_chunk_t`), apply the
**same accumulating-wait pattern** to K that already works for Q:

- Rearrange K's CB layout so each column subblock is **contiguous**
- Make `cb_wait_front` for K **accumulating** inside the `in1_subblock` loop
- Reader pushes K one column subblock at a time (`DHt * subblock_w` tiles per push)
- Forward (mcast/unicast) each K subblock as it's read — one signaling round per subblock
- Compute starts after the **first** K subblock + first Q subblock

No partial accumulation needed — each `(in0_sub, in1_sub)` produces **final** output.

All three core roles benefit:
- **Injector**: reads K sub0 from DRAM, mcasts/unicasts it, compute starts while sub1 is read
- **Receiver**: receives K sub0 from injector, compute starts while sub1 is in-flight
- **Non-participant**: reads K sub0 from DRAM, compute starts while sub1 is read

### 2.2 Every k_chunk, Not Just the First

The K subblock push applies to **every k_chunk iteration**, not just k_chunk==0.

**Why not first-only?** The overhead of per-subblock reads is negligible:
- Extra cost per k_chunk: one additional `cb_reserve_back` + `cb_push_back` per subblock
  (just counter increments — a handful of cycles)
- Each subblock is small (e.g., 32 tiles) — a single `noc_async_read_barrier` at the
  end of each subblock is sufficient. No intermediate barriers needed, so the total
  barrier count is just `qk_in1_num_subblocks` (e.g., 2) vs 1 for the current path.
  The extra barrier overhead is ~100-200 cycles — negligible.
- A first-only path would add branching complexity with no measurable benefit

**Why every k_chunk benefits:**
- K is re-read from DRAM (or re-received via mcast) every k_chunk
- Compute waits for K before starting each QK matmul
- Starting compute sooner on every k_chunk compounds across the K loop
- For non-causal with 10+ k_chunks, the cumulative savings are significant

### 2.3 Concrete Example

Q = `[7 x 4]` tiles (Sq_chunk_t=7, DHt=4), K^T = `[4 x 16]` tiles (Sk_chunk_t=16).
`determine_largest_subblock_size(7, 16, 8)` -> **(1, 8)**.
`qk_in0_num_subblocks=7, qk_in1_num_subblocks=2`.

```
k_chunk==0 (Q not yet loaded — Q sub0 interleaved after K sub0):

  Currently (all cores):
    Reader:  |--- read ALL K ---|fwd K|mask|---Q subs---|
    Compute:                                |--- QK ----|
                                            ↑ waits for ALL K + Q sub0

  Proposed (injector, mcast):
    Reader:  |K_s0|mcast0|Q_s0|K_s1|mcast1|mask|Q_s1..Q_s6|
    Compute:              ↑(0,0)                |QK continues...
                          ↑ 32+4 = 36 tiles

  Proposed (injector, unicast):
    Reader:  |K_s0|write+flush+sig|Q_s0|K_s1|write|mask ← overlap →|flush+sig|Q_s1..Q_s6|
    Compute:                      ↑(0,0)                                      |QK continues...

  Proposed (receiver):
    Reader:  |K_s0 recv|Q_s0|K_s1 recv|mask|Q_s1..Q_s6|
    Compute:            ↑(0,0)              |QK continues...

  Proposed (non-participant):
    Reader:  |K_s0 read|Q_s0|K_s1 read|mask|Q_s1..Q_s6|
    Compute:            ↑(0,0)              |QK continues...

k_chunk > 0 (Q already in CB — compute waits for K sub only):

  Proposed (injector, mcast):
    Reader:  |K_s0|mcast0|K_s1|mcast1|mask|
    Compute:              ↑(0,0)      ↑(0,1)

  Proposed (receiver):
    Reader:  |K_s0 recv|K_s1 recv|mask|
    Compute:            ↑(0,0)    ↑(0,1)

  Proposed (injector, unicast):
    Reader:  |K_s0|write+flush+sig|K_s1|write|mask ← overlap →|flush+sig|
    Compute:                       ↑(0,0)                      ↑(0,1)
```

### 2.4 Why This Doesn't Work With the Current CB Layout

K is read with `transpose=true`. Current transposed layout places
K^T[d, s] at CB index `s + d * Sk_chunk_t`:

```
CB[0..7]   = K^T[0, 0..7]       CB[8..15]  = K^T[0, 8..15]
CB[16..23] = K^T[1, 0..7]       CB[24..31] = K^T[1, 8..15]
CB[32..39] = K^T[2, 0..7]       CB[40..47] = K^T[2, 8..15]
CB[48..55] = K^T[3, 0..7]       CB[56..63] = K^T[3, 8..15]
              ↑ subblock 0 and subblock 1 tiles interleaved
```

Subblock 0's tiles are at CB indices {0..7, 16..23, 32..39, 48..55} — scattered
with gaps. `cb_push_back(32)` would claim CB[0..31] as ready, but tiles at
CB[16..23] belong to subblock 0 while CB[8..15] belong to subblock 1 (not yet
written). Compute would read garbage.

### 2.5 Proposed CB Layout: Subblock-Contiguous Transposed

Group tiles by column subblock so each subblock occupies a contiguous CB region:

K^T[d, s] at CB index: `sb * DHt * subblock_w + s_local + d * subblock_w`
where `sb = s / subblock_w`, `s_local = s % subblock_w`

```
Subblock 0 — CB[0..31]:
  CB[0..7]   = K^T[0, 0..7]
  CB[8..15]  = K^T[1, 0..7]
  CB[16..23] = K^T[2, 0..7]
  CB[24..31] = K^T[3, 0..7]

Subblock 1 — CB[32..63]:
  CB[32..39] = K^T[0, 8..15]
  CB[40..47] = K^T[1, 8..15]
  CB[48..55] = K^T[2, 8..15]
  CB[56..63] = K^T[3, 8..15]
```

Each subblock is a self-contained transposed block with inner stride `subblock_w`
(instead of `Sk_chunk_t`). Contiguous in CB -> reader can `cb_push_back` per subblock.

### 2.6 Backward Compatibility

When `in1_num_subblocks == 1` (e.g., WAN shapes with Sk=7, subblock_w=7):
- Single subblock covers full Sk_chunk_t
- `sb * DHt * subblock_w = 0` (only sb=0)
- Stride = `subblock_w = Sk_chunk_t = N`
- Layout and behavior are **identical** to current code — zero risk of regression

## 3. Implementation Details

### 3.1 Compute — `matmul_blocks` in `compute_common.hpp`

Three changes to the existing function:

```cpp
ALWI void matmul_blocks(
    const uint32_t& in0_cb, const uint32_t& in1_cb, const uint32_t& out_cb,
    const uint32_t& M, const uint32_t& N, const uint32_t& K,
    const uint32_t& num_blocks,
    const uint32_t& in0_num_subblocks, const uint32_t& in1_num_subblocks,
    const uint32_t& in0_block_w,
    const uint32_t& subblock_h, const uint32_t& subblock_w,
    const bool& transpose, ...) {

    // ...existing setup...

    reconfig_data_format(in1_cb, in0_cb);
    // CHANGE 1: Remove upfront K wait
    // was: cb_wait_front(in1_cb, K * N);
    cb_reserve_back(out_cb, output_num_tiles);

    const uint32_t in1_subblock_num_tiles = K * subblock_w;  // tiles per K column subblock

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        cb_wait_front(in0_cb, in0_wait_tiles);

        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            // CHANGE 1 (cont): Accumulating K wait
            cb_wait_front(in1_cb, (in1_subblock + 1) * in1_subblock_num_tiles);

            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            // CHANGE 2: Subblock base index
            uint32_t in1_index = in1_subblock * K * subblock_w;
            //   was: in1_index = in1_index_offset  (where offset += subblock_w each iter)

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index,
                    transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                // CHANGE 3: Inner-dim stride = subblock_w (not N)
                in1_index += subblock_w;
                //   was: in1_index += N;
            }

            // mask add, tile_regs_commit, pack output — ALL UNCHANGED
            // (out_col_offset = in1_subblock * subblock_w is still correct
            //  because output is in normal non-transposed layout)
            ...

            tile_regs_release();
        }
        in0_index_offset += subblock_h * in0_block_w;
        in0_wait_tiles += in0_subblock_num_tiles;
        cb_push_back(out_cb, in0_subblock_all_cols_num_tiles);
    }
    cb_pop_front(in1_cb, K * N);  // UNCHANGED: pop all K at end
}
```

**Summary of index changes:**

| | Current | Proposed |
|---|---------|----------|
| K wait | `cb_wait_front(in1_cb, K*N)` once before loops | `cb_wait_front(in1_cb, (in1_sub+1)*K*sw)` per in1_subblock |
| `in1_index` base | `in1_subblock * subblock_w` | `in1_subblock * K * subblock_w` |
| `in1_index` stride | `+= N` (= Sk_chunk_t) | `+= subblock_w` |

When `in1_num_subblocks == 1`: base = `0`, stride = `sw = N`. **Identical** to current.

**How the accumulating K wait progresses (example: 2 subblocks):**
```
(in0=0, in1=0): wait for  1 * K*sw =  32 -> blocks until reader pushes sub0
(in0=0, in1=1): wait for  2 * K*sw =  64 -> blocks until reader pushes sub1
(in0=1, in1=0): wait for  32 -> 64 available, instant
(in0=1, in1=1): wait for  64 -> 64 available, instant
...all subsequent in0_subblocks: instant (all K already in CB)
```

### 3.2 Reader — new `read_k_subblock` helper in `dataflow_common.hpp`

Reads one column subblock of K^T (= `subblock_w` rows of K), transposed
within the subblock. Returns CB start address (needed by forwarding path).

```cpp
// Read one K column subblock: subblock_w rows of K[Sk x DHt], stored transposed.
// K[s_local, d] -> CB offset (s_local + d * subblock_w) within the reserved region.
// Pushes subblock_w * DHt tiles independently.
// Returns base_write_ptr for use as forwarding source address.
// No intermediate read barriers — each subblock is small enough (e.g., 32 tiles)
// that a single barrier at the end is sufficient. Intermediate barriers would
// just add overhead without meaningful NOC cmd_buf pressure relief.
template <uint32_t tile_bytes, typename ReaderType>
FORCE_INLINE uint32_t read_k_subblock(
    const ReaderType& reader,
    const uint32_t cb_id,
    uint32_t start_tile_id,       // DRAM tile_id for first tile of this subblock
    const uint32_t src_rows,      // valid K rows in this subblock (may be < subblock_w)
    const uint32_t src_cols,      // DHt
    const uint32_t subblock_w,    // padded subblock width (= dst_rows)
    const uint32_t full_dh_cols)  // DHt (DRAM row width, for tile_id stride)
{
    const uint32_t num_tiles = subblock_w * src_cols;
    cb_reserve_back(cb_id, num_tiles);
    const uint32_t base_write_ptr = get_write_ptr(cb_id);

    for (uint32_t s = 0; s < src_rows; ++s) {
        uint32_t tile_id = start_tile_id + s * full_dh_cols;
        for (uint32_t d = 0; d < src_cols; ++d) {
            uint32_t write_offset = (s + d * subblock_w) * tile_bytes;
            noc_async_read_tile(tile_id + d, reader, base_write_ptr + write_offset);
        }
    }

    // Zero-pad rows beyond src_rows
    for (uint32_t s = src_rows; s < subblock_w; ++s) {
        for (uint32_t d = 0; d < src_cols; ++d) {
            fill_tile_zeros<tile_bytes, false>(cb_id, s + d * subblock_w);
        }
    }

    noc_async_read_barrier();
    cb_push_back(cb_id, num_tiles);
    return base_write_ptr;
}
```

### 3.3 Reader — main loop change in `reader_interleaved.cpp`

The current reader has separate sections for K read, K forward initiate,
mask read, K forward complete, Q subblock push, V read, V forward.

With per-subblock K, the read and forward are **merged into one loop**.
Each subblock is a self-contained round of the existing semaphore protocol.

**Three paths, preserving mask/K-forward overlap:**

The current code splits K forwarding into "initiate" and "complete" around
the mask read so the mask DRAM reads overlap with the K write in-flight:
```
Current:  [K write(async)] [mask read ← overlap →] [flush + signal]
```

We preserve this: for subblocks 0..N-2, do a full forward round (write + flush +
signal). For the **last** subblock, split into initiate (write only, no flush)
before mask read and complete (flush + signal) after mask read.

For **mcast** this is natural — linked + companion is fire-and-forget, so the
last subblock's data is in-flight during mask reads with no extra handling.

```cpp
const uint32_t k_sub_tiles = qk_subblock_w * DHt;  // tiles per K column subblock

// ─── K subblock loop (all 3 paths) with Q sub0 interleave on k_chunk==0 ───

for (uint32_t sb = 0; sb < qk_in1_num_subblocks; ++sb) {

    // ── K subblock read/receive ──
    if (should_receive) {
        cb_reserve_back(cb_k_in, k_sub_tiles);
        noc_semaphore_set(receiver_semaphore_addr_ptr, INVALID);
        noc_semaphore_inc(sender_semaphore_noc_addr, 1);
        noc_semaphore_wait(receiver_semaphore_addr_ptr, VALID);
        cb_push_back(cb_k_in, k_sub_tiles);
    } else {
        const uint32_t sb_k_start = sb * qk_subblock_w;
        const uint32_t sb_rows = std::min(qk_subblock_w,
            (k_row_tile_count > sb_k_start) ? k_row_tile_count - sb_k_start : 0u);
        const uint32_t sb_tile_id = k_start_tile_id + sb_k_start * DHt;

        uint32_t sub_start_addr = read_k_subblock<k_tile_bytes>(
            k_reader, cb_k_in, sb_tile_id,
            sb_rows, DHt, qk_subblock_w, DHt);

        // ── Forward (injector only) ──
        if (should_forward) {
            noc_semaphore_wait(sender_semaphore_addr_ptr, sender_wait_count);
            noc_semaphore_set(sender_semaphore_addr_ptr, 0);

            if constexpr (mcast_enabled) {
                uint64_t k_mcast_addr = mcast_base_noc_addr | sub_start_addr;
                noc_async_write_multicast(
                    sub_start_addr, k_mcast_addr,
                    k_sub_tiles * k_tile_bytes,
                    mcast_num_dests, true /* linked */);
                noc_semaphore_set_multicast(
                    valid_semaphore_addr, mcast_sem_noc_addr, mcast_num_dests);
            } else {
                uint64_t k_unicast_addr =
                    get_noc_addr(next_physical_x, next_physical_y, sub_start_addr);
                noc_async_write(
                    sub_start_addr, k_unicast_addr, k_sub_tiles * k_tile_bytes);

                if (sb < qk_in1_num_subblocks - 1) {
                    noc_async_writes_flushed();
                    noc_semaphore_set_remote(
                        valid_semaphore_addr, receiver_semaphore_noc_addr);
                }
                // Last subblock: defer flush+signal to after mask read
            }
        }
    }

    // ── Q sub0 interleave: push first Q subblock right after K sub0 ──
    // This is the key: on k_chunk==0, compute needs both K sub0 AND Q sub0.
    // After K sub0 is forwarded (linked resolved / flushed), Q DRAM reads are safe.
    // Pushing Q sub0 here lets compute start at just 4x8 K + 1x4 Q = 36 tiles.
    if constexpr (use_q_subblock_push) {
        if (k_chunk == 0 && sb == 0) {
            read_q_subblock<q_tile_bytes>(
                q_reader, cb_q_in, q_read_tile_id,
                0 * qk_subblock_h,    // sb_start_row
                qk_subblock_h,
                q_row_tile_count, DHt, DHt,
                barrier_threshold);
        }
    }
}

// ─── Mask read (overlaps with last K subblock write in-flight) ───
if constexpr (use_provided_mask) {
    // ... mask read unchanged ...
}

// ─── Last K subblock forward COMPLETE (unicast only) ───
if (should_forward) {
    if constexpr (!mcast_enabled) {
        noc_async_writes_flushed();
        noc_semaphore_set_remote(
            valid_semaphore_addr, receiver_semaphore_noc_addr);
    }
}

// ─── Remaining Q subblocks (k_chunk==0 only, sub1..subM) ───
if constexpr (use_q_subblock_push) {
    if (k_chunk == 0) {
        for (uint32_t q_sub = 1; q_sub < q_num_subblocks; ++q_sub) {
            read_q_subblock<q_tile_bytes>(
                q_reader, cb_q_in, q_read_tile_id,
                q_sub * qk_subblock_h,
                qk_subblock_h,
                q_row_tile_count, DHt, DHt,
                barrier_threshold);
        }
    }
}
```

**Why Q sub0 interleave is safe after K sub0 forwarding:**
- **Mcast**: K sub0's linked + companion resolves the linked constraint.
  `read_q_subblock` calls `noc_async_read_barrier()` — no deadlock.
- **Unicast** (non-last sub): K sub0's flush completes all writes.
  Q reads are safe.
- **Unicast** (only 1 K sub): K sub0's write is outstanding but unflushed.
  `noc_async_read_barrier()` only waits for reads, not writes — safe.
  The write flushes after mask read (same as current code).
- **Non-participant**: no forwarding, no writes. Q reads always safe.

**Why mask/K-forward overlap is preserved:**
```
Mcast:   [...sub0 mcast][Q_s0][...sub1 mcast(fire-and-forget)][mask ← overlap →]
Unicast: [...sub0 write+flush+sig][Q_s0][sub1 write(no flush)][mask ← overlap →][flush+sig]
```
The last K subblock's async write is still in-flight during mask reads.

**Why per-subblock mcast is safe between subblocks:**
- After each mcast linked write + companion: the linked constraint is resolved
- Next operation's `noc_async_read_barrier()` (in `read_k_subblock` or `read_q_subblock`)
  is safe — no outstanding linked writes remain
- Mcast data goes to receivers' CB; next subblock reads from DRAM to injector's CB
  — different source/dest, no conflict
- Injector and receiver CB write pointers advance in lockstep (both do
  `cb_reserve_back(sub_tiles)` + `cb_push_back(sub_tiles)` per subblock), so
  `mcast_base_noc_addr | sub_start_addr` targets the correct L1 address on receivers

### 3.4 Interaction with Q Subblock Push

Q subblock push (from commit `ec09e4c`) is now **split across two locations**:

- **Q sub0**: pushed inside the K subblock loop, immediately after K sub0
  is forwarded (k_chunk==0, sb==0). This is the key change that lets compute
  start at 36 tiles instead of waiting for all K + mask.
- **Q sub1..subM**: pushed after mask read and K forward complete (unchanged
  location, but starting from sub1 instead of sub0).

**Why Q sub0 after K sub0 is safe** (same justification as Q push after all K):
- Mcast: K sub0's companion resolves the linked constraint
- Unicast non-last: K sub0 flushed + signaled
- Unicast single-sub: outstanding write but `noc_async_read_barrier` only
  waits for reads — no conflict with NOC writes
- Non-participant / receiver: no outstanding writes

**Why remaining Q subs after mask + K forward complete are safe:**
- Mcast: all companions sent → no linked writes
- Unicast: last K sub flushed after mask → no outstanding writes

Combined flow on `k_chunk==0` **(compute starts after K sub0 + Q sub0 = 36 tiles)**:

```
Injector (mcast):
  Reader:  [K_s0 read+mcast] [Q_s0] [K_s1 read+mcast] [mask ← overlap →] [Q_s1..Q_s6]
  Compute:                    ↑(0,0)                                        ↑(1,0)...

Receiver:
  Reader:  [K_s0 recv] [Q_s0] [K_s1 recv] [mask] [Q_s1..Q_s6]
  Compute:              ↑(0,0)                     ↑(1,0)...

Non-participant:
  Reader:  [K_s0 read] [Q_s0] [K_s1 read] [mask] [Q_s1..Q_s6]
  Compute:              ↑(0,0)                     ↑(1,0)...

Injector (unicast):
  Reader:  [K_s0 write+flush+sig] [Q_s0] [K_s1 write] [mask ← overlap →] [flush+sig] [Q_s1..Q_s6]
  Compute:                         ↑(0,0)                                              ↑(1,0)...
```

Combined flow on `k_chunk > 0` (Q already in CB — full benefit, no Q interleave needed):

```
Injector (mcast):
  Reader:  [K_s0 read+mcast] [K_s1 read+mcast] [mask ← overlap →]
  Compute:                    ↑(0,0)             ↑(0,1)

Receiver:
  Reader:  [K_s0 recv] [K_s1 recv] [mask]
  Compute:              ↑(0,0)      ↑(0,1)

Injector (unicast):
  Reader:  [K_s0 write+flush+sig] [K_s1 write] [mask ← overlap →] [flush+sig]
  Compute:                         ↑(0,0)                           ↑(0,1)
```

### 3.5 Host — `sdpa_program_factory.cpp`

- Pass `qk_subblock_w` as a new reader compile-time arg (after `qk_subblock_h`)
- Kernel derives `qk_in1_num_subblocks = Sk_chunk_t / qk_subblock_w`
- Shift subsequent compile-time arg indices by 1
  (semaphore IDs, mcast_enabled, TensorAccessorArgs offsets)
- No CB size change — K CB still holds `Sk_chunk_t * DHt` tiles total
  (subblocks accumulate in the CB, popped all at once by compute)
- Paged attention (`is_chunked`) path is **unchanged** — it uses
  `read_paged_chunk_with_padding` which doesn't support forwarding.
  The subblock layout only applies to the non-paged path.

## 4. Expected Benefits

### 4.1 Compute start latency reduction

| Sq_chunk_t | Sk_chunk_t | DHt | subblock (h,w) | in1_subs | First K push | Total K | Reduction |
|------------|------------|-----|----------------|----------|-------------|---------|-----------|
| 7 | 7 | 4 | (1, 7) | 1 | 28 tiles | 28 tiles | **0%** (no change) |
| 7 | 14 | 4 | (1, 7) | 2 | 28 tiles | 56 tiles | **50%** |
| 7 | 16 | 4 | (1, 8) | 2 | 32 tiles | 64 tiles | **50%** |
| 7 | 24 | 4 | (1, 8) | 3 | 32 tiles | 96 tiles | **67%** |
| 7 | 32 | 8 | (1, 8) | 4 | 64 tiles | 256 tiles | **75%** |

### 4.2 Pipeline overlap per k_chunk (all core types)

**Injector:** compute `(in0=0, in1=0)` overlaps with DRAM read + mcast of K sub1.

**Receiver:** compute `(in0=0, in1=0)` overlaps with receiving K sub1 from injector.

**Non-participant:** compute `(in0=0, in1=0)` overlaps with DRAM read of K sub1.

Applies to **every k_chunk** (K is re-read/re-received each iteration).
Cumulative across all k_chunks per q_iter.

### 4.3 When this helps vs. doesn't

**Benefits when:** `Sk_chunk_t > subblock_w` -> multiple in1_subblocks.
Common for long sequences with large K chunks.

**No change when:** `Sk_chunk_t == subblock_w` -> single subblock.
All index expressions reduce to current values. Zero regression risk.

## 5. Risks

1. **`mm_block_init_short` kt_dim**: Uses `in0_block_w` (= DHt) as `kt_dim`.
   Unaffected — `in0_block_w` is unchanged, only in1 indexing changes.

2. **Output pack indexing**: `out_col_offset = in1_subblock * subblock_w`
   is unaffected — output is packed in normal non-transposed layout.

3. **Mcast destination alignment**: Injector and receiver CB write pointers
   must advance in lockstep. Both do `cb_reserve_back(sub_tiles)` +
   `cb_push_back(sub_tiles)` per subblock -> write pointers stay aligned.
   `mcast_base_noc_addr | sub_start_addr` hits the correct L1 address.

4. **Signaling overhead**: `qk_in1_num_subblocks` semaphore rounds per k_chunk
   for K (was 1). For 2 subblocks: 2 rounds instead of 1. Each round is
   ~100-200 cycles. Overhead: ~100-200 cycles per k_chunk — negligible
   compared to DRAM/compute latency.

5. **Linked write safety between subblocks**: After each subblock's
   `noc_async_write_multicast(linked=true)` + companion semaphore, the linked
   constraint is resolved. Next subblock's DRAM reads (with `noc_async_read_barrier`)
   are safe. No deadlock risk.

6. **V tensor**: Same optimization applies to V (in1 of `softmax(QK) x V`).
   V is not transposed but the same subblock-contiguous idea works.
   Defer to a follow-up.

## 6. Implementation Order

1. Compute: 3 index changes + accumulating K wait in `matmul_blocks`.
2. Reader helper: `read_k_subblock` in `dataflow_common.hpp`.
3. Reader main loop: unified K read+forward subblock loop (3 paths),
   split Q subblock push (sub0 inside K loop after sb==0, sub1..M after mask).
4. Host: pass `qk_subblock_w` as reader compile-time arg, shift indices.
5. Build: `./build_metal.sh --release`
6. Test: `pytest tests/tt_eager/python_api_testing/unit_testing/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy --timeout=30 -svv`
7. On hang: `./tools/tt-triage.py && tt-smi -r`
