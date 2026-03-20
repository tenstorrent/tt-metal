# ReduceScatter Ring Algorithm Deep Dive

This document explains how the **ring-based ReduceScatter** algorithm works in `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/`.

---

## 1. Overview: What is ReduceScatter?

**ReduceScatter** is a collective communication operation that:
1. Takes an input tensor split into `N` slices (where `N` = ring_size = number of devices)
2. Reduces (sums) all corresponding slices across all devices
3. Each device keeps only one final reduced slice (its "own" slice)

For an 8-device ring with input shape `[8, H, W]` scattered on dim=0:
- Device 0 keeps the reduced slice 0
- Device 1 keeps the reduced slice 1
- ...
- Device 7 keeps the reduced slice 7

---

## 2. Ring Topology and Bidirectional Flow

The algorithm uses a **ring topology** with **bidirectional data flow**:

```
        Forward direction (dir=1) --->

    [D0] --> [D1] --> [D2] --> [D3] --> [D4] --> [D5] --> [D6] --> [D7] --> [D0]

        <--- Backward direction (dir=0)
```

**Two sets of workers** operate in parallel:
- **Forward workers** (`dir=1`): Send data to the next device (D4 -> D5)
- **Backward workers** (`dir=0`): Send data to the previous device (D4 -> D3)

This doubles the throughput by utilizing both directions of the ring simultaneously.

---

## 3. The Algorithm from Device 4's Perspective

Let's trace the algorithm on **Device 4** with **ring_size = 8**, focusing on the **forward direction** (`dir=1`).

### 3.1 Initial Setup

From `ring_reduce_scatter_minimal_async_reader.cpp:115`:
```cpp
int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
// For Device 4, forward direction: slice_idx = 4 - 1 = 3
```

The reader starts at `slice_idx = 3` and works backward toward slice 4 (its own slice).

### 3.2 Iteration-by-Iteration Walkthrough

The main loop runs `ring_size` (8) iterations:

```cpp
for (uint32_t i = 0; i < ring_size; ++i) {
    const bool do_reduce = i != 0;  // First iteration: no reduction needed
    ...
}
```

#### **Iteration 0 (i=0): Bootstrap - No Reduction**

| Step | Action |
|------|--------|
| `slice_idx` | 3 |
| `do_reduce` | `false` (first iteration) |
| **Reader** | Reads local input slice 3 directly to `cb_reader_output_id` |
| **Writer** | Sends slice 3 to Device 5's intermediate buffer |

Code (`reader.cpp:126`):
```cpp
uint32_t cb_in0 = do_reduce ? cb_input_id : cb_reader_output_id;
// On i=0: cb_in0 = cb_reader_output_id (bypass reduction)
```

#### **Iteration 1 (i=1): First Reduction**

| Step | Action |
|------|--------|
| `slice_idx` | 2 (decremented) |
| `do_reduce` | `true` |
| **Reader** | Reads local slice 2 -> `cb_input_id` |
| **Reader** | Reads intermediate buffer (partial sum from D3) -> `cb_intermediate_id` |
| **Compute** | Adds them: `slice2_local + partial_from_D3` -> `cb_compute_output_id` |
| **Writer** | Sends reduced result to Device 5 |

**What's in the intermediate buffer?** At iteration i=1, D3 (in iteration i=0) wrote its local slice 2 to D4's intermediate buffer. So:
- `intermediate[2]` contains: D3's local slice 2

From `reader.cpp:214-234`:
```cpp
if (do_reduce) {
    // read next intermediate slice out of the intermediate buffer
    cb_reserve_back(cb_intermediate_id, tile_granularity);
    uint32_t intermediate_l1_write_addr = get_write_ptr(cb_intermediate_id);
    for (uint32_t j = 0; j < tiles_to_read_in_current_direction; ++j) {
        uint32_t intermediate_tile_id =
            intermediate_tile_id_start + intermediate_row_offset + intermediate_pages_read_in_row;
        uint64_t intermediate_noc_read_addr =
            get_noc_addr(intermediate_tile_id, intermediate_tensor_addrgen);
        noc_async_read(intermediate_noc_read_addr, intermediate_l1_write_addr, page_size);
        ...
    }
}
```

The intermediate buffer is indexed by `actual_slice_idx`, so Device 4 reads intermediate data for slice 2 when processing slice 2.

#### **Iterations 2-6: Continue Accumulating**

Each iteration:
1. Decrement `slice_idx` (wrapping: 1, 0, 7, 6, 5)
2. Read local slice at that index
3. Read intermediate buffer (accumulated partial from previous device)
4. Reduce (sum)
5. Write to next device (or to intermediate for next local slice)

#### **Iteration 7 (i=7): Final Write**

| Step | Action |
|------|--------|
| `slice_idx` | 4 (our own slice!) |
| **Reader** | Reads local slice 4 |
| **Reader** | Reads intermediate (accumulated sum from all other 7 devices) |
| **Compute** | Final reduction |
| **Writer** | Writes to OUTPUT buffer (not forwarding to next device) |

From `writer.cpp:410-458`:
```cpp
if (i < (ring_size - 1)) {
    // ... send to next device via fabric ...
} else {
    // Otherwise, on the last slice, write it to output buffer
    uint32_t output_tile_id_start = b * output_batch_num_pages;
    ...
    noc_async_write(l1_read_addr, local_noc_addr, page_size);
}
```

---

## 4. Slice Index Calculation

The `actual_slice_idx` handles wrap-around:

```cpp
// reader.cpp:129-133
uint32_t actual_slice_idx;
if (direction) {
    actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
} else {
    actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
}
```

For Device 4, forward direction, the slice indices are:
| Iteration | slice_idx (raw) | actual_slice_idx |
|-----------|-----------------|------------------|
| 0 | 3 | 3 |
| 1 | 2 | 2 |
| 2 | 1 | 1 |
| 3 | 0 | 0 |
| 4 | -1 | 7 |
| 5 | -2 | 6 |
| 6 | -3 | 5 |
| 7 | -4 | 4 (own slice) |

---

## 5. Tile-Level Interleaving (Forward/Backward Workers)

To maximize bandwidth, tiles within a slice are **interleaved** between forward and backward workers in an alternating pattern.

**Key mechanism**: Backward workers start with an offset, then both directions alternate processing chunks:

```cpp
// reader.cpp:160-169 - Backward direction starts with offset
if (!direction) {
    uint32_t backwards_offset = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
    for (uint32_t k = 0; k < backwards_offset; ++k) {
        input_pages_read_in_row++;
        // ... advance position ...
    }
    tiles_read += backwards_offset;
}
```

```cpp
// reader.cpp:191-195 - Forward reads half, backward reads remainder
if (direction) {
    tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
} else {
    tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read, tile_granularity);
}
```

**Result for 16 tiles with granularity=4:**
- **Forward workers** process tiles: 0-3, 8-11 (even chunks)
- **Backward workers** process tiles: 4-7, 12-15 (odd chunks)

---

## 6. Synchronization with Semaphores

### 6.1 Out-Ready Semaphore

Writers signal readers on the **next device** that data is ready:

```cpp
// writer.cpp:389-394
uint64_t out_ready_sem_noc_addr_in_pkt =
    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
    fabric_direction_connection,
    pkt_hdr_seminc,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
```

Readers wait on this semaphore:
```cpp
// reader.cpp:183-187
if (do_reduce && (chunk_count % chunks_per_sync == 0)) {
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
    sem_target++;
}
```

### 6.2 Batch-Ready Semaphore

At the end of each batch, a **multicast** semaphore signals all devices:
```cpp
// writer.cpp:461-467
uint64_t batch_ready_sem_noc_addr_in_pkt =
    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, batch_ready_sem, 0);
fabric_multicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
    fabric_direction_connection,
    pkt_hdr_mcastseminc,
    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{batch_ready_sem_noc_addr_in_pkt, 0});
```

---

## 7. Compute Kernel: The Reduction

The compute kernel (`ring_reduction.cpp`) performs element-wise addition:

```cpp
// ring_reduction.cpp:27-76
for (uint32_t b = 0; b < input_tensor_B; b++) {
    // Don't reduce on the first slice
    for (uint32_t i = 0; i < ring_size - 1; i++) {  // Note: ring_size - 1 reductions
        for (uint32_t c = 0; c < slice_C; c++) {
            while (tiles_read < tiles_to_read) {
                cb_wait_front(input_cb_id, tile_granularity);
                cb_wait_front(intermediate_cb, tile_granularity);
                cb_reserve_back(output_cb, tile_granularity);
                acquire_dst();
                for (uint32_t tile_id = 0; tile_id < num_pages_to_read; tile_id++) {
                    add_tiles(input_cb_id, intermediate_cb, tile_id, tile_id, tile_id);
                    pack_tile(tile_id, output_cb);
                }
                release_dst();
                cb_pop_front(input_cb_id, tile_granularity);
                cb_pop_front(intermediate_cb, tile_granularity);
                cb_push_back(output_cb, tile_granularity);
                ...
            }
        }
    }
}
```

Key insight: Only `ring_size - 1` reductions happen because the first iteration (i=0) directly passes data through without reduction.

---

## 8. Data Flow Diagram for Device 4

```
                        DEVICE 4 (Forward Direction)
                        ============================

    Iteration 0 (slice 3):
    ┌─────────────────┐
    │  Local Input    │────────────────────────────────┐
    │   Slice 3       │                                │
    └─────────────────┘                                │
                                                       v
                                              ┌─────────────────┐
                                              │  cb_reader_out  │
                                              │    (no reduce)  │
                                              └────────┬────────┘
                                                       │
                                                       v
                                              ┌─────────────────┐
                                              │    Writer       │──────> To D5
                                              └─────────────────┘


    Iteration 1+ (slice 2, 1, 0, 7, 6, 5):
    ┌─────────────────┐     ┌─────────────────┐
    │  Local Input    │     │  Intermediate   │  (from D3)
    │   Slice N       │     │   Slice N       │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             v                       v
        ┌─────────┐            ┌─────────┐
        │cb_input │            │cb_interm│
        └────┬────┘            └────┬────┘
             │                      │
             └──────────┬───────────┘
                        v
               ┌─────────────────┐
               │ Compute: ADD    │
               └────────┬────────┘
                        v
               ┌─────────────────┐
               │  cb_compute_out │
               └────────┬────────┘
                        v
               ┌─────────────────┐
               │    Writer       │──────> To D5 (intermediate)
               └─────────────────┘


    Iteration 7 (slice 4 - OWN SLICE):
    ┌─────────────────┐     ┌─────────────────┐
    │  Local Input    │     │  Intermediate   │  (accumulated from all 7 devices)
    │   Slice 4       │     │   Slice 4       │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             v                       v
             └──────────┬───────────┘
                        v
               ┌─────────────────┐
               │ Compute: ADD    │  (Final reduction!)
               └────────┬────────┘
                        v
               ┌─────────────────┐
               │  OUTPUT BUFFER  │  (Final result for slice 4)
               └─────────────────┘
```

---

## 9. Accumulation Pattern: How Partial Sums Build Up

This is the key insight to understanding the algorithm. Let's trace **slice 3** as it accumulates around the ring.

### Global View: All Devices at Each Iteration (Forward Direction)

| Iteration | D0 processes | D1 processes | D2 processes | D3 processes | D4 processes | D5 processes | D6 processes | D7 processes |
|-----------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| i=0 | slice 7 | slice 0 | slice 1 | slice 2 | **slice 3** | slice 4 | slice 5 | slice 6 |
| i=1 | slice 6 | slice 7 | slice 0 | slice 1 | slice 2 | **slice 3** | slice 4 | slice 5 |
| i=2 | slice 5 | slice 6 | slice 7 | slice 0 | slice 1 | slice 2 | **slice 3** | slice 4 |
| i=3 | slice 4 | slice 5 | slice 6 | slice 7 | slice 0 | slice 1 | slice 2 | **slice 3** |
| i=4 | **slice 3** | slice 4 | slice 5 | slice 6 | slice 7 | slice 0 | slice 1 | slice 2 |
| i=5 | slice 2 | **slice 3** | slice 4 | slice 5 | slice 6 | slice 7 | slice 0 | slice 1 |
| i=6 | slice 1 | slice 2 | **slice 3** | slice 4 | slice 5 | slice 6 | slice 7 | slice 0 |
| i=7 | slice 0 | slice 1 | slice 2 | **slice 3** | slice 4 | slice 5 | slice 6 | slice 7 |

**Bold** shows where slice 3's partial sum is being processed.

### Tracing Slice 3's Accumulation

| Iteration | Device | What happens | Partial sum after this step |
|-----------|--------|--------------|------------------------------|
| i=0 | D4 | Read local[3], send to D5 | D4[3] |
| i=1 | D5 | Read local[3] + intermediate[3], reduce, send to D6 | D4[3] + D5[3] |
| i=2 | D6 | Read local[3] + intermediate[3], reduce, send to D7 | D4[3] + D5[3] + D6[3] |
| i=3 | D7 | Read local[3] + intermediate[3], reduce, send to D0 | D4[3] + D5[3] + D6[3] + D7[3] |
| i=4 | D0 | Read local[3] + intermediate[3], reduce, send to D1 | D4[3] + D5[3] + D6[3] + D7[3] + D0[3] |
| i=5 | D1 | Read local[3] + intermediate[3], reduce, send to D2 | D4[3] + D5[3] + D6[3] + D7[3] + D0[3] + D1[3] |
| i=6 | D2 | Read local[3] + intermediate[3], reduce, send to D3 | D4[3] + D5[3] + D6[3] + D7[3] + D0[3] + D1[3] + D2[3] |
| i=7 | D3 | Read local[3] + intermediate[3], reduce, **write to output** | **FINAL: SUM of all D[3]** |

**D3 owns slice 3**, so after 7 hops (8-1 iterations), D3 has the fully reduced slice 3 and writes it to output.

### Why This Order Works

The key insight from the code comment (reader.cpp:118):
```cpp
// Loop over the slices, starting from the furthest, and working backwards until we get to ourselves
```

Device 4 processes slices: **3, 2, 1, 0, 7, 6, 5, 4**

- Slice 3 is "furthest" from slice 4 in terms of reduction hops
- Slice 4 (D4's own slice) comes last, after accumulating from all other devices

---

## 10. Bidirectional Operation: Forward + Backward

The algorithm runs **two independent reduction flows** simultaneously:

### Forward Workers (dir=1) on Device 4
- Send to: Device 5
- Receive from: Device 3
- Slice order: 3, 2, 1, 0, 7, 6, 5, **4** (own slice last)

### Backward Workers (dir=0) on Device 4
- Send to: Device 3
- Receive from: Device 5
- Slice order: 5, 6, 7, 0, 1, 2, 3, **4** (own slice last)

```cpp
// reader.cpp:115
int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;
// Forward (dir=1): slice_idx = 4 - 1 = 3, then decrement
// Backward (dir=0): slice_idx = 4 + 1 = 5, then increment
```

### Tile Interleaving Between Directions

Within each slice, tiles are split between forward and backward workers in an **alternating chunk pattern**:

**Backward workers start with an offset** (reader.cpp:160-169):
```cpp
if (!direction) {
    uint32_t backwards_offset = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
    for (uint32_t k = 0; k < backwards_offset; ++k) {
        // ... advance tile position ...
    }
    tiles_read += backwards_offset;
}
```

**Then both directions alternate** (reader.cpp:191-195):
```cpp
if (direction) {
    // Forward: read half of remaining tiles
    tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read / 2, tile_granularity);
} else {
    // Backward: read all remaining tiles (but started offset)
    tiles_to_read_in_current_direction = std::min(tiles_remaining_to_read, tile_granularity);
}
```

**Example**: 16 tiles with `tile_granularity=4`:

| Worker | Initial offset | Chunk 1 | Skip | Chunk 2 | Skip |
|--------|----------------|---------|------|---------|------|
| Forward (dir=1) | none | tiles 0-3 | 4-7 | tiles 8-11 | 12-15 |
| Backward (dir=0) | skip 0-3 | tiles 4-7 | 8-11 | tiles 12-15 | - |

**Result**: Forward handles tiles **0-3, 8-11**. Backward handles tiles **4-7, 12-15**.

Both directions write to the **same output buffer** at their respective tile positions, effectively doubling throughput.

---

## 11. Summary

| Component | File | Responsibility |
|-----------|------|----------------|
| **Reader** | `ring_reduce_scatter_minimal_async_reader.cpp` | Read local input + intermediate buffer, feed compute |
| **Compute** | `ring_reduction.cpp` | Element-wise add tiles |
| **Writer** | `ring_reduce_scatter_minimal_async_writer.cpp` | Send to next device OR write final output |

Key algorithm properties:
- **8 iterations** for ring_size=8 (N iterations for N devices)
- **7 reductions** per device (first iteration has no reduction)
- **Bidirectional** for 2x throughput
- **Pipelined** with semaphore-based synchronization
- **Final iteration** writes to output buffer instead of forwarding

---

## 12. Open Questions / Areas for Clarification

1. **Tile interleaving details**: How exactly are tiles partitioned between forward/backward workers? The `/2` logic suggests even/odd split.

2. **Chunks per sync**: What is the optimal `chunks_per_sync` value and how does it affect latency?

3. **Intermediate buffer sizing**: How is the intermediate buffer sized relative to input slices?

---

*Document prepared for understanding the ReduceScatter ring algorithm implementation.*
