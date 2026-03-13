# Ring Joint SDPA - Circular Buffer (CB) Usage Report

## How to Test & Build
- **Build**: `./build_metal.sh --release`
- **Run test**: `pytest tests/tt_eager/python_api_testing/unit_testing/test_ring_joint_attention_scaled_dot_product_attention_sprint.py::test_ring_joint_attention_sdpa_accuracy`
  - Use a 20s timeout. If it hangs, recover with `tt-smi -r`

## Test Configuration
- **Test**: `test_ring_joint_attention_scaled_dot_product_attention_sprint.py::test_ring_joint_attention_sdpa_accuracy`
- **Config**: wan2.2 compatible, seq_len=2240x4 devices, 10 heads, DH=128, BF16
- **Key Parameters**: `Sq_chunk_t=7`, `Sk_chunk_t=16`, `DHt=4`, `q_buffer_factor=1`
- **Streaming compute**: enabled (`use_streaming_compute=true`)
- **Tile size**: 2048 bytes (BF16 / Float16_b)

## CB Usage Ranked Largest to Smallest

| Rank | CB Index | Name                | Tiles | Tile Size (B) | Total (B) | Total (KB) | Formula                            |
|------|----------|---------------------|------:|---------------|----------:|-----------:|-------------------------------------|
| 1    | c_1      | K input             |   128 | 2048          |   262,144 |        256 | `Sk_chunk_t * DHt * 2` (double-buf) |
| 2    | c_2      | V input             |   128 | 2048          |   262,144 |        256 | `Sk_chunk_t * DHt * 2` (double-buf) |
| 3    | c_24     | QK intermediate     |   112 | 2048          |   229,376 |        224 | `Sq_chunk_t * Sk_chunk_t`           |
| 4    | c_0      | Q input             |    28 | 2048          |    57,344 |         56 | `Sq_chunk_t * DHt * q_buf_factor`   |
| 5    | c_7      | prev block output   |    28 | 2048          |    57,344 |         56 | `Sq_chunk_t * DHt`                  |
| 6    | c_25     | out intermediate A  |    28 | 2048          |    57,344 |         56 | `Sq_chunk_t * DHt`                  |
| 7    | c_26     | out accumulate B    |    28 | 2048          |    57,344 |         56 | `Sq_chunk_t * DHt`                  |
| 8    | c_16     | output              |    28 | 2048          |    57,344 |         56 | `Sq_chunk_t * DHt`                  |
| 9    | c_6      | stats input         |     7 | 2048          |    14,336 |         14 | `Sq_chunk_t`                        |
| 10   | c_27     | cur max A           |     7 | 2048          |    14,336 |         14 | `Sq_chunk_t`                        |
| 11   | c_28     | prev max B          |     7 | 2048          |    14,336 |         14 | `Sq_chunk_t`                        |
| 12   | c_29     | cur sum A           |     7 | 2048          |    14,336 |         14 | `Sq_chunk_t`                        |
| 13   | c_30     | prev sum B          |     7 | 2048          |    14,336 |         14 | `Sq_chunk_t`                        |
| 14   | c_31     | exp max diff        |     7 | 2048          |    14,336 |         14 | `Sq_chunk_t`                        |
| 15   | c_17     | stats output        |     7 | 2048          |    14,336 |         14 | `Sq_chunk_t`                        |
| 16   | c_10     | sum out (streaming) |     7 | 2048          |    14,336 |         14 | `Sq_chunk_t` (streaming only)       |
| 17   | c_11     | sum in (streaming)  |     7 | 2048          |    14,336 |         14 | `Sq_chunk_t` (streaming only)       |
| 18   | c_3      | mask                |     1 | 2048          |     2,048 |          2 | `total_lightweight_mask_tiles`      |
| 19   | c_4      | scale               |     1 | 2048          |     2,048 |          2 | 1 tile                              |
| 20   | c_5      | identity scale      |     1 | 2048          |     2,048 |          2 | 1 tile                              |
| 21   | c_8      | col identity        |     1 | 2048          |     2,048 |          2 | 1 tile                              |
| 22   | c_9      | recip scratch       |     1 | 2048          |     2,048 |          2 | 1 tile (streaming only)             |

## L1 Summary

| Metric              | Value        |
|---------------------|-------------|
| **Total CB usage**  | 1,179,648 B (1,152 KB) |
| **L1 per core**     | 1,572,864 B (1,536 KB) |
| **L1 remaining**    | 393,216 B (384 KB)     |
| **CB % of L1**      | 75.0%       |

## Breakdown by Category

| Category                  | Size (KB) | % of Total CB |
|---------------------------|----------:|--------------|
| K+V input (double-buf)    |       512 | 44.4%        |
| QK intermediate           |       224 | 19.4%        |
| Output-related (c_0,c_7,c_25,c_26,c_16) | 280 | 24.3% |
| Statistics (c_6,c_27-c_31,c_17,c_10,c_11) | 126 | 10.9% |
| Scalars/mask/scratch (c_3,c_4,c_5,c_8,c_9) | 10 | 0.9% |

---

## Deep Investigation: CB Lifetime Analysis & Reduction Techniques

### CB Lifecycle per Kernel (Streaming Path)

```
Timeline per K-chunk iteration within a ring iteration:
====================================================

READER kernel:
  1. cb_reserve_back(cb_k_in)  →  read/receive K tiles  →  cb_push_back(cb_k_in)
  2. [k_chunk==0 only] read Q subblocks into cb_q_in
  3. cb_reserve_back(cb_v_in)  →  read/receive V tiles  →  cb_push_back(cb_v_in)

COMPUTE kernel (sdpa_ring_v2 → sdpa_inner_loop_step):
  Phase 1 (Q@KT):
    - cb_wait_front(cb_q_in)        [Q stays fronted across all K chunks]
    - cb_wait_front(cb_k_in)        [consumed per K chunk]
    - writes QK scores into cb_qk_im via pack_tile
    - computes row-max into cur.max (c_27 or c_28)
    - computes sub_exp + row-sum into cur.sum (c_29 or c_30)
    - cb_pop_front(cb_k_in)         [K consumed]
  Phase 2 (QKT@V + SALAD):
    - cb_wait_front(cb_v_in)        [consumed per K chunk]
    - matmul QK @ V into cur.out (c_25 or c_26)
    - SALAD correction: reads prev.{max,sum,out} + cb_exp_max_diff
    - cb_pop_front(cb_v_in)         [V consumed]
  After last K chunk of last ring_iter:
    - normalize_row_streaming: reads cur.sum, cur.out → writes cb_out (c_16)
  After K-chunk loop:
    - cb_pop_front(cb_q_in)         [Q consumed]
    - Multi Q-chunk save: copy_block prev.out → cb_out (c_16)
                          copy_block prev.max → cb_max_out (c_17)
                          copy_block prev.sum → cb_sum_out (c_10)

WRITER kernel:
  Before K-chunk loop (ring_iter > 0, multi Q-chunk):
    - read_prev_accumulators: DRAM → cb_prev_out (c_7), cb_max_in (c_6), cb_sum_in (c_11)
  After K-chunk loop:
    - Last ring_iter: write_block from cb_out (c_16) to DRAM
    - Non-last, multi Q: write_accumulators from cb_out (c_16), cb_max_out (c_17), cb_sum_out (c_10)
```

### Technique 1: Alias c_7 (prev block out) with c_16 (output) — saves 56 KB

**Observation**: c_7 and c_16 have non-overlapping lifetimes.

- c_7 is written by writer (DRAM load) at the START of a ring iteration (before K-chunk loop)
- c_7 is consumed by compute (copy_block to prev.out) at the START of K-chunk loop
- c_16 is written by compute at the END of the K-chunk loop (either normalize or copy_block)
- c_16 is consumed by writer AFTER compute pushes it

Timeline:
```
  Writer fills c_7 → Compute reads c_7 → [K-chunk loop runs] → Compute fills c_16 → Writer drains c_16
                    ↑ c_7 FREE here                            ↑ c_16 starts here
```

**For single Q-chunk (q_per_core == 1)**: c_7 is NEVER used at all. Accumulators persist in L1 across ring iterations in c_25/c_26/c_27-c_30. The 56 KB for c_7 is completely wasted.

**For multi Q-chunk**: c_7 is consumed before c_16 is produced. They can share the same L1 address.

**Implementation**: Both use `out_df` format and `Sq_chunk_t * DHt` tiles. Use `set_globally_allocated_address` to alias c_16 to c_7's address, or allocate them at the same L1 offset. Alternatively, just use a single CB index for both roles (requires kernel code changes to use same index).

**Risk**: Low. Lifetimes are strictly non-overlapping.

### Technique 2: Alias c_6 (stats input / max_in) with c_31 (exp_max_diff) — saves 14 KB

**Observation**: c_6 and c_31 have non-overlapping lifetimes.

- c_6 (`cb_max_in`) is written by writer (DRAM load) and consumed by compute at the START via `copy_block(cb_max_in, q_prev.max, ...)` — this happens BEFORE the K-chunk loop
- c_31 (`cb_exp_max_diff`) is produced/consumed DURING the K-chunk loop's SALAD correction phase

```
  Writer fills c_6 → Compute copy_block c_6→prev.max → [c_6 FREE] → K-loop uses c_31 repeatedly
```

Both use `im_df`/`stats_df` (Float16_b) and hold `Sq_chunk_t` tiles.

**Implementation**: Same `set_globally_allocated_address` technique.

**Risk**: Low. c_6 is fully consumed (copy_block does cb_wait_front + cb_pop_front) before K-loop starts.

### Technique 3: Alias c_17 (stats output / max_out) with c_11 (sum_in) — saves 14 KB

**Observation**: In the streaming path:

- c_11 (`cb_sum_in`) is written by writer (DRAM load) and consumed by compute at the START via `copy_block(cb_sum_in, q_prev.sum, ...)` — BEFORE the K-chunk loop
- c_17 (`cb_max_out`) is written by compute and consumed by writer at the END via `write_accumulators` — AFTER the K-chunk loop

```
  Writer fills c_11 → Compute copy_block c_11→prev.sum → [c_11 FREE] → ... K-loop ... → Compute fills c_17 → Writer drains c_17
```

Both use `stats_df` (Float16_b) and hold `Sq_chunk_t` tiles.

**Risk**: Low. Strictly sequential lifetimes.

### Technique 4: Alias c_10 (sum_out) with c_6 (stats input) — saves 14 KB

In the streaming path:
- c_6 is consumed at the start (used as `cb_max_in`)
- c_10 is produced at the end (used as `cb_sum_out`)

They don't overlap. Both hold `Sq_chunk_t` stats tiles.

Note: This can be combined with Technique 2 (c_6 ↔ c_31) since c_31's usage during K-loop doesn't overlap with c_10's usage after the K-loop. However, all three (c_6, c_31, c_10) would need to share the same address. That works because:
- c_6 used → freed → c_31 used during K-loop → freed after loop → c_10 used after loop

### Technique 5: Eliminate c_25/c_26 duplication for single Q-chunk — saves 56 KB

**Observation**: In the streaming path with `q_per_core == 1` (common case — our test has it), the accumulators persist in L1 across ring iterations. The `acc_state` alternates between {c_25, c_27, c_29} and {c_26, c_28, c_30} via ping-pong.

However, after the K-chunk loop completes (all K chunks processed), only ONE of prev/cur holds the final result. The other is stale. On the LAST ring iteration, `normalize_row_streaming` reads from `cur.out` and writes to `cb_out` (c_16). Then `cur.max` is popped.

**Key insight**: For the last ring iteration, after normalization writes to c_16, BOTH c_25 and c_26 are freed. Before that, during the K-loop, both are needed simultaneously for the ping-pong.

So c_25/c_26 can't be directly merged. But c_7 (prev block out) could alias with the "inactive" accumulator half if we carefully manage which half is inactive at the right time. This is fragile and risky.

**Alternative approach (simpler)**: Since c_7 is not used in the single Q-chunk case, just don't allocate it when `max_q_per_core == 1`. The program factory already knows this value.

### Technique 6: Reduce K/V to 1.5x buffering — saves 128 KB

**Observation**: In the reader's inner loop, K and V are read sequentially:
1. Read K chunk → forward K → (Q on first iter) → Read V chunk → forward V

The V read starts AFTER the K read/forward completes. Meanwhile, compute does:
1. Wait K → QK matmul (consumes K) → Wait V → attn@V matmul (consumes V)

The QK matmul takes `Sq_chunk_t × Sk_chunk_t × DHt = 7 × 16 × 4 = 448` tile-level operations. During this time, the reader reads V (64 tiles from DRAM). If V read completes during QK matmul, V is ready when compute needs it.

After compute pops K_i, reader can start K_{i+1} (double-buffer slot). But V_i gets popped only after attn@V_i completes. Reader writes V_{i+1} only after starting K_{i+1} (sequential). So there's a natural gap.

**Proposal**: Keep K double-buffered (256 KB) but make V single-buffered (128 KB → save 128 KB).

- K needs double-buffering because reader starts K_{i+1} while compute processes K_i (QK matmul).
- V might not need double-buffering: by the time reader finishes K_{i+1} forward and starts V_{i+1}, compute has already popped V_i (attn@V already ran).

**Detailed pipeline analysis**:
```
Reader: K0 → fwd_K0 → Q → V0 → fwd_V0 → K1 → fwd_K1 → V1 → fwd_V1 → K2 ...
Compute:          wait_K0 → QK0 → wait_V0 → attnV0 → wait_K1 → QK1 → wait_V1 → attnV1 ...
```

V0 is popped after attnV0. Reader starts V1 after fwd_K1. Question: is attnV0 done before fwd_K1 finishes?
- attnV0 = 448 tile ops. fwd_K1 = 64 tiles over NOC/DRAM.
- If fwd_K1 takes less time than attnV0, reader waits for V to be popped → stall.
- If fwd_K1 takes more time than attnV0, V is already free → no stall.

For DRAM reads (non-chain case), K read = ~64 tiles × DRAM latency. This is likely slower than attnV matmul.
For L1 chain forwarding, K receive is very fast → V might not be free yet → stall risk.

**Risk**: Medium. Performance regression in L1-forwarding case. Needs benchmarking.

**Safer variant**: Make V 1.5x buffered (96 tiles = 192 KB, save 64 KB) using a larger-than-single but smaller-than-double buffer. But the CB API doesn't support fractional buffering directly; you'd need `Sk_chunk_t * DHt * 1.5` which only works if tiles divide evenly.

---

## Summary of Savings (sorted by risk: lowest first)

| Priority | Technique | What | Savings | Risk | Kernel Changes? |
|----------|-----------|------|---------|------|-----------------|
| 1 | 5a. Skip c_7 alloc (single Q) | Don't allocate prev_out when `max_q_per_core == 1` | **56 KB** | **None** | Program factory only | **DONE** ✅ |
| 2 | 1. Alias c_7 ↔ c_16 | prev_out ↔ output (for multi Q-chunk) | **56 KB** | **Low** | Program factory only | **DONE** ✅ |
| 3 | 2. Alias c_6 ↔ c_31 | max_in ↔ exp_max_diff | **14 KB** | **Low** | Program factory only | **DONE** ✅ |
| 4 | 3. Alias c_17 ↔ c_11 | max_out ↔ sum_in | **14 KB** | **Low** | Program factory only |
| 5 | 4. Alias c_10 ↔ c_6 | sum_out ↔ max_in (combine with #3) | **14 KB** | **Low** | Program factory only |
| 6 | 6. V single-buffer | Halve V CB | **128 KB** | **Medium** | Program factory + verify perf |
| 7 | 5b. Alias c_7 with inactive accum half | prev_out ↔ c_25 or c_26 | **56 KB** | **High** | Kernel + factory |

### Implementation order recommendation

**Step 1 — Zero risk (priority 1): 56 KB saved**
Just skip allocating c_7 when `max_q_per_core == 1`. The single Q-chunk streaming path never touches c_7. This is a one-line `if` guard in the program factory.

**Step 2 — Low risk (priorities 2-5): +42 KB = 98 KB total**
Alias CBs with non-overlapping lifetimes via `set_globally_allocated_address`. All changes are in the program factory — no kernel modifications needed. Priority 2 (c_7 ↔ c_16) covers the multi Q-chunk case. Priorities 3-5 alias the stats DRAM-load CBs with DRAM-save and K-loop-only CBs.

Note: Priority 1 and 2 together always save 56 KB regardless of Q-chunk count:
- `max_q_per_core == 1`: skip c_7 entirely (save 56 KB)
- `max_q_per_core > 1`: alias c_7 with c_16 (save 56 KB)

**Step 3 — Medium risk (priority 6): +128 KB = 226 KB total**
Make V single-buffered. Requires performance benchmarking — may cause stalls in the L1 chain-forwarding case where K receive is very fast.

**Step 4 — High risk (priority 7): +56 KB = 282 KB total**
Alias c_7 with an inactive accumulator half. Requires kernel changes and careful synchronization of the ping-pong state. Only worth pursuing if more L1 is needed after steps 1-3.

---

## Implementation Notes

### How to alias CBs in tt-metal

Use `set_globally_allocated_address` on the `CircularBufferConfig` to make two CB indices share the same L1 memory. Example pattern:

```cpp
// Allocate c_7 normally
auto c_in7_handle = CreateCircularBuffer(program, core_grid, c_in7_config);

// Make c_16 share c_7's address (they have non-overlapping lifetimes)
auto c_out0_config = CircularBufferConfig(out0_t * out_tile_size, {{tt::CBIndex::c_16, out_df}})
                         .set_page_size(tt::CBIndex::c_16, out_tile_size)
                         .set_globally_allocated_address(*c_in7_handle.globally_allocated_address());
CreateCircularBuffer(program, core_grid, c_out0_config);
```

Or alternatively, allocate a shared buffer and point both configs to it:

```cpp
auto shared_buffer = CreateBuffer(...);  // Allocate L1 buffer manually
auto c_in7_config = CircularBufferConfig(...)
    .set_globally_allocated_address(shared_buffer);
auto c_out0_config = CircularBufferConfig(...)
    .set_globally_allocated_address(shared_buffer);
```

### Prerequisite for aliasing

CBs being aliased must:
1. Have the same size (or the alias target must be >= the aliased CB)
2. Have strictly non-overlapping lifetimes (no concurrent reads/writes)
3. Not have outstanding async operations when the other starts using the memory
