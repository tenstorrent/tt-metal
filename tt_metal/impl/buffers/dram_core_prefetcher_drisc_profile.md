# DRAM-Core Prefetcher — DRISC Inner-Loop Profile

Profiling the DRISC prefetcher kernel's inner loop using the watcher ring
buffer + RISCV wall-clock cycle counter, with all other watcher features
disabled so the kernel runs at production speed.

## How the instrumentation works

The watcher ring buffer (`tt_metal/hw/inc/api/debug/ring_buffer.h`) gives every
kernel a 32-entry circular buffer in the watcher mailbox region of L1 that the
host can dump asynchronously into `generated/watcher/watcher.log`. Each entry
is a `uint32_t` of arbitrary meaning.

We use it to ship per-stage cycle counts out of the DRISC kernel without
serializing the inner loop on DPRINT.

### Header includes

```cpp
#include "api/debug/ring_buffer.h"           // WATCHER_RING_BUFFER_PUSH
#include "internal/tt-1xx/risc_common.h"     // get_timestamp_32b
```

`WATCHER_RING_BUFFER_PUSH(x)` compiles to a single push only when
`WATCHER_ENABLED && !WATCHER_DISABLE_RING_BUFFER` is set; otherwise it's a no-op
macro. `get_timestamp_32b()` reads the lower 32 bits of the RISC-V wall-clock
register (`RISCV_DEBUG_REG_WALL_CLOCK_L`) — one memory-mapped register read.

### Don't push every iter — accumulate

The ring buffer only holds 32 entries. The DRISC inner loop runs `num_blocks`
times (typically 16+), and we want timing for ~5 named stages per iter — that's
80+ values, which would wrap and we'd see only the last fraction. **Accumulate
in local registers, push the totals once at end of stream.**

```cpp
uint32_t cycles_issue = 0;
uint32_t cycles_wait = 0;
uint32_t cycles_reserve = 0;
uint32_t cycles_push = 0;
uint32_t cycles_flush = 0;

for (uint32_t b = 0; b < num_blocks; ++b) {
    const uint32_t t0 = get_timestamp_32b();
    // ... issue DMA(b+1) ...
    const uint32_t t1 = get_timestamp_32b();
    // ... wait DMA(b) ...
    const uint32_t t2 = get_timestamp_32b();
    // ... remote_cb_reserve_back ...
    const uint32_t t3 = get_timestamp_32b();
    // ... remote_cb_push_back_and_write_pages ...
    const uint32_t t4 = get_timestamp_32b();
    // ... noc_async_posted_writes_flushed ...
    const uint32_t t5 = get_timestamp_32b();

    cycles_issue += (t1 - t0);
    cycles_wait += (t2 - t1);
    if (b > 0) {  // skip iter 0's reserve -- matmul cold-start inflates it
        cycles_reserve += (t3 - t2);
    }
    cycles_push += (t4 - t3);
    cycles_flush += (t5 - t4);
}
```

Skip iter 0's `remote_cb_reserve_back` measurement: the receiver
(matmul reader) hasn't reached its first `remote_cb_wait_front` yet, so the
sender's `reserve_back` for block 0 includes the matmul's startup latency, not
the steady-state reserve cost.

### Self-labelled output

Tag each total in the upper 8 bits so the dump is decode-without-a-key:

```cpp
WATCHER_RING_BUFFER_PUSH(0xAA000000u | (cycles_issue   & 0x00FFFFFFu));
WATCHER_RING_BUFFER_PUSH(0xBB000000u | (cycles_wait    & 0x00FFFFFFu));
WATCHER_RING_BUFFER_PUSH(0xCC000000u | (cycles_reserve & 0x00FFFFFFu));
WATCHER_RING_BUFFER_PUSH(0xDD000000u | (cycles_push    & 0x00FFFFFFu));
WATCHER_RING_BUFFER_PUSH(0xEE000000u | (cycles_flush   & 0x00FFFFFFu));
WATCHER_RING_BUFFER_PUSH(0xFF000000u | (num_blocks     & 0x00FFFFFFu));
```

The 24-bit cycle range tops out at ~16M cycles ≈ 16ms at 1 GHz — plenty for a
16-block loop that takes ~5us total.

### Run with watcher ringbuf-only

Standard `TT_METAL_WATCHER=1` turns on every check (assert sanitization, NoC
sanitization, stack tracking, ...) which inflates per-iter cost massively. For
profiling we want **only** the ring buffer:

```sh
TT_METAL_WATCHER=1 \
TT_METAL_WATCHER_DISABLE_ASSERT=1 \
TT_METAL_WATCHER_DISABLE_PAUSE=1 \
TT_METAL_WATCHER_DISABLE_STACK_USAGE=1 \
TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1 \
TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1=1 \
TT_METAL_WATCHER_DISABLE_SANITIZE_WRITE_ONLY_L1=1 \
TT_METAL_WATCHER_DISABLE_WAYPOINT=1 \
TT_METAL_WATCHER_DISABLE_DISPATCH=1 \
TT_METAL_WATCHER_DISABLE_CB_SANITIZE=1
```

The watcher server polls cores every ~1s and dumps each core's ring buffer to
its log entry in `generated/watcher/watcher.log`. The test must **sleep a few
seconds after `synchronize_device`** so the watcher's polling thread sees the
final state before the device is closed:

```python
ttnn.dram_prefetcher(...)
ttnn.linear(...)
ttnn.synchronize_device(device)
time.sleep(3)  # let watcher dump the ring buffer
```

Without the sleep the device closes before the polling thread reads the
final ring-buffer state and you get stale or empty entries.

### Decoding

Watcher prints ring buffer entries newest-first. With our 6-push tail, an
example DRAM bank dump:

```
debug_ring_buffer=
[0xff000010,0xee000120,0xdd00098b,0xcc0002ea,0xbb000248,0xaa0003bf]
 ^num_blocks ^flush    ^push      ^reserve   ^wait       ^issue
```

Cycle counts (hex → decimal):
| Tag  | Stage         | Hex      | Total cycles | Iters | Avg/iter |
|------|---------------|----------|-------------:|------:|---------:|
| 0xAA | issue DMA     | 0x3bf    | 959          | 16    | 60       |
| 0xBB | wait DMA      | 0x248    | 584          | 16    | 37       |
| 0xCC | reserve_back  | 0x2ea    | 746          | 15    | 50       |
| 0xDD | push pages    | 0x98b    | 2443         | 16    | **153**  |
| 0xEE | flush writes  | 0x120    | 288          | 16    | 18       |
| 0xFF | num_blocks    | 0x10     | (= 16)       | -     | -        |
| —    | **per-iter sum** | —    | 5020         | -     | **318**  |

(Times in cycles ≈ ns at the BH 1 GHz wall clock.)

## What changed since the prior (pre-2026-05) profile

- The kernel was refactored: it no longer calls
  `remote_cb_push_back_and_write_pages`. The push loop is now a custom
  helper `prefetcher_write_chunk`, and the per-block credit-update is
  its own `prefetcher_finalize_block<skip_ptr_update=true>` (see
  `tt_metal/impl/buffers/kernels/dram_core_prefetcher.cpp:49,99`).
- "Iter 0" no longer exists as a single entity — the kernel iterates
  `(layer × tensor × block × sub_band × M)` chunks. We accumulate over
  every chunk in every layer (no cold-start exclusion); the trace
  warmup layer (`num_layers = trace_repeats + 1`) lets us start counting
  immediately.
- Workload moved from the old matmul-bench `M=32 K=512 N=1024 bf16
  ring=8` to the current BW-bench Llama shapes at `ring=64 bf8_b` (see
  `docs/dram_core_prefetcher_bw_results.md`).
- The receiver is now `dram_prefetcher_consumer`
  (`wait_front(1)/pop_front(1)` in a tight loop), not a matmul.

## Results — DRAM-core prefetcher, BW bench, ring=64 bf8_b

Workload: `test_bw_dram_core_prefetcher` from
`test_prefetcher_BH_bw_bench.py` at `BENCH_TRACE_REPEATS=20`,
`num_prefetch_layers=21` (1 warmup + 20 traced). Receiver is the BW-bench
discard kernel.

Two GCB sizes profiled to isolate receiver backpressure:
- **Small GCB**: `4 × page_bytes_per_recv` (original bench default).
- **Symmetric GCB**: `pages_per_layer × page_bytes_per_recv` (matches
  what the worker-core bench passes to `create_global_circular_buffer`).

Per-chunk cycle counts (averaged over `total_chunks`):

| Stage | Tag | 1B_WO (small / sym) | 70B_QKV_8d (small / sym) | 3B_FF1 (small / sym) |
|---|---|---:|---:|---:|
| total chunks | 0xFF | 1 344 | 1 344 | 2 688 |
| issue DMA setup | 0xA1 | 85 / 85 | 85 / 85 | 99 / 99 |
| wait DMA | 0xA2 | 25 / 25 | 34 / 34 | 29 / 29 |
| **`remote_cb_reserve_back`** | 0xA3 | **10 326 / 196** | **7 873 / 196** | **5 373 / 100** |
| `prefetcher_write_chunk` | 0xA4 | 810 / 810 | 1 867 / 1 867 | 917 / 912 |
| flush | 0xA5 | 33 / 33 | 33 / 33 | 31 / 31 |
| `prefetcher_finalize_block` | 0xA6 | 497 / 492 | 497 / 492 | 248 / 246 |
| **per-chunk total** | | **11 776 / 1 641** | **10 389 / 3 517** | **6 697 / 1 409** |
| trace_elapsed (ms) | | 1.63 / 1.55 | 2.62 / 2.50 | 2.80 / 2.73 |
| aggregate BW (GB/s) | | 54.6 / 57.7 | 136.2 / 142.6 | 254.9 / 261.1 |

Per-shape geometry (recap from
`dram_core_prefetcher_manager.cpp:compute_tensor_geom`):

- **1B_WO**: `k_block_w_tiles=1`, Case 1 fit, 1 chunk/block, 1 row × 1
  page × 8 receivers = 8 NoC writes/block, per-receiver page = 1 088 B.
- **70B_QKV_8d**: `k_block_w_tiles=4`, Case 1 fit, 1 chunk/block,
  4 rows × 1 page × 8 receivers = 32 writes/block, page = 4 352 B.
- **3B_FF1**: `k_block_w_tiles=2`, Case 2 fit (ring_half=35 984 B; one
  row = 34 816 B), 2 chunks/block, 1 row × 4 pages × 8 receivers per
  chunk = 32 writes/chunk = 64 writes/block, page = 8 704 B.

Per-shape geometry (recap from
`dram_core_prefetcher_manager.cpp:compute_tensor_geom`):

- **1B_WO**: `k_block_w_tiles=1`, Case 1 fit, 1 chunk/block, 1 row × 1
  page × 8 receivers = 8 NoC writes/block, per-receiver page = 1 088 B.
- **70B_QKV_8d**: `k_block_w_tiles=4`, Case 1 fit, 1 chunk/block,
  4 rows × 1 page × 8 receivers = 32 writes/block, page = 4 352 B.
- **3B_FF1**: `k_block_w_tiles=2`, Case 2 fit (ring_half=35 984 B; one
  row = 34 816 B), 2 chunks/block, 1 row × 4 pages × 8 receivers per
  chunk = 32 writes/chunk = 64 writes/block, page = 8 704 B.

## Takeaways

1. **`reserve_back` was *not* the bottleneck despite looking like 75–
   88% of the per-chunk cycle budget on the small GCB.** Shrinking the
   GCB to 4 pages (vs the worker's 64-page-per-receiver buffer) made
   the sender spin in `reserve_back` for ~10 000 cyc per chunk, but
   that spin time was almost entirely overlapped by other ops on the
   chip. Growing the GCB to symmetric (64 pages) reduced reserve_back
   by ~50× per chunk but only improved aggregate BW by 1–7% across the
   16 Llama shapes. **Conclusion: the original "receiver-bound" reading
   of the profile was an artifact of the spin happening in a
   serially-instrumented path — not a true wall-clock bottleneck.**

2. **The actual wall-clock bottleneck is push (`prefetcher_write_chunk`)**:
   it's the dominant per-chunk stage by far once `reserve_back` is
   normal-sized.
   - 1B_WO: 8 writes/chunk → 810 cyc → 101 cyc/write.
   - 70B_QKV_8d: 32 writes/chunk → 1 867 cyc → 58 cyc/write (better
     amortization of fixed per-chunk cost).
   - 3B_FF1: 32 writes/chunk → 912 cyc → 29 cyc/write (best — chunks
     share state across the K-row).
   Fixed per-chunk cost is roughly ~270 cyc (write `set_state` + loop
   prelude); per-receiver per-page cost is ~30 cyc once warm.

3. **`finalize_block` is non-trivial: ~250–500 cyc/block.** It's an
   8-receiver loop with one posted `noc_semaphore_inc` each. Multicast
   or batching the increments could halve it. ~15–25% of cycles on the
   symmetric-GCB profile.

4. **DMA pipeline is healthy.** Wait (25–34 cyc/chunk) is dwarfed by
   push (810–1 867); ping-pong depth=2 hides DMA latency completely on
   the bench shapes. Triple-buffering remains pointless here.

5. **`flush` is ~free (~33 cyc/chunk).**

6. **Per-chunk wall-clock from stage sums doesn't match
   `trace_elapsed`**: per-chunk 1 641 cyc × 1 344 chunks = 2.2 ms but
   `trace_elapsed` reports 1.55 ms (for 1B_WO symmetric). The kernel
   on each of the 8 banks runs in parallel, and `trace_elapsed`
   measures only the *consumer*-side trace replay window (the
   prefetcher daemon ran before/around it). The per-chunk numbers are
   still the correct unit for *relative* stage cost; the absolute
   sum-of-stages doesn't equal wall-clock.

## Recommendation

`prefetcher_write_chunk` is the dominant cost and the candidates from
the prior plan remain valid:

- **Dual-NoC receiver writes** — DRISC has 2 NIUs
  (`tt_metal/hw/inc/experimental/drisc_mode.h:70-95`), kernel uses only
  one. Splitting `num_receivers` across NoC0/NoC1 could overlap the
  8-receiver fanout across both NIUs. **Highest expected gain** for the
  push-dominated shapes (1B_WO, 70B_QKV_8d).
- **NoC `set_state` reuse** — `noc_async_write_one_packet_set_state`
  fires per receiver per chunk
  (`kernels/dram_core_prefetcher.cpp:67`); identical destination
  patterns across the M-chunks of a block could share state.
- **`finalize_block` multicast** — batch the per-receiver pages_sent
  increments into a single multicast inc. Lower priority (only 15–25%
  of cycles) but cheap to try.

Reserve_back is **not** worth optimizing on the kernel side; the
symmetric-GCB fix in the bench is the right place to address it.

## Reproducing

The instrumentation is checked in to
`tt_metal/impl/buffers/kernels/dram_core_prefetcher.cpp` behind a
`WATCHER_ENABLED && !WATCHER_DISABLE_RING_BUFFER` macro guard, so it's
zero-cost in production builds.

1. `~/bin-metal/build_metal.sh`.
2. Set the watcher-ringbuf-only env vars (see env block above; also
   committed in `out/bench_env` during the profiling window).
3. `~/bin-metal/tt-smi-reset`.
4. `rm -f generated/watcher/watcher.log` (fresh log).
5. Run a single-shape BW-bench invocation:
   ```bash
   ~/bin-metal/run_bench.sh 'tests/.../test_prefetcher_BH_bw_bench.py::test_bw_dram_core_prefetcher[1B_WO-...]'
   ```
   The BW bench's `time.sleep(3)` after `synchronize_device` (gated on
   `TT_METAL_WATCHER=1`) lets the watcher polling thread dump the ring
   buffers before the device closes.
6. Decode:
   ```bash
   grep -A1 "debug_ring_buffer" generated/watcher/watcher.log | grep "0xff" | head -3
   ```
   Each dump is `[0xff<chunks>, 0xa6<finalize>, 0xa5<flush>, 0xa4<push>,
   0xa3<reserve>, 0xa2<wait>, 0xa1<issue>]` — newest-first within the
   ring buffer, top byte is the stage tag, low 24 bits are cycle totals.

## File references

- Watcher ring buffer macro: `tt_metal/hw/inc/api/debug/ring_buffer.h`
- Wall-clock register access: `tt_metal/hw/inc/internal/tt-1xx/risc_common.h:247`
- Watcher dump (host-side, log writer):
  `tt_metal/impl/debug/watcher_device_reader.cpp:857` (`DumpRingBuffer`)
- DRISC kernel (current 2-buffer pipeline, pre-instrumentation):
  `ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/dram_core_prefetcher.cpp`
- Watcher env-var registry: `tt_metal/llrt/rtoptions.cpp:155–166`

---

## 2026-05-30 — Receiver-contiguous kernel (triple-buffer + dynamic batching)

Re-profiled the **current** recv-contig kernel (3-slot ping-pong stage, dynamic
per-receiver batching, deferred flush). Tags now: `A1` rounds, `A2` chunks,
`A3` poll, `A4` dma_iss (pure DMA-issue), `A5` dma_wait, `A6` set_state,
`A7` writes, `A8` final per-round flush, `A9` finalize, `AA` deferred in-loop
flush (NoC-write drain before slot reuse), `B0` stage_ring_size, `B1` stage_third.

Run with `BENCH_TRACE_REPEATS=20` (21 layers) — **important**: the prior
100-layer run silently **wrapped the 24-bit cycle counters**, which is what
produced the bogus "2 fast banks / 6 slow banks, 10× spread" reading in the
older notes. At 21 layers nothing wraps and the **per-bank spread is only
1.1–1.2×** — the banks are uniform; there is no GDDR-saturation outlier.

Per-chunk cycles (typical bank; wall clock ≈ ns at 1.35 GHz, profiling overhead
inflates absolutes — use ratios):

| Shape       | n_per_recv | BW (GB/s) | dma_iss | defer_fl | dma_wait | writes | bound by |
|-------------|-----------:|----------:|--------:|---------:|---------:|-------:|----------|
| 8B_FF1_1d   | 7 tiles    |   ~306    |   319   |   188    |    32    |   55   | DMA-read + NoC-drain (serialized) |
| 3B_FF1      | 4 tiles    |   ~305    |   208   |    78    |    31    |  213   | mixed (DMA + writes) |
| 1B_WO       | 1 tile     |   ~189    |   140   |    13    |    36    |  340   | NoC write-enqueue |
| 70B_QKV_8d  | 1 tile     |   ~281    |   143   |    13    |    27    |  339   | NoC write-enqueue |

Amortized per-round stages (divide by chunks/round): poll ~168/round,
set_state 23/visit, final-flush 120–350/round, finalize 440–497/round.

### Two regimes

1. **Large `n_per_recv` (≥4 tiles), big chunks** → bound by the DMA read
   (`dma_iss` blocks on DMA-engine acceptance) **plus** the deferred NoC-write
   drain (`defer_fl`), and the two are **serialized** in the loop: per chunk we
   pay `defer_fl` (drain gc-2's writes) → `dma_iss` (issue gc+1, blocks) →
   `dma_wait` → `writes` (enqueue gc). The DMA read of gc+1 does **not** overlap
   the NoC writes of gc, so we pay both. These are the shapes where worker-core
   wins (its ~1.5 MB L1 lets it buffer deep enough to overlap).

2. **`n_per_recv` = 1 tile, tiny blocks** → a chunk batches ~16 one-tile blocks
   = ~16 single-tile NoC packets; `writes` (~340 cyc, ~20 cyc/packet of
   `with_state` overhead) dominate and `defer_fl` is ~13 cyc (little data to
   drain). DMA is cheap. These are the shapes where recv-contig **beats** worker
   (worker is even worse at tiny per-receiver blocks).

### Actionable opportunity (not yet implemented)

On the large-N regime, `dma_iss` + `defer_fl` (~500 cyc) are both *waiting*
stages and they serialize. Reordering so chunk gc's posted NoC writes are
enqueued **before** issuing the DMA for gc+1 would let the ~188 cyc write-drain
overlap the ~319 cyc DMA read (instead of summing). Rough ceiling: ~560 → ~375
cyc/chunk, i.e. ~306 → ~410 GB/s on 8B_FF1_1d — which would clear worker-core
(390). The tiny-block regime wouldn't benefit (write-bound, not drain-bound);
there the lever is fewer/larger NoC packets (coalesce across blocks).

Reproduce: same watcher-ringbuf-only env block as above, `BENCH_TRACE_REPEATS=20`,
single-shape `test_bw_dram_core_prefetcher_recv_contig[<shape>-...]`, then
`python3 /tmp/decode_prof.py generated/watcher/watcher.log`.
