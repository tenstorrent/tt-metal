# Saturating DRAM bandwidth on Blackhole

This document describes how to saturate DRAM read bandwidth on Tenstorrent
Blackhole cards, reaching **~98% of the theoretical peak (500 GB/s of 512 GB/s)**
with a simple one-reader-per-bank kernel. It is the Blackhole companion to
[Saturating_DRAM_bandwidth.md](Saturating_DRAM_bandwidth.md), which covers the
original Wormhole/Grayskull work; the techniques (one reader per bank,
transaction-id double-buffering, bank-adjacent reader placement) carry over, but
the Blackhole numbers and the relative importance of each lever differ.

## Blackhole DRAM at a glance

| Property | Blackhole (p150) | Wormhole b0 (ref) |
|---|---|---|
| DRAM banks | 8 (GDDR6) | 12 |
| NOC subchannel endpoints per bank | 3 | 1 |
| Theoretical peak DRAM BW | **512 GB/s** | 384 GB/s |
| Core clock (AICLK) | ~1.35 GHz | ~1.0 GHz |
| NOC payload width | 512 bits (64 B/word) | 256 bits (32 B/word) |
| `NOC_MAX_BURST_SIZE` (max single packet) | **16 KB** (256 × 64 B) | 8 KB |

Two consequences of the wider Blackhole NOC matter for this workload:

- The **maximum single-packet read is 16 KB** (8 bf16 tiles), so each NOC
  transaction moves twice the payload of a Wormhole packet.
- The **per-bank ceiling** is `512 GB/s ÷ 8 banks = 64 GB/s`, i.e.
  `64e9 ÷ 1.35e9 ≈ 47.4 bytes/cycle` per reader core. Hitting peak means each of
  the 8 reader cores must sustain ~47 B/cyc from its bank.

## The three levers

A reader kernel that issues NOC reads to DRAM and discards them is limited by
three things. On Blackhole, in increasing order of impact:

### 1. Packet size — read the maximum burst

Issue reads of `NOC_MAX_BURST_SIZE` (16 KB) rather than per-tile (2 KB) reads.
Fewer, larger transactions amortize NOC issue overhead. A reader that streams
the bank's contiguous region in 16 KB packets is the baseline; the packet count
is rounded up so the final packet may over-read past the region end (harmless —
every byte is discarded).

### 2. Buffering depth — double-buffer with transaction ids

A naive reader issues a read, waits on a barrier, then issues the next — the
DRAM bank goes idle between the barrier and the next request. The fix (same as
the Wormhole report) is to tag each in-flight read with a NOC **transaction id
(trid)** and barrier on a trid only once a ring of N reads is outstanding, so
there is always a request in flight.

On Blackhole this lever saturates almost immediately. Sweeping the number of
outstanding transactions (`NUM_TRIDS`) on a 128 MiB read:

| outstanding reads | GB/s | utilization |
|---|---|---|
| 1 (single-buffer, barrier every read) | 186 | 36% |
| 2 (double-buffer) | 344 | 67% |
| 3, 4, 6, 8, 12, 15 | 344 | 67% |

**Double-buffering (depth 2) captures the entire latency-hiding benefit;**
depths 2–15 are byte-identical. Going deeper neither helps nor hurts — there is
no over-issue penalty up to the 15-trid hardware limit (`NOC_MAX_TRANSACTION_ID
= 0xF`), but there is also no reason to pay the extra L1. There is no need to
triple- or quadruple-buffer.

### 3. Reader core placement — the dominant lever on Blackhole

This is what gets you from 67% to ~98%. The reader core for each bank must be
placed **adjacent to that bank for the chosen NOC**, so that DRAM responses
travel one hop and reader routes do not overlap and congest the NOC. Arbitrary
placement (e.g. the first 8 worker cores) leaves long, overlapping routes and
caps throughput around 67%.

tt-metal exposes the optimal assignment directly:

```cpp
// One bank-adjacent worker core per DRAM bank, optimal for NOC0 reads.
std::vector<CoreCoord> cores =
    device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0);
```

`cores[bank_id]` is the worker core to run the reader for `bank_id` on. The
assignment differs for `NOC::NOC_1` (the two NOCs route in opposite directions,
so each bank's nearest core differs per NOC).

## Measured results

Microbenchmark: read a 8192×8192 bf16 (128 MiB) DRAM-interleaved tensor, one
reader core per bank (8 cores), discarding the data. Timed with the on-device
profiler (`DRAM_READ` kernel zone), `bytes ÷ slowest-core-cycles × AICLK`.
Blackhole p150, 8 banks, 1.35 GHz, peak 512 GB/s.

| stage | per-core B/cyc | aggregate B/cyc | GB/s | utilization |
|---|---|---|---|---|
| per-tile reads, barrier per block | 17.4 | 139 | 188 | 37% |
| 16 KB packets + trid double-buffer, arbitrary cores | 31.8 | 255 | 344 | 67% |
| **+ bank-adjacent optimal NOC0 placement** | **46.3** | **371** | **500** | **~98%** |

At the final stage each reader sustains 46.3 B/cyc against the ~47.4 B/cyc
per-bank ceiling — the banks themselves are the limit, and a single NOC already
reaches ~98% of peak.

## Takeaways

- **Core placement dominates on Blackhole.** With bank-adjacent NOC0 placement a
  single NOC reaches ~98% of peak; without it you are stuck near 67% no matter
  how you tune buffering or packet size. Always use
  `get_optimal_dram_bank_to_logical_worker_assignment`.
- **Double-buffering is enough.** Two outstanding transactions per core fully
  hide DRAM latency; more buffering is wasted L1.
- **Use the 16 KB max packet.** Larger transactions amortize issue overhead.
- **A second NOC is unnecessary for reads.** Since one NOC already reaches ~98%,
  adding NOC1 readers can only chase the last ~2%; the per-bank DRAM rate, not
  the NOC, is the binding constraint.

A reproducible end-to-end example (the `bh_dram_read` op, build/run/measure
steps, and the tuning sweep) is in
[bh_dram_read_microbenchmark.md](bh_dram_read_microbenchmark.md).
