# r02-b02-a01: path-aware dual-NoC output drain issued by BRISC itself (DM_DYNAMIC_NOC): only "short eastward" DRAM destinations, every other bank visit, go out on NoC0

## Motivation
The output drain is the largest exposed cost after the AG on the round root (r01-b04-a04). Medians over measured
calls (analysis/tails.py on reports/r01-b04-a04, devices 0-1, µs relative to the last W_AGWAIT end):

| shape | POST (AG -> TRISC end) | last drain end - AG | effective write rate |
|---|---|---|---|
| h3584 | 3.98 | 5.75-5.95 | ~200 GB/s |
| h4096 | 4.23 | 6.44-6.49 | ~202 GB/s |
| h6144 | 5.30 | 8.47-8.49 | ~232 GB/s |
| h7168 | 5.81 | 9.36-9.47 | ~245 GB/s |

The drain runs 1.8-3.6 µs past the end of compute. Reads in the same op reach ~420 GB/s.

**The drain also delays the next call.** analysis/gap.py (h7168, dev 1, call 58 vs 57) shows the cores whose drain
ended last in call N (x=5..11) start call N+1 up to 1.4 µs after the earliest core. A core's BRISC-FW zone restarts
~1.1 µs after its kernel ends, then the kernel starts ~0.7 µs later. Their input read, PRE and stick push all shift by
that amount, and F_COLLECT waits for them. So a late drain costs twice: the tail of call N and the start skew of N+1.

**Correction to earlier reflections:** on Blackhole `preferred_noc_for_dram_read` is NOC_0 and
`preferred_noc_for_dram_write` is NOC_1 (tt_metal/api/tt-metalium/kernel_types.hpp). So the reader (NCRISC) is on NoC0
and the writer (BRISC) drains on **NoC1**. r01-b02-a04 and r01-b03-a04 called these the other way round. Their "NoC1
half" was really NoC0.

**Link-load model (analysis/nocmodel.py, rules.py):**
- Inputs: physical NoC0 worker coords from the profiler (rows y=2,3; x=1..7,10,11,13,14), the BH soc-descriptor DRAM
  endpoints per NoC (dram_views), NoC0 routing +x then +y, and NoC1 routing -y then -x (the reverse of NoC0).
- Reads on NoC0 load the hottest link at 4.25 B (B = bytes per core). That gives ~90 GB/s per link at the measured read
  time, so the model is consistent.
- All-NoC1 writes (today) also load the hottest link at 4.25 B. The hot links are the westward row-1/4/7/10 links next
  to the x=0 and x=9 DRAM columns, which carry the wrap-around traffic of the cores on the far side.
- All-NoC0 writes would be 9 B, because the NoC0 endpoints sit in the workers' own rows. Most of that comes from
  left-half cores wrapping east through x=8..16 to the x=0 column. This matches r01-b02-a04's measured disaster: its
  NoC0 half was worst on the left cores.
- Sending only the **short eastward** destinations on NoC0, at half of their visits, cuts the hottest link to 3.0 B
  (-29%). Short eastward means x=9 banks (4-7) for left-half cores and x=0 banks (0-3) for right-half cores, i.e.
  1/4 of all tiles. Sending those destinations 100% on NoC0 overloads NoC0 (5.25 B).

## Mechanism
- Factory (`dit_fused_distributed_rmsnorm_program_factory.cpp`):
  - On the AG path (worker writer), on Blackhole, with a DRAM-interleaved output, put the worker reader and the
    worker writer kernels in `NOC_MODE::DM_DYNAMIC_NOC`. Both are required: all DM kernels on a core must share the
    noc mode. Each keeps its current default NoC (reader NoC0, writer NoC1).
  - Pass a new writer CT arg `dual_noc_drain`.
- Worker writer (`dit_rmsnorm_fused_worker_writer.cpp`), W_DRAIN:
  - For each output tile, `bank = page_id % NUM_DRAM_BANKS`.
  - The DRAM column is x=0 for banks 0-3 and x=9 for banks 4-7 (BH channel == bank).
  - The core is "left" if its NoC0 x < 9.
  - Write the tile on a second `Noc noc0(0)` iff the bank is in the short-eastward column for this core and
    `(page_id / NUM_DRAM_BANKS)` is even. Otherwise use the default NoC1 as today.
  - Before each block pop, flush both NoCs. The final write barrier covers both NoCs.
- Nothing else changes: same data, same compute, same AG protocol.

## Why this is not a repeat
- r01-b02-a04 (0.985, same 20-core layout) and r01-b03-a04 (+3%, 80-core column split) both moved a position-blind
  50/50 share of the drain to NCRISC on NoC0, with a cross-RISC drain_sem handshake. On this layout that sent the left
  cores' x=0-column writes on NoC0's long eastward wrap through rows 2/3, which the model rates the worst path.
- This node chooses the NoC per tile by destination:
  - Only the destinations whose NoC0 path is short go on NoC0, and only half of their visits (25% of tiles).
  - The far destinations stay on NoC1.
  - BRISC issues both NoCs itself in dynamic-NoC mode, so there is no NCRISC handshake.
- This is the "position-aware split, pick the NoC by the destination bank's column" that r01-b02-a04's reflection
  listed as the only dual-NoC variant worth trying. It was never tried.

## Expected effect and risk
- If the drain is NoC-link-bound (the x-graded drain ends suggest it is), the post-POST tail should shrink by ~1-2 µs at
  h6144/h7168 and ~0.5-1 µs at h3584/h4096.
- The next call's start skew should shrink too, because fewer cores drain late.
- Expected score ~1.25-1.30.
- Risks:
  - Dynamic-NoC mode adds L1-counter overhead on every NoC call in the reader (trid reads) and the writer, which could
    slow R_INPUT slightly.
  - If the model is wrong, the NoC0 share could congest row 2/3 as in r01-b02-a04. That would show as late left-half
    cores in W_DRAIN.
  - Correctness should be unaffected: same bytes and addresses, only the NoC changes. Both NoCs are flushed before
    each pop and barriered at kernel end, so a hang or PCC loss would point at the dynamic-mode plumbing.
- Judge with analysis/tails.py ("drainmax-AG" per shape) and analysis/startspread.py (per-core kstart and drain ends).
