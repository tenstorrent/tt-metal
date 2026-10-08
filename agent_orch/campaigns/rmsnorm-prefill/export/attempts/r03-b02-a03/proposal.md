# r03-b02-a03: two NoC command buffers in flight for the output drain: BRISC alternates each NoC's write cmd buf (0) and its idle read/atomic cmd buf (1), so the next 2 KB tile is issued while the previous one is still injecting

## Motivation
The drain is the post-AG tail on every shape and it is **per-core-bound**:
- r03-b04-a02: a wave of 10 writers alone on the chip drained at the same per-core rate (~7-8 tiles/µs incl. startup)
  as 20 writers together. So neither DRAM nor shared links set the per-core rate there.
- Parent r03-b02-a02 report (`/tmp` script, now `drain.py` in this dir): at h3584 all 24 worker cores drain at
  7.8-8.2 tiles/µs (W_DRAIN zone), with no position gradient. Measured from the first POST unpack
  (drain_e - C_POST start, TRISC_0) it is 2.94 µs for 28 tiles (~0.28 µs for the first block), so ~10.5 tiles/µs
  = ~95 ns = ~128 cycles per 2 KB tile = ~16 B/cycle per core. The pack thread produces POST tiles at 14 tiles/µs
  (T2 C_POST 1.97 µs / 28 tiles), so compute is not the limit; output_cb holds 2 rows.
- The drain issues every tile through ONE command buffer per NoC (`write_cmd_buf` = 0 under DM_DYNAMIC_NOC). A NoC
  command buffer stays busy while its request is being injected / back-pressured (blackhole noc.h:
  "noc_command_ready: no pending request that is being backpressured by the NOC"). So per tile the core serializes:
  wait cmd buf ready (previous 2 KB = 33 flits injected) -> BRISC address math + 2 L1 counter RMWs + 7 register
  writes -> NIU request setup -> injection. ~33 flits of injection plus ~60-90 cycles of issue/setup per tile fits the
  ~128 cycles/tile we see. Only ~25% of the tiles (the parent's NoC0 share) can overlap with a NoC1 tile today.

## Mechanism
Worker writer only (`device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`), W_DRAIN, when the kernel runs in
DM_DYNAMIC_NOC (`dual_noc_drain`, which is the campaign config on BH):
1. BRISC owns cmd bufs 0 (writes) and 1 (reads + atomics) on BOTH NoCs in dynamic mode. Nothing else uses cmd buf 1
   during the drain (the gathered-stick reads are barriered before it; no atomics until the next row).
2. Before the drain: program cmd buf 1's `NOC_TARG_ADDR_COORDINATE` (write source) to the local core on each NoC, as
   `dynamic_noc_init` does for cmd buf 0. Reads/atomics rewrite TARG coordinate per op, so no restore is needed.
3. In the drain, each tile still goes on the NoC the parent's path-aware rule picks, but alternates between cmd buf
   0 and cmd buf 1 of that NoC (per-NoC toggle), through `ncrisc_noc_fast_write_any_len<DM_DYNAMIC_NOC>` with the
   accessor's `get_noc_addr(page, 0, noc)`. Same VC, non-posted, the same per-NoC dynamic counters, so the per-block
   `async_writes_flushed` and the final `async_write_barrier` still cover every write.
4. After the drain: wait for cmd buf 1 to be idle on both NoCs and restore its `NOC_RET_ADDR_COORDINATE` to the local
   core (reads and atomics rely on that preset; the writes overwrote it with DRAM coords).
Non-dynamic builds keep the old loop (`if constexpr`).

## Why this is not a repeat
- All drain nodes so far changed WHERE the bytes go (NoC split by position/destination: r01-b02-a04, r01-b03-a04,
  r02-b01-a01, r02-b03-a01, r02-b02-a01) or flush cadence (r01-b02-a02, neutral). This keeps every tile on the exact
  NoC and path of the parent and changes only how many requests the core can have in the NIU at once.
- r03-b04-a02 found the per-core bound and suggested posted writes; posted writes keep a single cmd buf, so if the
  cap is cmd-buf serialization (the noc.h comment says it is), posted would not lift it. This tests the issue-side
  cap directly. Posted writes remain a separate child idea.

## Expected effect and risk
- If the per-core rate is cmd-buf/issue bound: per-tile cost drops toward max(injection, issue) ~60-70 cycles, and the
  drain becomes aggregate-bound (DRAM/link, ~420+ GB/s). Drain from first POST tile at h3584 2.9 -> ~2.2 µs, h7168
  5.6 -> ~4.8 µs. Kernel end moves with it: ~-0.5..-0.8 µs per shape, +3-5% score.
- If the cap is link/DRAM (or NIU injection) rather than issue: neutral.
- Risks: wrong TARG/RET coordinate programming -> writes sourced from the wrong core (accuracy_fail) or the next
  call's stick read / sem atomics misrouted (hang / accuracy_fail). Two writers may raise congestion and bring back a
  position gradient (would show in `drain.py` per-core rates). Accuracy otherwise bit-identical.
- Judge with `drain.py` / `drain2.py` (in this dir): per-core tiles/µs and drain_e - C_POST start vs the parent.
