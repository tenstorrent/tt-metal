# r03-b03-a02: unblock the weight-sender row's gate/up DRAM stream: the x owner multicasts while its weight reads are in flight, and the reader reads gate+up so the writer only multicasts (read of block k+1 overlaps the multicast of k)

## Motivation
The few-expert shapes are well short of both DRAM and FPU limits. Bytes / 440 GB/s against measured time:
mistral 129 vs 190 µs, minimax-m3 289 vs 372, dsv4-flash 257 vs 322, gpt-oss 127 vs 262. The DRAM-bound
shapes (dsv3, k2.7, kimi-k3, dsv4-pro) are within 6-16% of 440 GB/s.

From r03-b02-a01's MATH zones (`/tmp/r03b03a02/percore.py`), per gate/up K-block:
- mistral: ~13.6 µs per block (pcM 3, 110 KB of weights per column), against ~6.2 µs of FPU work.
- dsv4-flash (same h4096/i2048 dims, pcM 2): ~11.5 µs, against ~4.2 µs of FPU.
- So the block cycle is ~2x the compute, and the extra pcM row adds its FPU time on top instead of hiding
  under delivery. Delivery and compute look serial, not overlapped.
- gpt-oss: ~27 µs per 155 KB block. Aggregate DRAM is only ~230 GB/s, so this is not DRAM saturation.
- Every column sender moves only 6-9 GB/s. A lone 5-row group did 23 GB/s per column (r02-b03-a01).

Reading the kernels, I found two serial links in the weight-sender row (row 0 of every group, where all 11
cores are weight senders) that nobody has addressed:

1. **x multicast waits behind the owner's own weight reads.** On the row-major path, block kb's x
   multicast comes from a rotating owner core. In row 0 that owner is also a weight sender. Its reader
   multicasts x only after its weight work for kb: the gate read barrier, `mcast_go`, and the `up_done`
   wait for the writer's up read (reader.cpp, the `x_is_row_major` owner block sits after
   `if (is_in1_sender) {...}`).
   - Every other row-0 reader waits for that x (`in0_valid`) before it starts iteration kb+1. That start is
     where it grants `up_go(kb+1)` and issues `gate(kb+1)`.
   - So the next block's DRAM reads on all 11 row-0 cores start only after the owner's DRAM reads of kb,
     plus the x multicast, plus a semaphore round trip. Per block that is R + x_mcast + sems.
   - The owner's compute tilized the block one block earlier, so the x tiles were ready long before.
2. **The writer reads `up` and multicasts gate+up serially.** In the writer's UP_SPLIT + IN1_WRITER_MCAST
   loop: `up_go(kb)` -> read up(kb) -> barrier -> mcast_go -> in1_ready -> multicast(kb) -> flush ->
   next. So up(kb+1) is never in flight during multicast(kb). The reader's gate read does overlap the
   writer's multicast, but the up half doesn't. The cycle is Ru + Mc.

## Mechanism
Kernel and host changes, all in `unified_routed_expert_ffn/device/`:
- **reader.cpp, gate/up K-loop (row-major x):**
  - Split the weight-sender block into an issue part (wait for the slot, issue the gate (+ up) reads)
    and a completion part (read barrier, `mcast_go`, `up_done`, the reader-side multicast, the push).
  - The row-major x owner's multicast (wait `cb_x_tl`, wait `in0_ready`, multicast, valid) moves between
    the two, so it leaves while this core's weight reads are in flight.
  - The owner's look-ahead x stick reads (`issue_x_rm_read(kb + 2)`) are issued after the weight reads,
    so the weight requests reach DRAM first.
  - TILE x is untouched: its fixed in0 sender already sends x before the weight work.
- **program factory:** when IN1_WRITER_MCAST is on (the default), select `up_mode` 0. The reader then
  reads gate AND up on NoC 0 into its slots, and the writer does not read up.
  - The reader already runs ahead: after its gate+up read barrier it signals `mcast_go(kb)` and moves on
    to block kb+1 (gated only by `mcast_done(kb-2)` for slot reuse). Block kb+1's DRAM reads therefore
    overlap the writer's multicast of kb. The cycle becomes max(R_gu, Mc) instead of Ru + Mc.
  - With DS_NO_WRITER_MCAST set, the old UP_SPLIT mode (2) is kept unchanged.
- **writer.cpp:** the multicast loop no longer sits inside `if constexpr (writer_split_up)`. It runs
  whenever `writer_mcasts_in1` is set, and it reads `up` itself only in UP_SPLIT.
- **Measurement (profiler builds only, `#if defined(PROFILE_KERNEL)`):** accumulate wall-clock cycles per
  wait site and emit one `DeviceTimestampedData` per counter per RISC at kernel end (about 20 records per
  core per call).
  - Reader: slot reserve, pre-read wait, read issue, owner x multicast, read barrier, x valid wait,
    weight valid wait, plus GU/DN totals and DN read / activation / valid waits.
  - Writer: mcast_go wait, in1_ready wait, multicast, output drain.
  - This finally gives the per-block DM breakdown r03-b04-a01 asked for.

Numerics: same data, same blocks, same order, so bit-identical.

## Why this is not a repeat
- **r01-b03-a01 / r01-b04-a01** (read-ahead, both lost ~6% on the root schedule):
  - Both issued the next block's reads BEFORE the current multicast, on the RISC that multicasts.
  - Here the multicasting RISC (writer) issues no reads at all during gate/up. The reads move to the
    other RISC, so a multicast is never delayed by read issue.
  - It also runs on today's 2-3-row groups, not the root's 8-row single group.
- **r02-b02-a03 / r01-b04-a03** (look-ahead into the activation gap / deeper rings): ring depth and
  down-phase prefetch only. Neither touches the x-owner coupling or the writer serialization.
- **r02-b04-a02** (writer-owned down stream on NoC 1, -17%) moved down reads to NoC 1 and serialized
  read + multicast on one RISC. This node does the opposite: reads stay on NoC 0 (the reader), and the
  read and multicast run on different RISCs.
- The x-owner ordering has not been touched since r01-b01-a02 introduced row-shared tilize.

## Expected effect and risk
- Few-expert / not-DRAM-bound shapes:
  - gate/up block cycle -1..-2 µs (the x multicast + semaphore hop and the writer's serial multicast).
  - That is -5..-10% on mistral, minimax-m3, dsv4-flash and gpt-oss, and on dsv3/glm/k2.7/dsv4-pro's
    phase B. Expected geomean +1.5..3%.
- DRAM-bound shapes: neutral if DRAM stays the cap.
- Risks:
  - All gate/up weight read responses now land on NoC 0. If NoC 0 links near the DRAM columns congest,
    the DRAM-bound shapes lose. Watch the reader's read-barrier counter there.
  - The owner core's weight multicast is now delayed by its x multicast's waits (cb_x_tl and the 10
    in0_ready acks). The counters show it.
  - Hang risk: the x-owner path's waits now precede the read barrier. I checked the dependency cycle:
    tilize(kb) needs only gate/up(kb-1), and the in0_ready acks need only the acking cores' previous
    iteration.
- The eval's per-site counters will say which link binds now.
