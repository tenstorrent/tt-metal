# r03-b04-a02: two-wave AG pipeline: half the workers read first, and each wave gets its own fabric gather and go release, so wave A's stat AG overlaps wave B's input read and B's AG overlaps A's drain

## Motivation
Every node so far runs one rigid sequence per call: all 20 workers read their row together (DRAM ~350-400 GB/s,
R_INPUT end 3.4 µs at h3584, 6.5 µs at h7168). Then PRE runs, and the AG (F_COLLECT end -> go, ~3 µs; r02-b02-a03 and
r02-b02-a04 tables) follows, then the stick read (0.67 µs) and the combine (0.43 µs, r03-b04-a01). Only then does the
drain start. The drain is throughput-bound from its first tile. r02-b02-a04 found pack's POST blocked behind the drain,
and the r03 reflections found that kernel end moves 1:1 with drain start. It writes 2.24 MB/chip at h7168 in ~6.5 µs.
**DRAM is idle for the whole ~4-5 µs between the last input read and the first output write** on every shape. The
r03 nodes shave that gap; this node hides half of it.

## Mechanism
Split the 20 workers into two waves (even worker index = wave A, odd = wave B, so both waves span both worker rows
and all columns):

1. **Reader (NCRISC):** a wave-B core waits on a new start semaphore before its input read. Its wave-A partner (the
   adjacent worker) increments that semaphore when its own read is 3 blocks from landing. So the B read starts while
   A's tail is still in flight, and the DRAM read pipe stays full. Both waves therefore read with DRAM to themselves.
   Input lookahead goes 4 -> 8 blocks, so 10 cores keep the same total bytes in flight that 20 cores did
   (Little's law).
2. **Forwarder:** forked into the op dir as `dit_rmsnorm_wave_forwarder.cpp`, because the shared one serves GroupNorm
   too. Wave A owns packet slots [0,10) and wave B owns [10,20) in the same packet buffer and the same DRAM page, so the
   stats buffer geometry and the writer's gathered-stick read are unchanged. The forwarder becomes a two-event poll
   loop:
   - It sends wave w's fabric packet (its slot range, at the matching page offset) as soon as wave w's arrivals are
     complete.
   - It releases wave w's go-sems as soon as all peers' wave-w packets have landed.
   - Per-wave counting uses 16-bit fields of the existing semaphores. Workers increment arrival by `1 << (16*wave)`.
     The fused fabric atomic increments out_ready by `1 << (16*wave)` (the EDM does a full 32-bit NoC atomic add).
     So a fast peer's wave-B increment can never satisfy wave A's wait, and B's send is not held behind A's
     out_ready wait.
3. **Writer:** the only change is the arrival increment value, a new RT arg. Slot numbering comes from the factory.
4. **Factory:** waves are enabled only on the AG path with max_rounds == 1, one forwarder and >= 4 workers. Everything
   else keeps the old single-wave behavior (the fork's `num_waves == 1` path is the original loop).

Compute is unchanged.

Files: `device/dit_fused_distributed_rmsnorm_program_factory.cpp`, `device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp`,
`device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`, the new `device/kernels/dataflow/dit_rmsnorm_wave_forwarder.cpp`.

## Why this is not a repeat
- The r03 nodes (b01..b04 a01) all shortened the post-AG combine. This node keeps that combine and changes *when* each
  row's I/O happens.
- r02-b04-a01 moved cores (placement), and r01-b03 added cores (column split). Both kept one synchronous read -> AG ->
  drain wave. This node keeps the same 20 cores and placement and staggers them in time.
- r02-b02-a04 forked the forwarder only to multicast the go. That release-latency idea was flat. Here the forwarder
  becomes a two-wave pipeline.
- No node has overlapped the AG window with DRAM traffic.

## Expected effect and risk
Model at h7168:
- A reads 0 -> 3.3, A sticks ~4.9, A go ~7.9, A drain ~9 -> 12.3.
- B reads 3.3 -> 6.5, B sticks ~8.2, B go ~11.1 (the same as today's go). B then drains half the bytes, so it ends
  ~15.5-16 instead of ~18.7.

So expect -2 to -3 µs at h6144/h7168 and -1 to -1.5 µs at h3584/h4096, a score of about 1.45-1.5 if the read
bandwidth holds with 10 readers.

Risks:
- **10 cores may not saturate DRAM reads.** If not, B's chain ends later than today's single wave, and narrow shapes
  could regress. Check per-core R_INPUT start/end per wave.
- **Per-wave AG fixed cost.** If the fabric round for B starts late, B's go moves later. Check F_COLLECT/F_FABRIC zones
  per wave.
- **Hang risk** in the new forwarder loop or the start semaphore (wrong slot/partner mapping). That would show as a
  hang fail_class.
- Accuracy can only break through a slot/offset mapping bug (wrong row's stat). That would show as a large PCC drop.
