# r01-b04-a02 result: 0.8397 (ok)

## What happened vs expected
The run is valid with bit-identical accuracy (PCC 0.9999985, max_abs 0.022-0.024), but it is ~16% slower on every
shape: h3584 0.846, h4096 0.843, h6144 0.836, h7168 0.833 (parent r01-b04-a01: 1.116 / 1.126 / 1.052 / 1.016).
I expected -2 to -4 µs per shape and got +4.5 to +5.6 µs.

## Why (profiler evidence, chip 1, µs from kernel start; parent -> this node)
- h7168 (call 61441): R_INPUT end 7.6-8.9 -> **15.6-16.8**. The NCRISC kernel end was 15.4-16.8 in the parent
  (input, then gamma) and is 15.7-16.8 here, so input plus gamma takes the same total time as before. The
  difference is that the input row is now held hostage by the gamma reads. W_PUSH (PRE done) moved from
  8.6-10.0 to 17.3-18.6, and everything downstream shifted by ~8 µs: AG wait ends 21.1-21.4, TRISC ends
  26.6-27.2, W_DRAIN ends 27.9-30.9.
- h3584 (call 15361): R_INPUT end 3.8-4.9 -> 8.1-9.2, and the parent's NCRISC end was 7.7-8.8. Same pattern.
- So the gamma face-row reads cost ~4 µs (28 tiles) to ~8 µs (56 tiles), about 70 ns per 64 B read per core,
  whether they are issued after the input or interleaved with it. They are NOT latency-bound (deep issue did
  not shrink them). They look throughput-bound at the DRAM side: all 20 workers read the SAME [1,H] gamma pages
  (same banks, same rows) at the same moment, as 2 x 64 B transactions per tile = 2240 tiny reads per chip
  per call. Interleaving them with the input put that hot-spot traffic in front of the input blocks, which
  share the NCRISC read queue and the same DRAM banks.
- With gamma in the critical path I can't measure the input-only speedup from the trid pipelining. The PRE
  tail after the input is still ~1.7 µs, as in the parent.
- POST after the AG is ~5.5 µs for 56 tiles (~100 ns/tile, single pass), and the drain tail is another 1-4 µs.
  Both are unchanged from the parent.

## Classification
flawed idea (the gamma-interleave part). Gamma reads are a shared-bank hot spot, not a latency problem, so
putting them next to the input serializes the critical path. The trid-pipelined input read itself is
untested in isolation: repairable, but it needs to be split out.

## What a child of this node should try next
1. Keep the trid-pipelined input pass (reader `read_input_pass_pipelined`) but call it with `with_weight=false`
   and read gamma AFTER the input row (the parent's order, or r01-b01-a01's single-barrier batch). That isolates
   the input-pipelining effect. It is a one-line change from this node.
2. Take gamma off the critical path properly. Every core reads the same ~7 KB of gamma, so either
   (a) have each of the 20 workers read 1/20 of the gamma row and multicast/unicast it to the others (or one core
   reads it and mcasts it to the worker grid), or (b) let the idle BRISC (the writer waits ~8 µs for PRE)
   read gamma on NOC0 so it doesn't share NCRISC's queue with the input. Either way the 2240 tiny same-bank reads
   per chip disappear or move off the input path.
3. Don't interleave tiny broadcast reads with the bulk input stream again.
