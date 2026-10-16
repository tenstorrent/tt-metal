# r04-b01-a02: activated stream one block ahead in the down phase: receivers ack activated block kb+1 as soon as their own cb_in0_down_full slot for it is free, and its sender multicasts it while the row still waits for block kb (two parity valid flags), on r04-b02-a01's down weight look-ahead

## Motivation
- r04-b03-a01 (measurement node) settled where the compute time goes on this lineage. Its UNPACK-thread
  wait tallies show the down phase (DN) waiting on the **activated** stream for 30-58% of its window on
  every shape: dsv3 31.9%, k2.7 46.8%, k3 57.5%, glm 33.3%, mistral 34.3%, dsv4-flash 49.8%, dsv4-pro 49.7%,
  mm3 30.4%, gpt-oss 39.0%. The down weights add only 3-11% after that. The matmul under it runs at 21-25
  cycles per tile-mult (in-situ probe), so this is waiting, not compute.
- Why the activated stream is slow, from the reader (`unified_routed_expert_ffn_reader.cpp`, phase 4):
  - block kb's activated tiles come from row core gx == kb, and every block's data is READY as soon as that
    core's compute has run the activation (cb_activated is one block per core per chunk);
  - but a receiver acks block kb+1 only at the top of iteration kb+1, i.e. after BOTH activated(kb) and the
    down weights of kb have landed, and the sender of kb+1 waits for all 10 row acks;
  - so each of the 11 down blocks pays a full row-wide ack round trip plus the multicast latency, strictly
    in series, and is coupled to the slowest core's down-weight arrival of the previous block. With a
    2-slot ring there is no reason for that: the slot for kb+1 is free as soon as compute popped kb-1.
- Both measurement-driven reflections name this as the next step: r04-b02-a01 next #2 ("a counting act_valid
  ... receivers ack kb+1 once their cb_in0_down_full slot is free") and r04-b03-a01 next #1 ("decouple the
  row streams; down / activated is the biggest").
- Unlike gate/up (r04-b04-a01: early acks fired a lot but changed nothing, because the next block's
  PRODUCER was not ready), here the producer is always ready: the activated data exists before phase 4
  starts. Only the handshake holds it back.

## Mechanism
Kernel-only. Files: `kernels/dataflow/unified_routed_expert_ffn_reader.cpp`, `kernels/compute/fused_swiglu.cpp`
(revert to r03-b02-a02's), delete `compute/re_hwprof.hpp` and `compute/re_mm_unpack.h`.

1. **Base: r04-b02-a01's reader** (campaign best, 1.6275; sibling lineage from the same root, whose reader my
   parent never changed). It brings the down weight look-ahead with the spin re-check (DN -4..-30 µs mean
   core on all 9 shapes). Its gate/up look-ahead (measured neutral on 8 of 9 shapes) is switched off
   (`gu_lookahead = false`) to pay for the code below.
2. **Activated stream one block ahead (new).** In phase 4:
   - Two valid flags alternate by block parity: even blocks use `act_valid_sem`, odd blocks `in0_valid_sem`
     (the x-stream valid flag, idle for all of phase 4: x valid writes reach a core only after its whole row
     acked an x block, i.e. after the row left phase 4, and every activated write to a core lands before
     that core leaves phase 4). Two blocks can then be in flight to a core without one flag being
     overwritten by the next. Each flag is reset right before the ack that allows its next write, as before.
   - While a core waits for block kb (activated or down weights), it polls (`wait_polling`):
     - receiver of kb+1: once its own slot for kb+1 is free (`pages_reservable_at_back`), reset kb+1's
       flag and ack the sender of kb+1;
     - sender of kb+1: once its own slot for kb+1 is free, its cb_activated block is there and all 10 acks
       are in, multicast its block into the next ring slot (same address on every row core) with the
       linked valid write, exactly as today;
     - the in1_down sender keeps its weight look-ahead re-check.
   - Iteration kb+1 then skips the ack / multicast it already did. The sender still write-barriers its own
     loopback copy before pushing. Anything not done early falls back to today's path.
   - The sender code moves into one out-of-line helper (`send_act`) shared by both paths.
3. **Kernel-config-buffer budget.** The h7168 programs sit near the 70656 B limit. Local O2 / O3 compiles
   (the JIT's levels): reader +3.8 KB vs the root (r04-b02-a01's half +2.0 KB, this node +1.8 KB); compute
   -2.1 KB by dropping two measured-null, profiler-only/size-only pieces of my lineage: r03-b02-a03's RE-HW
   perf counters (TRISC1 -1.46 KB in profiler builds) and r04-b01-a01's CFGSHIFTMASK unpack MOP (TRISC0
   -0.68 KB, measured neutral). Net vs the root about +2.3 KB against an inferred ~3.0 KB of root headroom
   on kimi-k2.7 (r04-b04-a01's run-1 reader overflowed by 240 B at +3.3 KB in the same local measure).

Numerics: same tiles into the same slots in the same order; the compute is unchanged (r03-b02-a02's
kernel, bit-identical to every node since). PCC / max_abs / mean_abs must match the root exactly.

## Why this is not a repeat
- r04-b04-a01 decoupled the gate/up x and weight acks (neutral): there the next block's producers (x owner
  tilize, the writer's serial up read) were not ready, so an early ack only moved who waits. In phase 4 the
  activated producer is always ready, and nothing has touched the activated handshake yet.
- r04-b02-a01 / r03-b04-a04 decoupled the down WEIGHT reads from the acks. This node decouples the
  activated stream and, through it, the receivers' next weight ack (which is issued after the activated
  block lands).
- r02-b04-a02 (writer-owned down stream, -17%) moved the weight stream to another RISC with new sems; no
  activated change.
- The compute revert and the GU look-ahead switch-off are size housekeeping, both measured null before.

## Expected effect and risk
- If the activated handshake chain is what the DN window waits on: DN mean core -10..-30% on the shapes with
  the largest activated wait (kimi-k3 57%, dsv4-flash / dsv4-pro 50%, k2.7 47%), a few % elsewhere; the DN
  share of the critical row then moves the op time by roughly a third of that. Plus r04-b02-a01's down
  look-ahead gains on this lineage (~+1% geomean). Expected geomean ~1.63-1.66.
- If DN is really paced by down-weight DRAM share (as GU is): the activated wait turns into weight wait,
  DN flat, and only the ported down look-ahead gain remains (~1.62).
- Risks:
  - kernel config buffer overflow on the h7168 shapes (`Program size ... too large`; ~0.7 KB margin by my
    estimate) -> trim and re-run;
  - a protocol slip (wrong parity flag, an ack for a slot that is not free, a wrong next-slot address) ->
    hang or a PCC change; the bit-identical PCC/mean_abs is the check;
  - gpt-oss may lose part of r04-b02-a01's -5 µs GU from the switched-off gate/up look-ahead.
- How to read it: RE-DN (MATH zones) mean and crit core vs r04-b02-a01 and the parent.

## Addendum after the first eval run (kernel config buffer overflow, repaired before any timing)
- Run 1: `Program size (70816) too large for kernel config buffer (70656)` on kimi-k2.7 only, 160 B over, at
  program creation. The other 8 shapes ran and printed bit-identical PCC / max_abs / mean_abs (the protocol
  itself is fine). My local estimate (~0.7 KB margin) was off: the JIT's LTO build does not track local -O2
  object sizes closely. Run 1 is now the size anchor.
- Repair, inert for numerics and the schedule:
  1. Reader: the receiver ack and the sender multicast of the activated stream are one out-of-line step
     function (`try_act(kb, ahead)`) used by both the normal and the early path (-0.19 KB at -O2).
  2. `adaptive_chunk.hpp`: the launch planner region is built at -Os on the writer (BRISC) only, like the
     existing -O2 pragma on TRISC (-0.78 KB). The reader's -O2 planner, which releases the first weight
     reads, is unchanged; the writer's plan only gates the weight-sender writers' first `up` read.
  Net about -0.96 KB vs run 1, i.e. ~0.8 KB under the limit by the same local measure.
