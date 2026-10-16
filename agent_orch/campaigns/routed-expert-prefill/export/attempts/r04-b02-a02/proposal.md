# r04-b02-a02: run the phase-4 activated row stream on its own, decoupled from the down-weight handshakes

## Motivation
- r04-b03-a01 measured the compute's input waits. In the down phase the UNPACK thread waits on the
  **activated** block (cb_in0_down_full) for 30-58% of the RE-DN window on every shape. It waits on
  the down weights for only 3-11%.
- Today the activated stream is locked to the down-weight loop. In the reader each core runs one
  loop per down block kb: reserve both rings → ack the in1 sender and the act sender of kb → wait
  act_valid(kb) → wait in1_valid(kb) → push both.
  - So a core acks act(kb+1) only after its **column's** down weight kb has landed.
  - The act sender of kb+1 needs all 10 row acks. So activated block kb+1 starts only after the
    slowest of the 11 columns has delivered w(kb).
  - The act sender itself sends only in its own iteration kb+1, after its own w(kb) wait.
  - The activated data is ready early: each core's cb_activated is complete at the end of gate/up.
    Yet the stream moves at most one block ahead, and at the pace of the row's slowest column every
    block (a per-block max over 11 DRAM streams).
- Deeper d rings (r01-b04-a03) could not help, because this ack protocol never lets the activated
  stream run more than one block ahead.
- r03-b04-a04 next #1 and r04-b02-a01 next #2 both point here: let receivers ack activated kb+1 once
  their slot is free.

## Mechanism
File: `kernels/dataflow/unified_routed_expert_ffn_reader.cpp`, phase 4 only.
- **A non-blocking `pump_act()` state machine** for the activated stream, per down phase
  (`act_next`, `act_acked`):
  - Sender of block a (gx == a): once cb_activated is full, all GRID_X-1 acks are in and its own
    slot is free, it multicasts (same linked mcast + loopback as today), then sets
    act_valid = act_seq + a + 1 and multicasts it. Then the write barrier, push, pop cb_activated.
  - Receiver: as soon as block a-1 is pushed and a cb_in0_down_full slot is free, it acks block a's
    sender. It pushes once act_valid ≥ act_seq + a + 1.
  - act_valid becomes a running count instead of a 0/1 reset per block. Ordering argument: a sender
    sends only after every row core acked, and each core acks only after it has seen the previous
    block's value. So values arrive in order and no stale value can land late. act_ready keeps
    wait-all-then-reset; only one block's acks can be outstanding per sender.
- **The down-weight loop keeps its protocol** (same NoC split, slots and DOWN_SPLIT). Every wait in
  it spins `pump_act()` instead of blocking: the in1 slot reserve, in1_ready, the read barrier
  (`ncrisc_noc_reads_flushed`), down_done and in1_valid. A drain loop ends each down phase.
- r04-b02-a01's down look-ahead falls out naturally and its code is deleted. The sender's next read
  is issued as soon as its own slot frees: the top-of-block reserve now directly follows its
  multicast, with no act wait in between.
- Size: the parent's h7168 programs sit at the kernel config buffer limit, and the reader grows
  ~0.8 KB (local -Os profiler compile 14296 → 15076 B). To pay for it, the profiler-only MATH
  perf-counter block from r03-b02-a03 (`re_hwprof.hpp`) becomes opt-in (`RE_HWPROF`). That saves
  -1.46 KB of TRISC1 in profiler builds and changes nothing in non-profiler builds. This is a size
  repair, not a second mechanism.

Results stay bit-identical: same blocks, slots, K order and formats.

## Why this is not a repeat
- **r04-b04-a01** decoupled the x / weight acks in **gate/up** (neutral). There the next x sender
  (the owner) was not ready anyway: it was waiting for its tilize. Here the activated data is
  complete before phase 4 begins, so the act sender is ready at once. Here the sender is also
  decoupled, not just the acks.
- **r02-b04-a02** (writer-owned down stream, -11%) moved the down-weight reads to NoC 1 and onto
  one RISC. This node keeps the down-weight delivery exactly as it is and decouples only the
  activated stream.
- **r03-b04-a04 / r04-b02-a01** moved the down-weight read earlier. This node keeps that behaviour
  and attacks the other stream.

## Expected effect and risk
- **Expected:** shorter RE-DN on every shape, most on the many-small-block shapes (kimi-k3,
  dsv4-flash, dsv4-pro, kimi-k2.7, whose activated wait is 47-58%). Perhaps -5..-15% of DN, so
  -1..-4% chip time. DRAM-bound shapes may give part of it back (the conserved GU+DN signature).
- **Risks:**
  - Hang if the pump misses a wait or the ack / valid ordering is wrong.
  - Wrong data if a slot is overwritten while still read: every ack and send checks
    `pages_reservable_at_back`.
  - Kernel-config overflow if my size estimate is off.
- **Reading the eval:** bit-identical PCC / mean_abs; per-shape RE-DN mean core (zones.py) vs the
  parent.
