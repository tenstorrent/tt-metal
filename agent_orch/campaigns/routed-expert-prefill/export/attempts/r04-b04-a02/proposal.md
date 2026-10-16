# r04-b04-a02: activated-stream run-ahead in the down phase: each row core grants an activated ring slot (acks that block's sender) as soon as the slot is free, senders multicast as soon as all grants are in, per-slot valid flags allow out-of-order landing, and the activated ring gets up to 4 slots from leftover L1 (on r04-b02-a01's reader)

## Motivation
- r04-b03-a01 measured where the compute waits: in the down phase the UNPACK thread spends **30-58 %** of the window
  waiting for the activated block (`cb_in0_down_full`). It waits only 3-11 % more for the down weights after that:
  dsv3 31.9 %, k2.7 46.8, k3 57.5, glm 33.3, mistral 34.3, dsv4-flash 49.8, dsv4-pro 49.7, mm3 30.4, gpt-oss 39.0.
  This is the largest single wait share the campaign has measured, and DN is 25-40 % of every shape's per-core time.
- The data behind that wait is not late. Every core's `cb_activated` is complete when its gate/up + activation
  ends. What paces it is the handshake:
  - a receiver acks activated block kb only at the top of reader iteration kb;
  - it gets there only after block kb-1's activated AND its own column's down weights landed, and after a
    blocking reserve.
  - So block kb's multicast (from row core gx = kb) waits for all 11 row cores. That means 11 different column
    down-weight streams' block kb-1.
  - The row's activated stream therefore advances at the pace of the slowest of eleven weight streams, plus one
    ack round trip per block. With 2-3 ring slots it can never be more than one block ahead.
- r04-b02-a01 (best node, 1.6275) made the down WEIGHT stream start each block's DRAM read as soon as the sender's
  own slot frees: -4..-30 µs mean-core DN on all 9 shapes. Its next #2 proposes exactly this for the activated
  stream. r04-b03-a01 next #1 calls it the biggest lever.

## Mechanism
Starts from r04-b02-a01's reader (its down look-ahead is kept; this branch's parent r04-b04-a01 reader change, a
measured null, is dropped).

**Reader** (`kernels/dataflow/unified_routed_expert_ffn_reader.cpp`, phase 4 only).
- New `act_pump(pushed)`, non-blocking, called from every wait in the down loop:
  - the top of each iteration;
  - the in1 sender's `in1_ready` spin;
  - the activated wait;
  - the `in1_valid` spin.
- It does two things:
  1. **Grant:** for each next block j, if its ring slot is free (`pages_reservable_at_back` for blocks
     [pushed, j]), up to `ACT_SLOTS - 1` ahead of the block being pushed, ack block j's sender. For this
     core's own block it only marks the slot granted.
  2. **Send:** if this core's own block is granted locally, all `GRID_X - 1` grants are in and `cb_activated`
     holds the block, multicast it (INCL_SRC, linked) into the block's slot address. Then multicast the slot's
     flag (4 B write on the same command buffer, which closes the linked path) and pop `cb_activated`.
- **Per-slot valid flags** (`cb_act_flags`, one 16 B word per ring slot, same address on every core):
  - Blocks are numbered across chunks (`act_seq`). Block g's flag is word g % ACT_SLOTS and reads g + 1 once the
    block has landed.
  - A flag is rewritten only by block g + ACT_SLOTS. That needs every core's grant, and a core grants only after
    it consumed block g, so a waiting core never sees its target overwritten.
  - Flags are zeroed at kernel start, before this core grants anything.
  - The old binary `act_valid` sem cannot serve out-of-order arrival. It is left allocated and unused (15 of 16
    semaphores are used, so there is no room for per-slot sems).
- The sender's own copy is still awaited with `async_write_barrier` before it pushes its own block.
- The down-weight stream (acks, look-ahead, multicast) is unchanged.

**Host** (`unified_routed_expert_ffn_program_factory.cpp`).
- The activated ring depth `a_slots` is decoupled from `d_slots`. It starts at `d_slots` and deepens up to 4
  while the L1 fit leaves room (`cb_footprint_bytes` gains an activated-ring term).
- New `CB_ACT_FLAGS` (c_21, a_slots × 16 B).
- `ACT_SLOTS` / `ACT_FLAGS_CB` reader defines.
- Estimated: a = 4 on k3, glm, mistral, dsv4-flash; 3 on dsv3, k2.7, mm3; 2 on dsv4-pro, gpt-oss (no L1 left there).

**Size** (h7168 programs were at the 70656 B kernel-config limit in profiler builds after r04-b02-a01).
- The reader grows ~1.27 KB (local -Os profiler compile: 14296 → 15568 B).
- Paid for by making r03-b02-a03's profiler-only RE-HW MATH counters opt-in (`RE_HW_PROF`): -1460 B on TRISC1 in
  profiler builds, 0 B otherwise. They have served their purpose (r04-b01-a01 / r04-b03-a01 settled the question).

Numerics: same blocks, same slots, same push order. Bit-identical.

## Why this is not a repeat
- r04-b04-a01 (parent, neutral) decoupled the gate/up x and weight acks within a 2-slot ring.
  - There, the next block's producer (the x owner's tilize, the writer's serial up read) was not ready anyway.
  - Here every activated block exists at down-phase start, the ring can be up to 4 deep, and the sender also sends
    from any wait, not from its own iteration kb.
- r01-b04-a03 deepened all rings with the lock-step ack unchanged, so it had no way to run ahead.
- r02-b02-a03 / r03-b04-a04 / r04-b02-a01 are down-WEIGHT read timing. This node is the activated (row) stream.
- r02-b04-a02 (writer-owned down stream) moved the weights to another RISC.

## Expected effect and risk
- **Expected:**
  - mean-core DN -5..-20 % on the shapes with big activated waits (k3, dsv4-flash, dsv4-pro, k2.7, gpt-oss),
    less on dsv3 / glm / mm3 / mistral;
  - op time -1..-4 %, about +1-2 % geomean.
  - On DRAM-bound shapes part of it may move into GU (the conserved GU+DN pattern).
- **Risks:**
  - a hang from a grant / flag mis-sequence (the eval times out);
  - a stale-data PCC failure if a flag ever overtook its data;
  - a kernel-config-buffer overflow on k2.7 / dsv3 / dsv4-pro if the size estimate is off by more than
    ~200 B;
  - a CB address mismatch if the ring arithmetic is wrong (wrong rows, accuracy fail).
- **How to read the eval:** RE-DN mean-core per shape vs r04-b02-a01 (same reader otherwise).
