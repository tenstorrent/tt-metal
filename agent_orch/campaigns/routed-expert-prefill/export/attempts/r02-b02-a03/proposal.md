# r02-b02-a03: down-weight look-ahead across the activation gap: the reader streams down K-blocks Dw-2 ahead of the activated multicast, from a deeper down-weight-only ring sized from leftover L1

## Motivation
- Every expert has a gap between the end of the gate/up K-loop and the first down K-block: the
  activation phase. r02-b01-a02 measured it per expert as ACT ≈ 39.5 µs on kimi-k3, 38.7 µs on
  minimax-m3, 25.9 µs on gpt-oss, 22.9 µs on dsv4-pro and 18.3 µs on dsv4-flash. On the SiLU shapes
  it is the silu copy pass, which is unmeasured but smaller.
- The group's weight stream is almost idle during that gap. The reader's phase-4 loop is strictly
  sequential per down K-block kb: [reserve both slots, ack, down-weight read+mcast kb, WAIT for the
  activated block kb from core gx==kb, push]. So during the activation only down block 0 is fetched.
  Block 1's DRAM read starts only after compute has finished the whole activation and the first
  activated multicast has landed. r01-b04-a03 reached the same conclusion when its 3rd ring slot
  bought nothing: "the reader's down-phase iteration is still gated by the activated multicast of
  block 0, which needs compute to finish gate/up + activation."
- On several shapes DN is stream-bound, not compute-bound. For example kimi-k3 has DN ≈ 78 µs per
  expert over 11 blocks (≈ 7 µs/block) for ≈ 200 tile-mults per block (≈ 3 µs of FPU).
  dsv4-flash is ≈ 4.5 µs/block against ≈ 2 µs of FPU, and dsv4-pro ≈ 11.7 against ≈ 3-5. So
  every block streamed during the gap is roughly one block-time off the expert's DN phase.

## Mechanism
1. **Reader, phase 4** (`unified_routed_expert_ffn_reader.cpp`): split the down-weight protocol
   (reserve cb_in1_down, ack the column sender, sender DOWN_SPLIT read + mcast + valid, wait, push)
   from the activated protocol, with a look-ahead L = min(Dw - 2, num_blocks_d):
   - A prologue streams down-weight blocks 0..L-1 with no activated dependency.
   - Iteration kb then handles weight block kb+L (if any) and activated block kb, keeping today's
     ack-both-upfront / wait-both-at-the-end structure.
   - Weight block kb+L needs compute to have popped block kb+L-Dw = kb-2. That is the same
     condition today's iteration has, so the steady-state double buffering of the activated
     stream is unchanged and nothing waits on a block compute hasn't been handed.
   - At L = 0 (Dw = 2) the loop is exactly today's code.
2. **Factory**: give cb_in1_down its own depth `dw_slots`, separate from the activated ring
   (`d_slots`, cb_in0_down_full).
   - After today's ring-depth decisions (gate/up 3rd slot, then the down rings' 3rd slot),
     deepen only cb_in1_down as far as the leftover L1 allows, capped at num_blocks_d + 1 (and 8).
   - Pass the depth to the reader (L) and to the writer (the DOWN_SPLIT slot index
     `down_blk % IN1_D_SLOTS` must equal the cb_in1_down depth).
3. Also revert the parent's request-VC change (r02-b02-a02's reflection: neutral, may cost
   gpt-oss ~4%) back to r02-b02-a01's reads.

Numerics are untouched: same blocks, same slots, same K order and formats. Only when the weight
blocks are fetched changes.

## Why this is not a repeat
- r01-b03-a01 / r01-b04-a01 issued the next weight read before the current multicast inside the
  gate/up loop, which delayed the critical multicast. This change does not reorder anything inside
  a block's protocol. It only runs the down-weight block index ahead while the reader would
  otherwise be blocked on the activation.
- r02-b04-a02 moved the down stream to the writer on NoC 1 and serialised read+mcast on one RISC,
  and the run-ahead competed with the group's own gate/up stream. Here the reads stay split
  reader-NoC0 / writer-NoC1 exactly as today. The look-ahead starts only after the group's last
  gate/up block, so it never competes with its own gate/up stream.
- r01-b04-a03 deepened the rings to 3 slots, but the reader could never use the extra down slot
  during the gap (strictly sequential loop). This node both decouples the index and deepens only
  the down-weight ring.

## Expected effect and risk
- Shapes with leftover L1: dsv4-flash and mistral (Dw up to ~7) and kimi-k3 / glm (Dw ~4, by my
  replica of the fitter assuming ~100 KB reserved L1). dsv3 / kimi-k2.7 keep Dw = 3, i.e. L = 1, so
  2 blocks in the gap instead of 1. dsv4-pro / minimax-m3 / gpt-oss have no spare L1 and stay at
  Dw = 2, which runs today's exact loop.
- Expected: dsv4-flash -3..-8%, kimi-k3 -2..-6%, glm / mistral / dsv3 / kimi-k2.7 0..-2%, the rest
  unchanged (noise). That is roughly +1..+2% geomean.
- Risks:
  - **Deadlock** if the weight/activated interleave is wrong. Each core does both protocols in
    the same order, and weight block W's reserve only needs compute to have popped W-Dw = kb-2.
    A hang would show as fail_class hang.
  - **Writer slot mismatch** if IN1_D_SLOTS != the cb_in1_down depth. That gives wrong down
    weights, so accuracy_fail.
  - **L1 overflow**, guarded by the same footprint function.
  - DRAM contention: the early down reads overlap other groups' streams. On the DRAM-saturated
    shapes that should be neutral.
