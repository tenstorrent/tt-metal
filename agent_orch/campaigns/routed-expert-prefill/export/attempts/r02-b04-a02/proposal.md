# r02-b04-a02: writer-owned down-weight stream: the down weights run ahead of the activated multicast instead of in lock-step with it

## Motivation
Fitting time = F + U * per_core_M per expert to the measured groups gives a per-expert fixed cost F
that does not scale with per_core_M and is about half the time on the few-expert shapes:
- gpt-oss: F ~150 µs (r02-b04-a01 reflection, 3-row vs 2-row groups).
- mistral (1 expert per 2-row group, pcM 3): 173 µs measured vs ~83 µs of LoFi matmul at
  ~16 cycles/tile-mult (2328 tile-mults per pcM unit). So F ~90 µs.
- dsv4-flash (same weights as mistral, 2 experts per group at pcM 2): ~150 µs per expert, F ~95.
- minimax-m3 (1 expert per group, pcM 3): ~360 µs vs ~186 µs of matmul, F ~175 µs.
F roughly tracks weight bytes per expert (14 MB -> ~90 µs, 32 MB -> ~175 µs), so it looks like weight
delivery that is not overlapped with compute.

Reading the reader's phase-4 loop (`unified_routed_expert_ffn_reader.cpp`, down K-block loop) shows
the one weight stream that cannot run ahead:
- Gate/up weights already run ahead of compute: the reader reads `gate` for block kb+1 while the
  writer multicasts block kb (IN1_WRITER_MCAST), up to the 2-3 slot ring.
- The down weights are delivered inside the same per-K-block iteration as the activated multicast.
  The column sender reads block kb only after every receiver of its column has acked `in1_ready`
  for kb. A receiver acks kb only at the top of its reader iteration kb, i.e. after iteration kb-1
  has finished, including that block's activated multicast (`act_valid`). So:
  1. During the activation phase (all rows' `cb_activated` not yet pushed) the reader of every core
     sits in down iteration 0 waiting for the activated block, and only down block 0 is in flight.
     The down ring (2-3 slots) cannot fill while compute runs the activation, and no down DRAM traffic
     overlaps it.
  2. In the down loop, every block pays one serial chain: activated handshake (row) -> receivers'
     ack -> DRAM read -> weight multicast -> valid, and the next block's DRAM read cannot start
     before that chain ends. The ring only absorbs jitter, not this per-block latency.
A down K-block is small (in0_block_w_d = per_core_N_gu = 6-9 K-rows; ~1-6 µs of compute at
pcM 1-3), so the per-block chain is likely longer than the compute it feeds.

## Mechanism
Move the whole down-weight stream to the writer (BRISC, NoC 1), decoupled from the activated stream
that stays on the reader (NCRISC, NoC 0):
- **Writer** (`unified_routed_expert_ffn_writer.cpp`): after its gate/up work for a chunk (only the
  column sender has any), every core of a group runs the down-weight loop for that chunk, then drains
  `cb_out` as before. It is now the only producer of `cb_in1_down` (reserve/push):
  - sender: reserve its slot, read the whole block (all w_d K-rows; ND-shard runs as today) on
    NoC 1, barrier, wait for the column's receivers to ack `d_ready`, multicast the block on NoC 1
    (linked) and multicast `d_valid`, push.
  - receiver: reserve its slot, clear `d_valid`, ack `d_ready` to the sender (unicast NoC 1 atomic),
    wait `d_valid`, push.
  A receiver's writer has no gate/up work, so it acks down blocks as soon as its slots are free; the
  sender's writer starts the down reads right after its last gate/up multicast, i.e. while compute
  still runs the last gate/up blocks and the activation. The stream then stays up to `IN1_D_SLOTS`
  blocks ahead of compute and never waits on the activated multicast.
- **Reader**: the down loop keeps only the activated multicast (reserve `cb_in0_down_full`, ack
  `act_ready`, sender multicasts, wait `act_valid`, push). No more `cb_in1_down`, `in1_*` or
  DOWN_SPLIT traffic in phase 4.
- **Host** (`unified_routed_expert_ffn_program_factory.cpp`): the DOWN_SPLIT go/done semaphore pair
  (allocated only when w_d >= 2) becomes a dedicated, always-allocated `d_ready` / `d_valid` pair in
  the same runtime-arg slots. Semaphore count is unchanged (13). DOWN_SPLIT is retired
  (`down_split_k = in0_block_w_d`).
- **Measurement** (no behavioural effect): `DeviceZoneScopedN` zones on the compute MATH thread
  around the gate/up phase, the activation phase and the down phase of every chunk (3 zones per
  expert, at most 21 per op on kimi-k3, well below the 125-zone per-RISC budget), so the phase split
  of F can finally be read from profile_log_device.csv.

Numerics are untouched: the same weight bytes land in the same CB slots in the same order, and the
compute kernel's math is unchanged. Bit-identical results are expected. Every code path (TILE/RM x,
biases, all activations, 0 tokens, multi-chunk, experts_per_chip > 1, groups with no receivers) goes
through the same loop; it only changes which RISC moves the down weights.

## Why this is not a repeat
- r01-b03-a01 / r01-b04-a01 (read-ahead, regressed): they issued the NEXT block's DRAM read before
  the CURRENT block's multicast on the same RISC, delaying the multicast receivers wait on. Here the
  sender's per-block order stays read -> multicast; what changes is that the down chain no longer
  waits on the activated chain, and it runs on another RISC/NoC.
- r01-b04-a03 (3-slot rings, neutral): deeper rings cannot help a stream whose next read is gated
  on the previous block's activated handshake. This node removes that gate, so the existing 2-3
  slots can actually fill (during the activation in particular).
- r01-b03-a03 (writer feeds the next unit's first gate/up block, neutral): unit-boundary gate/up
  feed, not the down stream inside an expert.
- No previous node changed phase 4's delivery protocol.

## Expected effect and risk
- If the down chain is a large part of F: -5..-15% on the few-expert shapes (mistral, minimax-m3,
  gpt-oss, dsv4-flash) and a few % on kimi-k3 (6 experts per group, 66 down blocks per group). The
  DRAM-bound shapes (dsv3, kimi-k2.7, glm, dsv4-pro) should move little, maybe slightly, because the
  down reads now overlap the activation phase.
- Risk: hang if a semaphore/slot protocol detail is wrong (sender/receiver ack ordering, slot index,
  NoC-1 unicast address). Accuracy must be bit-identical; any change means a slot/ordering bug. The
  writer becomes busier on sender cores (up + down stream), which could delay the gate/up multicast
  of the next chunk/expert slightly; the zones will show per-phase times.
- The MATH zones give the gate/up / activation / down split per expert for every shape, which is
  needed to attribute what remains of F either way.
