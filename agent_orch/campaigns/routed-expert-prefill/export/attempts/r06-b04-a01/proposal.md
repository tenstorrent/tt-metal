# r06-b04-a01: wrap-free down-weight multicasts: weight senders sit only on their group's top or bottom row, and a bottom-row sender's down multicast runs on its writer's NoC 1 (-y) instead of the reader's NoC 0 (+y), so no down multicast wraps around the column torus

## Motivation
Each group column multicasts its down-weight blocks from the column's sender core to the group's other rows.
Today that is always the sender's READER on NoC 0, which routes +y. The sender row is `group_base + gx % rows`.

- A NoC-0 multicast whose sender is not the rectangle's top row has to reach the top row first by going +y.
  It wraps through every other group's rows and the two non-worker rows of that column (about 12 hops), and the
  path is `linked` (reserved) while 50-110 KB stream through it.
- In 2-row layouts every odd column has its sender on the bottom row, so in those columns all five groups' down
  multicasts wrap and overlap on every +y link of the column. Those links also carry the NoC-0 DRAM read
  responses of every sender in the column (NoC 0 runs +x along the bank row, then +y up the worker column).
- In 3/4/5/10-row groups, every sender except the top one wraps: 2/3, 3/4, 4/5, 9/10 of the columns.

Evidence:
- r05-b04-a01 (`analysis/wrap.py` on r04-b03-a01's UNPACK wait tallies): receivers in a column whose down
  multicast wraps wait for their down weights 2-11x longer than in a column whose multicast doesn't:
  kimi-k3 130.7 vs 46.1 µs, dsv4-flash 18.3 vs 9.0, minimax-m3 15.6 vs 4.1, mistral 6.7 vs 0.6, gpt-oss 4.1 vs 1.1.
  The column parity flips between gate/up (NoC 1, top senders wrap) and down (NoC 0, bottom senders wrap), so it
  is not a row or column position effect. The rows are lock-step across all 11 columns (activated stream), so the
  wrapped half paces the whole group.
- Link-load model (r05-b02-a01's `noc4.py`, extended in /tmp/r06b04/nocw2.py): with today's placement the hottest
  NoC-0 link in the down phase is a worker-column link: 56 GB/s on 5x2, 58 on 4x2, 63 on 4+3+3, 75 on 5+5, 92 on
  the 10-row layout. That is above the DRAM-region links r05-b02-a01 balanced to ~54. With wrap-free down
  multicasts the worker-column links drop to 22-52 and the down-phase maximum becomes the ~54 GB/s DRAM-region
  links on every layout. The 5+5 layout (phase B of dsv4-pro / kimi-k2.7, the two longest shapes) gains most.

## Mechanism
1. **Corner-only sender placement** (`adaptive_chunk::sender_row_in_group`, used by the program factory and,
   through `x_owner_key`, by the x-owner rotation in the reader and compute).
   - Column c's sender in group (base, rows) is the group's top row when (c + base) is even, else its bottom row.
   - For every 2-row group (all at even bases in the tables) this is exactly today's `base + c % 2`. So 5x2, 4x2
     and the 2-row groups of mixed layouts are unchanged.
   - The row-shared x-owner rotation keeps its closed-form helpers. A corner row skips one parity of columns:
     (rows, my_mt) = (2, parity). A middle row has no sender and skips none: (1, 0). 2-row groups give the same
     owners as today.
   - The host collision TT_FATAL (TILE path's fixed x sender (gy + 1) % GRID_X vs the weight sender) is kept for
     2-row groups. For 3+-row groups it is relaxed: any parity rule puts the bottom row's x-sender column on a
     bottom sender for odd heights. That core then runs both reads in sequence, as the original (0, 0) sender
     did, and each role keeps its semaphores. This costs TILE-path time only; row-major x skips sender columns.
2. **Bottom-row senders multicast down on the writer's NoC 1** (-y from the rectangle's NoC-1 start corner, so no
   wrap). Top-row senders keep the reader's NoC-0 multicast, which already starts at its corner. Applies when
   IN1_WRITER_MCAST and DOWN_SPLIT are both on, i.e. whenever the writer already reads the down tail.
   - Writer, per down block, after its tail read and `down_done`: wait `mcast_go` (reader's head landed) and the
     receivers' `in1_ready` acks, linked-multicast the whole slot, flush, multicast `in1_valid`, set `mcast_done`.
     This mirrors its gate/up multicast and reuses the same go/done pair: the sequence counter runs over every
     block the writer multicasts, in the order both RISCs walk them.
   - Reader:
     - skips the down ack wait and multicast;
     - signals `mcast_go` after its head read barrier;
     - before reading into a ring slot, waits until the writer's multicast out of that slot is done. The
       look-ahead becomes non-blocking on the same condition.
     - Block 0 keeps its "head read after the acks" order with a non-consuming `wait_min` on in1_ready.
     - At the end of a phase it drains the writer (`mcast_done >= mc_seq`), so a core that turns receiver in
       phase B cannot receive into a slot its writer still multicasts from.
   - As a side effect the bottom sender's head read of block kb+1 now overlaps the multicast of kb, as gate/up
     already does.
3. Gate/up is unchanged: the writer's NoC-1 multicast for every sender. Top senders still wrap there. Moving them
   to the reader would serialize the reader's 60% gate read behind its multicast.

4. Kernel-config size: the writer's copy of the launch planner is built at -Os (r04-b01-a02's measured-free
   pragma; locally the writer comes out 364 B smaller net), to pay for the reader's +420 B. The h7168 programs
   have only a few hundred bytes of headroom on this lineage.

Files: `adaptive_chunk.hpp`, `unified_routed_expert_ffn_program_factory.cpp` (placement), `..._reader.cpp`,
`..._writer.cpp`, `compute/fused_swiglu.cpp` (x-owner helper arguments).

## Why this is not a repeat
- r05-b04-a01 had the same diagnosis and hung. It moved the senders per phase (all gate/up senders on the bottom
  row, all down senders on the top row). That made the gate/up and down senders different cores and put x-owner
  duty on gate/up senders.
- This node keeps one sender per column for both phases and keeps x owners off senders. It changes which RISC/NoC
  carries a corner sender's down multicast, which is r05-b04-a01's repair option (c) applied to down. For 2-row
  groups the core roles are identical to the parent, so only the down multicast's NoC changes.
- No earlier node changed multicast direction. r02-b04-a02 moved the whole down stream (reads + multicast) to the
  writer for every sender, which made the reads free-running and cost 10-20% DRAM share. Here every read keeps
  its RISC, NoC, split and timing; only the bottom senders' multicast moves.

## Expected effect and risk
- Bit-identical outputs: same bytes into the same slots. Only the multicasting RISC and the sender row in
  3+-row groups change.
- Expected DN window: -5..-15% on the 5+5 phase-B shapes (dsv4-pro, kimi-k2.7) and on the 3+3+2+2 / 4+3+3
  shapes (mistral, minimax-m3, dsv4-flash, gpt-oss, dsv3/glm phase B). Smaller on kimi-k3 (5x2), where half the
  columns wrap today. Some of the DN gain may reappear in GU on DRAM-bound shapes (the conserved GU+DN pattern).
  Guess +1..+3% geomean.
- Risks:
  - Hang, if the reader and writer disagree on the predicate or the mc_seq order. Both compute it from the same
    (is_sender, my_mt, group_rows) and walk blocks in the same order.
  - Slot reuse race, guarded by the mcast_done waits.
  - Kernel-config size: the writer gains one multicast block, the reader a few predicates.
  - Row-link load on 10-row groups (all senders on 2 rows). The model puts it at the level of today's column
    hot spot.
- How to tell: per-shape DN / GU zone means vs r05-b02-a01, mean / max core end. In a profile, odd-column
  receivers' down waits should fall to the even columns' level.
