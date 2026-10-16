# r05-b03-a01: stagger the row-shared x owner rotation by M-row (group index + 2 x row-in-group), so a gate/up K-block's x stick reads, tilize and row multicast are spread over ~10 grid columns instead of landing on the same 1-2 columns in every row

## Motivation
The row-shared x tilize (r01-b01-a02, made sender-free in r03-b02-a02) gives gate/up K-block `j` of an M-row to
the owner column `adaptive_chunk::x_owner_col(j, grid_x, rows, my_mt)`. Slot `j % slots` does not depend on the
grid row, so for a given block every row of the grid picks the same slot:

- On the 5 x 2-row layout (phase A of dsv3 / glm / kimi-k2.7 / dsv4-pro, and kimi-k3 / mistral / minimax-m3):
  all 5 top rows use column `2r+1` and all 5 bottom rows use column `2r`. Ten owners are in **2 adjacent
  columns** for every block.
- On the 10-row layout, 7-9 of 10 rows share one column. On 5+5 it is 6-8, and on the mixed layouts 4-7
  (`/tmp/r05b03-a01-nikola/ana/owners.py`).

Each owner reads `per_core_M x 32` one-KB x sticks per block from DRAM on NoC 0 (96-128 KB at pcM 3-4). With x-first
routing, a NoC-0 read response turns into the owner's column and runs down that column's links. So per block
~1 MB of x responses plus those columns' gate-weight responses share 2 columns of vertical links. That is
~50-80 GB/s per column on a ~86 GB/s link, while the other 9 columns carry only their weight share.

The slow rows wait on x. Evidence:
- r04-b03-a01's UNPACK x-wait per grid row (`/tmp/r05b03-a01-nikola/ana/uwrow.py`). mistral y2/y3 wait
  39/39 µs, y8/y9 21/20 µs, and the GU window is 121 vs 101 µs. minimax-m3 y2-3 waits 65-68 µs, y8-9 41-45 µs. dsv3
  y2-3 waits 133-140 µs, y10-11 75-80 µs. The GU-window gap between rows equals the x-wait gap.
- r03-b04-a02's RE-GW zones: the first x block reaches the top rows 15-20 µs later than the bottom rows (block-1
  read starts at y2 ~37 µs vs y9 ~19 µs on mistral).
- r04-b02-a01's per-row GU (`/tmp/r04b01a02/rows.py`) is the same picture: mistral y2-3 120 µs vs y8-9 93; mm3
  242 vs 211; dsv4-flash y2-4 190 vs y10-11 163. The slow group is usually the chip-critical one (mistral y2-3 in 6
  of 10 calls, mm3 y4-5 in 6 of 10).
- Launch skew is ruled out: core start times differ by < 0.1 µs (`skew.py`).

The column hotspot has a second, steady-state effect. In each block the owner columns' gate-weight responses
share those links with the x responses, so those columns' weights arrive last. The row lockstep (the x multicast
needs all 11 acks, and senders ack only after their own block) then paces every row by that one slow column.

## Mechanism
Owner slot of block `j` on a row becomes `(j + x_off) % slots`, where `x_off = group_index + 2 * row_in_group`
(per phase, from the launch plan every RISC already computes). All cores of a row share the group index and the
row-in-group, so they still agree on every block's owner. The owner column set is unchanged, so an owner is never
the row's weight sender. Only the rotation phase differs per row.
- 5 x 2 layout: the 10 rows' owners of any block sit in 10 distinct columns (mean max rows per column 5.0 -> 1.0).
- 10-row, 5+5 and mixed layouts: max 2-3 rows per column (from 4-9).

Files (kernel-only, no host change):
- `kernels/adaptive_chunk.hpp`: `x_owner_offset(group, my_mt)` helper.
- `kernels/dataflow/unified_routed_expert_ffn_reader.cpp`: owner of block `kb` is `x_owner_col(x_blk + x_off + kb, ...)`.
- `kernels/compute/fused_swiglu.cpp`: UNPACK adds the group index to the per-expert mailbox word. The gate/up
  phase gets `x_blk_base = x_blk + x_off`, so the compute tilizes exactly the blocks its reader reads.

Measurement (profiler-only, compiled under PROFILE_KERNEL like the existing zones): one NCRISC zone `RE-X0`
around each unit's first x-valid wait (block 0 of every chunk), so the reflection can show x(0) arrival per row.

## Why this is not a repeat
- r03-b02-a02 moved the owner off the weight-sender core (which core in the row). r01-b01-a04 / r03-b04-a02 /
  r01-b03-a02 changed x read timing, transaction ids and batching. r04-b04-a01 decoupled the x/weight acks. None
  of them changed the fact that every row picks the same column for a block.
- No earlier node touched the cross-row placement of the x owners. This changes NoC link load, not handshakes,
  read timing or ring depth, which the history says not to retry.

## Expected effect and risk
- Bit-identical outputs: same x data, same tilize, same multicast payload and slot. Only the owner core changes.
- Expected: the top-row x-wait gap closes partly. First-unit x(0) arrives earlier on the slow rows, and per-block
  pacing is no longer set by one hot column.
  - mistral / minimax-m3 / dsv4-flash: -2..-6% (their slowest group is a top-row group).
  - dsv3 / glm / kimi-k2.7: -1..-2%.
  - kimi-k3 / dsv4-pro: ~0 (desynchronised groups and DRAM-bound).
  - gpt-oss: ~0 (its critical groups are the pcM-4 bottom ones).
  - Geomean maybe +0.5..+1.5%.
- Risk: owner/compute disagreement would hang or corrupt (the compute must tilize exactly the blocks the reader
  reads). Both derive the owner from the same `(x_blk + x_off)` with the same group and row-in-group.
- Size: a few instructions plus one profiler zone. The h7168 programs are ~0.5-0.9 KB under the 70656 B limit.
- How to tell:
  - `RE-X0` per row: the top-row first-x arrival should move toward the bottom rows'.
  - Per-row GU (`rows.py`): the y2-y5 vs y8-y11 gap should shrink.
  - Chip µs on mistral / mm3 / dsv4-flash.
- If GU per row does not move, the column hotspot is not what delays the top rows, and the cause is elsewhere
  (DRAM-endpoint geometry or the multicast chain).
