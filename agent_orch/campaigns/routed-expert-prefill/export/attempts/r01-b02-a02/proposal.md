# r01-b02-a02: collision-free DM reader placement for row groups (fixed staggered x sender, groups >= 2 rows, no bias reads for other groups' experts)

## Motivation
r01-b01-a01 and r01-b02-a01 run nearly the same row-group schedule: the same group size on every
shape except kimi-k3/dsv4-pro, the same per_core_M and the same weight-sender rows. Yet the parent
(b02) is clearly slower on two shapes:

| shape | b01 µs | b02 (parent) µs | parent group size |
|---|---|---|---|
| gpt-oss | 382 | 477 (+25%) | 4 rows, same as b01 |
| dsv4-pro | 1143 | 1211 (+6%) | 1 row (b01: 2) |

Per-core profile of the parent's gpt-oss (reports/r01-b02-a01, profile_log_device.csv, last call):
the 4-row groups end at ~425 µs (rows 0-3) and ~460 µs (rows 4-7). In b01 both groups end at
~385 µs. Both branches split 2+2 experts at per_core_M 2-3, so the parent takes longer per expert
with the same work. The main code difference on that path: when the group size is < 8, the
parent **rotates the x (in0) sender** across the row every K-block (`(my_mt + 1 + kb) % 11`). b01
keeps the host's fixed x sender at column gy+1. The host asserts that column never holds the
column's weight sender for 8-, 4- or 2-row groups (weight sender row = group_base + gx % rows,
so (gy+1) % rows != gy % rows). With rotation the x sender regularly lands on a weight-sender
core: ~1/4 of K-blocks at 4-row groups, ~1/2 at 2-row groups, and every K-block at 1-row groups,
where every core reads its own weights. That core's NCRISC then serially reads the x block
(gpt-oss: up to 96 row-major sticks), multicasts it, and only then reads its gate block. Its
column's gate multicast and the row's x multicast both wait on it.

A second leak, gpt-oss only (FUSE_BIAS): the parent's reader still reads every expert's
gate/up/down bias slice (27 bf16 tiles per core per expert), including experts dealt to the other
group. That is 2 wasted reads of ~54 KB per core, ~9.5 MB of extra DRAM traffic per call, issued on
the reader's critical path at each expert start. The compute kernel pops them without waiting.
b01 skips them.

## Mechanism
One theme: the DM RISCs of a row group never take DRAM reads that the classic placement would keep
off them.
1. `kernels/dataflow/unified_routed_expert_ffn_reader.cpp`: drop the per-K-block x-sender rotation
   and always use the host's fixed, staggered x sender (args 13/21/22). It is collision-free with
   the group weight senders for every group size >= 2.
2. `kernels/adaptive_chunk.hpp::group_rows`: the smallest group is 2 rows (new `kMinGroupRows`).
   A 1-row group makes every core a weight reader, so no x placement can avoid the collision. At
   2 rows the 3-4-tile shapes (dsv4-pro, kimi-k3) also balance: 12 experts on 4 groups is 3 each,
   where 8 groups gave 2 or 1. All three kernels call the same function, so they stay in agreement.
3. FUSE_BIAS: the reader reads and pushes an expert's biases only when its group runs it
   (`effective_chunks > 0`), and the compute kernel pops them under the same condition. Zero-count
   and other-group experts are now skipped without touching the bias CBs. Bias-free shapes are
   unaffected.

No host-side change, no CB, format or math change.

## Why this is not a repeat
This is a repair of the parent's (r01-b02-a01) runtime x-sender rotation and bias handling, not a
new schedule. It differs from r01-b01-a01, which never had rotation or 1-row groups, in keeping the
parent's simpler group-size rule and round-robin deal. So it also separates the two branches'
confounded differences: if gpt-oss and dsv4-pro reach b01's times while mistral and dsv4-flash keep
the parent's slightly better times, then rotation and the 1-row groups are what cost the parent.
The read-ahead and pipelining nodes (r01-b03-a01, r01-b04-a01) changed the order of reads within
one sender. This changes which core issues the reads.

## Expected effect and risk
- gpt-oss: 477 → ~385 µs (~1.24x over the parent). dsv4-pro: 1211 → ~1140 µs (~1.06x).
  kimi-k3 (1-row → 2-row groups at per_core_M_max 8): about neutral; b01 ran 2-row groups at
  1485 µs against the parent's 1487.
- The 2-row shapes (dsv3, kimi-k2.7, glm, mistral, dsv4-flash, minimax-m3) lose the rotation. If
  rotation was helping them (the parent beat b01 on mistral and dsv4-flash by ~3%), they may regress
  a few percent. The eval will show it, and it tells us whether rotation should come back only
  for the 2-row case.
- Accuracy: bit-identical expected, since every output tile keeps the same K order. Hang risk is
  low: the fixed-sender handshake is the root's own, and the bias push/pop condition is the same
  `effective_chunks > 0` on both sides.
