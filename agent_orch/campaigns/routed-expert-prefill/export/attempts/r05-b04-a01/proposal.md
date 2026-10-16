# r05-b04-a01: torus-aware weight-sender rows: the gate/up sender of every group column is the group's bottom row and the down sender its top row, so neither column weight multicast wraps around the NoC torus

## Motivation
Every group column multicasts its weight blocks to the group's rows:
- gate/up: the sender's WRITER, on NoC 1, which routes -y;
- down: the sender's READER, on NoC 0, which routes +y.

The sender row is `group_base + gx % rows` (host topology, staggered against the TILE x sender in r01-b02-a02).
A NoC-1 multicast from the group's TOP row has to reach the rectangle's NoC-1 start corner (the bottom row) by
going -y. So it wraps through the whole column of the torus: every other group's rows plus the non-worker rows,
about 10 hops. The NoC-0 down multicast from the BOTTOM row wraps the same way in +y. Both multicasts are `linked`,
so the wrapped path stays reserved while 50-110 KB stream through it. That happens in the same column links that
carry the other groups' wrapped multicasts and the NoC read responses into that column.

Measured on r04-b03-a01's UNPACK-thread wait tallies (RE-UW, the only node with per-core CB-wait times;
`analysis/wrap.py`). Receivers only, mean over the 10 measured calls:

| shape (layout) | gate/up weight wait: wrapping / non-wrapping column | down weight wait: wrapping / non-wrapping |
|---|---|---|
| kimi-k3 (2x5) | 156.5 / 108.7 µs | 130.7 / 46.1 µs |
| mistral (2x4) | 14.9 / 9.3 | 6.7 / 0.6 |
| minimax-m3 (2x4) | 31.1 / 23.7 | 15.6 / 4.1 |
| dsv4-flash (3,3,2,2) | 44.2 / 21.6 | 18.3 / 9.0 |
| gpt-oss (3,3,2,2) | 10.4 / 2.4 | 4.1 / 1.1 |

Every shape shows it, and in both phases: receivers whose column multicast wraps wait 1.4-4x longer for their
weights. The same parity shows up per column (`analysis/parity` in context.md): even gx (top-row sender) is late
in gate/up, odd gx (bottom-row sender) is late in down.

The rows are lock-step across all 11 columns (the x stream in gate/up, the activated stream in down), so the late
half of the columns paces every row of the group. The largest remaining waste in the op is also lateness: per-call
max minus median core is 20-90 µs on 7 of 9 shapes (`analysis/strag.py` on r04-b02-a01). Those tails come from
whole groups (and single cores) whose row/column streams run 1.5-2x slow for a phase (`analysis/slowuw.py`,
`grp2.py`). They are random per call, which fits contention between wrapped, path-reserving multicasts.

## Mechanism
Each phase's sender row is chosen so its multicast starts at the rectangle's own start corner:
- gate/up (NoC-1 writer multicast, -y): sender = the group's **bottom** row, in every column;
- down (NoC-0 reader multicast, +y): sender = the group's **top** row, in every column.

Each RISC keeps its NoC and its role: the reader still reads gate and half of down, multicasts down, and handles
x/act; the writer still reads up and the other half of down, and multicasts gate/up. Only which core of the
column plays the role changes, and it now differs per phase. Every group has >= 2 rows, so the two senders are
always different cores.

Files:
- `device/unified_routed_expert_ffn_program_factory.cpp`: per-layout topology sets carry the gate/up sender (bottom
  row) as `is_sender` and its coordinates. The down sender is the rectangle's start corner, so it needs no args.
  The TT_FATAL that kept the TILE-path x sender off the weight sender is dropped for the group topologies: every
  core of a bottom row is now a gate/up sender. That collision only costs time on the (unbenchmarked) TILE path;
  each role keeps its own semaphores.
- `kernels/dataflow/unified_routed_expert_ffn_reader.cpp`: gate/up uses `is_gu_sender` and acks the bottom core.
  Down uses `is_dn_sender = (my_mt == 0)` and acks the top core (rect start). The pre-down writer-multicast drain is
  keyed on the gate/up sender.
- `kernels/dataflow/unified_routed_expert_ffn_writer.cpp`: the up read + gate/up multicast run on the gate/up
  sender. The DOWN_SPLIT half-read runs on the down sender (`my_mt == 0`).
- `kernels/adaptive_chunk.hpp`: the row-shared x owner no longer skips "this row's weight-sender columns".
  - Every core of a bottom row is now a gate/up sender, and no core of the other rows is one.
  - So the owner is simply block j -> column j % GRID_X, as before r03-b02-a02, which measured that exclusion as
    neutral.
  - Reader and compute share the helper, so they stay in agreement.

Semaphores: gate/up acks now go only to bottom cores and down acks only to top cores, so a core's in1_ready only
ever counts one phase's acks. The writer-lag hazard (the gate/up sender's writer sets its own in1_valid after its
reader has moved on to down) keeps its existing drain (`mcast_done_sem.wait_min(mc_seq)` before phase 4), now on the
gate/up sender.

## Why this is not a repeat
- r01-b02-a02 / r03-b02-a02 placed senders to avoid RISC collisions (x vs weight work). Nothing so far has looked
  at NoC route direction or wrap-around.
- r02-b02-a02 (request VCs), r02-b04-a02 / r03-b03-a02 (moving a whole weight stream to one NoC) changed NoC usage,
  not the multicast geometry. This node keeps every stream on the NoC and RISC it uses today.
- The handshake / ack / look-ahead nodes (r04-*) changed when blocks are sent. This changes the path they take.

## Expected effect and risk
- Bit-identical outputs: same blocks, same slots, same order; only which core issues the reads and multicasts.
- The weight wait of the wrapping half of the columns should drop to the non-wrapping level. Expect:
  - gate/up and down windows -2..-5% on kimi-k3 / dsv4-flash / minimax-m3;
  - fewer slow-mode calls (the 20-90 µs per-call tails), if wrapped-path contention feeds them;
  - small or no gain on the DRAM-bound h7168 shapes.
  Overall guess: +1..+3%.
- Risks:
  - The bottom-row cores' NCRISC now does gate reads AND every x-owner duty of that row. r03-b02-a02 measured this
    neutral, but it is now 100% of the bottom row's owner blocks.
  - Hang, if a sender role or ack target is inconsistent between reader and writer.
  - Kernel-config size: the change mostly removes logic.
  - Host rebuild needed (topology args).
- How to tell:
  - re-run `analysis/rows.py` / `strag.py` (mean and max core end, per-call max - median);
  - GU/DN zone means vs r04-b02-a01.
