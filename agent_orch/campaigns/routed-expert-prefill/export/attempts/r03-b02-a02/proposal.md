# r03-b02-a02: keep the row-shared x owner off weight-sender cores. Gate/up block j of a row is read, tilized and multicast by a column whose core is not that column's weight sender, so no weight-sender NCRISC carries x duty

## Motivation
- The parent (r03-b02-a01, 1.5887) spends ~2-2.7x the ideal matmul time per gate/up K-block on the shapes that are not DRAM-bound. From its MATH zones (/tmp/r03b02a02/grid.py on reports/r03-b02-a01):
  - mistral: GU ~110 µs for 8 K-blocks at pcM 3, i.e. 13.7 µs per block against ~6.8 µs of LoFi matmul.
  - gpt-oss: GU 157-170 µs on the pcM-3 3-row groups and 164-190 on the pcM-4 2-row groups. That is ~26 µs per block (6 blocks) against 9.6 / 12.8 µs of matmul. +33% M-work costs only ~+10% GU, so GU is not compute-bound there.
  - These are also the shapes with the large "per-expert fixed cost F" (r02-b04-a01, r03-b04-a01): mistral, minimax-m3, gpt-oss, dsv4-flash.
- One structural coupling sits on that per-block path, and no node has removed it:
  - A group's weight sender for column gx is row `group_base + gx % rows` (factory, per layout).
  - The row-shared x owner of gate/up block j is column `j % 11`, in every row.
  - So in a group of R rows, on EVERY block, the row with `my_mt == (j % 11) % R` has its x owner on the column's weight-sender core. r01-b02-a03 estimated this at "~1/11 of K-blocks"; it is actually every block for some row of every group.
- What that core's reader (NCRISC) then does, from unified_routed_expert_ffn_reader.cpp:
  - At the top of iteration kb, the owner of kb+2 issues ~96 x-stick DRAM reads (pcM 3 x 32 sticks). Then, as weight sender, it reads gate(kb) and runs `noc_read.async_read_barrier()`. That barrier is global, so the gate hand-off to the writer (`mcast_go`) also waits for the 96 x reads. The column's gate+up multicast of block kb is delayed by the x read.
  - As the owner of kb, the same RISC then waits for its compute's tilize, the row's acks and the x multicast + write barrier. Only after that does its loop reach iteration kb+1's gate read.
  - The x stick reads are ~6x as many NoC transactions as the gate block (96 vs 16 per block), all issued from the RISC that streams the column's weights.
- In a lock-stepped group (2-slot x / weight rings), a late column delays that column's receiver core, and through the row's x acks the whole group. This happens on every block.

## Mechanism
Kernel-only (JIT). No host change and no CB change.
1. `kernels/adaptive_chunk.hpp`: new `x_owner_col(j, grid_x, rows, my_mt)`. Within an M-row (in-group row `my_mt` of an R-row group), the eligible columns are those with `c % rows != my_mt`: these cores are not their column's weight sender. Block j goes to the `(j % n_ok)`-th eligible column, where n_ok = 11 - #senders, i.e. 5-10 columns. The owner is still a deterministic function of (j, rows, my_mt). Every core of the row shares those, so they agree on the owner. rows < 2 falls back to `j % grid_x`, which never happens here.
2. Reader: `owns_x_block` and the non-owners' ack target use `x_owner_col` with the current phase's `group_rows` / `my_mt`.
3. Compute (`fused_swiglu.cpp`):
   - UNPACK already sends MATH/PACK each worked expert's group height. It now packs the in-group row next to it in the same mailbox word (`rows | my_mt << 16`), so there is no extra mailbox traffic.
   - `matmul_phase_fused_gu`'s two owner tests (pre-tilize of block 0, tilize of block+1) use `x_owner_col`.

Owners are still at least 5 blocks apart, and the 1-block cb_x_rm / cb_x_tl rings work for any spacing ≥ 1. At a two-phase layout switch a core may own close blocks. That only serializes its own tilize; it cannot deadlock (reader reserve -> compute pops after the tilize that precedes the previous matmul).

Numerics: the same RM bf16 -> bf8 tilize on a different core of the same row, with identical tiles and the same math. So results must be bit-identical.

## Why this is not a repeat
- r01-b02-a02 fixed the TILE/rotation collision by moving x to a fixed non-sender column (+25% on gpt-oss then). r01-b01-a02 then reintroduced rotation across ALL columns for the row-shared tilize. r01-b02-a03 ported that and judged the collision harmless at "~1/11 of blocks". Its next #4 ("keep the owner x read off weight-sender columns") was never tried.
- This is not DM read-ahead / ring depth / VC / pacing (r01-b03-a01, r01-b04-a01, r01-b04-a03, r02-b02-a02/a03, r01-b01-a03). No read is moved earlier and no buffering changes. Only which core of a row owns an x block changes.

## Expected effect and risk
- If the coupling is on the critical path: GU per block falls on the few-expert shapes (mistral, minimax-m3, gpt-oss, dsv4-flash). I'd guess -3..-10% there and 0..-2% on the DRAM-bound shapes (dsv3, glm, kimi-k2.7, kimi-k3, dsv4-pro), which are already at ~405-455 GB/s. If the coupling is hidden by the 2-slot slack, the result is neutral.
- Cost: owner tilize work concentrates on ~half the columns of each 2-row-group row (1 in 5-6 blocks instead of 1 in 11). It is still at most one owned block per core per K-block, so no core tilizes twice per block.
- Risks:
  - A reader/compute ownership mismatch would hang (cb_x_rm / cb_x_tl never filled) or garble x. The tool shows hang / accuracy_fail.
  - Binary size: r03-b01-a01 hit the 70656 B kernel-config buffer. This adds a few dozen instructions per RISC. If the program is too large, the eval shows a runtime_error on the h7168 shapes.
- How to tell: the RE-GU zone per core (grid.py) on mistral / gpt-oss / minimax-m3 against the parent (mistral ~110, gpt-oss 157-190, mm3 ~228 µs).

## Addendum after the first eval run (kernel config buffer overflow, repaired before any measurement)
- The first run failed at program creation on the three h7168 shapes (dsv3, kimi-k2.7, dsv4-pro): `Program size (71056 / 72608 / 71328) too large for kernel config buffer (70656)`, the same failure as r03-b01-a01. The parent was just under the limit, and this change pushed it over. No timing was measured for the mechanism.
- Size measurement (/tmp/r03b02a02/sz: the JIT's generated chlkc files plus representative CT args, built LTO -O3 + `-r` link, parent vs this node):
  - The owner test as first written (x_owner_col inlined at two sites in the gate/up phase) added ~+0.8-0.9 KB of compute code.
  - The reader growth is unmeasured.
- Repair (inert for numerics and the plan):
  1. Compute: UNPACK sends each expert's (group rows, row in group). Every thread computes the row's owner-slot count and its own slot once per expert (`x_owner_slots` / `x_owner_rank`). The gate/up phase then tests ownership with a wrapping slot counter, with no divide. The reader computes each block's owner once per K-block through an out-of-line `x_owner_col`. All three helpers agree, checked by brute force against an explicit list of eligible columns (/tmp/r03b02a02/own2.cpp).
  2. `adaptive_chunk.hpp`: on TRISC builds only (`COMPILE_FOR_TRISC`), the launch planner region is compiled with `#pragma GCC optimize("O2")`, i.e. at the dataflow kernels' level instead of -O3. In the JIT ELFs, -O3 made UNPACK's plan_launch + model_phase ~9.5 KB against the reader's ~4.5 KB.
     - The planner runs once per kernel. The plan is integer-only and identical at any opt level.
     - The reader's -O2 copy is the one compute waits on (first weight multicast).
     - Measured TRISC0: -4.4 / -4.9 KB. Compute total vs the parent: -3.8 KB (SiLU config) / -5.2 KB (clamped config).
