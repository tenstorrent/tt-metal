# r08-b04-a01: cross-group weight-stream pacing: each row group's weight senders hold their next block's DRAM reads while the group is more than one gate/up block ahead (by fraction of its own weight bytes) of the slowest busy group, read from a 5-counter progress board on core (0,0)

## Motivation
Since r07-b02-a01 removed the wrapped multicasts, the per-call variance is gone and what remains is a
**deterministic row gradient**: the topmost active row groups get a much smaller share of the shared
DRAM/NoC stream than the bottom ones, for identical work. From reports/r07-b02-a01 (RE-GU / RE-DN zones,
mean of the 10 measured calls, /tmp/r06b01a01/zones.py):

| shape | group, top -> bottom | first-expert GU (µs) | DN (µs) |
|---|---|---|---|
| mistral (1 expert per group) | y2-3, y4-5, y6-7, y8-9 | 122, 113, 93, 83 | 35, 34, 35, 33 |
| minimax-m3 | same | 247, 225, 188, 176 | 79, 80, 84, 81 |
| dsv3 phase A (1 expert per group) | y2-3 ... y10-11 | 295, 263, 200, 139, 131 | 67, 70, 77, 67, 60 |
| kimi-k3 (first expert) | y2-3 ... y10-11 | 177, 166, 146, 127, 122 | |

dsv3's top phase-A group ends its (identical) expert at ~373 µs, the bottom one at ~203 µs. The chip
time is the top group's end on every DRAM-bound shape: max row end vs mean row end is 509/454 (dsv3),
440/394 (glm), 167/146 (mistral), 333/295 (mm3), 1212/1070 (kimi-k3), 287/263 (dsv4-flash). That
max - mean gap (5-13%) is the largest waste left, as every reflection since round 1 says.

What is known about the gradient:
- it is a share effect, not a fixed per-row cost: r06-b02-a01 delayed the lower groups by 5-20 µs and the
  top group's GU fell 259 -> 210 µs (mm3); r07-b04-a01 left the top group idle and the next one down
  became the slow one. Whichever groups are highest among the ones streaming lose (r07-b04-a01 next #3:
  "the cure has to be dynamic: a run-time share cap or priority").
- static placements and offsets only rotate the loser (r06-b01, r06-b02, r07-b04); NoC/byte placement
  moves bandwidth between rows without raising the total (r06-b03, r07-b03).
- DN is equal on all rows, so a group's own pipeline is not slower; it gets less of the shared stream.

So: if the groups that are ahead hold their weight reads until the others catch up, the starved groups
get that share and every group should end near today's MEAN row end.

## Mechanism
Reader kernel (one file) plus a 64-byte CB and a define in the program factory. Writer and compute are
untouched (the writer's weight reads are already released by the reader's up_go / down_go).

1. **Progress board.** New CB c_21 (`RE_PACE_CB`, 64 B on every core). On core (0,0), the counts reader,
   bytes [0, 32) are the board: one cumulative counter per row group of the paced phase. Every core
   uses bytes [32, 64) as its local snapshot of the board. Every reader zeroes both at kernel start; on
   (0,0) that happens before the counts-valid multicast, which every other core waits for before it does
   anything, so no increment can precede the reset.
2. **Units = weight tiles per sender**: a gate/up block counts 2 x in0_block_w_gu x per_core_N_gu, a
   down block in0_block_w_d x per_core_N_d. Each group's total for the phase is
   chunks(expert, group rows) x (blocks_gu x U_gu + blocks_d x U_d) summed over its experts, computed by
   every core from the launch plan it already has. Groups with no experts have total 0 and never count.
3. **Publish**: one sender per group (the column-0 weight sender) does one NoC atomic increment of its
   group's board counter by the block's units whenever it starts a block (gate/up or down, main path or
   look-ahead). No multicast, nothing any write barrier waits on (r01-b01-a03's cost was an 87-way
   non-posted multicast per block from a critical core under its write barriers).
4. **Decide**: every weight sender, before it starts its next block (before the up_go / down_go release
   and its own reads), checks the snapshot: may go iff for every other busy group h
   (done_h < total_h): (done_me - U_gu) x total_h <= done_h x total_me (64-bit). So a group may run at
   most one gate/up block ahead of the slowest busy group in fraction of its own work. The snapshot is
   refreshed by one 32-byte NoC read issued with every block start (it lands under the block's existing
   read barrier). In the blocking path the sender polls (read + barrier) until allowed, capped at 64
   polls so a broken board can only slow a block, never hang. The non-blocking look-aheads just don't
   fire while held.
5. **Scope**: only the first non-empty phase (phase A of a two-phase plan, else the single phase), only
   layouts with 2+ groups. Phase B is unpaced: its groups start when their rows finish phase A, and with
   phase A paced they start together. `DS_NO_PACE=1` drops the define and restores today's kernel
   behaviour exactly.
6. **Size**: the h7168 programs are ~0.3 KB under the 70656 B kernel config buffer on this lineage. To
   make room I port r07-b03-a01's one-line change that makes the profiler-only RE-HW MATH counters
   opt-in (`RE_HWPROF`, ~1.0 KB TRISC1 in profiled builds). It has no effect outside profiler builds and
   only removes counter reads from MATH. This is a size enabler, not a second mechanism.

No byte, slot, order or format of any data changes, so the outputs stay bit-identical.

## Why this is not a repeat
- r01-b01-a03 (the only pacing attempt, on a lineage with random per-call tails): balanced the groups to
  ~1% but paid ~3-4 µs per block on every shape (gpt-oss +35% with 1% imbalance), from a per-block
  87-destination non-posted multicast on the column-0 core inside its write-barrier scope. This node
  publishes with one atomic and reads with one 32-B read per block, nothing inside a write barrier, and
  only the senders do anything. The target is also different: then the tail was random per call, now it
  is a measured, deterministic share gradient of 40-80%.
- r06-b02-a01 (static start stagger) and r07-b04-a01 (idle top group) are static and only rotated the
  loser; this is a closed-loop cap that follows whichever group is slowest.
- Not a planner or NoC-placement change (r05-r07's other branches).

## Expected effect and risk
- If the aggregate stream rate holds, every group ends near today's mean row end: dsv3 / glm -8..-12%,
  mistral / mm3 -8..-12%, kimi-k3 -8..-12%, dsv4-flash -5..-8%, kimi-k2.7 / dsv4-pro -2..-4% (phase A
  only), gpt-oss ~0 (its slow groups are compute-bound pcM-4 groups, so pacing only delays the others).
  Up to +5..+7% geomean if fully effective.
- Risks: (a) the aggregate rate drops when the leaders hold (fewer requests in flight), which r06-b02
  saw for a static stagger; (b) all groups aligning their GU -> DN transitions so DRAM idles in the ACT
  gap together (r01-b01-a03's "convoy" worry); (c) poll/read overhead; (d) kernel config overflow
  (runtime_error) if the reader grows more than the RE_HWPROF space; (e) hang only if the cap logic is
  wrong (bounded polls make it impossible by construction).
- How to tell: per-row GU / end tables (zones.py, ends.py): the row gradient in the first expert should
  collapse; the mean row end should stay at today's value. If the gradient collapses but the mean rises
  by as much as the max falls, the share is zero-sum with a loss and pacing is not the lever.

## Addendum (before the eval): absolute progress instead of fractions
The first build of points 2 and 4 (per-group totals from the plan, 64-bit fraction compares) grew the
reader by +1.2 KB locally, more than the RE_HWPROF space. The built version compares **absolute**
progress instead: a group may start a block while its counter is at most one gate/up block of weight
tiles above every other group's counter, and a group that has started all its blocks of the phase adds
2^30 (`kPaceDone`) so it never holds anyone (idle groups do so at once). No totals, no 64-bit math:
reader +964 B, TRISC1 -1020 B (local, profiler builds). In the benchmark every paced phase gives its
groups equal work except kimi-k3 (6/6/6/5/5 experts): there the 5-expert groups are held to the others'
pace and finish one expert early, and the last expert runs on 3 groups (fractions would have spread it).
The publish and the gate are in one out-of-line `Pacer` (start / publish).
