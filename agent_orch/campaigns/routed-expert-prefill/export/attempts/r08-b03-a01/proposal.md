# r08-b03-a01: dynamic cross-group pacing of the weight stream: each row group's column-0 weight sender holds its next DRAM block while another group of the phase is more than 2 blocks behind, so the groups that win the shared DRAM/NoC race back off and the starved top groups get the bandwidth

## Motivation
The biggest structured waste left is the spread between row groups that do the same work at the same
time. Per-row zones on the root's profile (reports/r07-b02-a01, /tmp/r06b01a01/zones.py), first gate/up of
identical experts, top group -> bottom group (µs):

| shape | GU0 per group, top -> bottom | max row end vs mean row end |
|---|---|---|
| dsv3 (phase A 5x2) | 295 / 263 / 200 / 138 / 130 | 509 vs 454 |
| kimi-k2.7 (phase A 5x2) | 364 / 307 / 185 / 124 / 119 | 699 vs 672 |
| dsv4-pro (phase A 5x2) | 473 / 398 / 260 / 193 / 179 | 996 vs 951 |
| minimax-m3 (4 x 2 rows) | 249 / 225 / 188 / 176 | 333 vs 295 |
| mistral (4 x 2 rows) | 122 / 113 / 93 / 83 | 167 vs 146 |

Identical work runs 1.4-3x longer on the top group. On the 1-expert-per-group shapes (mm3, mistral, and
dsv3 / glm phase A) the chip time is the top group, while the bottom groups idle for 40-170 µs. On the
multi-expert shapes the desync makes the top group the tail (r07-b02-a01 reflection point 3).

What earlier nodes established about this gradient:
- It is not the group's own NoC split (r06-b01-a01), not x-owner placement (r05-b03-a01), not fixed by
  extra NoC-1 endpoints (r07-b03-a01). Byte placement moves bandwidth between rows without raising the
  total.
- It is a first-mover / relative-position effect: a static start-up stagger hands the lead to whichever
  group starts first (r06-b02-a01: mm3 top GU 259 -> 210 µs when the lower groups were held 5-15 µs, but
  the held groups then lost), and dealing work away from the top just makes the next active group the
  loser (r07-b04-a01). Both reflections conclude the cure has to be dynamic: a run-time share cap.
- The total bandwidth is roughly conserved (~450-475 GB/s): when one group is held, the others speed up.

## Mechanism
Feedback pacing on the issue of weight DRAM reads (new header `kernels/group_pacer.hpp`, reader only):
- In every phase, the column-0 weight sender of each row group is that group's pacer. Every weight block
  it issues (gate/up block: gate + up head on NoC 0 and the writer's up release; down block: head read
  and the writer's tail release) is first "taken" from the pacer.
- Taking block j: wait until every other group g of the phase has taken at least min(T_g, j - 2) blocks,
  where T_g is g's total block count this phase (every core derives it from the shared plan: chunks x
  (num_blocks_gu + num_blocks_d) per expert). Then count j and send one NoC-0 atomic increment to each
  other group's pacer.
- The board is the 16-byte L1 slot of one new program semaphore (the 16th): word s counts the s-th other
  group's blocks, phase A in the low 16 bits, phase B in the high 16. The reader clears it at kernel start
  (before the counts broadcast; no peer can count into it before it has the counts and its plan).
- Look-ahead sites (try_gu_lookahead, try_down_lookahead) take non-blocking (the look-ahead simply does
  not fire while held); the top-of-block sites wait.
- One column per group is enough: a held column-0 sender stops its rows' x / activated streams within
  a block, so the whole group backs off.
- Safety: the group furthest behind never waits (its peers are all at or above its count, or finished),
  so there is no deadlock even across columns (only column 0 paces). A wait also gives up after 32768
  cycles (~24 µs). Peers in column 0 are addressed by stepping NoC y with the grid row; the host enables
  pacing only after checking that column 0's NoC y is linear in the row (DS_NO_PACE=1 turns it off).
- Data path, slots, byte counts and math are unchanged: bit-identical outputs expected.

Supporting change for kernel-config space (the reader grows ~1.0 KB in a local -O2 profiler compile):
the profiler-only MATH perf counters (re_hwprof.hpp) become opt-in (`RE_HWPROF`), as r07-b03-a01 did on
its lineage (~-1.0 KB TRISC1 in profiled builds). They only exist in profiler builds; non-profiled
binaries are unchanged by it.

Files: kernels/group_pacer.hpp (new), kernels/dataflow/unified_routed_expert_ffn_reader.cpp,
kernels/compute/re_hwprof.hpp, device/unified_routed_expert_ffn_program_factory.cpp (semaphore + define).

## Why this is not a repeat
- r01-b01-a03 (expert-group pacing via a grid-multicast progress board, 1.04): its board was published by
  a NON-POSTED 87-destination multicast from the column-0 core every block, inside that core's existing
  write barriers (x owner / activated loopback), which its reflection identifies as a ~3-4 µs per-block
  cost on every group, gpt-oss included (35% slower with 1% imbalance). Its slack was fraction-based.
  Here a pacer sends at most 4 unicast atomics per block to the 4 peer pacers in its own column, no
  multicast and nothing inside any write barrier; receivers read their own L1. The board is per peer
  (min-based, finished groups capped at their totals) and the wait is skipped by every core except one
  per group. The root has also changed completely since round 1 (row groups of 2-5 rows, look-ahead
  reads, wrap-free multicasts), and the gradient is now measured and deterministic (r07-b02-a01).
- r06-b02-a01 (static start-up stagger) and r07-b04-a01 (static deal toward the bottom): static; both
  reflections ask for exactly a dynamic share cap.
- r06-b01-a01 / r07-b03-a01 / r05-b02-a01 (NoC byte placement): this does not move any byte between NoCs.

## Expected effect and risk
- mm3, mistral: the 4 groups should end GU0 together near the mean (mm3 ~205-210 vs 249 on the top group,
  mistral ~100 vs 122): chip -8..-15%.
- dsv3 / glm / kimi-k2.7 / dsv4-pro: phase A ends together, phase B starts together and is paced too:
  -3..-8%. kimi-k3 (6/6/6/5/5): -3..-8% (spread 1070 mean vs 1212 max). dsv4-flash: -3..-6%.
- gpt-oss: its 2-row pcM-4 groups are compute-bound and slower; pacing holds the 3-row groups to them,
  which costs nothing at the makespan: ~neutral.
- Risks: (a) the aggregate rate drops if holding the leaders leaves too few reads in flight (then every
  group gets slower, as in r01-b01-a03; visible as mean row end up); (b) kernel-config overflow on the
  h7168 programs (runtime_error; repair by trimming); (c) a pacing bug can only delay reads (timeout),
  never change data, so any accuracy change would be a separate bug.
- Judge by per-row ends (/tmp/r06b01a01/ends.py) and per-row GU0 (/tmp/r06b01a01/zones.py): the
  top -> bottom spread must shrink, and the mean must not grow by more than the spread shrinks.
