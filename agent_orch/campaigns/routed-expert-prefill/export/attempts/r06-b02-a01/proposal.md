# r06-b02-a01: plan-aware start-up stagger: each lower row group holds its first DRAM read ~2.5 µs per grid row below the top group, so the top (critical) groups' first gate/up blocks land before the lower groups' burst competes with them

## Motivation
On the parent (r05-b02-a01) the top row groups are the slowest groups and set the chip time on 8 of 9 shapes.
Per-row end times (mean of the 10 measured calls, `/tmp/r04b01a02/rows.py`):

| shape | top rows end | bottom rows end | bottom slack |
|---|---|---|---|
| dsv3 | y2-5 503 | y9-11 409 | 94 µs |
| glm | y2-5 435 | y9-11 352 | 83 |
| kimi-k2.7 | y2-6 698 | y7-11 654 | 44 |
| kimi-k3 | y2-5 1216 | y8-11 943 | ~270 |
| mistral | y2-3 177 | y8-9 142 | 35 |
| dsv4-flash | y2-4 294 | y10-11 238 | 56 |
| dsv4-pro | y2-6 990 | y7-11 926 | 64 |
| minimax-m3 | y2-3 359 | y8-9 298 | 61 |
| gpt-oss | y2-4 227 | y10-11 247 | bottom is critical (pcM-4 groups) |

The groups do the same work: mistral / minimax-m3 run one expert at per_core_M 3 per 2-row group, and dsv3 / glm
phase A runs one expert at per_core_M 3 per 2-row group. Yet mistral's gate/up window is 127-130 µs on y2-3 and
83-84 µs on y8-9, and dsv3's phase-A gate/up window is 263 µs on y2-5 and 132-136 µs on y10-11.

Most of that gap is **start-up**, not steady state:
- r05-b03-a01's profiler-only RE-X0 zone (reports/r05-b03-a01, `/tmp/r06b02/x0.py`, `x0grid.py`) shows when x block 0
  lands per row. mistral: y2 43 µs, y3 37, y4 36, y5 22, y6 22, y7 19, y8 19, y9 13. dsv3: y2 76 µs down to
  y11 15 µs. gpt-oss: y2 ~45 µs down to y11 ~13 µs.
- The pure receiver cores of every row reach their first x wait at 5-8 µs on every row, so planning, counts and the
  `start` page are not staggered. The cores that are late are the ones that issued DRAM reads first: the weight
  senders (block-0 gate read + barrier, then the writer's `up` read) and the x owners (blocks 0/1 stick reads).
  In mistral those reads complete at 28-41 µs on y2 and 10-19 µs on y9.
- mistral's gate/up window minus x(0) landing is about the same on every row (top 78-85 µs, bottom 70-83 µs), so
  the ~35 µs row gap of the chip time is the start-up gap.
- The gradient follows contention, not just geometry. The bottom-most ACTIVE row is always first (mistral y9 at
  13 µs with y10-11 idle; dsv3 / gpt-oss y11 at 13-15 µs, while their y9 lands at 20-25 µs). With 5 groups
  (dsv3) the top row is 76 µs late; with 4 (mistral) 43 µs. A uniform-demand NoC-0 model also loads the column
  links into the top rows ~30% more than those into the bottom rows (responses from the row-11 / row-8/9
  endpoints wrap down into y2-4), so under a synchronized burst the top rows' first responses lose.
- The 60/40 NoC split (parent) and all-NoC-0 gate/up (r03-b03-a02) both made the top rows of the
  1-expert-per-group shapes slower and the bottom rows faster, i.e. extra traffic on the shared links lands on the
  top rows' start-up.

So at kernel start every group issues its first blocks at the same moment, the top groups' reads are served last,
and the top groups (the critical ones) start 20-60 µs after the bottom groups (which then idle at the end).

## Mechanism
One wait in the reader, before the core's first DRAM read of the launch:
- `adaptive_chunk::startup_stagger_cycles(plan, owner, tiles_of, my_gy, per_core_M_max)` (new, in
  `kernels/adaptive_chunk.hpp`, instantiated by the reader only):
  - takes the first phase that runs (phase A of a two-phase plan, else the single layout);
  - returns 0 for the top group (group base row 0);
  - for any other group, returns `kStaggerCyclesPerRow * group_base_row` (2.5 µs per grid row, 3375 cycles at
    1.35 GHz) **only if** the group's modelled load in that phase (summed `group_cost` of its experts) is not
    larger than the top group's. A group with more work than the top one (gpt-oss's 2-row per_core_M-4 groups, or
    kimi-k2.7's one 7-unit group) starts immediately.
  - All cores of a group share the group base and the loads, so a group's sender, receivers and x owners wait the
    same time. That keeps the group's own handshakes in step, and no group waits on another group in the first
    phase (weight multicasts are per group, x / activated multicasts per row).
- `kernels/dataflow/unified_routed_expert_ffn_reader.cpp`: after the plan, compute the stagger. At the first expert
  of that phase with `effective_chunks > 0`, `riscv_wait(stagger)` once (before the bias reads, the up-front x reads,
  and the first gate read / `up_go`), then clear it. It is also cleared at the end of the first phase, so phase B
  and later units never wait.
- The writer's DRAM reads (`up`, down tail) are released by the reader's go semaphores, so they follow the wait
  with no writer change. Compute waits for data as before.

Per shape (from the planner compiled natively, `/tmp/r06b02/plan/drv.cpp`), the delay of each group's first read:
- mistral / minimax-m3 (5 x 2, 4 experts): 0 / 5 / 10 / 15 µs;
- dsv3 / glm / dsv4-pro phase A (5 x 2): 0 / 5 / 10 / 15 / 20 µs;
- kimi-k2.7 phase A: 0 / 5 / 0 (more load) / 15 / 20 µs;
- kimi-k3 (5 x 2, 18/18/18/15/15): 0 / 5 / 10 / 15 / 20 µs;
- dsv4-flash (3+3+2+2, equal loads): 0 / 7.5 / 15 / 20 µs;
- gpt-oss (3+3+2+2, loads 4/4/5/5): 0 / 7.5 / 0 / 0 µs.

Every delayed group has 35-270 µs of slack today, more than its delay.

Numerics: unchanged, bit-identical. Same reads, same order per RISC, same slots; only a time offset before the first
read. Every code path (TILE x, bias, activations, 0 tokens, experts_per_chip > 1) runs the same wait or none. A
group with no expert in the first phase issues no reads, so its wait costs nothing.

## Why this is not a repeat
- r01-b01-a03 (expert-group pacing) held groups that were ahead throughout the call via a grid-multicast progress
  board (-0.43). This is a one-time, static, no-communication offset at kernel start, sized by row position and
  gated by the plan's loads. No semaphores, no polling of other cores.
- r05-b03-a01 spread the x owners over the columns (null). It measured the row gradient (RE-X0) and suggested this
  start-up stagger as "the cheap test" (its next #1).
- r05-b02-a01 (parent) and r03-b03-a02 / r02-b04-a02 changed which NoC carries the bytes. This changes when the
  lower groups start, not where any byte goes.
- Planner nodes (r02-r05) choose layouts and deals; this keeps the plan and changes only start timing.

## Expected effect and risk
- If the top groups' first reads are slow because of the lower groups' concurrent burst: the top groups' x(0) /
  first weights land 10-30 µs earlier, and the chip time (top groups) drops by about that much. mistral and
  minimax-m3 -5..-12%, dsv3 / glm / dsv4-flash -3..-8%, kimi-k2.7 / dsv4-pro / kimi-k3 -1..-2%, gpt-oss ~0.
  Geomean maybe +2..+4%.
- If the top rows are slow for a reason that does not depend on the other groups' traffic (path latency alone): the
  top groups don't move and the lower groups end up to 15-20 µs later, still inside their slack. Chip time ~0,
  except kimi-k2.7 (44 µs slack vs 20 µs delay) could tighten. Neutral.
- Over-correction: a lower group that becomes critical shows up as the bottom rows ending last in `rows.py`.
- Hang risk: none from the wait itself (bounded spin on the wall clock, no dependency). Code size: ~100-150 B on
  NCRISC only.
- How to tell: per-row GU / DN windows and ends (`/tmp/r04b01a02/rows.py`, `/tmp/r06b02/gu.py`). The top rows' gate/up
  window should shrink and the lower rows' should grow by about their delay.
