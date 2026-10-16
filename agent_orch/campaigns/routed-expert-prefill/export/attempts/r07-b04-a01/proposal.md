# r07-b04-a01: mirror the uniform-layout expert deal: list scheduling breaks load ties toward the bottom row group, so an uneven deal's extra experts (kimi-k3 28 on 5) go to the lower, faster groups and a short deal (mistral / minimax-m3, 4 on 5) leaves the top group idle

## Motivation
The row groups don't run at the same speed. When every group streams at once, the lower grid rows finish first. The top
groups' first expert runs up to ~2x longer (r06-b01-a01 zones; r06-b02-a01: a self-reinforcing first-mover effect
that a static start offset can't remove). Per-row kernel end on the parent r06-b04-a01
(`/tmp/r06b01a01/ends.py` on reports/r06-b04-a01, µs):

| shape | top group | ... | bottom group |
|---|---|---|---|
| kimi-k3 (5x2, 6/6/6/5/5 experts) | y2-3 1214 | y4-5 1201, y6-7 1146, y8-9 928 | y10-11 894 |
| mistral (5x2, 4 experts, y10-11 idle) | y2-3 171 | y4-5 164, y6-7 145 | y8-9 137 |
| minimax-m3 (same plan) | y2-3 326 | y4-5 329, y6-7 311 | y8-9 296 |

The uniform-layout planner (`schedule_in_order`, `GroupScheduler::assign`) gives each expert to the least-loaded
group, with ties going to the LOWEST index, i.e. the top rows. So:
- kimi-k3 deals 6 experts to each of the three slow top groups and 5 to the two fast bottom groups. The bottom
  groups then idle for ~300 µs, more than one expert (~190 µs).
- mistral / minimax-m3 leave the fastest group (y10-11) idle and run on the slowest one (y2-3).

r05-b02-a01 next #1, r06-b01-a01 next #2 and r06-b02-a01 next #3 all proposed exactly this. Nobody has tried it.

## Mechanism
`adaptive_chunk.hpp` only: in `schedule_in_order` and `GroupScheduler::assign`, change `loads[g] < loads[best]` to
`<=`, so a load tie goes to the highest-index (lowest) group.
- On a uniform layout all groups have the same height, so this is a pure relabelling g -> G-1-g of the same
  schedule. The makespans, and hence every layout / two-phase choice, are unchanged. Native replay of the kernel
  planner on the test counts (/tmp/r07b04/pt.cpp, diff vs the parent):
  - kimi-k3: per group (top -> bottom) 5/5/6/6/6 experts, was 6/6/6/5/5.
  - mistral, minimax-m3: experts on y4-11, y2-3 idle (was y2-9 busy, y10-11 idle).
  - The other 6 shapes: identical plans. Phase A deals round-robin, and phase B is by rank or mixed LPT, both
    untouched.
- Mixed-height LPT (`schedule_lpt`) is left alone. Its groups differ in height, so a tie flip there would not be
  a pure relabelling.
- Every RISC runs the same header, so reader, writer and compute agree on the new owners. No host change, no
  size change (one comparison opcode, two places).

## Why this is not a repeat
- r04-b03-a03 / r05-b01-a01 (idle-row tie pass) changed the LAYOUT for mistral / minimax-m3 to 3+3+2+2. That code
  isn't on this lineage. This node keeps the layout and only moves which group idles.
- r06-b01-a01 (row NoC tilt) and r06-b02-a01 (start-up stagger) tried to remove the row gradient. Both were
  neutral or negative, and both reflections recommend using the gradient in the planner instead. This node does
  that.
- No earlier node changed the expert -> group order.

## Expected effect and risk
- kimi-k3: the max moves from the top groups (1214) to roughly y6-7 (~1146) or y8-9 (928 + one expert ~ 1120),
  so about -4..-6% chip.
- mistral / minimax-m3: if the gradient is positional (it was on the NoC-0 request-path evidence), the critical
  group becomes y4-5 (164 / 329 today), about -3% / -0..-2%. If the gradient is relative ("whoever is on top
  loses"), about 0.
- Other shapes: bit-identical plans, so noise only.
- Geomean guess +0.5..+1%.
- Risks:
  - Accuracy: none. Same experts, same per_core_M, same numerics; only which rows run which expert changes.
  - Hang: idle groups and unequal expert counts are handled generically (count-0 skip), and nothing
    synchronises across groups in a single-phase plan.
  - The top group idling is new (y2-3 hosts nothing special in the role layout). The "big" and "mixed0" replay
    cases also mirror cleanly.
- How to tell: per-row ends (ends.py). kimi-k3's y2-5 should drop by ~190 µs and y8-11 rise by about as much.
