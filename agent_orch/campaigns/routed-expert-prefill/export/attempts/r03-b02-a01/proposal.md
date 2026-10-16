# r03-b02-a01: combine the best planner (r02-b02-a04) with r02-b01-a04's PACK-thread pipelined activation, and trim the planner prologue (counts cached once, uniform layouts scheduled once, divide-free model)

## Motivation
- The two best nodes are on disjoint code:
  - r02-b02-a04 (1.5697) is r02-b03-a03 plus a planner change in `adaptive_chunk.hpp` only (mixed-height phase B; dsv3 -6.4%, glm -7.9%).
  - r02-b01-a04 (1.5657) is r02-b03-a03 plus `fused_swiglu.cpp`'s PACK-thread activation pipelined into the last gate/up K-block: dsv4-pro -2.2%, mm3 -2.3%, gpt-oss -2.0%, kimi-k3 -1.4% vs r02-b03-a03. Its reader/writer change, the `start` fold into the counts multicast, was measured neutral (r02-b01-a02).
- These are the same two lines whose gains added up in r02-b01-a04 and r02-b03-a04. The planner-only and compute-only changes have not been combined yet.
- r02-b01-a04 measured the cost of the planner prologue directly: kernel start -> first RE-GU went from 6.4 to 29.2 µs on kimi-k3, and from 3-4 to 10-11 µs on dsv3/glm/k2.7/dsv4. That is critical-path time before any DRAM read, on every RISC. The likely causes:
  - ~10^3 `tiles_of` calls, each two volatile L1 loads plus band and clamp. On UNPACK there is also the CB-interface indirection and an ASSERT.
  - Software divides in `expert_cost` / `num_chunks` inside `model_phase` / `model_expert`. These RISC-Vs have no divider.
  - The 3 uniform layouts are scheduled twice: once in `pick_group_rows`, once in `plan_groups`.

## Mechanism
1. Take `compute/fused_swiglu.cpp` from r02-b01-a04 verbatim. It carries the PACK-thread pipelined short SFPU bodies and the RE-GU/ACT/DN MATH zones. It uses the same `adaptive_chunk` API as r02-b02-a04's header. Keep this lineage's reader/writer/factory, so the neutral `start` fold is left out.
2. In `adaptive_chunk.hpp` (planner prologue; every kernel calls `plan_launch`):
   - `plan_launch` reads the tile counts once into a local `uint16_t[E]` and runs every planner pass over that array.
   - The uniform makespans are computed once and shared by `make_plan` and `plan_groups`.
   - `PhaseDealer::deal` and `model_expert` use the divide-free `group_cost` (identical values to `expert_cost`). `model_expert` skips the `num_chunks` divide for single-chunk experts.
   - A host fuzz shows bit-identical plans: 400k random cases per grid height (8 and 10 rows), E in {8, 12, 16, 28}, comparing old and new `plan_launch` layouts and owner maps.

## Why this is not a repeat
- The combination itself is new. r02-b01-a04 combined the pipeline with r02-b03-a03, not with r02-b02-a04's mixed phase B.
- The prologue trim was suggested by r02-b03-a02, r02-b03-a04 and r02-b01-a04 but never tried.

## Expected effect and risk
- dsv3/glm keep r02-b02-a04's ~529/453 µs. The other shapes take r02-b01-a04's numbers (dsv4-pro ~1069, mm3 ~368, gpt-oss ~262, kimi-k3 ~1386).
- The prologue trim should cut kimi-k3's first-GU start by up to ~20 µs and the others' by a few µs.
- Expected score about 1.58-1.59.
- Numerics should be bit-identical: the same plans, and compute code already validated as bit-identical in r02-b01-a04.
- Risk: the compute file merge. The phase-B mixed layouts are planner-side only, and the compute kernel handles any `layout[phase]` through `layout_group`, as in r02-b02-a04. The zones show first-GU start, which separates the prologue effect from the pipeline effect.
