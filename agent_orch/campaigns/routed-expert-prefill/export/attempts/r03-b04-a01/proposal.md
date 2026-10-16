# r03-b04-a01: cut the launch planner's prologue cost

The change is bit-identical. Every RISC plans from a once-read byte cache of the tile counts and a largest-first order. Single-chunk costs no longer divide, the uniform layouts are scheduled once, phase A is modelled in one pass, and hopeless mixed layouts are pruned. The `start` page read overlaps the planning.

## Motivation
Every RISC of every core (reader, writer, UNPACK) runs `adaptive_chunk::plan_launch` before its first DRAM read. Its cost is therefore critical-path time on every call, with DRAM idle.

r02-b01-a04 measured the time from kernel start to the first RE-GU zone. The b03-lineage planner that the root still runs (make_plan + plan_groups) costs:
- kimi-k3: 29.2 µs, against 6.4 µs with the old uniform picker.
- dsv3 / glm / kimi-k2.7 / dsv4: 10-11 µs, against 3.2-3.9 µs.
- The 4-expert shapes: 5-6 µs, against 2.5 µs.

r02-b03-a02 saw the same cost as a +10..+21 µs median-core shift on kimi-k3 / dsv4-flash. Four reflections list "planner prologue trim" as an untried next step: r02-b03-a02 #1, r02-b03-a04 #3, r02-b02-a04 #3 and r02-b01-a04 #1. The root (r02-b02-a04) adds mixed phase-B scoring on top, so it costs slightly more.

Where the time goes: I compiled the root header for BH (riscv-tt-elf-g++ -mcpu=tt-bh -Os, the JIT default) and ran plan_launch on the test's routing in a small RV32IM interpreter (/tmp/rep-r03b04a01-rO2J/sim). On kimi-k3 it takes 50k instructions:
- 1312 L1 loads, because `tiles_of` re-reads `counts[idx[e]]` (two L1 words) about 47 times per expert.
- 893 `divu`, from expert_cost / num_chunks inside the two-phase model.
- The three uniform layouts are scheduled twice: by pick_group_rows and again by plan_groups.
- model_phase deals every expert twice per call.
- schedule_lpt walks all experts once per distinct tile count, for each mixed layout.

## Mechanism
`kernels/adaptive_chunk.hpp` is the only planner change. The plan it produces is unchanged:
1. **`PlanTiles`.** At entry, read each expert's tile count once into a local `uint8_t` cache. Counts >= 255 use an escape value that re-reads through `tiles_of`, so the cache is exact for any count. The same walk builds the active experts in largest-first order (tiles descending, index ascending) by stable insertion.
2. **Divide-free single-chunk costs.** `num_chunks`, `expert_cost` / `group_cost` (inlined) and `model_expert` take the single-chunk path without a divide. Multi-chunk experts keep the general path in a non-inlined helper.
3. **Uniform layouts scheduled once.** `uniform_makespans` schedules all three uniform layouts in one walk. `make_plan`'s uniform pick and `plan_groups` both reuse the result.
4. **Planning from the largest-first order.**
   - `schedule_lpt` makes one pass over the order instead of one pass per distinct count.
   - make_plan's `thresh` / `above` / leftover list come straight from the order. Phase A is the first n_a experts of the order, and phase B is the rest, with the k-th leftover on group k.
   - The phase-A model is one pass: round-robin groups all hold n_a / groups_a experts, so streams = groups_a.
   - The single-phase model records each expert's deal in pass 1 instead of re-dealing it in pass 2.
   - make_plan returns early, before any modelling, when no phase-B layout qualifies.
   - A mixed layout whose cheapest placements cannot fit under the cap is skipped (`lpt_may_fit`). If no assignment fits, LPT fails too, so the result is identical.
5. **`start` read overlapped.** In the reader and the writer, the `start` page DRAM read is issued before planning, and its barrier moves after `plan_launch`. Its latency (~1-2 µs) now runs under the planning.
6. **Measurement.** A `DeviceZoneScopedN("RE-PLAN")` zone wraps plan_launch on NCRISC (reader) and on UNPACK (compute), so the reflection can read the planner time directly.

Files changed: `kernels/adaptive_chunk.hpp`, `kernels/dataflow/unified_routed_expert_ffn_reader.cpp`, `kernels/dataflow/unified_routed_expert_ffn_writer.cpp` and `kernels/compute/fused_swiglu.cpp`. Only the zone changes in fused_swiglu.cpp.

Equivalence was checked on the host. A fuzz harness (/tmp/rep-r03b04a01-rO2J/fuzz.cpp) compiles the root header and the new one side by side. It compares `LaunchPlan.layout[0..1]` and `owner[2][E]` on 7.2M random cases per grid (10 and 8 rows):
- E = 1..64, plus 255/256/257/300;
- per_core_M_max 1/2/3/4/8;
- zero, small, large, multi-chunk and escape (>= 255) tile counts;
- 0.5M two-phase plans and 3.7M mixed single-layout plans.

The outputs are identical in every case.

Simulated planner instruction counts, old -> new (BH -Os, test routing):

| shape | old | new |
|---|---|---|
| kimi-k3 | 50.1k | 22.2k (L1 loads 1312 -> 56, divu 893 -> 0) |
| dsv3 / glm | 11.6k | 5.0k |
| kimi-k2.7 | 15.8k | 6.9k |
| dsv4-flash | 17.4k | 8.2k |
| dsv4-pro | 15.5k | 6.7k |
| mistral / mm3 | 4.6k | 3.0k |
| gpt-oss | 5.6k | 4.0k |

## Why this is not a repeat
- r02-b01-a02 folded the `start` read into the counts multicast. That was neutral: the prologue was then ~1 µs, before the two-phase planner existed. Its "don't spend attempts on the prologue" was written before plan_launch grew it to 5-29 µs. This node keeps the `start` read and only hides its latency.
- No node has touched the planner's cost. Every planner node changed which plan it picks: r02-b03-a01/a02, r02-b02-a01/a04 and r02-b04-a01/a03. Here the plan is bit-identical and only the time to compute it changes.
- It is orthogonal to the round-3 combinations in flight on the other branches (PACK-thread activation pipeline), so it stacks with them.

## Expected effect and risk
- Expected: the planner's share of first-GU-start drops by about 55-70%. kimi-k3 should save ~15-20 µs (-1..-1.4%), the 8-12-expert shapes ~4-7 µs (-0.5..-2%: dsv4-flash, glm and dsv3 most) and the 4-expert shapes ~1-3 µs (inside their ±5% noise). The `start` overlap adds ~1 µs everywhere. That is about +0.8..+1.2% geomean, so the result could land at the edge of the ±1% noise. The RE-PLAN zones are the direct check.
- Numerics: bit-identical. The same plan feeds the same kernels, so PCC / max_abs / mean_abs must match the root to every digit.
- Risks:
  - TRISC0 stack: +2E bytes (cache + order) and +E in the single-phase model, i.e. ~84 B at E = 28.
  - A JIT compile issue: the header was compiled clean for -mcpu=tt-bh / tt-bh-tensix with -std=c++17 -Wall -Werror at -Os and -O3.
  - A missed equivalence case would show as a hang or a wrong-rows accuracy failure; the fuzz makes that unlikely.
