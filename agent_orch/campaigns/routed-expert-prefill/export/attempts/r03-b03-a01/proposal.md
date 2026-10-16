# r03-b03-a01: cheap launch planner: every RISC caches the experts' tile counts once and plans on that array with division-free costs and one uniform pass, so the first DRAM read / matmul starts ~5-25 µs earlier

## Motivation
Before any weight read or matmul, every one of the ~550 RISCs (reader, writer and the UNPACK thread of
110 cores) runs the same launch planner (`adaptive_chunk::plan_launch`: `make_plan` + `plan_groups`).
Its cost is on every core's critical path and has grown with each schedule node:

- First-GU-start (kernel start -> first RE-GU zone on MATH) from r02-b03-a04's zones (same planner as
  the root, apart from the mixed phase-B scoring, which adds a bit; /tmp/r02b01a02/zones.py on its
  profile_log_device.csv):
  - kimi-k3 **29.0 µs**
  - dsv3 / kimi-k2.7 / glm / dsv4-flash / dsv4-pro 9.8-10.9 µs
  - mistral / minimax-m3 / gpt-oss 4.6-5.4 µs
- With the old single-layout picker (r02-b01-a02) the same quantity was 6.4 / 3.2-3.9 / 2.5 µs. So the
  planner now costs ~23 µs on kimi-k3, ~7 µs on the 8-12-expert shapes and ~2.5 µs on the 4-expert
  shapes, with no work overlapping it. On the DRAM-bound shapes, DRAM sits idle for that whole time.
- Why it is slow (code reading plus the JIT ELF disassembly in the campaign jitcache):
  - `tiles_of(e)` does two dependent volatile L1 loads (`counts_ptr[idx_ptr[e]]`), plus the band filter
    and clamp. On UNPACK it also re-reads both CB interfaces. It runs inside every pass of every
    schedule. That is several hundred calls on kimi-k3 (3 uniform passes in `pick_group_rows`, the
    thresh loop, the leftover walk, 2 x 2 `model_phase` passes, then 3 uniform + 3 mixed passes and
    the owner pass in `plan_groups`).
  - `model_phase` -> `PhaseDealer::deal` / `model_expert` -> `expert_cost` / `num_chunks` /
    `per_core_M_for_chunk` run about 6-13 hardware `divu`/`remu` per expert per pass (the ELF has 43
    of them), and `pos++ % groups` adds another per phase-A deal. On kimi-k3 that is ~750 divides,
    i.e. tens of thousands of cycles at ~32+ cycles each.
  - The 3 uniform layouts are scheduled twice, once in `make_plan` (`pick_group_rows`) and again in
    `plan_groups`.

## Mechanism
Only `device/kernels/adaptive_chunk.hpp`, plus small call-site and zone edits in the three kernels. No
host change. The plan stays exactly the same; only its cost changes.

1. **Cache the counts.** `plan_launch` reads `tiles_of(e)` once per expert into a local array and runs
   every planner pass on that array (a lambda over it). The array element type is
   `adaptive_chunk::TileCount<M_tiles_full>`: uint16 when M_tiles_full fits, else uint32. That is
   exact, with no saturation. The kernels pass M_tiles_full as a template argument.
2. **No divides in the planner's hot loops.** `PhaseDealer::deal` and `model_expert` use the existing
   division-free `group_cost` (== `expert_cost` for every input; the multi-chunk case falls back to
   it). `model_expert`'s chunk count is 0 / 1 without a divide in the single-chunk case. The phase-A
   round-robin uses a wrapping counter instead of `%`. The kernel-facing helpers (`num_chunks`,
   `expert_cost`, `per_core_M_for_chunk`) are untouched.
3. **One uniform pass, not two.** `make_plan` records the best uniform layout and its makespan
   (`pick_group_rows`' result). `plan_groups` reuses them instead of rescheduling the 3 uniform
   layouts. It keeps the same strict-< tie rules, so it makes the same pick.
4. **Measurement (inert):** one `DeviceZoneScopedN("RE-PLAN")` around `plan_launch` on the reader
   (NCRISC) and on UNPACK, so the reflection can read the residual planner cost directly.

Exactness check before the eval: a host build of the root's header and the new header, side by side,
fuzzed on random tile vectors (E 1-32, tiles 0-200, per_core_M_max 1-8, both grid heights) plus the
test's seed-0 routing. It must produce identical `LaunchPlan` layouts and identical owner maps.

## Why this is not a repeat
- r02-b01-a02 folded the `start` page into the counts multicast (prologue ~1 µs, neutral). It did not
  touch the planner, which came later (r02-b03-a01/a02, r02-b02-a04).
- Five reflections list "cut the planner prologue" as an open next step (r02-b03-a02 #1, r02-b01-a04
  #1, r02-b03-a04 #3, r02-b02-a04 #3, r02-b04-a01 #5). No node has tried it.
- It is orthogonal to the PACK-thread activation pipeline that other round-3 branches are porting, so
  the two should stack.

## Expected effect and risk
- First-GU-start should drop to ~3-6 µs on every shape. Expected savings: kimi-k3 ~15-20 µs (-1.2%),
  dsv3 / glm / dsv4-flash ~5-7 µs (-1.2..-2%), kimi-k2.7 / dsv4-pro ~5-7 µs (-0.6..-0.9%), the 4-expert
  shapes ~2 µs (-0.5..-1%). Geomean about +1%, comparable to the noise band per shape. So the
  RE-PLAN zones and first-GU-start, not the µs table, are the evidence.
- Numerics are bit-identical by construction: the same plan, the same math.
- Risks:
  - A planner mismatch between RISCs (all run the same code on the same counts) would show as a hang.
    A plan different from the root's would show as changed per-row end times. The host fuzz guards
    both.
  - TRISC stack: the cache adds 2 B per local expert (56 B on kimi-k3).
  - Zones: two per RISC per call, well within the profiler budget.
