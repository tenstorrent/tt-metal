# r03-b01-a03: make the reader's single launch planner fast: port r03-b04-a01's bit-identical planner trim onto the plan-once-per-core lineage, overlap both `start` reads with planning, and build the planner region at -O3 on the dataflow RISCs

## Motivation
- In the parent (r03-b01-a02) only the reader (NCRISC) runs `plan_launch`. The writer and compute
  (UNPACK) wait on `cb_plan_scratch`, and every core's first weight read and first matmul wait for it.
  The planner is therefore pure critical-path latency, once per call, with DRAM idle.
- The parent still runs the UNTRIMMED r02-b02-a04 planner. Its reflection measured first-GU-start
  45.8 µs on kimi-k3 (17.2 in r03-b02-a01 with a trimmed planner), 16.4 on dsv4-flash, 11-15 on
  dsv3 / kimi-k2.7, 7-8 on the 4-expert shapes. Its next #1: stack a planner trim; only one place
  to change now.
- r03-b04-a01 measured that trim on device: kimi-k3 median core -22 µs, dsv3 / dsv4-flash /
  dsv4-pro -5..-9 µs, RE-PLAN on NCRISC 17.7 µs on kimi-k3, bit-identical numerics.
- New evidence (this node): dataflow kernels build at -O2 (`program.cpp`: DM opt_level defaults to
  O2), compute at -O3. An RV32IM simulation of the reader's planner (r03-b04-a01's rvsim on the
  test routing, BH g++ 15.1) gives, in µs at 1.35 GHz:

  | shape | parent planner (-O2) | trimmed, -O2 | trimmed, planner region -O3 |
  |---|---|---|---|
  | kimi-k3 | 48.8 | 18.0 | 14.4 |
  | dsv4-flash | 15.9 | 6.6 | 5.3 |
  | kimi-k2.7 / dsv4-pro | 15.4 / 15.1 | 5.1 / 4.9 | 4.2 / 4.0 |
  | dsv3 / glm | 10.7 | 3.7 | 3.0 |
  | mistral / mm3 / gpt-oss | 3.8 / 3.8 / 5.5 | 2.6 / 2.6 / 4.6 | 2.3 / 2.3 / 4.1 |

  The parent column matches the parent's measured first-GU-start (kimi-k3 45.8), so the model is
  calibrated (r03-b04-a01's 17.7 µs measured vs 18.0 simulated at -O2 likewise).

## Mechanism
1. `kernels/adaptive_chunk.hpp`: replace the planner with r03-b04-a01's version (PlanTiles byte cache
   + largest-first order, divide-free single-chunk costs, uniform layouts scheduled once, one-pass
   phase-A model, early make_plan return, `lpt_may_fit` pruning). Keep this lineage's
   `plan_page_bytes` / `store_plan` / `load_plan`.
2. Same header: on dataflow builds (`COMPILE_FOR_NCRISC || COMPILE_FOR_BRISC`) wrap the planner region
   in `#pragma GCC push_options / optimize("O3")`, and make `plan_launch` `noinline` so the region's
   flags apply to its whole body (inlined into the -O2 kernel_main they would be lost; checked in
   the simulator). Compute no longer instantiates the planner, so TRISC binaries are unchanged.
3. Reader: the `start` page read is issued before planning, and its barrier + push move after the
   plan is published to `cb_plan_scratch`, so the plan reaches the writer/compute first.
   Plus an NCRISC `RE-PLAN` zone around `plan_launch` (profiler-only, measured to work by r03-b04-a01).
4. Writer: its `start` read barrier moves after `load_plan`, so the read's latency runs under the wait
   for the reader's plan.

Files: `kernels/adaptive_chunk.hpp`, `kernels/dataflow/unified_routed_expert_ffn_reader.cpp`,
`kernels/dataflow/unified_routed_expert_ffn_writer.cpp`. Kernel-only, no host rebuild.

Equivalence: host fuzz of the parent's `plan_launch` against the new one (r03-b04-a01's fuzz.cpp):
1.08M cases per grid height (10 and 8 rows), E = 1..64 + 255-300, per_core_M_max 1-8, including
two-phase and mixed plans: 0 mismatches.

## Why this is not a repeat
- r03-b04-a01 and r03-b02-a01 trimmed planners on lineages where three RISCs planned in parallel.
  Here only the reader plans (r03-b01-a02's plan sharing), so the trim lands in one place, the writer
  and UNPACK copies are already gone, and the binary has ~12 KB of headroom to spend on -O3.
- The -O3 planner region is new. r03-b02-a02 went the other way (-O2 on TRISC to save size), which
  cost it 0.3-4.7 µs of first-GU-start.
- This is the "stack a planner trim" next step from r03-b01-a02's reflection.

## Expected effect and risk
- Planner latency drops by ~34 µs on kimi-k3, ~10-11 µs on dsv4-flash / kimi-k2.7 / dsv4-pro, ~8 on
  dsv3 / glm, ~1.5 on the 4-expert shapes. r03-b01-a02 noted part of the parent's plan time was
  absorbed by a shorter first GU block (compute waits for the first weight multicast anyway), so I
  expect 50-80% of it on the op time: kimi-k3 -1.5..-2.5%, dsv4-flash -1.5..-3%, dsv3 / glm / k2.7 /
  dsv4-pro -0.5..-1.5%. About +0.8..+1.5% geomean (~1.60-1.61).
- Numerics bit-identical (same plan); PCC / max_abs / mean_abs must match to every digit.
- Risks: JIT error from the optimize pragma / attribute mix (cheap to fix before the device run);
  kernel-config-buffer size: the reader grows ~3-5 KB (simulated per-instantiation planner code 6.2 ->
  9.1-11.1 KB) against ~12-14 KB headroom. The RE-PLAN zone gives the direct check.
