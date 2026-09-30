# Faster bring-up testing: what was done, what remains

Written 2026-09-30, after the Hy4 Preview run (layers 0-5, 2x2). Time study with the numbers:
`hy4_bringup_time_study.html` (repo root).

## Where the time went (Hy4 run1)

- About 15 h of agent time in a 20.5 h run.
- Checking tests before they are frozen took most of it: swap-test reviews 7.8 h (about half of all agent time),
  component-test reviews 4.0 h. Writing the device code took only 1.5 h.
- Most agent time is device test runs (each one opens the mesh and loads the weights again), not thinking.

## Done

**F49: swap tests check themselves, no review agent (default).**
- A swap test runs one layer with steps 1..n on the device and the rest on the CPU. Before, it only checked the
  layer's final output, and a review agent then spent about 11 minutes per test adding checks by hand.
- Now every new swap test checks each device step itself: against the golden and against the CPU version of the step
  on the same inputs (error, worst row, row size, finite values, not secretly on the CPU). Code:
  `testing/component.py` (`run_swap_test(..., checks="steps")`), template in `testing/templates.py`.
- Swap tasks now freeze without a review agent. To bring the agent back, set in the model's spec:
  `agents.swap_review: all` (or a list of block types). The built-in checks stay either way.
- Proof: mistakes were injected into the last device step of the 4 reviewed Hy4 swap tests (7 kinds: scale ±2%,
  swapped halves, shifted rows, one chip's share zeroed, 1% noise, flipped signs). The old plain test caught about
  half, the reviewed tests all of them, the new built-in checks all of them (`dev/f49_mutation_proof.md`). On the
  device the new checks pass on the full layer 0 and layer 1 swap tests (no false alarms).
- Mistake injection for proving tests: `BRINGUP_IMPL=mutate:<kind>` with `BRINGUP_MUTATE_STEP=<step>`
  (`testing/mutate.py`).
- Expected saving: 7-8 h per bring-up of this size.

**F55: an agent is never blamed for files its own gate writes.** O.1 was retried for writing
`results/fork_calls.json`, which its gate writes anyway. One list (`core/gate.py: gate_outputs`) now decides what the
gate writes, deletes first and commits, and what the agent may change.

**Also fixed today:** F48 (gate commits include new fork files and knowledge notes); X.3 gate gives the profiler room
for 4000 programs (`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000`), since a 6-layer chunk runs more than 1000.

## Remains (not started)

1. **F56: component tests without a review agent** (about 4 h of review per bring-up; estimate 2-3 h to build and
   prove). Same idea as F49:
   - choose the built-in checks by output kind: float tensor (error, worst row, row size), top-k indices (set overlap,
     no future tokens), expert routing (expert overlap, weights sum to 1);
   - at freeze time, inject the standard mistakes on the CPU and require the test to fail on each;
   - a "can the test see it?" check: change each input and weight of the step slightly on the CPU; if the tested
     output barely moves, the test is blind to that part and must compare an intermediate or use a second input;
   - keep the review agent only for new output kinds or unresolved blind spots; prove it the F49 way on this run's
     reviewed component tests before switching.
2. **"Blocked" hand-off.** When an agent finds the fix is outside its allowed files (Hy4 X.3: the profiler setting),
   it should say so in a fixed way and the orchestrator should hand the task to the overseer instead of retrying.
   One wasted 15-20 min retry per such case today.
3. **Keep the mesh and weights loaded between test runs.** Every device test opens the mesh and loads weights again;
   a long-lived test worker would cut minutes off each of the ~400 device runs per bring-up. Largest change, needs
   care with hangs and resets.
4. **Do the test agents' mistake checks on the CPU,** not with 4-6 device runs per review.
5. Small items: dashboards now exceed the repo's 500 KB large-file hook (Hy4's are published but not committed);
   the rms_norm fork's C++ path does not refuse `memory_config` under `inplace` (test skipped, see its CHANGELOG).
