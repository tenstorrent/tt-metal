<!-- STATUS FOR THE NEXT SESSION: F56 below is DONE and proven; do not rebuild or re-prove it. -->
**F56 status (2026-09-30): built, proven, committed on branch `dnijemcevic/f56-component-checks`.** To bring it into
another branch, cherry-pick the `[F56]` commits (`git log --oneline --grep "\[F56\]"`); they touch only
`models/demos/common/bringup/` (testing/component_checks.py new; component.py, harness.py, templates.py, core/runs.py,
core/metrics.py, orchestrator.py, README.md, selftest/test_component_checks.py, dev/f56_*, dev/BREADCRUMBS.md, this
file). The proofs are recorded (dev/f56_mutation_proof.md, BREADCRUMBS F56 device table); there is no need to rerun
them after a cherry-pick. Run the selftests after it (`scripts/run_safe_pytest.sh --no-precompile --run-all
models/demos/common/bringup/selftest/`).

What is left: the owner's OK to switch the component review off by default (`orchestrator.py:
COMPONENT_REVIEW_DEFAULT = "all"` -> `"none"`, or per model `agents.component_review: none` in the spec). Until then
new component tests already carry the built-in checks and the review agent starts from them.

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

**F56: component tests check themselves (built; review skip off until the owner says OK).**
- Every new component test compares the device step with the CPU step on the same inputs and with the golden, with
  checks chosen by the kind of output (float numbers, router weights, top-k positions). The error limit follows
  the step's expected precision (the CPU step with bf16 intermediates), so it is tight for exact steps and loose
  for bfp8 experts. Each test also runs a few second inputs (chunk 0, another layer, shuffled, tiny and x2 inputs)
  for the bugs the golden cannot show: a wrong norm epsilon, iHC stream order at layer 0, a clamp.
- At freeze a CPU sweep injects every standard mistake and requires the test to catch each; if one slips through,
  the review agent starts with that log. Switch: `agents.component_review` (default `all`: review on).
- Proof: `dev/f56_mutation_proof.md` (CPU, 11 reviewed Hy4 tests) and BREADCRUMBS F56 (device, 6 Hy4 components,
  no false alarms).

## Remains (not started)

1. **Done as F56 (see above); was: component tests without a review agent** (about 4 h of review per bring-up; estimate 2-3 h to build and
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

## Recipe: F56, component tests without a review agent

Follow what F49 did for swap tests. Its commits (search `git log --grep "\[F49\]"`) and `dev/BREADCRUMBS.md`
section F49 are the worked example.

**Build (CPU only until step 6)**
1. `testing/component.py`: add `run_component_test(..., checks=None | "auto")`. `None` keeps today's behaviour, so
   existing frozen tests do not change. `"auto"` picks checks by the output's kind:
   - float tensor: PCC (as now) plus relative error, worst-row error, row-size ratio, finite values (reuse the helpers
     F49 added for swap steps);
   - integer top-k indices: per-row set overlap (the `topk_overlap` mode in `testing/harness.py`), no future
     positions, no repeats;
   - expert routing (indices + weights): expert overlap and weights summing to 1 per token.
   Take the limits from the spec (`thresholds.*`, add new ones with defaults like F49's `swap_step_*`).
2. **Mistake tests at freeze** (`core/runs.py: freeze_task`, which today runs the test with `BRINGUP_IMPL=reference`,
   must PASS, and `stub`, must FAIL): also run it with each `BRINGUP_IMPL=mutate:<kind>` from `testing/mutate.py`
   (CPU, no device) and require FAIL for every kind that applies to the output. Record which kinds were caught.
3. **"Can the test see it?"** (new, `testing/sensitivity.py`): on the CPU, scale each input and each weight of the
   step by 1.02 one at a time and measure the change of the tested output. If a change stays below the test's
   tolerance, the test is blind to that part: add a comparison of the intermediate that depends on it (the reference
   records every step output as `L{i}.<name>`) or a second input (another layer, or random data at a different
   scale). Report the blind parts in the freeze log.
4. `testing/templates.py`: component template gets `CHECKS = "auto"`.
5. `orchestrator.py`: component tasks skip the test-role agent unless `agents.component_review` (new, like
   `agents.swap_review`) names the block type, or step 3 found a blind spot it could not close, or the output kind is
   unknown.

**Prove before switching it on**
6. Copy `dev/f49_mutation_proof.py` to `dev/f56_mutation_proof.py` and point it at component tests: for a sample of
   Hy4's 42 reviewed component tests (at least one per output kind: attn_hc, attention, indexer, router, experts,
   a residual), run each injected mistake and compare: caught by the reviewed test / by the old plain test / by
   `checks="auto"`. Requirement: `auto` catches everything the reviewed test catches, and the unmutated reference
   passes. Hy4 goldens are in `/localdev/dnijemcevic/bringup/hy4_preview_d_p/golden` (CPU reference, run with
   `mesh=None`). Include the known hard cases from `knowledge/known_issues.md`: a wrong norm epsilon (invisible at large
   row RMS), iHC pre gates at layer 0, top-k compared by position.
7. Device check: a small pytest (like `generated/f49_device/test_f49_device.py`) that runs `checks="auto"` on the
   device for one component per output kind; it must pass (no false alarms from device rounding).
8. Selftests in `selftest/` (fixture model): `checks=None` unchanged, `auto` passes in reference mode, fails in stub
   mode and on each mistake kind, the review skip and `agents.component_review` work. Run
   `scripts/run_safe_pytest.sh --no-precompile --run-all models/demos/common/bringup/selftest/`.
9. Write BREADCRUMBS F56 with the proof table, and ask the owner before making the skip the default.

Expected saving: most of the ~4 h of component-test review per bring-up; the agent stays for new output kinds and
unresolved blind spots.
