---
name: perf-stability-debug
description: >
  Find out why an LLK perf number moves when it should not: the same build gives a different
  value on the next run, or a change that does no work (a NOP, a toolchain bump, a harness
  edit) moves values. Starts with cheap checks that tell a real regression from noise
  (same-code control, bit-exact compare, stage 2 at 5 code positions), then walks the methods
  the LLK perf team used on Wormhole and Blackhole: code-shift NOPs, SFPI pin/bump, bisect over
  perf-input commits, test order and worker count, warm-up and prelude kernels, L1 traffic,
  branch predictor, counters, loadgroup, board reset, and Versim/VCS. Use when a perf gate
  flags something the author says cannot be slower, when a same-code control is not
  bit-exact, or when a PR touches the perf harness, linker scripts, profiler or sfpi-version.
  Needs Tenstorrent hardware for most steps.
---

# Perf stability debug (LLK perf tests)

The LLK perf gate compares cycles of the same test between two commits. That is only useful
when one build gives one number. This skill finds out whether a move is real, and if it is
not, which mechanism moves it.

Two problems look the same in a gate report. Name the one you have first:

| Problem | Test | Typical cause |
|---|---|---|
| **P1 re-measure**: same build, other value | measure one build twice | another core or the host touches L1 at a random cycle; test order on a core (state left by the previous kernel) |
| **P2 no-work change**: a change that adds no work moves values | add one NOP before the loop, or bump only the toolchain | code address (branch predictor, instruction cache), start timing, packer rhythm |

A **real** regression is slower at every code position, on every repeat, with the change and
only with the change.

## Steps

1. **Collect the facts.** Arch, test node ids, run type (`L1_TO_L1` is what the gates measure),
   marker (`TILE_LOOP`, `INIT`, `KERNEL`), the two commits, and the gate run id. Read the
   gate's `perf_gate_report.points.csv` and note `TEXT_SIZE(...)` on both sides.
2. **Run the quick checks** in `references/quick-checks.md`, in order. Stop at the first check
   that answers the question. Most cases end here:
   - same-code control on one board, compared bit-exact with `scripts/perf_bitexact.py`;
   - the same control on a second board;
   - stage 2: the candidate at 5 code positions on base and head.
3. **Pick a method** from the table below by what the quick checks showed. Change one thing at
   a time. Keep the two builds that give the two values.
4. **Name the mechanism** with `references/mechanisms-and-fixes.md`. Check first whether a known
   fix already covers it (the warm-ups, the stability stack, loadgroup).
5. **Report**: what moves (count of points, max %), the control that proves it (run ids or
   board + tag), the mechanism or "unknown", and the next step. Never quote numbers measured
   on another branch or stack as the numbers of the PR under review.

| What the quick checks showed | Go to |
|---|---|
| Same build is not bit-exact on one board | `references/methods-order-and-state.md` (order, workers, warm-up, prelude, L1 traffic) |
| Bit-exact on one board, not across boards or CI runs | `references/methods-order-and-state.md` § board quality, then CI vs lab |
| Bit-exact, but a no-work change moves it | `references/methods-code-and-toolchain.md` (NOP shifts, SFPI pin, layout) |
| It started at some commit, cause unknown | `references/methods-code-and-toolchain.md` § bisect |
| Mechanism still unknown after the above | `references/methods-hardware.md` (BP off, counters, reset, Versim/VCS) |
| You want a CI check that would have caught it | `references/ci-quick-checks.md` |

## Reference map

- `references/quick-checks.md`: same-code control, bit-exact compare, second board, stage 2.
- `references/methods-order-and-state.md`: test order, 1 vs 15 workers, loadgroup, warm-up
  on/off, MATH_ISOLATE prelude, L1 traffic from host/BRISC, board quality.
- `references/methods-code-and-toolchain.md`: NOP and code-shift insertion (0/4/20/36/52 B),
  SFPI version pin/bump, disassembly diff, bisect over LLK-perf-input commits.
- `references/methods-hardware.md`: branch predictor off, counters on/off, board reset,
  Versim (Wormhole) and VCS (Blackhole) state diff.
- `references/mechanisms-and-fixes.md`: every known mechanism, what it looks like, and its fix.
- `references/ci-quick-checks.md`: design of cheap CI checks that flag a PR that can make
  stability worse.
- `scripts/perf_bitexact.py`: compare two `perf_data/runs/<tag>` directories point by point.

## Rules that save days

- Export `TT_LLK_DISABLE_ASSERTS=1` for every card or simulator run, as CI does. Without it
  `test_config.py` builds with `-DENABLE_LLK_ASSERT` and the INIT code is not the CI code.
- Mirror the gate: non-SoL (no `--speed-of-light`), `LLK_PERF_RUN_TYPES=L1_TO_L1`,
  `--compile-consumer --dist loadgroup -n 15` for the measuring pass.
- Set `PERF_RUN_TAG` for every pass. Reports go to `tt_metal/tt-llk/perf_data/runs/<tag>/`.
  Only the newest `PERF_KEEP_RUNS` (default 10) run directories are kept: copy a run out
  before you start the 11th.
- One measurement binds the state once per run. `run_count` > 1 and `min()` do not average
  it out: inside one run the repeats are identical.
- A move that is equal on all points of a module and has a different `TEXT_SIZE` is a code
  change. A move with the same `TEXT_SIZE` can still be a code shift (alignment absorbs 4 B).
- Lab boards are not CI runners. CI was 2.1–2.4× slower for the same sweep, and some boards
  are noisy. Repeat a surprising result on a second board before you build on it.
