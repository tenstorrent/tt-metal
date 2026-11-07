# Design: cheap CI checks for PRs that can make perf stability worse

Status: proposal, 2026-10-30. No workflow on main is changed by this document.

## Problem

The LLK perf gate compares two commits. It assumes that one build gives one number. Several
merged PRs broke that assumption without touching a kernel: #53642 (one `pytest.ini` line),
#62327 (SFPI 7.90, which exposed a WH order effect), harness and profiler changes. Each one
cost days of bisecting after the fact. The checks below flag such a PR before merge, for a
few runner minutes. They do not judge regressions; stage 1 and stage 2 do that.

## Inputs used for the costs

| Fact | Value | Source |
|---|---|---|
| Full L1_TO_L1 suite, non-SoL, gate SKU, one card | WH compile 7:52 + measure 2:33; BH compile 9:21 + measure 4:04 | run 34369520690 |
| Tests in the full L1_TO_L1 suite | WH 36,122; BH 38,984 | 2026-10 replay controls |
| Merge-gate leg cap | 15 min per leg, 3 legs per arch; the cap cannot be raised | `llk_perf_merge_gate_tests.yaml`; infra ruling 2026-09-22 |
| BH perf legs, successful runs 22–28 Oct | median 7.5, p90 9.2, max 9.8 min | merge-queue job scan |
| WH merge legs (one trial) | 6.0 / 7.1 / 7.0 min | run 40699802852 |
| Fixed cost of one pytest session in a leg | 8–10 s | run 40937155701 |
| Container init in a leg | ~50 s warm, up to ~3 min cold | run 40937155701; #62981 review |
| WH warm-up cost (matmul measure) | about +18 % | lab boards, 2026-10-30 |

**Sentinel set.** Several checks use a small fixed node list per arch: ~150
`perf_math_matmul` configs known to move (LoFi first, from the 2026-10 controls), plus 2
configs from every other perf module. About 200 nodes, kept in
`tests/perf_stability_sentinels.json` (proposed file). At the per-test rates above, one
measuring pass of 200 nodes is about 2–5 s of measure plus one session (8–10 s); compiling
them is under a minute. Estimates below round up.

## The checks

### C1. Path filter for stability-sensitive inputs (trigger only)

What: a script like `.github/scripts/llk-perf-inputs-changed.sh` with a narrower list. It
prints `true` when a PR touches what can change stability without touching a kernel:

```
tt_metal/sfpi-version  tt_metal/sfpi-info.sh
tt_metal/tt-llk/tests/python_tests/pytest.ini
tt_metal/tt-llk/tests/python_tests/helpers/llk_pytest_plugin.py
tt_metal/tt-llk/tests/python_tests/helpers/test_config.py
tt_metal/tt-llk/tests/python_tests/helpers/device.py
tt_metal/tt-llk/tests/python_tests/helpers/perf/**
tt_metal/tt-llk/tests/helpers/include/{counters.h,profiler.h,barrier.h,boot.h,perf.h}
tt_metal/tt-llk/tests/helpers/src/{brisc.cpp,trisc.cpp}
tt_metal/tt-llk/tests/helpers/ld/**
tt_metal/tt-llk/tests/run_llk_perf_*.sh  tt_metal/tt-llk/tests/perf_split_costs*.json
tt_metal/tt-llk/tests/requirements.txt
```

- Cost: < 10 s, in the existing `llk-perf-merge-gate-changes` job (merge gate) and the
  `branch-point` job (PR gate). No board.
- Catches: #53642 (`pytest.ini`), every SFPI bump, harness, profiler, firmware and linker-script
  changes.
- False alarms: none as a blocker, because it only triggers C3–C5 and adds a line to the report.
- Plugs into: `merge-gate.yaml` job `llk-perf-merge-gate-changes`, `llk-perf-gate.yaml` job
  `branch-point`; output `stability-inputs-changed`.

### C2. Fix code paths stay present (static, no board)

What: host-only tests in `tests/hardware_free/perf/` that fail when a known stability fix is
removed or weakened:
- the measuring pass of `run_llk_perf_{wormhole,blackhole}.sh` keeps `--dist loadgroup`, and
  `pytest.ini` keeps `--maxschedchunk` at 200 or less (until #63027 removes loadgroup on BH);
- runner scripts and the merge-gate test yaml export `TT_LLK_DISABLE_ASSERTS=1`;
- after #58068 lands: a `PerfConfig` built with `LLK_PERF_RUN_TYPES=L1_TO_L1` has an isolate
  kernel in its warm-up list, and every measured run config is warmed up;
- after #61357/#61358 land: the BRISC poll and the zone-record order (assert on the source
  text or a compiled symbol, whichever the fix makes testable).

- Cost: seconds, inside the existing `llk-hardware-free-impl.yaml` CPU job.
- Catches: a revert or refactor that drops a fix; the `maxschedchunk` 10 → 2000 class of change.
- False alarms: low. An intended change must update the test, which is the point: it forces a
  same-code control (C3) in the same PR.
- Plugs into: `tt_metal/tt-llk/tests/run_hardware_free.sh` (already runs `tests/hardware_free`).

### C3. Same-code double measure of the sentinel set (bit-exact)

What: in one leg per arch, after "Run tests": compile the sentinel set once, measure it twice
(`PERF_RUN_TAG=sentinel-a/-b`), compare with `perf_bitexact.py`. Report the count of points not
bit-exact and the largest move; fail the check only above a per-arch noise floor.

- Cost: about 1–2 runner-minutes per arch inside an existing leg (compile < 1 min, two
  sessions ~20 s, compare a few seconds). As a separate job: add 3–5 min for container init,
  checkout and image pull. Run it only when C1 fires (or nightly on main), so most PRs pay 0.
- Catches: P1 problems on the PR's own code: order effects, host/BRISC L1 traffic, harness
  timing. On the 2026-10 data: WH without warm-ups 696–853 TILE_LOOP points per matmul run;
  with them 0. BH full suite: 21 to 82 points not bit-exact per pair, all < 0.005 % (runs 40725553397,
  40861271282).
- False alarms: WH fires on main today (no warm-ups), so the WH check needs either #58068 or a
  floor taken from main's own sentinel runs. A noisy runner can fire; a fire is a report line,
  not a block, until the floor is known. Threshold proposal: BH 0 points over the gate rule;
  WH 0 after #58068.
- Plugs into: `llk-perf-impl.yaml` job `llk-perf-exec` as a step after "Run tests", gated on an
  input `stability-check: true` set by `merge-gate.yaml` / `llk-perf-gate.yaml` from C1. The
  leg has ~5 min headroom at the BH median and ~6 min at p90 (15 − 9.2); stage 2 also uses that
  time, so give C3 its own deadline and skip it when < 2 min remain.

### C4. Order perturbation of the sentinel set

What: measure the sentinel set in two different fixed orders and compare. Tools on main:
`--test-order-file` + `--rewind-runner` replay a given sequence (one worker); a crc32-salted
`xdist_group` per node (as `perf_bisect.py --stable-groups` does) gives a balanced fixed order
on 15 workers.

- Cost: one worker: ~200 nodes serially, about 1–2 min per order, so 3–5 runner-min per arch.
  15 workers with fixed groups: like C3 (1–2 min). Run only when C1 fires.
- Catches: a PR that makes the measured value depend on the previous kernel (mechanism #2 of
  `mechanisms-and-fixes.md`), which C3 can miss when the scheduler happens to repeat its order.
- False alarms: on WH main it fires (known order effect) until the warm-ups land. BH on the
  stack is order-independent, so a BH fire is a real finding.
- Plugs into: same step as C3, as a third pass. Needs a small plugin change (salted groups) or
  a generated order file; prototype on a branch first.
- Pitfall: do not use `--record-test-order` to produce the order: it reads the Tensix state
  after every test, which changes the WH carried state.

### C5. Code-size and placement diff of the sentinel kernels (no board)

What: compile the sentinel set at base and head on a CPU runner. For each ELF and thread,
compare `TEXT_SIZE`, the loop fingerprint (stage-2 `window_hash.py`, `LLK_WINDOW_HASH=1`), and
the TILE_LOOP start address modulo 16 and 64. Label each sentinel: "loop code changed",
"same loop, moved", "unchanged".

- Cost: two compile passes of ~200 nodes on a CPU runner, about 2–4 CPU-runner minutes per
  arch; no board time.
- Catches: SFPI bumps and harness changes that move or rewrite the measured loops (7.92
  changed 16 of 20 sampled math ELFs). It predicts which stage-2 rows will be "placement" and
  tells the author before any board runs.
- False alarms: high if used as a blocker: most code changes move addresses. Use it as a
  report section and as the input that turns on C3/C4 for kernel-only PRs.
- Plugs into: the existing CPU shadow-compile job of `llk-perf-impl.yaml`
  (`shadow-compile-runner`), or the `placement-plan` job that #62981 adds to
  `llk-perf-gate.yaml`. Needs #62981 (window hash, code shift).
- Pitfall: `TEXT_SIZE` alone misses a 4-byte shift (alignment absorbs it). Compare addresses.

## Rejected or deferred

- **Nightly same-code control of the full suite**: catches everything, but costs a full
  extra measure per arch per night (WH ~10 min, BH ~13 min on the gate SKU, plus compile).
  Keep as a weekly job if C3 proves too narrow.
- **Measure the MATH_ISOLATE prelude in the gate**: +39 % on WH legs, +72 % on BH with no gain.
  The warm-ups are cheaper.
- **Raise thresholds**: a threshold that admits a 28 % bistable swing is not a gate.

## Rollout

1. C1 and C2 first (seconds, no board). Report-only.
2. C3 on Blackhole when C1 fires; record the floor for 2 weeks; then fail on > 0 over the rule.
3. C5 with #62981's tooling, as a report section.
4. C3 and C4 on Wormhole after #58068 (warm-ups) is on main.

Acceptance for each step: one dispatch with a deliberately bad change (for example
`--maxschedchunk=2000` without loadgroup, or the warm-up commit reverted on WH) fires the
check, and one dispatch of an unchanged commit does not.
