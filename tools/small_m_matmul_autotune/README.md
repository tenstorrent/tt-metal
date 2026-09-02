# small_m_matmul offline autotuner

Offline configuration sweep for `ttnn.experimental.small_m_matmul`. It measures candidate configurations on
hardware and emits entries for `kTable` in
`ttnn/cpp/ttnn/operations/experimental/small_m_matmul/device/small_m_matmul_config.cpp`, which the shipped
picker consults. Tuning is offline by design: the operator itself does no file I/O, keeps no global state and
never measures on first call.

Requires a profiler-enabled build (the `build_metal.sh` default), a Blackhole device that nothing else holds,
and must run from the repo root with `TT_METAL_HOME` set.

```bash
# tune shapes, print the winning tuple per shape
python3 tools/small_m_matmul_autotune/autotune.py 512x6144x768 32x6144x6144

# wider shortlist, more confirmation relaunches, and write the kTable patch (prints the diff to review)
python3 tools/small_m_matmul_autotune/autotune.py 512x6144x768 --topk 16 --relaunches 3 --apply
```

| option | default | meaning |
| --- | --- | --- |
| `--topk` | 8 | shortlist depth per ranker (3 rankers, unioned) |
| `--relaunches` | 2 | fresh processes used to confirm a win |
| `--min-gain` | 1.5 | percent; below this the shipped pick is kept |
| `--apply` | off | write `kTable` via `apply_table.py` and show the diff |

## What it guarantees

1. **Feasible configurations only.** Candidates come from `autotune_feas.enumerate_full`, a Python mirror of the
   C++ `pick_plan` / `compute_cb_sizes` rules, so nothing that would `TT_FATAL` at program build is launched.
   The mirror hard-codes the Blackhole p150 constants (L1 budget, core limits); keep it in step with the C++.
2. **Correctness before timing.** Each candidate's first call is untimed and gated on PCC >= 0.999 against a
   torch reference and a zero-non-finite check; a wrong config can never win on speed.
3. **Warm, repeated, relaunched.** Device time comes from the profiler, not host wall. A winner is re-confirmed
   against the shipped pick across `--relaunches` fresh processes, and every relaunch must agree.
4. **The shipped configuration is always a candidate.** `config=None` is measured alongside the shortlist and a
   winner is reported only if it beats it by more than `--min-gain` percent.

## Files

- `autotune.py` — driver: shortlist by cost models, measure, confirm, optionally apply.
- `autotune_feas.py` — feasibility mirror of the C++ planner rules.
- `prod_sweep_worker.py` — one-shape measurement worker (opens a device, runs the op, parses the profiler CSV).
  Also used by `tests/ttnn/perf_tests/operations/matmul/test_small_m_matmul_perf.py`.
- `apply_table.py` — rewrites `kTable` entries in `small_m_matmul_config.cpp` in place.
