# Configuration sweep for `regime_a_matmul` (offline autotuner)

One command. Tuning is **offline by design** — it measures on hardware and emits entries for `kTable` in
`regime_a_matmul_config.cpp`, which the shipped picker already consults. The operator gets no file I/O, no
global state, and no first-call measurement.

## Run it

    # tune shapes, print the winning tuple per shape
    REGIME_A_TUNE_SHAPES=512x6144x768,32x6144x6144 \
      scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/matmul/test_regime_a_autotune.py -q -s

    # wider shortlist, more confirmation relaunches, and WRITE the kTable patch (prints the diff)
    REGIME_A_TUNE_SHAPES=512x6144x768 REGIME_A_TUNE_TOPK=16 REGIME_A_TUNE_RELAUNCHES=3 \
      REGIME_A_TUNE_APPLY=1 scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/matmul/test_regime_a_autotune.py -q -s

    # direct invocation (the test is a thin wrapper; prefer this in scripts)
    python3 tools/mm_sweep/picker_gen/autotune.py 512x6144x768 --topk 8 --relaunches 2 [--apply]

| env var | default | meaning |
|---|---|---|
| `REGIME_A_TUNE_SHAPES` | *(none -> test skips)* | `MxKxN[,MxKxN...]` |
| `REGIME_A_TUNE_TOPK` | 8 | shortlist depth **per ranker** (3 rankers, unioned) |
| `REGIME_A_TUNE_RELAUNCHES` | 2 | fresh processes used to confirm a win |
| `REGIME_A_TUNE_MIN_GAIN` | 1.5 | percent; below this the shipped pick is kept |
| `REGIME_A_TUNE_APPLY` | unset | write `kTable` and show the diff |

## What it guarantees

1. **Feasible configurations only.** Candidates come from `autotune_feas.enumerate_full`, an exact mirror of
   the C++ `pick_plan` / `compute_cb_sizes` rules, so nothing that would `TT_FATAL` at program build is
   launched. Keep that mirror in step with the C++ — a stale mirror once rejected configs the picker accepts,
   and a heuristic validated on the restricted set then regressed 4-34%.
2. **Correctness before timing.** Each candidate's FIRST call is untimed and gated on **PCC >= 0.999** against
   a torch reference **and** an explicit **zero-non-finite** check. A candidate failing either is discarded
   before its timing is considered, so a wrong config can never win on speed. Both gates are needed: a handful
   of NaN/Inf among millions of elements barely moves PCC (`BUG_rscatter_nonfinite.md`).
3. **Warm, repeated, and relaunched.** Per candidate: 2 blocks x [2 warmup + 12 timed] iterations on resident
   inputs, device time from the profiler (not host wall). The winner is then re-confirmed against the shipped
   pick across `RELAUNCHES` *fresh processes*, and every relaunch must agree. One reading is not enough on this
   hardware — that gate rejected 6 of 32 apparent wins in the original campaign, and see the worked example
   below.
4. **The shipped configuration is always a candidate.** `config=None` (the production picker) is measured
   alongside the shortlist, and a winner is reported only if it beats it by more than `MIN_GAIN`. The tool
   therefore cannot propose something slower than what ships.
5. **Winning tuple, and optionally a patch.** Prints the tuple per shape; `--apply` writes `kTable` via
   `apply_table.py` (verifying brace shape, updating in place or appending) and shows the diff to review.

## Why measure at all

The analytic picker is a good *ranker* and a poor *chooser*. Held out over ~17,000 timed configs on 27 shapes:
picking 1 config (what ships) is ~7.8% mean regret vs optimal; measuring its top 4 is ~3.1%, top 8 ~1.6%,
top 16 ~0.6%. Five attempts to improve the chooser formula all failed to generalise, so the leverage is in
measuring a handful. See `picker_gen/TIER2_COST_MODEL_ANALYSIS.md`.

It also fixes table staleness: 14 of the original 44 `kTable` rows were measured winners when added and were
invalidated by later kernel work. Re-running this after a kernel change re-measures instead of letting rows rot.

## Worked example (2026-08-11, bh-glx-120-c02u02)

    [32x2048x2048] 8 shortlisted; shipped pick 21.78 us already best (shortlist best 4,2,1,2,4 @ 22.13 us)
    [512x6144x768] shortlist 11 measured, best 6,1,2,2,3 @ 51.78 us -- keep shipped pick (-0.3%/+0.1%)
    # nothing to apply: the shipped pick was within 1.5% on every shape

Both outcomes are successes. The second is the relaunch gate earning its keep: one reading looked like a win,
two fresh relaunches disagreed in sign (-0.3%, +0.1%), so it was rejected as noise. `6,1,2,2,3` is in fact
already the `kTable` entry for that shape — the tool independently re-derived the shipped pick.

## Runtime

Minutes per shape. Budget roughly `shapes x (2*topk + 2*relaunches + 1)` process launches; each pays a device
open plus a JIT compile for a config never built before. The two-shape example above took 13m12s at
`topk=4, relaunches=2`.

The repo-wide 300s `pytest-timeout` is disabled for this test (`@pytest.mark.timeout(0)`) because of that.
Hang protection is not lost, it moves to where it belongs: `run_safe_pytest.sh` sets
`TT_METAL_OPERATION_TIMEOUT_SECONDS`, which fires per dispatch (~ms for these matmuls) rather than on total
wall time, and resets the device if it trips.

> On Galaxy, note `run_safe_pytest.sh` resets with `tt-smi -r`, which on bh-glx-120-c02u02 left ethernet links
> down and downgraded the mesh to 8x2 (16 of 32 chips); recovery needed `tt-smi -glx_reset`. See
> `GALAXY_TESTING.md`.
