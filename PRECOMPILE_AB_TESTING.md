# A/B testing: cold inline run vs `--precompile` warm run

How to measure the speedup `--precompile` actually buys you on a given suite + device — the **old
inline cold run** (kernels compile inline & serial during the run) vs the **precompile + warm run**
(kernels compile up-front, hardware-free, in parallel; the real run then reuses them).

This is the **performance** counterpart to `scripts/precompile_verify.sh`, which proves the two
produce *identical results* (correctness / R1). Confirm correctness there first; this doc is only
about wall-clock time. Mechanism + diagnostics live in `PRECOMPILE.md`.

## The metric

Both arms end with this line (added to `run_safe_pytest.sh`; it's on **stderr**, printed last):

```
SAFE_PYTEST_TOTAL_RUNTIME: 4m12s (252s total, device-lock-acquired -> exit)
```

It measures **device-lock-acquired → exit**, so for the `--precompile` arm it deliberately
**includes the warm phase and its own overhead** (cluster-descriptor capture + the real build_key
probe + the parallel warm pass). That is the honest end-to-end number — exactly what an A/B
comparison should compare. Speedup = `A_total / B_total`.

> Idle time spent queueing for the device lock is excluded (the clock starts once the lock is held),
> so a busy lock doesn't pollute the measurement — but see the load caveat below.

## The one thing that makes or breaks the test: cache state

The on-disk JIT cache persists across runs at `$TT_METAL_CACHE`, or `~/.cache/tt-metal-cache/` if
unset (`tt_metal/jit_build/build.cpp:90`). **If you don't reset it, your "cold" baseline is actually
warm from a previous run and the comparison is meaningless.** Both arms must start from the same
empty cache.

Isolate the cache into a throwaway dir so you can reset it cleanly without disturbing the shared one.
`run_safe_pytest.sh` never overrides `TT_METAL_CACHE` — both the warm pass and the real run inherit
whatever you export, so they share it:

```bash
export TT_METAL_CACHE=/tmp/ab_jit_cache   # both arms inherit this; script never overrides it
```

## Procedure

```bash
TESTS="tests/ttnn/unit_tests/operations/..."   # SAME selection + args for both arms
export TT_METAL_CACHE=/tmp/ab_jit_cache

# --- Arm A: cold (old inline, serial compile) ---
rm -rf "$TT_METAL_CACHE"
scripts/run_safe_pytest.sh --run-all $TESTS 2>&1 | tee /tmp/ab_cold.log
grep SAFE_PYTEST_TOTAL_RUNTIME /tmp/ab_cold.log

# --- Arm B: precompile (hardware-free parallel warm + warm run) ---
rm -rf "$TT_METAL_CACHE"
scripts/run_safe_pytest.sh --precompile --run-all $TESTS 2>&1 | tee /tmp/ab_warm.log
grep SAFE_PYTEST_TOTAL_RUNTIME /tmp/ab_warm.log
```

`--run-all` (not the default `-x` fail-fast) is important: a fail-fast stop on the first failure
would cut the run short and the timing wouldn't represent the full suite.

## Fair-test checklist

- **Same selection + same extra args** for both arms (the warm collect must mirror the real run).
- **Reset the cache before *each* arm** (`rm -rf "$TT_METAL_CACHE"`) — including before Arm A.
- **Confirm B actually warmed.** Look in `/tmp/ab_warm.log` for:
  ```
  PRECOMPILE: ✓ fingerprint matches your device (build_key …) — the warm cache WILL be reused.
  PRECOMPILE: ✓ warm pass complete in Ns …
  ```
  If you instead see `✗ build_key MISMATCH … running COLD`, B silently degraded to a cold run and
  you're comparing **cold vs cold** (no speedup expected — not a bug). Fix per `PRECOMPILE.md`
  (usually `rm -f /tmp/tt_precompile_cluster_desc.yaml` then re-run).
- **Descriptor cache is a one-time cost.** The *first* `--precompile` run in a container pays a
  one-off cluster-descriptor capture, cached at `/tmp/tt_precompile_cluster_desc.yaml` and reused
  after. To measure *steady state*, run B once to warm the descriptor, then reset **only the JIT
  cache** (not the descriptor) and measure B again.
- **Fixed per-run overhead amortizes.** Each `--precompile` run opens the device once for the real
  build_key probe (not cached) — a fixed cost folded into B's total. It amortizes over suite size:
  the larger the suite, the more the parallel-warm win dominates. **Tiny suites can be slower under
  `--precompile`** — that's expected, the up-front overhead isn't worth it for a handful of kernels.
- **Run each arm 2–3×** and compare **medians** — host/device load varies run to run.
- **Run on an otherwise-idle box.** The warm pass is `nproc`-way parallel; a competing job stealing
  CPU will inflate B's warm time (and A's inline compile too, but B has more to lose).
- **Sweep parallelism if you care.** B's warm defaults to xdist `-n nproc`; tune with
  `--precompile-workers N` to find the sweet spot (diminishing returns + redundant cross-process
  compiles eventually cap the gain).

## Attributing where the time went

The two `SAFE_PYTEST_TOTAL_RUNTIME` numbers are the bottom line. To break B down further:

```
PRECOMPILE: ✓ warm pass complete in 16s (build_key …) — the real run below reuses it.
```

So roughly `B_total ≈ (descriptor + real build_key probe, fixed) + (warm pass, e.g. 16s) +
(warm pytest run)`. In Arm A that compile cost is instead smeared *inline* across the pytest run, so
A's pytest summary wall-time is inflated by serial compilation while B's is near pure execution. The
detailed warm-collect log is at `/tmp/precompile_collect_$$.log` (per-worker `compiled N programs in
Xs` lines).

## Related

- `scripts/precompile_verify.sh <tests>` — correctness (R1): asserts B's pass/fail/skip counts are
  identical to A's. Run this before trusting any speedup.
- `PRECOMPILE.md` — what `--precompile` does, the diagnostics, and known limitations.
