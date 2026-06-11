# Per-op precompile A/B — methodology

How we measure what up-front precompile buys, **per operation**, on a representative set
of ops, across the **main** unit tests and the **nightly** suite. This is the agreed
procedure; the op table and the result template live in `OP_PERF_CATALOG.md`, and the
runner that executes all of this is `run_op_perf.sh` (driven by the `perf_set.sh` manifest).

This extends the suite-level `PRECOMPILE_AB_TESTING.md` in two ways the reviewer asked for:
**per-op reporting** and a **phase breakdown** of the precompile arm (collect / compile /
warm-run measured separately, not just one e2e number).

## Run it in tmux

The full sweep is long — every op pays a full cold compile in one arm and a full warm run in
the other, times two trees. **Always run inside tmux** so an SSH disconnect doesn't kill it:

```bash
tmux new -s perf
scripts/precompile_perf/run_op_perf.sh both 2>&1 | tee /tmp/perf_run.log
# detach: Ctrl-b d    reattach: tmux attach -t perf
```

The runner refuses to start outside tmux unless `PERF_ALLOW_NO_TMUX=1` is set.

## The two arms

Same test selection and same extra args in both arms — the precompile collect must mirror the
real run exactly. For each op:

- **Arm B — precompile ("the new thing"), runs FIRST.** `run_safe_pytest.sh --precompile`.
  Warms the on-disk JIT cache up front on the real device (collect + parallel compile), then
  runs the suite warm against that cache.
- **Arm A — cold baseline, runs SECOND.** `run_safe_pytest.sh` (no `--precompile`). The old
  inline path: kernels compile serially as the run reaches each op.

**B runs before A** deliberately: the new path is the thing under test, so it gets the first,
cleanest device window; if the box degrades or a run is interrupted partway through the sweep,
we still have the precompile numbers.

`--run-all` (not the default `-x` fail-fast) is used in both arms so a single failure doesn't
truncate the run and corrupt the timing.

## Always cold: delete both caches before each arm

A warm cache left over from a previous run silently turns the "cold" baseline into a warm one
and the comparison becomes meaningless. So **before each arm** the runner deletes both caches:

- **JIT kernel cache** — `rm -rf "$TT_METAL_CACHE"`. This is the on-disk compiled-kernel cache
  (`~/.cache/tt-metal-cache/` by default; `tt_metal/jit_build/build.cpp:90`).
- **ccache** — `ccache -C`. ccache is the C++ object cache the kernel build shells out to when
  `TT_METAL_CCACHE_KERNEL_SUPPORT` is set (`build.cpp:115`). We keep ccache **enabled** (so the
  measurement reflects the real ccache-on deployment) but **delete it before each arm** so every
  arm starts from a genuinely cold compiler cache.

Both caches are isolated to throwaway dirs (`TT_METAL_CACHE=/tmp/perf_jit_cache`,
`CCACHE_DIR=/tmp/perf_ccache`) so deleting them never touches your shared caches.

Within Arm B we do **not** clear between the warm-up and the warm run — the warm-up is supposed
to populate the cache the warm run then reuses. We clear once, before the arm.

> ccache state must be identical between the warm-up and the warm run, or the warm run silently
> misses (see `PRECOMPILE.md`). Because both inherit the same `CCACHE_DIR` and we only clear at
> arm start, they match by construction.

## The metric and the phase breakdown

Every `run_safe_pytest.sh` invocation prints, last, on stderr:

```
SAFE_PYTEST_TOTAL_RUNTIME: 4m12s (252s total, device-lock-acquired -> exit)
```

It measures **device-lock-acquired → exit** (idle lock-wait excluded), so Arm B's number
honestly includes its own warm-up and overhead. The headline per op is **speedup =
A_total / B_total**.

Arm B is then split into three phases. The runner reads them from the logs:

| Phase | What it is | Where it comes from |
|-------|-----------|---------------------|
| **collect** | NO_DISPATCH graph-capture pass that gathers the distinct programs (device open + running each test body to enqueue programs, no real dispatch) | `warmup_total − compile` |
| **compile** | parallel `up_front_compile` wall-clock that builds the collected programs | `UP_FRONT_COLLECT: compiled N programs in Xs` (collect log) |
| **warm-run** | the real pytest run reusing the warm cache | `B_total − warmup_total` |

where `warmup_total` is run_safe's `✓ warmup complete in Ns` (collect + compile combined) and
the collect log path is the `/tmp/precompile_collect_*.log` that run_safe prints.

This breakdown answers the two reviewer questions directly:

- **"How much parallelism, and the ceiling?"** — `compile` is the parallel wall; `programs` is
  how many were built. The serial reference (`UP_FRONT_COLLECT_WORKERS=1`) can be run once per
  op if a hard parallelism number is wanted, but the default sweep reports `compile` at the
  production worker count (`nproc`) only — we agreed not to sweep worker counts here.
- **"How much does waiting-until-everything-is-compiled cost?"** — `collect + compile` is exactly
  that barrier: the time Arm B spends before it executes a single test. That is the quantity a
  streaming design (overlap compile with execution) would hide. Compare it against Arm A, where
  the same compile cost is instead smeared *inside* the run.

**Residual inline compiles.** If `collect` missed some programs (a body the collector can't
capture faithfully — see the cold-compile carve-outs in `up_front_collect.py`), they compile
inline during the warm run and inflate `warm-run`. A warm-run wall well above Arm A's
execution-only floor flags residual misses; an optional second warm run (cache now fully warm)
isolates the delta.

## Cheap-compute collect routing ("fused-off" ops)

The collect pass shouldn't pay the host-side reference compute (it's irrelevant under
NO_DISPATCH — programs depend only on shapes/config). `tests/plugins/up_front_collect.py`
already routes the heavy ones to shape-correct stand-ins: `conv2d`, `conv3d`, `layer_norm`,
`group_norm`, `matmul`/`bmm`, `sdpa`, plus `from_torch` weight-prep. For any curated op whose
**collect** phase turns out to dominate (the phase breakdown makes this visible), add a stand-in
to that patch set — that is the "fuse off more ops" lever. It changes only collect cost, never
the programs collected, so it never affects correctness or the compile/warm-run numbers.

## Fair-test checklist

- Same selection + same extra args in both arms (the runner guarantees this).
- Caches deleted before **each** arm (JIT + ccache), isolated to throwaway dirs.
- Confirm Arm B actually warmed: `✓ build_key matches` / `✓ warmup complete`, not
  `✗ ... running COLD` (which means B silently degraded and you're comparing cold-vs-cold). The
  runner flags this in the `note` column.
- The cluster-descriptor capture is a one-time per-container cost; run B once to prime
  `/tmp/tt_precompile_cluster_desc.yaml`, then start the measured sweep.
- Run on an otherwise-idle box. The warm-up compile is `nproc`-parallel; a competing job steals
  CPU and inflates Arm B more than Arm A. Use the device lock (run_safe takes `flock` itself) and
  don't launch overlapping device jobs.
- Tiny per-op selections can be **slower** under precompile — the up-front overhead isn't worth
  it for a handful of kernels. That's expected and is itself a finding worth reporting.
- Run each op 2–3× and compare **medians** for anything headline; host/device load varies.

## Trace cases are excluded

Precompile cannot warm a traced command sequence, so trace-driven tests are deselected in both
arms via the manifest's `-k` filters. The known set and the pre-flight re-scan are documented in
`perf_set.sh` and `OP_PERF_CATALOG.md`.
