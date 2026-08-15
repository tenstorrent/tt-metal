# F1/F2 LLK corpus seed

`f1_candidates.tsv` is a deliberately small, auditable corpus seed for the
F1 cost-model and F2 differential-driver work.  It inventories only kernels
with a hand-written replay/MOP/SFPLOADMACRO advantage or a clearly convertible
SFPI boundary.  It is not a claim that every row already has a generated
replacement.

The runner measures simulator **device-cycle trace data** through
`craq-sim/scripts/perf/llk-sim-perf.sh`.  It does not use pytest elapsed time,
ELF size, or static instruction count as a score.  Existing `perf_*.py` paths
are recorded in the manifest for a later paired silicon run; they are not fed
to the simulator runner because craq-sim's functional-nodeid pipeline requires
the matching `test_*.py` nodeids.

## Reproduce a baseline

Build the simulator libraries, provision the normal tt-llk test virtualenv,
then run from this worktree:

```bash
CRAQ_SIM_ROOT=/localdev/nkapre/craq-sim \
  tt_metal/tt-llk/tests/corpus/run_craq_sim_corpus.sh --arch bh --tier 1 --sample 1
```

The launcher explicitly requires the `pytest_workerid_plugin` shim because
craq-sim intentionally omits xdist while some fixtures request `worker_id`.
Its default location is `/localdev/nkapre/sfpi-gcc-lreg-artifacts`; override
`PYTEST_WORKERID_PLUGIN_DIR` (and, if needed, `PYTEST_WORKERID_PLUGIN`) for a
different checked, importable shim.  The launcher exports both `PYTHONPATH` and
`PYTEST_PLUGINS` and records the selected values in `provenance.tsv`.

It defaults unconditionally to the tt-metal checkout containing this script,
even if the caller has `TT_METAL_HOME` exported.  Use
`CORPUS_TT_METAL_HOME=/path/to/checkout` or `--tt-metal-home /path/to/checkout`
only when deliberately measuring a different checkout; the resolved path is
recorded in `provenance.tsv`.

The runner also clears pytest's configured `addopts` for CRAQ collection and
execution.  This keeps a corpus checkout usable with a deliberately xdist-free
venv even when its normal `pytest.ini` contains xdist-only options; it does not
turn a failed test into a metric.

Use `--sample 0` only for the full parameterized corpus.  That is the
reproducible training/evaluation mode; it can be large.  `--list` records the
exact modules selected before spending simulator time.  The run directory
contains the selected manifest, the tt-metal/craq-sim revisions, `llk_nodeids.txt`,
and craq-sim's `llk_sim.tsv`; full pytest nodeids are the join key for a later
silicon collection.

For a real compiler A/B, add an identical-source generated selector to the
relevant test and run both selectors through this same path.  A hand-written
replay/MOP version versus an uncompressed generated implementation is a useful
Track-D gap measurement, but is not evidence for an F1 scheduler change.

## Current constraints

- The generated `vFloat` Welford body is the only existing paired SFPI body.
- `TTREPLAY`, `TTI_MOP`, and `SFPLOADMACRO` are not emitted by the compiler,
  so TopK/MoE rows are ranking targets rather than immediately convertible A/Bs.
- The simulator runner needs a tt-llk `.venv` in the selected checkout.  It
  fails before execution if that environment or the requested `libttsim.so`
  is absent; no host wall-clock fallback is permitted.
