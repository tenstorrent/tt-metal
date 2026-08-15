# F1/F2 LLK corpus seed

## Complete SFPU corpus

`sfpu_corpus_v1.tsv` is the authoritative, versioned inventory of every SFPU
header shipped for Blackhole (41) and Wormhole (32, a strict subset). Unlike
the 11-row `f1_candidates.tsv` prioritization seed, every row is present and
has either audited functional/performance mappings or an explicit `unmapped`
state. Static columns record raw TTI, typed SFPI, replay, and MOP presence.

Validate inventory drift in presubmit:

```bash
python3 tt_metal/tt-llk/tests/corpus/sfpu_corpus.py --validate
python3 tt_metal/tt-llk/tests/corpus/sfpu_corpus.py --arch bh --list
```

Regenerate only after auditing additions/removals and mappings:

```bash
python3 tt_metal/tt-llk/tests/corpus/sfpu_corpus.py --update --validate
```

The runner exposes `compile`, `craq`, and serialized `silicon` modes. Without
the required simulator or explicit hardware `--execute` authorization it
records machine-readable `SKIP_NO_SIMULATOR`, `SKIP_UNMAPPED`, or
`SKIP_HARDWARE_NOT_AUTHORIZED`; it never substitutes wall time or instruction
counts for device cycles. Each run writes pinned revision/manifest provenance,
JSON, TSV, and Markdown summaries under its run directory. Hardware collection
is intentionally nightly/manual and must be serialized by the lab runner; it
is not a pull-request gate.

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

Every measured run requires an explicit simulator: set
`CORPUS_SIMULATOR=/path/to/libttsim.so` or pass `--simulator /path/to/libttsim.so`.
This prevents a silent release/debug change from being inferred from
`CRAQ_SIM_ROOT`.  The runner validates and SHA-256 records the resolved artifact
in `provenance.tsv`.

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
