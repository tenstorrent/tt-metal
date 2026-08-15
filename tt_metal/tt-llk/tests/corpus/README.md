# F1/F2 LLK corpus seed

## Complete SFPU corpus

`sfpu_corpus_v2.tsv` inventories both active SFPU implementation surfaces:
legacy tt-llk common headers (41 BH/32 WH) and Metal ckernels (111 BH/106 WH),
plus Quasar (14 legacy/28 Metal): 164 logical implementations, 332 arch-specific
paths, and 143 combined basenames. Duplicate basenames remain distinct
rows through the `surface` and full path identity columns. Unlike
the 11-row `f1_candidates.tsv` prioritization seed, every row is present and
has either audited functional/performance mappings or an explicit `unmapped`
state. Static columns record raw TTI, typed SFPI, replay, and MOP presence.
Version 2 also makes the fresh semantic-C++ program auditable per logical row:
`semantic_cpp_class`, an exact readiness/blocker statement, paired-selector,
test and performance status, correctness metric/threshold/source, and the
current scoped silicon result/source.  The five readiness classes are
`ready`, `typed_wrapper_needed`, `macro_dependent`, `multithread_boundary`,
and `unmapped`.

Semantic status is never inferred from a basename or substring.  Every row is
keyed by its complete stable ID; a newly discovered ID fails validation until
it receives an explicit audit.  `unmapped` means exactly what it says: no
row-specific conversion assessment has been completed, not that a similarly
named test was guessed to cover it.

`sfpu_corpus_v1.tsv` is retained unchanged as the migration source and review
baseline.  The runner intentionally consumes only v2; `--update` refreshes
discovery fields while preserving reviewed v2 audit fields by stable ID.  This
avoids silently treating v1's missing audit columns as passing defaults.

Validate inventory drift in presubmit:

```bash
python3 tt_metal/tt-llk/tests/corpus/sfpu_corpus.py --validate
python3 tt_metal/tt-llk/tests/corpus/sfpu_corpus.py --arch bh --list
python3 tt_metal/tt-llk/tests/corpus/sfpu_corpus.py --arch bh --list --plan-format json
python3 tt_metal/tt-llk/tests/corpus/sfpu_corpus.py --arch bh --list --plan-format markdown
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

Every checked-in `win`, `parity`, or `loss` is correctness-gated.  That does
not mean every kernel uses PCC: the manifest records the actual contract.
Welford uses explicit mean/M2 tolerances; Reduce-SDPA, binary broadcast, and
Reciprocal use the shared element-tolerance plus PCC gate; TTNN Where uses
bit-exact selection (including NaN equality); MulInt32 uses exact integer tolerance.  TopK remains
blocked until values and companion indices are represented soundly and then
checked together.  The silicon runner rejects rows without an implemented
selector, a passing test status, and a non-`none` metric, and executes mapped
functional modules together with performance modules before accepting device
measurements.

## Fresh semantic-C++ conversion program

The corpus is a queue, not a promise that all 164 rows are immediately pure
C++.  Conversion work should move a row through these durable gates:

1. audit the semantic body and architectural boundaries;
2. add a test-only handwritten/generated selector with identical inputs;
3. compile every supported WH/BH/QSR path;
4. pass the row's recorded exact/PCC/tolerance contract (CRAQ where supported);
5. collect serialized, scoped Blackhole device cycles; and
6. turn repeated blockers into general compiler/API passes, never kernel-name
   peepholes.

The presubmit compile job uses `--require-executed-mapped`: it fails when no
mapped row really executes or when any mapped row does not pass.  QSR's 42
paths are published in all three plan formats, but QSR is deliberately absent
from the compile-gate matrix until it has a reviewed functional mapping; an
all-`SKIP_UNMAPPED` lane is not reported as green compilation.

Execution attribution is node-specific.  Before a shared pytest invocation,
the runner collects each mapped row's selectors independently and records the
exact concrete node IDs (including every parameterized instance).  The
checked-in pytest reporter then records setup/call/teardown outcomes by exact
node ID.  A nonzero aggregate pytest return code is provenance only: a failing
node fails its owning row, while an unrelated row whose nodes all ran cleanly
remains `PASS`.  Collection errors and nodes that never produced an outcome
are explicit `ERROR_COLLECTION` and `ERROR_NOT_RUN` states.  Expected skips do
not hide failures; a row with passing nodes and no failure passes while its
skip count remains in `results.json`, and a row with only skips is
`SKIP_ALL_TESTS`.

The same preflight records the resolved SFPI compiler, binary SHA-256,
`--version`, the repository pin from `tt_metal/sfpi-version`, and the installed
`tests/sfpi/sfpi.version`.  Pin drift is always visible in provenance and can
be made blocking with `--require-compiler-pin`.  Feature probes are attached
to stable corpus row IDs: for example, a compiler without indexed TopK
multi-result builtins blocks the TopK row as `BLOCKED_COMPILER_CAPABILITY` but
does not suppress unrelated Sigmoid, Exp, broadcast, or reduction rows.

For an identical-source compiler flag differential, select one or more exact
rows and give both control and candidate options.  The runner executes each
row twice in isolated `RUNNER_TEMP` trees, records exact pytest node outcomes,
extracts and hashes every ELF's executable `.text` by stable relative path,
and classifies the pair as `CHANGED_BINARY` or `BYTE_IDENTICAL`. Comparing
`.text` deliberately excludes different build-root paths in DWARF/debug data.
This is a structural discriminator, not a performance claim; CRAQ remains a
functional gate and scoped silicon cycles remain the performance authority.

```bash
python3 tt_metal/tt-llk/tests/corpus/sfpu_corpus.py \
  --mode compile --arch bh --execute \
  --row legacy__ckernel_sfpu_welfords \
  --compiler-ab-off-options=-mno-tt-tensix-optimize-lp \
  --compiler-ab-on-options=-mtt-tensix-optimize-lp \
  --require-changed-binary --require-compiler-pin \
  --run-root /tmp/sfpu-welford-compiler-ab
```

Use `--require-changed-binary` only for a row expected to exercise the pass.
Byte identity is the correct result for ineligible fallback fixtures and can
be archived without that gate.

Current evidence seeds the program honestly: Welford, Reduce-SDPA, accurate-BF16
Reciprocal, and typed Signbit with compiler-formed SFPLOADMACRO are scoped wins;
binary broadcast is exact cycle parity, Where and MulInt32 are macro-dependent
losses, and TopK requires typed multi-result architectural
modeling before a performance claim.  `SEMANTIC_CPP_CI_PLAN.md` defines the
presubmit/nightly artifact and promotion policy.

`f1_candidates.tsv` is a deliberately small, auditable corpus seed for the
F1 cost-model and F2 differential-driver work.  It inventories only kernels
with a hand-written replay/MOP/SFPLOADMACRO advantage or a clearly convertible
SFPI boundary.  It is not a claim that every row already has a generated
replacement.

The runner records simulator **modeled-cycle trace data** through
`craq-sim/scripts/perf/llk-sim-perf.sh`.  It does not use pytest elapsed time,
ELF size, or static instruction count as a score.  Existing `perf_*.py` paths
are recorded in the manifest for a later paired silicon run; they are not fed
to the simulator runner because craq-sim's functional-nodeid pipeline requires
the matching `test_*.py` nodeids.

CRAQ modeled cycles are a functional and optimization discriminator, not a
substitute for physical device cycles.  Only serialized silicon runs may update
the device-cycle baseline or make a performance acceptance claim.

Compare a captured `results.json` against the checked-in scoped silicon baseline:

```bash
python3 tt_metal/tt-llk/tests/corpus/sfpu_corpus.py \
  --compare-results /path/to/results.json \
  --baseline tt_metal/tt-llk/tests/corpus/sfpu_device_baseline_v1.tsv \
  --max-regression-pct 0
```

Repeated baseline samples with the same operation, architecture, metric, scope,
and selector use their minimum, matching the established three-process device
profiling convention.  Missing or non-device-cycle values are explicit skips.

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
