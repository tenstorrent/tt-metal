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
  --row metal__ckernel_sfpu_signbit \
  --compiler-ab-off-options="-mno-tt-tensix-optimize-latency-schedule -mno-tt-tensix-optimize-dst-iteration-fusion -mno-tt-tensix-optimize-replay-hoist -mno-tt-tensix-optimize-invariant-loadi -mno-tt-tensix-optimize-dst-autoincr" \
  --compiler-ab-on-options="-mtt-tensix-optimize-latency-schedule -mtt-tensix-optimize-dst-iteration-fusion -mtt-tensix-optimize-replay-hoist -mtt-tensix-optimize-invariant-loadi -mtt-tensix-optimize-dst-autoincr -mtt-tensix-macro-planner" \
  --require-changed-binary --require-compiler-pin \
  --run-root /tmp/sfpu-signbit-compiler-ab
```

Those are the canonical post-WP8 flag sets. The pre-WP8
`-mtt-tensix-{analyze,emit}-loadmacro` flags were **removed** with the
quarantined exact-calendar pass and now error on use; the planner ON leg is
`-mtt-tensix-macro-planner`. Any doc or script still naming the loadmacro
flags is stale.

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

Device baselines are keyed by **chip class** and never compared across
classes: `sfpu_device_baseline_v1.tsv` is the immutable p100a-era migration
source; `sfpu_device_baseline_p150_v1.tsv` carries the p150 cells seeded from
the post-WP8 sweep (its Reduce-SDPA `generated` row is an explicitly flagged
`measured_known_regression` — the profitability-gate fix lane owns that
update).  `--chip-class {p100a,p150}` selects the matching checked-in file
when `--baseline` is not given explicitly.

**Baseline update procedure (reviewed, manual):** sweeps only *report* drift
and never modify checked-in baselines.  To update: take the candidate cycles
from a green sweep evidence dir (`scoreboard.tsv`), verify the run's
`preflight.json` compiler sha and CRAQ verdicts, edit the chip-class baseline
TSV with provenance pointing at the evidence dir, and land it through normal
review.  A win→loss flip is never "updated over": it is a STOP event that
must be bisected first.

## One-command 2x2 sweep (HANDOFF §1/§3 protocol as code)

`sweep_2x2.py` regenerates the full `{semantic, hand} × {passes OFF, ON}`
silicon matrix from `sweep_2x2_ops.tsv` (exact pytest node ids, marker
discipline, CRAQ arch legs; absent rows are machine-readable SKIPs;
`kind=pinpair` rows run a paired gen-vs-hand A/B at the row's pinned flag
set — the Reduce-SDPA 834-vs-840 pin-coherence pair).  It
encodes the silicon protocol: pinned-compiler preflight (**cc1plus sha is
the primary pin**, resolved via `g++ -print-prog-name=cc1plus`; the driver
sha is secondary — the driver binary is identical across cc1plus-only
changes; plus the removed-flag probe), changed-binary classification
**before** any device job
(byte-identical OFF/ON pair ⇒ recorded refusal, no device run), paired CRAQ
through the generic-path simulator, every device job under both exclusive
flocks (`/tmp/tt-device.lock` outer, `/tmp/tt-llk-sfpu-silicon.lock` inner),
3 fresh profiler processes per selector per leg alternating OFF/ON with
unique `RUNNER_TEMP`s, raw+post CSVs copied in-lock, hand OFF==ON
byte-identity filling both hand cells from one physical run, and per-op
evidence (ELFs, `.text` hashes, `build.h`, logs, CSVs, compiler sha,
`SHA256SUMS`).  Markers: `KERNEL` for fire-and-forget replay-launch shapes
(SDPA — BODY is invalid there), `TILE_LOOP`/body markers otherwise; metric is
post-CSV `mean(MATH_ISOLATE)`/`tile_cnt` = cycles/tile.

```bash
python3 tt_metal/tt-llk/tests/corpus/sweep_2x2.py \
  --evidence-root ~/sfpi-uplift/sweep-2x2/evidence-$(date +%Y%m%d) \
  --cc1plus-sha 33221397 \
  --compiler-sha 4633999c \
  --sim-bh ~/sfpi-uplift/craq-sim/src/_out/release_bh/libttsim.so \
  --sim-wh ~/sfpi-uplift/craq-sim/src/_out/release_wh/libttsim.so \
  --allow-hardware \
  --baseline tt_metal/tt-llk/tests/corpus/sfpu_device_baseline_p150_v1.tsv
```

The run is idempotent and resumable per row/job (`--force` re-runs), supports
`--ops`, `--phases classify,craq,silicon,report`, `--dry-run`, and exits
nonzero on any RED (correctness failure, CRAQ gate, win→loss flip vs the
baseline).  `REPORT.md`, `SCOREBOARD.md`, `scoreboard.{json,tsv}` and
`SHA256SUMS` land in the evidence root.

Post-review hardening (PULL_ANALYSIS-20260817 §4, D2–D6):

* **Hash-matched resume** — cached device cells are reused only when their
  archived `.text` hash set equals what this run's compiler produces for the
  same node/flags; classify/CRAQ verdicts are keyed to the cc1plus (and
  simulator) sha256 and re-run on mismatch.  Stale-compiler cells re-measure
  instead of being silently trusted.
* **Class-aware report (D4)** — baseline rows carry `expected_class`
  (win/parity/loss/refusal); a prior WIN row that becomes a byte-identical
  refusal is RED, refusal→changed is a flagged YELLOW notice.  Proven by
  `selftest_sweep_2x2_report.py` (runs before every scheduled sweep).
* **Per-knob silicon (D3)** — weekly knob legs run the identical
  classify → paired BH CRAQ → correctness-then-perf pipeline as main legs,
  and only for rows whose main CRAQ gate is green.
* **DejaGnu gate (D2)** — extracted to `dejagnu_gate.sh` (clean→GREEN,
  failing→RED, missing-.sum→RED), self-tested by `selftest_dejagnu_gate.sh`.
* **Issue-slot sanity check (HANDOFF §1)** — rows with `issue_slot_lb`
  (BODY-family markers on macro-launch shapes, e.g. typecast at 128
  cycles/tile from the planner-dump payload structure) mark any reading
  below the bound INVALID_MARKER (RED; KERNEL marker required) and record
  the passing check in the result notes otherwise.
* **Scoreboard schema 2** — per-cell `compiler_sha` (cc1plus) and
  `craq_sim_sha` columns.

Sweep-hardening round 2 (adversarial review, 2026-08-16):

* **Keyed silicon phase** — the BH CRAQ gate accepts only verdicts keyed to
  THIS run's cc1plus+simulator+tt-metal head, and `--phases silicon` on an
  evidence root without classify evidence keyed to this run withholds the
  row RED (stale-toolchain greens can no longer authorize device jobs).
* **Cache identity** — cached device cells additionally key on the pytest
  node id + flags + extra_env (`jobkey.json`); an absent classify hash
  reference (`expected_texts=None`) re-runs, never reuses; `tt_metal_head`
  keys carry a `+dirty.<sha>` suffix for uncommitted tracked tt-llk edits.
* **Perf requires correctness** — every perf selector must have its
  correctness selector (ops-load validation, loud failure).
* **Magnitude-aware report** — per-cell ABSOLUTE cycle drift vs baseline
  (`--max-abs-drift-pct`, RED on slowdowns: uniform slowdowns preserve every
  ratio and hand legs on refusal rows fed no comparison), INVALID_METRIC
  (unparsable metric on a row with baseline history = RED), WIN→PARITY = RED
  by default (`--allow-win-to-parity` downgrades), loss growth beyond
  `--red-loss-growth-pct` = RED; YELLOW rows show `YELLOW`, never `ok`.
* **Toolchain identity** — pins are FULL 64-hex sha256 values (prefixes
  rejected); the reviewed `PINNED_*` values in `sweep_2x2.conf` cannot be
  silently overridden from the environment (wrappers take
  `--allow-pin-override` for a deliberate, logged one-off); the tests/sfpi
  symlink realpath is recorded in the manifest, a `--compiler` diverging
  from the harness toolchain aborts, and the harness-resolved cc1plus is
  re-verified at every phase entry.
* **DejaGnu gate round 2** — any non-PASS outcome class (UNRESOLVED, XPASS,
  UNTESTED, UNSUPPORTED, ERROR; XFAIL excepted) gates RED; `g++.sum` is
  deleted before each suite (a leftover is never counted); resume re-derives
  the verdict from the stored summary (a resumed RED stays RED, a partial
  summary is RED); suite patterns are never glob-expanded in the caller's
  cwd (`set -f`).

### Scheduled sweeps

`nightly_bh_sweep.sh` (02:00) runs validate → classify → CRAQ → BH silicon →
report against the chip-class baseline and the previous nightly run.
`weekly_bh_sweep.sh` (Sun 04:00) adds per-knob attribution (each optimization
flag toggled individually; per-knob silicon legs for the `HEADLINE_ROWS`
only), the WH CRAQ matrix for macro rows, and the DejaGnu byte-parity suites
(`loadmacro*`, `macro-planner*`) against the pinned toolchain build tree
(SKIP if absent).  All knobs/rows/paths live in `sweep_2x2.conf`
(env-overridable EXCEPT the reviewed `PINNED_*` toolchain pins, which reject
environment overrides unless the wrapper is passed `--allow-pin-override`),
not in script bodies.  Install the cron entries with
`install_sweep_cron.sh` (prints by default; `--install` writes the crontab —
an orchestrator/owner step, both entries flock-guarded and logging to
`~/sfpi-uplift/sweep-logs/`).

### Enforcement layer (ledger item 10 — by-memory rules turned into gates)

The wave-5/6 reviews found the same failure repeated: the measurement rules
existed only as prose and kept being skipped under pressure.  Four of them
are now mechanical; each has a self-test both wrappers run first, so a broken
gate can never bless a sweep:

1. **REVIEW_RECORD required** — a sweep whose phases include silicon and that
   passes `--allow-hardware` refuses in preflight unless
   `<evidence-root>/../REVIEW_RECORD-<cc1plus-12hex>.md` exists for the
   CURRENT pin, quotes the full cc1plus sha256, and names reviewer/commits/
   gates (template `REVIEW_RECORD_TEMPLATE.md`; per-pin records are also
   checked in under `review_records/`).  The record's sha256 lands in the
   run's `preflight.json`/`MANIFEST.txt`.  HANDOFF §1(4) as code.
2. **conf-lint** — `conf_lint.sh` (self-test `selftest_conf_lint.sh`) refuses
   the sweep, before the conf is even sourced, when the pin sha values, the
   conf's CURRENT PIN prose, the PIN HISTORY `(CURRENT)` entry, and the
   baseline TSV header anchors disagree — printing the exact disagreeing
   lines.  A re-pin must update prose+header in the same commit.
3. **issue_slot_lb required on macro-launch rows** — the classify phase
   disassembles every leg's `math.elf` (objdump is a preflight-verified
   tool); a row whose ON binary contains SFPLOADMACRO launches or ON-only
   fire-and-forget `ttreplay` launches with an empty `issue_slot_lb` is RED,
   named in the report with the §1 caveat (units rule in the
   `sweep_2x2_ops.tsv` header; self-test `selftest_enforcement_gates.py`).
4. **Sim sha pins** — `PINNED_SIM_BH/WH_SHA256` in `sweep_2x2.conf` under the
   same reviewed-guard discipline as the compiler pins; `sweep_2x2.py`
   verifies the libttsim sha256 at preflight and every phase entry
   (`--sim-bh-sha`/`--sim-wh-sha` from the wrappers).  A pinned-but-missing
   simulator refuses loudly instead of degrading to SKIP_NO_SIMULATOR.
5. **Measured rows stay wired (R7, Lane AZ corpus expansion)** — `conf_lint.sh`
   additionally refuses when any corpus manifest row with
   `perf_status=measured` has no `sweep_2x2_ops.tsv` row for its corpus id
   (the omission class that kept welford/recip/binary-bcast/mul_int measured
   but un-swept for two pin cycles); machine-readable `kind=skip` rows
   satisfy it, silence does not.

### Corpus-wide sweep surface (Lane AZ expansion)

`sweep_2x2_ops.tsv` carries EVERY mapped corpus row with a usable perf
vehicle: `kind=full2x2` rows are tier (a) (distinct hand + semantic forms),
`kind=semantic` rows are tier (b) *causal-only* rows — production typed-SFPI
bodies where hand == semantic source, so the honest axes are passes OFF vs ON
on the production body and no vs-hand claim exists.  Rows whose OFF/ON
binaries are byte-identical are recorded refusals (zero perf device jobs).
The `schedule` column implements the device-time budget as data: the nightly
wrapper passes `--schedule nightly`; weekly/manual sweeps run every row.
Per-selector `sem_extra_env`/`hand_extra_env` columns express same-node A/Bs
whose axis is a harness define (the mulint32 macro-vs-plain delivery pair).
Tier (c) corr-only and tier (d) blocked rows are documented in the Lane AZ
audit evidence, deliberately unwired.

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
- Post-WP8, the compiler DOES form `TTREPLAY` (replay-hoist under the
  profitability gate) and derives `SFPLOADMACRO` generically under
  `-mtt-tensix-macro-planner` (fires on the fresh Min/Max, Signbit, and
  UInt16→Float16_b Typecast shapes; refuses byte-identically elsewhere).
  `TTI_MOP` is still never compiler-emitted, so TopK/MoE rows remain ranking
  targets rather than immediately convertible A/Bs.
- The simulator runner needs a tt-llk `.venv` in the selected checkout.  It
  fails before execution if that environment or the requested `libttsim.so`
  is absent; no host wall-clock fallback is permitted.
