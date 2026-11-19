# Component profiling pipeline audit

CPU/source-only independent audit, 2026-11-14. No hardware, native large conversion, or implementation edits. The two OOM captures remain invalid. **Mid-run dump transport is compatible with the current native converter; complete-profile acceptance still has coverage gaps described below.**

## Compatible mechanisms

- `collect_full_model_components.sh:17–20` enables the traditional profiler, native op metadata, trace tracking and mid-run dumps while disabling device pushes to Tracy. The repaired real-time Tracy handler must be built/installed before the next run; this audit does not establish that installation.
- `profiler_analysis.cpp:661–687` appends native `cpp_device_perf_report.csv` rows and writes its header once. `profiler.cpp:1200–1262` does the same for the 15-field device marker CSV. Mid-run dumps clear marker sets (`:2524`) but do not clear `traces_replayed` or `runtime_ids_per_trace`. Replay sessions therefore remain cumulative per trace, not reset to one at each dump (`:1516–1528`, `:1925–1952`).
- The converter resolves the artifact root to an absolute path (`profile_full_model_components.py:224`), reads `raw/.logs`, and creates valid absolute symlinks for native artifacts in `postprocess/.logs`. Host root timing export uses `-u -c -f TT_`, avoiding malformed multiline automatic `Program_` source fields; metadata export keeps all messages, including `TT_SIGNPOST` and trace markers. It writes exports through `.partial` paths and refuses to reuse an interrupted partial.
- `compact_device` preserves one metadata row per `(device, encoded runtime ID, trace, replay session)`, rejects conflicting sub-device IDs, and verifies the resulting native lookup. It does not derive timings from the compacted marker file. Timings come from the untouched C++ report.
- Native `process_ops_logs.py:350–404` joins C++ results by `(device, runtime ID, trace ID, session)` and maps session N to the Nth host replay message for that same device/trace. Per-device encoded program IDs remain distinct. Mid-run file ordering across chips does not need to match host ordering: the converter sorts the final timeline using replay timestamps (`:1421–1446`). It is therefore appropriate to select signpost windows from the final report.
- Each measurement signpost encloses 16 complete replay/drain iterations (`run_full_model_performance.py:273–307`). Drains follow synchronization; start/end boundaries cannot intentionally split a kernel or trace. Reset, initial request seed, warmup and readbacks occur outside the measured windows. The C++ latency-between-ops field may restart at each dump; the component summary uses `DEVICE KERNEL DURATION [ns]`, not that latency field.
- The summary requires one ordered start/end pair, exact rank set `{0,1,2,3}`, positive finite durations, expected trace IDs, 16 nonempty sessions per component/rank, and identical recorded operation signatures across those sessions (`summarize_multichip_perf.py:115–140`; `profile_full_model_components.py:130–166`). Token-out contains model and sampler traces; the two controls contain only their respective trace.
- `metadata()` rejects unsuccessful, non-profiled, headline-eligible or wrong-shape artifacts. A CPU-only call against both failed v1/v2 metadata files raised `ValueError: Require successful diagnostic profile`, as intended. This is rejection evidence only, not a passing profiler fixture.

## Concrete acceptance gaps and proposed repairs

**1. Identical incomplete replays can currently pass.** The first observed replay is used as the completeness baseline. Native host import deliberately warns and skips malformed/truncated op JSON or cached references without prior metadata (`process_ops_logs.py:459–490`). If one captured operator's metadata disappears, that operator can be omitted from all 16 converted replays; their signatures still match. The native enrichment validates host-to-device matches, but does not require every native measured device row to have a converted host row. Native duplicate execution-UID rows also overwrite silently (`:382`).

Propose a stage-local validation before summary: reject these native import loss warnings, reject conflicting/duplicate measured execution UIDs, and compare the converted measured `(device, runtime ID, trace, session)` multiset against the native C++ rows for those same trace/session keys. Any deliberately excluded native record needs a precise, explicit reason. Compare session sets across all four ranks for each component as well as checking their counts. This does not alter native timing or model code. No implementation change was made in this audit.

**2. Required metadata coverage is not enforced.** `read_csv` requires duration/identity columns, but not the dtype/fidelity columns used in grouping; `row.get(..., "")` silently produces blank group fields. Blanks are honestly labeled unreported, yet `validated_profile` can still be emitted with all dtype/fidelity missing. Native JSON can drop `kernel_info` to fit its message limit, and native CSV's fidelity field uses the last compute-kernel entry when one op has several (`process_ops_logs.py:1500–1506`).

Propose mandatory operand dtype/shape/memory coverage for the actual compute/sampler/CCL operators and mandatory fidelity for operations with compute kernels. Allow explicitly named data-movement operations to have no fidelity. For mixed-fidelity multi-kernel operations, preserve/report the native JSON kernel list instead of implying one CSV fidelity describes every kernel. Report the coverage audit and missing fields before calling the profile validated; do not invent metadata from intended model policy.

**3. Direct converter invocation bypasses shell capture rejection.** The shell now rejects source-location instrumentation failures, dropped markers and partial-report warnings, including zero capture exit codes. `convert`/`summarize` invoked separately check model metadata but do not inspect `capture.log` or postprocessing loss warnings. A standalone conversion of a model-success/collector-failure artifact therefore relies on later indirect checks.

Propose sharing a small capture-health validator in `metadata()` (capture file present; reject instrumentation/drop warnings), and checking conversion warnings before report acceptance. The shell guard runs after workload completion; there is still no live collector-disconnect abort monitor. This is clearly documented rather than presented as OOM prevention.

## Attribution limits

The grouping is useful operation-work attribution. Per-rank duration sums and the maximum combined model/sampler rank are not wall latency; ranks are never summed into throughput. The reduced-stack sampler fraction cannot establish the all40 fraction. The fixed-token model-only control and fixed-logit sampler-only control have different activation/state regimes and are correctly labeled non-additive. Category assignment is a name/memory heuristic; fused collectives remain indivisible and generic custom kernels cannot be assigned exact sub-operation costs. Use raw op groups and kernel metadata when deciding whether LM-head/sampling dominates. Mid-run dumps reduce cross-device timestamp alignment accuracy, so no precise CCL overlap/skew claim follows from this capture.

## Next-run acceptance and pins

After the profiler library repair is installed, require successful native capture with no loss warnings, all 16 sessions × 4 ranks × requested components, unchanged trace/program identities, native-to-converted execution coverage, and explicit metadata coverage. Preserve failed runs, raw files and commands. Headline performance remains the separate unprofiled all40 result.

Inspected SHA256:

- `tests/profile_full_model_components.py`: `1aaf6597213636d96bc5893801ceeef00c95b57ba431dd11fafcbbc3fc3dcaa6`
- `tests/collect_full_model_components.sh`: `49752c4cad27b9d570e57c235d63ce50d3b87e0a044efa3dd1516b95b11190d4`
- `tools/tracy/process_ops_logs.py`: `c1637b72354446b20ea898c66d139abab7001b04de87873cf5584c1239f0c9ec`

## Stage-local repairs implemented after this audit

The parent authorized the three proposed acceptance repairs. `tests/profile_acceptance.py` now validates capture/conversion loss warnings for direct entry points; compares measured native and converted execution-UID multisets; rejects duplicate UIDs and differing session sets across ranks; and validates actual native operand/kernel metadata. The converter calls these checks after native processing and the summarizer repeats them before emitting a validated summary.

`native_program_metadata.json` preserves every selected program's native input/output operands, attributes and complete kernel inventory. Each converted operand's dtype/layout/shape/memory must match it. Missing/truncated inventories and unsupported metadata fail with concrete program/operator/field lists. Math fidelity is required for every native compute kernel. A non-compute operation is accepted without fidelity only when native JSON explicitly contains an empty compute list and a nonempty data-movement kernel inventory with source/name fields. No operation-name blanket exemptions or policy-inferred metadata are used.

Grouping now includes program hash and the full observed compute-kernel fidelity set; it does not rewrite the original `ops.csv` scalar. Multi-dtype operands and mixed HiFi2/LoFi kernel inventories are covered by CPU fixtures. Coverage/session evidence and the metadata artifact hash are included in `component_summary.json`, along with the acceptance helper source hash.

Validation: `python -m unittest models.autoports.qwen_qwen3_6_35b_a3b.tests.test_profile_acceptance -v` passed all 11 tests (0.071 seconds), including a complete synthetic 512-execution fixture, an operator missing from every replay, missing native rows, native/converted duplicate UIDs, rank/session mismatch, missing/truncated metadata, missing converted dtype/shape, explicit movement-only semantics, malformed JSON and capture/import warning rejection. Synthetic fixtures are not hardware evidence. `py_compile` and repository pre-commit checks passed for the three changed Python files. No device or large native conversion was run. Live collector abort monitoring remains outside these acceptance repairs.

Post-repair SHA256:

- `tests/profile_acceptance.py`: `0a1713254298701810235e0dd6a352a491bedcdb9a63c40375f86bdbd1ae00d9`
- `tests/profile_full_model_components.py`: `787928885fa26ab2299eaf650d5dd953fd9ebdd9ea0a7b69a6f5e42bf51488b5`
- `tests/test_profile_acceptance.py`: `7ed53e0afa502e8ab272265e8a15d3527fc56f235693c4b070409730da014f52`

## Bounded native-format compatibility check

`native_metadata_compatibility.json` records a CPU-only check against the existing optimized-multichip layer-0 capture. The 8,583,324-byte native metadata file contains 24,413 messages and 1,700 full metadata records/unique `(device_id, op_hash)` keys. Seventeen selected programs matched converted rows after scanning only the first 16,927 rows of the existing `ops.csv`; no converter, device or large capture processing was invoked. The check completed in 0.939 seconds.

All 17 selected programs passed the actual metadata loader and validator unchanged. Coverage includes copy, typecast, generic kernels, dense/sparse matmul, TopK, softmax, reduce-scatter, all-gather, all-reduce, dispatch/combine/post-combine, and grouped MoE on all four ranks. Native shape axes are strings such as `1[1]`, matching CSV values directly. Native device memory dictionaries map exactly to `DEV_<id>_<buffer>_<layout>`. Identical MoE program hashes on four devices remain four separate keys. Kernel source/name/fidelity fields validate; observed compute fidelities include HiFi4 and LoFi. Copy and all-gather have explicit empty compute inventories and nonempty movement inventories, correctly producing `[]` fidelity. No parser repair was needed.

The capture has no repeated uncached device/program key, so this check does not exercise repeated-record equivalence. Selected operands have device-dictionary storage, not string host storage. This is prior-stage format evidence only: it does not validate the upcoming mid-run dump capture, replay completeness or full-model performance acceptance.

Reproduce the bounded selected-program validation from the repository root:

```bash
python - <<'PY'
import csv, json
from pathlib import Path
from models.autoports.qwen_qwen3_6_35b_a3b.tests.profile_acceptance import (
    load_program_metadata, program_key, validate_metadata,
)
report = json.loads(Path("models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/performance/native_metadata_compatibility.json").read_text())
wanted = {(p["device_id"], p["program_hash"]) for p in report["programs"]}
programs = load_program_metadata(Path(report["metadata_path"]), wanted)
rows = {}
with Path(report["ops_path"]).open() as stream:
    for row in csv.DictReader(stream):
        if row.get("DEVICE ID") and row.get("PROGRAM HASH"):
            key = program_key(row)
            if key in wanted:
                rows.setdefault(key, row)
        if len(rows) == len(wanted):
            break
assert set(rows) == wanted
validate_metadata(list(rows.values()), programs)
print("PASS", len(rows), "native programs")
PY
```
