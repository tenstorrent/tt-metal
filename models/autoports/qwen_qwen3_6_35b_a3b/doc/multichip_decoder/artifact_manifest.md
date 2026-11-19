# Multichip decoder evidence manifest

All paths below are relative to this directory. Commands run from the tt-metal
root after `source python_env/bin/activate; source ../lane-env.sh`. Device jobs
are serial, bounded and ownership-checked. Binary tensors, raw Tracy captures
and full console/report logs remain local; compact JSON/provenance and selected
report tables are versioned. No artifact archive or per-artifact hash bundle is required.

| Evidence | Exact path or finite pattern | Producer |
|---|---|---|
| Default paired timing/correctness | `final_l{0,3}_b{1,32}.json` | `tests/run_multichip_validation.sh clean` |
| Nonaligned lengths | `boundary_l{0,3}_s{1,31,65,257}.json` | same script `boundary` |
| 128-step replay stress | `stress_l{0,3}_s129_d128.json` | same script `stress` |
| Watcher | `watcher_l{0,3}_b{1,32}.json`, `watcher/l{0,3}_b{1,32}/generated/watcher/watcher.log` | same script `watcher` |
| Natural activation source | `natural_b32.pt`, `natural_b32.metadata.json` | `tests/prepare_multichip_activations.py` |
| Natural batch-32 correctness | `natural_l{0,3}_b32.json` | `tests/run_multichip_acceptance.sh natural` |
| Mixed stack/shared physical KV pool | `mixed_stack_selected.json` | same script `all` |
| Context smokes | `context_l{0,3}_32k.json`, `context_fullmax.json` | same script `context` |
| Capacity | `memory_capacity_plan.json`, `../context_contract.json` | `tests/multichip_capacity.py` |
| B1 profiler | `tracy/layer_{0,3}/ops.csv`, `correctness.json`, `capture.log`, `ops_source.txt`, `{prefill,decode}_perf_report.{txt,csv}`, `{prefill,decode}_op_groups.csv` | `tests/collect_multichip_perf.sh`, `tests/summarize_multichip_perf.py` |
| B32 profiler | same paths under `perf_b32/` | `tests/collect_multichip_profile.sh`; summarizer `--batch 32 --doc .../perf_b32` |
| Narrow B32 device acceptance | `perf_b32/tracy/layer_{0,3}/correctness.{json,pt}`, `correctness.json.provenance.json` in each directory | `tests/run_multichip_profile.py`; both device runs PASS |
| Preserved narrow linear raw capture | `perf_b32/tracy/layer_0/raw/.logs/{tracy_profile_log_host.tracy,tracy_ops_times.csv,tracy_ops_data.csv,profile_log_device.csv,cpp_device_perf_report.csv}`, `perf_b32/tracy/layer_0/capture.log` | Tracy capture; historical CPU conversion OOM after device close |
| Streaming CPU conversion | `perf_b32/tracy/layer_{0,3}/postprocess.log`, `postprocess/.logs/`, `host_times_compaction.json`, `device_metadata_compaction.json`, `ops.csv`, `ops_source.txt` beneath each artifact directory | `tests/postprocess_multichip_profile.py`; both native CSV conversions PASS |
| Historical malformed-host conversion | `perf_b32/tracy/layer_0/postprocess_malformed_host/`, `perf_b32/tracy/layer_0/postprocess_malformed_host.log` | strict CSV row-width failure after more than 300 million unfiltered rows; original export preserved |
| Native filtered host export | `perf_b32/tracy/layer_{0,3}/raw/.logs/tracy_ops_root_times.csv` | saved trace exported with `-u -c -f TT_`; optional host child timings excluded explicitly |
| Profiler summary | `perf_summary.json`, `perf_b32/perf_summary.json`, `profile_findings.md` | CPU raw-rank accounting |
| Gate manifest | `gate_summary.json` | `tests/summarize_multichip_validation.py` |
| Dense/sparse geometry controls | `geometry/search.json`, `sparse_geometry/search.json` and named result/provenance files | `tests/run_multichip_geometry.py` |
| Local dispatch controls | `local_dispatch_l{0,3}_{false,true}.json` | paired runner with `--mesh-build-kwargs` |
| Previous profiler screening | `profiling_global_dispatch/` | preserved B1 global-dispatch/GU16 capture |
| Historical failed B32 instrumentation | `failures/profiler_b32_no_cache/`, `failures/profiler_b32_cached/` | preserved 32K Tracy source-location failures |
| Paired profiler state seeds | `seed_l{0,3}_b32.{json,pt}` | paired runner `--save-prefill-states`; complete acceptance then narrow profile |
| Recovery | `logs/reset_{1..11}.log`, corresponding smoke logs; `work_log.md` | bounded list/reset/list and trivial mesh operation |
| Debugging and topology controls | `AUTODEBUG.md`, `AUTOFIX.md`, `collective_candidate_audit.md`, `ag_minimal_candidate.md` | source diagnosis, adapted coherent layer candidates |

Each paired result has a `.json.provenance.json` sidecar with source versions,
command and environment. Context/mixed-stack summaries carry their own contracts
and source evidence. The `logs/` file with each run name is its console log.
The binary `.pt` beside a result contains exact tensors; saved `baseline_*.pt`
controls are matched by signature before reuse. Reused-baseline runs support
correctness only; final clean timing always executes both sides freshly.

The narrow B32 profiler's prefill is **one natural 32-token prompt**; its decode
is one complete B32 step seeded from 32 accepted post-prefill states. Both device
fixtures and native CPU CSV conversions passed. The historical first linear CPU
conversion exited 137 after mesh close:
approximately 35 GiB of host timings and 44 GiB of legacy device rows drove pandas
to approximately 249 GiB. No agent killed the already-exited process and no reset
was required. Streaming CPU preprocessing preserves raw inputs and the original
authoritative C++ timing CSV, then invokes unchanged native report conversion.
Its first streaming attempt caught unescaped commas/newlines in `Program_`
source-file fields after more than 300 million unfiltered host rows. That failed
processing directory/log is preserved above. The current export uses native
`-u -c -f TT_` filtering on the saved trace, followed by the native-relevant row
predicate. Optional CompileProgram and HWCommandQueue_write_buffer host child
timings are excluded and disclosed in `host_times_compaction.json`; all device
C++ timing and op metadata remain unchanged. Independent CPU audit confirmed that
the omitted nonroot host-child columns are optional and unused by device timing
and `tt-perf-report`; native root order is preserved. The tooling repair is
resolved without a device reset. Human-readable tables and final B32 profiler
summaries are complete; all 27 acceptance gates pass. The older broad-capture
failures above are historical. The broad collector now rejects B32 and directs
callers to the narrow collector; shell syntax and exit-2 checks passed on CPU.

Versioned compact CSVs are `latency_comparison.csv` and both profiler roots'
`tracy/layer_{0,3}/{prefill,decode}_{op_groups,perf_report_stacked}.csv`.
Full raw/merged CSVs and human text tables remain local at the paths above.
`moe_share.json`, `profile_findings_b1.md` and `profile_findings.md` record the
source-boundary MoE shares, actual precision/configuration and measured costs.

Historical `precision_policy` dictionaries retain inherited baseline sparse keys.
They are report-only and do not control the mesh sparse kernel. For mesh execution,
`mesh_sparse_geometry` plus source defaults define GU (8 cores, K block 64) and
down (64 cores, K block 4); both have one output tile per core. GU input is DRAM,
expert weights DRAM, sparse outputs/intermediate L1. `expert_memory=l1` describes
those outputs. Final raw profiler rows are the precision/layout evidence.

## Trace allocation positive control (round 1 closure)

`trace_alloc_tracking/positive_control.log` is the prescribed observed rejection
and recovery log; `positive_control.json` and its `.provenance.json` sidecar in
that directory record the target 1x4 mesh, startup environment, exact exception,
source identities and command. Buffer30 was rejected while alive and never
acknowledged; output buffer23 alone was acknowledged. After releasing buffer30,
replay produced exact output on all four devices. The acceptance checker requires
this artifact. Reset12 and four-device trivial-op smoke are in
`logs/reset_12{,_smoke}.log`; the earlier zero-interval watcher launch's failed
open/triage evidence is preserved in `failures/trace_control_watcher_zero/`.
