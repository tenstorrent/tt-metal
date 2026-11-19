# Final validated snapshot — clean-pass

All hard gates pass on decoder source SHA256
`476f9d99fa90a1268f264964037dbc6837b320adc9f48d172dc1156b8b35962d`.
[Acceptance index](acceptance.json) maps the final artifact set; earlier sections
below are chronological history and may describe failures now repaired.
The full advertised262144-token context is validated for both kinds, with no
capability reduction. All final hardware jobs exited0 and used physical chip0,
1×1 mesh, serialized execution and bounded timeouts. No later-stage work,
shared runtime edits, manual termination, operator reset or push occurred.

| Layer kind | Exact-max prefill PCC | Max-context traced decode PCC | Warm prefill kernel ms | Warm traced decode kernel ms |
|---|---:|---:|---:|---:|
| linear_attention | 0.999964119 | 0.999986067 | 16.051 | 2.735 |
| full_attention | 0.999876821 | 0.999574258 | 6.317 | 2.326 |

Timings are per layer, batch1, prefill512 and decode context513. Including
inter-op gaps: linear18.580/2.798ms, full6.479/2.371ms. Filtered CSV time units
are microseconds; raw ops and checksums are retained. Both final profiler logs
have zero dropped-marker warnings (`profiler_audit.json`). The earlier drained
run's timings are superseded. Human-readable tables are actual tt-perf-report
output, not CSV-mode status text.

Final artifact stems (each has JSON and console log unless specified):
- `gdn_contract_final`, `full_contract_final`: boundary lengths, seven resume
  windows, negative history controls and mixed rows; full runs with Watcher10.
- `gdn_trace_reuse_final`, `full_trace_reuse_final`: six requests each, one
  capture, no inter-request release/invalidation, identical retained trace ID.
- `{gdn,full}_final_context_{262143,262144}`: full TT output for every token,
  selected-row HF oracle, per-row contribution gate. Near-max runs include
  traced decode at262144 and bit-identical eager/trace results. Full-attention
  near-end continuation starts262111/262112 also passes.
- `layer_{0,3}_batch32_final`: real weights, traced decode, input refresh,
  repeated bit equality and allocation-tracking positive control.
- `layer_{0,3}_watcher_final`: four independent positions [33,1,63,32], traced
  replay and repeated bit equality under Watcher10. Raw logs are compressed;
  `watcher_audit.json` identifies clean final runs and historical failed probes.
- `position_fill`:35 exact-value Watcher cases spanning batch/alignment and
  high-position boundaries.
- `synthetic_pytest_final.log`:4 passed in23.84s. Two pytest warnings are dependency
  deprecations for SwigPyPacked/SwigPyObject lacking `__module__`; an exit-time
  swigvarlink deprecation has the same cause. No skipped or xfailed case.
- `runtime_audit.json`, `native_runtime_audit.md`: clean reachable-method and
  native-wrapper audits. No runtime host model fallback remains.
- `layer_{0,3}_profile_untracked`: accepted real-weight warmed performance runs.
  Tracking/tracebacks are explicitly disabled; `profile_final` runs and
  `perf_tracked/` retain the prior tracked correctness/profile evidence.
  Large original console files remain local and ignored; `.log.gz` copies are
  committed. `perf/{linear_attention,full_attention}/` holds tables, CSVs,
  raw compressed ops, exact renderer commands and full provenance.

Exact final command families, after activating python_env and lane-env.sh:

```bash
export TT_METAL_OPERATION_TIMEOUT_SECONDS=180
export TT_METAL_TRACE_ALLOC_TRACKING=1
# Run one device command at a time; substitute the recorded layer and output.
timeout 900 python -m models.autoports.qwen_qwen3_6_27b.tests.run_contract --layer 0 --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/gdn_contract_final.json
timeout 900 python -m models.autoports.qwen_qwen3_6_27b.tests.run_trace_reuse --layer 0 --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/gdn_trace_reuse_final.json
timeout 1200 python -m models.autoports.qwen_qwen3_6_27b.tests.run_long_context --layer 3 --length 262144 --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/full_final_context_262144.json
timeout 600 python -m models.autoports.qwen_qwen3_6_27b.tests.run_decoder --layer 3 --batch 32 --length 33 --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/layer_3_batch32_final.json
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/watcher_layer_3_final timeout 600 python -m models.autoports.qwen_qwen3_6_27b.tests.run_mixed_decode --layer 3 --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/layer_3_watcher_final.json
timeout 600 pytest -q --noconftest models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py
export TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_TRACEBACKS=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER TT_METAL_DPRINT_CORES TT_VISIBLE_DEVICES
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=5000 timeout 900 python -m tracy -r -p -v --no-web-server -o models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/tracy/layer_0_untracked -m models.autoports.qwen_qwen3_6_27b.tests.run_decoder --layer 0 --length 512 --profile-drain --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/layer_0_profile_untracked.json
python -m models.autoports.qwen_qwen3_6_27b.tests.render_perf --tracy-root models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/tracy/layer_0_untracked --result models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/layer_0_profile_untracked.json --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/perf/linear_attention
```

Layer3 profiling uses the corresponding `layer_3_untracked`,
`layer_3_profile_untracked.json` and `perf/full_attention` paths. Every result JSON
records its exact module arguments, environment, source/config hashes and
reference package versions. Watcher/profiler are unset for ordinary checks;
Watcher and profiler never overlap. `TT_VISIBLE_DEVICES` remains unset.

The final hook pass normalized trailing whitespace in rendered tables and a few
historical text logs, plus terminal newlines. Raw ops and compressed device logs
retain their original bytes; metrics and decoder source were unchanged.
Filtered CSV line endings were normalized from CRLF to LF for the checkpoint;
cell values and raw compressed source CSVs are unchanged.

## Review and checkpoint

Round 1 returned `more-work-needed`: [review](stage_review_1.md). Its sole
required finding was the measurement regime: accepted performance had allocation
tracking enabled. Repeated only the two B1/S512 profiles with tracking and
tracebacks disabled, explicit program-cache skip flag 0, Watcher disabled,
5000-program capacity and between-pass drains. Both exited 0 with PCC above
0.995, deterministic repeats, updated-input checks and no dropped markers.
Allocation positive-control is intentionally null in these untracked runs;
tracked correctness runs retain the passing control. No decoder or execution
harness change was made. `tests/evidence.py` now records both extra diagnostic
flags. Repository hooks pass for that provenance-only change (the environment's
standalone newer Black suggested formatting, but pinned repository Black passed
without changes). New logs include a pandas mixed-column DtypeWarning during
Tracy postprocessing; raw CSV generation and parsed timing totals succeed.

Accepted metrics now use `perf/`; historical tracked metrics and raw compressed
ops are preserved in `perf_tracked/`, with `profiler_audit_tracked.json` and
original `layer_{0,3}_profile_final.json/.log.gz`. New accepted profiler audit
records disabled flags and compressed-log hashes. Fresh diff-only round 2
[review](stage_review_2.md) returned **clean-pass**, with no Required Work.
All current and historical filtered timing CSVs are included through scoped
Git ignore exceptions. Local checkpoint SHAs are recorded below. No push.

### Local checkpoint

Repository: `/home/ttuser/qwen36-27b/tt-metal`; branch: `main`.
Implementation and reviewed evidence checkpoint: `5fe9ca54620d57743a2fce0122d52c7ed12b6656`.
A subsequent documentation-only commit records this SHA. Commit hooks passed;
all stage-owned files are committed locally. No push was performed.

### Handoff risks from round 1

- Long-context HF comparisons use selected boundary/final rows while TT executes
  every token and HF advances all state. This does not establish full-model or
  generated-text accuracy.
- Preserve the full-attention contribution gate during later optimization:
  native arithmetic is sensitive to reduction geometry; final minimum selected
  long-context contribution PCC is 0.9977730592.
- Serial mixed-request prefill and up to 511 exact-token steps for unaligned
  large-cache continuation are functional baselines. Aligned S512 profile rows
  do not characterize their latency.

## Chronological history

# Functional decoder work log

Stage 1 only, Qwen/Qwen3.6-27B, revision
`8f33b98937c6031bb49e713dee3e8e185aafd26f`. Starting tt-metal SHA
`a0b0bd54` on `main`; clean worktree at start. No push authorized.
Installed enabled skills: tt-model-bringup 0.1.23, tt-autodebug 0.1.8.
Startup inventory: `/home/ttuser/qwen36-27b/setup-evidence/plugins.json`.
All stage state remains on persistent storage under this autoport.

## Environment and commands

Every shell sources `python_env/bin/activate`, then
`/home/ttuser/qwen36-27b/lane-env.sh`. Transformers 5.12.1, source-built
TTNN. Weights were already cached before this stage; the layer runner reads
only relevant shard tensors. No full-model load or download was needed.
Hardware commands are serialized with bounded timeouts, on physical chip 0.
No resets, kills, or shared source changes have been performed.

- `fuser -v /dev/tenstorrent/*`: no owners before first device operation.
- `TT_VISIBLE_DEVICES=0 timeout 60 tt-smi -ls --local`: `device_list.log`.
- First open under `TT_VISIBLE_DEVICES=0` failed before execution:
  `device_smoke.log`, CUSTOM cluster requires a mesh descriptor because only
  one half of the P300 was visible. Source: `tt_cluster.cpp:305`.
- Resolved by keeping topology visible and opening `MeshShape(1,1)` with
  `physical_device_ids=[0]`. `device_smoke_physical.log` proves add/open/close.
  No multi-chip model computation is used.
- Repository hooks run before decoder experiments; see `hooks_*.log`.
- `timeout 300 python -m models.autoports.qwen_qwen3_6_27b.tests.run_decoder
  --length 1 --output .../gdn_smoke.json`: initial config class spelling failed;
  `WormholeComputeKernelConfig` is the checkout's supported class for Blackhole.
  Corrected rerun: `gdn_smoke_v2.log`, prefill PCC 0.99998498,
  decode PCC 0.99998629. Real layer 0 weights.
- Same command with `--layer 3 --length 33`: `full_smoke.log/json`,
  prefill PCC 0.99995124, decode PCC 0.99994940. Real layer 3 weights,
  permuted 32-token pages, partial tile tail.
- `TT_METAL_TRACE_ALLOC_TRACKING=1 timeout 600 python -m
  models.autoports.qwen_qwen3_6_27b.tests.run_decoder --layer 0 --length 33
  --output .../gdn_chunk_trace.json`: in progress; consult log for result.

## Architecture

HF Qwen3_5DecoderLayer uses linear attention on layers 0..2 then full
attention on layer 3, repeating across 64 layers. Hidden width 5120, MLP
17408. Both norms use `(1+weight)`. Linear attention has 16 Q/K heads,
48 V heads, all width 128, causal depthwise convolution width 4 and FP32
recurrent state. Full attention has 24 Q heads, 4 KV heads, width 256,
64 rotary dimensions, per-head Q/K RMSNorm and sigmoid output gate.
Residual order is pre-norm mixer + residual, pre-norm SwiGLU + residual.
The checkpoint's `output_gate_type=swish` field does not override the
installed HF full-attention implementation's explicit sigmoid.

## Remaining required work

Chunk/continuation, batched and full-context coverage; traced decode and
allocation-positive control for both kinds; real-weight precision gates;
negative history controls; deterministic replay and input refresh; warmed
Tracy/perf reports; runtime fallback audit; watcher; independent stage review;
local commits and SHA records. Current smoke evidence is not stage completion.

## Expanded validation and repair (2026-11-12)

All numbers below precede the FP32 folded-norm repair unless explicitly marked.
They establish partial progress, not final acceptance of current source.

| Artifact stem | Real layer / input | Result |
|---|---|---|
| `gdn_chunk_trace` | 0 / batch1 length33 | prefill .99998426; traced decode .99998653; repeated outputs bit equal |
| `gdn_batch32_trace` | 0 / batch32 length33 | prefill .99995780; traced decode .99996376; bit equal |
| `full_batch32_trace` | 3 / batch32 length33 | prefill old FP32 metric 1.0; traced decode .99994099; bit equal |
| `full_contract` | 3 / lengths1,31,32,33,127,128,129,511,512,513,1025,17 | pass; minimum contribution PCC .99985016 |
| `gdn_contract` | 0 / same increasing lengths | fails at511: minimum contribution PCC .99407857; whole output .99996442 |
| `full_long_harness` | 3 / length4097, decode context4098 | selected output PCC .9999797685; traced decode .9999799757 |

Artifact stems refer to `.log` and `.json` in this directory. Old float32
correlation could round to1.0; all new correlation calculations use float64.
Full-attention continuation starts1,32,33,128,511,512,544 passed per-row
contribution PCC, and each dropped-history negative control failed the same
.995 gate. Newly added mixed-row tests are not yet run.

Commands (after environment setup; every device job passed through installed
`$TT_MODEL_BRINGUP_ROOT/scripts/wait_for --cmd ... --file ... --timeout ...`):

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 timeout 600 python -m models.autoports.qwen_qwen3_6_27b.tests.run_decoder --layer 0 --length 33 --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/gdn_chunk_trace.json
TT_METAL_TRACE_ALLOC_TRACKING=1 timeout 600 python -m models.autoports.qwen_qwen3_6_27b.tests.run_decoder --layer 0 --batch 32 --length 33 --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/gdn_batch32_trace.json
TT_METAL_TRACE_ALLOC_TRACKING=1 timeout 600 python -m models.autoports.qwen_qwen3_6_27b.tests.run_decoder --layer 3 --batch 32 --length 33 --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/full_batch32_trace.json
timeout 900 python -m models.autoports.qwen_qwen3_6_27b.tests.run_contract --layer 0 --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/gdn_contract.json
timeout 900 python -m models.autoports.qwen_qwen3_6_27b.tests.run_contract --layer 3 --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/full_contract.json
timeout 900 python -m models.autoports.qwen_qwen3_6_27b.tests.run_long_context --layer 3 --length 4097 --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/full_long_harness.json
timeout 1800 python -m models.autoports.qwen_qwen3_6_27b.tests.run_long_context --layer 3 --length 262143 --output models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/full_context_262143.json
```

The long-context run started16:13 UTC. HF all-token K/V reference completed;
TT later stopped progressing. Read-only triage captured in `triage_long/`: worker
cores idle, prefetch waiting for commands, host waiting in
`SystemMemoryManager::fetch_queue_reserve_back` during `ttnn.add` enqueue.
The summary says pass, but detailed output includes Ethernet NoC counter
mismatches and triage halt-state warnings; it is not watcher-clean evidence.
Live GDB stacks were captured and detached. No reset or termination issued.
AutoTriage/AutoFix source investigation is in progress.

`AUTODEBUG.md` diagnoses precision differences; `AUTOFIX.md` tracks candidates.
The first change preserves folded norm scales in FP32 and awaits device
validation. The CPU inverse discriminator (`inverse_diagnostic/REPORT.md`)
reconstructed the exact length511 activation and found inverse-only minimum
contribution PCC .9999974874; this demotes inverse cancellation for that
input, without asserting TT arithmetic equivalence. The local `.pt` activation
is ignored; the committed script, input hash and JSON reproduce its evidence.

`full_trace_refresh.log` contains a successful unsafe-allocation rejection but
failed because the test expected the wrong error phrase. The test now matches
“device buffer” and “corrupted on replay”; its corrected pass is still pending.
`runtime_audit.json` is the reachable-method AST audit. Runtime sentinels cover
contract prefill and trace capture; final measured prefill now has the same
sentinel. Traced replay contains no Python model dispatch.

Installed `tt-perf-report==1.5.0` with `uv pip install --python
python_env/bin/python tt-perf-report` because this environment has no pip module.
This changed matplotlib3.11.2 to3.10.9. No warmed device-time report exists yet;
JSON wall times are provisional timing observations, not the required
Tracy/performance evidence.

## Triage interference caveat

The first full tt-triage capture occurred before the GDB/FetchQ captures.
Inspection of `tools/triage/triage.py:959` subsequently showed the normal
runner patches Blackhole/Wormhole debug control: it halts cores and suppresses
resume because halt/read/continue can break cores. Therefore the earlier phrase
“read-only triage” describes the intended diagnostic purpose, not absence of
execution-state changes. No standalone private-memory halt/resume helper was
run. Initial evidence before triage was only log silence and CPU/wait-state
observation; it did not prove a dispatch deadlock. The captured host-full /
device-waiting FetchQ state could have been created or perturbed by triage.
Do not attribute an independent runtime root cause from these snapshots.
No manual process termination or operator reset was issued. Await the original
1800-second timeout, then attempt a bounded normal mesh/add/close smoke.

Recovery plan refinement: because triage may have interfered, the first
post-recovery full-context rerun should use baseline execution with harness
progress markers, not infer that chunk drains fix a proven runtime bug. Only
if a pre-triage host/queue capture establishes the same mismatch should the
prepared `--drain-chunks` discriminator be used.

At16:43 UTC the original long-context command exited124 at1800 seconds;
`fuser -v /dev/tenstorrent/*` showed no owners afterward. Normal recovery
probe `TT_METAL_OPERATION_TIMEOUT_SECONDS=20 timeout 60 python -m
models.autoports.qwen_qwen3_6_27b.tests.smoke_device` passed open/add/read/close
in2 seconds (`recovery_smoke.log`). No operator reset was required or run.
Device validation resumes with the isolated FP32 norm repair.

## Final-source short-context validation

The FP32 norm repair passes `gdn_contract_fp32norm` and
`full_contract_fp32norm`. The native creation audit then replaced small host
`ttnn.full` with padded device `ttnn.moreh_full`; the failed tiny-row fill is
retained as `full_contract_device_fill.log`. The corrected
`full_contract_padded_fill` passes all contract cases under Watcher10, and
`full_trace_reuse_padded_fill` passes with one retained trace.

Both `*_final_trace` and `*_final_batch32` pass real-weight prefill/decode,
changed input/position replay, repeated bit equality and the unsafe-allocation
positive control under `TT_METAL_TRACE_ALLOC_TRACKING=1`. Batch32 PCC:
linear prefill .9999631641/decode .9999620187; full prefill
.9999702894/decode .9999571214. `*_mixed_decode` passes independent positions
[33,1,63,32]; `*_watcher` repeats it with Watcher10. `watcher_audit.json`
retains exit status, raw log hashes and compressed logs, and distinguishes
the failed native tiny-fill run. Synthetic real-config pytest:4 passed
(`synthetic_pytest.log`).

The first warmed length512 Tracy collection (`gdn_profile.log`) dropped
profiler markers because buffers filled across repeated passes. Model checks
completed; report merging failed. Added test-only `--profile-drain` to drain
between passes, never inside measured signposts. No decoder change. Rerun
required. The full-context clean retry starts16:56 UTC, timeout3600, layer3
length262143, `TT_METAL_OPERATION_TIMEOUT_SECONDS=180`, trace allocation
tracking1, no Watcher/profiler, output `full_context_262143_retry.json/log`.
Its progress markers advance normally beyond163840 tokens; no triage attached.

## Advertised-context and profiling progress

Linear real-weight full-context checks passed with no capability reduction:
`gdn_context_262143` gives prefill .9999624342, minimum selected contribution
.9996092708, traced decode at262144 .9999860667; `gdn_context_262144`
gives exact-max prefill .9999641191. Both execute the complete TT decoder for
every prompt token; the reduced HF oracle advances all recurrent state and
compares selected output rows. Watcher `position_fill` passes35 exact-value
cases spanning batch1,2,15,16,17,31,32 and positions1,31,33,511,262143.

`gdn_profile_drained` completes warmed Tracy collection and postprocessing.
Human-readable tables and filtered CSVs are in `perf/linear_attention/`;
raw ops are retained as `ops.csv.gz` with uncompressed SHA256 in metrics.json.
Filtered `Device Time [μs]` sums: prefill16007.277μs over1021 device ops,
traced decode2732.206μs over103 device ops. Including reported op gaps gives
19112.559μs and2795.465μs respectively. Batch1, prefill512, decode context513.
The tool is tt-perf-report1.5.0; `--arch blackhole` supplies its P150 nominal
roofline reference, not a measured P300 bandwidth claim. Measured durations
are device timings. The unclassified ReshapeCodegen warning affects category
labels, not raw timings. Tracy's optional WASM-copy warning does not prevent
the ops CSV; the log ends with successful report generation.

The full-attention context gate is still open. The clean baseline completes
but fails numerical accuracy; `long_attention_debug/AUTODEBUG.md` and
`HF_ORACLE.md` separate native reduction precision from a matched HF oracle.
A tail probe with HF-produced K/V reproduces the error for key chunk32;
chunk256 improves decode but leaves contribution rows below .995;
chunk512 passes prefill .9998759178, minimum contribution .9977399977 and
decode .9995686278. Complete all-token validation of that candidate is running.
The diagnostic hook changes only native key-block geometry, preserving full
context, logical output and real weights. A production repair must also align
chunked-prefill starts and reserve finite extra cache pages. No shared runtime
source edits are planned.

Profiler audit correction: the drained run generated reports successfully but
still reports dropped markers within a single prefill pass. Its numbers above
are provisional, not final accepted timings. Native runtime configuration
supports `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` (rtoptions.cpp:1347);
final collection will use5000 plus between-pass drains, and require no dropped
marker warnings. This changes profiler capacity, not model execution or math.
