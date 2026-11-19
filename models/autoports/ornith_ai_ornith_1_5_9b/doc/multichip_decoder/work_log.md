# Multichip decoder work log

Stage 4 in progress. Scope: multichip_decoder.py, tests and documentation only.
Baseline stage 3 is completed and independently clean-pass. Initial working tree
was clean. Never push; stage-owned local checkpoint will follow passing review.

Selected skills: multichip and tt-device-usage 0.1.23, dependency tt-autodebug
0.1.8 installed and enabled (codex plugin list). Environment uses the installed
plugin roots and repo python_env; no workflow files copied into production code.
All measurements serialized across the host; no device assignment to child agents.

## Initial planning and setup

- `fuser /dev/tenstorrent/*`: no owners. `timeout 60 tt-smi -ls --local`:
  IDs 0–3, P300c Blackhole. Persistent repo and Codex home under ornith-9b.
- `mesh_smoke.log`: FABRIC_1D, 1x4 open/close success, 11x10 grid.
- `mesh_plan.md` written before implementation: TP4, local heads and cache,
  BFP8 projections, dense MLP, topology comparison and full-stack capacity budget.
- Read llms.md §3.3, optimized baseline and completed-stage README/context contract,
  GPT-OSS config/CCL manager, common TT_CCL and Kolibri distributed-norm/CCL paths.
  Older TTTv2 modules in /home/ttuser/qwen38/tt-metal/models/common/modules were
  inspected for attention/MLP topology contracts (inspection only).
- Applying optimize for topology/performance and tt-enable-tracing for tracked
  capture/replay evidence; stage 4 is not a separate stage-5 completion claim.
- Hooks: hooks_initial.log / hooks_probe.log. Initial formatting fixed automatically.

## Commands and observations

All device commands use bounded timeout, logs retained in this directory.

`timeout 120 python` mesh smoke: mesh_smoke.log, passed.
`timeout 600 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe
--baseline --layer 0 --output .../baseline_smoke.json --reference
.../generated/multichip/linear.pt`: baseline_smoke.log, passed.

First TP4 harness attempt (tp4_smoke.log) rejected physical_device_ids=None at
binding validation, before mesh open. Corrected to []; no kernel ran. Retry:
`timeout 600 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe
--layer 0 --output .../tp4_smoke2.json --reference .../generated/multichip/linear.pt`.

## Remaining acceptance work

Implementation bringup, both layer-kind PCC/state coverage, non-aligned tails,
continuation/page maps, cross-kind stack, tracked changing-input trace replay,
long-context runtime evidence, deterministic stress, watcher, topology candidates,
serial latency and tt-perf-report, evidence audit, independent stage-review,
local stage commit with SHA. None are claimed passed without artifacts.

## Initial results and candidate adaptation

Both attention kinds pass contracts_sharded_{0,3}.json: logical lengths
1,2,3,4,31,32,33,1023,1024,1025,2051; 8 distinct traced decode inputs and positions
per case; shuffled page tables; complete local recurrent/KV/history reconstruction.
Tracking enabled with program-cache allocations included. No runtime torch/host
transfer permitted inside forward. Baseline outputs generated separately on chip0.

Serial S1024 initial linear measurements: baseline 0.64374 ms decode;
hidden-sharded 0.49180; replicated 0.47317. Prefill 3.17961 / 2.87029 / 2.71333 ms.
The sharded path retains local residual through distributed norm before gathering,
so this is a whole-layer comparison of coherent layouts.

Fused AG+MM initial candidate rejected sender-axis8/workers4 in host validation
(perf_gather_0.log); adapted workers to8 as required by the chosen grid. This is
not rejection of the topology family. Completed failed run released its devices;
no other owners. Reset required by selected multichip workflow after failed mesh
run: timeout180 tt-smi -r, reset_after_gather_validation.log. No processes killed.

Fused scatter first attempt used the older WormholeComputeKernelConfig binding;
new matmul_reduce_scatter_prefill accepts ComputeConfigDescriptor and negative
scatter_dim only. Adapted to descriptor with unchanged attention HiFi2 / MLP LoFi,
scatter_dim=-1; local M is internally padded to32 and sliced back to logical n.
Reset after clean process exit: reset_after_scatter_binding.log. No process kills.
Fused gather adapted workers8 passed accuracy but measured 1.0127 ms decode and
3.113 ms prefill, slower than both unfused coherent residual-layout controls.

## Geometry sweep hang / investigation in progress

The precision-locked sweep completed132 real-weight configurations; then
mlp.down_proj K3072,N4096,cores32,block1,readers8 stalled in MatmulDeviceOperation309.
Live evidence is triage_geometry/tt-triage.txt and triage-summary.txt. The device
job remains under its original1800-second timeout; it was not manually terminated.
AutoTriage isolated launcher first failed its sandbox preflight because this host
forbids unprivileged namespaces (autotriage_launch.log). Retried with the explicitly
supported AUTODEBUG_SKIP_CHILD_SANDBOX=1 in this already unrestricted authorized
session, retaining inspection-only child instructions (autotriage_retry.log).
Only model-local layout/config repairs are in scope; shared kernels remain untouched.


### AutoFix and stack capacity refinement (2026-11-12 19:43 UTC)

The isolated AutoTriage report completed in `AUTOTRIAGE.md`. Its current-source
factory/kernel chain corrects the initial informal guess: the stall is down
K3072, requested64 activation/output cores, block1, readers1; all32-core cases,
including block3/readers8, completed. `geometry_0_before_repair.json` preserves
those132 rows. AutoFix read-only agents independently checked exact48 input cores
and genuinely padded K4096/64 cores. No agent edited source or used devices.
The main process applied model-local `exact_k_shards`, kept output geometry
independent, and made the sweep resume/log before dispatch. Host regression:
`python models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_planner.py`
→ EXACT_K_PLANNER_PASS. Shared kernels remain unchanged. The original device
job is still waiting for its preexisting1800-second timeout; no signal/reset
has been sent. Device validation of the repair remains pending.

Host capacity review found persistent AR scratch must be shared across layers:
32 private copies would require2 MiB/core, above physical1.5 MiB/core. Added
`MultichipCollectives`, explicit `collectives=` constructor argument, and shared
workspace identity assertions in the cross-kind stack test. Total persistent
workspace is2 MiB/chip (64 KiB/core). Final stack execution will validate sharing.
Also expanded DRAM reserve from2 to8 GiB: full-context input+chunk outputs+concat
already require6 GiB at peak. Analytic total16.281 GiB still fits31.875 GiB and
retains262144 context. Context JSON and memory plan updated; no full-model code.

Final validation now checks replicated outputs on all four ranks, in addition to
local cache/state reconstruction. Candidate comparisons are consolidated in
`topology_comparison.csv`; they precede final geometry/workspace integration.

Auxiliary-weight accounting further includes tile-height padding on singleton norm
and convolution-tap tensors: conservative2 MiB/layer,64 MiB total/chip. Updated
analytic total16.340 GiB/chip (still below31.875 GiB); no capability change.

### Geometry timeout and recovery

At19:54:51 UTC the original1800-second geometry command ended with exit124.
No manual signal was sent. `fuser /dev/tenstorrent/{0,1,2,3}` showed no owners.
A fresh60-second `tests.multichip_mesh_smoke` initialized FABRIC_1D but did not
complete the trivial add; it too ended124 (`post_timeout_smoke.log`). No other
hardware work proceeded. `tt-smi -ls --local` then completed0; authorized bounded
`timeout180 tt-smi -r` is running. Pre-reset listing and reset logs are
`geometry_pre_reset_list.log` / `geometry_reset.log`.

Reset completed0; post-reset listing showed all four chips.
`geometry_recovered_smoke.log` passed, then the focused exact-K repair probe
passed both block1/2 cases with minimum all-chip PCC>0.9999999 and warmed trace.
No second reset required. Resumed600-second-per-layer geometry sweep is running.

Both resumed sweeps passed156 configurations each. Selected per-role defaults:
linear packed input32cores/block2/readers2; full packed input64/2/2; gate/up64/2/2;
down/output64/1/4. Exact input storage for down is48cores and output projection32.
Precision remainsBFP8/HiFi2 attention, BFP8/LoFi MLP. The down/output gains are
sub-percent; packed/full and gate/up improve materially. geometry_comparison.csv
contains all312 cases and geometry_source.py.gz preserves pre-integration source.
Final integrated source now awaits the complete acceptance and profiler battery.

Final baseline0 passed; first multichip final fixture failed before layer
construction because a metadata insertion also entered constructor kwargs and
referenced `layer` before assignment. Fixed the test-only insertion. Exact raw
failure: final_contracts_0_harness_error.log. Devices closed cleanly; following
multichip skill, resetting before retry. Decoder source unchanged.

### Native local layout and watcher AutoFix

`perf_native_{0,3}.json` showed local L1 sharded residual/norm/MLP execution improves
decode to0.3243/0.3047 ms, so integrated that optimized-baseline layout with the
shared direct-AR workspace. `pre_native/` preserves superseded final0 gate evidence.
Watcher then exposed the resident fused-prefill NoC accounting defect; see
`watcher_triage/AUTOTRIAGE.md` and `watcher_triage/AUTOFIX.md`, including original
failure artifacts, recovery, host planner enumeration and measured alternatives.
Default prefill now uses separate MM+RS+AG; optional fusion is constrained to64 KiB.
No shared source changed. Applied profiler's L1 input advice and measured the
whole layer:2.6894/2.7355 ms prefill with unchanged decode. Explicit minimal-matmul
M4/K16/N8 config remains; the report's generic no-program-config advice does not
recognize that operation's `config` field. Diagnostic native fused reports are
archived under `diagnostic_native_profiles/` with their exact source snapshot.
Final integrated source is being revalidated from the full baseline/TP4 battery.

## Final two-link acceptance (2026-11-12)

The final source SHA256 is `bc8386dc4e5c8f6fb18f24addca003e9cd37efe2b57341e6be9b06948aef88fc`. All four chips were used
serially; no competing device jobs. `ttnn.get_num_links(mesh,1)` returned2.
`perf_available_links_{0,3}.json` measured2-link prefill;
`perf_two_link_reduce_{0,3}.json` measured2-link prefill and direct decode AR.
Both improved the whole layer with PCC passing, so final CCLManager and AR use2.
`pre_two_links/` preserves the preceding fully passing one-link suite, profile
tables and implementation snapshot. No baseline precision or model capability
changed. The baseline source remains SHA256
`b678803e78b74ede66d61e7609c6ca9147b2977dbba702f8431eb621ef997c4e`.

Final run command (from checkout, embedded600/900-second job timeouts):

```bash
R=models/autoports/ornith_ai_ornith_1_5_9b
bash "$R/tests/run_multichip_final.sh" > "$R/doc/multichip_decoder/final_queue.log" 2>&1
python "$R/tests/summarize_multichip.py"
python "$R/tests/audit_multichip.py"
python "$R/tests/test_multichip_planner.py"
```

All returned success: `MULTICHIP_FINAL_PASS`, `PERF_SUMMARY_PASS`,
`MULTICHIP_AUDIT_PASS`, `EXACT_K_PLANNER_PASS`. `pre_device_hooks.log` records
format/lint checks before final measurement. Final artifacts are tied to the
source SHA above. `final_contracts_{0,3}` cover all11 logical lengths, all4 rank
contributions,8 evolving decode steps/length and local state reconstruction.
Each signature was captured once and replayed across3 restored fresh requests,
33 refreshes/kind, with tracking1 and program-cache skip0. Positive allocation
controls rejected deliberately unsafe lifetime allocation. `final_stress_{0,3}`
cover128 evolving steps; `final_long_{0,3}` execute complete262139/262143/262144
prefill, first/middle/end comparisons and3 bit-identical terminal-position replays.
`final_stack_{1,32}` covers layer0→3, shared workspace, distinct rows, ragged
prefill/unaligned continuation/shuffled pages/4 distinct replay inputs.
`watcher_{0,3}` pass separately with watcher10/NOINLINE1.
`final_audit.json` minimum PCC across every recorded final comparison is
0.99970448, above0.995. No dynamic torch/host transfer or static
runtime fallback found.

`performance.csv` records single-chip and TP4 latency, speedup and efficiency.
Final prefill linear/full:2.1632/2.2194 ms versus3.1796/3.6146 baseline
(1.470×/1.629×;36.75%/40.72% efficiency). Decode:.30786/.28846 ms versus
.64374/.62408 (2.091×/2.164×;52.28%/54.09%). Batch1 two-layer stack:
1.24764→.57825 ms; batch32 row-serial stack:39.77561→18.63515 ms.
All timings are warmed serial measurements, without profiler/watcher/tracking.
Decode medians use7 groups of50 single-layer replays or20 stack replays.
The earlier `perf_baseline_{0,3}` reference is the same unchanged optimized source,
precision and serial regime; latest `baseline_stack_perf_{1,32}` controls were
rerun in the final queue. No whole-model throughput claim.

Final `tracy/{0,3}/` has advice-enabled human-readable tables, CSVs and provenance
for both phases. `perf_summary.json` verifies actual BF16/BFP8 HiFi2/LoFi runtime
rows. Two links cut summed profiled prefill CCL from1213–1228 to661–664 µs and
decode CCL from44–45 to28 µs. Decode matmuls report72–85% DRAM utilization;
remaining data movement is55.3/52.0 µs. Prefill uses L1 input placement, explicit
M4/K16/N8 minimal-matmul config. Generic no-program-config advice is a report
recognition limitation. Above100% compute in DRAM-reader reports is the report's
reader-core denominator, not actual utilization. Full1024-token direct AR
needs4 MiB worker scratch even at2 links, above1.5 MiB L1 by exact source
allocation; no unsupported runtime test was launched. Smaller chunks remain an
unmeasured candidate. The measured native two-link final path reproduces the
selected candidate.

Capacity remains262144. Analytic full-stack DRAM budget is17,544,937,472 bytes
(16.340 GiB) against34,225,520,640 usable, plus separately accounted shared2 MiB
L1 workspace. Full-context layer-kind execution and two-layer stack are tested;
all32 layers were not loaded in this decoder-only stage. Batch32 uses independent
short caches; maximum context is per request.

The host-only watcher-plan investigation used:

```bash
R=models/autoports/ornith_ai_ornith_1_5_9b
g++ -std=c++20 -I. "$R/tests/multichip_prefill_plan.cpp" build_Release/lib/_ttnncpp.so -Wl,-rpath,/home/ttuser/ornith-9b/tt-metal/build_Release/lib -o "$R/generated/multichip/prefill_plan"
timeout 120 "$R/generated/multichip/prefill_plan" > "$R/doc/multichip_decoder/watcher_triage/planner_sweep.csv" 2> "$R/doc/multichip_decoder/watcher_triage/planner_sweep.log"
```

640 planner cases, both local K values and Mtiles1..32, prove64/96 KiB avoid the
resident split-NoC plan. Device constrained candidates and recovery commands are
in watcher_triage/AUTOFIX.md. Both final watcher gates now pass without suppression.

Evidence packaging: `early_evidence_archives.json` and `evidence_archives.json`
map original paths to SHA256, size and lossless gzip parts. All raw files remain
local. Large device-event CSV/Tracy binary captures remain local with hashes;
relevant operation CSVs, logs and all compact human-readable/CSV reports are
committed or losslessly archived. Concatenate listed parts, gzip-decompress to
the original path and verify SHA256 to restore an archive. Generated checkpoint
and reference tensors stay in ignored generated/multichip/.

Independent review and local checkpoint SHAs will be recorded below. Never push.

Final staged hooks normalized trailing whitespace/endings in generated text/YAML
and clang-formatted the host-only planner probe. Pre-normalization exact bytes are
losslessly preserved by report_whitespace_archives.json. No numerical data or
decoder source changed. The host probe was rebuilt and its CSV compared byte-for-
byte to the original planner sweep after formatting.

Independent review identified a documentation-only aggregation-label error.
The installed tt-perf-report uses maximum device duration for noncollectives
and mean device duration for AllGather/ReduceScatter/AllReduce. Corrected README
and summarize_multichip.py method wording, then regenerated perf_summary.json.
Underlying CSV/numbers and implementation are unchanged; no device rerun needed.

## Independent review and checkpoint

Fresh xhigh read-only reviewer `/root/stage_review_round1` returned **clean-pass**
in round1, with no required work. Exact report: `stage_review_round1.md`.
The profiler aggregation wording was fixed and inspected during that round;
no implementation or numerical change followed final measurement. Repository
hooks passed after the correction. Archive roundtrip verified203 raw evidence
files byte-for-byte; compact operation CSVs and all raw logs are preserved.
`git -c core.whitespace=cr-at-eol diff --cached --check` passed (CSV exports retain
their standard CRLF line endings). All staged files are within the target model.

### Next-stage interface notes

These are documented scope boundaries, not unresolved required work: construct
one shared MultichipCollectives for the full serial stack; validate generator-wide
multiple live trace signatures and prefill interleaving when that code exists;
invalidate runner reuse artifacts when test arguments/reference data change, since
the existing convenience skip keys only on implementation SHA. Full-layer-stack
accuracy/allocator peak and vectorized batching belong to subsequent stages.
Smaller-chunk direct prefill AR is an unmeasured optimization candidate.

The stage creates local checkpoints on `tt-metal/main`; no other repo was modified
and no push is performed. Implementation/review checkpoint SHA is recorded in the
following provenance entry after commit creation.

Implementation, tests, evidence and clean-pass review checkpoint:
- Repository: `/home/ttuser/ornith-9b/tt-metal`
- Branch: `main`
- Local commit: `f6b1e5faee57826c673eeddddeeb399be818b6ce`
- Push: never performed.

This provenance-only follow-up records the already-created checkpoint.
