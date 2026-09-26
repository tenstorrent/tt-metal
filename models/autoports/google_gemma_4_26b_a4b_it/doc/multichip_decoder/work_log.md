# Stage 04 work log

Model google/gemma-4-26B-A4B-it, revision4d7ae4984b7db7de8f8457170b3f1a419ee76d52.
Starting clean checkout2e3a1779d3, optimized implementation checkpointa9259624f2.
Only multichip decoder, stage tests/docs and context contract changed. No push.

## Startup and topology

Enabled tt-model-bringup/tt-autodebug plugin inventory verified in selected
Codex home's config. Packaged scripts/environment.py returned valid exports.
`timeout 60 tt-smi -ls --local`: exit0, four Blackhole P300c ASICs.
`set_fabric_config(FABRIC_1D); open_mesh_device(MeshShape(1,4)); close_mesh_device`:
exit0, MESH_SMOKE_OK. Unknown B850M-C motherboard warning falls back to bus IDs;
auto-discovery reported matching physical/logical degree histograms and all4
opened. No failure is inferred from that warning.
Read llms.md3.3, common Attention1D/MLP1D/RMSNorm1D, GPT-OSS CCL and sparse-expert
contracts, and the optimized decoder and evidence. Initial plan: mesh_plan.md.

## Initial experiments

Commands run as `HF_HUB_OFFLINE=1 timeout 240 python -m
models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder` plus flags
below. Reports/logs are named by output stem in this directory.

- probe_multichip_collective: BF16/FP32 reduce-scatter [1,1,32,2816] exact4.
- paired_initial: setup/runtime API mismatch, shared projection is callable;
  minimal fix calls it. Reset all devices after closed failed run:
  `timeout 180 tt-smi -r` exit0 (reset_initial.log), bounded list exit0
  (list_after_reset.log), next run opens full fabric successfully. No process
  killed and no locks cleared.
- paired_second: default layer0,length65,steps2; prefill.9996149855;
  decode.9998534455/.9999265616. Device-only guard passes.
- full_trace: --layer5 --trace --steps8 (length65). All replicas exact;
  deterministic repeated traces with refreshed positions; minimum PCC.9986253045.
- full_sharded_trace: same plus --sharded-residual. End-to-end sharded
  residual consumed by distributed norms/residuals and gathered only for input
  projections and comparison boundary. Minimum PCC.9984544464. Traced host
  latency about1.03ms versus.91ms replicated in these short controls; not a
  headline or final optimization claim.

The required4096/128 workload and final optimization/validation are in progress.
No clean-pass review or stage completion commit exists yet. Native profiler,
maximum context, cache ownership, batch, stack and watcher gates remain open.

## Geometry and topology experiments

`run_multichip_decoder --layer 0 --length 4096 --steps 128 --trace`
produced sliding_headline_v0/v1/v2.json. V1 uses 12 gate/up and 88 down
cores with prefill K-block 11 instead of 1. V2 uses explicit QKV and WO
projection configs. Sliding v2 minimum PCC .999855352, warmed host medians
224246 us prefill / 864.36 us decode, versus paired TP1 221219 / 824.99 us.
Full attention v1 (`--layer 5`) minimum PCC .99778229, TP4 host medians
191667 / 911.81 us, TP1 186599 / 876.54 us. Host times are not device metrics.

Native v0 profile collected with `python -m tracy -r -p -v
--op-support-count 100000 --no-op-info-cache
--disable-device-data-dump-to-files --disable-device-data-push-to-tracy
-o <stage>/profile_v0 -n tp4 -m <tests>.run_multichip_decoder
--tp 4 --layer 0 --length 4096 --trace --steps 1 --profile`.
Final CSV and tt-perf-report tables are under profile_v0; this one-decode
profile is diagnostic, not the required target measurement. AUTOTRIAGE.md
refutes the apparent shutdown hang: the run exited successfully without
intervention.

AutoDebug proposed EP4 to avoid narrow TP expert projections. The component
experiment owns 32 complete experts per rank and uses runtime sparsity counts;
fixed nnz=8 would be invalid for partitions with 0..8 selected experts.
Component correctness/trace/zero partitions passed, but the all-check watcher
run asserted in fabric Ethernet teardown after EP_PROBE_PASS. This is not
watcher-clean evidence. AUTOFIX_ep.md records recovery and scoped experiments;
AUTOTRIAGE_watcher.md is being prepared independently. No C++ changes.

## Attempt stop after scoped AutoFix failure

Paired EP runs used `--expert-parallel --length 4096 --steps 128 --trace
--check-cache`, layer 0 and 5, output sliding_ep_v2.json/full_ep_v2.json.
Both exited 0; per-layer host timing/speedup/efficiency and accuracy are in
candidate_summary.json. No final path is accepted.

CCL-only all-check Watcher control exited 134 after both reductions passed;
normal two-ERISC and supported single-ERISC modes both failed teardown.
Commands and distinctions are recorded in AUTOFIX_watcher.md. After each
failed process had exited, `timeout 180 tt-smi -r` and bounded list exited 0.
Final recovery files: ccl_single_erisc_reset.log, final_device_list.log,
final_mesh_smoke.log (FABRIC_1D1x4 open/close exit 0). No processes were killed
or locks removed in these controls. AutoFix found no supported scoped remedy;
C++ fabric repair requires a separately authorized scope.

An untested fused-tail edit was reverted. Current runtime is only formatting
different from the measured v2 EP snapshot; source_provenance.json confirms
AST equivalence except top-level import ordering. All 8 stage Python files pass py_compile and pre-commit
(black/autoflake/isort and other applicable hooks); precommit.log is retained.
No build needed. Prepared stack/batch/fused probes are explicitly unrun.
Telemetry JSON was produced locally from the exact supplied template.
Stage remains incomplete; no clean-pass or completion commit is claimed.

Final staged pre-commit run passed after the whitespace hooks normalized two
generated text reports. Preserved precommit_staged.log.gz records the successful
rerun. The local checkpoint is an incomplete/blocked attempt, not a stage-pass
commit. No push is authorized or performed.

## Local checkpoint

Repository tt-metal, branch gemma-4-26b-a4b-it:
`41a8c80072c3ce2f0480cd4bbfbefeccfedc8ddc` — incomplete/blocked attempt
checkpoint, not stage completion. Commit hooks passed. Identity was supplied
per command as Codex <codex@openai.com>, matching prior stage commits; no
global/repository identity configuration changed. Nothing pushed.

Independent fresh xhigh stage-review returned **more-work-needed**
(stage_review.md): fabric Watcher gate, final capability/context/batch/stack
validation, topology/geometry/default selection, and target native profiling.
The review verified numerical summaries, diagnostic windows, memory arithmetic
and provenance. No additional arithmetic defect was demonstrated. These gates
remain work after the separately scoped fabric repair; AutoFix's failed scoped
workaround is the stopping condition for this attempt, not the review verdict.
