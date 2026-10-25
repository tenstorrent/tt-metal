# Stage08 datatype sweep

Started from clean exp26-qwen38-bringup checkout, stage07 checkpoint
 db6a0a26defe9b41dcf333fa69f2ae662195ff5f and subsequent receipt commit.
No vLLM integration is authorized or started. Enabled plugin inventory is
.codex/config.toml; model-bringup 0.1.21 and tt-autodebug 0.1.8 environments
are set by tests/optimized_multichip_env.sh in qwen38-functional-build.

Acceptance: full-model top1 >=90%, top5 >=98%, top100 =100%; main AIME24
chat-template reference has100 generated positions. Native262144 capability,
32x32768 agent working set, batches1/8/16/32 remain the contract.
Candidate ranking requires traced teacher forcing. Final headline is separately
measured warmed autoregressive4096/128/B1/concurrency1, never teacher forcing.

Startup: docker process inspection found no live model jobs. Bounded
`docker exec qwen38-functional-build timeout 60 tt-smi -ls --local` reports
four p300c Blackhole ASICs. No reset needed. Baseline run opens the1x4 ring mesh.
Exact baseline command:
`docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m models.autoports.qwen_qwen3_8_27b.tests.run_full_validation --output-dir models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/baseline`
Output: baseline/run.log, prefill/teacher-forcing accuracy and performance,
shared qualitative suite, autoregressive AIME control and provenance.

Inspection: inherited decoder already uses BFP4/LoFi attention/gate-up/down,
BFP8 full-attention KV and most CCL, BF16 residual, FP32 recurrent state.
Fused prefill fidelity is hardcoded and needs explicit policy plumbing before
fidelity comparisons. Native batched fused CCL uses BF16 and must be documented
separately from B1 BFP8 CCL. No stage08 precision has been selected yet.

Before runtime edits, `tests/plot_precision_baseline.py` produced
baseline_run_view.png from stage07 benchmark JSON. It shows warmed host TTFT and
decode phases plus allocator checkpoints; it is not a device-activity trace.
Mechanism hypotheses: BFP8 head weights can reduce the remaining BF16 weight read;
BFP8 decode activations can reduce movement but add conversion; BFP4 KV can reduce
cache bandwidth/capacity cost but may lose accuracy. BFP8/HiFi2 canonical controls
and BFP4/HiFi2 establish fidelity cost and agreement on all projection families.
The inherited BFP4/LoFi full-model baseline covers attention, gate/up and down.
Expected winners must preserve the long-context/memory gates; prediction alone is
not a rejection or selection. Host matplotlib is absent; plotting uses the
existing container matplotlib3.10.9 with persistent MPLCONFIGDIR.

Precision contract distinguishes ordinary projection input activations, prefill
column inputs, fixed BF16 fused-row inputs, B1 CCL, fixed BFP8 prefill CCL and
fixed BF16 native batched CCL. Fixed representations are validated constraints;
unsupported values raise rather than being silently ignored. Recurrent state
remains FP32, residual/norm/sampling BF16. Candidate configs are not selections.

Baseline refresh exited0. Both phases99/100 top1,100/100 top5/top100.
Teacher-forcing runner decode35.5569769t/s/user, TTFT131.244735ms; generator
metrics include99 model replays and no new captures. This is203/100 teacher
forcing, not headline4096/128 autoregression. Shared qualitative outputs are
exact token-for-token matches to stage07 for all three prompts; control receipts
in baseline/qualitative_controls.json point to same-format HF outputs. Existing
128-token truncated explain/code controls remain unchanged; structured JSON finishes.
Using qualitative-check for the candidate/selected shared suite and long outputs.

Runtime plumbing added after baseline exited: precision.py, model.py,
generator.py, multichip_decoder.py. Default remains the inherited safe policy
until selected_precision_config.json exists. Explicit config or
QWEN38_PRECISION_CONFIG provides candidate/safe-baseline override. The constructor
validates fixed buffer/norm/sampler/context invariants. Head dtype/fidelity and
ordinary/fused projection fidelities are now explicit runtime inputs.
Python compile check passed. Reproducing baseline through this construction path
with tests/run_datatype_candidate.py; all64layers, native-size cache,3 warmed
traced teacher-forcing repeats, runtime tensor/policy inventory.

Policy-loader baseline finished successfully: top1=.99,top5=1,top100=1;
three-run median teacher-forcing decode35.4957801t/s/u, TTFT118.675430ms.
results/baseline_bfp4_lofi.json records all64 runtime tensor dtypes, resolved
per-layer policies, trace IDs/counters and per-repeat accuracy. This reproduces
the pre-plumbing baseline within observed timing variation. No candidate is yet
selection-eligible.

Sequential matrix command (inside container):
`bash models/autoports/qwen_qwen3_8_27b/tests/run_datatype_matrix.sh bfp4_hifi2 bfp8_lofi bfp8_hifi2 decode_ccl_bf16 decode_activation_bfp8 head_bfp8_lofi head_bfp8_hifi2`
The launcher aborts on runtime failure; accuracy failures are retained as rows.
KV candidates require a real-weight decoder smoke before their full-model run.

Baseline and the first HiFi2 run record the resolved runtime policy plus source
hashes; their compute-config Python repr contains only an address. The runner now
serializes actual math_fidelity/fp32 attributes and aggregates actual linear/fused
kernel input arguments during untimed preparation. Wrappers are removed before
accuracy/timing runs. This improves evidence without changing runtime kernels.
Final selected-policy verification must include these concrete kernel arguments.

Agent long-context fixture: agent_long_cases.json, exact32769-token chat history,
original deployment facts in the first user turn, intervening investigation log,
and a short final retrieval request. Expected JSON is project=Cedar,port=7319,
owner=Mira. tests/make_precision_long_case.py records rendered IDs and hashes.
This exercises an awkward1-token chunk remainder. Same fixture will be evaluated
with HF and selected full-model controls; no quality verdict exists yet.

First fidelity control: bfp4_hifi2 passes both phases99%/100%/100%, but median
traced teacher-forcing29.5099673t/s/u vs35.4957801 for LoFi. Reject HiFi2 as slower
with no measured accuracy benefit. BFP8 LoFi/HiFi2 controls are next. Candidate
comparison uses identical203/100/B1 readiness work and native262144 allocated cache.

kv_memory_forecast.json records physical tiled KV sizes: BFP4=4608,
BFP8=8704,BF16=16384bytes/token/device for16 full-attention layers. Capacity is
34,178,731,008bytes/device. Forecast uses measured stage07 B32 peak plus KV delta;
it is not an allocator measurement or proof of a physical limit, and changes no
capability. Only an actual selected configuration will update context_contract.json.

BFP8/LoFi first attempt fails during preparation's initial linear-attention mix
matmul. Actual captured args: BF16 input[1,1,1,5120],BFP8 weight[5120,4608],LoFi,
FP32 destination=false. Inherited BFP4 mix geometry uses8cores,K-block20,reader1.
Static CB region ends1,405,568bytes versus live L1 buffer at1,212,032; failure
occurs before any full-model accuracy/performance measurement. Preserved failure
JSON/log under failures/bfp8_lofi_initial.*. Devices close normally; no hang/reset.
This does not reject BFP8 or count as a Pareto point.

AutoDebug launcher sandbox preflight failed because unprivileged user namespaces
are unavailable. Retried with explicit AUTODEBUG_SKIP_CHILD_SANDBOX=1 under the
existing danger-full-access authorization. Child is inspection-only; exact prompt
and diagnostics are in autodebug_launch.log/autodebug_retry.log. AutoFix waits for
its diagnosis before source changes. Independent head/CCL/activation candidates
continue serially on hardware while source inspection runs separately.

Both BFP8 head candidates pass full-model gates: LoFi36.4197320t/s/u,HiFi2
36.3809209t/s/u; all top1=.99/top5=1/top100=1. Their0.107% difference overlaps
repeat variation, so there is no demonstrated fidelity-speed separation.
Both improve over baseline by about2.5%; long-context/memory/token-out gates pending.

AutoFix narrowed two L1 failures and passed reduced real-weight validation; see
AUTODEBUG.md,AUTOFIX.md,geometry_alternative_review.md and geometry_*.json.
BFP8 canonical and BF16 CCL full-model attempts are now retried after adaptation,
not rejected at the first runtime error. The repaired matrix command is
`docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/run_datatype_matrix.sh bfp8_lofi bfp8_hifi2 decode_ccl_bf16 decode_activation_bfp8 kv_bfp4 kv_bf16`.

KV smoke command: `docker exec qwen38-functional-build bash
models/autoports/qwen_qwen3_8_27b/tests/smoke_precision_kv.sh`.
Both BF16 and BFP4 K/V run real layer3 paged fill/update at nonaligned33 and4097
with exact trace replay. These smoke checks intentionally have no full-model
accuracy claim; full-model AIME evaluation follows.

Constructor validation additionally rejects an empty explicit dict, unsupported
fixed residual dtype, integer layer-exception keys and invalid fidelity names.
This prevents falsy overrides or non-JSON keys from being silently ignored.
Host checks passed without device access. Final default selected artifact is still
absent; ordinary model construction continues to use the safe inherited baseline.

Adaptive search addition, before measurement: BFP8 head saves about2.5% traced
full-model time while retaining the gate. Try BFP4 head with both LoFi and HiFi2
as the next plausible bandwidth reduction. Prediction: another small decode gain
but higher accuracy/long-generation risk. These two configs are planned, not
accepted. This makes the seriously considered BFP4 head group subject to the
same required LoFi/HiFi2 evidence as decoder projection groups. Keep BF16
embedding/norm/auxiliary boundaries and FP32 recurrence unchanged.

BFP8/HiFi2 full64 also passes99%/100%/100% but measures26.5816364t/s/u,
compared with27.0105877 for BFP8/LoFi and35.4957801 for BFP4/LoFi. The repaired
BFP8 control therefore establishes both legal same-dtype fidelity choices and a
measured performance rejection; it is not dismissed on synthetic PCC or an API
failure. All model warmup signatures complete with the dtype-aware geometry.

BF16 CCL targeted repair passed the four-layer real-owner probe, including all
prefill buckets and exact repeated nonaligned requests. Resumed serialized full
matrix via `docker exec qwen38-functional-build bash
models/autoports/qwen_qwen3_8_27b/tests/run_datatype_matrix.sh decode_ccl_bf16
decode_activation_bfp8 kv_bfp4 kv_bf16 head_bfp4_lofi head_bfp4_hifi2`.

Full64 repaired BF16 CCL passes99%/100%/100%, three traced repeats median
34.1809491t/s/u, TTFT120.675606ms. It is slower than inherited BFP8 CCL baseline
35.4957801t/s/u; rejected on speed after successful repair, not rejected because
of its initial runtime errors. Matrix continues with ordinary activation BFP8.

Ordinary activation BFP8 passes98%/100%/100% at35.0329144 traced t/s/u,
TTFT122.966537ms. It is slower and loses one top1 agreement position relative
to baseline; reject this independent change and do not add it to the prospective
combined policy. BF16 CCL likewise has no measured speed win. Remaining likely
combinations concern head and KV-cache precision.

CPU-only HF32769 multi-turn retrieval control launched at approximately
2026-10-25 06:13 UTC while TT BFP4-KV is still preparing early buckets (416/4096).
Command: `docker exec qwen38-functional-build bash
models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m
models.autoports.qwen_qwen3_8_27b.tests.run_long_quality_cases --hf --cases
models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/agent_long_cases.json --output
models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/agent_long_hf.json
--max-new-tokens 128`. This process never imports/opens TT hardware. Check and
record its completion before accepting subsequent TT timed runs; overlapping
performance measurements must be repeated without this CPU workload.

The BFP4-KV preparation delay was investigated: `zeros_like` falls back to host
full construction for BF4, and reset repeats this over the native pool. Stopped
the still-progressing candidate with SIGINT after preserving evidence; the
traceback lands in reset/zeros_like and mesh closes normally. No measured point
exists for this attempt. AutoFix source review and repair probes are in
kv_reset_review.md, kv_reset_autofix.md and bfp4_reset_probe.json. The narrow
BF4-only in-place device fill preserves cache addresses; all-rank exact-zero
primitive proof passes for one-page and native1024-page tensors. Real-layer
populated-cache reset/replay validation is running before full64 retry.

CPU HF32769 control completed successfully with exact JSON facts and EOS:
project Cedar, port7319, owner Mira. agent_long_hf.json records identical tokenizer
chat-template/input hash and explicit thinking=false. cpu_control_finished.json
was written atomically only after docker exec exit0 was confirmed. No completed
TF ranking measurement overlapped this CPU run: the only full candidate during
it was stopped in untimed preparation. Subsequent candidate runner supports an
explicit --wait-for-control receipt before accuracy/timing as an orchestration
safeguard. No CPU-control timing is a model-performance headline.

Reduced BF4 reset/replay passes all prepared buckets and five boundary cases,
including exact all-rank zero reset and unchanged cache/trace addresses. Resumed
full matrix: `docker exec -e QWEN38_CPU_CONTROL_RECEIPT=models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/cpu_control_finished.json
qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/run_datatype_matrix.sh
kv_bfp4 kv_bf16 head_bfp4_lofi head_bfp4_hifi2`. CPU HF has exited; the receipt
is also explicitly recorded before each candidate's measured runs.

Source chronology: geometry_alternative_review.md's current-hash match was checked
before the later BF4-only Generator.reset repair. The geometry source itself is
unchanged since the passing BFP8/BF16-CCL runs. Those controls use BF8 KV, so the
new BF4 conditional does not alter their executed reset path. Repaired BF4 KV
and subsequent candidates record the new generator source hash directly.

Repaired full64 KV BFP4 passes97%/100%/100%, median35.5352713 traced t/s/u,
TTFT122.411798ms. The approximately0.11% throughput difference versus baseline
is not a clear speed win; it loses two additional top1 agreement positions while
reducing KV bytes. Keep it as a prospective combined head/cache experiment,
not an automatic winner. Full64 completion closes the BF4-reset AutoFix defect.
BF16 KV control is now running, followed by both BFP4 head fidelities.

BF16 KV fails initial full64/native262144 SDPA decode preparation: static circular
buffer endpoint1083648 exceeds live L1 frontier1065344. Preserved
failures/kv_bf16_initial.json/.log. Small-context layer3 smoke was not sufficient
to expose this native page-table/config budget. Started fresh installed AutoDebug
launcher with bounded model/SDPA source focus; explicit child sandbox skip reuses
the already-diagnosed namespace-preflight exception. No target/hardware use by
child. No reset required; failed process closed mesh normally.

Exact-shape BF4 reset follow-up: original primitive used equal total volume but
1024x256 rows; production cache uses8192x32 rows. Reran probe_bfp4_reset.py with
actual production shape and explicit rank count4. Both one-page/native shapes
pass exact zero and stable addresses. Real generator/full64 tests always used
production shape, so their results did not depend on that primitive simplification.
Current bfp4_reset_probe.json records the corrected exact-shape proof.

While source-only AutoDebug runs, resumed serialized head comparisons via
run_datatype_matrix.sh head_bfp4_lofi head_bfp4_hifi2 with the completed CPU
control receipt. These use validated BF8 cache and do not depend on the BF16 fix.

BFP4/LoFi vocabulary head with inherited BF8 KV passes100%/100%/100% on the
one-problem readiness reference, median36.9184859 traced t/s/u,
TTFT118.365962ms. This is the fastest completed candidate so far, about4.0% over
baseline. Higher agreement on this small reference does not establish general
accuracy superiority. HiFi2 head comparison and combined head/KV trial remain.

Before combined measurements, add the two BFP4-head fidelities crossed with BFP4
KV: head_bfp4_lofi_kv_bfp4 and head_bfp4_hifi2_kv_bfp4. Together with the two BF8-KV
head trials, this is a small matched comparison of head fidelity and cache dtype.
It tests the plausible combined win and resolves the near-tie in cache-only
throughput without assuming independent numerical/performance effects add.

BF16 KV repair closes with a successful full64 native-context run:98%/100%/100%,
roughly35.41 traced t/s/u. The BF8-cache baseline is faster, has higher agreement
on this reference, and uses less KV storage. Reject BF16 KV on that measured
comparison; do not treat its earlier runtime failure or DRAM forecast as a
quality verdict or proven physical capacity limit. kv_bf16_autofix.md records
exact metrics and the native/batched controls. Matrix now tests both head/KV4
combinations. The BFP4 head HiFi2 control measured36.6369124t/s/u with100%/100%/100%,
versus36.9184859 for LoFi with the same BF8 cache.

Combined BFP4/LoFi head + BFP4 KV passes the gate but is dominated by head-only:
TF97%/100%/100%, prefill95%/100%/100%, median36.8162901t/s/u,
TTFT121.303796ms versus head-only100% throughout,36.9184859t/s/u. The combined
repeat range36.80527–36.85792 is below head-only36.85973–36.94936. Reject this
combined cache change on measured speed/agreement; memory savings alone do not
make it the requested fastest passing configuration. HiFi2 combination pending.

All14 candidates now have full-model traced results. Final combination
head_bfp4_hifi2_kv_bfp4 gives97%/100%/100% and36.8651789t/s/u, below head-only
BFP4/LoFi36.9184859 with100%/100%/100%. Thus head_bfp4_lofi is the literal fastest
median and highest-agreement evaluated policy; no noise exception is needed to
choose it. Installed its complete contract as selected_precision_config.json.
Final eligibility remains pending quality, context/memory and token-out gates.

Started normal default construction, with QWEN38_PRECISION_CONFIG explicitly
unset: `docker exec qwen38-functional-build bash
models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh env -u
QWEN38_PRECISION_CONFIG python -m models.autoports.qwen_qwen3_8_27b.tests.run_full_validation
--output-dir models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/selected_validation
--precision-long-cases models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/combined_long_cases.json`.
This captures actual prepared kernel arguments and reruns the main readiness,
shared short prompts, two4K open-ended prompts,4K exact-record retrieval and
32769-token multi-turn exact retrieval with preserved live traces.

Quality gate rejects the numerical winner head_bfp4_lofi. Normal default path
passes100% readiness and both factual retrieval cases, but benchmark_question_4096
spends all1024 output tokens revising its thinking and never reaches </think> or
final-answer text. Exact-input hash checks confirm HF512 reaches final text and
prior optimized TT1024 also reaches final text. Do not extend only the candidate's
budget to hide this regression. Archived complete validation/log under
quality_candidates/head_bfp4_lofi; verdict.json records controls and rule.

Next-fastest candidate is head_bfp4_hifi2_kv_bfp4 (36.8651789 TF t/s/u). Installed
its provisional selected artifact and started the same default-path validation,
unchanged prompts/1024 budget. Recomputed context_contract.json's
precision_candidate_evidence for BF4 KV payload4608bytes/token/device, native262144
and target32x32768; capability stays unchanged. Final measured memory/capacity
ledger is still pending. This is policy restoration/selection work, not a stage
pass or a reason to stop the attempt.

Quality also rejects head_bfp4_hifi2_kv_bfp4: both long narratives terminate
coherently (786/974 tokens), and both factual retrievals are exact with EOS, but
shared structured prompt emits only thinking at128. Exact same-prompt HF returns
complete4/3 JSON at128; baseline TT returns it with EOS at114. Archive all outputs,
control comparison and verdict under quality_candidates/head_bfp4_hifi2_kv_bfp4.
The missing short structured answer is a budget regression, not an arithmetic
error or a claim that the correct long replies are bad.

Continue descending measured speed with head_bfp4_lofi_kv_bfp4 (36.8162901t/s/u).
Default selected artifact and BF4 context-payload history updated. Added explicit
precision-mode gates to run_full_validation for exact short structured JSON
within128 and final-answer text on open-ended thinking-enabled long prompts
within1024; each failure saves quality_gate_failure.json and its actual output.
Controls and budgets are unchanged, and semantic review remains mandatory.

Third candidate head_bfp4_lofi_kv_bfp4 rejected by explicit structured128 quality gate; actual output and failure archived. Continue with head_bfp4_hifi2 (36.6369124 traced teacher-forcing t/s/u), original BF8 KV. Native capability remains262144.

Fourth candidate head_bfp4_hifi2 passes structured128 and readiness1/1/1 but fails long narrative final-answer gate at1024. Archive actual outputs/failure; restore head to BF8/LoFi, next-fastest evaluated36.419732t/s/u. KV remainsBF8, native262144 unchanged.

Fifth candidate head_bfp8_lofi rejected: structured128 begins correct final JSON but truncates before completing it. Exact controls complete within128. Archive output/failure; continue head_bfp8_hifi2 (36.3809209 traced t/s/u). No change to KV/context.

Selection validation uses the same command for each installed provisional config:
```sh
docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh env -u QWEN38_PRECISION_CONFIG python -m models.autoports.qwen_qwen3_8_27b.tests.run_full_validation --output-dir models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/selected_validation --precision-long-cases models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/combined_long_cases.json
```
The selected artifact is replaced only after the preceding hardware process
closes the mesh and exits. Rejected run directories/logs are moved intact to
quality_candidates/<config-id>; quality_gate_failure.json identifies deliberate
quality rejection, distinct from runtime crashes. Same prompts/budgets and
tracking enabled throughout. Scripts and stage-owned docs receive offline schema
and syntax checks; no hardware runs overlap.

head_bfp8_hifi2 quality passes: shared structured114/EOS, coherent4096/4095 narratives1024/760 bothEOS, factual4096/32769 exact47/33 bothEOS. Exact HF prompt provenance checked; all narrative outputs read. Start normal-default benchmark_selected_precision.sh, tracking0 after wrapper, no concurrent hardware.

Independent xhigh stage review initial verdict more-work-needed (stage_review_initial.md): no new implementation defect; finish pending benchmarks and reconcile reports. B1 token-out median36.17090288t/s/u and431.263307ms; native262144/nonaligned262141/final262143 finite and unchanged traces; allocator audit complete. B32 started after B1 process closed devices.

B32 selected token-out:4096 input median8.899521t/s/u (three repeats);32768 input7.46257t/s/u (one warmed repeat). Peak allocator28,142,107,136 bytes/device <34,178,731,008 capacity, exact event reconciliation and zero additional live bytes after audit. Actual native fused-decode weight/input/fidelity and BF16 batched collective buffers pass interim propagation audit. B8 begins after B32 closes devices.

All selected benchmarks exited0, serialized B1/B32/B8/B16. Offline audit_precision_runtime and summarize_precision_selection pass. Full actual dtype/fidelity inventories verified for all batches. Context strict check exits0 at262144; degenerate selected autoregressive output check exits0. Agentic checker exits1 for six inherited, evidence-backed representation-copy advisories only (no critical finding); physical payload counted once per named tensor family. Exact commands/output retained in final_checks.json. Final packet regenerated from supplied template with actual4096/128/B1 autoregressive values; device time/rooflines remain null.

Final independent xhigh stage-review returns **clean-pass**, no required work (stage_review.md). Reviewer independently reproduced all medians and hashed raw allocator peaks, checked actual output text, source hashes, lifetime/trace warning evidence, full runtime policy propagation and exact telemetry. Initial pending-evidence findings are closed. All changed/new Python and shell syntax checks and git diff --check pass. Stage08 complete; no vLLM integration or push. Local checkpoint follows this review.
