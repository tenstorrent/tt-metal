# Datatype sweep work log

Stage8 Qwen/Qwen3.8-27B revision1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
Starting HEAD624f6352a9c. Existing PIPELINE_BLOCKERS.md operator changes are unrelated and preserved.
Enabled installed tt-autodebug and tt-model-bringup verified in Codex config.toml;
packaged environment is loaded by tests/run_datatype_experiment.sh.

## Contract and baseline

Top1 >=90%, top5 >=98%, top100=100%; full64 layers, batch1 AIME24 chat S203/G100.
All selection measurements require model and sampling trace replay, native token delivery,
and reference token feedback. Token-out is measured separately using the Stage7 harness.
No vLLM integration or push is authorized in this stage.

2026-09-13: all four /proc/driver/tenstorrent/{0,1,2,3}/pids files empty.
Bounded tt-smi list exit0; exact configured Ring1x4 open/close exit0, MESH_SMOKE_OK.
Evidence: startup_list.log, startup_mesh.log. No reset or process termination needed.

Original optimized model refreshed before runtime edits: baseline_original.json,
.log, .source.sha256, .commit, .environment.json and .exit_status. Exact invocation
in commands.log. Prefill99/100 top1, decode98/100 top1; top5/top100100/100.
Warmed traced teacher forcing40.216222 t/s/user, TTFT77.932107ms;
warmed deferred token-out S128/G12840.387269 t/s/user, TTFT58.845697ms.
The readiness accuracy runner timing31.530251 t/s/user includes ranking/callback
work and is not interchangeable with that sampled-token benchmark.

## Search plan and runtime propagation

Inherited decoder projections are already BFP4/LoFi in all layers; paged KV BFP8.
Compare canonical BFP8 LoFi/HiFi2 controls, BFP4 HiFi2, BF16 KV, BFP8 CCL,
BFP8 projection activations, head BFP8 LoFi and head BFP4 LoFi/HiFi2.
Test compatible combinations after initial outcomes. Head LoFi is evaluated on
full-model gates even though Stage7 rejected a local PCC result.

Policy implementation: tt/precision.py validates fixed native operation contracts,
propagates group weights/fidelities/activations/CCL/KV and layer exceptions to the
existing decoder loader, and controls head weights/fidelity/logits. build_generator
and direct QwenModel construction share the loader. No serving adapter exists yet.
CCL workspace allocation now matches requested payload dtype; bind_cache validates
the layer-selected KV dtype. These changes require affected-device checks.

Stage is in progress; no selected policy or stage pass is claimed.

## Initial policy checks and harness correction

Policy-driven baseline reproduced99% prefill /98% decode top1, top5/top100100%.
Initial head_bfp8_lofi.json produced100% decode top1 and40.225259 t/s/user,
but its launcher exit_status=2 is invalid acceptance evidence: the launcher was
edited while Bash was waiting for Python; Bash subsequently read a changed file
offset and emitted a syntax error. Python had written results and closed all
devices (head_bfp8_lofi.log). This is an agent harness error, not a device/model
failure. The launcher is now frozen while running; *_evaluated experiments rerun
all accepted candidates with clean process status. Initial artifacts are retained.

The original model shared final-norm and head compute config. The sweep separates
final_norm_compute_fidelity=HiFi2 from head fidelity to isolate head matmul trials.
Residual dtype is explicitly consumed by residual-add output arguments; supported
native residual/norm/sampling assumptions remain BF16 and fail validation if changed.
BFP4/LoFi head reduced smoke passes S31/33/129, G4 on real layers0/3 and all4 chips.

Capacity arithmetic: tests/datatype_memory.py recomputed all12 planned configs.
All preserve262144 context. memory_*.json records per-resource bytes; even the
BFP8 decoder controls retain14,364,006,912 bytes of conservative DRAM headroom.
BFP4 KV is included as an additional likely bandwidth/capacity win. These host
calculations do not stand in for selected-policy capacity execution.

## First completed head comparisons

*_evaluated runs use frozen launch scripts and source snapshots. Baseline mean
of two warmed samples40.141822 t/s/user; head_bfp4_lofi40.878129 and
head_bfp4_hifi240.734779 (exact raw samples remain authoritative). Both BFP4 head
candidates have100% prefill top1,99% decode top1,100% top5/top100. These are
full-model gates with100 reference tokens, not a local projection PCC proxy.

The closest repository canonical Qwen36 policy was inspected directly;
canonical_policy_mapping.md records source and packed-group mapping. Added
canonical_qwen36_mixed_lofi (BF8 attention/output/down, BF4 gate/up, BF16 KV)
to the uniform BF8 controls. All mapped policies preserve the autoport geometry.

CCL BFP8 full-model candidate passes99% prefill/decode top1,100% top5/top100,
but measures39.649167 t/s/user, below the BF16 CCL baseline. BFP8 activation
candidate passes99% prefill /100% decode top1 and100% top5/top100, but measures
39.174354 t/s/user. Both are rejected for lower performance, not correctness.
Raw *_evaluated JSON and CSV aggregation remain authoritative for exact values.

## Initialization stall and AutoFix recovery

The first full kv_bfp4 attempt stalled after LOAD_LAYER63 and before MODEL_LOADED.
Device1 realtime-profiler synchronization had already timed out before layer0.
Fresh forked xhigh AutoTriage inspected current source and saved host stacks:
AUTOTRIAGE_kv_startup.md. The180-second live device capture timed out empty;
triage_kv_startup/host_gdb{,_main}.log and host_snapshot.json preserve successful
native host evidence. The main thread waits for a blocking pinned tensor upload;
KV allocation has not happened. No BFP4 KV accuracy or performance was inferred.

After evidence capture, only owned model PID599757 was sent SIGTERM (exit143).
A preserved tool-output snapshot at 22:58:20 shows all four driver owner lists
empty after model exit. The later recovery_actions.json owner field was captured
at 22:59:15 during the already-started reset and is mislabeled: device 1 lists
PID 602116. Its command line was not captured; reset ownership is an inference
from the serialized tool timeline, not a verified PID mapping. The raw JSON is
preserved, with exact tool events in triage_kv_startup/ownership_timeline_correction.json.
Bounded list failed(exit1), reset1 passed
(exit0), list_after1 passed(exit0, all4 p300c), and exact Ring1x4 open/close passed
(exit0, MESH_SMOKE_OK). No second reset, lock deletion or operator action needed.
Exact commands/statuses are in triage_kv_startup/recovery_actions.json and logs.

The candidate harness now schedules a180-second repeated Python stack dump during
model construction, cancelled before measurements, and records allocated cache,
logit and token dtypes. Model precision/runtime implementation is unchanged.
kv_bfp4_recovered retries the same config; an isolated AutoFix subagent is reviewing
recovery and source/config equivalence. The remaining matrix resumes sequentially.
No stage completion or native source repair is claimed by recovery alone.

AutoFix retry completed: kv_bfp4_recovered exit0,64 layers, actual BFP4 cache
allocation across16 full-attention layers,98% prefill/decode top1,100% top5/top100.
Warmed traced teacher-forcing samples40.177543/40.136699 t/s/user. Fresh isolated
AUTOFIX_kv_startup.md verifies model/native-source and precision equivalence.
Verdict recovered operation; native root cause remains unproven. Timed diagnostic
thread is a potential timing confound, not a claimed repair. The interrupted
attempt remains excluded from numerical ranking and is retained as failure evidence.

## Completed coarse matrix and combination

All 13 coarse full-model policies passed the required prefill/decode accuracy gates.
Uniform decoder BFP8/LoFi and BFP8/HiFi2 measured 30.7180 and 30.3664 t/s/user;
the mapped canonical mixed policy measured 34.3502. Decoder BFP4/HiFi2 measured
38.0612 versus the inherited BFP4/LoFi baseline 40.1418. Inner MLP BFP4/LoFi
with BFP8 first/last exceptions measured 36.1602. The head BFP4/LoFi candidate
currently leads at 40.8781; BFP4/HiFi2 head is 40.7347. These are warmed traced
teacher-forcing medians, not token-out serving measurements.

The next combination changes the head to BFP4/LoFi and KV to BFP4. This tests
the two passing memory-reduction opportunities together. CCL BFP8 and projection
activation BFP8 independently reduced throughput and remain rejected for speed.
No advertised capability is reduced. Matrix source snapshots retain the exact
runtime per run; final formatting only reordered the precision import in model.py.
The existing readiness and token-out harnesses now record the resolved precision
policy; their measured loops are unchanged.

The final normal-default teacher-forcing run is a separately labelled reproduction
check, excluded from the predeclared candidate-ranking rows. The winning candidate
must reproduce within 2% and retain accuracy/trace gates; a larger difference
requires investigation before selection can be accepted.

## Selection and normal-default reproduction

Selected head_bfp4_lofi, the fastest of 14 evaluated passing full-model configs:
40.878129 traced teacher-forcing t/s/user, TTFT76.607464ms, prefill100% and
decode99% top1, top5/top100100%. The head+KV BFP4 combination passed but measured
40.857371, within ordinary noise and below the selected BFP8-KV point. Thus both
the observed ranking and the simpler existing cache support the selected policy.

The selected JSON is consumed by default without an environment override.
selected_confirmation reproduced40.848672 t/s/user and identical accuracy;
actual weights, fidelities, cache/logit/token dtypes are recorded. selected_token_out
exit0: queued no-readback41.095462 t/s/user, TTFT58.417927ms; deferred complete
delivery41.075005 t/s/user, TTFT58.985590ms. Both use B1/S128/G128 and the Stage7
loop. Future serving comparisons must use the post-selection token-out number,
not substitute the AIME24 teacher-forcing ranking number.

selected_readiness exit0: standard accuracy gates pass; full-context cases
S262143/G2 and S262144/G1 reach position262144. Context contract records both
execution cases and selected BFP8 cache memory accounting, without capability
reduction. Full nonalignment/B32 and separate watcher checks are still running.

## Review remediation in progress

The reviewer independently verified all candidate policy/source/metric rows and
the Pareto plots. An owner-snapshot timestamp discrepancy was corrected using
the original tool transcript: see ownership_timeline_correction.json and
anomaly_ledger.md. A native-token versus HF-greedy count discrepancy is controlled
by reference_topk_order_audit.json: all stored native arrays reproduce the standard
readiness top-k metrics exactly; one HF tied maximum is ordered differently by
argmax and topk. Neither change alters the ranking or claims a model fix.

The selected Fibonacci answer uses a local list named sequence rather than fib.
The inherited output checker failed only on that spelling. Its restricted AST
check now permits append on a locally initialized list, retaining import/call
restrictions and behavioral tests for n=-1/0/1/2/8. The host qualitative checker
passes after that test correction.

Direct inspection found a 5/7/6 haiku versus 5/7/5 HF/Stage7 controls. The stage
remains open for focused same-prompt/cache precision controls under AutoFix;
no blanket qualitative pass or invented syllable count is claimed.

## Final validation and controlled quality disposition (2026-09-14 UTC)

All selected-policy device checks exit0. Full non-alignment covers seven lengths
with50 guarded captures; B32 fixed-slot/cache/continuation checks pass (PCC0.99928558).
Separate watcher plus trace-allocation tracking passes on real layers0/3 across
all four devices with Ethernet checks enabled and no device profiler.

The additional head_bfp4_lofi_no_fp32_acc candidate passes prefill100% and
decode99% top1, top5/top100100%, but its40.717337 t/s/user is slower. The final
matrix has15 passing accuracy configs and retains head_bfp4_lofi as fastest.

AutoFix haiku controls are complete. All final controlled processes exit0.
Selected native repeats and host-greedy match exactly at347 tokens with5/7/6
final lines. Baseline native repeats match the prior Stage7 control at418 tokens
and5/7/5. BFP4-head HiFi2 repeats match the selected347-token stream. The exact
S60/G1024/cache1088/history1023 geometry and physical table are identical. This
classifies a stable BFP4-head-sensitive quality limitation; it does not expose a
sampling/replay defect, and higher head fidelity alone does not repair it. The
selected config follows the explicit top1/top5 gates and fastest-traced-performance
contract. No blanket perfect instruction-quality or repaired-meter claim is made.

The probe initially dereferenced an intentionally invalidated host page-table
shadow after generation. The diagnostic now compares the actual device table;
original failed attempt and corrected clean retry are retained.

A review-found override-input bug is fixed: noncanonical layer keys are rejected
before allocation. Host regression checks cover rejected0/00 and applied string0.
policy_validation_equivalence.json proves all15 candidates plus selected resolve
identically under pre-fix and current validation; the measured runtime is unchanged.

Source snapshots are preserved byte-for-byte in per-run tar.gz archives with
archive/member SHA256 manifests. They reconstruct the original forensic paths.
The final policy/metric reports, reference top-k ordering audit, generated-code
and degeneracy checks pass. Independent final review and local checkpoints remain
the last closure steps; no push or vLLM integration has occurred.

## Independent review and local checkpoint

The fresh xhigh stage reviewer returned clean-pass in stage_review.md on
2026-09-14 UTC, with no required work before checkpointing. The review accepts
the explicit numeric selection rule with the retained, controlled haiku-meter
limitation; it does not claim the poem was repaired. All findings are fixed or
controlled with evidence and independently rechecked.

The checkpoint includes only this stage's model/test changes, context contract,
README and datatype_sweep evidence. The preexisting operator change in
PIPELINE_BLOCKERS.md is excluded. No vLLM repository or serving adapter changed.
No push is performed.

Authored Python and documentation passed explicit pre-commit checks. At checkpoint
time, only trailing-whitespace and end-of-file-fixer hooks are skipped to preserve
raw captured stdout/reference evidence byte-for-byte; those hooks already passed
the authored files. Other commit hooks run normally. No hook configuration changes.

### Recorded local checkpoint

| Repository | Branch | Stage checkpoint SHA | Subject |
| --- | --- | --- | --- |
| tt-metal | mvasiljevic/qwen38-full-bringup | 68de8c330c342b5827700a9acd6ece33100c338b | Select Qwen3.8-27B full-model precision policy |

The checkpoint completed successfully with all enabled commit hooks passing.
Exact command: `SKIP=trailing-whitespace,end-of-file-fixer git commit -m 'Select Qwen3.8-27B full-model precision policy'`.
This follow-up documentation commit records the stage checkpoint SHA; its own
SHA is reported in the handoff. Both commits remain local; no push occurred.
