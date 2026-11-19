# Optimized vLLM stage 10

Started 2026-11-17, from accepted stage 9 commit
`86130b35` (acceptance `85f277de`). Active runner wall cap: 8 hours.
No pushes. Preserve unrelated untracked prior-stage evidence.

Contract: real standalone TT plugin serving through `tt/generator_vllm.py`,
selected `combined_shared4_mixed` precision, P300x2 mesh 1x4, context 262144,
resident capacity 2, page 32, async scheduling, trace mode all, device sampling.
Prior capacity evidence excludes resident 32 by physical memory; 32 concurrent
HTTP requests remain required. No live-server profiling.

Stage 9 primary 128/128/1c1 decode was 9.675793 tokens/s/user, TTFT 6472.729ms;
it had no benchmark request warmup. It is historical context, not the stage 10
warmed before measurement. New before and after use the same shared runner
with `--additional-benchmark-args '--num-warmups 1'` for both primary 128/128/1c1
and secondary CI 100/100/32 uncapped concurrency.

Startup: enabled tt-autodebug verified in Codex config; installed package roots
come from lane-env.sh/environment.py. Four Blackhole p300c devices visible,
fuser empty, no pre-existing serving/test process. Bounded `tt-smi -ls --local`
passed. Plugin tree clean at `629f07cb`; upstream vLLM 0.31.0 unchanged.
Selected skills: vllm-integration, optimize, tt-device-usage, tt-enable-tracing,
qualitative-check and stage-review; AutoFix when needed. Independent CPU-only
optimization audit requested; only root owns device work.

Baseline launch: `TRACKING=0 bash doc/vllm_integration/run_server.sh
readiness_vllm/stage10_before` with paths rooted at this model directory.
Full command is the existing stage 9 script; logs are `logs/before_server.log`
and `../../readiness_vllm/stage10_before/server.log`.
No implementation changes before baseline.

Pending: warmed baseline, operation-topology audit and candidate decisions,
three scheduler budgets and mixed-traffic ITL, final sampling/quality/benchmarks,
P4/P5 timing and eval forecast, trace/async checks for affected paths, gates,
independent review and local checkpoint.

Candidate 1 changes only fixed-address scratch transfers in SlotStateMover.
Generator-wide preparation warms every copy shape before recording two traces
per resident slot; generator owns IDs and releases them at explicit teardown.
No new tensor, acknowledgment, allocation scope, dtype, layout, cache ownership,
sampling strategy or plugin interface. Same CQ0 ordering as the eager copies.
All source/destination backing allocations predate every live trace; hash
history aliases token_history, while shared attention K/V is excluded. Prefill
consumes the copied scratch state before copy-back; decode reads the destination
after copy-back. Pending decode is drained by the existing plugin transition
contract. Existing async host outputs are distinct completed DMA buffers.

Independent CPU audit caught stale replay callbacks after explicit generator
close. Fixed before hardware: clear callbacks after releasing IDs; an explicitly
closed/reused owner can use the existing warmed eager transfer, never dead IDs.
Strengthened host tests with distinct three-slot state, scratch edges plus eager
interior migration, stable addresses, shared-KV exclusion and close cleanup.
32 CPU checks pass; hooks recorded in logs/candidate1_precommit_v2.log.
This is development evidence, not stage review or device acceptance.

Warmed primary baseline (128in/128out/1request,c1,capacity2,C262144,budget256):
TTFT4648.148ms, TPOT103.516583ms, ITL median103.493736ms,
output throughput7.192993tokens/s. Exact raw/normalized results and provenance
are in readiness_vllm/stage10_before. CI burst and serial prefill burst are
collected on that same original-code server before candidate device work.

Candidate1 local checkpoint:4e5a1ddd3e4ede3a313d293025257c74f4ee8b3f.
Primary128/128/1c1 warmed after:TTFT4651.392ms,TPOT103.517868ms,
ITLmedian103.490717ms,throughput7.191605tokens/s,decode9.660168t/s/u.
Before9.660288t/s/u; change-0.00124%. No meaningful gain; not accepted.
CI before100/100/32 completed32:medianTTFT214182.769ms,
TPOT201.637464ms,medianITL201.829421ms,throughput7.262191tokens/s.
CI is queued capacity evidence only, never headline per-user decode.

Watcher correctness discovery triggered installed AutoDebug and AutoFix.
First AutoDebug child sandbox preflight failed (bwrap user namespaces unavailable);
retry used documented AUTODEBUG_SKIP_CHILD_SANDBOX=1 in current unrestricted
session. Report is AUTODEBUG.md, parent commands in logs/autodebug_copy_trace*.
See AUTOFIX.md and copy_control_comparison_v2.json for discrimination matrix.
Production-path traced and eager controls both pass all exact histories without
watcher. Watcher10 fails both; immediate-twice and watcher3600 also fail with
different first tokens. No claim of a watcher implementation defect yet.
All completed device jobs closed cleanly; no device reset or profiler used.

2026-11-17 04:18 UTC: native FP32 reduction controls passed twice, but
intermediate copies localized a transient error before reduction: gathered
remote rank1 data observed on rank3 is corrupt while later reducer output is
correct. Alternating-input isolated gather stress passes1000 iterations.
Independent source audit finds even-ring all_gather payload/completion route
coverage mismatch; a diagnostic matching payload routes to completion routes
is under exact watcher-enabled transition validation. No production repair
is accepted yet. See AUTOFIX.md and watcher_singleton_collectives_v2.state.json.

AutoFix root cause localized and repaired: multicast all_gather completion
missed alternating antipodal routes. Fixed dual-route completion and fenced
completion epochs for preallocated outputs. No plugin/vLLM changes. Fullwatcher
async and intermediate-state checks pass; final combined source is being checked.
Candidate2 skips the unused32×promptwidth int64 history when penalties are
inactive, preserving seed resets;33 CPU tests and source hooks pass.
Full-model128/128/B1 comparison is rerun with repaired shared CCL so serving
comparison does not conflate decoder-runtime changes with adapter overhead.

Core repair local checkpoint:572c94156bc4e594f0d2159a8e21808b6ea34ad7.
Includes ordinary and preallocated-output correctness controls, diagnostic
reproducers and topology proof. Incremental build/install logs and binary
hashes are retained. No push and no plugin/vLLM source changes.

Candidate2 checkpoint:be261ee9 (full SHA recorded by git). Final combined
watcher async probe passed both histories at80.30s. First fresh full-model
comparison invocation used vllm-env, whose Transformers lacks standalone
qwen4_exp registration; it failed setup before model construction. Rerun uses
the original full-model python_env (Transformers5.22.2), preserving workload
and selected policy. This is an environment setup failure, not device math.

Final matched comparison plan: after picking scheduler budget, measure old
generator/slot-state source with repaired shared CCL as a correctness-fixed
baseline, then final candidate at the identical served configuration. Preserve
initial pre-repair256 measurements as historical evidence; do not compare
different scheduler budgets or attribute CCL correctness changes to adapter
optimization. No source changes while an in-flight measured job uses them.

04:42 UTC: fresh full-model comparison passes (full48, B1, context262144,
128 input / 128 generated, 127 replayed decode steps). Resident token-out
9.785589898 t/s/u,102.191079991 ms; public caller-read9.770679444 t/s/u.
Fullwatcher final combined async transition check passes both strict histories.
Fresh TT shared quality controls are running with full contract context.
Old HF controls remain immutable; their old1024 field was a request-length
guard, while actual HF model configuration retained checkpoint max positions.
Chunk sweep orchestrator independently audited: fresh result guard, exact
4096/32768 rows, fuser error handling, owned PID/start-time identity and cleanup.
Three serial budgets256/33/17, one decoder (resident capacity2), ITLbound1000ms.
Required eval/correctness probes continue after the optimization wall cap if
needed; the cap stops optional optimization, not acceptance requirements.

04:54 UTC: checkpoint ad6c615433c8ed645354a28b767f65d9b1476921 records
repair diagnosis/controls, fresh full-context TT qualitative controls and guarded
chunk sweep. All12 TT raw256-token outputs exactly match prior TT controls;
independent HF/TT semantic review is quality_controls_review.md.
Fresh eval audit found required top_k40 is host-routed by advertised device cap32.
The common formatter also clamps32 and native sampling selects/walks only32.
This is required work, not an accepted host fallback or permission to alter eval
profile. AutoDebug eval-sampling retry uses bundled current Codex executable
(the stale ~/.local binary could not find code-mode host); no target execution
from investigation. Current three-arm greedy chunk sweep is unaffected.
Prepared configurable synchronous launch control and async-history prompt lengths:
when budget17 is selected, use5/7-token simultaneous inputs and160 outputs to
retain B2 from first decode and cross multiple32-token pages. The separate mixed
arrival suite still uses129/130/131/257/513/515 and96 outputs. Default65/67/96
probe remains intact for budgets admitting both prompts together. No weakened
width, page-table, token equality, overlap, or cleanup assertions.

05:22 UTC: required eval k40 repair applied after devices became free.
Native sorted64 and common/Qwen single-owner candidate families built locally;
no plugin/vLLM change. Native focused probe and Qwen seeded lifecycle pass;
capacity64 narrow path reproduces all old capacity32 token checks exactly.
Full watcher immediate/queued controls running for greedy and eval k40.
See AUTOFIX_eval_sampling.md for exact failures, setup fixes and artifacts.
Original chunk budget256/alignment128 measured worstITL19.9806s for32768input
with one concurrent decoder, resident2/C262144; budget33 failed alignment128
startup guard. Devices closed without reset. Existing config alignment0 is now
explicit for both serving/benchmark; new33/17/9 sweep will use final sampler.

05:30 UTC: wide sampler focused repair passes native distribution/high-ID/stale
inputs, exact old narrow tokens, seeded lifecycle, greedy and eval full-watcher
queued async histories. Common live CPU123 and model47+55subtests pass; all
pre-commit hooks pass. Invalid reset ordering fixed with six before-fail/after-pass
cases. Final full48 serving gates remain pending. Plugin/vLLM unchanged.

Local checkpoint755bf42020a43ebfcef6ff14320d5563fe0efa4d contains the k40
repair, regression tests, focus artifacts and alignment0 harness; no push.
Full-model comparison v3 starts from this source. Trace allocation positive
control rejects a deliberate live allocation and succeeds after deallocation.

05:34 UTC: full_model_comparison_v3 passes, full48/C262144/B1,128 input/
128 generated (127 decode steps), selected policy, no watcher/profiler/tracker.
Resident model+sample102.270285ms =9.778011t/s/u; public serving-boundary
resident102.240057ms =9.780902t/s/u; caller-read102.376513ms =9.767865t/s/u.
TTFT4744.060672ms includes reset/prefill/first sample+read. Source hashes
unchanged throughout. Devices closed cleanly before fresh TT quality control.

05:44 UTC: fresh TT qualitative control after755bf420 passes12/12; all256 raw
tokens and logical text exactly match previous stage10 control, no cleanup error.
Comparison script checks complete primary/CI/serial-burst shapes, identical
server contracts, selected policy, clean device routes and warmed controls.
Final serving measurements remain required.

05:44 UTC: checkpointacc20fc1b953ee0b3ea049bff17d81478cd384ec records final
full-model/TT controls and comparison analysis. Corrected sweep launched with
33,17,9, alignment0, resident2/C262144 and one concurrent decoder, source hashes
in chunk_sweep_execution.json; source unchanged during measurements. Serving
contract max_device_top_k/top_k rows now reflect40 with historical32 records
preserved; final real-vLLM row evidence will be updated after acceptance.

06:10 UTC: source inspection identifies existing supported scheduler knobs:
_align_prefill_chunk_end leaves final/non-aligned prompt tails unchanged,
and decode_interleave_prefill_steps defaults2. The unfinished alignment0
33-token diagnostic used many decomposed physical prefill passes. Cancelled
only its owned HTTP client, drained to0running/0waiting, then gracefully
stopped its server; cancellation manifest preserves ownership/cleanup. It
provides no completed32K measurement and is not a final sweep arm. New matched
configuration alignment8 plus one-prefill/one-decode interleave avoids odd
partial chunks while retaining all final prompt lengths. Restart all three
33/17/9 arms consistently; repeat tracked mixed/history gates afterward.
These are existing TT configuration values, not plugin/source/default changes.

Alignment validation: vllm-env python -m pytest -q -o addopts= plugin/tests/
test_prefill_chunk_alignment.py:12 passed. Actual pinned TTScheduler chunk
walks across caps9/17/33, twenty prompt lengths including129/130/131/513/515,
and continuation starts0/1/3/17/31:246 cases pass, no stall or lost tail.
The served mixed/history tests remain mandatory after this scheduler setting.

Replacement sweep source checkpointf29418e64a4e69f71a20f8f51b0a1b0feb4e492d;
launch follows the vllm-env/lane-env setup in run_server.sh:

```bash
python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_optimized_vllm_chunk_sweep --budgets 33,17,9 --evidence-prefix stage10_chunk_v3
```

Model source is
unchanged from755bf420. Structural first4K event comparison: alignment0 used
571 physical prefill replays and4158 processed positions (62 restored-prefix
replays); alignment8 uses128 replays and4096 positions. Both cover every0..4095
position. This is topology/count evidence, not a latency result.

07:46 UTC: alignment8/interleave1/1 budget33 completed with worst client ITL
2557.9/2599.3 ms at4K/32K; budget17 completed at1587.8/1620.5 ms. Both fail
1000 ms with no request errors. Budget9 continues unchanged. AutoDebug and an
independent source-only AutoFix hypothesis check identify the final-prefill
publication barrier: synchronous final prefill blocks publication of an already
materialized background decode. Four maxima match adjacent final prefill-result
intervals within0.8 ms; exact SSE pair attribution remains inferred because the
packaged client discards timestamps. See AUTODEBUG_mixed_itl.md and
AUTOFIX_mixed_itl.md. Existing interleave1/2 is the next discriminating experiment;
no plugin change or capability reduction is proposed.

chunk_v3_background_coverage_partial.json verifies an active background decoder
at every foreground prefill submission for both completed budgets,127/1023
interleaved decode submissions at budget33 and255/2047 at17, and zero capture
events in all four windows. These server events supplement, not replace, the
official client-content ITL. Source-only reviewer used no hardware or profiling.

08:08 UTC: alignment8/interleave1/1 sweep completed; none meets1000 ms. Budget9 TTFT300922.5/2462077.7 ms and worst ITL1064.0/1098.8 ms for4K/32K. All three arms drain0/0, clean owned-process/device-handle checks. Archived complete raw results in chunk_interleave1/. AutoFix now tests existing interleave1/2 with consistent17,33,9 sweep. No model/plugin/native source change.

08:22 UTC: local checkpoint176eda533691150616c0709f030f1aa1e969c757
starts interleave1/2 sweep17,33,9 under stage10_chunk_v4. The full1/1
background-coverage audit is now chunk_interleave1/background_coverage.json:
all six4K/32K windows have a decoder active on every prefill call and zero
captures. Budget9 has511/4095 decode submissions in those windows.

Prepared run_optimized_vllm_measurement.py for the matched greedy arms. It
backs up/restores only generator.py and slot_state.py, retains failed servers
and a recovery record, checks port/device ownership, preserves both runner
summaries, validates the actual launch command, records clean plugin HEAD,
vLLM version/Python tree, native source and loaded TT binary hashes, worker
configuration and all model TT Python source hashes. The comparator now
requires matching shared identities plus successful cleanup/restoration.
An independent source-only reviewer verified old-generator/narrow-sampler
compatibility and found no remaining blockers after the provenance fixes.
Syntax/CLI checks pass; real execution is pending the selected budget.

Planned invocations from the vllm-env/lane-env setup, with FINAL_BUDGET set
from the completed consistent sweep:

```bash
python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_optimized_vllm_measurement --budget "$FINAL_BUDGET" --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_matched_before --baseline-revision 86130b35
python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_optimized_vllm_measurement --budget "$FINAL_BUDGET" --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_matched_after --serial-reference models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_matched_before/vllm_prefill_burst_benchmark.json
```

This is a combined-generator-change comparison against the corrected shared
runtime, not an isolated candidate2 gain or an exact replay of pre-repair
historical86130b35 performance. The official run_vllm_server benchmark and
prefill_burst stages supply the measurements; no new timing harness is used.

08:50 UTC: interleave1/2 budget17 fails4K worst1184.1ms but passes32K877.5ms, compared with1587.8/1620.5ms under1/1. TTFT243952.3/1987248.8ms, no errors/clean shutdown.33/9 continue consistently. New matched measurement harness/comparator checkpoint07b83817bd1dfe22f5fcb40a07719cf938aff785 passes hooks and source-only review; no final measurement claim yet.

09:06 UTC: prepared run_optimized_vllm_replay_checks.py for the final tracked
sync/async checks. It launches the existing server script, runs the existing
160-token equal-width history and nine-wave mixed greedy/k20/k40 checks,
requires device-only routes, and records source/runtime equality and owned
cleanup. Async mode requires the completed synchronous execution with the same
budget/source/runtime. No new model/plugin behavior; CLI and repository hooks
pass. Hardware execution remains pending the chunk-budget selection.

```bash
python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_optimized_vllm_replay_checks --budget "$FINAL_BUDGET" --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_final_tracked_sync
python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_optimized_vllm_replay_checks --budget "$FINAL_BUDGET" --output-dir models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_final_tracked_async --async-control models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_final_tracked_sync/history.json
```

Replay orchestration checkpoint: b0fc736735962b82167941f61f61b9b29e1ad23a. The follow-up provenance set also includes both common-sampler Python files, so sync/async comparisons reject a shared sampler source change.

09:22 UTC: interleave1/2 budget33 completes with4K/32K TTFT183815.3/1489260.9ms and worst ITL1339.8/1364.7ms. Both windows fail1000ms; no request errors, idle0/0, no remaining owned processes and devices free before9 launch. Budget9 now runs unchanged. Replay provenance follow-up commit 09566d3dd4a270a58ee7ea144240a850c4d0b2f0.

09:26 UTC: chunk_v4_background_coverage_partial.json confirms an active decoder
at every foreground prefill call for completed17/33 arms, with zero captures,
retirements, host routes or OOV events in all four windows. Decode submissions
are510/4094 for17 and254/2046 for33, consistent with two decode submissions
between physical chunks. These counts do not replace official client ITL.
The new CPU-only audit_optimized_vllm_chunk_coverage.py also reproduces the
complete earlier1/1 sweep coverage (six windows); run it on the final manifest
after9 closes. Model/runtime source remains unchanged.

09:35 UTC: acceptance.sh prepares the final clean-server attachment sequence:
full unchanged sampling conformance (reasoning cap1528 and chat observer),
12 matched qualitative outputs, nine-wave mixed traffic, P5 timing at
1024/8192/32768/131072 plus127/128 boundaries, then P4 eval/plain comparisons
and32-user6000-token-capped EOS-respecting eval load. Each device-only phase
gets a separate log-offset route audit. Full conformance compatibility routes
remain explicitly outside those claims. Existing canonical timing/load results
must be preserved before running; server ownership/cleanup stays with caller.
P4 request timeout36000s/outer43200s accommodates the worst queued resident2
load without reducing users, context, profile or token cap. No B2 roofline is
invented; missing floor stays advisory. bash -n passes. CPU-only tokenizer
preflight in vllm-env Transformers5.14.1 matches all six HF-control rendered
prompts and token IDs, with no TTNN import. Actual acceptance remains pending.

```bash
bash models/autoports/qwen_qwen3_8_flash_next/doc/optimized_vllm/acceptance.sh "$FINAL_SERVER_LOG" models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_final_acceptance "$FINAL_BUDGET"
```

09:45 UTC: closure audit found that the current default vLLM shutdown_timeout0
force-kills EngineCore while generator.close is releasing traces:5/201 releases
for17,6/201 for33. Idle0/0 and no remaining owned PIDs/handles are verified,
but previous references to clean shutdown mean only that process/handle scope;
they do not establish complete orderly trace/device closure. The existing
--shutdown-timeout10 is planned after the live sweep closes, on both final
measurement arms. New harness checks require all captured IDs released and
no forced process kill/teardown errors. Also fixed a harness expectation:
run_vllm_server merges stages, so final benchmark summary is serve+benchmark+
prefill_burst, not serve alone. Added shared readiness Python source identity.

Source inspection also found unconditional ReadDeviceProfiler in plugin
worker.close_mesh_device. Profiling-disabled native code returns before queue
synchronization/readback, so no actual collection is alleged; nevertheless
it conflicts with the explicit no-call closure rule. A minimal unapplied
shared removal/test patch and all-eight-item proposed record are in
../vllm_integration/plugin_changes.md. CPU AST/stub reproduction fails on
pinned helper and preserves exact close/fabric order after in-memory removal.
Plugin HEAD629f07cb remains clean. Exact authorization was requested; do not
apply the proposal or start final orderly-closure runs until it arrives.
Budget9 continues unchanged. No new profiler call, device reset or live-server
profiling was used for this investigation.

10:20 UTC: consistent interleave1/2 sweep finished with exit0. Budget9 passes:
4K-input/4-output foreground TTFT355182.9ms and worst background client
ITL592.1ms;32K-input/4-output TTFT2918032.7ms and worst ITL615.7ms. Background
is one128-input decoder capped65536 outputs; resident2/C262144/P300x2,
alignment8/interleave1/2, greedy all-device.9 is the largest tested passing
budget;17 and33 remain rejected. chunk_selection.json verifies actual launch
settings, all source hashes unchanged, pinned clean plugin629f07cb, and complete
probe/idle/ownership results. chunk_v4_background_coverage.json passes all six
windows with continuing decode and zero captures/retirements/host/OOV.

Final fuser/process/start-tick/port checks pass: no owned processes, no device
handles, port8000 closed. Orderly trace teardown remains a separate failure:
9 has0/201 recorded releases before default-timeout force-kill,17 has5/201,
33 has6/201. Do not turn this into a full cleanup pass. No new server started.
Exact authorization for the shared profiler-call removal remains pending.
Prepared final harnesses require shutdown grace10 and complete release/no-kill
evidence; run_server.sh intentionally remains unchanged until the current
proposal is resolved. Final primary/CI/burst before/after, tracked histories,
sampling/quality/mixed, P4/P5, final documentation and stage review are unfinished.
No Stop Rule or stage pass is claimed. All optional untried checklist items
remain explicit. Local preparation/proposal commit:
0a3edb2e7656469613b02f81a726071a774dd0ad; nothing pushed.

Local checkpoints for this continuation (no pushes):

```text
b0fc736735962b82167941f61f61b9b29e1ad23a Add owned tracked serving replay checks and refresh ITL findings
09566d3dd4a270a58ee7ea144240a850c4d0b2f0 Include common sampler sources in replay control identity
843c3b072e964e293d87a739dc6c63507f665622 Audit mixed chunk coverage and preserve rejected budget 33 evidence
0a3edb2e7656469613b02f81a726071a774dd0ad Prepare acceptance and propose profiler-free orderly serving shutdown
b2b3bb93c30d93455a7628147c1ec2972564485e Select serving budget 9 and preserve pending cleanup authorization
```

Final continuation checks: six-window coverage audit passes, source/config identities verified, no owned processes/device handles and port8000 closed. Python compilation, bash syntax, git diff --check, explicit pre-commit and commit hooks pass. The proposed plugin patch passes git apply --check without modifying the plugin. Exact shared-default authorization remains unanswered; final serving and stage-review acceptance are not claimed.

## Owner shutdown decision and Stop Rule — 2026-11-17 11:13 UTC

Owner explicitly declined the plugin profiler-call removal. Decision recorded in owner_decision_shutdown_2026_11_17.json and eval_readiness/limitations.json. Plugin remains clean at629f07cb385494b14d0ee3d138de039d71d09546; patch is unapplied. Preserve default shutdown_timeout0 and existing owned cleanup. Process/handle cleanup remains mandatory; complete orderly trace release is an accepted plugin-owned limitation for upstream follow-up. Source confirms the unconditional hook; profiling-disabled native collection returns early, and observed worker termination can precede that hook. No unsupported causal diagnosis is asserted. This supersedes earlier pending-authorization/grace10 plans.

The eight-hour optional optimization cap elapsed at11:06:35UTC (start03:06:35). Stop Rule3 now applies. Mandatory matched measurements, correctness/eval gates, review and commits continue.

## Not tried (stop rule)

All gains below are unknown in this serving configuration. Selected datatype policy stays fixed.

- Broader residual-layout/collective families, row-parallel output decompositions, fractured consumer layouts.
- Dominant matmul program geometry, Blackhole readers-per-bank, precision/geometry cross-product.
- Packed versus split QKV and MLP gate/up; phase-specific activation shards.
- Reduced KV dtype and new attention/groupwise precision/fidelity sweeps.
- Additional generator persistent-output CCL paths, shorter compressed-index copies, per-slot whole-decode traces, traced reset.
- Omitting LM-head work on intermediate prefill chunks.
- Optional batched dense prefill; serial burst ratio remains a required measurement, with no adoption claim.

11:21 UTC: first matched baseline startup was healthy, but the new harness incorrectly required EngineCore's `/proc/PID/environ` to retain all initial keys. Actual setproctitle CPU reproduction proves these bytes can disappear while `os.environ` values remain unchanged. Recorded proc_environment_reproducer.json; verify the owned API parent's environment and pinned spawn source, preserve EngineCore observations without asserting they represent runtime getenv. No plugin/vLLM changes. Failed attempt stage10_matched_before_startup_probe_v1 ran no benchmarks, cleanup left no owned processes/handles, baseline files restored. During this aborted load a commit hook briefly stashed those two baseline files; the attempt supplies no measured evidence. Future source-replacement arms run without git hooks until restoration. Final gates driver now serializes after, comparison, tracked sync/async, acceptance and stops on failure. Owner decision checkpoint b9f328a5; full SHA will be recorded with final local commits.

## Owner stage-cap instruction and required lifecycle repair

2026-11-17: owner says the stage time/token cap is reached: keep the best passing version, run remaining required gates once, and carry untried ideas. Optional work remains stopped; the preceding Not tried list is the hand-off. A mandatory serial burst exposed a pre-existing unfinished-prefill slot relocation defect in baseline86130b35, also present in the optimized source. Primary and CI completed, but this failed arm is not accepted as the before comparison. Evidence retained in readiness_vllm/stage10_matched_before_slot_overlap_v1 and AUTODEBUG_slot_overlap.md. Source restored from verified saved HEAD65077302 bytes; no owned processes/device handles remain. No plugin change or device reset. AutoFix applies the same minimal lifecycle correctness repair to both comparison arms before the remaining gates. Unresolved correctness cannot be carried as a pass.

12:01 UTC: minimal pending-prefill repair CPU gates pass (60 tests; eight new regressions also pass against corrected baseline AST). Formatting and commit hooks pass. Local optimized repair commit `acbeff148d916e4c697624e4187596e2c4c1bbfd`; local corrected-baseline comparison commit `f80ac7492f10d5840943ccfcaa636ecb18e7871c` on stage10-corrected-baseline, not checked out or pushed. The latter differs only in generator.py/slot_state.py and includes the same correctness repair. Details: corrected_baseline.json and pending_prefill_correctness.patch. Remaining required serving gates will run once on these frozen sources; hardware acceptance is not yet claimed.

## Final matched measurement results

Both corrected-baseline and optimized arms pass the relocation-versus-solo32-token control, official warmed primary128in/128out/c1, CI100in/100out/32requests/uncapped concurrency, and serial-prefill burst128in/32out/32requests/c32. All use resident2/C262144/mesh1x4/page32/budget9/alignment8/interleave1:2/async/device sampling/selected datatype. The strict comparator passes matching server CLI, workload, shared Python/native runtime, selected config, source consistency and owned cleanup. See perf_summary.json and each readiness_vllm/stage10_matched_{before,after}/execution.json/provenance.json.

Primary before→after: TTFT7367.275664→7368.667097ms; TPOT103.625797→103.580787ms; mean ITL equals TPOT; median ITL103.607503→103.553426ms; aggregate output6.235355→6.236666tok/s; TPOT-derived9.650107→9.654300t/s/u. Decode change+0.0435% is noise, not a meaningful speedup. Final serving TPOT/full-model resident TPOT=1.012814 (full-model B1/128in/128out/127decode,C262144,selected policy). CI capacity before→after: mean TTFT251479.417→251345.289ms; TPOT236.006829→235.875647ms; aggregate output6.217375→6.220888tok/s. Burst mean-TTFT after/before=0.999543; serial prefill retained, no dense batched-prefill adoption.

Both route audits have zero host submissions/OOV. Both relocation probes exercised slot1→0 at committed position120 and matched solo tokens exactly without capture/retirement. Existing cleanup leaves zero owned PIDs/handles. Baseline releases3/198 traces; optimized4/202 before forced teardown. This remains the owner-accepted plugin limitation, not an orderly-closure claim. Full-model resident decode evidence is unchanged by the new prefill-only transition; no new decode trace/body or sampler computation was added. Final tracked and acceptance gates are running, so no stage pass yet.

Retention decision under the owner cap: keep the selected, passing implementation. The bounded four setup-owned scratch-copy traces and omission of unused prompt history do not show a material end-to-end gain in the matched primary, CI or burst measurements; do not claim otherwise. Timing differences are within noise, and neither arm establishes a meaningful performance advantage. Trace copies eliminate repeated host copy dispatch and preserve fixed operands; unused history allocation is removed by construction. Removing these changes now would create another source variant requiring repeated gates. Reassessing their setup/maintenance tradeoff is a non-blocking hand-off, not additional tuning in this stage.

## Owner extension and remaining acceptance — 2026-11-17

The owner loosened the wall cap for the remaining required gates and slow long-context work and authorized termination of owned leftover processes and `tt-smi -r` if needed. Optional optimization remains stopped; no reset has been needed in this continuation. See `owner_wall_cap_extension_2026_11_17.json`. The plugin-change rejection remains binding.

The first final sampling profile completed: **63 passed, 75 skipped, 4 failed** in1144.80s. Failures are `test_mixed_params_batch`, `test_same_seeds_reproduce_across_batches`, `test_specific_seed_reproducible[0]`, and `test_top1_is_greedy`; full assertions and XML remain under `readiness_vllm/stage10_final_acceptance/checks/sampling/` and `checks/sampling.xml`. This is not a full-suite pass. Correctness/logprob tests passed; the four reproducibility/isolation failures require diagnosis and current standalone/served logit evidence before classification. AutoDebug is inspecting source and the saved log only; no plugin change is authorized.

The original serial driver stopped at the sampling failure and retained its owned healthy server. Remaining independent gates continue on that same unchanged server using `bash doc/optimized_vllm/remaining_acceptance.sh <server/server.log> <checks> 9` (paths relative to the model root in this description; the saved script has the full invocation environment). The script preserves the failed sampling artifacts and runs the exact quality, mixed-traffic, P5 timing and P4 load commands from `acceptance.sh`, with route audits and final readiness check. Its existence does not waive sampling. Source/native/plugin identity, actual launch configuration and owned process start ticks remain in the original execution manifest.

Final HTTP qualitative collection completed12/12. Manual review and exact current TT-control prefix comparison pass; prompt tokenization matches the pinned HF controls. All returned tokens through EOS or256 cap are identical. See `final_quality_review.md` and `final_quality_regression.json`. Known haiku meter and capped-answer limits are retained. Quality route audit reports device-only sampling; mixed-traffic check follows on the same server.

Clean mixed traffic passed18 requests/9 staggered pairs at logical129/130/131/257/513/515 input,96 output,c2,C262144/resident2/budget9. Device-only sampling, continuation offsets, actual concurrent decode, intermediate no-output, no OOV, at-most-one capture per signature and zero retirements pass. Exactly one deferred first width64 sampling capture (0.3421s) occurred; all prior traces retained. TTFT p50/p95/p99=11029.859/40704.346/40730.734ms; text-stream-gap p50/p95/p99/max=201.722/572.661/584.013/1944.979ms; completed-request rate0.0435872/s. These gaps are text-chunk arrivals, not exact token ITL; maximum exceeds1000ms and is disclosed, not replaced by p99. The separate chunk-budget interference windows pass their required bound. No nonoverlapped greedy comparator was supplied to this mixed client; strict history/control equality is separately established by the tracked sync/async suites. See `final_mixed_summary.json` and raw acceptance `checks/mixed.json`.

AutoDebug completed through its isolated launcher (no hardware or source mutation); report `AUTODEBUG_sampling.md`, previous root report restored. Its reconstructed failing windows show no host/device route switching and no partial-prefill relocation. Current singleton-decode interleaving differs from the earlier passing stage9 admission histories. Source confirms the inherited singleton FP32-accumulating versus batched BF16 expert reduction. `AUTOFIX_sampling.md` records the leading hypothesis and alternatives. Test-only `check_sampling_repro_logits.py` prepares fresh standalone and served10-token greedy comparisons for all four prompt families, with exact repeats/row swaps and descriptive shared-prefix cross-width differences. Hardware evidence is pending after the serial P5/P4 sequence; no production fix or benign classification is claimed yet.

Local evidence/diagnostic checkpoint `ed9fd9c4f76adf08e7c69d7cb00edff079c6f5fd` records completed matched/trace/quality/mixed summaries and the test-only logit diagnostic. Pre-commit checks pass after import ordering correction. Production model/plugin/native source remains unchanged from the measured optimized arm. Hardware diagnostic execution remains pending. No push.

Independent CPU-only AutoFix probe audit (`/root/logit_probe_audit`) found no concrete bugs in `check_sampling_repro_logits.py`: public generator signatures, positions, per-layer tables, checkpoint/context/policy, repeat/reversed-row comparisons and token/logprob checks reviewed. Replay mode is host-only. Tracking0 lifecycle inference and selected-top20 limits remain explicit. No hardware or HTTP run by the reviewer; this is not the final stage review.

Sampling skip audit:70 speculative tests and1 speculative-artifact test are outside the non-speculative server; one all-vocabulary logprob test is skipped at the served top20 API limit. Three supported chunked-prefill recall tests also skipped because the initial full-profile invocation omitted `--tt-chunked-prefill-budget`. These are remaining required coverage, not unsupported capability: run `tests/tt/test_chunked_prefill.py --tt-chunked-prefill-budget=9` once after P5/P4. Dedicated mixed/trace coverage does not replace the content-recall assertions.

15:52 UTC owner-message resume: the interrupted turn ended the outer remaining_acceptance shell, while the bounded P5 timing client (926113 under timeout926112) and original server/EngineCore survived. At15:50 the131K prefill reached62552 tokens without recent errors. A detached, bounded serial continuation `resume_after_timing.sh` now waits for that exact client, verifies fresh complete timing JSON, audits its log interval, runs the unchanged P4 command once, checks readiness, then runs the three skipped chunked-recall tests and served logit diagnostic. PID/start ticks and exact command: readiness_vllm/stage10_final_acceptance/resumed_driver.json. No active device job was terminated, no reset, no model/plugin/native changes, and no completed gate is rerun.

17:03 UTC: P5 timing PASS, all four checks. Canonical doc/eval_readiness/long_context_timing.json records C262144/resident2/budget9, input1024/8192/32768/131072 with32 output tokens: TTFT59783.0/482918.2/1954437.8/8004681.3ms and TPOT104.60/110.63/118.00/129.54ms. No timing request failed and no TPOT cliff. The127/128-input probes (4 output tokens, three measured repeats after warmup) give median TTFT7569.3/7392.8ms, ratio1.024<1.5. The selected budget9 sweep gate passes. Full timing interval audit:21760 prefill submissions,148 device decodes,160 device-sampled submissions,0 host/OOV/captures;12 full input reloads,132 resident decodes,4 page-only reloads. The resumed driver recovered this complete result; its original process exit code is unknown because the parent shell was interrupted, so acceptance comes from fresh canonical checks and log, not an invented exit status. P4 launched once at17:03 on the same server.

18:42 UTC: P4 completed32/32 with all7 critical checks passing and3 advisories (functional floor unspecified at resident2, median answer428.5<3000, forecast lacks50% bucket coverage). Main profile lm_eval_reasoning,32 offered/resident2,C262144,cap6000,ignore_eosFalse:37315 output tokens in4006.0s,29 natural stops/3 length stops, wall9.314778tok/s (capacity, not headline t/s/u). Comparison32 requests/profile,256 output/ignore_eosTrue: median plain/eval TPOT213.02/213.23ms,ratio1.001<1.15. Peak running2/waiting31;preemptions0;health200;running/waiting/KV return0;main-load captures0. Whole P4 route audit (both comparison arms plus main):96 device final-prefills,28327 device decodes,0 host/OOV/new captures,8 pending-prefill relocations. Four saved answer tails are coherent and their displayed math/logic conclusions correct; the harness did not retain all full answers, so no32-answer accuracy claim. final_eval_summary.json and canonical long_answer_load.json preserve results.

Corrected planning forecast in eval_forecast_wall.json uses actual wall capacity, never bucket per-user×32. Default4027500 output tokens imply120.104851h: GPQA70.855152h, MMLU subset25.049801h, IFEval24.199897h. One shared14.4h budget is allocated proportionally, yielding illustrative23/33/64 questions; these are scheduling illustrations, not adopted reductions of required eval counts, output caps or context. Long-answer extrapolation and prompt-mix limits remain explicit. All three P4 advisories are recorded in doc/eval_readiness/limitations.json; no critical waiver is requested.

18:44 UTC: the three chunked-prefill recall tests ran with the explicit budget9 but failed under their default legacy-completion24-token cap. Every observed response ended in the initial `<think>` deliberation, before any final answer; preserve chunked_recall.xml/log. This is a concrete prompt-format/output-budget problem in the invocation, not evidence yet of missing cache content. Use the tests' existing --tt-recall-reasoning-effort low and --tt-recall-reasoning-token-budget1528 options for the focused rerun after the served logit diagnostic. Their final-answer/nonempty/finish-stop/passphrase assertions stay intact; no plugin or test source edits.

18:48 UTC: all three supported chunked-prefill recall tests PASS (194.21s) with existing chat/reasoning options; all14 requests satisfy unchanged final-content/passphrase/finish-stop assertions. Route audit:770 device submissions,0 host/OOV/capture. Both raw failed24-token invocation and corrected chat XML/logs are retained. Script:chunked_recall_chat.sh.

18:50 UTC: final owned server cleanup PASS, final_serving_cleanup.json. SIGTERM to the ownership-verified runner allowed existing default-timeout cleanup; zero owned PIDs/device holders, idle queues, source/runtime unchanged. Observed0/205 trace releases and forced worker termination remain accepted plugin-owned limitation; no reset, no plugin edits, no claim that the disabled profiler helper caused this specific kill. Original failed execution manifest is unchanged.

18:55 UTC: standalone logit discriminator started serially after verifying free devices. Command: `timeout 3600 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.check_sampling_repro_logits --mode standalone --execution-manifest models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_final_acceptance/execution.json --output models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_sampling_logits_standalone.json`. Lane environment, explicit mesh descriptor and OMP_NUM_THREADS8; watcher/profiler off. Raw stdout/stderr and exit saved beside JSON. Served24-run logit diagnostic already passes repeats/row-swap/actual physical-slot checks; native comparison pending.

Canonical readiness_vllm benchmark/CI/burst/run-summary JSON now comes from stage10_matched_after, and qualitative outputs from stage10_final_acceptance/checks. The matched run summary contains the actual owned server launch. Prior stage9 canonical content remains in git at ed9fd9c4f76adf08e7c69d7cb00edff079c6f5fd; no extra artifact archives are created. The final correctness server launch remains separately in final_serving_cleanup.json.

Final runner-side stage check executed once after canonical promotion: `MODEL_DIR=models/autoports/qwen_qwen3_8_flash_next bash /home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/prompts/model_bringup_multigoal/10-optimized-vllm.check.sh`. Exit1(advisory), no critical/error. No degenerate output; context262144 passes, historical reduced diagnostics disclosed; optional serial-prefill ratios miss batched-prefill advisory targets; P5 passes4checks; P4 criticals pass and three declared advisories remain. Exact output:final_stage_check.log/exit. The script includes check_eval_readiness, so no redundant standalone rerun is needed.

Standalone discriminator attempt1 failed before generation: generic diagnostic pool reserved16384 pages per indexed layer (two full contexts), exceeding available DRAM. Server evidence explicitly uses8866 pages/283712 pool tokens at unchangedC262144/B2 (acceptance server.log:175). No owned device handles remain after finally cleanup; no reset. AutoFix corrects only the test setup via --pool-blocks8866 and reruns to stage10_sampling_logits_standalone_v2.json/log/exit. Production source is unchanged. Probe revision differences are explicit in replay provenance; all model/shared/config hashes still must match and CPU negative-control rejection passes. The failed first JSON/log remains, with no result rewrite.

## Final logit acceptance — owner resume2026-11-18

The owner delegated full autonomy, loosened the cap and requested completion of comparison, review and local commits; prior no-plugin-change decision stands. The interrupted native retry had completed successfully: standalone_v2 JSON passes all four checks,24 runs/all48/B2/C262144/8866 pages, full-vocabulary repeat/row-swap exact. Log records device close; parent interruption left no exit sidecar, so do not invent exit0. All devices are free and no vLLM/EngineCore processes remain.

CPU-only replay command: `python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.check_sampling_repro_logits --mode served --execution-manifest models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_final_acceptance/execution.json --replay-artifact models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_sampling_logits_served.json --baseline models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_sampling_logits_standalone_v2.json --output models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/stage10_sampling_logits_comparison.json` exits0. All400 selected top20 maps equal exactly, physical positions and all repeat/swap checks pass. Native and served width-dependent token divergence matches (Text0 token7, Number token6); no cross-width determinism claim.

Four original pytest output-repeat failures are separately classified under vllm-integration's reproducibility-only rule, with other correctness/logprob/crash-free/qualitative checks passing. Raw63pass/75skip/4fail XML is unchanged. Three supported chunked-recall skips are covered by chat follow-up3pass;72expected skips remain. Current greedy diagnostics do not replay original stochastic prefixes, and original-HF controls are not claimed matched to these four cases. See final_sampling_summary.json, final_logit_summary.json and AUTOFIX_sampling.md. Canonical sampling_full_acceptance.json is now current and truthfully retains4rawfailures.

## Independent review round1 and correction

Fresh xhigh no-fork reviewer returned more-work-needed with oneP2: direct run_server.sh fallback still256 although every accepted launch explicitly used9. Report stage_review_round1.md. Corrected only fallback to9; explicit TOKEN_BUDGET overrides retained. CPU-only actual shell expansion stubs environment sourcing and final exec, verifies default9 and explicit17 plus C262144/B2: launcher_default_check.json PASS. No server/device launch, no altered historical source hashes, no need to repeat unchanged hardware gates. Current serving-contract supplement now points to final stage10 evidence and production source.

The reviewer independently checked all400 native/served token-step maps,960 sync/async output tokens, raw sampling/eval counts, captures/signatures and generated outputs. Other inspected gates and stated limitations accepted within their scope. The round1 report requests a small diff-only follow-up; final verdict still pending. Staged pre-commit checks pass. Generated JSON received only EOF normalization; raw XML was restored unchanged and remains local rather than being reformatted for git.

## Stage10 completion

Fresh independent xhigh round2 review returns **clean-pass**, with no required work: stage_review_round2.md. The one launcher finding is fixed and independently verified; no further device rerun or review required. Final README/checklist/manifest/serving-contract status now records completion with the scoped sampling/eval/shutdown limitations. Final applicable pre-commit hooks pass; all required evidence runs are complete. No owned processes/device handles remain, no reset was needed in this continuation, and the plugin remains unchanged. No push.

## Carried to next stage

- Untried optional optimization ideas are listed under Not tried (stop rule); gains unknown. Reassess four copy-trace setup/maintenance cost without claiming current speedup.
- Reproducibility across arbitrary active-width/admission histories is not guaranteed. Preserve raw four failures and current400-map exact control; original stochastic prefixes and exact-case HF controls are unmeasured.
- P4 has limited long-answer coverage and no B2 theoretical floor. Plan actual eval runtime from measured capacity;120.104851h is an extrapolation, not actual eval. Do not reduce required question counts/context based on illustrative fit counts.
- Plugin-owned orderly teardown remains incomplete under the explicit owner decision. Keep no-owned-process/handle cleanup and pursue upstream follow-up; no unauthorized plugin patch.
- Preserve persistent local raw large trace/logit JSON and server logs through hand-off. Compact summaries and manifests are committed; no new artifact archives/hashes were created.
- Shared native CCL/sampling hardware evidence covers this1×4 Blackhole configuration; other topology hardware validation remains separate.

## Local commit ledger

Repo `/home/ttuser/qwen38-flash/tt-metal`, branch `main`. Stage9 parent86130b35fd4abe56ca55f357168c97e6bc42ac97; all following checkpoints local, never pushed. Corrected comparison-only baseline branch points to f80ac7492f10d5840943ccfcaa636ecb18e7871c. Plugin unchanged at629f07cb385494b14d0ee3d138de039d71d09546; no plugin/vLLM commit.

- `4e5a1ddd3e4ede3a313d293025257c74f4ee8b3f` — Prepare Qwen serving state-copy trace candidate and warmed benchmark harness
- `572c94156bc4e594f0d2159a8e21808b6ea34ad7` — Fix multicast all-gather completion across alternate routes and epochs
- `be261ee9a013f3af35dd932512d355274286c840` — Skip unused prompt history for device sampling without penalties
- `ad6c615433c8ed645354a28b767f65d9b1476921` — Record serving runtime repair controls and guarded chunk sweep
- `03cacef7c1094dbe93ca29d7c74a18a285ba6693` — Prepare matched async controls and document eval sampling gap
- `755bf42020a43ebfcef6ff14320d5563fe0efa4d` — Support Qwen eval top-k 40 with persistent device sampling families
- `acc20fc1b953ee0b3ea049bff17d81478cd384ec` — Record repaired full-model controls and matched serving analysis
- `f29418e64a4e69f71a20f8f51b0a1b0feb4e492d` — Tune existing prefill alignment and decode interleave settings
- `176eda533691150616c0709f030f1aa1e969c757` — Test decode publication cadence after mixed-traffic ITL diagnosis
- `07b83817bd1dfe22f5fcb40a07719cf938aff785` — Bind matched serving measurements to runtime identity and cleanup
- `b0fc736735962b82167941f61f61b9b29e1ad23a` — Add owned tracked serving replay checks and refresh ITL findings
- `09566d3dd4a270a58ee7ea144240a850c4d0b2f0` — Include common sampler sources in replay control identity
- `843c3b072e964e293d87a739dc6c63507f665622` — Audit mixed chunk coverage and preserve rejected budget 33 evidence
- `0a3edb2e7656469613b02f81a726071a774dd0ad` — Prepare acceptance and propose profiler-free orderly serving shutdown
- `b2b3bb93c30d93455a7628147c1ec2972564485e` — Select serving budget 9 and preserve pending cleanup authorization
- `d1c6aca6c19e3e4d9c0e37398d0212e3ed178959` — Log local serving checkpoints and pending authorization handoff
- `b9f328a5adebdec5a6bf30a9619e802dcc65ee27` — Honor declined plugin change and preserve owned serving cleanup
- `65077302816adbb29109909dd675e1bfebcae5ed` — Repair process environment verification and serialize final serving gates
- `acbeff148d916e4c697624e4187596e2c4c1bbfd` — Preserve unfinished prefill state when serving slots move
- `ff80569624d60dfdce236aee2cb432e849b3515e` — Record corrected serving comparison baseline and cap handoff
- `ed9fd9c4f76adf08e7c69d7cb00edff079c6f5fd` — Record matched serving gates and prepare sampling reproducibility diagnostic

Final closure checkpoint SHAs are also recorded in the persistent runner receipt `/home/ttuser/qwen38-flash/multigoal_logs/10-optimized-vllm-commits.json` (kept outside the repository to avoid a self-referential commit hash).

Final evidence/launcher/diagnostic/documentation checkpoint: `8d7af6e236c90849319ab75a63e9ed787d771c18` on main. Commit hooks passed; all stage-owned tracked changes included, large raw artifacts retained locally by manifest. Follow-up receipt commit changes only this SHA ledger; both full SHAs are in the persistent runner receipt linked above.
