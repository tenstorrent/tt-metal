# vLLM integration work log

## Starting point — 2026-11-16

Model `Qwen/Qwen3.6-27B`, checkpoint
`8f33b98937c6031bb49e713dee3e8e185aafd26f`, autoport
`models/autoports/qwen_qwen3_6_27b`. Clean tt-metal `main` at start.
Completed datatype sweep selects `gdn_attn8_head8_mlp4_lofi`, consumed by
normal Model/build_generator construction. No precision search or context
reduction is part of this stage. Supported context:262144; target mesh:1x4
Blackhole P300x2, chips0..3. Baseline token-out:B1/P128/G128 at75.380767 ms TTFT
and37.817704 t/s/u; teacher-forcing37.645632 t/s/u is only a lower bound on
serving work. No serving comparison is measured yet.

Applied vllm-integration, tt-device-usage and tt-enable-tracing. AutoDebug and
AutoFix are used for cross-boundary plugin findings. Enabled plugin dependency
was verified in the active Codex config. Package exports are captured in
`env.sh`; installed skill packages remain read-only. No device was opened or
reset during the setup/CPU audit. Initial `fuser /dev/tenstorrent/*` showed no
holders.

## Serving stack installation

Reused the pair pinned in `doc/full_model/serving_contract.json`:
- Plugin:https://github.com/tenstorrent/vllm-tt-plugin,
  SHA042a94863ad27d88207bcf71abdf67122c6f4c27, clean source initially.
- Upstream:https://github.com/vllm-project/vllm, releasev0.31.0.
- Recommendation:plugin README and `docs/install-vllm-tt.sh` at that SHA,
  together with `docs/vllm-overrides.txt`. No upstream source checkout.

Read pinned interface, setup, execution, lifecycle, capabilities and decode
reload contracts, plus repository AGENTS.md before touching plugin files.
Installation:

```bash
source models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/env.sh
cd "$VLLM_TT_PLUGIN_ROOT"
timeout 1800 bash -e -c 'source docs/install-vllm-tt.sh'
```

`install.log`:exit0, installed vllm0.31.0+empty and editable standalone
vllm-tt-plugin0.1.0. `environment_before.txt` preserves the original environment;
`environment.json` records package versions and resolved import locations.
Torch2.11.0+cpu and transformers5.12.1 retained. Plugin host-test prerequisites:
`uv pip install 'pytest>=8,<9' 'ruff==0.14.0'`; pytest9.0.3→8.4.2, as required
by plugin AGENTS.md. Receipt:`plugin_test_dependencies.log`.

`uv pip check` exits1 (`pip_check.log`): vLLM's declared opencv>=4.13 conflicts
with its documented TT override4.11.0.86; unrelated installed FiftyOne requires
sse-starlette<1, whereas serving dependencies installed3.7.0. Do not claim a
clean dependency check. No FiftyOne functionality is used or validated here.
The interpreter has no pip module; uv's installed-metadata check was used.

## Initial adapter and CPU audit

`tt/generator_vllm.py` is a draft, not hardware-validated. It selects the pinned
policy via Model, constructs QwenGenerator only after scheduler cache allocation,
and delegates all model/sampling execution. Async callbacks enqueue a per-call
host transfer before subsequent replay; capability stays False until evidence.
Host compatibility is explicit and optional. The model-local
`QWEN_VLLM_DIAGNOSTIC_LAYERS=0,3` switch prepares the representative inner loop;
it is absent for final model runs and adds no plugin interface.

Runtime registration uses:
`TT_MODEL_CLASS_OVERRIDES=TTQwen3_5ForConditionalGeneration=models.autoports.qwen_qwen3_6_27b.tt.generator_vllm:QwenVLLM`.
The first host probe imported the platform too early (circular import);
the second omitted upstream0.31's required model_config argument. These probe
errors are retained in `registration.log` and `registration_retry.log`.
`registration_final.log` uses the current registry signature. Engine-process
resolution still needs the actual server launch.

Independent CPU audit `/root/serving_contract_audit` identified PC-01 and PC-02.
`probe_plugin_contract.py` executes exact AST-extracted plugin methods and the
intermediate-mask routing branch without TT device access. Its JSON proves
continuation slots2/1 become1/2 while decode owns0, and an otherwise eligible
mixed intermediate/final batch loses device sampling and receives no finality
metadata. This is a CPU reproducer, not a serving test.

Three plugin tests in `tests/test_block_model_runner.py` exercise reversed
continuations, mixed new/continuing rows, and a continuation alone. Command:

```bash
cd "$VLLM_TT_PLUGIN_ROOT"
PYTHONPATH=tests/host-stubs timeout 180 "$VIRTUAL_ENV/bin/python" -m pytest \
  tests/test_block_model_runner.py -k continuing_prefill_retains_state_slots -q
```

`slot_regression_before.log`:3 failures on pinned source, as expected. No
production plugin change was made before the independent diagnosis.

## AutoDebug and authorization

The installed AutoDebug launcher first failed sandbox preflight because the
host forbids unprivileged namespaces (`autodebug_launcher.log`); no model was
started. Retried with `AUTODEBUG_SKIP_CHILD_SANDBOX=1`, within this run's existing
unrestricted execution authorization. The investigator remains inspection-only;
`autodebug_retry.log` preserves the invocation and output. Its conclusion must
be read before keeping any repair.

PC-01 in `plugin_changes.md` requests exactly the capability
`supports_prefill_sampling_mask` and keyword `prefill_forward(sampling_mask=...)`.
The installed skill explicitly requires human authorization for these shared
interface additions. An asynchronous approval request is pending. No approval
is inferred from elapsed time, and the extension is not implemented.

## Acceptance state

No device/server benchmarks, max-context request, sampling suite, qualitative
outputs, eval probes, mixed-shape trace check or final review have run yet.
Primary128/128/1 and secondary100/100/32 metrics are unavailable. Target server
settings remain context262144, block32, max-num-seqs32, decode trace, on-device
sampling all, async scheduling only after proof. Final command/TT config must
be recorded from the actual successful runner, not this plan.

No stage-completion commit or push. Remaining requirements are work; PC-01 is
the exact pending operator decision. Plugin PC-02 validation/commit receipts
will be appended once the diagnosis and tests complete.

### Host setup results

Starting tt-metal SHA:`818f5e81574b25c1a0784051172e7f91dc96ae52`.
`registration_final.log` passes: registry resolves
`TTQwen3_5ForConditionalGeneration` to this autoport's `QwenVLLM`.
This is host-process verification, not engine-subprocess evidence.
`server_cli_help.txt` confirms upstream CLI accepts `--additional-config` and
`--async-scheduling`; no obsolete plugin-config shim was added.

Pinned plugin host baseline:
`PYTHONPATH=tests/host-stubs timeout 600 "$VIRTUAL_ENV/bin/python" -m pytest tests/ --ignore=tests/tt -q -k 'not continuing_prefill_retains_state_slots'`:
**2303 passed,16 skipped,3 deselected** in84.69s, receipt
`plugin_host_baseline.log`. Only the three newly added regressions are excluded
from this pre-fix baseline; they separately fail in `slot_regression_before.log`.

Serving-suite collection initially lacked required CLI options; receipt
`sampling_collection.log`. With explicit server URL/model/max-num-seqs32,
`sampling_collection_final.log` collects141 tests and errors on
`tests/tt/test_tile_size.py`, whose direct platform import triggers upstream's
partially initialized `vllm.platforms.current_platform`. This is outstanding
collection work, not a test skip or serving pass. No server was contacted.

### AutoDebug conclusion and kept host repairs

The installed isolated AutoDebug launcher completed with exit0; report copied
as `AUTODEBUG.md`, with diagnosis pinned to plugin042a948 despite the owner's
concurrent candidate edits. It independently confirms the existing state-slot
allocator defect and the documented chunked-prefill host-output limitation.
A model can replay an entire prefix to reconstruct state, but doing so conceals
false plugin ownership and duplicates computation. Partitioning plugin calls
can keep final rows device-sampled, but intermediate calls still require host
logits. PC-01 remains a justified shared extension under this goal's stricter
no-logits-readback requirement, not the only imaginable design.

AutoFix record:`AUTOFIX.md`. PC-02 now reserves all existing live slots before
assigning new requests and reuses each continuation's claim. The full plugin
host suite passes **2306 tests,16 skipped** in63.67s
(`plugin_host_fixed.log`). The additional companion-finish history control
also passes: `slot_regression_final.log` has **4 focused passes**. No device
result is inferred from these host tests.

PC-03 explicitly initializes vLLM before the tile-size test imports the plugin
platform. `sampling_collection_fixed.log` collects **142 tests**, exit0; no
assertions or test coverage were removed. Plugin pre-commit passes after
formatting (`plugin_hooks_history_final.log`), and `git diff --check` passes.
Exact local candidate patch:`plugin_candidates.patch`. The initial ruff
format run changed only the added test formatting; the full host suite ran
with the formatted production source. The later history test adds coverage
without altering production code.

These candidate plugin changes remain uncommitted pending affected serving
validation and the independent stage review. Their records explicitly mark
that incomplete status; no completed plugin-change or stage acceptance is
claimed. No upstream vLLM source was edited. The sampling-mask capability and
keyword are still absent: PC-01 is awaiting exact human authorization.

No server/device workload was launched. Therefore no successful runner command,
benchmark, sampling/qualitative output or model-output verdict exists yet.
The final process audit is `cleanup.json`; no vLLM/EngineCore device holder
remains. All original serving gates remain required after approval.

### Authorization revalidation

Automatic continuation supplied no human approval for PC-01. Verified the
proposed capability and keyword remain absent from plugin source and no device
is held. Restored the root AUTODEBUG.md to its pre-stage tracked contents after
verifying the new report is preserved byte-for-byte in this stage directory;
this keeps the previous stage report out of the integration diff. PC-01 remains
pending, and the goal is not complete.

### Blocked audit — third consecutive authorization recurrence

The original goal turn and two automatic continuations all retain the same
pending PC-01 human decision. No operator message authorizes the capability
or keyword. Revalidated the proposed record and source: neither extension is
implemented. The prior turn preserved the integration diagnosis and restored
the historical root report; that cleanup is complete. Independent setup and
host repairs are preserved, but do not satisfy the required device-serving
contract. No device holder remains. Marking the goal blocked on PC-01; this is
not completion or a stage-review pass. Resume requires explicit authorization
for the proposal or an operator decision changing that proposed interface.

## 2026-11-16 PC-01 owner approval

The owner explicitly approved the exact `supports_prefill_sampling_mask` default-False capability and `prefill_forward(sampling_mask=...)` semantics recorded in plugin_changes.md. Implementation resumed, with independent CPU-side plugin work and generator-side mask handling. All prior blocked statements are historical; stage 9 is active and its serving gates remain pending.

PC-01 plugin candidate: 223 regression cases; full plugin host suite 2530 passed, 16 skipped (`pc01_host_suite.log`); pre-commit passed (`pc01_hooks_final.log`). Pinned-source regression failed in 12 expected cases (`pc01_before.log`). Parent inspected the shared diff.

Representative hardware command (diagnostic layers 0,3; all 4 chips; full serving context/pool geometry; bounded preparation subset):
```bash
source models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/env.sh
QWEN_VLLM_DIAGNOSTIC_LAYERS=0,3 TT_METAL_TRACE_ALLOC_TRACKING=1 timeout 1800 python -m models.autoports.qwen_qwen3_6_27b.tests.probe_vllm_adapter --output models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/adapter_reduced.json > models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/adapter_reduced.log 2>&1
```

`adapter_reduced_final.json`/`.log`: representative adapter probe passes with trace allocation tracking and deliberate late-allocation positive control. Intermediate-only prefill has zero sampling, token reads and logits reads; mixed final row equals seeded solo continuation and preserves masked controls. Deferred read handles retain known values123/456 independently; stale host token/position/table values are ignored on resident decode; page-only reload advances position once without token refresh. First attempt stopped on an inconclusive equal-token assertion; retry stopped because the positive-control assertion expected a different error phrase. Both test defects were corrected; all original logs retained. No reset or hang. Caps enabled after this direct proof; scheduler overlap and automatic allocator growth remain distinct pending checks.

Live triage was captured during a long gap near B4 warmup length288 (`triage_reduced_warmup/`). Progress subsequently reached320; no hang established, no reset/kill. Triage operation reporting was incomplete because inspector had no program metadata; basic hardware checks passed. `triage_reduced_warmup/AUTOTRIAGE.md` records that evidence limit.

Reduced server first launch: full B1/B4 preparation completed and HTTP health passed. First nonaligned129-token request failed at the documented mRoPE tuple boundary, after model execution; adapter corrected to return zero text-only deltas. Failure logs preserved under readiness_vllm/reduced/. No stale server/EngineCore holders remained; chip0 post-server arithmetic smoke passed without reset. Retry command:
```bash
QWEN_VLLM_DIAGNOSTIC_LAYERS=0,3 TT_METAL_TRACE_ALLOC_TRACKING=1 bash models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/launch_server.sh models/autoports/qwen_qwen3_6_27b/readiness_vllm/reduced_retry 4
```

Reduced retry passed the same-server lifecycle (`readiness_vllm/reduced_retry/lifecycle.json`): actual changed continuation starts and partial tails with interleaved decode; three stable trace signatures, no retirement/recapture; greedy equality across reordering/concurrency. Diagnostic B4 primary128/128/1: TTFT15.592530ms, mean TPOT2.466073755ms,405.502876t/s/u, ITL P50/P992.462353/2.572969ms, output388.885912tok/s. CI100/100/32 is secondary:1491.606012output tok/s. These two-layer numbers are not full-model performance. Server was stopped only after metrics showed no running/waiting requests; no device holder remained.

Source audit identified avoidable fixed-capacity GDN clone/restore and recurrence work on single-user decode. Candidate model-owned buckets `[1,max_num_seqs]` preserve one generator/cache/sampler. Prefix kernels restrict writes to active prefix users; attention borrows identical scheduler K/V with a one-row table refreshed only at actual reload. All kernels warm before captures. CPU independent review found no concrete wiring defect. `adapter_prefix.json`/`.log` passed on layers0,3/full serving geometry with allocation tracking positive control, B1-vs-B4 greedy equality, inactive GDN byte equality, B4→B1→B4→B1 transitions, stale inputs/deferred reads, zero full-logits reads. Real frozen TTSamplingParams are copied into adapter-owned parameter values without mutating them. Reduced live benchmark and full-model correctness/performance remain pending; no performance adoption claim yet.

```bash
QWEN_VLLM_DIAGNOSTIC_LAYERS=0,3 TT_METAL_TRACE_ALLOC_TRACKING=1 bash models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/launch_server.sh models/autoports/qwen_qwen3_6_27b/readiness_vllm/reduced_prefix 4
```

Full-serving plan: preserve262144 context,32slots, selected precision, async final. Native uniform prefill is **not adopted**: previous B3 evidence is numerically close but not bit equal, and scheduler512-token budget does not form B32x128 native groups. Stage9 measures serial burst; optional batched adoption remains Stage10 work. Adapter default now serial; model-owned `QWEN_VLLM_BATCHED_PREFILL=1` explicitly opts into the existing experimental native path. Standalone generator default is unchanged. Full synchronous control started with the equivalent older `QWEN_VLLM_SERIAL_PREFILL=1`; final launch uses current default-off implementation. Inactive uniform preparation can be omitted when serial to avoid unnecessary allocation/warmup.

Operator defaults short chat to `--default-chat-template-kwargs '{"enable_thinking":false}'`; P3 explicitly tests thinking on. HF controls are running thinking-off,256tokens,sampled.7/.9/k20 matching packaged suite. CPU-only reference runs concurrent with preparation, so no performance is claimed from that interval. Optional `QWEN_VLLM_HOST_COMPATIBILITY=1` enables documented unsupported features for full sampling/P2/P3; it does not replace supported device sampling. Added literal `host_sampling` marker only for actual host steps so readiness probes count fallback accurately.

Temporarily suspended CPU-only HF control PID2053702 during full server compilation/control measurements to remove CPU contention; resume with SIGCONT after the control server checks. The HF timeout remains10800seconds. No TT process was paused or terminated.

Candidate deferred page refresh: B1 decode updates its shared one-row page table and marks the full per-layer tables dirty; the next full decode or prefill flushes them, including unchanged-map prefill. The independent source audit found no concrete correctness defect. Strengthened `probe_vllm_adapter.py` now compares active and inactive recurrent tensors exactly and checks dirty-map flushing; hardware verification is pending. Detailed program-cache telemetry is enabled for the tracked lifecycle; `QWEN_VLLM_TELEMETRY=0` suppresses that query/JSON overhead during performance. Per-step sampling-mode audit remains enabled. `capture_phase.py` records command exit, server-log byte boundaries and device/host counts for each measured phase.

Full synchronous control command (all64 layers,32slots,262144context, allocation tracking):
```bash
QWEN_VLLM_HOST_COMPATIBILITY=1 QWEN_VLLM_SERIAL_PREFILL=1 TT_METAL_TRACE_ALLOC_TRACKING=1 bash models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/launch_server.sh models/autoports/qwen_qwen3_6_27b/readiness_vllm/full_sync_serial 32 --no-async-scheduling
```
This process loaded the adapter before the deferred-page/telemetry changes. It is the non-overlapped numerical control, not final-source acceptance or performance evidence. It still prepares the unused nativeB32 prefill path; subsequent serial launches omit that preparation.

Full synchronous control failed after successful preparation on its first decode at the synchronous host-formatting boundary. See `AUTOFIX.md` for the diagnosed callback mismatch and minimal repair. No model/kernel/device execution failure was indicated. Both server processes exited; four-chip arithmetic smoke passed without reset (`post_sync_server_smoke.log`). Resumed CPU HF PID2053702. Strengthened representative probe launched with the command above, output `adapter_prefix_final.json`; this also validates the deferred-page candidate.

The same-code TT qualitative control completed all12 greedy/sampled cases at256token budget, thinkingoff,.7/.9/k20; raw output is `readiness_vllm/qualitative_controls/tt.json`. Read all12: coherent, on topic, English except requested French translation, no repetitive/gibberish or request-contaminated prose. Explanations and stories can end at the token cap. Final cross-backend verdict awaits all HF and served outputs. The raw standalone stopping criterion used text_config EOS248044; HF/vLLM generation_config also stops at chat EOS248046. Short raw TT outputs therefore include `im_end`, a newline, and `endoftext`. `tt_serving_stop.json` is an explicitly derived prefix through the first checkpoint EOS, preserving raw counts and citing the unchanged raw artifact; this normalization cannot change any token before stopping. The control script now uses the checkpoint generation_config EOS list for future runs. No model forward/sampling path changed. Special token spellings in raw tokenizer decoding are distinguished from visible API output.

Corrected full synchronous server queued only after TT control exit and empty device-holder check; command: `QWEN_VLLM_HOST_COMPATIBILITY=1 TT_METAL_TRACE_ALLOC_TRACKING=1 bash doc/vllm_integration/launch_server.sh readiness_vllm/full_sync_serial_retry 32 --no-async-scheduling` (paths relative to this model folder in this abbreviated record; launch script changes to repo root). Exact runner command is in that directory's runner.log.

Full synchronous retry lifecycle passed (`readiness_vllm/full_sync_serial_retry/lifecycle.json`):16requests, all greedy repeat/reordering comparisons exact, actual continuation tails with concurrent decode,4stable signatures(decode_b1 B1; decode/sample/reset B32; all8224pages), zero retirements, program cache constant24853. 27 allocator-grown page columns were actually consumed;23page-only updates;663unchanged-map steady steps. Audit713device steps/0host steps (`lifecycle_audit.json`). This is tracked synchronous diagnostic traffic while the CPU HF control was running; its latency is not headline performance.
Full synchronous logprob control passed (`logits.json`): top20IDs and values exact across6solo prompts and two concurrent/reordered repeats; all6standalone top20sets/values exact, maximum logprob error0. Optional host compatibility was explicit;9prefillcalls/0decodecalls in `logits_retry_audit.json`. Initial client attempt failed before any HTTPrequest because Transformers5 returned BatchEncoding from tokenize=True; corrected test to the same render-then-encode procedure as the controls.

Sampling smoke on full synchronous retry:2passed,1failed,1skipped (`sampling_tests.log`, `sampling_smoke_audit.json`). `test_mixed_params_batch` fails seeded stochastic equality after shuffled admission; greedy/topk1 and min_p pass. All-vocab chat logprobs skips because the server retains the upstream max_logprobs20 cap; top20 correctness already passed exactly. This is not final full-profile acceptance.

Focused `probe_vllm_sampling_route.py` on the same server submits a129token/64output supported seeded request, then only after its first streamed token admits a137token/8output presence-penalty companion. `sampling_route.json` records request bodies, times, IDs and exact log byte ranges. Device→host→device mode switches occur during the first request; sampled output differs from its solo control at index2, while the equivalent greedy pair is identical. This is a routing defect, not a waived reproducibility-only result. Fresh inspection-only AutoDebug runs through the installed launcher (`sampling_autodebug.log`); existing documented sandbox preflight failure justifies the already-authorized explicit sandbox skip. No new plugin interface is implemented.

## 2026-11-16 sampling investigation and PC-04 proposal

Fresh AutoDebug completed exit0. Saved report `sampling_AUTODEBUG.md`; restored
root AUTODEBUG.md to its prior-stage tracked content. The report confirms the
controlled companion probe's route switch but explicitly cannot identify the
first singleton in the original shuffled smoke. Do not conflate those claims.
The plugin documents batch-wide `auto` routing, so this is a stage-contract gap,
not a false claim that the plugin promised request-lifetime affinity.

Parent source check confirms canonical generator penalties are disabled and
must be implemented in model code. A separately authorized routing workaround
would not substitute for that missing feature. Correct penalties across replay
also need the original prompt boundary, which current prefill arguments omit.
`probe_prefill_history.py` executes the real current CPU builders/submission:
replay prompt[0]+generated[1,2] and fresh prompt[0,1,2] have identical adapter
kwargs, despite different required output counts. It exits0 with
`missing_history_boundary_confirmed`; JSON and log retained. Command is in PC-04.
An independent source reviewer checked front/lane ordering, authoritative
num_prompt_tokens, finality and release semantics and agreed no existing
prefill mechanism carries the boundary. The proposed exact capability and
keyword in `plugin_changes.md` have NOT been implemented. PC-01 approval does
not cover PC-04. This is the first new authorization recurrence; the active
goal is not marked complete or prematurely marked blocked.

Local plugin checkpoints (never pushed):
- `ea251ee12377d247fd4b80af67c2cf7b8ac4d7a1`: PC-03 test collection correction.
  Explicit pre-commit pass: `pc03_commit_hooks.log`.
- `1e35b653ae57f28914e60b3759ad07bc38f32a21`: approved PC-01 finality metadata
  and PC-02 continuing-state ownership. Explicit pre-commit pass:
  `pc01_pc02_commit_hooks.log`. Combined host suite2530passed/16skipped,
  before-change failures, representative mask/async probe and all-layer
  synchronous affected lifecycle evidence are recorded in plugin_changes.md.
  These are checkpoints, not final stage-review or complete serving acceptance.
`plugin_candidates.patch` now records the entire pinned-base→local-HEAD diff,
including new tests. Upstream vLLM source is unchanged.

Shutdown: saved zero-running/zero-waiting metrics before sending SIGTERM to
API2192335. API exits0; upstream's timeout0 shutdown manager reports a forced
EngineCore2193661 kill during teardown, so complete trace-release events are
not claimed. The serve-only wrapper reports exit1 because its child ended
while it was holding. This is intentional diagnostic shutdown, not a passing
final runner invocation. No server/EngineCore holder remained. Bounded
four-chip arithmetic open/add/read/close passes without reset in
`post_routing_server_smoke.log`. `cleanup.json` is the current audit; the prior
pre-approval audit is preserved as `cleanup_before_pc01_approval.json`.
No watcher/profiler was used; stage9 reset count remains0.

CPU-only HF controls were resumed after compilation and completed six outputs
(first three prompts, greedy+sampled); all six were read. They are coherent,
on-topic and free of mechanical repetition/gibberish/language drift or request
contamination. Explanation/story truncation occurs at256tokens in both HF and
TT. To leave no unexplained background job while awaiting PC-04, terminated
only CPU process2053702; session exits143. Completed JSON rows are preserved;
no TT process was interrupted. Resume exactly with:
```bash
source models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/env.sh
timeout 10800 python -u models/autoports/qwen_qwen3_6_27b/doc/full_model/qualitative/run_suite.py \
  --backend hf --thinking off --temperature 0.7 --top-p 0.9 --top-k 20 \
  --max-new-tokens 256 --resume \
  --output models/autoports/qwen_qwen3_6_27b/readiness_vllm/qualitative_controls/hf.json \
  > models/autoports/qwen_qwen3_6_27b/readiness_vllm/qualitative_controls/hf_resume.log 2>&1
```
`run_suite.py` verifies metadata and rendered input IDs before reusing rows.
Served quality remains unjudged until collection; no partial-control pass
substitutes for it. Max-context acceptance, final async/full sampling,
P2/P3/P5accuracy, full-model performance and stage review remain required.

Checkpoint hooks formatted Python imports/line wrapping only (no execution
semantics changed) and normalized trailing whitespace/end-of-file on some
text artifacts. Raw server logs and their byte-indexed probe intervals were
unchanged. The CPU history witness reran after import formatting. Two complete
inspection transcripts exceed the repository500KB limit and remain raw local
artifacts outside the commit: `autodebug_retry.log` and `sampling_autodebug.log`.
Their completed reports are committed. All raw benchmark/server artifacts remain
on disk at their recorded paths; no per-artifact compression or archive was made.

Checkpoint verification: repository pre-commit passes in
`checkpoint_hooks_final.log`; source/document diff whitespace checks pass with
the generated literal plugin patch excluded (its context lines intentionally
contain a single space). Serving-contract structure now includes the proposed
history interface as a `plugin_owned` row. The first structural check rejected
a Markdown fragment in its file-path field; the file-only record path passes
(`serving_contract_checkpoint_final.log`,82rows). This structural pass is not
stage acceptance. PC-04 remains unimplemented and pending exact authorization.

Local tt-metal implementation/evidence checkpoint:
`29c6436249c56f29ae6f36990d09191654505379` (base
`818f5e81574b25c1a0784051172e7f91dc96ae52`). Pre-commit and commit hooks pass.
No push. PC-04 remains proposed; this commit does not close stage9 or replace
its required final gates/review. Raw oversized inspection logs remain local.

## PC-04 continuation audit and independent control resume

The automatic continuation contains no exact approval for PC-04. Revalidated
that its record remains proposed, plugin worktree is clean, and no server or
TT device holder exists. Previous turn made progress (PC-01 checkpoints,
confirmed history witness, cleanup and PC-04 proposal); it was not a wait on a
live job. Stage completion remains unproven.

Independent required work can continue: resumed HF controls with the recorded
`--resume` command, bounded10800seconds, session4935, Python PID2316633
(parent timeout2316632). Confirmed live via ps and six REUSED log lines;
`hf_resume_status.json` records this observation, not a future completion claim.
Completed HF inputs/rendered prompts and model/revision/tokenizer/settings match
all corresponding TT controls exactly. Remaining prompt IDs3/4/5 are running.
Do not restart this job because an observation call times out: inspect PID/session.
The earlier no-background-job cleanup was true at its timestamp; this CPU-only
job was launched afterward and holds no TT devices.

An independent CPU-only AutoFix check is testing fixed-logit host sampler RNG
ownership through row/layout changes. It cannot implement PC-04 or infer the
missing original smoke request identity. Results will be recorded under
`host_rng_probe*`; no new shared interface is authorized by this continuation.

Independent CPU host RNG audit completed:24fixtures/408calls pass exact tokens
and per-request RNG state against native and solo controls;105existing focused
plugin tests pass. Parent reviewed the exercised InputBatch lifecycle and
sampler assertions. `host_rng_probe.py/.json/.log`, command script and existing
test log record all scope and limits. No host RNG leak reproduced; this does
not resolve original smoke attribution or the confirmed device/host switch.
No production source or new interface changed. The HF resume remains live;
the preceding goal turn made progress by starting that independent required
work, and this turn verified the same session4935/PID2316633 rather than
restarting it. PC-04 has now recurred across three goal turns, but a blocked
state is not appropriate while required independent control collection can
still make progress.

Acceptance command inspection: P3's packaged renderer already uses
apply_chat_template(tokenize=False) then explicit input_ids, so the earlier
Transformers5 BatchEncoding client bug does not affect that path. No runtime
P2/P3/P5 pass is inferred from this source check.

### PC-04 blocked audit after the independent CPU audit

PC-04 authorization remains absent in the original proposal turn and both
subsequent automatic continuations. The CPU sampler investigation is now
complete and found no repair to make. The exact proposal, ownership argument,
CPU indistinguishability witness and validation plan are concrete and recorded.
No presently available parent-side implementation can close this boundary
without that owner decision; future new HF results are an external completion
event, not authorization. Marking the goal blocked on PC-04 after the required
three-turn recurrence. This supersedes the earlier within-turn statement while
the independent CPU investigation was still active; it is not stage completion.

The bounded HF CPU job remains live (session4935, PID2316633, timeout10800) and
writes completed rows to hf.json itself. It may finish while awaiting the owner.
Do not infer completion from this log: on resume inspect that session/process
and the resulting12records before issuing a qualitative verdict. No TT devices
are held, and no server/EngineCore process is running. Do not kill or restart
this CPU job merely because the goal is blocked. The remaining stage gates are
unchanged and will resume after the exact PC-04 decision.


## 2026-11-16 PC-04 owner approval and implementation

The owner explicitly approved PC-04 exactly as proposed: default-False
`supports_prefill_sampling_history` and `original_prompt_lens` semantics.
Prior blocked/pending statements above are historical. Implementation resumed
with independent plugin CPU regressions and canonical generator penalty state.
No new sampling routing policy is included.

The bounded HF resume completed all remaining six cases and exited. All twelve
HF outputs have now been read: coherent, on topic, no mechanical repetition,
gibberish, language drift or request contamination. French is requested; both
Fibonacci functions are valid. Thermodynamics outputs reach only the start of
the second law at the fixed 256-token limit, as also seen in TT; this is budget
truncation, not a claimed complete answer. No HF process remains. Served quality
and matched final verdict remain pending.

PC-04 plugin host regression suite passes (183 focused,2713 full,16 skipped);
see `pc04_validation.log`. Canonical penalty probe passes under trace allocation
tracking: `penalties_probe_v4.json/.log`, five fixed signatures and zero
retirements. It covers positive/negative repetition, presence/frequency and
combined penalties, count masks, raw logits, release/remap, and identical
concatenated histories with different original prompt boundaries. Neutral
regression and affected live serving follow; no final capability acceptance yet.
All failed inner-loop attempts are retained and explained in `AUTOFIX.md`.

Reduced async32-slot live PC04 validation passed on the same server: route
companion258device/0host and seeded/greedy exact, mixed_params_batch21device/0host
and pass, lifecycle712device/0host with5stable signatures and0retirements,28
consumed allocator-grown columns,24page-only updates,659unchanged-table decodes.
Artifacts: `readiness_vllm/pc04_reduced_server/`. These are diagnostic reduced
model results, not full-model performance/quality. A top-k64 companion still
reproduces the independent lifetime-routing gap (first sampled difference4,
16host forwards). PC05 records an exact opt-in policy proposal; async owner
question submitted, no authorization or implementation.

Reduced server drained to zero active/waiting requests before API shutdown;
no vLLM/EngineCore/device holders remain from it. Four-chip arithmetic passed
without reset (`post_pc04_reduced_smoke.log`). Full64/B32 synchronous control
launched with tracking1/telemetry1/hostcompat1, no diagnostic layers, via
`launch_server.sh readiness_vllm/pc04_full_sync 32 --no-async-scheduling`.
All-layer preparation completed with5stable traces; penalty companion passes
258device/0host with exact seeded/greedy agreement. Targeted penalty tests run
next on that same server.

All-layer PC04 affected validation passed: penalty companion258device/0host,
all8penalty/isolation tests905device/0host (`pc04_full_sync/route*` and
`penalties_tests*`). Canonical sampler behavior and history reset now reach
actual full-model serving; these checks do not establish PC05 lifetime policy.
Plugin explicit final pre-commit passed (`pc04_hooks_before_commit.log`).
Local plugin commit `cf2ce6d499040f9bce0c0a0e263329da7771f1e7`; no push.

### PC-04 full sampling, 2026-11-16

`check_server.sh readiness_vllm/pc04_full_checks 32 sampling,qualitative full
readiness_vllm/pc04_full_sync/server.log` (paths relative to the model directory
except script) exited1 after full sampling:66passed,75skipped,1failed in610.54s.
The failing `TestSeedingAndVariety::test_topk[19]` had4/10 changed unseeded
pairs against a5/10 minimum; greedy agreement and within-batch variety passed.
The unchanged targeted rerun passed in7.21s (21device/0host forwards), recorded
in `pc04_full_sync/topk19_retry.log` and `_audit.json`. No test was weakened;
no final full-profile pass is claimed. The raw prompt is continuation stress,
not a chat-format quality verdict; no HF-equivalence claim is made for it.
The full profile's failure stopped the runner before qualitative collection,
so that stage is run separately against the same server.

PC-04 full-model qualitative collection completed separately on the same
synchronous server:12outputs read, no mechanical repetition/gibberish/language
drift/request contamination. Record minor story pronoun drift, imperfect
sampled haiku form, and token-capped explanatory answers rather than claim
perfect instruction fidelity. All6greedy texts equal prior full-generator
controls after terminal chat-token removal. `/tokenize` matches all6rendered
prompt-token sequences. Degeneracy check passes. Exact artifacts under
`readiness_vllm/pc04_full_checks/`: `vllm_qualitative_outputs.json`,
`qualitative_prompt_format.json`, `qualitative_verdict.json`,
`degenerate_output.json`. Fresh same-code TT control is outstanding.

Updated17-request mixed-shape/lifecycle check passes on that same server in
51.98s:5prepared signatures,0retirements, program cache24885 unchanged,
774device/0host forwards,29consumed allocator-grown page columns,
24page-only updates,722unchanged-table decode steps. Actual changed chunk
starts/partial tails with concurrent decode and nonaligned129/130/131 lengths
are in `pc04_full_sync/lifecycle.json`; audit/log retain command and interval.
Workload17requests including1027/1539chunked prompts: TTFT P50/P95/P99
522.31/19128.49/19130.85ms, ITL P50/P99/max56.72/364.04/11905.93ms,
throughput0.3270completed requests/s. Diagnostic sync/tracking regime, not
primary decode throughput. This is the new synchronous control for final async.

PC-05 all-layer confirmation: `pc04_full_sync/topk_route.json` admits top-k64
companion after first streamed token of top-k20 request; seeded output differs
atindex2, greedy identical,242device/16host forwards. PC-05 remains proposed
and unimplemented; approval request is pending. No new policy is authorized
by PC-04. The server drained (running=waiting=0) before SIGTERM to owned API
2573163; shutdown evidence is in that run directory. No reset issued.

Final PC-04 cleanup: serve-only wrapper exited1 because its server was
intentionally stopped. Upstream shutdown manager force-killed EngineCore during
teardown after zero active/waiting requests; `cleanup_pc04.json` verifies no
vLLM/EngineCore or device holders remain. `post_pc04_full_smoke.log` passes
open/add/read/close on all4chips without reset. `lifecycle_summary.json` beside
each raw lifecycle result retains requests, trace signatures and chunk events;
raw multi-megabyte page-table telemetry remains unchanged locally and is listed
in `pc04_artifacts.json`. Repo file-size policy excludes oversized raw files
from Git; no evidence was compressed, hashed or deleted.

Checkpoint formatting checks: `pc04_model_hooks_before_commit_retry.log`
passes all source checks after Black formatted the standalone teacher-forcing
guard. Whole-checkpoint hooks add terminal newlines to two JSON artifacts;
JSON content is unchanged. Raw console logs with trailing whitespace remain
outside Git unchanged, alongside oversized raw artifacts, with exact paths
and reasons in `pc04_artifacts.json`. No remaining server processes/devices.
PC05 is the authorization boundary for continuing integration; stage remains
active/incomplete and no stage-review pass is claimed.

Local PC-04 checkpoint: tt-metal
`f7de43f9c2a14f5e8a4fa966610997673c259ce2` (canonical device history,
adapter capabilities and validation artifacts); plugin
`cf2ce6d499040f9bce0c0a0e263329da7771f1e7` (approved shared metadata).
Whole-checkpoint explicit pre-commit and commit hooks passed. No push.
Documentation-only follow-up records these SHAs. Remaining raw local evidence
is enumerated in `pc04_artifacts.json`; no outstanding source edits.

### Continuation: same-code standalone control and capture fix

PC05 authorization is still pending. Independent current-code control work
found a real PC04 regression: `qualitative_controls/tt_pc04.log` exited1 during
standalone prefill_128 capture, before outputs. Public release_request had
acquired a host-writing sampling-history reset. The captured graph now calls
the factored device-only recurrent reset; upload clears sampler history outside
capture, while public release preserves both operations. Read-only review
confirmed all replay callers upload first. See AUTOFIX.md for the stack.
No serving policy or plugin change was made.

`probe_prefill_penalty_trace.py` passed on real layers0/3 with tracking1 and
program-cache tracking enabled:5captures,0retirements,249stable programs,
dirty counts cleared before prefill, exact repeated neutral/penalty output,
and allocation positive control. Fresh all64 retry uses the same source at
local fix commit `00cf43edc62990114938adc55bb680792a6a4525`, tracking enabled,
256tokens and the prior controls' exact settings; command/provenance are in
`qualitative_controls/tt_pc04_retry.command.json`. No push. The legacy runtime
boundary audit's expected trace set also needs the new sample_penalties entry;
adding that required key preserves its host-boundary checks.

Fresh all64 control retry completed with exit0 and all12outputs, on same1x4
chips, selected precision and256-token budget. Tracking stayed enabled through
all captures and generation. Five traces captured, zero runtime retirements;
24760programs at preparation, five orderly shutdown releases. All12outputs
were read: coherent/on-topic, no mechanical repetition, gibberish, language
drift or request contamination. Minor pronoun/article issues and token-cap
endings are recorded. All12token sequences exactly match the prior TT control
through chat EOS; all6greedy texts exactly match served outputs. HF/control
prompt metadata matches for all12rows. Served sampled streams are unseeded
versus control seed0, so only quality—not stream equivalence—is compared.
Artifacts: `readiness_vllm/qualitative_controls/tt_pc04_retry.json`,
`tt_pc04_retry.command.json`, `tt_pc04_retry.log`, `tt_pc04_verdict.json`,
`tt_pc04_cleanup.json`. fuser confirms all devices free, no reset needed.
PC05 still requires exact owner approval; no routing-policy change implemented.

### PC-05 authorization blocked audit, 2026-11-16

The preceding goal turn made progress: fix00cf43ed and evidence checkpoint
`f1f3511ac3fa9ca64d18b441d5f57712411223e7` complete the same-code control and
repair standalone trace capture. This continuation revalidated the remaining
boundary. PC05 exact owner approval has been absent for three consecutive goal
turns (initial PC04 implementation turn and two automatic continuations).
Plugin source still accepts only auto/host; plugin worktree is clean and PC05
remains proposed. Full64/B32 `pc04_full_sync/topk_route.json` still proves the
seeded request differs at token2 when a host-only companion joins, with
242device/16host forwards. The twelve current-code controls pass. No devices
are held and no job remains to wait on.

Blocked audit passes: the required shared routing behavior cannot be implemented
under existing PC01/PC04 authorization. Final serving acceptance must verify
the corrected request-lifetime policy; rerunning unchanged final gates cannot
close this confirmed contract failure. Set goal blocked pending exact approval
of PC05 item5: opt-in tt.seeded_sampling_policy=request, lifetime sampler owner
and fair homogeneous cohorts; auto/host unchanged. Proposal and planned
validation remain in plugin_changes.md. Stage9 is not complete. No push.

### PC-05 owner approval and resumption (2026-11-16)

The owner approved the exact PC-05 proposal: opt-in request-stable sampler
ownership via `tt.seeded_sampling_policy="request"`, default off, other models
unaffected. Stage9 resumes with AutoFix source audits, host regression tests,
then representative serving before full-model acceptance. Prior failure and
blocked records remain historical evidence; no acceptance gates are waived.

PC-05 implementation now has71 dedicated CPU regressions (43 classifier/runner,
28 scheduler), all passing in `pc05_new_tests.log`. The wrapper-bypass witness
fails16 mixed-owner cases as expected (`pc05_cohorts_before.log`).536 existing
focused regressions pass. Source-only audit: `pc05_source_review.md` (not a stage
review). The first full suite loaded a test fixture before its missing
`supports_chunked_prefill=True` declaration was corrected; it reports2783pass,
16skip,1fixturefailure in `pc05_host_suite.log`. The final71-case run includes
the corrected fixture. A complete-suite rerun uses frozen source/tests.

Operator launch helpers now explicitly set approved `seeded_sampling_policy`
`request`; an optional final argument `auto` retains the unchanged-policy
comparison path. Both helpers record the selected policy in typed TT config.
The direct admission probe adds logprobs/logit-bias/min-tokens companions,
both arrival orders and simultaneous admission, plus an explicit failing exit
when `--require-stable` detects drift in either the active or companion request.
Historical diagnostic mode and artifacts remain intact.

PC-05 plugin committed locally as `da1587235b84b13f2826cfd7e8b448b340322a54` after explicit staged pre-commit.
No push. Full host suite2784pass/16skip. Reduced15admission cases and17-request
lifecycle allpass; five stable signatures,0retirements,777device/0host lifecycle
steps. Drained running/waiting counters0; API2825127/EngineCore2825565 exited,
fuser empty, no reset. Final all64 asynchronous acceptance starts with the same
policy/cache/context and allocation tracking. User stage-cap instruction now
limits work to one remaining acceptance pass; untried ideas will be handed off.

### Owner stage cap and untried work (2026-11-16)

The owner ended optional experiments and requested one pass through the remaining
required gates. Preserve the implemented PC05 checkpoint, run final all64 async
acceptance and a separate tracking-disabled performance pass, then review the
actual results. Failed correctness gates are not relabeled as passes.

#### Carried to next stage: untried ideas

- Measure homogeneous `auto` versus `request` policy overhead on identical
  server/workload settings. CPU regression coverage establishes unchanged
  default behavior; no same-harness performance equivalence is claimed.
- Consider narrowing lane overlap predicates to the active sampler cohort when
  a paused host-only request exists. The current conservative predicate can
  reduce mixed-traffic overlap; pure-device traffic does not have that condition.
- Batched prefill remains disabled. The final serial burst is the baseline;
  do not claim an off/on speedup. Adoption requires current-code batched-equals-
  solo evidence and the same-workload performance comparison.
- Wider-than32 device sampling requires a real multi-tile sampler, candidate
  membership tests and its own correctness work. Merely raising the declared
  ceiling is unsafe. Explicit host compatibility handles such requests today.
- Chunk-budget tuning and long-answer throughput are stage10 work; retain512
  for this acceptance. No new sweep is attempted after the owner's cap.

Final acceptance command is `source doc/vllm_integration/env.sh` (relative to
this model), then `python doc/vllm_integration/run_final_acceptance.py` from the
checkout with full model paths. Exact expanded commands and exit codes are in
`readiness_vllm/pc05_final_async/acceptance_commands.json`.

Final full sampling passes67tests with75expected framework/capability skips.
All12 qualitative outputs read; all6greedy texts exactly equal same-code TT
controls. No repetition/gibberish/language drift/request contamination. P3 passes
all15checks, including thinking and tools.

P2 accepted every profile, but `ratio_lm_eval_reasoning_32u` fails: top_k40 uses
the explicit host route (real device ceiling32),356.8ms/token versus83.5ms plain,
ratio4.28. Its1-user ratio1.62 is advisory. Non-reasoning, model-card and agentic
profiles remain on device and pass load ratios. Raw result:
`readiness_vllm/eval_probes/final_pass/harness_requests.json` and its log.
The owner cap forbids further experiments. No fictitious2-hour repair time or
HF-equivalence waiver is claimed; this required gate remains unresolved pending
final review. Continue P5, max context and the required performance pass once.

P5 passes all9needle cases at8K/32K/128K and depths.1/.5/.9; P3 passes all15checks.
The one full-context request passed:262111prompt+33output=262144total,
prompt modulo32=31,544device/0host sampling operations. No context reduction.
`pc05_final_async/max_context.json` verifies usage and all33returned token IDs.
Its535.46s client duration/TTFT includes slow byte-at-a-time parsing of a large
prompt-token-ID SSE event; the server had already completed and was idle while
the client drained. This is capacity evidence, not a performance claim. Carry
larger client stream chunks as a harness improvement; do not rerun the request.

Tracked server API2844638/EngineCore2849732 drained to0running/0waiting and
exited after SIGTERM; fuser empty, no reset. `pc05_final_async/cleanup.json`
records mode counts and OOV audit. All required acceptance phases ran once;
only P2's ratio remains critical. Performance server `pc05_perf` starts with
tracking0 and detailed telemetry0; per-step sampling-mode audit remains enabled.

Shutdown audit clarification: upstream vLLM's abort timeout is0s. After the
API received SIGTERM with no active requests, its manager logged SIGTERM then
force-killed EngineCore2849732. Both exited and all devices were released;
this is not evidence of graceful adapter trace teardown. The same unchanged
upstream policy will be audited at final performance shutdown. No reset used.

### Final tracking-disabled performance pass

`pc05_perf/benchmark_audit.json` records the exact attached command and48.30s
client-phase duration. All profiles completed;282device/0host forwards, no OOV
or error lines. Both allocation tracking and detailed telemetry disabled.

- Primary128input/128output/1request/concurrency1: TTFT P50/P99282.487ms;
  TPOT mean/P50/P9926.492ms; ITL P50/P9926.491/27.250ms; aggregate35.094tok/s;
  1000/meanTPOT=37.747t/s/u. Raw`readiness_vllm/vllm_result.json`.
- Secondary CI100/100/32, uncapped admission: TTFT P50/P993788.295/6170.506ms;
  TPOT mean/P50/P99101.832/101.343/125.124ms; ITL P50/P9956.153/1147.959ms;
  aggregate226.867tok/s; TPOT-derived9.820t/s/u (not headline decode).
  Raw`readiness_vllm/vllm_ci_serving_result.json`; normalized config/metrics in
  `vllm_ci_serving_benchmark.json`.
- Serial prefill128/32/32/concurrency32: TTFT P50/P994070.534/6784.091ms;
  TPOT mean/P50/P99169.854/166.250/252.033ms; ITL P50/P9956.099/1594.357ms;
  aggregate106.578tok/s; TPOT-derived5.887t/s/u. Raw
  `readiness_vllm/vllm_prefill_burst_result.json`. No batched speedup claimed.

Performance API2965022/EngineCore2965406 drained then exited; fuser empty,
no reset. `pc05_perf/cleanup.json` discloses upstream zero-timeout abort and
force-kill. Readiness serve-only wrapper exits1 on intentional shutdown;
attached sampling/qualitative and benchmark runner phases each succeeded.
Headline metrics are at the top of README, workload shapes beside every row.

### Aggregate gate and metadata correction

Ran the aggregate09-vllm.check.sh once, saved`readiness_vllm/stage9_gate.log`,
exit2. Degeneracy passes; serving contract84rows passes; serial prefill's absent
off/on ratio is advisory; P3/P5pass; P2 remains critical as recorded above.
The context checker additionally misread summary command-return-code key
`max_context:0` as a context cap. Renamed those process results to a list of
`phase`/`exit_code` records. No served configuration or request changed.
One CPU-only context metadata check then exited0, confirming target/supported
262144 (`readiness_vllm/context_metadata_check.log`). Its advisory quotes the
preserved historical false-positive log. No serving gate or measurement rerun.
All required evidence runs are now finished; fresh independent review follows.

### Local checkpoint under the owner cap

Preserved the best tested implementation and completed acceptance/performance
artifacts in local tt-metal commit
`98aa9992e837570df6ed57ae9455a32a0dee9f03` (no push). The serving plugin is
locally committed at `da1587235b84b13f2826cfd7e8b448b340322a54` (no push).
The explicit staged pre-commit invocation and commit-time hooks both passed.
`pc05_checkpoint_hooks_retry.log` records the explicit passing invocation.
The first invocation only removed an extra EOF blank line from a raw historical
hook log; restored its original bytes and excluded it from the index, then
reran mandatory commit hooks. No acceptance test or hardware experiment repeated.

The 153-entry manifest verifies exact hashes for final and reused evidence.
Raw files that exceed the repository size limit or require whitespace/newline
rewrites remain at their recorded workspace paths with manifest hashes; their
bytes were preserved. This is local filesystem persistence, not an external
backup claim. Required source, reports, normalized metrics, and suitable raw
evidence are committed. No vLLM/EngineCore process retains a device.

### Independent stage review and capped disposition

Fresh xhigh stage-review round 1 returned **more-work-needed** in
`stage_review_round1.md`. Its sole blocking finding is P2's unwaived
32-user reasoning ratio 4.275190600 against 1.15. It independently checked
raw lifecycle token controls, qualitative TT controls, benchmark arithmetic,
registration/cache/precision and device-only performance routing. No additional
blocking correctness defect was found. Review performed no new hardware runs.

Reconciled the review's stale status sentences in `plugin_changes.md` and
`doc/full_model/serving_contract.json` with final acceptance and async evidence.
Optional policy comparison, prefill/overlap tuning, sampler widening and client
parsing ideas remain untried hand-off notes above. Sampler widening may address
P2, but the required failed gate itself cannot be carried as non-blocking.
No fictitious waiver, repair effort, clean-pass or pass-with-carried is asserted.
The owner cap ends experiments with the current checkpoint; stage 9 remains
not passed. Goal completion is not claimed while this required gate is open.


### Owner-resumed P2 repair (2026-11-16)

The owner resumed stage9 and authorized cleanup/reset of owned leftover jobs.
Initial process/device audit found no holders; tt-smi listed all four devices;
no reset was needed. `$autofix` and independent `$autodebug` investigated the
sole round1 blocker. Details and failed hypotheses are in
`p2_repair/AUTOFIX.md` and `p2_repair/AUTODEBUG.md`.

The canonical shared sampler now supports opt-in64 candidates. The legacy
TTNN default32 remains unchanged. TTSampling collects64 per TP shard,
compacts the union to64 on device while preserving token IDs, and uses the same
TTNN sampling operator and persistent token-out feedback. The64 mode explicitly
requires width64, preserves the complete two-tile sort, and traverses both tiles
with matching masks, wait/pop counts and face offsets. No adapter sampler,
host argmax, full-logit readback or Python token feedback was added. The
Qwen adapter's documented max_device_top_k is64. The late unsupported companion
probe now requests128; the previous64 companion is supported after this fix.
The existing public-temperature upload also now converts T to1/T exactly once.

Build: `cmake --build build_Release --target ttnn -j8`; initial `_ttnn`
target was invalid and is retained in the log. Updated local `_ttnn.so` and
`_ttnncpp.so` runtime copies from the build outputs. Source hooks pass.
`host_probe_retry.log` passes default32/cap64, reciprocal T, logical reload and
invalid65 rejection. `legacy_width_retry.json` proves the old mode never draws
ranks33–40. `width64.json` passes distribution, mixed k/p/T, seeded eager/trace
replay and an allocation-tracker positive control. `adapter_width64.json`
passes real reduced TP4 layers0/3, external8196-page cache, seeded replay,
inactive rows, greedy and penalty sibling, five stable traces and fixed
recurrent token-output addresses, with no full-logits readbacks. Both hardware
probes included program-cache allocations in tracking and closed devices.
The first single-op attempt with TT_VISIBLE_DEVICES=0 failed topology discovery
before opening a device; the recorded retry opened device0 with the full
physical topology visible. Independent source review found no concrete defect.

The original failed gate and overwritten raw/root artifacts are preserved in
`p2_repair/prior_artifacts/`. The owner-resumed full-model tracked server is
`readiness_vllm/p2_final_async`, same all64 precision/context262144/max-seqs32,
serial prefill, page32, chunk512, async and request-stable ownership policy.
P2 is rerun without profile/user/token/threshold overrides. Current status and
subsequent outcomes are recorded below; this entry alone does not claim a pass.

P2 completed with exit0 and all15 checks passing. Reasoning top-k40/T1/p.95/
seed42 at32users measured78.238609ms TPOT versus82.333068ms plain,
ratio0.950269573 (limit1.15); the old ratio was4.275190600. Other32-user ratios:
non-reasoning1.036004900, model-card0.946781826, agentic0.949326704.
Each run completed its requested128tokens per request. All one-user ratios
also pass. `p2_final_async/p2_audit.json` records2617device/0host operations,
from startup through the end of P2, before lifecycle begins. The shared runner's
DEVICE_LOG regex misses the adapter's operation=... mode=device marker and
reports0 device lines; the byte-scoped raw audit supplies actual route proof.
The resolved failure was removed from active limitations; its unchanged record
remains in prior_artifacts, with no waiver claimed.

Affected current same-server checks also pass: lifecycle17 requests,16 exact
synchronous greedy controls,775device/0host forwards, unchanged5 trace IDs,
0 captures/retirements during workload,24888 stable program-cache entries,
actual chunk starts, concurrent decode,31 consumed allocator-growth examples,
26 page-only updates and718 unchanged-table steps. Current fixed-logit and
late top-k128 host-companion checks pass. All use the original public async
adapter path; current full shared sampling/qualitative acceptance is ongoing.

Full shared sampling completed:67passed/75skipped in543.59s, no failures.
The unchanged exclusions are71 speculative cases,3 test-budget-flag-dependent
chunk cases and1 all-vocabulary logprob case; current lifecycle independently
covers real mixed chunk/decode behavior. During this run, a pre-commit invocation
stashed the tracked log pathname while pytest still held its original stdout
inode. Held `/proc/4030146/fd/5` open and copied its exact bytes through completion
back to the intended pathname. `p2_repair/log_recovery.json` records the original
inode and18201 recovered bytes. No output was reconstructed or edited and no
test rerun was needed. Future hook runs wait until artifact writers exit.

Current full qualitative runner succeeded (12outputs). Read all texts and
recorded `readiness_vllm/qualitative_verdict.json`: coherent/on-topic, no
mechanical repetition, gibberish, unrequested language drift or contamination.
Six greedy outputs exactly equal the retained TT controls after terminal-token
removal. Both stories have a pronoun inconsistency; greedy matches its control.
Verbose answers truncate at256tokens (thermodynamics is incomplete). Sampled
T.7 now has correct reciprocal scaling; no exact unseeded/control comparison or
population-level quality claim is made.

Current all64 runtime config equals the complete selected precision JSON;
`p2_final_async/runtime_contract.json` confirms64 layers, adapter registration
and external8224-page cache. P3's15checks and P5's9needle cases remain from the
completed final_pass: both use greedy; model/cache/positions/precision unchanged,
and current greedy controls and mixed-shape tests match. The once-only
262111+33=262144 capacity request remains valid: fixed cache/layout/context
unchanged and the current server allocates that same pool plus its fully
prepared sampling buffers. It is not repeated or used as timing evidence.

Acceptance API4000628/EngineCore4002125 drained to0running/0waiting and exited;
fuser empty, no reset. `stop_server.py` records exact owned PIDs/command,
metrics and mode counts in `p2_final_async/cleanup.json`:7592device/516explicit
host-compatibility forwards, no OOV. The host total is diagnostic/shared-test
coverage; P2 and lifecycle audits have zero host forwards. Upstream again
force-killed EngineCore during zero-timeout shutdown; graceful teardown remains
unproven. The attached full-profile sampling/qualitative runner exited0; the
serve-only wrapper exits1 on deliberate shutdown, as before.

Performance refresh runs in separate `readiness_vllm/p2_perf`, allocation
tracking0/detailed telemetry0, decode traces enabled, otherwise same serving
configuration. Production source hooks passed; helper import sorting was applied before measurement and is rechecked at closeout. No further
source change or optimization experiment is planned.

Final performance refresh passed,48.267s attached phase;282device/0host operations, no errors.

- primary: workload{'prompt_len': 128, 'output_len': 128, 'num_requests': 1, 'concurrency': 1, 'temperature': 0.0, 'generation_config_mode': 'explicit', 'dataset_name': 'random', 'ignore_eos': True}; TTFT P50/P99275.746611/275.746611ms; TPOT mean/P50/P9926.572380/26.572380/26.572380ms; ITL P50/P9926.567944/27.368671ms; output35.060375tok/s; TPOT-derived37.633061t/s/u. Raw`/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/readiness_vllm/vllm_result.json`.

- ci: workload{'prompt_len': 100, 'output_len': 100, 'num_requests': 32, 'concurrency': None, 'temperature': 0.0, 'generation_config_mode': 'explicit', 'dataset_name': 'random', 'ignore_eos': True}; TTFT P50/P993759.765476/6121.489942ms; TPOT mean/P50/P99100.748347/100.262387/123.879620ms; ITL P50/P9956.229123/1141.898990ms; output229.093610tok/s; TPOT-derived9.925721t/s/u. Raw`/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/readiness_vllm/vllm_ci_serving_result.json`.

- prefill: workload{'prompt_len': 128, 'output_len': 32, 'num_requests': 32, 'concurrency': 32, 'temperature': 0.0, 'generation_config_mode': 'explicit', 'dataset_name': 'random', 'ignore_eos': True}; TTFT P50/P993985.526106/6704.322407ms; TPOT mean/P50/P99169.077425/166.326086/248.821084ms; ITL P50/P9956.262361/1584.187115ms; output107.532538tok/s; TPOT-derived5.914450t/s/u. Raw`/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/readiness_vllm/vllm_prefill_burst_result.json`.

CI and prefill-burst metrics are secondary; the primary128/128/1 result is the headline. Performance API4064752/EngineCore4065169 drained and exited; fuser empty, no reset. Shutdown again used upstream force-kill; not graceful trace teardown evidence. No hardware job remains. Untried carry items: homogeneous auto/request policy comparison, cohort overlap, prefill batching/chunk-budget tuning and faster capacity-client parsing. Wider sampling is now implemented and P2 is closed, not a carry item.

Aggregate09-vllm.check.sh completed once for the repaired snapshot, exit1:
no degenerate outputs; context262144OK; all84 serving-contract rowsOK; eval
readiness passes P2/P3/P5. Advisories are the unadopted batched-prefill serial
comparison and historical prose quoting the previous false context-cap parse.
These do not represent a current context reduction or failed required gate.
`readiness_vllm/stage9_gate.log` is current; the old failed log is archived.
Initial final-source/helper hook passes applied isort-only formatting; their
shell tail output masked the hook exit status. Production sampler/kernel hooks
had already passed. Final closeout reruns hooks with the actual exit preserved;
no semantic implementation or hardware rerun is needed for import ordering.

Local repair/source/evidence commit:
`450c36d7c4864f69f2e8365b7c62e4ff3506e495`.
Explicit final closeout hooks and commit-time hooks both exit0; exact logs in
`p2_repair/closeout_hooks.log` and `p2_repair/commit.log`. Plugin remains clean
at`da1587235b84b13f2826cfd7e8b448b340322a54`; upstream vLLM unchanged. Nothing
pushed. Raw oversized/untracked runtime artifacts remain local at manifest
paths; managed source, tests, summaries and suitable raw JSONs are committed.
Fresh xhigh stage-review round2 is inspecting only round1 findings and the
changed code/evidence, with all gates finished and devices free.


### Final review closeout — 2026-11-17

Fresh independent xhigh round2 returned **clean-pass** in
`stage_review_round2.md`. It closed round1's sole required P2 finding using the
unchanged top-k40 profile and independently checked the canonical wider sampler,
actual device-route counts, exact greedy controls, all12 qualitative outputs,
trace lifecycle, refreshed benchmarks and cleanup. No required work remains.
Stage9 is complete; the aggregate exit1 advisories are documented and non-blocking.
No additional device test or experiment was run for this documentation closeout.

The preserved `p2_final_async/sampling_qualitative_run_summary.json` contains
merged stage metadata, including historical benchmark/prefill entries. Actual
phase provenance is `acceptance_commands.json` and the completed phase log;
current benchmark provenance is `p2_perf`. This is metadata scope, not a missing
acceptance phase, and is now explicit in README.

Untried hand-off topics remain optional: homogeneous auto/request policy
comparison, cohort overlap, prefill batching/chunk-budget tuning, faster
capacity-client parsing, graceful EngineCore teardown, and broader architecture
or soak coverage. No further experiment is required by the review.

Local source/evidence SHA is `450c36d7c4864f69f2e8365b7c62e4ff3506e495`;
plugin SHA is `da1587235b84b13f2826cfd7e8b448b340322a54`. This review and final
status reconciliation are committed separately as documentation only. Final
closeout hook/commit logs are `p2_repair/review_closeout_hooks.log` and
`p2_repair/review_closeout_commit.log`. Nothing is pushed. Both serving process
pairs have exited and the recorded final device-holder audit is empty.
