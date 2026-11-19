# AutoFix: serving contract gaps

## Failing contract and observations

A continuation prefill must read the same recurrent state it previously wrote.
The CPU reproducer executes the pinned plugin's allocator with decode in slot0
and continuations in slots2/1. It reassigns them to1/2 and sends no prefill
state-remap. Three real plugin regression cases fail on the original source.
Separately, intermediate chunks force host sampling and the model receives no
row-finality information. This second behavior matches the documented plugin
interface but cannot satisfy this stage's stricter device-sampling requirement.

The independent AutoDebug invocation is preserved in `autodebug_retry.log`;
the completed report is preserved as `AUTODEBUG.md` in this directory. Source-level
claims were also checked against the pinned plugin and exact CPU method runs
in `probe_plugin_contract.json`.

## Hypotheses and discriminating checks

1. Preserve existing prefill slot ownership in the plugin, reserving all
   continuing slots before placing new requests. Check: reordered, single,
   and mixed new/continuing prefills retain their old slots, with no aliases.
2. Move recurrence inside the adapter. Rejected: the adapter has destination
   slots only, not prior ownership or request IDs. Guessing from physical pages
   is unsafe because the scheduler reclaims/reassigns them.
3. Split mixed prefill batches using existing calls. This could keep final
   rows sampled on device, but intermediate calls still require host logits;
   it does not close the stage's no-logits-transfer contract.
4. Use a documented opt-in final-row sampling mask. Correct owner is the
   plugin; PC-01 received exact owner authorization on 2026-11-16. Implementation
   now validates the mask in the low-level generator and preserves non-final sampler state.

## Kept candidate repairs

Plugin `src/vllm_tt_plugin/model_runner.py`: reserve all live claimed slots and
reuse a continuing request's claimed slot. Only new requests choose a free row.
`docs/MODEL_LIFECYCLE.md` clarifies that prefill has no remap command and keeps
continuation ownership. Three tests added to `tests/test_block_model_runner.py`.
No capability, configuration option or new hook was added.

A separate collection failure came from `tests/tt/test_tile_size.py` directly
importing the platform during vLLM initialization. Explicitly import vLLM first;
all test assertions remain unchanged. PC-03 records this test-only repair.

## Validation

- `slot_regression_before.log`:3/3 new cases fail on pinned source.
- `plugin_host_baseline.log`:2303 passed,16 skipped,3 new cases deselected.
- `plugin_host_fixed.log`:2306 passed,16 skipped,0 failures; all new cases pass.
- `sampling_collection_final.log`:141 collected with one import-cycle error.
- `sampling_collection_fixed.log`:142 collected, no collection error.
- `slot_regression_final.log`:4 focused cases pass, including a physical-history
  ledger after the competing decoder finishes.
- `plugin_hooks_final.log` and `plugin_hooks_history_final.log`: formatter/linter results.

Commands are in `work_log.md`; all runs are CPU-only using plugin host stubs.
These results prove host ownership and collection behavior, not device model
correctness, async correctness, performance, or mixed-shape serving acceptance.

## Outcome and remaining risk

Host defect repairs pass. No AutoFix failure is claimed. PC-01 was explicitly
approved by the owner on 2026-11-16; implementation and its validation resumed. Serving must still prove recurrence preservation with actual
requests and all original goal gates. Candidate plugin edits are uncommitted
until their affected serving validation and stage review are complete.

## Live-server adapter mRoPE mismatch

The reduced server completed full preparation and reported healthy. Its first129-token request executed device prefill and then failed in plugin submit_prefill while unpacking `(output, rope_deltas)` (`readiness_vllm/reduced/server.log`). This is a model adapter gap, not a plugin defect: MODEL_EXECUTION.md explicitly documents the tuple when ModelConfig.uses_mrope is true. Qwen3.6 keeps `mrope_section` metadata in text-only mode. The adapter now returns per-row zero text-only deltas and rejects nonzero decode deltas; the generator execution/sampling path is unchanged. The direct adapter probe now checks the tuple. A focused served request is the discriminating validation before resuming the lifecycle workload.

After the server exception, all run processes exited and no devices remained held. The bounded post_server_smoke.log passed (chip0 open/add/read/close); no reset. Reduced retry uses the shared launch_server.sh with the same context, pool, sampling and trace config.

## Synchronous readback boundary (2026-11-16)

Full64/32slot synchronous control completed preparation and trace capture, then failed on its first decode in `process_decode_output_host`: a replicated multi-device tensor reached `ttnn.to_torch` without a composer. The pinned plugin only invokes `read_decode_output` for `async_read=True`; synchronous finalization passes the device object directly to host formatting. Reduced previous serving was async and did not exercise this boundary.

Minimal adapter repair: if the formatter receives device storage, call the existing `read_decode_output` first. That helper selects replica0 for token output; logits retain their existing mesh-concatenation path. Host outputs from async readback do not incur a second copy. Independent CPU review confirmed the causal chain and async safety. The strengthened representative probe now directly formats a known device-token output without first calling readback. Hardware verification is running in `adapter_prefix_final.log`.

The failed server and EngineCore exited, releasing all devices. `post_sync_server_smoke.log` passed four-chip add/read/close without a reset. HF CPU controls resumed. The failed control artifacts remain under `readiness_vllm/full_sync_serial/`; they are not a passing non-overlapped baseline.

`adapter_prefix_final.json` and `.log` passed (exit0): direct synchronous device-token formatting, independent async reads, prefix active/inactive state byte equality, deferred full-page uploads flushed by unchanged-map prefill, stale inputs and tracking positive control. No full-logit reads occurred. The reduced target is layers0,3 with serving context262144 and8196physicalpages; it does not substitute for the pending full-model control.

## Sampling ownership and missing prefill history (2026-11-16)

The corrected full synchronous lifecycle and logprob controls pass. Sampling
smoke fails mixed-params isolation. A separate identity-preserving live probe
confirms a supported seeded request changes device→host→device when a
presence-penalty companion joins/leaves; first sampled difference is index2,
the first host token. Its greedy control is exact. The original smoke's first
device-prefill request cannot be identified uniquely from the logs, so this
does not fully attribute that earlier mismatch.

Fresh inspection-only AutoDebug completed successfully; report preserved in
`sampling_AUTODEBUG.md`, launcher output in `sampling_autodebug.log`; the root
prior-stage report was restored. Parent verified the cited sampler rejection,
batch-wide routing, prefill arguments and combined replay source. Independent
CPU review confirmed the original-history boundary is absent in both layouts.

Hypotheses and checks:

1. Stale feedback/state as a general explanation: lifecycle, page growth and
   exact first-token logprobs argue against it, but do not exclude every later
   sampled prefix. Preserve this uncertainty; no broad numerical waiver.
2. Model-owned device penalties remove the known penalty fallback trigger.
   Required work: canonical penalty parameters/counts, masked counted sampling,
   release/remap/reload history, raw-logit preservation and counted warmup before
   capture. Do not flip the capability before the complete contract passes.
3. Existing metadata can reconstruct resumed output counts: refuted by
   `probe_prefill_history.py/.json/.log`. Actual builders/submission deliver
   identical adapter arguments for `[0] + generated[1,2]` and fresh `[0,1,2]`.
   Correct counts differ. This CPU witness is not a device penalty test.
4. New routing policy/cohort splitting: not implemented. It would change a
   shared contract and cannot substitute for the model's missing penalties.
   Host-only features still leave request-lifetime routing unresolved.

The next necessary shared interface is recorded as proposed PC-04, with exact
semantics and validation. It provides the original prompt boundary only.
PC-01 approval does not cover it. AutoFix has identified an authorization
boundary, not exhausted experiments or declared the failed gate acceptable.
No PC-04 code exists. After approval, implement metadata plus canonical model
penalties, run fixed-logit/history/mask regressions and reduced-device checks,
then the focused live companion/unchanged sampling test before full acceptance.

### Fixed-logit host RNG ownership hypothesis

`host_rng_probe.json` passes24fixtures/408sampling calls using actual InputBatch
add/remove/condense/readmission, runner generator selection and the optimized
and ordinary CPU samplers. Exact tokens and committed RNG states match both
ordinary upstream sampling and independent solo-request controls. Coverage
includes FP32/FP64 noise, vocab4101/248320 (and2048 fallback), greedy companions,
row reordering/subselection, intermediate-prefill generator clones, cache
capacity0/1/8, fresh ID reuse and explicit state changes. Completed speculative
prefetch does not advance live request RNG. Existing focused plugin tests pass
105cases (`host_rng_probe_existing_tests.log`). Commands and script retained.
The inspected implementation symbols match the pinned plugin AST.

Parent inspected the assertions and state comparisons. This refutes a simple
host RNG ownership leak for these fixtures; it does not prove the live original
smoke's logits/metadata/penalties or identify its singleton. No production repair
is justified by this check. The next live same-route investigation requires
captured request identity, effective parameters and same-prefix logits, as the
AutoDebug report specifies. PC-04 authorization remains pending.


## PC-04 approved: canonical penalty history repair (2026-11-16)

The approved metadata addresses the indistinguishable fresh/replay prefill
arguments established in `probe_prefill_history.json`. Plugin host regression
commands/results are in `pc04_validation.log` (183 focused and 2713 full-suite
passes,16 skips). No sampling-routing policy was changed.

Generator repair: reuse `SamplingGenerator._run_sampling` with a second prepared
`sample_penalties` trace; the original neutral trace has no added vocabulary
copy/count work. `tt/sampling_history.py` owns persistent staging for canonical
TTPenalties buffers, masks count increments by emitted rows, preserves raw logits,
replaces authoritative history at prefill/decode transitions, and remaps/releases
request state. The adapter translates the same penalty parameter fields.

Discriminating tracked device test:
```bash
source models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/env.sh
export QWEN_VLLM_DIAGNOSTIC_LAYERS=0,3 TT_METAL_TRACE_ALLOC_TRACKING=1
export QWEN_VLLM_HOST_COMPATIBILITY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
timeout 1200 python -u models/autoports/qwen_qwen3_6_27b/tests/probe_vllm_penalties.py \
 --output models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/penalties_probe_v4.json
```
`penalties_probe_v4.log/.json` pass six synthetic arithmetic cases, raw-logit
preservation, inactive count masks, replacement without double count, remap,
release, actual adapter fresh/replay prefill, and boundary beyond intermediate
width. Five captured signatures remain stable with allocation tracking.
Earlier retained failures: `penalties_probe.log` Torch Shape needed tuple;
`penalties_probe_retry.log` gather keyword is `out`, not `output_tensor`;
`penalties_probe_v3.log` exposed that dim0 gather transforms its output storage.
The fix copies the returned gather tensor, matching existing generator usage;
source confirmation in gather.cpp pre/post transforms. No shared TTNN changes.
The first neutral regression attempt failed on an accidental reset tuple edit
while removing unused scratch; corrected immediately, CPU helper checks rerun.
No device hang/reset occurred; each failed process closed all chips.

Independent source review caught standalone history being initialized before
prefill release; moved it after prefill. Four-rank CPU helper test and explicit
limits: `sampling_history_host_probe.py` / `sampling_history_host.json`.
Live serving, full sampling suite, async and final acceptance remain work.

Final read-only source review found one standalone-only combination to reject:
`generate(next_input=..., nonneutral device penalties)` would count sampled
tokens while teacher forcing replaces feedback with different tokens. Added an
early ValueError rather than advertise this unsupported combination. Extracted
actual-method CPU guard checks cover all three penalties and neutral/non-forced
controls (`teacher_forcing_guard.log`). No serving method changed after the
current full server started; teacher-forcing baselines use neutral penalties.

### PC-04 standalone prefill trace regression, 2026-11-16

Fresh same-code all64 qualitative control (`qualitative_controls/tt_pc04.log`)
failed before generation with `Writes are not supported during trace capture`
in prefill_128. The stack directly identifies `release_request` calling the
new penalty-history host reset. Capturing sample_penalties marks history dirty;
the next prefill capture therefore takes a host-writing release branch. This
was absent from B32 serving, whose first-chunk trace is disabled. No TT hang;
devices closed, fuser empty, bounded mesh arithmetic passed without reset
(`post_pc04_capture_failure_smoke.log`).

Read-only source review confirmed every prefill replay first calls upload.
Split release into `_release_sampling_history` (host boundary) and
`_release_recurrent_slot` (device-only); public release calls both. Prefill
upload clears sampling history after validation, graph clears only recurrent
state. The final generator reset still clears synthetic capture counts.
Do not merely clear the dirty set, which would leave stale device counts.
`probe_prefill_penalty_trace.py` exercises capture with dirty history, explicit
history clear before prefill replay, neutral/penalty/neutral output equality,
stable trace/program counts, and an allocation-tracker positive control on
representative layers0/3 before repeating the full control.

Reduced post-fix result: `prefill_penalty_trace.json` passed with5captures,
0retirements and249constant program entries. Explicit dirty-count reset and
repeated neutral/penalty outputs passed. Tracker positive control raised on
the deliberately live allocation, then replay succeeded after its release.
Local fix commit: `00cf43edc62990114938adc55bb680792a6a4525`.

All64 tracked qualitative retry also passes:12outputs, identical prior TT
tokens through chat EOS and exact6served greedy texts. Details are in
`readiness_vllm/qualitative_controls/tt_pc04_verdict.json`. The capture fix is
verified on reduced and full targets; no reset was necessary.

## PC-05 request-stable sampling (owner approved 2026-11-16)

Failing contract: a device-supported seeded request must retain device RNG
ownership when a genuinely host-only companion arrives. PC04 top-k64 companion
controls diverged at token4 on the reduced async target and token2 on full64
sync, while greedy remained exact. Canonical penalties already passed device
isolation. The approved fix changes the opt-in shared routing policy only.

Hypothesis checks: independent source audits verified that pre-budget scheduler
queue filtering can preserve requests/KV/async placeholders, whereas filtering
already-produced SchedulerOutput would corrupt scheduler accounting. Existing
front-packed slot remaps preserve paused recurrence. Lane decode needs explicit
inactive rows because its prior path executes all occupied slots. Per-owner
prefill/decode cadence must also be retained so alternating cohorts cannot
reset each other's fairness counters. Empty attempts still let EngineCore run
another scheduling turn; they cannot discard cleanup events.

Implemented in local plugin commit da15872: shared intrinsic classifier and persistent cached
request owner; fair host/device cohorts selected before phase negotiation;
existing reload/drain contracts on transitions; lane inactive-row masking.
Default auto/host behavior remains unchanged. Existing focused host regressions
passed536 cases (`pc05_legacy_focused.log`). Final host suite2784pass/16skip;71new mechanism tests pass. All15representative mixed-owner cases and the full64 active/companion check preserve both solo streams. Final full64 async lifecycle matches all17synchronous controls with five retained signatures and0retirements. Final full sampling67pass/75skip. The PC05 routing defect is fixed; P2 separately exposes the real top_k40 host-route performance cost (ratio4.28 at32users). No wider device sampler or invented waiver is added after the owner stage cap. Exact artifacts are in pc05_artifacts.json and readiness_vllm/pc05_final_async/.
