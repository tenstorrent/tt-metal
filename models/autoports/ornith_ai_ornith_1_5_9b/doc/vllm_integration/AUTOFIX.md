# AutoFix: Ornith serving boundary

Diagnosis: [AUTODEBUG.md](AUTODEBUG.md), independently produced from pinned sources and failed request logs. Failing contract: plugin requests must translate to the canonical generator, preserving sampling values, output row order, text rotary positions and cache ownership.

Prioritized repairs and discriminating checks:

1. Convert plugin sampling dataclasses to the canonical SamplingParams without formatting/inverting/clamping values. The existing host test reproduced the mismatch before the change; it passes afterward.
2. Preserve the canonical host-prefill tensor directly. Host prefill must not re-read logits or interpret the tensor as PrefillBatch.
3. Return zero text M-RoPE deltas alongside prefill output when the actual HF config uses M-RoPE; consume only zero/None decode delta metadata. Reject real media and nonzero deltas.

Only tt/generator_vllm.py implementation changed. No plugin, vLLM, generator or numeric model source change was needed for these four errors. Tests: tests/vllm_interface_test.py, interface_before.log (3 failures), interface_after.log (5 passed). Runtime: diagnostic4/request.json gives HTTP200 for129 input tokens and4 output tokens. diagnostic4/mixed.json gives six HTTP200 requests at lengths129/1033/131/130/129/2051 with70 outputs, concurrency3, tracking on and context262144. Repeated129 requests have identical70-token sequences. diagnostic4/host_logprobs.json passes three concurrent70-token requests with logprobs=0, exercising explicit host compatibility. This is a two-layer diagnostic, not full-model quality or performance.

The diagnostic4 server was stopped with SIGINT to its runner only after /metrics showed zero running and waiting requests. It shut down and released all devices. No reset was required.

Remaining investigations: source-confirmed risks in seed continuity on transitions and penalty histories on resumed prefill are assigned separate read-only AutoFix hypothesis investigations. Direct adapter stale-input/page-table/remap/readback validation is running. No final stage pass or failure verdict yet.

Update: direct adapter decode battery passed. Diagnostic6 async mixed requests matched the synchronous control, but host-to-device mixed continuation later failed; see server log and events summary. Both seed continuity and missing resumed-prefill histories are confirmed source-level defects, with separate reports. The latter requires an explicitly authorized shared interface; proposed record is plugin_changes.md. AutoFix has not been declared exhausted. Stage is incomplete, awaiting that exact authorization, with all further local findings preserved for continuation.

## Ordinary-request seed continuity repair

Implemented the model-owned repair from `seed_continuity_investigation.md` in
`tt/generator.py`; no plugin/shared interface change. The earlier statement that
only the adapter changed describes the initial four-error repair, not this
subsequent seed fix. Existing trace instrumentation is preserved.

Successful ordinary prefill records its prefix end per cache/slot for both host
and device sampling. At an authoritative decode input reload, missing output
history now obtains its RNG offset from `position - prefill_end + 1`. Explicit
output histories, including empty histories and rewinds, override that inferred
offset and update its future position relationship. Small host submission
counters retain dormant-slot offsets between transitions; they inspect the
already-admitted active mask, never token readback or stale resident inputs.
Metadata follows remaps and is cleared on release/reset/fresh admission. The
four reload commands and the existing sampling kernels remain authoritative.

Validation performed without importing or running a TT model:

```text
python_env/bin/python -m unittest \
  models.autoports.ornith_ai_ornith_1_5_9b.tests.vllm_seed_continuity_test \
  models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_sampling_state_test \
  models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_decode_contract_test -v
```

Result: **25 tests passed**, including ten new seed-continuity tests. The new
tests AST-extract actual generator prefill/decode/preflight/reset/execution
methods, replacing TT operations with host fakes. They cover first decode,
resident membership reload, active/dormant remaps, host chunks and host intervals,
explicit histories/empty rewinds, scoped resets, release/reuse/cache identity,
and rejection before writes when an origin or inferred offset is invalid.
Poison host objects verify resident steps, remap-only steps, and page-only
reloads never inspect stale tokens/positions. The test also checks that input,
parameter, and seed reloads still execute in order.

Before/after discrimination: reloaded the same extracted methods from
`git show HEAD:models/autoports/ornith_ai_ornith_1_5_9b/tt/generator.py` and
reran four selected new test methods. All four failed with five numeric seed
assertions (the rewind test has two subcases), with zero harness errors. For
example the first decode reconstructed seed 143 instead of 144; a later
membership transition reconstructed 143 instead of 147. The working-tree
methods pass all ten cases.

Remaining limits: device validation is pending; no hardware result or performance
claim is made here. `tests/vllm_seed_decode_probe.py` was inspected read-only and
its offsets match the repair (post-prefill baseline +1, +2, remapped +3, host idle,
then +5). Capture must still precede live admission as in that probe. Arbitrary
resumed prefill containing prior generated outputs remains unimplemented: this
interface cannot identify the original prompt/output boundary, and an incoming
prefill is still treated as ordinary prompt admission. No `original_prompt_lens`
interface was added. Explicit histories at decode remain usable, but cannot
repair an earlier resumed-prefill draw or missing penalty token history.
Unspecified (`seed=None`) requests retain the prior random-base reset behavior;
deterministic continuity claims apply to explicit seeds.

### Parent device verification of seed repair

The parent reran the 25 host tests into `seed_host_tests.log` and the original-HEAD four-case discrimination into `seed_host_before.log` (5 numeric failures, no harness errors). `seed_precommit.log` records passing repository hooks before hardware execution.

Under `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, `timeout 300 python $MODEL_DIR/tests/vllm_seed_decode_probe.py --output $MODEL_DIR/doc/vllm_integration/adapter_seed_decode_probe.json` passed all six checks on TP4, real layers0/3, B3, context256. Checks: first decode seed includes prefill draw; resident advance once; remap/reset; device seed idle during explicit host mode; host-logits geometry; return-to-device uses logical offset. The probe reads seed tensors solely as a test oracle. Runtime token feedback remains canonical/device-resident. Initial test fixture attempts used incompatible scalar temperature with list seeds, then failed to unpack the existing async `(handle, events)` return; both fixture mistakes were corrected, logs retained as `adapter_seed_probe_bad_fixture.log` and `adapter_seed_probe_bad_read_fixture.log`. They are not model failures.

`timeout 300 python $MODEL_DIR/tests/vllm_adapter_decode_probe.py --output $MODEL_DIR/doc/vllm_integration/adapter_decode_after_seed.json` also passed, with the same allocation tracking. This covers two pending submissions, stale host objects unread, changed/unchanged tables, whole-pool and resident remaps, scoped reset, inactive state, host-logits ownership, and invalid input rejection. Both jobs closed all devices normally. No reset, profiler, server or plugin change was used for the seed repair. Full-model serving evidence remains pending; original prompt/output boundaries for unknown resumed prefill remain impossible to infer from the current interface.

## Owner-approved boundary repair and forced-resume follow-up

Owner authorized Entry1; plugin local215effd carries immutable original prompt boundaries and repairs shared host-prefill histories. Canonical generator restores prompt/output histories with the same existing sampler and reconstructs the seed offset from the supplied original boundary. 37 combined host tests pass (`entry1_all_model_host.log`); 14 reduced adapter device checks pass (`original_boundary_device.json`); the adversarial decode battery still passes (`adapter_decode_after_history.json`, captures6/releases0).

Real forced scheduler preemption in diagnostic8 completes but exact trajectory comparison fails. The reduced direct numerical probe atB3/context256/oneactive slot demonstrates identical seeds and all four penalty tables before output23; decode versus replay logits PCC0.9999695532,maxabs0.0625,top32orderidentical,host/deviceprefillbitwiseidentical. Both direct paths sampletoken1. This does not explain away the served failure: servedcontext262144 andpagepooldiffer; its sampled prefix diverges from this direct control at17. Source inspection additionally proves a front-packed immediate-resume page-table append defect, recorded separately as proposed Entry3; actual plugin host regression and external AutoDebug are in progress. No runtime workaround or sampling fallback has been introduced.


## Diagnostic9 read-only preemption audit, 2026-11-13

Saved `diagnostic9_pages/preemption_analysis.json` and `.md`, comparing diagnostic8/9 and the served-context canonical control. Entry3 `2a1504554aa7babb1d426dfbca8c7db8c0a795ef` visibly replaces resumed prefill page rows: greedy/host `[8,7,6,5,4]` becomes `[4,5,6,7,8]`; seeded reverses that ordering. Every replay uses start0/end151/original129/slot0. All baseline sequences are unchanged and all six requests complete100 tokens. Forced host raw scores first change between plugin versions at output index32, absolute decode position160 and first newly needed logical page5; seeded old/new token sequences first change there too. This supports the ownership repair; decode page rows were not recorded.

**All three strict uninterrupted/resumed token comparisons remain failed.** Diagnostic9 greedy device and host first differ at index49 (53 versus498), seeded at index22 (264 versus51). Canonical served-context recomputation independently reproduces only the seeded first difference with identical four penalty tables and seed275437, differing raw logits (PCC0.9999462981, max_abs0.25), and identical host/device prefill logits. The control ends at output23 and does not compare recurrent/convolution/KV state. Greedy index49 has not yet been independently reproduced; no exact equivalence or numerical-tolerance acceptance is claimed. A targeted canonical greedy replay through output50 remains useful. Full32-layer/B32 acceptance is separately underway; this audit ran no device job and edited no runtime code.


### Prepared targeted greedy numerical control

Extended `tests/vllm_replay_numerics_probe.py` with optional `--greedy`, `--follow-outputs 50`, and `--served-reference .../diagnostic9_pages/preemption.json`; defaults remain sampled output23. It generates the uninterrupted branch to the requested total, replays the original129 plus22 outputs, then follows the resumed branch to the same total. Each additional decision checks all four sampler tables against its authoritative history before sampling. At the first unequal token it records both state oracles/signatures, raw-logit metrics/top64 and full raw vectors in `<output>.first_divergence.pt`. Optional served comparisons retain separate baseline/forced equality results. These are diagnostic reads, not corrective runtime inputs; the usual explicit first decode reload after prefill is preserved.

Host-only fake control-flow validation exercised default23 and greedy50, including a synthetic divergence49, exact state checks, served-reference matching and raw artifact output. Compile and Ruff checks passed (`replay_followup_host_validation.log`). No device execution or actual greedy reproduction is claimed; parent will run after the full-model server releases devices. Suggested flags: `--context 262144 --greedy --follow-outputs 50 --served-reference $MODEL_DIR/doc/vllm_integration/diagnostic9_pages/preemption.json --output $MODEL_DIR/doc/vllm_integration/replay_greedy_follow50.json`.

## 2026-11-13 — prefill horizon/discard repair, verification ongoing

The first model-owned prefill refinement forwards authoritative original horizons
and avoids discarded intermediate-logit readback.42 CPU tests passed, but an
independent audit found an unknown-to-known transition they did not validate;
see the new AUTODEBUG section. AutoFix is repairing the earliest helper condition
and adding real-validator regressions. No assertion suppression or plugin change
is proposed. The corrected tracked launch is guarded until host review passes.
Hardware claims remain pending; the current long-context server has the prior
source loaded and continues its bounded acceptance request.

The unknown-to-known correction is implemented: compatibility and boundary
checks now run for every continuation unless missing-history behavior was
explicitly requested.44 CPU tests pass. Restoring the old condition makes both
new actual-runtime validation regressions fail with the expected profile/tail
exceptions, recorded in `intermediate_prefill_transition_negative_control.log`.
Independent follow-up inspection confirms the original finding is closed.
`prefill_transition_validated.json` releases the corrected-server launch guard;
it explicitly records hardware validation as pending. No plugin source changed.


## Optional model-state probe: diagnostic slice preparation

The first aligned-state run failed before model-state comparisons. The allocation tracker rejected28 live `program_cache: SliceDeviceOperation` buffers before the sampling replay (`replay_model_state151_before_slice_warmup/replay_model_state151.log:153-206`). The optional diagnostic first sliced active KV pages0..4 and inactive sentinel pages8192/16384 after model/sampling trace capture. Integer slice offsets and rank-local shard placement require their own program-cache preparation. This is a diagnostic preparation defect; no model-state conclusion is available from the failed run.

Failure log, JSON and original probe source are preserved in `replay_model_state151_before_slice_warmup/`. Only `tests/vllm_replay_numerics_probe.py` and its CPU test changed: the exact snapshot reader is shared by preparation and collection, and the optional path executes it twice before the first capture. The second pass must add no programs. It uses the actual cache tensors, selected offsets and all TP shards, discards CPU copies, and writes no model state. Preparation rejects existing model traces or differing decode/prefill mappings. Defaults remain unchanged; no allocation exemption or tracker/program-cache suppression was added.

Validation:10 CPU tests pass, including a negative control with56 previously unseen source/rank/page signatures after simulated capture versus zero new signatures after preparation; the mock signature count is not a measured TTNN program count. Actual pooled layout/page-offset/alignment checks remain. Ruff and py_compile pass; commands/results are in `replay_model_state_slice_warmup_host.log`. Hardware rerun with tracking1/skip0 remains required; the report now records both warm-pass program counts. No hardware was run for this repair, and strict trajectory failures/model-state attribution remain unresolved.


### Aligned151 rerun completed

The parent executed the corrected probe with tracking/program-cache checks retained; complete JSON and normal device closure are recorded. All six structural checks pass, host/device replay48tensors are exact, and sampled inactive32tensors are exact in each comparison. CPU analysis of the raw tensors finds all original-prompt KV rows0..128 exact, with differences beginning at the decode transition129. This demonstrates a canonical prefill/decode computation-route difference; no concrete remaining adapter state defect is identified. Strict original forced-preemption token parity remains failed, and a rounding-only explanation or tolerance is not established. Detailed metrics, scope and limitations: `replay_model_state151_interpretation.json` and `.md`. No additional runtime fix was made.
