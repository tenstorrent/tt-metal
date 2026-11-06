# Model-owned sampled-token retention

Validity goal: retain already-host-visible sampled decode token IDs for diagnosing capped segments lost by the pinned mini-swe malformed-response parser. Unblocks SWE cap classification; does not infer request identities or repair upstream retention. Forecast0.12 deviceh, reserved box0.25h after queued Astropy, no device work authorized to this preparer.

1. Read prior malformed-response diagnosis and current collector/adapter completion ownership. Fresh bundled personal-login AutoDebug inspection establishes the narrow safe insertion point before edits.
2. If supported, implement opt-in bounded private token-segment records on existing phase step IDs, physical rows and authoritative positions. No extra readback/device synchronization, request/content changes, wrapper or shared plugin edits. Explicitly report omitted host-sampled prefill tokens, unknown request IDs, padding/slot ambiguity, export/buffer loss, and inability to prove whole response completeness.
3. Host tests cover asynchronous readback association, slot remap/position snapshots, disabled behavior and failed/bounded export. Verify output identity/value preservation. Return candidate feasibility to coordinator before live validation.
4. Coordinator may allocate reserved live test for paired exact native output and full retained decode IDs under real async serving. No final RESULT until effort ends; previous effort evidence and root reports untouched.

Host candidate ready (not accepted): fresh AutoDebug confirms exact host hook and missing-request scope. Added default-off GEMMA4_RETAIN_DECODE_TOKENS; enable alongside GEMMA4_BENCHMARK_PHASE_LOG and positive GEMMA4_PHASE_AUTOEXPORT_SECONDS. Existing O_EXCL/0600/capped writer is required. Completion records contain sampled_decode_token_ids indexed by physical row; position means processed input position and sample predicts position+1. No prefill/host-sampled token or client-emission coverage. Association/conversion failure disables token diagnostics, clears bounded pending references and marks loss; model validation remains outside diagnostic guards. Setup failure warns without token contents and yields no recorder (missing evidence is incomplete). Timing-only behavior remains strict.

Host checks:14 token-retention unittest cases,13 existing recorder cases, and existing check_serving_host_contract.py passed. New checks include reverse async completion with reused device object, copies surviving tensor mutation, positions/remap snapshots, B1/8/16/32 inactive rows, direct sync adapter samples, async output identity under capture failure, model-validation exception preservation, duplicate/unknown associations, pending/event/file bounds, failed export and reset refusal before drain. No generator.py edits, device job or final result written. Next: coordinator review and bounded real async native-output pair after Astropy queue finishes; require same-submission tuple witness and zero loss/pending at final drain before acceptance.

## Prepared live pair (coordinator launch only)

Private driver: `bringup/artifacts/model-token-retention/live_pair.py`; launcher: `bringup/artifacts/model-token-retention/launch-arm.sh`, derived from the already used native-token-retention-confirmation full60 B8 launcher. Both arms use the same worktree/source and server arguments, phase export enabled every1s, private64MiB cap; only GEMMA4_RETAIN_DECODE_TOKENS differs0/1. Explicitly unset reduced-layer selection. Source pair is diagnostic-off versus diagnostic-on, not two model implementations.

Forecast0.12deviceh for both arms, based on prior autoexport pair server holds0.05168+0.05223h plus~0.016h for the longer staggered requests and shutdown allowance. This is an estimate, not a runtime result. Each server bounded420s, combined840s=0.2333h within0.25h box; coordinator includes actual cleanup/queue-device ownership and stops if remaining box cannot fit second arm. No launch before queued Astropy completes.

Use a fresh private run directory (do not overwrite existing files), for example `$RUN` below is an operator/coordinator-selected absolute persistent directory under `bringup/artifacts/model-token-retention/`:

```bash
mkdir -m 700 -p "$RUN/baseline" "$RUN/candidate"
# Coordinator queues baseline server; client is a separate host command while it holds device.
bash bringup/artifacts/model-token-retention/launch-arm.sh "$RUN/baseline" 0
python_env/bin/python bringup/artifacts/model-token-retention/live_pair.py arm "$RUN/baseline/client"
# Client ends only after all HTTP requests complete, engine metrics idle, and final phase drain.
# Coordinator checks owned server identity, shuts it down idle, and waits for device close.
# Only then queue candidate, same commands/configuration except retention flag:
bash bringup/artifacts/model-token-retention/launch-arm.sh "$RUN/candidate" 1
python_env/bin/python bringup/artifacts/model-token-retention/live_pair.py arm "$RUN/candidate/client"
# After identical idle close:
python_env/bin/python bringup/artifacts/model-token-retention/live_pair.py audit "$RUN/baseline" "$RUN/candidate" "$RUN/paired-audit.json"
```

The separate server/client commands are intentionally not an autonomous device queue or shutdown wrapper. Coordinator owns launch handles, owner checks and safe idle shutdown. No client-side tool commands execute. Native return_token_ids is on in both arms. Fixed-seed greedy requests A/C use existing SWE tool protocol, B uses existing Terminal protocol plus inert8192-token-ish padding and384output cap/ignore_eos to maintain overlap. C has distinct2048-token-ish padding. All bodies identical across arms, saved privately. A starts, B staggers150ms later; after A completes C starts while B is expected active. At most2clients. These are synthetic diagnostics, not scored tasks or changes to release caps.

Audit compares complete native message/reasoning/tool arguments, finish/stop reasons, usage, prompt and generated token IDs, excluding only randomized tool-call IDs. It verifies identical request bodies. It checks every native decode ID against retained physical-row samples by measured disjoint prompt/output position intervals plus exact token values; this is not an arbitrary-row request-ID assignment. Native token index0 is explicitly excluded as prefill coverage. Disjoint intervals must be proven from native prompt token counts, otherwise audit fails. All samples at an expected position must agree; missing/mismatched positions fail. Duplicates and samples outside native ranges are reported (possible async suppression/overshoot), never relabeled emitted tokens. Reports inactive rows, observed overlap, nonidentity remaps and host-sampled prefill steps. No remap or mixed-prefill observation means that part of traffic was not exercised, not an inferred pass. Final errors/loss/pending must be zero for acceptance. No claims of full malformed-response reconstruction.

## Position identity safeguard

Pinned stock plugin model_runner.py:1091–1096 and2958–2959 sets decode_layout_changed on persistent-batch change; async_decode.py:450–472 forces reload_inputs on that transition. However `_decode_state_slot_remap` also derives state ownership independently, so these local observations do not prove every future nonidentity remap implies reload. The candidate now conservatively flags `position_identity_uncertain=true`, marks loss and disables diagnostics if nonidentity slot_remap arrives without reload_inputs. It does not alter model inputs or apply a guessed permutation. Identity remap/no reload remains valid; reload snapshots supplied positions. New focused host test covers this case.15token tests +13existing recorder tests +host contract pass; driver compiled and shell syntax checked, no live execution. Final AUTOFIX/RESULT remains pending live outcome.

## Accepted closure and publication preparation extension

Coordinator accepted live pair after both bounded servers closed. Prepare source-only publication commits, no pushes/dispatch: copy only accepted benchmark_phases.py/generator_vllm.py to dedicated TT publication branch based439c22b0; TTI on877e162 adds model-specific GEMMA4_RETAIN_DECODE_TOKENS=1 and enables original frozen SWE5 at existingC2/nativeIDs, GPQA/Terminal nightly0. Preserve all other source/settings and Terminal retention metadata. Run focused/full hooks and exact-source audits. New runtime SHA requires new CI image; old439c image cannot validate it. The next normal same-five rung remains coordinator-owned and its validity is unproven by this synthetic pair. Final evidence is written now once; publication candidates are separate preparation evidence.
