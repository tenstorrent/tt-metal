# AutoDebug: model-owned sampled decode retention

## Finding

Minimum viable retention is an **opt-in copy of already-materialized sampled decode IDs into the existing completion event**, paired with its existing submission step and physical-row positions. No generator/device operation or plugin/request change is necessary. This is a partial diagnostic trace, not a reconstructed response or a repair for malformed generation.

Two concrete discrepancies matter before implementation: phase events currently contain no sampled IDs; and the recorder's manual-export mode is neither private-by-construction nor byte-capped. Existing recorder association exceptions also propagate into serving, so simply adding payloads does not satisfy a diagnostic-failure-isolation requirement.

Inspection baseline: execution checkout HEAD `0903d710c82b61a9e2f7a2a790f9d4a185eb8fb5`, current working-tree sources. Read startup contracts and prior `malformed-response-retention/{AUTODEBUG,AUTOFIX,RESULT}.md`. Plugin inspected at `/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09`, HEAD `90f33d2e392eb8bfda6948df9dabe712a724cb1a`; `git diff ca7a18246e4111629ed6cb4a7d1192002749fccf -- src` is empty. No delegation, hardware, publication, source edits, or root-report edits.

## Proven completion/association chain

Paths below are relative to `models/autoports/google_gemma_4_31b_it`, except plugin references.

| Boundary | Source evidence and implication |
| --- | --- |
| Submission | `tt/generator_vllm.py:375–389` snapshots positions and obtains `step_id` before dispatch. `benchmark_phases.py:begin` stores that record in `pending`; `request_ids` is explicitly null. |
| Reused device storage | `tt/generator.py:568–575` replays model/sampler and returns persistent `self.tokens` when reads are deferred. Device object identity alone cannot distinguish successive steps. |
| Existing transfer | `generator.py:577–581` queues `output.cpu(blocking=not async_read)` and records an event. Adapter `generator_vllm.py:440–448` associates FIFO submission with a distinct host object. Recorder retains the host object alongside its identity, preventing ordinary identity reuse while pending. |
| Actual async completion | Stock plugin `src/vllm_tt_plugin/async_decode.py:1656–1680` waits each submission's read events before invoking `process_decode_output_host`. This is not merely enqueue completion. |
| Materialized IDs | `generator.py:583–590` selects the first replicated device tensor, takes the first configured `batch` lanes, and returns int64 `[batch, 1]`. Adapter `generator_vllm.py:451–459` validates that result and completes the record associated with the original host object. Capture the returned CPU result here, not by converting `output` again. |
| Direct synchronous path | `generator_vllm.py:433–439` already constructs/validates `[batch, 1]` sampled IDs and completes by explicit `phase_step`. Capture here too. The deferred no-read-hook conversion path uses the recorder's oldest-unbound fallback. |

The existing overlap test uses the same device object for two submissions and distinct host objects. It proves metadata association in that sequence, not equality of retained IDs to actual async outputs. A late completion must use the record popped for **that host object**, never `sequence`, latest positions, latest remap, or current batch ownership. Copy primitive integers immediately; do not retain mutable tensor views.

## Positions, remapping, and missing coverage

`benchmark_phases.py:decode_positions` snapshots authoritative supplied positions on reload, otherwise advances its shadow, preserving negative inactive rows. This matches `generator.py:148–153,543–565`: no-reload ignores supplied inputs; model decode increments nonnegative resident positions. These positions describe the **input token processed**. The sample predicts the next token (normally position `p+1`); label this convention explicitly.

Adapter `generator_vllm.py:396–404` applies `slot_remap` to sampler seeds. It does not permute the just-produced host result. Plugin `model_runner.py:1338` defines new row `i` as taking old state slot `remap[i]`; request ownership lives in plugin state. Retain an immutable submission-time remap, if present, as context only. Do not apply it again to output IDs or positions. Pair row `i` with that step's position `i`; keep inactive `-1` rows identified as inactive and never present them as emitted request tokens. A concrete delayed-completion case is step A at `[20,30]`, followed by reloaded step B at `[31,21]` with remap `[1,0]`: A's rows remain paired with `[20,30]` even if A completes after B's submission.

The model hook receives no request-ID mapping; plugin `async_decode.py:1360–1405` passes sampling/remap data, not `row_req_ids`. Equal positions or stable rows do not prove request continuity. Slot reuse, cancellation, or plugin suppression can make a sampled row differ from a token eventually published to a client.

Prefill completes at `generator_vllm.py:349–356` without retaining IDs. All-final device-prefill IDs exist there but are outside the requested decode-only minimum; mixed/intermediate prefill returns logits for subsequent plugin host sampling. Host-sampled decode likewise exposes logits, not the final sampled IDs. Record coverage explicitly: `decode_device_only`, `prefill_tokens_retained=false`, `host_sampled_tokens_retained=false`, `request_ids_available=false`, `client_emission_verified=false`. Do not reconstruct missing IDs from text, logits argmax, later inputs, or guessed request mappings.

Prior malformed-response reports concern a separate downstream loss: mini's parser can discard a completed response on `FormatError`. This model-side trace neither fixes that dependency nor recovers missing historical responses. No evidence here establishes repetition, a model loop, or the cause of malformed output.

## Smallest proposed implementation contract

1. Add a separate default-off token-retention option to `PhaseRecorder`; phase logging alone must not opt into token storage. Require an explicit bounded private export configuration for this mode. Keep `generator.py` unchanged.
2. Extend completion calls to accept optional already-host sampled IDs. Adapter passes them only on the two device-sampled decode completion paths above. The collector attaches a bounded integer list to the matching completion record, with explicit row indexing, original positions, sampling mode, and submission-time remap. Emit no duplicate payload on submit. Disabled mode performs no extra tensor conversion/copy.
3. Preserve phase timestamp semantics: take the completion timestamp at the existing boundary before diagnostic conversion/export work. Use only CPU data; no additional readback, synchronization, sampler call, or change to returned tensor, shape, dtype, values, or validation. Keep existing model validation failures visible.
4. Make retention setup, association, payload conversion, and export failures non-fatal to model outputs in this opt-in mode. Existing `bind_readback`/`complete_output` raise, and adapter does not catch them. On ambiguity, mark loss and disable/drop affected diagnostic state rather than guessing a pairing. Guard only diagnostic operations, not model execution or token validation. Preserve legacy timing-only behavior unless separately intended.
5. Bound per-event rows (configured batch, at most 32), buffered events/bytes, pending/unbound/readback references, error storage, export chunks, and total file bytes. Current `max_events` bounds only `events`; pending maps/queues/errors have no independent cap. Release diagnostic object references after failure/cap. Persist bounded counts/reasons for payload loss, association failure, buffer overflow, file exhaustion, and export failure, with pending/incomplete status. Missing payload must never mean an empty successful sample.
6. Reuse the secure auto-export writer (`O_EXCL`, mode 0600, retained descriptor, reserved terminal-status space), including for manual drains in retention mode, or require auto-export explicitly. Current `_write_export`'s non-auto branch uses ordinary append, follows existing paths, and bypasses the byte cap; it is unsuitable for token evidence. Keep IDs out of stdout/error text. Use a private evidence directory and exclude files from publication. An unwritable disk cannot reliably persist its own failure: expose bounded status through control state/log status without token contents, and treat missing final status as incomplete, never as proof of losslessness.

## Verification and remaining uncertainty

Executed only `PYTHONDONTWRITEBYTECODE=1 python models/autoports/google_gemma_4_31b_it/tests/test_benchmark_phases.py`: **13 passed**. Tests cover overlap association, position shadow, concurrency, privacy and caps in auto-export, and failure status. They do not exercise token retention, adapter payload forwarding, or real event/value correspondence.

Required host checks for a future patch: exact `(step_id, row, position, token_id)` equality for reused-device/distinct-host overlap, reversed completion order, sync/direct/deferred paths, reload/remap while older work is pending, stale inputs on no-reload, inactive holes and B1/8/16/32. Mutate caller lists/tensors after capture to prove copies. Assert opt-out emits no IDs; host sampling/prefill report explicit noncoverage. Inject unknown/duplicate bindings, conversion errors, queue overflow, short writes, disk failure, stale files and symlinks; returned model results and existing model exceptions must remain unchanged. Confirm all bounds and omission counters, including final drain and interrupted export.

A later authorized real async check must compare each retained tuple against the **same submission's already-materialized host result**, before any plugin row filtering, under overlapping replay/readback and slot churn. An independent test witness must retain submission identity and positions; aggregate token ranges, text similarity, or unindexed logs cannot prove pairing. Verify unchanged readback/trace counters and no added wait calls, and inspect pending/loss status after the final existing completion. Do not add production instrumentation or device synchronization to establish this. Host tests cannot prove runtime CQ ordering, transfer contents, or unchanged timing; those remain unverified here.

Claim-review pass: rechecked both completion routes, plugin event wait, position advancement, remap polarity, export branches, and test assertions. No async misassociation is proven in the existing happy path. The headline is the retention opportunity and concrete privacy/failure-isolation gaps, not an alleged generation correctness defect.
