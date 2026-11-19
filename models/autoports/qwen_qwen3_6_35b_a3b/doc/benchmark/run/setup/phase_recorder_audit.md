# Independent CPU audit of adapter phase recorder

Date: 2026-11-18. Scope: root `AUTODEBUG.md`, `tt/benchmark_phases.py`, modified `tt/generator_vllm.py`, and their generator/serving/output-pool and pinned-plugin contracts. No source edits or device calls. The recorder was loaded directly with `importlib.util.spec_from_file_location`; synthetic handles were `types.SimpleNamespace`, so TTNN was never imported. A temporary snapshot directory was removed after the test, and the prior SIGUSR1 handler was restored.

## Result

The selected pinned-plugin ordinary decode path preserves the intended request/position ledger and joins completion by unique pool generation, including two outstanding outputs and reused leases. Existing host completion is marked after the pool processes/converts the output. No added device synchronization, device read, token inspection, or per-step file write was found in the recorder. Recorder state is protected by an RLock; accepted member records copy scalar metadata rather than retain mutable slot dictionaries. This is a CPU/source result, not live-serving validation or proof of collector attribution/roofline accounting.

## Focused synthetic checks

Executed one inline `python` process; all **10 assertions passed**:

1. Two prefill lifetimes with holes, followed by a contiguous second chunk: request identity survives and position becomes the cumulative end.
2. Resident decode with an opaque, unusable position object: no advisory position access; positions increment from 128/100 to 129/101.
3. Two generations complete in reverse order: each completion joins its own submission call and the pending map drains.
4. Destination-to-source remap `[2,1,0,3]`: requests and positions move together.
5. Lease 0 reused by generation 3: no alias to generation 1.
6. Authoritative reload with rows `[2,0]`, positions `[77,-1]`: only the active row is counted and inactive ownership remains available for later valid remapping.
7. Release then prefill into the same physical slot: a new monotonic request lifetime is assigned.
8. Subsequent position changes/release/reuse do not mutate prior accepted-member records.
9. Noncontiguous continued prefill is rejected.
10. Completion on a worker thread followed by snapshot: pending is empty and event sequence numbers are contiguous.

## Limits and repair considerations

- **Direct synchronous adapter calls are unsupported by the new instrumentation.** `QwenGenerator.decode_forward_v1` defaults `read_from_device=True` and then returns host tensor data, while the adapter unconditionally passes that result to `decode_accepted`, which requires `generation`, `lease`, and `rows`. This would raise with recording enabled. The current pinned plugin passes `read_from_device=False` in both ordinary async and ordinary synchronous paths, so this does not invalidate those measured paths. If retaining the wider adapter API is required, instrument the synchronous return separately or reject the unsupported selector before submission.
- **The prefill recorder only follows `empty_slots`, not the valid generator alias `slots`.** The current pinned plugin supplies `empty_slots`, so its path is correct. A direct call using `slots=[2]` would record the request in slot 0 and later fail/misassociate. Normalize the same slot selector if the broader API is to be covered.
- **Failed submissions mutate the shadow before model validation and lack explicit abort events.** A rejected decode leaves an unmatched `decode_start`; a partially rejected prefill may mutate earlier rows before it raises. The collector must reject unmatched/failed timelines and never reuse their state as valid evidence. Reinitializing the server after execution errors is sufficient for the selected successful benchmark path; recovery within the same recorder needs explicit failure/poison handling.
- An authoritative reload replaces the shadow without logging the prior position or checking discrepancies. This follows authoritative execution semantics, but it does not independently prove absence of a prior shadow drift. A successful live cohort must still reconcile accepted positions/counts, including unused speculative final steps.
- Phase starts occur after some existing adapter validation/logging and sampling translation (decode); prefill completion precedes construction of the returned mRoPE tuple. Full-phase accounting therefore still needs the collector's justified host-gap/boundary treatment. This audit does not certify complete end-to-end phase envelopes.

Source boundary checks: pinned `model_runner.py:4095-4102,4128-4140` supplies prefill starts and empty slots; `async_decode.py:1191-1195,1447-1452` supplies nonblocking decode; `model_runner.py:4191-4195` does the same for ordinary synchronous scheduling. `serving.py:230-257` defines authoritative versus resident positions and remap direction; `serving_state.py:480-555` proves generation uniqueness and natural host conversion completion.
