# Async output and dormant-slot source audit

CPU source inspection only; no hardware result is claimed here. The pending
runtime coverage is `tests/probe_full_model_serving.py`.

## Host tensor and event contract

`tt/generator.py:read_decode_output` calls `DecodeSubmission.tensor.cpu` with
`blocking=False` for asynchronous reads, constructs a new host submission, then
records a CQ0 event. It does not defer the copy until host processing.

The underlying implementation supports this use:

- `ttnn/core/tensor/tensor_ops.cpp:233`: `cpu` enqueues a read of device storage
  through its mesh command queue and propagates tensor topology onto the host
  result. Host input is returned directly.
- `tt_metal/impl/tensor/tensor_apis.cpp:55`: each uniform `enqueue_read_tensor`
  creates a fresh distributed host buffer and allocates each host shard before
  enqueuing the read. Two calls do not reuse one persistent host output buffer.
- `tt_metal/impl/tensor/distributed_tensor_apis.cpp:118`: the nonuniform branch
  also creates and allocates a distinct distributed host buffer for each read.
- `ttnn/core/events.cpp:16`: Python `record_event(mesh, 0)` maps to
  `enqueue_record_event_to_host`; `event_synchronize` waits for that mesh event.
  The event follows the read on the same CQ used by decode.
- `ttnn/core/distributed/api.cpp:68`: despite its name, `get_device_tensors`
  explicitly accepts CPU tensors and returns their host shards.
- `ttnn/core/distributed/host_ccl.cpp:all_gather`: on this single-host mesh,
  host-shard gathering validates local shards and returns the host tensor;
  this token-processing call does not fetch logits or tokens from the device.
- `ttnn/ttnn/distributed/distributed.py:ConcatMeshToTensor` constructs the C++
  composer; logits processing concatenates the host shards along vocabulary
  dimension 3. The sampler's token tensor is replicated, so selecting its
  first host shard is consistent with its ownership.

No source-level defect was found in the current asynchronous host snapshot
mechanism. The caller must enqueue `read_decode_output` before submitting a
later decode that writes the same resident output, and synchronize the returned
events before processing host data. A raw `DecodeSubmission` is a reference to
resident device output, not an already-created immutable host snapshot. The
probe follows this ordering and deliberately delays host processing until after
a later decode/read, then processes the two snapshots in reverse order.

This establishes neither safe scheduler overlap nor permission to rebuild the
next request's inputs from stale host state. The probe uses explicit scheduler
inputs for its asynchronous comparison and compares exact synchronous controls.

## Dormant-slot regression coverage

The old authoritative-input reload uploaded a zero-filled 32-slot tensor and
only populated active rows, losing dormant rows' sampled feedback. The current
source first copies `decode_tokens` to `saved_tokens`, uploads active input
values, then calls `_restore_inactive_tokens(active_slots)` before decode.

The new `active_reload_preserves_dormant_feedback` probe case:

1. Prefills and samples slots 0 and 1, requiring a nonzero sampled slot1 token
   so the previous zero-filled overwrite is observable.
2. Reloads authoritative input for active slot0 while marking slot1 inactive.
3. Compares all dormant persistent token rows on every rank, all mutable slot1
   cache tensors and its exact QSA/current-position state.
4. Resumes slot1 with `reload_inputs=False` and compares its output and cache
   exactly with an uninterrupted fresh control.

The async case now also asserts both returned TTNN tensors have HOST storage
and are distinct host tensor objects. Existing delayed-processing checks cover
token `[B,1]` and logits `[B,1,V]` results, row order, inactive rows, and repeated
processing of the earlier snapshot. These assertions await device execution;
formatting and Python compilation passed after the update.
