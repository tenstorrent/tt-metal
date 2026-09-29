# AUTOTRIAGE

## Diagnosis

The probe-only indexed expert-union mask attempted a host device write during
trace capture. Its later cleanup raised while releasing an unfinished trace;
the Python process remained alive and held the UMD lock. No selected production
runtime is implicated by this experiment.

## Triage Evidence

The full live capture is preserved as
`models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/tsu_optimization/experts_trace_failure_triage.txt`.
Inspector identifies the failing probe's runtime directory. All four op meshes
are idle and there are no reported running operations or lightweight asserts.
Core magic passes. Dispatch stacks wait for commands; ARC uptimes agree and
clocks are1350MHz. Some Ethernet counters differ from software counters.
The full report reports binary-integrity mismatches on logical(0,0) on all ranks
although the separate summary labels every script pass: the summary alone is
not a clean bill of health. Binary anomalies remain unresolved diagnostic
observations; no affected kernel stack is used to assert a hardware diagnosis.

## Source Evidence

`tests/tsu_batch_candidate.py:expert_union_decode` originally called
`ones_like(ids, dtype=bfloat16)`, where ids has an integer dtype.
`ttnn/cpp/ttnn/operations/creation/creation.cpp:full_like_impl` only uses the
device fill when source and destination dtype match and are supported tiled
floating types. The dtype override falls back to host `full_impl`.
`tt_metal/distributed/fd_mesh_command_queue.cpp:write_shard_to_device` rejects
all host writes while trace_id is set. This exact rejection appears four times
at13:35:01, after the serialized baseline succeeded. The generator warms,
begins capture, calls `_forward`, and cannot reach end_trace_capture on error.
Its finalizer then reports an unregistered trace. This explains the warm/capture
contrast without a CB/semaphore producer-count change.

## Downstream Effects

ContainerPID162423 remains alive after the exceptions and retains TT device
FDs. RetryPID162549 has not reached model execution: UMD explicitly waits for
the first PID's chip lock. Host-side fuser incorrectly appeared empty across
this namespace; future cleanup checks must include the owned container's
process/FD view and completion of the exec session, not host fuser alone.

## Proposed Fix

Create the mask with `ones_like(routes[..., :top_k])`, an existing tiled BF16
tensor, so source/output dtype match and device fill is selected. The rest of
the scatter/union contract is unchanged. This probe fix is present but not yet
verified by a completed retry. Preserve the capture before terminating the
owned failed/waiting probes, then retry only after container process/FD checks.

## Uncertainty

No native host stack was available to identify the precise teardown futex.
The capture-time rejection is proven; the cleanup's exact internal deadlock and
the idle-core binary diagnostics are not fully diagnosed. No reset is justified
solely by this report. A clean successful retry and correctness check are required.

## Validation Addendum

After the live capture, only the failed and waiting owned processes were
terminated; no reset was performed. The same-dtype device-fill retry exits0,
matches all32 logits exactly, and passes six recorded layer0/5 output/full-KV
cases under Watcher/allocation tracking. The complete30-layer probe also matches
all32 logits exactly. This validates removal of the triggering host-write path;
it does not independently diagnose the generic unfinished-trace teardown bug.
