# AutoDebug: idle phase exports invalidate the whole collector archive

## Finding

**Confirmed source defect: the periodic recorder appends a JSONL record on every timer tick even when neither events nor meaningful status changed.** In GPQA20 run 42256121890, the retained log reports a phase file changing during tar collection after the engine became idle. The stock named-volume collector treats any nonzero tar return as collection failure and never extracts its stdout, so one changing phase file prevents collection of every member in that archive. This is archive rejection, not deletion of the source evidence.

The smallest model-owned intervention is an **automatic-export-only changed-state guard** in `tt/benchmark_phases.py`: export queued events unconditionally; when none are queued, write only if the semantic snapshot differs from the last successfully persisted snapshot. Preserve explicit drain/flush, reset semantics, bounded chunking, token payloads, and terminal failure handling. No collector/plugin modification, file rewrite, or rotation is needed.

## Scope and provenance

Inspection only: no tests, target imports, device commands, or implementation changes were performed. This report is the only file created. Root evidence was read, not modified. Startup contracts were read. A second investigator independently examined the proposed fix and existing tests.

Paths below use these abbreviations:

- `P`: `models/autoports/google_gemma_4_31b_it/tt/benchmark_phases.py` in this worktree, HEAD `e7e947f5a8c5128ea0512805b5e3ac46d21e693d`.
- `C`: `/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/workflows/run_docker_server.py`.
- `R`: sibling `run.py` in that inference-server checkout.
- `L`: `/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/bringup/artifacts/stage12-gpqa20-c4-retained-complete.log`.
- Test and adapter paths are relative to `models/autoports/google_gemma_4_31b_it/`.

`L:64–66` records dispatched metal SHA `11a387692d2b434198e79dd92fbb8ed308c3ca47` and inference-server SHA `37ee508969c730320c01259fafe340e6c2c14d4e`. The latter is the inspected inference-server HEAD. The dispatched metal object is absent from this execution checkout; reading its blob ID from the dedicated `tt-metal-stage12-publish` checkout gives `e6f228a726baaf0a66b7f5bd691aad9f84b7dd9f`, exactly equal to `git hash-object` of P. Thus the recorder analysis is revision-consistent; no removed Gemma implementation was consulted.

## Direct observations versus interpretation

Direct observations from L:

- `129461–129463`: phase path is under `cache_root/logs/gemma4-phase-{pid}-{time_ns}.jsonl`; autoexport interval is 1 second; retained decode tokens enabled. These are replayed startup logs; their embedded timestamp is 16:55:26, not the outer replay timestamp.
- `128905`: engine reports `EngineCore waiting for work` at 18:29:25.544.
- `128926–128929`: workflow/eval commands complete with rc=0 around 18:29:26.880–26.909.
- `128930`: at 18:29:27.678 the collector reports `tar: ./gemma4-phase-147-1793984275796374080.jsonl: file changed as we read it`.
- `128935` onward: atexit begins stopping the inference container at 18:29:27.693, after collection failed; subsequent shutdown logs name EngineCore pid 147.
- `129246` identifies the requested run ID in its artifact URL.

Interpretation: unnecessary periodic appends are a sufficient, source-backed explanation for mutation while otherwise idle. The log does not expose the exact writer call or the tail of the rejected JSONL; it cannot prove that the colliding write contained zero events rather than a last completion/backlog/status record. Idle engine status also does not prove the recorder had already drained all host-side events. The diagnosis is strong for the recurring idle race, while attribution of that single colliding append remains inferred.

## Complete causal chain

1. `tt/generator_vllm.py:66–81` constructs P when `GEMMA4_BENCHMARK_PHASE_LOG` is set. P reads the interval environment variable at `49–50` and starts a daemon socket/export thread at `96–99`.
2. `P:356–369` periodically invokes `_export(automatic=True)` while neither export failure nor exhaustion is set. No new model request is required. The interval is a polling schedule, not a precise hard deadline; socket requests and I/O may delay it.
3. `_export` snapshots state and chooses up to 1024 events by default (`P:45, 312–318`). With an empty queue, count is zero; the only early return is file exhaustion. `_write_export` still runs at `321`.
4. Each snapshot gets a fresh `boundary_ns` (`P:238`). `_write_export` serializes a full record plus newline and appends through its retained, unbuffered descriptor (`P:274–278, 302–307`). Even an otherwise identical idle status changes file length and modification time. The recorder's export lock serializes its own writers, not an external tar reader.
5. The workflow invokes collection after `runner.run()` (`R:1422–1434`); observed shutdown occurs later. In the named-volume path, C launches a separate tar container over the live volume and archives all of `/cache_root/logs` (`C:369–391`). A read-only mount for tar does not stop the server's writer.
6. On nonzero tar status, C logs stderr and returns zero (`C:393–400`) **before** opening/extracting stdout (`401–407`). Hence a warning about one phase JSONL suppresses extraction of all phase files and any other members in this collection attempt. It does not establish that the volume lost its files, nor that unrelated workflow artifacts were discarded.

No accelerator synchronization change is implied. The adapter records decode completion after its existing host conversion/validation (`tt/generator_vllm.py:445–470`); the recorder performs host-only file I/O. Waiting for more device work cannot stop the independent export timer.

## Recommended minimal fix contract

Add one last-successful semantic fingerprint, initially unset. A small canonical immutable serialization of the snapshot metadata is sufficient; there is no reason to hash events or reread the file.

Within the existing `export_lock`, take the snapshot and queued-event slice under `self.lock`. Compute the candidate fingerprint from that snapshot, excluding only `boundary_ns` and `events`. Retain `window_start_ns`, `buffered_events`, `pending_steps`, errors, truncated status, and the entire token-retention metadata (including disabled/failure reason/count, uncertain positions, loss, export failure, exhaustion, and pending-incomplete). Keeping the remaining identity/schema fields also avoids accidentally omitting future status fields. Freeze the comparison value before releasing the state lock; do not retain aliases to mutable metadata.

Skip writing only when all of the following hold: this is an automatic export, the event slice is empty, a successful prior snapshot exists, and its fingerprint equals the candidate. Return the fresh result as appropriate without touching the file. Preserve existing exhausted/error behavior; a skip must not hide a previously failed exporter when an explicit command would raise.

On a normal successful write, preserve deletion of exactly the exported queue prefix and remember **the pre-I/O candidate actually written**. Do not mark current post-I/O state as persisted: a producer may change errors or pending state while I/O is in flight. Leave submission/completion free of file-I/O locks. If writing fails, do not advance the fingerprint or delete queued events. New events always force export, even if their count equals the previous count.

Including `buffered_events` has an intentional consequence: after the final event-bearing record, the next poll emits one empty status record with `buffered_events=0`; subsequent unchanged polls write nothing. This preserves final queue status without rewriting a large file. The initial unset fingerprint similarly permits one initial empty status/file. A reset changes `window_start_ns` and therefore permits one new status record. Neither continuously changing `boundary_ns` nor current wall time belongs in the comparison.

Explicit `drain`/`flush` must still export even when empty (`P:337–339`), returning fresh timing boundaries; update the last-written fingerprint after these successful writes too. `check` remains read-only. `reset` retains all current rejection conditions (`P:342–348`) and does not pretend its returned status was written. `tools/collect_benchmark_roofline.py:86–105` uses explicit capture boundaries, reinforcing the need to preserve those commands.

**Limit:** this produces a stable file after queued events and metadata settle, not an atomic snapshot against new requests, explicit drains/resets, or the final status poll. With one-second polling, even the final event chunk and final empty status can overlap an early collector. Do not claim guaranteed success for this exact run timing without a controlled follow-up. Eliminating all collection races would require a collection/quiescence handshake or snapshot protocol, beyond this minimal scope.

## Exhaustion, failure, and alternatives

- Preserve the cap path (`P:279–301`): it replaces an oversized normal record with a compact terminal loss/exhaustion record, then `_export` clears the queue and the timer stops writing (`331–334, 360`). Do not describe that as successfully persisting the original events or full candidate metadata. A fingerprint implementation should leave terminal handling separate; no future automatic write is needed once exhausted. The existing compact terminal format does not contain every normal status field.
- Preserve write-failure behavior (`P:322–328`): mark loss/error and retain events; the timer stops retrying, explicit drains raise through `_write_export`, and `check` exposes state. Actual disk failure can prevent persisting its own failure status, and a partial write can leave an incomplete final JSON line. A dedup guard cannot promise on-disk error metadata in that case. These are existing limitations, not evidence of the observed tar warning's cause.
- Event-only `if automatic and not events: return` is insufficient. For example, a non-retaining recorder can export a submit, then encounter an unknown completion (`P:209–218`) which adds an error without adding an event. Reset also changes status without events. A last-event-only scheme would also leave pre-drain `buffered_events` as the final value. Retention loss/status must not depend on a later successful event to become visible.
- Comparing the full snapshot including `boundary_ns` never suppresses idle writes. Comparing only pending count misses errors and token-loss metadata.
- A dirty flag could work, but every meaningful mutation, including failure cleanup, pending transitions, and reset, must maintain it correctly. Snapshot comparison is a smaller audit surface here.
- Increasing the interval only reduces collision probability and delays token persistence. Disabling periodic export conflicts with token retention's constructor requirement (`P:59–60`). Rewriting/replacing the whole file or rotating it adds cost and lifecycle complexity outside the requested boundary. Ignoring tar failures changes stock collector policy and is outside scope.

## Discriminating host tests to add/run later (not executed)

Existing CPU-only tests import the recorder directly without device fixtures. `tests/test_benchmark_phases.py:136–175,177–219,233–276` covers background creation, concurrent drains, lock separation, cap, stale files, and write failure; `tests/test_token_retention.py:138–157,195–225` covers retention loss and reset. None asserts settled idle file stability.

1. **Idle stability:** initial automatic export, then many automatic calls with changing clocks; assert identical bytes, size, and mtime after the first record. After events, allow the final zero-buffer record, then require stability. Exercise the actual timer in a separate host test.
2. **Chunk and token completeness:** use more than one chunk, verify every submit/complete and token payload appears exactly once and in order. Refill with the same number of events to defeat count-only dedup. Verify final pending/error/loss status.
3. **Metadata-only changes:** after a settled snapshot, trigger unknown-output error without a new event; require one new status record. Exercise pending/readback cleanup and retention-disabled/loss/reason/uncertainty transitions, including a dropped diagnostic event under overflow. Each persisted change must be followed by stable idle polls.
4. **Commands:** repeat empty drain/flush and verify fresh appended boundaries. Check must not write. Reset must preserve its rejection guards and cause the next automatic status to reflect the new window exactly once.
5. **Concurrent metadata change during write:** block `_write_export` with a host synchronization fixture, mutate status while it is blocked, then release it. The next empty automatic export must persist that mutation. This specifically rejects saving a recomputed post-I/O fingerprint. Retain existing no-submission-lock-during-I/O and no-loss/no-duplication tests.
6. **Failures/cap:** stale file and symlink refusal; short, zero-progress, and partial-then-failing writes; no fingerprint advance or event deletion on failure; explicit drains still fail and check reports loss. At cap, exactly one bounded terminal loss record, no later file growth, no false claim that omitted events were exported.
7. **Collector contrast:** in a temporary host directory only, coordinate a sufficiently long tar read with old automatic empty writes and observe tar status/stdout; repeat with the changed-state implementation after it settles and verify extraction of all members. Use deterministic synchronization rather than hoping for a timer collision. Separately race the final real event/status or an explicit drain: document the remaining boundary instead of claiming total race elimination.

## Post-draft claim review

Rechecked the headline against the exact recorder blob, the collector return-before-extraction branch, and the log ordering. The source proves ongoing idle appends and whole-archive rejection; it does not prove which specific append collided. Kept terminal metadata sparsity, partial-write recovery, and residual final-drain races as limitations/other potential issues, not alternative proven causes. The proposed guard changes only unnecessary automatic writes and preserves event/token evidence and explicit control semantics within existing cap/error guarantees.
