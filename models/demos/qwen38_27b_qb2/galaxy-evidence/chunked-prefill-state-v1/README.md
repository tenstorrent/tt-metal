# Chunked-prefill state prerequisite, Oct 9 2026 UTC

The current qualified-source launch **disables scheduler chunked prefill**.
The plugin already interleaves decode after bounded counts of prefill steps,
but one current step can include a whole long prompt. Internal model chunking
limits activation memory; it does not return control to the scheduler between
chunks. This is a source-supported explanation for long decode pauses during
new prompt admission, not a complete timing attribution.

The pinned plugin also has a correctness defect that must be fixed before
enabling chunked prefill for a recurrent model. `_alloc_prefill_state_slots`
excluded all scheduled prefills from held slots and assigned them new slots
based on their current host rows. A prompt continuation needs the state from
its previous chunk. Changing its ownership map without a device gather makes
it consume another request's state. New arrivals could also overwrite the
state of a continuation scheduled later in the same batch.

## Reproduction and fix

Four host regressions fail against `b7e4292e4193cba20abe9c7c68ce489201b2e36b`:
an earlier request finishing, reordered continuations, a new arrival preceding
continuations, and a continuation following an interleaved decode gather.
Tests model actual device-slot contents independently of the ownership map.
The fix reserves every live owner slot and keeps a continuation at its existing
slot. Only new/preempted requests allocate free slots. Decode retains its
existing explicit gather/permutation path.

The corrected original-code run reports **4 failed, 15 passed**. The fixed
code reports **167 passed** across state ownership, chunked-prefill policy,
decode interleaving, model-runner and block-scheduler tests. Pre-commit passed.
An intermediate test incorrectly assumed one exact decode permutation; it was
changed to derive expected ownership from the independently gathered state,
then both original and fixed arms were rerun. An earlier invocation named a
nonexistent host test file and collected no tests. These failures are retained.

Upstream push access was denied; GitHub permissions confirmed read-only access.
The fix is pushed to the user's public fork:

- Repository: <https://github.com/anatarajan-tt/vllm-tt-plugin>.
- Branch: `anatarajan/qwen38-chunked-state-slots-20261009`.
- Commit: `e5b02d58bda26ee326fe0e7cbed9e4828adb9426`.

The same allocator is present on fetched upstream main `c62035d8f16eb591b258a7d5f2b329495829a74c`.
That main has advanced to vLLM 0.29; this isolated fix stays on the deployed
0.26 plugin pin to avoid changing unrelated runtime dependencies.
No upstream PR or deployment promotion is claimed.

## What remains before a scheduling experiment

1. Qualify Qwen's partial-prefill state/position handling, including unaligned
   boundaries, reordered slots, intervening decode and sampling-stream continuity.
2. Declare the model's existing `supports_chunked_prefill` capability and test
   through the actual plugin consumer. Intermediate chunks already use host
   sampling with cloned RNG state and suppress output in this pinned plugin.
3. In a separate deployment, enable chunking and compare a smaller scheduler
   token budget plus decode-interleave cadence against the unchanged control.
   Smaller quanta can reduce decode gaps but add launches, trace/state switching
   and reduce prefill batch efficiency; no throughput uplift is measured yet.
4. Retain matching source pins, G0, full GPQA and physical HTTP/agentic checks
   before selecting a new release configuration.

This defect is **not an explanation for the current GPQA misses**: the relevant
chunked-prefill path is disabled there. Existing hardware queues, plugin/image
pins and model capabilities remain unchanged.

Evidence: [summary](report.json), [original failures](state-slots-before-v2.xml),
[fixed tests](state-slots-after-v3.xml).
