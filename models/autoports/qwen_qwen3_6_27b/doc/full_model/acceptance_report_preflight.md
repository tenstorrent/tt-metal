# Acceptance reporting preflight

Source/CPU-only review, 2026-11-14, after the parent added UINT16 CCL accounting and the slot probe's provisional report. Inspected `tests/full_model_evidence.py`, `tests/probe_full_slots.py`, `tests/full_model_context_acceptance.py`, `tests/profile_full_terminal.py`, and `doc/full_model/native_acceptance_queue.py`, including its imported queue implementation and pinned readiness runner exit behavior. No devices, long tests, production changes, or test changes were made.

**No additional deterministic late-reporting failure was found for the currently queued B32/full64 command, maximum-context command, or performance command.** Two concrete slot-probe CLI defects remain outside the current queue's arguments. They do not justify interrupting the active B32 run.

## Findings

1. **A fresh output directory loses the completed slot report.** `probe_full_slots.py:24` accepts any `--output` path, but the first write at line123 does not create its parent directory. Unlike the context/performance probes, this probe performs all preparation and assertions before that write. For `--output <new-directory>/result.json`, `Path.write_text` raises `FileNotFoundError` after the hardware work, and no provisional result is saved. Minimal repair: create/validate the output parent immediately after parsing arguments. The native queue writes into the already-existing `doc/full_model`, so its current command is unaffected.

2. **`--batch 1` is accepted but cannot run the slot assertions.** `probe_full_slots.py:23` accepts any integer and model/generator B1 is valid. After full preparation and an initial decode, line76 assigns `position[1] = -1`; subsequent assertions and `release_request(1)` likewise require slot1. A B1 invocation therefore fails with an indexing error after expensive preparation. Minimal repair: reject batch sizes below2 before opening the mesh, or explicitly document/enforce this probe's multi-slot domain. The current queue supplies32 and is unaffected. This is an invalid probe invocation, not missing B1 model support.

## Acceptance-status boundary verified

`native_acceptance_queue.py` delegates to `acceptance_queue.py`, which records `passed_exit0` solely from subprocess return status. The pinned0.1.24 `run_prefill_check.py` and `run_teacher_forcing.py` print aggregate top-1/top-5/top-100 and return normally without enforcing the user's thresholds. Consequently, `passed_exit0` is an execution result and **cannot itself certify top-5>=98% / top-100=100%**. Read and record the actual aggregate values before declaring those gates passed. The queue currently labels this status explicitly; no finding is made that the parent has confused it with accuracy acceptance.

The resumed-prefill runner does return a nonzero status for a failed/inconclusive record; its default artifact path matches the queue. The autoregressive runner's default metadata/completion directory also matches the queue. Its qualitative acceptance remains the parent's explicit reading of both completions, as intended. The reused performance tuple resolves to `performance_final` with `--performance --all-layers`, tracking/profiler disabled by the queue, rather than accidentally reusing a preceding native job.

## Rechecked after the UINT16 failure

- `ccl_inventory` now recognizes BF16, BF8_B, FP32, UINT32, INT32, and UINT16; UINT16 tile bytes are2048. The known failing dtype is covered. Its selected-policy DRAM/interleaved/tiled assertion remains intentional. Deduplication by simultaneous buffer address avoids double-counting aliases on the selected one-mesh lane.
- The slot probe writes `functional_assertions_passed=True` with `passed=False` before invoking CCL accounting, then promotes `passed=True` only after the inventory succeeds. A future inventory error preserves the completed functional evidence. Source identity and memory sampling precede that provisional write; their current referenced paths and return types were checked.
- `source_identity()` resolves its seven model/native paths correctly from this checkout. The added native paths use the repository root, not the model directory. The returned data are JSON-serializable.
- The context probe samples CCL inventory before the long maximum-context prefill. It saves the completed prefill measurement before decode and saves success/error state in `finally`. Prompt construction yields exactly262143 tokens; one decode consumes the final legal position262143 and expects next position262144. Its page1..8192 mapping deliberately allocates8193 pages with page0 reserved. KV payload formulas correctly use BF8_B tile bytes rather than BF16 element size.
- Performance arguments reject nonpositive repeats before mesh construction, so median calculation has a populated run list. The recorded timing keys agree with `QwenGenerator.last_timings`. Counter expectations match128 sampled tokens/127 decode steps and one initial position/RoPE setup. Output parent creation and incremental/final saves are present. Profile setup hooks are removed before measured windows, and capture-active guards prevent profiler drains within capture.
- Report structures inspected contain Python scalars/lists/dicts rather than TT tensors or opaque trace objects. The profiler snapshot explicitly stringifies trace IDs; context records only trace equality. Mesh memory fields are explicitly converted to Python ints.

## CPU checks

All five requested files parsed successfully with `ast.parse`. Executed `source_identity()` without TTNN/device imports and verified seven existing source paths and JSON serialization. Executed `ccl_inventory()` with a standard-library fake TTNN module and six one-tile fake tensors, one of each supported dtype, plus a duplicate-address alias. Expected allocator-byte sum, six distinct buffers, and JSON serialization all passed. No native/device operation was executed.

The active acceptance remains owned by the parent. This note supplies no new hardware pass or stage verdict.

## Closure of the two CLI findings

Re-read `probe_full_slots.py` after the parent's repair. The parser now uses `choices=range(2, 33)` for `--batch`, rejecting B1 before mesh construction. Immediately after argument parsing it calls `args.output.parent.mkdir(parents=True, exist_ok=True)`, before any model/device work. Both reported CLI findings are closed by source inspection. No hardware rerun was needed for this closure.
