# AutoDebug: deferred token delivery

Investigation: source only, 2026-09-12. No device run, implementation edit,
decoder-policy change, dtype search, or vLLM work was performed. This report
applies the AutoFix and TT Enable Tracing investigation guidance. Hardware
claims below are hypotheses until the focused probes pass.

## Starting evidence and verified source finding

Stage 7 requires complete token output while the steady decode loop performs no
per-token host synchronization or readback. The existing low-level path already
has separate model/sampling traces, nonblocking replay, persistent token
feedback, and device position advancement:

- `tt/generator.py:216–229`: `_model_step` advances positions and RoPE on device;
  `_sampling_step` passes `tt_out_tok=self.tokens` and advances sampler seeds.
- `tt/generator.py:343–356`: `decode_forward` submits both traces with
  `blocking=False`, but its default `read_from_device=True` then calls
  `_read_tokens()`.
- `tt/generator.py:480–483`: `generate` uses that default on every step and
  converts the returned token to a Python integer immediately. Thus the
  nonblocking submissions still serialize with every output read.
- `tt/generator.py:58–60`: `_read_tokens` converts device 0's persistent token
  buffer to Torch. Saving that same device tensor repeatedly in a Python list
  would save aliases, not past sampled values.
- `models/common/sampling/tt_sampling.py:1109–1122`: common split sampling seeds
  and draws on device, writing directly into the supplied `output_tensor`.
  The force-argmax path likewise honors the output buffer at lines 884–891.

`doc/full_model/performance.md` reports 39.055 tokens/s and explicitly excludes
steady feedback uploads/full-logit reads, but does not establish absence of
per-token output reads. Its accounting names output-token delivery as a terminal
cost. The README's Stage 6 numbers therefore remain baseline evidence, not proof
of the stronger Stage 7 delivery contract.

## Primary hypothesis: device-indexed history in the sampling trace

Allocate a persistent, replicated UINT32 ROW_MAJOR DRAM history with shape
`[C, 1, 1, 32]` and a replicated UINT32 ROW_MAJOR DRAM cursor of shape `[1]`.
Here `C` is the reserved number of sampled decode outputs. The existing token
buffer already has compatible shape `[1, 1, 1, 32]`. A generator-owned append
after common sampling can use existing APIs:

```python
updated = ttnn.indexed_fill(history_cursor, token_history, self.tokens, dim=0)
ttnn.copy(updated, token_history)
ttnn.plus_one(history_cursor)
```

Capture these three operations in the sampling trace, after `tt_out_tok` has
been written. The model trace remains separate and the next model replay still
reads `self.tokens`. Do not replace token feedback with a history slice.

Source support:

- `ttnn/cpp/ttnn/operations/data_movement/indexed_fill/indexed_fill_nanobind.cpp`
  exposes `indexed_fill(batch_id, input_tensor_a, input_tensor_b, *, dim=0)`.
- `.../indexed_fill/indexed_fill.cpp:34–79` checks matching ranks/non-indexed
  dimensions and dispatches ROW_MAJOR dim 0 directly, without a permute.
- `.../indexed_fill/device/indexed_fill_device_operation.cpp:41–106` supports
  rank 4 ROW_MAJOR data and INT32/UINT32 ROW_MAJOR index tensors. This path imposes
  no BF16-only restriction on data; the factory sizes pages from the actual
  element size at `.../indexed_fill_program_factory.cpp:211–249`.
- `.../indexed_fill/device/kernels/dataflow/indexed_fill_reader.cpp:53–59`
  reads index values from the device tensor. Its generic path at lines 236–261
  chooses the replacement input using those values on every execution. A cursor
  value is therefore data, not a captured host offset. UINT32 token values are
  moved as bytes, with no lossy BF16 conversion required.
- `.../indexed_fill/device/indexed_fill_device_operation.cpp:196–198` allocates
  a new output. **The copy-back is necessary:** assigning the Python history
  name to the returned tensor alone would leave replay bound to the original
  input allocation.
- `ttnn/cpp/ttnn/operations/data_movement/copy/copy.cpp:13–15` passes the
  destination as a preallocated output. Its device operation accepts UINT32
  and ROW_MAJOR tensors and returns that destination (`device/...cpp:76–128,
  191–196`). Both operands must retain the same shape/layout/dtype.
- `ttnn/cpp/ttnn/operations/experimental/plusone/device/plusone_device_operation.cpp:12–33`
  supports INT32/UINT32 ROW_MAJOR rank 1–4 input and returns the same tensor.

No existing exact UINT32/TP4/history trace test was found. The current
`tests/ttnn/unit_tests/operations/data_movement/test_indexed_fill.py` covers BF16
data and integer indices, so the exact-shape probe below is required.

The implementation copies the complete history in `indexed_fill` and then
copies it back: O(C * 32) bytes per append, not O(32). At C=128 the logical history
is 16 KiB/device; at C=262144 it is 32 MiB/device, before scratch output. For a
fixed requested generation length G this becomes O(G²) aggregate traffic.
Reserve request-sized capacity, retain a reusable capacity only while sensible,
and measure both S128/G128 and a larger capacity. This is a correctness-first
candidate using existing ops, not an unmeasured performance improvement.

## Minimal integration boundaries

1. Keep the first-token read at `generate:465` if preserving the current TTFT
   definition. Defer the G−1 decode outputs only; C then needs G−1 entries.
   A G1 request needs no decode history. Alternatively include the first token
   in history, but explicitly synchronize/read a first-token boundary before
   recording TTFT; measuring only submission time would be incorrect.
2. For the normal device-sampling path without a `next_input` callback, call
   `decode_forward(..., read_from_device=False)` and do not access its token
   contents in the loop. Retain compatibility/teacher-forcing semantics:
   `next_input(step, predicted)` requires the actual predicted integer and
   cannot be deferred without changing that API.
3. After all decode submissions, read history once and select
   `history[:num_decode_outputs, 0, 0, :batch_size]` on the host. For B1 combine
   column 0 with the separately read first token. The final read/drain belongs
   in `decode_s`; otherwise the reported throughput measures enqueue time.
4. Count per-step token reads separately from final history readbacks and
   request/TTFT reads. A final drain outside the replay-loop counter interval
   must still be inside the timed decode interval.
5. Preserve low-level fixed-slot decode as its own supported interface. A
   history row naturally holds all 32 physical slots without reordering.
   Returned low-level values should still select the bound batch slots;
   inactive slot outputs must not be treated as active request tokens.
   History capacity is independent of absolute prompt/current positions and
   page-table IDs. Do not use `positions` as the output-history index.

## Trace lifetime, reset, and capacity requirements

- Allocate persistent history/cursor before either trace is captured. Growing
  or replacing history must first release traces that hold its old address.
  Reuse stable history storage/cursor across matching requests.
- Recording enabled/disabled is a sampling graph variant. Include that mode,
  history allocation identity/geometry, and current sampling branch in trace
  validity. The existing `_release_traces` can conservatively release both
  traces on a variant change. Retain existing invalidation for cache rebinding,
  new prefill shapes, active-slot changes, and argmax/split branch changes.
  Ordinary k/p/temperature updates in the split branch already update the
  sampler's persistent parameter tensors in place (`tt_sampling.py:593–636`).
- A low-level call that does not request output recording must not unknowingly
  replay a recording trace whose history budget is exhausted. Either bind a
  recording mode explicitly or keep history management outside that public
  path. Do not silently impose G128 as a low-level decode limit.
- Reset cursor and host-side valid-output count at request boundaries. Enforce
  the remaining history capacity on the host from submitted step counts,
  independently of `remaining_steps` for the KV cache. Reject overflow before
  replay; the generic indexed-fill reader simply fails to match an out-of-range
  index, so missing output can otherwise be silent.
- `_capture` warms `_sampling_step`, which will append one extra token. Restore
  the history cursor alongside tokens/positions/RoPE/seeds before capture.
  Full history backup is unnecessary if warmup writes only the next unused row:
  restore the cursor and let the first real replay overwrite that row, while
  exposing only the valid prefix. Prove this with capture at a nonzero cursor.
  Do not increment host valid count during warmup/capture.
- Warm indexed-fill/copy/plus-one with the exact history shape and trace mode.
  Keep the temporary indexed-fill output local; only the history and cursor
  must persist. Repeated requests can reset only the cursor and read the valid
  prefix; stale tail entries must never escape as output.
- Preserve arbitrary prompt lengths and public context validation in the
  existing prefill/cache code. A history change does not require modifications
  to `tt/model.py` or any decoder precision policy.

## Alternatives and source-based refutations

| Candidate | Assessment |
| --- | --- |
| Save `self.tokens` or the return from non-reading decode in a list | Refuted: every entry aliases one overwritten feedback buffer. |
| Append a newly allocated `ttnn.clone(self.tokens)` after each replay | Allocation-lifetime hazard: `tt_metal/impl/allocator/allocator.cpp:118–133` warns that allocations made with live traces can be overwritten by replay. `trace_allocation_tracker.cpp:117–133` records these survivors. Preallocated independent output slots plus eager `ttnn.copy` avoid that specific hazard and provide a useful controlled alternative, but add host-dispatched work per token and many allocations. |
| `ttnn.copy(self.tokens, history[i:i+1])` | Refuted as an in-place parent update: TTNN slices are new outputs; the model itself documents this at `tt/model.py:253` and copies reconstructed state back. |
| One repeated `experimental.slice_write` trace with an incrementing Python start | Refuted: binding only accepts host start/end/step vectors; `slice_write/device/slice_write_device_operation_types.hpp:14–18` stores Shapes, and the factory bakes their start offset into runtime arguments. Replay cannot observe the Python loop index. An eager call per step is an alternative worth measuring, but requires offset variants/warmup and host dispatch. Its BF16-only docstring is stale relative to permissive native validation; UINT32 support still needs a probe. |
| `ttnn.scatter` directly into persistent history | No preallocated-output API: `scatter.hpp:14–21` returns an out-of-place result (`device/scatter_device_operation.cpp:77–80`), still requiring copy-back. Non-last-axis scatter also adds transpose work; indexed-fill dim0 is simpler here. |
| Use common sampling's output state | Refuted as ordered history: `tt_penalties.py:294–360` maintains vocabulary counts/masks, not the generated sequence. |
| Reuse paged KV update as a UINT32 history writer | Unsupported direct match: `experimental/paged_cache/device/update_cache/paged_update_cache_device_operation.cpp:42–47,295–297` requires TILE floating-point cache/input. BF16 token storage would lose token-ID precision. |

## Focused verify/refute experiments

These are proposed experiments, not executed commands/results. The coordinating
agent should run them through the existing device-ownership/recovery workflow.

1. **Exact history component probe.** Allocate the proposed replicated history,
   scalar cursor, and exact `[1,1,1,32]` UINT32 token buffer on TP4. Initialize
   tokens to `70000 + arange(32)` to catch any floating-point conversion. Capture
   `plus_one(tokens); indexed_fill; copy-back; plus_one(cursor)` after exact warmup
   and reset. Submit 129 nonblocking replays with no intervening reads. Read
   every replica after the last submission and assert exact equality to
   `70000 + arange(32)[None, :] + arange(1,130)[:,None]`, cursor=129, unchanged
   unwritten tail, and unchanged tensor buffer addresses. Use C=129/C=257 for
   those 129 replays and also test C=1 with one replay. If UINT32
   transfer/cursor/capture fails, refute this candidate before
   changing production generation. Enable program-cache-miss rejection during
   capture and allocation tracking in this correctness-only probe.
2. **Warmup/capture and reset probe.** Append a real row eagerly, then warm and
   capture at cursor=1 using the proposed restore logic. Replay several steps;
   assert no extra/skipped row and preservation of row 0. Reset only the cursor,
   change initial token values, replay a shorter request, and compare only the
   new valid prefix. Verify the last valid append and rejection of one extra
   submission without a device read. Grow history after releasing traces and
   verify the replacement binding. Exercise recording off/on switches.
3. **Reduced real generation A/B.** Use real layers 0/3, embeddings, selected
   head, split sampler, and identical seeds. Compare every token from the
   existing immediate-read path with deferred history for S128/G128 and
   unaligned prompts such as S1/S33/S129, G1/G2/G129/G257. Alternate greedy and
   sampled requests, long and short requests, and repeated identical requests.
   Require exact token lists and lengths. Inspect token/position/RoPE/seed
   state only after complete runs, and prove replay N's token feeds replay N+1.
4. **Low-level contract regression.** Bind external caches at B1/B32; include
   nonzero fixed slots, inactive slots, nonzero starting positions, unchanged
   and changed page tables. Compare the existing immediate-read outputs against
   recorded history with the same supplied token/position schedule. A request
   history budget must not alter cache capacity, slot identities, or resets.
5. **Performance and full-model acceptance.** Run the existing S128/G128 harness
   with two matching requests so trace capture is excluded from warmed timing.
   Require G−1 model replays, G−1 sampling replays, zero reads/uploads/syncs in
   the normal steady replay loop, one final history read, and decode timing
   that includes device completion. Repeat with C257 and a larger C to expose
   the O(C) append cost. Profile the component/reduced model separately, then
   confirm complete full-model token lists, accuracy and qualitative gates.
   Do not infer a speedup merely from removal of `_read_tokens` calls.

## Verdict

The root host serialization is verified in source. Device-indexed persistent
history plus copy-back is the smallest identified existing-op strategy that
keeps the steady loop as two trace submissions and preserves all sampled
outputs. Its exact UINT32 TP4 trace behavior and net latency remain unverified.
The component probe should decide whether to integrate it; capacity scaling
must remain a measured limitation rather than a hidden context restriction.
