# B1: paired real PP routing and timing capture

Read-only source review of `akhan/mistral4-prefill-followups`, 2026-09-11. This is an instrumentation design, not an executed experiment or proof of the historical layer variance.

## Minimal instrumentation points

Paths below are relative to the repository. Line numbers refer to the reviewed source.

- `models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py:642` (`prefill`): bind a capture identity containing request_id, cache/user identity, actual_start, actual_end, rank, and global layer slice. The eager transformer call at line 807 passes `actual_isl=actual_end-actual_start` at line 817. Preserve these exact values. Do not infer valid tokens from padded shape or layer ordinal.
- `.../tt/tt_prefill_transformer.py:473`: current `forward_layer_{i}_start/end` markers use LOCAL layer index. Add/request-log the global offset `self.first_layer_idx`; on a uniform 36-layer PP4 split, rank 2 local 0 and 5 are global L18 and L23. Verify the actual split from `models/demos/common/prefill/runners/prefill_runner.py:480` before analysis.
- `.../tt/moe/tt_moe.py:673`: gate produces scores, indices, logits. Routing setup consumes exactly these indices at lines 681–686. Do not perturb this path with a host read.
- `.../tt/moe/tt_moe.py:805`: indices are already moved to DRAM in ordinary execution, after dispatch; this is the smallest place to retain the actual dispatch indices in a diagnostic collector. Keep a strong tensor reference for ONLY the selected L18/L23 forwards. No clone, conversion, tensor logging, or host synchronization here. Carry global layer identity from constructor `layer_idx` (line 218); it is otherwise used for weight setup rather than stored explicitly for this hook.
- Drain the collector only after the entire selected request has finished on all ranks and the measurement boundary has closed; synchronize then convert per device. Record logical coordinate and physical device ID for each shard rather than relying on implicit concatenation ordering. Release references after export.

## Why existing debugging flags are unsuitable

`TtMoe.forward(return_intermediates=True)` performs host reads of expert counts and offsets at lines 909 onward, retains large tensors, and alters allocation behavior. `TtPrefillBlock._moe_path` at line 788 discards the returned intermediates anyway. `debug_token_count` also performs in-forward `to_torch` reads at lines 693–704. `TT_PREFILL_BLOCK_TIMING=1` inserts synchronizations at block lines 613, 723, 751 and reports host timings. Keep all three disabled for kernel profiling. Retaining only already-created DRAM indices adds buffer lifetime and some Python overhead; it is low intrusion, not zero intrusion. Compare paired-capture-enabled versus disabled runs under identical inputs before treating timings as representative.

## Mapping and validation

`.../tt/moe/init_helpers.py:195`, `ExpertMapping.create_dispatch_table(128,8,1)`, maps all expert IDs 0..127 to destination logical chip `expert_id // 16`; table column 128 is the padding sentinel mapped to -1. Export the actual table and runtime topology as provenance. PP stage routing must not call `load_captured_routing`, whose existing purpose is remapping one TP4 column and discarding other columns' routes.

For every selected forward save raw per-shard indices, source logical/physical IDs, original shape, top-k=4, per-shard valid-token bounds, padding side, and actual start/end. Validate four distinct real experts per valid token, IDs in 0..127, and total valid assignments = valid tokens × 4. Exclude sentinel 128 and padded rows; do not reinterpret real expert 127 as padding. `get_sp_mesh_composer` at line 926 can compose gate outputs (replicated across TP), but explicit shard exports make source ordering auditable.

Count assignments per expert, destination device, and source→destination pair. Also count unique token-destination pairs because dispatch multicast can share payload across experts. Neither assignment count is itself network-byte traffic.

## Timing join and experiment sequence

Use a small warmed EAGER PP4 run first. Add host signposts containing rank, request/chunk ID, and global layer around the existing boundaries; preserve every device's Dispatch, Combine and routed-FFN kernel duration. Join by request/layer, operation ordering/global-call-count, and device ID. Compare L18 versus L23 within the same request and rank, across repeated requests/chunks. Report both per-device measurements and the historical-compatible sum of per-operation device maxima, explicitly not layer wall-clock latency.

After this paired dataset establishes whether imbalance accompanies the difference, replay captured routes under fixed shapes, buffers, topology and warmup; test balanced and controlled expert/destination permutations separately. This distinguishes sensitivity from causality and does not establish an end-to-end improvement by itself.

Traced capture needs separate design: Python hooks run during capture, not replay; retaining one address only observes its latest contents, and replay-specific request identity cannot come from a capture-time closure. Runtime documents this at lines 738–741 and uses `_trace_request_id`. A one-replay diagnostic snapshot drained after completion can be valid if the retained buffer is truly live through the trace and not recycled. Multi-replay routing history requires dedicated output slots/device copies or serialized post-replay exports, each altering the measurement regime. Do not silently apply the eager collector to traced throughput runs.
