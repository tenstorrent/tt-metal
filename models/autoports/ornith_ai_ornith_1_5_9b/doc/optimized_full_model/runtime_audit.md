# Runtime and lifetime audit

The selected complete-model path is `build_generator(... decode_head="dram",
selective_inactive_restore=True)`. It executes 24 linear-attention and eight
full-attention layers on mesh 1x4. `decode_resident` queues the existing model
trace followed by the existing common-sampler trace, both nonblocking on CQ0.
It returns a borrowed final device token buffer; it does not accumulate output
history for a caller. Streaming `decode_forward` retains its explicit async
submission/read/process interface.

## Boundary accounting

| Boundary | Selected behavior | Evidence |
|---|---|---|
| Prompt and scheduler | Explicit host admission, prompt lengths, fixed-slot capacity, positions and page tables; validation precedes mutation | Existing serving contract; host tests; full B32 capacity probe |
| Embedding | Device token lookup of hidden-sharded BF16 weights, one hidden gather | Reduced full-model decode report |
| Residual stack | Existing BF16 locally width-sharded L1 layout, no inter-layer conversion | Unchanged optimized decoder source; layer report |
| Final norm/head | Local RMSNorm in residual layout, BF8 DRAM-sharded HiFi2 head, BF16 vocab shards | Head sweep and final report attributes |
| Sampling | Common `TtSampling`, physical top32 candidates per shard, semantic k1 for greedy; top-k/top-p, penalties and seed modes remain prepared | sampler_choices.json; sampling_transitions_final.json |
| Token feedback | Common sampler writes the same persistent `tt_out_tok` used by the next model trace | Resident final-token comparison; fallback tripwires |
| Positions/RoPE | Device `plus_one` preserves negative inactive positions; RoPE consumes device position indices | Exact position assertions; mixed B32 slot checks |
| Page tables | Persistent; only explicit scheduler reload commands upload tables (all fixed rows) | Resident counters remain zero; existing changed-table contract |
| Inactive recurrent state | Bit-copy only inactive FP32 rows before/after native arithmetic | inactive_copy_tracked_final.json; selective_batch32_tracked.json |
| Host boundary | No conversion, H2D copy, readback or synchronization inside resident submission loop | `full_model_perf_probe` patches these TTNN APIs to raise; final synchronization occurs after the entire window |

`host_sampling=True` and `model.host_logits` remain explicit diagnostic/reference
interfaces. The measured token-out path does not select them. Greedy is never
silently redirected to torch argmax. `runtime_fallback_final.json` additionally
traps torch/Tensor argmax and full-logits readback during real generation. The
readback throughput row intentionally includes the caller's scalar token reads;
it is separately labeled from resident decode and teacher forcing.

## Trace ownership

Weights, sampler buffers, token/position/RoPE inputs, scheduler-managed page tables,
CCL scratch, masked state snapshots and the int32 active-row mask all predate
capture. The new raw-copy operation adds no allocation acknowledgment, tracker
exemption or host interaction. Its dataflow CBs are program-owned and tracked.
One shared snapshot serves serial layers; no snapshot alias escapes a layer.
The copy is bitwise FP32 and skips active rows. Scheduler permutations use a
separate **full** snapshot, so the selective optimization cannot drop active
state during remapping. Full-pool restore is retained as a diagnostic control.

The decoder stack's exact-output acknowledgments and CCL ownership invariants
are unchanged from the reviewed full-model stage. Tracking-enabled evidence
includes program-cache allocations (`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`).
Full B32 replay, scheduler lifecycle, changed masks and the isolated copy probe
all exercise the new branch. Watcher is separate from profiler collection.

`decode_resident` refuses uninitialized/reset state, host-logit-only state,
outstanding reads, closed generators, context overruns and unmapped causal pages
anywhere in the requested window. Ordinary resident `decode_forward` performs
the same pre-mutation admission. `resident_page_guard.json` verifies sparse-page
rejection, mapped boundaries and changed-table continuation with tracking. Prefill and slot
reset invalidate resident eligibility for the affected cache binding. It
preserves the existing current sampling state; admission/reconfiguration stays
caller-owned. Host tests cover these guard conditions and external cache
ownership. No serving adapter or vLLM integration was introduced.
