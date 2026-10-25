# Recurrent snapshot copy contract

`tt/serving_state.py:ServingStatePool(model, active_states, num_blocks_by_layer)`
allocates a separate conv/recurrent pool for every selected linear layer in
original model-layer coordinates. Each layer may have its own scheduler block
count; `.pools[original_layer_index]` exposes its `LinearState`. Attention has no
entry. Live request storage remains in `active_states`; pool storage never aliases
live state. The parent adapter passes vLLM-owned block IDs and owns hit/boundary
semantics. This helper does not infer a cache hit from tokens or timestamps.

Methods:

- `warmup()` compiles save and restore before model trace capture. It saves live
  slot0 to each new pool's block0 and restores it; live state is unchanged.
- `save(slot, block_ids)` copies that live row to the supplied scheduler block for
  every selected linear layer.
- `restore(slot, block_ids)` copies each exact saved boundary into independent
  live storage. Snapshot storage and other live slots remain unchanged.

The adapter must save only complete semantic boundaries, restore only scheduler
hits at those exact boundaries, and preserve active state between boundaries.
Block IDs may differ across layer groups. Pool lifetime and reassignment belong
to vLLM. A prior snapshot is immutable until vLLM reuses its physical block.

The generic dataflow kernel copies raw pages without arithmetic: convolution is
three5120-byte BF16 row-major pages per state; recurrence is1924096-byte FP32
pages per state. Up to8 workers copy disjoint pages. One persistent `[L,8]` uint32
metadata tensor contains dynamic source/destination row IDs. Addresses are runtime
arguments; geometry and allocations remain fixed. No entire-pool gather, transpose,
scratch clone, host state readback, or per-decode snapshot copy occurs. Metadata
uploads and Python descriptor construction happen only at save/restore boundaries.

All pool buffers and metadata exist before capture. The generic operation creates
no device outputs; it writes only the destination row of preexisting storage.
Both directions must be warmed before capture. No corruptible-buffer exemptions
or broad allocation scopes were introduced. Copying/restoring on the same CQ is
ordered with model execution; callers must drain pending scheduler transitions
before changing request ownership.

`tests/probe_serving_state_pool.py` is the isolated validation: two synthetic
linear layers, different physical block IDs, exact FP32/BF16 restoration into a
different live slot, preservation of another live slot, prior-snapshot
immutability, and a retained trace replay after changed copy IDs. It reports no
model accuracy or performance. The first run found TTNN Shape does not accept
Python slices; converting to tuples fixed that host setup error. Kernel and
allocation-tracker validation remain pending until `state_pool_probe.json`
contains `status: pass` from the stage owner's serialized run.

The stage owner's corrected run passed: `readiness_vllm/state_pool_probe.json`
and `state_pool_probe_retry.log`. Exact restoration and peer preservation passed
for both fields in both layers; old snapshots remained unchanged. The retained
trace replayed after changing physical snapshot IDs, with one capture and no
retirement during the workload. Two five-block pools occupied8,017,920bytes per
device. This validates the primitive and its allocation lifetime, not scheduler
hit correctness or all-layer capacity.
