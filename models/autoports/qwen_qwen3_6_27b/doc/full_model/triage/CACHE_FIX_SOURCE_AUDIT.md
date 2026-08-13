# AutoFix Cache Repair Source Audit

## Scope and starting evidence

Source-only independent audit; no TT device was opened and no implementation was
modified.  The starting implementation was:

- `tt/multichip_decoder.py`: inactive decode K/V suppression and inactive
  prefill page-table handling;
- `tt/model.py`: selective linear-state reset;
- `tt/generator.py`: reset/refill lifecycle and split-trace teardown;
- `tests/full_attention_inactive_kv.py` and
  `tests/full_model_mixed_slots.py`: intended focused device proofs;
- TTNN paged-cache device validation, program factories, and dataflow kernels.

The current hardware logs do not contain a successful
`FULL_ATTENTION_INACTIVE_KV_OK` or post-repair
`INACTIVE_KV_EXACT RESET_REUSE_OK` result.  The focused official-weight run was
blocked by the independently recorded mesh command-queue stall, so the two
model tests remain proposed experiments, not passing evidence.

## Hypothesis experiments

### 1. `INT32 -1` skips `paged_update_cache`

**Hypothesis.**  `ttnn.where(active_mask, cache_positions, -1)` produces one
INT32 update index per fixed slot, and the `-1` rows perform no cache read or
write during traced decode.

**Source experiment and result.**  The sentinel portion is **verified**:

- `multichip_decoder.py:863-895` typecasts positions to INT32 and passes the
  selected tensor as `update_idxs_tensor` to both key and value updates.
- `paged_update_cache_device_operation.cpp:198-251` requires the tensor to be
  INT32, row-major, DRAM-interleaved when unsharded, and to contain one index
  per padded decode user.
- Both paged reader and writer explicitly compare the per-user index to
  `(uint32_t)-1` before resolving the page table.  The reader skips cache reads
  (`reader_paged_fused_update_cache_interleaved_start_id.cpp:85-103,158-165`),
  and the writer suppresses cache writes
  (`writer_paged_fused_update_cache_interleaved_start_id.cpp:70-117,143-160`).
  The row-major input variants contain the same guard.

The exact `where` producer contract is **still uncertain**:

- The eager and trace active mask is BF16 row-major, while `cache_positions` is
  INT32 row-major (`generator.py:181-185,217-223` and
  `model.py:331-334`).  `where` therefore uses a BF16 predicate with an INT32
  true tensor and scalar `-1`.
- `ternary.cpp:101-126` only typecasts an *integer* predicate when true/false
  values are floating point.  It does not typecast this BF16 predicate to
  INT32.  The output dtype is selected from the INT32 true tensor, but the
  existing INT32 `where` unit coverage uses INT32 predicates and values
  (`tests/ttnn/unit_tests/operations/eltwise/test_where.py:101-127`), not this
  mixed BF16-predicate/INT32-value combination.
- The operation is device-only and has no host read/write, so it is structurally
  trace-compatible once its exact program is warmed.  That does not prove the
  mixed-dtype values are correct on this target.

**Verdict:** sentinel semantics verified; exact producer dtype/shape and traced
behavior still require a focused device proof.  A safer source contract would
make the predicate INT32 (or construct INT32 update indices by another proven
integer path), but no implementation change is made by this audit.

**Required experiment:** run
`python models/autoports/qwen_qwen3_6_27b/tests/full_attention_inactive_kv.py`
and add/read back `update_positions` in a minimal B2 eager control to prove
exact values `[position, -1]`.  Then capture/replay at least two steps with
changing positions and assert inactive key/value blocks remain bit-exact while
the active blocks change.

### 2. `-1` page-table entries skip `paged_fill_cache`

**Hypothesis.**  Filling every page-table entry for a zero-length slot with
`-1` prevents prefill from overwriting that slot's K/V cache.

**Source experiment and result.  Verified.**

- `generator.py:142-159` clones the generator-owned INT32 table, fills every
  entry of each zero-length row with `-1`, and supplies it only as
  `cache_page_table`; the ordinary page table remains the attention-read table.
- `optimized_decoder.py:1660-1678` stores that separate write table for the
  prefill call.
- The short prefill calls batched `paged_fill_cache` with that table and the
  persistent row indices (`multichip_decoder.py:948-960`).  The long path slices
  the same table and calls the same operation for every K/V chunk
  (`multichip_decoder.py:985-1014`); because inactive rows are entirely `-1`,
  slicing preserves the sentinel.
- `writer_fill_cache_interleaved.cpp:12-28,181-199` defines `(uint32_t)-1` as
  `SKIP_PAGE_TABLE_ENTRY`, consumes/discards input tiles for such blocks, and
  performs no write.
- The repository has a direct batched regression for this behavior:
  `test_paged_fill_cache_batched_skip_entries` in
  `tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py:1194-1215`.

This proves kernel and call-site semantics.  The model-level peer-preservation
run is still required because it also covers the Qwen mesh mapper, BFP8 cache,
long-lived cache identities, and batch-index tensor.

### 3. Selective linear conv/recurrent reset

**Hypothesis.**  Multiplying each linear layer's conv and recurrent state by a
per-slot 0/1 mask zeros selected rows and preserves live rows.

**Source experiment and result.  Shape/logic verified; runtime proof pending.**

- `model.py:170-203` validates/deduplicates slots and creates a replicated BF16
  row-major mask.  It reshapes the same mask to `(1,B,1,1)` for conv state and
  `(B,1,1,1)` for recurrent state, matching the batch axes used in
  `multichip_decoder.py:663-705`.
- Only linear-attention layers are touched; paged full-attention K/V ownership
  is unchanged.  Multiplication by exactly one preserves BF16 values exactly
  in principle, and multiplication by zero clears selected BF16/FP32 rows.
- The method synchronizes before deallocating the mask, so its temporary mask
  lifetime is safe and `model.reset_slots()` is blocking after its reset
  kernels have been submitted.
- `full_model_mixed_slots.py:76-105` is correctly shaped to prove live linear
  state preservation, reset-slot zeroing, mandatory refill, and live K/V
  preservation during peer refill, but no passing post-repair artifact exists.

The remaining uncertainty is the exact in-place broadcast implementation for
both the conv cache dtype and the recurrent cache dtype on TP4, particularly
when recurrent state is FP32.  Source shape reasoning is not a replacement for
the bit-exact focused run.

### 4. Reset/refill/decode completion and lifecycle

**Hypothesis.**  A caller may stop nonblocking token-out replay, reset a slot,
refill it, and resume without racing trace buffers or state updates.

**Source experiment and result.  Refuted for the general asynchronous caller
contract.**

- `token_out_decode_step` submits the model trace nonblocking and then submits
  the sampler trace (`generator.py:316-329`).  A caller can therefore have
  outstanding trace work when it immediately calls `reset_slots()` or
  `reset()`.
- Both reset methods call `_release_traces()` first
  (`generator.py:434-445`), but `_release_traces()` has no preceding device
  synchronization.
- `ttnn::operations::trace::release_trace` simply calls
  `MeshDevice::release_mesh_trace` (`ttnn/cpp/ttnn/operations/trace.cpp:31-34`).
  Mesh release immediately erases the trace buffer and marks allocations safe
  (`tt_metal/distributed/mesh_device.cpp:1261-1277` and
  `tt_metal/impl/sub_device/sub_device_manager.cpp:136-143`); it is not a queue
  completion barrier.
- Consequently, releasing after a nonblocking replay is not source-proven safe.
  The reset-slot synchronization occurs only *after* trace release.  It orders
  and completes the subsequent multiply kernels but cannot retroactively make
  freeing the outstanding trace buffer safe.
- `model.reset_slots()` is blocking at return, as noted above.  In contrast,
  whole-model `reset_cache()` enqueues in-place zero multiplies but has no final
  synchronization (`model.py:164-168`), so `generator.reset()` does not promise
  reset completion at return.
- Eager `decode_forward()` returns through a host logits readback, which is a
  natural completion boundary.  Public prefill likewise reads logits before
  returning.  The unsafe transition is specifically nonblocking token-out
  replay directly followed by release/reset (or teardown) without an explicit
  caller synchronization.
- The refill gate itself is logically sound: reset adds slots to
  `_slots_requiring_prefill`; eager and trace setup reject active decode for
  those slots; successful positive-length prefill removes only the refilled
  slots (`generator.py:164-178,293-306,440-445`).

**Required fix/proof:** synchronize the mesh before releasing live model and
sampler traces (or prove and use an explicit runtime primitive that fences a
trace before release).  Define whether `reset()` is blocking; if it is the
serving reset boundary, synchronize its cache clears before return.  Add a
focused sequence that submits `token_out_decode_step(readback=False)`, calls
`reset_slots()` immediately, refills the reset slot, resumes decode, and proves
both live-slot preservation and reset-slot freshness.  Run it repeatedly with
watcher enabled in a separate non-profiled job.

## Final status

**Still failing source audit.**  The underlying `-1` update and fill sentinels
are genuine, and the selective reset masks have the correct semantic axes.
However, completion cannot be claimed until:

1. the mixed BF16-predicate/INT32-value `where` path is proven or replaced by a
   proven INT32 producer;
2. the Qwen TP4 inactive K/V and reset/refill focused tests pass on recovered
   hardware; and
3. trace release is fenced before freeing buffers after nonblocking replay,
   with a repeated lifecycle test proving the transition.

No implementation change was made during this source-only audit.
