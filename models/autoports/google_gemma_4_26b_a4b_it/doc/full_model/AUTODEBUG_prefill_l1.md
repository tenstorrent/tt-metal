# AutoDebug: full-stack prefill L1 collision

## Verdict

The verified reproducing reservation is the **four persistent decode-router
buffers per decoder**, not sampler logits, KV cache, or learned norm weights.
Each `GeneralizedRouter` keeps four 2,048-byte shards on worker `(10,9)`:
`bias`, `indices`, `output`, and `output_indices`. Thirty layers retain
245,760 bytes; the reduced two-layer model omits 229,376 bytes of this state.
The normal L1 allocator's shared address frontier makes these shards constrain
SDPA's `(0,0)-(7,7)` CB grid despite the different physical worker placement.

The parent's exact reservation control reproduces the same SDPA collision:
`probe_l1_routers_retry.log` reports CB end **1,315,840** and lowest occupied
L1 address **1,301,504**, a 14,336-byte overlap. Adding only the missing 28×4
router buffers to the passing reduced/sampler/full-semaphore control therefore
proves this retained router footprint is sufficient to cause the failure.
The 3,840-byte frontier difference from the original full model is consistent
with remaining allocation/history differences; this control does not identify
each smaller allocation. The exception reports the lowest occupied L1
address, **not an allocation ID or Python owner**. Identifying the allocation
whose address is exactly 1,297,664 requires a live memory dump; it need not be a
router buffer itself. Router storage explains the large omitted reservation
that pushes that frontier downward.

A model-owned pool for these four router buffers preserves the decoder's
operations, placements, dtypes, fidelities, CCL policy, and semaphore ownership.
The parent applied this pool in `tt/model.py:52–65`; full-model validation is
pending. Its safety is supported by the output-copy lifetime analysis below. Sharing
only the two immutable input buffers is a narrower alternative capacity
control. Moving these directly tensor-backed gate CBs into DRAM is not an
equivalent repair.

This was the delegated fresh-context, source-only AutoFix diagnosis. No
hardware, TTNN imports, implementation edits, or builds were performed by the
diagnosis agent. No further agent/CLI was launched because the parent explicitly
prohibited delegation. Only this report and a stage work-log note were written.

## Evidence and exact SDPA allocation

- `readiness.log` finishes constructing layers 0–29 and `MODEL_READY`, then
  ordinary sliding SDPA throws: CB region end **1,315,840**; lowest occupied L1
  address **1,297,664**; deficit **18,176 bytes**. The stack ends at
  `tt/optimized_decoder.py:1402`, before full-model logits or sampler execution.
- Parent's fresh chat reference has 161 prompt tokens plus 100 continuation
  tokens, hence 261 logical prefill tokens. The model runner uses TP4 and
  `max_seq_len=8192`, with `TT_METAL_TRACE_ALLOC_TRACKING=1`.
- `all_layer_smoke.log` previously passed the 19-token all-layer case. This
  changes SDPA geometry and does not test the same CB footprint.
- `probe_l1_retry.log` passes a 261-token reduced layers `(0,5)` run with the
  common sampler and 28 additional `_MeshCCLManager` objects. It reports
  `SPLIT_TRACE_READY` and three model/sampler replays with allocation tracking.
  Therefore the sampler plus 30 semaphore sets **alone** does not reproduce
  the full-stack collision.
- `probe_l1_routers.log`'s first router-reservation attempt used `ttnn.clone` on
  `(10,9)`-sharded tensors and failed in clone's runtime-argument validation
  before prefill. That is an invalid experiment, not a refutation of the L1
  hypothesis. Use allocation-only `ttnn.empty` with matching specs.
- `probe_l1_routers_retry.log` uses the corrected exact-spec allocation-only
  reservation and reproduces the original SDPA program's 1,315,840-byte CB end
  after `MODEL_READY`, with a 1,301,504-byte occupied frontier. This is the
  positive causal control; the clone failure is separate.

`OptimizedDecoder.prefill_forward` rounds 261 rows to 288
(`tt/optimized_decoder.py:701–709`). The imported
`models/demos/gemma4/tt/attention/operations.py:222–261` selects sliding
head-dimension 256, Q256/K128, grid 8×8. The selected prefill compute retains
FP32 destination accumulation (`tt/optimized_decoder.py:1239–1247`), so SDPA
uses the legacy compute/mask path. It rounds the 288 rows to two Q chunks.
The causal pair schedule assigns two chunks to an active core and therefore
double-buffers Q even though the TP4 local query-head count is only four
(`sdpa_program_factory.cpp:453–474`).

The exact per-core CB inventory follows
`ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp:502–515,775–889`:

| Buffer group | Bytes |
| --- | ---: |
| BF16 Q, double buffered, 8 Q tiles × 8 head-width tiles | 262,144 |
| BF16 K and V, each double buffered, 4 K tiles × 8 width tiles | 262,144 |
| Legacy sliding mask, 64 BFP4 tiles × 576 bytes | 36,864 |
| Two BF16 scalar tiles | 4,096 |
| FP32 QK, 8×4 tiles | 131,072 |
| Two BF16 output intermediates plus final output | 393,216 |
| Three BF16 statistic vectors plus two FP32 sum vectors | 114,688 |
| **Total CB storage** | **1,204,224** |

The observed region end minus this inventory is **111,616 bytes**. This
calibrates the CB base/reserved region for this run; it is not an assumed
universal reservation. The sum reproduces the exception's 1,315,840-byte end
exactly. For 19 tokens, physical32/Q32/K32 with single-buffered Q yields only
154,752 bytes of CB storage, explaining why that smoke can pass.

`tt_metal/impl/program/program.cpp:1823–1829,1925–1932` compares this CB end
against `lowest_occupied_compute_l1_address`. The normal allocator has a common
L1 frontier; only the explicit HYBRID/per-core branch at 1908–1923 does a
CB-range-specific physical-bank check. Moving an ordinary sharded tensor to
worker `(10,9)` does not itself provide a separate allocator budget.

## Persistent allocation ownership

`tt/optimized_decoder.py:895–912` creates the router's four tensors in a
single-core `(32,32)` L1 shard. Bias and output are BF16; input and output
indices are uint16. All four therefore occupy 2,048 bytes per shard, regardless
of the bias/index tensors' logical 16×16 shape. `tt/multichip_decoder.py:880–893`
relocates them to `(10,9)` without reducing their size. All 30 model layers are
constructed before any prefill (`tt/model.py:51–58`). These buffers are
decode-only: `GeneralizedRouter.__call__` sends non-single-token inputs to the
existing prefill router at `optimized_decoder.py:917–919`.

Other allocations should remain distinguished:

- `_MeshCCLManager` creates 12 global semaphores per layer: six RS, four AG, two
  barriers (`models/demos/gpt_oss/tt/ccl.py:39–56`), covering the full worker
  grid. `GlobalSemaphoreImpl::setup_buffer` allocates one uint32 per shard/core
  (`tt_metal/impl/buffers/global_semaphore.cpp:83–100`). Alignment adds allocator
  cost. The parent has already included all 30 sets in the passing control.
- The common sampler's `TT_CCL` adds 36 global semaphores, while its persistent
  parameter, vocabulary-index, log-prob and penalty tensors are DRAM in the
  inspected constructor paths. The sampler is present in both failed and
  passing controls. It does not execute in the failing readiness prefill.
- `_ExpertParallelExperts.route_indices` keeps two additional L1 tensors per
  layer (`multichip_decoder.py:585–595`). A later exact full-footprint control
  can reserve these matching buffers too. Their much smaller per-bank
  interleaved footprint does not replace the missing 28×8 KiB router state.
- RMSNorm `tt_weight` is explicitly DRAM
  (`models/demos/gemma4/tt/rms_norm.py:25–35`); `_build_sharded_cfg` returns only
  input memory/program metadata. `precision_ops.norm_weight` transforms these
  existing DRAM weights and does not request persistent L1. Learned norm
  weights are layer-specific and must not be aliased across layers.
- Model embedding/head/norm/RoPE/zero tensors and all paged caches use the
  model's explicit DRAM upload helper (`tt/model.py:79–95`).

## Minimal discriminating controls

Keep the exact 261-token padded shape, real layers `(0,5)`, sampler, full
semaphore reservations, Q256/K128 sliding program, precisions, and allocations
alive across the call. Use fresh processes between controls.

1. **Reserve the missing router footprint — verified.** Add 28 copies of each router
   tensor using `ttnn.empty` with `shape`, `dtype`, `layout`, `device=mesh`, and
   `memory_config=router.memory` matching the actual tensor. Retain all 112
   tensors. Do not use `clone`: its distinct failure above prevents testing
   SDPA. Result: the same SDPA allocation collision appears with a CB end
   of 1,315,840 and frontier 1,301,504 in `probe_l1_routers_retry.log`.
   The lowest frontier can differ from the original full model
   because reservation order and the small omitted EP buffers differ.
2. **Remove only that reservation — passing baseline**, leaving sampler and all
   30 semaphore sets. `probe_l1_retry.log` is the parent's passing control.
3. **Full model, pool router buffers before first prefill — applied by parent,
   validation pending.** Share the four
   exact buffers across serial layers, keeping CCL semaphores and weights
   private. Storage savings: **29×8,192 = 237,568 bytes**. Compare the original
   standard readiness prefill, then decode accuracy and trace replay. Sharing
   only bias/indices saves **118,784 bytes**, already larger than the observed
   18,176-byte deficit; it is a useful alternative to isolate immutable reuse.
   Allocator fragmentation means these are capacity savings, not a guarantee
   that the minimum address moves upward by exactly the same amount.
4. **Q128/K128 diagnostic only if needed.** On the same shape and fixed policy,
   reducing Q alone produces three Q chunks; four local heads over 64 cores
   gives single-buffered Q. Calculated CB storage falls to 669,696 bytes,
   predicted end 781,312 with the observed base. This tests CB-pressure
   sensitivity but does not identify the persistent owner, and is unnecessary
   if pooling proves the issue while preserving the selected Q256/K128 policy.

For direct ownership evidence, dump L1 memory immediately after construction
and immediately before SDPA; list each retained router tensor's
`buffer_address()`, shape, dtype, layout, and memory config. The existing
`ttnn.dump_device_memory_state(mesh, prefix=...)` / `get_memory_view` wrappers
are in `ttnn/ttnn/device.py:207–222`. Capture those addresses in the same
process as the failure. `TT_METAL_TRACE_ALLOC_TRACKING=1` guards allocations
made while traces are active; it is not a general prefill allocation-owner
report and cannot identify this frontier by itself.

## Router-pool lifetime review

Sharing the two constants is safe by source inspection: each layer uploads the
same zero BF16 bias and transposed uint16 `arange(256)` indices. The gate's
single-block kernel reads both into registers and writes only the output CBs
(`generalized_moe_gate/device/unified_kernels/generalized_moe_gate.hpp:215–250`).
No per-layer learned value belongs to these two buffers.

Sharing output scratch is also consistent with the current serial model:

1. The gate returns `self.output` / `self.output_indices` aliases, but
   `GeneralizedRouter.__call__` immediately converts both from HEIGHT_SHARDED
   L1 to INTERLEAVED L1 (`optimized_decoder.py:969–970`). The changed memory
   config causes an actual copy; `ShardedToInterleavedDeviceOperation` creates
   a new tensor when no preallocated output is supplied
   (`.../sharded_to_interleaved_device_operation.cpp:125–133`).
2. The slices/views at 971–972 and `last_decode_indices` at 973–974 refer to
   that interleaved copy, **not** the pooled gate output. Retaining per-layer
   decode indices therefore does not retain an alias of shared scratch.
3. Indexed experts consume these copied indices for selected weights and both
   sparse matmuls before the next layer runs (`optimized_decoder.py:210–232`).
4. `Gemma4Model.decode_forward` invokes layers in order on the same command
   queue. The full layer's data dependency and intervening CCL separate router
   invocations. The pool must stay model-owned, serial, and allocated before
   decode warmup/capture. It must not be exposed to concurrent independent
   models/requests or separate command queues.

The full-model wrapper is an appropriate scope for the pool: it knows and owns
this serial scheduling contract. It can alias the existing routers immediately
after construction without changing the accepted decoder's standalone API.
Keep all four `router.memory` specifications equal, keep per-layer learned
weights/program/fidelity policy untouched, and retain a clear owner for the
pooled tensors through every trace lifetime. Do not explicitly deallocate a
tensor after another router has been made to share it.

Required verification remains the parent's original readiness check plus
full-stack decode accuracy, repeated/advancing trace replays with allocation
tracking, and any required watcher run. Passing a reduced model's generated
text is a capacity/trace control only, not full-model correctness evidence.
