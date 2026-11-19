# Bounded fixed-slot decode traces

CPU-only source audit; no TTNN import, device experiment, or implementation
change. This is a proposed repair and test plan, not passing device evidence.
Inspected the live `tt/model.py`, `tt/generator.py`,
`tt/multichip_decoder.py`, and the current common sampler and paged-cache/SDPA
kernel source. The source can continue changing while this document is read.

## Finding

`QwenGenerator.prepare(slots=...)` captures exactly one Python slot tuple and
stores `prepared_slots`. `decode_forward` rejects any different tuple. In
`QwenModel.decode`, Python indexing binds each slot's position, recurrent
state, and page-table tensors, and static token slices. `decode_batch`'s loop
count and batched MoE/projection geometry depend on that tuple's length.
Therefore changing slot count, order, or membership changes captured work;
merely removing the equality guard produces wrong request ownership.
Canonicalizing tuple order removes permutations but still leaves 2^32 active
sets. Capturing each set is not a bounded serving strategy.

Two other changes are necessary with an activity mask:

- `_refresh` currently constructs an all-zero 32-token vector and fills only
  selected slots. This erases parked requests' persistent sampled tokens.
- Sampling writes all 32 output tokens, and the sampling trace increments all
  32 seeds. Excluding a row from model decode alone does not freeze its token
  or RNG state. `configure_sampling` similarly replaces every row's seed and
  parameters, which cannot implement independent request admission.

## Smallest implementation

Keep physical request slots fixed for the generator's lifetime. Let M be the
configured `max_batch_size`, with 1 <= M <= 32. Allocate all M slot states and
all sampling/token/position/mask buffers before warmup. Use only these model
execution signatures:

| Signature | Static decoder rows | Selection |
|---|---|---|
| `solo0` | slot 0 only | only slot 0 active |
| `fixed_M` | canonical slots 0..M-1 | every other nonempty active set |

For M=1 these are the same signature, captured once. There is one common
sampling trace over the fixed 32 sampler rows. Thus there are at most three
traces for a configured model/cache, independent of request identities, row
orders, activity changes, prompt lengths, and page mappings. Preserve the
existing optimized batch-1 path in `solo0`; do not add recurrent masking or
state copies to that model path. `fixed_M` still uses the selected
`decode_batch` batched attention/collectives/MoE path and its residual, dtype,
fidelity and packing policies. It does not become M serial full-model calls.

A singleton in physical slot 7 uses `fixed_M`; it must never be silently
renumbered to slot 0, because its cache, RNG and token feedback belong to slot
7. The primary standalone single-user path continues to use slot 0. If the
measured sparse-batch cost warrants more prepared shapes, add fixed **prefix**
buckets 1/2/4/8/16/32 and choose the smallest covering the largest active slot.
That is an optional bounded optimization (at most six model signatures), not
a need to enumerate active sets. Measure trace-region capacity before adding
those variants. Do not implement a trace eviction fallback.

The model signature includes configured M, stable cache object/resource
bindings, physical tensors and model policy. A scheduler slot tuple is runtime
data and output ordering only. Reject a genuinely different cache allocation
unless it was registered and prepared before capture; changing page-table
**values** within the same allocation is supported without recapture.

## Persistent activity and position state

Allocate a stable device mask `active[32]` and stable per-slot absolute
positions. Update activity only on scheduler changes, comparing against the
last host mask; unchanged steps do not copy it again. Reordered API rows with
the same active set do not even require an activity update. Canonical model
and sampler row s always correspond to physical slot s; read returned rows in
the caller's requested order.

Keep the parked position as its real value; do not replace persistent position
state with -1. For each fixed-row full-attention call derive on device:

```
rope_position[s] = active[s] ? position[s] : 0
attention_position[s] = active[s] ? position[s] : -1
position[s] <- position[s] + active[s]  # in model trace, after attention
```

Use `rope_position` for the embedding lookup and `attention_position` for both
paged cache update and SDPA. Casting -1 to uint32 before RoPE embedding would
index outside the table. Active positions must remain below context capacity.

The current native source explicitly supports the -1 sentinel:

- `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp`:
  `raw_update_idx == (uint32_t)-1` skips cache writes.
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp`,
  `writer_decode_all.cpp`, and `compute/sdpa_flash_decode.cpp` return for
  `cur_pos == UINT32_MAX`.

An SDPA early return may leave output storage untouched. Explicitly select
zero for inactive attention/hidden rows before a batched consumer; do not rely
on uninitialized or stale inactive results being finite. Keep valid allocated
page-table buffers for every physical slot, even when inactive. Do not point
an inactive writable row at an active request's pages as a masking shortcut.

## Recurrent-state preservation

A zero token or a zero hidden vector does **not** make a linear-attention step
an identity. In `LinearTP4Mixin._linear_attention` the convolution kernel
updates `state['conv']` in place, and `_recurrent_step` writes
`state['recurrent']` via `output_state`. Both commits require the same runtime
activity predicate in `fixed_M`.

The smallest targeted repair is an optional activity argument in the batch
path, keeping the existing unmasked fast path unchanged:

1. For `TP4DecodeConvolution`, read a stable per-slot device activity scalar
   and predicate only its three final history writes. The current
   `DECODE_CONV_READER` fetches all old history before those writes; retain its
   BF16 bit transport, convolution arithmetic, and producer/consumer counts.
   Do not return early from only the reader while compute/writer kernels wait
   on circular buffers. Gate the history writes and let the operation finish.
2. Compute the recurrent candidate with `output_state=None`, retaining its
   current FP32 arithmetic and round-state policy. Commit the candidate into
   the persistent state only for an active row. A device conditional copy
   must preserve inactive FP32 bytes exactly; do not achieve an identity by
   setting decay=1/beta=0, because `round_state=True` can still change an
   inactive FP32 state produced by prefill.
3. Implement the commit with a supported exact-dtype select/output operation
   after verifying its alias/layout contract. If native select cannot cover
   FP32 TILE or UINT32 ROW_MAJOR safely, a small model-local predicated byte
   copy built with the existing `generic_op` pattern is a concrete fallback
   for this **state commit operation**, not a model/host fallback: read the
   device scalar, copy the candidate's bytes only when active, otherwise
   leave the destination untouched. Never let a runtime address come from a
   host per-token decision.
4. Do not snapshot every layer's full recurrent cache and restore it after
   forward. That adds unnecessary large copies and persistent storage. The
   candidate state is already a normal operation output; only its commit is
   conditional. Keep candidate and transient aliases alive correctly through
   their consumers inside capture.

This adds no new precision policy or replicated residual boundary. Masked
commit cost belongs in the `fixed_M` performance evidence. Kernel changes are
unverified until the reduced real-state tests below pass.

## Token feedback, RNG and scheduler updates

Keep the common sampler's canonical call:

```
sampler._run_sampling(sampler_logits, ..., tt_out_tok=persistent_tokens)
```

Inside the sampling trace, preserve the old 32-token vector in a preallocated
small device scratch, run the sampler directly into `persistent_tokens`, then
restore only inactive token entries from that scratch. These are device
copies/selection over 128 bytes per chip; no host token loop is introduced.
Advance per-slot seeds by `active`, not by unconditional `plus_one`. The common
sampler calls `manual_seed(seeds=self.seeds_tt_tensor)` before sampling, so
preserved seed values ensure a resumed inactive row does not consume additional
per-request draws. Validate this with seeded controls; do not assume global
RNG behavior from intent. Preserve penalty history too when enabled; current
`penalties_on=False` does not yet provide that contract.

For scheduler-supplied tokens/positions, write a persistent staging buffer and
merge only explicitly refreshed slots on device. Do not build a host token
shadow from previous outputs, and do not zero the whole token vector. A
request admission/resume may refresh its slots once. Fixed-step continuation
uses `refresh_inputs=False`; it submits the model trace and sampling trace
without reconstructing inputs. Parameter/seed updates must likewise target
selected slots, leaving other requests' on-device RNG progression intact.

`slots` controls which logical rows are returned and which activity mask is
selected, not where a sampled token is copied. Python may enqueue two traces
per step and optionally read output required by the caller; it never supplies
the next sampled token back to the model. Reject duplicate/out-of-range slots
and mismatched per-row lengths before mutating device state. An empty active
set may return without replay and must leave every state unchanged.

## Preparation and trace lifetime

Warm both model signatures and every relevant sampling/commit/mask/position
operation before capturing either model trace. Allocate active masks, scratch,
position representations, sampler outputs and CCL resources first. Restore
warmup-mutated KV, recurrent, token, position and RNG state before capture/use.
Bind both signatures to the same stable slot-0/cache and sampler input buffers.

Store traces by execution signature, not `prepared_slots`. Capture each once;
`solo0 -> fixed_M -> solo0` must reuse the original IDs. Exact-tensor allocation
acknowledgments require a lifetime proof against **all** live traces: each
model replay completely rewrites the sampler logits it advertises, sampling
consumes them before the next model replay, and feedback tokens/masks/positions
survive every alternate trace. No broad capture/forward exemption is valid.
Lazy `_plan` prefill preparation remains a separate source of late allocation:
all supported eager prefill variants must also be prepared before the first
capture. Fixing batch keys alone does not establish that larger warmup gate.

## Focused acceptance tests

Use the reduced full-model probe with one real layer of each kind and the real
terminal sampler first. All following sequences use one generator, stable
cache allocations, and existing traces; do not reset counters between cases.

1. With M=32 prepare `solo0` and `fixed_M`. Run active `(0,)`, then `(3,0)`,
   `(0,3)`, `(31,)`, `(31,7,0)`, back to `(0,)`. Assert <=2 model captures and
   one sampling capture, identical IDs on return, and zero releases until
   teardown. Canonical model/cache row ownership is unchanged by API order.
2. Prefill distinct real prompts into slots 0, 3, 7 and 31, with disjoint
   nonidentity page mappings and different positions. Compare active output
   logits/tokens and per-slot KV/recurrent states against the same slots run
   alone. Include all 32 active once to cover the selected batched decoder
   geometry. Do not require bit equality between fundamentally different
   batch geometries unless the model's measured arithmetic supports it; report
   numerical differences and token agreement. Reordered identical geometry
   must be bit-identical per slot.
3. Fill an inactive slot's conv/recurrent state with recorded real nonzero
   values, including an FP32 recurrent value not exactly representable in
   BF16. Pause it while other slots run several steps. Assert bit-exact
   preservation of conv, recurrent, KV, absolute position, token and seed;
   resume it and compare to a no-pause control. Also test inactive slots after
   prefill, where accidental state rounding is most easily exposed.
4. Record exact persistent tokens/positions around two or more replay pairs;
   token N is the decode input for N+1 and active position increments once.
   Observe those values for the test only. Production feedback has zero host
   token/position/RoPE refreshes after admission. Inactive counters stay zero.
5. Same active set but reordered rows performs zero page-table refreshes when
   mappings are unchanged. Then update one slot's mapping (with equivalent
   populated KV pages) and verify only its stable table values refresh, no
   capture occurs, and outputs remain correct. Re-test the unchanged case.
6. Seeded active-row sampling matches a run where inactive slots were absent
   from scheduling; greedy -> sampled -> greedy retains trace IDs and greedy
   semantics. Changing one request's sampling seed must not reset another's.
7. Run the sequence with `TT_METAL_TRACE_ALLOC_TRACKING=1` and
   `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, positive control separately,
   ending on a decode replay. Include supported eager prefill between model
   traces. No live unsafe allocation or new program signature is accepted.
8. With tracking disabled, measure slot-0 batch-1 128/128 before/after and
   `fixed_M` active counts 2/8/16/32; measure the masked recurrent commit and
   token mask overhead in the reduced profiler. Preserve the selected TP4
   decoder policy and report physical versus active rows. Unmeasured extra
   prefix buckets are optimization candidates, not claimed results.

No hardware claims are made by this design audit.

## Model-side implementation checkpoint

Implemented optional `QwenModel.decode(..., activity=None)`; `activity` is a
list of stable INT32/UINT32 ROW_MAJOR DRAM scalars indexed by physical slot.
`None` retains the unmasked path. `slot_state.py` provides exact conditional
DRAM copies, safe runtime position inputs and conditional position increment.
Native ternary validation requires TILE operands/output, so these helpers use
small byte-preserving dataflow programs instead of converting FP32 recurrent
state or ROW_MAJOR request state for a select operation.

The selected decoder's convolution has an optional default-off predicate only
around its three history writes. The recurrent candidate is committed only
when active; no previous-state snapshot is allocated. Full-attention output
is zeroed before arithmetic on skipped SDPA rows. Masked residual rows get
independent output storage, because sliced decoder rows may alias tile backing
storage. Constructor sharing/removal work owned by the root agent is retained.

CPU validation command:

```bash
python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/slot_state_static_check.py
```

This passed exact byte/page partition tests for FP32 `[1,8,128,128]`, BF16
ROW_MAJOR `[1,3,2048]`, BF16 TILE `[1,32,2048]`, and UINT32 token rows. It also
checks that the optional convolution source guards exactly its three commits
without introducing an early return. Python syntax compilation passed for
model, decoder and helper. These checks substitute a descriptor stub and do
not import TTNN. They do **not** validate device compilation or operation.

Next device check before integrating fixed-slot traces: allocate activity 0
and 1, position 123, distinct non-BF16-representable FP32 recurrent values and
BF16 conv values. For each activity, verify conditional copy bit patterns;
verify `position_inputs` gives `(0,-1)` for inactive and `(123,123)` for active;
verify advance gives 123/124. Check in-place zeroing of a BF16 TILE buffer.
This specifically tests the four-byte scalar DMA/alignment and accessor
bindings in the new generic programs. Then run the reduced real two-layer
masked batch, checking inactive conv/recurrent/KV bytes and active output
against unmasked controls, followed by the traced activity-switch sequence
above. No device check was run by this implementation subtask.
