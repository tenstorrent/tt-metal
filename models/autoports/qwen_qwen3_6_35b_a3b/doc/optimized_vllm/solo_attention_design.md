# Retained solo-slot0 attention specialization

Source-only design, 2026-11-17. No implementation changes for this hypothesis,
device jobs, server requests or measured claims. This design assumes the
separate inactive-route candidate first passes exactness; it does not assume
that candidate improves latency or reaches the functional floor.

## Concrete proposal

Add opt-in `QwenModel(..., specialize_solo_slot0_attention=False)`, requiring
`suppress_inactive_moe_routes=True` when enabled. The generator prepares one
additional retained model trace `model_solo0`, with physical rows2 and only
slot0's independent attention work. Keep ordinary model_2/model_4/model_fixed
and sampling/sampling_64 traces. Select the new trace only when the generator's
authoritative active-slot set is exactly `(0,)`; every other set selects the
existing bucket path. Capacity, external decode buckets and plugin interfaces
stay unchanged. The plugin still submits its ordinary B2 carrier for solo.

This is **not** physical B1 decode: it must retain the same B2 full-attention
input projection, output projection, reduction, residual, post-attention norm,
router, shared expert and terminal shapes. Preserve grouped MoE with runtime
activity masking. Never call the singleton `decode_forward` or its indexed
expert path for the active row as a substitute.

## Minimal code points

| Location | Exact change and preserved boundary |
| --- | --- |
| `QwenModel.__init__` | Validate the flag/dependency and prepare two shared immutable zero partial tensors before any trace: linear partial FP32 logical[1,1,2048]/padded[1,32,2048], full-head partial BF16 logical[1,1,1024]/padded[1,32,1024], both DRAM TILE. They can be reused read-only across all layers. |
| `QwenModel.decode` | Add an internal explicit `solo_slot0_attention=False` graph selector. Require exactly physical slots(0,1), supplied persistent activity/mask and supported selected layer topology. Keep embedding, position/RoPE preparation, per-layer inactive masks and position advancement unchanged initially. Pass the graph selector/zero partials to layers. |
| `MultichipDecoder.decode_batch` full-attention branch | Preserve concatenate2→normalized_input2→packed projection2. Run `_full_attention(... projected=projected[:,0:1,:], defer_output=True)` only for row0. Supply the immutable BF16 zero head partial for row1. Concatenate the same two-row shape and retain `_full_output(..., batch_decode=True)` plus `_reduce(..., batched=True)` unchanged. |
| `decode_batch` linear branch | Retain row0's existing singleton input norm and exact `_linear_attention(...decode=True,activity=positions[0]['activity'])`. To minimize scope, retain both input-normalization dispatches initially, but skip row1's complete independent `_linear_attention`, supplying the FP32 zero partial. Concatenate2 and keep the same FP32/BF16 direct-reduction boundary, residual and downstream grouped path. |
| `QwenGenerator._decode_into_sampler` | Add a static internal graph-selector keyword forwarded to model decode. Keep both physical logits rows, 32-row sampler carrier and persistent output copy unchanged. |
| `QwenGenerator.prepare` | Warm the specialized graph, its exact constants/layouts and any zero/concat variants in phase1. Only then capture `model_solo0` once in phase2 with all other prepared model/sampler traces. |
| `QwenGenerator.model_trace_key` | If the optional trace was prepared, physical B2 remains in the current selection set, and authoritative active slots equal(0,), select `model_solo0`; otherwise use existing extent selection. This lets a discriminator temporarily force current selection to(32,) without accidentally selecting solo0. |
| `_record_capture` | Record distinct graph key, physical_rows2, attention_slots[0], prepared constants and stable buffer bindings. Sampling signatures remain independent. |

The selected code has `batch_attention_reduce=True`,
`batch_full_projections=True`, replicated residual and canonical separate full
input/output projection. The new path should reject unsupported constructor
topologies before preparation, rather than silently omit work under an
unreviewed fallback branch. Keep the current method default behavior intact.

Full-attention zero shape/dtype comes from `_full_attention`'s existing
`nlp_concat_heads_decode`/reshape result. Linear zero must be FP32 because
ordered output scanning returns FP32 partials before direct all-reduce; using
BF16 zeros can change concatenation/reduction selection or introduce a cast.
Derive and assert actual metadata in the reduced control before finalizing
constants; source reasoning is not evidence the device accepts every layout.

## Safety and exactness reasoning

The new graph is a prepared execution signature, not a Python decision made
inside replay. Its omitted slot1 work is safe only when slot1 is inactive.
Select it from the generator's command-owned active set after remap/reload
validation, never from a possibly stale `start_pos` payload or client batch
count. Resident steps retain that authoritative active set and device mask;
page-only refresh cannot change it. Existing reload/remap operations drain
pending outputs before mutating request layout. A return to `(0,)` reuses the
same trace ID; owner1-only, owner17-only and two-live-row sets must never use it.

The remaining row0 state and cache operations still read persistent device
activity and positions. Row1's recurrent/convolution tensors are untouched
because no call reaches them; row1's paged-update/SDPA work is omitted. Keep
existing conditional position advancement, so dormant position1 is unchanged.
Do not reset dormant state to zero to make the specialization convenient.

The ordinary inactive linear-attention row can produce nonzero intermediate
output from dormant recurrence even though its commit is suppressed. Replacing
that intermediate by zero changes row1's residual/norm/router/shared values.
This is acceptable only if active results remain exact: runtime route masking
must exclude every inactive row before grouped counts/dispatch. B2 dense
matmul/norm and collectives are rowwise in the intended contract, but changing
row1 data can still expose tile packing, BFP8 scale sharing, or an unexpected
cross-row kernel dependency. Treat exact active-logit comparison as mandatory.
Do not conclude equivalence merely because row1 is masked at the layer output.

Full attention already zeros an inactive SDPA result before gating/concat;
the zero placeholder is therefore particularly direct there. Nonetheless,
preserve packed projection2 and batched output2, since switching either to
singleton can change the active row's numerical family. Do not replace them
with the historical B1 path.

The immutable zero tensors must be initialized before the first live trace,
kept alive by the model, and never reused as output buffers. They can be read
by every trace; no acknowledgment/exemption should be necessary. Their payload
cost is327680bytes/chip (262144 FP32 plus65536 BF16) before allocator alignment.
Update the context-memory accounting with actual allocations. No cache dtype,
layout, max context or physical pool changes are proposed.

## Preparation, replay and trace budget

Warm all ordinary buckets, then solo0 and both sampler variants before any
capture. Set activity to(0,), initialize token/position0 for the warm graph,
restore every touched request/KV/RNG input using existing setup reset, and
capture from that prepared state. Reset alone is setup, not a request-time
trace release. The solo graph can be captured alongside the others in any
validated order; no late warming after an ordinary trace is live.

The new trace consumes the same persistent token, position, page, activity,
cache and sampler buffers. It produces both physical logits rows and retains
the separate canonical sampling trace. `serving.decode_with_commands` already
submits the selected model key nonblocking, so it needs no plugin or output-
lease changes. Existing immutable lease copies and deferred event boundary
remain mandatory.

One additional trace is required; no previous trace can be dropped to fit.
With B2/B4/B32 plus two samplers the total becomes six traces. The new command
buffer should be smaller than ordinary B2 because it omits30 independent
linear-attention paths and10 full row1 cache/SDPA/head paths, but its actual
allocation is unknown. Existing 768MiB reservation is a fixed benchmark config:
measure remaining trace allocation and bounded capture. If it does not fit,
record the exact bytes/failure and optimize trace/resource storage; do not
silently grow the TT config or evict B4/B32. Historical467MB B32 size is not
sufficient to prove current six-trace capacity.

## Reduced discriminator before all40 serving

Use real layers0/3, capacity32, prepared buckets2/4/32, tracking1/skip0 and the
same persistent inputs/output leases. Compare route-masked ordinary B2/B32
against route-masked solo0; also retain the already-passing unmasked-route
reference so two candidate changes are not credited without a chain of exact
controls. Start from identical prefilled state, then:

1. Teacher-force changed tokens/positions on slot0; compare full active logits,
   all-rank recurrent/conv, active required KV pages and all dormant state.
   Use nonzero prefills for slot1 and high-slot owners, not zero-only dormant
   fixtures. Confirm slot1 token/position/seed/cache bytes remain identical.
2. Replay several resident steps without token/position/page uploads; compare
   exact greedy and seeded native32/native64 trajectories and two pending
   output snapshots. Keep dormant-wide sampling-mode selection covered.
3. Transition `(0,)→(0,1)→(0,)→(1,)→(0,1,3)→(0,17)→(0,)`, plus a real ownership
   permutation with drained outputs. Confirm ordinary B2/B4/B32 keys appear
   where required and solo0 returns without recapture. Include changed required
   page mappings and unchanged tables while retaining authoritative positions.
4. Check inactive slot0/empty-active dispatch never wrongly samples solo0;
   direct internal graph misuse with missing mask or wrong physical slots
   should reject before device mutation.
5. Observe exact constants, output geometry, program count stability, six
   captures/no releases until teardown, all-trace lifetime and tracker positive
   control. End the workload with decode replay. Run Watcher separately if
   needed; no serving profiler.

After this passes, use all40 exact controls and actual plugin mixed traffic,
including concurrent chunked-prefill arrivals, to prove selection and overlap.
Record clean same-harness primary and CI metrics with the original config;
the candidate's expected savings are an unmeasured hypothesis. Meeting or
missing21.63129875ms remains an independent measured gate.
