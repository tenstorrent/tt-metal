# Stage 10 serving topology audit

CPU-only independent source/artifact inspection, 2026-11-17, starting checkout
`5e2dca2e09485be6c4f18c688a1b7b532e45a99b`. No TTNN import, hardware job,
profiler, server request or implementation edit was made by this auditor. This
is a candidate audit, not a stage-review verdict or performance acceptance.
All paths below are relative to the model root unless prefixed otherwise.

## Baseline and comparison boundary

`readiness_vllm/final64_perf/vllm_benchmark.json` records real vLLM greedy
128-input/128-output/one-request serving: TTFT 3006.653 ms, mean TPOT 53.79794 ms,
median ITL 53.77115 ms, P99 ITL 54.66383 ms, output throughput 13.00891 token/s,
and TPOT-derived **18.58807 t/s/u**. Configured capacity is 32, context 262144,
mesh 1x4 Blackhole; solo serving executes the prepared B2 model trace.
These are stage-9 historical measurements, not this stage's fresh baseline.

The separate CI 100-input/100-output/32-request burst completes all 32 requests:
median/P99 TTFT 57696.973/88723.513 ms, mean TPOT 655.88026 ms, median/P99 ITL
452.25248/899.12939 ms, output throughput 23.95398 token/s. Burst admission and
serial prefills affect those intervals; its reciprocal TPOT is not headline
single-user throughput.

`readiness_vllm/serving_overhead/comparison.json` provides the repaired native
capacity32/B2 control: 53.80649 ms native serial versus 53.93798 ms adapter
serial and 53.75049 ms queued async2, with all 18 sequences exact. It used the
older canonical32 sampler; final64 HTTP measurements in stage9 are comparable
only to their specifically recorded matched native control. This demonstrates
no material wrapper gap in that regime. Stage7's 21.41687 ms B1 token-out row
is a different singleton expert execution family and predates a fused-reader
completion repair; it is not an appropriate subtraction from current B2.

## Policy actually selected by construction

`tt/generator_vllm.py::initialize_vllm_model` constructs `QwenModel` without a
precision override. `tt/model.py` calls `resolve_precision(None)`, whose
`SELECTED_PATH` is `doc/datatype_sweep/selected_precision_config.json`.
That file selects `baseline_bfp8`, with no layer exceptions: BFP8 expert GU,
down and grouped EP weights, LoFi (FP32 accumulation for indexed experts,
BF16 destination for grouped EP); BFP8/HiFi2/FP32 accumulation for attention,
shared projections and LM head. Activations, residual, CCL, KV, logits and
sampling boundaries are BF16; linear attention has BF16 weights and ordered
FP32 arithmetic/state, BF16 convolution state. Serving does not silently
reselect a faster datatype candidate. Runtime propagation evidence remains in
the datatype and stage9 artifacts; source inspection alone is not a new
runtime precision measurement.

## Operation topology before tuning

| Boundary | Actual source path and repeated work | Constraint and action |
| --- | --- | --- |
| Scheduler to generator | Adapter delegates to `decode_forward_v1`; `serving.decode_with_commands` validates reload commands and reserves an immutable output lease. Resident steps ignore stale host token/position payloads. | Keep explicit reload contract. No plugin changes needed for model-side candidates. |
| Token/position/page state | Persistent device tokens, per-slot positions/activity, cache and sampler tensors; `_refresh` only on authoritative input reload; `_bind_pages` compares CPU shadows and skips unchanged tables. `advance_position` runs in model trace. | Already removes per-token host reconstruction. Page-only updates must preserve token/position state and coexist with pending output leases. |
| Embedding and initial state | Each physical decode row gathers its hidden-sharded embedding to replicated H2048, builds device RoPE inputs, then masks inactive rows. | B2 still does work for its inactive second row. B1 substitution is not exact by construction. |
| Thirty linear-attention layers | Each physical row takes ordered norm/projection, convolution and recurrent-state path; row partials concatenate, direct AR restores replicated residual; grouped MoE follows. | Attention cost scales with physical bucket. Preserve conditional state commits and ordered arithmetic. Intermediate buckets are the strongest bounded candidate. |
| Ten full-attention layers | Batched norm/packed QKVG projection; per-row paged cache/SDPA; concatenate, output projection, reduction and residual. | Keep page32 BF16 KV and paired head ownership. No attention-math replacement is proposed. |
| MoE in serving B>=2 | `_moe` pads rows to multiple32; ordered router/top8 scores, local EP dispatch, fused grouped SwiGLU, combine/reduce, packed shared expert branch. Decode sets `stochastic_rounding=False`. | Active experts remain routed. This is not the indexed singleton `sparse_matmul` path. Changing bucket does not automatically prove exact row outputs: require controls. |
| Residual/collective boundary | Replicated BF16 residual across layers; direct AR with shared preallocated scratch/semaphores. Batch outputs are converted once to row-major before row extraction. Model applies per-row inactive masking after every layer. | No wrapper all-gather between layers. Preserve shared workspace serialization. Carrying a new sharded residual requires next-consumer adaptation, not a local knob. |
| Terminal | FP32 RMSNorm then BF16; vocabulary-parallel BFP8 DRAM-sharded LM head (readers3, block4, cores16, per-core N32); local logits copied into persistent sampler-ready buffer. | No full-vocabulary all-gather on measured device path. Existing geometry is explicit, not default dispatch. |
| Sampling | Two retained common-sampler traces, native32/native64 draw, shared physical top64 candidate pipeline, greedy k1/p0/temp1; `tt_out_tok` writes persistent tokens; active-row merge and seed advancement preserve dormant owners. | No force-argmax or host greedy fallback. Wide trace selection includes dormant owners; do not select solely on active rows. |
| Async return | Model and sampler replay `blocking=False`; output copied into one of two preallocated leases. Deferred `.cpu(blocking=False)` records an event; plugin waits before host formatting. | Snapshot copy protects pending results and cannot simply be deleted. Token processing reads first replicated device's already-transferred output; host-logit concatenation is explicit compatibility only. |
| Prefill | `BoundedPrefill` serial per request: embed64; linear attention chunks<=64; full attention aligned chunks32..256 or singleton tail; MoE chunks<=2048; only final valid row gets terminal head. Three capacity-sized row-major DRAM residual workspaces. | Per attention/MoE boundary `_read` allocates/loads/tilizes and `_write` converts/untilizes/transports. This is real material work; no new profiler is needed to identify topology. Preserve exact grouping, nonaligned tails and stable trace lifetime. |

The highest-latency prefill tail is visible directly in `attention_chunks`:
when `absolute % 256 != 0`, full attention advances singleton rows until the
next 256 boundary, even if many tokens remain. Changing that rule is not a
mere scheduling optimization: cache-fill/SDPA and numerical contracts require
fresh boundary/continuation controls. Three-second 128-token TTFT is not
explained by adapter argument conversion alone.

## Ranked bounded candidates

1. **Try B4 alone between existing B2/B32 before considering B8/B16.**
   Current `tt_supported_decode_batch_sizes` exposes only `{2, capacity}` and
   `model_trace_key` selects by highest active physical slot, not number of
   live requests. Any slot2 owner therefore pays B32 attention today. Add
   intermediate prefixes in the adapter declaration; generator preparation,
   capture and selection already iterate this tuple. Keep B2/B32 and both
   sampler traces. Expected benefit is mixed/burst capacity at partial
   occupancy, not a primary single-user improvement; size is unmeasured.
   Require exact active logits/tokens/state against B32 at extents3/4/5/8/9/16,
   inactive holes and remaps, plus automatic allocator growth, queued async
   outputs, unchanged/changed tables, all sampler modes, tracking1/skip0 and
   mixed chunked traffic. Additional traces must fit the same trace reservation
   and retain all prior traces. Sparse high slots still correctly select B32;
   do not compact ownership implicitly. Source `_moe` pads these shapes to32
   and disables grouped decode stochastic rounding, making this plausible,
   but shape-specific attention projection arithmetic still needs proof.
   Preserve the current 805306368-byte trace reservation. The recorded
   stage6 B32 model trace is467361792bytes
   (`doc/full_model/serving_audit.md`, underlying
   `logs/full_batch32_final.log:3761`); current final64 logs record capture IDs
   and times but not per-trace bytes. With the explicitly approximate affine
   command-size model T(B)=F+B*V, F>=0, the old B32 size implies B2+B4+B32
   needs554992128+1.8125Fbytes before the two sampling traces. That leaves
   250314240bytes for fixed overhead and sampling; B4 retention is plausible,
   but this is not a measured capacity proof. By contrast, adding all4/8/16
   predicts at least905513472bytes even with F=0 and no samplers, exceeding
   the unchanged reservation. Thus do not prepare every intermediate bucket
   blindly. B4 targets active slot extents3/4 and leaves primary B2 and full
   B32 traffic unchanged. Obtain actual current trace allocation sizes or
   bounded capture evidence before accepting retention; this historical
   affine estimate neither authorizes a larger TT config nor proves legality.
2. **Eliminate redundant host token clone after advanced indexing/conversion.**
   `DecodeOutputPool.process` indexes `host[list(handle.rows)]`, converts to
   int64, reshapes, then clones. Advanced indexing already owns new storage;
   the int64 conversion from uint32 owns another. Removing only the terminal
   clone can preserve writable-result ownership with a focused alias check.
   Expected gain is tiny and likely below noise; benchmark rather than claim.
   This is lower risk than deleting the required device snapshot or async
   event. It does not justify changing route logging or OOV validation.
3. **Cache immutable prefill cache geometry validation.**
   `BoundedPrefill.prefill` reconstructs shapes/layout/memory-config tuples for
   every layer on each invocation, though serving forbids cache rebinding.
   A generation-owned identity/version check could avoid repeated invariant
   traversal while retaining initial full validation. Expected gain is small;
   it cannot plausibly remove most 128-token TTFT. Reject if it weakens detection
   of an externally changed cache or needs a new plugin contract.
4. **Short-prefill tiled transport family, if remaining wall budget permits.**
   Keep attention chunk arithmetic and MoE row grouping, but retain temporary
   tiled attention outputs until the same MoE consumer instead of writing and
   reading them through row-major DRAM. This potentially removes tilize,
   untilize and row copies. It is larger work: bounded live temporary storage,
   prewarmed signatures, exact padding and cleanup before replay must be
   proven for tails and resumed chunks, not only128. Do not introduce an
   aligned-only public path or request-dependent persistent allocations. No
   speedup is established by this audit.

Optional batched prefill is a separate candidate: the current executor is an
explicit serial workspace and prefill grouped MoE uses stochastic rounding.
Packing requests changes grouping and can change results. The required
bit-equal last-logit/KV comparison is a real gate, not satisfied by PCC or by
existing serial burst completion. Record the serial burst ratio even if this
candidate is not adopted.

## Existing investigated items and exact scope of blockers

| Optimize item | Existing evidence and current disposition |
| --- | --- |
| Async, persistent inputs, no host sampling | Source above plus stage9 `final64_gate/lifecycle_summary.json` and final route counts. Already enabled; preserve and repeat affected checks. No justified large adapter-gap hypothesis. |
| Single-row bucket | `multichip_decoder.decode_batch` changes to singleton indexed MoE at B1; B>=2 uses grouped EP, distinct rounding/projection paths. `doc/full_model/AUTODEBUG_MIXED.md` documents the mismatch. B2/B32 controls do not authorize B1 substitution. |
| Hidden-sharded residual/MM-RS family | Stage5 `batch_sharded_l0/l3`, `sharded_bfp8_l0`, and `autofix_fused_rs_l3` retain true sharded residual through next norm. Measured slower, not rejected only after restoring old replication. Combined AG-MM family failed exact paired K-cache cells; canonical-input fused output control passed but lost at5.892128ms layer3 B32. These are scoped historical exclusions, not universal impossibility. |
| Reduced KV/CCL or attention fidelity | Stage8 selected config remains BF16 KV/CCL and HiFi2 ordinary projections. Earlier real layer BFP8 KV passes but shows no material decode gain and slower prefill; canonical full-model policy was selected after accuracy/P1. Changing KV invalidates context acceptance and is not a cheap serving tweak. |
| LM-head geometry / Blackhole readers | `tt/lm_head.py` wires explicit readers3/block4/cores16/N32. Stage7 `candidate_search.csv` and terminal controls measured this role. Other geometries are untried handoff candidates, not proved impossible. Current B2 time does not establish LM head dominance. |
| Persistent sampler CCL | Stage7 `sampler_persistent_broadcast_rejection.md` includes raw poisoned gather: remote96/128 columns unwritten, 3072/4096 elements wrong/rank. Native broadcast scatter fails to flush one-page tail with4096B packet/2048B tile. Global2048B payload workaround conflicts with selected FP32 all-reduce tile size. Width padding/native repair remain untried, not accepted rejection of all persistent approaches. Canonical64 geometry differs from that top32 repro and would need its own control. |
| Packed QKV/shared GU | Existing packed attention and shared projections are selected. Replacing with split adds launches/changes numerics and needs measurement; no current serving evidence justifies doing so first. |
| Sparse expert path / logical padding | Source preserves routed active experts; grouped EP is selected for B>=2. Padding stays internal to MoE. Inactive state commit/mask is mandatory; dropping masks is unsafe. |
| Profiling / roofline | No live-serving profiling. Earlier reduced non-serving stage7 artifacts provide context, not a fresh B2 decomposition or serving roofline. Current benchmark JSON and contract checks are the required evidence. |
| Prefill chunks/tails | Existing attention planner preserves arbitrary lengths via scalar tails; historical stage7 boundary TTFT64/255/256/257/1025/2049 shows no robust improvement. No exact blocker proves a better chunk planner impossible; it remains untried substantial work. |

Apply the optimize Stop Rule to fresh measurements. An untried candidate above
must be recorded as such, never relabeled a measured rejection. The owner must
produce same-server-config before/after single-user and CI rows, fresh final
sampling/qualitative and eval-load gates, and mixed trace/allocation evidence
for bucket changes before any stage acceptance claim.
