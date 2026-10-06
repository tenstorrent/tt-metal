# Precision boundaries in dedicated fusion candidates

Source investigation, 2026-09-25. No hardware commands or implementation changes
were made by this investigator. The associated diagnostic is
`tests/probe_fused_precision.py`; its device experiments remain unrun here.

## Finding

Dedicated RMSNorm, RoPE and decode SDPA deserve real-activation experiments with
their strongest supported precision settings. FP32 tensor storage alone does
not establish equivalence to the functional decoder's SFPU graph. Existing
controls prove that attention PCC above 0.9998 can still change top-8 expert
selection and reduce final-layer PCC below 0.995. Keep both same-input component
comparisons and integrated real-checkpoint checks. A first validation error or
one synthetic case is insufficient grounds to reject a candidate.

The accepted context remains 262144. Functional-stage `README.md` records
passing sliding/full maximum-context and nonaligned-262143 checks. No candidate
may shorten the cache, replace caller page ownership, reduce precision of
physical addresses, or introduce a full-attention identity gather.

## Graph and contracts

| Boundary | Current graph and movement | Dedicated candidate |
| --- | --- | --- |
| Input, Q/K/V, post-attention and router norms | Cast FP32; square; mean; add epsilon; exact rsqrt; multiply input; optional multiply FP32 gamma. Heads move from sharded split output to DRAM first. | `ttnn.rms_norm` with explicit exact HiFi4/FP32 destination configuration. |
| Q/K RoPE | Select supplied table rows; decode repeats over heads; split/negate/concat; FP32 cosine/sine products; add. | `rotary_embedding` or newer `rotary_embedding_hf`, preserving supplied table values. |
| Decode attention | Device page selection; integer physical row indices; embedding/reshape/permute K/V; per-KV-head QK; mask; max/subtract/exp/sum/divide; PV; concatenate. | `paged_scaled_dot_product_attention_decode` with BF16 Q adaptation, exact exponential, HiFi4 and FP32 destination accumulation. |
| Attention softmax | FP32 max/subtract/exp/sum/divide inside the preceding graph. | `ttnn.softmax(..., numeric_stable=True, compute_kernel_config=...)` is a narrower independent candidate if whole SDPA fails. |

Prefill already uses dedicated native SDPA. Its accurate norm/RoPE front end
also generates the decode cache, so a decode-only success does not authorize
changing prefill. Decode QKV and router projections use FP32 SFPU dot products;
substituting an ordinary FP32-output matmul reintroduces a separately proven
Src narrowing boundary.

## Concrete adaptations and failure discriminators

### RMSNorm

The API accepts tiled FP32 activations and FP32 gamma
(`normalization/layernorm/device/layernorm_device_operation.cpp:49,78,110`).
The current `[1,1,1,D]` tiled FP32 norm weights satisfy the gamma shape contract;
row-major activations are rejected by `normalization/rmsnorm/rmsnorm.cpp:34`.
Default RMSNorm disables FP32 accumulation and enables approximate math
(`rmsnorm.cpp:16–19`), so the default invocation is an inadequate control.

Try `norm_fp32` first, then `norm_external_weight`: the latter calls unweighted
native RMSNorm and keeps the final gamma product in equal-FP32 SFPU arithmetic.
Use `--norm-site input|head|post|router` and `--phase prefill|decode` to isolate a
failure before combining sites. Compare each named boundary on identical real
inputs and run the unchanged full-layer gate.

Neither adaptation removes all narrowing: the nonsharded factory assigns
Float32 consumer buffers `UnpackToSrc`, except selected accumulator/Welford
aliases; Welford is excluded for RMSNorm
(`layernorm_op_multi_core.cpp:437,511,883–926`). Epsilon is stored as BF16.
Therefore failure after explicit config is not evidence that the input layout
is unsupported. If an additional sharded variant is tried, inventory and verify
its actual kernel rather than assuming a layout change restores FP32 operands.
The unit test `tests/ttnn/unit_tests/operations/fused/test_rms_norm.py:121`
explicitly acknowledges TF32 unpack for its FP32 case.

### RoPE

Cast the *existing BF16 tables* to FP32 on device; do not upload newly calculated
FP32 tables. That keeps position semantics constant while widening product
buffers. The older op stores products in the table dtype and its multi-tile
core group 1 uses a default compute config; requested FP32 settings only reach
group 2 (`experimental/transformer/rotary_embedding/device/
rotary_embedding_program_factory.cpp:602–621,812–828`). This was a causal
functional-stage issue, not merely a speculative limitation.

The newer HF op's interleaved multi-tile factory applies FP32 destination config
to both groups and stores cosine/sine products in their table dtype
(`rotary_embedding_hf_multi_core_program_factory.cpp:468–488,573–594`). Its
interleaved reader swaps complete half-width tile ranges; the compute kernel
negates the first half and adds the two products. This matches the current
rotate-half expression, including supplied partial-RoPE tables.

`rope_hf_fp32` uses this interleaved **prefill kernel for both phases**. During
decode the already-selected single position is repeated over the logical head
rows, yielding `[1,1,H,D]` tables beside `[1,1,H,D]` input. It avoids sharded
decode constraints and preserves device-owned positions. This is a concrete
adaptation to test after a decode-layout error.

Gemma4's shared `attention/operations.py:208` warns that the HF op mis-rotated
global D512 partial-RoPE caches in its prior per-user decode use (~0.72 PCC).
Do not assume the separate interleaved route has the same failure. Validate
D256 sliding and D512 full layers independently, including a nonzero position.
Both new and old ops still use FPU binary products/addition and thus Src
narrowing. Passing a trivial position-zero test is especially weak evidence.

### Decode SDPA and softmax

Native decode and prefill SDPA validators permit BF16/BFP8/BFP4, **not FP32 Q**
(`transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:38–43`;
`transformer/sdpa/device/sdpa_device_operation.cpp:40–43`). Passing FP32 and
stopping on that error does not test fusion. `sdpa_exact` explicitly rounds Q
to BF16, keeps the current BF16 cache and INT32 page/position tensors, and uses
scale 1.0, the configured sliding window, exact exponential, HiFi4, FP32
destination accumulation and a bounded grid with K chunk 64.

The diagnostic compares three pairs on the same real cache: precise FP32-Q
versus fused; precise FP32-Q versus precise BF16-Q; precise BF16-Q versus fused.
These distinguish required input rounding from kernel error. Fused output is
cast back to FP32 before the existing output projection; this cannot recover
lost bits, but preserves the downstream dtype contract.

Decode's factory hard-codes BF16 intermediate and statistic buffers even with
FP32 destination accumulation (`sdpa_decode_program_factory.cpp:439–440`).
There is no exposed flag to make those buffers FP32. A genuine all-FP32 decode
SDPA adaptation would need C++ changes outside this stage's allowed scope.
Do not describe the tested configuration as FP32 SDPA.

If SDPA fails, native softmax is a distinct graph candidate: it accepts FP32
input and its factory selects FP32 intermediates with FP32 accumulation.
However, its attention factory also explicitly selects `UnpackToSrc` for input,
exponentials and statistics (`normalization/softmax/device/
softmax_program_factory_attention_optimized.cpp:459–491`). Check the actual
262144-wide large-tensor path and real route outcomes. Do not replace the
router's logits-top-k with probability-top-k; the functional controls proved
that rounded probabilities change close ranks.

## Diagnostic use and acceptance

The new probe patches only test-time module bindings after loading a functional
layer. It keeps a bounded first same-input comparison per phase/boundary as
device clones, then reports PCC, maximum absolute error and relative L2 before
device close. It does not perform host reads inside audited forwards or trace
capture. Its extra comparisons make its timing unsuitable for performance
claims. JSON embeds the selected harness's complete result and records failures.

Example commands for the runtime owner (not run by this investigator):

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_fused_precision --candidate norm_fp32 --norm-site head --runner batched --batch 32 --layer 0 --output models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/norm_head_sliding.json
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_fused_precision --candidate norm_external_weight --norm-site head --runner batched --batch 32 --layer 5 --output models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/norm_external_full.json
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_fused_precision --candidate rope_hf_fp32 --runner request_reuse --layer 5 --output models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/rope_hf_full_reuse.json
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_fused_precision --candidate sdpa_exact --layer 0 --length 4096 --steps 128 --output models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/sdpa_sliding_128.json
```

Initial controls should cover original seed42 full S33, seed19 reuse boundary
cases S31/S1025/S2049 and real batch32 for both kinds. Retained candidates still
need 4096/128-step, request reuse, continuation, maximum-context 262144/262143,
changed-position trace replay and warmed performance evidence. First-boundary
local comparisons are diagnostic sampling, not a substitute for this matrix.

Only the new Python probe and this report were added. Python compilation and
Black formatting were checked; no C++ build is required. No numerical or
performance success is claimed for these candidates by this investigator.

## Follow-up: paged cache reads and grouped matmuls

This follow-up is also source-only. No runtime files or tests were changed for
the following candidates.

### Tiled cache gathering: legal candidate, unfavorable implementation

`embedding/embedding.cpp:30–32` unconditionally converts tiled weights to
row-major before selecting rows. Thus the current flat-cache embedding really
does untilize the complete caller cache on each invocation, including sliding
attention's much smaller selected window.

A minimal **correctness candidate** is:

```python
# rows is the existing UINT32 [1, selected_pages * kv_heads * 32].
flat = ttnn.reshape(cache, (1, physical_pages * kv_heads * 32, head_dim))
indices = ttnn.to_layout(rows, ttnn.TILE_LAYOUT)
gathered = ttnn.tosa_gather(flat, indices)
# Continue with the existing selected-pages/heads/block reshape and permute.
```

Keep `cache_row_indices` unchanged: integer left shift plus integer offsets
avoids loss of physical addresses above 2**24. Keep full attention's direct
page-table path unchanged as well. `tosa_gather` requires rank-3 `[N,K,C]`
values and rank-2 `[N,W]` indices, which these shapes satisfy. It expands indices
over C and delegates to ordinary gather (`gather/tosa/gather_tosa.cpp:18–32,70`).
The native gather accepts UINT32/UINT16 indices, requires matching input/index
layouts and outputs the index shape (`gather/device/gather_device_operation.cpp:
70–77,96–105,146`). BF16 data is moved as bits, without FP32 address arithmetic.
`test_tosa_gather.py` checks exact equality for tiled BF16 values and UINT32
indices, but its largest tested gather extent is only 2048, not this workload.

This removes the explicit cache untilization, but **does not implement a sparse
cache read**:

- `gather/gather.cpp:89` transposes the gather dimension to the last axis, so
  gathering flat rows transposes the entire cache and the expanded index tensor.
- For a gather extent above 1920 elements, the tiled multi-core factory is
  selected (`gather_device_operation.cpp:16–43`). Its writer loops over every
  source-width tile for every assigned output-index tile
  (`gather_writer_single_row_multi_core.cpp:68–97`). The reader's bitmap skips
  element scans, but still consumes those already-read input tiles
  (`gather_reader_single_row_multi_core.cpp:146–162`).
- With a single full-context full-attention cache, flat input has 524288 rows
  and D512. Expanded UINT32 indices alone occupy 1 GiB. The source loop reads
  the 512 MiB BF16 cache once per selected-row tile: 16384 times, about 8 TiB
  aggregate input traffic per K or V gather, before additional movement.
  This is a source-derived traffic count, not measured latency. More physical
  pages make it worse. Page-level reshaping can reduce repeated scans, but
  still expands one index per output element and scans all physical pages.

The tiled multi-core reader uses UINT32 global indices and a within-tile local
offset; its UINT16 local offset does not itself truncate the global row ID.
The single-core path's larger UINT16 local offset is bounded by the <=60-tile
selector. The earlier corrupt **row-major page-table** multi-core result is not
proof that this tiled path is corrupt; these are different kernels.

Therefore tiled TOSA gather is worth a bounded exact-mapping control, but the
source gives a concrete scalability concern rather than a promising default
replacement. First compare gathered K/V bitwise on random physical pages,
duplicates at clamped sliding tails, positions 31/32/33 and 1023/1024/1025. Measure
small and medium geometries before attempting the potentially enormous full
shape. Do not claim a performance win merely because `untilize` disappears.

No existing standalone dynamic paged tile loader was found in the relevant
op library. `experimental.nlp_kv_cache_load_slice` takes host uint32 start/end
scalars, no page table or device index tensor, and returns a contiguous tiled
slice in L1; its validator also limits batch*heads to the core count. It cannot
substitute arbitrary physical pages or changed device positions on trace
replay. The `experimental.paged_cache` public APIs update/fill caches, rather
than load them. Matmul's `gather_in0` gathers distributed operand shards and
has no arbitrary row-index tensor. A direct sparse paged cache-read kernel
would need changes outside the permitted runtime scope.

### Group QK and PV matmuls while retaining per-head softmax

This is the smaller promising candidate. Let N be KV heads, G=16/N and
L=selected_pages*32. Sliding has N=8/G=2/D=256; full has N=2/G=8/D=512. The gathered K/V
already have shape `[1,N,L,D]`. Group Q from `[1,1,16,D]` to `[1,N,G,D]`, then:

```python
grouped_q = ttnn.reshape(q, (1, num_kv_heads, group, head_dim))
scores = ttnn.matmul(
    grouped_q, keys, transpose_b=True,
    dtype=ttnn.float32, compute_kernel_config=self.compute,
)  # [1,N,G,L]
probabilities = []
for head in range(num_kv_heads):
    per_head = scores[:, head:head + 1, :, :]  # [1,1,G,L]
    # Keep the existing where/max/subtract/exact-exp/sum/divide verbatim.
    probabilities.append(existing_per_head_softmax(per_head, allowed))
grouped_probabilities = ttnn.concat(probabilities, dim=1)
grouped_output = ttnn.matmul(
    grouped_probabilities, values,
    dtype=ttnn.float32, compute_kernel_config=self.compute,
)  # [1,N,G,D]
output = ttnn.reshape(grouped_output, (1,1,16,head_dim))
```

This preserves the current grouping: query heads `head*G:(head+1)*G` use KV
head `head`. It needs no K/V repetition. Both operand ranks and leading batch
dimensions match the nonbroadcast matmul validator
(`matmul/device/matmul_device_operation.cpp:244–310`). Ordinary `ttnn.matmul`
supports transpose-B and batched input; do not confuse it with the separate
`matmul_batched_weights` API, which rejects transpose. Relevant tests include
`test_matmul.py:test_matmul_with_transpose_a_or_b` and its batched configuration
tests. Keep FP32 output, HiFi4, exact math and disabled packer accumulation.

The reshape is **not necessarily a view**: each logical G2/G8 head group gets
32 padded rows. The public reshape implementation dispatches a device tiled
repack, and that primitive explicitly accepts FLOAT32
(`reshape_view/device/reshape_device_operation.cpp:24–30`). Do not force a
physical-volume reshape that reinterprets padding as real heads. An alternative
control is concatenating the existing query slices along dimension1, which
preserves their current padding and then exercises only grouped matmul. Check
reshape/concat output bitwise before attributing a difference to arithmetic.

First group **QK only**, leaving the original softmax and PV loop; then group
PV after the former passes. On identical gathered inputs, compare each group's
scores, probabilities and final output against the current loop, including
maximum errors and final-layer routing outcomes. Different matmul program
selection may change accumulation even with identical compute flags, so
algebraic equivalence is not sufficient.

At L=262144/N=2, a grouped FP32 `[1,N,32,L]` score tensor occupies 64 MiB; the
probability tensor has the same padded size. The original per-head loop can
release each head's temporaries before the next. Record the changed memory
peak and use warmed timings, including the small Q/output repacks and
probability concatenation. This candidate preserves the full context and all
device-owned addressing because page selection/cache reads remain untouched.

## Final candidate audit: actionable follow-ups

This is a source/recorded-evidence audit of the experimental `fused_decoder.py`
and `PATTERNS.md`, not a stage review. Hardware ownership remains with the
parent. The following items are hypotheses or evidence qualifications, not a
stage acceptance decision.

1. **Try the dedicated fused K/V cache writer.** This pattern was absent from
   the initial ledger. `ttnn.experimental.paged_fused_update_cache(k_cache,
   k_update, v_cache, v_update, update_idxs_tensor=cache_pos,
   page_table=page_table)` combines the two current cache writes. It has no
   block-size/head-count override; its tiled factory derives those values from
   cache1's declared shape, which already matches this stage's actual 32-token
   pages and KV-head count. Both caches have matching geometry/dtype.

   The critical adaptation is **disjoint input shard grids** with equal core
   counts (`experimental/paged_cache/device/fused_update_cache/
   paged_fused_update_cache_device_operation.cpp:351–365`). Reusing today's
   `cache_memory` for both updates fails this contract. For the existing B1
   slot loop, prepare two one-core ROW_MAJOR-oriented HEIGHT_SHARDED L1 configs
   with shard `[32,D]`, e.g. cores `(0,0)` and `(1,0)`. Retain the explicit SFPU
   BF16 casts and move each update directly to its designated grid. These
   replace the existing two memory conversions; they need not add two more.
   The op updates both original cache buffers in place and returns them.
   INT32 row-major page/position tensors already match its validators.

   Check K/V rows and untouched bytes exactly at positions 31/32/33, changed
   physical page tables, and trace replay for D256/N8 and D512/N2. Its existing
   unit tests check BF16 input/cache bitwise equality, but use D128 and pages 64
   or 128; they do not establish the target contract. Include reshard costs in
   timing. No cache alias or skipped BF16 conversion is part of this proposal.

2. **Reuse the unweighted K/V norm when projection weights are tied.** Full
   attention's loader sets `v_w = k_w`
   (`models/demos/gemma4/tt/attention/weights.py:78–82`), and `TiedQKV` now
   duplicates the same computed K columns. The current head graph nevertheless
   computes `normalize(k, eps, k_weight)` and `normalize(v, eps)` separately.
   With the selected external-weight normalization, these share an identical
   unweighted normalization. After checking pre-norm K/V equality, compute
   `base_kv = normalize(k, eps)`, use it directly as V, and multiply it by
   `k_weight` for K. Q is unchanged; only K subsequently gets RoPE. The unused
   split V's conversion to DRAM can also be omitted. This is a local
   common-subexpression candidate, not permission to tie post-RoPE K to V.
   Compare head outputs/cache rows exactly and rerun the combined full-kind
   checks; the existing tied-projection headline alone does not test this.

3. **Keep RoPE decisions specific to kind and phase.** The newer HF op passes
   full-kind batch32 and the 4096/128-step standalone control (minimum decode
   0.99977276 in `rope_rope_hf_fp32_headline_full.json`). Sliding prefill-only
   fails 0.99312926, while sliding decode-only passes the recorded headline at
   0.99902595 (`rope_hf_{prefill,decode}_headline_sliding.json`). Thus full-kind
   both phases plus sliding decode-only is a concrete policy to test in the
   final combination. It is unsound to reject all newer HF RoPE from the
   sliding all-phases failure; it is equally unsound to accept the combined
   policy from these isolated checks. Old native RoPE was retried after its
   logical-shape repair and fails the full headline at 0.99302951; that is a
   valid numerical rejection of that tested policy, not a layout-only error.

4. **Localize the combined sliding reuse failure without conflating norms.**
   The parent reports S2049 decode 0.9885523 despite passing prefill, batch32
   and full reuse. At position 2049 the retained paging expressions select
   pages 32..64 and allow absolute tokens 1026..2049, consistent with a 1024-token
   window; no new addressing discrepancy is evident in the grouped source.
   First remove RoPE policy alone. Then remove `common` alone while retaining
   native input/head/post/router norms. Finally remove both `norm` and
   `common` to restore the functional normalization boundaries while keeping
   grouped attention. Removing `norm` alone is not the latter control:
   `common` still replaces two weighted shared/expert norms with one unweighted
   norm plus external weights, now using the SFPU implementation. If needed,
   keep batched QK/PV while restoring per-head softmax shapes to distinguish
   changed matmul lowering from changed reduction geometry. Use the exact
   reused request and page map, rather than a fresh random S2049 prompt.

   The parent's subsequent real ablations localize this to native normalization:
   dropping RoPE alone still fails near 0.98862, additionally dropping common
   normalization fails near 0.98872, and restoring functional norms passes all
   nine reuse cases. Native external-weight norm sites are now being tested
   separately at this exact request. This supersedes the earlier ambiguity
   between RoPE, grouping and normalization; it does not identify a particular
   norm site or prove softmax.

5. **Native softmax is not disproved by the failing combined RoPE run.**
   `softmax_sliding_headline.json` also enables the independently failing
   sliding RoPE policy. The isolated `softmax_only_sliding.json` has minimum
   headline decode 0.99553139. Continue the current real-kind/combined matrix
   and timing control before making a universal rejection. The margin above
   threshold is small; component probability PCC cannot replace routed-layer
   comparison.

   For exact FP32 `[1,2,8,262144]`, native stable softmax is **not statically
   blocked by allocating the entire width in L1**. The rank4/last-dimension
   selector uses `SoftmaxProgramFactoryAttentionOptimized`. Its 90%-L1 estimate
   triggers the large kernel at Wt8192; IN0, EXPS and X are each capped at 80
   FP32 tiles (`softmax_program_factory_attention_optimized.cpp:151–161`).
   Including OUT0's 8 tiles, seven single FP32 tiles and one BF16 padding tile
   gives 1,046,528 bytes/core of declared buffers, before framework and other
   resident allocations. Work splitting uses two padded tile rows, hence two
   cores for this shape. The large reader scans scores three times for stable
   softmax. Its 103 passes cover 102*80+32 width tiles and explicitly realign CB
   pointers after the final partial pass.

   The more concrete concern is numerical: the large kernel reloads the prior
   max/sum through `copy_tile`, and the factory pins those FP32 buffers to
   `UnpackToSrc`; the denominator can be narrowed at every width pass
   (`softmax_large_tensor.cpp:258–285`). A 4096 headline can select the small
   kernel, so it does not establish large-path equivalence. Compare
   same-input probabilities, absolute/relative errors, row sums and PV outputs
   at widths 8192 and 262144, followed by the routed-layer gate. There is no
   context truncation in that experiment.

   If actual resident-buffer pressure prevents the default route, a narrow
   layout adaptation is reshape to `[1,1,2,8,L]` (rank5), native softmax on the
   last axis, then reshape back. The selector routes rank5 to GeneralWLarge,
   whose declared buffers total 13 FP32 tiles (~52 KiB), independent of L.
   This preserves logical values/context and avoids increasing the padded
   head count. It uses a different reduction kernel with its own Src narrowing;
   it is a layout/L1 alternative requiring fresh numerical and performance
   checks, not an accuracy guarantee. Do not shard a complete 262144-wide row
   into L1 or reduce the context to make the first route fit.

6. **Grouped attention has no source-level context cap in its current
   addressing/reduction graph.** It preserves full attention's direct table,
   integer cache-row mapping and original mask. Generic FP32 max/sum remain
   on the accurate SFPU route after batching
   (`reduction/generic/device/reduce_op.cpp:160–168`); the leading KV-head
   dimension does not enter the width reduction. The large score/probability
   allocations and changed matmul program selection still need the planned
   262144/262143 tests. If only batched matmul fails or regresses there,
   bounded `MatmulMultiCoreReuseProgramConfig` output blocks are an adaptation
   to try before rejecting all grouping; keep context and compute precision.

7. **Keep the GELU timing claim narrow and its probe reproducible.** Recorded
   component medians favor separate decode GELU by about 1.4–1.9 microseconds,
   while fused prefill GELU is about 20.7 microseconds faster. Both paths are
   recorded bitwise equal. These are traced host timings, not device-kernel
   durations, and the small decode difference is not a substantial standalone
   performance claim. The current selected runtime forces decode GELU separate
   even when `PackedExperts(fused_gelu=True)`. Consequently a future unchanged
   `probe_fused_components` run labels two identical decode paths as `packed`
   and `packed_gelu`; preserve the original evidence and add an explicit
   experimental override if that comparison is repeated. The selected
   phase-specific implementation remains subject to overall warmed timing.

The direct tiled-gather rejection now also has a bounded device comparison:
`tiled_cache_gather.json` records exact output for both alternatives, but
traced host time 24,390.02 microseconds versus 658.65 for embedding at cache shape
`[160,8,32,256]`. This supports rejecting that tested gather lowering and is
consistent with the source traffic analysis above. It does not reject a future
purpose-built sparse tile loader.

## Softmax adaptation after the reuse ablations

There is no source reason to call rank-5 GeneralW more numerically faithful than
rank-4 AttentionOptimized. Both general factories explicitly force all consumed
FP32 buffers to `UnpackToSrc`. GeneralWSmall also packs `x-minus-max` and reloads
it before exp (`moreh_softmax_w.cpp:96–137`), whereas AttentionOptimized performs
subtraction and exp in the same DST residency (`attention/compute/softmax.cpp:35–59`).
GeneralWLarge repeatedly writes and reloads an elementwise running sum with
`add_tiles_to_cb` for every width tile (`moreh_softmax_w_large.cpp:87–118`): at
262144 context this creates 8192 accumulator reloads before the final row
reduction. The attention large path has approximately 103 partial reductions.
The general path saves L1, but neither that saving nor the rank change addresses
the identified loss of FP32 mantissa bits.

Two concrete adapted candidates remain, in this order:

1. Keep the existing FP32 SFPU max/subtraction, then call dedicated softmax on
   the centered tensor with `numeric_stable=False`, exact math and FP32
   accumulation. This avoids subtracting two individually Src-rounded original
   values. In particular, scores close to their row maximum enter the dedicated
   kernel as their already-computed small differences. The dedicated kernel
   still narrows centered inputs, exponents and reciprocal normalization, so
   this is a discriminating candidate, not a numerical guarantee. Use rank 4
   here: the general width kernels compute their own max/subtraction regardless
   of this flag. Compare the identical saved scores from the exact failing
   reuse request, then both real layer kinds and full context if it passes.

2. If the dedicated normalization remains below the gate, fuse only subtraction
   with exact exp and preserve the current FP32 max, sum and division:

   ```python
   exps = ttnn.subtract(
       scores,
       ttnn.max(scores, dim=-1, keepdim=True),
       activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)],
   )
   probabilities = ttnn.div(exps, ttnn.sum(exps, dim=-1, keepdim=True))
   ```

   Leave `fast_and_approximate_mode` unset: explicitly passing `False` is
   rejected for FP32 output by `binary_ng_device_operation.cpp:20–48`. That
   explicit flag requests BF16 rounding behavior, not more precise FP32
   arithmetic. Binary-NG still chooses the SFPU route for FP32 SUB and configures direct FP32
   unpack to DST (`binary_ng_device_operation.cpp:49–56`,
   `binary_ng_program_factory.cpp:1219–1241`). The column-broadcast kernel
   executes post-activations on that same DST result before packing
   (`kernels_ng/compute/eltwise_binary_sfpu_col_bcast.cpp:119–133`). EXP parameter
   `0.0` generates `exp_tile<0>` (`unary_op_utils.cpp:365–368`), matching the
   existing exact exponent path. Do not substitute the string `"exp"`: its
   resolver requests approximate exp (`unary_op_utils.cpp:1021`). This removes
   one op launch and one FP32 intermediate read/write without routing the
   reduction or operands through native-softmax Src registers. Compare the
   same-input exponent/probability tensors and routed-layer output; removal of
   an intermediate pack still deserves the real numerical gate and warmed
   timing before selection.

Subsequent parent-run results now resolve these proposals: centered native
softmax fails the guarded sliding headline at minimum decode PCC
0.9934474790 (`centered_softmax_sliding.json`). Guarded native softmax passes
the full headline, but the parent reports approximately 5688 microseconds
versus 5651 without it; this is no demonstrated speedup. The initial exp-merge
run hit the explicit-flag validator above. Removing that flag gives passing
headline records for both kinds (`merged_exp_{sliding,full}_retry.json`,
minimum decode PCC 0.9989820301 and 0.9997895484 respectively). These isolated
records do not establish the final combined coverage matrix.

## Remaining structural audit and page-axis gather control

The current graph covers the material listed skill patterns: dedicated head
split/concat, native prefill attention, packed gate/up, grouped QK/PV, tied
full-kind K/V projection and unweighted norm reuse, fused K/V cache writes,
common residual normalization, input GELU in binary multiplication, residual
RMSNorm, and output scale/cast merging. Native decode SDPA, native softmax and
universal native head RMSNorm have had concrete dtype/layout or precision
adaptations followed by real-layer checks. The remaining precise norm boundary
has two smaller graph candidates, now being tested by the parent:

- Fuse epsilon addition with `UnaryWithParam(RSQRT, 0.0)` on the FP32 binary
  output, leaving square/mean and subsequent products unchanged. Omit the
  explicit BF16-only accurate flag here too. The parent reports the rsqrt
  candidate preserves the sliding headline PCC and about 5142 versus 5148
  microseconds; this small observed difference needs the overall warmed gate.
- Pack Q/K/V head rows before their common unweighted precise RMS math, then
  slice and apply the distinct Q/K weights. This combines independent row
  reductions without mixing their reduction axes. Decode concatenation is
  along head/row dimension 2; prefill heads lie on dimension 1. Logical rows
  and tile padding must be preserved. `headpack_sliding.json` now records the
  same minimum headline decode PCC as the ungrouped precise-head candidate;
  broader coverage and timing selection remain with the parent.

If the selected policy retains HF RoPE, Q/K are another independent peer pair:
concatenate normalized Q/K along the decode head axis (dimension 2), perform
one rotation with shared table preparation, and split the result. In prefill
the head axis is dimension 1. Both tensors use the same D and rotation
convention, but the added concat/slices and changed tile padding can cancel
the saved launch; test same-input head/cache values and warmed total time.
This candidate has no live target if measured policy removes native RoPE.

The cache read remains the largest structural concern in this part of the
source. It cannot become a persistent row-major cache just by selecting a
different update factory: ordinary paged update requires TILE cache
(`update_cache/paged_update_cache_device_operation.cpp:42–43`), and the fused
operation also requires TILE cache (`fused_update_cache/
paged_fused_update_cache_device_operation.cpp:184`). Its row-major factory
name refers to the **update input** layout. `indexed_fill` supports row-major
data but creates a new output tensor and copies the unmodified data, so it
does not offer a cheap in-place sparse mirror update. Changing the public
cache representation or adding a second cache ownership/coherence protocol
would exceed a local equivalent graph rewrite.

One useful layout adaptation was not measured by the original flattened-row
gather experiment: gather **physical pages** along dimension 0 directly from
`[physical_pages,Nkv,32,D]`. Expand the selected UINT32 page IDs into
`[selected_pages,Nkv,32,D]`, then call `ttnn.gather(cache, dim=0, index=...)`.
The output is already the same page/head/token order consumed by the existing
post-embedding reshape/permute. This does not assume consecutive pages and
preserves arbitrary device-owned page selection.

At the original 160-page/33-selected-page sliding geometry, the gather's
internal transpose gives source width 160 (5 tiles), rather than the
flattened source width 40960 (1280 tiles). This chooses `SingleRowSingleCore`
instead of `SingleRowMultiCore`; despite its name, work is distributed across
tile rows. The former reads each input tile row once, holds it in L1, and
reuses it across both output-width tiles
(`gather_writer_single_row_single_core.cpp:65–91`,
`gather_reader_single_row_single_core.cpp:166–182`). This removes the repeated
source-cache scans responsible for the earlier poor result. It still needs
full-cache transposition, an index per output element, and scalar tile
indexing, so faster execution is not established from source alone.

For full context with 8192 physical/selected pages this still takes the
multi-core path and can rescan the full cache 256 times. It is therefore a
bounded sliding candidate, not a justified universal replacement. Keep the
existing embedding path as the full-kind/max-context control; do not reduce
the context capacity or narrow physical page IDs.

`tests/probe_fused_cache_gather.py` now preserves the original `embedding` and
`tiled_gather` controls and adds:

- `page_axis_gather`: pre-expanded UINT32 indices, isolating the changed
  gather axis and all gather-internal movement.
- `page_axis_gather_dynamic`: includes device repeat/index expansion in the
  captured graph. IDs remain device tensors; repeat may use its legal
  row-major fallback for the singleton input shape.

Both compare output bitwise to the same random cache/page map, validate trace
replay output, and report median plus individual warmed host timing samples.
The default seed, original cache values/page choices and existing candidate
names are unchanged. Shape arguments support bounded controls; no hardware
was run by this source-audit agent. A passing movement-only probe would still
need insertion into the real sliding reuse/batch/headline graph and comparison
with index construction included before selection.

The parent has now run all four movement controls in
`cache_gather_adapted.json`. All outputs and trace replays are bitwise equal.
The warmed host medians are 138.87 microseconds for embedding, 24391.75 for
flattened gather, 3980.21 for page-axis gather, and 4106.67 with page-index
expansion included. Thus page-axis adaptation repairs the worst repeated-read
behavior but remains much slower; both gather policies are rejected for this
measured geometry. The earlier 658.65-microsecond embedding number came from
a shorter initial timing window and should not be presented as its warmed
steady baseline. The newer embedding samples converge to 138.87 and 138.51
microseconds after a 310.65-microsecond first sample.

The remaining Q/K RoPE peer merge now has a standalone test-only control in
`tests/probe_fused_rope_merge.py`. It calls the live `FusedAttention.rotary`
method separately for FP32 Q `[1,1,16,256]` and K `[1,1,8,256]`, versus one
call on their concatenated 24 logical rows followed by the two slices. Its
random inputs are row-normalized in host FP32 and use paired BF16 cos/sin
tables; source-equivalent compute flags are exact HiFi4 with FP32 destination
accumulation and no packer accumulation. The captured merge includes concat,
shared table preparation and both output slices. It reports Q/K PCC, exact
equality, maximum error, replay equality, and median plus individual warmed
trace timing samples after a separate warmup batch. This is an explicit
same-input component control, not real-weight final-layer validation. No
hardware was run by this source-audit agent.

## Direct-M expert peer batching

The original grouped-prefill grid warning is not a source blocker for a
different, simpler contract: feed `[1,1,M,H]` directly into packed sparse
gate/up, set `per_core_M=M/32`, and keep the all-ones E-entry sparsity tensor
with `nnz=E`. The sparse output-shape helper explicitly takes M from the last
two A dimensions, places B's expert batch dimensions immediately before M/N,
then prepends A's batch dimensions (`matmul/device/sparse/
sparse_matmul_device_operation.cpp:29–72`). The resulting
`[1,1,1,E,M,2J]` can reshape directly to `[1,E,M,2J]`; both-sparse down produces
`[1,E,M,H]`. No token-group/expert transpose is needed.

The factory computes `Mt` with `fuse_batch=false` and
`num_blocks_y=ceil(Mt/per_core_M)`
(`sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:156–170`). Matching M in
the existing `_build_sparse_matmul_config` keeps that count at one for
M32/64/128/256/1024. The N grid, K blocking, subblocks, output dtype and compute
flags need not change. Its CB sizes use `out_block_h`, which remains one,
rather than `per_core_M` (`:196–228`); increasing this M loop does not itself
multiply the per-core CB allocation. Larger output/intermediate DRAM tensors
remain a real cost to measure.

Do not equate this with demonstrated weight-traffic reuse. The in1 reader
loops over output-height blocks before its K-block weight reads
(`reader_bmm_tile_layout_in1_sender_writer_padding.cpp:321–446`), so keeping
`out_block_h=1` still reads the weights for each M tile. The concrete rewrite
combines independent tile dispatches and removes repeated Python/operator
boundaries; numerical and speed equivalence need the experiment.

`tests/probe_fused_expert_batch.py` provides `DirectMExperts`, a test-only
subclass sharing the original packed buffers. It captures actual expert input
and routing tensors from the selected `FusedDecoder` on the first 1024 rows
of recorded `headline_inputs.pt` using the pinned real layer weights. Every
candidate then consumes those identical tensors, compared with the live
32-token loop. The default M list is exactly 32/64/128/256/1024; only M batching
and matching `per_core_M` change. It records raw sparse output shapes, PCC,
exact equality, maximum error, trace replay equality, and warmed median/individual
host timing samples. `--batch-tokens` narrows an isolated retry, and layer 0/5
select the two real layer kinds. Black and Python compilation were checked;
this source-audit agent did not run hardware or change runtime code.

## Dedicated expert weighted reduction

The final expert mixing suffix is a contraction and therefore a dedicated
matmul candidate: for each token, `[1,E] @ [E,H]`. The existing decoder first
multiplies each BF16 expert output by its route and then sums experts. That
intermediate product rounding is not generally identical to a native matmul's
accumulation, even with HiFi4 and FP32 destination accumulation. Keep the
same-input comparison against the live graph; mathematical equivalence alone
does not satisfy the numerical contract.

`tests/probe_fused_expert_mix.py` captures actual sparse-down outputs, routing
tensors, and mixed outputs from the current selected `PackedExperts` during
real-weight S64 prefill and S1 decode at position 64. Inputs come from the
recorded headline fixture; decode follows that prefill using its populated
cache. A test-only subclass wraps the live sparse operation only during capture,
retains its down output, and lets the original expert suffix finish normally.
Before benchmarking, the copied suffix must match that captured live output
bitwise. Neither projection outputs nor routes are synthetically generated.

Two bounded native controls use exact HiFi4, no packer L1 accumulation, BF16
output, and BF16 versus FP32 destination accumulation. Native matmul accepts
tilized floating-point operands with equal logical and padded K dimensions
(`matmul_device_operation.cpp:33–94`); no new geometry or dtype sweep is added.
Decode includes the existing movement from `[1,E,1,H]` to `[1,1,E,H]`. Prefill
includes down movement `[1,E,64,H]` to `[1,64,E,H]`, route movement to
`[1,64,1,E]`, and output movement back to `[1,1,64,H]`. Permutations preserve
the padding contract when the token dimension becomes a batch dimension.
These movements are part of every candidate's measured trace, not preparation
excluded from its latency.

The JSON contains exact equality, PCC, maximum error, finite-output and replay
checks, warmed host trace medians/samples, and speedup against the existing
multiply/reduce. `--layer 0` and `--layer 5` cover the two real layer kinds;
`--candidate matmul_bf16_acc` or `matmul_fp32_acc` isolates a supported control.
Only a numerically passing and faster component is eligible for subsequent
integrated decoder checks. This source-audit agent ran Black and Python
compilation only; candidate acceptance remains pending the parent's serial
hardware measurements.
