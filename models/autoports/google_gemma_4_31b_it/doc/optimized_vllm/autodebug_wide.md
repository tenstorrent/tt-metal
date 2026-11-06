# AutoDebug: wider batched prefill loses solo bit equality

## Finding

**The leading diagnosis is a batch-dependent change in the first input RMSNorm's numerical algorithm.** `MultichipDecoder._norm` uses block-sharded RMSNorm only through 1024 total padded rows, then falls back to interleaved RMSNorm. The 4096-row experiment crosses this boundary; solo and B8×128 do not. These implementations use different reduction partitions **and different reciprocal precision**, before the first QKV projection. This is a concrete mismatch with the intended solo-bit-equality contract, not proof that either TTNN RMSNorm implementation is mathematically invalid.

Confidence: high in the dispatch/numerical discrepancy; moderate that it is the first actual divergent tensor in the recorded run. No intermediate tensors were recorded, and no device execution was performed here. The final 9.15625 logit delta is not quantitatively explained by static inspection. A focused norm comparison is needed before claiming a complete root cause or accepting a wider fix.

The existing 1024-row group cap is the evidence-backed safe choice. For a wider experiment, preserve the established norm computation using bounded batch slices while leaving the wide path in DRAM. Do not simply force all 4096 rows into L1 or remove the norm threshold.

## Scope and evidence

Inspection date: 2026-11-06. HEAD: `f2abf54e1db6c1c7bea6ae5f530ae8ae02a5b9ce`. This checkout has existing modifications and untracked stage-10 code/evidence; the diagnosis covers the inspected working tree, not just HEAD. The probe JSONs do not contain complete source hashes, so historical identity of every implementation detail cannot be independently established. The work log and current call paths agree on the threshold contrast.

Read `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`, the AutoDebug skill, the autoport and generic TTNN sources. No excluded Gemma implementation was consulted. No source changes, hardware commands, credentials changes, launcher invocation, or publication were performed. This report is the only intended new artifact.

Direct observations from `models/autoports/google_gemma_4_31b_it/doc/optimized_vllm/`:

- `probe_b8_s128.json`: all eight requests, all four devices, logits and both selected layers' K/V tensors are exactly equal to solo.
- `probe_b32_wide.json`: batch 32, length 128, `max_group_rows=4096`; every request fails logits and K/V equality on all four devices for layers 0 and 5. Per-request maximum logit deltas range from 2.15625 to 9.15625.
- `tracked_b32_control.json`: passing B32×128 uses four groups of eight, with exact logits/cache and trace/allocation checks. Logical batch 32 is therefore not itself the failure boundary.
- `probe_b8_tails.json`: lengths 5,31,32,33,63,64,65,96,97,127 pass. These remain at or below 1024 padded group rows.
- The selected precision file names `head_bfp4`: BF16 residuals/norm weights/embedding, BFP4 LoFi projections and head, BF8 prefill CCL/activation/KV. Norm compute itself is HiFi4, exact-mode, FP32 destination accumulation (`tt/multichip_decoder.py:89`).

“All K/V differ” means each compared tensor has at least one unequal element. The artifacts do not supply cache error magnitudes, mismatch coordinates, or counts. They do not establish widespread element corruption or the first divergent operation.

## Earliest inconsistent transition

Paths below are relative to `models/autoports/google_gemma_4_31b_it/` unless explicitly rooted at `ttnn/`.

`tt/batched_prefill.py:15-19` admits `(count,size)` according to `count*size <= max_rows`. Length128 needs no rounding. `tt/optimized_decoder.py:606-609` selects L1/clones the residual only for `B*padded_S <=1024`. `_qkv` then immediately calls `_norm(x,"input_layernorm")` at `:383`.

`tt/multichip_decoder.py:317-347` computes total padded rows and selects:

| Case | Physical input | Rows | Residual placement at first norm | Norm configuration |
|---|---|---:|---|---|
| Solo | `[1,1,128,5376]` | 128 | private L1 clone | grid 8×4, shard `[32,672]`, block_h=1, block_w=21, subblock_w=3 |
| B8 | `[8,1,128,5376]` | 1024 | private L1 clone | grid 8×8, shard `[128,672]`, block_h=4, block_w=21, subblock_w=3 |
| B32 wide | `[32,1,128,5376]` | 4096 | DRAM | default interleaved norm over full width 5376 |

The ordinary model leaves `residual_sharded=False` (`tt/multichip_decoder.py:30`, model constructor `tt/model.py:46-65`). Thus the wide path calls `OptimizedDecoder._norm` at `tt/optimized_decoder.py:247-253`; it does **not** run distributed RMSNorm pre/post-all-gather. All eight width shards in the passing norm together cover 5376 columns, with no padding on the reduction axis. The output returns to its input memory config.

### Lowered numerical difference

The sharded call explicitly supplies `LayerNormShardedMultiCoreProgramConfig`. Its factory and compute path are:

- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core_sharded.cpp:265-278`: local scale `winv = num_blocks/logical_K = 8/5376 = 1/672`, converted to BF16; global scale `cinv = 1/8`, also BF16 (exact).
- `.../device/kernels/compute/layernorm_sharded.cpp:456-498`: reduce squared values across 21 tiles per width shard, then reduce the eight partial results across cores.
- `.../device/sharded_layernorm_factory_helpers.cpp:116,675-685`: partial/statistic buffers use Float32 with this kernel config. It would be incorrect to describe these as BF16 partial buffers.

BF16(1/672) = 0.00148773193359375. Multiplying this by 672 gives 0.999755859375. Even before reduction-order effects, this differs from the interleaved reciprocal calculation.

The interleaved call goes through `.../device/layernorm_op_multi_core.cpp`. The normal compute kernel invokes `numeric::row_wise_mean` in `.../device/kernels/compute/layernorm.cpp:285-287`; `ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/numeric.h:228-240` applies FP32 `1/N` after summation. The streaming large-tensor kernel likewise applies `1/W` after summation (`.../device/kernels/compute/layernorm_large_tensor.cpp:350-355`). Here N/W is 5376, or 168 tiles.

The precise normal-versus-streaming interleaved choice depends on usable L1 capacity (`layernorm_op_multi_core.cpp:291-349`). The artifacts do not capture that allocator state; do not assume one of these variants solely from the Python shape. Both differ from the passing sharded scale and reduction tree. RMSNorm is not the Welford LayerNorm path.

This discrepancy occurs before any first-layer projection, head norm, RoPE, or cache write. A small change in the BF16 normalized hidden values can change BFP4/LoFi projection outputs and BF8 cache quantization; later residual/MLP/attention operations can propagate it into layer 5 and logits. That chain is plausible, not measured. The reciprocal discrepancy alone corresponds to roughly a 0.000122 relative normalization scale change for negligible epsilon; it does not prove the observed final error magnitude.

## Other paths adjudicated

1. **Batch preparation/cache ownership: no contradiction found.** `tt/batched_prefill.py:23-24,59-67` repeats positions 0..127 independently, selects the requested slots' pages, fills four pages per request, and supplies zero starts. Terminal indices `row*128+127` are correct (`:56,81-85`). The probe snapshots batched caches before solo overwrites and compares the same page IDs and valid tokens (`tests/probe_batched_prefill.py:48-77`). The paged-fill reader/writer map work rows to batch/head/sequence tiles; each produced width-tile block has a matching wait/write/barrier/pop. No B8-versus-B32 page remapping defect was identified.
2. **First projection: secondary candidate, no proven violation.** `_linear` uses explicit minimal-matmul blocks M=4/K=8/N=16 and subblock1×8 (`tt/optimized_decoder.py:260-287`). The wrapper preserves input/weight order (`ttnn/cpp/ttnn/operations/experimental/minimal_matmul/minimal_matmul.cpp:26-59`). The descriptor folds batch into M using physical volume/K (`.../device/minimal_matmul_program_descriptor.cpp:250-260`), derives input/weight/output formats independently (`:218-235`), and selects the Metal2 data-movement/compute kernels (`:698-700,837`). First-layer sliding QKV has K=5376, N=4096 per chip; M=1024 or4096. The strict `M>N` transpose test at `:302` is false in both cases. The full-attention QK N=4608 also does not cross that test. L1/DRAM access and M work counts change, but no concrete lost-batch or format mismatch was found.
3. **RoPE, SDPA, output collectives, MLP and head cannot individually explain first-layer V inequality.** V is produced/normalized/cast before SDPA, and V does not pass through RoPE (`tt/optimized_decoder.py:447-466,610-628`). These operations may still introduce additional later divergence after a norm-only repair. In-place versus out-of-place residual/GELU multiplication also changes with `prefill_l1` (`tt/multichip_decoder.py:504,514`), but happens after layer-0 cache production.
4. **Final-logit extraction is not sufficient as a root cause.** Cache inequality is an independent observation preceding extraction. The reduced probe's terminal batching also stays within the same one-physical-M-tile head path for solo/B8/B32.
5. **Completion boundaries:** forward operations enqueue work; `host()` uses per-device `ttnn.to_torch` to materialize logits/cache before CPU equality checks. This is not a test that declares success merely because enqueueing finished. No trace replay or timing occurs before the failing assertion. A lifetime/race issue remains possible in principle, but no source-level evidence supports it over the norm discrepancy.

## Minimal discriminating probe (follow-up, not executed)

Use the same selected weights/policy and one saved layer-0 embedding tensor for the exact 32×128 tokens. Avoid changing the global memory policy or sampler.

1. Verify the wide embedding equals the corresponding solo embeddings on every device. This checks the only earlier arithmetic chain (embedding/gather/scale).
2. On that **same** `[32,1,128,5376]` BF16 tensor, compute the current wide input norm and four batch-axis B8 norm slices using the existing sharded configuration. Return both outputs to DRAM. Compare each against solo norm results. Record exactness, differing-element counts, max/RMS errors and first mismatching `(device,batch,token,channel)` with values.
3. If the norm is the first mismatch, retain the original wide projection config and feed it the concatenated bounded-norm output. Compare QKV before splitting, normalized V and K before cache casting, then BF8 K/V before fill. Read cache only if those match. This distinguishes the norm producer from a downstream consumer sensitive to shape or storage provenance.
4. If bounded norms restore first-layer K/V, apply that policy to **all four named layernorms** per layer for a full reduced-probe A/B. Keep `prefill_l1=False` for the wide computation. Input norm alone cannot establish later equality because post-attention/pre-FF/post-FF norms cross the same boundary.
5. If norm outputs already match, demote this diagnosis and locate the first difference through projection, split, head normalization and fill in that order. Repeated identical runs distinguish deterministic numerical differences from a race.

Only after the focused probe, test boundary cases B8×128 (1024), B16×64 (1024), B16×96 (1536), B16×128 (2048), B32×32 (1024), B32×64 (2048), B32×128 (4096), plus existing tail and reordered-slot cases. Unmeasured cases are predictions, not claimed failures. Preserve all-device exact checks rather than substituting PCC.

## Minimal fixes preserving solo behavior

- **Established fallback:** retain `max_rows=1024`; four B8 groups already pass the B32 control. This preserves logical batch and context and does not require scheduler restrictions.
- **Preferred wider experiment:** chunk only named full-hidden layernorms along the batch axis into <=1024 physical rows and concatenate their outputs in DRAM. Keep the established width partition/scaling and all solo behavior. This is a proposed intervention, not an accepted fix; warmed allocation/trace behavior and downstream exactness must be revalidated.
- **Do not merely extend the sharded guard to4096.** Existing geometry would yield `[512,672]` shards on8×8 cores (336 tiles/core). One full FP32 shard scratch is 1,376,256 bytes; multiple such scratch buffers plus a 688,128-byte BF16 shard exceed the intended per-core budget. Do not couple norm selection to forcing `prefill_l1=True` either.
- **Do not make all solos interleaved or change precision/tolerances to hide the mismatch.** That changes the established numerical reference. A new shared kernel that reproduces the sharded arithmetic is a larger alternative only if bounded norm slicing cannot meet performance needs.

## Report review

Rechecked the headline against both branches and their lowered numeric code. The direct finding is the caller's change of numerical contract at1024 rows. No claim is made that the sharded buffers are BF16, that a specific interleaved allocator variant ran, that all cache elements differ, or that the entire9.15625 error is explained. Batch/page mapping and first-projection orientation were checked as competing explanations and did not yield a concrete contradiction. The larger-batch norm scratch calculation rules out presenting threshold removal as a trivial safe fix. Hardware confirmation of the first divergent intermediate remains the decisive next step.
