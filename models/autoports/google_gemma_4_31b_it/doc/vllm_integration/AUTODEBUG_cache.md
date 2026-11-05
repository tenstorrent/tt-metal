# AutoDebug: stage09 hybrid KV cache startup failure

## Findings

**The observed startup failure is an adapter contract mismatch.** vLLM deliberately normalizes the global-attention scheduler block from 32 to 64 tokens to equalize hybrid page bytes. The pinned plugin forwards that normalized shape correctly. The failing adapter's `allocate_kv_cache_per_layer()` assumes that scheduler blocks are already the decoder's physical 32-token pages and rejects `(8449, 1, 64, 512)`.

**A second, independently proven defect in that snapshot is that `tensor_idx` is discarded.** It identifies shared storage across hybrid groups. Allocating a separate pool for every layer loses the physical sharing assumed by the planner. This is not what raises the observed exception, but must be addressed in a complete fix that preserves hybrid memory accounting.

The recommended adapter-only correction is to retain the current scheduler specs, allocate one K/V backing pair per distinct `tensor_idx`, expose physical page32 views, and expand each global scheduler block ID `b` to physical IDs `[2*b, 2*b+1]`. Keep local IDs unchanged. Trim the plugin's known right padding to each layer's native scheduler-table width before expansion. This keeps the stock plugin, vLLM block ownership, hybrid groups, chunked prefill, and the advertised **262144-token context**.

This investigation performed no hardware commands, model execution, source changes, plugin changes, or network access. Only this report was written. No removed or stock Gemma implementation was read or copied. The investigation was not relaunched.

## Evidence and revision scope

- Checkout HEAD: `9df04b7df6909d3c5e5e8e9b286b539855df7af5`. The adapter is an untracked working-tree file, so HEAD alone does not identify it. Its SHA256 was `53ebac31dbbdb21811f2bb5f6beec3fc73beb197055e285e49c417d4b37e29fe`.
- Pinned plugin: `/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09`, clean worktree at `ca7a18246e4111629ed6cb4a7d1192002749fccf`.
- Installed vLLM: the log identifies `0.31.0`. Inspected the actual installed `python_env/lib/python3.10/site-packages/vllm/v1/core/kv_cache_utils.py` and `v1/kv_cache_interface.py`; their SHA256 values were respectively `1b20c836aece4982dd87f7e06e14f5ae07a1fc2636101074cfad11584b1a0dcf` and `99ca1d725c949b216d6911fb846c6f3333873fe79852c579a8505fdfcd447e47`.
- Read startup contracts `bringup/run_contract.json` and `bringup/agentic_evals/inputs.json`. This diagnosis leaves the pinned HF revision, sampling decisions, and plugin pin unchanged.

**Concurrent edit discovered during final review:** another workspace actor changed the adapter while this inspection was in progress (file mtime `2026-11-05 19:13:13 UTC`, observed SHA256 `fe3e5d245d437605221be239f58527f5e6de41e17cef80582e8b6cb8a8c59ea2`). That version contains `cache_storage[tensor_idx]`, physical page factors, `ttnn.experimental.view`, and `_physical_pages()`. The findings, quoted line numbers, and failing AST checks below describe the original hashed snapshot that matches the startup log, not a claim that those defects remain in the concurrent revision. The replacement's hardware correctness and readiness have not been validated here. Its presence does not invalidate the demonstrated failure chain.

The original adapter boundary read:

```python
shape, _, tensor_idx = per_layer_specs[index]
if tuple(shape[1:]) != (layer.nkv, 32, layer.head_dim):
    raise ValueError(f"KV specification does not match layer{index}: {shape}")
cache.append(layer.allocate_cache(num_blocks=shape[0]))
```

Direct observations from [server.log](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/readiness_vllm/reduced_server/server.log:15):

| Observation | Location |
|---|---|
| `block_size=32`, context262144, batch1, pinned model/tokenizer revision, authorized `top_k=32` | line15 |
| Chunked prefill budget2048 | lines31,47 |
| Actual registered class is this run's autoport, reduced layers `[0,5]`, BFP8 KV | line77 |
| LBNHC layout, 8449 blocks, reported483487 tokens / 1.84x concurrency | lines78–80 |
| Plugin calls the adapter allocator; adapter line96 rejects global layer5 `(8449,1,64,512)` | lines103–111 |

The following sections derive these values from source. They are not inferred merely from the log labels.

## 1. Complete sizing chain: why 32 becomes 64

The adapter's [get_kv_cache_spec](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/tt/generator_vllm.py:68) supplies **unsharded** BF16 accounting specs, all initially with block32. The pinned geometry is explicitly checked in [multichip_decoder.py](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/tt/multichip_decoder.py:62): sliding KV heads16/head256, global KV heads4/head512, TP4, sliding window1024. This gives:

| Attention | Initial BF16 K+V bytes per scheduler block | Normalized scheduler block | Plugin shape per device |
|---|---:|---:|---|
| Sliding | `2 * 32 * 16 * 256 * 2 = 524288` | 32 | `(N,4,32,256)` |
| Global | `2 * 32 * 4 * 512 * 2 = 262144` | 64 | `(N,1,64,512)` |

[AttentionSpec.page_size_bytes](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/python_env/lib/python3.10/site-packages/vllm/v1/kv_cache_interface.py:520) accounts for both K and V. [unify_kv_cache_spec_page_size](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/python_env/lib/python3.10/site-packages/vllm/v1/core/kv_cache_utils.py:1383), particularly1428–1432, multiplies the smaller spec's block size by the integer page-byte ratio, here2. CLI `--block-size 32` does not prohibit this subsequent per-group normalization.

The plugin explicitly uses each normalized group's block size when constructing input block tables ([model_runner.py:660–704](/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09/src/vllm_tt_plugin/model_runner.py:660)). `_kv_cache_shape()` divides KV heads by the internal TT mesh width and preserves that block size ([lines768–780](/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09/src/vllm_tt_plugin/model_runner.py:768)). It therefore correctly produces the failing shape.

The block count is separately explained by [worker.py:796–833](/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09/src/vllm_tt_plugin/worker.py:796):

```text
max_tokens_all_users                     = 262144
one output-block reservation, batch1     =     32
sliding heuristic: 1024 * batch1 * 8      =   8192
ceil((262144 + 32 + 8192) / CLI_block32)  =   8449
```

The admission floor does not increase that count for the current specs. The 60-layer 5:1 sliding/global pattern forms five sliding groups and one global group, each containing10 layers. With normalized global64 and a2048-token in-flight budget, the engine's maximum per-request demand is:

```text
global: 262144 / 64                         = 4096 blocks
each sliding: ceil((1024 - 1 + 2048)/32) + 1 =   97 blocks
total: 4096 + 5*97                          = 4581 blocks
8449 / 4581                                = 1.844357127...
int(262144 * 8449 / 4581)                    = 483487 tokens
```

This exactly matches the log's capacity and concurrency, as well as its exception shape. The extra-retention distinction between the worker's admission floor and the engine is discussed below; neither calculation changes8449 here.

The earliest incompatible assumption is the adapter boundary, not the final validator's wording: [adapter lines94–97](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/tt/generator_vllm.py:94) assume a scheduler block is a physical page32. The physical decoder requirement is real: [OptimizedDecoder.allocate_cache](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/tt/optimized_decoder.py:212) rejects other page sizes, and [generator._validate_page_maps](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/tt/generator.py:374) independently requires page32. Removing one guard cannot fix this.

## 2. `tensor_idx` is a physical sharing contract

The plugin's [MODEL_SETUP.md](/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09/docs/MODEL_SETUP.md:119) explicitly specifies `(shape,dtype,tensor_idx)` and says equal indices share a buffer. The [builder](/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09/src/vllm_tt_plugin/model_runner.py:856) derives the index from `offset + position * layer_stride` (lines923–949). Upstream [KVCacheTensor](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/python_env/lib/python3.10/site-packages/vllm/v1/kv_cache_interface.py:1417) defines overlapping group allocations: each block ID belongs to one group at a time. These are shared physical slots, **not shared attention values between layers**.

For this pattern, [group construction](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/python_env/lib/python3.10/site-packages/vllm/v1/core/kv_cache_utils.py:1558) interleaves the50 sliding layers into five groups of10, plus the10 global layers. [Placement](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/python_env/lib/python3.10/site-packages/vllm/v1/core/kv_cache_utils.py:1775) overlays those six groups from byte0. The plugin consequently emits10 distinct storage indices: for this exact periodic pattern, `tensor_idx = layer_index // 6`. Reduced layers0 and5 both have index0. Implementation must consume the emitted indices, rather than hardcode that pattern.

The failing adapter reads `tensor_idx` and then allocates separately for every layer. The reduced startup actually allocates layer 0 and then fails at layer 5; it does not demonstrate a full-model OOM. Nevertheless, a patch that only expands global shapes while retaining per-layer allocation would allocate 60 equal-sized K/V pairs instead of 10, a factor 6 excess.

With the logged BFP8 KV dtype, each32x32 tile is1088 bytes ([tile_size](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/tt_metal/api/tt-metalium/tt_backend_api_types.hpp:122)). The shared full-model payload would be:

```text
10 arenas * 8449 IDs * 32 tiles per K * 2(K,V) * 1088
= 5883207680 bytes/device = 5.479 GiB/device
```

Independent allocations after page expansion would be32.875 GiB/device for KV alone. These are payload calculations, excluding allocator/bank padding, weights, workspaces, traces, and serving reserves; they are not measured free-memory or fit claims. The planner's torch.BF16 accounting spec is also not a measurement of the selected TT BFP8 storage. The uniform dtype scaling preserves this page-byte ratio, but real device accounting must use actual TT allocations.

## 3. Minimum complete adapter-owned mapping fix

Keep `get_kv_cache_spec()` and the pinned plugin unchanged for the recommended approach. Change allocation and page-table translation together in `generator_vllm.py`:

1. Record each layer's incoming scheduler block size `B`, block count `N`, and storage index. Validate the expected head geometry, supported page multiple, and common physical dtype/byte capacity within each shared arena. For the observed geometry, `B` is32 or64 and `r = B/32` is1 or2.
2. Allocate a K/V backing pair once per distinct `tensor_idx`. Return per-layer physical views of shape `(N*r, heads_per_device, 32, head_dim)`. Retain the backing and view references for the entire trace lifetime. Reuse one view for identical layouts.
3. Translate absolute scheduler-table columns and IDs before handing them to the generator:

   ```text
   physical_table[j*r + k] = scheduler_table[j]*r + k, 0 <= k < r
   sliding: b -> [b]
   global:  b -> [2*b, 2*b+1]
   ```

   Do not alter token positions, rotate a sliding ring, allocate IDs independently, or compact away absolute logical columns. Continue using `page_tables_per_layer` for hybrid groups.
4. Copy translated contents into the already prepared fixed device tables through the existing `refresh_page_tables()` path. Use the same cache object for warmup, prefill, and decode. The existing external-cache path sets `owns_cache=False` ([generator.py:57–72](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/tt/generator.py:57)); the standalone request allocator then exits without assigning pages (lines353–355). This preserves vLLM ownership.

### Storage views must actually alias

Per pool ID and per K or V:

| View | Physical page count | Tiles per physical page | Tiles for scheduler ID `b` |
|---|---:|---:|---|
| Sliding `(N,4,32,256)` | N | `4*1*8 = 32` | `[32*b,32*b+31]` |
| Global `(2*N,1,32,512)` | 2N | `1*1*16 = 16` | pages2b and2b+1 together cover `[32*b,32*b+31]` |

The two views have equal total padded volume and identical tile format. Group-disjoint block IDs therefore occupy disjoint tile spans in their shared arena.

**Do not use ordinary `ttnn.reshape` to implement this alias.** Its metadata-only path requires equal last dimensions ([reshape.cpp:709–722](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp:709));256→512 does not satisfy that condition. A materialized reshape would defeat sharing. The existing [ttnn.experimental.view](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/ttnn/cpp/ttnn/operations/experimental/reshape/view.cpp:12) delegates to the raw tensor view. For tiled device storage, [tensor_ops.cpp:445–467](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/ttnn/core/tensor/tensor_ops.cpp:445) constructs new metadata at the original buffer address and shares its storage. This is intentional reinterpretation of an arena, not a value-preserving reshape of one layer's data. Source supports this implementation route; address identity, allocation size and writes through both views still need device verification.

The lowered consumers support the coordinate proof:

- [Paged fill writer:20–37](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_fill_cache_interleaved.cpp:20) computes physical tile offset as `physical_block * (heads*block_tiles*width_tiles)` plus head/row offsets.
- [Paged decode update reader:138–146](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/reader_paged_fused_update_cache_interleaved_start_id.cpp:138) selects logical column `position/block_size` and uses the same physical stride.
- [Paged SDPA reader helper:53–68](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp:53) consumes the same coordinate system; decode uses this common mapping as well.
- [PreparedPrefill.stage:54–68](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/tt/prefill.py:54) slices fill tables in page32 units and derives attention tables from the same host maps. Translating only decode, only prefill, or only allocation would be incomplete.

For example, scheduler global IDs `[7,11]` must become physical `[14,15,22,23]`. At token32, the physical decoder must still access scheduler block7's second half; blindly copied IDs access11 instead. At token64 it must move to physical22. At the final context token262143, the physical logical column is8191, derived from scheduler column4095 and half1. No context reduction is required.

### Trim known scheduler padding before expansion

The plugin builds global native tables with4096 columns at context262144/block64, but [pads per-layer tables](/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09/src/vllm_tt_plugin/model_runner.py:830) to `min(ceil(context/CLI_block32),N) = 8192`. [input_batch.py:778–808](/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09/src/vllm_tt_plugin/input_batch.py:778) explicitly clears columns beyond the live prefix and right-pads with zero.

Use at most `ceil(context/B)` native columns for each layer before expansion, checking any discarded suffix is the expected zero padding. Expand the global4096-column prefix into8192 physical columns. Expanding all8192 incoming columns produces16384 and fails the adapter's prepared capacity check. Preserve the full8192-column absolute sliding map even though few of its blocks are live.

Null ID0 is reserved by vLLM. For global64 it maps to physical pages0 and1, both inside reserved pool slot0; normal scheduler ID1 starts at physical2. Unused destination tails can remain zero; expanded null entries themselves are `[0,1]`. Do not introduce negative sentinel IDs: the generator rejects them, and the SDPA reader does not implement fill-cache's special `-1` skip semantics. Validate scheduler IDs against `0 <= id < N` before translation.

## 4. Alternative: keep scheduler block32 with padded page bytes

Setting the global `FullAttentionSpec.page_size_padded=524288` makes both spec pages equal in bytes while leaving both scheduler blocks32. This is a supported field in installed upstream vLLM, and the unifier immediately returns already-equal pages. It can remove the initial shape rejection with a small spec edit.

It is **not a complete fix by itself**. The plugin's per-layer triple includes shape, dtype and `tensor_idx`, but not padded page stride. Physical storage and addresses must still reserve a full arena slot for each global ID. With the shared tile arena above, global block32 ID`b` must map to physical page`2*b`, leaving page`2*b+1` unused. A compact `(N,1,32,512)` global view indexed directly by`b` occupies only half the required stride and can overlap another group's valid ID. Separate per-layer allocations avoid that aliasing error but retain the accounting defect.

| Property | Recommended scheduler64→physical32 expansion | Global scheduler32 + padded bytes |
|---|---|---|
| Spec change | None | Global padded page bytes524288 |
| Global physical table mapping | `b -> [2b,2b+1]` | `b -> 2b` |
| Global logical blocks for context262144 | 4096 | 8192 |
| Shared arena per scheduler ID |32 tiles per K/V, fully used by global64 |32 tiles per K/V, half unused by global32 |
| Current batch1 pool after worker floor |8449 |8684 |
| Full shared BFP8 KV payload/device |5.479 GiB |5.632 GiB |
| Approximate reported max-context concurrency |1.844x |1.001x |

Padding does not reduce the context, but increases scheduler demand. Under the pinned worker's admission helper, global8192 plus five sliding98-block demands plus one null block and one output block gives **8684**. Do not force the old8449 pool after changing specs: engine demand alone is8677. The existing worker recalculates the floor automatically ([worker.py:846–870](/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09/src/vllm_tt_plugin/worker.py:846)). Padding adds about2.78% payload at this particular heuristic pool size, while roughly halving its reported full-context concurrency. These figures are for batch1 and the observed2048 prefill budget, not the entire serving batch matrix.

Expansion is preferable here because it preserves the existing planner's capacity and grouping, uses every global arena slot, and requires no new spec convention. Both approaches need deliberate sharing and physical address translation; a one-line padding workaround does not satisfy the full contract.

## 5. Static checks performed and report adjudication

Two isolated investigations independently followed planner/accounting and decoder/mapping. Host-only checks used AST-extracted functions/classes and fake device-facing objects; none imported TTNN or executed hardware operations. The adapter check used `python_env/bin/python` and CPU PyTorch. The independent planner check used only Python's standard library.

- Extracting the actual adapter class and calling its allocator with the derived specs reproduced exactly `KV specification does not match layer5: (8449, 1, 64, 512)`, after the mocked layer0 allocation of8449 blocks.
- Calling the failing snapshot's `_pages()` with padded global IDs `[7,11,0,...]` returned `[7,11,0,0]`, proving that accepting the larger shape without table translation is insufficient.
- AST extraction of upstream spec/unification/grouping/layout logic and the plugin's per-layer builder reproduced six groups of10, ten shared storage indices, both reduced layers' index0, both incoming shapes, and483487 reported capacity tokens. Unused model-specific branches were stubbed; this was not a full engine run.
- Integer coordinate checks covered scheduler IDs0,1,37,8448 and positions0,31,32,63,64,262143. The two global page32 spans exactly match one sliding-sized pool slot, and the maximum expanded ID16897 fits capacity16898.
- The native-width versus padded-width check produced8192 versus16384 physical columns, confirming the normalization needed before expansion.

After drafting, headline claims were checked against the exception path, upstream calculations, plugin transfer, physical allocator, and lowered readers/writers. The observed exception is deterministic host validation, before generator construction/warmup in this allocator. Prior layer0 initialization may have enqueued device work, but no asynchronous-completion hypothesis is needed to explain the Python error. No claim is made that later warmup, tracing, full-model memory fit, or long-context execution has passed.

## Other potential issues and validation gaps

- **Extra-retention mismatch, not this startup cause.** The adapter requests `extra_retained_tokens=32`. Upstream [kv_cache_utils.py:2738–2743](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/python_env/lib/python3.10/site-packages/vllm/v1/core/kv_cache_utils.py:2738) unconditionally replaces it with `max(0,num_prefill_lookahead_tokens-1)`, zero for this non-speculative run. The plugin's [worker helper:710–719](/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09/src/vllm_tt_plugin/worker.py:710) replaces it only when nonzero, so its floor counts98 sliding blocks versus the engine's97. This explains the five-block floor discrepancy. It neither generates block64 nor causes the observed rejection. Do not rely on the adapter's extra32 surviving engine planning; a claim of resulting attention corruption would require a separate proof of a necessary retained page.
- **Existing adapter test bypasses the failing preparation.** [check_vllm_adapter.py:24–28](/home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/tests/check_vllm_adapter.py:24) synthesizes page32 specs and distinct tensor indices, and uses context4096. It cannot validate normalized block64, sharing, or the padded native table width encountered by the server. Its result must not be treated as evidence that this production boundary works.
- **Physical memory fit remains unverified.** The plugin supplies a synthetic block budget, not a memory profile. Preserve context262144 while measuring actual allocations; this inspection establishes no evidence-backed hardware reason to reduce it. Mixed precision across layers sharing an arena would require explicit byte-layout handling or a checked rejection; the observed KV dtype is uniform BFP8.

## Follow-up validation for the implementing agent

1. Add host coverage that passes this adapter's specs through the installed planner and pinned plugin builder, rather than manufacturing page32 specs. Check shared index counts, native block sizes, padded table widths, bounds and null IDs for batches1/8/16/32.
2. Check mapping under noncontiguous IDs, group-disjoint ownership, recycled IDs, partial batch rows, sliding eviction, and token boundaries31/32/63/64 and262143. Verify discarded global-table suffixes are only padding.
3. On the device, allocate the shared pair once, assert aliases retain its addresses and capacity, and write recognizable data through local/global views for disjoint IDs. Read it back after synchronization. Verify no additional KV allocation/materialized reshape occurs during refresh or trace replay.
4. Repeat the reduced real-plugin server startup, then mixed chunked prefill/decode and full-model long-context validation with all required batches. Use allocator evidence for the memory ledger. Preserve the stock plugin and authorized sampling recipe; nothing in this cache failure calls for PC-01, disabled chunking, or a scheduler/context restriction.
