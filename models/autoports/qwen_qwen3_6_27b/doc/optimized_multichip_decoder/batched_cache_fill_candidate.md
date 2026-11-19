# Native batched prefill K/V fill

A native bulk path already exists: `ttnn.experimental.paged_fill_cache(..., batch_idx_tensor=indices)` accepts `[B,H,T,D]` input and one index per input batch. Two calls fill K and V for the entire batch. This preserves separate BFP8 cache buffers and the existing page layout. No new native kernel or combined K/V allocation is needed.

Implemented isolated files:

- `tt/batched_cache_fill_candidate.py`: `batched_fill_pair`, two native calls, no setup allocations or host readbacks.
- `tests/probe_batched_cache_fill.py`: exact complete-cache reference/oracle, changed-input and changed-page-table traces, BFP8 controls and trace/eager timing.

No production/shared/native files were edited; no Torch/TTNN import or device execution by the preparing agent. CPU mapping checks and source assertions passed; device validation is pending.

## Actual caller and material scope

This checkout has no `_prefill_cache` helper. The identical per-user loop is spelled out in `flat_prefill._attention_flat` (~line86) and inherited `OptimizedDecoder._attention_prefill` (~line1391). Both pad K/V to32 rows, cast to cache dtype, slice the page table at `start_pos//32`, then issue two `paged_fill_cache` calls per user. The QKVG packing and batched prefill RoPE are upstream and remain untouched.

Profile4 full-attention B32S33 prefill contains188 operations:64 cache fills and76 slices. Exactly64 slices immediately precede those64 fills. Summing **rounded report rows**, those128 targeted kernels total about128 us; their preceding operation gaps total about1,636 us (550 before fills and1,086 before their slices). Whole report totals are3,076 us kernels +2,231 us gaps. This demonstrates an eager dispatch opportunity, not a promised1.6ms speedup: gaps include other host/runtime costs and the bulk kernels still take time. The replacement removes64 per-user slices and reduces64 fills to2; shared page-table slicing and the remaining12 slices stay.

## Native source audit

Relevant source under `ttnn/cpp/ttnn/operations/experimental/paged_cache/`:

- `paged_cache_nanobind.cpp:117–159` documents `[input_batch]` ROW_MAJOR INTERLEAVED DRAM `batch_idx_tensor`; scalar fallback requires input_batch1. This is an exposed native API, not a hypothetical batching wrapper.
- `device/fill_cache/paged_fill_cache_device_operation.cpp:33–47` requires cache/input dtype equality and explicitly calls this raw tile copy. BFP8 is supported; **keep the existing prefill cast before this helper**. No quantization precision change is proposed.
- `device/fill_cache/paged_fill_cache_program_factory.cpp:128–189` uses work size `input_batch * num_heads * input_seq_len_tiles`, verifies exactly one physical index element per input batch, and rejects multiuser input without indices. Local H1,D256,block32 matches its geometry contract.
- `device/kernels/dataflow/reader_fill_cache_interleaved.cpp` reads contiguous complete tile pages using the input dtype's actual byte size. The writer decodes `cur_batch=row_id/num_blocks_per_batch`, selects `batch_idx_arr[cur_batch]`, reloads the corresponding page-table row, and writes complete raw tiles after a NoC barrier. Existing -1 page entries are skipped.
- `tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py:1042–1273` already covers batched fill, permuted target users, skip entries and program-cache reuse for H1. Its existing batched fixtures are BF16; the new probe specifically uses this model's BFP8, D256, block32 and nonaligned batch/sequence lengths.

`paged_fused_update_cache` is a **decode** path: input dim0 must be1, it accepts BF16/FP32 update inputs and repacks into cache dtype. It is not a two-buffer prefill fill API. Combining K/V into one call via concatenation would require additional input/cache representation work and is unnecessary for this bounded experiment. Two batched calls already remove the material per-user dispatch cost.

## Layout, ownership and public-start contract

Helper contract is deliberately narrow: prepared K/V `[B,1,T,256]`, T divisible by32, B1–32, BFP8 TILE interleaved DRAM; cache `[physical_blocks,1,32,256]` with existing shape/dtype; original ROW_MAJOR INT32 page table; preallocated ROW_MAJOR INT32/UINT32 index tensor with exactly B physical elements. The probe uses identity or reversed unique target-user indices. Setup for the model uses replicated `arange(B,dtype=int32)`; do not allocate/upload it inside forward or capture. The same tensor can serve both K and V and remain immutable for that execution slot.

**Disjoint destination ownership matters.** The legacy Python loop has serial last-writer behavior if active users alias the same destination physical page; a bulk program can race those writes. The current model allocator creates disjoint per-user page blocks, and physical-page remapping preserves that property. The helper requires `disjoint_write_pages=True` as an explicit caller invariant; it does not read a device page table to invent a proof. If an external table has shared writable pages or its ownership is unknown, retain the ordered loop. Indices must refer to valid table rows; setup owns that range/uniqueness check.

Aligned chunks only enter this helper. `_attention_flat` and inherited `_attention_prefill` already assert `start_pos % _attention_chunk(state)==0`. The existing scheduler walks an unaligned public start token by token using `_attention_decode`, then resumes bulk prefill at the next attention-chunk boundary. Keep that scheduler and fallback unchanged. The helper independently rejects `start_pos%32 !=0`; using `start_pos//32` for a nonaligned public start would overwrite prior cache rows and is forbidden. Tail lengths are padded and copied exactly as the original loop, including zero-filled final-page suffix. Arbitrary public lengths/starts are therefore preserved by the existing dispatch, not by treating partial-page writes as whole-page fills.

Owner-safe production proposal after validation: create indices in instance setup/preparation for each supported active B, or in a caller-owned execution context; introduce a default-off `batched_prefill_cache` policy and one explicit-owner method used by **both** current loop sites. Call it only when the allocator/context certifies disjoint active pages. Clone/resource variants must receive their own context or immutable indices on the same mesh; do not install a closure bound to an original model instance. Retain original loops as fallback. No cache buffers or cache dtype/layout change, no new runtime host synchronization.

## Focused proof and parent commands

CPU mode passed216 combinations of B1/2/3/17/31/32, S1/31/32/33/64/513, aligned starts0/32/512, identity/reversed user mapping. It proves the bulk row decomposition equals the per-user mapping and gives unique tile ownership for disjoint pages. It asserts actual native source/API patterns and hashes those sources into hardware reports. CPU mapping is not hardware correctness.

The device probe initializes nonzero K/V caches, uses distinct rank/user values, casts prepared padded inputs to BFP8 once, and compares both native routes against an independent decoded-value page oracle. It checks **all cache elements**, including untouched prefixes/suffixes, zero tail padding and skip-page retention. Reads compare exact decoded values; this is not a raw-byte dump of BFP8 exponent storage. Native source establishes that neither route repacks tiles.

Every changed step uses new K/V and a new permutation of physical pages. Both implementations have separate caches and share source fixtures. Before capture, all fixture banks/cache/input/table/index buffers exist, every copy shape and both native paths have warmed, and program-cache entry count is recorded. Both traces replay sequential state histories against saved eager results; no new program-cache entry is allowed. All traces release before any owned buffer is freed. Dispatch-inclusive eager timings are measured after trace release; traced timings are reported separately. No allocation exemptions or tracker acknowledgements.

Run only through the parent's serialized device ownership, using fresh output paths:

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_batched_cache_fill --cpu-check

timeout 180s env TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_batched_cache_fill --batches 32 --lengths 33 --starts 0 --output "$D/batched_fill_smoke.json"

timeout 600s env TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_batched_cache_fill --batches 1 3 17 32 --lengths 33 513 --starts 0 32 512 --output "$D/batched_fill_broad.json"

timeout 300s env TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_batched_cache_fill --batches 3 32 --lengths 33 513 --starts 32 --reverse-users --skip-page --output "$D/batched_fill_remap_skip.json"
```

After focused success, parent integration must additionally run the unchanged full-layer output/cache/per-user PCC gates, remapped page-table checks, and **public unaligned starts** (e.g.1,31,33) with lengths crossing the next scheduler boundary. Those public tests exercise retained decode fallback plus newly selected bulk fill; the focused helper intentionally does not claim such an end-to-end scheduler test. Measure full prefill B32S33/S513 and B1 regression controls, retaining identical packing/RoPE policy. Profile only the winning coherent implementation and verify two fills per bulk chunk rather than accepting an API-only microbenchmark as whole-layer evidence.
