# Compact full-attention decode path

Source-only, 2026-11-12. Prototype: `candidates/compact_attention_decode.py`, syntax checked only. No hardware run or main implementation edits.

## Proposed path

Fold decode input `[B,1,5120] -> [1,B,5120]` **before** projection. Existing `_linear` treats B token rows as decode when B<=32, so the projection stays compact. Reshape packed `[1,B,8192] -> [1,1,B,8192]` and invoke `nlp_create_qkv_heads_decode(num_heads=24,num_kv_heads=4)`. It returns Q `[1,B,24,256]`, K/V `[1,B,4,256]`, with the **head** axis padded to32. None has a per-head padded sequence axis. At B32 Q storage drops from 12 MiB to 0.5 MiB; K storage drops from 2 MiB to0.5 MiB. These are tensor-size calculations, not timings.

The creator's documentation is stale about requiring sharded input/B32: its current validator accepts interleaved input and logical B<=32, and its factory has an explicit interleaved path. Interleaved input always selects overlapping Q/K/V grids. Specify one user per core on the same prefix-B grid used for V cache update; output shard shape is `[32,256]`. Input width8192 infers head_dim256 from `(24+2*4)` exactly. Logical B is retained by `compute_output_specs`, independently of physical tile padding.

## RMSNorm: direct height-sharded call is illegal; compact interleaved is legal

`layernorm_device_operation.cpp:223-230` explicitly rejects HEIGHT_SHARDED input for RMSNorm. There is no program-config switch that makes the native splitter output directly legal. The prototype converts compact Q/K to interleaved, applies the **existing** `rms_norm` with current FP32 affine weights `(1 + HF weight)`, epsilon and compute config, and retains `[1,B,H,256]` throughout. It normalizes each head's 256 channels, not the combined head width. This preserves the current mathematical boundary without 24-way sequence padding.

A subsequent lower-transfer candidate can reshard to BLOCK_SHARDED with one width core and several height cores, e.g. B32: a 1x8 rectangle, shard `[128,256]`, `block_h=4,block_w=8,subblock_w=4`; then use the sharded RMSNorm program and reshard for rotary. This satisfies current validator width/height decomposition. A plain relabel of the B-core HEIGHT grid as BLOCK on an8x4 rectangle is **not** legal: native BLOCK decomposition divides width across x, while the old height shards each own full256. The compact interleaved candidate should be measured first; sharded adaptation must preserve FP32 affine and compare real-weight PCC.

## Partial HF rotary and mixed positions

Normalize all256 channels first; rotate only prefix64; concatenate untouched tail192 afterward. Use **HF** rotate-half64, not Llama interleaved-pair rotary256. The existing per-user cos/sin `[B,1,1,64]` permutes to `[1,B,1,64]`, retaining distinct rows for mixed positions. It is not replaced by the first user's position.

`rotary_embedding_hf` decode explicitly supports input `[1,B,H,64]`, HEIGHT_SHARDED, and trig `[1,B,1,64]` on exactly the same grid/orientation. For one user per core, both shard heights are32 and widths64. The factory creates programs on the **complete grid bounding rectangle**, not a sparse CoreRangeSet; a prefix-B irregular grid is unsafe even if the current validator accepts it.

The prototype selects the largest rectangular core count n that divides B and fits the device grid, then shards both prefix and trig with height `(B/n)*32`. The factory supports `batch_per_core=B/n` and repeats each user's trig over its own padded head tile. B32 uses32 cores; B1 uses1; an awkward prime B13 on an11x10 device uses one core with13 users. This is legal without increasing public/logical batch, and should be measured for awkward batches. B26 can use2 cores with13 users each. No public batch divisibility restriction is introduced. A faster prime-batch option could internally pad users to a supported rectangle and crop before cache/SDPA, but that introduces extra ownership/position work and is not in the initial prototype.

## Cache and attention

Keep V directly on the native split output grid (prefix B cores). After norm/rotary K is resharded to the next B cores, on the same8-wide row-major enumeration. `paged_fused_update_cache` requires K/V core grids to be disjoint and equal sized (`paged_fused_update_cache_device_operation.cpp:237-246`), and accepts BF16/FLOAT32 input even for a lower-precision destination cache. Both are `[1,B,4,256]`, shard `[32,256]`. The prototype keeps page table and per-user positions unchanged. Do **not** try using native `overlap_qk_coregrid=False` on interleaved input: the wrapper forces overlap true, and for sharded input the disjoint mode also requires input shard width dividing head_dim.

Q stays compact and feeds paged decode SDPA directly; no Q batch/head permutation remains. Preserve current cache dtype, scale256**-0.5, key-chunk selection and FP32 compute config. With4 KV heads, SDPA's GQA validation currently forbids sharded output (`sdpa_decode_device_operation.cpp:545-547`), so the initial candidate requests/defaults interleaved output and reshards its compact `[1,B,padded_heads,256]` output once for `nlp_concat_heads_decode`.

## Concat and gate order

`nlp_concat_heads_decode(num_heads=24)` requires one height shard per user and emits WIDTH_SHARDED `[1,1,32,6144]` even when B<32. Its native packing discards padded heads. Pass full device `sub_core_grids` so arbitrary prefix-B grids use the subcoregrid factory legally. Convert to interleaved, **crop the batch axis to logical B**, and reshape `[1,B,6144]` before sigmoid-gating or output projection.

Gate is still the separately extracted HF q-projection gate in head-major channel order, sigmoid applied elementwise after attention concat and before `o_proj`. Full packed QKVG and split attention policy branches remain represented. The output projection runs on compact B rows and only the final result reshapes to `[B,1,5120]` for residual addition.

## Validation needed

Compare B1/B8/B32 and awkward B13/B26, including distinct per-user positions, nonidentity page tables, short/long cache histories, identical-row users and changed trace inputs. Compare normalized Q/K, rotated prefix/tail, cache pages, output before/after gate, and final layer PCC. Require deterministic repeated replay and original cache/state identities. Benchmark each compact norm/rotary/concat transfer against current path, then selected full decoder; run watcher separately and allocation-tracked replay. The helper is a candidate, not a correctness or performance claim.
