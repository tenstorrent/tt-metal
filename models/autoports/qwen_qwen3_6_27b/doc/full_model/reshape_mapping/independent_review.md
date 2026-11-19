# Independent reshape mapping review

2026-11-14; CPU/source only. The parent owns implementation and hardware validation. This review changed only documentation artifacts. It is a focused AutoFix review, not the full-model stage review or a device acceptance result.

## Reproducible proof

From the checkout root:

```bash
python models/autoports/qwen_qwen3_6_27b/doc/full_model/reshape_mapping/cpu_mapping_proof.py > models/autoports/qwen_qwen3_6_27b/doc/full_model/reshape_mapping/cpu_mapping_proof.json
```

Result: **PASS,2322 exact comparisons**. The script imports only the standard library and neither imports TTNN nor opens a device. The JSON records the proof script and reviewed native source SHA256 values.

The independent implementation follows native `reshape_map_output_page`: enumerate output elements in tile order, calculate source logical indices and face offsets, group contiguous segments by source tile, sort source tile groups, and pad groups with zero-count entries. It compares the complete original map against the column-zero map with the source page index shifted only for nonzero segments.

Coverage: B1/3/32; logical lengths1,2,15,16,17,31,32,33,34,48,63,64,65,127,161,511,512; widths32/64/1280/5120; packing and unpacking; unique first/middle/last output tile rows and first/second/last tile columns. Standard32x32 tiles with16x16 faces are used. The proof does not execute native code or assert device timing/correctness.

## Concrete allocation owner and histogram attribution

`reshape_tiled_program_factory.cpp` uploads a uint32 segment mapping tensor and moves its owning MeshTensor into `ProgramArtifacts.op_owned_tensors`. `mesh_device_operation_adapter.hpp::SpecSharedVariables` keeps those tensors alive for the program-cache entry's lifetime. The mapping contains one page per output tile in the original implementation. This is retained tensor metadata, not a Python output list or merely a small kernel binary.

The original native mapping geometry independently reproduces the major new buffer sizes in `../prepare_memory_baseline.json` between uniform lengths32 and64:

| Per-bank bytes | Count across matching lengths | Geometry and source call |
| --- | ---: | --- |
|2785280|16|Width5120 unpack, odd lengths33..63; terminal `_norm` returns original user/sequence axes in `multichip_decoder.py:633`|
|2621440|8|Width5120 unpack, lengths34,38,...,62|
|2293760|4|Width5120 unpack, lengths36,44,52,60|
|1114112|16|Width2048 unpack, odd lengths33..63; temporal QKV reshape in `multichip_decoder.py:891`|
|1048576|8|Width2048 unpack, lengths34,38,...,62|
|696320|16|Width1280 unpack, odd lengths33..63; residual result reshape in `flat_prefill.py:148`|
|655360|8|Width1280 unpack, lengths34,38,...,62; other shapes also contribute at this size|

For example, an odd-length width5120 unpack has10240 output tiles and136 padded mapping segments per tile. Its mapping allocation is `10240 *136 *16 /8 =2785280` bytes per bank. Width1280 and2048 scale the page count by1/4 and2/5. Both size and repetition count match the reported histogram. This identifies the major growth owner; it does not assign every buffer or every remaining histogram count.

The JSON includes geometry histograms for every length33..64, plus predicted compressed histograms. A geometry entry alone does not prove public dispatch allocates that mapping: aligned shapes may take a view or another reshape route. In particular, do not count the length64 modeled map as necessarily retained. The unchanged fixed CCL inventory reported by the parent and these exact native mapping matches exclude expanding long-shape CCL as the explanation for the identified buffers.

## Review of the parent's two-file repair

Reviewed the actual diff in `reshape_tiled_program_factory.cpp` and `reader_reshape_tiled.cpp`; source identities are in `cpu_mapping_proof.json`.

The factory selects `mapping_column_stride=W/tile_width` only when input and output logical widths are equal and tile-aligned. It constructs one map for each output tile row by generating the old map at `row*stride`. Other shapes retain stride1 and the original map construction. The reader loads mapping page `out_page/stride`, then adds `out_page%stride` to source tile IDs. Work partitioning, output page numbers, segment offsets, element counts, dtype, tile placement, and op-owned mapping lifetime remain unchanged.

For fixed equal widthW, changing output tile column byc changes the input tile column byc, without changing input logical row or batch. Therefore every source page ID increases byc; source and destination face offsets and segment counts remain identical. Source page grouping and sort order are preserved by adding one constant. The maximum segment count per mapping page is also unchanged, so compressing across columns preserves the mapping FIFO's page size and kernel resource requirements.

The unchanged writer compares raw source IDs only to decide which already-loaded tile to consume. Equality within an output tile is invariant under the constant shift. Its `previous_input_page_idx` resets toUINT_MAX for each output page; therefore adjacent columns sharing the compressed base map still trigger the required page transition. Reader and writer retain matching FIFO consumption counts. Core work ranges may begin or end mid-column group because the reader derives row and column from the absolute output page index.

**Focused source review: no correctness finding in the two-file change for the supported native tiled reshape contract.** Width1-tile reduces to stride1. Unequal or partial widths retain the old behavior. Existing shape validation must continue excluding empty dimensions; valid output tile counts are divisible by the selected width-tile count. The independent proof targets standard tile geometry used by this model; it does not broaden the native operation's supported tile geometries or custom-padding contract.

## Required device checks and limits

The parent should retain forced-native pack/unpack checks for nonaligned lengths, multiple widths/batches, both core groups and ranges beginning mid-column, then replay cache hits with new input addresses and contents. Check BF16/FP32 and supported integer types, plus unequal-width/partial-width cases to cover unchanged stride1 behavior. Verify the compressed metadata allocation reduction and stable cache-hit allocations before full B32 preparation. Do not replace this with cache clearing, skipped shapes, CCL policy changes, or context reduction.

At review time no hardware result is claimed here. The parent's focused device test, all64 capacity retry, allocation-tracked generator checks, and stage acceptance remain separate evidence.
