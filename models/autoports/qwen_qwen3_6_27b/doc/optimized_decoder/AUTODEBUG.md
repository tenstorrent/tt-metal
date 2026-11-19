# AutoDebug: DRAM weight stride mismatch corrupts packed GDN prefill

## Finding

**The failing decoder stores `packed_gdn` with 66 tiles per DRAM-bank row, but the 2D prefill matmul reads it with a 65-tile row stride and 65-tile bank boundaries.** This changes the weight matrix being multiplied, beginning in the first K block. It is a concrete layout/addressing mismatch, not an inference from the size of the PCC drop.

The smallest repair for the reported configuration is decoder-local: explicitly zero-extend `packed_gdn` from `[5120, 16512]` to `[5120, 16896]` before DRAM width sharding. Keep its bank shard `[5120, 2112]`, two readers, K block four, and prefill grid `[11, 10]`. Derive program geometry from the extended tensor width and retain the original projection width for consumers. No shared matmul changes are needed for this repair.

Confidence is high in the source-level addressing diagnosis. The final repaired decoder's full PCC/state/trace gates still require hardware validation; this investigation did not run hardware or modify source.

## Scope and revision correspondence

Investigation scope is the stage-3 single-device decoder and its dense matmul consumers. Checkout HEAD is `4c79322a4c142187308864977f00cbe5012c0352`.

The decoder and `run_decoder.py` hashes initially matched `dram2_0.json` exactly:

- Decoder SHA256: `54d2cc9a15a9e4a9e15925cbaff225eb84002bec7939c682ea44c352b46319a8`.
- Harness SHA256: `1476f3cf1bc26b5b3b173564a495f9711d0564931faa0d5ba6dcda8379130f98`.
- Projection-search harness initially inspected: `268d12c6f25fb2f80d7ab7b9021528ecd7df0de98723f63a87c4b2d33e7ac258`, matching its artifact; that live file also subsequently changed. Search-harness line references describe the initially inspected code.

The live `tt/optimized_decoder.py` changed externally during inspection. To avoid attributing the old failure to newer edits, **all `D:<line>` references below refer to the artifact-matching archived decoder**, [source_archives/54d2cc9a15a9e4a9e15925cbaff225eb84002bec7939c682ea44c352b46319a8.py](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/source_archives/54d2cc9a15a9e4a9e15925cbaff225eb84002bec7939c682ea44c352b46319a8.py). That archive's hash was verified. The later live edits and newly added policy combinations are not the source revision diagnosed here. Baseline BFP8 evidence uses the separately preserved `f5a90408...` decoder archive; its diff adds the DRAM conversion and projection dispatch but does not change GDN packing or recurrence.

For compact source references, the following names denote repository-relative files:

| Name | File |
| --- | --- |
| `F2` | `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp` |
| `R2` | `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp` |
| `FD` | `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp` |
| `RD` | `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` |
| `V` | `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp` |
| `H` | `models/autoports/qwen_qwen3_6_27b/tests/run_decoder.py` |

## Direct observations

These are recorded results from existing artifacts, not executions performed by this investigator.

| Artifact / control | Observed result | Limit |
| --- | --- | --- |
| `bfp8_hifi2_0.json`, layer 0, length 512 | HF prefill PCC `0.9999076513`, decode `0.9999888950`; functional-control state PCC `0.9999581177`, `0.9999145459` | Earlier archived decoder; interleaved weights and earlier dispatch |
| `dram2_0.json` / `.log`, layer 0, batch 1, length 512 | HF prefill `0.5764608413`, traced decode `0.5164517993`; eager functional-control prefill `0.5764815108`, decode `0.5159121151` | Exact failing decoder/harness hashes matched |
| Same failing run | State PCC `0.3796763281`, `-0.0078889284`; prefill and repeated traced outputs bit-identical | State is compared **after the eager decode**, not immediately after prefill (`H:144–158`) |
| `search_hifi2_0.json` | All five exact decode geometries used by integration—8 input/storage cores, 2 readers, block 4—pass isolated projections, each PCC above `0.99999` | Batch 32, captured zero-initial-state decode activations; not a 512-row 2D prefill test |
| Additional `shard_probe.json` / `.log` | `packed_gdn`: original 512-row projection PCC `0.0289911245`; explicit width 16896 gives `0.9999945430`. Other four layer-0 projections already give about `0.9999945` | Supplementary artifact has no embedded provenance/source hash; qualified below |

The complete projection search has 234 rows: 98 successful rows, all marked correct, and 136 rows with errors. Per projection, successful/error counts are packed GDN `20/34`, linear-attention output `17/10`, MLP down `17/28`, MLP gate `22/32`, MLP up `22/32`. There is no recorded successfully completed low-PCC projection in this search. This does not turn rejected configurations into supported ones.

The exact integrated decode candidates report PCCs `0.9999941944` (packed GDN), `0.9999981329` (attention output), `0.9999980549` (MLP down), `0.9999950205` (MLP gate), and `0.9999947851` (MLP up). Search code captures activations from an interleaved-weight decoder and compares to `host.float() @ read(weight).float()` (`tests/run_projection_search.py:55–99`), then independently constructs the DRAM-sharded decode call (`:117–155`). It does not test the prefill reader.

The supplementary probe source initially inspected was `tests/run_shard_probe.py`, SHA256 `fbdf2c5f9f91d3fe0416cee17e06a7ee8d7858f8c75f53d22af1d37175afd92e`; it too was subsequently edited externally. In that inspected version, lines 19–34 select the reported sharding/grid policy, compare a 512-row `_linear` call against `x @ read(weight)`, then re-upload a zero-extended matrix with the same shard config and crop its result. Its key-labelled records match the JSON/log. The changed width also changes derived per-core N/subblock geometry, so this is not a fixed-program-geometry A/B test. The artifacts do not bind their run to that source hash; this is corroborating evidence, not a provenance-proved reproduction. The source-address contradiction below stands independently.

## Lowered path and first divergent calculation

### Weight preparation and effective tensor

`D:100–109` concatenates QKV (10240 columns), Z (6144), B padded from 48 to 64, and A padded from 48 to 64. After transposition the packed weight is:

```text
K = 5120 = 160 tiles
N = 10240 + 6144 + 64 + 64 = 16512 = 516 tiles
```

`D:175–197` changes the weight memory configuration without explicitly extending tensor N. With eight banks and two readers its formula gives:

```text
stored bank width S = 2 * ceil(516 / (8*2)) = 66 tiles = 2112 elements
total bank capacity = 8 * 66 = 528 tiles = 16896 elements
tensor N remains 516 tiles
```

Eight banks are consistent with the recorded search's `[5120, 2112]` GDN shard and `[5120, 2176]` MLP input-projection shard, the existing Blackhole log, and `tt_metal/soc_descriptors/blackhole_140_arch.yaml:11–21`.

`to_memory_config` selects the sharding op or general copy fallback according to capability/capacity (`ttnn/cpp/ttnn/operations/core/to_memory_config/to_memory_config_op.cpp:340–363`). Neither is a request to extend the logical matrix. Their output-spec constructors preserve logical shape (`operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp:121–133` and `operations/data_movement/copy/device/copy_device_operation.cpp:154–176`, under `ttnn/cpp/ttnn/`). Tile-layout default alignment is the tile shape rather than the bank capacity (`tt_metal/impl/tensor/spec/layout/page_config.cpp:51–53`), so N16512 remains N16512 in the padded tensor shape. Shard allocation capacity must not be substituted for tensor N. The physical mapping advances bank column starts by the actual shard width and places pages at `row * shard_width + column` (`tt_metal/impl/buffers/buffer.cpp:320–348,459–469`).

### Python dispatch, canonicalization, and validation

The 512-token prefill is one 512-row chunk (`D:404–421`). `_linear` selects the 2D config because `b*seq > 32`, uses DRAM-interleaved BF16 A, BFP8 tile-layout DRAM-width-sharded B, BF16 DRAM-interleaved output, HiFi2 with FP32 destination/packer accumulation, no bias, and no operand transpose (`D:55–64,232–275`). Its effective geometry for packed GDN is:

```text
grid = (11,10), Mt = 16, Kt = 160, Nt = 516
per_core_M = ceil(16/10) = 2
per_core_N = ceil(516/11) = 47
out_subblock = (1,1), in0_block_w = 4
active output grid = (11,8), K blocks = 40
```

`matmul/matmul.cpp:259–313,341–350,449–496` preserves this operand order/config; there is no transpose, swap, or hidden reshaping that repairs B's layout. `V:2748–2749` selects `MatmulMultiCoreReuseMcast2DProgramFactory`. Validation permits width-sharded DRAM B (`V:1914–1930`); the width-equality check at `V:1985–2000` applies only to non-DRAM B. It does not reject the 66-versus-65 mismatch.

`F2:1838–1842` derives Nt from B's padded tensor shape. **The earliest divergent calculation is `F2:586–593`:**

```text
per_core_N_storage = ceil(Nt / device.num_dram_channels())
                   = ceil(516/8) = 65 tiles
```

This ignores `B.shard_spec().shape[1] / 32 = 66`. `F2:1315–1320` passes 65 both as the bank-local K-row stride and as the basis of the K-block byte advance. `F2:686–689` selects `IN1_DRAM_WIDTH_SHARDED`; its selected source is `R2`.

### Concrete wrong addresses

Let `r` be the K tile row, `c` the output column tile, and `T` the BFP8 tile size. The contracts are:

```text
correct: bank = floor(c/66), bank-local byte offset = (r*66 + c%66)*T
prefill: bank = floor(c/65), bank-local byte offset = (r*65 + c%65)*T
```

`F2`'s bank walker assigns worker 0 47 tiles from bank 0, then worker 1 18 tiles from bank 0 starting at local column 47 and 29 tiles from bank 1 (`F2:219–275`). The correct split for that second worker is 19 plus 28.

| Requested weight tile `(r,c)` | Correct `(bank, local tile)` | Prefill reads | Consequence |
| --- | --- | --- | --- |
| `(0,0)` | `(0,0)` | `(0,0)` | Initial tile agrees |
| `(1,0)` | `(0,66)` | `(0,65)` | Reads actual weight tile `(0,65)` instead of `(1,0)` |
| `(4,0)` | `(0,264)` | `(0,260)` | Next K4 block starts four tiles early |
| `(0,65)` | `(0,65)` | `(1,0)` | Reads actual global column 66 instead of 65 |

`R2:439–478` uses `BankView` and these bank-local offsets directly, increments each row by `in1_block_w_dram_bytes`, and advances K blocks by `in1_dram_block_size_bytes`. The accessor does not reinterpret these as logical tile indices: its documented contract names allocator bank and bank-local bytes (`tt_metal/hw/inc/api/tensor/tensor_accessor.h:912–918`). Thus generic shard-aware accessor translation cannot compensate.

### Queue and completion ledger

For a full 47-column worker, each of 40 K iterations reserves and fills `4*47 = 188` B tiles, completes its NoC read barrier, multicasts, and publishes 188 tiles (`R2:412–479,539–547`). Compute waits for and consumes the same 188 (`kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp:418–419,589–590`). The first iteration's K rows are read at offsets `0,65,130,195` tiles instead of `0,66,132,198`; the next starts at 260 instead of 264. This is wrong content within a normally counted block, so successful waits and deterministic completion are compatible with corruption. The final worker's single padded output column does not cause the first worker's already-wrong reads.

After padding, the same accounting becomes 40 iterations of `4*48 = 192` B tiles, with the correct 66-tile bank-row stride. No queue imbalance is needed to explain either case.

The harness synchronizes warm/timed prefill and uses blocking trace replay (`H:163–181,219–231`). Eager functional-control comparisons read outputs before tracing (`H:139–158`). Therefore the report is about completed wrong outputs, not just an enqueue-side success or a failure first appearing at teardown.

## Causal chain and contrasts

The first dense projection after input norm is packed GDN. Its incorrect output affects QKV/conv history, Z, beta, and A/g, rather than only unused tail columns (`D:326–375`). Conv history is copied into `state.conv` at `D:357`; Q/K/V and gates feed the recurrent state update at `D:365`. Bad attention then propagates through the residual and MLP (`D:314–324`). This explains a catastrophic full-prefill PCC loss and corrupted persistent state without a new recurrence or MLP bug.

The failing layer is **linear attention**. Its two state PCC entries describe conv and recurrent state, not paged KV. Eager decode consumes the state left by prefill (`H:144–146`), and the traced path snapshots/restores that same bad-prefill state (`H:183–199,219–226`). Inherited state corruption is therefore a sufficient route to poor decode and is consistent with the high isolated dense-decode PCCs. The exact `0.516` decode value and split between inherited-state versus any additional batch-1/composition error cannot be proved by static inspection; no clean-state or reference-state integrated decode control is recorded here.

Why the passing contrasts avoid this mismatch:

- Interleaved BFP8 uses the interleaved reader, avoiding the inferred per-bank row stride entirely. The older baseline is not a perfect one-variable experiment, but the archive diff and isolated probe localize the relevant change.
- The DRAM-sharded decode reader explicitly takes the actual bank shard width (`FD:157–158,1037–1039`) and reads `(block*rows_per_block+row)*bank_row_stride + column_offset` (`RD:140–155`). Here two readers each read 33 of the actual 66 tiles; `FD:188–196` validates that relationship. The reported config does not opt into `use_multicore_pipeline`, whose default is false (`device/config/matmul_program_config_types.hpp:75–89` under the matmul directory). Two readers therefore use this split-bank path; the result is not evidence about the separately selectable multicore pipeline. Its passing search does not validate the distinct prefill reader.
- Other active layer-0 dense weights have stored width equal to the 2D inference. They cannot independently explain the first GDN corruption.

| Projection(s) | K | Original N / tiles | Actual bank tiles, readers=2 | 2D-inferred bank tiles | Recommended tensor N |
| --- | ---: | ---: | ---: | ---: | ---: |
| Packed GDN | 5120 | 16512 / 516 | **66** | **65** | **16896** |
| Linear output / full-attention O | 6144 | 5120 / 160 | 20 | 20 | 5120 |
| MLP gate and up | 5120 | 17408 / 544 | 68 | 68 | 17408 |
| MLP down | 17408 | 5120 / 160 | 20 | 20 | 5120 |
| Full-attention packed QKV | 5120 | 8192 / 256 | 32 | 32 | 8192 |
| Full-attention gate | 5120 | 6144 / 192 | 24 | 24 | 6144 |

The last two rows are source-derived coverage checks, not failing full-attention measurements. This diagnosis predicts the reader-two packed-GDN case. With one reader, GDN's stored width is 65 and matches the inference; this particular mismatch does not explain unrelated one-reader failures. With three readers, any projection rounded beyond `ceil(Nt/8)` needs the same compatibility check.

## Minimal decoder-local repair proposal

For the exact failing policy, pad only packed GDN's **rightmost tail** by 384 columns. Preserve all existing QKV/Z/B/A positions, including the internal B/A padding. Use the same prepared sharded weight in prefill and decode. This does not enlarge its already allocated 66-tile-per-bank storage footprint.

For a general local preparation helper, with B banks, R readers, C activation/output-storage cores:

```python
# Run during loading, on the final packed/transposed host weight.
logical_n = weight.shape[-1]
align_n = 32 * math.lcm(B * R, C)  # B*R suffices for the reported C=8
physical_n = ((logical_n + align_n - 1) // align_n) * align_n
weight = torch.nn.functional.pad(weight, (0, physical_n - logical_n))
# Upload/quantize this tensor, then retain one DRAM WIDTH_SHARDED copy:
# shard shape = (K, physical_n // B)
# decode per_core_N = ceil(physical_n / (32*C))
# prefill per_core_N = ceil(physical_n / (32*gx))
# Keep logical_n separately for the returned activation shape.
```

Using host padding before BFP8 conversion is straightforward and preserves the original quantized data in the existing full tiles. An initialization-only `ttnn.pad` of the interleaved BFP8 weight before `to_memory_config` is also an available local option; do not pad the already sharded weight by changing metadata alone. No host work belongs in forward.

For the reported packed GDN the resulting legal settings are:

```text
tensor weight       [5120,16896], Ntiles=528
DRAM bank shard     [5120,2112], S=66 tiles
decode input shard  [32,640], unchanged, 8 storage cores
decode readers      2, width33 each, 66 == 2*33
decode output       per_core_N66, 8 output-storage cores
prefill grid        [11,10], per_core_M2, per_core_N48, subblock[1,4]
K/block             160/4 = 40 iterations, unchanged
```

Recompute decode `per_core_N` from physical N so the intended eight-core output geometry is retained. Leaving 65 would allocate nine output-storage cores for 528 tiles (`V:3124–3131`); that is a geometry change, **not a proved corruption or validation failure**, because output-storage and reader grids can differ.

For a reusable helper, crop BF16 results to `logical_n` after conversion to DRAM-interleaved output and before batch reshape/downstream consumption. The exact packed-GDN-only repair can also rely on its existing fixed consumer slices: all used regions end by column 16496 (`D:329,358–359,374`), so the added tail is ignored. Explicit crop metadata is preferable when extending the helper to other projections.

No K/input padding is required for these dimensions. K tile counts are 160, 192, and 544; with eight cores their input shard widths are 20, 24, and 68 tiles, all divisible by decode block four and prefill block four. Cropping a preceding padded projection keeps these original K dimensions.

No change to the `[11,10]` grid is required. The existing bank walker can straddle banks, and its narrow-worker clamp is present (`F2:229–247`). Padded GDN uses exactly 11 active columns and eight active rows, subblock width four divides per-core width 48, and K160 is divisible by four. Grid selection remains subject to the actual device extent and performance measurement. Padding only to a multiple of `B*32` is insufficient for two readers: N16640 would leave inferred width65 while reader rounding still requires an actual width66.

## Other Potential Issues and unresolved dimensions

| Candidate | Adjudication / remaining uncertainty |
| --- | --- |
| Independent DRAM decode arithmetic defect | All five exact search geometries pass; integrated decode inherits corrupted state. Batch-1 full composition remains unisolated, so do not claim it is fully cleared. |
| BFP8 quantization, HiFi2, or FP32 accumulation alone | Does not account for the direct row-stride contradiction; earlier BFP8/HiFi2 evidence and the padded probe pass. Numerical quality must still be validated after repair. |
| Narrow worker versus bank width | Previously suspicious shared path already clamps the initial span (`F2:232`); current issue remains even in worker 0. Matching grid width to bank count would not fix its wrong K-row stride. |
| Trace replay/lifetime corruption | Eager and prefill failures precede trace replay, and repetitions are bit-identical. No evidence here requires this additional cause. This does not establish general trace-allocation safety. |
| Corrupted upload / conversion | Decode search and the probe's reference read the stored matrix successfully. Probe comparison is against the read-back matrix, so it localizes a consumer discrepancy; it is not a bitwise upload-integrity audit. |
| Later live decoder edits | Concurrent changes are outside the failing artifact's source hash. The observed local padding direction agrees with this diagnosis, but this report does not certify that in-progress revision or its other policies. |

Existing shared 2D tests miss reader-multiple storage padding: `tests/ttnn/unit_tests/operations/matmul/test_matmul_dram_sharded_multishard.py:125–179` prepares the minimum inferred per-bank width; `test_matmul_2d_config_bitexact.py:15–23,35–54` also prepares minimum bank padding and is Wormhole-only. Neither proves that 2D accepts a larger actual bank width required by two Blackhole readers.

## Review and follow-up validation

Claims were checked against the artifact-matching decoder archive, wrapper dispatch, validation, selected factory, bank walker, reader kernel, bank-address API, and physical page mapping. An independent xhigh investigator reviewed the first draft against those contracts and found no material causal or repair error after corrections. A cheap standard-library calculation expanded all table shapes and original/repaired addresses and checked K divisibility; no TTNN model was imported or executed for this check. No hardware reproduction, reset, test, or benchmark was performed by this investigator.

Before accepting the repair, the implementing session should verify:

1. A direct packed-GDN 512-row projection against the same interleaved quantized weight, plus the unchanged other projection controls. Assert the bank-width equality explicitly in the local preparation path.
2. The original layer-0 length-512 prefill, eager/traced decode, contribution PCC, and conv/recurrent-state gates. Compare state immediately after prefill as well as after decode.
3. Clean/reference-state batch-1 decode and batch-32 decode after padding, distinguishing remaining decode issues from inherited-state corruption.
4. Existing short/irregular-length and batch-2 decoder coverage, and the full-attention layer control; ensure extra output columns never enter consumers. Reader-one and reader-three policies need their own legal-shape controls if retained.

These are proposed follow-ups, not completed validation. The report's primary conclusion is the proved incompatible 66-versus-65 layout contract and its local, one-weight-copy padding repair.
