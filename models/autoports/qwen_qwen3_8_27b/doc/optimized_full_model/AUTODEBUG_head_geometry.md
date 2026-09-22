# AutoDebug: LM-head reader geometry

2026-09-13. Source-only AutoFix hypothesis audit; the coordinator owns all
hardware experiments. No implementation changes by this investigator.

## Finding

The original probe hard-codes DRAM-bank width rounding for **two** readers while
sweeping `num_workers_per_dram_bank`. For one reader, the final 4736-column chunk
has a physical 20-tile bank row but the native reader derives a 19-tile row and
reads it contiguously. This is a concrete geometry incompatibility in the
probe, not evidence that one-reader matmul inherently loses numerical accuracy.
The compatible-padding hardware control is pending at report creation.

`head_8192_reader1.log` and `head_8192_reader1_active.log` both fail with active-row
PCC `0.7906657457351685`, while local argmax agrees. Restricting comparison to
the active row did not remove the failure. Selected readers2
`head_baseline_fixed.log` passes all four ranks at PCC >=0.99999988.

## Geometry and source chain

The captured head activation is BF16 `[1,1,32,5120]` physically, with one active
row, width-sharded over eight storage cores with `[32,640]` shards. This comes
after the model's 40-core RMSNorm boundary (`tt/model.py:173–187`). Both reader
counts use K=160 tiles, `in0_block_w=10`, and two K blocks per input shard.
These satisfy `validate_matmul_dram_sharded_config` at
`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:1303`.

In `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`:

- Lines 131–149 read physical bank width and validate divisibility by reader count.
- Lines 161–169 derive reader width as `ceil(N_tiles / num_workers)` and require
  `bank_width == readers * reader_width` only when readers >1.
- `per_core_N` is passed as `per_core_N_storage` (lines 63 and 1054 onward).
  It controls output storage, independently of the derived compute/reader width.
  Output specs at `matmul_device_operation.cpp:2510–2553` likewise derive output
  shard count/width from this storage parameter. Do not divide `per_core_N` by
  reader count as a supposed fix.
- Lines 375–376 enable `SPLIT_DRAM_BANK` only when readers >1.

The split path in `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:131–146`
uses `bank_row_stride_tiles`. The single-reader path at lines 148–160 instead
advances one contiguous address by each page read; it assumes its computed row
width matches the physical bank row.

| Chunk N | N tiles | Readers | Original bank tiles | Derived reader tiles | Required bank tiles |
| --- | --- | --- | --- | --- | --- |
| 8192 | 256 | 1 | 32 | 32 | 32 |
| 4736 | 148 | 1 | 20 | 19 | **19** |
| 8192 | 256 | 2 | 32 | 16 | 32 |
| 4736 | 148 | 2 | 20 | 10 | 20 |
| 16384 | 512 | 3 | 64 | 22 | **66** |
| 12928 | 404 | 3 | 52 | 17 | **51** |

For the failing one-reader tail, the second conceptual K row starts at tile19
instead of physical tile20. This also shifts later block starts: a block10
advances190 tiles instead of200. Full8192 chunks have matching widths and are
predicted to pass separately. Aggregate PCC alone has not localized the error
to that tail, so this prediction should be checked if compatible padding fails.

The compatible bank-width formula for these shapes is:

```python
quantum = readers * 32
bank_width = ceil(chunk_width / (banks * quantum)) * quantum
```

Keep logical N and all precision/compute settings unchanged. One-reader tail
bank width becomes608 columns, versus640 originally. The probe derives its
output-storage `per_core_N` from that bank width, giving19 versus20.

Preserving an additional64-column factor with `lcm(64, readers*32)` is too much
padding for some three-reader tails. For chunk16384, it fixes the full chunk to
66 bank tiles but rounds tail12928 to54, violating `54 != 3*17`. The coordinator
reported this exact tail assertion during the source audit. Use the exact
reader quantum; changing divisibility alone is insufficient.

## Other hypotheses

**Reader count omitted from cache key: refuted by source.**
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` is an aggregate with
`num_workers_per_dram_bank` at `matmul_program_config_types.hpp:80`.
`MatmulParams` contains the complete config.
`MatmulDeviceOperation::compute_descriptor_program_hash` hashes full attributes
at `matmul_device_operation.cpp:2738`; aggregate reflection hashes every field
at `tt_stl/tt_stl/reflection.hpp:1423`. There is no custom hash omitting reader
count. The one-reader failure also compiled new kernels on its first attempt.

**Recurrence of stage6 oversized NoC read: not supported by this source path.**
The stage6 repair changes split-reader row reads to use actual constexpr byte
size as the NoC template bound. Single-reader code instead reads pages sized
by `get_max_page_size_and_num_pages` (`matmul_utilities.cpp:357`), bounded by
the architecture's NoC maximum. The repaired split path is still present.
Stage6's wide BF8 test covers N8192/K5120 and readers1/2/3, but uses block2 and
reader-dependent padding; it does not exercise this incorrectly padded N4736
tail. Its success is consistent with the present diagnosis.

## Focused verify/refute experiment

1. Coordinator reruns `probe_lm_head_geometry.py --chunk 8192 --block 10
   --readers 1` with exact reader-dependent bank padding, same captured real
   activation, weights, precision, and active-row gate. This control is underway.
2. If it passes PCC>=0.999 and argmax on every rank, retain the probe geometry
   correction and record the original reader1 result as invalid geometry.
   It does not justify rejecting readers1 performance before measuring it.
3. If it still fails, report per-chunk/per-rank PCC before concatenation and
   log logical/padded shapes, bank shard width, input/output shard grids and
   program fields. The first prediction is seven full chunks pass and tail
   fails under old padding; new padding should eliminate that tail mismatch.
   A stricter one-variable control can change only tail bank width20→19 while
   leaving output-storage `per_core_N=20`, since native write-back supports
   distinct reader/storage widths.
4. Rerun readers3 using the same exact-padding formula. This is a geometry
   validity/control test; no correctness or performance outcome is claimed yet.

No native kernel repair, precision increase, cache bypass, or relaxed correctness
gate is warranted by current evidence. Device verification remains coordinator
work; this report's numerical geometry table was checked with host-only integer
arithmetic.

## Verified AutoFix control

The coordinator's `head_8192_reader1_recovered.json` now verifies the compatible
one-reader geometry: tail `per_core_N=19`, active-row PCC0.99999988–1.0 and exact
local greedy outputs on all four ranks. The intervening model-initialization
stall was recovered separately and occurred before candidate execution.
This passing geometry-only control verifies the padding cause; no kernel or
precision change was required.

The corrected reader1 component costs1150.478us per traced replay versus
999.725us for8192/block10/readers2 and949.427us for16384/block5/readers2.
Reader1 is therefore rejected on measured performance after fixing its input
contract, not on the earlier invalid-geometry PCC result.
