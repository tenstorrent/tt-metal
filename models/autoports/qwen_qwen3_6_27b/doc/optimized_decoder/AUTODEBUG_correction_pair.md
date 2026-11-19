# Packed primary/residual attention projections

Source-only AutoFix topology candidate, 2026-11-12. No device run or main-source edit. Candidate correctness/performance is unverified. The separate C32/R4/block1 path remains the measured baseline.

## Existing measured topology

`perf/profile80_bfp8_0/decode_perf_report.csv`, B32:

| Projection | Primary FP32 matmul | Residual FP32 matmul | FP32 add→BF16 | BF16 export |
|---|---:|---:|---:|---:|
| packed GDN input | 193.587 us | 192.935 us | 11.189 us | 4.053 us |
| GDN output | 67.846 us | 67.981 us | 3.844 us | 1.388 us |

The two terms already share one input reshard per projection (1.607/1.492 us). Packing cannot claim to eliminate an input reshard that is already shared. Its opportunities are one matmul dispatch/program prologue instead of two, some shared A work, lower padded output arithmetic, and 2.94% fewer GDN input-weight bytes.

## Candidate and exact weight construction

`candidates/correction_pair_decode.py` is frozen from `default81.py` by `make_correction_pair.py`. Setup concatenates the **existing quantized BFP8** primary and residual tensors before DRAM sharding/padding. It never returns to the original BF16 weights to quantize the pair. Both logical boundaries are tile aligned, so the concat factory copies complete pages including BFP8 exponents. Groups of 16 values cannot straddle a new boundary.

Relevant implementation evidence:

- `concat/device/concat_device_operation.cpp:149-183`: equal layouts/dtypes; tile-padding warning applies only when logical and padded concat dimensions differ. Here K and N are tile aligned and equal to padded dimensions.
- `concat/device/concat_program_factory.cpp:59-74`: TILE pages use the dtype's tile byte size, including the complete BFP8 representation.
- `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:173-196`: each DRAM bank width must divide the reader count, and bank width must equal readers × worker width. The proposed geometries meet both constraints.
- `slice/device/slice_device_operation.cpp:456-458`: TILE slicing uses TensorAccessor for all sharded buffer types. It supports direct L1-sharded input to L1-sharded output; a full FP32 DRAM export is unnecessary.
- `slice/device/slice_program_factory_tile.cpp:39-73`: the slice stages two tiles per worker and uses whole-tile page sizes; both primary/residual column starts are tile aligned.

Default C32/R4/block1 geometry:

| Projection | Logical pair N | Physical pair N | Tiles/bank | Tiles/reader | Separate physical total N |
|---|---:|---:|---:|---:|---:|
| packed GDN input | 33024 | 33792 | 132 | 33 | 34816 |
| GDN output | 10240 | 10240 | 40 | 10 | 10240 |

K remains 5120/6144, input width per C32 core remains 160/192, and K blocking remains one tile. The candidate inherits the selected attention HiFi4, FP32 destination and packer policy. DRAM-sharded factory auto-subblock selection can still change when N changes: separate17-tile workers pad their compute span to20; pair33-tile workers pad to36. Therefore exact weight preservation does **not** establish bit-exact matmul outputs. Real-weight PCC and long recurrent stress remain required.

`correction_pair_verify_setup:true` checks exact dequantized equality of both halves before sharding and again after final DRAM padding/sharding, plus exact zero padding. These diagnostic readbacks happen only during setup and are excluded from runtime/performance claims. Successful checks print `CORRECTION_PAIR_SETUP_VERIFY <key> exact physical_n <N>`.

## Runtime and phase contract

The packed path performs:

1. Existing compact BF16 activation → pair input L1 shards.
2. One native DRAM-sharded matmul with FP32 output.
3. Two TILE slices directly into matching L1-width-sharded primary/residual outputs.
4. Existing FP32 add with BF16 output rounding.
5. One BF16 sharded-to-interleaved export.

The slices are real data movement. For B32 input projection they copy two logical 2,113,536-byte FP32 halves within L1/NoC; output projection halves are 655,360 bytes each. Compared with the separate path, the candidate adds these slice reads/writes and two dispatches. It must win **end-to-end traced decode**, including this cost. A narrower source representation alone is insufficient.

`correction_pair_split:"dram"` is a legal comparison/repair path: slice directly from the packed L1 output into DRAM, then add there. It avoids exporting the unused packed tail but adds DRAM FP32 traffic. Default is `"sharded"`. If L1 slice output derivation or placement fails, inspect the exact kernel/assertion and compare this explicit destination; do not reject packing after the first API error.

Phase propagation is explicit: `decode_forward` passes `decode=True` through `_gdn`/`_gdn_compact`, which passes `decode_pair=True` only to the packed GDN and output projections. Defaults are False. There is no persistent mutable phase flag. `prefill_forward` AST and `_linear_raw` AST are unchanged; prefill S1 and final S1 tails keep the separate weights and original arithmetic. The candidate requires compact GDN when enabled. Both ordinary [B,1,W] and compact [1,B,W] decode are supported.

## Storage and experiment controls

Original terms remain resident for identical prefill. Additional BFP8 storage is:

- Input pair: 183,828,480 bytes = 175.3125 MiB.
- Output pair: 66,846,720 bytes = 63.75 MiB.
- Both: 250,675,200 bytes = 239.0625 MiB, excluding allocator alignment/setup temporaries.

This must be included in context/capacity accounting if selected. The input pair reads 5.3125 MiB fewer weights per token than two separately padded tensors; output packing saves no weight bytes.

Use separate policies to isolate cost and legality:

```json
{"pack_correction_pair":true,"correction_pair_keys":["packed_gdn"],"correction_pair_verify_setup":true}
```

```json
{"pack_correction_pair":true,"correction_pair_keys":["linear_attn.out_proj.weight"],"correction_pair_verify_setup":true}
```

```json
{"pack_correction_pair":true,"correction_pair_keys":["packed_gdn","linear_attn.out_proj.weight"],"correction_pair_verify_setup":true}
```

Optional `correction_pair_geometry` is keyed by original projection name and overrides `cores/readers/block/pipeline` for the pair only. Begin with inherited C32/R4/block1, then tune if necessary. Retain the best correct separate path in every table. Do not lower fidelity or enable the previously failing packer accumulation while attributing the topology change.

To rebase onto a newer frozen passing prefill candidate without modifying it:

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_correction_pair.py \
  --source models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/<source>.py \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/<new_name>.py
```

The generator refuses to replace an existing output and records source/output hashes. First run real-weight B1/B32 short correctness with setup verification, then the exact B32 128-prefix/512-step schedule and watcher run. Compare prefill outputs/states bitwise against the same source with `pack_correction_pair:false`; the prefix must stay unchanged. If retained, collect traced-decode profiler rows for the packed matmul, both slices, add and export, and report the full before/after latency.

## CPU checks completed

`tests/test_correction_pair_decode.py` passed 32 algebra/shape/phase cases: N16512/N5120, B1/B3/B31/B32, ordinary/compact inputs, sharded/DRAM splits. It checks one packed matmul versus two separate calls, exact dyadic-control BF16 sums, column order, bank/reader divisibility, explicit phase propagation, unchanged prefill/raw-matmul AST, and absence of runtime host conversions. `correction_pair_cpu_checks.json` records results. The dyadic CPU control intentionally isolates layout/algebra; it is not a substitute for native matmul precision testing.

```bash
python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/test_correction_pair_decode.py
python_env/bin/python -m py_compile models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/correction_pair_decode.py
```

No speedup, native bit-exactness, allocation success or watcher result is claimed yet.
