# Optional DRAM-sharded lane QKV candidate

`LanePartitionQKV` now has an optional DRAM-sharded decode path. Factory defaults
are unchanged: `qkv_dram=False`. Controls are `qkv_dram_readers=1` and
`qkv_dram_block=1`. The existing fidelity, BF16 activation decomposition, lane
mask, FP32 destination/output, FP32 final row reduction, and tied-K restoration
remain in effect. Prefill uses the existing source projection. This candidate
has source/host validation only; no hardware or performance result is claimed.

## Source contracts and adaptation

- `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:1303`
  requires input A and output to use matching width-sharded L1 layouts, row-major
  activation shards, tiled K divisible by the K block, and each activation shard's
  tiled width divisible by that block. It explicitly requires **one padded M
  tile**. Therefore M=32 runs once; M=48 runs as 32 and 16 logical rows, converted
  back to interleaved outputs and rejoined in original order before the FP32 sum.
- The same file at line 2512 derives output storage from `per_core_M` and
  `per_core_N`. Setting `per_core_M=1` and `per_core_N=N_padded/32/input_cores`
  makes output coverage exact on the same number of storage cores as input A.
- `device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:143`
  requires per-bank weight tiles divisible by reader count. With multiple
  readers, bank width must equal readers times each worker's exact N coverage.
  The candidate explicitly zero-pads logical N to a multiple of
  `32*lcm(banks*6,input_cores)`. All reader counts 1/2/3 therefore share the same
  matrix storage and satisfy these constraints, without partial/padding-only
  banks. Output columns are cropped before row reduction and before tied-K
  duplication.
- That factory at line 170 selects output subblocks using `fp32_dest_acc_en`
  and caps their width at four tiles; line 206 uses Float32 intermediates when
  FP32 destination accumulation is enabled. Output CB format is derived from
  actual requested output dtype. The candidate requests `ttnn.float32`; there
  is no BF16 output cast before the row sum.
- `device/utilities/matmul_utilities.cpp:341` permits readers 1 through 3, and
  device validation allows more than one only on Blackhole. Constructor checks
  these constraints. DRAM plus separate Q/K/V is explicitly rejected for now.

Weights are constructed once from the original host state using the same
single-device column order in `models/demos/gemma4/tt/attention/weights.py:111`:
Q, K, V; tied global attention uses stored Q, K. They are uploaded in BF16.
The source callable and original weights remain available for prefill. There
is no device-to-host weight transfer and no host operation in decode.

For K=2816, the constructor selects the largest of 8/6/4/3/2/1 input storage
cores fitting the device's first row, dividing 88 K tiles, and making each
shard's K width divisible by the chosen block. With eight DRAM banks and block
1, both sliding N=8192 and tied-full N=9216 use physical N=9216, eight input
storage cores, input shards [32,352], weight shards [2816,1152], output shards
[32,1152], and `per_core_N=36`. The report's `qkv_dram_geometry` field records
actual geometry from the device.

The independent DRAM block starts at 1 because BF16 weight tiles are 2048 bytes
and FP32 result/intermediate tiles are 4096 bytes. For the geometry above, the
tripled weight CB alone would consume `3*11*36*2048=2,433,024` bytes per worker
at block 11 with one reader. That is a poor first configuration. Block 1 needs
221,184 bytes for that CB; three readers reduce the per-worker width to 12,
making block 11 a later candidate. These are source-derived CB sizes, not an
exhaustive L1 allocation proof or a speed measurement.

## Proposed device-owner checks

Use the existing same-input projection probe or short whole-layer check first,
then the unchanged 4096+128 gate. Verify actual BF16 weight/operand and FP32
product/output dtypes, tied-K output shape, exact trace repeatability, runtime
audit, program-cache guard, and component/full-layer latency before retaining.

```bash
python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_decoder \
  --defaults --default-overrides '{"qkv_dram":true,"qkv_dram_readers":1,"qkv_dram_block":1}' \
  --layer 0 --length 33 --real --decode --steps 8 --verify-program-cache \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv_dram_smoke_layer0.json
```

Candidate-mode CLI also exposes `--qkv-dram`, `--qkv-dram-readers` and
`--qkv-dram-block`, alongside `--qkv-lanes`. With `--defaults`, use the JSON
override mechanism as above. Explicit interleaved `qkv_grid`, `qkv_block_w`, and
`qkv_subblock_w` do not configure the DRAM kernel.

Python compilation, Black, and `git diff --check` pass. No C++ or kernel changes
were made, so no build is required. No hardware commands were run by this
investigator.
