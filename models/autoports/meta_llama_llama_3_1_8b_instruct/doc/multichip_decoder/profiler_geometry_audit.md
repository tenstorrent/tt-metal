# DRAM-sharded projection geometry audit

This audit resolves the `tt-perf-report` `SLOW` / `No output subblock size
found` warning on every final TP4 projection row.  It distinguishes the public
program-config fields from the geometry that the DRAM-sharded factory derives
internally, and compares the material interleaved 1-D alternative under the
same BFP4/LoFi policy.

## What the configured 16 means

`OptimizationConfig.qkv_cores`, `output_cores`, `gate_up_cores`, and
`down_cores` are used by `_dram_matmul_program_config` to set the input L1
shard and `per_core_N` storage partition.  They are not the final kernel-launch
core count.  The final default uses a 16-way storage/N partition for each
projection and a 16-core residual/norm grid.

The raw Tracy rows report 80 active kernel cores for each DRAM-sharded
projection.  This agrees with the factory: it assigns weight readers from the
eight Blackhole DRAM banks, unions those cores with the input storage cores,
then launches kernels on the rectangular bounding box of that union.  In this
placement that rectangle contains 80 cores.  Eight DRAM-bank workers own the
unique output-N regions; the bounding-box launch also includes multicast and
storage participants.  Therefore neither “16 projection compute cores” nor
the report's launch count alone describes the full mapping.

## Why the report cannot see output subblocks

The public `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` contains
only `in0_block_w`, `per_core_M`, `per_core_N`, and `fused_activation`; it has
no output-subblock field.  That is exactly the attribute set serialized into
the retained raw profiler CSV, so `tt-perf-report` leaves its output-subblock
columns blank and emits its generic warning.

The selected factory does choose a real output subblock after dispatch:

1. `get_optimal_dram_bank_to_reader_assignment` selects the eight P300c DRAM
   bank workers.
2. `per_core_N_compute = ceil(N_tiles / num_dram_banks)`.
3. `get_matmul_subblock_params` chooses a divisor with at most eight BF16
   destination tiles, then the factory's preference loop can widen it if that
   reduces the number of N subblocks.
4. The selected `out_subblock_h`, `out_subblock_w`, and tile count are passed
   directly in the compute-kernel arguments.

Source:

- `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config_types.hpp`
- `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp`
- `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`

For batch-32 decode (`per_core_M=1`) the exact derived geometry is:

| Projection | Local KxN | N tiles / 8 banks | Factory output subblock | Selected input block |
| --- | ---: | ---: | ---: | ---: |
| packed QKV | `4096x1536` | 6 | `1x6` | 8 K tiles |
| O | `1024x4096` | 16 | `1x8` | 2 K tiles |
| gate | `4096x3584` | 14 | `1x7` | 8 K tiles |
| up | `4096x3584` | 14 | `1x7` | 8 K tiles |
| down | `3584x4096` | 16 | `1x8` | 7 K tiles |

Every output subblock is the largest exact divisor permitted by the factory's
eight-tile BF16 destination rule for these one-tile-M shapes.  Every selected
input block is the complete K width of one 16-way input shard.  The blank
profiler column is therefore a parser-visibility limitation, not a missing or
unit-width subblock in the measured kernel.

## Same-policy alternative measurements

Two additional full-decoder candidates were run after the profiler warning:

| Candidate | TP4 prefill | TP4 decode | Output PCC | Disposition |
| --- | ---: | ---: | ---: | --- |
| final 16-way DRAM-sharded | 0.733909 ms | 0.320058 ms | 0.9999998071 | selected |
| eight-way DRAM storage/N geometry | 0.753826 ms | 0.331311 ms | 0.9999998070 | decode 3.52% slower |
| interleaved BFP4/LoFi 1-D | 0.989849 ms | 0.405666 ms | 0.9999998039 | prefill 34.87%, decode 26.75% slower |

The interleaved family is shape-adapted through the complete decoder, not an
isolated matmul: QKV uses 24 output workers with `per_core_N=2`; O and down use
64 with `per_core_N=2`; gate/up use 28 with `per_core_N=4`; explicit output
subblocks are respectively 2, 2, 4, 4, and 2.  The same BFP4 weights, LoFi
math, local attention/MLP ownership, two BF16 all-reduces, cache updates, and
warmed decode trace run to an output PCC of `0.9999998039`.

The retained `SLOW` label reflects low modeled utilization for a one-tile-M
decode and the report's inability to introspect the factory-derived subblock.
It identifies a real optimization pressure, but the material public-config and
interleaved alternatives are slower.  The selected final default is the
fastest measured correct full-decoder family in the same precision regime.

Exact logs and all derived metrics are in `candidate_results.csv`,
`logs/perf_explore_geometry8.log`, and
`logs/perf_explore_interleaved_1d.log`.
