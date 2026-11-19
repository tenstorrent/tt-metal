# Higher readers with lower padding and explicit C64 storage

V4 completed in `geometry119.json`: reformatted C32/R4/B1 measured 0.424386 ms, versus R8 0.448022, R9 0.488083, R10 0.545508, R12 0.465675, and R13 0.668678 ms. Every case preserved weight elements exactly. Every R>4 changed both FP32 partials and final BF16 output bitwise; final own-weight PCC remained approximately 0.9999985728. C32 higher-reader variants are rejected for latency. Fewer DST passes did not produce a win.

The next immutable probe is `tests/run_attention_geometry_probe_v5.py`. It retains the same original native C32/R4/B1 control and a reformatted matched C32/R4 row, then crosses C8 and C16 storage grids with R8/9/10/12/13. It also includes explicit C64/R9 and C64/R12 input-shard tests. There are 13 candidate rows plus the native baseline. K block1, precision configuration, actual captured compact input, FP32 partial outputs, BF16 final add, references, timing and error recording remain unchanged.

The smaller storage grids change N padding independently of the compute worker count. For example, C8/R13 uses N16640 instead of C32/R13's N26624, while both use 104 compute workers with one DST pass. C8/R13's ideal pair weight-read floor is approximately 0.354 ms, versus approximately 0.566 ms for C32/R13. C8/16 with R10 use N17920 instead of C32/R10's N20480. Actual NoC, DRAM and L1 effects must be measured; V4's changed output bits mean equivalence cannot be assumed from unchanged K groups.

## C64 legality and empty-shard check

The existing decoder geometry builder uses shard width `k//cores`, which gives 80 for C64/K5120 and is not tile-aligned. V5 explicitly constructs a 96-value (three-tile) shard width on 64 grid cores while leaving logical and padded K at 5120. This is supported by the following source chain:

- `tt_metal/impl/tensor/spec/tensor_spec.cpp::get_shape_fits_shard_grid_error` permits a width-sharded grid with more cores than real shards.
- `ttnn/core/tensor/tensor_utils.cpp::get_optimal_worker_cores_for_sharded_tensor` returns `cores_with_data()` for L1, explicitly excluding unused grid cores.
- `interleaved_to_sharded_program_factory.cpp` uses that active-core list, so only 54 real shards are initialized and the final active shard gets its actual width. It does not launch the copy kernel on the ten empty grid cores.
- `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp` requires activation capacity `x_shard_w * x_cores.size() >= Kt`; it retains original Kt160.
- `reader_dram_sharded_multicore.cpp::read_x` addresses `kXCores[k/SW]`. With K tiles 0–159 and SW3, the largest addressed core index is 53. No empty trailing grid core is read. Output N is padded to a multiple of 64, so all 64 output-storage cores have real output tiles.

Before benchmarking **every** new geometry, V5 checks an untimed activation round trip bitwise, asserts unchanged logical shape and padded K, and queries the actual number of active input cores. C64 must report 54; otherwise that geometry gets a setup-error row and is not timed. Every weight still passes exact prequantized-value readback checks. Runtime API errors remain per-row and the sweep continues after recoverable failures. No main decoder, old probe or shared source is changed.

Run under the parent's existing delivery118 environment and device lock:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_attention_geometry_probe_v5 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/geometry_lower_padding_results.json \
  > models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/geometry_lower_padding_results.log 2>&1
```

Syntax and `--plan-only` pass. `attention_geometry_v5_plan.json` is the immutable plan. The host-only `candidates/check_geometry_lower_padding_v5_cpu.py` checks unchanged timed-function ASTs, full column ownership, K addressing into active shards only, and the C64 54/10 active/unused split. `geometry_lower_padding_v5_cpu.json` records the source hash, checks and theoretical byte floors. Device correctness and latency are pending.

If V5 does not improve latency, an untried intermediate-reader hypothesis is C8/R5 (N16640, 40 workers, 13 columns/worker, two passes) or C8/16 R6 (N16896, 48 workers, 11 columns/worker, two passes). They may balance reduced padding with less NoC contention than the tested high-reader cases. This is a hand-off hypothesis, not a completed or rejected measurement.
