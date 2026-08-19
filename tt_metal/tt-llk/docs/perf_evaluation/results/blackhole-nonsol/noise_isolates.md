# LLK perf run-to-run noise

- runs: **5** (same commit, same machine)
- points compared: **311352** (test x marker x run_type x sweep-config)
- absolute floor applied to the recommendation: **100 cycles**
- arch: blackhole
- commit: 587c53ba4114d897840065635caa6a795cf229bf
- speed_of_light: off
- host: bh-qbge-15-special-nstojic-for-reservation-69347

Runs analyzed:

- `/home/nstojic/perf_noise_isolates/run_1`
- `/home/nstojic/perf_noise_isolates/run_2`
- `/home/nstojic/perf_noise_isolates/run_3`
- `/home/nstojic/perf_noise_isolates/run_4`
- `/home/nstojic/perf_noise_isolates/run_5`

## Gate false-positive floor -- median-of-1 vs median-of-1

6160960 simulated comparisons of identical code. Any threshold at or below a percentile below fires on that fraction of points **with no code change at all**.

| percentile | |delta| |
| --- | --- |
| p50 | 0.00% |
| p90 | 0.00% |
| p95 | 0.00% |
| p99 | 0.17% |
| max | 24.62% |

## Gate false-positive floor -- median-of-2 vs median-of-2

9241440 simulated comparisons of identical code. Any threshold at or below a percentile below fires on that fraction of points **with no code change at all**.

| percentile | |delta| |
| --- | --- |
| p50 | 0.00% |
| p90 | 0.00% |
| p95 | 0.00% |
| p99 | 0.27% |
| max | 10.96% |

## Recommended threshold

- **median-of-1**: `1.00%` clears the p99 of pure noise; `25.00%` clears every observed noise sample.
- **median-of-2**: `1.00%` clears the p99 of pure noise; `11.00%` clears every observed noise sample.

Pair the relative threshold with an absolute floor of **100 cycles** -- flag a point only when it is both more than the relative threshold slower AND more than the floor slower in absolute cycles.

### Noise by run type (median-of-1)

| run type | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| MATH_ISOLATE | 1990620 | 0.00% | 0.34% | 5.43% |
| PACK_ISOLATE | 2134840 | 0.00% | 0.24% | 24.62% |
| UNPACK_ISOLATE | 2035500 | 0.00% | 0.17% | 2.61% |

### Noise by marker (median-of-1)

| marker | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| INIT | 2053500 | 0.00% | 0.99% | 5.43% |
| KERNEL | 2053500 | 0.00% | 0.02% | 16.44% |
| TILE_LOOP | 2053500 | 0.00% | 0.01% | 24.62% |
| UNINIT | 460 | 0.00% | 0.00% | 0.00% |

### Noise by run type (median-of-2)

| run type | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| PACK_ISOLATE | 3202260 | 0.00% | 0.46% | 10.96% |
| MATH_ISOLATE | 2985930 | 0.00% | 0.34% | 3.78% |
| UNPACK_ISOLATE | 3053250 | 0.00% | 0.09% | 2.61% |

### Noise by marker (median-of-2)

| marker | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| INIT | 3080250 | 0.11% | 0.79% | 4.04% |
| KERNEL | 3080250 | 0.00% | 0.02% | 7.60% |
| TILE_LOOP | 3080250 | 0.00% | 0.01% | 10.96% |
| UNINIT | 690 | 0.00% | 0.00% | 0.00% |

## 25 least stable points

Candidates to stabilize, exclude from the gate, or give a per-point threshold. `spread` is (max-min)/median across the runs.

| test | marker | run_type | median | spread | abs spread | config |
| --- | --- | --- | --- | --- | --- | --- |
| perf_pack_dest_bank | TILE_LOOP | PACK_ISOLATE | 1787.0 | 19.75% | 353.0 | dest_acc=DestAccumulation.No, dst_index=0, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, input_num_blocks=2, input_num_tiles_in_block=4, l1_acc=L1Accumulation.Yes, loop_factor=8, num_blocks=2, num_faces=4, num_faces_A=4, num_faces_B=4, num_tiles_in_block=4, output_num_blocks=2, output_num_tiles_in_block=4, speed_of_light=False, tile_cnt=8, tilize=Tilize.No, unpack_to_dest=False |
| perf_fast_tilize_full | UNINIT | MATH_ISOLATE | 42.0 | 14.29% | 6.0 | block_ct_dim=6, block_rt_dim=1, dest_acc=DestAccumulation.No, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=6, full_rt_dim=1, loop_factor=32, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=6, unpack_to_dest=False |
| perf_pack_dest_bank | KERNEL | PACK_ISOLATE | 2472.0 | 14.12% | 349.0 | dest_acc=DestAccumulation.No, dst_index=0, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, input_num_blocks=2, input_num_tiles_in_block=4, l1_acc=L1Accumulation.Yes, loop_factor=8, num_blocks=2, num_faces=4, num_faces_A=4, num_faces_B=4, num_tiles_in_block=4, output_num_blocks=2, output_num_tiles_in_block=4, speed_of_light=False, tile_cnt=8, tilize=Tilize.No, unpack_to_dest=False |
| perf_fast_tilize_full | UNINIT | MATH_ISOLATE | 36.0 | 13.89% | 5.0 | block_ct_dim=8, block_rt_dim=1, dest_acc=DestAccumulation.No, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Bfp4_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=8, full_rt_dim=1, loop_factor=32, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=8, unpack_to_dest=False |
| perf_eltwise_unary_sfpu | INIT | MATH_ISOLATE | 194.0 | 5.15% | 10.0 | approx_mode=ApproximationMode.No, clamp_negative=False, dest_acc=DestAccumulation.No, fast_mode=FastMode.No, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, iterations=32, loop_factor=16, mathop=MathOperation.GeluTanh, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, stable_sort=StableSort.No, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=16, block_rt_dim=2, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float32, formats.register_A=Tf32, formats.register_B=Tf32, formats.sfpu_math=Tf32, full_ct_dim=16, full_rt_dim=2, loop_factor=128, speed_of_light=False, tile_cnt=32, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=16, block_rt_dim=4, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float32, formats.register_A=Tf32, formats.register_B=Tf32, formats.sfpu_math=Tf32, full_ct_dim=16, full_rt_dim=4, loop_factor=128, speed_of_light=False, tile_cnt=64, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=3, block_rt_dim=1, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float32, formats.register_A=Tf32, formats.register_B=Tf32, formats.sfpu_math=Tf32, full_ct_dim=3, full_rt_dim=1, loop_factor=32, speed_of_light=False, tile_cnt=3, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=3, block_rt_dim=1, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float32, formats.register_A=Tf32, formats.register_B=Tf32, formats.sfpu_math=Tf32, full_ct_dim=3, full_rt_dim=1, loop_factor=8, speed_of_light=False, tile_cnt=3, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=4, block_rt_dim=1, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float32, formats.register_A=Tf32, formats.register_B=Tf32, formats.sfpu_math=Tf32, full_ct_dim=4, full_rt_dim=1, loop_factor=8, speed_of_light=False, tile_cnt=4, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=4, block_rt_dim=1, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=4, full_rt_dim=1, loop_factor=32, speed_of_light=False, tile_cnt=4, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=5, block_rt_dim=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=5, full_rt_dim=2, loop_factor=32, speed_of_light=False, tile_cnt=10, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=5, block_rt_dim=2, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Bfp4_b, formats.input_B=Bfp4_b, formats.output=Float32, formats.register_A=Bfp4_b, formats.register_B=Bfp4_b, formats.sfpu_math=Bfp8_b, full_ct_dim=5, full_rt_dim=2, loop_factor=128, speed_of_light=False, tile_cnt=10, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=5, block_rt_dim=2, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=5, full_rt_dim=2, loop_factor=128, speed_of_light=False, tile_cnt=10, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=5, block_rt_dim=2, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, full_ct_dim=5, full_rt_dim=2, loop_factor=8, speed_of_light=False, tile_cnt=10, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=5, block_rt_dim=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Full, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=5, full_rt_dim=4, loop_factor=32, speed_of_light=False, tile_cnt=20, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=5, block_rt_dim=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=5, full_rt_dim=4, loop_factor=32, speed_of_light=False, tile_cnt=20, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=5, block_rt_dim=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=5, full_rt_dim=4, loop_factor=8, speed_of_light=False, tile_cnt=20, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=5, block_rt_dim=4, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Bfp4_b, formats.input_B=Bfp4_b, formats.output=Float32, formats.register_A=Bfp4_b, formats.register_B=Bfp4_b, formats.sfpu_math=Bfp8_b, full_ct_dim=5, full_rt_dim=4, loop_factor=32, speed_of_light=False, tile_cnt=20, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=5, block_rt_dim=4, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, full_ct_dim=5, full_rt_dim=4, loop_factor=32, speed_of_light=False, tile_cnt=20, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=5, block_rt_dim=4, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=5, full_rt_dim=4, loop_factor=8, speed_of_light=False, tile_cnt=20, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=5, block_rt_dim=4, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, full_ct_dim=5, full_rt_dim=4, loop_factor=32, speed_of_light=False, tile_cnt=20, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=6, block_rt_dim=2, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float32, formats.register_A=Tf32, formats.register_B=Tf32, formats.sfpu_math=Tf32, full_ct_dim=6, full_rt_dim=2, loop_factor=8, speed_of_light=False, tile_cnt=12, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=6, block_rt_dim=4, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float32, formats.register_A=Tf32, formats.register_B=Tf32, formats.sfpu_math=Tf32, full_ct_dim=6, full_rt_dim=4, loop_factor=128, speed_of_light=False, tile_cnt=24, unpack_to_dest=False |
| perf_fast_untilize | INIT | MATH_ISOLATE | 126.0 | 4.76% | 6.0 | block_ct_dim=7, block_rt_dim=2, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float32, formats.register_A=Tf32, formats.register_B=Tf32, formats.sfpu_math=Tf32, full_ct_dim=7, full_rt_dim=2, loop_factor=128, speed_of_light=False, tile_cnt=14, unpack_to_dest=False |
