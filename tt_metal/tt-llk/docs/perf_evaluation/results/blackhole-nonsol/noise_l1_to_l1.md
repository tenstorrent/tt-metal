# LLK perf run-to-run noise

- runs: **5** (same commit, same machine)
- points compared: **108377** (test x marker x run_type x sweep-config)
- absolute floor applied to the recommendation: **100 cycles**
- arch: blackhole
- commit: b8db2cbeb5c8fcf6c34f3f607268a28e45c97ccb
- speed_of_light: off
- host: bh-qbge-15-special-nstojic-for-reservation-69347

Runs analyzed:

- `/home/nstojic/perf_noise_baseline/run_1`
- `/home/nstojic/perf_noise_baseline/run_2`
- `/home/nstojic/perf_noise_baseline/run_3`
- `/home/nstojic/perf_noise_baseline/run_4`
- `/home/nstojic/perf_noise_baseline/run_5`

## Gate false-positive floor -- median-of-1 vs median-of-1

2167540 simulated comparisons of identical code. Any threshold at or below a percentile below fires on that fraction of points **with no code change at all**.

| percentile | |delta| |
| --- | --- |
| p50 | 0.00% |
| p90 | 0.00% |
| p95 | 0.00% |
| p99 | 0.41% |
| max | 7.91% |

## Gate false-positive floor -- median-of-2 vs median-of-2

3251310 simulated comparisons of identical code. Any threshold at or below a percentile below fires on that fraction of points **with no code change at all**.

| percentile | |delta| |
| --- | --- |
| p50 | 0.00% |
| p90 | 0.00% |
| p95 | 0.00% |
| p99 | 0.51% |
| max | 4.89% |

## Recommended threshold

- **median-of-1**: `1.00%` clears the p99 of pure noise; `8.00%` clears every observed noise sample.
- **median-of-2**: `1.00%` clears the p99 of pure noise; `5.00%` clears every observed noise sample.

Pair the relative threshold with an absolute floor of **100 cycles** -- flag a point only when it is both more than the relative threshold slower AND more than the floor slower in absolute cycles.

### Noise by run type (median-of-1)

| run type | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| L1_TO_L1 | 2167540 | 0.00% | 0.41% | 7.91% |

### Noise by marker (median-of-1)

| marker | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| INIT | 711520 | 0.00% | 1.36% | 7.91% |
| UNINIT | 32980 | 0.00% | 0.93% | 3.57% |
| KERNEL | 711520 | 0.00% | 0.01% | 1.90% |
| TILE_LOOP | 711520 | 0.00% | 0.01% | 1.91% |

### Noise by run type (median-of-2)

| run type | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| L1_TO_L1 | 3251310 | 0.00% | 0.51% | 4.89% |

### Noise by marker (median-of-2)

| marker | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| INIT | 1067280 | 0.23% | 1.01% | 4.89% |
| UNINIT | 49470 | 0.20% | 0.64% | 1.79% |
| KERNEL | 1067280 | 0.00% | 0.01% | 1.21% |
| TILE_LOOP | 1067280 | 0.00% | 0.01% | 1.21% |

## 25 least stable points

Candidates to stabilize, exclude from the gate, or give a per-point threshold. `spread` is (max-min)/median across the runs.

| test | marker | run_type | median | spread | abs spread | config |
| --- | --- | --- | --- | --- | --- | --- |
| perf_fast_untilize | INIT | L1_TO_L1 | 316.0 | 7.91% | 25.0 | block_ct_dim=9, block_rt_dim=4, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, full_ct_dim=9, full_rt_dim=4, loop_factor=32, speed_of_light=False, tile_cnt=36, unpack_to_dest=False |
| perf_fast_untilize | INIT | L1_TO_L1 | 324.0 | 6.48% | 21.0 | block_ct_dim=7, block_rt_dim=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Full, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=7, full_rt_dim=1, loop_factor=128, speed_of_light=False, tile_cnt=7, unpack_to_dest=False |
| perf_fast_untilize | INIT | L1_TO_L1 | 324.0 | 6.48% | 21.0 | block_ct_dim=7, block_rt_dim=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Full, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=7, full_rt_dim=1, loop_factor=8, speed_of_light=False, tile_cnt=7, unpack_to_dest=False |
| perf_fast_untilize | INIT | L1_TO_L1 | 325.0 | 6.46% | 21.0 | block_ct_dim=7, block_rt_dim=1, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=7, full_rt_dim=1, loop_factor=8, speed_of_light=False, tile_cnt=7, unpack_to_dest=False |
| perf_eltwise_binary_sfpu | INIT | L1_TO_L1 | 351.0 | 6.27% | 22.0 | approx_mode=ApproximationMode.No, dest_acc=DestAccumulation.Yes, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Bfp8_b, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, iterations=32, loop_factor=16, mathop=MathOperation.SfpuElwadd, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_eltwise_binary_sfpu | INIT | L1_TO_L1 | 351.0 | 6.27% | 22.0 | approx_mode=ApproximationMode.Yes, dest_acc=DestAccumulation.Yes, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Bfp8_b, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, iterations=32, loop_factor=16, mathop=MathOperation.SfpuElwmul, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_eltwise_binary_sfpu | INIT | L1_TO_L1 | 351.0 | 6.27% | 22.0 | approx_mode=ApproximationMode.Yes, dest_acc=DestAccumulation.Yes, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Bfp8_b, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, iterations=32, loop_factor=16, mathop=MathOperation.SfpuElwrsub, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_eltwise_binary_sfpu | INIT | L1_TO_L1 | 300.0 | 6.00% | 18.0 | approx_mode=ApproximationMode.Yes, dest_acc=DestAccumulation.Yes, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float16_b, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, iterations=32, loop_factor=16, mathop=MathOperation.SfpuElwadd, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_eltwise_binary_sfpu | INIT | L1_TO_L1 | 300.0 | 6.00% | 18.0 | approx_mode=ApproximationMode.Yes, dest_acc=DestAccumulation.Yes, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float16_b, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, iterations=32, loop_factor=16, mathop=MathOperation.SfpuElwrsub, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_fast_untilize | INIT | L1_TO_L1 | 323.0 | 5.57% | 18.0 | block_ct_dim=9, block_rt_dim=4, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Bfp4_b, formats.input_B=Bfp4_b, formats.output=Float32, formats.register_A=Bfp4_b, formats.register_B=Bfp4_b, formats.sfpu_math=Bfp8_b, full_ct_dim=9, full_rt_dim=4, loop_factor=128, speed_of_light=False, tile_cnt=36, unpack_to_dest=False |
| perf_fast_untilize | INIT | L1_TO_L1 | 323.0 | 5.57% | 18.0 | block_ct_dim=9, block_rt_dim=4, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Full, formats.input_A=Bfp4_b, formats.input_B=Bfp4_b, formats.output=Float32, formats.register_A=Bfp4_b, formats.register_B=Bfp4_b, formats.sfpu_math=Bfp8_b, full_ct_dim=9, full_rt_dim=4, loop_factor=8, speed_of_light=False, tile_cnt=36, unpack_to_dest=False |
| perf_fast_untilize | INIT | L1_TO_L1 | 341.0 | 5.28% | 18.0 | block_ct_dim=9, block_rt_dim=2, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, full_ct_dim=9, full_rt_dim=2, loop_factor=8, speed_of_light=False, tile_cnt=18, unpack_to_dest=False |
| perf_eltwise_binary_fpu | INIT | L1_TO_L1 | 442.0 | 5.20% | 23.0 | dest_acc=DestAccumulation.Yes, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Bfp8_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, loop_factor=8, math_fidelity=MathFidelity.LoFi, mathop=MathOperation.Elwadd, speed_of_light=False, tile_cnt=16, unpack_to_dest=False |
| perf_fast_untilize | INIT | L1_TO_L1 | 376.0 | 5.05% | 19.0 | block_ct_dim=12, block_rt_dim=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Full, formats.input_A=Bfp4_b, formats.input_B=Bfp4_b, formats.output=Float16_b, formats.register_A=Bfp4_b, formats.register_B=Bfp4_b, formats.sfpu_math=Bfp8_b, full_ct_dim=12, full_rt_dim=4, loop_factor=8, speed_of_light=False, tile_cnt=48, unpack_to_dest=False |
| perf_fast_untilize_baseline_compare | INIT | L1_TO_L1 | 326.0 | 4.91% | 16.0 | block_ct_dim=6, block_rt_dim=2, dest_acc=DestAccumulation.No, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=12, full_rt_dim=2, loop_factor=32, speed_of_light=False, tile_cnt=24, unpack_to_dest=False |
| perf_fast_untilize_baseline_compare | INIT | L1_TO_L1 | 310.0 | 4.84% | 15.0 | block_ct_dim=8, block_rt_dim=1, dest_acc=DestAccumulation.No, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=8, full_rt_dim=1, loop_factor=32, speed_of_light=False, tile_cnt=8, unpack_to_dest=False |
| perf_fast_untilize_baseline_compare | INIT | L1_TO_L1 | 310.0 | 4.84% | 15.0 | block_ct_dim=8, block_rt_dim=2, dest_acc=DestAccumulation.No, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=8, full_rt_dim=2, loop_factor=32, speed_of_light=False, tile_cnt=16, unpack_to_dest=False |
| perf_fast_untilize_baseline_compare | INIT | L1_TO_L1 | 311.0 | 4.82% | 15.0 | block_ct_dim=3, block_rt_dim=1, dest_acc=DestAccumulation.No, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=3, full_rt_dim=1, loop_factor=32, speed_of_light=False, tile_cnt=3, unpack_to_dest=False |
| perf_fast_untilize_baseline_compare | INIT | L1_TO_L1 | 311.0 | 4.82% | 15.0 | block_ct_dim=3, block_rt_dim=4, dest_acc=DestAccumulation.No, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=3, full_rt_dim=4, loop_factor=16, speed_of_light=False, tile_cnt=12, unpack_to_dest=False |
| perf_fast_untilize | INIT | L1_TO_L1 | 315.0 | 4.76% | 15.0 | block_ct_dim=16, block_rt_dim=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Full, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=16, full_rt_dim=2, loop_factor=128, speed_of_light=False, tile_cnt=32, unpack_to_dest=False |
| perf_fast_untilize_baseline_compare | INIT | L1_TO_L1 | 357.0 | 4.76% | 17.0 | block_ct_dim=4, block_rt_dim=2, dest_acc=DestAccumulation.No, formats.input_A=Bfp4_b, formats.input_B=Bfp4_b, formats.output=Float16_b, formats.register_A=Bfp4_b, formats.register_B=Bfp4_b, formats.sfpu_math=Bfp8_b, full_ct_dim=4, full_rt_dim=2, loop_factor=32, speed_of_light=False, tile_cnt=8, unpack_to_dest=False |
| perf_fast_untilize_baseline_compare | INIT | L1_TO_L1 | 357.0 | 4.76% | 17.0 | block_ct_dim=5, block_rt_dim=2, dest_acc=DestAccumulation.No, formats.input_A=Bfp4_b, formats.input_B=Bfp4_b, formats.output=Float16_b, formats.register_A=Bfp4_b, formats.register_B=Bfp4_b, formats.sfpu_math=Bfp8_b, full_ct_dim=5, full_rt_dim=2, loop_factor=32, speed_of_light=False, tile_cnt=10, unpack_to_dest=False |
| perf_eltwise_binary_sfpu | INIT | L1_TO_L1 | 311.0 | 4.66% | 14.5 | approx_mode=ApproximationMode.No, dest_acc=DestAccumulation.Yes, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float16, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, iterations=32, loop_factor=16, mathop=MathOperation.SfpuElwmul, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_eltwise_binary_sfpu | INIT | L1_TO_L1 | 311.0 | 4.66% | 14.5 | approx_mode=ApproximationMode.No, dest_acc=DestAccumulation.Yes, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float16, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, iterations=32, loop_factor=16, mathop=MathOperation.SfpuElwsub, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_eltwise_binary_sfpu | INIT | L1_TO_L1 | 311.0 | 4.66% | 14.5 | approx_mode=ApproximationMode.Yes, dest_acc=DestAccumulation.Yes, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float16, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, iterations=32, loop_factor=16, mathop=MathOperation.SfpuElwadd, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
