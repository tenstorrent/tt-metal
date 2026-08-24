# LLK perf run-to-run noise

- runs: **10** (same commit, same machine)
- points compared: **100971** (test x marker x run_type x sweep-config)
- absolute floor applied to the recommendation: **100 cycles**
- arch: wormhole
- commit: 8b9db213408b812cebe7c873341102a6c92910e2
- speed_of_light: off
- host: wh-lb-35-special-nstojic-for-reservation-72888

Runs analyzed:

- `/home/nstojic/wh_l1_x10/run_1`
- `/home/nstojic/wh_l1_x10/run_2`
- `/home/nstojic/wh_l1_x10/run_3`
- `/home/nstojic/wh_l1_x10/run_4`
- `/home/nstojic/wh_l1_x10/run_5`
- `/home/nstojic/wh_l1_x10/run_6`
- `/home/nstojic/wh_l1_x10/run_7`
- `/home/nstojic/wh_l1_x10/run_8`
- `/home/nstojic/wh_l1_x10/run_9`
- `/home/nstojic/wh_l1_x10/run_10`

## Gate false-positive floor -- median-of-1 vs median-of-1

9087390 simulated comparisons of identical code. Any threshold at or below a percentile below fires on that fraction of points **with no code change at all**.

| percentile | |delta| |
| --- | --- |
| p50 | 0.00% |
| p90 | 0.06% |
| p95 | 0.35% |
| p99 | 1.26% |
| max | 8.83% |

## Gate false-positive floor -- median-of-2 vs median-of-2

127223460 simulated comparisons of identical code. Any threshold at or below a percentile below fires on that fraction of points **with no code change at all**.

| percentile | |delta| |
| --- | --- |
| p50 | 0.00% |
| p90 | 0.12% |
| p95 | 0.34% |
| p99 | 0.87% |
| max | 4.74% |

## Recommended threshold

- **median-of-1**: `2.00%` clears the p99 of pure noise; `9.00%` clears every observed noise sample.
- **median-of-2**: `1.00%` clears the p99 of pure noise; `5.00%` clears every observed noise sample.

Pair the relative threshold with an absolute floor of **100 cycles** -- flag a point only when it is both more than the relative threshold slower AND more than the floor slower in absolute cycles.

### Noise by run type (median-of-1)

| run type | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| L1_TO_L1 | 9087390 | 0.35% | 1.26% | 8.83% |

### Noise by marker (median-of-1)

| marker | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| INIT | 3029130 | 0.97% | 1.85% | 5.99% |
| TILE_LOOP | 3029130 | 0.05% | 0.20% | 8.83% |
| KERNEL | 3029130 | 0.05% | 0.20% | 8.43% |

### Noise by run type (median-of-2)

| run type | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| L1_TO_L1 | 127223460 | 0.34% | 0.87% | 4.74% |

### Noise by marker (median-of-2)

| marker | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| INIT | 42407820 | 0.68% | 1.27% | 4.74% |
| TILE_LOOP | 42407820 | 0.04% | 0.15% | 4.48% |
| KERNEL | 42407820 | 0.04% | 0.15% | 4.46% |

## 25 least stable points

Candidates to stabilize, exclude from the gate, or give a per-point threshold. `spread` is (max-min)/median across the runs.

| test | marker | run_type | median | spread | abs spread | config |
| --- | --- | --- | --- | --- | --- | --- |
| perf_matmul | TILE_LOOP | L1_TO_L1 | 12594.0 | 8.83% | 1112.0 | c_dimm=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Bfp8_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=1, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=4, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | KERNEL | L1_TO_L1 | 13184.0 | 8.43% | 1111.0 | c_dimm=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Bfp8_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=1, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=4, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 218468.0 | 6.39% | 13965.0 | c_dimm=5, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, dst_index=0, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16_b, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, in0_c_dim=32, in0_r_dim=16, in1_c_dim=32, in1_r_dim=32, k_dimm=1, loop_factor=1024, math_fidelity=MathFidelity.HiFi2, num_faces=2, num_faces_A=2, num_faces_B=4, partial_a=True, partial_b=False, partial_face_math=False, partial_face_pack=True, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=5, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_math_matmul | KERNEL | L1_TO_L1 | 219223.0 | 6.37% | 13965.0 | c_dimm=5, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, dst_index=0, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16_b, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, in0_c_dim=32, in0_r_dim=16, in1_c_dim=32, in1_r_dim=32, k_dimm=1, loop_factor=1024, math_fidelity=MathFidelity.HiFi2, num_faces=2, num_faces_A=2, num_faces_B=4, partial_a=True, partial_b=False, partial_face_math=False, partial_face_pack=True, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=5, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 168975.0 | 6.05% | 10225.0 | c_dimm=3, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, dst_index=1, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, in0_c_dim=32, in0_r_dim=1, in1_c_dim=32, in1_r_dim=32, k_dimm=1, loop_factor=1024, math_fidelity=MathFidelity.LoFi, num_faces=2, num_faces_A=2, num_faces_B=4, partial_a=True, partial_b=False, partial_face_math=True, partial_face_pack=True, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=3, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 168948.0 | 6.04% | 10198.0 | c_dimm=3, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, dst_index=1, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16_b, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, in0_c_dim=32, in0_r_dim=1, in1_c_dim=32, in1_r_dim=32, k_dimm=1, loop_factor=1024, math_fidelity=MathFidelity.LoFi, num_faces=2, num_faces_A=2, num_faces_B=4, partial_a=True, partial_b=False, partial_face_math=True, partial_face_pack=True, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=3, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_math_matmul | KERNEL | L1_TO_L1 | 169729.0 | 6.03% | 10227.0 | c_dimm=3, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, dst_index=1, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, in0_c_dim=32, in0_r_dim=1, in1_c_dim=32, in1_r_dim=32, k_dimm=1, loop_factor=1024, math_fidelity=MathFidelity.LoFi, num_faces=2, num_faces_A=2, num_faces_B=4, partial_a=True, partial_b=False, partial_face_math=True, partial_face_pack=True, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=3, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_math_matmul | KERNEL | L1_TO_L1 | 169696.0 | 6.01% | 10194.0 | c_dimm=3, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, dst_index=1, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16_b, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, in0_c_dim=32, in0_r_dim=1, in1_c_dim=32, in1_r_dim=32, k_dimm=1, loop_factor=1024, math_fidelity=MathFidelity.LoFi, num_faces=2, num_faces_A=2, num_faces_B=4, partial_a=True, partial_b=False, partial_face_math=True, partial_face_pack=True, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=3, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_eltwise_binary_sfpu | INIT | L1_TO_L1 | 460.0 | 5.65% | 26.0 | approx_mode=ApproximationMode.No, dest_acc=DestAccumulation.No, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float32, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, iterations=32, loop_factor=16, mathop=MathOperation.SfpuElwsub, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_eltwise_binary_sfpu | INIT | L1_TO_L1 | 376.0 | 5.59% | 21.0 | approx_mode=ApproximationMode.Yes, dest_acc=DestAccumulation.No, formats.input_A=UInt32, formats.input_B=UInt32, formats.output=UInt32, formats.register_A=UInt32, formats.register_B=UInt32, formats.sfpu_math=UInt32, iterations=32, loop_factor=16, mathop=MathOperation.SfpuAddTopRow, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=8, unpack_to_dest=True, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_pack_untilize | INIT | L1_TO_L1 | 513.0 | 5.26% | 27.0 | block_ct_dim=3, block_rt_dim=6, dest_acc=DestAccumulation.Yes, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, full_ct_dim=3, full_rt_dim=6, loop_factor=32, speed_of_light=False, tile_cnt=18, unpack_to_dest=False |
| perf_pack_untilize | INIT | L1_TO_L1 | 513.0 | 5.26% | 27.0 | block_ct_dim=3, block_rt_dim=7, dest_acc=DestAccumulation.Yes, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float16, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, full_ct_dim=3, full_rt_dim=7, loop_factor=32, speed_of_light=False, tile_cnt=21, unpack_to_dest=False |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 212824.5 | 4.87% | 10358.0 | c_dimm=5, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, dst_index=0, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, in0_c_dim=32, in0_r_dim=8, in1_c_dim=32, in1_r_dim=32, k_dimm=1, loop_factor=1024, math_fidelity=MathFidelity.LoFi, num_faces=2, num_faces_A=2, num_faces_B=4, partial_a=True, partial_b=False, partial_face_math=True, partial_face_pack=True, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=5, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_math_matmul | KERNEL | L1_TO_L1 | 213564.5 | 4.85% | 10358.0 | c_dimm=5, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, dst_index=0, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, in0_c_dim=32, in0_r_dim=8, in1_c_dim=32, in1_r_dim=32, k_dimm=1, loop_factor=1024, math_fidelity=MathFidelity.LoFi, num_faces=2, num_faces_A=2, num_faces_B=4, partial_a=True, partial_b=False, partial_face_math=True, partial_face_pack=True, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=5, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 524.0 | 4.77% | 25.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=2, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=2, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=1, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=2, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=3, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=6, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=64, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=4, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=4, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=8, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=16, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=3, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=1, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=3, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=3, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=1, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=3, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=3, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=2, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=6, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=3, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=96, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
