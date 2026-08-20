# LLK perf run-to-run noise

- runs: **5** (same commit, same machine)
- points compared: **100971** (test x marker x run_type x sweep-config)
- absolute floor applied to the recommendation: **100 cycles**
- arch: wormhole
- commit: 3a0c85c851758d3aba7429260829dfc8f44f820a
- speed_of_light: off
- host: wh-lb-35-special-nstojic-for-reservation-69726

Runs analyzed:

- `/home/nstojic/wh_noise_l1/run_1`
- `/home/nstojic/wh_noise_l1/run_2`
- `/home/nstojic/wh_noise_l1/run_3`
- `/home/nstojic/wh_noise_l1/run_4`
- `/home/nstojic/wh_noise_l1/run_5`

## Gate false-positive floor -- median-of-1 vs median-of-1

2019420 simulated comparisons of identical code. Any threshold at or below a percentile below fires on that fraction of points **with no code change at all**.

| percentile | |delta| |
| --- | --- |
| p50 | 0.00% |
| p90 | 0.07% |
| p95 | 0.39% |
| p99 | 1.27% |
| max | 5.99% |

## Gate false-positive floor -- median-of-2 vs median-of-2

3029130 simulated comparisons of identical code. Any threshold at or below a percentile below fires on that fraction of points **with no code change at all**.

| percentile | |delta| |
| --- | --- |
| p50 | 0.00% |
| p90 | 0.15% |
| p95 | 0.35% |
| p99 | 0.88% |
| max | 4.71% |

## Recommended threshold

- **median-of-1**: `2.00%` clears the p99 of pure noise; `6.00%` clears every observed noise sample.
- **median-of-2**: `1.00%` clears the p99 of pure noise; `5.00%` clears every observed noise sample.

Pair the relative threshold with an absolute floor of **100 cycles** -- flag a point only when it is both more than the relative threshold slower AND more than the floor slower in absolute cycles.

### Noise by run type (median-of-1)

| run type | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| L1_TO_L1 | 2019420 | 0.39% | 1.27% | 5.99% |

### Noise by marker (median-of-1)

| marker | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| INIT | 673140 | 0.98% | 1.91% | 5.99% |
| TILE_LOOP | 673140 | 0.05% | 0.20% | 4.84% |
| KERNEL | 673140 | 0.05% | 0.20% | 4.82% |

### Noise by run type (median-of-2)

| run type | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| L1_TO_L1 | 3029130 | 0.35% | 0.88% | 4.71% |

### Noise by marker (median-of-2)

| marker | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| INIT | 1009710 | 0.69% | 1.29% | 4.71% |
| TILE_LOOP | 1009710 | 0.04% | 0.15% | 3.52% |
| KERNEL | 1009710 | 0.04% | 0.15% | 3.51% |

## 25 least stable points

Candidates to stabilize, exclude from the gate, or give a per-point threshold. `spread` is (max-min)/median across the runs.

| test | marker | run_type | median | spread | abs spread | config |
| --- | --- | --- | --- | --- | --- | --- |
| perf_eltwise_binary_sfpu | INIT | L1_TO_L1 | 460.0 | 5.65% | 26.0 | approx_mode=ApproximationMode.No, dest_acc=DestAccumulation.No, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16_b, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, iterations=32, loop_factor=16, mathop=MathOperation.SfpuElwadd, num_faces=4, num_faces_A=4, num_faces_B=4, speed_of_light=False, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_pack_untilize | INIT | L1_TO_L1 | 513.0 | 5.26% | 27.0 | block_ct_dim=3, block_rt_dim=3, dest_acc=DestAccumulation.Yes, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float16, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, full_ct_dim=3, full_rt_dim=3, loop_factor=32, speed_of_light=False, tile_cnt=9, unpack_to_dest=False |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=3, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=2, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=6, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=3, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=3, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=9, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=3, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=4, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=12, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=6, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=1, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=6, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=7, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=2, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=14, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=7, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=224, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=8, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=1, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 527.0 | 4.74% | 25.0 | c_dimm=8, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=4, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=32, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 510.0 | 4.71% | 24.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, k_dimm=1, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=3, speed_of_light=False, throttle_level=0, tile_cnt=3, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 510.0 | 4.71% | 24.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, k_dimm=2, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=2, speed_of_light=False, throttle_level=0, tile_cnt=4, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 510.0 | 4.71% | 24.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, k_dimm=2, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=6, speed_of_light=False, throttle_level=0, tile_cnt=12, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 510.0 | 4.71% | 24.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, k_dimm=2, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=8, speed_of_light=False, throttle_level=0, tile_cnt=16, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 510.0 | 4.71% | 24.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, k_dimm=3, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=3, speed_of_light=False, throttle_level=0, tile_cnt=9, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 510.0 | 4.71% | 24.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=6, speed_of_light=False, throttle_level=0, tile_cnt=192, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 510.0 | 4.71% | 24.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, k_dimm=4, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=2, speed_of_light=False, throttle_level=0, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 510.0 | 4.71% | 24.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, k_dimm=8, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=5, speed_of_light=False, throttle_level=0, tile_cnt=40, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 510.0 | 4.71% | 24.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Bfp8_b, formats.input_B=Bfp8_b, formats.output=Float32, formats.register_A=Bfp8_b, formats.register_B=Bfp8_b, formats.sfpu_math=Bfp8_b, k_dimm=8, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=7, speed_of_light=False, throttle_level=0, tile_cnt=56, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_math_matmul | TILE_LOOP | L1_TO_L1 | 193975.0 | 4.62% | 8956.0 | c_dimm=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, dst_index=4, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, in0_c_dim=32, in0_r_dim=16, in1_c_dim=32, in1_r_dim=32, k_dimm=1, loop_factor=1024, math_fidelity=MathFidelity.LoFi, num_faces=2, num_faces_A=2, num_faces_B=4, partial_a=True, partial_b=False, partial_face_math=False, partial_face_pack=True, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=4, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_math_matmul | KERNEL | L1_TO_L1 | 194730.0 | 4.60% | 8959.0 | c_dimm=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, dst_index=4, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, in0_c_dim=32, in0_r_dim=16, in1_c_dim=32, in1_r_dim=32, k_dimm=1, loop_factor=1024, math_fidelity=MathFidelity.LoFi, num_faces=2, num_faces_A=2, num_faces_B=4, partial_a=True, partial_b=False, partial_face_math=False, partial_face_pack=True, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=4, unpack_to_dest=False, unpack_transpose_faces=Transpose.No, unpack_transpose_within_face=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 526.0 | 4.56% | 24.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=64, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 526.0 | 4.56% | 24.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=4, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 526.0 | 4.56% | 24.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=8, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=16, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | INIT | L1_TO_L1 | 526.0 | 4.56% | 24.0 | c_dimm=3, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float32, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=3, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=9, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
