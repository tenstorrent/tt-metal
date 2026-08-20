# LLK perf run-to-run noise

- runs: **5** (same commit, same machine)
- points compared: **290952** (test x marker x run_type x sweep-config)
- absolute floor applied to the recommendation: **100 cycles**
- arch: wormhole
- commit: 59602f6f3fa27d01199a769488866e9cfdc241ed
- speed_of_light: off
- host: wh-lb-35-special-nstojic-for-reservation-70325

Runs analyzed:

- `/home/nstojic/wh_noise_isolates/run_1`
- `/home/nstojic/wh_noise_isolates/run_2`
- `/home/nstojic/wh_noise_isolates/run_3`
- `/home/nstojic/wh_noise_isolates/run_4`
- `/home/nstojic/wh_noise_isolates/run_5`

## Gate false-positive floor -- median-of-1 vs median-of-1

5819040 simulated comparisons of identical code. Any threshold at or below a percentile below fires on that fraction of points **with no code change at all**.

| percentile | |delta| |
| --- | --- |
| p50 | 0.00% |
| p90 | 0.00% |
| p95 | 0.09% |
| p99 | 1.36% |
| max | 24.75% |

## Gate false-positive floor -- median-of-2 vs median-of-2

8728560 simulated comparisons of identical code. Any threshold at or below a percentile below fires on that fraction of points **with no code change at all**.

| percentile | |delta| |
| --- | --- |
| p50 | 0.00% |
| p90 | 0.02% |
| p95 | 0.23% |
| p99 | 0.92% |
| max | 22.96% |

## Recommended threshold

- **median-of-1**: `2.00%` clears the p99 of pure noise; `25.00%` clears every observed noise sample.
- **median-of-2**: `1.00%` clears the p99 of pure noise; `23.00%` clears every observed noise sample.

Pair the relative threshold with an absolute floor of **100 cycles** -- flag a point only when it is both more than the relative threshold slower AND more than the floor slower in absolute cycles.

### Noise by run type (median-of-1)

| run type | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| PACK_ISOLATE | 1988280 | 0.15% | 2.15% | 24.75% |
| MATH_ISOLATE | 1893840 | 0.01% | 1.06% | 5.91% |
| UNPACK_ISOLATE | 1936920 | 0.18% | 0.91% | 3.67% |

### Noise by marker (median-of-1)

| marker | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| INIT | 1939680 | 0.71% | 1.81% | 5.91% |
| KERNEL | 1939680 | 0.01% | 0.24% | 24.63% |
| TILE_LOOP | 1939680 | 0.01% | 0.23% | 24.75% |

### Noise by run type (median-of-2)

| run type | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| PACK_ISOLATE | 2982420 | 0.31% | 1.74% | 22.96% |
| MATH_ISOLATE | 2840760 | 0.05% | 0.78% | 4.93% |
| UNPACK_ISOLATE | 2905380 | 0.26% | 0.71% | 3.41% |

### Noise by marker (median-of-2)

| marker | n | p95 | p99 | max |
| --- | --- | --- | --- | --- |
| INIT | 2909520 | 0.60% | 1.13% | 4.93% |
| TILE_LOOP | 2909520 | 0.02% | 0.26% | 22.96% |
| KERNEL | 2909520 | 0.02% | 0.24% | 20.80% |

## 25 least stable points

Candidates to stabilize, exclude from the gate, or give a per-point threshold. `spread` is (max-min)/median across the runs.

| test | marker | run_type | median | spread | abs spread | config |
| --- | --- | --- | --- | --- | --- | --- |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 9350.0 | 24.27% | 2269.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16_b, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=2, speed_of_light=False, throttle_level=0, tile_cnt=128, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 9350.0 | 24.27% | 2269.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16_b, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=4, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=2, speed_of_light=False, throttle_level=0, tile_cnt=16, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 9352.0 | 23.72% | 2218.0 | c_dimm=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=128, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 9352.0 | 23.72% | 2218.0 | c_dimm=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=4, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=16, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 9352.0 | 23.72% | 2218.0 | c_dimm=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=3, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=12, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 4704.0 | 23.51% | 1106.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=3, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=6, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 7050.0 | 23.26% | 1640.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=1, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=3, speed_of_light=False, throttle_level=0, tile_cnt=3, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 7026.0 | 22.96% | 1613.0 | c_dimm=1, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float16_b, formats.register_A=Tf32, formats.register_B=Tf32, formats.sfpu_math=Tf32, k_dimm=3, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=3, speed_of_light=False, throttle_level=0, tile_cnt=9, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 7026.0 | 22.96% | 1613.0 | c_dimm=1, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float16_b, formats.register_A=Tf32, formats.register_B=Tf32, formats.sfpu_math=Tf32, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=3, speed_of_light=False, throttle_level=0, tile_cnt=96, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 4738.0 | 22.63% | 1072.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=4, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=2, speed_of_light=False, throttle_level=0, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 4738.0 | 22.63% | 1072.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=8, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=2, speed_of_light=False, throttle_level=0, tile_cnt=16, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | KERNEL | PACK_ISOLATE | 10141.0 | 22.49% | 2281.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16_b, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=2, speed_of_light=False, throttle_level=0, tile_cnt=128, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | KERNEL | PACK_ISOLATE | 10141.0 | 22.49% | 2281.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16_b, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=4, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=2, speed_of_light=False, throttle_level=0, tile_cnt=16, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | KERNEL | PACK_ISOLATE | 10068.0 | 22.20% | 2235.0 | c_dimm=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=3, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=12, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | KERNEL | PACK_ISOLATE | 10072.0 | 22.12% | 2228.0 | c_dimm=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=128, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | KERNEL | PACK_ISOLATE | 10072.0 | 22.12% | 2228.0 | c_dimm=4, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=4, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=16, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 4701.0 | 21.76% | 1023.0 | c_dimm=1, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=2, speed_of_light=False, throttle_level=0, tile_cnt=64, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 4701.0 | 21.57% | 1014.0 | c_dimm=1, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=2, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=2, speed_of_light=False, throttle_level=0, tile_cnt=4, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 4701.0 | 21.57% | 1014.0 | c_dimm=1, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Float16_b, formats.input_B=Float16_b, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=4, loop_factor=64, math_fidelity=MathFidelity.HiFi3, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=2, speed_of_light=False, throttle_level=0, tile_cnt=8, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | KERNEL | PACK_ISOLATE | 7767.0 | 21.31% | 1655.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=1, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=3, speed_of_light=False, throttle_level=0, tile_cnt=3, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 4738.0 | 20.94% | 992.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float16_b, formats.register_A=Float16_b, formats.register_B=Float16_b, formats.sfpu_math=Float16_b, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=2, speed_of_light=False, throttle_level=0, tile_cnt=64, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | KERNEL | PACK_ISOLATE | 7734.0 | 20.80% | 1609.0 | c_dimm=1, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float16_b, formats.register_A=Tf32, formats.register_B=Tf32, formats.sfpu_math=Tf32, k_dimm=3, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=3, speed_of_light=False, throttle_level=0, tile_cnt=9, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | KERNEL | PACK_ISOLATE | 7734.0 | 20.80% | 1609.0 | c_dimm=1, dest_acc=DestAccumulation.Yes, dest_sync=DestSync.Half, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float16_b, formats.register_A=Tf32, formats.register_B=Tf32, formats.sfpu_math=Tf32, k_dimm=32, loop_factor=64, math_fidelity=MathFidelity.LoFi, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=3, speed_of_light=False, throttle_level=0, tile_cnt=96, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | KERNEL | PACK_ISOLATE | 5425.0 | 20.55% | 1115.0 | c_dimm=2, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float16, formats.input_B=Float16, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=3, loop_factor=64, math_fidelity=MathFidelity.HiFi2, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=6, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
| perf_matmul | TILE_LOOP | PACK_ISOLATE | 2430.0 | 20.25% | 492.0 | c_dimm=1, dest_acc=DestAccumulation.No, dest_sync=DestSync.Half, formats.input_A=Float32, formats.input_B=Float32, formats.output=Float16, formats.register_A=Float16, formats.register_B=Float16, formats.sfpu_math=Float16, k_dimm=2, loop_factor=64, math_fidelity=MathFidelity.HiFi4, num_faces=4, num_faces_A=4, num_faces_B=4, r_dimm=1, speed_of_light=False, throttle_level=0, tile_cnt=2, unpack_to_dest=False, unpack_transpose_faces=Transpose.No |
