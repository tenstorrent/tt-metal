# Matrix Multiply FLOPS


## Introduction

Across many families of neural networks and applications, the common denominator is the use of the generalized matrix multiply operation. Depending on the size and the precision of the input and output matrices, different underlying effects, and more importantly performance metrics, can be observed. Classically, this comes down to the hardware's ability to execute an operation, and its ability to fetch the data for that operation intercept.

If the data is small and already in registers, the cost to operate on that data is negligible. If the data is in cache, performance is dictated by how quickly the data can be funnelled thought caches to the compute units. In there worst case scenarios, the data needed is in device memory, host memory, or stored on a disk.

Thankfully, matrix multiplication requires more compute operations (2N^3) than memory operations (3n^2). As such, for a given device, there will always be points at which a device is limited by the underlying compute units, not the underlying memory system. We call this point the roofline.
However, said inversion point depends on the size and crossover point of each cache level/memory technology and the datatype in use. The amount of 8 bit elements that can be moved per unit time is nearly an order of magnitude more than 64 bit elements.

Therefore, the peak achieved flops changes based on the datatype, the size of the data, and the layout of the data.


# Test it yourself!

Assuming you have access to a device (if not, they're available for purchase at Tenstorrent.com!), you can test and see the matrix multiply TFLOPS results for yourself by running:

`pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf` (available in the tt-metal repository) on a N150 card.

Alternatively, to test on an N300 card, use the following command:

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf` on a N300 card.

To do so, make sure to have followed the setup instructions guide available at https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md

NB: You'll need to comment out `#@pytest.mark.skip(reason="WH didt hang, need to skip CI and run locally only")` line.

## Points of interest in the tests

The parameters of interest are 3 fold:
1. Dimensions: the sizes of the matrix on each edge, denoted as m, n and k
2. The fidelity of the computation, referred to as lofi, hifi2, hifi3, and hifi4. This affects how many bits of each input datatype is actually ingested during the computation.
3. Datatype of input/output space. It has been shown that a network layer need not always use all of the bits of a given datatype. But some layers do need the full resolution provided by a given data type, and its higher memory footprint.

For example, when changing the precision of the matrix, for a given size of matrix the output performance is expected to be different.

![A simple bar chart of the TFLOPS on WH when changing the precision of matrcies](images/effects_of_precision.png "Variance in performance of TFLOPS on WH from SRAM due to changing precision")



## Operations

### Matrix Multiplication

The WH matrix engine performs 8x16 x 16x16 = 8x16 in a single cycle. \
This is 2*8\*16\*16 = 4096 muladds in a single cycle. At 1GHz, this is 4 TFLOPs per matrix engine. \
The 8x16 is the smallest matrix that can be fed into in0, and 16x16 is the
smallest matrix that can be fed into in1.

If the input matrices fed into the engine are "shorter" than 8x16, for example 1x16, the engine will still perform 8x16 x 16x16 = 8x16, but the effective throughput will be 1/8.
Thus, for 1x16 x 16x16 matrices, the effective throughput is 0.5 TFLOP per matrix engine.

MATH_FIDELITY is used for higher precision, and TFLOPs are calculated by dividing by the MATH_FIDELITY value.

LoFi ->  ~4 TFLOPs \
HiFi2 -> ~2 TFLOPs \
HiFi3 -> ~1.33 TFLOPs \
HiFi4 -> ~1 TFLOPs

### Peak Machine FLOPS

Each N300s card is made up of 2 Wormhole ASICs. Each ASIC provides a usable grid of 8 * 8 tensix Cores.

Depending on the fidelity, datatype, and matrix shape chosen, different peak teraflop values can be achieved.

Below is the table generated from running the benchmark script, showcasing the performance of matrix multiplication (matmul) operations using square matrices of sizes ranging from 512x512x512 to 16384x16384x16384. The results include evaluations across various data formats, paired with different levels of math fidelity (bfloat16-HiFi2, bfloat16-HiFi4, bfloat8_b-LoFi, and bfloat4_b-LoFi).

We also show the results with and without trace (see [AdvancedPerformanceOptimizationsForModels](tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md) for details of trace). With trace, we can minimize the overhead of host which can reflect the actual device performance better.

Finally, we present the results in terms of device time, device throughput in TFLOPs, device utilization compared to the user-specified grid size and device utilization compared to the full grid size (8x8 in Wormhole). Utilization is calculated with 

Utilization = ideal cycles / actual cycles. 

Ideal cycles = (m * k * n) / (tile_height * tile_width * tile_height) * (cycle_per_tile / num_cores). 

Cycle_per_tile is the ideal compute cycle for each tile, which depends on math fidelity (LoFi: 16, HiFi2: 32, HiFi3: 48, HiFi4: 64). For utilization of user-specified grid size, num_cores is the user-specified number of cores. For utilization of full grid size, num_cores is the maximum number of cores available for compute. 


|     m |     k |     n | use_trace   | grid_size   | in0_sharded   | out_sharded   | in0_storage_type   | in1_storage_type   | out_storage_type   | dtype              | math_fidelity      |   inference_time_avg (ns) |   TFLOPs (avg) | Utilization (vs user grid)   | Utilization (vs 8x8 full grid)   |
|------:|------:|------:|:------------|:------------|:--------------|:--------------|:-------------------|:-------------------|:-------------------|:-------------------|:-------------------|--------------------------:|---------------:|:-----------------------------|:---------------------------------|
|   512 |   512 |   512 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          378654           |           0.71 | 0.54%                        | 0.54%                            |
|   512 |  1024 |  1024 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          363193           |           2.96 | 2.26%                        | 2.26%                            |
|   512 |  1024 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          362425           |           5.93 | 4.52%                        | 4.52%                            |
|  1024 |  1024 |  1024 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          360315           |           5.96 | 4.55%                        | 4.55%                            |
|  1024 |  1024 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          360370           |          11.92 | 9.09%                        | 9.09%                            |
|  1024 |  2048 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          361652           |          23.75 | 18.12%                       | 18.12%                           |
|  2048 |  2048 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          360396           |          47.67 | 36.37%                       | 36.37%                           |
|  2048 |  2048 |  3072 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          357599           |          72.06 | 54.98%                       | 54.98%                           |
|  2048 |  3072 |  3072 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          509491           |          75.87 | 57.88%                       | 57.88%                           |
|  3072 |  3072 |  3072 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          790913           |          73.31 | 55.93%                       | 55.93%                           |
|  3072 |  3072 |  4096 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi2 |               1.1777e+06  |          65.64 | 50.08%                       | 50.08%                           |
|  3072 |  4096 |  4096 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi2 |               1.44314e+06 |          71.43 | 54.49%                       | 54.49%                           |
|  4096 |  4096 |  4096 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi2 |               2.07709e+06 |          66.17 | 50.48%                       | 50.48%                           |
|  8192 |  8192 |  8192 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi2 |               1.44337e+07 |          76.18 | 58.12%                       | 58.12%                           |
| 16384 | 16384 | 16384 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi2 |               1.07906e+08 |          81.52 | 62.19%                       | 62.19%                           |
|   512 |   512 |   512 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          354750           |           0.76 | 1.15%                        | 1.15%                            |
|   512 |  1024 |  1024 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          354664           |           3.03 | 4.62%                        | 4.62%                            |
|   512 |  1024 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          357769           |           6    | 9.16%                        | 9.16%                            |
|  1024 |  1024 |  1024 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          358937           |           5.98 | 9.13%                        | 9.13%                            |
|  1024 |  1024 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          361688           |          11.87 | 18.12%                       | 18.12%                           |
|  1024 |  2048 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          365498           |          23.5  | 35.86%                       | 35.86%                           |
|  2048 |  2048 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          364318           |          47.16 | 71.95%                       | 71.95%                           |
|  2048 |  2048 |  3072 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          494421           |          52.12 | 79.53%                       | 79.53%                           |
|  2048 |  3072 |  3072 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          746684           |          51.77 | 78.99%                       | 78.99%                           |
|  3072 |  3072 |  3072 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |               1.14083e+06 |          50.82 | 77.55%                       | 77.55%                           |
|  3072 |  3072 |  4096 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi4 |               1.71129e+06 |          45.18 | 68.93%                       | 68.93%                           |
|  3072 |  4096 |  4096 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi4 |               2.20082e+06 |          46.84 | 71.47%                       | 71.47%                           |
|  4096 |  4096 |  4096 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi4 |               2.8713e+06  |          47.87 | 73.04%                       | 73.04%                           |
|  8192 |  8192 |  8192 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi4 |               2.18014e+07 |          50.43 | 76.95%                       | 76.95%                           |
| 16384 | 16384 | 16384 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi4 |               1.71243e+08 |          51.37 | 78.38%                       | 78.38%                           |
|   512 |   512 |   512 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          355988           |           0.75 | 0.29%                        | 0.29%                            |
|   512 |  1024 |  1024 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          357294           |           3.01 | 1.15%                        | 1.15%                            |
|   512 |  1024 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          355995           |           6.03 | 2.30%                        | 2.30%                            |
|  1024 |  1024 |  1024 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          359674           |           5.97 | 2.28%                        | 2.28%                            |
|  1024 |  1024 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          364969           |          11.77 | 4.49%                        | 4.49%                            |
|  1024 |  2048 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          372298           |          23.07 | 8.80%                        | 8.80%                            |
|  2048 |  2048 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          362940           |          47.34 | 18.06%                       | 18.06%                           |
|  2048 |  2048 |  3072 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          362773           |          71.04 | 27.10%                       | 27.10%                           |
|  2048 |  3072 |  3072 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          364785           |         105.97 | 40.42%                       | 40.42%                           |
|  3072 |  3072 |  3072 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          408983           |         141.77 | 54.08%                       | 54.08%                           |
|  3072 |  3072 |  4096 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          541277           |         142.83 | 54.48%                       | 54.48%                           |
|  3072 |  4096 |  4096 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          706060           |         145.99 | 55.69%                       | 55.69%                           |
|  4096 |  4096 |  4096 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT8_B | MathFidelity.LoFi  |               1.03637e+06 |         132.62 | 50.59%                       | 50.59%                           |
|  8192 |  8192 |  8192 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT8_B | MathFidelity.LoFi  |               7.4627e+06  |         147.33 | 56.20%                       | 56.20%                           |
| 16384 | 16384 | 16384 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT8_B | MathFidelity.LoFi  |               5.71668e+07 |         153.87 | 58.70%                       | 58.70%                           |
|   512 |   512 |   512 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          357399           |           0.75 | 0.29%                        | 0.29%                            |
|   512 |  1024 |  1024 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          359850           |           2.98 | 1.14%                        | 1.14%                            |
|   512 |  1024 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          358658           |           5.99 | 2.28%                        | 2.28%                            |
|  1024 |  1024 |  1024 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          359278           |           5.98 | 2.28%                        | 2.28%                            |
|  1024 |  1024 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          358381           |          11.98 | 4.57%                        | 4.57%                            |
|  1024 |  2048 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          356746           |          24.08 | 9.19%                        | 9.19%                            |
|  2048 |  2048 |  2048 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          361416           |          47.53 | 18.13%                       | 18.13%                           |
|  2048 |  2048 |  3072 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          357502           |          72.08 | 27.50%                       | 27.50%                           |
|  2048 |  3072 |  3072 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          360315           |         107.28 | 40.92%                       | 40.92%                           |
|  3072 |  3072 |  3072 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          359447           |         161.31 | 61.53%                       | 61.53%                           |
|  3072 |  3072 |  4096 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          386701           |         199.92 | 76.26%                       | 76.26%                           |
|  3072 |  4096 |  4096 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          528603           |         195    | 74.39%                       | 74.39%                           |
|  4096 |  4096 |  4096 | False       | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          704734           |         195.02 | 74.40%                       | 74.40%                           |
|  8192 |  8192 |  8192 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT4_B | MathFidelity.LoFi  |               5.80098e+06 |         189.54 | 72.30%                       | 72.30%                           |
| 16384 | 16384 | 16384 | False       | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT4_B | MathFidelity.LoFi  |               4.62353e+07 |         190.25 | 72.57%                       | 72.57%                           |
|   512 |   512 |   512 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |           13051           |          20.57 | 15.69%                       | 15.69%                           |
|   512 |  1024 |  1024 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |           29254           |          36.7  | 28.00%                       | 28.00%                           |
|   512 |  1024 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |           52824           |          40.65 | 31.02%                       | 31.02%                           |
|  1024 |  1024 |  1024 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |           36845.2         |          58.28 | 44.47%                       | 44.47%                           |
|  1024 |  1024 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |           65000.1         |          66.08 | 50.41%                       | 50.41%                           |
|  1024 |  2048 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          120201           |          71.46 | 54.52%                       | 54.52%                           |
|  2048 |  2048 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          218179           |          78.74 | 60.08%                       | 60.08%                           |
|  2048 |  2048 |  3072 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          330052           |          78.08 | 59.57%                       | 59.57%                           |
|  2048 |  3072 |  3072 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          499430           |          77.4  | 59.05%                       | 59.05%                           |
|  3072 |  3072 |  3072 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |          768714           |          75.43 | 57.55%                       | 57.55%                           |
|  3072 |  3072 |  4096 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi2 |               1.13666e+06 |          68.01 | 51.89%                       | 51.89%                           |
|  3072 |  4096 |  4096 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi2 |               1.37919e+06 |          74.74 | 57.02%                       | 57.02%                           |
|  4096 |  4096 |  4096 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi2 |               1.98296e+06 |          69.31 | 52.88%                       | 52.88%                           |
|  8192 |  8192 |  8192 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi2 |               1.3634e+07  |          80.64 | 61.53%                       | 61.53%                           |
| 16384 | 16384 | 16384 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi2 |               1.01918e+08 |          86.31 | 65.85%                       | 65.85%                           |
|   512 |   512 |   512 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |           13451.6         |          19.96 | 30.45%                       | 30.45%                           |
|   512 |  1024 |  1024 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |           30355.5         |          35.37 | 53.97%                       | 53.97%                           |
|   512 |  1024 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |           55108.1         |          38.97 | 59.46%                       | 59.46%                           |
|  1024 |  1024 |  1024 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |           46296.1         |          46.39 | 70.78%                       | 70.78%                           |
|  1024 |  1024 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |           88396.1         |          48.59 | 74.14%                       | 74.14%                           |
|  1024 |  2048 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          169585           |          50.65 | 77.29%                       | 77.29%                           |
|  2048 |  2048 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          326509           |          52.62 | 80.29%                       | 80.29%                           |
|  2048 |  2048 |  3072 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          493205           |          52.25 | 79.73%                       | 79.73%                           |
|  2048 |  3072 |  3072 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |          723646           |          53.42 | 81.51%                       | 81.51%                           |
|  3072 |  3072 |  3072 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi4 |               1.10006e+06 |          52.71 | 80.43%                       | 80.43%                           |
|  3072 |  3072 |  4096 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi4 |               1.63843e+06 |          47.19 | 72.00%                       | 72.00%                           |
|  3072 |  4096 |  4096 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi4 |               2.10337e+06 |          49.01 | 74.78%                       | 74.78%                           |
|  4096 |  4096 |  4096 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi4 |               2.71992e+06 |          50.53 | 77.10%                       | 77.10%                           |
|  8192 |  8192 |  8192 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi4 |               2.06619e+07 |          53.21 | 81.20%                       | 81.20%                           |
| 16384 | 16384 | 16384 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT16  | MathFidelity.HiFi4 |               1.61645e+08 |          54.42 | 83.03%                       | 83.03%                           |
|   512 |   512 |   512 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |           10769.4         |          24.93 | 9.51%                        | 9.51%                            |
|   512 |  1024 |  1024 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |           20384.8         |          52.67 | 20.09%                       | 20.09%                           |
|   512 |  1024 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |           32320           |          66.44 | 25.35%                       | 25.35%                           |
|  1024 |  1024 |  1024 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |           23441.3         |          91.61 | 34.95%                       | 34.95%                           |
|  1024 |  1024 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |           38425.9         |         111.77 | 42.64%                       | 42.64%                           |
|  1024 |  2048 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |           65989.5         |         130.17 | 49.66%                       | 49.66%                           |
|  2048 |  2048 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          119739           |         143.48 | 54.73%                       | 54.73%                           |
|  2048 |  2048 |  3072 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          179234           |         143.78 | 54.85%                       | 54.85%                           |
|  2048 |  3072 |  3072 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          263422           |         146.74 | 55.98%                       | 55.98%                           |
|  3072 |  3072 |  3072 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          398071           |         145.66 | 55.56%                       | 55.56%                           |
|  3072 |  3072 |  4096 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          528693           |         146.23 | 55.78%                       | 55.78%                           |
|  3072 |  4096 |  4096 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT8_B | MathFidelity.LoFi  |          689044           |         149.6  | 57.07%                       | 57.07%                           |
|  4096 |  4096 |  4096 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT8_B | MathFidelity.LoFi  |          999753           |         137.47 | 52.44%                       | 52.44%                           |
|  8192 |  8192 |  8192 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT8_B | MathFidelity.LoFi  |               7.07856e+06 |         155.33 | 59.25%                       | 59.25%                           |
| 16384 | 16384 | 16384 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT8_B | MathFidelity.LoFi  |               5.40224e+07 |         162.82 | 62.11%                       | 62.11%                           |
|   512 |   512 |   512 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |           10182.9         |          26.36 | 10.06%                       | 10.06%                           |
|   512 |  1024 |  1024 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |           17058.8         |          62.94 | 24.01%                       | 24.01%                           |
|   512 |  1024 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |           26030.5         |          82.5  | 31.47%                       | 31.47%                           |
|  1024 |  1024 |  1024 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |           19059.2         |         112.67 | 42.98%                       | 42.98%                           |
|  1024 |  1024 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |           30508           |         140.78 | 53.70%                       | 53.70%                           |
|  1024 |  2048 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |           50296.8         |         170.78 | 65.15%                       | 65.15%                           |
|  2048 |  2048 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |           90451.2         |         189.94 | 72.45%                       | 72.45%                           |
|  2048 |  2048 |  3072 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          134931           |         190.99 | 72.86%                       | 72.86%                           |
|  2048 |  3072 |  3072 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          196865           |         196.35 | 74.90%                       | 74.90%                           |
|  3072 |  3072 |  3072 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          290744           |         199.43 | 76.08%                       | 76.08%                           |
|  3072 |  3072 |  4096 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          384467           |         201.08 | 76.71%                       | 76.71%                           |
|  3072 |  4096 |  4096 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          515461           |         199.97 | 76.28%                       | 76.28%                           |
|  4096 |  4096 |  4096 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT4_B | MathFidelity.LoFi  |          682909           |         201.26 | 76.77%                       | 76.77%                           |
|  8192 |  8192 |  8192 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT4_B | MathFidelity.LoFi  |               5.52199e+06 |         199.12 | 75.96%                       | 75.96%                           |
| 16384 | 16384 | 16384 | True        | (8, 8)      | False         | False         | DRAM               | DRAM               | DRAM               | DataType.BFLOAT4_B | MathFidelity.LoFi  |               4.36456e+07 |         201.53 | 76.88%                       | 76.88%                           |


#### Square matrices

For most hardware, peak performance is achieved with square matrices that best align with the underlying hardware, for example WH performs best when using Square input matrices, we achieve highest device utilization with bfloat16 and HiFi4.

![A simple bar chart of the TFLOPS on WH when using various square matrcies](images/TFLOPS_WH_SQUARE.png "Square Matrix TFLOPS on WH from SRAM")

#### Rectangular matrices

When deviating from Square matrices, the total balance of compute can be thrown off, lowering peak performance. For example, processing matrices with equal amounts of elements, but different shapes can reduce peak TFLOPS.

Given input matrix A of 512x1024 and B of 1024x2048 to produce output matrix 512x2048 requires the same amount of computation as if the input matrices were of dimensions 1024^2. However, the performance results are measurably different:

|     m |     k |     n | use_trace   | grid_size   | in0_sharded   | out_sharded   | in0_storage_type   | in1_storage_type   | out_storage_type   | dtype              | math_fidelity      |   inference_time_avg (ns) |   TFLOPs (avg) | Utilization (vs user grid)   | Utilization (vs 8x8 full grid)   |
|------:|------:|------:|:------------|:------------|:--------------|:--------------|:-------------------|:-------------------|:-------------------|:-------------------|:-------------------|--------------------------:|---------------:|:-----------------------------|:---------------------------------|
|   512 |  1024 |  2048 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |           52824           |          40.65 | 31.02%                       | 31.02%                           |
|  1024 |  1024 |  1024 | True        | (8, 8)      | True          | True          | L1                 | DRAM               | L1                 | DataType.BFLOAT16  | MathFidelity.HiFi2 |           36845.2         |          58.28 | 44.47%                       | 44.47%       

![A simple bar chart of the TFLOPS on WH when using square vs rectangular matrcies](images/effects_of_shapes.png "Square vs rectangular Matrix TFLOPS on WH from SRAM")



### Understanding device scaling: SRAM vs DRAM

When a tensix core executes an operation, it does so by reading in data from SRAM, forwarding that to a register, executing the operation, and then writing the result back to SRAM.

Each Tensix core on a WH ASIC has ~1.5MB of SRAM. When feeding data from SRAM, each tensix can operate unencumbered. However some problems require more working memory than is available via SRAM. When this happens, Tensix will instead map data to device memory or DRAM. Accessing data from DRAM is slower both in terms of bandwidth and latency than SRAM. Simultaneously, because of the interconnected nature of the WH ASIC, a clever programmer may often find that the result of one tensix unit is what is needed for the input of another tensix core. Instead of writing that data back to device memory, the data can instead be forwarded directly over the NOC.
