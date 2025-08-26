# Matrix Multiply FLOPS


## Introduction

Machine learning computation, especially neural networks, relies heavily on linear algebra operations. One of these crucial and particularly computationally intensive operations is matrix multiplication. General Matrix Multiplication (GEMM), despite being extremely optimized in software, is still extremely time and compute intensive, and having fast matrix multiplication is key to efficient and fast inference of modern neural networks. Many core operations, such as fully connected layers, convolutions, and attention mechanisms in transformers, can be reduced to and represented as one or many GEMMs.


| Machine Learning Component      | How GEMM/matmul is used                               |
|---------------------------------|------------------------------------------------------|
| Dense Layer (MLP)               | Input × Weight matrix                                |
| Convolution       | Input patches × Filter matrix                        |
| RNN/GRU/LSTM Cell               | Input/State vector × Weight matrix                   |
| Attention (Transformers)        | Q, K, V projections; Attention scores computation    |
| Output Projection               | Hidden activations × Output weight matrix            |


If tensors used for the calculations are small and already in registers, the cost to operate on that data is negligible. If the data is in cache, performance is dictated by how quickly the data can be funnelled through caches to the compute units. In their worst case scenarios, the data needed is in device memory, host memory, or stored on a disk.

Thankfully, matrix multiplication requires more compute operations (2N^3) than memory operations (3n^2). As such, for a given device, there will always be points at which a device is limited by the underlying compute units, not the underlying memory system. We call this point the roofline.
However, this inversion point depends on the size and crossover point of each cache level/memory technology and the datatype in use. The amount of 8-bit elements that can be moved per unit time is nearly an order of magnitude more than 64-bit elements.

Therefore, the peak achieved flops changes based on the datatype, the size of the data, and the layout of the data.


## Test it yourself!

## Prerequisites

> [!IMPORTANT]
> These benchmarks require tt-metal to be built with profiler support enabled. Device-based utilization metrics are only available in profiler builds.

### Build tt-metal with Profiler Support

```bash
./build_metal.sh --enable-profiler
```
### Running Benchmarks

The matrix multiply TFLOPS results can be tested on any Wormhole or Blackhole card using:

**For manually selected matmul configurations (best performance):**
```bash
TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf
```

**For out-of-box matmul configurations (default settings):**
```bash
TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf_out_of_box
```

**For comprehensive sweep (to run a sweep to find the ideal config):**
```bash
TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf_sweep_all
```

### N300 Card Instructions

Alternatively, to test on an N300 card, use the following commands. 

**For manually selected matmul configurations:**
```bash
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf
```

**For out-of-box matmul configurations:**
```bash
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf_out_of_box
```

**For comprehensive sweep:**
```bash
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml TT_METAL_DEVICE_PROFILER=1 pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf_sweep_all
```

Python scripts for reproducing the plots are included in this directory.

## Design of Experiments
The parameters of interest are 3 fold:
1. Dimensions: The sizes of the matrices along each axis, denoted as m , n and k . (m, k) represents the size of the input tensor, while (k, n) is the size of the activation tensor. 
Larger tensors require more computation since the number of operations needed to perform matrix multiplication increases as O(m*k*n).
2. Computation Fidelity: Referred to as LoFi, HiFi2, HiFi3 and HiFi4. Internally, the matrix engine can adjust the number of bits being processed, which affects both the precision of the results and the computation speed.
3. Input/Output Datatype: Larger datatypes require more memory for storage. As a result, more precise datatypes can become bottlenecked if stored in DRAM.

For more details please refer to the tech reports [Matrix Engine](../matrix_engine/matrix_engine.md) and [Data Formats](../data_formats/data_formats.md)

For example, when changing the precision of the matrix, for a given size of matrix the output performance is expected to be different.


## MicroBenchmarks

### Matrix Multiplication TFLOPS on Wormhole/Blackhole (WH/BH)

The WH matrix engine performs 8x16 x 16x16 = 8x16 in a single cycle.
- This is 2*8\*16\*16 = 4096 multiply-adds in a single cycle.
- At 1GHz, this is 4 TFLOPS per matrix engine.fed into in0, and 16x16 is the smallest matrix that can be fed into in1.

If the input matrices fed into the engine are "shorter" than 8x16, for example 1x16, the engine will still perform 8x16 x 16x16 = 8x16, but the effective throughput will be 1/8.
Thus, for 1x16 x 16x16 matrices, the effective throughput is 0.5 TFLOPS per matrix engine.
MATH_FIDELITY is used for higher precision, and TFLOPS are calculated by dividing by the MATH_FIDELITY value.

- LoFi ->  ~4 TFLOPS
- HiFi2 -> ~2 TFLOPS
- HiFi3 -> ~1.33 TFLOPS
- HiFi4 -> ~1 TFLOPS


### Utilization Derivation Formula
```
Utilization = ideal cycles / actual cycles. tile_width * tile_height) * (cycle_per_tile / num_cores)
```
- Cycle_per_tile is the ideal compute cycle for each tile, which depends on math fidelity (LoFi: 16, HiFi2: 32, HiFi3: 48, HiFi4: 64).
- For utilization of full grid size, num_cores is the maximum number of cores available for compute. Currently the max for Wormhole is 8x8 with Blackhole supporting up to 13x10.

### Manually Tuned Performance
Here we show the peak results we can get from manually selected matmul configurations, including packer L1 enablement, math fidelity, input/output sharding, and input/output L1/DRAM selection.
#### Peak FLOPS

Depending on the fidelity, datatype, and matrix shape chosen, different peak teraflop values can be achieved.
Below is the results generated from running the benchmark script, showcasing the performance of matrix multiplication (matmul) operations using matrices of sizes ranging from 512x512x512/640,832,832 to 16384x16384x16384/20480,26624,26624 . The results include evaluations across various data formats, paired with different levels of math fidelity (bfloat16-HiFi2, bfloat16-HiFi4,  bfloat8_b-HiFi2, bfloat8_b-LoFi, and bfloat4_b-LoFi).
We also show the results with and without trace (see [AdvancedPerformanceOptimizationsForModels](../AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md) for details of trace). With trace, we can minimize the overhead of host which can reflect the actual device performance better.
Finally, we present the results in terms of device time, device throughput in TFLOPS, device utilization compared to the user-specified grid size and device utilization compared to the full grid size (8x8 in Wormhole/13x10 in Blackhole).
As seen below, while Wormhole cards can perform matrix multiplications at around 180TFlops, Blackhole cards have even more impressive throughput at 560TFlop. Lower fidelity computations with less precise datatypes computations complete faster than "full fidelity" Float16 computations. HiFi2/BFloat8 is roughly **1.5x to 1.8x faster** than HiFi4/Float16, with LoFi/Float4 coming in at **2x to 3.5x** faster without tracing.
#### Performance scatter plot across all matrix sizes and configurations

![](images/flops_vs_matrix_elements_comparison.png)


#### Performance bar plot across all matrix sizes and configurations

![](images/flops_by_matrix_size_and_type_sorted.png)

### Utilization

#### Utilization plot across all matrix sizes and configurations, based on the chip TFLOPS calculated per each Math Fidelity

Wormhole cards more easily achieve full utilization of their Tensix cores, performing at up to 83% of the theoretical value. While the larger core grid of Blackhole makes it harder to achieve high utilization, it still delivers performance north of 74% utilization at peak.

![](images/utilization_vs_matrix_elements_comparison.png)

### Understanding Device Scaling: SRAM vs DRAM

When a Tensix core executes an operation, it reads data from SRAM, forwards it to a register, performs the computation, and then writes the result back to SRAM. Each Tensix core on a WH ASIC has approximately 1.5MB of SRAM. When data fits within this SRAM, each Tensix can operate without contention. However, some problems require more working memory than SRAM can provide. In these cases, the Tensix core will instead map data to device memory or DRAM. Accessing data from DRAM is slower than SRAM, both in terms of bandwidth and latency.

In this report, the developed Python scripts evaluate three separate configurations:
1. All matrices stored on L1 (SRAM)
2. One matrix on L1 and one on DRAM
3. Both matrices on DRAM

In most cases, storing all matrices on L1 is ideal, as it completely avoids accessing the slower DRAM. The configuration with one matrix on L1 and one on DRAM incurs a small performance penalty, typically in the single-digit percentage range at worst. DRAM-only performance is highly variable: small matrices suffer the largest performance penalty when stored in DRAM, while larger tensors achieve performance closer to an L1-only configuration.

### Tracing

Tracing in the TT-Metallium stack is a performance optimization that records commands for dispatching operations into the DRAM buffer and replays them later for execution, removing host overhead of dispatching operations during a loop iteration.

#### Tracing on N150

![](images/trace_comparison_n150.png)

#### Tracing on P150

![](images/trace_comparison_p150.png)

As shown here, on both Wormhole and Blackhole, Trace helps recover more lost performance on smaller tensor matrix multiplications compared to larger ones. This is likely because smaller matrix operations take less time to execute than larger ones, meaning that host overhead is, percentage-wise, more harmful to overall runtime and maximum throughput compared to larger tensors.


### Out of the Box Performance

In this tech report, we fine-tuned the parameters of each matrix multiplication kernel to extract the maximum possible performance. However, here, we will compare the performance loss with the default configuration to demonstrate the hardware’s ability to perform well without significant hand tuning.


#### OOB vs Hand Tuned on N150

![](images/oob_comparison_n150.png)

#### OOB vs Hand Tuned on P150

![](images/oob_comparison_p150.png)

As shown here, on both Wormhole and Blackhole, hand tuned configs helps recover more lost performance on smaller tensor matrix multiplications compared to larger ones. Similar to tracing, the configuration matters more for smaller tensors, as it is harder to saturate the core grid with smaller workloads compared to larger ones.




### Rectangular Matrix

Both architectures perform most ideally when the input tensors are closest to square shapes, but they still perform well on rectangular matrices. However, as the tensors become more rectangular, performance takes a larger hit.


#### Rectangular Matrix on N150

![](images/aspect_ratio_comparison_n150.png)

#### Rectangular Matrix on P150

![](images/aspect_ratio_comparison_p150.png)

### All Data

This section contains the best performing configurations for each tensor size. For each unique matrix dimension (M×K×N), only the configuration achieving the highest TFLOPS is shown.


<details>
<summary><strong>N150 Best Configurations</strong> (click to expand)</summary>

| M | K | N | Aspect Ratio M | Aspect Ratio K | Aspect Ratio N | Use Trace | Grid Size | Dtype | Math Fidelity | Tflops | In0 Sharded | Out Sharded | In0 Block W Div | Num Out Blocks H | Num Out Blocks W | In0 Block W | Out Subblock H | Out Subblock W | Out Block H | Out Block W | Per Core M | Per Core N | Inference Time Avg | Host Based Utilization (Vs User Selected Grid 8X8) | Host Based Utilization (Vs Full Available Grid 8X8) | Device Based Utilization (Vs User Selected Grid 8X8) | Device Based Utilization (Vs Full Available Grid 8X8) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 512 | 512 | 1 | 1 | 1 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 7.04 | False | False | 1 | 2 | 1 | 2 | 1 | 2 | 1 | 2 | 2 | 2 | 38154.92 | 2.68 | 2.68 | 2683795435.95 | 2683795435.95 |
| 512 | 1024 | 1024 | 1 | 1 | 1 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 29.47 | False | False | 1 | 1 | 1 | 4 | 2 | 4 | 2 | 4 | 2 | 4 | 36430.36 | 11.24 | 11.24 | 11243369884.82 | 11243369884.82 |
| 512 | 1024 | 2048 | 1 | 1 | 1 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 56.15 | False | False | 1 | 1 | 1 | 4 | 2 | 4 | 2 | 8 | 2 | 8 | 38242.34 | 21.42 | 21.42 | 21421283271.82 | 21421283271.82 |
| 512 | 1024 | 4096 | 1 | 2 | 8 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 67.83 | False | False | 1 | 1 | 1 | 4 | 2 | 4 | 2 | 16 | 2 | 16 | 63323.97 | 25.87 | 25.87 | 25873296963.86 | 25873296963.86 |
| 512 | 2048 | 4096 | 1 | 2 | 4 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 89.14 | False | False | 1 | 1 | 1 | 8 | 2 | 4 | 2 | 16 | 2 | 16 | 96368.79 | 34.00 | 34.00 | 34002709913.90 | 34002709913.90 |
| 512 | 2048 | 8192 | 1 | 2 | 8 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 95.10 | False | False | 1 | 1 | 1 | 8 | 2 | 4 | 2 | 32 | 2 | 32 | 180649.76 | 36.28 | 36.28 | 36277934135.41 | 36277934135.41 |
| 512 | 2048 | 16384 | 1 | 2 | 8 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 97.98 | False | False | 1 | 1 | 1 | 8 | 2 | 4 | 2 | 64 | 2 | 64 | 350666.05 | 37.38 | 37.38 | 37378012910.52 | 37378012910.52 |
| 1024 | 1024 | 1024 | 1 | 1 | 1 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 47.56 | False | False | 1 | 1 | 1 | 4 | 4 | 2 | 4 | 4 | 4 | 4 | 45156.48 | 18.14 | 18.14 | 18141361334.74 | 18141361334.74 |
| 1024 | 1024 | 2048 | 1 | 1 | 1 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 83.44 | False | False | 1 | 1 | 1 | 4 | 4 | 2 | 4 | 8 | 4 | 8 | 51474.57 | 31.83 | 31.83 | 31829308353.87 | 31829308353.87 |
| 1024 | 2048 | 2048 | 1 | 1 | 1 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 116.67 | False | False | 1 | 1 | 1 | 8 | 4 | 2 | 4 | 8 | 4 | 8 | 73623.66 | 44.51 | 44.51 | 44507433119.17 | 44507433119.17 |
| 1024 | 2048 | 4096 | 1 | 2 | 4 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 134.36 | False | False | 1 | 1 | 1 | 8 | 4 | 2 | 4 | 16 | 4 | 16 | 127863.88 | 51.25 | 51.25 | 51254504371.43 | 51254504371.43 |
| 1024 | 2048 | 8192 | 1 | 2 | 8 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 142.89 | False | False | 1 | 1 | 1 | 8 | 4 | 2 | 4 | 32 | 4 | 32 | 240468.98 | 54.51 | 54.51 | 54506822713.46 | 54506822713.46 |
| 1024 | 2048 | 16384 | 1 | 2 | 8 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 137.64 | False | False | 1 | 1 | 2 | 8 | 4 | 2 | 4 | 32 | 4 | 64 | 499272.35 | 52.51 | 52.51 | 52505211201.76 | 52505211201.76 |
| 1024 | 4096 | 8192 | 1 | 2 | 4 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 161.08 | False | False | 1 | 1 | 1 | 16 | 4 | 2 | 4 | 32 | 4 | 32 | 426626.21 | 61.45 | 61.45 | 61445826968.59 | 61445826968.59 |
| 1024 | 4096 | 16384 | 1 | 2 | 8 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 168.95 | False | False | 1 | 1 | 2 | 16 | 4 | 2 | 4 | 32 | 4 | 64 | 813484.19 | 64.45 | 64.45 | 64449685098.24 | 64449685098.24 |
| 2048 | 2048 | 2048 | 1 | 1 | 1 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 141.07 | False | False | 1 | 1 | 1 | 8 | 4 | 2 | 8 | 8 | 8 | 8 | 121784.21 | 53.81 | 53.81 | 53813215924.82 | 53813215924.82 |
| 2048 | 2048 | 3072 | 1 | 1 | 1 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 136.32 | False | False | 1 | 1 | 1 | 8 | 4 | 2 | 8 | 12 | 8 | 12 | 189042.09 | 52.00 | 52.00 | 52001117469.54 | 52001117469.54 |
| 2048 | 3072 | 3072 | 1 | 1 | 1 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 154.00 | False | False | 1 | 1 | 1 | 12 | 4 | 2 | 8 | 12 | 8 | 12 | 251007.08 | 58.75 | 58.75 | 58745753288.75 | 58745753288.75 |
| 2048 | 4096 | 8192 | 1 | 2 | 4 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 164.14 | False | False | 1 | 1 | 4 | 16 | 4 | 2 | 8 | 8 | 8 | 32 | 837326.05 | 62.61 | 62.61 | 62614557390.43 | 62614557390.43 |
| 2048 | 4096 | 12288 | 1 | 2 | 4 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 167.23 | False | False | 1 | 1 | 4 | 16 | 4 | 2 | 8 | 12 | 8 | 48 | 1232767.11 | 63.79 | 63.79 | 63794044856.07 | 63794044856.07 |
| 2048 | 4096 | 16384 | 1 | 2 | 8 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 172.61 | False | False | 1 | 1 | 4 | 16 | 4 | 2 | 8 | 16 | 8 | 64 | 1592493.06 | 65.84 | 65.84 | 65844933842.92 | 65844933842.92 |
| 2048 | 4096 | 24576 | 1 | 2 | 8 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 172.10 | False | False | 1 | 1 | 4 | 16 | 4 | 2 | 8 | 24 | 8 | 96 | 2395772.93 | 65.65 | 65.65 | 65651630741.16 | 65651630741.16 |
| 2048 | 6144 | 12288 | 1 | 2 | 4 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 174.95 | False | False | 1 | 1 | 2 | 24 | 4 | 2 | 8 | 24 | 8 | 48 | 1767611.50 | 66.74 | 66.74 | 66736836550.16 | 66736836550.16 |
| 2048 | 6144 | 24576 | 1 | 2 | 8 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 181.70 | False | False | 1 | 1 | 4 | 24 | 4 | 2 | 8 | 24 | 8 | 96 | 3403759.00 | 69.31 | 69.31 | 69314425555.35 | 69314425555.35 |
| 3072 | 3072 | 3072 | 1 | 1 | 1 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 153.53 | False | False | 1 | 1 | 1 | 12 | 4 | 2 | 12 | 12 | 12 | 12 | 377655.03 | 58.57 | 58.57 | 58567735854.55 | 58567735854.55 |
| 3072 | 3072 | 4096 | 1 | 1 | 1 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 163.54 | False | False | 1 | 1 | 2 | 12 | 4 | 2 | 12 | 8 | 12 | 16 | 472736.36 | 62.38 | 62.38 | 62384031735.32 | 62384031735.32 |
| 3072 | 4096 | 4096 | 1 | 1 | 1 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 171.29 | False | False | 1 | 1 | 1 | 16 | 4 | 2 | 12 | 16 | 12 | 16 | 601768.49 | 65.34 | 65.34 | 65343401016.80 | 65343401016.80 |
| 3072 | 6144 | 12288 | 1 | 2 | 4 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 180.08 | False | False | 1 | 1 | 4 | 24 | 4 | 2 | 12 | 12 | 12 | 48 | 2575874.33 | 68.69 | 68.69 | 68694034501.00 | 68694034501.00 |
| 3072 | 6144 | 16384 | 1 | 2 | 4 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 182.69 | False | False | 1 | 1 | 4 | 24 | 4 | 2 | 12 | 16 | 12 | 64 | 3385353.09 | 69.69 | 69.69 | 69691282959.49 | 69691282959.49 |
| 3072 | 6144 | 24576 | 1 | 2 | 8 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 180.88 | False | False | 1 | 1 | 8 | 24 | 4 | 2 | 12 | 12 | 12 | 96 | 5128955.84 | 69.00 | 69.00 | 68999307259.89 | 68999307259.89 |
| 3072 | 6144 | 32768 | 1 | 2 | 8 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 184.87 | False | False | 1 | 1 | 8 | 24 | 4 | 2 | 12 | 16 | 12 | 128 | 6690835.95 | 70.52 | 70.52 | 70523205669.90 | 70523205669.90 |
| 3072 | 8192 | 16384 | 1 | 2 | 4 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 183.01 | False | False | 1 | 1 | 8 | 32 | 4 | 2 | 12 | 8 | 12 | 64 | 4505848.88 | 69.81 | 69.81 | 69814325348.63 | 69814325348.63 |
| 3072 | 8192 | 32768 | 1 | 2 | 8 | False | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 178.04 | False | False | 1 | 2 | 8 | 32 | 2 | 4 | 6 | 16 | 12 | 128 | 9263563.16 | 67.92 | 67.92 | 67916155953.86 | 67916155953.86 |
| 4096 | 4096 | 4096 | 1 | 1 | 1 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 171.33 | False | False | 1 | 1 | 1 | 16 | 4 | 2 | 16 | 16 | 16 | 16 | 802183.15 | 65.36 | 65.36 | 65357642975.45 | 65357642975.45 |
| 4096 | 8192 | 16384 | 1 | 2 | 4 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 184.62 | False | False | 1 | 1 | 8 | 32 | 4 | 2 | 16 | 8 | 16 | 64 | 5955553.05 | 70.43 | 70.43 | 70426775840.96 | 70426775840.96 |
| 4096 | 8192 | 32768 | 1 | 2 | 8 | True | 8x8 | DataType.BFLOAT4_B | MathFidelity.LoFi | 183.39 | False | False | 1 | 2 | 8 | 32 | 4 | 2 | 8 | 16 | 16 | 128 | 11990737.92 | 69.96 | 69.96 | 69959063899.47 | 69959063899.47 |

_Best configurations only: 36 unique tensor sizes (highest TFLOPS per size)._

</details>


<details>
<summary><strong>P150 Best Configurations</strong> (click to expand)</summary>

| M | K | N | Aspect Ratio M | Aspect Ratio K | Aspect Ratio N | Use Trace | Grid Size | Dtype | Math Fidelity | Tflops | In0 Sharded | Out Sharded | In0 Block W Div | Num Out Blocks H | Num Out Blocks W | In0 Block W | Out Subblock H | Out Subblock W | Out Block H | Out Block W | Per Core M | Per Core N | Inference Time Avg | Host Based Utilization (Vs User Selected Grid 13X10) | Host Based Utilization (Vs Full Available Grid 13X10) | Device Based Utilization (Vs User Selected Grid 13X10) | Device Based Utilization (Vs Full Available Grid 13X10) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 640 | 832 | 832 | 1 | 1 | 1 | False | 13x10 | DataType.BFLOAT16 | MathFidelity.HiFi4 | 37.13 | False | False | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 23865.70 | 20.66 | 20.66 | 27889398025.97 | 27889398025.97 |
| 640 | 1664 | 1664 | 1 | 1 | 1 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 139.58 | False | False | 1 | 1 | 1 | 4 | 2 | 4 | 2 | 4 | 2 | 4 | 25391.58 | 19.42 | 19.42 | 26213415421.60 | 26213415421.60 |
| 640 | 1664 | 3328 | 1 | 1 | 1 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 197.94 | False | False | 1 | 1 | 1 | 4 | 2 | 4 | 2 | 8 | 2 | 8 | 35810.47 | 27.54 | 27.54 | 37173485251.66 | 37173485251.66 |
| 640 | 1664 | 6656 | 1 | 2 | 8 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 259.43 | False | False | 1 | 1 | 1 | 4 | 2 | 4 | 2 | 16 | 2 | 16 | 54645.54 | 36.09 | 36.09 | 48721269500.87 | 48721269500.87 |
| 640 | 3328 | 6656 | 1 | 2 | 4 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 329.79 | False | False | 1 | 1 | 1 | 8 | 2 | 4 | 2 | 16 | 2 | 16 | 85973.74 | 45.88 | 45.88 | 61935191179.15 | 61935191179.15 |
| 640 | 3328 | 13312 | 1 | 2 | 8 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 357.39 | False | False | 1 | 1 | 1 | 8 | 2 | 4 | 2 | 32 | 2 | 32 | 158667.56 | 49.72 | 49.72 | 67118947976.56 | 67118947976.56 |
| 640 | 3328 | 26624 | 1 | 2 | 8 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 375.71 | False | False | 1 | 1 | 1 | 8 | 2 | 4 | 2 | 64 | 2 | 64 | 301861.76 | 52.27 | 52.27 | 70559450088.30 | 70559450088.30 |
| 1280 | 1664 | 1664 | 1 | 1 | 1 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 247.14 | False | False | 1 | 1 | 1 | 4 | 4 | 2 | 4 | 4 | 4 | 4 | 28681.76 | 34.38 | 34.38 | 46412780422.28 | 46412780422.28 |
| 1280 | 1664 | 3328 | 1 | 1 | 1 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 319.69 | False | False | 1 | 1 | 1 | 4 | 4 | 2 | 4 | 8 | 4 | 8 | 44345.86 | 44.47 | 44.47 | 60037177255.91 | 60037177255.91 |
| 1280 | 3328 | 3328 | 1 | 1 | 1 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 413.50 | False | False | 1 | 1 | 1 | 8 | 4 | 2 | 4 | 8 | 4 | 8 | 68569.18 | 57.52 | 57.52 | 77655876005.56 | 77655876005.56 |
| 1280 | 3328 | 6656 | 1 | 2 | 4 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 480.79 | False | False | 1 | 1 | 1 | 8 | 4 | 2 | 4 | 16 | 4 | 16 | 117945.67 | 66.88 | 66.88 | 90292419402.47 | 90292419402.47 |
| 1280 | 3328 | 13312 | 1 | 2 | 4 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 497.85 | False | False | 1 | 1 | 1 | 8 | 4 | 2 | 4 | 32 | 4 | 32 | 227808.95 | 69.26 | 69.26 | 93495886715.65 | 93495886715.65 |
| 1280 | 3328 | 26624 | 1 | 2 | 8 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 500.18 | False | False | 1 | 1 | 1 | 8 | 4 | 2 | 4 | 64 | 4 | 64 | 453495.98 | 69.58 | 69.58 | 93933357611.90 | 93933357611.90 |
| 1280 | 6656 | 13312 | 1 | 2 | 4 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 527.87 | False | False | 1 | 1 | 2 | 16 | 4 | 2 | 4 | 16 | 4 | 32 | 429701.81 | 73.43 | 73.43 | 99134794159.46 | 99134794159.46 |
| 1280 | 6656 | 26624 | 1 | 2 | 8 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 527.38 | False | False | 1 | 1 | 2 | 16 | 4 | 2 | 4 | 32 | 4 | 64 | 860214.23 | 73.36 | 73.36 | 99041374453.22 | 99041374453.22 |
| 2560 | 3328 | 3328 | 1 | 1 | 1 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 482.54 | False | False | 1 | 1 | 1 | 8 | 4 | 2 | 8 | 8 | 8 | 8 | 117516.52 | 67.13 | 67.13 | 90622154348.55 | 90622154348.55 |
| 2560 | 3328 | 4992 | 1 | 1 | 1 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 509.02 | False | False | 1 | 1 | 1 | 8 | 4 | 2 | 8 | 12 | 8 | 12 | 167107.58 | 70.81 | 70.81 | 95593508086.17 | 95593508086.17 |
| 2560 | 4992 | 4992 | 1 | 1 | 1 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 536.93 | False | False | 1 | 1 | 1 | 12 | 4 | 2 | 8 | 12 | 8 | 12 | 237631.80 | 74.69 | 74.69 | 100834990194.04 | 100834990194.04 |
| 2560 | 6656 | 13312 | 1 | 2 | 4 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 546.71 | False | False | 1 | 1 | 1 | 16 | 4 | 2 | 8 | 32 | 8 | 32 | 829792.02 | 76.05 | 76.05 | 102672474148.72 | 102672474148.72 |
| 2560 | 6656 | 19968 | 1 | 2 | 4 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 550.74 | False | False | 1 | 1 | 2 | 16 | 4 | 2 | 8 | 24 | 8 | 48 | 1235580.44 | 76.61 | 76.61 | 103429283447.98 | 103429283447.98 |
| 2560 | 6656 | 26624 | 1 | 2 | 8 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 550.40 | False | False | 1 | 1 | 2 | 16 | 4 | 2 | 8 | 32 | 8 | 64 | 1648449.90 | 76.57 | 76.57 | 103365956242.23 | 103365956242.23 |
| 2560 | 6656 | 39936 | 1 | 2 | 8 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 552.67 | False | False | 1 | 1 | 4 | 16 | 4 | 2 | 8 | 24 | 8 | 96 | 2462530.14 | 76.88 | 76.88 | 103791785632.28 | 103791785632.28 |
| 2560 | 9984 | 19968 | 1 | 2 | 4 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 564.81 | False | False | 1 | 1 | 2 | 24 | 4 | 2 | 8 | 24 | 8 | 48 | 1807188.99 | 78.57 | 78.57 | 106072359504.90 | 106072359504.90 |
| 2560 | 9984 | 39936 | 1 | 2 | 8 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 567.02 | False | False | 1 | 1 | 8 | 24 | 4 | 2 | 8 | 12 | 8 | 96 | 3600299.36 | 78.88 | 78.88 | 106487145050.57 | 106487145050.57 |
| 3840 | 4992 | 4992 | 1 | 1 | 1 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 562.14 | False | False | 1 | 1 | 1 | 12 | 4 | 2 | 12 | 12 | 12 | 12 | 340461.73 | 78.20 | 78.20 | 105569574292.44 | 105569574292.44 |
| 3840 | 4992 | 6656 | 1 | 1 | 1 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 535.77 | False | False | 1 | 1 | 2 | 12 | 4 | 2 | 12 | 8 | 12 | 16 | 476288.80 | 74.53 | 74.53 | 100617945363.57 | 100617945363.57 |
| 3840 | 6656 | 6656 | 1 | 1 | 1 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 548.18 | False | False | 1 | 1 | 2 | 16 | 4 | 2 | 12 | 8 | 12 | 16 | 620675.09 | 76.26 | 76.26 | 102948549637.15 | 102948549637.15 |
| 3840 | 9984 | 19968 | 1 | 2 | 4 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 565.37 | False | False | 1 | 1 | 8 | 24 | 4 | 2 | 12 | 6 | 12 | 48 | 2708125.11 | 78.65 | 78.65 | 106176482935.27 | 106176482935.27 |
| 3840 | 9984 | 26624 | 1 | 2 | 4 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 570.37 | False | False | 1 | 1 | 4 | 24 | 4 | 2 | 12 | 16 | 12 | 64 | 3579163.55 | 79.35 | 79.35 | 107115976820.19 | 107115976820.19 |
| 3840 | 9984 | 39936 | 1 | 2 | 8 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 566.98 | False | False | 1 | 1 | 8 | 24 | 4 | 2 | 12 | 12 | 12 | 96 | 5400896.07 | 78.87 | 78.87 | 106478331056.97 | 106478331056.97 |
| 3840 | 9984 | 53248 | 1 | 2 | 8 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 567.08 | False | False | 1 | 1 | 8 | 24 | 4 | 2 | 12 | 16 | 12 | 128 | 7199859.62 | 78.89 | 78.89 | 106498076429.37 | 106498076429.37 |
| 3840 | 13312 | 26624 | 1 | 2 | 4 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 572.53 | False | False | 1 | 1 | 8 | 32 | 4 | 2 | 12 | 8 | 12 | 64 | 4754185.68 | 79.65 | 79.65 | 107522262438.92 | 107522262438.92 |
| 3840 | 13312 | 53248 | 1 | 2 | 8 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 541.85 | False | False | 1 | 2 | 8 | 32 | 2 | 4 | 6 | 16 | 12 | 128 | 10046815.87 | 75.38 | 75.38 | 101759762795.07 | 101759762795.07 |
| 5120 | 6656 | 6656 | 1 | 1 | 1 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 549.12 | False | False | 1 | 1 | 1 | 16 | 4 | 2 | 16 | 16 | 16 | 16 | 826144.22 | 76.39 | 76.39 | 103125820041.90 | 103125820041.90 |
| 5120 | 13312 | 26624 | 1 | 2 | 4 | False | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 574.52 | False | False | 1 | 1 | 8 | 32 | 4 | 2 | 16 | 8 | 16 | 64 | 6317019.46 | 79.92 | 79.92 | 107894934317.81 | 107894934317.81 |
| 5120 | 13312 | 53248 | 1 | 2 | 8 | True | 13x10 | DataType.BFLOAT4_B | MathFidelity.LoFi | 566.39 | False | False | 1 | 2 | 8 | 32 | 4 | 2 | 8 | 16 | 16 | 128 | 12815284.73 | 78.79 | 78.79 | 106368982728.48 | 106368982728.48 |

_Best configurations only: 36 unique tensor sizes (highest TFLOPS per size)._

</details>
