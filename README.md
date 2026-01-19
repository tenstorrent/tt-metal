[![tt-metal CI](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml/badge.svg)](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tenstorrent/tt-metal)

<div align="center">

<h1>

[Hardware](https://tenstorrent.com/hardware/blackhole) | [Install](./INSTALLING.md) |  [Discord](https://discord.gg/tvhGzHQwaj) | [Join Us](https://boards.greenhouse.io/tenstorrent/jobs/4155609007) | [Bounty $](https://github.com/tenstorrent/tt-metal/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty)

</h1>

<img src="https://raw.githubusercontent.com/tenstorrent/tt-metal/main/docs/source/common/_static/tt_nn_w_logo.png" alt="ttnn logo" height="180"/>

**TT-NN** is a Python & C++ Neural Network OP library.

<h3>

[API Reference](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html) | [Model Demos](./models/demos/)

</h3>

</div>

## Featured Models

The Models team is focused on developing the following models, optimizing them for performance, accuracy, and compatibility. Follow each model link for more details.

>[!IMPORTANT]
> For a **full model list** see the **[Model Matrix](https://github.com/tenstorrent/tt-metal/tree/main/models/README.md)**, or visit the **[Developer Hub](https://tenstorrent.com/developers)**.

>[!NOTE]
> Performance Metrics:
> - Time to First Token (TTFT) measures the time (in milliseconds) it takes to generate the first output token after input is received.
> - T/S/U (Tokens per Second per User): Represents the throughput of first-token generation after prefill. It is calculated as 1 / inter-token latency.
> - T/S (Tokens per Second): Represents total token throughput, calculated as T/S = T/S/U x batch size.
> - TP (Tensor Parallel) and DP (Data Parallel): Indicate the parallelization factors across multiple devices.
> - Reported LLM Performance: Based on an input sequence length of 128 tokens for all models.
> - Performance Data Source: Metrics were collected using the tt-metal model demos (linked above). Results may vary when using other runtimes such as the vLLM inference server.

### [Llama 3.3 70B (TP=32)](./models/demos/llama3_70b_galaxy)
| Batch | Hardware | TTFT (MS) | T/S/U | Target<br>T/S/U | T/S | TT-Metalium Release | vLLM Tenstorrent Repo Release |
|-------|----------|-----------|-------|-----------------|-----|---------------------|-------------------------------|
| 32    | [Galaxy (Wormhole)](https://tenstorrent.com/hardware/galaxy) | 53      | 72.5  | 80              | 2268.8  | [v0.65.0-rc7](https://github.com/tenstorrent/tt-metal/tree/v0.65.0-rc7) | [59be953](https://github.com/tenstorrent/vllm/tree/59be953f2bbd21e227f9ef4b779f545f9c3bf599/tt_metal) |

### [Qwen 2.5 7B (TP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)
| Batch | Hardware | TTFT (MS) | T/S/U | Target<br>T/S/U | T/S  | TT-Metalium Release | vLLM Tenstorrent Repo Release |
|-------|----------|-----------|-------|-----------------|------|---------------------|-------------------------------|
| 32 | [n300 (Wormhole)](https://tenstorrent.com/hardware/wormhole) | 109 | 22.1 | 30 | 707.2 | [v0.62.0-rc35](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc35) | [ced0161](https://github.com/tenstorrent/vllm/tree/ced0161dc223e6d8aca5f44a6c43d13070c3fba6/tt_metal) |

### [Qwen 2.5 72B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)
| Batch | Hardware | TTFT (MS) | T/S/U | Target<br>T/S/U | T/S | TT-Metalium Release | vLLM Tenstorrent Repo Release |
|-------|----------|-----------|-------|-----------------|-----|---------------------|-------------------------------|
| 32 | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 223 | 15.4 | 20 | 492.8 | [v0.62.0-rc25](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc25) | [e7c329b](https://github.com/tenstorrent/vllm/tree/e7c329b1664f8591ae8b4269bed9690726e52a24/tt_metal) |

### [Whisper (distil-large-v3)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/whisper)
| Batch | Hardware | TTFT (MS) | T/S/U | Target<br>T/S/U | T/S | TT-Metalium Release |
|-------|----------|-----------|-------|-----------------|-----|---------------------|
| 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 163       | 105.0  | 45           | 105.0   | [v0.65.0-dev20251208](https://github.com/tenstorrent/tt-metal/tree/v0.65.0-dev20251208) |
| 1     | [p150 (Blackhole)](https://tenstorrent.com/hardware/blackhole)        | 63       | 263.4  |            | 263.4   | [v0.65.0-dev20251208](https://github.com/tenstorrent/tt-metal/tree/v0.65.0-dev20251208) |

### [Mixtral 8x7B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)
| Batch | Hardware | TTFT (MS) | T/S/U | Target<br>T/S/U | T/S | TT-Metalium Release |
|-------|----------|-----------|-------|-----------------|-----|---------------------|
| 32 | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 122 | 24.9 | 33 | 796.8 | [v0.62.0-dev20251015](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-dev20251015) |

Blackhole software optimization is under active development.  Please join us in shaping the future of open source AI! <br> [\[Discord\]](https://discord.gg/tenstorrent) [\[Developer Hub\]](https://tenstorrent.com/developers)

For more information regarding vLLM installation and environment creation visit the [Tenstorrent vLLM repository](https://github.com/tenstorrent/vllm/blob/dev/tt_metal/README.md).

## Model Updates

For the latest model updates and features, please see [MODEL_UPDATES.md](models/docs/MODEL_UPDATES.md)

## Model Bring-Up and Testing

For information on initial model procedures, please see [Model Bring-Up and Testing](https://github.com/tenstorrent/tt-metal/blob/main/models/docs/model_bring_up.md)

## TT-NN Tech Reports

- [Advanced Performance Optimizations for Models](./tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md) (updated March 4th, 2025)
- [Programming Mesh of Devices](./tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md) (updated Sept 9th, 2024)
- [ViT Implementation in TT-NN on GS](./tech_reports/ViT-TTNN/vit.md)  (updated Sept 22nd, 2024)
- [LLMs Bring up in TT-NN](./tech_reports/LLMs/llms.md)  (updated Oct 29th, 2024)
- [CNN Bring up & Optimization in TT-NN](./tech_reports/CNNs/cnn_optimizations.md) (updated Jan 22nd, 2025)

## Benchmarks

- [Matrix Multiply FLOPS on Wormhole and Blackhole](./tech_reports/GEMM_FLOPS/GEMM_FLOPS.md)  (updated June 17th, 2025)

---

<div align="center">

<img src="https://raw.githubusercontent.com/tenstorrent/tt-metal/main/docs/source/common/images/tt_refresh_metalium_w_icon.png" alt="TT-Metalium logo" height="180"/>

**TT-Metalium** is our low-level programming model, enabling kernel development for Tenstorrent hardware.

<h3>

[Programming Guide](./METALIUM_GUIDE.md) | [API Reference](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/index.html)

</h3>
</div>

## Getting started

Get started with [simple kernels](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/index.html).

## TT-Metalium Tech Reports

- [Matrix Engine](./tech_reports/matrix_engine/matrix_engine.md) (updated Sept 6th, 2024)
- [Data Formats](./tech_reports/data_formats/data_formats.md) (updated Sept 7th, 2024)
- [Reconfiguring Data Formats](./tech_reports/data_formats/reconfig_data_format.md) (updated Oct 17th, 2024)
- [Handling special floating-point numbers](./tech_reports/Handling_Special_Value/special_values.md) (updated Oct 5th, 2024)
- [Allocator](./tech_reports/memory/allocator.md) (Updated Dec 19th, 2024)
- [Tensor Layouts](./tech_reports/tensor_layouts/tensor_layouts.md) (updated Sept 6th, 2024)
- [Saturating DRAM Bandwidth](./tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md) (updated Sept 6th, 2024)
- [Flash Attention on Wormhole](./tech_reports/FlashAttention/FlashAttention.md) (updated Sept 6th, 2024)
- [CNNs on TT Architectures](./tech_reports/CNNs/ttcnn.md) (updated Sept 6th, 2024)
- [Ethernet and Multichip Basics](./tech_reports/EthernetMultichip/BasicEthernetGuide.md) (Updated Sept 20th, 2024)
- [Blackhole Bring-Up Programming Guide](./tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md) (Updated Dec 18th, 2024)
- [Sub-Devices](./tech_reports/SubDevices/SubDevices.md) (Updated Jan 7th, 2025)

## TT-Metalium Programming Examples

### Hello World

- [Hello World! Compute Kernel](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/hello_world_compute_kernel/hello_world_compute.md)
- [Hello World! Data Movement Kernel](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/hello_world_datamovement_kernel/hello_world_data_movement.md)

### Add Integers

- [Add 2 Integers in Baby RiscV](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/add_2_integers_in_riscv/add_2_integers_in_riscv.md)
- [Add 2 Integers in Compute Kernel](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/add_2_integers_in_compute/add_2_integers_in_compute.md)

### Simple Tensor Manipulation

- [Sharding](./tech_reports/prog_examples/shard_data_rm/shard_data_rm.md)
- [Padding](./tech_reports/prog_examples/pad_multi_core/pad_multi_core.md)

### DRAM Data Movement

- [Dram Loopback Data Movement](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/loopback/dram_loopback.md)

### Eltwise

- [Eltwise Unary OP in Vector Engine (SFPU)](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/eltwise_sfpu/eltwise_sfpu.md)
- [Eltwise Binary OP in Matrix Engine (FPU)](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/eltwise_binary/eltwise_binary.md)

### Matmul

- [Matmul OP on a Single_core](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_single_core/matmul_single_core.md)
- [Matmul OP on Multi_core (Basic)](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_multi_core/matmul_multi_core.md)
- [Matmul Multi_core Reuse (Optimized)](./tech_reports/prog_examples/matmul_multi_core_optimized/data_reuse.md)
- [Matmul Multi_core Multi-Cast (Optimized)](./tech_reports/prog_examples/matmul_multi_core_optimized/data_mcast.md)

### Tools and Instruments

#### [TT-NN Visualizer](https://github.com/tenstorrent/ttnn-visualizer)
A comprehensive tool for visualizing and analyzing model execution, offering interactive graphs, memory plots, tensor details, buffer overviews, operation flow graphs, and multi-instance support with file or SSH-based report loading.

#### [TT-Exalens](https://github.com/tenstorrent/tt-exalens)
The TT-Exalens repository describes TT-Lensium, a low-level debugging tool for Tenstorrent hardware. It allows developers to access and communicate with Wormhole and Blackhole devices.

#### [TT-SMI](https://github.com/tenstorrent/tt-smi)
The TT-SMI repository describes the Tenstorrent System Management Interface. This command line utility can interact with Tenstorrent devices on host. TT-SMI provides an easy to use interface displaying device, telemetry, and firmware information.

#### [Model Explorer](https://github.com/tenstorrent/model-explorer)
The Model Explorer is an intuitive and hierarchical visualization tool using model graphs. It organizes model operations into nested layers and provides features for model exploration and debugging.

#### [Tracy Profiler](https://github.com/tenstorrent/tracy)
The Tracy Profiler is a real-time nanosecond resolution, remote telemetry, hybrid frame, and sampling tool. Tracy supports profiling CPU, GPU, memory allocation, locks, context switches, and more.

#### [Kernel Print Debug](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/kernel_print.html)
DPRINT can print variables, addresses, and circular buffer data from kernels to the host terminal or log file. This feature is useful for debugging issues with kernels.

#### [Watcher](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tools/watcher.rst)
Watcher monitors firmware and kernels for common programming errors, and overall device status. If an error or hang occurs, Watcher displays log data of that occurrence.

#### [Inspector](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tools/inspector.rst)
Inspector provides insights into host runtime. It logs necessary data for investigation and allows queries to host runtime data.

## Related Tenstorrent Projects
- [TT-Forge](https://github.com/tenstorrent/tt-forge/tree/main)
- [TT-Forge-FE](https://github.com/tenstorrent/tt-forge-fe)
- [TT-Torch](https://github.com/tenstorrent/tt-torch)
- [TT-XLA](https://github.com/tenstorrent/tt-xla)
- [TT-MLIR](https://github.com/tenstorrent/tt-mlir)
- [TT-TVM](https://github.com/tenstorrent/tt-tvm)

## Latest Releases

| Release | Release Date | FW Version |
|:---------:|:--------------:|:------------:|
| 0.65.0 | ETA Dec 15, 2025 | 19.2.0 |
| [0.64.5](https://github.com/tenstorrent/tt-metal/releases/tag/v0.64.5) | Dec  1, 2025 | 18.12.0 |
| [0.64.4](https://github.com/tenstorrent/tt-metal/releases/tag/v0.64.4) | Nov 24, 2025 | 18.12.0 |
| [0.64.3](https://github.com/tenstorrent/tt-metal/releases/tag/v0.64.3) | Nov 14, 2025 | 18.12.0 |
| [0.64.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.64.0) | Oct 29, 2025 | 18.12.0 |
| [0.63.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.63.0) | Sep 22, 2025 | 18.8.0 |
| [0.62.2](https://github.com/tenstorrent/tt-metal/releases/tag/v0.62.2) | Aug 20, 2025 | 18.6.0 |
| 0.61.0  | Skipped | - |
| [0.60.1](https://github.com/tenstorrent/tt-metal/releases/tag/v0.60.1) | Jul 22, 2025 | 18.6.0 |
| [0.59.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.59.0) | Jun 18, 2025 | - |
| [0.58.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.58.0) | May 13, 2025 | - |
| [0.57.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.57.0) | Apr 15, 2025 | - |
| [0.56.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.56.0) | Mar 7, 2025  | - |

Visit the [releases](https://github.com/tenstorrent/tt-metal/tree/main/releases) folder for details on releases, release notes, and estimated release dates.

## Tenstorrent Bounty Program Terms and Conditions
This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-metal, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both “bounty” and difficulty level!

## License
TT-Metalium and TTNN are licensed under the Apache 2.0 License, as detailed in [LICENSE](LICENSE) and [LICENSE_understanding.txt](LICENSE_understanding.txt).

Some distributable forms of this project—such as manylinux-compliant wheels—may need to bundle additional libraries beyond the standard Linux system libraries. For example:

- libnuma
- libhwloc
- openmpi (when built with multihost support)
- libevent (when built with multihost support)

These libraries are bound by their own license terms.
