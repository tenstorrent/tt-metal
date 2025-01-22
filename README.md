<div align="center">

<h1>

[Buy hardware](https://tenstorrent.com/cards/) | [Install](./INSTALLING.md) | [Discord](https://discord.gg/tvhGzHQwaj) | [Join Us](https://boards.greenhouse.io/tenstorrent/jobs/4155609007)

</h1>

<img src="./docs/source/common/_static/tt_nn_w_logo.png" alt="ttnn logo" height="180"/>

**TT-NN** is a Python & C++ Neural Network OP library.

<h3>

[API Reference](https://docs.tenstorrent.com/ttnn/latest/index.html) | [Model Demos](./models/demos/)

</h3>

</div>

---

## LLMs

| Model                                                         | Batch | Hardware                                                 | ttft (ms) | t/s/u | Target<br>t/s/u | t/s    | TT-Metalium Release                                            | vLLM Tenstorrent Repo Release                                                                                |
|---------------------------------------------------------------|-------|----------------------------------------------------------|-----------|-------|-----------------|--------|---------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [Falcon 7B (decode only)](./models/demos/ttnn_falcon7b)       | 32    | [e150](https://tenstorrent.com/hardware/grayskull)       |           | 4.2   | 4.4             | 134.4  |                                               |                                                                            |
| [Falcon 7B](./models/demos/wormhole/falcon7b)                 | 32    | [n150](https://tenstorrent.com/hardware/wormhole)        | 71        | 18.0  | 26              | 576.0  | [v0.54.0-rc3](https://github.com/tenstorrent/tt-metal/tree/v0.54.0-rc3) |                                                                                                   |
| [Mistral 7B](./models/demos/wormhole/mistral7b)               | 32    | [n150](https://tenstorrent.com/hardware/wormhole)        |           | 9.9   | 25              | 316.8  | [v0.51.0-rc28](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc28) |                                                                                                   |
| [Mamba 2.8B](./models/demos/wormhole/mamba)                   | 32    | [n150](https://tenstorrent.com/hardware/wormhole)        | 48        | 12.3  | 41              | 393.6  | [v0.51.0-rc26](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc26) |                                                                                                   |
| [Llama 3.1 8B](./models/demos/llama3)                         | 32     | [n150](https://tenstorrent.com/hardware/wormhole)        | 169       | 23.4  | 23              | 748.8   | [v0.54.0-rc20](https://github.com/tenstorrent/tt-metal/tree/v0.54.0-rc20)  | [2f33504](https://github.com/tenstorrent/vllm/tree/2f33504bad49a6202d3685155107a6126a5b5e6e) |
| [Llama 3.2 1B](./models/demos/llama3)                         | 32     | [n150](https://tenstorrent.com/hardware/wormhole)        | 68        | 58.6  | 160             | 1875.2   | [v0.54.0-rc20](https://github.com/tenstorrent/tt-metal/tree/v0.54.0-rc20)  | [2f33504](https://github.com/tenstorrent/vllm/tree/2f33504bad49a6202d3685155107a6126a5b5e6e) |
| [Llama 3.2 3B](./models/demos/llama3)                         | 32     | [n150](https://tenstorrent.com/hardware/wormhole)        | 110       | 36.1  | 60              | 1155.2   | [v0.54.0-rc20](https://github.com/tenstorrent/tt-metal/tree/v0.54.0-rc20)  | [2f33504](https://github.com/tenstorrent/vllm/tree/2f33504bad49a6202d3685155107a6126a5b5e6e) |
| [Llama 3.2 11B Vision (TP=2)](./models/demos/llama3)          | 16     | [n300](https://tenstorrent.com/hardware/wormhole)        | 2880       | 14.8  | 17              | 236.8   | [v0.54.0-rc20](https://github.com/tenstorrent/tt-metal/tree/v0.54.0-rc20)  | [2f33504](https://github.com/tenstorrent/vllm/tree/2f33504bad49a6202d3685155107a6126a5b5e6e) |
| [Falcon 7B (DP=8)](./models/demos/t3000/falcon7b)             | 256   | [QuietBox](https://tenstorrent.com/hardware/tt-quietbox) | 96        | 14.7  | 26              | 3763.2 | [v0.54.0-rc3](https://github.com/tenstorrent/tt-metal/tree/v0.54.0-rc3) |                                                                                                   |
| [Llama 3.1 70B (TP=8)](./models/demos/t3000/llama3_70b)       | 32    | [QuietBox](https://tenstorrent.com/hardware/tt-quietbox) | 190       | 15.1  | 20              | 483.2  | [v0.54.0-rc2](https://github.com/tenstorrent/tt-metal/tree/v0.54.0-rc2) | [9531611](https://github.com/tenstorrent/vllm/tree/953161188c50f10da95a88ab305e23977ebd3750)      |
| [Falcon 40B (TP=8)](./models/demos/t3000/falcon40b)           | 32    | [QuietBox](https://tenstorrent.com/hardware/tt-quietbox) |           | 5.3   | 36              | 169.6  | [v0.53.1-rc23](https://github.com/tenstorrent/tt-metal/tree/v0.53.1-rc23)  |                                                                                                   |
| [Mixtral 8x7B (TP=8)](./models/demos/t3000/mixtral8x7b)       | 32    | [QuietBox](https://tenstorrent.com/hardware/tt-quietbox) | 230       | 14.6  | 33              | 467.2  | [v0.53.0-rc44](https://github.com/tenstorrent/tt-metal/tree/v0.53.0-rc44) |                                                                                                   |
| [Falcon 7B (DP=32)](./models/demos/tg/falcon7b)               | 1024  | [Galaxy](https://tenstorrent.com/hardware/galaxy)        | 242       | 4.4   | 26              | 4505.6 | [v0.53.0-rc33](https://github.com/tenstorrent/tt-metal/tree/v0.53.0-rc33) |                                                                                                   |
| [Llama 3.1 70B (DP=4, TP=8)](./models/demos/t3000/llama3_70b) | 128   | [Galaxy](https://tenstorrent.com/hardware/galaxy)        | 190       | 14.3  | 20              | 1835.5 | [v0.52.0-rc31](https://github.com/tenstorrent/tt-metal/tree/v0.52.0-rc31) |                                                                                                   |
| [Llama 3.1 70B (TP=32)](./models/demos/llama3)                | 32   | [Galaxy](https://tenstorrent.com/hardware/galaxy)        | 763       | 13.5  | 80              | 432.0 | [v0.54.0-rc20](https://github.com/tenstorrent/tt-metal/tree/v0.54.0-rc20) | [2f33504](https://github.com/tenstorrent/vllm/tree/2f33504bad49a6202d3685155107a6126a5b5e6e) |

> **Last Update:** January 13, 2025
>
> **Notes:**
>
> - ttft = time to first token | t/s/u = tokens/second/user | t/s = tokens/second; where t/s = t/s/u * batch.
> - TP = Tensor Parallel, DP = Data Parallel; Defines parallelization factors across multiple devices.
> - The reported LLM performance is for an input sequence length (number of rows filled in the KV cache) of 128 for all models except Mamba (which can accept any sequence length).
> - The t/s/u reported is the throughput of the first token generated after prefill, i.e. 1 / inter token latency.

## CNNs

| Model                                                                       | Batch | Hardware                                                 | fps     | Target fps | Release     |
|-----------------------------------------------------------------------------|-------|----------------------------------------------------------|---------|------------|-------------|
| [ResNet-50 (224x224)](./models/demos/grayskull/resnet50)                    | 20    | [e150](https://tenstorrent.com/hardware/grayskull)       | 5,100   | 10,000     |             |
| [ResNet-50 (224x224)](./models/demos/wormhole/resnet50)                     | 16    | [n150](https://tenstorrent.com/hardware/wormhole)        | 4,700   | 7,000      |             |
| [ResNet-50 (224x224) (DP=2)](./models/demos/wormhole/resnet50)              | 32    | [n300](https://tenstorrent.com/hardware/wormhole)        | 9,200   | 14,000     |             |
| [ResNet-50 (224x224) (DP=8)](./models/demos/t3000/resnet50)                 | 128   | [QuietBox](https://tenstorrent.com/hardware/tt-quietbox) | 35,800  | 56,000     |             |
| [ResNet-50 (224x224) (DP=32)](./models/demos/tg/resnet50)                   | 512   | [Galaxy](https://tenstorrent.com/hardware/galaxy)        | 96,800  | 224,000    |             |
| [ResNet-50 (224x224) (DP=64)](./models/demos/tgg/resnet50)                  | 1024  | [Two Galaxies](https://tenstorrent.com/hardware/galaxy)  | 145,000 | 448,000    |             |
| [ViT (224x224)](./models/demos/grayskull/vit)                               | 9     | [e150](https://tenstorrent.com/hardware/grayskull)       | 1,360   | 2,000      |             |
| [ViT (224x224)](./models/demos/wormhole/vit)                                | 8     | [n150](https://tenstorrent.com/hardware/wormhole)        | 912     | 1,600      |             |
| [Stable Diffusion 1.4 (512x512)](./models/demos/wormhole/stable_diffusion)  | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 0.167   | 0.3        |             |
| [YOLOv4 (320x320)](./models/demos/yolov4)                                   | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 95      | 300        |             |
| [SegFormer Semantic Segmentation (512x512)](./models/demos/segformer)       | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 90      | 300        |             |

## NLPs

| Model                                               | Batch | Hardware                                           | sen/sec | Target sen/sec | Release |
|-----------------------------------------------------|-------|----------------------------------------------------|---------|----------------|---------|
| [BERT-Large](./models/demos/metal_BERT_large_11/)   | 12    | [e150](https://tenstorrent.com/hardware/grayskull) | 370     | 410            |         |
| [BERT-Large](./models/demos/metal_BERT_large_11/)   | 8     | [n150](https://tenstorrent.com/hardware/wormhole)  | 270     | 400            |         |
| [T5 small](.models/demos/grayskull/t5)              |       | [e150](https://tenstorrent.com/hardware/grayskull) | 140     |                |         |
| [Bloom](.models/demos/grayskull/functional_bloom)   |       | [e150](https://tenstorrent.com/hardware/grayskull) | 70      |                |         |

## Model Updates

For the latest model updates and features, please see [MODEL_UPDATES.md](models/MODEL_UPDATES.md)

## TT-NN Tech Reports

- [Advanced Performance Optimizations for Models](./tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md) (updated Dec 4th, 2024)
- [Programming Mesh of Devices](./tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md) (updated Sept 9th, 2024)
- [ViT Implementation in TT-NN on GS](./tech_reports/ViT-TTNN/vit.md)  (updated Sept 22nd, 2024)
- [LLMs Bring up in TT-NN](./tech_reports/LLMs/llms.md)  (updated Oct 29th, 2024)
- [YOLOv4 Implementation in TT-NN on WH](./tech_reports/YoloV4-TTNN/yolov4.md)  (updated November 8th, 2024)

## Benchmarks

- [Matrix Multiply FLOPS on WH](./tech_reports/GEMM_FLOPS/GEMM_FLOPS.md)  (updated November 13th, 2024)

---

<div align="center">

<img src="./docs/source/common/_static/tt_metalium_w_logo.png" alt="TT-Metalium logo" height="180"/>

**TT-Metalium** is our low-level programming model, enabling kernel development for Tenstorrent hardware.

<h3>

[Programming Guide](./METALIUM_GUIDE.md) | [API Reference](https://docs.tenstorrent.com/tt-metalium/latest/tt_metal/apis/index.html)

</h3>
</div>

## Getting started

Get started with [simple kernels](https://docs.tenstorrent.com/tt-metalium/latest/tt_metal/examples/index.html).

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
- [CNN Bring-up & Optimization in TT-NN](./tech_reports/CNNs/cnn_optimizations.md.md) (updated Jan 22nd, 2025)
- [Ethernet and Multichip Basics](./tech_reports/EthernetMultichip/BasicEthernetGuide.md) (Updated Sept 20th, 2024)
- [Collective Communication Library (CCL)](./tech_reports/EthernetMultichip/CclDeveloperGuide.md) (Updated Sept 20th, 2024)
- [Blackhole Bring-Up Programming Guide](./tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md) (Updated Dec 18th, 2024)
- [Sub-Devices](./tech_reports/SubDevices/SubDevices.md) (Updated Jan 7th, 2025)

## TT-Metalium Programming Examples

### Hello World

- [Hello World! Compute Kernel](./tech_reports/prog_examples/hello_world_compute/hello_world_compute.md)
- [Hello World! Data Movement Kernel](./tech_reports/prog_examples/hello_world_data_movement/hello_world_data_movement.md)

### Add Integers

- [Add 2 Integers in Baby RiscV](./tech_reports/prog_examples/add_2_integers_in_riscv/add_2_integers_in_riscv.md)
- [Add 2 Integers in Compute Kernel](./tech_reports/prog_examples/add_2_integers_in_compute/add_2_integers_in_compute.md)

### Simple Tensor Manipulation

- [Sharding](./tech_reports/prog_examples/shard_data_rm/shard_data_rm.md)
- [Padding](./tech_reports/prog_examples/pad_multi_core/pad_multi_core.md)

### DRAM Data Movement

- [Dram Loopback Data Movement](./tech_reports/prog_examples/dram_loopback/dram_loopback.md)

### Eltwise

- [Eltwise Unary OP in Vector Engine (SFPU)](./tech_reports/prog_examples/eltwise_sfpu/eltwise_sfpu.md)
- [Eltwise Binary OP in Matrix Engine (FPU)](./tech_reports/prog_examples/eltwise_binary/eltwise_binary.md)

### Matmul

- [Matmul OP on a Single_core](./tech_reports/prog_examples/matmul_single_core/matmul_single_core.md)
- [Matmul OP on Multi_core (Basic)](./tech_reports/prog_examples/matmul_multi_core/matmul_multi_core.md)
- [Matmul Multi_core Reuse (Optimized)](./tech_reports/prog_examples/matmul_multi_core_optimized/data_reuse.md)
- [Matmul Multi_core Multi-Cast (Optimized)](./tech_reports/prog_examples/matmul_multi_core_optimized/data_mcast.md)
