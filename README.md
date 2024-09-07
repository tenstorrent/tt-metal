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
| Model                                                      | Batch | Hardware |ttft (s) | t/s/u | Target t/s/u | Release     |
|----------------------------------------------------------  |-------|----------|------------|-------|--------------|-------------|
| [Falcon7B-decode](./models/demos/ttnn_falcon7b)            | 32    | [e150](https://tenstorrent.com/hardware/grayskull)     |            | 135   | 140          |
| [Falcon7B](./models/demos/wormhole/falcon7b)               | 32    | [n150](https://tenstorrent.com/hardware/wormhole)     | 0.08       | 16.7  | 26           | [v0.51.0-rc24](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc24) | 
| [Mistral-7B](./models/demos/wormhole/mistral7b)            | 32    | [n150](https://tenstorrent.com/hardware/wormhole)     |            | 9.9   | 25           | [v0.51.0-rc28](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc28) |
| [Mamba-2.8B](./models/demos/wormhole/mamba)                | 32    | [n150](https://tenstorrent.com/hardware/wormhole)     | 0.04       | 12.3  | 41           | [v0.51.0-rc26](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc26) | 
| [LLaMA-3.1-8B](./models/demos/wormhole/llama31_8b)         | 32    | [n150](https://tenstorrent.com/hardware/wormhole)     |            | 8.3   | 23           | [v0.51.0-rc28](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc28) | 
| [Falcon7B (data parallel)](./models/demos/t3000/falcon7b)          |  32 | [LoudBox](https://tenstorrent.com/hardware/tt-loudbox) | 0.11 | 13.4 | 26 |      [v0.51.0-rc36](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc36) | 
| [LLaMA-2-70B - (tensor parallel)](./models/demos/t3000/llama2_70b)     | 32 | [LoudBox](https://tenstorrent.com/hardware/tt-loudbox) |   | 10.4 | 20 | [v0.51.0-rc36](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc36) | 
| [LLaMA-3.1-70B (tensor parallel)](./models/demos/t3000/llama3_70b)   | 32 | [LoudBox](https://tenstorrent.com/hardware/tt-loudbox) |   | 10.4 | 20 | [v0.51.0-rc36](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc36) | 
| [Falcon40B (tensor parallel)](./models/demos/t3000/falcon40b)        | 32 | [LoudBox](https://tenstorrent.com/hardware/tt-loudbox) |  | 5.3 | 36 | [v0.51.0-rc35](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc35) |
| [Mixtral7Bx8 (tensor parallel)](./models/demos/t3000/mixtral8x7b)    | 32 | [LoudBox](https://tenstorrent.com/hardware/tt-loudbox) | 0.19 | 15.7 | 33 | [v0.51.0-rc33](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc33) | 
| [Falcon7B (data parallel)](./models/demos/tg/falcon7b)     |1024 | [Galaxy](https://tenstorrent.com/hardware/galaxy) | 0.30 | 4.0 | 26 |  [v0.51.0-rc30](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc30) | 

## CNNs
| Model                                               | Batch | Hardware |fps    | Target fps | Release     |
|-----------------------------------------------------|-------|----------|-------|------------|-------------|
| [ResNet-50 (224x224)](./models/demos/ttnn_resnet)           | 20    | [e150](https://tenstorrent.com/hardware/grayskull)     | 5,100  | 10,000    |              |
| [ResNet-50 (224x224)](./models/demos/ttnn_resnet)           |  16 | [n150](https://tenstorrent.com/hardware/wormhole) | 4,100 | 7,000 |   | 
| [ResNet-50 (224x224) (data parallel)](./models/demos/ttnn_resnet)       | 128 | [Galaxy](https://tenstorrent.com/hardware/galaxy) | 31,250 | 56,000 |  | 
| [ViT](./models/demos/grayskull/vit)                         | 8    | [e150](https://tenstorrent.com/hardware/grayskull) | 860 | 2,000 |  | 
| [Stable Diffusion 1.4 (512x512)](./models/demos/wormhole/stable_diffusion)  | 1 | [n150](https://tenstorrent.com/hardware/wormhole) | 0.167 | 0.3 |  | 

## NLPs
| Model                                               | Batch | Hardware |sen/sec    | Target sen/sec | Release     |
|-----------------------------------------------------|-------|----------|-------|------------|-------------|
| [BERT-Large](./models/demos/bert)                   | 12 | [e150](https://tenstorrent.com/hardware/grayskull) | 370 | 410 | |
| [BERT-Large](./models/demos/metal_BERT_large_11/)   | 8 | [n150](https://tenstorrent.com/hardware/wormhole) | 270 | 400 | | 
| [T5 small](.models/demos/grayskull/t5)              |   | [e150](https://tenstorrent.com/hardware/grayskull) | 140 | | | 
| [Bloom](.models/demos/grayskull/functional_bloom)   |  | [e150](https://tenstorrent.com/hardware/grayskull) | 70 | | | 



## Model Updates
For the latest model updates and features, please see [MODEL_UPDATES.md](models/MODEL_UPDATES.md)

## TT-NN Tech Reports
- [Advanced Performance Optimizations for Models](./tech_reports/AdvancedPerformanceOperationsForModels/AdvancedPerformanceOptimizationsForModels.md) (updated Sept 6th)
- [Programming Mesh of Devices](./tech_reports/Programming%20Mesh%20of%20Devices/Programming%20Mesh%20of%20Devices%20with%20TT-NN.md) (updated Sept 6th)
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
- [Matrix Engine](./tech_reports/matrix_engine/matrix_engine.md) (updated Sept 6th)
- [Tensor Layouts](./tech_reports/tensor_layouts/tensor_layouts.md) (updated Sept 6th)
- [Saturating DRAM Bandwidth](./tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md) (updated Sept 6th)
- [Flash Attention on Wormhole](./tech_reports/FlashAttention/FlashAttention.md) (updated Sept 6th)
- [CNNs on TT Architectures](./tech_reports/CNNs/ttcnn.md) (updated Sept 6th)

