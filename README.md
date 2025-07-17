[![tt-metal CI](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml/badge.svg)](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tenstorrent/tt-metal)

<div align="center">

<h1>

[Install](./INSTALLING.md) | [Buy Hardware](https://tenstorrent.com/hardware/blackhole) | [Bounty $](https://github.com/tenstorrent/tt-metal/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty) | [Join Us](https://boards.greenhouse.io/tenstorrent/jobs/4155609007) | [Discord](https://discord.gg/tvhGzHQwaj)

</h1>

<img src="https://raw.githubusercontent.com/tenstorrent/tt-metal/main/docs/source/common/_static/tt_nn_w_logo.png" alt="ttnn logo" height="180"/>

**TT-NN** is a Python & C++ Neural Network OP library.

<h3>

[API Reference](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html) | [Model Demos](./models/demos/)

</h3>

</div>

## Latest Releases

| Release | Release Date |
|---------|--------------|
| 0.61.0  | ETA Aug 13, 2025 |
| [0.60.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.60.0) | Jul 16, 2025 |
| [0.59.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.59.0) | Jun 18, 2025 |
| [0.58.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.58.0) | May 13, 2025 |
| [0.57.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.57.0) | Apr 15, 2025 |
| [0.56.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.56.0) | Mar 7, 2025 |

## LLMs

| Model                                                         | Batch | Hardware                                                 | ttft (ms) | t/s/u | Target<br>t/s/u | t/s    | TT-Metalium Release                                            | vLLM Tenstorrent Repo Release                                                                                |
|---------------------------------------------------------------|-------|----------------------------------------------------------|-----------|-------|-----------------|--------|---------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [Qwen 3 32B (TP=8)](./models/tt_transformers)                       | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 109       | 22.1  | 30              | 707.2  | [v0.59.0-rc52](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc52) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal) |
| [QwQ 32B (TP=8)](./models/tt_transformers)                       | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 133       | 25.2  | 30              | 806.4  | [v0.56.0-rc51](https://github.com/tenstorrent/tt-metal/tree/v0.56.0-rc51) | [e2e0002](https://github.com/tenstorrent/vllm/tree/e2e0002ac7dcbc5793983c0f967474d4dcab21f8/tt_metal)      |
| [DeepSeek R1 Distill Llama 3.3 70B (TP=8)](./models/tt_transformers)       | 32    | [QuietBox  (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 159       | 15.9  | 20    | 508.8  | [v0.59.0-rc53](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc53) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal)      |
| [Llama 3.1 70B (TP=32)](./models/demos/llama3_subdevices)        | 32    | [Galaxy](https://tenstorrent.com/hardware/galaxy) | 105      | 59.1  | 80              | 1891.2  | [v0.59.0-rc52](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc52) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal) |
| [Llama 3.1 70B (TP=8)](./models/tt_transformers)                 | 32    | [QuietBox  (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 159       | 15.9  | 20              | 508.8  | [v0.59.0-rc53](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc53) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal)      |
| [Llama 3.1 70B (TP=4)](./models/tt_transformers)                 | 32    | [QuietBox (Blackhole)](https://tenstorrent.com/hardware/tt-quietbox) | 195*       | 14.9*  |               | 476.5*  | [v0.59.0-rc53](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc53) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal)      |
| [Llama 3.2 11B Vision (TP=2)](./models/tt_transformers)          | 16     | [n300](https://tenstorrent.com/hardware/wormhole)        | 2550       | 15.8  | 17              | 252.8   | [v0.56.0-rc6](https://github.com/tenstorrent/tt-metal/tree/v0.56.0-rc6)  | [e2e0002](https://github.com/tenstorrent/vllm/tree/e2e0002ac7dcbc5793983c0f967474d4dcab21f8/tt_metal) |
| [Qwen 2.5 7B (TP=2)](https://github.com/tenstorrent/tt-metal/tree/v0.56.0-rc33/models/demos/llama3)   | 32     | [n300](https://tenstorrent.com/hardware/wormhole)        | 126      | 32.5  | 38              | 1040.0   | [v0.56.0-rc33](https://github.com/tenstorrent/tt-metal/tree/v0.56.0-rc33) | [e2e0002](https://github.com/tenstorrent/vllm/tree/e2e0002ac7dcbc5793983c0f967474d4dcab21f8/tt_metal) |
| [Qwen 2.5 72B (TP=8)](./models/tt_transformers)               | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 319       | 14.6  | 20              | 467.2  | [v0.59.0-rc52](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc52) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal) |
| [Falcon 7B](./models/demos/wormhole/falcon7b)                 | 32    | [n150](https://tenstorrent.com/hardware/wormhole)        | 70        | 18.5  | 26              | 592.0  | [v0.59.0-rc52](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc52) |                                                                                                   |
| [Falcon 7B (DP=8)](./models/demos/t3000/falcon7b)             | 256   | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 87        | 15.9  | 26              | 4070.4 | [v0.59.0-rc38](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc38) |                                                                                                   |
| [Falcon 7B (DP=32)](./models/demos/tg/falcon7b)               | 1024  | [Galaxy](https://tenstorrent.com/hardware/galaxy)        | 125       | 12.9   | 26              | 13209.6 | [v0.59.0-rc52](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc52) |                                                                                                   |
| [Falcon 40B (TP=8)](./models/demos/t3000/falcon40b)           | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) |           | 11.9   | 36              | 380.8  | [v0.59.0-rc38](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc38)  |                                                                                                   |
| [Llama 3.1 8B](./models/tt_transformers)                         | 32     | [p100](https://tenstorrent.com/hardware/blackhole)        | 87*       | 26.5*   |               | 848.0*   | [v0.59.0-rc3](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc3) | [739dcaa](https://github.com/tenstorrent/vllm/tree/739dcaa2915fa29d757c25a02c17aadce0c58055/tt_metal) |
| [Llama 3.1 8B](./models/tt_transformers)                         | 32     | [p150](https://tenstorrent.com/hardware/blackhole)        | 69*       | 29.1*   |               | 931.2*   | [v0.59.0-rc3](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc3) | [739dcaa](https://github.com/tenstorrent/vllm/tree/739dcaa2915fa29d757c25a02c17aadce0c58055/tt_metal) |
| [Llama 3.1 8B (DP=2)](./models/tt_transformers)                  | 64     | [2 x p150](https://tenstorrent.com/hardware/blackhole)        | 64*       | 18.6*   |               | 1190.4*  | [v0.59.0-rc3](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc3) | [739dcaa](https://github.com/tenstorrent/vllm/tree/739dcaa2915fa29d757c25a02c17aadce0c58055/tt_metal) |
| [Llama 3.1 8B](./models/tt_transformers)                         | 32     | [n150](https://tenstorrent.com/hardware/wormhole)        | 104       | 24.8  | 23              | 793.6   | [v0.59.0-rc52](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc52) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal) |
| [Llama 3.2 1B](./models/tt_transformers)                         | 32     | [n150](https://tenstorrent.com/hardware/wormhole)        | 23        | 72.6  | 160             | 2323.2   | [v0.59.0-rc52](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc52) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal) |
| [Llama 3.2 3B](./models/tt_transformers)                         | 32     | [n150](https://tenstorrent.com/hardware/wormhole)        | 53       | 43.5  | 60              | 1392.0   | [v0.59.0-rc52](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc52) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal) |
| [Mamba 2.8B](./models/demos/wormhole/mamba)                   | 32    | [n150](https://tenstorrent.com/hardware/wormhole)        | 35        | 14.1  | 41              | 451.2  | [v0.59.0-rc38](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc38) |                                                                                                   |
| [Mistral 7B](./models/tt_transformers)                        | 32    | [n150](https://tenstorrent.com/hardware/wormhole)        |  101 | 28.3  | 23     | 905.6 | [v0.59.0-rc52](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc52)    |  [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal) |
| [Mixtral 8x7B (TP=8)](./models/demos/t3000/mixtral8x7b)       | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 207       | 16.6  | 33              | 531.2  | [v0.59.0-rc53](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc53) |                                                                                                   |

> **Last Update:** June 23, 2025
>
> **Notes:**
>
> - ttft = time to first token | t/s/u = tokens/second/user | t/s = tokens/second; where t/s = t/s/u * batch.
> - TP = Tensor Parallel, DP = Data Parallel; Defines parallelization factors across multiple devices.
> - The reported LLM performance is for an input sequence length (number of rows filled in the KV cache) of 128 for all models except Mamba (which can accept any sequence length).
> - The t/s/u reported is the throughput of the first token generated after prefill, i.e. 1 / inter token latency.
> - Performance numbers were collected using the tt-metal model demos (accessible via the model links). If running with a vLLM inference server, performance may be different.
> - \* Blackhole software optimization is under active development.  Please join us in shaping the future of open source AI! <br> [\[Discord\]](https://discord.gg/tenstorrent) [\[Developer Hub\]](https://tenstorrent.com/developers)
> - For more information regarding vLLM installation and environment creation visit the [Tenstorrent vLLM repository](https://github.com/tenstorrent/vllm/blob/dev/tt_metal/README.md).

## Speech-to-Text

| Model                                                | Batch | Hardware                                                 | ttft (ms) | t/s/u | Target t/s/u | t/s    | TT-Metalium Release                                                       |
|------------------------------------------------------|-------|----------------------------------------------------------|-----------|-------|--------------|--------|---------------------------------------------------------------------------|
| [Whisper (distil-large-v3)](./models/demos/whisper)  | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 232       | 58.1  | 45           | 58.1   | [v0.59.0-rc52](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc52) |


## Diffusion Models
| Model                                                                       | Batch | Hardware                                                 | Sec/Image     | Target Sec/Image | Release     |
|-----------------------------------------------------------------------------|-------|----------------------------------------------------------|---------|------------|-------------|
| [Stable Diffusion 1.4 (512x512)](./models/demos/wormhole/stable_diffusion)  | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 6.25   | 3        |           |
| [Stable Diffusion 3.5 Medium (512x512)](https://github.com/tenstorrent/tt-metal/blob/mbahnas/sd35_medium_512_spacelike_feb05/models/experimental/stable_diffusion3)  | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 16   | 10        |             |


**Notes:**
- Stable Diffusion sec/image is based on the time elapsed from submitting the input prompt to receiving the image from the VAE decoder.



## CNNs and Vision Transformers

### Classification models

| Model                                                                       | Batch | Hardware                                                 | Image/sec    | Target Image/sec | Release     |
|-----------------------------------------------------------------------------|-------|----------------------------------------------------------|---------|------------|-------------|
| [ResNet-50 (224x224)](./models/demos/wormhole/resnet50)                     | 16    | [n150](https://tenstorrent.com/hardware/wormhole)        | 4,700   | 7,000      | [v0.59.0](https://github.com/tenstorrent/tt-metal/tree/v0.59.0) |
| [ResNet-50 (224x224) (DP=2)](./models/demos/t3000/resnet50)                 | 32    | [n300](https://tenstorrent.com/hardware/wormhole)        | 9,200   | 14,000     | [v0.59.0](https://github.com/tenstorrent/tt-metal/tree/v0.59.0) |
| [ResNet-50 (224x224) (DP=8)](./models/demos/t3000/resnet50)                 | 128   | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 35,800  | 56,000     | [v0.59.0](https://github.com/tenstorrent/tt-metal/tree/v0.59.0) |
| [ResNet-50 (224x224) (DP=32)](./models/demos/tg/resnet50)                   | 512   | [Galaxy](https://tenstorrent.com/hardware/galaxy)        | 96,800  | 224,000    | [v0.59.0](https://github.com/tenstorrent/tt-metal/tree/v0.59.0) |
| [ViT-base (224x224)](./models/demos/wormhole/vit)                                | 8     | [n150](https://tenstorrent.com/hardware/wormhole)        | 1,370    | 1,600      | [v0.60.0-rc4](https://github.com/tenstorrent/tt-metal/tree/v0.60.0-rc4) |
| [ViT-base (224x224)  (DP=2)](./models/demos/t3000/vit)                                | 16     | [n300](https://tenstorrent.com/hardware/wormhole)        | 1,900   | 3,200      | [v0.60.0-rc4](https://github.com/tenstorrent/tt-metal/tree/v0.60.0-rc4) |
| [ViT-base (224x224)  (DP=8)](./models/demos/t3000/vit)                                | 64     | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox)        | 7,700    | 12,800      | [v0.60.0-rc4](https://github.com/tenstorrent/tt-metal/tree/v0.60.0-rc4) |
| [MobileNet-v2 (224x224)](./models/demos/mobilenetv2)                     | 10    | [n150](https://tenstorrent.com/hardware/wormhole)        |  2,808  | 3,500      |  |

### Object Detection
| Model                                                                       | Batch | Hardware                                                 | Frame/sec (FPS)     | Target FPS | Release     |
|-----------------------------------------------------------------------------|-------|----------------------------------------------------------|---------|------------|-------------|
| [YOLOv4 (320x320)](./models/demos/yolov4)                                   | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 120     | 320        |             |
| [YOLOv4 (640x640)](./models/demos/yolov4)                                   | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 50      | 180        |             |
| [YOLOv8x (640x640)](./models/demos/yolov8x)                                   | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 45      | 100        |             |
| [YOLOv8s (640x640)](./models/demos/yolov8s)                                   | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 175      | 320        |             |
| [YOLOv8s_world (640x640)](./models/demos/yolov8s_world)                                   | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 57      | 200        |             |
| [YOLOv9c (640x640)](./models/demos/yolov9c)                                   | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 55      | 320        |             |
| [YOLOv10x (640x640)](./models/demos/yolov10x)                                   | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 26      | 200        |             |

### Segmentation
| Model                                                                       | Batch | Hardware                                                 | Frame/sec (FPS)     | Target FPS | Release     |
|-----------------------------------------------------------------------------|-------|----------------------------------------------------------|---------|------------|-------------|
| [UNet - VGG19 (256x256)](./models/demos/vgg_unet)                                   | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 77      | 150        |             |
| [SegFormer Semantic Segmentation (512x512)](./models/demos/segformer)       | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 84      | 300        |              |
| [YOLOv9c (640x640)](./models/demos/yolov9c)                                   | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 40      | 240        |             |
| [UFLD - v2 (320x800)](./models/demos/ufld_v2)                                   | 1     | [n150](https://tenstorrent.com/hardware/wormhole)        | 255      | 2000        |             |


## NLPs

| Model                                               | Batch | Hardware                                           | Sentence/sec | Target sentence/sec | Release |
|-----------------------------------------------------|-------|----------------------------------------------------|---------|----------------|---------|
| [BERT-Large](./models/demos/metal_BERT_large_11/)   | 8     | [n150](https://tenstorrent.com/hardware/wormhole)  | 270     | 400            |         |
| [Sentence-Bert (backbone: bert-base)](./models/demos/sentence_bert/)   | 8     | [n150](https://tenstorrent.com/hardware/wormhole)  | 233     | 360            |         |

## Model Updates

For the latest model updates and features, please see [MODEL_UPDATES.md](models/MODEL_UPDATES.md)

## Model Bring-Up and Testing

For information on initial model procedures, please see [Model Bring-Up and Testing](https://github.com/tenstorrent/tt-metal/tree/main/models/bringup_testing)

## TT-NN Tech Reports

- [Advanced Performance Optimizations for Models](./tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md) (updated March 4th, 2025)
- [Programming Mesh of Devices](./tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md) (updated Sept 9th, 2024)
- [ViT Implementation in TT-NN on GS](./tech_reports/ViT-TTNN/vit.md)  (updated Sept 22nd, 2024)
- [LLMs Bring up in TT-NN](./tech_reports/LLMs/llms.md)  (updated Oct 29th, 2024)
- [YOLOv4 Implementation in TT-NN on WH](./tech_reports/YoloV4-TTNN/yolov4.md)  (updated November 8th, 2024)
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

### Tools and Instruments

#### [TT_NN Visualizer](https://github.com/tenstorrent/ttnn-visualizer)
A comprehensive tool for visualizing and analyzing model execution, offering interactive graphs, memory plots, tensor details, buffer overviews, operation flow graphs, and multi-instance support with file or SSH-based report loading.
Install via pip or build from source:
```bash
pip install ttnn-visualizer
```



### Tenstorrent Bounty Program Terms and Conditions
This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-metal, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both “bounty” and difficulty level!

## License
TT-Metalium and TTNN are licensed under the Apache 2.0 License, as detailed in [LICENSE](LICENSE) and [LICENSE_understanding.txt](LICENSE_understanding.txt).

Some distributable forms of this project—such as manylinux-compliant wheels—may need to bundle additional libraries beyond the standard Linux system libraries. For example:

- libnuma
- libhwloc
- openmpi (when built with multihost support)
- libevent (when built with multihost support)

These libraries are bound by their own license terms.
