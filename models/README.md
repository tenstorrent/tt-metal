# TT-Metalium / TT-NN Models

## LLMs

| Model                                                         | Batch | Hardware                                                 | ttft (ms) | t/s/u | Target<br>t/s/u | t/s    | TT-Metalium Release                                            | vLLM Tenstorrent Repo Release                                                                                |
|---------------------------------------------------------------|-------|----------------------------------------------------------|-----------|-------|-----------------|--------|---------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [Qwen 3 32B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                       | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 109       | 22.1  | 30              | 707.2  | [v0.59.0-rc52](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc52) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal) |
| [QwQ 32B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                       | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 133       | 25.2  | 30              | 806.4  | [v0.56.0-rc51](https://github.com/tenstorrent/tt-metal/tree/v0.56.0-rc51) | [e2e0002](https://github.com/tenstorrent/vllm/tree/e2e0002ac7dcbc5793983c0f967474d4dcab21f8/tt_metal)      |
| [DeepSeek R1 Distill Llama 3.3 70B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)       | 32    | [QuietBox  (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 159       | 15.9  | 20    | 508.8  | [v0.59.0-rc53](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc53) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal)      |
| [Llama 3.1 70B (TP=32)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/llama3_70b_galaxy)        | 32    | [Galaxy (Wormhole)](https://tenstorrent.com/hardware/galaxy) | 53      | 72.5  | 80              | 2268.8  | [v0.62.2](https://github.com/tenstorrent/tt-metal/tree/v0.62.2) | [c348d08](https://github.com/tenstorrent/vllm/tree/c348d085a463340a66194bbee9cd4bfc5f9c697a/tt_metal) |
| [Llama 3.1 70B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                 | 32    | [QuietBox  (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 159       | 15.9  | 20              | 508.8  | [v0.59.0-rc53](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc53) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal)      |
| [Llama 3.1 70B (TP=4)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                 | 32    | [QuietBox (Blackhole)](https://tenstorrent.com/hardware/tt-quietbox) | 195*       | 14.9*  |               | 476.5*  | [v0.59.0-rc53](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc53) | [f028da1](https://github.com/tenstorrent/vllm/tree/f028da11b5b8205272bf18a478de93bd2dd3e29e/tt_metal)      |
| [Llama 3.2 11B Vision (TP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)          | 16     | [n300 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 2550       | 15.8  | 17              | 252.8   | [v0.56.0-rc6](https://github.com/tenstorrent/tt-metal/tree/v0.56.0-rc6)  | [e2e0002](https://github.com/tenstorrent/vllm/tree/e2e0002ac7dcbc5793983c0f967474d4dcab21f8/tt_metal) |
| [Qwen 2.5 7B (TP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)   | 32     | [n300 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 86      | 24.6  | 38              | 787.2   | [v0.62.0-rc35](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc35) | [ced0161](https://github.com/tenstorrent/vllm/tree/ced0161dc223e6d8aca5f44a6c43d13070c3fba6/tt_metal) |
| [Qwen 2.5 72B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)               | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 223       | 15.4  | 20              | 492.8  | [v0.62.0-rc25](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc25) | [e7c329b](https://github.com/tenstorrent/vllm/tree/e7c329b1664f8591ae8b4269bed9690726e52a24/tt_metal) |
| [Falcon 7B](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/wormhole/falcon7b)                 | 32    | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 70        | 18.5  | 26              | 592.0  | [v0.62.0-rc25](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc25) |                                                                                                   |
| [Falcon 7B (DP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/t3000/falcon7b)             | 256   | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 87        | 15.9  | 26              | 4070.4 | [v0.62.0-rc25](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc25) |                                                                                                   |
| [Falcon 7B (DP=32)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/tg/falcon7b)               | 1024  | [Galaxy (Wormhole)](https://tenstorrent.com/hardware/galaxy)        | 121       | 13.2   | 26              | 13516.8 | [v0.62.0-rc25](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc25) |                                                                                                   |
| [Falcon 40B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/t3000/falcon40b)           | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) |           | 11.9   | 36              | 380.8  | [v0.59.0-rc38](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc38)  |                                                                                                   |
| [Llama 3.1 8B](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                         | 32     | [p100 (Blackhole)](https://tenstorrent.com/hardware/blackhole)        | 61*       | 28.7*   |               | 918.4*   | [v0.62.0-rc35](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc35) | [ced0161](https://github.com/tenstorrent/vllm/tree/ced0161dc223e6d8aca5f44a6c43d13070c3fba6/tt_metal) |
| [Llama 3.1 8B](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                         | 32     | [p150 (Blackhole)](https://tenstorrent.com/hardware/blackhole)        | 57*       | 32.8*   |               | 1049.6*   | [v0.62.0-rc35](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc35) | [ced0161](https://github.com/tenstorrent/vllm/tree/ced0161dc223e6d8aca5f44a6c43d13070c3fba6/tt_metal) |
| [Llama 3.1 8B (DP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                  | 64     | [2 x p150 (Blackhole)](https://tenstorrent.com/hardware/blackhole)        | 64*       | 18.6*   |               | 1190.4*  | [v0.59.0-rc3](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc3) | [739dcaa](https://github.com/tenstorrent/vllm/tree/739dcaa2915fa29d757c25a02c17aadce0c58055/tt_metal) |
| [Llama 3.1 8B](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                         | 32     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 104       | 26.0  | 23              | 832.0   | [v0.62.0-rc35](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc35) | [ced0161](https://github.com/tenstorrent/vllm/tree/ced0161dc223e6d8aca5f44a6c43d13070c3fba6/tt_metal) |
| [Llama 3.2 1B](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                         | 32     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 23        | 79.1  | 160             | 2531.2   | [v0.62.0-rc35](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc35) | [ced0161](https://github.com/tenstorrent/vllm/tree/ced0161dc223e6d8aca5f44a6c43d13070c3fba6/tt_metal) |
| [Llama 3.2 3B](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                         | 32     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 52       | 46.6  | 60              | 1491.2   | [v0.62.0-rc35](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc35) | [ced0161](https://github.com/tenstorrent/vllm/tree/ced0161dc223e6d8aca5f44a6c43d13070c3fba6/tt_metal) |
| [Mamba 2.8B](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/wormhole/mamba)                   | 32    | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 35        | 14.1  | 41              | 451.2  | [v0.59.0-rc38](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc38) |                                                                                                   |
| [Mistral 7B](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers)                        | 32    | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        |  99 | 28.7  | 23     | 918.4 | [v0.62.0-rc35](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc35)    |  [ced0161](https://github.com/tenstorrent/vllm/tree/ced0161dc223e6d8aca5f44a6c43d13070c3fba6/tt_metal) |
| [Mixtral 8x7B (TP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/t3000/mixtral8x7b)       | 32    | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 196       | 16.6  | 33              | 531.2  | [v0.62.0-rc8](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc8) |                                                                                                   |

> **Last Update:** Sept 8, 2025
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
| [Whisper (distil-large-v3)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/whisper)  | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 232       | 58.1  | 45           | 58.1   | [v0.59.0-rc52](https://github.com/tenstorrent/tt-metal/tree/v0.59.0-rc52) |

## Diffusion Models
| Model                                                                       | Batch | Hardware                                                 | Sec/Image     | Target Sec/Image | Release     |
|-----------------------------------------------------------------------------|-------|----------------------------------------------------------|---------|------------|-------------|
| [Stable Diffusion 1.4 (512x512)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/wormhole/stable_diffusion))  | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 6.25   | 3        |           |
| [Stable Diffusion 3.5 Medium (512x512)](https://github.com/tenstorrent/tt-metal/tree/main/models/experimental/stable_diffusion_35_large)  | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 16   | 10        |             |

**Notes:**
- Stable Diffusion sec/image is based on the time elapsed from submitting the input prompt to receiving the image from the VAE decoder.

## CNNs and Vision Transformers

### Classification models

| Model                                                                       | Batch | Hardware                                                 | Image/sec    | Target Image/sec | Release     |
|-----------------------------------------------------------------------------|-------|----------------------------------------------------------|---------|------------|-------------|
| [ResNet-50 (224x224)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/wormhole/resnet50)                     | 16    | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 4,700   | 7,000      | [v0.59.0](https://github.com/tenstorrent/tt-metal/tree/v0.59.0) |
| [ResNet-50 (224x224) (DP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/t3000/resnet50)                 | 32    | [n300 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 9,200   | 14,000     | [v0.59.0](https://github.com/tenstorrent/tt-metal/tree/v0.59.0) |
| [ResNet-50 (224x224) (DP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/t3000/resnet50)                 | 128   | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox) | 35,800  | 56,000     | [v0.59.0](https://github.com/tenstorrent/tt-metal/tree/v0.59.0) |
| [ResNet-50 (224x224) (DP=32)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/tg/resnet50)                   | 512   | [Galaxy (Wormhole)](https://tenstorrent.com/hardware/galaxy)        | 96,800  | 224,000    | [v0.59.0](https://github.com/tenstorrent/tt-metal/tree/v0.59.0) |
| [ViT-base (224x224)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/wormhole/vit)                                | 8     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 1,370    | 1,600      | [v0.60.0-rc4](https://github.com/tenstorrent/tt-metal/tree/v0.60.0-rc4) |
| [ViT-base (224x224)  (DP=2)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/t3000/vit)                                | 16     | [n300 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 1,900   | 3,200      | [v0.60.0-rc4](https://github.com/tenstorrent/tt-metal/tree/v0.60.0-rc4) |
| [ViT-base (224x224)  (DP=8)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/t3000/vit)                                | 64     | [QuietBox (Wormhole)](https://tenstorrent.com/hardware/tt-quietbox)        | 7,700    | 12,800      | [v0.60.0-rc4](https://github.com/tenstorrent/tt-metal/tree/v0.60.0-rc4) |
| [MobileNet-v2 (224x224)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/mobilenetv2)                     | 10    | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        |  2,808  | 3,500      |  |

### Object Detection
| Model                                                                       | Batch | Hardware                                                 | Frame/sec (FPS)     | Target FPS | Release     |
|-----------------------------------------------------------------------------|-------|----------------------------------------------------------|---------|------------|-------------|
| [YOLOv4 (320x320)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov4)                                   | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 120     | 320        |             |
| [YOLOv4 (640x640)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov4)                                   | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 50      | 180        |             |
| [YOLOv8x (640x640)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov8x)                                   | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 45      | 100        |             |
| [YOLOv8s (640x640)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov8s)                                   | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 175      | 320        |             |
| [YOLOv8s_world (640x640)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov8s_world)                                   | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 57      | 200        |             |
| [YOLOv9c (640x640)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov9c)                                   | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 55      | 320        |             |
| [YOLOv10x (640x640)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov10x)                                   | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 26      | 200        |             |

### Segmentation
| Model                                                                       | Batch | Hardware                                                 | Frame/sec (FPS)     | Target FPS | Release     |
|-----------------------------------------------------------------------------|-------|----------------------------------------------------------|---------|------------|-------------|
| [UNet - VGG19 (256x256)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/vgg_unet)                                   | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 77      | 150        |             |
| [SegFormer Semantic Segmentation (512x512)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/segformer)       | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 84      | 300        |              |
| [YOLOv9c (640x640)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/yolov9c)                                   | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 40      | 240        |             |
| [UFLD - v2 (320x800)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/ufld_v2)                                   | 1     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)        | 255      | 2000        |             |

## NLPs

| Model                                               | Batch | Hardware                                           | Sentence/sec | Target sentence/sec | Release |
|-----------------------------------------------------|-------|----------------------------------------------------|---------|----------------|---------|
| [BERT-Large](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/metal_BERT_large_11)   | 8     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)  | 270     | 400            |         |
| [Sentence-Bert (backbone: bert-base)](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/sentence_bert)   | 8     | [n150 (Wormhole)](https://tenstorrent.com/hardware/wormhole)  | 233     | 360            |         |

## Model Demos

- [Demo models on Grayskull](../README.md#grayskull-gs-models)
- [Demo models on Wormhole](../README.md#wormhole-wh-models)
- [Demo models on TT-QuietBox & TT-LoudBox (Wormhole)](../README.md#tt-quietbox--tt-loudbox-2x4-mesh-of-whs-models)
- [Demo models on Single Galaxy (Wormhole)](../README.md#single-galaxy-8x4-mesh-of-whs-models)

## Latest Releases

| Release | Release Date |
|---------|--------------|
| [0.62.2](https://github.com/tenstorrent/tt-metal/releases/tag/v0.62.2)  | Aug 20, 2025 |
| 0.61.0  | Skipped |
| [0.60.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.60.0) | Jul 16, 2025 |
| [0.59.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.59.0) | Jun 18, 2025 |
| [0.58.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.58.0) | May 13, 2025 |
| [0.57.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.57.0) | Apr 15, 2025 |
| [0.56.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.56.0) | Mar 7, 2025 |

Visit the [releases](https://github.com/tenstorrent/tt-metal/tree/main/releases) folder for details on releases, release notes, and estimated release dates.

## Release methodology

- Models that are advertised as part of release, usually the demo models, are treated as first-class citizens, and therefore are treated as tests.
- Model writers are responsible for ensuring their demo model tests are always passing. Any failure is treated highest-priority (or P0) failure.
- Model writers are responsible for advertising which release tag (including release candidates) contains passing tests for their demo models.
- Model writers are responsible for updating their perf metrics for the demo models at a regular cadence. Currently, the cadence is at least every 2 weeks.
