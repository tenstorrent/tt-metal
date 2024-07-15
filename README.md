<div align="center">

<h1>

[Buy hardware](https://tenstorrent.com/cards/) | [Install](./INSTALLING.md) | [Discord](https://discord.gg/tvhGzHQwaj)

</h1>

<img src="./docs/source/common/_static/tt_nn_w_logo.png" alt="ttnn logo" height="150"/>

**TT-NN** is a Python & C++ Neural Network OP library.

<h3>

[API Reference](https://tenstorrent.github.io/tt-metal/latest/ttnn) | [Model Demos](./models/demos/)

</h3>

</div>

---

## Grayskull (GS) Models

| Model                                                      | Batch               | End-to-end throughput [1]    | Device throughput [2]       | Target                              |
|----------------------------------------------------------  |---------------------|------------------------------|-----------------------------|-------------------------------------|
| [ResNet-50](./models/demos/resnet) (fps)                   | 20                  | 5,100                        | 7,700                       | 10,000                              |
| [BERT-Large](./models/demos/bert) (sen/s)                  | 12                  | 370                          | 406                         | 410                                 |
| [Falcon7B-decode](./models/demos/ttnn_falcon7b) (t/s)      | 32                  | 135                          | 135                         | 140                                 |
| [ViT](./models/demos/grayskull/vit) (fps)                  | 8                   | 860                          | 1570                        | 2000                                |
| [T5 small](.models/demos/grayskull/t5) (sen/s)             |                     | 140                          |                             |                                     |
| [Bloom](.models/demos/grayskull/functional_bloom) (sen/s)  |                     | 70                           |                             |                                     |
| U-Net                                                      | coming soon         |                              |                             |                                     |

[1] - Observed from the host. Includes dispatch overhead and kernel execution time.

[2] - Ignoring host overhead. Kernel execution time only.

## Wormhole (WH) Models

> [!NOTE]
>
> All model demos in this table function on both N150 and N300 Wormhole cards, unless otherwise stated.

| Model                                                                                | Gen. Token [3]     |  Batch               | End-to-end throughput [1]    | Device throughput [2]       | Target         |
|--------------------------------------------------------------------------------------|--------------------|----------------------|------------------------------|-----------------------------|----------------|
| [Falcon7B-decode](./models/demos/wormhole/falcon7b)                                  | 129th              | 32                   | 13.3 t/s/u - 425 t/s         | 15.4 t/s/u - 493 t/s        | 26             |
| [Mistral-7B-decode](./models/demos/wormhole/mistral7b)                               | 129th              | 32                   | 9.9 t/s/u - 317 t/s         | 11.0 t/s/u - 352 t/s        | 25             |
| [Mamba-2.8B-decode](./models/demos/mamba)                                            | any                | 32                   | 11.6 t/s/u - 370 t/s         | 16.5 t/s/u - 528 t/s        | 41             |
| [BERT-Large](./models/demos/metal_BERT_large_11/) (sen/s) [4]                        |                    | 8                    | 270                          | 340                         | 400            |
| [Stable Diffusion 1.4](./models/demos/wormhole/stable_diffusion) 512x512  (sec/img)  |                    | 1                    | 6                            | 5                           |                |

[1] - Observed from the host. Includes dispatch overhead and kernel execution time.

[2] - Ignoring host overhead. Kernel execution time only.

[3] - Generating the `i`'th token in a sequence while the kv_cache is filled with `i-1` rows.

[4] - This model demo does not work on N150. It does work on N300.

## T3000 (2x4 mesh of WHs) Models

| Model                                                     |   Technique        | Gen. Token [3]      |  Batch                | End-to-end throughput [1]    | Device throughput [2]        | Target          |
|-----------------------------------------------------------|--------------------|---------------------|-----------------------|------------------------------|------------------------------|-----------------|
| [Falcon7B-decode](./models/demos/t3000/falcon7b)          | Data Parallel      | 129th               |  256                  |  7.4 t/s/u - 1901 t/s        |  15.5 t/s/u - 3968 t/s       |   26 t/s/u      |
| [LLaMA-2-70B-decode](./models/demos/t3000/llama2_70b)     | Tensor Parallel    | 129th               |  32                   | 10.4 t/s/u - 333 t/s         |  16.6 t/s/u - 532 t/s        |   20 t/s/u      |
| [LLaMA-3-70B-decode](./models/demos/t3000/llama3_70b)     | Tensor Parallel    | 129th               |  32                   | 10.4 t/s/u - 333 t/s         |  15.8 t/s/u - 506 t/s        |   20 t/s/u      |
| [Falcon40B-decode](./models/demos/t3000/falcon40b)        | Tensor Parallel    | 129th               |  32                   | work-in-progress             |  10.0 t/s/u - 320 t/s        |   36 t/s/u      |
| [Mixtral7Bx8-decode](./models/demos/t3000/mixtral8x7b)    | Tensor Parallel    | 129th               |  32                   | 15.1 t/s/u - 483 t/s          |  27.1 t/s/u - 868 t/s        |   33 t/s/u      |
| ResNet50                                                  | Data Parallel      | coming soon         |                       |                              |                              |                 |

## Using TT-NN ops and tensors

```python
import ttnn
import torch

with ttnn.manage_device(device_id=0) as device:
   a = torch.ones((5, 7))
   b = torch.ones((1, 7))

   a = ttnn.from_torch(a, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
   b = ttnn.from_torch(b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

   output = a + b
   output = ttnn.to_torch(output)

print(output)
```

---

<div align="center">

<img src="./docs/source/common/_static/tt_metalium_w_logo.png" alt="TT-Metalium logo" height="150"/>

**TT-Metalium** is our low-level programming model, enabling kernel development for Tenstorrent hardware.


<h3>

[Programming Guide](./METALIUM_GUIDE.md) | [API Reference](https://tenstorrent.github.io/tt-metal/latest/tt-metalium)

</h3>
</div>

## Getting started

Get started with [simple kernels](https://tenstorrent.github.io/tt-metal/latest/tt-metalium/tt_metal/examples/index.html).
