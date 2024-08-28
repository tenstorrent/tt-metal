<div align="center">

<h1>

[Buy hardware](https://tenstorrent.com/cards/) | [Install](./INSTALLING.md) | [Discord](https://discord.gg/tvhGzHQwaj) | [Join Us](https://boards.greenhouse.io/tenstorrent/jobs/4155609007)

</h1>

<img src="./docs/source/common/_static/tt_nn_w_logo.png" alt="ttnn logo" height="150"/>

**TT-NN** is a Python & C++ Neural Network OP library.

<h3>

[API Reference](https://docs.tenstorrent.com/ttnn/latest/index.html) | [Model Demos](./models/demos/)

</h3>

</div>

---

## Grayskull (GS) Models

| Model                                                      | Batch               | End-to-end throughput [1]    | Device throughput [2]       | Target throughput                             |
|----------------------------------------------------------  |---------------------|------------------------------|-----------------------------|-------------------------------------|
| [ResNet-50](./models/demos/resnet) (fps)                   | 20                  | 5,500                        | 7,700                       | 10,000                              |
| [BERT-Large](./models/demos/bert) (sen/s)                  | 12                  | 370                          | 406                         | 410                                 |
| [Falcon7B-decode](./models/demos/ttnn_falcon7b) (t/s)      | 32                  | 135                          | 135                         | 140                                 |
| [ViT](./models/demos/grayskull/vit) (fps)                  | 8                   | 860                          | 1570                        | 2000                                |
| [T5 small](.models/demos/grayskull/t5) (sen/s)             |                     | 140                          |                             |                                     |
| [Bloom](.models/demos/grayskull/functional_bloom) (sen/s)  |                     | 70                           |                             |                                     |
| U-Net                                                      | coming soon         |                              |                             |                                     |

[1] - Observed from the host. Includes dispatch overhead and kernel execution time. For LLMs, token-to-token decode throughput is reported.

[2] - Ignoring host overhead. Kernel execution time only. For LLMs, token-to-token decode throughput is reported.

## Wormhole (WH) Models

> [!NOTE]
>
> All model demos in this table function on both N150 and N300 Wormhole cards, unless otherwise stated.
>
> Furthermore, all performance numbers here are run or based off an N300 Wormhole card.

| Model                                                                                  | Last verified release                                                     | Gen. Token [3]     |  Batch               | Time to first token [4] | End-to-end throughput [1]      | Device throughput [2]        | Target throughput        |
|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------|--------------------|----------------------|-------------------------| ---------------------------------|------------------------------|----------------|
| [Falcon7B](./models/demos/wormhole/falcon7b)                                           | [v0.51.0-rc24](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc24) | 129th              | 32                   | 0.08 s                  | 16.7 t/s/u - 534 t/s           | 19.6 t/s/u - 627 t/s         | 26             |
| [Mistral-7B](./models/demos/wormhole/mistral7b)                                        | [v0.51.0-rc28](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc28) | 129th              | 32                   | coming soon             | 9.9 t/s/u - 317 t/s            | 11.0 t/s/u - 352 t/s         | 25             |
| [Mamba-2.8B](./models/demos/wormhole/mamba)                                            | [v0.51.0-rc26](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc26) | any                | 32                   | 0.04 s                  | 12.3 t/s/u - 394 t/s           | 17.1 t/s/u - 547 t/s         | 41             |
| [LLaMA-3.1-8B](./models/demos/wormhole/llama31_8b)                                     | [v0.51.0-rc28](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc28) | 129th              | 1                    | coming soon             | 8.3 t/s/u - 8.3 t/s           | 9.7 t/s/u - 9.7 t/s         | 23             |
| [BERT-Large](./models/demos/metal_BERT_large_11/) (sen/s) [5]                          |                                                                           | -                   | 8                    | -                       | 270                            | 340                          | 400            |
| [Stable Diffusion 1.4](./models/demos/wormhole/stable_diffusion) 512x512 (sec/img) [6] |                                                                           | -                   | 1                    | -                       | 6                              | 5                            | 3              |
| [ResNet-50](./models/demos/ttnn_resnet) (fps)                                          |                                                                           | -                   | 16                   | -                       | 4,300                          | 5,550                        | 7,000          |

[1] - Observed from the host. Includes dispatch overhead and kernel execution time. For LLMs, token-to-token decode throughput is reported.

[2] - Ignoring host overhead. Kernel execution time only. For LLMs, token-to-token decode throughput is reported.

[3] - Generating the `i`'th token in a sequence while the kv_cache is filled with `i-1` rows.

[4] - Time to fill the kv_cache and generate the first output token (1st user).

[5] - This model demo does not work on N150. It does work on N300.

[6] - This model demo does not work on N300. It does work on N150.

##  TT-QuietBox & TT-LoudBox (2x4 mesh of WHs) Models

| Model                                              | Last verified release                                                     |   Technique        | Gen. Token [3]      |  Batch                | Time to first token [4] | End-to-end throughput [1]    | Device throughput [2]        | Target throughput          |
|----------------------------------------------------|---------------------------------------------------------------------------|--------------------|---------------------|-----------------------|-------------------------|------------------------------|------------------------------|-----------------|
| [Falcon7B](./models/demos/t3000/falcon7b)          | [v0.51.0-rc36](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc36) | Data Parallel      | 129th               |  256                  | 0.11 s                  | 13.4 t/s/u - 3430 t/s         |  19.6 t/s/u - 5018 t/s       |   26 t/s/u      |
| [LLaMA-2-70B](./models/demos/t3000/llama2_70b)     | [v0.51.0-rc36](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc36) | Tensor Parallel    | 129th               |  32                   | coming soon             | 10.4 t/s/u - 333 t/s         |  16.6 t/s/u - 531 t/s        |   20 t/s/u      |
| [LLaMA-3.1-70B](./models/demos/t3000/llama3_70b)   | [v0.51.0-rc36](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc36) | Tensor Parallel    | 129th               |  32                   | coming soon             | 10.4 t/s/u - 333 t/s         |  15.8 t/s/u - 506 t/s        |   20 t/s/u      |
| [Falcon40B](./models/demos/t3000/falcon40b)        | [v0.51.0-rc35](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc35) | Tensor Parallel    | 129th               |  32                   | coming soon             | 5.3 t/s/u - 168 t/s          |  12.2 t/s/u - 390 t/s        |   36 t/s/u      |
| [Mixtral7Bx8](./models/demos/t3000/mixtral8x7b)    | [v0.51.0-rc33](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc33) | Tensor Parallel    | 129th               |  32                   | 0.19 s                  | 15.7 t/s/u - 502 t/s         |  21.4 t/s/u - 685 t/s        |   33 t/s/u      |
| [ResNet-50](./models/demos/ttnn_resnet) (fps)      |                                                                           | Data Parallel      | -                   |  128                  | -                       | 31,700                       |  44,400                      |   56,000        |

## Single Galaxy (8x4 mesh of WHs) Models

| Model                                              | Last verified release                                                     |   Technique        | Gen. Token [3]      |  Batch                | Time to first token [4] | End-to-end throughput [1]    | Device throughput [2]        | Target throughput |
|----------------------------------------------------|---------------------------------------------------------------------------|--------------------|---------------------|-----------------------|-------------------------|------------------------------|------------------------------|-------------------|
| [Falcon7B](./models/demos/tg/falcon7b)             | [v0.51.0-rc30](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc30) | Data Parallel      | 129th               |  1024                 | 0.30 s                  | 4.0 t/s/u - 4096 t/s         | 17.7 t/s/u - 18125 t/s       |   26 t/s/u        |



## Model Updates
For the latest model updates and features, please see [MODEL_UPDATES.md](models/MODEL_UPDATES.md)


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

[Programming Guide](./METALIUM_GUIDE.md) | [API Reference](https://docs.tenstorrent.com/tt-metalium/latest/tt_metal/apis/index.html)

</h3>
</div>

## Getting started

Get started with [simple kernels](https://docs.tenstorrent.com/tt-metalium/latest/tt_metal/examples/index.html).
