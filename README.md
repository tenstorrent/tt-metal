<div align="center">

<h1>

[Buy hardware](https://tenstorrent.com/cards/) | [Install](./INSTALLING.md) | [Discord](https://discord.gg/tvhGzHQwaj)

</h1>

<img src="./docs/source/common/_static/tt_nn_w_logo.png" alt="ttnn logo" height="150"/>

**TT-NN** is python & C++ Neural Network OP library.

<h3>

[API Reference](https://tenstorrent.github.io/tt-metal/latest/ttnn) | [Model Demos](./models/demos/)

</h3>

</div>

---

## Grayskull (GS) Models

| Model                                                    | Batch               | End-to-end throughput [1]    | Device throughput [2]       | Target                              |
|----------------------------------------------------------|---------------------|------------------------------|-----------------------------|-------------------------------------|
| [ResNet-50](./models/demos/resnet) (fps)                 | 20                  | 2,070                        | 7,200                       | 10,000                              |
| [BERT-Large](./models/demos/bert) (sen/s)                | 12                  | 362                          | 406                         | 410                                 |
| [Falcon7B-decode](./models/demos/ttnn_falcon7b) (t/s)    | 32                  | 135                          | 135                         | 140                                 |
| U-Net                                                    | coming soon         |                              |                             |                                     |
| T5 small                                                 | coming soon         |                              |                             |                                     |
| Bloom                                                    | coming soon         |                              |                             |                                     |

[1] - Observed from the host. Includes dispatch overahed and kernel execution time.

[2] - Ignoring host overhead. Kernel execution time only.

## Wormhole (WH) Models

| Model                                                       | Gen. Token         |  Batch               | End-to-end throughput [1]   | Device throughput [2]       | Target         |
|-------------------------------------------------------------|--------------------|----------------------|-----------------------------|-----------------------------|----------------|
| [Falcon7B-decode](./models/demos/wormhole/falcon7b) (t/s/u) | 129th              | 32                   | 9.9                         | 13.5                        | 14             |
| [Mistral-7B-decode](./models/demos/mistral7b) (t/s/u)       |  33rd              | 32                   | 7.9                         | 10.9                        | 14             |
| [Mamba-2.8B-decode](./models/demos/mamba) (t/s/u)           |  any               | 32                   | 1.7                         | 2.0                         | 17             |
| Stable Diffusion 1.4 512x512                                | 1                  | coming soon          |                             |                             |                |


## T3000 (2x4 mesh of WHs) Models

| Model                                                         | Gen. Token         |  Batch               | End-to-end throughput [1]   | Device throughput [2]       | Target         |
|---------------------------------------------------------------|--------------------|----------------------|-----------------------------|-----------------------------|----------------|
| [LLaMA-2-70B-decode](./models/demos/t3000/llama2_70b) (t/s/u) | 129th              |  32                  | 0.8                         |  8.4                        |   11.4         |
| [Falcon40B-decode](./models/demos/falcon40b)                  | coming soon        |                      |                             |                             |                |
| Mixtral7Bx8-decode                                            | coming soon        |                      |                             |                             |                |
| ResNet50 (data parallel)                                      | coming soon        |                      |                             |                             |                |

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
