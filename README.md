<div align="center">

<h1>
   
[Buy hardware](https://tenstorrent.com/cards/) | [Discord](https://discord.gg/tvhGzHQwaj)

</h1>
   
<img src="./docs/source/common/_static/tt_nn_w_logo.png" alt="ttnn logo" height="250"/>

**TT-NN** is python & C++ Neural Network OP library.

<h3>

[TT-NN API Reference](https://tenstorrent-metal.github.io/tt-metal/latest/ttnn) | [Model Demos](./models/demos/) 

</h3>

</div>

---

## Grayskull (GS) Models

| Model                                                    | Batch               | End-to-end throughput        | Device throughput [1] | Target                              |
|----------------------------------------------------------|---------------------|------------------------------|-----------------------------|-------------------------------------|
| [ResNet-50](./models/demos/resnet) (fps)                 | 20                  | 2,070                        | 7,200                       | 10,000                              |
| [BERT-Large](./models/demos/bert) (sen/s)                | 12                  | 362                          | 406                         | 410                                 |
| [Falcon7B-decode](./models/demos/ttnn_falcon7b) (t/s)    | 32                  | 135                          | 135                         | 140                                 |
| U-Net                                                    | coming soon         |                              |                             |                                     |
| T5 small                                                 | coming soon         |                              |                             |                                     |
| Bloom                                                    | coming soon         |                              |                             |                                     |

[1] - Throughput on device ignores the overhead of host runtime, which is being currently optimized.

## Wormhole (WH) Models

| Model                                                    | Batch               | End-to-end throughput        | Device throughput [1] | Target                              |
|----------------------------------------------------------|---------------------|------------------------------|-----------------------------|-------------------------------------|
| Falcon-7B-decode (t/s/u)                                 | 32                  | 6.6                          | 11.6                        | 14                                  |
| Mistral-7B-decode (t/s/u)                                | 32                  | 3.3                          | 12.6                        | 14                                  |
| Mamba-2.8B-decode (t/s/u)                                | 32                  | coming soon                  |                             | 17                                  |
| Stable Diffusion 1.4 512x512                             | 1                   | coming soon                  |                             |                                     |


## LoudBox (2x4 mesh of WHs) Models 

| Model                                    | Batch                    | Throughput |
|------------------------------------------|--------------------------|----------------------------|
| [Falcon40B](./models/demos/falcon40b)    | coming soon              |               |
| [LLaMA-2-70B](./models/demos/llama2_70b) | coming soon              |               |
| Mixtral7Bx8                              | coming soon              |               |




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

<img src="./docs/source/common/_static/tt_metalium_w_logo.png" alt="TT-Metalium logo" height="250"/>

**TT-Metalium** is our low-level programming model, enabling kernel development for Tenstorrent hardware.


<h3>

[TT-Metalium Programming Guide](./METALIUM_GUIDE.md)
   
[TT-Metalium API Reference](https://tenstorrent-metal.github.io/tt-metal/latest/tt-metalium)

</h3>
</div>

## Table of contents

<!-- toc -->

- [Installing](#installing)
- [Getting started](#getting-started)
- [Documentation](#documentation)
- [Troubleshooting and debugging tips](#troubleshooting-and-debugging-tips)
- [Contributing](#contributing)
- [Communication](#communication)

<!-- tocstop -->

Table of Contents generated with
[markdown-toc](https://github.com/luciopaiva/markdown-toc).

## Installing

**Note**: Currently, all features are only fully tested on Grayskull E150
accelerators. We are currently working on functionality for other Tenstorrent
architectures.

To find through all necessary instructions for setting up your Tenstorrent
accelerator and this software, please refer to our full [installation
instructions](./INSTALLING.md).

You should look ahead to [Getting started](#getting-started) to further use
this project.

## Getting started

### Environment setup

If you just came reading from building from source, you can read ahead to
[running an example](#running-example-programs).

Otherwise, you must set up the necessary environment variables to use this
project every time:

```
export ARCH_NAME=<arch name>
export TT_METAL_HOME=<appropriate value based on installation method above>
```

where ``<arch name>`` is your target, which could be:

- ``grayskull``
- ``wormhole_b0``

etc...

If you're setting up an environment from source, you must further set up and
activate the environment with:

```
export PYTHONPATH=<this repo dir>
export TT_METAL_ENV=dev
source build/python_env/bin/activate
```

### Running example programs

After installing, please refer to our [Getting Started
page](https://tenstorrent-metal.github.io/tt-metal/latest/tt-metalium/get_started/get_started.html)
in our documentation.

Note that example programs are only available through source installation at
this time.

## Documentation

Please refer to our documentation:

- [TT-Metalium](https://tenstorrent-metal.github.io/tt-metal/latest/tt-metalium)
- [TT-NN](https://tenstorrent-metal.github.io/tt-metal/latest/ttnn)

## Troubleshooting and debugging tips

In addition to our documentation above, you can check out relevant sections in
the [contribution
standards](https://github.com/tenstorrent-metal/tt-metal/blob/main/CONTRIBUTING.md)
if you ever need hardware troubleshooting help or debugging tips.

## Contributing

We are excited to move our development to the public, open-source domain.
However, we are not adequately staffed to review contributions in an expedient
and manageable time frame at this time. In the meantime, please review the
[contributor's guide](CONTRIBUTING.md) for more information about contribution
standards.

If you would like to contribute, your submissions **must** pass post-commit
regressions. If you would like more information on running tests locally and
CI, please refer to the relevant section in the the [contributor's
guide](CONTRIBUTING.md) and read it in its entirety.

## Communication

Announcements from the Tenstorrent team regarding this project will be in the
[discussions
page](https://github.com/tenstorrent-metal/tt-metal/discussions/categories/general-announcements).

We also have a Discord channel that you can join. You may discuss with other
members of the community and developers there. You may use this invite
[link](https://discord.gg/tvhGzHQwaj).
If you would like to formally propose a new feature, report a bug, or have
issues with permissions, please file through [GitHub
issues](https://github.com/tenstorrent-metal/tt-metal/issues/new/choose).
