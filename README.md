<div align="center">

<img src="./docs/source/_static/tt_metalium_w_logo.png" alt="TT-Metalium logo" height="250"/>
<img src="./docs/source/_static/tt_nn_w_logo.png" alt="ttnn logo" height="250"/>

**TT-Metalium** is our low-level programming model, enabling kernel development for Tenstorrent hardware.

**TT-NN** is our neural network OP library with a PyTorch-like Python, and C++ API.

<h3>

[Documentation](https://tenstorrent-metal.github.io/tt-metal/latest/index.html) | [Working models](./models/demos/) | [Discord](https://discord.gg/tvhGzHQwaj) | [Tenstorrent website](https://tenstorrent.com)

</h3>

</div>

---

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

You can use simple conversion APIs to use PyTorch tensors with Tenstorrent hardware.

## Tracing graphs for TT-NN operators

```python
import ttnn
import torch

with ttnn.manage_device(device_id=0) as device, ttnn.tracer.trace():
   a = torch.ones((5, 7))
   b = torch.ones((1, 7))

   a = ttnn.from_torch(a, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
   b = ttnn.from_torch(b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

   output = a + b
   output = ttnn.to_torch(output)

ttnn.tracer.visualize(output)
```

We also provide tools to review the graphs you create and run in TT-NN.

<img src="./docs/source/_static/add.svg" alt="ttnn tracer example" height="250"/>

## Running performant out-of-the-box models

We have working demos for models such as [ResNet](./models/demos/resnet), [BERT](./models/demos/bert), [Falcon7B](./models/demos/ttnn_falcon7b) and [Falcon40B](./models/demos/falcon40b).

These are some performance metrics for our models running on Grayskull (GS). We constantly improve these
and publish these metrics on [GitHub
Actions](https://github.com/tenstorrent-metal/tt-metal/actions/workflows/perf-models.yaml).

| Model                            | Batch size          | GS end-to-end throughput [1] | GS on-device throughput [2] | Target GS end-to-end throughput [1] |
|----------------------------------|---------------------|------------------------------|-----------------------------|-------------------------------------|
| ResNet-50 (fps)                  | 20                  | 2070                         | 6943                        | 10000                               |
| BERT-Large (sen/s)                  | 12                  | 362                          | 406                         | 410                                 |
| TT-NN Falcon-7B decode (t/s)     | 32                  | 135                          | coming soon      | 140                                 |
| ViT                              | Coming end of March |                              |                             |                                     |
| U-Net                            | Coming end of March |                              |                             |                                     |

[1] - Throughput is measured by taking batch size and dividing by accelerator inference time, and reported per sec.

[2] - Throughput on device is measured by directly counting the clock cycles for operations done on device.

## What's coming for models

We are writing efficient versions of the following models to run on our Wormhole (WH) architecture (N300 2x WH card):

| Model            | N300 (2x WH) Throughput |
|------------------|-------------------------|
| Falcon-7B        | Coming end of March     |
| Mistral-7B        | Coming end of March     |
| Mamba-2.8B       | Coming end of March     |
| Stable Diffusion | Coming end of March     |

And are also mapping models to run on our T3000 workstation with a 2x4 mesh of Wormhole devices (8x WH), and Galaxy systems with a 4x8 mesh of Wormhole devices (32x WH):

| Model       | T3000 (8x WH) Throughput  | Galaxy (32x WH) Throughput |
|-------------|---------------------------|----------------------------|
| Falcon-40B  | Coming end of March       | Coming end of March        |
| LLaMA-2-70B | Coming end of March       | Coming end of March        |
| Mixtral7Bx8 | Coming end of March       | Coming end of March        |

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
page](https://tenstorrent-metal.github.io/tt-metal/latest/get_started/get_started.html)
in our documentation.

Note that example programs are only available through source installation at
this time.

## Documentation

Please refer to our
[documentation](https://tenstorrent-metal.github.io/tt-metal/latest/index.html).

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
