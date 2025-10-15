# Ultra-Fast-Lane-Detection-v2

## Platforms:
    Wormhole (n150, n300), Blackhole (p150)

## Introduction
The Ultra-Fast-Lane-Detection-v2 is a PyTorch-based implementation designed for fast and efficient deep lane detection using hybrid anchor-driven ordinal classification. It enhances the speed and accuracy of lane detection models with a focus on real-time applications.

Resource link - [source](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run:

Find ufld_v2 instructions for the following devices:

- Wormhole (n150, n300): [models/demos/wormhole/ufld_v2](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/wormhole/ufld_v2)

- Blackhole (p150): [models/demos/blackhole/ufld_v2](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/blackhole/ufld_v2)
