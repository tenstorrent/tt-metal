# NanoGPT on Tenstorrent - Quick Start

## Setup

```bash
export TT_METAL_HOME=/home/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/tt-metal
```

## Build

```bash
cd /home/tt-metal/tt-train/build
cmake .. && make nano_gpt -j$(nproc)
```

## Run

```bash
tt-smi -r 0
./build/sources/examples/nano_gpt/nano_gpt
```

Config: `configs/training_shakespeare_nanogpt.yaml`
