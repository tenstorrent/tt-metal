# tt-train: CPP ML training framework

## Overview
This repository contains a high-performance training framework developed in C++ designed to efficiently leverage the computational capabilities of Tenstorrent hardware. The framework is optimized to accelerate model training tasks, providing a seamless interface for training deep learning models on Tenstorrent's advanced hardware architecture.

# Prerequisites

tt-train is built as part of tt-metal. Before building, ensure you have:

1. Cloned the tt-metal repository with submodules:
```bash
git submodule update --init --recursive
```

2. Followed the tt-metal setup instructions in the main repository README.

# Building the project

tt-train is built using the `build_metal.sh` script from the tt-metal root directory.

## Terminal

```bash
# Release build (default)
./build_metal.sh --build-tt-train

# Debug build
./build_metal.sh --build-tt-train --debug

# Development build (RelWithDebInfo)
./build_metal.sh --build-tt-train --development
```

## VSCode

1. Open the tt-metal root directory in VSCode
2. Install the [CMake](https://marketplace.visualstudio.com/items?itemName=twxs.cmake) extension
3. Set `BUILD_TT_TRAIN=ON` in your CMake configure settings
4. Build all targets using the CMake extension


# Run
## MNIST
### Training
```
# Navigate to the root directory of the repository
./build/sources/examples/mnist_mlp/mnist_mlp --model_path mnist_mlp.msgpack --num_epochs 10
```
### Evaluation
```
# Navigate to the root directory of the repository
./build/sources/examples/mnist_mlp/mnist_mlp --model_path mnist_mlp.msgpack -e 1
```

## NanoGPT Shakespeare
### Training
```
# Navigate to the root directory of the repository
TT_LOGGER_LEVEL=FATAL ./build/sources/examples/nano_gpt/nano_gpt
```

Training loss example from [wandb project](https://wandb.ai/tenstorrent-ml/tt_train_nano_gpt):
![NanoGPT training wandb chart](./images/nano-gpt-training-example.png)


For more information on training with the C++ version, please see the [NanoGPT Example README](sources/examples/nano_gpt/README.md).

For both training and evaluation, we encourage you to use the Python version of NanoGPT, which can be used via Jupyter Notebook found in the NanoGPT example directory.

More information on available configuration options can be found in the [configs directory](configs/README.md).

### Nightly only tests
TT-Train nightly tests are all gtests whose names contain `NIGHTLY_` (see
`tt-train/tests/model/nano_gpt_test.cpp`). They run only on Wormhole B0 when
**both** are true:

- `is_nigthly_tt_train_tests_enabled` is set to `true` in `nano_gpt_test.cpp`
  (spelling matches the source constant)
- `ENABLE_NIGHTLY_TT_TRAIN_TESTS` is set in the environment (for example
  `ENABLE_NIGHTLY_TT_TRAIN_TESTS=1`)

If CI fails but local runs pass, confirm both conditions before changing
expected values or thresholds. To run them in GitHub Actions, search for
`Nightly tt-metal L2 tests` (tt-train jobs export `ENABLE_NIGHTLY_TT_TRAIN_TESTS=1`
via `tests/pipeline_reorg/train_unit_tests.yaml`; the compile-time flag must
still be enabled in code).

### wandb support
If you don't have an account to wandb (or don't want to use it), use `-w 0` argument or run `wandb offline` beforehand (creates `wandb/settings` file)

### GPU baseline
[This repository](https://github.com/philei-tt/tt-train_nanoGPT-gpu-baseline) can be used to compare speed and accuracy to GPU implementation of nanoGPT and gpt2s

# Profiler
Use of profiler requires additional setup. Follow instructions [here](./docs/PROFILER.md).

# Contributing
* Create a new branch.
* Make your changes and commit them.
* Add new tests and run existing ones
* Open a pull request (PR).
* Ensure the PR is approved by at least one code owner before merging.
