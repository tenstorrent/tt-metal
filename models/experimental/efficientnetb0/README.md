# Efficientnetb0

## Platforms:
Wormhole (n150, n300)

## Introduction
EfficientNet-B0 is a lightweight and efficient convolutional neural network architecture developed by Google AI. Model is know for its efficiency in image classification tasks. It's a member of the EfficientNet family, which utilizes a compound scaling method to balance model size, accuracy, and computational cost. EfficientNetB0 was trained on the massive ImageNet dataset and can classify images into 1000 object categories.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`
- Login to huggingface using your token: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
  - To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens


## How to run

- Use the following command to run the `EfficientNetb0` model:

  ```sh
  pytest models/experimental/efficientnetb0/tests/pcc/test_ttnn_efficientnetb0.py::test_efficientnetb0_model
  ```

### Model performant running with Trace+2CQs

#### Single Device (BS=1)

- For `224x224`, end-2-end perf is `74` FPS :

  ```sh
  pytest models/experimental/efficientnetb0/tests/perf/test_e2e_performant.py::test_e2e_performant
  ```

#### Multi Device (DP=2, N300)

- For `224x224`, end-2-end perf is `146` FPS :

  ```sh
  pytest models/experimental/efficientnetb0/tests/perf/test_e2e_performant.py::test_e2e_performant_dp
  ```

## Model Demo with Trace+2CQs

#### Single Device (BS=1)

- Use the following command to run demo for 224x224 resolution:

  ```sh
  pytest models/experimental/efficientnetb0/demo/demo.py::test_demo
  ```

#### Multi Device (DP=2, N300)

- Use the following command to run demo for 224x224 resolution:

  ```sh
  pytest models/experimental/efficientnetb0/demo/demo.py::test_demo_dp
  ```

## Performant Data Evaluation with Trace+2CQ

#### Single Device (BS=1):

- Use the following command to run the performant data evaluation with Trace+2CQs:

  ```
  pytest models/demos/classification_eval/classification_eval.py::test_efficientnetb0_image_classification_eval
  ```
#### Multi Device (DP=2, N300):

- Use the following command to run the performant data evaluation with Trace+2CQs:

  ```
  pytest models/demos/classification_eval/classification_eval.py::test_efficientnetb0_image_classification_eval_dp
  ```

## Details
- The entry point to efficientnetb0 is in models/experimental/efficientnetb0/tt/ttnn_efficientnetb0.py.
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution - `(224, 224)` - (Height, Width).
- Dataset used for evaluation - **imagenet-1k**
