# Vovnet

## Platforms:
Wormhole (n150, n300)

## Introduction
**VoVNet** is a convolutional neural network designed for efficient and high-performance image recognition tasks. It introduces a novel One-Shot Aggregation (OSA) module that aggregates features from several layers at once, reducing redundancy and improving efficiency. Unlike DenseNet, which uses dense connections, VoVNet performs a single aggregation at the end of each block. This design leads to faster inference and lower memory usage while maintaining strong accuracy.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`
- Login to huggingface with: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
   - To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens

## How to Run:
- Use the following command to run the model:

  ```
  pytest models/experimental/vovnet/tests/pcc/test_tt_vovnet.py
  ```

### Model Performant Model with Trace+2CQ

#### Single Device (BS=1):

- For `224x224`, end-2-end perf is `127` FPS

  ```
  pytest models/experimental/vovnet/tests/perf/test_e2e_performant.py::test_vovnet_e2e_performant
  ```

#### Multi Device (DP=2, N300):

- For `224x224`, end-2-end perf is `245` FPS

  ```
  pytest models/experimental/vovnet/tests/perf/test_e2e_performant.py::test_vovnet_e2e_performant_dp
  ```

### Performant Demo on ImageNet:

- Make sure your HuggingFace token is set ([See Prerequisites](#prerequisites) for instructions)
- Use the following command to run the Demo using `ImageNet` dataset:

  ```bash
  pytest models/experimental/vovnet/demo/demo.py
  ```

## Testing

### Performant Data Evaluation with Trace+2CQ

#### Single Device (BS=1):

- Use the following command to run the performant data evaluation with Trace+2CQs:

  ```
  pytest models/demos/classification_eval/classification_eval.py::test_vovnet_image_classification_eval
  ```

#### Multi Device (DP=2, N300):

- Use the following command to run the performant data evaluation with Trace+2CQs:

  ```
  pytest models/demos/classification_eval/classification_eval.py::test_vovnet_image_classification_eval_dp
  ```
## Details

- The post-processing is performed using PyTorch.
- The entry point to the `vovnet` is located at: `models/experimental/vovnet/tt/vovnet.py`
- Batch Size : `1`(Single Device), `2`(Multi Device)
- Supported Input Resolution - `(224, 224) `- (Height, Width)
- Dataset used for evaluation - **imagenet-1k**
