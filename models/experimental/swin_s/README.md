# Swin-S

## Platforms:
Wormhole (n150, n300)

## Introduction:
The Swin-S (Small) model is a hierarchical Vision Transformer that serves as a general-purpose backbone for various computer vision tasks. It introduces a shifted window mechanism to efficiently compute self-attention within local regions while enabling cross-window connections, maintaining linear computational complexity with respect to image size. With its scalable and flexible architecture, Swin-S achieves strong performance on tasks like image classification, object detection, and semantic segmentation, offering a balanced trade-off between accuracy and efficiency.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`
- login to huggingface with: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
   - To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens

## How to Run:
Use the following command to run the swin_s model:
```
pytest models/experimental/swin_s/tests/pcc/test_ttnn_swin_transformer.py
```

## Performant Model with Trace+2CQ

### Single Device (BS=1):
-  For overall rutime inference (end-2-end), use the following command to run for single device:
```sh
pytest models/experimental/swin_s/tests/perf/test_e2e_performant.py::test_e2e_performant
```
- end-2-end perf is 6 FPS

### Multi Device (DP=2, N300):

-  For overall rutime inference (end-2-end), use the following command to run for multi device:
```sh
pytest  models/experimental/swin_s/tests/perf/test_e2e_performant.py::test_e2e_performant_dp
```
- end-2-end perf is 13 FPS

## Performant demo with Trace+2CQ

- When running the ImageNet demo for the first time, you need to authenticate with Hugging Face by either running `huggingface-cli login` or setting the token directly using `export HF_TOKEN=<your_token>`.
- To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens.

- Use the following command to run the demo for Imagenet-1K:

### Single Device (BS=1):
- Use the following command to run demo for single device:
```sh
pytest models/experimental/swin_s/demo/demo.py::test_swin_s_demo
```

### Multi Device (DP=2, N300):
- Use the following command to run demo for multi device:
```sh
pytest models/experimental/swin_s/demo/demo.py::test_swin_s_demo_dp
```

## Performant Data Evaluation with Trace+2CQ

### Single Device (BS=1):

- Use the following command to run the performant data evaluation with Trace+2CQs:

  ```
  pytest models/demos/classification_eval/classification_eval.py::test_swin_s_image_classification_eval
  ```
### Multi Device (DP=2, N300):

- Use the following command to run the performant data evaluation with Trace+2CQs:

  ```
  pytest models/demos/classification_eval/classification_eval.py::test_swin_s_image_classification_eval_dp
  ```

## Details
The entry point to Swin_S model is swin_s in `models/experimental/swin_s/tt/tt_swin_transformer.py`. The
- model picks up weights from "IMAGENET1K_V1" as mentioned [here](https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py) under Swin_S_Weights
- Batch Size: 1
- Resolution: 512x512
- Dataset used for evaluation - **imagenet-1k**

### Owner: [HariniMohan0102](https://github.com/HariniMohan0102)
