# Swin-S Model

## Platforms:
    WH N150, N300

## Introduction:
The Swin-S (Small) model is a hierarchical Vision Transformer that serves as a general-purpose backbone for various computer vision tasks. It introduces a shifted window mechanism to efficiently compute self-attention within local regions while enabling cross-window connections, maintaining linear computational complexity with respect to image size. With its scalable and flexible architecture, Swin-S achieves strong performance on tasks like image classification, object detection, and semantic segmentation, offering a balanced trade-off between accuracy and efficiency.

## Details
The entry point to Swin_S model is swin_s in `models/experimental/swin_s/tt/tt_swin_transformer.py`. The
model picks up weights from "IMAGENET1K_V1" as mentioned [here](https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py) under Swin_S_Weights

## How to Run:
If running on Wormhole N300 (not required for N150 or Blackhole), the following environment variable needs to be set as the model requires at least 8x8 core grid size:
```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

To obtain the perf reports through profiler, please build with the following command:
```
./build_metal.sh -p
```

### To run the Swin_S model of 512x512 resolution:
```sh
pytest --disable-warnings tests/ttnn/integration_tests/swin_s/test_ttnn_swin_transformer.py
```

## Model performant running with Trace+2CQ
Use the following command to run the e2e perf:

-  For overall rutime inference (end-2-end), use the following command to run the demo:
```sh
pytest --disable-warnings models/experimental/swin_s/tests/perf/test_e2e_performant.py
```
- end-2-end perf is 5 FPS

## Model demo with trace

- When running the ImageNet demo for the first time, you need to authenticate with Hugging Face by either running `huggingface-cli login` or setting the token directly using `export HF_TOKEN=<your_token>`.
- To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens.

- Use the following command to run the demo for Imagenet-1K:
```sh
pytest --disable-warnings models/demos/swin_s/demo/demo.py::test_run_swin_s_trace_2cqs_inference
```

### Owner: [HariniMohan0102](https://github.com/HariniMohan0102)
