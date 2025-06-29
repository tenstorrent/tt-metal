# Mobilenetv2 Model

## Platforms:
    WH N150/N300
**Note:** On N300 ,Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

## Introduction
The MobileNetV2 model is a convolutional neural network (CNN) architecture designed for efficient mobile and embedded vision applications. It was introduced in the paper ["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381). </br>
The MobileNetV2 model has been pre-trained on the ImageNet dataset and can be used for various tasks such as image classification, object detection, and semantic segmentation. It has achieved state-of-the-art performance on several benchmarks 1 for mobile and embedded vision applications.

## Details
- The entry point to mobilenetv2 model is MobileNetV2 in `models/demos/mobilenetv2/tt/ttnn_monilenetv2.py`.
- Supported Input Resolution - (224,224) (Height,Width)
- Batch Size :8

### Model performant running with Trace+2CQ

#### For 224x224:

- end-2-end perf is 2808 FPS

```bash
pytest models/demos/mobilenetv2/tests/test_e2e_performant.py
```

### Demo on ImageNet:

You will need a huggingFace account to download ImageNet dataset as part of this test. You may create a token from:
```bash
https://huggingface.co/settings/tokens
```
or

```bash
https://huggingface.co/docs/hub/security-tokens
```

If running the test from terminal you may log into HuggingFace by running:
```bash
huggingface-cli login
```

Use the following command to run the Demo on ImageNet dataset:
```bash
pytest  models/demos/mobilenetv2/demo/demo.py::test_mobilenetv2_imagenet_demo
```
