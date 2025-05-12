# Mobilenetv2 Model

## Platforms:
    WH N300

**Note:** Make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

## Introduction
The MobileNetV2 model is a convolutional neural network (CNN) architecture designed for efficient mobile and embedded vision applications. It was introduced in the paper ["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381). </br>
The MobileNetV2 model has been pre-trained on the ImageNet dataset and can be used for various tasks such as image classification, object detection, and semantic segmentation. It has achieved state-of-the-art performance on several benchmarks 1 for mobile and embedded vision applications.

## Details
The entry point to mobilenetv2 model is MobileNetV2 in
```
models/demos/mobilenetv2/tt/ttnn_monilenetv2.py
```

- Use the following command to run the mobilenetv2 demo:
```bash
pytest -k "pretrained_weight_true" models/demos/mobilenetv2/demo/demo.py::test_mobilenetv2_imagenet_demo
```

Note: This demo receives inputs from Imagenet dataset.

If you wish to run the model with a desired input, use the following command by providing the image path in the method "test_mobilenetv2_demo": `pytest models/demos/mobilenetv2/demo/demo.py::test_mobilenetv2_demo`


- Use the following command to run the e2e perf with trace(430 FPS):
```bash
pytest models/demos/mobilenetv2/tests/test_e2e_performant.py
```

- Use the following command to run the device perf :
```bash
pytest models/demos/mobilenetv2/tests/test_perf_mobilenetv2.py::test_perf_device_bare_metal_mobilenetv2
```

#### Note:
The post-processing is performed using PyTorch. and also that the first time the Imagenet demo is run, you need to login to huggingface using your token: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
- To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens
