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
The entry point to mobilenetv2 model is MobileNetV2 in `models/experimental/mobilenetv2/tt/ttnn_monilenetv2.py`.

Use the following command to run the model :
`pytest tests/ttnn/integration_tests/mobilenetv2/test_mobilenetv2.py`

Use the following command to run the e2e perf(11 FPS):
`pytest models/experimental/mobilenetv2/tests/test_perf_mobilenetv2.py::test_mobilenetv2`

Use the following command to run the e2e perf with trace(430 FPS):
`pytest models/experimental/mobilenetv2/tests/test_e2e_performant.py`

Note : The model supports a batch size of 8 for a resolution of 224. If you prefer to use a different batch size, it is recommended to modify the batch_size accordingly in the test file.

### Owner: [Sabira](https://github.com/sabira-mcw)
