# Mobilenetv2 Model


## Platforms:
    WH N150/N300


To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Introduction
The MobileNetV2 model is a convolutional neural network (CNN) architecture designed for efficient mobile and embedded vision applications. It was introduced in the paper ["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381). </br>
The MobileNetV2 model has been pre-trained on the ImageNet dataset and can be used for various tasks such as image classification, object detection, and semantic segmentation. It has achieved state-of-the-art performance on several benchmarks 1 for mobile and embedded vision applications.

### Details
- The entry point to mobilenetv2 model is MobileNetV2 in `models/demos/mobilenetv2/tt/ttnn_mobilenetv2.py`.
- Supported Input Resolution - (224,224) (Height,Width)
- Batch Size :10
- Dataset used for evaluation - **imagenet-1k**

### How to Run:

Use the following command to run the model :

```
pytest --disable-warnings models/demos/mobilenetv2/tests/pcc/test_mobilenetv2.py::test_mobilenetv2
```

### Performant Model with Trace+2CQ

#### Single Device (BS=10):

- end-2-end perf is `2470` FPS

Use the following command to run the performant Model with Trace+2CQs:

```
pytest --disable-warnings models/demos/mobilenetv2/tests/perf/test_e2e_performant.py::test_mobilenetv2_e2e
```
#### Multi Device (DP=2, N300):

- end-2-end perf is `4933` FPS

Use the following command to run the performant Model with Trace+2CQs:

```
pytest --disable-warnings models/demos/mobilenetv2/tests/perf/test_e2e_performant.py::test_mobilenetv2_e2e_dp
```

### Performant Demo with Trace+2CQ

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

#### Single Device (BS=10):

Use the following command to run the performant Demo with Trace+2CQs:

```
pytest --disable-warnings models/demos/mobilenetv2/demo/demo.py::test_mobilenetv2_imagenet_demo
```

#### Multi Device (DP=2, N300):

Use the following command to run the performant Demo with Trace+2CQs:

```
pytest --disable-warnings models/demos/mobilenetv2/demo/demo.py::test_mobilenetv2_imagenet_demo_dp
```

### Performant Data evaluation with Trace+2CQ

#### Single Device (BS=10):

Use the following command to run the performant evaluation with Trace+2CQs:

```
pytest --disable-warnings models/experimental/classification_eval/classification_eval.py::test_mobilenetv2_image_classification_eval
```

#### Multi Device (DP=2, N300):

Use the following command to run the performant evaluation with Trace+2CQs:

```
pytest --disable-warnings models/experimental/classification_eval/classification_eval.py::test_mobilenetv2_image_classification_eval_dp
```

Note: The model is evaluated with 512 samples.

#### Note:
- The post-processing is performed using PyTorch.
- The first time the Imagenet demo is run, you need to login to huggingface using your token: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`.
- To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens.
