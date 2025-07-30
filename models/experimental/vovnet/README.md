# Vovnet - Model

### Platforms:

Wormhole N150, N300

**Note:** On N300, make sure to use `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` with the pytest.

Or, make sure to set the following environment variable in the terminal:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Introduction

**VoVNet** is a convolutional neural network designed for efficient and high-performance image recognition tasks. It introduces a novel One-Shot Aggregation (OSA) module that aggregates features from several layers at once, reducing redundancy and improving efficiency. Unlike DenseNet, which uses dense connections, VoVNet performs a single aggregation at the end of each block. This design leads to faster inference and lower memory usage while maintaining strong accuracy.


### Details

- The entry point to the vovnet is located at:`models/experimental/vovnet/tt/vovnet.py`
- Batch Size :1
- Supported Input Resolution - (224,224) (Height,Width)

### How to Run:

Use the following command to run the model :

```
pytest --disable-warnings models/experimental/vovnet/tests/pcc/test_tt_vovnet.py
```

### Performant Model with Trace+2CQ

#### Single Device (BS=1):

- end-2-end perf is `110` FPS

Use the following command to run the performant Model with Trace+2CQs:

```
pytest --disable-warnings models/experimental/vovnet/tests/perf/test_e2e_performant.py::test_vovnet_e2e_performant
```
#### Multi Device (DP=2, N300):

- end-2-end perf is `205` FPS

Use the following command to run the performant Model with Trace+2CQs:

```
pytest --disable-warnings models/experimental/vovnet/tests/perf/test_e2e_performant.py::test_vovnet_e2e_performant_dp
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

Use the following command to run the Demo on ImageNet dataset:
```bash
pytest  models/experimental/vovnet/demo/demo.py::test_vovnet_imagenet_demo
```

#### Note:
- The post-processing is performed using PyTorch.
- The first time the Imagenet demo is run, you need to login to huggingface using your token: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`.
- To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens.

#### Single Device (BS=1):

Use the following command to run the performant Demo with Trace+2CQs:

```
pytest --disable-warnings models/experimental/vovnet/demo/demo.py::test_vovnet_imagenet_demo
```

#### Multi Device (DP=2, N300):

Use the following command to run the performant Demo with Trace+2CQs:

```
pytest --disable-warnings models/experimental/vovnet/demo/demo.py::test_vovnet_imagenet_demo_dp
```

### Performant Data evaluation with Trace+2CQ

#### Single Device (BS=1):

Use the following command to run the performant evaluation with Trace+2CQs:

```
pytest --disable-warnings models/experimental/classification_eval/classification_eval.py::test_vovnet_image_classification_eval
```

#### Multi Device (DP=2, N300):

Use the following command to run the performant evaluation with Trace+2CQs:

```
pytest --disable-warnings models/experimental/classification_eval/classification_eval.py::test_vovnet_image_classification_eval_dp
```

Note: The model is evaluated with 512 samples.
