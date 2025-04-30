# Efficientnetb0 Model

## Platforms:
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

## Introduction
EfficientNet-B0 is a lightweight and efficient convolutional neural network architecture developed by Google AI. Model is know for its efficiency in image classification tasks. It's a member of the EfficientNet family, which utilizes a compound scaling method to balance model size, accuracy, and computational cost. EfficientNetB0 was trained on the massive ImageNet dataset and can classify images into 1000 object categories.

## Details
The entry point to efficientnetb0 is in `models/experimental/efficientnetb0/tt/ttnn_efficientnetb0.py`.
- Batch Size: 1
- Resolution: 224x224

## How to run
Use the following command to run the EfficientNetb0 model :
```python
pytest --disable-warnings tests/ttnn/integration_tests/efficientnetb0/test_ttnn_efficientnetb0.py
```
