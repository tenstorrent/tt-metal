# Efficientnetb0

## Platforms:
Wormhole N150, N300

## Introduction
EfficientNet-B0 is a lightweight and efficient convolutional neural network architecture developed by Google AI. Model is know for its efficiency in image classification tasks. It's a member of the EfficientNet family, which utilizes a compound scaling method to balance model size, accuracy, and computational cost. EfficientNetB0 was trained on the massive ImageNet dataset and can classify images into 1000 object categories.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to run
Use the following command to run the EfficientNetb0 model:
```python
pytest --disable-warnings models/experimental/efficientnetb0/tests/pcc/test_ttnn_efficientnetb0.py
```

## Model performant running with Trace+2CQ
Use the following command to run the e2e perf:

-  For overall rutime inference (end-2-end), use the following command to run:
```sh
pytest --disable-warnings models/experimental/efficientnetb0/tests/perf/test_e2e_performant.py
```
- end-2-end perf is 74 FPS.

## Details
The entry point to efficientnetb0 is in `models/experimental/efficientnetb0/tt/ttnn_efficientnetb0.py`.
- Batch Size: 1
- Resolution: 224x224
