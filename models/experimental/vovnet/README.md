# Vovnet

## Platforms:
    Wormhole N150, N300

## Introduction
**VoVNet** is a convolutional neural network designed for efficient and high-performance image recognition tasks. It introduces a novel One-Shot Aggregation (OSA) module that aggregates features from several layers at once, reducing redundancy and improving efficiency. Unlike DenseNet, which uses dense connections, VoVNet performs a single aggregation at the end of each block. This design leads to faster inference and lower memory usage while maintaining strong accuracy.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run:
- Use the following command to run the model:
```
pytest --disable-warnings models/experimental/vovnet/tests/pcc/test_tt_vovnet.py
```

## Model Performant Model with Trace+2CQ
- end-2-end perf is 84 FPS

Use the following command to run the performant Model with Trace+2CQs:

```
pytest --disable-warnings models/experimental/yolov5x/tests/perf/test_e2e_performant.py
```

## Details
- The entry point to the vovnet is located at:`models/experimental/functional_vovnet/tt/vovnet.py`
- Batch Size :1
- Supported Input Resolution - (224,224) (Height,Width)
