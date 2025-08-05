# Vovnet - Model

### Platforms:

Wormhole N150, N300

To obtain the perf reports through profiler, please build with following command:
```
./build_metal.sh -p
```

### Introduction

**VoVNet** is a convolutional neural network designed for efficient and high-performance image recognition tasks. It introduces a novel One-Shot Aggregation (OSA) module that aggregates features from several layers at once, reducing redundancy and improving efficiency. Unlike DenseNet, which uses dense connections, VoVNet performs a single aggregation at the end of each block. This design leads to faster inference and lower memory usage while maintaining strong accuracy.


### Details

- The entry point to the vovnet is located at:`models/experimental/functional_vovnet/tt/vovnet.py`
- Batch Size :1
- Supported Input Resolution - (224,224) (Height,Width)

### How to Run:

Use the following command to run the model :

```
pytest --disable-warnings models/experimental/vovnet/tests/pcc/test_tt_vovnet.py
```
