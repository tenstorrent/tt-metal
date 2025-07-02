# Ultra-Fast-Lane-Detection-v2

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

The Ultra-Fast-Lane-Detection-v2 is a PyTorch-based implementation designed for fast and efficient deep lane detection using hybrid anchor-driven ordinal classification. It enhances the speed and accuracy of lane detection models with a focus on real-time applications.

Resource link - [source](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

### Details

- The entry point to the UFLD_v2 is located at:`models/demos/ufld_v2/ttnn/ttnn_ufld_v2.py`
- The model picks up trained weights from the **tusimple_res34.pth** file located at:`models/demos/ufld_v2/reference/tusimple_res34.pth`
- Batch Size :1
- Supported Input Resolution - (320,800) (Height,Width)

### How to Run:

Use the following command to run the model :

```
pytest --disable-warnings tests/ttnn/integration_tests/ufld_v2/test_ttnn_ufld_v2.py::test_ufld_v2_model
```

### Performant Model with Trace+2CQ
- end-2-end perf is 1414 FPS

Use the following command to run the performant Model with Trace+2CQs:

```
pytest --disable-warnings models/demos/ufld_v2/tests/test_ufld_v2_e2e_performant.py
```

### Performant Demo with Trace+2CQ

Use the following command to run the performant Demo with Trace+2CQs:

```
pytest --disable-warnings models/demos/ufld_v2/demo/demo.py
```

To run the demo on your data:

- Add your images to the 'images' directory and list their filenames in 'input_images.txt' under demo folder
- Annotate the corresponding ground truth labels in 'ground_truth_labels.json' using the required format.
