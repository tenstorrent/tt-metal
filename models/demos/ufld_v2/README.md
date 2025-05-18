# Ultra-Fast-Lane-Detection-v2

### Platforms:
    WH N300,N150

### Introduction

The Ultra-Fast-Lane-Detection-v2 is a PyTorch-based implementation designed for fast and efficient deep lane detection using hybrid anchor-driven ordinal classification. It enhances the speed and accuracy of lane detection models with a focus on real-time applications.

Resource link - [source](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

### Model Details

- The entry point to the UFLD_v2 is located at:`models/demos/ufld_v2/ttnn/ttnn_ufld_v2.py`
- The model picks up trained weights from the **tusimple_res34.pth** file located at:`models/demos/ufld_v2/reference/tusimple_res34.pth`
- Batch Size :1
- Supported Input Resolution - (320,800) (Height,Width)

Export the following command before running pytests on N300:

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml`

Use the following command to run the model :

`pytest tests/ttnn/integration_tests/ufld_v2/test_ttnn_ufld_v2.py::test_ufld_v2_model`

Use the following command to run the e2e perf(5.3 FPS):

`pytest models/demos/ufld_v2/tests/test_ufld_v2_perf.py::test_ufld_v2_perf`

Use the following command to run the e2e perf with trace(107 FPS):

`pytest models/demos/ufld_v2/tests/test_ufld_v2_e2e_performant.py`

Use the following command to generate device perf (306 FPS):

`pytest models/demos/ufld_v2/tests/test_ufld_v2_perf.py::test_perf_device_bare_metal_ufld_v2`

### Demo

Use the following command to run the demo :

`pytest models/demos/ufld_v2/demo/demo.py`

To run the demo on your data:

- Add your images to the 'images' directory and list their filenames in 'input_images.txt' under demo folder
- Annotate the corresponding ground truth labels in 'ground_truth_labels.json' using the required format.
