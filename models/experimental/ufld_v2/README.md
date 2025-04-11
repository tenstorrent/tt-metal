# Ultra-Fast-Lane-Detection-v2

### Platforms:
    WH N300

### Introduction

The Ultra-Fast-Lane-Detection-v2 is a PyTorch-based implementation designed for fast and efficient deep lane detection using hybrid anchor-driven ordinal classification. It enhances the speed and accuracy of lane detection models with a focus on real-time applications.

Resource link - [source](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

### Model Details

- The entry point to the UFLD_v2 is located at:`models/experimental/ufld_v2/ttnn/ttnn_UFLD_v2.py`
- The model picks up trained weights from the **tusimple_res34.pth** file located at:`models/experimental/ufld_v2/reference/tusimple_res34.pth`
- Batch Size :1

Export the following command before running pytests on N300:

`WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml`

Use the following command to run the model :

`pytest tests/ttnn/integration_tests/ufld_v2/test_ttnn_UFLD_v2.py::test_UFD_V2_Model`

Use the following command to run the e2e perf:

`pytest models/experimental/ufld_v2/tests/test_UFLD_v2_perf.py::test_ufld_v2_perf`

Use the following command to run the e2e perf with trace:

`pytest models/experimental/ufld_v2/tests/test_UFLD_v2_e2e_performant.py`

#### Owner: [vguduruTT](https://github.com/vguduruTT)
