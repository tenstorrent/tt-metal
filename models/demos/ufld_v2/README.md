# Ultra-Fast-Lane-Detection-v2

## Platforms:
Wormhole (n150, n300)

## Introduction
The Ultra-Fast-Lane-Detection-v2 is a PyTorch-based implementation designed for fast and efficient deep lane detection using hybrid anchor-driven ordinal classification. It enhances the speed and accuracy of lane detection models with a focus on real-time applications.

Resource link - [source](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
  - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`

## How to Run:

Use the following command to run the model:
  ```
  pytest --disable-warnings models/demos/ufld_v2/tests/pcc/test_ttnn_ufld_v2.py::test_ufld_v2_model
  ```

### Performant Model with Trace+2CQ

#### Single Device (BS=1):
- end-2-end perf is `365` FPS (**On N150**), _On N300 single device, the FPS will be low as it uses ethernet dispatch_

  ```
  pytest --disable-warnings models/demos/ufld_v2/tests/perf/test_ufld_v2_e2e_performant.py::test_ufldv2_e2e_performant
  ```
#### Multi Device (DP=2, N300):
- end-2-end perf is `572` FPS

  ```
  pytest --disable-warnings models/demos/ufld_v2/tests/perf/test_ufld_v2_e2e_performant.py::test_ufldv2_e2e_performant_dp
  ```

### Performant Demo with Trace+2CQ

#### Single Device (BS=1):
- Use the following command to run the performant Demo with Trace+2CQs:

  ```
  pytest --disable-warnings models/demos/ufld_v2/demo/demo.py::test_ufld_v2_demo
  ```

#### Multi Device (DP=2, N300):
- Use the following command to run the DP performant Demo with Trace+2CQs:

  ```
  pytest --disable-warnings models/demos/ufld_v2/demo/demo.py::test_ufld_v2_demo_dp
  ```

### To run the demo on your data:
- Add your images to the `images` directory under `demo` folder.
- Annotate the corresponding ground truth labels in `ground_truth_labels.json` using the required format.
- The Demo outputs are saved inside this directories: `models/demos/ufld_v2/demo/reference_model_results` and `models/demos/ufld_v2/demo/ttnn_model_results`

## Testing
### Performant Data Evaluation with Trace+2CQ
- dataset source: [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple)
- Adjust the `num_of_images` parameter to control the number of dataset samples used during evaluation. (default number of images taken - 100)

#### Single Device (BS=1):

- Use the following command to run the performant data evaluation with Trace+2CQs:

  ```
  pytest --disable-warnings models/demos/ufld_v2/demo/dataset_evaluation.py::test_ufld_v2_dataset_inference
  ```

#### Multi Device (DP=2, N300):

- Use the following command to run the performant data evaluation with Trace+2CQs:

  ```
  pytest --disable-warnings models/demos/ufld_v2/demo/dataset_evaluation.py::test_ufld_v2_dataset_inference_dp
  ```

## Details
- The entry point of the model is located at ```models/demos/ufld_v2/ttnn/ttnn_ufld_v2.py```
- Batch Size : `1` (Single Device), `2` (Multi Device).
- Supported Input Resolution : `(320, 800)` - (Height, Width).
- Dataset used for evaluation : [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple)
