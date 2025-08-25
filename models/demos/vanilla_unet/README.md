# Unet Vanilla

## Platforms:
Wormhole (n150, n300)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
   - To obtain the perf reports through profiler, please build with: `./build_metal.sh -p`


## How to run (480x640 resolution)

Use the following command to run the inference pipeline:

    ```
    pytest models/demos/vanilla_unet/tests/pcc/test_ttnn_unet.py::test_unet
    ```

### Model performant running with Trace+2CQs
#### Single Device (BS=1):

- For `480x640`, end-2-end perf is `72` FPS

    ```sh
    pytest models/demos/vanilla_unet/tests/perf/test_e2e_performant.py::test_e2e_performant
    ```

#### Multi Device (DP=2, N300):

- For `480x640`, end-2-end perf is `263` FPS

    ```sh
    pytest models/demos/vanilla_unet/tests/perf/test_e2e_performant.py::test_e2e_performant_dp
    ```

### Performant Demo with Trace+2CQ

#### Single image
- Use the following command to run the demo for `480x640` resolution:

    ```
    pytest models/demos/vanilla_unet/demo/demo.py::test_unet_demo_single_image
    ```

- Output images will be saved in the `models/demos/vanilla_unet/demo/pred` folder


### Evaluation test:

#### Single Device (BS=1):

- Use the following command to run the performant evaluation with Trace+2CQs:

    ```sh
    pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet
    ```

#### Multi Device (DP=2, N300):

- Use the following command to run the performant evaluation with Trace+2CQs:

    ```sh
    pytest models/demos/segmentation_evaluation/test_segmentation_eval.py::test_vanilla_unet_dp
    ```

**Note:** If vanilla unet evaluation test fails with the error: `ValueError: Sample larger than population or is negative`
Try deleting the `imageset` folder in `models/demos/segmentation_evaluation` directory and try running again.


## Details
- Entry point for the model is `models/demos/vanilla_unet/ttnn/ttnn_unet.py`
- Batch Size: 1 (Single Device).
- Support Input Resolution: (480, 640) - (Height, Width).
- Dataset - [mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
