# UNet3D

## Platforms
- Wormhole (single device and 2-device mesh)

## Introduction
UNet3D is a volumetric segmentation model for 3D data. This demo runs a TTNN implementation
and includes a Torch reference, a downloadable checkpoint converted to safetensors, and an HDF5
prediction pipeline for confocal boundary data.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium / TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Installing demo dependencies:
  ```bash
  pip install safetensors h5py tqdm scikit-image
  ```

## Assets
- Download the checkpoint and convert to safetensors:
  ```bash
  python models/demos/unet_3d/demo/download_models.py
  ```

- Download the validation dataset used by the default config:
  ```bash
  python models/demos/unet_3d/demo/download_datasets.py
  ```

  Available dataset names: `cell_boundary`, `confocal_boundary`
  Available dataset types: `validation`, `test`

## How to Run
### PCC Tests
- Full model PCC:
  ```bash
  pytest models/demos/unet_3d/tests/pcc/test_model.py
  ```

- Component PCCs:
```bash
  pytest models/demos/unet_3d/tests/pcc/test_conv3d.py
  pytest models/demos/unet_3d/tests/pcc/test_conv_block.py
  pytest models/demos/unet_3d/tests/pcc/test_encoder.py
  pytest models/demos/unet_3d/tests/pcc/test_decoder.py
  pytest models/demos/unet_3d/tests/pcc/test_group_norm3d.py
  pytest models/demos/unet_3d/tests/pcc/test_max_pool3d.py
  pytest models/demos/unet_3d/tests/pcc/test_upsample3d.py
  ```

### Performant Model with Trace+2CQ
#### Single Device
```bash
pytest models/demos/unet_3d/tests/perf/test_e2e_performant.py::test_unet3d_e2e
```

#### 2-Device Mesh
```bash
pytest models/demos/unet_3d/tests/perf/test_e2e_performant.py::test_unet3d_e2e_dp
```

## Demo: HDF5 Validation
- Run the validation pipeline (uses `configs/test_confocal_boundary.json`):
  ```bash
  python models/demos/unet_3d/demo/validate.py
  ```

  The default config expects:
  - Model weights at `models/demos/unet_3d/data/models/confocal_boundary.safetensors`
  - Validation dataset at `models/demos/unet_3d/data/datasets/confocal_boundary/validation/N_420_ds2x.h5`
  - Output H5 files in `models/demos/unet_3d/data/predictions/confocal_boundary/validation`

  Update `models/demos/unet_3d/configs/test_confocal_boundary.json` to point at custom HDF5 files or to
  change patch/stride settings.

  Validation logs IoU and Dice coefficient metrics.

## Demo: Tracy Perf Capture
- Run the Tracy-enabled demo to extract a CSV perf sheet:
  ```bash
  python -m tracy -r -n unet3d -m pytest models/demos/unet_3d/demo/demo.py
  ```

## Details
- TTNN model: `models/demos/unet_3d/ttnn_impl/model.py`
- Torch model: `models/demos/unet_3d/torch_impl/model.py`
- Runner entry point: `models/demos/unet_3d/runner/performant_runner.py`
- Default config: `models/demos/unet_3d/configs/test_confocal_boundary.json`
- Input size: We are choosing [32, 32, 64] (for the single batch run) since it is the most we can fit in the L1 memory of the N300s device during model running.

## Workarounds
- 3D op handling: This model uses natively 3D ops (for example, `max_pool3d` and `upsample`). As a temporary
  workaround, the implementation applies the 2D versions twice: first along the H/W dimensions while treating T
  as batch, then combining H/W into a single dimension and treating T as another dimension. Alternatives for the
  future reader:
  - Use the torch version as a fallback
  - Open an issue for native 3D implementations of these ops

## References
- [UNet3D Paper](https://arxiv.org/abs/1606.06650)
- Reference implementation of torch model adapted from [here](https://github.com/wolny/pytorch-3dunet)
