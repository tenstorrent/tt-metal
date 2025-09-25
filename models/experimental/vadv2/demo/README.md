## VADV2 Demo: Setup & Usage Guide

This guide explains how to set up the environment, install dependencies, prepare datasets, and run the VADV2 demo on the NuScenes dataset.

---

### Create a New Python Environment

```bash
python3 -m venv vadv2_env
source vadv2_env/bin/activate
```
### Build and Install Dependencies

```bash
rm -rf build built build_Release build_Release_tracy
git submodule foreach 'git lfs fetch --all && git lfs pull'
git submodule update --init --recursive
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)  
export PYTHONPATH=$(pwd)
export TT_METAL_ENV=dev
./build_metal.sh -p
```
- Note: Modify create_venv.sh file comment 17-25 lines and then run
```bash
./create_venv.sh
source python_env/bin/activate
```
```bash
pip install mmdet3d==1.4.0
pip install mmdet==3.3.0
pip install mmcv==2.1.0 --no-cache-dir
```

### Prepare Data for the Demo

- Create Required Folder Structure
```bash
mkdir -p models/experimental/vadv2/demo/data
```
- Download NuScenes Mini Dataset → Place it in:
```bash
models/experimental/vadv2/demo/data/
```
- Download CAN Bus Data → Place it in:
```bash
models/experimental/vadv2/demo/data/
```

### Register Custom Transforms & Datasets
- Make sure to register all custom transforms and datasets mentioned in the config file ```models/experimental/vadv2/demo/config.py``` in virtual environment:
```bash
mmdet3d/datasets/__init__.py
```

### Generate PKL files
```bash
python models/experimental/vadv2/demo/vad_nuscenes_converter.py nuscenes \
  --root-path models/experimental/vadv2/demo/data/nuscenes \
  --out-dir models/experimental/vadv2/demo/data/nuscenes \
  --extra-tag vad_nuscenes \
  --version v1.0 \
  --canbus models/experimental/vadv2/demo/data
```
- Note: Ensure the dataset contains all files required for both train and test splits.

### Test the DEMO:
- To run reference demo
```bash
pytest models/experimental/vadv2/demo/demo.py::test_torch_demo
```
- To run tt demo - WIP
```bash
pytest models/experimental/vadv2/demo/demo.py::test_tt_demo
```
