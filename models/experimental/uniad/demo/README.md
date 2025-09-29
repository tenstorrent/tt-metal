## Uniad Demo: Setup & Usage Guide

This guide explains how to set up the environment, install dependencies, prepare datasets, and run the Uniad demo on the NuScenes dataset.

---

### Create a New Python Environment

```bash
python3 -m venv uniad_env
source uniad_env/bin/activate

### Build and Install Dependencies
```bash
rm -rf build built build_Release build_Release_tracy
git submodule foreach 'git lfs fetch --all && git lfs pull'
git submodule update --init --recursive
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)  
export PYTHONPATH=$(pwd)
export TT_METAL_ENV=dev
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
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
pip install motmetrics
pip install shapely==1.8.5.post1
```

### Prepare Data for the Demo

- Create Required Folder Structure
```bash
mkdir -p models/experimental/uniad/demo/data
```
- Download NuScenes Mini Dataset → Place it in:
```bash
models/experimental/uniad/demo/data/nuscenes
```
- Download CAN Bus Data → Place it in:
```bash
models/experimental/uniad/demo/data/nuscenes
```

### Register Custom Transforms & Datasets
- Make sure to register all custom transforms and datasets mentioned in the config file ```models/experimental/uniad/demo/config.py``` in virtual environment:

### For Dataset:
```bash
uniad_env/lib64/python3.10/site-packages/mmdet3d/datasets/__init__.py
```
### For Transforms:
```bash
uniad_env/lib64/python3.10/site-packages/mmdet3d/datasets/transforms/__init__.py
```

### Generate PKL files
```bash
python models/experimental/uniad/demo/uniad_nuscenes_converter.py nuscenes \
    --root-path models/experimental/uniad/demo/data/nuscenes \
    --out-dir models/experimental/uniad/demo/data/infoss \
    --extra-tag uniad_nuscenes \
    --version v1.0-mini \
    --canbus models/experimental/uniad/demo/data \
    --canbus models/experimental/uniad/demo/data/nuscenes
```
- Note: Ensure the dataset contains all files required for both train and test splits.

### Test the DEMO:

The test runs the Uniad reference model with Nuscenes Dataset for 81 Iterations.

- To run reference demo
```bash
pytest models/experimental/uniad/demo/demo.py
```

TODO:
- Prepare evaluation script for the reference model - WIP.
- Set up the pipeline to run the TT Uniad model in the demo.
