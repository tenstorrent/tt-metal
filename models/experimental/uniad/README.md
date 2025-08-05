# UNIAD

## Prerequisite

### Python Package dependencies
    - MMCV : Use `pip install mmcv` command to install
    - Casadi: Use `pip install casadi` command to install

### Pretrained weights
- To run UNIAD model sub_modules test, you should ensure that you have download pre-trained weights.
- Use the following command to download the weights, `wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth`. Place it in the following path `models/experimental/uniad/`.

## Branch
- Checkout to the branch, https://github.com/tenstorrent/tt-metal/tree/punith/ttnn_uniad. Use the below command to checkout to the branch,


        git checkout punith/ttnn_uniad


## Run the following commands to test the individual submodules,

**DetectionTransformerDecoder** -
```
pytest models/experimental/uniad/tests/pcc/test_tt_decoder.py
```
**BEVFormerEncoder** -
```
pytest models/experimental/uniad/tests/pcc/test_tt_encoder.py
```
**MotionHead** -
```
pytest models/experimental/uniad/tests/pcc/test_tt_motion_head.py
```

**MemoryBank** -
```
pytest models/experimental/uniad/tests/pcc/test_memory_bank.py
```

**BEVFormerTrackHead** -
```
pytest models/experimental/uniad/tests/pcc/test_tt_head.py
```

**OccHead** -
```
pytest models/experimental/uniad/tests/pcc/test_occ_head.py
```

**ResNet** -
```
pytest models/experimental/uniad/tests/pcc/test_tt_resnet.py::test_uniad_resnet
```

**PlanningHeadSingleMode** -
```
pytest models/experimental/uniad/tests/pcc/test_tt_planning_head.py
```

Note:
- e2e ttnn pipeline is in progress
- **BEVFormerTrackHead** is tested with random weights
