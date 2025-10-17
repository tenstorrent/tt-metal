# UNIAD

## Prerequisite

### Python Package dependencies
    - MMCV : Use `pip install mmcv` command to install
    - Casadi: Use `pip install casadi` command to install
    - mmcv-full: Use `pip install mmcv-full` command to install

Note: UniAD model tests run successfully only on Python 3.10.12.

### Pretrained weights
- The pre-trained weights will be downloaded automatically during sub-module testing.
- If the weights are not downloaded automatically, you can manually fetch them using the command, `wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth`. Place it in the following path `models/experimental/uniad/`.

## Branch
- Checkout to the branch, https://github.com/tenstorrent/tt-metal/tree/punith/ttnn_uniad. Use the below command to checkout to the branch,


        git checkout punith/ttnn_uniad


## Run the following commands to test the individual submodules,

**DetectionTransformerDecoder** -
```
pytest models/experimental/uniad/tests/pcc/test_ttnn_decoder.py
```
**BEVFormerEncoder** -
```
pytest models/experimental/uniad/tests/pcc/test_ttnn_encoder.py
```
**MotionHead** -
```
pytest models/experimental/uniad/tests/pcc/test_ttnn_motion_head.py
```

**MemoryBank** -
```
pytest models/experimental/uniad/tests/pcc/test_ttnn_memory_bank.py
```

**BEVFormerTrackHead** -
```
pytest models/experimental/uniad/tests/pcc/test_ttnn_head.py
```

**OccHead** -
```
pytest models/experimental/uniad/tests/pcc/test_ttnn_occ_head.py
```

**ResNet** -
```
pytest models/experimental/uniad/tests/pcc/test_ttnn_resnet.py::test_uniad_resnet
```

**PlanningHeadSingleMode** -
```
pytest models/experimental/uniad/tests/pcc/test_ttnn_planning_head.py
```

**QueryInteractionModule** -
```
pytest models/experimental/uniad/tests/pcc/test_ttnn_query_interaction.py
```

**PansegformerHead** -
```
pytest models/experimental/uniad/tests/pcc/test_ttnn_pan_segformer_head.py.py
```

## Run the following command to test full model(UniAD) integration

```
pytest models/experimental/uniad/tests/pcc/test_ttnn_uniad.py
```

Note:
- Raised issue for fallback torch ops and added the issue links to resp
