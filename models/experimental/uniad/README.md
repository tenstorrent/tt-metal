# UNIAD

## Prerequisite

UniAD relies on the standard tt-metal Python environment (`./create_venv.sh`).
The host-side DCNv2 reference is implemented via `torchvision.ops.deform_conv2d`,
so no additional MMCV / Casadi install is required.

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

**ModulatedDeformConv (device)** -
```
pytest models/experimental/uniad/tests/unit/test_dcn_device.py
```

## Run the following command to test full model(UniAD) integration

```
pytest models/experimental/uniad/tests/pcc/test_ttnn_uniad.py
```

## Performance

Warm-iteration wall time on Blackhole p150b (`test_ttnn_uniad.py::test_uniad`,
`TT_UNIAD_TIMING=1 TT_UNIAD_WARM_ITERS=2`, measured on `call#3`):

| Phase                              |  ms |
| ---------------------------------- | --: |
| `extract_img_feat` (ResNet101+FPN) | 372 |
| `get_bev_features` (BEV encoder)   | 422 |
| `get_detections` (DETR decoder)    |  89 |
| `simple_test_track` (subtotal)     | 959 |
| `seg_head.forward_test`            | 262 |
| `motion_head.forward_test`         | 122 |
| `planning_head.forward_test`       |  39 |
| `occ_head`                         |   1 |
| **forward total**                  | **~1383** |

Per-run noise is ±2-3 % on each phase. PCC `sdc_traj` ≥ 0.99 (gate
floor). The committed `tests/perf_baseline.json` pins each phase to
±5 % / 20 ms abs floor and asserts inter-warm variance under 3 %
whenever `TT_UNIAD_TIMING=1 TT_UNIAD_WARM_ITERS≥1`; the e2e test
fails on a regression.

### Modulated deformable convolution

The ResNet101 backbone runs 26 modulated deformable convs in
`layer3`/`layer4`. By default this routes through a device-side
`TtModulatedDeformConv2dDevice` (built on `ttnn.grid_sample`), which
removes ~3.5 sec of per-forward host CPU compute vs. the legacy
mmcv fallback. Set `TT_DCN_DEVICE=0` to switch back to the host
path (`torchvision.ops.deform_conv2d`) for numerical bisection.

### Environment variables

| Variable                  | Default      | Purpose |
| ------------------------- | ------------ | ------- |
| `TT_DCN_DEVICE`           | `1` (device) | Route modulated deformable conv through the device path. Set `0` for host CPU fallback. |
| `TT_DCN_TIMING`           | unset (off)  | Print per-forward DCN-path breakdown (transfer / cpu / back / device). Diagnostic only. |
| `TT_UNIAD_TIMING`         | unset (off)  | Print a per-sub-phase wall-clock breakdown of `TtUniAD.forward_test`. Enables the per-phase perf gate in `test_ttnn_uniad.py`. No-op when unset. |
| `TT_UNIAD_WARM_ITERS`     | `0`          | Test-side: extra warm-up forward passes (with `reset_test_state()` between them) before the PCC-asserted call. Use `≥1` so timing lands on a warm cache. |
| `TT_UNIAD_PERF_GATE`      | `1` (on)     | Silence the per-phase regression gate while keeping timing prints. Only consulted when `TT_UNIAD_TIMING=1` AND `TT_UNIAD_WARM_ITERS≥1`. |
| `TT_UNIAD_DISABLE_TRACE`  | unset (off)  | Bypass the BEV/DETR trace-replay capture. Required when running Tracy's device profiler, which is incompatible with `begin_trace_capture`. |

Note:
- Raised issue for fallback torch ops and added the issue links to resp
