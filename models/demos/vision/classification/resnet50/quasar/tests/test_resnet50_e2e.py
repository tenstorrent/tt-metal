# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
FULL resnet50 end-to-end on the 2-core Quasar emulator (#48552).

This is the successor to test_resnet50_layers_1_3.py: it runs the WHOLE network (stem -> maxpool -> layers 1-4
-> avgpool -> fc) and checks the final 1000-class logits against the torch golden, now that LAYER 4 has a working
2-core path.

Layer4 route (the new part): every layer4 conv runs HEIGHT_SHARDED through the SPLIT path (Program A gather+tilize
-> Program B plain K-spill matmul), NOT the fused conv_bmm -- so it dodges BOTH the fused block-sharded 0x0119
deadlock and the full-N HEIGHT_SHARDED weights overflow. conv2 (3x3, K=144) K-spills the split matmul; the s2
downsample takes the split via force_1x1_nonmm_split. Validated op-by-op by test_conv2d_layer4_l1_fit.py.

Still host-fallbacked (separate, unfixed bugs -- NOT layer4):
  * stem conv1 (folded 4x4): DRAM-slice slice_write per-core TILE-pad mismatch.
  * downsample (all): the layer3_module1 512->1024 s2 @28x28 is BLOCK_SHARDED -> fused conv_bmm 0x19 (the s2
    downsamples also N-halve in HS). The LAYER4 downsample has a working device path (split), but _CONV_ON_DEVICE
    is a single global flag, so all downsamples are host-computed here for now.

REQUIRES (until the K-spill 0x10000 has its full real fix): run with the DPRINT mask on --
  unset TT_METAL_LLK_ASSERTS; TT_METAL_DPRINT_CORES=all
so the K-spill matmul's mm_partials wait->pop hazard stays masked (the copy_tile interpose is only a partial fix).

Run (emulator):
  unset TT_METAL_LLK_ASSERTS
  TT_METAL_DPRINT_CORES=all TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}' \
  pytest -q models/demos/vision/classification/resnet50/quasar/tests/test_resnet50_e2e.py
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.vision.classification.resnet50.quasar.tests.common.resnet50_test_infra import create_test_infra
from models.demos.vision.classification.resnet50.quasar.tt import ttnn_functional_resnet50 as _rn
from models.demos.vision.classification.resnet50.quasar.tt.ttnn_functional_resnet50 import _pcc

PCC = 0.98


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("use_pretrained_weight", [True, False], ids=["pretrained", "random"])
def test_resnet50_e2e(device, use_pretrained_weight, model_location_generator):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")

    saved = {k: os.environ.get(k) for k in ("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "RESNET_PCC_LOG")}
    # No TT_METAL_QSR_RESNET_STOP_AFTER_LAYER3 -> the model runs the FULL network including layer4/avgpool/fc.
    os.environ["TT_METAL_QSR_CONV_SPLIT_PROGRAM"] = "1"  # HS convs -> split path (off the fused conv_bmm 0x19)
    os.environ["RESNET_PCC_LOG"] = "1"  # per-op [GOLDENPCC] so the FIRST diverging op is visible on failure
    # Host-fallback the stem + all downsamples (see module docstring): they have separate unfixed 2-core bugs.
    _saved_cod = dict(_rn._CONV_ON_DEVICE)
    _rn._CONV_ON_DEVICE["stem"] = False
    _rn._CONV_ON_DEVICE["downsample"] = False
    try:
        test_infra = create_test_infra(
            device,
            batch_size=1,
            act_dtype=ttnn.bfloat16,
            weight_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            use_pretrained_weight=use_pretrained_weight,
            model_location_generator=model_location_generator,
        )
        tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
        test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
        dev_out = test_infra.run()  # full network -> [1, 1, 1, 1000]-ish logits on device

        dev = ttnn.to_torch(dev_out).float().reshape(1, -1)[:, :1000]
        golden = test_infra.torch_output_tensor.float().reshape(1, -1)[:, :1000]
        pcc, finite_frac = _pcc(dev, golden)
        # top-1 agreement is the true classification check (PCC can be soft on 1000-way logits under bf16/LoFi).
        dev_top1 = int(torch.argmax(dev, dim=1).item())
        golden_top1 = int(torch.argmax(golden, dim=1).item())
        logger.info(
            f"[e2e] final logits pcc={pcc:.6f} finite_frac={finite_frac:.6f} "
            f"dev_top1={dev_top1} golden_top1={golden_top1}"
        )
        assert finite_frac > 0.999, f"final logits have non-finite values (finite_frac={finite_frac})"
        assert pcc >= PCC, f"final logits PCC {pcc:.6f} < {PCC} (see per-op [GOLDENPCC] logs for the first divergence)"
        if use_pretrained_weight:
            assert dev_top1 == golden_top1, f"top-1 mismatch: device={dev_top1} golden={golden_top1}"
    finally:
        _rn._CONV_ON_DEVICE.update(_saved_cod)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
