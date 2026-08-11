# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Layers-1-3 end-to-end on the 2-core Quasar emulator (#48552).

The full resnet50 e2e cannot complete on the 2-core emulator: layer4 conv2 + the layer3_module1 downsample have
no working path there (fused conv_bmm bank-reuse 0x0119, and the DRAM-weight K-spill matmul TILE_COUNTERS fault --
both deep LLK/sim bugs, tracked via test_conv2d_block_sharded.py and test_matmul_dram_weights_kspill.py). But the
whole front of the network -- stem (host-folded conv1) + host maxpool + layers 1-3 (HEIGHT_SHARDED on the split
path) -- IS validated op-by-op. This test runs the real model end-to-end THROUGH layer3 and checks the layer3
output against the torch golden, so the working portion is guarded as one on-device run.

Mechanism: TT_METAL_QSR_RESNET_STOP_AFTER_LAYER3=1 makes the model return the layer3 output and skip
layer4/avgpool/fc; TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 routes the HS conv2 off the fused conv_bmm 0x19 to the split
(tilize + matmul::linear) path. RESNET_PCC_LOG=1 additionally prints per-op [GOLDENPCC] lines (stem, maxpool,
every layer1-3 module .add) so the FIRST diverging op is visible if the PCC assert fails.

Run (craq-sim / emulator):
  TT_METAL_FORCE_JIT_COMPILE=1 \
  TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}' \
  pytest -q models/demos/vision/classification/resnet50/quasar/tests/test_resnet50_layers_1_3.py
"""

import os

import pytest
from loguru import logger

import ttnn
from models.demos.vision.classification.resnet50.quasar.tests.common.resnet50_test_infra import create_test_infra
from models.demos.vision.classification.resnet50.quasar.tt import ttnn_functional_resnet50 as _rn
from models.demos.vision.classification.resnet50.quasar.tt.ttnn_functional_resnet50 import _pcc

PCC = 0.98


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("use_pretrained_weight", [True, False], ids=["pretrained", "random"])
def test_resnet50_layers_1_3(device, use_pretrained_weight, model_location_generator):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")

    saved = {
        k: os.environ.get(k)
        for k in ("TT_METAL_QSR_RESNET_STOP_AFTER_LAYER3", "TT_METAL_QSR_CONV_SPLIT_PROGRAM", "RESNET_PCC_LOG")
    }
    os.environ["TT_METAL_QSR_RESNET_STOP_AFTER_LAYER3"] = "1"  # return the layer3 output; skip layer4/avgpool/fc
    os.environ["TT_METAL_QSR_CONV_SPLIT_PROGRAM"] = "1"  # HS conv2 -> split path (off the fused conv_bmm 0x19)
    os.environ["RESNET_PCC_LOG"] = "1"  # print per-op [GOLDENPCC] so the first diverging op is visible on failure
    # [#48552] Host-fallback the STEM and the DOWNSAMPLE (existing bypass, same mechanism as maxpool). Neither
    # has a working 2-core device path here:
    #   * stem conv1 (folded 4x4, 112x112 output) routes to the DRAM slice path and trips slice_write's per-core
    #     TILE-pad mismatch (196->224 rows/core: logical 392 tiles vs sharded 448) -- conv2d_DRAM per-core-pad bug.
    #   * projection 1x1 s2 downsamples: layer3_module1's 512->1024 s2 is block-sharded -> fused conv_bmm 0x19;
    #     the s2 downsamples also N-halve in HS.
    # Computing both on host lets the layers-1-3 e2e complete while the conv1/conv2/conv3 of every layer1-3 module
    # still run ON DEVICE (the split-path HS convs we validated). Restore after.
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
        dev_l3 = test_infra.run()  # with the gate set, run() returns the layer3 output [1,1,NHW,1024]

        # Golden = the torch layer3 output the infra already captured via its RESNET_PCC_LOG forward hooks
        # (block .add hook on layer3's last module). This is the exact reference the model's own [GOLDENPCC]
        # line for op "layer3_module6.add" used, so we assert on the same comparison.
        golden = _rn._GOLDEN.get("layer3_module6.add")
        assert golden is not None, (
            "layer3 golden not captured -- RESNET_PCC_LOG must be '1' BEFORE create_test_infra so the infra "
            "registers the torch forward hooks (set_golden_intermediates)."
        )
        dev = ttnn.to_torch(dev_l3).float()
        pcc, finite_frac = _pcc(dev, golden)
        logger.info(f"[layers_1_3] layer3 output pcc={pcc:.6f} finite_frac={finite_frac:.6f} dev={tuple(dev.shape)}")
        assert finite_frac > 0.999, f"layer3 output has non-finite values (finite_frac={finite_frac})"
        assert (
            pcc >= PCC
        ), f"layer3 output PCC {pcc:.6f} < {PCC} (see the per-op [GOLDENPCC] logs for the first divergence)"
    finally:
        _rn._CONV_ON_DEVICE.update(_saved_cod)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
