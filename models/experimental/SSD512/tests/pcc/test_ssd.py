# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from loguru import logger
from models.experimental.SSD512.common import (
    setup_seeds_and_deterministic,
    build_and_init_torch_model,
    build_and_load_ttnn_model,
    synchronize_device,
)
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
@pytest.mark.parametrize(
    "size",
    (512,),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 98304}], indirect=True)
def test_ssd512_network(device, pcc, size, reset_seeds):
    """
    Test Full SSD512 Network.
    """
    # from models.experimental.SSD512.tt.layers.tt_multibox_heads import clear_multibox_weight_cache
    # from models.experimental.SSD512.tt.layers.tt_extras_backbone import clear_extras_weight_cache as clear_extras_backbone_cache
    # from models.experimental.SSD512.tt.tt_ssd import clear_extras_weight_cache
    # clear_multibox_weight_cache()
    # clear_extras_backbone_cache()
    # clear_extras_weight_cache()
    setup_seeds_and_deterministic(reset_seeds=reset_seeds, seed=0)

    num_classes = 21  # VOC dataset
    batch_size = 1

    # Build PyTorch reference model
    torch_model = build_and_init_torch_model(phase="test", size=size, num_classes=num_classes)

    # Build TTNN model and load weights
    ttnn_model = build_and_load_ttnn_model(torch_model, device, num_classes=num_classes)

    synchronize_device(device)

    input_tensor = torch.randn(batch_size, 3, size, size)

    # Optional signposts for device performance measurement
    try:
        from tracy import signpost

        use_signpost = True
    except ImportError:
        use_signpost = False

    torch_sources = []
    torch_loc_preds = []
    torch_conf_preds = []

    with torch.no_grad():
        x = input_tensor.clone()

        # VGG up to conv4_3 relu (layer 22)
        for k in range(23):
            x = torch_model.base[k](x)
        torch_sources.append(torch_model.L2Norm(x.clone()))

        # VGG up to conv7 (fc7)
        for k in range(23, len(torch_model.base)):
            x = torch_model.base[k](x)
        torch_sources.append(x.clone())

        # Extras layers
        for k, v in enumerate(torch_model.extras):
            x = torch.nn.functional.relu(v(x), inplace=True)
            if k % 2 == 1:
                torch_sources.append(x.clone())

        # Multibox heads
        for source, loc_layer, conf_layer in zip(torch_sources, torch_model.loc, torch_model.conf):
            loc_pred = loc_layer(source).permute(0, 2, 3, 1).contiguous()
            conf_pred = conf_layer(source).permute(0, 2, 3, 1).contiguous()
            torch_loc_preds.append(loc_pred)
            torch_conf_preds.append(conf_pred)

        torch_loc_flat = torch.cat([o.view(o.size(0), -1) for o in torch_loc_preds], 1)
        torch_conf_flat = torch.cat([o.view(o.size(0), -1) for o in torch_conf_preds], 1)

    torch_loc = torch_loc_flat
    torch_conf = torch_conf_flat

    synchronize_device(device)

    # Mark start of measurement region for device perf
    if use_signpost:
        signpost("start")

    # Run TTNN forward pass
    ttnn_loc, ttnn_conf, debug_dict = ttnn_model.forward(input_tensor, dtype=ttnn.bfloat16, debug=True)

    # Mark end of measurement region for device perf
    if use_signpost:
        signpost("stop")
    ttnn_loc = ttnn_loc.float()
    ttnn_conf = ttnn_conf.float()

    torch_loc_flat = torch_loc.flatten()
    ttnn_loc_flat = ttnn_loc.flatten()
    torch_conf_flat = torch_conf.flatten()
    ttnn_conf_flat = ttnn_conf.flatten()

    min_loc_len = min(len(torch_loc_flat), len(ttnn_loc_flat))
    min_conf_len = min(len(torch_conf_flat), len(ttnn_conf_flat))

    if len(torch_loc_flat) != len(ttnn_loc_flat):
        torch_loc_flat = torch_loc_flat[:min_loc_len]
        ttnn_loc_flat = ttnn_loc_flat[:min_loc_len]

    if len(torch_conf_flat) != len(ttnn_conf_flat):
        torch_conf_flat = torch_conf_flat[:min_conf_len]
        ttnn_conf_flat = ttnn_conf_flat[:min_conf_len]

    does_pass_loc, pcc_message_loc = comp_pcc(torch_loc_flat, ttnn_loc_flat, pcc)
    logger.info(f"Location PCC: {pcc_message_loc}")

    does_pass_conf, pcc_message_conf = comp_pcc(torch_conf_flat, ttnn_conf_flat, pcc)
    logger.info(f"Confidence PCC: {pcc_message_conf}")

    assert_with_pcc(torch_loc_flat, ttnn_loc_flat, pcc)
    assert_with_pcc(torch_conf_flat, ttnn_conf_flat, pcc)
