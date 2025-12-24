# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from models.experimental.SSD512.common import (
    SSD512_L1_SMALL_SIZE,
    SSD512_NUM_CLASSES,
    load_torch_model,
    create_ssd512_input_tensors,
)
from models.experimental.SSD512.tt.tt_ssd import TtSSD
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


# End-to-end SSD512 model test
@pytest.mark.parametrize("pcc", ((0.97),))
@pytest.mark.parametrize("size", (512,))
@pytest.mark.parametrize("device_params", [{"l1_small_size": SSD512_L1_SMALL_SIZE}], indirect=True)
def test_ssd512(device, pcc, size, reset_seeds):
    torch_model = load_torch_model(phase="test", size=size, num_classes=SSD512_NUM_CLASSES)
    torch_input, ttnn_input = create_ssd512_input_tensors(batch=1, input_height=size, input_width=size)

    with torch.no_grad():
        torch_sources = []
        torch_loc_preds = []
        torch_conf_preds = []
        x = torch_input.clone()

        # Extract features from VGG backbone: first 23 layers, then L2Norm
        for k in range(23):
            x = torch_model.base[k](x)
        torch_sources.append(torch_model.L2Norm(x.clone()))

        for k in range(23, len(torch_model.base)):
            x = torch_model.base[k](x)
        torch_sources.append(x.clone())

        # Extract features from extras network at odd indices
        for k, v in enumerate(torch_model.extras):
            x = torch.nn.functional.relu(v(x), inplace=True)
            if k % 2 == 1:
                torch_sources.append(x.clone())

        # Generate location and confidence predictions for each feature scale
        for source, loc_layer, conf_layer in zip(torch_sources, torch_model.loc, torch_model.conf):
            loc_pred = loc_layer(source).permute(0, 2, 3, 1).contiguous()
            conf_pred = conf_layer(source).permute(0, 2, 3, 1).contiguous()
            torch_loc_preds.append(loc_pred)
            torch_conf_preds.append(conf_pred)

    ttnn_model = TtSSD(torch_model, torch_input, batch_size=1, device=device)

    ttnn.synchronize_device(device)

    tt_loc_preds, tt_conf_preds = ttnn_model(device=device, input=ttnn_input)

    # Compare predictions for each of the 7 feature map scales
    for source_idx in range(7):
        tt_loc = ttnn.to_torch(tt_loc_preds[source_idx]).float()
        tt_conf = ttnn.to_torch(tt_conf_preds[source_idx]).float()

        torch_loc = torch_loc_preds[source_idx]
        torch_conf = torch_conf_preds[source_idx]

        if tt_loc.shape != torch_loc.shape:
            min_shape = [min(s1, s2) for s1, s2 in zip(torch_loc.shape, tt_loc.shape)]
            torch_loc = torch_loc[tuple(slice(0, s) for s in min_shape)]
            tt_loc = tt_loc[tuple(slice(0, s) for s in min_shape)]

        if tt_conf.shape != torch_conf.shape:
            min_shape = [min(s1, s2) for s1, s2 in zip(torch_conf.shape, tt_conf.shape)]
            torch_conf = torch_conf[tuple(slice(0, s) for s in min_shape)]
            tt_conf = tt_conf[tuple(slice(0, s) for s in min_shape)]

        _, pcc_message_loc = comp_pcc(torch_loc, tt_loc, pcc)
        logger.info(f"Location head {source_idx} PCC: {pcc_message_loc}")

        _, pcc_message_conf = comp_pcc(torch_conf, tt_conf, pcc)
        logger.info(f"Confidence head {source_idx} PCC: {pcc_message_conf}")

        assert_with_pcc(torch_loc, tt_loc, pcc)
        assert_with_pcc(torch_conf, tt_conf, pcc)
