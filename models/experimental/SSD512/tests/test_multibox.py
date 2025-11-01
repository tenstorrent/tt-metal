"""Tests for TTNN multibox head implementation."""

import pytest
import torch
import torch.nn as nn
from loguru import logger

from models.common.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.SSD512.tt.layers.multibox import TtMultiBoxHead
from models.experimental.SSD512.reference.ssd import multibox as torch_multibox


@pytest.mark.parametrize("pcc", [0.97])
def test_multibox_head(device, pcc):
    """Test TTNN multibox head against PyTorch implementation."""

    num_classes = 21  # 20 classes + background

    # Create base VGG and extra layers for PyTorch multibox
    base_vgg = [
        nn.Conv2d(512, 512, 3, padding=1),  # conv4_3
        nn.Conv2d(1024, 1024, 1),  # conv7
    ]
    extras = [
        nn.Conv2d(512, 256, 1),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.Conv2d(256, 128, 1),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.Conv2d(256, 128, 1),
        nn.Conv2d(128, 256, 4, padding=1),
    ]

    # Create PyTorch multibox head
    _, _, (torch_loc, torch_conf) = torch_multibox(base_vgg, extras, [4, 6, 6, 6, 4, 4], num_classes)
    torch_loc = nn.ModuleList(torch_loc)
    torch_conf = nn.ModuleList(torch_conf)

    # Create TTNN multibox head
    tt_model = TtMultiBoxHead({}, num_classes, device=device)

    # Create test feature maps with example sizes
    feature_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    channels = [512, 1024, 512, 256, 256, 256]
    batch_size = 1

    torch_features = []
    tt_features = []
    for size, c in zip(feature_sizes, channels):
        f = torch.randn(batch_size, c, size[0], size[1])
        torch_features.append(f)
        tt_features.append(torch_to_tt_tensor_rm(f, device))

    # Run forward passes
    torch_loc = []
    torch_conf = []
    for x, l, c in zip(torch_features, torch_loc, torch_conf):
        loc = l(x).permute(0, 2, 3, 1).contiguous()
        conf = c(x).permute(0, 2, 3, 1).contiguous()
        torch_loc.append(loc.view(loc.size(0), -1))
        torch_conf.append(conf.view(conf.size(0), -1))

    torch_loc = torch.cat(torch_loc, 1).view(batch_size, -1, 4)
    torch_conf = torch.cat(torch_conf, 1).view(batch_size, -1, num_classes)

    tt_loc, tt_conf = tt_model(tt_features)
    tt_loc = tt_to_torch_tensor(tt_loc)
    tt_conf = tt_to_torch_tensor(tt_conf)

    # Compare localization outputs
    loc_pass, loc_pcc = comp_pcc(torch_loc, tt_loc, pcc)
    logger.info(f"Multibox loc PCC: {loc_pcc}")
    loc_close = comp_allclose(torch_loc, tt_loc)
    logger.info(f"Multibox loc allclose: {loc_close}")

    # Compare classification outputs
    conf_pass, conf_pcc = comp_pcc(torch_conf, tt_conf, pcc)
    logger.info(f"Multibox conf PCC: {conf_pcc}")
    conf_close = comp_allclose(torch_conf, tt_conf)
    logger.info(f"Multibox conf allclose: {conf_close}")

    assert loc_pass, f"Multibox loc output does not meet PCC requirement {pcc}"
    assert conf_pass, f"Multibox conf output does not meet PCC requirement {pcc}"
