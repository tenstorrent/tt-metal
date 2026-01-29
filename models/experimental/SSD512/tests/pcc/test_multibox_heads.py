# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import pytest
import ttnn
from loguru import logger

from models.experimental.SSD512.reference.ssd import multibox, base, mbox, vgg
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
)
from models.experimental.SSD512.tt.tt_multibox_heads import TtMultiBoxHEAD
from models.experimental.SSD512.reference.ssd import add_extras, extras
from models.experimental.SSD512.common import SSD512_NUM_CLASSES


@pytest.mark.parametrize("pcc", ((0.99),))
@pytest.mark.parametrize("size", (512,))
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_multibox_heads(device, pcc, size, reset_seeds):
    num_classes = SSD512_NUM_CLASSES

    vgg_layers = vgg(base[str(size)], i=3, batch_norm=False)
    extra_layers = add_extras(extras[str(size)], i=1024, batch_norm=False)

    torch_vgg_model = nn.ModuleList(vgg_layers)
    torch_extras_model = nn.ModuleList(extra_layers)

    extra_source_indices_all = [1, 3, 5, 7, 9, 11] if size == 512 else [1, 3, 5, 7]
    extra_source_indices = []
    extra_channels = []
    for idx in extra_source_indices_all:
        if idx < len(torch_extras_model):
            extra_channels.append(torch_extras_model[idx].out_channels)
            extra_source_indices.append(idx)

    _, _, (torch_loc_layers, torch_conf_layers) = multibox(
        torch_vgg_model, torch_extras_model, mbox[str(size)], num_classes
    )

    torch_loc_model = nn.ModuleList(torch_loc_layers)
    torch_conf_model = nn.ModuleList(torch_conf_layers)
    torch_loc_model.eval()
    torch_conf_model.eval()

    num_heads = 7
    batch_size = 1
    sources = []

    sources.append(torch.randn(batch_size, 512, 64, 64))
    sources.append(torch.randn(batch_size, 1024, 32, 32))
    extra_dims = [(512, 16, 16), (256, 8, 8), (256, 4, 4), (256, 2, 2), (256, 1, 1), (256, 1, 1)]

    num_extra_sources = num_heads - 2
    for i in range(num_extra_sources):
        if i < len(extra_dims):
            channels, h, w = extra_dims[i]
            sources.append(torch.randn(batch_size, channels, h, w))
        else:
            sources.append(torch.randn(batch_size, 256, 1, 1))
    loc_config_layers = []
    conf_config_layers = []
    loc_kernel_layers = []
    conf_kernel_layers = []
    for source_idx, source in enumerate(sources):
        if isinstance(torch_loc_model[source_idx], nn.Conv2d):
            loc_config_layers = Conv2dConfiguration.from_torch(
                torch_loc_model[source_idx],
                input_height=source.shape[-2],
                input_width=source.shape[-1],
                batch_size=source.shape[0],
            )

            conf_config_layers = Conv2dConfiguration.from_torch(
                torch_conf_model[source_idx],
                input_height=source.shape[-2],
                input_width=source.shape[-1],
                batch_size=source.shape[0],
            )
            loc_kernel_layers.append(
                TtMultiBoxHEAD(
                    device=device,
                    conv_config_layer=loc_config_layers,
                )
            )
            conf_kernel_layers.append(
                TtMultiBoxHEAD(
                    device=device,
                    conv_config_layer=conf_config_layers,
                )
            )
    torch_loc_preds = []
    torch_conf_preds = []

    with torch.no_grad():
        for source_idx, source in enumerate(sources):
            loc_pred = torch_loc_model[source_idx](source)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            torch_loc_preds.append(loc_pred)

            conf_pred = torch_conf_model[source_idx](source)
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
            torch_conf_preds.append(conf_pred)

    tt_loc_preds = []
    tt_conf_preds = []

    for source_idx, source in enumerate(sources):
        ttnn_input_tensor = ttnn.from_torch(
            source.permute(0, 2, 3, 1), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
        )
        loc_pred = loc_kernel_layers[source_idx](device, ttnn_input_tensor)
        tt_loc_preds.append(loc_pred)

        conf_pred = conf_kernel_layers[source_idx](device, ttnn_input_tensor)
        tt_conf_preds.append(conf_pred)

    for source_idx in range(len(sources)):
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
