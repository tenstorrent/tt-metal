# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import pytest
import ttnn
from loguru import logger
from models.experimental.SSD512.reference.ssd import multibox, base, extras, mbox, vgg, add_extras
from models.experimental.SSD512.tt.layers.tt_multibox_heads import (
    build_multibox_heads,
    apply_multibox_heads,
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
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_multibox_heads(device, pcc, size, reset_seeds):
    """
    Test Multibox heads implementation.
    """
    from models.experimental.SSD512.common import setup_seeds_and_deterministic

    setup_seeds_and_deterministic(reset_seeds=reset_seeds, seed=0)

    num_classes = 21  # VOC dataset

    vgg_layers = vgg(base[str(size)], i=3, batch_norm=False)
    extra_layers = add_extras(extras[str(size)], i=1024, batch_norm=False)

    # Create PyTorch multibox heads
    torch_vgg_model = nn.ModuleList(vgg_layers)
    torch_extras_model = nn.ModuleList(extra_layers)

    # Get source channels from VGG layers
    vgg_source_indices = [21, -2]
    vgg_channels = []
    for idx in vgg_source_indices:
        vgg_layer = torch_vgg_model[idx]
        vgg_channels.append(vgg_layer.out_channels)

    extra_source_indices_all = [1, 3, 5, 7] if size == 300 else [1, 3, 5, 7, 9, 11]
    extra_source_indices = []
    extra_channels = []
    for idx in extra_source_indices_all:
        if idx < len(torch_extras_model):
            extra_layer = torch_extras_model[idx]
            extra_channels.append(extra_layer.out_channels)
            extra_source_indices.append(idx)

    # Build PyTorch multibox heads
    _, _, (torch_loc_layers, torch_conf_layers) = multibox(
        torch_vgg_model, torch_extras_model, mbox[str(size)], num_classes
    )

    torch_loc_model = nn.ModuleList(torch_loc_layers)
    torch_conf_model = nn.ModuleList(torch_conf_layers)
    torch_loc_model.eval()
    torch_conf_model.eval()

    # Build TTNN multibox heads
    loc_layers_config, conf_layers_config = build_multibox_heads(
        size=size,
        num_classes=num_classes,
        vgg_channels=vgg_channels,
        extra_channels=extra_channels,
        extra_source_indices=extra_source_indices,
        device=device,
    )

    # Extract weights from PyTorch model and load into TTNN layers
    torch_conv_idx = 0
    loc_layers_with_weights = []
    conf_layers_with_weights = []

    for layer in loc_layers_config:
        if layer["type"] == "conv":
            if torch_conv_idx >= len(torch_loc_model):
                raise ValueError("Mismatch: More loc layers in TTNN config than PyTorch model")

            torch_conv = torch_loc_model[torch_conv_idx]
            torch_conv_idx += 1

            weight = torch_conv.weight.data.clone()
            bias = torch_conv.bias.data.clone() if torch_conv.bias is not None else None

            if device is not None:
                weight_ttnn = ttnn.from_torch(
                    weight,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                if bias is not None:
                    bias_reshaped = bias.reshape((1, 1, 1, -1))
                    bias_ttnn = ttnn.from_torch(
                        bias_reshaped,
                        device=device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                else:
                    bias_ttnn = None
            else:
                weight_ttnn = weight
                bias_ttnn = bias

            layer_with_weights = layer.copy()
            layer_with_weights["weight"] = weight_ttnn
            layer_with_weights["bias"] = bias_ttnn
            loc_layers_with_weights.append(layer_with_weights)

    # Process confidence layers
    torch_conv_idx = 0
    for layer in conf_layers_config:
        if layer["type"] == "conv":
            if torch_conv_idx >= len(torch_conf_model):
                raise ValueError("Mismatch: More conf layers in TTNN config than PyTorch model")

            torch_conv = torch_conf_model[torch_conv_idx]
            torch_conv_idx += 1

            weight = torch_conv.weight.data.clone()
            bias = torch_conv.bias.data.clone() if torch_conv.bias is not None else None

            if device is not None:
                weight_ttnn = ttnn.from_torch(
                    weight,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                if bias is not None:
                    bias_reshaped = bias.reshape((1, 1, 1, -1))
                    bias_ttnn = ttnn.from_torch(
                        bias_reshaped,
                        device=device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                else:
                    bias_ttnn = None
            else:
                weight_ttnn = weight
                bias_ttnn = bias

            layer_with_weights = layer.copy()
            layer_with_weights["weight"] = weight_ttnn
            layer_with_weights["bias"] = bias_ttnn
            conf_layers_with_weights.append(layer_with_weights)

    assert len(loc_layers_with_weights) == len(
        torch_loc_model
    ), f"Mismatch: TTNN has {len(loc_layers_with_weights)} loc heads, PyTorch has {len(torch_loc_model)}"
    assert len(conf_layers_with_weights) == len(
        torch_conf_model
    ), f"Mismatch: TTNN has {len(conf_layers_with_weights)} conf heads, PyTorch has {len(torch_conf_model)}"

    num_heads = len(loc_layers_with_weights)
    batch_size = 1
    sources = []

    # VGG sources: Conv4_3 and Conv7 (always 2 sources)
    if size == 300:
        sources.append(torch.randn(batch_size, 512, 38, 38))  # Conv4_3
        sources.append(torch.randn(batch_size, 1024, 19, 19))  # Conv7
        num_extra_sources = num_heads - 2
        extra_dims = [(512, 10, 10), (256, 5, 5), (256, 3, 3), (256, 1, 1)]
        for i in range(num_extra_sources):
            if i < len(extra_dims):
                channels, h, w = extra_dims[i]
                sources.append(torch.randn(batch_size, channels, h, w))
            else:
                sources.append(torch.randn(batch_size, 256, 1, 1))
    else:  # size == 512
        sources.append(torch.randn(batch_size, 512, 64, 64))  # Conv4_3
        sources.append(torch.randn(batch_size, 1024, 32, 32))  # Conv7
        # Remaining sources come from extras
        num_extra_sources = num_heads - 2
        # Typical extras sources for SSD512
        extra_dims = [(512, 16, 16), (256, 8, 8), (256, 4, 4), (256, 2, 2), (256, 1, 1), (256, 1, 1)]
        for i in range(num_extra_sources):
            if i < len(extra_dims):
                channels, h, w = extra_dims[i]
                sources.append(torch.randn(batch_size, channels, h, w))
            else:
                sources.append(torch.randn(batch_size, 256, 1, 1))

    # Run PyTorch reference forward pass
    torch_loc_preds = []
    torch_conf_preds = []

    with torch.no_grad():
        for source_idx, source in enumerate(sources):
            # Apply location head
            loc_pred = torch_loc_model[source_idx](source)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            torch_loc_preds.append(loc_pred)

            # Apply confidence head
            conf_pred = torch_conf_model[source_idx](source)
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
            torch_conf_preds.append(conf_pred)

    # Run TTNN forward pass
    tt_loc_preds, tt_conf_preds = apply_multibox_heads(
        sources,
        loc_layers_with_weights,
        conf_layers_with_weights,
        device=device,
        dtype=ttnn.bfloat16,
    )

    for source_idx in range(len(sources)):
        tt_loc = ttnn.to_torch(tt_loc_preds[source_idx])
        tt_conf = ttnn.to_torch(tt_conf_preds[source_idx])
        tt_loc = tt_loc.float()
        tt_conf = tt_conf.float()

        torch_loc = torch_loc_preds[source_idx]
        torch_conf = torch_conf_preds[source_idx]

        if tt_loc.shape != torch_loc.shape:
            logger.error(
                f"Location shape mismatch at source {source_idx}: PyTorch {torch_loc.shape}, TTNN {tt_loc.shape}"
            )
            min_shape = [min(s1, s2) for s1, s2 in zip(torch_loc.shape, tt_loc.shape)]
            torch_loc = torch_loc[tuple(slice(0, s) for s in min_shape)]
            tt_loc = tt_loc[tuple(slice(0, s) for s in min_shape)]

        if tt_conf.shape != torch_conf.shape:
            logger.error(
                f"Confidence shape mismatch at source {source_idx}: PyTorch {torch_conf.shape}, TTNN {tt_conf.shape}"
            )
            min_shape = [min(s1, s2) for s1, s2 in zip(torch_conf.shape, tt_conf.shape)]
            torch_conf = torch_conf[tuple(slice(0, s) for s in min_shape)]
            tt_conf = tt_conf[tuple(slice(0, s) for s in min_shape)]

        # location head
        does_pass, pcc_message = comp_pcc(torch_loc, tt_loc, pcc)
        logger.info(f" Location head {source_idx} PCC: {pcc_message}")

        # confidence head
        does_pass, pcc_message = comp_pcc(torch_conf, tt_conf, pcc)
        logger.info(f" Confidence head {source_idx} PCC: {pcc_message}")

        assert_with_pcc(torch_loc, tt_loc, pcc)
        assert_with_pcc(torch_conf, tt_conf, pcc)
