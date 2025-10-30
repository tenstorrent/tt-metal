# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from models.experimental.SSD512.reference.ssd import build_ssd
from models.common.utility_functions import comp_pcc

import torch.nn as nn


@pytest.mark.parametrize("pcc", ((0.99),))
def test_ssd_multibox_head(device, pcc, reset_seeds):
    """
    Test SSD512 multibox head
    """

    # Build PyTorch SSD512 model
    torch_ssd = build_ssd("test", size=512, num_classes=21)
    torch_ssd.eval()

    # SSD512 configuration
    num_classes = 21
    mbox = [4, 6, 6, 6, 4, 4, 4]
    source_channels = [512, 1024, 512, 256, 256, 256, 256]
    feature_sizes = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]

    # Create mock source features (7 sources at different scales)
    batch_size = 1
    sources = []
    for channels, (h, w) in zip(source_channels, feature_sizes):
        sources.append(torch.randn(batch_size, channels, h, w))

    # Run PyTorch multibox head
    with torch.no_grad():
        torch_loc_preds = []
        torch_conf_preds = []

        # Apply loc and conf layers from reference SSD
        for source, loc_layer, conf_layer in zip(sources, torch_ssd.loc, torch_ssd.conf):
            # Location predictions
            loc_pred = loc_layer(source)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_pred = loc_pred.view(batch_size, -1, 4)
            torch_loc_preds.append(loc_pred)

            # Confidence predictions
            conf_pred = conf_layer(source)
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
            conf_pred = conf_pred.view(batch_size, -1, num_classes)
            torch_conf_preds.append(conf_pred)

        torch_loc = torch.cat(torch_loc_preds, dim=1)
        torch_conf = torch.cat(torch_conf_preds, dim=1)

        # Export ONNX for torch_loc and torch_conf using the current computation graph
        class LocConfExporter(nn.Module):
            def forward(self, *sources):
                locs = []
                confs = []
                for source, loc_pred, conf_pred in zip(sources, torch_loc_preds, torch_conf_preds):
                    locs.append(loc_pred)
                    confs.append(conf_pred)
                loc = torch.cat(locs, dim=1)
                conf = torch.cat(confs, dim=1)
                return loc, conf

        exporter = LocConfExporter()
        exporter.eval()
        dummy_inputs = tuple(sources)
        torch.onnx.export(
            exporter,
            dummy_inputs,
            "ssd512_multibox_head_via_concat.onnx",
            input_names=[f"source_{i}" for i in range(len(sources))],
            output_names=["loc", "conf"],
            opset_version=12,
            dynamic_axes={f"source_{i}": {0: "batch"} for i in range(len(sources))},
        )
        logger.info("Exported ssd512_multibox_head_via_concat.onnx")

    logger.info(f"PyTorch loc shape: {torch_loc.shape}")
    logger.info(f"PyTorch conf shape: {torch_conf.shape}")

    # Create TTNN multibox head
    from models.experimental.SSD512.tt.tt_ssd_multihead import TtSSDMultiboxHeadHybrid

    tt_multibox_head = TtSSDMultiboxHeadHybrid(
        num_classes=21,
        mbox=[4, 6, 6, 6, 4, 4, 4],
        source_channels=[512, 1024, 512, 256, 256, 256, 256],
        state_dict=torch_ssd.state_dict(),
        base_address="",
        device=device,
    )
    print(tt_multibox_head)
    # Convert sources to TTNN
    tt_sources = []
    for i, s in enumerate(sources):
        tt_tensor = ttnn.from_torch(
            s, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_sources.append(tt_tensor)

    # Run TTNN multibox head
    tt_loc, tt_conf = tt_multibox_head(tt_sources)

    logger.info(f"TTNN loc shape: {tt_loc.shape}")
    logger.info(f"TTNN conf shape: {tt_conf.shape}")

    # Compare outputs
    does_pass_loc, pcc_message_loc = comp_pcc(torch_loc, tt_loc, pcc)
    does_pass_conf, pcc_message_conf = comp_pcc(torch_conf, tt_conf, pcc)

    logger.info(f"Location PCC: {pcc_message_loc}")
    logger.info(f"Confidence PCC: {pcc_message_conf}")

    if does_pass_loc and does_pass_conf:
        logger.info("SSD Multibox Head Passed!")

    assert does_pass_loc, f"Location predictions do not meet PCC requirement {pcc}: {pcc_message_loc}"
    assert does_pass_conf, f"Confidence predictions do not meet PCC requirement {pcc}: {pcc_message_conf}"
