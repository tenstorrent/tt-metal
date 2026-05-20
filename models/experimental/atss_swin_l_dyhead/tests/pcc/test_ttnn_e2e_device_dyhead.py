# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""E2E PCC test for TtATSSModel with hybrid_dyhead='device' (full on-device path)."""

import pytest
import torch
import ttnn

from loguru import logger
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_atss_e2e_device_pcc(device, atss_ckpt_path, atss_ref_model):
    """E2E: backbone+FPN (TTNN) → DyHead (FULLY on TTNN device) → ATSS Head (TTNN)."""
    from models.experimental.atss_swin_l_dyhead.tt.tt_atss_model import TtATSSModel

    torch.manual_seed(42)
    INPUT_H, INPUT_W = 640, 640
    sample_input = torch.randint(0, 256, (1, 3, INPUT_H, INPUT_W), dtype=torch.float32)
    x_ref = atss_ref_model.preprocess(sample_input)
    padded_h, padded_w = x_ref.shape[2], x_ref.shape[3]

    logger.info(f"Building TTNN model (hybrid_dyhead='device', input={padded_h}x{padded_w})...")
    ttnn_model = TtATSSModel.from_checkpoint(
        atss_ckpt_path, device, input_h=padded_h, input_w=padded_w, hybrid_dyhead="device"
    )

    x_ttnn_input = ttnn_model.preprocess(sample_input)
    x_on_device = ttnn.from_torch(
        x_ttnn_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # forward_device — should be host-roundtrip-free with device DyHead.
    logger.info("Running forward_device (host-free)...")
    cls_scores_ttnn, bbox_preds_ttnn, centernesses_ttnn = ttnn_model.forward_device(x_on_device)

    # Reference outputs
    with torch.no_grad():
        ref_cls, ref_reg, ref_cent = atss_ref_model(x_ref)

    # Compare. NHWC ttnn → NCHW torch.
    cls_pccs, reg_pccs, cent_pccs = [], [], []
    for i in range(5):
        for name, ref_list, tt_list, pccs in [
            ("cls", ref_cls, cls_scores_ttnn, cls_pccs),
            ("reg", ref_reg, bbox_preds_ttnn, reg_pccs),
            ("cent", ref_cent, centernesses_ttnn, cent_pccs),
        ]:
            tt_t = ttnn.to_torch(ttnn.from_device(tt_list[i])).float().permute(0, 3, 1, 2)
            _, pcc = comp_pcc(ref_list[i], tt_t, 0.0)
            pccs.append(pcc)
            logger.info(f"  Level {i} {name}: shape={list(tt_t.shape)}, PCC={pcc:.6f}")

    logger.info(f"cls PCC range: [{min(cls_pccs):.4f}, {max(cls_pccs):.4f}]")
    logger.info(f"reg PCC range: [{min(reg_pccs):.4f}, {max(reg_pccs):.4f}]")
    logger.info(f"cent PCC range: [{min(cent_pccs):.4f}, {max(cent_pccs):.4f}]")
    # Loose threshold — bf16 compounding through 6 DyHead blocks.
    assert min(cls_pccs + reg_pccs + cent_pccs) >= 0.70, "E2E PCC too low"
