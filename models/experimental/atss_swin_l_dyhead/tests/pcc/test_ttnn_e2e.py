# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end PCC test: full TTNN ATSS pipeline vs PyTorch reference.

Validates the complete data path:
  Swin-L (TTNN) → FPN (TTNN) → DyHead (PyTorch) → ATSS Head (TTNN)

Compares intermediate and final outputs against the standalone PyTorch reference.

Run with:
  cd $TT_METAL_HOME
  source python_env/bin/activate
  export ARCH_NAME=wormhole_b0
  export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
  export TT_METAL_HOME=$(pwd)
  export PYTHONPATH=$(pwd):$HOME/.local/lib/python3.10/site-packages
  pytest models/experimental/atss_swin_l_dyhead/tests/pcc/test_ttnn_e2e.py -v -s
"""

import pytest
import torch
import ttnn

from loguru import logger
from models.common.utility_functions import comp_pcc
import tracy


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_atss_e2e_pcc(device, atss_ckpt_path, atss_ref_model):
    """
    Full E2E comparison: TTNN model vs PyTorch reference.
    Both models use the same checkpoint and the same preprocessed input.
    Compares outputs at every stage: backbone, FPN, head.
    """
    from models.experimental.atss_swin_l_dyhead.tt.tt_atss_model import TtATSSModel

    # --- Shared input (640x640 → padded to 640x640, already multiple of 128) ---
    torch.manual_seed(42)
    INPUT_H, INPUT_W = 640, 640
    sample_input = torch.randint(0, 256, (1, 3, INPUT_H, INPUT_W), dtype=torch.float32)
    x_ref = atss_ref_model.preprocess(sample_input)
    padded_h, padded_w = x_ref.shape[2], x_ref.shape[3]

    # --- Build TTNN model with matching input dimensions ---
    logger.info(f"Building TTNN ATSS model (input={padded_h}x{padded_w})...")
    ttnn_model = TtATSSModel.from_checkpoint(atss_ckpt_path, device, input_h=padded_h, input_w=padded_w)

    x_ttnn_input = ttnn_model.preprocess(sample_input)
    assert torch.allclose(x_ref, x_ttnn_input, atol=1e-5), "Preprocessing mismatch"
    logger.info(f"Preprocessed shape: {x_ref.shape}")

    # ========================
    # Stage 1: Backbone
    # ========================
    logger.info("--- Stage 1: Backbone ---")
    with torch.no_grad():
        ref_backbone_feats = atss_ref_model.backbone(x_ref)

    x_on_device = ttnn.from_torch(
        x_ttnn_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tracy.signpost("start")
    ttnn_backbone_feats = ttnn_model.backbone(x_on_device)

    assert len(ref_backbone_feats) == len(ttnn_backbone_feats) == 3
    for i, (rf, tf) in enumerate(zip(ref_backbone_feats, ttnn_backbone_feats)):
        t_out = ttnn.to_torch(ttnn.from_device(tf))
        t_out = torch.permute(t_out, (0, 3, 1, 2))  # NHWC(backbone output) -> NCHW
        passing, pcc = comp_pcc(rf, t_out, 0.96)
        logger.info(f"  Backbone stage {i+1}: shape={list(rf.shape)}, PCC={pcc:.6f}")
        assert passing, f"Backbone stage {i+1} PCC {pcc:.6f} < 0.96"

    # ========================
    # Stage 2: FPN
    # ========================
    logger.info("--- Stage 2: FPN ---")
    with torch.no_grad():
        ref_fpn_feats = atss_ref_model.fpn(tuple(ref_backbone_feats))

    ttnn_fpn_feats = ttnn_model.fpn(ttnn_backbone_feats)

    assert len(ref_fpn_feats) == len(ttnn_fpn_feats) == 5
    for i, (rf, tf) in enumerate(zip(ref_fpn_feats, ttnn_fpn_feats)):
        t_out = ttnn.to_torch(ttnn.from_device(tf))
        t_out = torch.permute(t_out, (0, 3, 1, 2))  # NHWC(FPN output) -> NCHW
        passing, pcc = comp_pcc(rf, t_out, 0.96)
        logger.info(f"  FPN P{i+3}: shape={list(rf.shape)}, PCC={pcc:.6f}")
        assert passing, f"FPN P{i+3} PCC {pcc:.6f} < 0.96"

    # ========================
    # Stage 3: DyHead (PyTorch on both sides)
    # ========================
    logger.info("--- Stage 3: DyHead ---")
    with torch.no_grad():
        ref_dy_feats = atss_ref_model.dyhead(list(ref_fpn_feats))

    ttnn_dy_feats = ttnn_model.forward_dyhead(ttnn_fpn_feats)

    assert len(ref_dy_feats) == len(ttnn_dy_feats) == 5
    for i, (rf, tf) in enumerate(zip(ref_dy_feats, ttnn_dy_feats)):
        tf = torch.permute(tf, (0, 3, 1, 2))  # NHWC(Dyhead output) -> NCHW
        passing, pcc = comp_pcc(rf, tf, 0.96)
        logger.info(f"  DyHead level {i}: shape={list(rf.shape)}, PCC={pcc:.6f}")
        assert passing, f"DyHead level {i} PCC {pcc:.6f} < 0.96"

    # ========================
    # Stage 4: ATSS Head
    # ========================
    logger.info("--- Stage 4: ATSS Head ---")
    with torch.no_grad():
        ref_cls, ref_reg, ref_cent = atss_ref_model.head(tuple(ref_dy_feats))

    ttnn_cls, ttnn_reg, ttnn_cent = ttnn_model.forward_head(ttnn_dy_feats)
    tracy.signpost("stop")
    ttnn_cls_rp = []
    ttnn_reg_rp = []
    ttnn_cent_rp = []

    for i in range(5):
        ttnn_cls_rp.append(ttnn_cls[i])
        passing, pcc = comp_pcc(ref_cls[i], ttnn_cls[i], 0.96)
        logger.info(f"  Head level {i} cls: PCC={pcc:.6f}")
        assert passing, f"Head cls level {i} PCC {pcc:.6f} < 0.96"

        ttnn_reg_rp.append(ttnn_reg[i])
        passing, pcc = comp_pcc(ref_reg[i], ttnn_reg[i], 0.96)
        logger.info(f"  Head level {i} reg: PCC={pcc:.6f}")
        assert passing, f"Head reg level {i} PCC {pcc:.6f} < 0.96"

        ttnn_cent_rp.append(ttnn_cent[i])
        passing, pcc = comp_pcc(ref_cent[i], ttnn_cent[i], 0.96)
        logger.info(f"  Head level {i} cent: PCC={pcc:.6f}")
        assert passing, f"Head cent level {i} PCC {pcc:.6f} < 0.96"

    # ========================
    # Stage 5: Post-processing comparison
    # ========================
    logger.info("--- Stage 5: Post-processing ---")
    from models.experimental.atss_swin_l_dyhead.reference.postprocess import atss_postprocess

    ref_results = atss_postprocess(
        [c for c in ref_cls],
        [r for r in ref_reg],
        [c for c in ref_cent],
        img_shape=(640, 640),
        score_thr=0.05,
    )
    ttnn_results = atss_postprocess(
        ttnn_cls_rp,
        ttnn_reg_rp,
        ttnn_cent_rp,
        img_shape=(640, 640),
        score_thr=0.05,
    )

    ref_n = ref_results["bboxes"].shape[0]
    ttnn_n = ttnn_results["bboxes"].shape[0]
    logger.info(f"  Reference detections: {ref_n}")
    logger.info(f"  TTNN detections:      {ttnn_n}")

    if ref_n > 0 and ttnn_n > 0:
        n_common = min(ref_n, ttnn_n)
        bbox_pcc_pass, bbox_pcc = comp_pcc(ref_results["bboxes"][:n_common], ttnn_results["bboxes"][:n_common], 0.90)
        score_pcc_pass, score_pcc = comp_pcc(ref_results["scores"][:n_common], ttnn_results["scores"][:n_common], 0.90)
        logger.info(f"  BBox PCC (top {n_common}): {bbox_pcc:.6f}")
        logger.info(f"  Score PCC (top {n_common}): {score_pcc:.6f}")

    logger.info("=== E2E PCC TEST PASSED ===")
