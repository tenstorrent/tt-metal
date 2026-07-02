# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC test: TTNN detection heads (cls + bbox) vs PyTorch reference (mmdet bbox_head).

Tests the forward_heads method of TtDINO by comparing class logits and
bbox predictions against the mmdet DINOHead.forward output.

Run with:
  export PYTHONPATH=/home/ubuntu/tt-metal:$HOME/.local/lib/python3.10/site-packages
  pytest models/experimental/dino_5scale_swin_l/tests/pcc/test_ttnn_heads.py -v
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.dino_5scale_swin_l.common import (
    DINO_INPUT_H,
    DINO_INPUT_W,
    NUM_QUERIES,
    DECODER_NUM_LAYERS,
)
from loguru import logger


def _mmdet_importable():
    try:
        from mmdet.apis import init_detector  # noqa: F401

        return True
    except ImportError:
        return False


def _get_ckpt_and_config():
    import os
    from pathlib import Path

    base = Path(os.environ.get("TT_METAL_HOME", Path.cwd()))
    config = base / "models/experimental/dino_5scale_swin_l/reference/dino_5scale_swin_l.py"
    ckpt_dir = base / "models/experimental/dino_5scale_swin_l/checkpoints/dino_5scale_swin_l"
    ckpt = ckpt_dir / "dino_5scale_swin_l.pth"
    if not ckpt.is_file():
        ckpt = ckpt_dir / "dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"
    return str(config), str(ckpt)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_heads_pcc(device, reset_seeds):
    """Compare TTNN detection head outputs vs PyTorch reference."""
    if not _mmdet_importable():
        pytest.skip("mmdet not importable")

    config_path, ckpt_path = _get_ckpt_and_config()
    from pathlib import Path

    if not Path(ckpt_path).is_file():
        pytest.skip("Checkpoint not found")

    # --- Run full PyTorch reference through decoder ---
    from models.experimental.dino_5scale_swin_l.reference.dino_staged_forward import DINOStagedForward

    ref = DINOStagedForward(config_path, ckpt_path)
    torch_input = torch.rand(1, 3, DINO_INPUT_H, DINO_INPUT_W)

    with torch.no_grad():
        backbone_feats = ref.forward_backbone(torch_input)
        neck_out = list(ref.model.neck(backbone_feats))

    ref_decoder_out = ref.forward_decoder(neck_out)
    ref_hidden_states = ref_decoder_out["hidden_states"]
    ref_references = ref_decoder_out["references"]

    # Run reference head
    with torch.no_grad():
        ref_cls, ref_coords = ref.model.bbox_head(ref_hidden_states, ref_references)

    logger.info(f"Reference cls shape: {ref_cls.shape}, coords shape: {ref_coords.shape}")

    # --- TTNN detection heads ---
    from models.experimental.dino_5scale_swin_l.tt.model_preprocessing import load_decoder_weights
    from models.experimental.dino_5scale_swin_l.tt.tt_decoder import TtRegBranch, inverse_sigmoid_torch

    decoder_params = load_decoder_weights(ckpt_path, device)

    # Prepare hidden_states as ttnn tensors on device
    hidden_states_tt = []
    for i in range(DECODER_NUM_LAYERS):
        hs = ref_hidden_states[i]  # [B, num_queries, 256]
        hs_tt = ttnn.from_torch(
            hs.to(torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        hidden_states_tt.append(hs_tt)

    # references[0] = init_reference, references[1..6] = inter_references
    references_torch = ref_references  # list of 7 tensors [B, num_queries, 4]

    # Build head components
    cls_branches = decoder_params["cls_branches"]
    reg_branches_head = [TtRegBranch(decoder_params["reg_branches"][i], device) for i in range(DECODER_NUM_LAYERS)]

    # Run TTNN heads
    all_cls = []
    all_coords = []
    for layer_id in range(DECODER_NUM_LAYERS):
        hidden_state = hidden_states_tt[layer_id]
        reference = references_torch[layer_id]

        cls_w = cls_branches[layer_id]["weight"]
        cls_b = cls_branches[layer_id]["bias"]
        cls_out_tt = ttnn.linear(hidden_state, cls_w, bias=cls_b)
        cls_out = ttnn.to_torch(cls_out_tt).float()[:, :NUM_QUERIES, :]
        ttnn.deallocate(cls_out_tt)

        reg_out_tt = reg_branches_head[layer_id](hidden_state)
        reg_out = ttnn.to_torch(reg_out_tt).float()[:, :NUM_QUERIES, :]
        ttnn.deallocate(reg_out_tt)

        ref_inv = inverse_sigmoid_torch(reference, eps=1e-3)
        coords = (reg_out + ref_inv).sigmoid()

        all_cls.append(cls_out)
        all_coords.append(coords)

    tt_cls = torch.stack(all_cls, dim=0)
    tt_coords = torch.stack(all_coords, dim=0)

    logger.info(f"TTNN cls shape: {tt_cls.shape}, coords shape: {tt_coords.shape}")

    # Compare PCC
    pcc_threshold_cls = 0.98
    pcc_threshold_bbox = 0.98

    for layer_id in range(DECODER_NUM_LAYERS):
        passing_cls, pcc_cls = comp_pcc(ref_cls[layer_id], tt_cls[layer_id], pcc_threshold_cls)
        passing_bbox, pcc_bbox = comp_pcc(ref_coords[layer_id], tt_coords[layer_id], pcc_threshold_bbox)
        logger.info(f"Layer {layer_id}: cls PCC={pcc_cls:.6f}, bbox PCC={pcc_bbox:.6f}")
        assert passing_cls, f"Layer {layer_id} cls PCC {pcc_cls:.6f} < {pcc_threshold_cls}"
        assert passing_bbox, f"Layer {layer_id} bbox PCC {pcc_bbox:.6f} < {pcc_threshold_bbox}"
