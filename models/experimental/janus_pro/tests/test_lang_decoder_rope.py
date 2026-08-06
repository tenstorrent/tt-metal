# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.janus_pro.tt.model_config import ModelArgs
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta
from models.tt_transformers.tt.rope import get_rot_mats


@torch.no_grad()
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_rope_inference(seq_len, mesh_device, dummy_weights, reset_seeds, ensure_gc):
    pcc_required = 0.999

    model_args = ModelArgs(
        mesh_device, max_batch_size=1, max_seq_len=max(seq_len, 512), dummy_weights=dummy_weights, cache_hf=True
    )

    # TT RoPE cos/sin matrices (Meta interleaved style), built from the Janus text-decoder config.
    tt_cos, tt_sin = get_rot_mats(
        head_dim=model_args.head_dim,
        device=mesh_device,
        seq_len=seq_len,
        theta=model_args.rope_theta,
        rope_scaling=model_args.rope_scaling,
    )
    tt_cos = ttnn.to_torch(tt_cos, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1].float()
    tt_sin = ttnn.to_torch(tt_sin, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1].float()

    # HF reference: language_model.rotary_emb produces HF-style (half-dim duplicated) cos/sin.
    reference_model = model_args.reference_transformer(wrap=False)
    rotary_emb = reference_model.model.rotary_emb
    position_ids = torch.arange(seq_len).unsqueeze(0)
    dummy = torch.zeros(1, seq_len, model_args.head_dim)
    hf_cos, hf_sin = rotary_emb(dummy, position_ids)

    # Convert HF style to Meta style so it can be compared against the TT matrices.
    hf_cos_meta, hf_sin_meta = convert_rope_style_hf_to_meta(hf_cos.unsqueeze(1).float(), hf_sin.unsqueeze(1).float())

    all_tests_pass = True
    for name, ref, tt in (("cos", hf_cos_meta, tt_cos), ("sin", hf_sin_meta, tt_sin)):
        passing, pcc_message = comp_pcc(ref, tt, pcc_required)
        logger.info(comp_allclose(ref, tt))
        logger.info(f"RoPE {name} PCC: {pcc_message}")
        if passing:
            logger.info(f"RoPE {name} Passed!")
        else:
            logger.warning(f"RoPE {name} Failed!")
            all_tests_pass = False

    assert all_tests_pass, f"RoPE matrices do not meet PCC requirement {pcc_required}. Check Warnings!"
