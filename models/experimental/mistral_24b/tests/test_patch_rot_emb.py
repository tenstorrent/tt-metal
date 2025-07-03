from loguru import logger

import torch
import pytest
import os
import ttnn

# models/tt_transformers/tt/common.py
from models.experimental.mistral_24b.tt.vision_rope import VisionRotarySetup as RotarySetup

from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
from models.tt_transformers.tt.model_config import ModelArgs


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("device"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (128,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
def test_rot_emb(seq_len, batch_size, reset_seeds, device):
    dtype = ttnn.bfloat16
    mode = "decode" if seq_len <= 32 else "prefill"

    tt_model_args = ModelArgs(
        device,
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    tt_model_args.n_layers = 1
    partial_state_dict = {}

    reference_model = tt_model_args.reference_vision_rot_emb()
    reference_model.load_state_dict(partial_state_dict)

    image_size = tt_model_args.vision_image_size
    patch_size = tt_model_args.vision_patch_size
    dim = tt_model_args.vision_head_dim
    num_patches_per_dim = image_size // patch_size
    num_patches = num_patches_per_dim * num_patches_per_dim
    position_ids = torch.arange(4096, dtype=torch.long)

    x = torch.randn(batch_size, 4096, 1024)

    cos, sin = reference_model(x, position_ids)
    tt_model = RotarySetup(
        device,
        batch_size,
        dim,
        image_size,
        patch_size,
        num_patches,
        tt_model_args.vision_rope_theta,
        scale_factor=None,
        orig_context_len=num_patches,
        datatype=dtype,
    )

    cos2, sin2 = tt_model.get_rot_mats(position_ids)
    cos2 = ttnn.from_device(cos2)
    cos2 = ttnn.to_torch(cos2)
    cos2 = cos2.squeeze(0)

    sin2 = ttnn.from_device(sin2)
    sin2 = ttnn.to_torch(sin2)
    sin2 = sin2.squeeze(0)

    passing, pcc_message = comp_pcc(cos, cos2)

    logger.info(comp_allclose(cos, cos2))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"COS PCC value is lower than {0.99} for some of the outputs. Check Warnings!"

    passing, pcc_message = comp_pcc(sin, sin2)

    logger.info(comp_allclose(sin, sin2))
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"SIN PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
