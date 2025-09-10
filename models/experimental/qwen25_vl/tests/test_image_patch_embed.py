""""Test for Qwen 2.5 VL Patch Embed"""

from loguru import logger

import torch
import pytest
import os
import ttnn
from models.experimental.qwen25_vl.tt.patch_embed import TTQwen2_5_VisionPatchEmbed

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
def test_embed_inference(seq_len, batch_size, reset_seeds, device):
    dtype = ttnn.bfloat16
    mode = "decode" if seq_len <= 32 else "prefill"

    tt_model_args = ModelArgs(
        device,
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    tt_model_args.n_layers = 1
    state_dict = tt_model_args.load_state_dict()

    reference_model = tt_model_args.reference_vision_qwen_patch_embed()  # Qwen Patch embed
    first_layer_prefix = "visual.patch_embed."

    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model.load_state_dict(partial_state_dict)

    tt_model = TTQwen2_5_VisionPatchEmbed(
        device=device,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        embed_dim=1280,
        state_dict=state_dict,
        weight_key=first_layer_prefix,
        layer_num=None,
        state_dict_prefix="",
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        mode=mode,
    )

    input = torch.rand(1, 1, 1380, 1176)
    reference_output = reference_model(input)

    tt_input = ttnn.from_torch(
        input, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    tt_output = tt_model(tt_input)

    tt_output_torch = ttnn.to_torch(
        tt_output,
    )

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("Patch embed Passed!")
    else:
        logger.warning("Patch embed Failed!")

    assert passing, f"Patch embed output does not meet PCC requirement {0.99}."
