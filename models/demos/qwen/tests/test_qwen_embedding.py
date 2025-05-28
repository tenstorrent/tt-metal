# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen.reference.tokenizer import Tokenizer
from models.demos.qwen.tt.model_config import TtModelArgs
from models.demos.qwen.tt.qwen_common import HostEmbedding
from models.demos.qwen.tt.qwen_embedding import TtQwenEmbedding
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_qwen_embedding(mesh_device, use_program_cache, reset_seeds, ensure_gc):
    if mesh_device.shape != (1, 1):
        pytest.skip("Only N150 is supported")
    dtype = ttnn.bfloat16

    model_args = TtModelArgs(mesh_device)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    tokenizer = Tokenizer(model_args.tokenizer_path)

    reference_emb = HostEmbedding(model_args)

    layer_name = "model.embed_tokens.weight"
    reference_emb.load_state_dict({"emb.weight": state_dict[layer_name]})

    tt_emb = TtQwenEmbedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=dtype,
    )

    prompts = ["Joy"] * 32
    pt_input = torch.tensor([tokenizer.encode(prompt, bos=False) for prompt in prompts])
    reference_output = reference_emb(pt_input)
    logger.info(f"reference_output: {reference_output.shape}")

    tt_input = ttnn.from_torch(
        pt_input.squeeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_output = tt_emb(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).view(
        reference_output.shape
    )
    logger.info(f"tt_output_torch: {tt_output_torch.shape}")

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Qwen_embedding Passed!")
    else:
        logger.warning("Qwen_embedding Failed!")

    assert passing, f"Qwen_embedding output does not meet PCC requirement {0.99}."
