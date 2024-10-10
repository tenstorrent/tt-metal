# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_embedding import TtLlamaEmbedding
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3.tt.llama_common import HostEmbedding


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
def test_llama_embedding(mesh_device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat16

    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))
    tokenizer = Tokenizer(model_args.tokenizer_path)

    reference_emb = HostEmbedding(model_args)
    reference_emb.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    tt_emb = TtLlamaEmbedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=dtype,
    )

    prompts = ["Joy"] * 32
    pt_input = torch.tensor([tokenizer.encode(prompt, bos=False, eos=False) for prompt in prompts])
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
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0].view(
        reference_output.shape
    )
    logger.info(f"tt_output_torch: {tt_output_torch.shape}")

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Llama_embedding Passed!")
    else:
        logger.warning("Llama_embedding Failed!")

    assert passing, f"Llama_embedding output does not meet PCC requirement {0.99}."
