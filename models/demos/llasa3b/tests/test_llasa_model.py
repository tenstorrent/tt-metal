# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs


@torch.no_grad()
@pytest.mark.parametrize("seq_len", [1])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_llasa3b_model_inference(seq_len, batch_size, mesh_device, reset_seeds):
    """
    End-to-End PCC validation for a single Transformer Decode layer (Llasa-3B).
    Because Llasa-3B heavily borrows from LLaMA-3.2-3B, this ensures that the
    underlying tt_transformers pipeline executes cleanly across the single-layer
    boundaries using Llasa's extended checkpoint weights and hidden dimensions.
    """
    os.environ["HF_MODEL"] = "HKUSTAudio/Llasa-3B"

    dtype = ttnn.bfloat8_b

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len, cache_hf=True, prefetcher=None)
    # Test a single layer of the stack to ensure weight caching and precision bounds
    # operate correctly without waiting 30 mins to compile 28 layers.
    model_args.n_layers = 1

    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", None)

    # Snip the reference state dict for 1 layer
    reference_state_dict = {
        k[len(state_dict_prefix) :]: v
        for k, v in state_dict.items()
        if (
            f"{state_dict_prefix}layers.0." in k
            or any(
                [
                    f"{state_dict_prefix}{name}" in k
                    for name in ["tok_embeddings.weight", "norm.weight", "output.weight"]
                ]
            )
        )
    }

    reference_model = model_args.reference_transformer()
    reference_model.load_state_dict(reference_state_dict)

    # Initialize Paged Attention Block
    paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
    )

    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
    )

    # Generate random sequence for input
    # Limit tokens to standard llama text space (128256) instead of padded max bound to avoid OOB config errors
    torch_input_tokens = torch.randint(0, 128256, (batch_size, seq_len))
    embd = model_args.reference_embedding(reference_model)
    weight = state_dict[f"{state_dict_prefix}tok_embeddings.weight"]
    embd.load_state_dict({"emb.weight": weight})

    pt_decode_input = embd(torch_input_tokens)

    # Instantiate Target Model
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
        prefetcher=None,
    )

    # Generate initial current_pos
    current_pos = torch.tensor([0])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
    )

    decode_input = model_args.prepare_residual_tensor_decode(
        pt_decode_input,
        model_args.get_residual_mem_config(Mode.DECODE, None),
    )
    rot_mats = tt_model.rope_setup.get_rot_mats(current_pos, None)

    logger.info("Running Reference PT Model Decode...")
    ref_output = reference_model(pt_decode_input, current_pos[0])

    logger.info("Running TTNN Model Decode...")
    tt_out = tt_model(
        decode_input,
        current_pos_tensor,
        rot_mats_global=rot_mats,
        mode=Mode.DECODE,
        page_table=page_table_tt,
    )

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, -1), mesh_shape=model_args.cluster_shape)
    tt_output_torch = (
        ttnn.to_torch(tt_out, mesh_composer=mesh_composer)
        .permute(2, 1, 0, 3)
        .squeeze(2)[:batch_size, 0:1, : model_args.vocab_size]
    )

    pcc_required = 0.94
    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(ref_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("Llasa-3B Transformer Decode Layer Passed!")
    else:
        logger.warning("Llasa-3B Transformer Decode Layer Failed!")

    assert passing, f"Transformer output does not meet PCC requirement {pcc_required}: {pcc_message}."
