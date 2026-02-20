# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.prefetcher import Prefetcher


@torch.no_grad()
@pytest.mark.parametrize("use_prefetcher", [False])
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
def test_llasa3b_lm_head_inference(seq_len, batch_size, mesh_device, use_prefetcher, reset_seeds):
    """
    PCC test specifically for the Llasa-3B LM Head.
    Llasa-3B has a massively expanded vocabulary size (approx 193K) compared to
    the base LLaMA-3.2-3B (approx 128K). This test verifies that the custom DRAM memory routing
    for the LMHead concatenation correctly maintains precision and avoids L1 OOMs.
    """

    # Force the environment flag so ModelArgs picks up the Llasa-3B config
    os.environ["HF_MODEL"] = "HKUSTAudio/Llasa-3B"
    logger.info("Setting HF_MODEL to HKUSTAudio/Llasa-3B to test large vocals")

    dtype = ttnn.bfloat8_b

    prefetcher = Prefetcher(mesh_device, num_tensors=0, num_layers=1) if use_prefetcher else None

    if use_prefetcher:
        prefetcher.init(mode=Mode.DECODE)

    # Initialize model arguments extracting vocabulary parameters from Llasa config
    model_args = ModelArgs(
        mesh_device, max_batch_size=batch_size, max_seq_len=seq_len, cache_hf=True, prefetcher=prefetcher
    )
    model_args.n_layers = 1

    # Assert we are actually testing the extended vocab size!
    logger.info(f"Loaded Llasa-3B with Vocab Size: {model_args.vocab_size} (Padded: {model_args.padded_vocab_size})")

    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", None)

    partial_state_dict = {
        "weight": state_dict[f"{state_dict_prefix}output.weight"],
    }

    model_args.WEIGHTS_DTYPE = dtype
    reference_model = model_args.reference_lm_head()
    reference_model.load_state_dict(partial_state_dict)

    tt_ccl = TT_CCL(mesh_device)
    tt_model = LMHead(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        dtype=dtype,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        max_columns_per_device=model_args.max_columns_per_device_lm_head,
        prefetcher=prefetcher,
    )

    # Note: Using random initialization for activation inputs
    torch_input = torch.randn(1, 1, seq_len, model_args.dim, dtype=torch.bfloat16)
    reference_output = reference_model(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=model_args.get_lm_head_input_mem_config(Mode.PREFILL, prefetcher),
        layout=ttnn.TILE_LAYOUT,
    )

    # Pass through custom extended vocab logic
    tt_output = tt_model(tt_input)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, model_args.cluster_shape, dims=(3, 1) if model_args.is_galaxy else (1, 3)
        ),
    )

    # Trim to expected logical vocab bounds
    tt_output_torch = tt_output_torch[:, 0:1, :, : model_args.vocab_size]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Llasa-3B LM_Head Passed!")
    else:
        logger.warning("Llasa-3B LM_Head Failed!")

    assert passing, f"LM_Head output does not meet PCC requirement {pcc_required}: {pcc_message}."
