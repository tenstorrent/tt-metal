# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Correctness test for EXAONE 4.0 decoder layer on Tenstorrent hardware.

Compares the TT implementation against the HuggingFace CPU reference layer-by-layer.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig, precompute_freqs
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import HfRotarySetup, RotarySetup


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (True,),
    ids=("paged_attention",),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("max_seq_len", (256,))
@pytest.mark.parametrize("generation_length", (10,))
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_exaone4_decoder_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
    generation_length,
):
    dtype = ttnn.bfloat8_b
    mode = Mode.DECODE

    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        cache_hf=True,
        use_hf_rope=False,
    )
    model_args.n_layers = 1

    logger.info(f"Model: {model_args.model_name}, is_post_norm: {model_args.is_post_norm}")
    assert model_args.is_post_norm, "EXAONE 4.0 should be detected as post-norm"

    state_dict = model_args.load_state_dict()
    reference_model = model_args.reference_decoder(load_checkpoint=True)

    generation_start_pos = 0
    all_tests_pass = True

    # Setup RoPE
    rope_setup = RotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
        model_args.use_qk_fused,
    )

    if model_args.rope_theta_local is not None:
        rope_setup_local = RotarySetup(
            mesh_device,
            model_args.max_batch_size,
            model_args.head_dim,
            model_args.max_seq_len,
            model_args.rope_theta_local,
            None,
        )
    else:
        rope_setup_local = None

    transformation_mats = rope_setup.get_both_trans_mats()

    # Paged attention setup
    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
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
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if (model_args.is_galaxy and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # Initialize TT model
    tt_ccl = TT_CCL(mesh_device)
    tt_model = TransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
    )

    seqlen = 1

    # Precompute freqs_cis for reference model
    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.rope_scaling.factor if model_args.rope_scaling else None,
        model_args.rope_scaling.original_max_position_embeddings if model_args.rope_scaling else None,
        model_args.rope_scaling.rope_type.value if model_args.rope_scaling else "llama3",
    )
    freqs_cis = torch.complex(cos, sin)

    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    for i in range(generation_length):
        logger.info(f"[EXAONE4 Decoder] Generating token {i}")

        # Reference model
        ref_input = torch.randn(1, seqlen, model_args.dim)
        ref_output = reference_model(ref_input, current_pos[0:1], freqs_cis)

        # TT model
        tt_input = ref_input.clone()
        tt_input = ttnn.from_torch(
            tt_input,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            memory_config=model_args.get_residual_mem_config(mode, None),
        )

        rot_mat_idxs = rope_setup.get_rot_idxs(current_pos, on_host=False)
        rot_mats_global = rope_setup.get_rot_mats(rot_mat_idxs)
        rot_mats_local = rope_setup_local.get_rot_mats(rot_mat_idxs) if rope_setup_local else None

        tt_output = tt_model(
            tt_input,
            current_pos_tensor,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            mode=mode,
            page_table=page_table_tt,
        )

        # Compare outputs
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
        )[0, 0, :batch_size, : model_args.dim]

        ref_output_squeezed = ref_output.squeeze(0)

        passing, pcc_message = comp_pcc(ref_output_squeezed, tt_output_torch, pcc=0.97)
        logger.info(f"Token {i}: {pcc_message}")

        if not passing:
            all_tests_pass = False
            logger.warning(f"PCC check failed at token {i}")

        # Advance position
        current_pos += 1
        ttnn.plus_one(current_pos_tensor)

    assert all_tests_pass, "One or more PCC checks failed for EXAONE 4.0 decoder"
    logger.info("EXAONE 4.0 decoder correctness test PASSED")
