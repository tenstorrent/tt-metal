# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.gemma3.tt.model_config import ModelArgs as Gemma3ModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import precompute_freqs_cis
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import PagedAttentionConfig, get_rot_transformation_mat
from models.tt_transformers.tt.rope import get_rot_mats


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
# Model and attention prefill tests should run both with and without paged attention to debug any issues that may occur with default attention
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        False,
    ),
    ids=(
        "paged_attention",
        "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "max_seq_len",
    (
        256,  # 4096,
        # 1024 * 32,
        # 1024 * 64,
    ),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_attention_inference(
    max_seq_len,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99
    batch_size = 1  # For prefill we only support batch_size = 1

    model_args = Gemma3ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True)

    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("Attention", 0) + "."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model = model_args.reference_attention()
    reference_model.load_state_dict(partial_state_dict)

    # pre-compute the rotational embedding matrix and send to device
    rot_mats = get_rot_mats(
        head_dim=model_args.head_dim,
        device=mesh_device,
        seq_len=max_seq_len,
        theta=model_args.rope_theta,
        rope_scaling=model_args.rope_scaling,
    )
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)

    transformation_mats_prefill = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"prefill": transformation_mats_prefill}

    generation_start_pos = 0
    generation_length = 3
    all_tests_pass = True

    # Setup page table
    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    tt_ccl = TT_CCL(mesh_device)
    tt_model = Attention(
        mesh_device,
        tt_ccl,
        model_args,
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        paged_attention_config=paged_attention_config,
    )

    pt_attention_input = (
        torch.rand(
            batch_size, max_seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
        )
        * 2
    ) - 1
    tt_attention_input = pt_attention_input.clone()
    attention_input = model_args.prepare_residual_tensor_prefill(
        tt_attention_input,
        force_replicated=False if model_args.is_galaxy else True,
    )

    tt_out = tt_model(
        attention_input,
        current_pos=None,
        rot_mats=rot_mats,
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
    )
    tt_out = ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape)
    )
    tt_output_torch = tt_out[:, 0:1, :, : model_args.dim].view(batch_size, max_seq_len, -1)  # [ batch, seq, hidden_dim]
    positions = torch.LongTensor(range(max_seq_len))
    freqs_cis_i = precompute_freqs_cis(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.rope_scaling.factor if model_args.rope_scaling else None,
    )[positions]
    attn_mask = torch.full((max_seq_len, max_seq_len), torch.finfo(torch.float32).min)
    attn_mask_torch = torch.triu(attn_mask, diagonal=1)
    reference_output = reference_model(pt_attention_input, positions[0], freqs_cis_i, mask=attn_mask_torch)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info(f"Attention Passed!")
    else:
        logger.warning(f"Attention Failed!")
        all_tests_pass = False

    check_kv_cache = True  # May want to disable: Issue #10648
    if check_kv_cache:
        # PyTorch output --------------------------------------------------------------------
        pytorch_layer_present = [
            reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch_size, n_kv_heads, seq, head_dim]
            reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch_size, n_kv_heads, seq, head_dim]
        ]
        # TT hardware execution -------------------------------------------------------------
        if paged_attention:
            tt_layer_present = [
                (
                    ttnn.to_torch(
                        cache,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device,
                            dims=(1, 3) if model_args.is_galaxy else (0, 1),
                            mesh_shape=model_args.cluster_shape,
                        ),
                    )[reverse_permutation][:, : model_args.n_kv_heads, :, : model_args.head_dim]
                    .reshape(
                        model_args.max_batch_size,
                        paged_attention_config.max_num_blocks // model_args.max_batch_size,
                        model_args.n_kv_heads,
                        paged_attention_config.block_size,
                        model_args.head_dim,
                    )
                    .transpose(1, 2)
                    .reshape(model_args.max_batch_size, model_args.n_kv_heads, -1, model_args.head_dim)[
                        :batch_size, ...
                    ]
                )
                for cache in tt_model.layer_past
            ]
        else:
            tt_layer_present = [
                ttnn.to_torch(
                    cache,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device,
                        dims=(1, 0) if model_args.is_galaxy else (0, 1),
                        mesh_shape=model_args.cluster_shape,
                    ),
                )[:batch_size, :, :, :]
                for cache in tt_model.layer_past
            ]

        for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
            cache_length_to_check = min(model_args.max_seq_len, generation_start_pos + generation_length + 1)
            cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
            cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
            does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
            if i == 0:
                logger.info(f"K cache output: {output_pcc}")
            else:
                logger.info(f"V cache output: {output_pcc}")

            if does_pass:
                logger.info(f"KV Cache Passed!")
            else:
                logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
                all_tests_pass = False

    if all_tests_pass:
        logger.info("Attention output Passed!")
    else:
        logger.warning("Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
