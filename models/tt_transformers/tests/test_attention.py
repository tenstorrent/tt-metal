# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import PagedAttentionConfig, precompute_freqs
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import RotarySetup


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
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_attention_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True)
    model_args.n_layers = 1  # For the unit test, just run a single layer

    state_dict = model_args.load_state_dict()

    first_layer_prefix = model_args.get_state_dict_prefix("Attention", 0) + "."
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    reference_model = model_args.reference_attention()
    reference_model.load_state_dict(partial_state_dict)

    seq_len = 1

    generation_start_pos = 0
    generation_length = 10
    all_tests_pass = True

    # Setup RoPE transformation matrices
    rope_setup = RotarySetup(
        mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
        model_args.use_qk_fused,
    )

    transformation_mats = rope_setup.get_both_trans_mats()

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
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if (model_args.is_galaxy and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
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

    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.rope_scaling.factor if model_args.rope_scaling else None,
        model_args.rope_scaling.original_max_position_embeddings if model_args.rope_scaling else None,
        model_args.rope_scaling.rope_type.value if model_args.rope_scaling else "llama3",
    )
    freqs_cis = torch.complex(cos, sin)

    # Initial positions
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
        # 70B attention block typically sees tensors with mean 0 and std 0.03 - 0.05 in layer 1
        pt_attention_input = torch.randn(
            batch_size, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
        )  # Qwen2.5 0.5B sees 0.1 to 2.1

        tt_attention_input = pt_attention_input.clone()

        attention_input = model_args.prepare_residual_tensor_decode(
            tt_attention_input,
            model_args.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
            force_replicated=False if model_args.is_galaxy else True,
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = rope_setup.get_rot_mats(current_pos)

        tt_out = tt_model(
            attention_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        # multi-device attention module returns replicated output
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        tt_output_torch = tt_out[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)

        # In this test all users have the same position (if using batch > 1)
        freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

        reference_output = reference_model(pt_attention_input, current_pos[0], freqs_cis_i, mask=None)

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        if passing:
            logger.info(f"[pos={current_pos[0]}] Attention Passed!")
        else:
            logger.warning(f"[pos={current_pos[0]}] Attention Failed!")
            all_tests_pass = False

        # Increment position
        current_pos = torch.tensor([generation_start_pos + i + 1 for _ in range(batch_size)])
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

        check_kv_cache = True
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
            for label, cache_pt, cache_tt in zip(["K", "V"], pytorch_layer_present, tt_layer_present):
                cache_length_to_check = min(model_args.max_seq_len, generation_start_pos + i + 1)
                cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
                cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
                does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
                logger.info(f"{label} cache output: {output_pcc}")
                if does_pass:
                    logger.info(f"{label} cache Passed!")
                else:
                    logger.warning(f"{label} Cache Failed! PCC value is lower than {pcc}")
                    all_tests_pass = False

    if all_tests_pass:
        logger.info("Attention output Passed!")
    else:
        logger.warning("Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
