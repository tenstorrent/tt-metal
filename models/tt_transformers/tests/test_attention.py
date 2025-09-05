# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import PagedAttentionConfig, precompute_freqs
from models.tt_transformers.tt.model_config import ModelArgs, CheckpointType
from models.tt_transformers.tt.rope import RotarySetup


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
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
@pytest.mark.parametrize(
    "rope_embeddings",
    ["global", "local"]
)
def test_attention_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
    rope_embeddings,
):
    from models.demos.gemma3.tt.model_config import CheckpointType, ModelArgs

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

    reference_model = model_args.reference_attention(rope_embeddings)
    # reference_model = model_args.reference_attention()
    reference_model.load_state_dict(partial_state_dict)

    seq_len = 1

    generation_start_pos = 0
    generation_length = 100
    all_tests_pass = True

    # Setup RoPE transformation matrices
    rope_setup = RotarySetup(
        mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
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
    local_transformation_mats = rope_setup_local.get_both_trans_mats() if rope_setup_local is not None else None

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
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        paged_attention_config=paged_attention_config,
    )

    if model_args.checkpoint_type == CheckpointType.Meta:
        cos, sin = precompute_freqs(
            model_args.head_dim,
            model_args.max_seq_len * 2,
            model_args.rope_theta,
            model_args.rope_scaling.factor if model_args.rope_scaling else None,
            model_args.rope_scaling.original_max_position_embeddings if model_args.rope_scaling else None,
        )
        freqs_cis = torch.complex(cos, sin)
    else:
        freqs_cis = None

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
    pccs = {"attention": [], "K": [], "V": []}
    atols = {"attention": [], "K": [], "V": []}
    rtols = {"attention": [], "K": [], "V": []}
    abs_max_vals_ref = {"attention": [], "K": [], "V": []}
    abs_max_vals_tt = {"attention": [], "K": [], "V": []}
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
        rot_mats_local = None if rope_setup_local is None else rope_setup_local.get_rot_mats(current_pos)

        tt_out = tt_model(
            attention_input,
            current_pos_tensor,
            rot_mats=rot_mats_local if rope_embeddings == "local" else rot_mats,
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
        freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0) if freqs_cis is not None else None

        reference_output = reference_model(
            pt_attention_input.to(torch.bfloat16), current_pos[0], freqs_cis_i, mask=None
        )
        # reference_output = reference_model(pt_attention_input, current_pos[0], freqs_cis_i, mask=None)

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)
        pccs["attention"].append(pcc_message)
        abs_max_vals_ref["attention"].append(torch.max(torch.abs(reference_output.to(torch.float32))))
        abs_max_vals_tt["attention"].append(torch.max(torch.abs(tt_output_torch.to(torch.float32))))
        passing_allclose, atol_delta, rtol_delta = comp_allclose(reference_output, tt_output_torch)
        atols["attention"].append(atol_delta)
        rtols["attention"].append(rtol_delta)
        # logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"Max ATOL Delta: {atol_delta}, Max RTOL Delta: {rtol_delta}")
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
                # cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
                cache_pt = cache_pt[:, :, :cache_length_to_check-generation_start_pos, :]
                cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
                abs_max_vals_ref[label].append(torch.max(torch.abs(cache_pt.to(torch.float32))))
                abs_max_vals_tt[label].append(torch.max(torch.abs(cache_tt.to(torch.float32))))
                does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
                passing_allclose, atol_delta, rtol_delta = comp_allclose(cache_pt, cache_tt)
                pccs[label].append(output_pcc)
                atols[label].append(atol_delta)
                rtols[label].append(rtol_delta)
                # logger.info(f"{label} cache output: {output_pcc}")
                logger.info(f"Max ATOL Delta: {atol_delta}, Max RTOL Delta: {rtol_delta}")
                logger.info(f"{label} cache output: {output_pcc}")
                if does_pass:
                    logger.info(f"{label} cache Passed!")
                else:
                    logger.warning(f"{label} Cache Failed! PCC value is lower than {pcc}")
                    all_tests_pass = False

    import matplotlib.pyplot as plt
    import json
    json.dump(pccs, open(f"attention_pccs_startpos{generation_start_pos}_rope{rope_embeddings}_{model_args.model_name}.json", "w"))
    fig, ax = plt.subplots(3, 3)
    ax[0][0].plot(pccs["attention"])
    ax[0][0].set_title("Attention")
    ax[0][0].set_ylim(0, 1.05)
    ax[1][0].plot(pccs["K"])
    ax[1][0].set_title("Keys")
    ax[1][0].set_ylim(0, 1.05)
    ax[2][0].plot(pccs["V"])
    ax[2][0].set_title("Values")
    ax[2][0].set_ylim(0, 1.05)

    ax[0][1].plot(atols["attention"])
    ax[0][1].set_title("Attention")
    # ax[1][0].set_ylim(0, 1.05)
    ax[1][1].plot(atols["K"])
    ax[1][1].set_title("Keys")
    # ax[1][1].set_ylim(0, 1.05)
    ax[2][1].plot(atols["V"])
    ax[2][1].set_title("Values")
    # ax[1][2].set_ylim(0, 1.05)

    ax[0][2].plot(rtols["attention"])
    ax[0][2].set_title("Attention")
    # ax[2][0].set_ylim(0, 1.05)
    ax[1][2].plot(rtols["K"])
    ax[1][2].set_title("Keys")
    # ax[2][1].set_ylim(0, 1.05)
    ax[2][2].plot(rtols["V"])
    ax[2][2].set_title("Values")
    # ax[2][2].set_ylim(0, 1.05)

    
    plt.tight_layout()
    plt.savefig(f"attention_pccs_startpos{generation_start_pos}_rope{rope_embeddings}_{model_args.model_name}.png")

    fig, ax = plt.subplots(3)
    ax[0].plot(abs_max_vals_ref["attention"], label="Reference")
    ax[0].plot(abs_max_vals_tt["attention"], label="TT")
    ax[0].set_title("Attention")
    ax[1].plot(abs_max_vals_ref["K"], label="Reference")
    ax[1].set_title("Keys")
    ax[1].plot(abs_max_vals_tt["K"], label="TT")
    ax[2].set_title("Values")
    ax[2].plot(abs_max_vals_ref["V"], label="Reference")
    ax[2].plot(abs_max_vals_tt["V"], label="TT")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(f"attention_abs_max_vals_startpos{generation_start_pos}_rope{rope_embeddings}_{model_args.model_name}.png")


    if all_tests_pass:
        logger.info("Attention output Passed!")
    else:
        logger.warning("Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("max_seq_len", [256])
@pytest.mark.parametrize("batch_size", [1])
def test_kv_cache(mesh_device, max_seq_len, batch_size):
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True)

    core_grid = mesh_device.compute_with_storage_grid_size()
    batch_grid = (
        ttnn.CoreGrid(y=4, x=8)
        if ttnn.get_arch_name() == "blackhole"
        else ttnn.num_cores_to_corerangeset(batch_size, core_grid, row_wise=True)
    )
    mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    q_heads_1BQD = torch.randn(1, 1, 32, 128)
    q_heads_1BQD = ttnn.from_torch(
        q_heads_1BQD,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[1, 2], mesh_shape=(1, 8)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    keys = torch.zeros(1, 16, 256, 128)
    keys = ttnn.from_torch(
        keys,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[0, 1], mesh_shape=(1, 8)),
        memory_config=ttnn.create_sharded_memory_config(
            keys.shape,
            core_grid=ttnn.CoreGrid(y=1, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    values = torch.zeros(1, 16, 256, 128)
    values = ttnn.from_torch(
        values,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[0, 1], mesh_shape=(1, 8)),
    )

    generation_start_pos = 0
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
        memory_config=ttnn.create_sharded_memory_config(
            current_pos_tensor.shape,
            core_grid=ttnn.CoreGrid(y=1, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    k_heads_1BKD = torch.randn(1, 1, 16, 128)
    k_heads_1BKD = ttnn.from_torch(
        k_heads_1BKD,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[1, 2], mesh_shape=(1, 8)),
        memory_config=ttnn.create_sharded_memory_config(
            k_heads_1BKD.shape,
            core_grid=ttnn.CoreGrid(y=1, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    v_heads_1BKD = torch.randn(1, 1, 16, 128)
    v_heads_1BKD = ttnn.from_torch(
        v_heads_1BKD,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[1, 2], mesh_shape=(1, 8)),
        memory_config=ttnn.create_sharded_memory_config(
            v_heads_1BKD.shape,
            core_grid=ttnn.CoreGrid(y=1, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )

    ttnn.experimental.paged_update_cache(keys, k_heads_1BKD, update_idxs_tensor=current_pos_tensor, page_table=None)
    ttnn.experimental.paged_update_cache(values, v_heads_1BKD, update_idxs_tensor=current_pos_tensor, page_table=None)
    scale = 1.0
    model_config = model_args.get_model_config()
    attn_output_1G4D = ttnn.transformer.scaled_dot_product_attention_decode(
        q_heads_1BQD,
        keys,
        values,
        cur_pos_tensor=current_pos_tensor,
        scale=scale,
        program_config=model_config["SDPA_DECODE_PROGCFG"],
        compute_kernel_config=model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_DECODE, configuration=model_config
        ),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  # FIXME: why not L1 height sharded e.g. SCORES_BATCHED_MM_OUTPUT_MEMCFG?
    )
