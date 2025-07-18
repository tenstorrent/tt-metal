# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import llama_models.llama3.reference_impl.multimodal.model as llama_reference_mod
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import get_prefill_rot_mat, get_single_rot_mat
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_cross_attention_transformer_text import (
    TtLlamaCrossAttentionTransformerText,
)
from models.utility_functions import comp_allclose, comp_pcc, nearest_32, skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "text_seq_len",
    (2048,),
)
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
    "batch",
    (1,),
    ids=[
        "batch_1",
    ],
)
@torch.no_grad()
def test_cross_attention_transformer_text_inference(
    text_seq_len,
    batch,
    mesh_device,
    reset_seeds,
    is_ci_env,
):
    dtype = ttnn.bfloat8_b
    prefill_pcc_required = 0.98
    decode_pcc_required = 0.965

    model_args = ModelArgs(mesh_device, max_batch_size=batch)
    # Limit the max seqlen to 4k to avoid OOM on host
    model_args.max_seq_len = 4096
    kv_cache_dtype = torch.float32
    n_iter = 10
    if model_args.is_90b:
        # [INFO] use bfloat16 for in reference model to avoid OOM on host
        torch.set_default_dtype(torch.bfloat16)
        kv_cache_dtype = torch.bfloat16
        # [INFO] n_iter = 3 is sufficient to exercise both prefill and decode phases
        n_iter = 3
        if is_ci_env:
            model_args.n_layers = 1
            model_args.vision_num_cross_attention_layers = 1
            logger.info(
                f"Load and test {model_args.n_layers} layers and {model_args.vision_num_cross_attention_layers} cross attention layer in CI for Llama 90B model"
            )

    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "text_model."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    if model_args.is_90b and is_ci_env:
        # removing extra cross attention layers from the state dict as the Ref model decrees
        x_atten_prefix = "cross_attention_layers."
        partial_state_dict = {
            k: v
            for k, v in partial_state_dict.items()
            if (not k.startswith(x_atten_prefix))
            or (k.startswith(x_atten_prefix) and int(k.split(".")[1]) < model_args.vision_num_cross_attention_layers)
        }

    dim = model_args.dim
    head_dim = model_args.head_dim
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads

    reference_model = llama_reference_mod.CrossAttentionTransformerText(args=model_args)
    reference_model.setup_cache(model_args.max_batch_size, kv_cache_dtype)
    reference_model.load_state_dict(partial_state_dict)

    num_chunks = 4
    vision_seq_len = num_chunks * nearest_32(model_args.vision_chunk_ntok)

    all_tests_pass = True

    tt_model = TtLlamaCrossAttentionTransformerText(
        mesh_device,
        state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
    )
    vision_tokens = torch.randn((batch, vision_seq_len, dim))

    tt_vision_tokens = vision_tokens.clone()

    """
    Test compute_xattn_kv_cache
    """
    xattn_caches = torch.stack(
        [layer.compute_xattn_kv_cache(vision_tokens) for layer in reference_model.cross_attention_layers]
    )
    # unstack layers
    pt_xattn_cache_chunks = torch.chunk(xattn_caches, len(reference_model.cross_attention_layers), dim=0)
    # unstack k/v
    pt_xattn_cache_chunks = [torch.chunk(x, 2, dim=1) for x in pt_xattn_cache_chunks]
    pt_xattn_cache_chunks = [x for xx in pt_xattn_cache_chunks for x in xx]
    pt_xattn_cache_chunks = [
        x.view(batch, n_heads, vision_seq_len, head_dim)[:, :: n_heads // n_kv_heads] for x in pt_xattn_cache_chunks
    ]

    # Iterate over batch
    # Preallocate K and V caches
    tt_xattn_cache = tt_model.setup_cache(max_batch_size=batch)

    # Test forward pass of the model

    prev_pos = 0
    # tokens = torch.randint(100, 1000, (batch, text_seq_len+n_iter), dtype=torch.long)#, device="cuda"
    tokens = torch.randint(0, model_args.vocab_size, (batch, text_seq_len + n_iter), dtype=torch.long)
    if model_args.is_90b and is_ci_env:
        ref_file_path = model_args.CKPT_DIR + "/refpt/llama3_cross_attention_transformer_text_reference_output.pt"
        logger.info(f"Loading reference model results from file: {ref_file_path}")
        results_to_save = torch.load(ref_file_path, map_location="cpu")
        get_ref_model_logits = lambda iter_idx, *args, **kwargs: results_to_save[iter_idx]["logits"]
        get_ref_model_xattn_cache = lambda iter_idx: results_to_save[iter_idx]["xattn_cache"]
    else:
        logger.info(f"Running reference model for validation")
        get_ref_model_logits = lambda _, *args, **kwargs: reference_model.forward(*args, **kwargs)
        get_ref_model_xattn_cache = lambda _: pt_xattn_cache_chunks

    for i in range(n_iter):
        # Test prefill and decode
        mode = "prefill" if i == 0 else "decode"
        seq_len = text_seq_len if mode == "prefill" else 1
        cur_pos = seq_len + prev_pos

        # Prepare pytorch inputs
        position_ids = torch.arange(prev_pos, cur_pos, dtype=torch.long)  # , device="cuda"

        logger.info(f"mode: {mode}, seq_len: {seq_len}, cur_pos: {cur_pos}")
        logger.info(f"position_ids: {position_ids}")

        # Common mask prep
        xattn_mask = torch.bernoulli(
            torch.full(
                (
                    batch,
                    seq_len,
                    vision_seq_len,
                ),
                0.25,
            )
        )
        xattn_mask = xattn_mask.unsqueeze(1)
        xattn_mask = xattn_mask * -1e9

        xattn_mask_expand = xattn_mask.expand(-1, n_heads // model_args.num_devices, -1, -1)

        full_text_mask = torch.bernoulli(
            torch.full(
                (
                    batch,
                    seq_len,
                ),
                0.75 if seq_len != 1 else 1.0,
            )
        )
        full_text_mask = full_text_mask.unsqueeze(1).unsqueeze(-1)
        full_text_mask_expand_1NSH = full_text_mask.expand(-1, n_heads // model_args.num_devices, -1, head_dim)

        h = reference_model.get_partially_trainable_embedding(tokens[:, position_ids])

        TEXT_ONLY = False

        logits = get_ref_model_logits(
            i,
            position_ids,
            h,
            xattn_mask,
            full_text_mask,
            xattn_caches,
            text_only_inference=TEXT_ONLY,
        )

        # Prepare TT inputs
        if mode == "prefill":
            full_text_mask_expand_11SD = full_text_mask.expand(-1, -1, -1, dim)
            outputs = []
            for b in range(batch):
                tt_tensor_vision_tokens = model_args.prepare_residual_tensor_prefill(
                    tt_vision_tokens[b : b + 1],
                    force_replicated=True,
                )
                tt_h = model_args.prepare_residual_tensor_prefill(
                    h[b : b + 1],
                )
                tt_xattn_mask = ttnn.from_torch(
                    xattn_mask[b : b + 1],
                    device=mesh_device,
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                tt_full_text_mask_expand_1NSH = ttnn.from_torch(
                    full_text_mask_expand_1NSH[b : b + 1],
                    device=mesh_device,
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                tt_full_text_mask_expand_11SD = ttnn.from_torch(
                    full_text_mask_expand_11SD[b : b + 1],
                    device=mesh_device,
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                )

                rot_mats = get_prefill_rot_mat(
                    model_args.head_dim,
                    mesh_device,
                    seq_len,
                    model_args.rope_theta,
                    model_args.rope_scaling,
                )
                tt_out = tt_model(
                    tt_h,
                    xattn_mask=tt_xattn_mask,
                    full_text_row_masked_out_mask_1NSH=tt_full_text_mask_expand_1NSH,
                    full_text_row_masked_out_mask_11SD=tt_full_text_mask_expand_11SD,
                    xattn_caches=tt_xattn_cache,
                    current_pos=None,
                    rot_mats=rot_mats,
                    user_id=b,
                    mode=mode,
                    text_only_inference=TEXT_ONLY,
                    vision_tokens=tt_tensor_vision_tokens,
                )

                tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
                tt_output_torch = tt_output_torch[0, ..., :seq_len, :].view(1, seq_len, -1)
                outputs.append(tt_output_torch)

            tt_out = torch.cat(outputs, dim=0).view(batch, seq_len, -1)
            pcc_required = prefill_pcc_required

        else:
            tt_h = model_args.prepare_residual_tensor_decode(
                h,
                model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
            )
            position_ids = position_ids.reshape(1).expand(batch)
            tt_position_id = ttnn.from_torch(
                position_ids,
                device=mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

            rot_mats, _ = get_single_rot_mat(
                model_args.head_dim,
                mesh_device,
                model_args.num_devices,
                start_pos=cur_pos - 1,
                theta=model_args.rope_theta,
                rope_scaling=model_args.rope_scaling,
            )
            tt_rope_id = tt_model.rope_setup.get_rot_idxs(position_ids)
            rot_mats = tt_model.rope_setup.get_rot_mats(tt_rope_id)

            xattn_mask_expand = xattn_mask_expand.permute(2, 0, 1, 3).contiguous()
            tt_xattn_mask = ttnn.from_torch(
                xattn_mask_expand,
                device=mesh_device,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            tt_xattn_mask = ttnn.reshape(
                tt_xattn_mask,
                [1, batch, n_heads // model_args.num_devices, vision_seq_len],
                [1, batch, 32, vision_seq_len],
            )
            full_text_mask_expand_1NSH = full_text_mask.expand(-1, n_heads // model_args.num_devices, -1, head_dim)
            full_text_mask_expand_1NSH = full_text_mask_expand_1NSH.permute(2, 0, 1, 3).contiguous()
            tt_full_text_mask_expand_1NSH = ttnn.from_torch(
                full_text_mask_expand_1NSH,
                device=mesh_device,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            tt_full_text_mask_expand_1NSH = ttnn.reshape(
                tt_full_text_mask_expand_1NSH,
                [1, batch, n_heads // model_args.num_devices, head_dim],
                [1, batch, 32, head_dim],
            )

            full_text_mask_expand_11SD = full_text_mask.transpose(0, 2)
            if batch < 32:
                full_text_mask_expand_11SD = torch.cat(
                    [full_text_mask_expand_11SD, torch.zeros(1, 1, 32 - batch, 1)], dim=2
                )
            full_text_mask_expand_11SD = full_text_mask_expand_11SD.expand(-1, -1, -1, dim // model_args.num_devices)
            tt_full_text_mask_expand_11SD = ttnn.from_torch(
                full_text_mask_expand_11SD,
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            tt_full_text_mask_expand_11SD = ttnn.to_layout(tt_full_text_mask_expand_11SD, ttnn.TILE_LAYOUT)

            tt_out = tt_model(
                tt_h,
                xattn_mask=tt_xattn_mask,
                full_text_row_masked_out_mask_1NSH=tt_full_text_mask_expand_1NSH,
                full_text_row_masked_out_mask_11SD=tt_full_text_mask_expand_11SD,
                xattn_caches=tt_xattn_cache,
                current_pos=tt_position_id,
                rot_mats=rot_mats,
                mode=mode,
                text_only_inference=TEXT_ONLY,
            )

            tt_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

            tt_out = tt_out[0, :, :batch, :].reshape(logits.shape)
            pcc_required = decode_pcc_required

        passing, pcc_message = comp_pcc(logits, tt_out, pcc_required)
        logger.info(comp_allclose(logits, tt_out))
        logger.info(f"PCC: {pcc_message}")
        assert passing, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
        prev_pos = cur_pos

        if mode == "prefill":
            tt_xattn_cache_torch = [
                ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)).view(
                    batch,
                    n_kv_heads,
                    vision_seq_len,
                    head_dim,
                )
                for kv_cache in tt_xattn_cache
                for x in kv_cache
            ]

            for pt, tt in zip(get_ref_model_xattn_cache(i), tt_xattn_cache_torch):
                passing, pcc_message = comp_pcc(pt, tt, prefill_pcc_required)

                logger.info(comp_allclose(pt, tt))
                logger.info(f"PCC: {pcc_message}")

                if passing:
                    logger.info(f"compute_xattn_kv_cache Passed!")
                else:
                    logger.warning(f"compute_xattn_kv_cache Failed!")
                    all_tests_pass = False

        assert all_tests_pass, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
