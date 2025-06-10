# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import llama_models.llama3.reference_impl.multimodal.model as llama_reference_mod
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.llama_cross_block import TtLlamaCrossAttentionTransformerBlock
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
    (1, 2),
    ids=[
        "batch_1",
        "batch_2",
    ],
)
def test_cross_attention_transformer_block_inference(
    text_seq_len, batch, mesh_device, use_program_cache, reset_seeds, ensure_gc
):
    dtype = ttnn.bfloat16
    pcc_required = 0.99

    model_args = ModelArgs(mesh_device, max_batch_size=batch)
    # Limit the max seqlen to 4k to avoid OOM on host
    model_args.max_seq_len = 4096
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "text_model.cross_attention_layers.0."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    dim = model_args.dim
    head_dim = model_args.head_dim
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads
    reference_model = llama_reference_mod.CrossAttentionTransformerBlock(args=model_args, layer_id=0, no_ffn=False)
    reference_model.load_state_dict(partial_state_dict)

    num_chunks = 4
    vision_seq_len = num_chunks * nearest_32(model_args.vision_chunk_ntok)

    all_tests_pass = True

    tt_model = TtLlamaCrossAttentionTransformerBlock(
        mesh_device,
        state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        no_ffn=False,
    )

    pt_xattn_tokens = (torch.rand(batch, vision_seq_len, dim) * 2) - 1
    tt_xattn_tokens = pt_xattn_tokens.clone()

    """
    Test compute_xattn_kv_cache
    """
    pt_xattn_cache = reference_model.compute_xattn_kv_cache(pt_xattn_tokens)
    pt_xattn_cache_chunks = torch.chunk(pt_xattn_cache, 2, dim=0)
    pt_xattn_cache_chunks = [
        x.view(batch, n_heads, vision_seq_len, head_dim)[:, :: n_heads // n_kv_heads] for x in pt_xattn_cache
    ]

    # Preallocate K and V caches
    tt_xattn_cache = [
        ttnn.from_torch(
            torch.zeros(batch, n_kv_heads, vision_seq_len, head_dim),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        for _ in range(2)
    ]

    """
    Test forward, prefill and decode!
    """
    n_iter = 10
    for i in range(n_iter):
        mode = "prefill" if i == 0 else "decode"
        seq_len = text_seq_len if mode == "prefill" else 1
        pt_x = (torch.rand(batch, seq_len, dim) * 2) - 1
        tt_x = pt_x.clone()

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

        pt_out = reference_model.forward(
            pt_x, xattn_mask=xattn_mask, full_text_row_masked_out_mask=full_text_mask, xattn_cache=pt_xattn_cache
        )

        if mode == "prefill":
            full_text_mask_expand_11SD = full_text_mask.expand(-1, -1, -1, dim)
            outputs = []
            for b in range(batch):
                tt_tensor_xattn_tokens = model_args.prepare_residual_tensor_prefill(
                    tt_xattn_tokens[b : b + 1],
                    force_replicated=True,
                )
                tt_tensor_x = model_args.prepare_residual_tensor_prefill(
                    tt_x[b : b + 1],
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
                tt_out = tt_model(
                    tt_tensor_x,
                    xattn_mask=tt_xattn_mask,
                    full_text_row_masked_out_mask_1NSH=tt_full_text_mask_expand_1NSH,
                    full_text_row_masked_out_mask_11SD=tt_full_text_mask_expand_11SD,
                    xattn_cache=tt_xattn_cache,
                    mode=mode,
                    user_id=b,
                    vision_tokens=tt_tensor_xattn_tokens,
                )

                tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
                tt_output_torch = tt_output_torch[..., :seq_len, :].view(1, seq_len, dim)
                outputs.append(tt_output_torch)
            tt_output_torch = torch.cat(outputs, dim=0).view(batch, seq_len, dim)

        else:
            tt_x = model_args.prepare_residual_tensor_decode(
                tt_x,
                model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
            )
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
                tt_x,
                xattn_mask=tt_xattn_mask,
                full_text_row_masked_out_mask_1NSH=tt_full_text_mask_expand_1NSH,
                full_text_row_masked_out_mask_11SD=tt_full_text_mask_expand_11SD,
                xattn_cache=tt_xattn_cache,
                mode=mode,
            )

            tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
            tt_output_torch = tt_output_torch[:, :, :batch, :].reshape(batch, seq_len, dim)

        passing, pcc_message = comp_pcc(pt_out, tt_output_torch, pcc_required)
        logger.info(comp_allclose(pt_out, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        all_tests_pass = all_tests_pass and passing

        if mode == "prefill":
            tt_xattn_cache_torch = [
                ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)).view(
                    batch,
                    n_kv_heads,
                    vision_seq_len,
                    head_dim,
                )
                for x in tt_xattn_cache
            ]

            for pt, tt in zip(pt_xattn_cache_chunks, tt_xattn_cache_torch):
                passing, pcc_message = comp_pcc(pt, tt, pcc_required)

                logger.info(comp_allclose(pt, tt))
                logger.info(f"PCC: {pcc_message}")
                if passing:
                    logger.info(f"compute_xattn_kv_cache Passed!")
                else:
                    logger.warning(f"compute_xattn_kv_cache Failed!")
                    all_tests_pass = False

    assert all_tests_pass, f"PCC value is lower than {pcc_required} for some of the outputs. Check Warnings!"
