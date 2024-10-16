# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
import importlib

llama_reference_mod = importlib.import_module(
    "models.demos.t3000.llama2_70b.reference.llama-models.models.llama3.reference_impl.multimodal.model"
)
from models.demos.llama3.tt.llama_cross_attention_transformer_text import TtLlamaCrossAttentionTransformerText
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.llama3.tt.llama_common import (
    prepare_inputs_ttnn_prefill,
    prepare_inputs_ttnn,
    get_prefill_rot_mat,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
    get_single_rot_mat,
)
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "vision_seq_len",
    (4224,),
)
@pytest.mark.parametrize(
    "text_seq_len",
    (2048,),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_llama_cross_attention_transformer_text_inference(
    vision_seq_len,
    text_seq_len,
    mesh_device,
    use_program_cache,
    reset_seeds,
    no_grad,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.98

    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "text_model."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    tt_model = TtLlamaCrossAttentionTransformerText(
        mesh_device,
        state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
    )

    dim = model_args.dim
    head_dim = model_args.head_dim
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads
    # norm_eps = model_args.norm_eps
    reference_model = llama_reference_mod.CrossAttentionTransformerText(args=model_args)
    reference_model.setup_cache(model_args.max_batch_size, torch.float32)
    reference_model.load_state_dict(partial_state_dict)

    batch = 1

    all_tests_pass = True

    vision_tokens = torch.randn((batch, vision_seq_len, dim))

    tt_vision_tokens = vision_tokens.clone()
    tt_vision_tokens = prepare_inputs_ttnn_prefill(
        tt_vision_tokens,
        mesh_device,
    )

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
    # slice out replicated k/v heads
    pt_xattn_cache_chunks = [
        x.view(batch, n_heads, vision_seq_len, head_dim)[:, :: n_heads // n_kv_heads] for x in pt_xattn_cache_chunks
    ]

    tt_xattn_cache = [layer.compute_xattn_kv_cache(tt_vision_tokens) for layer in tt_model.cross_attention_layers]
    tt_xattn_cache_torch = [
        ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)).view(
            batch, n_kv_heads, vision_seq_len, head_dim
        )
        for kv_cache in tt_xattn_cache
        for x in kv_cache
    ]

    for pt, tt in zip(pt_xattn_cache_chunks, tt_xattn_cache_torch):
        passing, pcc_message = comp_pcc(pt, tt, pcc)

        logger.info(comp_allclose(pt, tt))
        logger.info(pcc_message)

        if passing:
            logger.info(f"compute_xattn_kv_cache Passed!")
        else:
            logger.warning(f"compute_xattn_kv_cache Failed!")
            all_tests_pass = False

    assert all_tests_pass

    # Test forward pass of the model
    n_iter = 2000
    prev_pos = 0
    # tokens = torch.randint(100, 1000, (batch, text_seq_len+n_iter), dtype=torch.long)#, device="cuda"
    tokens = torch.randint(
        0, model_args.vocab_size, (batch, text_seq_len + n_iter), dtype=torch.long
    )  # , device="cuda"
    for i in range(n_iter):
        # Test prefill and decode
        mode = "prefill" if i == 0 else "decode"
        seq_len = text_seq_len if mode == "prefill" else 1
        cur_pos = seq_len + prev_pos

        # Prepare pytorch inputs
        position_ids = torch.arange(prev_pos, cur_pos, dtype=torch.long)  # , device="cuda"

        print(f"mode: {mode}, seq_len: {seq_len}, cur_pos: {cur_pos}")
        print(f"position_ids: {position_ids}")
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

        h = reference_model.get_partially_trainable_embedding(tokens[:, position_ids])

        logits = reference_model.forward(
            position_ids,
            h,
            xattn_mask,
            full_text_mask,
            xattn_caches,
            text_only_inference=True,
        )

        # Prepare TT inputs

        if mode == "prefill":
            tt_h = prepare_inputs_ttnn_prefill(
                h,
                mesh_device,
            )
        else:
            tt_h = prepare_inputs_ttnn(
                h,
                model_args.dim,
                mesh_device,
            )

        tt_position_id = ttnn.from_torch(
            position_ids.reshape(batch, seq_len),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        if mode == "prefill":
            rot_mats = get_prefill_rot_mat(model_args.head_dim, model_args.max_seq_len, mesh_device, seq_len=seq_len)
            transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
            transformation_mats = ttnn.as_tensor(
                transformation_mat_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            rot_mats, rot_matrix = get_single_rot_mat(
                model_args.head_dim,
                mesh_device,
                model_args.num_devices,
                start_pos=cur_pos - 1,
            )
            transformation_mats = None

        xattn_mask_expand = xattn_mask.expand(-1, n_heads // model_args.num_devices, -1, -1)
        tt_xattn_mask = ttnn.from_torch(
            xattn_mask_expand,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        if mode == "decode":
            tt_xattn_mask = ttnn.reshape(
                tt_xattn_mask,
                shape=ttnn.Shape(
                    [batch, n_heads // model_args.num_devices, seq_len, vision_seq_len],
                    [batch, n_heads // model_args.num_devices, 32, vision_seq_len],
                ),
            )
        full_text_mask_expand_1NSH = full_text_mask.expand(-1, n_heads // model_args.num_devices, -1, head_dim)
        tt_full_text_mask_expand_1NSH = ttnn.from_torch(
            full_text_mask_expand_1NSH,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        if mode == "decode":
            tt_full_text_mask_expand_1NSH = ttnn.reshape(
                tt_full_text_mask_expand_1NSH,
                shape=ttnn.Shape(
                    [batch, n_heads // model_args.num_devices, seq_len, head_dim],
                    [batch, n_heads // model_args.num_devices, 32, head_dim],
                ),
            )

        full_text_mask_expand_11SD = full_text_mask.expand(-1, -1, -1, dim)
        tt_full_text_mask_expand_11SD = ttnn.from_torch(
            full_text_mask_expand_11SD,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        if mode == "decode":
            tt_full_text_mask_expand_11SD = ttnn.reshape(
                tt_full_text_mask_expand_11SD,
                shape=ttnn.Shape(
                    [batch, 1, seq_len, head_dim],
                    [batch, 1, 32, head_dim],
                ),
            )

        tt_out = tt_model(
            tt_h,
            xattn_mask=tt_xattn_mask,
            full_text_row_masked_out_mask_1NSH=tt_full_text_mask_expand_1NSH,
            full_text_row_masked_out_mask_11SD=tt_full_text_mask_expand_11SD,
            xattn_caches=tt_xattn_cache,
            current_pos=tt_position_id,
            rot_mat=rot_mats,
            transformation_mats=transformation_mats,
            user_id=0,
            mode=mode,
            text_only_inference=True,
        )

        tt_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        # breakpoint()
        if mode == "prefill":
            tt_out = tt_out[0].reshape(logits.shape)
        else:
            tt_out = tt_out[0, ..., :batch, :].transpose(0, 1).view(logits.shape)
        passing, pcc_message = comp_pcc(logits, tt_out, pcc)
        logger.info(comp_allclose(logits, tt_out))
        logger.info(pcc_message)
        prev_pos = cur_pos
        # assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"

    # pt_xattn_tokens = (torch.rand(batch, vision_seq_len, dim) * 2) - 1
    # tt_xattn_tokens = pt_xattn_tokens.clone()
    # tt_xattn_tokens = prepare_inputs_ttnn_prefill(
    #     tt_xattn_tokens,
    #     mesh_device,
    # )

    # """
    # Test compute_xattn_kv_cache
    # """
    # pt_xattn_cache = reference_model.compute_xattn_kv_cache(pt_xattn_tokens)
    # pt_xattn_cache_chunks = torch.chunk(pt_xattn_cache, 2, dim=0)
    # pt_xattn_cache_chunks = [
    #     x.view(batch, n_heads, vision_seq_len, head_dim)[:, :: n_heads // n_kv_heads] for x in pt_xattn_cache
    # ]

    # tt_xattn_cache = tt_model.compute_xattn_kv_cache(tt_xattn_tokens)
    # tt_xattn_cache_torch = [
    #     ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)).view(
    #         batch, n_kv_heads, vision_seq_len, head_dim
    #     )
    #     for x in tt_xattn_cache
    # ]

    # for pt, tt in zip(pt_xattn_cache_chunks, tt_xattn_cache_torch):
    #     passing, pcc_message = comp_pcc(pt, tt, pcc)

    #     logger.info(comp_allclose(pt, tt))
    #     logger.info(pcc_message)

    #     if passing:
    #         logger.info(f"compute_xattn_kv_cache Passed!")
    #     else:
    #         logger.warning(f"compute_xattn_kv_cache Failed!")
    #         all_tests_pass = False

    # """
    # Test forward, prefill and decode!
    # """
    # for i in range(10):
    #     seq_len = text_seq_len if i == 0 else 1
    #     mode = "prefill" if i == 0 else "decode"
    #     pt_x = (torch.rand(batch, seq_len, dim) * 2) - 1
    #     tt_x = pt_x.clone()
    #     if mode == "prefill":
    #         tt_x = prepare_inputs_ttnn_prefill(
    #             tt_x,
    #             mesh_device,
    #         )
    #     else:
    #         tt_x = prepare_inputs_ttnn(
    #             tt_x,
    #             model_args.dim,
    #             mesh_device,
    #         )

    #     xattn_mask = torch.bernoulli(
    #         torch.full(
    #             (
    #                 batch,
    #                 seq_len,
    #                 vision_seq_len,
    #             ),
    #             0.25,
    #         )
    #     )
    #     xattn_mask = xattn_mask.unsqueeze(1)
    #     xattn_mask = xattn_mask * -1e9

    #     xattn_mask_expand = xattn_mask.expand(-1, n_heads // model_args.num_devices, -1, -1)
    #     tt_xattn_mask = ttnn.from_torch(
    #         xattn_mask_expand,
    #         device=mesh_device,
    #         dtype=ttnn.bfloat8_b,
    #         layout=ttnn.TILE_LAYOUT,
    #         memory_config=ttnn.DRAM_MEMORY_CONFIG,
    #         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    #     )
    #     tt_xattn_mask = ttnn.reshape(
    #         tt_xattn_mask,
    #         shape=ttnn.Shape(
    #             [batch, n_heads // model_args.num_devices, seq_len, vision_seq_len],
    #             [batch, n_heads // model_args.num_devices, 32, vision_seq_len],
    #         ),
    #     )

    #     full_text_mask = torch.bernoulli(
    #         torch.full(
    #             (
    #                 batch,
    #                 seq_len,
    #             ),
    #             0.75 if seq_len != 1 else 1.0,
    #         )
    #     )
    #     full_text_mask = full_text_mask.unsqueeze(1).unsqueeze(-1)
    #     full_text_mask_expand_1NSH = full_text_mask.expand(-1, n_heads // model_args.num_devices, -1, head_dim)
    #     tt_full_text_mask_expand_1NSH = ttnn.from_torch(
    #         full_text_mask_expand_1NSH,
    #         device=mesh_device,
    #         dtype=ttnn.bfloat8_b,
    #         layout=ttnn.TILE_LAYOUT,
    #         memory_config=ttnn.DRAM_MEMORY_CONFIG,
    #         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    #     )
    #     if mode == "decode":
    #         tt_full_text_mask_expand_1NSH = ttnn.reshape(
    #             tt_full_text_mask_expand_1NSH,
    #             shape=ttnn.Shape(
    #                 [batch, n_heads // model_args.num_devices, seq_len, head_dim],
    #                 [batch, n_heads // model_args.num_devices, 32, head_dim],
    #             ),
    #         )

    #     full_text_mask_expand_11SD = full_text_mask.expand(-1, -1, -1, dim)
    #     tt_full_text_mask_expand_11SD = ttnn.from_torch(
    #         full_text_mask_expand_11SD,
    #         device=mesh_device,
    #         dtype=ttnn.bfloat8_b,
    #         layout=ttnn.TILE_LAYOUT,
    #         memory_config=ttnn.DRAM_MEMORY_CONFIG,
    #         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    #     )
    #     if mode == "decode":
    #         tt_full_text_mask_expand_11SD = ttnn.reshape(
    #             tt_full_text_mask_expand_11SD,
    #             shape=ttnn.Shape(
    #                 [batch, 1, seq_len, head_dim],
    #                 [batch, 1, 32, head_dim],
    #             ),
    #         )

    #     pt_out = reference_model.forward(
    #         pt_x, xattn_mask=xattn_mask, full_text_row_masked_out_mask=full_text_mask, xattn_cache=pt_xattn_cache
    #     )

    #     tt_out = tt_model(
    #         tt_x,
    #         xattn_mask=tt_xattn_mask,
    #         full_text_row_masked_out_mask_1NSH=tt_full_text_mask_expand_1NSH,
    #         full_text_row_masked_out_mask_11SD=tt_full_text_mask_expand_11SD,
    #         xattn_cache=tt_xattn_cache,
    #         mode=mode,
    #     )

    #     tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    #     if mode == "prefill":
    #         tt_output_torch = tt_output_torch[0, ..., :seq_len, :].view(batch, seq_len, dim)
    #     else:
    #         tt_output_torch = tt_output_torch[0, ..., :batch, :].transpose(0, 1).view(batch, seq_len, dim)
    #     passing, pcc_message = comp_pcc(pt_out, tt_output_torch, pcc)
    #     logger.info(comp_allclose(pt_out, tt_output_torch))
    #     logger.info(pcc_message)
    #     all_tests_pass = all_tests_pass and passing

    # if all_tests_pass:
    #     logger.info("Llama Attention output Passed!")
    # else:
    #     logger.warning("Llama Attention output Failed!")
    #     assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
