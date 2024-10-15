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
from models.demos.llama3.tt.llama_cross_attention import TtLlamaCrossAttention
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.llama3.tt.llama_common import (
    prepare_inputs_ttnn_prefill,
    prepare_inputs_ttnn,
)
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "vision_seq_len",
    (5120,),
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
def test_llama_cross_attention_inference(
    vision_seq_len, text_seq_len, mesh_device, use_program_cache, reset_seeds, ensure_gc
):
    dtype = ttnn.bfloat16
    pcc = 0.99

    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device)
    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = "text_model.cross_attention_layers.0.attention."
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    dim = model_args.dim
    head_dim = model_args.head_dim
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads
    norm_eps = model_args.norm_eps
    reference_model = llama_reference_mod.CrossAttention(
        dim=dim, head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=norm_eps
    )
    reference_model.load_state_dict(partial_state_dict)

    batch = 1

    all_tests_pass = True

    tt_model = TtLlamaCrossAttention(
        mesh_device,
        state_dict,
        state_dict_prefix=first_layer_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        dtype=dtype,
        configuration=model_args,
        dim=dim,
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        norm_eps=norm_eps,
    )

    pt_xattn_tokens = (torch.rand(batch, vision_seq_len, dim) * 2) - 1
    tt_xattn_tokens = pt_xattn_tokens.clone()
    tt_xattn_tokens = prepare_inputs_ttnn_prefill(
        tt_xattn_tokens,
        mesh_device,
    )

    """
    Test compute_xattn_kv_cache
    """
    pt_xattn_cache = reference_model.compute_xattn_kv_cache(pt_xattn_tokens)
    pt_xattn_cache_chunks = torch.chunk(pt_xattn_cache, 2, dim=0)
    pt_xattn_cache_chunks = [
        x.view(batch, n_heads, vision_seq_len, head_dim)[:, :: n_heads // n_kv_heads] for x in pt_xattn_cache
    ]

    tt_xattn_cache = tt_model.compute_xattn_kv_cache(tt_xattn_tokens)
    tt_xattn_cache_torch = [
        ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)).view(
            batch, n_kv_heads, vision_seq_len, head_dim
        )
        for x in tt_xattn_cache
    ]

    for pt, tt in zip(pt_xattn_cache_chunks, tt_xattn_cache_torch):
        passing, pcc_message = comp_pcc(pt, tt, pcc)

        logger.info(comp_allclose(pt, tt))
        logger.info(f"PCC: {pcc_message}")
        if passing:
            logger.info(f"compute_xattn_kv_cache Passed!")
        else:
            logger.warning(f"compute_xattn_kv_cache Failed!")
            all_tests_pass = False

    """
    Test forward, prefill and decode!
    """
    for i in range(10):
        seq_len = text_seq_len if i == 0 else 1
        mode = "prefill" if i == 0 else "decode"
        pt_x = (torch.rand(batch, seq_len, dim) * 2) - 1
        tt_x = pt_x.clone()
        if mode == "prefill":
            tt_x = prepare_inputs_ttnn_prefill(
                tt_x,
                mesh_device,
            )
        else:
            tt_x = prepare_inputs_ttnn(
                tt_x,
                model_args.dim,
                mesh_device,
            )

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
        full_text_mask_expand = full_text_mask.expand(-1, n_heads // model_args.num_devices, -1, head_dim)
        tt_full_text_mask = ttnn.from_torch(
            full_text_mask_expand,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        if mode == "decode":
            tt_full_text_mask = ttnn.reshape(
                tt_full_text_mask,
                shape=ttnn.Shape(
                    [batch, n_heads // model_args.num_devices, seq_len, head_dim],
                    [batch, n_heads // model_args.num_devices, 32, head_dim],
                ),
            )

        pt_out = reference_model.forward(
            pt_x, xattn_mask=xattn_mask, full_text_row_masked_out_mask=full_text_mask, xattn_cache=pt_xattn_cache
        )

        tt_out = tt_model(
            tt_x,
            xattn_mask=tt_xattn_mask,
            full_text_row_masked_out_mask_1NSH=tt_full_text_mask,
            xattn_cache=tt_xattn_cache,
            mode=mode,
        )

        tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        if mode == "prefill":
            tt_output_torch = tt_output_torch[0, ..., :seq_len, :].view(batch, seq_len, dim)
        else:
            tt_output_torch = tt_output_torch[0, ..., :batch, :].transpose(0, 1).view(batch, seq_len, dim)
        passing, pcc_message = comp_pcc(pt_out, tt_output_torch, pcc)
        logger.info(comp_allclose(pt_out, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        all_tests_pass = all_tests_pass and passing

    if all_tests_pass:
        logger.info("Llama Attention output Passed!")
    else:
        logger.warning("Llama Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
