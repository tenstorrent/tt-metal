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
from models.demos.wormhole.llama31_8b_N300.tt.llama_cross_attention import TtLlamaCrossAttention
from models.demos.wormhole.llama31_8b_N300.tt.model_config import TtModelArgs
from models.demos.wormhole.llama31_8b_N300.tt.llama_common import (
    prepare_inputs_ttnn_prefill,
)
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (5120,),
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
def test_llama_cross_attention_inference(seq_len, mesh_device, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat16
    pcc = 0.99

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

    pt_xattn_tokens = (torch.rand(batch, seq_len, dim) * 2) - 1
    tt_xattn_tokens = pt_xattn_tokens.clone()
    tt_xattn_tokens = prepare_inputs_ttnn_prefill(
        tt_xattn_tokens,
        mesh_device,
    )

    """
    Test compute_xattn_kv_cache
    """
    # pt_xattn_cache = reference_model.compute_xattn_kv_cache(pt_xattn_tokens)
    # # pt_xattn_cache = [x.view(batch, n_heads, seq_len, head_dim)[:,::n_heads//n_kv_heads] for x in pt_xattn_cache]
    # # pt_xattn_cache = pt_xattn_cache.view(batch, seq_len, -1)

    # tt_xattn_cache = tt_model.compute_xattn_kv_cache(tt_xattn_tokens)
    # tt_xattn_cache_torch = ttnn.to_torch(tt_xattn_cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))#.view(batch, seq_len, -1)
    # # breakpoint()
    # passing, pcc_message = comp_pcc(pt_xattn_cache, tt_xattn_cache_torch, pcc)

    # logger.info(comp_allclose(pt_xattn_cache, tt_xattn_cache_torch))
    # logger.info(pcc_message)

    pt_xattn_cache = reference_model.compute_xattn_kv_cache(pt_xattn_tokens)
    pt_xattn_cache_chunks = torch.chunk(pt_xattn_cache, 2, dim=0)
    pt_xattn_cache_chunks = [
        x.view(batch, n_heads, seq_len, head_dim)[:, :: n_heads // n_kv_heads] for x in pt_xattn_cache
    ]
    # pt_xattn_cache_chunks = [x.view(batch, n_kv_heads, seq_len, head_dim) for x in pt_xattn_cache_chunks]

    tt_xattn_cache = tt_model.compute_xattn_kv_cache(tt_xattn_tokens)
    tt_xattn_cache_torch = [
        ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)).view(
            batch, n_kv_heads, seq_len, head_dim
        )
        for x in tt_xattn_cache
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

    if passing:
        logger.info(f"Llama_Attention Passed!")
    else:
        logger.warning(f"Llama_Attention Failed!")
        all_tests_pass = False

    if all_tests_pass:
        logger.info("Llama Attention output Passed!")
    else:
        logger.warning("Llama Attention output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
