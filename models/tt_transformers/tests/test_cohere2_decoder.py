# SPDX-FileCopyrightText: © 2026 Tenstorrent AI, Inc.

# SPDX-License-Identifier: Apache-2.0
# Cohere2 (Command-R7B) hybrid decoder coverage: layer 0 exercises the
# sliding-attention + local-rotary path, layer 3 the full-attention NoPE
# path. The generic test_decoder.py only builds layer 0, so a NoPE routing
# regression (e.g. rotary applied on, or missing from, the wrong layers)
# would go unnoticed. Reference is the true HF Cohere2DecoderLayer.
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
from models.tt_transformers.tt.model_config import HfDecoderWrapper, ModelArgs
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
    (True,),
    ids=("paged_attention",),
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
    (256,),
)
@pytest.mark.parametrize(
    "generation_length",
    (10,),
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "layer_num",
    (0, 3),  # 0: sliding + local rotary, 3: full attention + NoPE
    ids=("sliding-layer0", "full-layer3"),
)
def test_cohere2_decoder_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
    generation_length,
    layer_num,
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
    # Keep layers 0..3 so the full-attention layer 3 weights survive trimming
    model_args.n_layers = 4

    state_dict = model_args.load_state_dict()

    # Raw HF model for per-layer references (self-loading from checkpoint)
    hf_model = model_args.reference_transformer(wrap=False, load_checkpoint=True)

    # Setup RoPE transformation matrices; NoPE global layers (full attention)
    # take identity cos/sin exactly like Transformer does.
    global_rope_kwargs = {"nope": True} if model_args.use_global_nope else {}
    rope_setup = RotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
        model_args.use_qk_fused,
        **global_rope_kwargs,
    )
    if model_args.rope_theta_local is not None:
        rope_setup_local = RotarySetup(
            mesh_device,
            model_args.max_batch_size,
            model_args.head_dim,
            model_args.max_seq_len,
            model_args.rope_theta_local,
            None,
            use_qk_fused=model_args.use_qk_fused,
        )
    else:
        rope_setup_local = None
    transformation_mats = rope_setup.get_both_trans_mats()

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
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
    )

    tt_ccl = TT_CCL(mesh_device)
    tt_model = TransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=layer_num,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
    )

    hf_layer = hf_model.model.layers[layer_num]
    reference_model = HfDecoderWrapper(
        hf_layer,
        model_args.head_dim,
        hf_model.model.rotary_emb,
        None,
        use_hf_rope=False,
        skip_qkv_permute=model_args.skip_qkv_permute,
    )

    seqlen = 1
    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.rope_scaling.factor if model_args.rope_scaling else None,
        model_args.rope_scaling.original_max_position_embeddings if model_args.rope_scaling else None,
        model_args.rope_scaling.rope_type.value if model_args.rope_scaling else "llama3",
    )
    freqs_cis = torch.complex(cos, sin)

    current_pos = torch.tensor([0 for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
    )

    all_tests_pass = True
    for i in range(generation_length):
        logger.info(f"[Cohere2 layer {layer_num}] Generating token {i}")
        pt_decode_input = (
            torch.rand(
                batch_size, seqlen, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
            )
            * 2
        ) - 1
        decode_input = model_args.prepare_residual_tensor_decode(
            pt_decode_input.clone(),
            model_args.get_residual_mem_config(mode),
        )

        rot_mats = rope_setup.get_rot_mats(current_pos)
        rot_mats_local = None if rope_setup_local is None else rope_setup_local.get_rot_mats(current_pos)

        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats_global=rot_mats,
            rot_mats_local=rot_mats_local,
            mode=mode,
            page_table=page_table_tt,
        )
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        tt_output_torch = tt_out[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)

        freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)
        ref_output = reference_model(pt_decode_input, current_pos[0], freqs_cis_i, mask=None)
        if ref_output.dim() == 2:
            ref_output = ref_output.unsqueeze(1)

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch[: ref_output.shape[0]])
        logger.info(comp_allclose(ref_output, tt_output_torch[: ref_output.shape[0]]))
        logger.info(f"PCC: {pcc_message}")
        if passing:
            logger.info(f"Cohere2 layer {layer_num} Passed!")
        else:
            logger.warning(f"Cohere2 layer {layer_num} Failed!")
            all_tests_pass = False

        current_pos = torch.tensor([i + 1 for _ in range(batch_size)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape
            ),
        )

    assert all_tests_pass, f"Cohere2 layer {layer_num} PCC below 0.99 on some iteration. Check Warnings!"
