# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import bz2
import os
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    HostEmbedding,
    PagedAttentionConfig,
)
from models.tt_transformers.tt.rope import get_rot_mats
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_embedding import TtLlamaEmbedding
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from transformers import AutoTokenizer
from models.common.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@torch.no_grad()
@pytest.mark.timeout(900)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
# Model and attention prefill tests should run both with and without paged attention to debug any issues that may occur with default attention
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        # False,
    ),
    ids=(
        "paged_attention",
        # "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 4096}],
)
@pytest.mark.parametrize(
    "seq_len",
    (2048, 4096, 8192),
    ids=[
        # "128",
        "2048",
        "4096",
        "8192",
        # "32k",
        # "64k",
    ],
)
@pytest.mark.parametrize(
    "max_seq_len",
    (131072,),
    ids=[
        "max128k",
    ],
)
@pytest.mark.parametrize(
    "num_layers",
    (64,),
    ids=[
        "1layer",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
def test_qwen_model_prefill_inference(
    paged_attention,
    page_params,
    seq_len,
    max_seq_len,
    num_layers,
    mesh_device,
    reset_seeds,
    ensure_gc,
    is_ci_env,
    request,
):
    test_id = request.node.callspec.id
    if is_ci_env:
        if "accuracy" in test_id:
            pytest.skip("CI test only runs performance mode to reduce CI pipeline load")

        # TODO: Save ref outputs to avoid running reference model for large seq_len
        if seq_len > 8192:
            pytest.skip("CI test only runs up to 8192 seq_len to avoid out of ram issues for ref model")
        if num_layers != 1 and seq_len != 4096:
            pytest.skip("CI only runs full model for 4k seq len to reduce CI pipeline load")

    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
    cache_pcc = num_layers == 1  # Flag to measure KV cache PCC. Avoid running for all layers to speed up test time.
    dtype = ttnn.bfloat8_b
    batch_size = 1  # For prefill we only support batch_size = 1

    # Use instruct weights instead of general weights
    instruct = False
    dummy_weights = False

    # Load Qwen model using TtQwenModelArgs
    model_args = TtQwenModelArgs(
        mesh_device,
        instruct=instruct,
        dummy_weights=dummy_weights,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )

    if num_layers is not None:
        model_args.n_layers = num_layers

    # Load tt_transformers reference model args for reference transformer
    model_args_ref = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True)
    model_args_ref.n_layers = model_args.n_layers

    # Use Qwen tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(model_args.TOKENIZER_PATH)

    # This sets the minimum PCC for each iteration based on optimization mode
    if num_layers == 1:
        expec_out_pcc = 0.97
        expec_kv_cache_pcc = 0.99
    else:
        if "accuracy" in test_id:
            expec_out_pcc = 0.915  # TODO Look on improving PCC
        else:  # performance mode
            expec_out_pcc = 0.869  # TODO Look on improving PCC
        expec_kv_cache_pcc = 0.88

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        if paged_attention
        else None
    )

    logger.info("Loading weights...")
    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    reference_state_dict = {
        k[len(state_dict_prefix) :]: v
        for k, v in state_dict.items()
        if (
            any([f"{state_dict_prefix}layers.{i}." in k for i in range(model_args.n_layers)])
            or any(
                [
                    f"{state_dict_prefix}{name}" in k
                    for name in ["tok_embeddings.weight", "norm.weight", "output.weight"]
                ]
            )
        )
    }

    # Load reference model state dict
    state_dict_ref = model_args_ref.load_state_dict()
    state_dict_prefix_ref = model_args_ref.get_state_dict_prefix("", None)
    reference_state_dict_ref = {
        k[len(state_dict_prefix_ref) :]: v
        for k, v in state_dict_ref.items()
        if (
            any([f"{state_dict_prefix_ref}layers.{i}." in k for i in range(model_args_ref.n_layers)])
            or any(
                [
                    f"{state_dict_prefix_ref}{name}" in k
                    for name in ["tok_embeddings.weight", "norm.weight", "output.weight"]
                ]
            )
        )
    }
    logger.info("Finished loading weights...")

    # Load TTNN model
    logger.info(f"Loading TT model...")
    tt_embd = TtLlamaEmbedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
    )
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
        mode="prefill",
        allocate_prefill_buffers=True,
    )
    logger.info("Finished loading TT model.")

    # Create page table if paged attention is enabled
    if paged_attention:
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
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )
    else:
        page_table_tt = None

    # Load prompt
    current_file_path = os.path.abspath(__file__)
    current_file_dir = os.path.dirname(current_file_path)
    prompt_file = os.path.join(current_file_dir, "tale-of-two-cities.txt.bz2")
    with bz2.open(prompt_file, "rt", encoding="utf-8") as f:
        prompt = f.read()
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=True)[:seq_len]
    logger.info(f"Prompt length: {len(encoded_prompt)} tokens")

    # Load reference model
    if run_ref_pt:
        logger.info("Loading reference model...")
        reference_model = model_args_ref.reference_transformer()
        reference_model.load_state_dict(reference_state_dict_ref)
        # Embedding on host
        embd = HostEmbedding(model_args)
        embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})
        logger.info("Finished loading reference model.")

    # Pre-compute the rotational embedding matrix and send to device
    rot_mats = get_rot_mats(
        head_dim=model_args.head_dim,
        device=mesh_device,
        seq_len=max_seq_len,
        theta=model_args.rope_theta,
        rope_scaling=model_args.rope_scaling_factor,
    )

    # Select the first token from the prompt for initial decoding
    encoded_prompt_tensor = torch.tensor(encoded_prompt)
    tt_prefill_input = encoded_prompt_tensor.unsqueeze(0)
    start_pos = 0

    # Prepare input for TT model
    if run_ref_pt:
        pt_prefill_input = embd(encoded_prompt_tensor).view(batch_size, seq_len, -1)
    else:
        # Create dummy input for TT model when not running reference
        pt_prefill_input = torch.randn(batch_size, seq_len, model_args.dim)

    tt_prefill_input = model_args.prepare_residual_tensor_prefill(
        pt_prefill_input,
    )

    # Run TT model
    logger.info(f"Running TT model...")
    tt_out = tt_model(
        tt_prefill_input,
        current_pos=None,
        rot_mats=rot_mats,
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
    )

    # Convert transformer output to torch tensor first
    tt_output_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=(1, 3) if model_args.is_galaxy else (1, 3), mesh_shape=model_args.cluster_shape
        ),
    )
    tt_output_torch = tt_output_torch[:, 0:1, :, : model_args.dim].view(
        batch_size, seq_len, -1
    )  # [batch, seq, hidden_dim]
    tt_output_torch = tt_output_torch * torch.rsqrt(
        tt_output_torch.pow(2).mean(-1, keepdim=True) + tt_model.norm.norm.eps
    )
    tt_output_torch = tt_output_torch * tt_model.norm_weight

    # Apply LM head on host (CPU)
    # Get the last token output for prefill
    last_token_output = tt_output_torch[:, -1:, :]  # [batch, 1, hidden_dim]

    # Load the LM head weight from state dict and apply it on host
    lm_head_weight = state_dict[f"{state_dict_prefix}output.weight"]  # [vocab_size, hidden_dim]
    tt_output_torch = torch.matmul(last_token_output, lm_head_weight.T)  # [batch, 1, vocab_size]
    logger.info(f"Finished running TT model.")

    if run_ref_pt:
        # Run reference model
        logger.info(f"Running reference model...")
        ref_input_dtype = get_ref_model_dype(reference_model, model_args_ref.model_name)
        ref_output = reference_model(pt_prefill_input.to(torch.bfloat16), start_pos)
        ref_output = ref_output[:, -1:, :]  # Get last token since TT model only returns the last token
        logger.info(f"Finished running reference model.")

        # Measure PCC if also running reference model
        all_tests_pass = True

        # Check output pcc
        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, expec_out_pcc)
        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"Output PCC: {pcc_message}")
        if not passing:
            all_tests_pass = False
            logger.warning(f"Output PCC {pcc_message} is lower than {expec_out_pcc}")

        if passing:
            logger.info("Qwen Model Prefill TTT Passed!")
        else:
            logger.warning("Qwen Model Prefill TTT Failed!")

        # Compare KV caches
        if cache_pcc:
            for i in range(model_args.n_layers):
                pytorch_layer_present = [
                    reference_model.layers[i]
                    .attention.cache_k.clone()
                    .permute(0, 2, 1, 3),  # [batch_size, n_kv_heads, seq, head_dim]
                    reference_model.layers[i]
                    .attention.cache_v.clone()
                    .permute(0, 2, 1, 3),  # [batch_size, n_kv_heads, seq, head_dim]
                ]

                tt_layer_present = []
                if paged_attention:
                    for layer_past in tt_model.layers[i].attention.layer_past:
                        tt_layer_present.append(
                            ttnn.to_torch(
                                layer_past,
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
                else:
                    for layer_past in tt_model.layers[i].attention.layer_past:
                        tt_layer_present.append(
                            ttnn.to_torch(
                                layer_past,
                                mesh_composer=ttnn.ConcatMesh2dToTensor(
                                    mesh_device,
                                    dims=(1, 0) if model_args.is_galaxy else (0, 1),
                                    mesh_shape=model_args.cluster_shape,
                                ),
                            )
                        )

                for j, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                    cache_length_to_check = seq_len
                    cache_pt = cache_pt[:, :, 0:cache_length_to_check, :]
                    cache_tt = cache_tt[:, :, 0:cache_length_to_check, :]
                    pcc_passed, output_pcc = comp_pcc(cache_pt, cache_tt, expec_kv_cache_pcc)
                    kv_str = "K" if j == 0 else "V"
                    logger.info(f"[layer={i+1}] {kv_str} cache PCC: {output_pcc}")
                    if not pcc_passed:
                        all_tests_pass = False
                        logger.warning(f"[layer={i+1}] {kv_str} PCC {output_pcc} is lower than {expec_kv_cache_pcc}")

        if all_tests_pass:
            logger.info("All PCC checks passed!")
        else:
            assert all_tests_pass, f"PCC is lower than expected for some of the outputs. Check warnings!"
