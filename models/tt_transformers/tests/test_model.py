# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig, sample_host
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import CheckpointType, DecodersPrecision, ModelArgs
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "weights, layers",
    [
        ("random", 1),
        ("instruct", None),
    ],
    ids=["quick", "full"],
)
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
@pytest.mark.parametrize(
    "optimizations",
    [
        lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
        lambda model_args: DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name),
    ],
    ids=["performance", "accuracy"],
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
def test_model_inference(
    weights,
    layers,
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    optimizations,
    mesh_device,
    reset_seeds,
    ensure_gc,
    request,
):
    model_name_env = os.getenv("HF_MODEL")
    if model_name_env and "Mistral-7B" in model_name_env and weights == "instruct":
        pytest.skip(
            "Skipping Mistral-7B full model test for now. See issue https://github.com/tenstorrent/tt-metal/issues/19806"
        )

    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
    dtype = ttnn.bfloat8_b

    test_id = request.node.callspec.id
    mode_accuracy = "accuracy" in test_id
    instruct = False  # True if weights == "instruct" else False
    dummy_weights = True if weights == "random" else False

    # Flag to measure KV cache PCC. Avoid running for all layers to speed up test time.
    # Also avoid comparing PCC for dummy weights
    cache_pcc = layers == 1 and not dummy_weights

    model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        dummy_weights=dummy_weights,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )

    # Define minimum PCC for each iteration
    if layers == 1:
        pcc = 0.88 if mode_accuracy else 0.86
    else:
        pcc = 0.94 if mode_accuracy else 0.86

    model_name = model_args.base_model_name
    if layers == 1:  # quick mode has tight PCC checks for known models
        if model_args.checkpoint_type == CheckpointType.HuggingFace:
            model_name = model_args.base_model_name
        else:
            model_name = {
                (16, False): "llama32_1b",
                (28, False): "llama32_3b",
                (32, False): "llama31_8b",
                (32, True): "llama32_11b",
                (80, False): "llama31_70b",
                (80, True): "llama32_90b",
            }[(model_args.n_layers, model_args.is_vision())]

        # Define tight final PCC thresholds for quick mode
        final_model_pcc = {
            "llama32_1b": 0.9991 if mode_accuracy else 0.9864,
            "llama32_3b": 0.9989 if mode_accuracy else 0.9837,
            "llama31_8b": 0.9987 if mode_accuracy else 0.9850,
            "llama32_11b": 0.9987 if mode_accuracy else 0.9850,
            "llama31_70b": 0.9843 if mode_accuracy else 0.97607,
            "llama32_90b": 0.9759,
            "Mistral-7B": 0.95 if mode_accuracy else 0.95,
        }[model_name]

        final_k_cache_pcc = {
            "llama32_1b": 0.9998,
            "llama32_3b": 0.9998,
            "llama31_8b": 0.9997,
            "llama32_11b": 0.9995,
            "llama31_70b": 0.9997,
            "llama32_90b": 0.9995,
            "Mistral-7B": 0.68,
        }[model_name]
        final_v_cache_pcc = {
            "llama32_1b": 0.9996,
            "llama32_3b": 0.9998,
            "llama31_8b": 0.9997,
            "llama32_11b": 0.9996,
            "llama31_70b": 0.9997,
            "llama32_90b": 0.9996,
            "Mistral-7B": 0.68,
        }[model_name]

        quick_iterations = {
            "llama32_1b": 2,
            "llama32_3b": 4,
            "llama31_8b": 6,
            "llama32_11b": 6,
            "llama31_70b": 6,
            "llama32_90b": 6,
            "Mistral-7B": 2,
        }[model_name]

        iterations = quick_iterations
    else:
        iterations = 9

    if layers is not None:
        model_args.n_layers = layers
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

    prompts = ["This is a test"] * model_args.max_batch_size
    if dummy_weights:
        # "This is a test" encoded prompt
        if model_name == "Mistral-7B":
            encoded_prompts = [[1619, 1117, 1032, 2137]] * model_args.max_batch_size
        else:
            encoded_prompts = [[128000, 2028, 374, 264, 1296]] * model_args.max_batch_size
        assert not instruct, "Instruct prompt not implemented with dummy weights"
    else:
        tokenizer = model_args.tokenizer
        if instruct:
            encoded_prompts = [model_args.encode_prompt(prompt) for prompt in prompts]
        else:
            encoded_prompts = [model_args.encode_prompt(prompt, instruct=False) for prompt in prompts]

    reference_model = None
    if run_ref_pt:
        reference_model = model_args.reference_transformer()
        reference_model.load_state_dict(reference_state_dict)

    # Embedding on host
    embd = model_args.reference_embedding(reference_model)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    generation_start_pos = 0
    generation_length = iterations

    page_table_tt = None
    paged_attention_config = None

    # Prepare page table for paged attention
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
                dims=(None, -2) if batch_size > 1 else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # Load TTNN model
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    logger.info("Model and caches loaded.")

    if run_ref_pt:
        all_tests_pass = True
        final_tests_pass = True
        kv_cache_tests_pass = True

    seqlen = 1  # Generating one token per user at a time
    batch = model_args.max_batch_size

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
    tt_decode_input = pt_decode_input

    # Keep track of generated outputs to print out later
    all_outputs = []
    if run_ref_pt:
        all_outputs_ref = []

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch)])
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
        logger.info(f"[Model] Generating token {i}")

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)

        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        # Convert ttnn tensor to torch tensor
        mesh_composer = ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=(3, 1) if model_args.is_galaxy else (1, -1), mesh_shape=model_args.cluster_shape
        )
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=mesh_composer)
            .permute(2, 1, 0, 3)
            .squeeze(2)[: model_args.max_batch_size, 0:1, : model_args.vocab_size]
        )

        ttnn.deallocate(tt_out)

        if run_ref_pt:  # Run reference model
            # In this test all users have the same position
            ref_output = reference_model(pt_decode_input, current_pos[0])

        # Increment position
        current_pos = torch.tensor([generation_start_pos + i for _ in range(batch)])
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

        # Append the generated token to the list of outputs
        if i in range(len(encoded_prompts[0])):
            # While in "prefill" mode, use the prompt tokens as the output
            all_outputs.append(encoded_prompts[0][i])  # Update list of TT outputs
            if run_ref_pt:
                all_outputs_ref.append(encoded_prompts[0][i])  # Update list of ref outputs

            tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
            if run_ref_pt:
                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            # Greedy decode (temperature = 0) the generated token and save it to print out later
            if run_ref_pt:
                # Sample from reference model first
                _, pt_out_tok = sample_host(ref_output, temperature=0, top_p=0.8)
                pt_decode_input = embd(pt_out_tok)
                all_outputs_ref.append(pt_out_tok.squeeze(1).tolist()[0])

                # Use the same token for TT model (teacher forcing)
                tt_decode_input = pt_decode_input
                all_outputs.append(pt_out_tok.squeeze(1).tolist()[0])
            else:
                # If not running reference model, sample from TT model directly
                _, tt_out_tok = sample_host(tt_output_torch, temperature=0, top_p=0.8)
                tt_decode_input = embd(tt_out_tok)
                all_outputs.append(tt_out_tok.squeeze(1).tolist()[0])

        # Measure PCC if also running reference model
        if run_ref_pt:
            if layers == 1 and i == iterations - 1:  # On last iteration in the quick test, set a tighter PCC
                passing, pcc_message = comp_pcc(ref_output, tt_output_torch, final_model_pcc)
                if not passing:
                    final_tests_pass = False
            else:
                passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

            logger.info(comp_allclose(ref_output, tt_output_torch))
            logger.info(f"PCC: {pcc_message}")

            if passing:
                logger.info("Model Passed!")
            else:
                logger.warning("Model Failed!")
            if not passing:
                all_tests_pass = False

            # Compare KV caches
            if cache_pcc:
                for l in range(model_args.n_layers):
                    if model_args.checkpoint_type == CheckpointType.HuggingFace:
                        pytorch_layer_present = [
                            reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                            reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                        ]
                    else:
                        pytorch_layer_present = [
                            reference_model.layers[l]
                            .attention.cache_k.clone()
                            .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                            reference_model.layers[l]
                            .attention.cache_v.clone()
                            .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                        ]
                    tt_layer_present = []
                    if paged_attention:
                        for layer_past in tt_model.layers[l].attention.layer_past:
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
                                    :batch, ...
                                ]
                            )
                    else:
                        for layer_past in tt_model.layers[l].attention.layer_past:
                            tt_layer_present.append(
                                ttnn.to_torch(
                                    layer_past,
                                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                                        mesh_device,
                                        dims=(1, 0) if model_args.is_galaxy else (0, 1),
                                        mesh_shape=model_args.cluster_shape,
                                    ),
                                )[:batch, :, :, :]
                            )

                    for kv_cache, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                        cache_length_to_check = min(model_args.max_seq_len, generation_start_pos + i + 1)
                        cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
                        cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
                        if (
                            layers == 1 and i == iterations - 1
                        ):  # On last iteration in the quick test, set a tighter PCC
                            if kv_cache == 0:  # K cache
                                does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, final_k_cache_pcc)
                            else:  # V cache
                                does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, final_v_cache_pcc)
                        else:
                            does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
                        if kv_cache == 0:
                            logger.info(f"K cache output: {output_pcc}")
                        else:
                            logger.info(f"V cache output: {output_pcc}")

                        if does_pass:
                            logger.info(f"KV Cache Passed!")
                        else:
                            logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
                            all_tests_pass = False

        if not dummy_weights:
            logger.info("[ttnn generation User 0] " + tokenizer.decode(all_outputs).replace("\n", "\\n"))
            if run_ref_pt:
                logger.info("[Ref generation User 0] " + tokenizer.decode(all_outputs_ref).replace("\n", "\\n"))

    if run_ref_pt:
        if all_tests_pass:
            logger.info(f"All {generation_length} decode iterations Passed!")
        else:
            logger.warning("One or more iterations of decode had bad PCC")
            if layers == 1:
                assert final_tests_pass, f"PCC value is lower than {final_model_pcc} for final output. Check Warnings!"
            assert kv_cache_tests_pass, f"KV Cache PCC value is lower expected for some of the outputs. Check Warnings!"
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
