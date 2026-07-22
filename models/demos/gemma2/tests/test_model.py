# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
#
# Gemma-2 decode accuracy (PCC) test. Adapted from
# models/tt_transformers/tests/test_model.py with two Gemma-2 changes:
#   1. Build the dedicated Gemma-2 ``ModelArgs`` subclass (adds RMSNorm unit
#      offset + sqrt(hidden) embed scale).
#   2. Provide ``rot_mats_local`` in the decode loop so interleaved
#      sliding-window layers get their local rope (the generic test only passes
#      the global rope, which crashes for interleaved-attention models).
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.gemma2.tt.model_config import ModelArgs
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig, sample_host
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision


@torch.no_grad()
@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("use_prefetcher", ([False]))
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
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
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
    use_prefetcher,
):
    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
    dtype = ttnn.bfloat8_b

    use_hf_rope = False  # Gemma-2 uses the default (non-HF) rope path
    test_id = request.node.callspec.id
    mode_accuracy = "accuracy" in test_id
    instruct = False
    dummy_weights = True if weights == "random" else False

    # Flag to measure KV cache PCC. Avoid running for all layers to speed up test time.
    cache_pcc = layers == 1 and not dummy_weights

    prefetcher = None

    model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        dummy_weights=dummy_weights,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        cache_hf=True,
        prefetcher=prefetcher,
        use_hf_rope=use_hf_rope,
    )

    # Define minimum PCC for each iteration
    if layers == 1:
        pcc = 0.88 if mode_accuracy else 0.86
    else:
        pcc = 0.94 if mode_accuracy else 0.86

    model_name = model_args.base_model_name

    if layers == 1:  # quick mode has tight PCC checks for known models
        final_model_pcc = {}.get(model_name, 0.88 if mode_accuracy else 0.86)
        final_k_cache_pcc = {}.get(model_name, 0.9995)
        final_v_cache_pcc = {}.get(model_name, 0.9995)
        iterations = 6
    else:
        iterations = 9

    if layers is not None:
        model_args.n_layers = layers
    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    reference_state_dict = None
    if dummy_weights:
        reference_state_dict = {
            k[len(state_dict_prefix) :]: v
            for k, v in state_dict.items()
            if (
                any([f"{state_dict_prefix}layers.{i}." in k for i in range(model_args.n_layers)])
                or any(
                    [
                        f"{state_dict_prefix}{name}" in k
                        for name in [
                            "tok_embeddings.weight",
                            "learnable_embedding.weight",
                            "norm.weight",
                            "output.weight",
                        ]
                    ]
                )
            )
        }

    prompts = ["This is a test"] * model_args.max_batch_size
    if dummy_weights:
        encoded_prompts = [[128000, 2028, 374, 264, 1296]] * model_args.max_batch_size
        assert not instruct, "Instruct prompt not implemented with dummy weights"
    else:
        tokenizer = model_args.tokenizer
        encoded_prompts = [model_args.encode_prompt(prompt, instruct=False) for prompt in prompts]

    reference_model = None
    if run_ref_pt:
        reference_model = model_args.reference_transformer(load_checkpoint=not dummy_weights)
        if dummy_weights:
            reference_model.load_state_dict(reference_state_dict)

    # Embedding on host
    embd = model_args.reference_embedding(reference_model)
    weight = state_dict[f"{state_dict_prefix}tok_embeddings.weight"]
    embd.load_state_dict({"emb.weight": weight})

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
        prefetcher=None,
    )

    logger.info("Model and caches loaded.")

    if run_ref_pt:
        all_tests_pass = True
        final_tests_pass = True
        kv_cache_tests_pass = True

    seqlen = 1  # Generating one token per user at a time
    batch = model_args.max_batch_size

    encoded_prompts_tensor = torch.tensor(encoded_prompts)
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
    tt_decode_input = pt_decode_input

    all_outputs = []
    if run_ref_pt:
        all_outputs_ref = []

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
            model_args.get_residual_mem_config(Mode.DECODE, prefetcher),
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = tt_model.rope_setup.get_rot_mats(current_pos, prefetcher)
        # Gemma-2 interleaves sliding-window layers, which use a separate local rope.
        rot_mats_local = (
            tt_model.rope_local_setup.get_rot_mats(current_pos, prefetcher)
            if hasattr(tt_model, "rope_local_setup")
            else None
        )

        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats_global=rot_mats,
            rot_mats_local=rot_mats_local,
            mode=Mode.DECODE,
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
            all_outputs.append(encoded_prompts[0][i])
            if run_ref_pt:
                all_outputs_ref.append(encoded_prompts[0][i])
            tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
            if run_ref_pt:
                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            if run_ref_pt:
                _, pt_out_tok = sample_host(ref_output, temperature=0, top_p=0.8)
                pt_decode_input = embd(pt_out_tok)
                all_outputs_ref.append(pt_out_tok.squeeze(1).tolist()[0])
                tt_decode_input = pt_decode_input
                all_outputs.append(pt_out_tok.squeeze(1).tolist()[0])
            else:
                _, tt_out_tok = sample_host(tt_output_torch, temperature=0, top_p=0.8)
                tt_decode_input = embd(tt_out_tok)
                all_outputs.append(tt_out_tok.squeeze(1).tolist()[0])

        # Measure PCC if also running reference model
        if run_ref_pt:
            if layers == 1 and i == iterations - 1:
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
                assert (
                    final_tests_pass
                ), f"PCC value {pcc_message} is lower than {final_model_pcc} for final output. Check Warnings!"
            assert (
                all_tests_pass
            ), f"PCC value {pcc_message} is lower than {pcc} for some of the outputs. Check Warnings!"
