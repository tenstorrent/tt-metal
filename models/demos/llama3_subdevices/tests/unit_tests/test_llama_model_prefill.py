# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import bz2
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3_subdevices.tt.llama_common import (
    get_prefill_rot_mat,
    HostEmbedding,
    encode_prompt_llama_instruct,
    PagedAttentionConfig,
)
from models.demos.llama3_subdevices.tt.llama_model import TtTransformer
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs, LlamaOptimizations
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Transformer
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
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
    (128,),
)
@pytest.mark.parametrize(
    "optimizations",
    [
        pytest.param(LlamaOptimizations.accuracy, id="accuracy"),
        # pytest.param(LlamaOptimizations.performance, id="performance"),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
def test_llama_model_inference(
    seq_len,
    paged_attention,
    page_params,
    optimizations,
    mesh_device,
    reset_seeds,
    ensure_gc,
    is_ci_env,
):
    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
    cache_pcc = True  # Flag to measure KV cache PCC for all layers

    dtype = ttnn.bfloat8_b
    batch_size = 1  # For prefill we only support batch_size = 1

    # This sets the minimum PCC for each iteration based on optimization mode
    if optimizations == LlamaOptimizations.accuracy:
        pcc = 0.917  # TODO Look on improving PCC
    else:  # performance mode
        assert optimizations == LlamaOptimizations.performance
        pcc = 0.917  # TODO Look on improving PCC

    # Use instruct weights instead of general weights
    instruct = True

    model_args = TtModelArgs(
        mesh_device, max_batch_size=batch_size, optimizations=optimizations, max_seq_len=seq_len, dummy_weights=True
    )
    model_args.use_prefetcher = False
    model_args.n_layers = 1
    tokenizer = Tokenizer(model_args.tokenizer_path)

    logger.info("Loading weights...")
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    state_dict = model_args.load_state_dict()
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
    logger.info("Finished loading weights...")

    current_file_path = os.path.abspath(__file__)
    current_file_dir = os.path.dirname(current_file_path)
    prompt_file = os.path.join(current_file_dir, "tale-of-two-cities.txt.bz2")

    with bz2.open(prompt_file, "rt", encoding="utf-8") as f:
        prompt = f.read()

    if instruct:
        encoded_prompt = encode_prompt_llama_instruct(tokenizer, prompt)[:seq_len]
    else:
        encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False)[:seq_len]

    if run_ref_pt:
        reference_model = Transformer(model_args)
        reference_model.load_state_dict(reference_state_dict)
    # Embedding on host
    embd = HostEmbedding(model_args)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    # pre-compute the rotational embedding matrix and send to device
    rot_mats = get_prefill_rot_mat(
        model_args.head_dim,
        model_args.max_seq_len,
        mesh_device,
        seq_len=seq_len,
        scale_factor=model_args.rope_scaling_factor,
    )
    # Setup page table
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
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # Load TTNN model
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

    logger.info("Model and caches loaded.")

    if run_ref_pt:
        all_tests_pass = True

    # Select the first token from the prompt for initial decoding
    encoded_prompt_tensor = torch.tensor(encoded_prompt)  # [:,0]
    pt_prefill_input = embd(encoded_prompt_tensor).view(batch_size, seq_len, -1)

    tt_prefill_input = pt_prefill_input

    tt_prefill_input = model_args.prepare_residual_tensor_prefill(
        pt_prefill_input,
    )
    for i in range(1):
        start_pos = 0
        # Run TT model
        tt_out = tt_model(
            tt_prefill_input,
            current_pos=None,
            rot_mats=rot_mats,
            user_id=i,
            mode="prefill",
            page_table=page_table_tt,
        )
        # Convert ttnn tensor to torch tensor
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=(1, 3) if model_args.is_galaxy else (1, 3), mesh_shape=model_args.cluster_shape
            ),
        )
        tt_output_torch = tt_out[:, 0:1, :, : model_args.dim].view(batch_size, seq_len, -1)  # [ batch, seq, hidden_dim]

        if run_ref_pt:  # Run reference model
            ref_output = reference_model(pt_prefill_input, start_pos, mode="prefill")

        # Measure PCC if also running reference model
        if run_ref_pt:
            passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

            logger.info(comp_allclose(ref_output, tt_output_torch))
            logger.info(f"PCC: {pcc_message}")

            if passing:
                logger.info("Llama Model Passed!")
            else:
                logger.warning("Llama Model Failed!")
            if not passing:
                all_tests_pass = False

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
                        tt_layer_present = [
                            (
                                ttnn.to_torch(
                                    cache,
                                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                                        mesh_device,
                                        dims=(1, 0) if model_args.is_galaxy else (0, 1),
                                        mesh_shape=model_args.cluster_shape,
                                    ),
                                )[reverse_permutation]
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
                            for cache in tt_model.layers[i].attention.layer_past
                        ]
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

                    for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                        cache_length_to_check = model_args.max_seq_len
                        cache_pt = cache_pt[0, :, 0:cache_length_to_check, :]
                        cache_tt = cache_tt[0, :, 0:cache_length_to_check, :]
                        does_pass, output_pcc = comp_pcc(cache_pt, cache_tt)
                        if i == 0:
                            logger.info(f"K cache output: {output_pcc}")
                        else:
                            logger.info(f"V cache output: {output_pcc}")

                        if does_pass:
                            logger.info(f"V Cache Passed!")
                        else:
                            logger.warning(f"V Cache Failed! PCC value is lower than {0.99}")
                        # if not does_pass:
                        # all_tests_pass = False

    if run_ref_pt:
        if all_tests_pass:
            logger.info(f"All Llama prefill iterations Passed!")
        else:
            logger.warning("One or more iterations of Llama prefill had bad PCC")
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
