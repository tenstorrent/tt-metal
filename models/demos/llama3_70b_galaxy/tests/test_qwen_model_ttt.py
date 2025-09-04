# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_common import (
    sample_host,
    HostEmbedding,
    PagedAttentionConfig,
)
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_embedding import TtLlamaEmbedding
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.demos.llama3_70b_galaxy.tt.sampling import TTSampling
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from transformers import AutoTokenizer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "weights, layers, iterations",
    [
        ("instruct", 3, 2000),
    ],
    ids=["quick"],
)
@pytest.mark.parametrize(
    "sampling_params",
    [
        {"top_k": 1, "top_p": 0.00, "temperature": 1.0, "seed": 42},
        # {"top_k": 32, "top_p": 0.9, "temperature": 0.7, "seed": 42}
    ],
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
    [{"page_block_size": 64, "page_max_num_blocks": 256}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (2048,),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
def test_qwen_model_ttt_inference(
    weights,
    layers,
    iterations,
    sampling_params,
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
    cache_pcc = layers == 2  # Flag to measure KV cache PCC. Avoid running for all layers to speed up test time.
    dtype = ttnn.bfloat8_b

    top_k = sampling_params["top_k"]
    if isinstance(top_k, int):
        top_k = [top_k] * batch_size
    top_p = sampling_params["top_p"]
    if isinstance(top_p, float):
        top_p = [top_p] * batch_size
    temperature = sampling_params["temperature"]
    if isinstance(temperature, float):
        temperature = [temperature] * batch_size
    seed = sampling_params["seed"]

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

    model_name = {
        (80): "qwen3_70b",
        (3): "qwen3_1layer",
    }[(layers if layers is not None else model_args.n_layers)]

    # Define minimum PCC for each iteration
    if layers == 1:
        pcc = 0.921936
    else:
        pcc = 0.94

    # Define tight final PCC thresholds for quick mode
    final_model_pcc = {"qwen3_70b": 0.921936, "qwen3_1layer": 0.921936}[model_name]

    final_k_cache_pcc = {
        "qwen3_70b": 0.9997,
        "qwen3_1layer": 0.9997,
    }[model_name]
    final_v_cache_pcc = {
        "qwen3_70b": 0.9997,
        "qwen3_1layer": 0.9997,
    }[model_name]

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

    # Load tt_transformers reference model args for reference transformer
    model_args_ref = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, cache_hf=True)
    model_args_ref.n_layers = model_args.n_layers

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

    prompts = ["What is your favorite condiment? "] * model_args.max_batch_size
    if dummy_weights:
        encoded_prompts = [
            [128000, 2028, 374, 264, 1296]
        ] * model_args.max_batch_size  # "This is a test" encoded prompt
        assert not instruct, "Instruct prompt not implemented with dummy weights"
    else:
        # Use Qwen tokenizer from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained(model_args.TOKENIZER_PATH)
        # if instruct:
        #     encoded_prompts = [encode_prompt_llama_instruct(tokenizer, prompt) for prompt in prompts]
        # else:
        encoded_prompts = [tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]

    if run_ref_pt:
        # Use tt_transformers reference transformer
        reference_model = model_args_ref.reference_transformer()
        reference_model.load_state_dict(reference_state_dict_ref)

    # Embedding on host
    embd = HostEmbedding(model_args)
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
        paged_cache_max_seq_len = (
            paged_attention_config.block_size
            * paged_attention_config.max_num_blocks
            / model_args.batch_size_per_device_group
        )
        is_valid_token_position = 256 + generation_start_pos <= paged_cache_max_seq_len
        assert_msg = f"max_generated_tokens ({256}) + start_pos ({generation_start_pos}) <= paged_cache_max_seq_len ({paged_cache_max_seq_len})"
        assert is_valid_token_position, assert_msg

        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.batch_size_per_device_group,
            paged_attention_config.max_num_blocks // model_args.batch_size_per_device_group,
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        )
        logger.info("Page table tensor done")

    # Load TTNN model
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
        enable_prefetcher_performance_mode=True,
    )
    tt_sampling = TTSampling(
        args=model_args,
        mesh_device=mesh_device,
        tt_ccl=tt_model.tt_ccl,
    )
    logger.info("Qwen Model and caches loaded.")

    if run_ref_pt:
        all_tests_pass = True
        final_tests_pass = True
        kv_cache_tests_pass = True

    seqlen = 1  # Generating one token per user at a time
    batch = model_args.max_batch_size

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
    # Prepare the encoded prompts for the decode input
    tt_out_tok = ttnn.from_torch(
        encoded_prompts_tensor[:, :1].reshape(1, 1, 1, batch_size),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_decode_input = tt_embd(tt_out_tok)

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
    all_pccs = []

    # Get cos/sin matrices for the current position of each user
    rot_mats, rot_mat_idxs = tt_model.rope_setup.get_rm_rot_mats(current_pos, return_rot_idxs=True)

    try:
        for i in range(generation_length):
            logger.info(f"[Qwen Model TTT] Generating token {i}")
            rot_mats = tt_model.rope_setup.get_rm_rot_mats(rot_mat_idxs)

            # Run TT model
            tt_out = tt_model(
                tt_decode_input,
                current_pos_tensor,
                rot_mats=rot_mats,
                mode="decode",
                page_table=page_table_tt,
            )

            # Sampling
            tt_out_tok = tt_sampling(tt_out[0], seed)
            tt_decode_input = tt_embd(tt_out_tok)

            # Convert ttnn tensor to torch tensor
            mesh_composer = ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=(3, 1) if model_args.is_galaxy else (1, -1), mesh_shape=model_args.cluster_shape
            )

            outs = [ttnn.to_torch(out, mesh_composer=mesh_composer) for out in tt_out]
            outs = torch.concat(outs, dim=-1)
            tt_output_torch = outs.permute(2, 1, 0, 3).squeeze(2)[
                : model_args.max_batch_size, 0:1, : model_args.vocab_size
            ]

            if run_ref_pt:  # Run reference model
                # Use tt_transformers reference model with appropriate dtype
                ref_input_dtype = get_ref_model_dype(reference_model, model_args_ref.model_name)
                ref_output = reference_model(pt_decode_input.to(torch.bfloat16), current_pos[0])
                # breakpoint()

            # Increment position
            ttnn.plus_one(
                current_pos_tensor,
                sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            )
            current_pos = torch.tensor([i + 1 for _ in range(batch)])
            ttnn.plus_one(
                rot_mat_idxs,
                sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            )

            # Append the generated token to the list of outputs
            if i in range(len(encoded_prompts[0])):
                # While in "prefill" mode, use the prompt tokens as the output
                all_outputs.append(encoded_prompts[0][i])  # Update list of TT outputs
                if run_ref_pt:
                    all_outputs_ref.append(encoded_prompts[0][i])  # Update list of ref outputs

                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
                tt_out_tok = ttnn.from_torch(
                    encoded_prompts_tensor[:, i].reshape(1, 1, 1, batch_size),
                    device=mesh_device,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape
                    ),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                tt_decode_input = tt_embd(tt_out_tok)

            else:
                tt_out_tok_device0 = ttnn.get_device_tensors(tt_out_tok)[0]
                tt_out_tok_cpu = tt_out_tok_device0.cpu(blocking=True, cq_id=0)
                tt_out_tok = ttnn.to_torch(
                    tt_out_tok_cpu,
                ).view(32, 1)

                all_outputs.append(tt_out_tok.squeeze(1).tolist()[0])  # Update generated token to list of TT outputs

                if run_ref_pt:
                    pt_out_tok = sample_host(ref_output, None, temperature=1.0, top_p=0.0)
                    pt_decode_input = embd(pt_out_tok)
                    all_outputs_ref.append(
                        pt_out_tok.squeeze(1).tolist()[0]
                    )  # Update generated token to list of ref outputs

                    # Teacher forcing
                    # tt_out_tok = ttnn.from_torch(
                    #     pt_out_tok.reshape(1, 1, 1, batch_size),
                    #     device=mesh_device,
                    #     dtype=ttnn.uint32,
                    #     layout=ttnn.ROW_MAJOR_LAYOUT,
                    #     mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
                    #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    # )
                    # tt_decode_input = tt_embd(tt_out_tok)

            # Measure PCC if also running reference model
            if run_ref_pt:
                if i == iterations - 1:  # On last iteration in the quick test, set a tighter PCC
                    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, final_model_pcc)
                    if not passing:
                        final_tests_pass = False
                else:
                    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

                logger.info(comp_allclose(ref_output, tt_output_torch))
                logger.info(f"PCC: {pcc_message}")
                all_pccs.append(pcc_message)
                print("All PCCs: ", all_pccs)

                if passing:
                    logger.info("Qwen Model TTT Passed!")
                else:
                    logger.warning("Qwen Model TTT Failed!")
                if not passing:
                    all_tests_pass = False

                # Compare KV caches
                # for l in range(model_args.n_layers):
                #     pytorch_layer_present = [
                #         reference_model.cache_k(l).clone()
                #         .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                #         reference_model.cache_v(l).clone()
                #         .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                #     ]
                #     tt_layer_present = []
                #     for layer_past_i in tt_model.layers[l].attention.layer_past:
                #         intermediate_tt = ttnn.to_torch(
                #                 layer_past_i,
                #                 mesh_composer=ttnn.ConcatMesh2dToTensor(
                #                     mesh_device,
                #                     dims=(1, 3) if model_args.is_galaxy else (0, 1),
                #                     mesh_shape=model_args.cluster_shape,
                #                 ),
                #             )
                #         tt_layer_present.append(
                #             intermediate_tt[reverse_permutation][:, : model_args.n_kv_heads, :, : model_args.head_dim]
                #             .reshape(
                #                 model_args.max_batch_size,
                #                 paged_attention_config.max_num_blocks // model_args.max_batch_size,
                #                 model_args.n_kv_heads,
                #                 paged_attention_config.block_size,
                #                 model_args.head_dim,
                #             )
                #             .transpose(1, 2)
                #             .reshape(model_args.max_batch_size, model_args.n_kv_heads, -1, model_args.head_dim)[
                #                 :batch_size, ...
                #             ]
                #         )

                #     for kv_cache, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                #         cache_length_to_check = min(
                #             model_args.max_seq_len, generation_start_pos + generation_length + 1
                #         )
                #         # cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
                #         # cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
                #         cache_pt = cache_pt[:, :, i, :]
                #         cache_tt = cache_tt[:, :, i, :]
                #         does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
                #         if kv_cache == 0:
                #             logger.info(f"K cache output: {output_pcc}")
                #         else:
                #             logger.info(f"V cache output: {output_pcc}")

                #         if does_pass:
                #             logger.info(f"KV Cache Passed!")
                #         else:
                #             logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
                #             kv_cache_tests_pass = False

            if not dummy_weights:
                logger.info(
                    "[ttnn generation User 0] "
                    + tokenizer.decode(all_outputs, skip_special_tokens=True).replace("\n", "\\n")
                )
                if run_ref_pt:
                    logger.info(
                        "[Ref generation User 0] "
                        + tokenizer.decode(all_outputs_ref, skip_special_tokens=True).replace("\n", "\\n")
                    )
    finally:
        tt_model.tt_ccl.close()

    if run_ref_pt:
        if all_tests_pass:
            logger.info(f"All {generation_length} Qwen TTT decode iterations Passed!")
        else:
            logger.warning("One or more iterations of Qwen TTT decode had bad PCC")
            assert final_tests_pass, f"PCC value is lower than {final_model_pcc} for final output. Check Warnings!"
            assert kv_cache_tests_pass, f"KV Cache PCC value is lower expected for some of the outputs. Check Warnings!"
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
