# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
import os
from models.demos.llama3_subdevices.tt.llama_common import (
    sample_host,
    HostEmbedding,
    PagedAttentionConfig,
)
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs, LlamaOptimizations
from models.demos.llama3_subdevices.tt.llama_model import TtTransformer
from models.demos.llama3_subdevices.tt.sampling import TTSampling
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Transformer
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull

is_RING_6U = os.environ.get("RING_6U", "0") == "1"


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.timeout(1800)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "weights, layers, iterations",
    [
        ("random", 1, 6),
        ("instruct", 80, 9),
    ],
    ids=["quick", "full"],
)
@pytest.mark.parametrize(
    "sampling_params",
    [{"top_k": 1, "top_p": 0.00, "seed": 42}],
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        # True,
        False,
    ),
    ids=(
        # "paged_attention",
        "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 4096}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize(
    "optimizations",
    [
        pytest.param(LlamaOptimizations.accuracy, id="accuracy"),
        # pytest.param(LlamaOptimizations.performance, id="performance"),
    ],
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
            "worker_l1_size": 1344544,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING if is_RING_6U else ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_llama_model_inference(
    weights,
    layers,
    iterations,
    sampling_params,
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    optimizations,
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
):
    generate_ref_pt_cache = True  # Flag to run reference PyTorch model and save cached outputs to disk
    generate_tt_model_logits_cache = False
    use_cached_ref_pt = False  # Flag to use cached reference outputs from disk
    ref_pt_cache_path = "models/demos/llama3_subdevices/tests/ref_outputs/test_llama_model/test_llama_model_ref_output_Llama3.3-70B-Instruct"
    compare_ref_pt = use_cached_ref_pt or generate_ref_pt_cache
    cache_pcc = layers == 1  # Flag to measure KV cache PCC. Avoid running for all layers to speed up test time.
    dtype = ttnn.bfloat8_b

    mode_accuracy = optimizations == LlamaOptimizations.accuracy
    instruct = True if weights == "instruct" else False
    dummy_weights = True if weights == "random" else False
    model_args = TtModelArgs(
        mesh_device,
        instruct=instruct,
        dummy_weights=dummy_weights,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )

    model_name = {
        (80, False): "llama31_70b",
    }[(model_args.n_layers, model_args.is_vision())]

    # Define minimum PCC for each iteration
    if layers == 1:
        pcc = 0.921936
    else:
        pcc = 0.93

    # Define tight final PCC thresholds for quick mode
    final_model_pcc = {"llama31_70b": 0.921936}[model_name]

    final_k_cache_pcc = {
        "llama31_70b": 0.9997,
    }[model_name]
    final_v_cache_pcc = {
        "llama31_70b": 0.9997,
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

    prompts = ["Life is "] * model_args.max_batch_size
    if dummy_weights:
        encoded_prompts = [
            [128000, 2028, 374, 264, 1296]
        ] * model_args.max_batch_size  # "This is a test" encoded prompt
        assert not instruct, "Instruct prompt not implemented with dummy weights"
    else:
        tokenizer = Tokenizer(model_args.tokenizer_path)
        # if instruct:
        #     encoded_prompts = [encode_prompt_llama_instruct(tokenizer, prompt) for prompt in prompts]
        # else:
        encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]

    if generate_ref_pt_cache:
        reference_model = Transformer(model_args)
        reference_model.load_state_dict(reference_state_dict)

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
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # Load TTNN model
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    tt_sampling = TTSampling(
        args=model_args,
        mesh_device=mesh_device,
        sampling_params=sampling_params,
        tt_ccl=tt_model.tt_ccl,
    )
    logger.info("Model and caches loaded.")

    if compare_ref_pt:
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
    if compare_ref_pt:
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
    # Get cos/sin matrices for the current position of each user
    rot_mats, rot_mat_idxs = tt_model.rope_setup.get_rot_mats(current_pos, return_rot_idxs=True)

    all_pccs = []

    try:
        for i in range(generation_length):
            logger.info(f"[Llama3 Model] Generating token {i}")
            decode_input = model_args.prepare_residual_tensor_decode(
                tt_decode_input,
                model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
            )

            # Get cos/sin matrices for the current position of each user
            rot_mats = tt_model.rope_setup.get_rm_rot_mats(current_pos)

            # Run TT model
            tt_out = tt_model(
                decode_input,
                current_pos_tensor,
                rot_mats=rot_mats,
                mode="decode",
                page_table=page_table_tt,
            )
            # Sampling
            tt_out_tok = tt_sampling(tt_out[0])

            if generate_tt_model_logits_cache:
                tt_model_output_cache_path = f"models/demos/llama3_subdevices/tests/ref_outputs/test_llama_model/tt_model_layers_{layers}_output_logits_tok_{i}.bin"
                ttnn.dump_tensor(file_name=tt_model_output_cache_path, tensor=tt_out[0])

            # Convert ttnn tensor to torch tensor
            mesh_composer = ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=(3, 1) if model_args.is_galaxy else (1, -1), mesh_shape=model_args.cluster_shape
            )

            outs = [ttnn.to_torch(out, mesh_composer=mesh_composer) for out in tt_out]
            # logger.info(f"TT output shape: {outs[0].shape}")
            outs = torch.concat(outs, dim=-1)
            # for t in tt_out:
            #     t.deallocate(True)
            tt_output_torch = outs.permute(2, 1, 0, 3).squeeze(2)[
                : model_args.max_batch_size, 0:1, : model_args.vocab_size
            ]

            k_cache = None
            v_cache = None
            if generate_ref_pt_cache:  # Run reference model
                # In this test all users have the same position
                ref_output = reference_model(pt_decode_input, current_pos[0])
                # Save reference output to file
                with open(ref_pt_cache_path + f"_layers_{layers}_" + f"_tok_{i}.pt", "wb") as f:
                    torch.save(ref_output, f)
                if cache_pcc:
                    for l in range(model_args.n_layers):
                        k_cache = reference_model.layers[l].attention.cache_k
                        with open(ref_pt_cache_path + f"_k_cache_" + f"_layer_{l}_" + f"_tok_{i}.pt", "wb") as f:
                            torch.save(k_cache, f)
                        v_cache = reference_model.layers[l].attention.cache_v
                        with open(ref_pt_cache_path + f"_v_cache_" + f"_layer_{l}_" + f"_tok_{i}.pt", "wb") as f:
                            torch.save(v_cache, f)
            elif use_cached_ref_pt:
                with open(ref_pt_cache_path + f"_layers_{layers}_" + f"_tok_{i}.pt", "rb") as f:
                    ref_output = torch.load(f)
                if cache_pcc:
                    for l in range(model_args.n_layers):
                        with open(ref_pt_cache_path + f"_k_cache_" + f"_layer_{l}_" + f"_tok_{i}.pt", "rb") as f:
                            k_cache = torch.load(f)
                        with open(ref_pt_cache_path + f"_v_cache_" + f"_layer_{l}_" + f"_tok_{i}.pt", "rb") as f:
                            v_cache = torch.load(f)

            # Measure PCC between TT model logits and reference model logits
            if compare_ref_pt:
                if layers == 1 and i == iterations - 1:  # On last iteration in the quick test, set a tighter PCC
                    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, final_model_pcc)
                    if not passing:
                        final_tests_pass = False
                else:
                    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)

                logger.info(pcc_message)
                logger.info(f"PCC: {pcc_message}")
                all_pccs.append(pcc_message)
                print("All PCCs: ", all_pccs)

                if passing:
                    logger.info("Llama Model Passed!")
                else:
                    logger.warning("Llama Model Failed!")
                if not passing:
                    all_tests_pass = False

                # Compare KV caches
                if cache_pcc:
                    for l in range(model_args.n_layers):
                        pytorch_layer_present = [
                            k_cache.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                            v_cache.clone().permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
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
                            cache_length_to_check = min(
                                model_args.max_seq_len, generation_start_pos + generation_length + 1
                            )
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
                if compare_ref_pt:
                    all_outputs_ref.append(encoded_prompts[0][i])  # Update list of ref outputs

                tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
                if compare_ref_pt:
                    pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
            else:
                tt_out_tok_device0 = ttnn.get_device_tensors(tt_out_tok)[0]
                tt_out_tok_cpu = tt_out_tok_device0.cpu(blocking=True, cq_id=0)
                tt_out_tok = ttnn.to_torch(
                    tt_out_tok_cpu,
                ).view(32, 1)

                all_outputs.append(tt_out_tok.squeeze(1).tolist()[0])  # Update generated token to list of TT outputs
                tt_decode_input = embd(tt_out_tok)

                if compare_ref_pt:
                    pt_out_tok = sample_host(ref_output, None, temperature=0, top_p=sampling_params["top_p"])
                    pt_decode_input = embd(pt_out_tok)
                    all_outputs_ref.append(
                        pt_out_tok.squeeze(1).tolist()[0]
                    )  # Update generated token to list of ref outputs
                    passing, pcc_message = comp_allclose(pt_out_tok, tt_out_tok)
                    logger.info(f"PT out tok: {pt_out_tok}")
                    logger.info(f"TT out tok: {tt_out_tok}")
                    logger.info(pcc_message)
                    if not passing:
                        logger.warning(f"Sampling failed!")
                        all_tests_pass = False

            if not dummy_weights:
                logger.info("[ttnn generation User 0] " + tokenizer.decode(all_outputs).replace("\n", "\\n"))
                if compare_ref_pt:
                    logger.info("[Ref generation User 0] " + tokenizer.decode(all_outputs_ref).replace("\n", "\\n"))
    finally:
        pass

    if compare_ref_pt:
        if all_tests_pass:
            logger.info(f"All {generation_length} Llama decode iterations Passed!")
        else:
            logger.warning("One or more iterations of Llama decode had bad PCC")
            assert final_tests_pass, f"PCC value is lower than {final_model_pcc} for final output. Check Warnings!"
            assert kv_cache_tests_pass, f"KV Cache PCC value is lower expected for some of the outputs. Check Warnings!"
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
