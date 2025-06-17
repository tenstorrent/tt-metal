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
        ("instruct", 80, 5),
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
    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
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
        pcc = 0.94

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

    if run_ref_pt:
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
                tt_out_tok_device0 = ttnn.get_device_tensors(tt_out_tok)[0]
                tt_out_tok_cpu = tt_out_tok_device0.cpu(blocking=True, cq_id=0)
                tt_out_tok = ttnn.to_torch(
                    tt_out_tok_cpu,
                ).view(32, 1)

                all_outputs.append(tt_out_tok.squeeze(1).tolist()[0])  # Update generated token to list of TT outputs
                tt_decode_input = embd(tt_out_tok)

                if run_ref_pt:
                    pt_out_tok = sample_host(ref_output, None, temperature=0, top_p=0.8)
                    pt_decode_input = embd(pt_out_tok)
                    all_outputs_ref.append(
                        pt_out_tok.squeeze(1).tolist()[0]
                    )  # Update generated token to list of ref outputs
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

            if not dummy_weights:
                logger.info("[ttnn generation User 0] " + tokenizer.decode(all_outputs).replace("\n", "\\n"))
                if run_ref_pt:
                    logger.info("[Ref generation User 0] " + tokenizer.decode(all_outputs_ref).replace("\n", "\\n"))
    finally:
        tt_model.tt_ccl.close()

    if run_ref_pt:
        if all_tests_pass:
            logger.info(f"All {generation_length} Llama decode iterations Passed!")
        else:
            logger.warning("One or more iterations of Llama decode had bad PCC")
            assert final_tests_pass, f"PCC value is lower than {final_model_pcc} for final output. Check Warnings!"
            assert kv_cache_tests_pass, f"KV Cache PCC value is lower expected for some of the outputs. Check Warnings!"
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
