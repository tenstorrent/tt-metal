# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_common import (
    precompute_freqs,
    get_single_rot_mat,
    prepare_inputs_ttnn,
    sample,
    encode_prompt_llama_instruct,
    HostEmbedding,
)
from models.demos.llama3.tt.llama_model import TtTransformer
from models.demos.llama3.tt.model_config import TtModelArgs
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
    "weights, layers",
    [
        ("random", 1),
        ("general", None),
    ],
    ids=["quick", "full"],
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
def test_llama_model_inference(mesh_device, weights, layers, use_program_cache, reset_seeds, ensure_gc):
    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
    cache_pcc = layers == 1  # Flag to measure KV cache PCC. Avoid running for all layers to speed up test time.

    dtype = ttnn.bfloat8_b

    mesh_device.enable_async(True)

    # This sets the minimum PCC for each iteration
    # TODO: In the full model test, iterations 4 and 8 have lower PCCs of 0.9077 and 0.9593 respectively.
    pcc = 0.94

    instruct = True if weights == "instruct" else False
    dummy_weights = True if weights == "random" else False
    model_args = TtModelArgs(mesh_device, instruct=instruct, dummy_weights=dummy_weights)

    model_from_layers = {
        16: "llama32_1b",
        28: "llama32_3b",
        32: "llama31_8b",
    }[model_args.n_layers]

    final_model_pcc = {"llama32_1b": 0.9991, "llama32_3b": 0.9990, "llama31_8b": 0.99765}[model_from_layers]
    final_k_cache_pcc = {"llama32_1b": 0.9998, "llama32_3b": 0.9998, "llama31_8b": 0.9995}[model_from_layers]  # TODO
    final_v_cache_pcc = {"llama32_1b": 0.9996, "llama32_3b": 0.9998, "llama31_8b": 0.9996}[model_from_layers]  # TODO
    quick_iterations = {"llama32_1b": 2, "llama32_3b": 4, "llama31_8b": 6}[model_from_layers]

    iterations = quick_iterations if layers == 1 else 9

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
        encoded_prompts = [[128000, 2028, 374, 264, 1296]]  # "This is a test" encoded prompt
        assert not instruct, "Instruct prompt not implemented with dummy weights"
    else:
        tokenizer = Tokenizer(model_args.tokenizer_path)
        if instruct:
            encoded_prompts = [encode_prompt_llama_instruct(tokenizer, prompt) for prompt in prompts]
        else:
            encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]

    if run_ref_pt:
        reference_model = Transformer(model_args)
        reference_model.load_state_dict(reference_state_dict)

    # Embedding on host
    embd = HostEmbedding(model_args)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    generation_start_pos = 0
    generation_length = iterations

    # pre-compute the rotational embedding matrix and send to device
    current_rot_mat, rot_matrix = get_single_rot_mat(
        model_args.head_dim,
        mesh_device,
        model_args.num_devices,
        start_pos=0,
    )

    # Load TTNN model
    tt_model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )
    logger.info("Model and caches loaded.")

    if run_ref_pt:
        all_tests_pass = True

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

    for i in range(generation_length):
        current_pos = generation_start_pos + i

        decode_input = prepare_inputs_ttnn(
            tt_decode_input,
            model_args.dim,
            tt_model.mesh_device,
        )
        current_pos_tensor = ttnn.from_torch(
            torch.tensor([current_pos] * batch),
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Run TT model
        tt_out = tt_model(decode_input, current_pos_tensor, rot_mat=current_rot_mat)
        # Convert ttnn tensor to torch tensor
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
            .permute(2, 1, 0, 3)
            .squeeze(1)[: model_args.max_batch_size, :, :]
        )  # [seq, batch, hidden_dim]

        ttnn.deallocate(tt_out)

        # Update rotation matrix for next iteration
        current_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)

        if run_ref_pt:  # Run reference model
            # freqs_cis_i = freqs_cis[current_pos, :].unsqueeze(0)
            # positions = torch.tensor([current_pos])
            # mask = ttnn.to_torch(attn_mask[0])
            ref_output = reference_model(pt_decode_input, current_pos)

        # While in "prefill" mode, use the prompt tokens as the output
        if i in range(len(encoded_prompts[0])):
            all_outputs.append(encoded_prompts[0][i])  # Update list of TT outputs
            if run_ref_pt:
                all_outputs_ref.append(encoded_prompts[0][i])  # Update list of ref outputs

            tt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
            if run_ref_pt:
                pt_decode_input = embd(encoded_prompts_tensor[:, i]).view(batch, seqlen, -1)
        else:
            # Greedy decode (temperature = 0) the generated token and save it to print out later
            tt_out_tok = sample(tt_output_torch, temperature=0, top_p=0.8)
            tt_decode_input = embd(tt_out_tok)
            all_outputs.append(tt_out_tok.squeeze(1).tolist()[0])  # Update generated token to list of TT outputs
            if run_ref_pt:
                pt_out_tok = sample(ref_output, temperature=0, top_p=0.8)
                pt_decode_input = embd(pt_out_tok)
                all_outputs_ref.append(
                    pt_out_tok.squeeze(1).tolist()[0]
                )  # Update generated token to list of ref outputs

        # Measure PCC if also running reference model
        if run_ref_pt:
            if layers == 1 and i == iterations - 1:  # On last iteration in the quick test, set a tighter PCC
                passing, pcc_message = comp_pcc(ref_output, tt_output_torch, final_model_pcc)
            else:
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
                    for layer_past in tt_model.layers[l].attention.layer_past:
                        tt_layer_present.append(
                            ttnn.to_torch(layer_past, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
                        )

                    for kv_cache, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                        cache_length_to_check = min(
                            model_args.sliding_window, generation_start_pos + generation_length + 1
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
                            logger.info(f"V Cache Passed!")
                        else:
                            logger.warning(f"V Cache Failed! PCC value is lower than {pcc}")
                            all_tests_pass = False

        if not dummy_weights:
            logger.info("[ttnn generation User 0] " + tokenizer.decode(all_outputs).replace("\n", "\\n"))
            if run_ref_pt:
                logger.info("[Ref generation User 0] " + tokenizer.decode(all_outputs_ref).replace("\n", "\\n"))

    if run_ref_pt:
        if all_tests_pass:
            logger.info(f"All {generation_length} Llama decode iterations Passed!")
        else:
            logger.warning("One or more iterations of Llama decode had bad PCC")
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
