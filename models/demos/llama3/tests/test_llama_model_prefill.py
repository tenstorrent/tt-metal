# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import bz2
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_common import (
    get_prefill_rot_mat,
    prepare_inputs_ttnn_prefill,
    get_rot_transformation_mat,
    sample,
    HostEmbedding,
    encode_prompt_llama_instruct,
)
from models.demos.llama3.tt.llama_model import TtTransformer
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Transformer, precompute_freqs_cis
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
    "seq_len",
    (
        # 128,
        # 1024,
        4096,
    ),
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
def test_llama_model_inference(mesh_device, seq_len, use_program_cache, reset_seeds):
    run_ref_pt = True  # Flag to run reference PyTorch model and compare PCC
    cache_pcc = False  # Flag to measure KV cache PCC for all layers

    dtype = ttnn.bfloat8_b
    pcc = 0.93

    mesh_device.enable_async(True)

    # Use instruct weights instead of general weights
    instruct = False

    model_args = TtModelArgs(mesh_device, instruct=instruct)
    # model_args.n_layers = 1

    tokenizer = Tokenizer(model_args.tokenizer_path)

    logger.info("Loading weights...")
    state_dict_prefix = model_args.get_state_dict_prefix("", None)

    state_dict = torch.load(model_args.consolidated_weights_path, map_location=torch.device("cpu"))
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
        prompts = f.read()

    if instruct:
        encoded_prompts = [encode_prompt_llama_instruct(tokenizer, prompt) for prompt in prompts]
    else:
        encoded_prompts = tokenizer.encode(prompts, bos=True, eos=False)[:seq_len]

    if run_ref_pt:
        reference_model = Transformer(model_args)
        reference_model.load_state_dict(reference_state_dict)
    # Embedding on host
    embd = HostEmbedding(model_args)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    # pre-compute the rotational embedding matrix and send to device
    rot_mats = get_prefill_rot_mat(model_args.head_dim, model_args.max_seq_len, mesh_device, seq_len=seq_len)
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

    batch = 1

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor).view(batch, seq_len, -1)

    tt_decode_input = pt_decode_input

    decode_input = prepare_inputs_ttnn_prefill(
        tt_decode_input,
        tt_model.mesh_device,
    )
    for i in range(1):
        start_pos = 0
        # Run TT model
        tt_out = tt_model(decode_input, None, rot_mats, transformation_mats, user_id=i, mode="prefill")
        # Convert ttnn tensor to torch tensor
        tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))[
            :, 0, :, :
        ].view(
            batch, seq_len, -1
        )  # [ batch, seq, hidden_dim]

        if run_ref_pt:  # Run reference model
            ref_output = reference_model(pt_decode_input, start_pos, mode="prefill")

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
                        .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                        reference_model.layers[i]
                        .attention.cache_v.clone()
                        .permute(0, 2, 1, 3),  # [batch, n_kv_heads, seq, head_dim]
                    ]

                    tt_layer_present = []
                    for layer_past in tt_model.layers[i].attention.layer_past_list[0]:
                        tt_layer_present.append(
                            ttnn.to_torch(layer_past, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
                        )

                    for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
                        cache_length_to_check = model_args.sliding_window
                        cache_pt = cache_pt[:, :, 0:cache_length_to_check, :]
                        cache_tt = cache_tt[:, :, 0:cache_length_to_check, :]
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
            logger.info(f"All Llama decode iterations Passed!")
        else:
            logger.warning("One or more iterations of Llama decode had bad PCC")
            assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
