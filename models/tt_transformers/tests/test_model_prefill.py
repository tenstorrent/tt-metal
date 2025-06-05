# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import bz2
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision
from models.utility_functions import comp_pcc, skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.timeout(900)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
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
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "seq_len",
    (128, 3072, 4096, 8192, 16384, 32768),
    ids=["128", "3k", "4k", "8k", "16k", "32k"],
)
@pytest.mark.parametrize(
    "max_seq_len",
    (128 * 1024,),
    ids=[
        "max128k",
    ],
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
    "num_layers",
    (1, None),
    ids=["1layer", "all_layers"],
)
def test_model_inference(
    paged_attention,
    page_params,
    optimizations,
    seq_len,
    max_seq_len,
    num_layers,
    mesh_device,
    use_program_cache,
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
    dtype = ttnn.bfloat8_b
    batch_size = 1  # For prefill we only support batch_size = 1

    # Use instruct weights instead of general weights
    instruct = True

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        if paged_attention
        else None
    )

    # Load TTNN model
    logger.info(f"Loading TT model...")
    model_args, tt_model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device,
        instruct=instruct,
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        dtype=dtype,
        num_layers=num_layers,
    )

    if model_args.base_model_name.startswith("Mistral-") or model_args.base_model_name.startswith("Qwen3-"):
        # TODO: Per layer KV cache fetching is not implemented for all models
        # See issue https://github.com/tenstorrent/tt-metal/issues/19806"
        cache_pcc = False
    else:
        cache_pcc = True

    # This sets the minimum PCC for each iteration based on optimization mode
    # TODO: See issue https://github.com/tenstorrent/tt-metal/issues/19806
    perf_out_pcc_map = {"Mistral-7B-Instruct-v0.3": 0.73}
    acc_out_pcc_map = {"Mistral-7B-Instruct-v0.3": 0.75}
    kv_cache_pcc_map = {"Mistral-7B-Instruct-v0.3": 0.75}

    if num_layers == 1:
        expec_out_pcc = 0.97
        expec_kv_cache_pcc = 0.99
    else:
        if "accuracy" in test_id:
            default_expec_out_pcc = 0.91  # TODO Look on improving PCC
            expec_out_pcc = acc_out_pcc_map.get(model_args.model_name, default_expec_out_pcc)
        else:  # performance mode
            assert "performance" in test_id
            default_expec_out_pcc = 0.869  # TODO Look on improving PCC
            expec_out_pcc = perf_out_pcc_map.get(model_args.model_name, default_expec_out_pcc)

        default_expec_kv_cache_pcc = 0.88
        expec_kv_cache_pcc = kv_cache_pcc_map.get(model_args.model_name, default_expec_kv_cache_pcc)

    tokenizer = model_args.tokenizer
    generator = Generator([tt_model], [model_args], mesh_device, tokenizer=tokenizer)
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
    else:
        page_table = None

    # Load prompt
    current_file_path = os.path.abspath(__file__)
    current_file_dir = os.path.dirname(current_file_path)
    prompt_file = os.path.join(current_file_dir, "tale-of-two-cities.txt.bz2")
    with bz2.open(prompt_file, "rt", encoding="utf-8") as f:
        prompt = f.read()
    encoded_prompt = model_args.encode_prompt(prompt, instruct=instruct)[:seq_len]
    logger.info(f"Prompt length: {len(encoded_prompt)} tokens")

    # Load reference model
    if run_ref_pt:
        logger.info("Loading reference model...")
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
        reference_model = model_args.reference_transformer()
        reference_model.load_state_dict(reference_state_dict)
        # Embedding on host
        embd = model_args.reference_embedding()
        embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})
        logger.info("Finished loading reference model.")

    # Select the first token from the prompt for initial decoding
    encoded_prompt_tensor = torch.tensor(encoded_prompt)  # [:,0]
    tt_prefill_input = encoded_prompt_tensor.unsqueeze(0)
    prompt_lens = [seq_len]
    start_pos = 0

    # Run TT model
    logger.info(f"Running TT model...")
    tt_output_torch = generator.prefill_forward_text(
        tt_prefill_input,
        page_table=page_table,
        kv_cache=[tt_kv_cache],
        prompt_lens=prompt_lens,
    )
    logger.info(f"Finished running TT model.")

    if run_ref_pt:
        # Run reference model
        logger.info(f"Running reference model...")
        pt_prefill_input = embd(encoded_prompt_tensor).view(batch_size, seq_len, -1)
        ref_output = reference_model(pt_prefill_input, start_pos)
        ref_output = ref_output[:, -1:, :]  # Get last token since TT model only returns the last token
        logger.info(f"Finished running reference model.")

        # Measure PCC if also running reference model
        all_tests_pass = True

        # Check output pcc
        passing, pcc_message = comp_pcc(ref_output, tt_output_torch, expec_out_pcc)
        logger.info(f"Output PCC: {pcc_message}")
        if not passing:
            all_tests_pass = False
            logger.warning(f"Output PCC {pcc_message} is lower than {expec_out_pcc}")

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
                    for layer_past in tt_model.layers[i].attention.layer_past_list[0]:
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
