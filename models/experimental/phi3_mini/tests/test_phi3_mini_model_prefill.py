# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import bz2
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.experimental.phi3_mini.tt.phi3_mini_common import create_tt_model
from models.tt_transformers.tt.model_config import DecodersPrecision
from models.experimental.phi3_mini.tt.phi3_mini_generator import Phi3MiniGenerator
from models.utility_functions import (
    comp_pcc,
)
from models.utility_functions import skip_for_grayskull
from models.tt_transformers.tt.model_config import HfModelWrapper
from models.tt_transformers.tt.common import PagedAttentionConfig


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
    (None,),
    ids=["all_layers"],
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

    # This sets the minimum PCC for each iteration based on optimization mode
    if num_layers == 1:
        expec_out_pcc = 0.97
    else:
        if "accuracy" in test_id:
            expec_out_pcc = 0.91  # TODO Look on improving PCC for low seq_len
        else:  # performance mode
            assert "performance" in test_id
            expec_out_pcc = 0.82  # TODO Look on improving PCC for low seq_len

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
    model_args, tt_model, tt_kv_cache, _ = create_tt_model(
        mesh_device,
        instruct=instruct,
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        dtype=dtype,
        num_layers=num_layers,
    )
    tokenizer = model_args.tokenizer
    generator = Phi3MiniGenerator([tt_model], [model_args], mesh_device, tokenizer=tokenizer)
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
        reference_transformer_model = model_args.reference_transformer(wrap=False)
        reference_model = HfModelWrapper(reference_transformer_model, model_args.head_dim)

        # Embedding on host
        embd = model_args.reference_embedding(reference_transformer_model)
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

        if all_tests_pass:
            logger.info("All PCC checks passed!")
        else:
            assert all_tests_pass, f"PCC is lower than expected for some of the outputs. Check warnings!"
