# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import bz2
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3.tt.llama_common import (
    PagedAttentionConfig,
    get_block_size,
    num_blocks_in_seq,
    HostEmbedding,
)
from models.demos.llama3.tt.llama_model import TtTransformer
from models.demos.llama3.tt.model_config import TtModelArgs, LlamaOptimizations
from models.demos.llama3.tt.generator import LlamaGenerator
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Transformer
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (True,),
    ids=("paged_attention",),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 64, "page_max_num_blocks": 2048}],
)
@pytest.mark.parametrize(
    "seq_len, prefill_chunk_size",
    [(4096, 2048)],
)
@pytest.mark.parametrize(
    "optimizations",
    [
        pytest.param(LlamaOptimizations.accuracy, id="accuracy"),
    ],
)
def test_chunked_prefill_single_user(
    seq_len,
    prefill_chunk_size,
    paged_attention,
    page_params,
    optimizations,
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
    is_ci_env,
):
    mesh_device.enable_async(True)

    dtype = ttnn.bfloat8_b
    batch_size = 1  # For prefill we only support batch_size = 1

    # This sets the minimum PCC for each iteration based on optimization mode
    if optimizations == LlamaOptimizations.accuracy:
        pcc = 0.91  # TODO Look on improving PCC
    else:  # performance mode
        assert optimizations == LlamaOptimizations.performance
        pcc = 0.869  # TODO Look on improving PCC

    model_args = TtModelArgs(mesh_device, max_batch_size=batch_size, optimizations=optimizations, max_seq_len=seq_len)
    model_args.max_prefill_chunk_size = prefill_chunk_size

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

    reference_model = Transformer(model_args)
    reference_model.load_state_dict(reference_state_dict)
    embd = HostEmbedding(model_args)
    embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

    # Setup page table
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    # Implied shuffling of blocks
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    # Page table which maps virtual blocks to physical
    reverse_permutation = torch.argsort(permutation)
    static_page_table = reverse_permutation.reshape(
        model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
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
    generator = LlamaGenerator(tt_model, model_args, mesh_device)

    logger.info("Model and caches loaded.")

    # Select the first token from the prompt for initial decoding
    pt_prefill_input = torch.randint(0, 32000, (batch_size, seq_len), dtype=torch.long)
    tt_prefill_input = pt_prefill_input

    pt_prefill_input = embd(pt_prefill_input).view(batch_size, seq_len, -1)

    tt_kv_cache = [l.attention.layer_past for l in tt_model.layers]
    # Slice out relevant part of page table
    block_size = get_block_size(tt_kv_cache)
    num_blocks = num_blocks_in_seq(seq_len, block_size)
    static_page_table = static_page_table[:, :num_blocks]

    start_pos = 0
    logger.info("Running reference model")
    ref_output = reference_model(pt_prefill_input, start_pos, mode="decode")

    # Run TT model, collecting various last_token_idxs
    logger.info("Running TT model")
    for last_token_idx in range(prefill_chunk_size - 10, seq_len, prefill_chunk_size):
        logger.info(f"Running TT model for last_token_idx: {last_token_idx}")
        tt_output_torch = generator.prefill_forward_single_user_text(
            tt_prefill_input,
            page_table=static_page_table,
            user_id=0,
            last_token_idx=last_token_idx,
            kv_cache=tt_kv_cache,
        )
        tt_output_torch = tt_output_torch.reshape(batch_size, 1, -1)

        ref_output_slice = ref_output[:, last_token_idx : last_token_idx + 1, :]

        passing, pcc_message = comp_pcc(ref_output_slice, tt_output_torch, pcc)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
        assert passing
