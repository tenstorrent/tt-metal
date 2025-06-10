# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.t3000.llama2_70b.demo.demo_continuous_batching_paged_attention import (
    ModelArgs,
    PagedAttentionConfig,
    TTArgs,
)
from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.tt.llama_common import (
    check_mesh_device,
    comp_pcc,
    load_llama_state_dict,
    setup_llama_env,
)
from models.demos.t3000.llama2_70b.tt.llama_generation import (
    TtLlamaModelForGeneration,
    get_block_size,
    num_blocks_in_seq,
)


def run_chunked_prefill_single_user(model_args, tt_args, chunk_size):
    # Set up paged attention config
    paged_attention_config = PagedAttentionConfig()
    bsz = model_args.max_batch_size

    # Create static page table (same as in demo)
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    static_page_table = reverse_permutation.reshape(bsz, paged_attention_config.max_num_blocks // bsz)

    # Build reference generator
    ref_generator = Llama.build(
        ckpt_dir=model_args.ckpt_dir,
        tokenizer_path=model_args.tokenizer_path,
        max_seq_len=model_args.max_seq_len,
        max_batch_size=model_args.max_batch_size,
        skip_model_load=model_args.skip_model_load,
        n_layers=model_args.num_layers,
    )

    # Load state dict for TT model
    state_dict = load_llama_state_dict(model_args.ckpt_dir, n_layers=model_args.num_layers)

    # Build TT generator with paged attention
    tt_model = TtLlamaModelForGeneration(
        configuration=ref_generator.model.params,
        state_dict=state_dict,
        model_args=model_args,
        tt_args=tt_args,
        paged_attention_config=paged_attention_config,
    )

    # For testing, override the max prefill length
    tt_model.model_config["MAX_PREFILL_SEQ_LEN"] = chunk_size

    # Extract the model's KV cache such that we can pass it in to the forward function
    kv_cache = [l.attention.layer_past for l in tt_model.tt_model.layers]

    # Create random input
    seq_len = model_args.max_seq_len
    input_tokens = torch.randint(0, 32000, (1, seq_len), dtype=torch.long)

    # Slice out relevant part of page table
    block_size = get_block_size(kv_cache)
    num_blocks = num_blocks_in_seq(seq_len, block_size)
    static_page_table = static_page_table[:, :num_blocks]

    # Run both models
    with torch.no_grad():
        tt_logits = tt_model.prefill_forward_single_user(
            input_tokens,
            start_pos=0,
            user_id=0,
            page_table=static_page_table,
            kv_cache=kv_cache,
        )
        ref_logits = ref_generator.model.forward(input_tokens, start_pos=0)

    # Compare outputs
    does_pass, pcc = comp_pcc(ref_logits, tt_logits, pcc=0.99)
    logger.info(f"PCC between reference and TT model logits: {pcc}")
    assert does_pass, f"Logits PCC {pcc} below threshold of 0.99"

    ref_kv_cache = [[l.attention.cache_k, l.attention.cache_v] for l in ref_generator.model.layers]
    # Compare KV caches
    for layer_idx in range(len(kv_cache)):
        tt_cache = kv_cache[layer_idx]
        ref_cache = ref_kv_cache[layer_idx]

        # Unshuffle paged cache and review it as unpaged cache (similar to paged_update_cache test)
        tt_got_back_shuffled = [
            ttnn.to_torch(kv, mesh_composer=ttnn.ConcatMeshToTensor(tt_args.mesh_device, dim=1)) for kv in tt_cache
        ]
        tt_got_back_unshuffled = [shuffled[reverse_permutation] for shuffled in tt_got_back_shuffled]

        # Reshape to match reference cache dimensions
        max_num_blocks = tt_got_back_shuffled[0].shape[0]
        block_size = tt_got_back_shuffled[0].shape[2]
        num_heads = tt_got_back_shuffled[0].shape[1]
        head_dim = tt_got_back_shuffled[0].shape[3]
        tt_got_back = [
            unshuffled.reshape(1, max_num_blocks, num_heads, block_size, head_dim)
            .transpose(1, 2)
            .reshape(1, num_heads, -1, head_dim)
            for unshuffled in tt_got_back_unshuffled
        ]

        for i in range(len(tt_got_back)):
            ref_cache_slice = ref_cache[i][:1, :seq_len, :, :].permute(0, 2, 1, 3)
            # Compare caches
            does_pass_cache, pcc_cache = comp_pcc(ref_cache_slice, tt_got_back[i][:, :, :seq_len, :])
            logger.info(f"PCC between reference and TT model KV cache at layer {layer_idx}: {pcc_cache}")
        assert does_pass_cache, f"KV cache PCC {pcc_cache} below threshold at layer {layer_idx}"

    return does_pass, pcc


@torch.no_grad()
@pytest.mark.timeout(240000)
@pytest.mark.parametrize(
    "llama_version",
    ["llama3"],
)
@pytest.mark.parametrize(
    "num_layers",
    [
        1,
    ],
    ids=[
        "1L",
    ],
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len, chunk_size",
    [(1, 128 * 1024, 32 * 1024), (16, 8 * 1024, 2 * 1024), (32, 2 * 1024, 1 * 1024)],
    ids=["1BSZ", "16BSZ", "32BSZ"],
)
def test_chunked_prefill_single_user(
    t3k_mesh_device, llama_version, num_layers, max_batch_size, max_context_len, chunk_size
):
    """
    This test ensures that chunked prefill, when used by calling `prefill_forward_single_user`,
    matches the reference implementation.
    """
    if max_context_len == 128 * 1024:
        pytest.skip("Skipping test for max_context_len = 128*1024 since reference runs OOM")
    # Set up environment
    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
    )

    # Check device compatibility
    check_mesh_device(t3k_mesh_device, model_config)

    # Create args
    model_args = ModelArgs(
        implementation="tt",
        llama_version=llama_version,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_batch_size=max_batch_size,
        num_layers=num_layers,
        max_seq_len=max_context_len,
        max_kv_context_len=max_context_len,
    )

    tt_args = TTArgs(
        mesh_device=t3k_mesh_device,
        n_devices=8,
        cache_path=cache_path,
    )

    # Run test
    does_pass, pcc = run_chunked_prefill_single_user(model_args, tt_args, chunk_size)
    assert does_pass, f"Test failed with PCC {pcc}"


def run_batch_prefill_test(model_args, tt_args, chunk_size, batch):
    # Set up paged attention config
    paged_attention_config = PagedAttentionConfig()
    # Create static page table (same as in demo)
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    static_page_table = reverse_permutation.reshape(
        model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
    )
    # Build reference generator
    ref_generator = Llama.build(
        ckpt_dir=model_args.ckpt_dir,
        tokenizer_path=model_args.tokenizer_path,
        max_seq_len=model_args.max_seq_len,
        max_batch_size=model_args.max_batch_size,
        skip_model_load=model_args.skip_model_load,
        n_layers=model_args.num_layers,
    )

    # Load state dict for TT model
    state_dict = load_llama_state_dict(model_args.ckpt_dir, n_layers=model_args.num_layers)

    # Build TT generator with paged attention
    tt_model = TtLlamaModelForGeneration(
        configuration=ref_generator.model.params,
        state_dict=state_dict,
        model_args=model_args,
        tt_args=tt_args,
        paged_attention_config=paged_attention_config,
    )

    # For testing, override the max prefill length
    tt_model.model_config["MAX_PREFILL_SEQ_LEN"] = chunk_size

    # Extract the model's KV cache such that we can pass it in to the forward function
    kv_cache = [l.attention.layer_past for l in tt_model.tt_model.layers]

    # Create random input with varying sequence lengths
    max_seq_len = model_args.max_seq_len
    prompt_lens = torch.randint(
        chunk_size, max_seq_len + 1, (batch,)
    )  # Random lengths between chunk_size and max_seq_len
    input_tokens = torch.randint(0, 32000, (batch, max_seq_len), dtype=torch.long)
    logger.info(f"Prompt lengths: {prompt_lens}")
    batch_page_table = static_page_table[:batch]
    # Run both models
    with torch.no_grad():
        tt_logits = tt_model.prefill_forward(
            input_tokens,
            start_pos=0,
            page_table=batch_page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
        )
        logger.info(f"TT logits shape: {tt_logits.shape}")

        # Run reference model
        batch_logits = ref_generator.model.forward(input_tokens, start_pos=0)
        ref_logits = batch_logits[torch.arange(batch), prompt_lens - 1, :].unsqueeze(1)  # Only keep last token's logits
        ref_kv_cache = [[l.attention.cache_k, l.attention.cache_v] for l in ref_generator.model.layers]

    # Compare outputs
    does_pass, pcc = comp_pcc(ref_logits, tt_logits, pcc=0.99)
    logger.info(f"PCC between reference and TT model: {pcc}")
    assert does_pass, f"PCC {pcc} below threshold of 0.99"

    # Compare KV caches
    for layer_idx in range(len(kv_cache)):
        tt_cache = kv_cache[layer_idx]
        ref_cache = ref_kv_cache[layer_idx]

        # Unshuffle paged cache and review it as unpaged cache (similar to paged_update_cache test)
        tt_got_back_shuffled = [
            ttnn.to_torch(kv, mesh_composer=ttnn.ConcatMeshToTensor(tt_args.mesh_device, dim=1)) for kv in tt_cache
        ]
        tt_got_back_unshuffled = [shuffled[reverse_permutation] for shuffled in tt_got_back_shuffled]

        # Reshape to match reference cache dimensions
        max_num_blocks = tt_got_back_shuffled[0].shape[0]
        block_size = tt_got_back_shuffled[0].shape[2]
        num_heads = tt_got_back_shuffled[0].shape[1]
        head_dim = tt_got_back_shuffled[0].shape[3]
        tt_got_back = [
            unshuffled.reshape(
                model_args.max_batch_size, max_num_blocks // model_args.max_batch_size, num_heads, block_size, head_dim
            )
            .transpose(1, 2)
            .reshape(model_args.max_batch_size, num_heads, -1, head_dim)
            for unshuffled in tt_got_back_unshuffled
        ]

        for b in range(batch):
            valid_seq_len = prompt_lens[b]
            logger.info(f"valid seq len: {valid_seq_len}")
            for i in range(len(tt_got_back)):
                logger.info(f"layer {i}, batch {b}")
                logger.info(f"ref cache shape: {ref_cache[i].shape}")
                logger.info(f"tt cache shape: {tt_got_back[i].shape}")
                ref_cache_slice = ref_cache[i][b : b + 1, :valid_seq_len, :, :].permute(0, 2, 1, 3)
                tt_cache_slice = tt_got_back[i][b : b + 1, :, :valid_seq_len, :]
                # Compare caches
                does_pass_cache, pcc_cache = comp_pcc(ref_cache_slice, tt_cache_slice)
                logger.info(f"PCC between reference and TT model KV cache at layer {layer_idx}: {pcc_cache}")
                assert does_pass_cache, f"KV cache PCC {pcc_cache} below threshold at layer {layer_idx}"

    return does_pass, pcc


@torch.no_grad()
@pytest.mark.timeout(240000)
@pytest.mark.parametrize(
    "llama_version",
    ["llama3"],
)
@pytest.mark.parametrize(
    "num_layers",
    [
        1,
    ],
    ids=[
        "1L",
    ],
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len, chunk_size, batch",
    [(1, 128 * 1024, 32 * 1024, 1), (16, 8 * 1024, 2 * 1024, 4), (32, 2 * 1024, 1 * 1024, 4)],
    ids=["1BSZ", "16BSZ", "32BSZ"],
)
def test_batch_prefill(t3k_mesh_device, llama_version, num_layers, max_batch_size, max_context_len, chunk_size, batch):
    """
    This test ensures that batch prefill matches the reference implementation
    when processing multiple sequences of different lengths.
    """
    if max_context_len == 128 * 1024:
        pytest.skip("Skipping test for max_context_len = 128*1024 since reference runs OOM")
    # Set up environment
    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
    )

    # Check device compatibility
    check_mesh_device(t3k_mesh_device, model_config)

    # Create args
    model_args = ModelArgs(
        implementation="tt",
        llama_version=llama_version,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_batch_size=max_batch_size,
        num_layers=num_layers,
        max_seq_len=max_context_len,
        max_kv_context_len=max_context_len,
    )

    tt_args = TTArgs(
        mesh_device=t3k_mesh_device,
        n_devices=8,
        cache_path=cache_path,
    )

    # Run test
    does_pass, pcc = run_batch_prefill_test(model_args, tt_args, chunk_size, batch)
    assert does_pass, f"Test failed with PCC {pcc}"
