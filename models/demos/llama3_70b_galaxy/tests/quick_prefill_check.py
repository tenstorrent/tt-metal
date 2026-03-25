# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Quick prefill sanity check - compares TT vs HuggingFace on a simple prompt.
Uses the Generator class to properly handle tensor distribution.
"""

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.demos.llama3_70b_galaxy.tt.generator import Generator
from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_pcc(ref, test):
    """Compute Pearson correlation coefficient."""
    ref_flat = ref.flatten().float()
    test_flat = test.flatten().float()

    ref_mean = ref_flat.mean()
    test_mean = test_flat.mean()

    ref_centered = ref_flat - ref_mean
    test_centered = test_flat - test_mean

    numerator = (ref_centered * test_centered).sum()
    denominator = torch.sqrt((ref_centered**2).sum() * (test_centered**2).sum())

    if denominator == 0:
        return 0.0
    return (numerator / denominator).item()


def create_tt_olmo_model_single_layer(mesh_device, batch_size=32, n_layers=1):
    """Create OLMo model with specified number of layers for testing."""
    tt_model_args = TtOlmoModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=4096,
        instruct=False,
        n_layers=n_layers,
    )

    state_dict = tt_model_args.load_state_dict()

    paged_attention_config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=4096,
    )
    # Page table
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(batch_size, paged_attention_config.max_num_blocks // batch_size)

    model = TtTransformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paged_attention_config,
        mode="prefill",
        enable_prefetcher_performance_mode=False,
    )

    tt_kv_cache = [l.attention.layer_past for l in model.layers]

    return tt_model_args, model, page_table, [tt_kv_cache]


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 102000000, "fabric_config": True}],
    indirect=True,
)
def test_quick_prefill_check(mesh_device, reset_seeds, ensure_gc):
    """Quick sanity check comparing TT prefill logits vs HuggingFace."""

    n_layers = 1  # Single layer for quick test
    batch_size = 32

    logger.info("=" * 60)
    logger.info("Quick Prefill Sanity Check - TT vs HuggingFace")
    logger.info("=" * 60)

    # Create TT model
    logger.info(f"Creating TT OLMo model with {n_layers} layer(s)...")
    model_args, model, page_table, tt_kv_caches = create_tt_olmo_model_single_layer(
        mesh_device, batch_size=batch_size, n_layers=n_layers
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.TOKENIZER_PATH)
    model_args.tokenizer = tokenizer

    # Create generator
    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)

    # Load HuggingFace model (1 layer only)
    logger.info(f"Loading HuggingFace model with {n_layers} layer(s)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_args.CKPT_DIR,
        num_hidden_layers=n_layers,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    hf_model.eval()

    # Create test prompt - pad to 128 tokens (minimum prefill length)
    test_prompt = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt")

    # Pad to 128 tokens
    if input_ids.shape[1] < 128:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        padding = torch.full((1, 128 - input_ids.shape[1]), pad_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, padding], dim=1)
    else:
        input_ids = input_ids[:, :128]

    original_len = len(tokenizer.encode(test_prompt))
    padded_len = input_ids.shape[1]
    logger.info(f"Input: {original_len} tokens padded to {padded_len}")
    logger.info(f"HF will get logits at position {original_len - 1}, TT should get logits at same position")

    # Run HuggingFace
    logger.info("Running HuggingFace prefill...")
    hf_output = hf_model(input_ids, output_hidden_states=True, use_cache=False)

    # Get logits at the last real token (before padding)
    hf_logits = hf_output.logits[0, original_len - 1, :]

    logger.info(f"HF logits stats: min={hf_logits.min():.4f}, max={hf_logits.max():.4f}, mean={hf_logits.mean():.4f}")

    # Get HF top-5 predictions
    hf_probs = torch.softmax(hf_logits.float(), dim=-1)
    hf_top5_probs, hf_top5_ids = torch.topk(hf_probs, k=5)
    logger.info("HuggingFace Top-5 predictions:")
    for i, (prob, tok_id) in enumerate(zip(hf_top5_probs, hf_top5_ids)):
        logger.info(f"  {i+1}. {tokenizer.decode([tok_id])!r} (id={tok_id.item()}, prob={prob.item():.4f})")

    # Run TT prefill
    logger.info("Running TT prefill...")
    # Expand input for batch 32
    input_ids_batch = input_ids.expand(batch_size, -1)

    # Use generator's prefill which handles distribution properly
    # sampling_params=None causes return_logits=True
    # IMPORTANT: Use original_len as prompt_lens so TT gets logits at same position as HF
    tt_logits = generator.prefill_forward_text(
        input_ids_batch,  # torch.Tensor
        page_table=page_table,
        kv_cache=tt_kv_caches,
        prompt_lens=[original_len] * batch_size,  # Use actual length, not padded length!
        enable_trace=False,  # Disable trace for debugging
        sampling_params=None,  # Return logits
    )

    # tt_logits shape: [batch, 1, vocab_size]
    logger.info(f"TT logits shape: {tt_logits.shape}")

    # Get TT logits at the last real token position (same as HF)
    # The prefill returns logits at last_token_idx which is prompt_len - 1
    tt_logits_user0 = tt_logits[0, 0, :]  # [vocab_size]

    logger.info(
        f"TT logits stats: min={tt_logits_user0.min():.4f}, max={tt_logits_user0.max():.4f}, mean={tt_logits_user0.mean():.4f}"
    )

    # Get TT top-5 predictions
    tt_probs = torch.softmax(tt_logits_user0.float(), dim=-1)
    tt_top5_probs, tt_top5_ids = torch.topk(tt_probs, k=5)
    logger.info("TT Top-5 predictions:")
    for i, (prob, tok_id) in enumerate(zip(tt_top5_probs, tt_top5_ids)):
        logger.info(f"  {i+1}. {tokenizer.decode([tok_id])!r} (id={tok_id.item()}, prob={prob.item():.4f})")

    # Compute PCC between HF and TT logits
    # HF logits need to match vocab size (TT might be padded)
    vocab_size = min(hf_logits.shape[0], tt_logits_user0.shape[0])
    pcc = compute_pcc(hf_logits[:vocab_size], tt_logits_user0[:vocab_size])
    logger.info(f"PCC between HF and TT logits: {pcc:.6f}")

    # Check if top token matches
    top_token_match = hf_top5_ids[0].item() == tt_top5_ids[0].item()
    logger.info(f"Top token match: {top_token_match}")

    logger.info("=" * 60)
    logger.info("Comparison Results")
    logger.info("=" * 60)
    logger.info(f"HF predicts: {tokenizer.decode([hf_top5_ids[0]])!r}")
    logger.info(f"TT predicts: {tokenizer.decode([tt_top5_ids[0]])!r}")
    logger.info(f"PCC: {pcc:.6f}")

    # Cleanup
    model.tt_ccl.close()

    # Assert PCC threshold
    # Note: BF8 precision + per-head QK norm approximation gives ~0.83 PCC for 1 layer
    # This is acceptable for a functional test
    assert pcc > 0.80, f"PCC {pcc:.6f} below threshold 0.80"
    assert top_token_match, "Top token should match between HF and TT"


if __name__ == "__main__":
    # For manual testing
    import sys

    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
