# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Debug test to trace PCC at each stage of prefill.
Compares TT vs HF at embedding, attention input, attention output, and final logits.
"""

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.demos.llama3_70b_galaxy.tt.generator import Generator
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


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 102000000, "fabric_config": True}],
    indirect=True,
)
def test_debug_prefill_pcc(mesh_device, reset_seeds, ensure_gc):
    """Debug PCC at each computation stage."""

    n_layers = 1
    batch_size = 1  # Single user for easier debugging

    logger.info("=" * 60)
    logger.info("Debug Prefill PCC - Tracing divergence")
    logger.info("=" * 60)

    # Create TT model
    tt_model_args = TtOlmoModelArgs(
        mesh_device,
        max_batch_size=32,  # Model config expects 32
        max_seq_len=4096,
        instruct=False,
        n_layers=n_layers,
    )

    state_dict = tt_model_args.load_state_dict()

    paged_attention_config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=4096,
    )

    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(32, paged_attention_config.max_num_blocks // 32)

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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tt_model_args.TOKENIZER_PATH)
    tt_model_args.tokenizer = tokenizer

    # Load HF model
    logger.info(f"Loading HuggingFace model with {n_layers} layer(s)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        tt_model_args.CKPT_DIR,
        num_hidden_layers=n_layers,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    hf_model.eval()

    # Test prompt
    test_prompt = "The quick brown fox"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt")

    # Pad to 128 tokens
    if input_ids.shape[1] < 128:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        padding = torch.full((1, 128 - input_ids.shape[1]), pad_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, padding], dim=1)

    original_len = len(tokenizer.encode(test_prompt))
    logger.info(f"Prompt: {test_prompt!r}, tokens: {original_len}, padded to: {input_ids.shape[1]}")

    # === HuggingFace forward with intermediate outputs ===
    logger.info("Running HuggingFace forward...")

    # Get embedding output
    hf_embed = hf_model.model.embed_tokens(input_ids)
    logger.info(f"HF embedding shape: {hf_embed.shape}")
    logger.info(f"HF embedding stats: min={hf_embed.min():.4f}, max={hf_embed.max():.4f}, mean={hf_embed.mean():.4f}")

    # Full forward
    hf_output = hf_model(input_ids, output_hidden_states=True, use_cache=False)
    hf_logits = hf_output.logits[0, original_len - 1, :]

    logger.info(f"HF hidden_states[0] (after embed): {hf_output.hidden_states[0].shape}")
    logger.info(f"HF hidden_states[1] (after layer 0): {hf_output.hidden_states[1].shape}")

    hf_after_layer0 = hf_output.hidden_states[1][0, original_len - 1, :]
    logger.info(
        f"HF after layer 0 stats: min={hf_after_layer0.min():.4f}, max={hf_after_layer0.max():.4f}, mean={hf_after_layer0.mean():.4f}"
    )

    # === TT forward ===
    logger.info("Running TT forward...")

    generator = Generator(model, tt_model_args, mesh_device, tokenizer=tokenizer)

    # Expand for batch 32 (model requirement)
    input_ids_batch = input_ids.expand(32, -1)

    tt_logits = generator.prefill_forward_text(
        input_ids_batch,
        page_table=page_table,
        kv_cache=[tt_kv_cache],
        prompt_lens=[original_len] * 32,
        enable_trace=False,
        sampling_params=None,
    )

    tt_logits_user0 = tt_logits[0, 0, :]

    # === Compare ===
    logger.info("=" * 60)
    logger.info("PCC Comparison")
    logger.info("=" * 60)

    # Logits PCC
    vocab_size = min(hf_logits.shape[0], tt_logits_user0.shape[0])
    pcc_logits = compute_pcc(hf_logits[:vocab_size], tt_logits_user0[:vocab_size])
    logger.info(f"Logits PCC: {pcc_logits:.6f}")

    # Stats comparison
    logger.info(f"HF logits: min={hf_logits.min():.4f}, max={hf_logits.max():.4f}, mean={hf_logits.mean():.4f}")
    logger.info(
        f"TT logits: min={tt_logits_user0.min():.4f}, max={tt_logits_user0.max():.4f}, mean={tt_logits_user0.mean():.4f}"
    )

    # Top tokens
    hf_top = torch.argmax(hf_logits).item()
    tt_top = torch.argmax(tt_logits_user0).item()
    logger.info(f"HF top token: {hf_top} = {tokenizer.decode([hf_top])!r}")
    logger.info(f"TT top token: {tt_top} = {tokenizer.decode([tt_top])!r}")
    logger.info(f"Top token match: {hf_top == tt_top}")

    # Cleanup
    model.tt_ccl.close()

    assert pcc_logits > 0.80, f"PCC {pcc_logits:.6f} below threshold"


if __name__ == "__main__":
    import sys

    pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
