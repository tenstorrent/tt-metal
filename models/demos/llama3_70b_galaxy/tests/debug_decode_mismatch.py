"""
Debug script to compare HuggingFace vs TT decode step by step.
"""

import torch
import pytest
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.demos.llama3_70b_galaxy.tt.generator import Generator


@pytest.fixture
def mesh_device(request):
    """Fixture for mesh device from pytest params."""
    return request.getfixturevalue("mesh_device")


def test_decode_comparison(mesh_device):
    """Compare HuggingFace vs TT decode outputs."""

    # Model setup
    model_name = "/home/cust-team/share/olmo/OLMo-3.1-32B-Think"
    batch_size = 1  # Use single user for easier debugging

    logger.info(f"Loading HuggingFace model: {model_name}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a short prompt
    prompt = "Hello<|reserved_special_token_170|>"  # <think> token
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    original_len = input_ids.shape[1]
    logger.info(f"Input: {prompt!r}, tokens: {input_ids.tolist()}, len={original_len}")

    # === HuggingFace decode ===
    logger.info("Running HuggingFace generation for 5 tokens...")
    with torch.no_grad():
        hf_outputs = hf_model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,  # Greedy
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    hf_generated = hf_outputs.sequences[0, original_len:].tolist()
    logger.info(f"HuggingFace generated: {hf_generated}")
    logger.info(f"HuggingFace decoded: {tokenizer.decode(hf_generated)!r}")

    # Log HF logits for each decode step
    for i, scores in enumerate(hf_outputs.scores):
        top_token = scores[0].argmax().item()
        top_logit = scores[0].max().item()
        logger.info(
            f"HF decode step {i}: top token={top_token} ({tokenizer.decode([top_token])!r}), logit={top_logit:.4f}"
        )

    # === TT decode ===
    logger.info("Setting up TT model...")
    # Pad input to 128 for prefill
    padded_len = 128
    input_ids_padded = torch.nn.functional.pad(
        input_ids[0], (0, padded_len - original_len), value=tokenizer.pad_token_id or 0
    ).unsqueeze(0)

    # Create generator
    from models.demos.llama3_70b_galaxy.tt.olmo_model_config import TtOlmoModelArgs

    model_args = TtOlmoModelArgs(mesh_device, is_decode=False)

    generator = Generator(mesh_device, model_args)

    # Create KV cache

    tt_model = generator.model
    paged_attention_config = tt_model.paged_attention_config

    tt_kv_caches = []
    for layer_idx in range(model_args.n_layers):
        layer = tt_model.layers[layer_idx]
        layer_kv_cache = [layer.attention.layer_past[0], layer.attention.layer_past[1]]
        tt_kv_caches.append(layer_kv_cache)

    # Create page table
    page_table = torch.zeros((batch_size, paged_attention_config.max_num_blocks // batch_size), dtype=torch.int32)
    for i in range(batch_size):
        num_blocks_per_user = paged_attention_config.max_num_blocks // batch_size
        page_table[i] = torch.arange(i * num_blocks_per_user, (i + 1) * num_blocks_per_user)

    # Prefill
    logger.info("Running TT prefill...")
    tt_prefill_logits = generator.prefill_forward_text(
        input_ids_padded.repeat(32, 1),  # Batch to 32 for TT
        page_table=page_table.repeat(32, 1)[:batch_size],  # Use actual batch
        kv_cache=tt_kv_caches,
        prompt_lens=[original_len] * batch_size,
        enable_trace=False,
        sampling_params=None,
    )

    # Extract prefill token
    tt_prefill_token = tt_prefill_logits[:batch_size].argmax(dim=-1)
    logger.info(f"TT prefill token: {tt_prefill_token.item()} ({tokenizer.decode([tt_prefill_token.item()])!r})")

    # Switch to decode mode
    tt_model.switch_mode("decode")

    # Decode loop
    logger.info("Running TT decode for 5 tokens...")
    current_pos = torch.tensor([original_len] * 32, dtype=torch.long)  # Pad to 32
    out_tok = tt_prefill_token.view(-1, 1).repeat(32, 1)

    tt_generated = []
    for i in range(5):
        logger.info(f"TT decode step {i}: input token={out_tok[0, 0].item()}, pos={current_pos[0].item()}")

        # Run decode
        decode_result = generator.decode_forward(
            out_tok[:batch_size],
            current_pos[:batch_size],
            enable_trace=False,
            page_table=page_table,
            kv_cache=tt_kv_caches,
            read_from_device=True,
            async_read=False,
            sampling_params=None,  # Return logits
            reset_inputs=i == 0,
            is_cur_pos_sharded=False,
            is_page_table_sharded=False,
        )

        # Get top token from logits
        if isinstance(decode_result, tuple):
            tt_logits = decode_result[0]
        else:
            tt_logits = decode_result

        if hasattr(tt_logits, "shape"):
            top_token = int(tt_logits[0].argmax().item())
            top_logit = float(tt_logits[0].max().item())
        else:
            top_token = int(tt_logits[0])
            top_logit = 0.0

        logger.info(
            f"TT decode step {i}: output token={top_token} ({tokenizer.decode([top_token])!r}), logit={top_logit:.4f}"
        )
        tt_generated.append(top_token)

        # Update for next iteration
        out_tok[0, 0] = top_token
        current_pos += 1

    logger.info(f"TT generated: {tt_generated}")
    logger.info(f"TT decoded: {tokenizer.decode(tt_generated)!r}")

    # Compare
    logger.info("=== COMPARISON ===")
    logger.info(f"HF: {hf_generated}")
    logger.info(f"TT: {tt_generated}")
    match = hf_generated == tt_generated
    logger.info(f"Match: {match}")

    # Cleanup
    generator.model.tt_ccl.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-k", "test_decode_comparison"])
