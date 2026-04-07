# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Validate Gemma 4 TT implementation against CPU reference (HuggingFace).
Compares logits output for the same input prompt.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import torch
import ttnn
from loguru import logger


def compute_pcc(ref, test):
    """Compute Pearson Correlation Coefficient between two tensors."""
    ref = ref.float().flatten()
    test = test.float().flatten()
    # Remove any NaN/Inf values
    mask = torch.isfinite(ref) & torch.isfinite(test)
    ref = ref[mask]
    test = test[mask]
    if len(ref) == 0:
        return 0.0
    ref_mean = ref.mean()
    test_mean = test.mean()
    ref_std = ref.std()
    test_std = test.std()
    if ref_std == 0 or test_std == 0:
        return 1.0 if torch.allclose(ref, test) else 0.0
    pcc = ((ref - ref_mean) * (test - test_mean)).mean() / (ref_std * test_std)
    return pcc.item()


def get_cpu_reference_logits(prompt, model_name="google/gemma-4-31B-it"):
    """Get reference logits from HuggingFace model on CPU."""
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    logger.info("Loading CPU reference model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    model.eval()
    t1 = time.time()
    logger.info(f"CPU model loaded in {t1-t0:.1f}s")

    tokens = tokenizer.encode(prompt, return_tensors="pt")
    logger.info(f"Prompt: '{prompt}' -> {tokens.shape[1]} tokens")

    with torch.no_grad():
        outputs = model(input_ids=tokens)
        cpu_logits = outputs.logits  # [batch, seq_len, vocab_size]

    # Get logits for the last token
    last_token_logits = cpu_logits[0, -1, :]  # [vocab_size]
    next_token = torch.argmax(last_token_logits).item()
    logger.info(f"CPU reference next token: {next_token} = '{tokenizer.decode([next_token])}'")

    return cpu_logits, tokens, tokenizer, model


def get_tt_logits(prompt, model_args, mesh_device, model, tokenizer):
    """Get logits from TT implementation."""

    tokens = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = tokens.shape[1]
    padded_len = max(256, ((seq_len + 127) // 128) * 128)
    padded_tokens = torch.nn.functional.pad(tokens, (0, padded_len - seq_len), value=0)

    last_token_idx = seq_len - 1
    tt_inputs, rot_mats_global, rot_mats_local, _, _ = model.prepare_inputs_prefill(
        padded_tokens, start_pos=0, last_token_idx=last_token_idx,
    )

    get_last = (last_token_idx // 32) * 32
    tt_out = model.ttnn_prefill_forward(
        tt_inputs,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        get_last_token=get_last,
    )

    tt_out_host = ttnn.from_device(tt_out)
    tt_logits = model.process_output_prefill(tt_out_host, last_token_idx % 32)

    next_token = torch.argmax(tt_logits).item()
    logger.info(f"TT next token: {next_token} = '{tokenizer.decode([next_token])}'")

    return tt_logits


def test_cpu_validation():
    prompt = "The capital of France is"

    # Step 1: Get CPU reference
    cpu_logits, tokens, tokenizer, cpu_model = get_cpu_reference_logits(prompt)
    cpu_last_logits = cpu_logits[0, -1, :]
    del cpu_model  # Free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    # Step 2: Get TT implementation output
    logger.info("Opening TT mesh device...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))

    try:
        from models.demos.multimodal.gemma4.tt.model_config import ModelArgs
        from models.demos.multimodal.gemma4.tt.gemma4_model import Gemma4Transformer

        model_args = ModelArgs(
            mesh_device=mesh_device, instruct=True, max_batch_size=1, max_seq_len=2048,
        )

        state_dict = model_args.load_state_dict()
        model = Gemma4Transformer(
            args=model_args, dtype=ttnn.bfloat16, mesh_device=mesh_device,
            state_dict=state_dict, weight_cache_path=model_args.weight_cache_path(ttnn.bfloat16),
        )

        tt_logits = get_tt_logits(prompt, model_args, mesh_device, model, tokenizer)

        # Step 3: Compare
        # Compare top-k tokens
        k = 10
        cpu_topk = torch.topk(cpu_last_logits, k)
        tt_topk = torch.topk(tt_logits[:cpu_last_logits.shape[0]], k)

        logger.info(f"\nTop-{k} comparison:")
        logger.info(f"{'Rank':<6} {'CPU Token':<12} {'TT Token':<12} {'CPU Logit':<12} {'TT Logit':<12} {'Match'}")
        matches = 0
        for i in range(k):
            cpu_tok = cpu_topk.indices[i].item()
            tt_tok = tt_topk.indices[i].item()
            match = "✓" if cpu_tok == tt_tok else "✗"
            if cpu_tok == tt_tok:
                matches += 1
            logger.info(f"{i+1:<6} {cpu_tok:<12} {tt_tok:<12} {cpu_topk.values[i].item():<12.4f} {tt_topk.values[i].item():<12.4f} {match}")

        # Compute PCC on full logits
        min_vocab = min(cpu_last_logits.shape[0], tt_logits.shape[0])
        pcc = compute_pcc(cpu_last_logits[:min_vocab], tt_logits[:min_vocab])
        logger.info(f"\nPearson Correlation (logits): {pcc:.6f}")
        logger.info(f"Top-{k} token match rate: {matches}/{k}")
        logger.info(f"Top-1 match: {'YES' if cpu_topk.indices[0] == tt_topk.indices[0] else 'NO'}")

        accuracy_pct = pcc * 100
        logger.info(f"\nAccuracy: {accuracy_pct:.2f}%")

        if pcc >= 0.99:
            logger.info("VALIDATION PASSED (PCC >= 0.99)")
        elif pcc >= 0.90:
            logger.info("VALIDATION MARGINAL (0.90 <= PCC < 0.99)")
        else:
            logger.info("VALIDATION FAILED (PCC < 0.90)")

    finally:
        ttnn.close_mesh_device(mesh_device)
        logger.info("Mesh device closed")


if __name__ == "__main__":
    test_cpu_validation()
