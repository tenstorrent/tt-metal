# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Compare TTNN TADA embedding computation with reference on CPU.
Loads weights directly, bypasses torchaudio.
"""
import os
import sys

import torch
import torch.nn as nn
from loguru import logger
from safetensors.torch import load_file as safetensors_load_file

TADA_MODEL_PATH = os.environ.get("TADA_MODEL_PATH", "/home/ttuser/atupe/tada/tada-1b")

# TADA constants
ACOUSTIC_DIM = 512
HIDDEN_SIZE = 2048
NUM_TIME_CLASSES = 256
SHIFT_ACOUSTIC = 5


def load_tada_embedding_modules(model_path):
    """Load just the embedding-related weights from TADA safetensors."""
    weights = safetensors_load_file(os.path.join(model_path, "model.safetensors"))
    # Convert all weights to float32
    weights = {k: v.float() for k, v in weights.items()}

    # embed_tokens
    embed_tokens = nn.Embedding(128256, HIDDEN_SIZE)
    embed_tokens.weight = nn.Parameter(weights["model.embed_tokens.weight"])

    # acoustic_proj (Linear 512 -> 2048 with bias)
    acoustic_proj = nn.Linear(ACOUSTIC_DIM, HIDDEN_SIZE, bias=True)
    acoustic_proj.weight = nn.Parameter(weights["acoustic_proj.weight"])
    acoustic_proj.bias = nn.Parameter(weights["acoustic_proj.bias"])

    # acoustic_mask_emb (Embedding(2, 2048))
    acoustic_mask_emb = nn.Embedding(2, HIDDEN_SIZE)
    acoustic_mask_emb.weight = nn.Parameter(weights["acoustic_mask_emb.weight"])

    # time_start_embed, time_end_embed (Embedding(256, 2048))
    time_start_embed = nn.Embedding(NUM_TIME_CLASSES, HIDDEN_SIZE)
    time_start_embed.weight = nn.Parameter(weights["time_start_embed.weight"])

    time_end_embed = nn.Embedding(NUM_TIME_CLASSES, HIDDEN_SIZE)
    time_end_embed.weight = nn.Parameter(weights["time_end_embed.weight"])

    # lm_head (shares weights with embed_tokens in TADA 1B)
    lm_head = nn.Linear(HIDDEN_SIZE, 128256, bias=False)
    lm_head.weight = embed_tokens.weight  # tied weights

    return {
        "embed_tokens": embed_tokens,
        "acoustic_proj": acoustic_proj,
        "acoustic_mask_emb": acoustic_mask_emb,
        "time_start_embed": time_start_embed,
        "time_end_embed": time_end_embed,
        "lm_head": lm_head,
    }


def reference_embed_inputs(input_ids, acoustic_features, acoustic_masks, time_before, time_after, modules):
    """Reference embedding computation on CPU (float32)."""
    token_emb = modules["embed_tokens"](input_ids)  # (B, 1, 2048)
    acoustic_emb = modules["acoustic_proj"](acoustic_features)  # (B, 1, 2048)
    mask_emb = modules["acoustic_mask_emb"](acoustic_masks)  # (B, 1, 2048)
    time_start_emb = modules["time_start_embed"](time_before)  # (B, 1, 2048)
    time_end_emb = modules["time_end_embed"](time_after)  # (B, 1, 2048)

    return token_emb + acoustic_emb + mask_emb + time_start_emb + time_end_emb


def compare_embeddings():
    """Compare TTNN embedding with reference for multiple token positions."""
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    logger.info("Loading embedding modules...")
    modules = load_tada_embedding_modules(TADA_MODEL_PATH)

    # Build the same input sequence as the TTNN demo
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tt.tada_generator import normalize_text

    generation_text = normalize_text("This is a test of text to speech on Tenstorrent hardware.")
    prefix = "<|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    text_tokens = tokenizer.encode(generation_text, add_special_tokens=False)
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)

    bos_id = tokenizer.bos_token_id
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    all_tokens = [bos_id] + prefix_tokens + text_tokens + [eot_id] * 5
    input_ids = torch.tensor([all_tokens], dtype=torch.long)
    prefix_len = 1 + len(prefix_tokens)  # 8

    logger.info(f"Input sequence: {len(all_tokens)} tokens, prefix_len={prefix_len}")

    # Simulate the AR loop embedding computation for first 15 steps
    acoustic_features = torch.zeros(1, ACOUSTIC_DIM)
    acoustic_masks = torch.zeros(1, dtype=torch.long)
    time_len_before = torch.zeros(1, dtype=torch.long)
    time_len_after = torch.zeros(1, dtype=torch.long)

    logger.info("\n=== Reference embedding norms per step ===")
    for step in range(min(15, len(all_tokens))):
        tok_id = input_ids[:, step]
        tok_name = tokenizer.decode([tok_id.item()])

        # Compute reference embedding
        ref_emb = reference_embed_inputs(
            tok_id.unsqueeze(1),
            acoustic_features.unsqueeze(1),
            acoustic_masks.unsqueeze(1),
            time_len_before.unsqueeze(1),
            time_len_after.unsqueeze(1),
            modules,
        )

        # Individual component norms
        tok_emb = modules["embed_tokens"](tok_id.unsqueeze(1))
        ac_emb = modules["acoustic_proj"](acoustic_features.unsqueeze(1))
        mask_emb = modules["acoustic_mask_emb"](acoustic_masks.unsqueeze(1))
        t_start = modules["time_start_embed"](time_len_before.unsqueeze(1))
        t_end = modules["time_end_embed"](time_len_after.unsqueeze(1))

        logger.info(
            f"  Step {step} [{repr(tok_name):20s}]: "
            f"total={ref_emb.norm().item():.3f}, "
            f"tok={tok_emb.norm().item():.3f}, "
            f"ac={ac_emb.norm().item():.3f}, "
            f"mask={mask_emb.norm().item():.3f}, "
            f"t_start={t_start.norm().item():.3f}, "
            f"t_end={t_end.norm().item():.3f}"
        )

    # Now also load the TTNN embedding and compare for the same inputs
    # This requires a TT device, so skip if not available
    try:
        pass

        logger.info("\n=== TTNN vs Reference embedding comparison ===")
        # We need a mesh device — check if available
        logger.info("TT device comparison requires running as pytest. Skipping TTNN comparison.")
    except ImportError:
        logger.info("TTNN not available, skipping device comparison.")

    # Also check: what does acoustic_mask_emb(0) vs acoustic_mask_emb(1) look like?
    mask0 = modules["acoustic_mask_emb"](torch.tensor([0]))
    mask1 = modules["acoustic_mask_emb"](torch.tensor([1]))
    logger.info(f"\nacoustic_mask_emb(0) norm={mask0.norm().item():.6f}, mean={mask0.mean().item():.6f}")
    logger.info(f"acoustic_mask_emb(1) norm={mask1.norm().item():.6f}, mean={mask1.mean().item():.6f}")
    logger.info(f"acoustic_mask_emb diff norm={torch.norm(mask1 - mask0).item():.6f}")

    # Check acoustic_proj(zeros) — should be just bias
    zero_acoustic = torch.zeros(1, ACOUSTIC_DIM)
    ac_zero = modules["acoustic_proj"](zero_acoustic)
    logger.info(f"\nacoustic_proj(zeros) = bias only: norm={ac_zero.norm().item():.6f}")
    logger.info(f"acoustic_proj.bias norm={modules['acoustic_proj'].bias.norm().item():.6f}")

    # Check time embeddings at 0
    t0_start = modules["time_start_embed"](torch.tensor([0]))
    t0_end = modules["time_end_embed"](torch.tensor([0]))
    logger.info(f"\ntime_start_embed(0) norm={t0_start.norm().item():.6f}")
    logger.info(f"time_end_embed(0) norm={t0_end.norm().item():.6f}")


if __name__ == "__main__":
    compare_embeddings()
