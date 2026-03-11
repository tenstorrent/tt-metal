#!/usr/bin/env python3
"""
Compare Llama hidden states: TTNN (bfloat16) vs reference (float32 on CPU).
Runs the full Llama backbone on CPU for a few steps and compares.
"""
import json
import os

import torch
import torch.nn as nn
from loguru import logger
from safetensors.torch import load_file as safetensors_load_file

TADA_MODEL_PATH = os.environ.get("TADA_MODEL_PATH", "/home/ttuser/atupe/tada/tada-1b")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

HIDDEN_SIZE = 2048
ACOUSTIC_DIM = 512
NUM_TIME_CLASSES = 256


def load_full_tada_model_cpu(model_path):
    """Load the full TADA model (Llama + TADA embeddings) on CPU."""
    from transformers import LlamaConfig, LlamaModel

    with open(os.path.join(model_path, "config.json")) as f:
        config_dict = json.load(f)

    llama_config = LlamaConfig(
        vocab_size=config_dict["vocab_size"],
        hidden_size=config_dict["hidden_size"],
        num_hidden_layers=config_dict["num_hidden_layers"],
        num_attention_heads=config_dict["num_attention_heads"],
        num_key_value_heads=config_dict["num_key_value_heads"],
        intermediate_size=config_dict["intermediate_size"],
        max_position_embeddings=config_dict.get("max_position_embeddings", 131072),
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-5),
        rope_theta=config_dict.get("rope_theta", 500000.0),
    )

    logger.info(f"Loading Llama model: {llama_config.num_hidden_layers} layers, {llama_config.hidden_size} dim")

    model = LlamaModel(llama_config)

    # Load weights
    weights = safetensors_load_file(os.path.join(model_path, "model.safetensors"))
    model_state = {}
    for k, v in weights.items():
        if k.startswith("model."):
            model_state[k[6:]] = v.float()

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    logger.info(f"Llama: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        logger.info(f"  Missing: {missing[:5]}...")

    model.eval()

    # Load TADA embeddings
    embed_tokens = model.embed_tokens
    acoustic_proj = nn.Linear(ACOUSTIC_DIM, HIDDEN_SIZE, bias=True)
    acoustic_proj.weight = nn.Parameter(weights["acoustic_proj.weight"].float())
    acoustic_proj.bias = nn.Parameter(weights["acoustic_proj.bias"].float())

    acoustic_mask_emb = nn.Embedding(2, HIDDEN_SIZE)
    acoustic_mask_emb.weight = nn.Parameter(weights["acoustic_mask_emb.weight"].float())

    time_start_embed = nn.Embedding(NUM_TIME_CLASSES, HIDDEN_SIZE)
    time_start_embed.weight = nn.Parameter(weights["time_start_embed.weight"].float())

    time_end_embed = nn.Embedding(NUM_TIME_CLASSES, HIDDEN_SIZE)
    time_end_embed.weight = nn.Parameter(weights["time_end_embed.weight"].float())

    tada_modules = {
        "embed_tokens": embed_tokens,
        "acoustic_proj": acoustic_proj,
        "acoustic_mask_emb": acoustic_mask_emb,
        "time_start_embed": time_start_embed,
        "time_end_embed": time_end_embed,
    }

    return model, tada_modules


def build_inputs_embeds(input_id, acoustic, mask, t_before, t_after, modules):
    """Build single-step inputs_embeds on CPU."""
    token_emb = modules["embed_tokens"](input_id.unsqueeze(1))
    ac_emb = modules["acoustic_proj"](acoustic.unsqueeze(1))
    mask_emb = modules["acoustic_mask_emb"](mask.unsqueeze(1))
    t_start = modules["time_start_embed"](t_before.unsqueeze(1))
    t_end = modules["time_end_embed"](t_after.unsqueeze(1))
    return token_emb + ac_emb + mask_emb + t_start + t_end


def main():
    from transformers import AutoTokenizer

    # Load TTNN hidden states
    debug_path = os.path.join(OUTPUT_DIR, "debug_hidden_states.pt")
    if not os.path.exists(debug_path):
        logger.error("No TTNN hidden states found. Run demo first.")
        return

    data = torch.load(debug_path, weights_only=False)
    ttnn_hidden = data["hidden_states"]
    input_ids_full = data["input_ids"]
    logger.info(f"Loaded {len(ttnn_hidden)} TTNN hidden states")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # Load full model on CPU
    logger.info("Loading Llama model on CPU (float32)...")
    llama_model, tada_modules = load_full_tada_model_cpu(TADA_MODEL_PATH)
    logger.info("Model loaded.")

    # Run Llama on CPU for first N steps and compare hidden states
    N = min(15, len(ttnn_hidden))
    acoustic_features = torch.zeros(1, ACOUSTIC_DIM)
    acoustic_masks = torch.zeros(1, dtype=torch.long)
    time_before = torch.zeros(1, dtype=torch.long)
    time_after = torch.zeros(1, dtype=torch.long)

    past_key_values = None
    cache_position = torch.tensor([0], dtype=torch.long)

    logger.info(f"\n=== Comparing hidden states for first {N} steps ===")
    for step in range(N):
        tok_id = input_ids_full[:, step]
        tok_name = tokenizer.decode([tok_id.item()])

        # Build embedding
        inputs_embeds = build_inputs_embeds(
            tok_id, acoustic_features, acoustic_masks, time_before, time_after, tada_modules
        )

        # Run Llama forward
        with torch.no_grad():
            outputs = llama_model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )
        past_key_values = outputs.past_key_values
        cpu_hidden = outputs.last_hidden_state  # (1, 1, 2048)
        cache_position = cache_position + 1

        # Compare with TTNN hidden state
        ttnn_h = ttnn_hidden[step].float()  # (1, 1, 2048)

        # Cosine similarity
        cos_sim = nn.functional.cosine_similarity(cpu_hidden.flatten(), ttnn_h.flatten(), dim=0).item()

        # PCC
        cpu_flat = cpu_hidden.flatten()
        ttnn_flat = ttnn_h.flatten()
        pcc = torch.corrcoef(torch.stack([cpu_flat, ttnn_flat]))[0, 1].item()

        # Relative error
        rel_err = (cpu_hidden - ttnn_h).norm() / cpu_hidden.norm()

        logger.info(
            f"  Step {step:2d} [{repr(tok_name):20s}]: "
            f"cpu_norm={cpu_hidden.norm():.3f}, ttnn_norm={ttnn_h.norm():.3f}, "
            f"cos_sim={cos_sim:.6f}, pcc={pcc:.6f}, rel_err={rel_err:.6f}"
        )

    logger.info("\nDone. If cos_sim/pcc are close to 1.0, Llama hidden states match well.")
    logger.info("If they diverge, bfloat16 precision in TTNN Llama is causing the distortion.")


if __name__ == "__main__":
    main()
