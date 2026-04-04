#!/usr/bin/env python3
"""
Generate Gemma 4 E4B reference outputs WITHOUT per_layer_input gating or KV sharing.
Validates the core attention + MLP + norms pipeline.
"""

import getpass
import json
import os
from pathlib import Path

getpass.getuser = lambda: "node"

import torch

MODEL_PATH = os.environ.get(
    "HF_MODEL",
    "/workspace/group/gemma4_weights/models--google--gemma-4-E4B-it/snapshots/292a7e278a400932df35f9fd4b1501edd04133a5",
)
OUTPUT_DIR = Path(__file__).parent / "reference_outputs_no_gating"


def generate_reference():
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Tokens: {[tokenizer.decode([t]) for t in input_ids[0]]}")

    text_model = model.model.language_model

    # Patch each decoder layer to skip per_layer_input gating
    for layer in text_model.layers:
        layer.hidden_size_per_layer_input = 0

    # Patch the text model to skip per_layer_input computation entirely
    text_model.hidden_size_per_layer_input = 0

    # Also patch the config objects (checked by outer model forward)
    model.config.text_config.hidden_size_per_layer_input = 0
    if hasattr(text_model, "config"):
        text_model.config.hidden_size_per_layer_input = 0
    if hasattr(model.model, "config"):
        model.model.config.text_config.hidden_size_per_layer_input = 0

    # Note: KV sharing is ENABLED (essential for correct model behavior)
    # Only per-layer input gating is disabled for bring-up

    print("Running forward pass (no per_layer_input gating, KV sharing enabled)...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits  # [1, seq_len, vocab_size]
    print(f"Logits shape: {logits.shape}")

    last_pos_logits = logits[0, -1, :].float()
    predicted_token = last_pos_logits.argmax().item()
    print(f"Predicted token: {predicted_token} ({tokenizer.decode([predicted_token])})")
    print(
        f"Logits stats: min={last_pos_logits.min():.4f}, max={last_pos_logits.max():.4f}, mean={last_pos_logits.mean():.4f}"
    )

    top5 = torch.topk(last_pos_logits, 5)
    print(f"Top-5 tokens: {top5.indices.tolist()}")
    print(f"Top-5 decoded: {[tokenizer.decode([t]) for t in top5.indices.tolist()]}")
    print(f"Top-5 values: {[f'{v:.4f}' for v in top5.values.tolist()]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(last_pos_logits, OUTPUT_DIR / "logits_last_pos.pt")

    metadata = {
        "prompt": prompt,
        "input_ids": input_ids.tolist(),
        "next_token_id": predicted_token,
        "next_token": tokenizer.decode([predicted_token]),
        "logits_shape": list(logits.shape),
        "per_layer_gating": False,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_reference()
