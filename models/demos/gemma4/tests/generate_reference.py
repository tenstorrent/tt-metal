# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Generate PyTorch reference outputs for Gemma 4 E4B.
Saves per-layer hidden states and final logits for PCC validation.
"""

import json
import os
from pathlib import Path

GEMMA4_WEIGHTS = os.environ.get(
    "GEMMA4_WEIGHTS",
    os.path.expanduser(
        "~/gemma4_weights/models--google--gemma-4-E4B-it/snapshots/292a7e278a400932df35f9fd4b1501edd04133a5"
    ),
)
REFERENCE_DIR = (
    Path(os.environ.get("TT_METAL_HOME", ".")) / "models" / "demos" / "gemma4" / "tests" / "reference_outputs"
)


def generate_reference(prompt="The capital of France is", max_new_tokens=1):
    """Generate reference hidden states and logits for PCC validation."""
    os.environ["USER"] = os.environ.get("USER", "node")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {GEMMA4_WEIGHTS}...")
    tokenizer = AutoTokenizer.from_pretrained(GEMMA4_WEIGHTS)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA4_WEIGHTS,
        dtype=torch.bfloat16,
    )
    model.eval()

    print(f"Tokenizing prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Input shape: {input_ids.shape}, tokens: {input_ids.tolist()}")

    # Forward pass with output hidden states
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    logits = outputs.logits
    hidden_states = outputs.hidden_states  # tuple of (n_layers + 1) tensors

    print(f"Logits shape: {logits.shape}")
    print(f"Number of hidden state layers: {len(hidden_states)}")
    print(f"Hidden state shape: {hidden_states[0].shape}")

    # Get the predicted token
    next_token_logits = logits[0, -1, :]  # Last position
    next_token_id = torch.argmax(next_token_logits).item()
    next_token = tokenizer.decode([next_token_id])
    print(f"Predicted next token: '{next_token}' (id={next_token_id})")

    # Apply softcapping to logits (for comparison)
    softcapped_logits = torch.tanh(logits / 30.0) * 30.0
    print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"Softcapped range: [{softcapped_logits.min():.4f}, {softcapped_logits.max():.4f}]")

    # Save reference outputs
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "prompt": prompt,
        "input_ids": input_ids.tolist(),
        "next_token_id": next_token_id,
        "next_token": next_token,
        "logits_shape": list(logits.shape),
        "n_hidden_layers": len(hidden_states),
    }
    with open(REFERENCE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save logits (last position only to save space)
    torch.save(logits[0, -1, :].float().cpu(), REFERENCE_DIR / "logits_last_pos.pt")

    # Save embedding output and select hidden states (every 6th layer + last)
    for i in [0, 6, 12, 18, 24, 30, 36, 42]:
        if i < len(hidden_states):
            torch.save(hidden_states[i][0, -1, :].float().cpu(), REFERENCE_DIR / f"hidden_state_layer_{i}_last_pos.pt")

    print(f"\nReference outputs saved to {REFERENCE_DIR}")
    print(f"Files: {[f.name for f in REFERENCE_DIR.iterdir()]}")

    return metadata


if __name__ == "__main__":
    generate_reference()
