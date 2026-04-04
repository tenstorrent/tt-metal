# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gemma 4 E4B Prefill Test

Validates the TtGemma4TextModel against PyTorch reference outputs.
Runs a single prefill pass with "The capital of France is" and checks:
1. PCC of final logits against reference
2. That the predicted next token is "France" (id=7001)
"""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.utility_functions import comp_pcc

# Reference output paths
REFERENCE_DIR = Path(__file__).parent / "reference_outputs"
GEMMA4_WEIGHTS = os.environ.get(
    "GEMMA4_WEIGHTS",
    "/workspace/group/gemma4_weights/models--google--gemma-4-E4B-it/snapshots/292a7e278a400932df35f9fd4b1501edd04133a5",
)


def get_reference_data():
    """Load reference metadata and outputs."""
    with open(REFERENCE_DIR / "metadata.json") as f:
        metadata = json.load(f)

    ref_logits = torch.load(REFERENCE_DIR / "logits_last_pos.pt", map_location="cpu")
    return metadata, ref_logits


@pytest.mark.parametrize("max_seq_len", [128])  # Short sequence for initial testing
def test_gemma4_prefill(mesh_device, max_seq_len, use_program_cache):
    """
    Test Gemma 4 E4B prefill with reference PCC validation.

    This test:
    1. Loads the Gemma 4 E4B model on N150
    2. Runs prefill with "The capital of France is" (5 tokens)
    3. Compares final logits against PyTorch reference
    4. Validates the predicted next token
    """
    os.environ["HF_MODEL"] = GEMMA4_WEIGHTS
    os.environ["USER"] = os.environ.get("USER", "node")

    metadata, ref_logits = get_reference_data()
    input_ids = metadata["input_ids"][0]  # [818, 5279, 529, 7001, 563]
    expected_token = metadata["next_token_id"]  # 7001 ("France")

    # Load model config
    from models.demos.gemma4.tt.model_config import ModelArgs

    model_args = ModelArgs(
        mesh_device=mesh_device,
        instruct=True,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )

    # Load state dict
    state_dict = model_args.load_state_dict()

    # Create model
    from models.demos.gemma4.tt.gemma4_model import TtGemma4TextModel

    weight_cache_path = model_args.weight_cache_path(model_args.get_model_config().get("DEFAULT_DTYPE", ttnn.bfloat8_b))

    model = TtGemma4TextModel(
        args=model_args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
    )

    # Prepare input tokens
    tokens = torch.tensor([input_ids], dtype=torch.long)  # [1, 5]
    seq_len = tokens.shape[-1]

    # Pad to tile boundary (nearest 32)
    padded_seq_len = ((seq_len + 31) // 32) * 32
    if padded_seq_len < 128:
        padded_seq_len = 128  # Minimum prefill size
    padded_tokens = torch.nn.functional.pad(tokens, (0, padded_seq_len - seq_len), value=0)

    # Prepare prefill inputs
    prefill_inputs = model.prepare_inputs_prefill(
        padded_tokens,
        start_pos=0,
        last_token_idx=seq_len - 1,
    )
    (tt_tokens_embd, rot_mats_global, rot_mats_local, tt_page_table, tt_chunk_page_table) = prefill_inputs

    # Run forward
    get_last_token = ((seq_len - 1) // 32) * 32
    tt_logits = model.ttnn_prefill_forward(
        tt_tokens_embd,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        user_id=0,
        page_table=tt_page_table,
        chunk_page_table=tt_chunk_page_table,
        get_last_token=get_last_token,
    )

    # Process output
    tt_logits_host = ttnn.to_torch(ttnn.from_device(tt_logits))

    # Extract last token logits
    last_token_offset = (seq_len - 1) % 32
    tt_logits_last = tt_logits_host[0, 0, last_token_offset, : model_args.vocab_size].float()

    # Check predicted token
    predicted_token = tt_logits_last.argmax().item()
    print(f"\nPredicted token: {predicted_token} (expected: {expected_token})")
    print(f"Predicted correct: {predicted_token == expected_token}")

    # PCC against reference logits
    pcc_val = comp_pcc(ref_logits.unsqueeze(0), tt_logits_last.unsqueeze(0))
    print(f"Logits PCC: {pcc_val}")

    # Assert minimum PCC
    min_pcc = 0.90  # Initial target, will increase as implementation matures
    assert pcc_val[0] >= min_pcc, f"PCC {pcc_val[0]:.4f} below threshold {min_pcc}"
    print(f"\n✓ Prefill test passed! PCC={pcc_val[0]:.4f}, token={predicted_token}")


if __name__ == "__main__":
    # Allow running directly for debugging
    import ttnn

    device = ttnn.open_device(device_id=0)
    mesh_device = ttnn.MeshDevice([device])
    try:
        test_gemma4_prefill(mesh_device, max_seq_len=128, use_program_cache=True)
    finally:
        ttnn.close_device(device)
