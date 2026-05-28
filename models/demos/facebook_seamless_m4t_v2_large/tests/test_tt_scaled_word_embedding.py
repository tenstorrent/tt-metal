# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 TTNN ScaledWordEmbedding block.

Loads the saved golden metadata from
`models/demos/facebook_seamless_m4t_v2_large/reference/golden/scaled_word_embedding.pt`
and reconstructs the embedding weight from the recorded ``weight_seed`` so the
test doesn't have to carry a ~1 GB table on disk. Runs the TTNN block on the
open p150 (blackhole) device and asserts PCC > 0.99 against the reference
output.

Can also be run as a standalone script (`python test_tt_scaled_word_embedding.py`)
which is how the bring-up orchestrator invokes it. It opens its own device,
runs the comparison, prints PCC as JSON, and exits.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ScaledWordEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.scaled_word_embedding import ScaledWordEmbedding

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "scaled_word_embedding.pt"


def _reconstruct_weight(golden) -> torch.Tensor:
    """Re-init the HF embedding with the recorded seed to get the matching weight.

    The golden file stores ``weight_seed`` (the value passed to
    ``torch.manual_seed`` before constructing the HF module). Re-running that
    sequence reproduces the exact same weight tensor; we verify with the
    stored ``weight_checksum`` for safety.
    """
    cfg = golden["config"]
    vocab_size = int(cfg["vocab_size"])
    hidden_size = int(cfg["hidden_size"])
    padding_idx = int(golden["padding_idx"])
    scale = float(golden["scale"])
    seed = int(golden["weight_seed"])

    torch.manual_seed(seed)
    hf_module = (
        SeamlessM4Tv2ScaledWordEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx,
            embed_scale=scale,
        )
        .to(torch.float32)
        .eval()
    )
    weight = hf_module.weight.detach().clone()

    expected_checksum = float(golden["weight_checksum"])
    actual_checksum = float(weight.to(torch.float64).sum().item())
    # Sanity check: seed reproduction should be exact.
    assert math.isclose(
        actual_checksum, expected_checksum, rel_tol=0, abs_tol=1e-3
    ), f"weight checksum mismatch: got {actual_checksum} expected {expected_checksum}"
    return weight


def _run_scaled_word_embedding_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    input_ids: torch.Tensor = golden["input_ids"]
    scale: float = float(golden["scale"])
    padding_idx: int = int(golden["padding_idx"])
    ref_out: torch.Tensor = golden["output"]

    weight = _reconstruct_weight(golden)

    tt_embed = ScaledWordEmbedding(
        device=device,
        weight=weight,
        scale=scale,
        padding_idx=padding_idx,
        weight_dtype=ttnn.bfloat16,
    )

    # ttnn.embedding expects uint32 ROW_MAJOR ids.
    tt_input_ids = ttnn.from_torch(
        input_ids.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_embed(tt_input_ids)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)
    tt_output_torch = tt_output_torch.reshape(ref_out.shape)

    passing, pcc_message = comp_pcc(ref_out, tt_output_torch, 0.99)
    msg_str = str(pcc_message).strip()
    try:
        pcc_value = float(msg_str)
    except ValueError:
        import re

        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        pcc_value = float(match.group(0)) if match else float("nan")
    print(f"comp_pcc: passing={passing}, message={pcc_message}")
    return pcc_value


def test_tt_scaled_word_embedding():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        pcc_value = _run_scaled_word_embedding_pcc(device)
    finally:
        ttnn.close_device(device)
    assert pcc_value > 0.99, f"PCC {pcc_value} <= 0.99"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    pcc_value = float("nan")
    try:
        pcc_value = _run_scaled_word_embedding_pcc(device)
    finally:
        ttnn.close_device(device)
    result = {"pcc": pcc_value, "passed": pcc_value > 0.99}
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
