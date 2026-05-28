# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 TTNN SinusoidalPositionalEmbedding block.

Loads the saved golden file from
`models/demos/facebook_seamless_m4t_v2_large/reference/golden/sinusoidal_positional_embedding.pt`
which carries the full 4098x1024 sinusoidal weight table plus both encoder
(batch=1, seq=128) and decoder-incremental (batch=1, seq=1, past_kv=64)
input/output pairs. Runs the TTNN block on the open p150 (blackhole) device
and asserts PCC > 0.99 for both modes.

Can also be run as a standalone script
(`python test_tt_sinusoidal_positional_embedding.py`) which is how the
bring-up orchestrator invokes it. It opens its own device, runs the
comparison, prints PCC as JSON, and exits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "sinusoidal_positional_embedding.pt"


def _extract_pcc(message) -> float:
    msg_str = str(message).strip()
    try:
        return float(msg_str)
    except ValueError:
        import re

        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_pcc(device) -> dict:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    weights: torch.Tensor = golden["weights"]
    padding_idx: int = int(golden["config"]["padding_idx"])

    tt_embed = SinusoidalPositionalEmbedding(
        device=device,
        weights=weights,
        padding_idx=padding_idx,
        weight_dtype=ttnn.bfloat16,
    )

    results = {}

    # Encoder mode: batch=1, seq=128, past_key_values_length=0.
    enc = golden["encoder"]
    enc_out_tt = tt_embed(
        input_ids=enc["input_ids"],
        past_key_values_length=int(enc["past_key_values_length"]),
    )
    enc_out_torch = ttnn.to_torch(enc_out_tt).to(torch.float32).reshape(enc["output"].shape)
    passing_enc, msg_enc = comp_pcc(enc["output"], enc_out_torch, 0.99)
    enc_pcc = _extract_pcc(msg_enc)
    print(f"[encoder] passing={passing_enc} pcc_msg={msg_enc}")
    results["encoder_pcc"] = enc_pcc

    # Decoder-incremental mode: batch=1, seq=1, past_key_values_length=64.
    dec = golden["decoder_incremental"]
    dec_out_tt = tt_embed(
        input_ids=dec["input_ids"],
        past_key_values_length=int(dec["past_key_values_length"]),
    )
    dec_out_torch = ttnn.to_torch(dec_out_tt).to(torch.float32).reshape(dec["output"].shape)
    passing_dec, msg_dec = comp_pcc(dec["output"], dec_out_torch, 0.99)
    dec_pcc = _extract_pcc(msg_dec)
    print(f"[decoder_incremental] passing={passing_dec} pcc_msg={msg_dec}")
    results["decoder_pcc"] = dec_pcc

    return results


def test_tt_sinusoidal_positional_embedding():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        results = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    assert results["encoder_pcc"] > 0.99, f"encoder PCC {results['encoder_pcc']} <= 0.99"
    assert results["decoder_pcc"] > 0.99, f"decoder PCC {results['decoder_pcc']} <= 0.99"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    results = {"encoder_pcc": float("nan"), "decoder_pcc": float("nan")}
    try:
        results = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    results["passed"] = results["encoder_pcc"] > 0.99 and results["decoder_pcc"] > 0.99
    print(json.dumps(results))
    sys.exit(0 if results["passed"] else 1)
