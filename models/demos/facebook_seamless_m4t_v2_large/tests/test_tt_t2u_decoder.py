# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 T2U decoder TTNN block.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/t2u_decoder.pt``,
runs the TTNN ``T2uDecoder`` block on the open p150 (blackhole) device, and
asserts PCC > 0.99 against the saved reference outputs.

The golden uses ``num_layers=2`` and ``batch=1``, ``encoder_seq_len=4``,
``char_seq_len=8`` (the same configuration the reference itself verified
against HuggingFace bit-for-bit). The reference returns ``dur_out`` as
all-ones (each char produces one unit), so the upsampled unit length is 8.

Can also be run as a standalone script
(``python test_tt_t2u_decoder.py``) which opens its own device, runs the
PCC comparison, prints a single-line JSON, and exits 0 on pass / 1 on fail.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_decoder import T2uDecoder

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "t2u_decoder.pt"


def _pcc_from_message(passing: bool, pcc_message) -> float:
    msg_str = str(pcc_message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    char_input_ids: torch.Tensor = golden["char_input_ids"]
    char_count_per_id: torch.Tensor = golden["char_count_per_id"]
    encoder_hidden_states: torch.Tensor = golden["encoder_hidden_states"]
    state_dict = golden["state_dict"]
    char_positional_weights: torch.Tensor = golden["char_positional_weights"]
    positional_weights: torch.Tensor = golden["positional_weights"]
    ref_last_hidden: torch.Tensor = golden["output_last_hidden_state"]
    ref_padding_mask: torch.Tensor = golden["output_padding_mask"]
    ref_dur_out: torch.Tensor = golden["output_dur_out"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    eps = float(cfg["eps"])
    embed_scale = float(cfg["embed_scale"])
    padding_idx = int(cfg["padding_idx"])
    conv_kernel_size = int(cfg["conv_kernel_size"])
    variance_predictor_kernel_size = int(cfg["variance_predictor_kernel_size"])

    tt_block = T2uDecoder(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        state_dict=state_dict,
        char_positional_weights=char_positional_weights,
        positional_weights=positional_weights,
        embed_scale=embed_scale,
        padding_idx=padding_idx,
        eps=eps,
        variance_predictor_kernel_size=variance_predictor_kernel_size,
        conv_kernel_size=conv_kernel_size,
        weight_dtype=ttnn.bfloat16,
    )

    out = tt_block(
        char_input_ids=char_input_ids,
        char_count_per_id=char_count_per_id,
        encoder_hidden_states=encoder_hidden_states,
    )
    out_torch = ttnn.to_torch(out["last_hidden_state"]).to(torch.float32).reshape(ref_last_hidden.shape)

    # Sanity-check dur_out and padding_mask parity before PCC.
    assert torch.equal(
        out["dur_out"], ref_dur_out
    ), f"dur_out mismatch: tt={out['dur_out'].tolist()} ref={ref_dur_out.tolist()}"
    assert torch.equal(
        out["padding_mask"].to(ref_padding_mask.dtype), ref_padding_mask
    ), f"padding_mask mismatch: tt={out['padding_mask']} ref={ref_padding_mask}"

    passing, pcc_message = comp_pcc(ref_last_hidden, out_torch, 0.99)
    print(f"comp_pcc(t2u_decoder.last_hidden_state): passing={passing}, message={pcc_message}")
    return _pcc_from_message(passing, pcc_message)


def test_tt_t2u_decoder():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        pcc = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    assert pcc > 0.99, f"PCC {pcc} <= 0.99"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    pcc = float("nan")
    try:
        pcc = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    result = {
        "pcc": pcc,
        "passed": pcc > 0.99,
    }
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
