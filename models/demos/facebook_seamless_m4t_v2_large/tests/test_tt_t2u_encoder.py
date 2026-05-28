# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 T2U encoder TTNN block.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/t2u_encoder.pt``,
runs the TTNN ``T2uEncoder`` block on the open p150 (blackhole) device
for both the unmasked and masked paths, and asserts PCC > 0.99 against the
saved reference outputs (the worst of the two is reported).

The golden uses ``encoder_layers=2`` (the saved state_dict has 2 layers); the
full T2U encoder has 6, but per-layer correctness against HF is already
covered by ``test_tt_text_encoder_layer.py`` (the T2U encoder layer is
structurally identical to the text encoder layer). The 2-layer config still
exercises the stacked-layers + final-LN composition.

The T2U encoder consumes pre-embedded ``inputs_embeds`` of shape
``[B, T, hidden]`` directly -- no token lookup, no positional embed. The
golden therefore stores the embeds (and the resulting outputs) only; there
is no embed-tokens table to reconstruct.

Can also be run as a standalone script
(``python test_tt_t2u_encoder.py``) which opens its own device, runs the
PCC comparison, prints the result and exits 0 on pass / 1 on fail.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Tuple

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_encoder import T2uEncoder

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "t2u_encoder.pt"


def _pcc_from_message(passing: bool, pcc_message) -> float:
    msg_str = str(pcc_message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_pcc(device) -> Tuple[float, float]:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    inputs_embeds: torch.Tensor = golden["inputs_embeds"]  # [B, T, hidden] float32
    attention_mask_4d: torch.Tensor = golden["attention_mask_4d"]  # [B, 1, T, T] float32
    state_dict = golden["state_dict"]
    ref_unmasked: torch.Tensor = golden["output_unmasked"]
    ref_masked: torch.Tensor = golden["output_masked"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    eps = float(cfg["eps"])

    tt_block = T2uEncoder(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        layers_state_dict=state_dict["layers"],
        final_layer_norm_state_dict=state_dict["final_layer_norm"],
        eps=eps,
        weight_dtype=ttnn.bfloat16,
    )

    def _run_path(attn_mask: torch.Tensor | None, ref: torch.Tensor) -> float:
        # Pass mask via the host-side ``attention_mask_torch`` entry so the
        # block can tile-pad with the additive log "minus" fill. SDPA reads
        # the mask in 32x32 tiles and a sub-tile mask leaks stale bf16
        # values into the softmax without explicit padding.
        out_tt = tt_block(inputs_embeds, attention_mask_torch=attn_mask)
        out_torch = ttnn.to_torch(out_tt).to(torch.float32).reshape(ref.shape)
        passing, pcc_message = comp_pcc(ref, out_torch, 0.99)
        print(
            f"comp_pcc(attn_mask={'yes' if attn_mask is not None else 'no'}):"
            f" passing={passing}, message={pcc_message}"
        )
        return _pcc_from_message(passing, pcc_message)

    pcc_unmasked = _run_path(None, ref_unmasked)
    pcc_masked = _run_path(attention_mask_4d, ref_masked)
    return pcc_unmasked, pcc_masked


def test_tt_t2u_encoder():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        pcc_unmasked, pcc_masked = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    worst = min(pcc_unmasked, pcc_masked)
    assert worst > 0.99, f"PCC unmasked={pcc_unmasked}, masked={pcc_masked}; worst={worst} <= 0.99"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    pcc_unmasked = float("nan")
    pcc_masked = float("nan")
    try:
        pcc_unmasked, pcc_masked = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    worst = min(pcc_unmasked, pcc_masked)
    result = {
        "pcc_unmasked": pcc_unmasked,
        "pcc_masked": pcc_masked,
        "pcc": worst,
        "passed": worst > 0.99,
    }
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
