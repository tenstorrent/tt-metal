# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 NLLB-style text decoder TTNN block.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/text_decoder.pt``,
runs the TTNN ``TextDecoder`` block on the open p150 (blackhole) device for
both the unmasked and masked paths, and asserts PCC > 0.99 against the saved
reference outputs (the worst of the two is reported).

The golden uses ``decoder_layers=2``. The per-layer block is separately
verified by ``test_tt_text_decoder_layer.py``; this test exercises the
embedding + positional + mask plumbing + final LayerNorm stack around the
existing layers, plus the cross-attention mask/src-len tile padding path.

Can also be run as a standalone script
(``python test_tt_text_decoder.py``) which opens its own device, runs the
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
from models.demos.facebook_seamless_m4t_v2_large.tt.text_decoder import TextDecoder

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "text_decoder.pt"


def _pcc_from_message(passing: bool, pcc_message) -> float:
    msg_str = str(pcc_message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_pcc(device) -> Tuple[float, float]:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    input_ids: torch.Tensor = golden["input_ids"]  # [B, T] int64
    encoder_hidden_states: torch.Tensor = golden["encoder_hidden_states"]  # [B, S, H]
    decoder_attention_mask: torch.Tensor = golden["decoder_attention_mask"]  # [B, T]
    encoder_attention_mask: torch.Tensor = golden["encoder_attention_mask"]  # [B, S]
    state_dict = golden["state_dict"]
    ref_unmasked: torch.Tensor = golden["output_unmasked"]
    ref_masked: torch.Tensor = golden["output_masked"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    eps = float(cfg["eps"])
    padding_idx = int(cfg["padding_idx"])

    tt_block = TextDecoder(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        embed_tokens_weight=state_dict["embed_tokens"]["weight"],
        embed_positions_weights=state_dict["embed_positions_weights"],
        layers_state_dict=state_dict["layers"],
        final_layer_norm_state_dict=state_dict["layer_norm"],
        eps=eps,
        padding_idx=padding_idx,
        weight_dtype=ttnn.bfloat16,
    )

    def _run_path(
        attn_mask_2d,
        enc_mask_2d,
        ref: torch.Tensor,
        label: str,
    ) -> float:
        out_tt = tt_block(
            input_ids,
            encoder_hidden_states_torch=encoder_hidden_states,
            attention_mask=attn_mask_2d,
            encoder_attention_mask=enc_mask_2d,
        )
        out_torch = ttnn.to_torch(out_tt).to(torch.float32).reshape(ref.shape)
        passing, pcc_message = comp_pcc(ref, out_torch, 0.99)
        print(f"comp_pcc({label}): passing={passing}, message={pcc_message}")
        return _pcc_from_message(passing, pcc_message)

    pcc_unmasked = _run_path(None, None, ref_unmasked, "unmasked")
    pcc_masked = _run_path(
        decoder_attention_mask,
        encoder_attention_mask,
        ref_masked,
        "masked",
    )
    return pcc_unmasked, pcc_masked


def test_tt_text_decoder():
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
