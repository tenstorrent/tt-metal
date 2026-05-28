# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 NLLB-style text-decoder-layer TTNN block.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/text_decoder_layer.pt``,
runs the TTNN ``TextDecoderLayer`` block on the open p150 (blackhole) device
for the self-only, unmasked self+cross, and masked self+cross paths, and
asserts PCC > 0.99 against the saved reference outputs (the worst of the
three is reported).

Can also be run as a standalone script
(``python test_tt_text_decoder_layer.py``) which opens its own device,
runs the PCC comparison, prints the result and exits 0 on pass / 1 on fail.
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
from models.demos.facebook_seamless_m4t_v2_large.tt.text_decoder_layer import TextDecoderLayer

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "text_decoder_layer.pt"


def _pcc_from_message(passing: bool, pcc_message) -> float:
    msg_str = str(pcc_message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_pcc(device) -> Tuple[float, float, float]:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]  # [B, T, C] float32
    enc_torch: torch.Tensor = golden["encoder_hidden_states"]  # [B, S, C] float32
    self_attention_mask_torch: torch.Tensor = golden["self_attention_mask"]  # [B,1,T,T]
    encoder_attention_mask_torch: torch.Tensor = golden["encoder_attention_mask"]  # [B,1,T,S]
    state_dict = golden["state_dict"]
    ref_self_only: torch.Tensor = golden["output_self_only"]
    ref_unmasked: torch.Tensor = golden["output_unmasked"]
    ref_masked: torch.Tensor = golden["output_masked"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    eps = float(cfg["eps"])

    tt_block = TextDecoderLayer(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        state_dict=state_dict,
        eps=eps,
        weight_dtype=ttnn.bfloat16,
    )

    def _to_tt(t: torch.Tensor, *, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _run_path(
        enc_torch_path,
        self_mask_torch,
        enc_mask_torch,
        ref: torch.Tensor,
        label: str,
    ) -> float:
        x_tt = _to_tt(x_torch)
        enc_tt = _to_tt(enc_torch_path) if enc_torch_path is not None else None
        self_mask_tt = _to_tt(self_mask_torch) if self_mask_torch is not None else None
        enc_mask_tt = _to_tt(enc_mask_torch) if enc_mask_torch is not None else None

        out_tt = tt_block(
            x_tt,
            encoder_hidden_states=enc_tt,
            self_attention_mask=self_mask_tt,
            encoder_attention_mask=enc_mask_tt,
        )
        out_torch = ttnn.to_torch(out_tt).to(torch.float32).reshape(ref.shape)
        passing, pcc_message = comp_pcc(ref, out_torch, 0.99)
        print(f"comp_pcc({label}): passing={passing}, message={pcc_message}")
        return _pcc_from_message(passing, pcc_message)

    pcc_self_only = _run_path(None, None, None, ref_self_only, "self_only")
    pcc_unmasked = _run_path(enc_torch, None, None, ref_unmasked, "unmasked")
    pcc_masked = _run_path(
        enc_torch,
        self_attention_mask_torch,
        encoder_attention_mask_torch,
        ref_masked,
        "masked",
    )
    return pcc_self_only, pcc_unmasked, pcc_masked


def test_tt_text_decoder_layer():
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        pcc_self_only, pcc_unmasked, pcc_masked = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    worst = min(pcc_self_only, pcc_unmasked, pcc_masked)
    assert worst > 0.99, (
        f"PCC self_only={pcc_self_only}, unmasked={pcc_unmasked}, " f"masked={pcc_masked}; worst={worst} <= 0.99"
    )


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    pcc_self_only = float("nan")
    pcc_unmasked = float("nan")
    pcc_masked = float("nan")
    try:
        pcc_self_only, pcc_unmasked, pcc_masked = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    worst = min(pcc_self_only, pcc_unmasked, pcc_masked)
    result = {
        "pcc_self_only": pcc_self_only,
        "pcc_unmasked": pcc_unmasked,
        "pcc_masked": pcc_masked,
        "pcc": worst,
        "passed": worst > 0.99,
    }
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
