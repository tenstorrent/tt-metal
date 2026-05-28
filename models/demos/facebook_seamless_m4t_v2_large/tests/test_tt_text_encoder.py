# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 NLLB-style text encoder TTNN block.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/text_encoder.pt``,
runs the TTNN ``TextEncoder`` block on the open p150 (blackhole) device
for both the unmasked and masked paths, and asserts PCC > 0.99 against the
saved reference outputs (the worst of the two is reported).

The golden uses ``encoder_layers=2`` (the saved state_dict has 2 layers); the
full model has 24, but per-layer correctness against HF is already covered by
``test_tt_text_encoder_layer.py``. The 2-layer config still exercises the
embed + positional + stacked-layers + final-LN composition.

The reference golden stores only the embedding-table rows actually indexed by
the test ``input_ids`` (``state_dict["embed_tokens_rows_used"]``, keyed by
token id) — it does NOT store the full vocab×hidden table (which would be
~1 GB at fp32). We rebuild a small embed table here that exactly reproduces
the rows the input_ids touch (zeros everywhere else): this is sufficient for
PCC because the gather only ever indexes those rows.

Can also be run as a standalone script
(``python test_tt_text_encoder.py``) which opens its own device, runs the
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
from models.demos.facebook_seamless_m4t_v2_large.tt.text_encoder import TextEncoder

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "text_encoder.pt"


def _pcc_from_message(passing: bool, pcc_message) -> float:
    msg_str = str(pcc_message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _rebuild_embed_table(
    embed_tokens_rows_used: dict,
    hidden_size: int,
) -> torch.Tensor:
    """Build a minimal embed table that reproduces the rows the golden uses.

    The reference test saved ``{tok_id: row}`` for every unique id in
    ``input_ids`` (to keep the golden small — the real vocab is 256k×1024).
    We reconstruct a table where row ``tok_id`` matches the saved row and all
    other rows are zero. The encoder only ever indexes the saved rows, so
    PCC is unaffected by the zero filler.
    """
    if not embed_tokens_rows_used:
        raise ValueError("embed_tokens_rows_used is empty; cannot rebuild embed table")
    max_id = max(int(k) for k in embed_tokens_rows_used.keys())
    # Allocate enough rows to cover the highest id we'll look up. dtype is
    # taken from one of the stored rows so we match exactly bit-for-bit.
    sample_row = next(iter(embed_tokens_rows_used.values()))
    table = torch.zeros((max_id + 1, hidden_size), dtype=sample_row.dtype)
    for tok_id, row in embed_tokens_rows_used.items():
        table[int(tok_id)] = row
    return table


def _run_pcc(device) -> Tuple[float, float]:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    input_ids: torch.Tensor = golden["input_ids"]  # [B, T] int64
    attention_mask_4d: torch.Tensor = golden["attention_mask_4d"]  # [B, 1, T, T] float32
    state_dict = golden["state_dict"]
    ref_unmasked: torch.Tensor = golden["output_unmasked"]
    ref_masked: torch.Tensor = golden["output_masked"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    eps = float(cfg["eps"])
    padding_idx = int(cfg["padding_idx"])
    embed_scale = float(cfg["embed_scale"])

    # Rebuild the (small) embed table covering the rows our input_ids touch.
    embed_tokens_weight = _rebuild_embed_table(
        state_dict["embed_tokens_rows_used"],
        hidden_size=embed_dim,
    )

    tt_block = TextEncoder(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        embed_tokens_weight=embed_tokens_weight,
        embed_positions_weights=state_dict["embed_positions_weights"],
        layers_state_dict=state_dict["layers"],
        final_layer_norm_state_dict=state_dict["final_layer_norm"],
        eps=eps,
        padding_idx=padding_idx,
        embed_scale=embed_scale,
        weight_dtype=ttnn.bfloat16,
    )

    def _run_path(attn_mask: torch.Tensor | None, ref: torch.Tensor) -> float:
        # Pass mask via the host-side ``attention_mask_torch`` entry so the
        # block can tile-pad with the additive log "minus" fill. SDPA reads
        # the mask in 32x32 tiles and a sub-tile mask leaks stale bf16
        # values into the softmax without explicit padding.
        out_tt = tt_block(input_ids, attention_mask_torch=attn_mask)
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


def test_tt_text_encoder():
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
