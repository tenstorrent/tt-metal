# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 top-level T2TT TTNN composition.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/seamless_m4t_v2.pt``,
runs the TTNN :class:`SeamlessM4Tv2` composition on the open p150 (blackhole)
device for both the unmasked and masked paths, and asserts PCC > 0.99 against
the saved reference outputs (the worst of the two is reported).

The golden uses ``encoder_layers=2, decoder_layers=2`` (per-layer blocks are
separately verified). This test exercises the top-level
``text_encoder -> text_decoder -> lm_head`` plumbing.

The golden file does NOT save the full state_dict — at hidden=1024,
vocab=256102 the embed table alone is ~1 GB at fp32. Instead, we
reconstruct the HF ``SeamlessM4Tv2ForTextToText`` module at runtime
(with the same ``torch.manual_seed(0)`` + ``encoder_layers=2,
decoder_layers=2``) used to build the golden, and pull the nested
state_dict out of it. This matches the approach in
``reference/test_functional_seamless_m4t_v2.py``.

Can also be run as a standalone script
(``python test_tt_seamless_m4t_v2.py``) which opens its own device, runs
the PCC comparison, prints the result and exits 0 on pass / 1 on fail.
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
from models.demos.facebook_seamless_m4t_v2_large.tt.seamless_m4t_v2 import SeamlessM4Tv2

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "seamless_m4t_v2.pt"


def _pcc_from_message(passing: bool, pcc_message) -> float:
    msg_str = str(pcc_message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _ln_sd(ln) -> dict:
    return {
        "weight": ln.weight.detach().clone(),
        "bias": ln.bias.detach().clone(),
    }


def _linear_sd(lin) -> dict:
    sd = {"weight": lin.weight.detach().clone()}
    if lin.bias is not None:
        sd["bias"] = lin.bias.detach().clone()
    return sd


def _attn_sd(attn) -> dict:
    return {
        "q_proj": _linear_sd(attn.q_proj),
        "k_proj": _linear_sd(attn.k_proj),
        "v_proj": _linear_sd(attn.v_proj),
        "out_proj": _linear_sd(attn.out_proj),
    }


def _ffn_sd(ffn) -> dict:
    return {
        "fc1": _linear_sd(ffn.fc1),
        "fc2": _linear_sd(ffn.fc2),
    }


def _extract_text_encoder_layer_state_dict(layer) -> dict:
    return {
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "self_attn": _attn_sd(layer.self_attn),
        "ffn_layer_norm": _ln_sd(layer.ffn_layer_norm),
        "ffn": _ffn_sd(layer.ffn),
    }


def _extract_text_decoder_layer_state_dict(layer) -> dict:
    return {
        "self_attn_layer_norm": _ln_sd(layer.self_attn_layer_norm),
        "self_attn": _attn_sd(layer.self_attn),
        "cross_attention_layer_norm": _ln_sd(layer.cross_attention_layer_norm),
        "cross_attention": _attn_sd(layer.cross_attention),
        "ffn_layer_norm": _ln_sd(layer.ffn_layer_norm),
        "ffn": _ffn_sd(layer.ffn),
    }


def _extract_text_encoder_state_dict(encoder) -> dict:
    return {
        "embed_tokens": {"weight": encoder.embed_tokens.weight.detach().clone()},
        "embed_positions_weights": encoder.embed_positions.weights.detach().clone(),
        "layers": [_extract_text_encoder_layer_state_dict(layer) for layer in encoder.layers],
        "final_layer_norm": _ln_sd(encoder.layer_norm),
    }


def _extract_text_decoder_state_dict(decoder) -> dict:
    return {
        "embed_tokens": {"weight": decoder.embed_tokens.weight.detach().clone()},
        "embed_positions_weights": decoder.embed_positions.weights.detach().clone(),
        "layers": [_extract_text_decoder_layer_state_dict(layer) for layer in decoder.layers],
        "layer_norm": _ln_sd(decoder.layer_norm),
    }


def _build_hf_state_dict():
    """Rebuild the HF model deterministically and extract the state dicts.

    Uses the same seed + config tweaks as
    ``reference/test_functional_seamless_m4t_v2.py``, so the resulting
    state dict bit-matches the one that produced the golden tensors.
    """
    from transformers import SeamlessM4Tv2Config
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ForTextToText

    torch.manual_seed(0)

    config = SeamlessM4Tv2Config()
    config.encoder_layers = 2
    config.decoder_layers = 2

    hf = SeamlessM4Tv2ForTextToText(config)
    hf.eval()
    # Zero out every dropout so a behavioural change in dropout-during-eval
    # wouldn't surprise us (matches reference test).
    hf.text_encoder.dropout = 0.0
    for layer in hf.text_encoder.layers:
        layer.attn_dropout.p = 0.0
        layer.ffn_dropout.p = 0.0
        layer.self_attn.dropout = 0.0
        layer.ffn.dropout.p = 0.0
    hf.text_decoder.dropout = 0.0
    for layer in hf.text_decoder.layers:
        layer.attn_dropout.p = 0.0
        layer.ffn_dropout.p = 0.0
        layer.self_attn.dropout = 0.0
        layer.cross_attention.dropout = 0.0
        layer.ffn.dropout.p = 0.0

    return {
        "text_encoder": _extract_text_encoder_state_dict(hf.text_encoder),
        "text_decoder": _extract_text_decoder_state_dict(hf.text_decoder),
        "lm_head": _linear_sd(hf.lm_head),
    }


def _run_pcc(device) -> Tuple[float, float]:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    input_ids: torch.Tensor = golden["input_ids"]  # [B, S]
    decoder_input_ids: torch.Tensor = golden["decoder_input_ids"]  # [B, T]
    encoder_attention_mask: torch.Tensor = golden["encoder_attention_mask"]  # [B, S]
    decoder_attention_mask: torch.Tensor = golden["decoder_attention_mask"]  # [B, T]
    ref_unmasked_logits: torch.Tensor = golden["logits_unmasked"]
    ref_masked_logits: torch.Tensor = golden["logits_masked"]
    cfg = golden["config"]

    embed_dim = int(cfg["hidden"])
    num_heads = int(cfg["num_heads"])
    head_dim = int(cfg["head_dim"])
    eps = float(cfg["eps"])
    encoder_padding_idx = int(cfg["encoder_padding_idx"])
    decoder_padding_idx = int(cfg["decoder_padding_idx"])

    state_dict = _build_hf_state_dict()

    tt_block = SeamlessM4Tv2(
        device=device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        text_encoder_state_dict=state_dict["text_encoder"],
        text_decoder_state_dict=state_dict["text_decoder"],
        lm_head_state_dict=state_dict["lm_head"],
        eps=eps,
        encoder_padding_idx=encoder_padding_idx,
        decoder_padding_idx=decoder_padding_idx,
        weight_dtype=ttnn.bfloat16,
    )

    def _run_path(
        attn_mask_2d,
        dec_mask_2d,
        ref: torch.Tensor,
        label: str,
    ) -> float:
        out_tt = tt_block(
            input_ids,
            decoder_input_ids,
            attention_mask=attn_mask_2d,
            decoder_attention_mask=dec_mask_2d,
        )
        out_torch = ttnn.to_torch(out_tt).to(torch.float32).reshape(ref.shape)
        passing, pcc_message = comp_pcc(ref, out_torch, 0.99)
        print(f"comp_pcc({label}): passing={passing}, message={pcc_message}")
        return _pcc_from_message(passing, pcc_message)

    pcc_unmasked = _run_path(None, None, ref_unmasked_logits, "unmasked")
    pcc_masked = _run_path(
        encoder_attention_mask,
        decoder_attention_mask,
        ref_masked_logits,
        "masked",
    )
    return pcc_unmasked, pcc_masked


def test_tt_seamless_m4t_v2():
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
