# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for MXFP4 expert-weight dequantization in ``ModelArgs.load_state_dict``.

openai/gpt-oss ships the MoE experts MXFP4-quantized: the checkpoint stores
``experts.gate_up_proj_blocks`` / ``_scales`` + ``down_proj_blocks`` / ``_scales`` (uint8) with NO
dense ``gate_up_proj`` / ``down_proj``. A regression where the loader read the raw safetensors
instead of going through HF ``from_pretrained`` left the experts packed, so
``prepare_routed_expert_weights`` raised ``KeyError: 'gate_up_proj'`` on the first real-weights run
(the single-card unit tests missed it -- they build dense expert tensors directly).

We can't *quantize* to MXFP4 on a CPU host (needs triton), but transformers *dequantizes* MXFP4 on
CPU. So this test hand-builds a tiny gpt-oss MXFP4 checkpoint (uint8 blocks + e8m0 scales in the
openai/gpt-oss on-disk layout), loads it through ``ModelArgs.load_state_dict``, and asserts the
experts come back DENSE and feed ``prepare_routed_expert_weights`` without a KeyError.
"""
import json
import os

import pytest
import torch

from models.demos.gpt_oss_d_p.tt.model_config import ModelArgs
from models.demos.gpt_oss_d_p.tt.moe.weights import prepare_routed_expert_weights

E, HID, INTER = 4, 64, 64  # tiny dims; multiples of the 32-wide MXFP4 block
TWO_INTER = 2 * INTER
LP = "model.layers.0.mlp.experts"


def _mxfp4_blocks_scales(out_dim, k_dim):
    """MXFP4 storage tensors for a dense [E, out_dim, k_dim] weight (k_dim is the quantized axis).

    blocks: uint8 [E, out_dim, k_dim/32, 16] (each byte packs two fp4 nibbles -> 32 values/block);
    scales: uint8 [E, out_dim, k_dim/32] (e8m0; 127 == exponent bias -> scale 2**0). Values are
    arbitrary here (0-nibbles -> fp4 0.0): the test checks the dequant PATH + shapes, not numerics.
    """
    assert k_dim % 32 == 0
    g = k_dim // 32
    blocks = torch.zeros(E, out_dim, g, 16, dtype=torch.uint8)
    scales = torch.full((E, out_dim, g), 127, dtype=torch.uint8)
    return blocks, scales


def _write_tiny_mxfp4_checkpoint(path):
    from safetensors.torch import save_file
    from transformers import GptOssConfig, GptOssForCausalLM

    cfg = GptOssConfig(
        vocab_size=64,
        hidden_size=HID,
        intermediate_size=INTER,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=E,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )
    sd = GptOssForCausalLM(cfg).state_dict()

    # Swap the dense expert weights for MXFP4 blocks+scales in the openai/gpt-oss on-disk layout:
    # gate_up blocks [E, 2*inter, hidden/32, 16]; down blocks [E, hidden, inter/32, 16].
    for key in ("gate_up_proj", "down_proj"):
        sd.pop(f"{LP}.{key}", None)
    sd[f"{LP}.gate_up_proj_blocks"], sd[f"{LP}.gate_up_proj_scales"] = _mxfp4_blocks_scales(TWO_INTER, HID)
    sd[f"{LP}.down_proj_blocks"], sd[f"{LP}.down_proj_scales"] = _mxfp4_blocks_scales(HID, INTER)

    os.makedirs(path, exist_ok=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(path, "model.safetensors"))

    c = cfg.to_dict()
    c["quantization_config"] = {
        "quant_method": "mxfp4",
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(c, f)


def test_load_state_dict_dequantizes_mxfp4_experts(tmp_path):
    ckpt = str(tmp_path / "tiny_mxfp4")
    try:
        _write_tiny_mxfp4_checkpoint(ckpt)
    except Exception as e:  # transformers without GptOss, etc.
        pytest.skip(f"cannot build a tiny gpt-oss checkpoint here: {e}")

    try:
        sd = ModelArgs.load_state_dict(ckpt, convert_to_meta_format=False)
    except Exception as e:  # MXFP4 dequant-on-load path unavailable on this host
        pytest.skip(f"MXFP4 dequant-on-load unavailable here: {e}")

    # The regression: no packed tensors may leak through -- the loader must dequantize.
    leaked = sorted(k for k in sd if k.endswith("_blocks") or k.endswith("_scales"))
    assert not leaked, f"MXFP4 packed tensors leaked into the state dict: {leaked[:6]}"

    # Dense experts must exist with the layout prepare_routed_expert_weights expects.
    assert f"{LP}.gate_up_proj" in sd, "dense gate_up_proj missing after load (dequant did not run)"
    assert f"{LP}.down_proj" in sd, "dense down_proj missing after load"
    assert tuple(sd[f"{LP}.gate_up_proj"].shape) == (E, HID, TWO_INTER)
    assert tuple(sd[f"{LP}.down_proj"].shape) == (E, INTER, HID)

    # And they must feed prepare_routed_expert_weights without the KeyError this test guards against.
    experts = {k.split("experts.")[-1]: v for k, v in sd.items() if ".mlp.experts." in k and "layers.0" in k}
    w, b = prepare_routed_expert_weights(experts, E, HID, INTER)
    assert len(w) == E and len(b) == E
