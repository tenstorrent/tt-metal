# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the MXFP8 / NVFP4 checkpoint dequantization (utils/checkpoint.py).

Synthetic tensors are quantized with a small reference encoder written from the format specs, then
dequantized and compared. With HF_MODEL pointing at a quantized checkpoint, one MXFP8 linear and one
NVFP4 expert are also dequantized from it and range-checked.
"""

import json
import math
import os

import pytest
import torch
from safetensors.torch import save_file

from models.demos.minimax_m3.reference.model import SafetensorsWeights
from models.demos.minimax_m3.utils.checkpoint import (
    E2M1_VALUES,
    check_supported_sparse_attention,
    dequant_mxfp8,
    dequant_nvfp4,
    iter_dequantized,
    read_text_backbone,
    text_config_of,
    unpack_e2m1,
)
from models.demos.minimax_m3.utils.model_variant import load_config

FP8_MAX = 448.0
FP4_MAX = 6.0


# ---- reference encoders (format specs, independent of the code under test) -----------------------------
def quantize_mxfp8(w: torch.Tensor, block: int = 32):
    """Per 32-element block along K: UE8M0 scale 2^e with e = ceil(log2(amax / 448)), fp8 = w / 2^e."""
    n, k = w.shape
    blocks = w.float().view(n, k // block, block)
    amax = blocks.abs().amax(-1, keepdim=True).clamp(min=2.0**-100)
    e = torch.ceil(torch.log2(amax / FP8_MAX))
    fp8 = (blocks / torch.exp2(e)).to(torch.float8_e4m3fn).view(n, k)
    return fp8, (e.squeeze(-1) + 127).to(torch.uint8)


def round_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 code (sign in bit 3) for each value of ``x`` (already in [-6, 6])."""
    mags = torch.tensor(E2M1_VALUES[:8])
    idx = (x.abs().unsqueeze(-1) - mags).abs().argmin(-1)
    return idx | ((x < 0).to(torch.long) << 3)


def pack_low_nibble_first(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def quantize_nvfp4(w: torch.Tensor, group: int = 16):
    """compressed-tensors NVFP4: global = 448*6/amax, per-16 fp8 scale = group_amax/6 * global,
    codes = e2m1(w * global / scale), two codes per byte (element 2j low nibble)."""
    n, k = w.shape
    w = w.float()
    global_scale = torch.tensor(FP8_MAX * FP4_MAX / w.abs().max().item(), dtype=torch.float32)
    groups = w.view(n, k // group, group)
    scale = (groups.abs().amax(-1) / FP4_MAX * global_scale).to(torch.float8_e4m3fn)
    scaled = groups * global_scale / scale.float().clamp(min=2.0**-9).unsqueeze(-1)
    codes = round_to_e2m1_codes(scaled.clamp(-FP4_MAX, FP4_MAX)).view(n, k)
    return pack_low_nibble_first(codes), scale, global_scale


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# ---- format primitives ----------------------------------------------------------------------------------
def test_unpack_e2m1_table_and_nibble_order():
    packed = torch.tensor([[0x21, 0xF8, 0x7A]], dtype=torch.uint8)
    # 0x21: low 1 (0.5), high 2 (1.0); 0xF8: low 8 (-0), high 15 (-6); 0x7A: low 10 (-1), high 7 (6)
    expected = torch.tensor([[0.5, 1.0, -0.0, -6.0, -1.0, 6.0]])
    assert torch.equal(unpack_e2m1(packed), expected)


def test_mxfp8_exact_on_representable_values():
    torch.manual_seed(0)
    n, k = 4, 64
    fp8 = (torch.randn(n, k) * 100).to(torch.float8_e4m3fn)
    e = torch.randint(127 - 20, 127 + 5, (n, k // 32), dtype=torch.uint8)
    out = dequant_mxfp8(fp8, e)
    expected = fp8.float().view(n, 2, 32) * torch.exp2(e.float() - 127).unsqueeze(-1)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, expected.view(n, k).to(torch.bfloat16))


def test_mxfp8_round_trip():
    torch.manual_seed(1)
    w = (torch.randn(96, 256) * 0.02).to(torch.bfloat16)
    fp8, scale = quantize_mxfp8(w)
    out = dequant_mxfp8(fp8, scale)
    assert out.shape == w.shape and out.dtype == torch.bfloat16
    assert rel_err(out, w) < 0.04  # e4m3 carries 3 mantissa bits
    # a wrong exponent bias would be off by a power of two
    assert abs(out.float().std().item() / w.float().std().item() - 1.0) < 0.02


def test_nvfp4_exact_on_representable_values():
    torch.manual_seed(2)
    n, k = 3, 32
    codes = torch.randint(0, 16, (n, k))
    scale = (torch.rand(n, k // 16) * 400 + 1).to(torch.float8_e4m3fn)
    global_scale = torch.tensor(1234.5)
    out = dequant_nvfp4(pack_low_nibble_first(codes), scale, global_scale)
    values = torch.tensor(E2M1_VALUES)[codes].view(n, 2, 16)
    expected = values * scale.float().unsqueeze(-1) / global_scale
    assert torch.equal(out, expected.view(n, k).to(torch.bfloat16))


def test_nvfp4_round_trip():
    torch.manual_seed(3)
    w = (torch.randn(64, 512) * 0.03).to(torch.bfloat16)
    packed, scale, global_scale = quantize_nvfp4(w)
    out = dequant_nvfp4(packed, scale, global_scale)
    assert out.shape == w.shape and out.dtype == torch.bfloat16
    assert rel_err(out, w) < 0.15  # 4-bit values, 16-element groups
    assert abs(out.float().std().item() / w.float().std().item() - 1.0) < 0.05
    # swapped nibbles must not reproduce the weight
    swapped = ((packed & 0x0F) << 4) | (packed >> 4)
    assert rel_err(dequant_nvfp4(swapped, scale, global_scale), w) > 0.5


@pytest.mark.parametrize(
    "fn, args",
    [
        (dequant_mxfp8, (torch.zeros(2, 64).to(torch.float8_e4m3fn), torch.zeros(2, 3, dtype=torch.uint8))),
        (dequant_mxfp8, (torch.zeros(2, 64), torch.zeros(2, 2, dtype=torch.uint8))),
        (
            dequant_nvfp4,
            (torch.zeros(2, 16, dtype=torch.uint8), torch.zeros(2, 4).to(torch.float8_e4m3fn), torch.tensor(1.0)),
        ),
        (dequant_nvfp4, (torch.zeros(2, 16, dtype=torch.uint8), torch.zeros(2, 2), torch.tensor(1.0))),
    ],
    ids=["mxfp8_scale_shape", "mxfp8_weight_dtype", "nvfp4_scale_shape", "nvfp4_scale_dtype"],
)
def test_dequant_rejects_mismatched_inputs(fn, args, expect_error):
    with expect_error(ValueError, "must be|does not match"):
        fn(*args)


# ---- key rewriting --------------------------------------------------------------------------------------
P = "language_model.model.layers.0."


def _synthetic_checkpoint(tmp_path):
    """Two shards; one MXFP8 scale and one NVFP4 scale sit in the other shard than their weight."""
    torch.manual_seed(4)
    q_w = torch.randn(128, 64, dtype=torch.bfloat16) * 0.02
    e_w = torch.randn(32, 64, dtype=torch.bfloat16) * 0.02
    q_fp8, q_scale = quantize_mxfp8(q_w)
    e_packed, e_scale, e_gs = quantize_nvfp4(e_w)
    ex = P + "block_sparse_moe.experts.0.w1."
    shard_a = {
        P + "self_attn.q_proj.weight": q_fp8,
        P + "input_layernorm.weight": torch.ones(64, dtype=torch.bfloat16),
        P + "block_sparse_moe.gate.weight": torch.randn(8, 64, dtype=torch.float32),
        ex + "weight_packed": e_packed,
        ex + "weight_global_scale": e_gs,
        "vision_tower.blocks.0.weight": torch.randn(4, 4),
    }
    shard_b = {
        P + "self_attn.q_proj.weight_scale_inv": q_scale,
        ex + "weight_scale": e_scale,
        ex + "input_global_scale": torch.tensor(17.0),
        "language_model.lm_head.weight": torch.randn(16, 64, dtype=torch.bfloat16),
    }
    weight_map = {}
    for name, shard in (("a.safetensors", shard_a), ("b.safetensors", shard_b)):
        save_file(shard, str(tmp_path / name))
        weight_map.update({k: name for k in shard})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return weight_map, q_fp8, q_scale, (e_packed, e_scale, e_gs)


def test_read_text_backbone_rewrites_keys_and_drops_aux(tmp_path):
    weight_map, q_fp8, q_scale, nvfp4 = _synthetic_checkpoint(tmp_path)
    sd = read_text_backbone(tmp_path, weight_map, ["a.safetensors", "b.safetensors"])
    assert set(sd) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "lm_head.weight",
    }
    gate = "model.layers.0.block_sparse_moe.gate.weight"
    assert sd[gate].dtype == torch.float32  # the MoE router stays fp32 (test_router_precision.py)
    assert all(t.dtype == torch.bfloat16 for k, t in sd.items() if k != gate)
    assert torch.equal(sd["model.layers.0.self_attn.q_proj.weight"], dequant_mxfp8(q_fp8, q_scale))
    assert torch.equal(sd["model.layers.0.block_sparse_moe.experts.0.w1.weight"], dequant_nvfp4(*nvfp4))
    # a shard subset reads only its own keys but still finds scales in the other shard
    assert set(read_text_backbone(tmp_path, weight_map, ["a.safetensors"])) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
    }


def test_iter_dequantized_passes_plain_tensors_through():
    plain = {"x.weight": torch.randn(2, 2, dtype=torch.bfloat16), "x.bias": torch.randn(2)}
    out = dict(iter_dequantized(plain, plain.__getitem__))
    assert out.keys() == plain.keys() and all(out[k] is plain[k] for k in plain)


def test_reference_weights_read_quantized_checkpoint(tmp_path):
    _, q_fp8, q_scale, nvfp4 = _synthetic_checkpoint(tmp_path)
    w = SafetensorsWeights(tmp_path, dtype=torch.float32)
    expert = "model.layers.0.block_sparse_moe.experts.0.w1.weight"
    assert w.has(expert) and w.has("model.layers.0.self_attn.q_proj.weight")
    assert not w.has("model.layers.0.block_sparse_moe.experts.1.w1.weight")
    assert torch.equal(w.get(expert), dequant_nvfp4(*nvfp4).float())
    assert torch.equal(w.get("model.layers.0.self_attn.q_proj.weight"), dequant_mxfp8(q_fp8, q_scale).float())


# ---- config ---------------------------------------------------------------------------------------------
def test_text_config_of_wraps_dict():
    class Wrapper:
        text_config = load_config("MiniMax-M3.1")

    tc = text_config_of(Wrapper())
    assert (
        tc.hidden_size == 6144
        and tc.rope_theta == 1e7
        and tc.sparse_attention_config["sparse_attention_freq"] == [1] * 60
    )


@pytest.mark.parametrize(
    "cfg, ok",
    [
        (
            {"sparse_attention_freq": [1, 1], "sparse_disable_index_value": [1, 1], "sparse_enable_kv_share": False},
            True,
        ),
        ({}, True),
        ({"sparse_attention_freq": [0, 1], "sparse_disable_index_value": [0, 1]}, True),  # dense layer: flag unused
        ({"sparse_attention_freq": [1, 1], "sparse_disable_index_value": [1, 0]}, False),
        ({"sparse_enable_kv_share": True}, False),
        ({"sparse_enable_vo_rope": True}, False),
    ],
)
def test_check_supported_sparse_attention(cfg, ok, expect_error):
    if ok:
        check_supported_sparse_attention(cfg)
    else:
        with expect_error(NotImplementedError, "unsupported sparse attention options"):
            check_supported_sparse_attention(cfg)


# ---- real checkpoint (optional) -------------------------------------------------------------------------
def _quantized_checkpoint():
    path = os.getenv("HF_MODEL")
    if not path or not os.path.isfile(os.path.join(path, "config.json")):
        return None
    with open(os.path.join(path, "config.json")) as f:
        return path if "quantization_config" in json.load(f) else None


@pytest.mark.skipif(_quantized_checkpoint() is None, reason="HF_MODEL is not a quantized checkpoint")
def test_real_checkpoint_dequant_ranges():
    w = SafetensorsWeights(_quantized_checkpoint(), dtype=torch.float32)
    q = w.get("model.layers.3.self_attn.q_proj.weight")
    assert q.shape == (8192, 6144) and torch.isfinite(q).all()
    assert 0.005 < q.std().item() < 0.2

    expert = "model.layers.3.block_sparse_moe.experts.0.w1."
    e = w.get(expert + "weight")
    assert e.shape == (3072, 6144) and torch.isfinite(e).all()
    assert 0.005 < e.std().item() < 0.2
    # global_scale = 448 * 6 / amax(w), so the dequantized amax sits at (or just under) that bound
    bound = FP8_MAX * FP4_MAX / w._raw(w.prefix + expert + "weight_global_scale").item()
    assert math.isclose(e.abs().max().item(), bound, rel_tol=0.05)
