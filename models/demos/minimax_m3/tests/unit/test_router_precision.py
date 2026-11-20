# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the MoE router's precision handling (no device).

The vendor routes in fp32: fp32 gate GEMM, sigmoid + fp32 correction bias, top-4 (ties -> lowest expert id),
renormalize, x routed_scaling_factor. The bias sits at ~10-37 with a spread of ~0.05, so the TT path keeps it
fp32 and mean-centered, keeps the gate weight fp32 on the host, and runs the gate GEMM with fp32
accumulation. ``test_selection_agreement_*`` models the device arithmetic in torch on synthetic tensors shaped
like the checkpoint's (bias level / spread and logit scale of layers 3 and 59) and checks the top-4 agreement
with the fp32 rule.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from models.demos.minimax_m3.utils.checkpoint import centered_correction_bias, is_router_fp32_key, read_text_backbone
from models.demos.minimax_m3.utils.general_utils import ROUTER_BIAS_CACHE_NAME, cache_file_exists

E, H, K, SCALE = 128, 6144, 4, 2.0


# ---- number formats / device arithmetic models ----------------------------------------------------------
def _bf16(x):
    return x.float().to(torch.bfloat16).float()


def _tf32_trunc(x):
    """fp32 with the low 13 mantissa bits zeroed: how moe_grouped_topk's FPU reads fp32 tiles (SrcA)."""
    return (x.float().contiguous().view(torch.int32) & ~((1 << 13) - 1)).view(torch.float32)


def _route(logits, bias, *, op_keys):
    """Top-4 ids (sorted) and dense combine weights. op_keys=True models moe_grouped_topk: the sigmoid enters
    the FPU bias add truncated to TF32, and the sort keys are TF32 too. Ties go to the lowest expert id."""
    s = torch.sigmoid(logits.double())
    key = (_tf32_trunc(s) + bias.float()) if op_keys else (s + bias.double())
    if op_keys:
        key = _tf32_trunc(key)
    ids = torch.sort(key.double(), dim=-1, descending=True, stable=True).indices[:, :K]
    w = torch.gather(s, 1, ids)
    w = w / w.sum(-1, keepdim=True) * SCALE
    return ids.sort(-1).values, torch.zeros_like(s).scatter_(1, ids, w)


def _gemm_default_ttnn(h, w):
    """ttnn.linear's default for bf16 x bf16: HiFi2 (activations lose their low mantissa bit), bf16 running sum
    rounded after every 32-wide K tile, bf16 logits."""
    a = (h.float().contiguous().view(torch.int32) & ~((1 << 17) - 1)).view(torch.float32)
    wb = _bf16(w)
    acc = torch.zeros(h.shape[0], w.shape[0])
    for k0 in range(0, h.shape[1], 32):
        acc = _bf16(acc + a[:, k0 : k0 + 32] @ wb[:, k0 : k0 + 32].t())
    return acc.double()


def _set_agreement(a, b):
    return (a[0] == b[0]).all(-1).float().mean().item() * 100.0


def _synthetic_layer(bias_level, bias_std, logit_std, tokens=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(E, H, generator=g) * (logit_std / H**0.5)
    bias = bias_level + bias_std * torch.randn(E, generator=g)
    h = _bf16(torch.randn(tokens, H, generator=g))  # bf16 activations, upcast exactly by the vendor
    return h, w, bias


# Layer 3 (bias 16.16..16.47, std 0.057, logit std ~3.8) and layer 59 (bias 36.6..37.0, std 0.049, ~1.5).
LAYERS = {"L3": (16.3, 0.057, 3.8), "L59": (36.8, 0.049, 1.5)}


@pytest.mark.parametrize("layer", LAYERS)
def test_selection_agreement_with_fp32_router(layer):
    h, w, bias = _synthetic_layer(*LAYERS[layer])
    ref = _route(h.double() @ w.double().t(), bias.double(), op_keys=False)

    old = _route(_gemm_default_ttnn(h, w), _bf16(bias), op_keys=True)
    fp32_acc = h.double() @ _bf16(w).double().t()  # bf16 weight, exact products, fp32 accumulation
    new = _route(_tf32_trunc(fp32_acc), centered_correction_bias(bias), op_keys=True)
    uncentered = _route(_tf32_trunc(fp32_acc), bias, op_keys=True)

    old_pct, new_pct, unc_pct = (_set_agreement(r, ref) for r in (old, new, uncentered))
    assert old_pct < 50.0, f"{layer}: bf16 bias + default GEMM agrees on {old_pct:.1f}% (expected far below)"
    assert unc_pct < 90.0, f"{layer}: an un-centered fp32 bias agrees on {unc_pct:.1f}% (expected key-limited)"
    assert new_pct > 95.0, f"{layer}: centered fp32 bias + fp32-acc GEMM agrees on only {new_pct:.1f}%"


def test_centered_bias_keeps_selection_and_weights():
    h, w, bias = _synthetic_layer(*LAYERS["L3"], tokens=1024)
    logits = h.double() @ w.double().t()
    centered = centered_correction_bias(bias)
    assert centered.dtype == torch.float32
    assert abs(centered.double().mean().item()) < 1e-6
    ids, weights = _route(logits, bias.double(), op_keys=False)
    ids_c, weights_c = _route(logits, centered.double(), op_keys=False)
    assert (ids == ids_c).float().mean().item() > 0.999
    same = (ids == ids_c).all(-1)
    assert torch.allclose(weights[same], weights_c[same], atol=0, rtol=0), "combine weights use the unbiased sigmoid"


def test_centered_bias_upcasts_before_centering():
    bias = torch.tensor([16.2, 16.3, 16.4, 16.5], dtype=torch.bfloat16)
    out = centered_correction_bias(bias)
    assert out.dtype == torch.float32
    assert torch.allclose(out, bias.float() - bias.float().mean())


# ---- loader -----------------------------------------------------------------------------------------------
def test_loader_keeps_router_fp32(tmp_path):
    p = "language_model.model.layers.3."
    gate = torch.randn(E, 64, dtype=torch.float32)
    bias = 16.3 + 0.057 * torch.randn(E, dtype=torch.float32)
    shard = {
        p + "block_sparse_moe.gate.weight": gate,
        p + "block_sparse_moe.e_score_correction_bias": bias,
        p + "post_attention_layernorm.weight": torch.randn(64, dtype=torch.float32),
        p + "self_attn.o_proj.weight": torch.randn(64, 64, dtype=torch.bfloat16),
    }
    save_file(shard, str(tmp_path / "a.safetensors"))
    weight_map = {k: "a.safetensors" for k in shard}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    sd = read_text_backbone(tmp_path, weight_map, ["a.safetensors"])
    q = "model.layers.3."
    assert sd[q + "block_sparse_moe.gate.weight"].dtype == torch.float32
    assert torch.equal(sd[q + "block_sparse_moe.gate.weight"], gate)
    assert sd[q + "block_sparse_moe.e_score_correction_bias"].dtype == torch.float32
    assert torch.equal(sd[q + "block_sparse_moe.e_score_correction_bias"], bias)
    # every other fp32 tensor is still cast to bf16
    assert sd[q + "post_attention_layernorm.weight"].dtype == torch.bfloat16
    assert sd[q + "self_attn.o_proj.weight"].dtype == torch.bfloat16
    # bf16 would collapse the bias to a few distinct values; fp32 keeps every one
    assert bias.to(torch.bfloat16).unique().numel() <= 8 < sd[q + "block_sparse_moe.e_score_correction_bias"].numel()


@pytest.mark.parametrize(
    "key, expected",
    [
        ("model.layers.3.block_sparse_moe.gate.weight", True),
        ("language_model.model.layers.59.block_sparse_moe.e_score_correction_bias", True),
        ("model.layers.3.mlp.gate_proj.weight", False),
        ("model.layers.3.block_sparse_moe.shared_experts.gate_proj.weight", False),
        ("model.layers.3.block_sparse_moe.experts.0.w1.weight", False),
    ],
)
def test_is_router_fp32_key(key, expected):
    assert is_router_fp32_key(key) is expected


# ---- weight cache --------------------------------------------------------------------------------------
def _touch(root, rel):
    path = root / f"{rel}_dtype_FLOAT32_layout_TILE.tensorbin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_bias_cache_name_never_matches_the_bf16_entry(tmp_path):
    assert ROUTER_BIAS_CACHE_NAME != "e_score_correction_bias"
    router = tmp_path / "router"
    router.mkdir()
    (router / "e_score_correction_bias_dtype_BFLOAT16_layout_TILE.tensorbin").write_bytes(b"")
    assert not cache_file_exists(str(router / ROUTER_BIAS_CACHE_NAME))
    (router / f"{ROUTER_BIAS_CACHE_NAME}_dtype_FLOAT32_layout_TILE.tensorbin").write_bytes(b"")
    assert cache_file_exists(str(router / ROUTER_BIAS_CACHE_NAME))


def test_weight_cache_requires_centered_router_bias(tmp_path):
    import ttnn
    from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

    cfg = SimpleNamespace(moe_layer_freq=[1], sparse_attention_config=None)
    base = "model.layers.0"
    for rel in (
        "input_layernorm/weight",
        "post_attention_layernorm/weight",
        "self_attn/wqkv",
        "self_attn/o_proj",
        "self_attn/q_norm",
        "self_attn/k_norm",
        "mlp/router/weight",
        "mlp/router/e_score_correction_bias",
        "mlp/shared_expert/gate_proj",
        "mlp/experts_ep/layer_0.gate.weight",
        "mlp/experts_ep/layer_0.routed_expert.local_0_gate_dtype_BFLOAT4_B",
    ):
        _touch(tmp_path, f"{base}/{rel}")

    def complete():
        return weight_cache_is_complete(tmp_path, cfg, 1, ttnn.bfloat4_b, is_first_rank=False, is_last_rank=False)

    assert not complete(), "a cache with only the stale bf16 bias must not count as complete"
    _touch(tmp_path, f"{base}/mlp/router/{ROUTER_BIAS_CACHE_NAME}")
    assert complete()
