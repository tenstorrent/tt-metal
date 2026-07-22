# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Focused parity coverage for DeepSeek MLA fused SDPA.

This intentionally avoids the full DeepSeek model path so unrelated MoE test
failures do not mask whether MLA can use the fused SDPA op directly.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.models.deepseek import DeepSeekConfig
from ttml.models.deepseek.autograd_ops import autograd_concat, autograd_slice, split_heads
from ttml.models.deepseek.mla import MultiHeadLatentAttention


SEED = 2026


def _make_config(seq_len: int = 64) -> DeepSeekConfig:
    """Build a minimal DeepSeek config that exercises MLA's fused SDPA shape path."""
    return DeepSeekConfig(
        vocab_size=64,
        dim=64,
        inter_dim=64,
        moe_inter_dim=64,
        n_layers=1,
        n_dense_layers=1,
        n_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        max_seq_len=seq_len,
        rope_theta=10000.0,
    )


def _make_inputs(batch_size: int = 2, seq_len: int = 64, dim: int = 64):
    """Create deterministic MLA activations plus the explicit mask used by the old composite path."""
    rng = np.random.default_rng(SEED)
    x_np = rng.standard_normal((batch_size, 1, seq_len, dim), dtype=np.float32)
    mask_np = np.tril(np.ones((seq_len, seq_len), dtype=np.float32)).reshape(1, 1, seq_len, seq_len)
    x = ttml.autograd.Tensor.from_numpy(x_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    mask = ttml.autograd.Tensor.from_numpy(mask_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    return x, mask


def _old_composite_forward(module: MultiHeadLatentAttention, x: ttml.autograd.Tensor, mask: ttml.autograd.Tensor):
    """Run the pre-fused MLA attention path as the reference implementation."""
    B, _, S, _ = list(x.get_value().shape)
    n_heads = module.n_heads
    qk_nope = module.qk_nope_head_dim
    qk_head = module.qk_head_dim
    kv_lora = module.kv_lora_rank

    if module.q_lora_rank == 0:
        q = module.wq(x)
    else:
        q = module.wq_b(module.q_norm(module.wq_a(x)))
    q = split_heads(q, n_heads)

    q_nope = autograd_slice(q, [0, 0, 0, 0], [B, n_heads, S, qk_nope])
    q_pe = autograd_slice(q, [0, 0, 0, qk_nope], [B, n_heads, S, qk_head])
    q_pe = ttml.ops.rope.rope(q_pe, module.rope_params)

    kv_full = module.wkv_a(x)
    kv = autograd_slice(kv_full, [0, 0, 0, 0], [B, 1, S, kv_lora])
    k_pe = autograd_slice(kv_full, [0, 0, 0, kv_lora], [B, 1, S, kv_lora + module.qk_rope_head_dim])
    k_pe = ttml.ops.rope.rope(k_pe, module.rope_params)
    k_pe = autograd_concat([k_pe] * n_heads, dim=1)

    kv_up = module.wkv_b(module.kv_norm(kv))
    kv_up = split_heads(kv_up, n_heads)
    k_nope = autograd_slice(kv_up, [0, 0, 0, 0], [B, n_heads, S, qk_nope])
    v = autograd_slice(kv_up, [0, 0, 0, qk_nope], [B, n_heads, S, qk_nope + module.v_head_dim])

    q_full = autograd_concat([q_nope, q_pe], dim=3)
    k_full = autograd_concat([k_nope, k_pe], dim=3)
    attn = ttml.ops.attention.scaled_dot_product_attention_composite(q_full, k_full, v, mask)
    attn = ttml.ops.multi_head_utils.heads_fusion(attn)
    return module.wo(attn)


def _to_numpy(tensor: ttml.autograd.Tensor) -> np.ndarray:
    """Convert a TTML tensor to an fp32 numpy array for host-side comparison."""
    return np.asarray(tensor.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)


def _grad_to_numpy(param) -> np.ndarray:
    """Convert an initialized TTML parameter gradient to an fp32 numpy array."""
    return np.asarray(param.get_grad_tensor().to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)


def _compare_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Return absolute-error and cosine-similarity metrics for two arrays."""
    af = np.asarray(a, dtype=np.float32).reshape(-1)
    bf = np.asarray(b, dtype=np.float32).reshape(-1)
    diff = np.abs(af - bf)
    norm_a = float(np.linalg.norm(af))
    norm_b = float(np.linalg.norm(bf))
    denom = norm_a * norm_b + 1e-8
    return {
        "abs_max": float(diff.max()),
        "abs_mean": float(diff.mean()),
        "cos_sim": float(np.dot(af, bf) / denom),
        "norm_a": norm_a,
        "norm_b": norm_b,
    }


def _assert_close(name: str, metrics: dict[str, float], abs_mean_limit: float = 5e-3) -> None:
    """Assert BF16-tolerant closeness from the metrics produced by `_compare_arrays`."""
    assert metrics["abs_mean"] < abs_mean_limit, f"{name} mean abs diff too high: {metrics}"
    assert metrics["abs_max"] < 5e-2, f"{name} max abs diff too high: {metrics}"

    # Very small gradients can have unstable cosine despite BF16-sized absolute
    # differences, so treat sub-1e-5 mean error as already close enough.
    if metrics["abs_mean"] >= 1e-5:
        assert metrics["cos_sim"] > 0.995, f"{name} cosine similarity too low: {metrics}"


@pytest.fixture(autouse=True)
def reset_graph():
    """Clear the TTML autograd graph after each test case."""
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
def test_mla_fused_sdpa_matches_composite_forward_and_backward():
    """Compare fused-SDPA MLA against the old composite path for output and parameter gradients."""
    ctx = ttml.autograd.AutoContext.get_instance()

    try:
        device = ctx.get_device()
        cfg = _make_config()
        rope_params = ttml.ops.rope.build_rope_params(cfg.max_seq_len, cfg.qk_rope_head_dim, cfg.rope_theta)
        module = MultiHeadLatentAttention(cfg, rope_params)

        ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
        x, mask = _make_inputs()
        out_composite = _old_composite_forward(module, x, mask)
        ttnn.synchronize_device(device)
        out_composite_np = _to_numpy(out_composite)
        ctx.reset_graph()

        x, _ = _make_inputs()
        out_fused = module(x)  # MLA is causal-only: no mask argument
        ttnn.synchronize_device(device)
        out_fused_np = _to_numpy(out_fused)
        ctx.reset_graph()

        _assert_close("forward", _compare_arrays(out_composite_np, out_fused_np))

        ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
        opt_cfg = ttml.optimizers.SGDConfig.make(0.0, 0.0, 0.0, 0.0, False)
        zero_grad = ttml.optimizers.SGD(module.parameters(), opt_cfg)

        zero_grad.zero_grad()
        x, mask = _make_inputs()
        loss = ttml.ops.unary.mean(_old_composite_forward(module, x, mask))
        loss.backward(False)
        ttnn.synchronize_device(device)
        composite_grads = {
            name: _grad_to_numpy(param) for name, param in module.parameters().items() if param.is_grad_initialized()
        }
        ctx.reset_graph()

        zero_grad.zero_grad()
        x, _ = _make_inputs()
        loss = ttml.ops.unary.mean(module(x))  # MLA is causal-only: no mask argument
        loss.backward(False)
        ttnn.synchronize_device(device)
        fused_grads = {
            name: _grad_to_numpy(param) for name, param in module.parameters().items() if param.is_grad_initialized()
        }
        ctx.reset_graph()

        assert set(composite_grads) == set(fused_grads)
        for name in sorted(composite_grads):
            _assert_close(f"grad {name}", _compare_arrays(composite_grads[name], fused_grads[name]))
    finally:
        ctx.close_device()
