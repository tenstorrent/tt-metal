# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Value-level parity tests: ttml DeepSeek vs torch GPU baseline.

Builds a tiny DeepSeek-V3 configuration, creates both the torch reference
(from ``ttml.models.deepseek.gpu_baseline``) and the ttml implementation
(``ttml.models.deepseek.DeepSeek``), copies the torch weights into the
ttml model, and then verifies that forward logits and per-parameter
weight gradients match up to bf16 precision.

Both models run in bf16 to mirror the training configuration used in
``tt-train_nanoGPT-gpu-baseline/train_deepseek_torch.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import ml_dtypes

import ttnn
import ttml
from ttml.common.data import build_causal_mask
from ttml.models import RunnerType
from ttml.models.deepseek import DeepSeek, DeepSeekConfig
from ttml.models.deepseek.torch_baseline import (
    ModelArgs,
    MoE as TorchMoE,
    Transformer as TorchTransformer,
)


SEED = 2026


# =============================================================================
# Tiny tile-aligned config
# =============================================================================

TINY_KWARGS = dict(
    vocab_size=64,
    dim=64,
    inter_dim=64,
    moe_inter_dim=64,
    n_layers=2,
    n_dense_layers=1,
    n_heads=2,
    n_routed_experts=4,
    n_shared_experts=1,
    n_activated_experts=2,
    n_expert_groups=2,
    n_limited_groups=1,
    score_func="sigmoid",
    route_scale=2.5,
    q_lora_rank=32,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=32,
    v_head_dim=32,
    max_seq_len=64,
    rope_theta=10000.0,
)


def make_torch_args() -> ModelArgs:
    return ModelArgs(**TINY_KWARGS)


def make_ttml_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        runner_type=RunnerType.Default,
        **TINY_KWARGS,
    )


# =============================================================================
# Weight copy: torch -> ttml
# =============================================================================


def _torch_weight_to_bf16_4d(tensor: torch.Tensor, target_shape) -> np.ndarray:
    """Convert a torch weight to a 4D ml_dtypes.bfloat16 numpy array matching
    ``target_shape``. Transposes a 2D tensor if the rows/cols are swapped."""
    arr = tensor.detach().to(torch.float32).cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(1, 1, 1, -1)
    elif arr.ndim == 2:
        r, c = arr.shape
        tr, tc = target_shape[-2], target_shape[-1]
        if r == tr and c == tc:
            pass
        elif c == tr and r == tc:
            arr = arr.T
        else:
            raise RuntimeError(f"weight shape mismatch: source ({r}x{c}) vs target ({tr}x{tc})")
        arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
    else:
        raise RuntimeError(f"unexpected weight rank {arr.ndim}")
    return arr.astype(ml_dtypes.bfloat16)


def _assign(param, arr_bf16_4d: np.ndarray) -> None:
    tgt = tuple(param.shape())
    src = arr_bf16_4d.shape
    assert src == tgt, f"shape mismatch: source {src} vs target {tgt}"
    restored = ttml.autograd.Tensor.from_numpy(arr_bf16_4d, layout=ttnn.Layout.TILE)
    param.assign(restored)


def build_torch_to_ttml_mapping(cfg: DeepSeekConfig) -> dict[str, str]:
    """Return torch-state-dict-name -> ttml-parameter-name mapping."""
    mapping: dict[str, str] = {
        "embed.weight": "DeepSeek/tok_emb/weight",
        "head.weight": "DeepSeek/head/weight",
        "norm.weight": "DeepSeek/norm/gamma",
    }
    for b in range(cfg.n_layers):
        tp = f"DeepSeek/blocks/{b}"
        hp = f"layers.{b}"

        # Block-level norms
        mapping[f"{hp}.attn_norm.weight"] = f"{tp}/attn_norm/gamma"
        mapping[f"{hp}.ffn_norm.weight"] = f"{tp}/ffn_norm/gamma"

        # MLA
        mapping[f"{hp}.attn.wq_a.weight"] = f"{tp}/attn/wq_a/weight"
        mapping[f"{hp}.attn.wq_b.weight"] = f"{tp}/attn/wq_b/weight"
        mapping[f"{hp}.attn.q_norm.weight"] = f"{tp}/attn/q_norm/gamma"
        mapping[f"{hp}.attn.wkv_a.weight"] = f"{tp}/attn/wkv_a/weight"
        mapping[f"{hp}.attn.wkv_b.weight"] = f"{tp}/attn/wkv_b/weight"
        mapping[f"{hp}.attn.kv_norm.weight"] = f"{tp}/attn/kv_norm/gamma"
        mapping[f"{hp}.attn.wo.weight"] = f"{tp}/attn/wo/weight"

        # FFN: dense MLP or MoE
        if b < cfg.n_dense_layers:
            for w in ("w1", "w2", "w3"):
                mapping[f"{hp}.ffn.{w}.weight"] = f"{tp}/ffn/{w}/weight"
        else:
            mapping[f"{hp}.ffn.gate.weight"] = f"{tp}/ffn/gate/weight"
            for e in range(cfg.n_routed_experts):
                for w in ("w1", "w2", "w3"):
                    mapping[f"{hp}.ffn.experts.{e}.{w}.weight"] = f"{tp}/ffn/experts/{e}/{w}/weight"
            for w in ("w1", "w2", "w3"):
                mapping[f"{hp}.ffn.shared_experts.{w}.weight"] = f"{tp}/ffn/shared_experts/{w}/weight"
    return mapping


def copy_torch_to_ttml(
    torch_model: TorchTransformer,
    ttml_model: DeepSeek,
    cfg: DeepSeekConfig,
) -> dict[str, str]:
    """Copy weights from a torch DeepSeek into a ttml DeepSeek.

    Returns the name mapping so the caller can use it to walk gradients.
    """
    mapping = build_torch_to_ttml_mapping(cfg)
    ttml_params = ttml_model.parameters()
    torch_sd = dict(torch_model.state_dict())

    for torch_name, ttml_name in mapping.items():
        assert torch_name in torch_sd, f"missing torch weight: {torch_name}"
        assert ttml_name in ttml_params, f"missing ttml param: {ttml_name}"
        param = ttml_params[ttml_name]
        arr = _torch_weight_to_bf16_4d(torch_sd[torch_name], param.shape())
        _assign(param, arr)

    return mapping


# =============================================================================
# Comparison helpers
# =============================================================================


def _flat(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).reshape(-1)


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    af = _flat(a)
    bf = _flat(b)
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    if na < eps or nb < eps:
        return 1.0 if na < eps and nb < eps else 0.0
    return float(np.dot(af, bf) / (na * nb + eps))


def _compare_arrays(a: np.ndarray, b: np.ndarray, rel_eps: float = 1e-2) -> dict:
    """Per-tensor comparison metrics, mirroring qwen3/gradients.py."""
    af = _flat(a)
    bf = _flat(b)
    diff = np.abs(af - bf)
    rel = diff / (np.abs(af) + rel_eps)
    return {
        "abs_max": float(diff.max()),
        "abs_mean": float(diff.mean()),
        "rel_max": float(rel.max()),
        "rel_mean": float(rel.mean()),
        "cos_sim": _cosine_similarity(af, bf),
    }


def _ttml_grad_to_numpy(param) -> np.ndarray:
    """Extract a ttml parameter's gradient as an fp32 numpy array."""
    assert param.is_grad_initialized(), "parameter has no gradient"
    grad_tensor = param.get_grad_tensor()
    return np.asarray(grad_tensor.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)


def _torch_grad_to_numpy(tensor: torch.Tensor, target_shape) -> np.ndarray:
    """Convert a torch gradient to the same 4D layout ttml uses."""
    arr = tensor.detach().to(torch.float32).cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(1, 1, 1, -1)
    elif arr.ndim == 2:
        r, c = arr.shape
        tr, tc = target_shape[-2], target_shape[-1]
        if r == tr and c == tc:
            pass
        elif c == tr and r == tc:
            arr = arr.T
        else:
            raise RuntimeError(f"grad shape mismatch: source ({r}x{c}) vs target ({tr}x{tc})")
        arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
    return arr


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_graph():
    """Reset the autograd graph between tests."""
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.fixture
def seeded():
    """Seed numpy and torch deterministically."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)


@pytest.fixture
def models(seeded):
    """Build and weight-copy both models. Return (torch_model, ttml_model, cfg)."""
    ttml_cfg = make_ttml_config()
    torch_args = make_torch_args()

    torch_model = TorchTransformer(torch_args).to(dtype=torch.bfloat16)
    ttml_model = DeepSeek(ttml_cfg)

    copy_torch_to_ttml(torch_model, ttml_model, ttml_cfg)

    return torch_model, ttml_model, ttml_cfg


def _make_inputs(cfg: DeepSeekConfig, batch_size: int):
    seq_len = cfg.max_seq_len
    tokens = np.random.randint(0, cfg.vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    torch_tokens = torch.from_numpy(tokens)

    ttml_tokens = tokens.astype(np.uint32).reshape(batch_size, 1, 1, seq_len)
    ttml_input = ttml.autograd.Tensor.from_numpy(
        ttml_tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )

    mask_np = build_causal_mask(seq_len)
    ttml_mask = ttml.autograd.Tensor.from_numpy(mask_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)

    return torch_tokens, ttml_input, ttml_mask


# =============================================================================
# Tests
# =============================================================================


class TestDeepSeekVsTorchBaseline:
    """Value-level parity tests between ttml DeepSeek and the torch baseline."""

    def test_logits_match(self, models):
        """Forward logits produced by both models should be close in bf16."""
        torch_model, ttml_model, cfg = models

        torch_model.eval()
        ttml_model.eval()

        batch_size = 2
        torch_tokens, ttml_input, ttml_mask = _make_inputs(cfg, batch_size)

        with torch.no_grad():
            logits_torch_bf16 = torch_model(torch_tokens)
        logits_torch = logits_torch_bf16.to(torch.float32).cpu().numpy()
        logits_torch_4d = logits_torch.reshape(batch_size, 1, cfg.max_seq_len, -1)

        logits_ttml_t = ttml_model(ttml_input, ttml_mask)
        logits_ttml = np.asarray(logits_ttml_t.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)

        assert np.all(np.isfinite(logits_torch_4d)), "torch logits not finite"
        assert np.all(np.isfinite(logits_ttml)), "ttml logits not finite"

        metrics = _compare_arrays(logits_torch_4d, logits_ttml)
        print(
            "\n[test_logits_match] "
            f"abs_max={metrics['abs_max']:.4f} "
            f"abs_mean={metrics['abs_mean']:.4f} "
            f"rel_mean={metrics['rel_mean']:.4f} "
            f"cos_sim={metrics['cos_sim']:.4f}"
        )

        assert metrics["cos_sim"] > 0.995, (
            f"logits cosine similarity too low: {metrics['cos_sim']:.4f} " f"(expected > 0.995)"
        )
        assert metrics["abs_mean"] < 5e-2, (
            f"logits mean abs diff too high: {metrics['abs_mean']:.4f} " f"(expected < 5e-2)"
        )
        assert metrics["abs_max"] < 2e-1, (
            f"logits max abs diff too high: {metrics['abs_max']:.4f} " f"(expected < 2e-1)"
        )

    def test_gradient_match(self, models):
        """Per-parameter weight gradients should match in bf16."""
        torch_model, ttml_model, cfg = models

        torch_model.train()
        ttml_model.train()

        batch_size = 2
        torch_tokens, ttml_input, ttml_mask = _make_inputs(cfg, batch_size)

        # Torch backward
        torch_model.zero_grad(set_to_none=True)
        logits_torch_bf16 = torch_model(torch_tokens)
        loss_torch = logits_torch_bf16.to(torch.float32).mean()
        loss_torch.backward()

        # ttml backward
        logits_ttml = ttml_model(ttml_input, ttml_mask)
        loss_ttml = ttml.ops.unary.mean(logits_ttml)
        loss_ttml.backward(False)

        mapping = build_torch_to_ttml_mapping(cfg)
        torch_params = dict(torch_model.named_parameters())
        ttml_params = ttml_model.parameters()

        rows = []
        skipped: list[str] = []
        for torch_name, ttml_name in mapping.items():
            t_param = torch_params[torch_name]
            m_param = ttml_params[ttml_name]

            if t_param.grad is None:
                # Expert parameters may legitimately have no grad if all
                # routing mass went to other experts. Skip but remember.
                if not m_param.is_grad_initialized():
                    skipped.append(torch_name)
                    continue
                # ttml has a grad but torch doesn't: treat torch grad as zeros
                t_grad_np = np.zeros(tuple(m_param.shape()), dtype=np.float32)
            else:
                t_grad_np = _torch_grad_to_numpy(t_param.grad, m_param.shape())

            if not m_param.is_grad_initialized():
                # ttml produced no grad but torch did: this is always a bug
                raise AssertionError(f"ttml parameter {ttml_name} has no gradient but torch " f"{torch_name} does")

            m_grad_np = _ttml_grad_to_numpy(m_param)

            assert np.all(np.isfinite(t_grad_np)), f"torch grad not finite: {torch_name}"
            assert np.all(np.isfinite(m_grad_np)), f"ttml grad not finite: {ttml_name}"

            metrics = _compare_arrays(t_grad_np, m_grad_np)
            metrics["torch_name"] = torch_name
            metrics["ttml_name"] = ttml_name
            rows.append(metrics)

        assert len(rows) > 0, "no parameters were compared"

        cos_sims = np.array([r["cos_sim"] for r in rows])
        rel_means = np.array([r["rel_mean"] for r in rows])

        # Per-row summary for debugging
        print("\n[test_gradient_match] per-parameter metrics:")
        for r in rows:
            print(
                f"  {r['torch_name']:<55s} "
                f"cos={r['cos_sim']:.4f} rel_mean={r['rel_mean']:.4f} "
                f"abs_max={r['abs_max']:.4f}"
            )
        if skipped:
            print(
                f"  (skipped {len(skipped)} params without grads on either side: "
                f"{skipped[:4]}{'...' if len(skipped) > 4 else ''})"
            )

        mean_cos = float(cos_sims.mean())
        min_cos = float(cos_sims.min())
        mean_rel = float(rel_means.mean())

        assert mean_cos > 0.95, f"mean grad cosine similarity too low: {mean_cos:.4f} " f"(expected > 0.95)"
        assert min_cos > 0.80, (
            f"min grad cosine similarity too low: {min_cos:.4f} "
            f"(expected > 0.80); worst param: "
            f"{rows[int(cos_sims.argmin())]['torch_name']}"
        )
        assert mean_rel < 0.20, f"mean grad relative diff too high: {mean_rel:.4f} " f"(expected < 0.20)"

    def test_moe_routing_indices_match(self, models):
        """Top-k expert indices from the MoE router should agree between torch
        and ttml when both are fed the same hidden state and have identical
        gate weights.

        Calls ttml's ``MoE.compute_routing`` helper directly (bypassing the
        rest of the MoE forward) so a failure here points squarely at the
        gate weight layout, sigmoid, or top-k / group-mask plumbing.
        """
        torch_model, ttml_model, cfg = models
        torch_model.eval()
        ttml_model.eval()

        moe_layer_idx = cfg.n_dense_layers
        assert moe_layer_idx < cfg.n_layers

        torch_moe: TorchMoE = torch_model.layers[moe_layer_idx].ffn
        ttml_moe = ttml_model.blocks[moe_layer_idx].ffn
        assert isinstance(torch_moe, TorchMoE)
        assert hasattr(ttml_moe, "compute_routing"), (
            "ttml MoE is missing compute_routing helper; this test requires "
            "the routing logic to be exposed as a standalone method"
        )

        # Deterministic hidden state. Run both gates in bf16 so the comparison
        # is limited to device-vs-cpu bf16 rounding noise, not fp32-vs-bf16.
        B, S = 1, cfg.max_seq_len
        hidden_np = (np.random.randn(B, S, cfg.dim) * 0.1).astype(np.float32)

        # Torch side: bf16 gate on flattened hidden
        hidden_torch = torch.from_numpy(hidden_np).to(torch.bfloat16)
        with torch.no_grad():
            flat = hidden_torch.view(-1, cfg.dim)
            _w, torch_indices = torch_moe.gate(flat)
        # shape [B*S, n_activated]
        torch_indices_sorted = np.sort(torch_indices.cpu().numpy(), axis=-1)

        # ttml side: feed the same hidden into compute_routing
        hidden_bf16 = hidden_np.reshape(B, 1, S, cfg.dim).astype(ml_dtypes.bfloat16)
        ttml_hidden = ttml.autograd.Tensor.from_numpy(hidden_bf16, layout=ttnn.Layout.TILE)
        _scores, _topk_values, topk_indices_ttnn = ttml_moe.compute_routing(ttml_hidden)

        # ttnn.topk returns UINT16 indices which ttml's to_numpy doesn't bind;
        topk_indices_u32 = ttnn.typecast(topk_indices_ttnn, ttnn.DataType.UINT32)
        ttml_idx_np = np.asarray(ttml.autograd.Tensor(topk_indices_u32, False).to_numpy(ttnn.DataType.UINT32))
        # ttml returns [B, 1, S, n_activated]; flatten to [B*S, n_activated]
        ttml_idx_np = ttml_idx_np.reshape(B * S, cfg.n_activated_experts).astype(np.int64)
        ttml_indices_sorted = np.sort(ttml_idx_np, axis=-1)

        # Diagnostics: how often do the two disagree, and is the disagreement
        # localized (suggests precision tie-breaking) or widespread (layout bug)?
        per_token_equal = np.all(torch_indices_sorted == ttml_indices_sorted, axis=-1)
        num_agree = int(per_token_equal.sum())
        num_total = int(per_token_equal.size)
        print(
            f"\n[test_moe_routing_indices_match] tokens matching: "
            f"{num_agree}/{num_total} ({100 * num_agree / num_total:.1f}%)"
        )
        if num_agree < num_total:
            bad = np.where(~per_token_equal)[0][:8]
            print(
                f"  first disagreeing tokens: {bad.tolist()} "
                f"torch={torch_indices_sorted[bad].tolist()} "
                f"ttml={ttml_indices_sorted[bad].tolist()}"
            )

        # With identical bf16 weights and bf16 activations on both sides, the
        # vast majority of tokens should route identically. Residual noise is
        # allowed because ttnn's bf16 topk/sum on device can break near-ties
        # differently from torch's CPU bf16, but a layout bug would make the
        # two disagree for essentially every token.
        agreement = num_agree / max(num_total, 1)
        assert agreement >= 0.95, (
            f"torch and ttml routed to different experts for {num_total - num_agree} "
            f"out of {num_total} tokens (agreement {agreement:.2%}). This is far "
            f"above bf16 tie-break noise and almost certainly a gate-weight "
            f"layout or top-k plumbing bug."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
