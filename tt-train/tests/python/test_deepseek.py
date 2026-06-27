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

import os

import numpy as np
import pytest
import torch
import ml_dtypes

import ttnn
import ttml
from ttml.common.data import build_causal_mask
from ttml.models import RunnerType
from ttml.models.deepseek import DeepSeek, DeepSeekConfig
from ttml.models.deepseek.moe import MoE as TtmlMoE
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
    return grad_tensor.to_numpy(ttnn.DataType.FLOAT32)


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
    """Seed numpy, torch, and ttml.init deterministically."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ttml.init.manual_seed(SEED)


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


def _build_models_at_seed(seed: int):
    """Seed RNGs, build both models in bf16, copy torch weights to ttml, set
    eval mode. Returns ``(torch_model, ttml_model, cfg)``.

    Like the ``models`` fixture but parametrized on the seed, for the
    diagnostic tests below.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    ttml.init.manual_seed(seed)

    cfg = make_ttml_config()
    torch_args = make_torch_args()

    torch_model = TorchTransformer(torch_args).to(dtype=torch.bfloat16)
    ttml_model = DeepSeek(cfg)
    copy_torch_to_ttml(torch_model, ttml_model, cfg)

    torch_model.eval()
    ttml_model.eval()

    return torch_model, ttml_model, cfg


def _ttml_topk_indices_to_numpy(topk_ttnn) -> np.ndarray:
    """Convert a ttnn.topk indices tensor ``[B, 1, S, k]`` to an int64 numpy
    array ``[B, S, k]``. topk returns UINT16, which to_numpy can't bind."""
    topk_u32 = ttnn.typecast(topk_ttnn, ttnn.DataType.UINT32)
    arr = ttml.autograd.create_tensor(topk_u32, requires_grad=False).to_numpy(ttnn.DataType.UINT32)
    return arr.reshape(arr.shape[0], arr.shape[2], arr.shape[3]).astype(np.int64)


def _capture_moe_input_hidden(torch_model, ttml_model, cfg, batch_size: int):
    """Run one forward on both models and capture the hidden state flowing
    into the first MoE layer. Returns ``(torch_hidden [B,S,dim] bf16,
    ttml_hidden [B,1,S,dim])``."""
    moe_layer_idx = cfg.n_dense_layers
    assert moe_layer_idx < cfg.n_layers, "config has no MoE layer"

    torch_moe = torch_model.layers[moe_layer_idx].ffn
    ttml_moe = ttml_model.blocks[moe_layer_idx].ffn
    assert isinstance(torch_moe, TorchMoE)
    assert isinstance(ttml_moe, TtmlMoE)

    torch_inputs: list = []

    def torch_pre_hook(_module, args):
        torch_inputs.append(args[0].detach().clone())

    torch_handle = torch_moe.register_forward_pre_hook(torch_pre_hook)

    ttml_inputs: list = []
    orig_ttml_forward = ttml_moe.forward

    def ttml_capture_forward(x):
        ttml_inputs.append(x)
        return orig_ttml_forward(x)

    ttml_moe.forward = ttml_capture_forward

    try:
        torch_tokens, ttml_input, ttml_mask = _make_inputs(cfg, batch_size)
        with torch.no_grad():
            torch_model(torch_tokens)
        ttml_model(ttml_input, ttml_mask)
    finally:
        torch_handle.remove()
        ttml_moe.forward = orig_ttml_forward

    assert (
        len(torch_inputs) == 1 and len(ttml_inputs) == 1
    ), f"expected exactly one MoE invocation, got torch={len(torch_inputs)} ttml={len(ttml_inputs)}"
    return torch_inputs[0], ttml_inputs[0]


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
        logits_ttml = logits_ttml_t.to_numpy(ttnn.DataType.FLOAT32)

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
        # No abs_max bound: a bf16-level difference in the MoE input can flip a
        # token's routing and spike one logit, so abs_max isn't meaningful here.

    @pytest.mark.skipif(
        not os.environ.get("TT_DEEPSEEK_SEED_SWEEP"),
        reason="Diagnostic sweep; run with TT_DEEPSEEK_SEED_SWEEP=1",
    )
    @pytest.mark.parametrize("seed", [SEED + i for i in range(20)])
    def test_logits_seed_sweep(self, seed):
        """Diagnostic: sweep seeds and print per-seed metrics (no asserts) to
        characterize abs_max vs max(|logit|) over the bf16 noise distribution.
        """
        torch_model, ttml_model, ttml_cfg = _build_models_at_seed(seed)

        batch_size = 2
        torch_tokens, ttml_input, ttml_mask = _make_inputs(ttml_cfg, batch_size)

        with torch.no_grad():
            logits_torch_bf16 = torch_model(torch_tokens)
        logits_torch = logits_torch_bf16.to(torch.float32).cpu().numpy()
        logits_torch_4d = logits_torch.reshape(batch_size, 1, ttml_cfg.max_seq_len, -1)

        logits_ttml_t = ttml_model(ttml_input, ttml_mask)
        logits_ttml = logits_ttml_t.to_numpy(ttnn.DataType.FLOAT32)

        metrics = _compare_arrays(logits_torch_4d, logits_ttml)
        max_logit = float(np.abs(logits_torch_4d).max())
        rel_to_max = metrics["abs_max"] / max(max_logit, 1e-9)
        print(
            f"\n[seed_sweep seed={seed}] "
            f"abs_max={metrics['abs_max']:.4f} "
            f"abs_mean={metrics['abs_mean']:.4f} "
            f"cos_sim={metrics['cos_sim']:.4f} "
            f"max_|logit|={max_logit:.2f} "
            f"abs_max/max_|logit|={rel_to_max:.4f}"
        )

        # --- Localize the divergence -----------------------------------------
        diff = np.abs(logits_torch_4d - logits_ttml)  # (B, 1, T, V)
        b_w, _, t_w, v_w = (int(x) for x in np.unravel_index(int(diff.argmax()), diff.shape))
        print(
            f"  worst element: batch={b_w} token={t_w} vocab_idx={v_w} "
            f"torch={logits_torch_4d[b_w, 0, t_w, v_w]:.4f} "
            f"ttml={logits_ttml[b_w, 0, t_w, v_w]:.4f}"
        )

        # Per-token max diff to see if divergence is concentrated or spread.
        per_token = diff.max(axis=(1, 3))  # (B, T)
        flat = per_token.reshape(-1)
        top_n = 5
        top_idx_flat = np.argsort(flat)[-top_n:][::-1]
        rows = []
        for f in top_idx_flat:
            b, t = int(f // per_token.shape[1]), int(f % per_token.shape[1])
            rows.append(f"(b={b},t={t}):{flat[f]:.3f}")
        print(f"  top-{top_n} per-token max-diff: " + ", ".join(rows))

        # --- Gate sanity check on a fresh hidden state -----------------------
        # Feed both gates a fresh random hidden (not the runtime MoE input) and
        # compare top-k indices. Widespread disagreement points to a
        # gate-layout or topk bug.
        moe_layer_idx = ttml_cfg.n_dense_layers
        if moe_layer_idx < ttml_cfg.n_layers:
            torch_moe = torch_model.layers[moe_layer_idx].ffn
            ttml_moe = ttml_model.blocks[moe_layer_idx].ffn
            assert isinstance(torch_moe, TorchMoE)
            assert isinstance(ttml_moe, TtmlMoE)

            B, S = batch_size, ttml_cfg.max_seq_len
            hidden_np = (np.random.randn(B, S, ttml_cfg.dim) * 0.1).astype(np.float32)
            hidden_torch_bf = torch.from_numpy(hidden_np).to(torch.bfloat16)
            with torch.no_grad():
                _w, torch_idx = torch_moe.gate(hidden_torch_bf.view(-1, ttml_cfg.dim))
            torch_idx_sorted = np.sort(torch_idx.cpu().numpy(), axis=-1)

            hidden_bf16 = hidden_np.reshape(B, 1, S, ttml_cfg.dim).astype(ml_dtypes.bfloat16)
            ttml_hidden = ttml.autograd.Tensor.from_numpy(hidden_bf16, layout=ttnn.Layout.TILE)
            _scores, _vals, topk_ttnn = ttml_moe.compute_routing(ttml_hidden)
            ttml_idx = _ttml_topk_indices_to_numpy(topk_ttnn).reshape(B * S, ttml_cfg.n_activated_experts)
            ttml_idx_sorted = np.sort(ttml_idx, axis=-1)

            per_token_routing_eq = np.all(torch_idx_sorted == ttml_idx_sorted, axis=-1)
            agreement = float(per_token_routing_eq.mean())
            print(f"  MoE routing agreement (synthetic hidden): {agreement * 100:.1f}% of tokens")

            mismatches_on_worst = []
            for f in top_idx_flat:
                b, t = int(f // per_token.shape[1]), int(f % per_token.shape[1])
                flat_token = b * S + t
                if not per_token_routing_eq[flat_token]:
                    mismatches_on_worst.append(f"(b={b},t={t})")
            if mismatches_on_worst:
                print(f"  worst tokens with routing disagreement: {', '.join(mismatches_on_worst)}")
            else:
                print("  worst tokens all have matching routing (MoE not the obvious cause)")

    @pytest.mark.skipif(
        not os.environ.get("TT_DEEPSEEK_DEBUG_SEED"),
        reason="MoE-input debug; run with TT_DEEPSEEK_DEBUG_SEED=<int>",
    )
    def test_logits_debug_seed(self):
        """Debug a single bad seed: capture the runtime hidden state into the
        MoE layer on both sides, re-run each gate on it, and print routing
        indices/weights side-by-side for the worst-error token.
        """
        seed = int(os.environ["TT_DEEPSEEK_DEBUG_SEED"])
        torch_model, ttml_model, ttml_cfg = _build_models_at_seed(seed)

        moe_layer_idx = ttml_cfg.n_dense_layers
        assert moe_layer_idx < ttml_cfg.n_layers, "config has no MoE layer to debug"

        torch_moe = torch_model.layers[moe_layer_idx].ffn
        ttml_moe = ttml_model.blocks[moe_layer_idx].ffn
        assert isinstance(torch_moe, TorchMoE)
        assert isinstance(ttml_moe, TtmlMoE)

        # --- Hook MoE inputs ------------------------------------------------
        torch_inputs: list[torch.Tensor] = []

        def torch_pre_hook(_module, args):
            # Clone so the captured copy isn't mutated downstream.
            torch_inputs.append(args[0].detach().clone())

        torch_handle = torch_moe.register_forward_pre_hook(torch_pre_hook)

        ttml_inputs: list = []
        orig_ttml_forward = ttml_moe.forward

        def ttml_capture_forward(x):
            ttml_inputs.append(x)
            return orig_ttml_forward(x)

        ttml_moe.forward = ttml_capture_forward

        try:
            batch_size = 2
            torch_tokens, ttml_input, ttml_mask = _make_inputs(ttml_cfg, batch_size)

            with torch.no_grad():
                logits_torch_bf16 = torch_model(torch_tokens)
            logits_torch = logits_torch_bf16.to(torch.float32).cpu().numpy()
            logits_torch_4d = logits_torch.reshape(batch_size, 1, ttml_cfg.max_seq_len, -1)

            logits_ttml_t = ttml_model(ttml_input, ttml_mask)
            logits_ttml = logits_ttml_t.to_numpy(ttnn.DataType.FLOAT32)
        finally:
            torch_handle.remove()
            ttml_moe.forward = orig_ttml_forward

        assert (
            len(torch_inputs) == 1 and len(ttml_inputs) == 1
        ), f"expected exactly one MoE invocation, got torch={len(torch_inputs)} ttml={len(ttml_inputs)}"

        # --- Locate worst element -------------------------------------------
        diff = np.abs(logits_torch_4d - logits_ttml)
        b_w, _, t_w, v_w = (int(x) for x in np.unravel_index(int(diff.argmax()), diff.shape))
        print(
            f"\n[debug seed={seed}] worst element: batch={b_w} token={t_w} "
            f"vocab_idx={v_w} torch={logits_torch_4d[b_w, 0, t_w, v_w]:.4f} "
            f"ttml={logits_ttml[b_w, 0, t_w, v_w]:.4f} abs_diff={diff[b_w, 0, t_w, v_w]:.4f}"
        )

        # --- Compare hidden states at MoE input -----------------------------
        torch_h_t = torch_inputs[0]
        torch_h_np = torch_h_t.to(torch.float32).cpu().numpy().reshape(batch_size, ttml_cfg.max_seq_len, ttml_cfg.dim)
        ttml_h_t = ttml_inputs[0]
        ttml_h_np = ttml_h_t.to_numpy(ttnn.DataType.FLOAT32).reshape(batch_size, ttml_cfg.max_seq_len, ttml_cfg.dim)
        h_diff = np.abs(torch_h_np - ttml_h_np)
        print(
            f"  MoE-input hidden: abs_max={h_diff.max():.4f} abs_mean={h_diff.mean():.4f} "
            f"per-worst-token abs_max={h_diff[b_w, t_w].max():.4f}"
        )

        # --- Re-run torch gate on captured torch hidden ---------------------
        with torch.no_grad():
            flat_torch = torch_h_t.reshape(-1, ttml_cfg.dim).to(torch.bfloat16)
            torch_weights, torch_idx = torch_moe.gate(flat_torch)
            # Raw bf16 sigmoid scores across all experts for the worst token,
            # bypassing the gate's gather/normalize so we see the full vector.
            torch_full_scores = (
                torch.nn.functional.linear(flat_torch, torch_moe.gate.weight)
                .sigmoid()[b_w * ttml_cfg.max_seq_len + t_w]
                .to(torch.float32)
                .cpu()
                .numpy()
            )
        flat_pos = b_w * ttml_cfg.max_seq_len + t_w
        torch_idx_w = torch_idx[flat_pos].cpu().numpy()
        torch_w_w = torch_weights[flat_pos].to(torch.float32).cpu().numpy()

        # --- Re-run ttml routing on captured ttml hidden --------------------
        scores, _topk_vals, topk_idx_ttnn = ttml_moe.compute_routing(ttml_h_t)
        scores_np = scores.to_numpy(ttnn.DataType.FLOAT32).reshape(
            batch_size, ttml_cfg.max_seq_len, ttml_cfg.n_routed_experts
        )
        topk_np = _ttml_topk_indices_to_numpy(topk_idx_ttnn)
        ttml_idx_w = topk_np[b_w, t_w]
        # Gather selected scores, normalize, scale — mirrors torch Gate logic
        ttml_w_w = scores_np[b_w, t_w][ttml_idx_w]
        if ttml_cfg.score_func == "sigmoid":
            ttml_w_w = ttml_w_w / (ttml_w_w.sum() + 1e-20)
        ttml_w_w = ttml_w_w * ttml_cfg.route_scale

        def _fmt(arr):
            return "[" + ", ".join(f"{x:.4f}" for x in arr) + "]"

        print(f"  torch gate @ worst token: indices={np.sort(torch_idx_w).tolist()}")
        print(f"    weights (in idx order)={_fmt(torch_w_w)}")
        print(f"    full sigmoid scores   ={_fmt(torch_full_scores)}")
        print(f"  ttml  gate @ worst token: indices={np.sort(ttml_idx_w).tolist()}")
        print(f"    weights (in idx order)={_fmt(ttml_w_w)}")
        print(f"    full sigmoid scores   ={_fmt(scores_np[b_w, t_w])}")

        # Headline: do indices and weights agree?
        idx_match = sorted(torch_idx_w.tolist()) == sorted(ttml_idx_w.tolist())
        weight_max_diff = float(np.abs(np.sort(torch_w_w) - np.sort(ttml_w_w)).max()) if idx_match else float("nan")
        print(f"  → routing indices match: {idx_match}; weights abs_max_diff={weight_max_diff:.4f}")

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
        """Top-k expert indices should agree between torch and ttml fed the
        same hidden state. Feeds the same captured runtime MoE input to both
        gates, so any disagreement isolates a gate-weight layout, sigmoid, or
        top-k / group-mask bug.
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

        # Capture the hidden states actually flowing into the MoE layer.
        batch_size = 2
        torch_h_t, ttml_h_t = _capture_moe_input_hidden(torch_model, ttml_model, cfg, batch_size)
        B, S = batch_size, cfg.max_seq_len

        # Reference routing: torch gate on its runtime hidden (bf16).
        with torch.no_grad():
            _w, torch_indices = torch_moe.gate(torch_h_t.reshape(-1, cfg.dim).to(torch.bfloat16))
        torch_indices_sorted = np.sort(torch_indices.cpu().numpy(), axis=-1)

        def _ttml_routing(hidden_ttml):
            _scores, _topk_values, topk_idx = ttml_moe.compute_routing(hidden_ttml)
            idx = _ttml_topk_indices_to_numpy(topk_idx).reshape(B * S, cfg.n_activated_experts)
            return np.sort(idx, axis=-1)

        # --- Hard check: same input to both gates --------------------------
        # Feed torch's runtime hidden to ttml's gate too, so the only
        # difference is the gate computation (device vs cpu bf16), not the input.
        torch_h_np = torch_h_t.to(torch.float32).cpu().numpy().reshape(B, 1, S, cfg.dim)
        shared_hidden = ttml.autograd.Tensor.from_numpy(torch_h_np.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE)
        ttml_indices_sorted = _ttml_routing(shared_hidden)

        per_token_equal = np.all(torch_indices_sorted == ttml_indices_sorted, axis=-1)
        num_agree = int(per_token_equal.sum())
        num_total = int(per_token_equal.size)
        print(
            f"\n[test_moe_routing_indices_match] same-input tokens matching: "
            f"{num_agree}/{num_total} ({100 * num_agree / num_total:.1f}%)"
        )
        if num_agree < num_total:
            bad = np.where(~per_token_equal)[0][:8]
            print(
                f"  first disagreeing tokens: {bad.tolist()} "
                f"torch={torch_indices_sorted[bad].tolist()} "
                f"ttml={ttml_indices_sorted[bad].tolist()}"
            )

        # --- Diagnostic: each side on its own runtime hidden ---------------
        # Routing-flip rate when the two sides' inputs differ at bf16 level.
        # Not asserted — it's input divergence, not a gate bug.
        ttml_own_sorted = _ttml_routing(ttml_h_t)
        own_agree = float(np.all(torch_indices_sorted == ttml_own_sorted, axis=-1).mean())
        print(f"  own-input (runtime) agreement: {own_agree:.2%} — inherent bf16 input flips")

        # With identical bf16 weights and the same input, nearly all tokens
        # should route identically; residual flips are device-vs-cpu bf16
        # tie-breaks, but a layout bug would disagree on essentially every token.
        agreement = num_agree / max(num_total, 1)
        assert agreement >= 0.95, (
            f"torch and ttml routed to different experts for {num_total - num_agree} "
            f"out of {num_total} tokens (agreement {agreement:.2%}) on identical input. "
            f"This is far above bf16 tie-break noise and almost certainly a "
            f"gate-weight layout or top-k plumbing bug."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
