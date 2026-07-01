# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers and fixtures for VibeVoice LM prefill/decode PCC tests."""

import contextlib
import math
import sys
import time
from pathlib import Path

import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.tt.ttnn_vibevoice_lm import (
    TTVibeVoiceLM,
    preprocess_lm_weights,
    _HIFI4,
    _SDPA_DECODE_CFG,
    _apply_rope_ttnn,
    _fused_sdpa_decode_safe,
    _k_chunk_from_cache_seq,
    _reshape_tt,
)

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SEQ_LEN = 32
PCC_THRESHOLD = 0.99
DECODE_GENERATION_LENGTH = 10
DECODE_LAYERWISE_FAIL_STEP = 7
# Fused scaled_dot_product_attention_decode baseline (seed=0, Blackhole, decode_pcc_run.log).
FUSED_MULTI_STEP_DECODE_PCC_BASELINE = [
    0.99972,
    0.99609,
    0.99352,
    0.99330,
    0.99435,
    0.99986,
    0.99204,
    0.98992,
    0.98704,
    0.99048,
]
PREFILL_ISL_SWEEP_LENGTHS = [32, 64, 128, 256, 512, 1024]
PREFILL_ISL_EXTENDED_SWEEP_LENGTHS = [
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
]
PREFILL_ISL_EXTENDED_TARGET = 65536
PREFILL_CHUNK_SIZE = 256
L0_ATTENTION_STAGE_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "q_rope",
    "k_rope",
    "sdpa_out",
    "o_proj_out",
)
L0_DECODE_PATH_COMPARE_STAGES = (
    "qk_scores",
    "scaled_scores",
    "softmax_probs",
    "weighted_v",
    "sdpa_out",
    "o_proj_out",
    "hidden_after_residual",
)
FUSED_OPAQUE_SDPA_STAGES = frozenset({"qk_scores", "scaled_scores", "softmax_probs", "weighted_v"})
FUSED_VS_MANUAL_PCC_SIGNIFICANT_DROP = 0.995
L0_ATTENTION_STAGE_PCC_NEAR_PERFECT = 0.9999
L0_SDPA_STAGE_NAMES = (
    "qk_scores",
    "scaled_scores",
    "masked_scores",
    "softmax_probs",
    "weighted_v",
    "sdpa_out",
)

_HF_MODEL_CACHE: dict[tuple[int, str], torch.nn.Module] = {}
DEFAULT_HF_DECODE_ATTN_IMPLEMENTATION = "sdpa"


def _remap_lm_state_to_hf(lm_state: dict) -> dict:
    hf_state = {}
    for k, v in lm_state.items():
        hf_k = k
        hf_k = hf_k.replace("tok_embeddings.", "embed_tokens.")
        hf_k = hf_k.replace(".attention.wq", ".self_attn.q_proj")
        hf_k = hf_k.replace(".attention.wk", ".self_attn.k_proj")
        hf_k = hf_k.replace(".attention.wv", ".self_attn.v_proj")
        hf_k = hf_k.replace(".attention.wo", ".self_attn.o_proj")
        hf_k = hf_k.replace(".feed_forward.w1", ".mlp.gate_proj")
        hf_k = hf_k.replace(".feed_forward.w3", ".mlp.up_proj")
        hf_k = hf_k.replace(".feed_forward.w2", ".mlp.down_proj")
        hf_k = hf_k.replace(".attention_norm", ".input_layernorm")
        hf_k = hf_k.replace(".ffn_norm", ".post_attention_layernorm")
        hf_k = hf_k.replace("norm.weight", "norm.weight")
        hf_state[hf_k] = v
    return hf_state


def _get_hf_reference_model(
    lm_state: dict,
    vv_config,
    *,
    attn_implementation: str = DEFAULT_HF_DECODE_ATTN_IMPLEMENTATION,
) -> torch.nn.Module:
    """Build and cache one HF Qwen2Model per ``(lm_state, attn_implementation)``."""
    cache_key = (id(lm_state), attn_implementation)
    if cache_key not in _HF_MODEL_CACHE:
        from transformers import Qwen2Config, Qwen2Model

        cfg_dec = vv_config.decoder
        hf_cfg = Qwen2Config(
            hidden_size=cfg_dec.hidden_size,
            num_hidden_layers=cfg_dec.num_hidden_layers,
            num_attention_heads=cfg_dec.num_attention_heads,
            num_key_value_heads=cfg_dec.num_key_value_heads,
            intermediate_size=cfg_dec.intermediate_size,
            vocab_size=cfg_dec.vocab_size,
            rope_theta=cfg_dec.rope_theta,
            rms_norm_eps=cfg_dec.rms_norm_eps,
            max_position_embeddings=cfg_dec.max_position_embeddings,
            attn_implementation=attn_implementation,
        )
        model = Qwen2Model(hf_cfg)
        model.load_state_dict(_remap_lm_state_to_hf(lm_state), strict=False)
        model.to(torch.bfloat16)
        model.eval()
        _HF_MODEL_CACHE[cache_key] = model
    return _HF_MODEL_CACHE[cache_key]


def reference_lm_forward(lm_state: dict, input_ids: torch.Tensor, vv_config) -> torch.Tensor:
    """Run reference Qwen2 forward using transformers (bf16 — matches TT numerics)."""
    model = _get_hf_reference_model(lm_state, vv_config)
    with torch.no_grad():
        out = model(input_ids)
    return out.last_hidden_state.float()  # [B, S, hidden]


def reference_lm_prefill_cache(
    lm_state: dict,
    input_ids: torch.Tensor,
    vv_config,
    *,
    attn_implementation: str = DEFAULT_HF_DECODE_ATTN_IMPLEMENTATION,
):
    """Prefill HF Qwen2 and return ``past_key_values`` for incremental decode."""
    model = _get_hf_reference_model(lm_state, vv_config, attn_implementation=attn_implementation)
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    return out.past_key_values


def reference_lm_decode_hidden(
    lm_state: dict,
    input_ids: torch.Tensor,
    vv_config,
    past_key_values,
    *,
    attn_implementation: str = DEFAULT_HF_DECODE_ATTN_IMPLEMENTATION,
):
    """Single HF decode step; returns (last hidden [B, 1, H], updated past_key_values)."""
    model = _get_hf_reference_model(lm_state, vv_config, attn_implementation=attn_implementation)
    with torch.no_grad():
        out = model(input_ids, past_key_values=past_key_values, use_cache=True)
    return out.last_hidden_state[:, -1:].float(), out.past_key_values


def reference_lm_decode_layer_hiddens(
    lm_state: dict, input_ids: torch.Tensor, vv_config, past_key_values
) -> tuple[list[torch.Tensor], torch.Tensor, object]:
    """Single HF decode step with ``output_hidden_states=True``.

    Returns (per-layer hiddens, final hidden, updated past_key_values).
    ``per-layer`` aligns with TT probe: index 0 = embeddings, 1..N = after each block.
    """
    model = _get_hf_reference_model(lm_state, vv_config)
    with torch.no_grad():
        out = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
    layer_hiddens = [h[:, -1:].float() for h in out.hidden_states]
    final = out.last_hidden_state[:, -1:].float()
    return layer_hiddens, final, out.past_key_values


def reference_lm_decode_l0_attention(
    lm_state: dict, input_ids: torch.Tensor, vv_config, past_key_values, layer_idx: int = 0
) -> tuple[torch.Tensor, torch.Tensor, object]:
    """HF decode step with hooks on layer ``layer_idx`` self-attn input/output.

    Returns (attn_input, attn_output, updated past_key_values) for the last token.
    """
    model = _get_hf_reference_model(lm_state, vv_config)
    captured: dict[str, torch.Tensor] = {}

    def _norm_hook(_module, _inp, out):
        captured["attn_in"] = out[:, -1:].float()

    def _attn_post_hook(_module, _inp, out):
        attn_out = out[0] if isinstance(out, tuple) else out
        captured["attn_out"] = attn_out[:, -1:].float()

    layer = model.layers[layer_idx]
    norm_handle = layer.input_layernorm.register_forward_hook(_norm_hook)
    post_handle = layer.self_attn.register_forward_hook(_attn_post_hook)
    try:
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)
        return captured["attn_in"], captured["attn_out"], out.past_key_values
    finally:
        post_handle.remove()
        norm_handle.remove()


def _hf_attn_stage_flat(t: torch.Tensor) -> torch.Tensor:
    """Flatten decode-step attention intermediate to 1D float32 for PCC."""
    t = t.float()
    if t.dim() == 4:
        return t[:, :, -1:, :].reshape(-1)
    if t.dim() == 3:
        return t[:, -1:, :].reshape(-1)
    return t.reshape(-1)


def _tt_attn_stage_flat(x: ttnn.Tensor, layout: str) -> torch.Tensor:
    """Convert TT attention intermediate to 1D float32 torch for PCC."""
    t = ttnn.to_torch(ttnn.typecast(x, ttnn.float32))
    if layout == "proj":
        return t.squeeze(1).reshape(-1)
    if layout == "heads":
        return t[:, :, -1:, :].reshape(-1)
    if layout == "hidden":
        return t.squeeze(1).reshape(-1)
    raise ValueError(f"unknown TT stage layout: {layout}")


def _flat_sdpa_stage_hf(t: torch.Tensor) -> torch.Tensor:
    """Flatten SDPA intermediate [B, n_heads, 1, K|hd] to 1D float32 for PCC."""
    return t.float().reshape(-1)


def _flat_sdpa_stage_tt(x: ttnn.Tensor) -> torch.Tensor:
    """Flatten TT SDPA intermediate to 1D float32 torch for PCC."""
    return ttnn.to_torch(ttnn.typecast(x, ttnn.float32)).reshape(-1)


def _stage_capture_from_tt(x: ttnn.Tensor) -> dict:
    """Capture a TT tensor as float32 torch with shape metadata."""
    t = ttnn.to_torch(ttnn.typecast(x, ttnn.float32)).to(torch.float32)
    return {"tensor": t, "shape": tuple(t.shape), "flat": t.reshape(-1)}


def _stage_capture_from_torch(t: torch.Tensor) -> dict:
    """Capture a torch tensor with shape metadata."""
    t = t.detach().float()
    return {"tensor": t, "shape": tuple(t.shape), "flat": t.reshape(-1)}


def _cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    if a_f.numel() == 0 or b_f.numel() == 0:
        return float("nan")
    if a_f.shape != b_f.shape:
        return float("nan")
    return torch.nn.functional.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def _compare_stage_tensors(ref: dict, cmp: dict | None) -> dict:
    """Compare two stage captures; ``ref`` is manual path, ``cmp`` is fused (or None)."""
    ref_flat = ref["flat"]
    if cmp is None:
        return {"comparable": False, "reason": "fused kernel does not expose this stage"}

    cmp_flat = cmp["flat"]
    if ref_flat.shape != cmp_flat.shape:
        return {
            "comparable": False,
            "reason": f"shape mismatch ref={tuple(ref_flat.shape)} cmp={tuple(cmp_flat.shape)}",
        }

    err = ref_flat - cmp_flat
    abs_err = err.abs()
    _, pcc = comp_pcc(ref_flat, cmp_flat, pcc=0.0)
    return {
        "comparable": True,
        "pcc": pcc,
        "max_abs_error": abs_err.max().item(),
        "mean_abs_error": abs_err.mean().item(),
        "rms_error": err.pow(2).mean().sqrt().item(),
        "cosine_similarity": _cosine_similarity_flat(ref_flat, cmp_flat),
    }


def _tensor_health_stats(t: torch.Tensor) -> dict:
    """Numerical health and range stats for one tensor."""
    x = t.detach().float().reshape(-1)
    finite = torch.isfinite(x)
    nan_count = torch.isnan(x).sum().item()
    inf_count = torch.isinf(x).sum().item()
    if finite.any():
        xf = x[finite]
        stats = {
            "min": xf.min().item(),
            "max": xf.max().item(),
            "mean": xf.mean().item(),
            "std": xf.std(unbiased=False).item(),
        }
        abs_x = xf.abs()
        stats["near_bf16_saturation_frac"] = (abs_x > 60000.0).float().mean().item()
        stats["exact_zero_frac"] = (xf == 0).float().mean().item()
    else:
        stats = {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
        stats["near_bf16_saturation_frac"] = float("nan")
        stats["exact_zero_frac"] = float("nan")

    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "numel": x.numel(),
        **stats,
    }


def _error_distribution_stats(ref_flat: torch.Tensor, cmp_flat: torch.Tensor) -> dict:
    """Characterize whether error is uniform or dominated by outliers."""
    err = (ref_flat.float() - cmp_flat.float()).abs()
    total = err.sum().item()
    if total <= 0 or err.numel() == 0:
        return {
            "top1pct_error_fraction": 0.0,
            "top10_max_abs_errors": [],
            "top10_indices": [],
            "character": "no_error",
        }

    k1pct = max(1, int(math.ceil(0.01 * err.numel())))
    top1pct, _ = torch.topk(err, k1pct)
    top1pct_frac = top1pct.sum().item() / total

    k10 = min(10, err.numel())
    top10, top10_idx = torch.topk(err, k10)
    character = "dominated_by_outliers" if top1pct_frac > 0.35 else "relatively_uniform"

    return {
        "top1pct_error_fraction": top1pct_frac,
        "top10_max_abs_errors": top10.tolist(),
        "top10_indices": top10_idx.tolist(),
        "character": character,
    }


def _softmax_distribution_stats(probs_2d: torch.Tensor) -> dict:
    """Softmax distribution stats over the key dimension; ``probs_2d`` is [n_heads, valid_len]."""
    p = probs_2d.float().clamp(min=0)
    row_sum = p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    p = p / row_sum
    ent = -(p * (p + 1e-12).log()).sum(dim=-1)
    max_p = p.max(dim=-1).values
    k = min(5, p.shape[-1])
    topk_vals, _ = torch.topk(p, k=k, dim=-1)

    return {
        "entropy_mean": ent.mean().item(),
        "entropy_std": ent.std(unbiased=False).item(),
        "entropy_min": ent.min().item(),
        "entropy_max": ent.max().item(),
        "max_prob_mean": max_p.mean().item(),
        "max_prob_max": max_p.max().item(),
        "max_prob_min": max_p.min().item(),
        "topk_prob_means": topk_vals.mean(dim=0).tolist(),
    }


def _torch_bf16_softmax_from_scaled(scaled_2d: torch.Tensor) -> torch.Tensor:
    """Simulate bf16 softmax on scaled scores for expected-numerics comparison."""
    s = scaled_2d.to(torch.bfloat16).float()
    return torch.softmax(s, dim=-1)


def _per_head_pcc(ref_2d: torch.Tensor, cmp_2d: torch.Tensor) -> list[float]:
    """Per-head PCC for tensors shaped [n_heads, feature_dim]."""
    if ref_2d.shape != cmp_2d.shape or ref_2d.dim() != 2:
        return []
    return [comp_pcc(ref_2d[h], cmp_2d[h], pcc=0.0)[1] for h in range(ref_2d.shape[0])]


def reference_lm_decode_l0_sdpa_stages(
    lm_state: dict, input_ids: torch.Tensor, vv_config, past_key_values, layer_idx: int = 0
) -> tuple[dict[str, torch.Tensor], object]:
    """HF decode step with eager-attention SDPA stage capture on layer ``layer_idx``."""
    from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv

    model = _get_hf_reference_model(lm_state, vv_config)
    stages: dict[str, torch.Tensor] = {}
    eager_call_idx = [0]
    orig_eager = qwen2_mod.eager_attention_forward
    prev_impl = model.config._attn_implementation

    def _patched_eager(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        qk = torch.matmul(query, key_states.transpose(2, 3))
        scaled = qk * scaling
        masked = scaled if attention_mask is None else scaled + attention_mask
        attn_w = torch.nn.functional.softmax(masked, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_w = torch.nn.functional.dropout(attn_w, p=dropout, training=module.training)
        weighted = torch.matmul(attn_w, value_states)
        attn_output = weighted.transpose(1, 2).contiguous()

        if eager_call_idx[0] == layer_idx:
            stages["qk_scores"] = _flat_sdpa_stage_hf(qk)
            stages["scaled_scores"] = _flat_sdpa_stage_hf(scaled)
            stages["masked_scores"] = _flat_sdpa_stage_hf(masked)
            stages["softmax_probs"] = _flat_sdpa_stage_hf(attn_w)
            stages["weighted_v"] = _flat_sdpa_stage_hf(weighted)
            stages["sdpa_out"] = _flat_sdpa_stage_hf(attn_output)

        eager_call_idx[0] += 1
        return attn_output, attn_w

    qwen2_mod.eager_attention_forward = _patched_eager
    model.config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)
        return {name: stages[name] for name in L0_SDPA_STAGE_NAMES}, out.past_key_values
    finally:
        qwen2_mod.eager_attention_forward = orig_eager
        model.config._attn_implementation = prev_impl


def compare_named_stage_pccs(
    ref_stages: dict[str, torch.Tensor],
    tt_stages: dict[str, torch.Tensor],
    stage_names: tuple[str, ...],
    label: str,
) -> list[tuple[str, float, float]]:
    """Compare HF vs TT named stages; returns [(stage, pcc, max_abs_err), ...]."""
    results: list[tuple[str, float, float]] = []
    for name in stage_names:
        ref = ref_stages[name]
        tt = tt_stages[name]
        if ref.shape != tt.shape:
            raise AssertionError(f"{label} stage {name} shape mismatch: HF={tuple(ref.shape)} TT={tuple(tt.shape)}")
        _, pcc = comp_pcc(ref, tt, pcc=0.0)
        max_err = (ref - tt).abs().max().item()
        results.append((name, pcc, max_err))
    return results


def compare_l0_sdpa_stage_pccs(
    ref_stages: dict[str, torch.Tensor], tt_stages: dict[str, torch.Tensor]
) -> list[tuple[str, float, float]]:
    """Compare HF vs TT L0 decode SDPA stages."""
    return compare_named_stage_pccs(ref_stages, tt_stages, L0_SDPA_STAGE_NAMES, "L0 SDPA")


def print_l0_sdpa_stage_pcc_table(
    decode_step: int,
    position: int,
    cache_prefix_len: int,
    stage_pccs: list[tuple[str, float, float]],
    *,
    tt_path_note: str = "manual fp32 matmul/softmax (test probe)",
) -> None:
    """Print stage-by-stage L0 decode SDPA PCC table and first divergence."""
    print(f"\n[L0 decode SDPA stages] step={decode_step}  position={position}")
    print(f"  HF cache prefix (before token): {cache_prefix_len} tokens")
    print(f"  TT start_pos / RoPE index      : {position}")
    print(f"  valid attention prefix (incl. Q): {position + 1} tokens")
    print(f"  TT SDPA path                   : {tt_path_note}")
    print("Stage          | PCC     | max_abs_err")
    print("---------------|---------|------------")
    for name, pcc, max_err in stage_pccs:
        print(f"{name:14s} | {pcc:.5f} | {max_err:.6f}")

    first_div = next(
        (name for name, pcc, _ in stage_pccs if pcc < L0_ATTENTION_STAGE_PCC_NEAR_PERFECT),
        None,
    )
    if first_div is None:
        print(f"All SDPA stages PCC >= {L0_ATTENTION_STAGE_PCC_NEAR_PERFECT:.4f}")
    else:
        div_pcc = next(pcc for n, pcc, _ in stage_pccs if n == first_div)
        print(f"First SDPA stage below {L0_ATTENTION_STAGE_PCC_NEAR_PERFECT:.4f}: " f"{first_div} (PCC={div_pcc:.5f})")


def reference_lm_decode_l0_attention_stages(
    lm_state: dict, input_ids: torch.Tensor, vv_config, past_key_values, layer_idx: int = 0
) -> tuple[dict[str, torch.Tensor], object]:
    """HF decode step with hooks on L0 self-attn intermediates (test-only diagnostics)."""
    from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod

    model = _get_hf_reference_model(lm_state, vv_config)
    layer = model.layers[layer_idx]
    attn = layer.self_attn
    stages: dict[str, torch.Tensor] = {}
    handles = []

    def _save_proj(name: str):
        def hook(_module, _inp, out):
            stages[name] = _hf_attn_stage_flat(out)

        return hook

    for name, mod in (("q_proj", attn.q_proj), ("k_proj", attn.k_proj), ("v_proj", attn.v_proj)):
        handles.append(mod.register_forward_hook(_save_proj(name)))

    def _o_pre_hook(_module, args):
        stages["sdpa_out"] = _hf_attn_stage_flat(args[0])

    def _o_post_hook(_module, _inp, out):
        stages["o_proj_out"] = _hf_attn_stage_flat(out)

    handles.append(attn.o_proj.register_forward_pre_hook(_o_pre_hook))
    handles.append(attn.o_proj.register_forward_hook(_o_post_hook))

    orig_rope = qwen2_mod.apply_rotary_pos_emb
    rope_call_idx = [0]

    def _patched_rope(q, k, cos, sin, unsqueeze_dim=1):
        q2, k2 = orig_rope(q, k, cos, sin, unsqueeze_dim)
        if rope_call_idx[0] == layer_idx:
            stages["q_rope"] = _hf_attn_stage_flat(q2)
            stages["k_rope"] = _hf_attn_stage_flat(k2)
        rope_call_idx[0] += 1
        return q2, k2

    qwen2_mod.apply_rotary_pos_emb = _patched_rope
    try:
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)
        return {name: stages[name] for name in L0_ATTENTION_STAGE_NAMES}, out.past_key_values
    finally:
        qwen2_mod.apply_rotary_pos_emb = orig_rope
        for handle in handles:
            handle.remove()


def compare_l0_attention_stage_pccs(
    ref_stages: dict[str, torch.Tensor], tt_stages: dict[str, torch.Tensor]
) -> list[tuple[str, float, float]]:
    """Compare HF vs TT L0 attention stages; returns [(stage, pcc, max_abs_err), ...]."""
    results: list[tuple[str, float, float]] = []
    for name in L0_ATTENTION_STAGE_NAMES:
        ref = ref_stages[name]
        tt = tt_stages[name]
        if ref.shape != tt.shape:
            raise AssertionError(f"L0 stage {name} shape mismatch: HF={tuple(ref.shape)} TT={tuple(tt.shape)}")
        _, pcc = comp_pcc(ref, tt, pcc=0.0)
        max_err = (ref - tt).abs().max().item()
        results.append((name, pcc, max_err))
    return results


def print_l0_attention_stage_pcc_table(
    decode_step: int,
    position: int,
    cache_prefix_len: int,
    stage_pccs: list[tuple[str, float, float]],
) -> None:
    """Print stage-by-stage L0 decode attention PCC table and first divergence."""
    print(f"\n[L0 decode attention stages] step={decode_step}  position={position}")
    print(f"  HF cache prefix (before token): {cache_prefix_len} tokens")
    print(f"  TT start_pos / RoPE index      : {position}")
    print(f"  valid attention prefix (incl. Q): {position + 1} tokens")
    print("Stage      | PCC     | max_abs_err")
    print("-----------|---------|------------")
    for name, pcc, max_err in stage_pccs:
        print(f"{name:10s} | {pcc:.5f} | {max_err:.6f}")

    first_div = next(
        (name for name, pcc, _ in stage_pccs if pcc < L0_ATTENTION_STAGE_PCC_NEAR_PERFECT),
        None,
    )
    if first_div is None:
        print(f"All stages PCC >= {L0_ATTENTION_STAGE_PCC_NEAR_PERFECT:.4f}")
    else:
        div_pcc = next(pcc for n, pcc, _ in stage_pccs if n == first_div)
        print(f"First stage below {L0_ATTENTION_STAGE_PCC_NEAR_PERFECT:.4f}: " f"{first_div} (PCC={div_pcc:.5f})")


def print_l0_decode_attention_pcc(
    decode_step: int,
    position: int,
    cache_prefix_len: int,
    ref_attn_in: torch.Tensor,
    ref_attn_out: torch.Tensor,
    tt_attn_in: torch.Tensor,
    tt_attn_out: torch.Tensor,
) -> tuple[float, float]:
    """Print L0 decode attention PCC diagnostics; returns (attn_in_pcc, attn_out_pcc)."""
    _, in_pcc = comp_pcc(ref_attn_in, tt_attn_in, pcc=0.0)
    _, out_pcc = comp_pcc(ref_attn_out, tt_attn_out, pcc=0.0)
    out_err = (ref_attn_out - tt_attn_out).abs()

    print(f"\n[L0 decode attention] step={decode_step}  position={position}")
    print(f"  HF cache prefix (before token): {cache_prefix_len} tokens")
    print(f"  TT start_pos / RoPE index      : {position}")
    print(f"  valid attention prefix (incl. Q): {position + 1} tokens")
    print(f"  attn_input  (pre-norm) PCC     : {in_pcc:.5f}")
    print(f"  attn_output (post sdpa+proj)   : {out_pcc:.5f}")
    print(f"  attn_output max_abs_err={out_err.max().item():.6f}  " f"mean_abs_err={out_err.mean().item():.6f}")
    print(f"  shapes: HF={tuple(ref_attn_out.shape)} TT={tuple(tt_attn_out.shape)}")
    return in_pcc, out_pcc


class _TTVibeVoiceLMLayerProbe(TTVibeVoiceLM):
    """Test-only subclass: capture hidden states after embed and each transformer layer."""

    def forward_with_layer_hiddens(
        self,
        inputs_embeds: ttnn.Tensor,
        start_pos: int = 0,
        kv_cache=None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        cfg = self.cfg
        cos_tt, sin_tt = self._cos_tt, self._sin_tt

        x = inputs_embeds
        if x.dtype == ttnn.float32:
            x = ttnn.typecast(x, ttnn.bfloat16)

        layer_hiddens: list[torch.Tensor] = [_tt_tensor_to_hidden_torch(x)]
        for layer_idx in range(cfg.num_hidden_layers):
            x = self._transformer_layer(x, layer_idx, (cos_tt, sin_tt), kv_cache, start_pos)
            layer_hiddens.append(_tt_tensor_to_hidden_torch(x))

        x = ttnn.rms_norm(
            x,
            weight=self.w.norm_w,
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        final = _tt_tensor_to_hidden_torch(x)
        return layer_hiddens, final

    def forward_decoder_layer_hidden(
        self,
        hidden: torch.Tensor,
        start_pos: int,
        kv_cache,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Run one decoder layer on ``hidden`` [B, 1, H]; returns [B, 1, H] float32 (Devstral-style)."""
        x = hidden_torch_to_tt(hidden, self.device)
        x = self._transformer_layer(x, layer_idx, (self._cos_tt, self._sin_tt), kv_cache, start_pos)
        return _tt_tensor_to_hidden_torch(x)

    def l0_decode_attention(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        kv_cache,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run embed + L0 pre-norm + attention only; returns (attn_input, attn_output) as torch."""
        inputs_embeds = self._embed(input_ids)
        x = inputs_embeds
        if x.dtype == ttnn.float32:
            x = ttnn.typecast(x, ttnn.bfloat16)

        lw = self.w.layers[layer_idx]
        cos_tt, sin_tt = self._cos_tt, self._sin_tt
        x_norm = ttnn.rms_norm(
            x,
            weight=lw.attn_norm_w,
            epsilon=self.cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = self._attention_layer(x_norm, lw, (cos_tt, sin_tt), kv_cache, layer_idx, start_pos)
        return _tt_tensor_to_hidden_torch(x_norm), _tt_tensor_to_hidden_torch(attn_out)

    def l0_decode_attention_stages(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        kv_cache,
        layer_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Run embed + L0 pre-norm + decode attention step-by-step; capture stage outputs."""
        cfg = self.cfg
        lw = self.w.layers[layer_idx]
        cos_tt, sin_tt = self._cos_tt, self._sin_tt

        x = self._embed(input_ids)
        if x.dtype == ttnn.float32:
            x = ttnn.typecast(x, ttnn.bfloat16)
        x_norm = ttnn.rms_norm(
            x,
            weight=lw.attn_norm_w,
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        B = x_norm.shape[0]
        S = x_norm.shape[2]
        head_dim = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        assert S == 1, "l0_decode_attention_stages supports decode (S=1) only"

        stages: dict[str, torch.Tensor] = {}

        q = ttnn.linear(x_norm, lw.wq, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.linear(x_norm, lw.wk, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.linear(x_norm, lw.wv, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if lw.q_bias is not None:
            q = ttnn.add(q, lw.q_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if lw.k_bias is not None:
            k = ttnn.add(k, lw.k_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if lw.v_bias is not None:
            v = ttnn.add(v, lw.v_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        stages["q_proj"] = _tt_attn_stage_flat(q, "proj")
        stages["k_proj"] = _tt_attn_stage_flat(k, "proj")
        stages["v_proj"] = _tt_attn_stage_flat(v, "proj")

        q = ttnn.permute(_reshape_tt(q, [B, S, n_heads, head_dim]), (0, 2, 1, 3))
        k = ttnn.permute(_reshape_tt(k, [B, S, n_kv, head_dim]), (0, 2, 1, 3))
        v = ttnn.permute(_reshape_tt(v, [B, S, n_kv, head_dim]), (0, 2, 1, 3))

        c = ttnn.slice(
            cos_tt, [0, 0, start_pos, 0], [1, 1, start_pos + S, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        s = ttnn.slice(
            sin_tt, [0, 0, start_pos, 0], [1, 1, start_pos + S, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q = _apply_rope_ttnn(q, c, s)
        k = _apply_rope_ttnn(k, c, s)
        stages["q_rope"] = _tt_attn_stage_flat(q, "heads")
        stages["k_rope"] = _tt_attn_stage_flat(k, "heads")

        assert kv_cache is not None and kv_cache.keys[layer_idx] is not None, "decode needs an allocated KV cache"
        ttnn.update_cache(kv_cache.keys[layer_idx], k, start_pos)
        ttnn.update_cache(kv_cache.values[layer_idx], v, start_pos)

        cache_seq = kv_cache.max_seq or kv_cache.keys[layer_idx].shape[2]
        valid_len = start_pos + S
        k_chunk = _k_chunk_from_cache_seq(cache_seq)

        if _fused_sdpa_decode_safe(valid_len, k_chunk):
            q_dec = ttnn.permute(q, (0, 2, 1, 3))
            attn = ttnn.transformer.scaled_dot_product_attention_decode(
                q_dec,
                kv_cache.keys[layer_idx],
                kv_cache.values[layer_idx],
                cur_pos=[start_pos],
                scale=self.scale,
                program_config=_SDPA_DECODE_CFG,
                compute_kernel_config=_HIFI4,
            )
            out = _reshape_tt(attn, [B, 1, S, n_heads * head_dim])
        else:
            k_all = ttnn.slice(
                kv_cache.keys[layer_idx],
                [0, 0, 0, 0],
                [B, n_kv, valid_len, head_dim],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            v_all = ttnn.slice(
                kv_cache.values[layer_idx],
                [0, 0, 0, 0],
                [B, n_kv, valid_len, head_dim],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            repeat = n_heads // n_kv
            k_slices, v_slices = [], []
            for kv_idx in range(n_kv):
                kh = ttnn.slice(
                    k_all,
                    [0, kv_idx, 0, 0],
                    [B, kv_idx + 1, valid_len, head_dim],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                vh = ttnn.slice(
                    v_all,
                    [0, kv_idx, 0, 0],
                    [B, kv_idx + 1, valid_len, head_dim],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(repeat):
                    k_slices.append(kh)
                    v_slices.append(vh)
            k_rep = ttnn.concat(k_slices, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            v_rep = ttnn.concat(v_slices, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q_f32 = ttnn.typecast(q, ttnn.float32)
            k_f32 = ttnn.typecast(k_rep, ttnn.float32)
            v_f32 = ttnn.typecast(v_rep, ttnn.float32)
            k_t = ttnn.permute(k_f32, (0, 1, 3, 2))
            scores = ttnn.matmul(q_f32, k_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            scores = ttnn.mul(scores, self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn_w = ttnn.softmax(scores, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn = ttnn.matmul(attn_w, v_f32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.typecast(attn, ttnn.bfloat16)
            out = _reshape_tt(out, [B, 1, S, n_heads * head_dim])

        stages["sdpa_out"] = _tt_attn_stage_flat(out, "proj")
        out = ttnn.linear(out, lw.wo, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        stages["o_proj_out"] = _tt_attn_stage_flat(out, "hidden")
        return stages

    def l0_decode_sdpa_stages(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        kv_cache,
        layer_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Run L0 decode SDPA step-by-step via manual fp32 matmul/softmax; capture stages."""
        cfg = self.cfg
        lw = self.w.layers[layer_idx]
        cos_tt, sin_tt = self._cos_tt, self._sin_tt

        x = self._embed(input_ids)
        if x.dtype == ttnn.float32:
            x = ttnn.typecast(x, ttnn.bfloat16)
        x_norm = ttnn.rms_norm(
            x,
            weight=lw.attn_norm_w,
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        B = x_norm.shape[0]
        S = x_norm.shape[2]
        head_dim = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        assert S == 1, "l0_decode_sdpa_stages supports decode (S=1) only"

        q = ttnn.linear(x_norm, lw.wq, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.linear(x_norm, lw.wk, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.linear(x_norm, lw.wv, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if lw.q_bias is not None:
            q = ttnn.add(q, lw.q_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if lw.k_bias is not None:
            k = ttnn.add(k, lw.k_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if lw.v_bias is not None:
            v = ttnn.add(v, lw.v_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        q = ttnn.permute(_reshape_tt(q, [B, S, n_heads, head_dim]), (0, 2, 1, 3))
        k = ttnn.permute(_reshape_tt(k, [B, S, n_kv, head_dim]), (0, 2, 1, 3))
        v = ttnn.permute(_reshape_tt(v, [B, S, n_kv, head_dim]), (0, 2, 1, 3))

        c = ttnn.slice(
            cos_tt, [0, 0, start_pos, 0], [1, 1, start_pos + S, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        s = ttnn.slice(
            sin_tt, [0, 0, start_pos, 0], [1, 1, start_pos + S, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q = _apply_rope_ttnn(q, c, s)
        k = _apply_rope_ttnn(k, c, s)

        assert kv_cache is not None and kv_cache.keys[layer_idx] is not None, "decode needs an allocated KV cache"
        ttnn.update_cache(kv_cache.keys[layer_idx], k, start_pos)
        ttnn.update_cache(kv_cache.values[layer_idx], v, start_pos)

        valid_len = start_pos + S
        k_all = ttnn.slice(
            kv_cache.keys[layer_idx],
            [0, 0, 0, 0],
            [B, n_kv, valid_len, head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_all = ttnn.slice(
            kv_cache.values[layer_idx],
            [0, 0, 0, 0],
            [B, n_kv, valid_len, head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        repeat = n_heads // n_kv
        k_slices, v_slices = [], []
        for kv_idx in range(n_kv):
            kh = ttnn.slice(
                k_all,
                [0, kv_idx, 0, 0],
                [B, kv_idx + 1, valid_len, head_dim],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            vh = ttnn.slice(
                v_all,
                [0, kv_idx, 0, 0],
                [B, kv_idx + 1, valid_len, head_dim],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(repeat):
                k_slices.append(kh)
                v_slices.append(vh)
        k_rep = ttnn.concat(k_slices, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_rep = ttnn.concat(v_slices, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        stages: dict[str, torch.Tensor] = {}
        q_f32 = ttnn.typecast(q, ttnn.float32)
        k_f32 = ttnn.typecast(k_rep, ttnn.float32)
        v_f32 = ttnn.typecast(v_rep, ttnn.float32)
        k_t = ttnn.permute(k_f32, (0, 1, 3, 2))
        qk = ttnn.matmul(q_f32, k_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        stages["qk_scores"] = _flat_sdpa_stage_tt(qk)

        scaled = ttnn.mul(qk, self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        stages["scaled_scores"] = _flat_sdpa_stage_tt(scaled)
        stages["masked_scores"] = _flat_sdpa_stage_tt(scaled)

        attn_w = ttnn.softmax(scaled, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        stages["softmax_probs"] = _flat_sdpa_stage_tt(attn_w)

        weighted = ttnn.matmul(attn_w, v_f32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        stages["weighted_v"] = _flat_sdpa_stage_tt(weighted)

        out = ttnn.typecast(weighted, ttnn.bfloat16)
        out = _reshape_tt(out, [B, 1, S, n_heads * head_dim])
        stages["sdpa_out"] = _flat_sdpa_stage_tt(out)
        return stages

    def l0_decode_sdpa_stages_fused(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        kv_cache,
        layer_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Capture production fused ``scaled_dot_product_attention_decode`` output only."""
        stages = self.l0_decode_attention_stages(input_ids, start_pos, kv_cache, layer_idx=layer_idx)
        return {"sdpa_out": stages["sdpa_out"]}

    def capture_l0_decode_path_stages(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        kv_cache,
        layer_idx: int = 0,
        *,
        sdpa_mode: str,
    ) -> dict:
        """Capture L0 decode path stages for manual fp32 SDPA or fused SDPA (test-only).

        Returns a dict with ``meta`` and ``stages`` where each stage entry is either
        ``None`` (fused opaque stage) or ``{"tensor", "shape", "flat"}``.
        """
        if sdpa_mode not in {"manual", "fused"}:
            raise ValueError(f"sdpa_mode must be 'manual' or 'fused', got {sdpa_mode!r}")

        cfg = self.cfg
        lw = self.w.layers[layer_idx]
        cos_tt, sin_tt = self._cos_tt, self._sin_tt

        x = self._embed(input_ids)
        if x.dtype == ttnn.float32:
            x = ttnn.typecast(x, ttnn.bfloat16)
        x_residual_in = x

        x_norm = ttnn.rms_norm(
            x,
            weight=lw.attn_norm_w,
            epsilon=cfg.rms_norm_eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        B = x_norm.shape[0]
        S = x_norm.shape[2]
        head_dim = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        assert S == 1, "capture_l0_decode_path_stages supports decode (S=1) only"

        q = ttnn.linear(x_norm, lw.wq, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.linear(x_norm, lw.wk, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.linear(x_norm, lw.wv, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if lw.q_bias is not None:
            q = ttnn.add(q, lw.q_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if lw.k_bias is not None:
            k = ttnn.add(k, lw.k_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if lw.v_bias is not None:
            v = ttnn.add(v, lw.v_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        q = ttnn.permute(_reshape_tt(q, [B, S, n_heads, head_dim]), (0, 2, 1, 3))
        k = ttnn.permute(_reshape_tt(k, [B, S, n_kv, head_dim]), (0, 2, 1, 3))
        v = ttnn.permute(_reshape_tt(v, [B, S, n_kv, head_dim]), (0, 2, 1, 3))

        c = ttnn.slice(
            cos_tt, [0, 0, start_pos, 0], [1, 1, start_pos + S, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        s = ttnn.slice(
            sin_tt, [0, 0, start_pos, 0], [1, 1, start_pos + S, head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q = _apply_rope_ttnn(q, c, s)
        k = _apply_rope_ttnn(k, c, s)

        assert kv_cache is not None and kv_cache.keys[layer_idx] is not None, "decode needs an allocated KV cache"
        ttnn.update_cache(kv_cache.keys[layer_idx], k, start_pos)
        ttnn.update_cache(kv_cache.values[layer_idx], v, start_pos)

        valid_len = start_pos + S
        stages: dict[str, dict | None] = {name: None for name in L0_DECODE_PATH_COMPARE_STAGES}
        stages["x_embed"] = _stage_capture_from_tt(x_residual_in)

        if sdpa_mode == "manual":
            k_all = ttnn.slice(
                kv_cache.keys[layer_idx],
                [0, 0, 0, 0],
                [B, n_kv, valid_len, head_dim],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            v_all = ttnn.slice(
                kv_cache.values[layer_idx],
                [0, 0, 0, 0],
                [B, n_kv, valid_len, head_dim],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            repeat = n_heads // n_kv
            k_slices, v_slices = [], []
            for kv_idx in range(n_kv):
                kh = ttnn.slice(
                    k_all,
                    [0, kv_idx, 0, 0],
                    [B, kv_idx + 1, valid_len, head_dim],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                vh = ttnn.slice(
                    v_all,
                    [0, kv_idx, 0, 0],
                    [B, kv_idx + 1, valid_len, head_dim],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(repeat):
                    k_slices.append(kh)
                    v_slices.append(vh)
            k_rep = ttnn.concat(k_slices, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            v_rep = ttnn.concat(v_slices, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            q_f32 = ttnn.typecast(q, ttnn.float32)
            k_f32 = ttnn.typecast(k_rep, ttnn.float32)
            v_f32 = ttnn.typecast(v_rep, ttnn.float32)
            k_t = ttnn.permute(k_f32, (0, 1, 3, 2))
            qk = ttnn.matmul(q_f32, k_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            scaled = ttnn.mul(qk, self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn_w = ttnn.softmax(scaled, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            weighted = ttnn.matmul(attn_w, v_f32, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            stages["qk_scores"] = _stage_capture_from_tt(qk)
            stages["scaled_scores"] = _stage_capture_from_tt(scaled)
            stages["softmax_probs"] = _stage_capture_from_tt(attn_w)
            stages["weighted_v"] = _stage_capture_from_tt(weighted)

            out = ttnn.typecast(weighted, ttnn.bfloat16)
            out = _reshape_tt(out, [B, 1, S, n_heads * head_dim])
            stages["sdpa_out"] = _stage_capture_from_tt(out)
        else:
            q_dec = ttnn.permute(q, (0, 2, 1, 3))
            attn = ttnn.transformer.scaled_dot_product_attention_decode(
                q_dec,
                kv_cache.keys[layer_idx],
                kv_cache.values[layer_idx],
                cur_pos=[start_pos],
                scale=self.scale,
                program_config=_SDPA_DECODE_CFG,
                compute_kernel_config=_HIFI4,
            )
            out = _reshape_tt(attn, [B, 1, S, n_heads * head_dim])
            stages["sdpa_out"] = _stage_capture_from_tt(out)

        o_proj = ttnn.linear(out, lw.wo, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        stages["o_proj_out"] = _stage_capture_from_tt(o_proj)
        hidden_after = ttnn.add(x_residual_in, o_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        stages["hidden_after_residual"] = _stage_capture_from_tt(hidden_after)

        return {
            "meta": {
                "sdpa_mode": sdpa_mode,
                "layer_idx": layer_idx,
                "start_pos": start_pos,
                "valid_len": valid_len,
                "n_heads": n_heads,
                "n_kv": n_kv,
                "head_dim": head_dim,
                "hidden_dim": cfg.hidden_size,
                "scale": self.scale,
            },
            "stages": stages,
        }


def _tt_tensor_to_hidden_torch(x: ttnn.Tensor) -> torch.Tensor:
    """Convert [B, 1, S, H] device tensor to float32 torch [B, S, H]."""
    return ttnn.to_torch(ttnn.typecast(x, ttnn.float32)).to(torch.float32).squeeze(1)


def as_layer_probe(lm_tt: TTVibeVoiceLM) -> _TTVibeVoiceLMLayerProbe:
    """Rebind an existing LM instance as a layer probe (test-only, no extra weight copy)."""
    probe = _TTVibeVoiceLMLayerProbe.__new__(_TTVibeVoiceLMLayerProbe)
    probe.__dict__.update(lm_tt.__dict__)
    return probe


def hidden_torch_to_tt(hidden: torch.Tensor, device) -> ttnn.Tensor:
    """``hidden`` [B, 1, H] bf16 → TT ``[B, 1, 1, H]`` TILE."""
    if hidden.dim() != 3 or hidden.shape[1] != 1:
        raise ValueError(f"expected hidden [B, 1, H], got {tuple(hidden.shape)}")
    B, _, H = hidden.shape
    return ttnn.from_torch(
        hidden.reshape(B, 1, 1, H).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def advance_tt_decode(lm_tt: TTVibeVoiceLM, decode_tokens: torch.Tensor, prefill_len: int, kv_cache, num_steps: int):
    """Run ``num_steps`` TT decode steps starting at ``prefill_len``."""
    for step in range(num_steps):
        tt_decode_hidden(lm_tt, decode_tokens[:, step : step + 1], prefill_len + step, kv_cache)


def advance_hf_decode(
    lm_state: dict, decode_tokens: torch.Tensor, prefill_len: int, vv_config, hf_cache, num_steps: int
):
    """Run ``num_steps`` HF decode steps; returns updated ``past_key_values``."""
    for step in range(num_steps):
        _, hf_cache = reference_lm_decode_hidden(lm_state, decode_tokens[:, step : step + 1], vv_config, hf_cache)
    return hf_cache


def prepare_failing_decode_step_context(
    mesh_device, lm_state, vv_config, decode_step: int = DECODE_LAYERWISE_FAIL_STEP
):
    """Prefill + advance to ``decode_step`` on shared TT/HF caches (seed=0 token stream)."""
    torch.manual_seed(0)
    cfg = vv_config.decoder

    prompt = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), dtype=torch.long)
    decode_tokens = torch.randint(0, cfg.vocab_size, (1, DECODE_GENERATION_LENGTH), dtype=torch.long)
    token = decode_tokens[:, decode_step : decode_step + 1]
    position = SEQ_LEN + decode_step

    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    kv_cache = lm_tt.alloc_kv_cache(SEQ_LEN + DECODE_GENERATION_LENGTH + 8)
    tt_prefill_hidden(lm_tt, prompt, kv_cache)
    advance_tt_decode(lm_tt, decode_tokens, SEQ_LEN, kv_cache, decode_step)

    hf_cache = reference_lm_prefill_cache(lm_state, prompt, vv_config)
    hf_cache = advance_hf_decode(lm_state, decode_tokens, SEQ_LEN, vv_config, hf_cache, decode_step)

    return {
        "cfg": cfg,
        "token": token,
        "decode_step": decode_step,
        "position": position,
        "cache_prefix_len": hf_cache_seq_length(hf_cache),
        "lm_tt": lm_tt,
        "kv_cache": kv_cache,
        "hf_cache": hf_cache,
    }


def capture_fused_vs_manual_l0_decode_at_step(
    mesh_device,
    lm_state,
    vv_config,
    decode_step: int = DECODE_LAYERWISE_FAIL_STEP,
) -> dict:
    """Build twin TT contexts at ``decode_step`` and capture manual vs fused L0 path stages."""
    ctx_manual = prepare_failing_decode_step_context(mesh_device, lm_state, vv_config, decode_step=decode_step)
    ctx_fused = prepare_failing_decode_step_context(mesh_device, lm_state, vv_config, decode_step=decode_step)

    probe_manual = as_layer_probe(ctx_manual["lm_tt"])
    probe_fused = as_layer_probe(ctx_fused["lm_tt"])

    manual_capture = probe_manual.capture_l0_decode_path_stages(
        ctx_manual["token"],
        ctx_manual["position"],
        ctx_manual["kv_cache"],
        sdpa_mode="manual",
    )
    fused_capture = probe_fused.capture_l0_decode_path_stages(
        ctx_fused["token"],
        ctx_fused["position"],
        ctx_fused["kv_cache"],
        sdpa_mode="fused",
    )

    embed_manual = manual_capture["stages"]["x_embed"]["flat"]
    embed_fused = fused_capture["stages"]["x_embed"]["flat"]
    _, embed_pcc = comp_pcc(embed_manual, embed_fused, pcc=0.0)

    return {
        "decode_step": decode_step,
        "position": ctx_manual["position"],
        "token": ctx_manual["token"],
        "manual_capture": manual_capture,
        "fused_capture": fused_capture,
        "input_embed_pcc": embed_pcc,
    }


def _print_stage_health(label: str, stage: dict | None) -> None:
    if stage is None:
        print(f"    {label}: N/A")
        return
    h = _tensor_health_stats(stage["flat"])
    print(
        f"    {label}: shape={stage['shape']}  min={h['min']:.6g}  max={h['max']:.6g}  "
        f"mean={h['mean']:.6g}  std={h['std']:.6g}  nan={h['nan_count']}  inf={h['inf_count']}  "
        f"near_bf16_sat={h['near_bf16_saturation_frac']:.4f}"
    )


def print_fused_vs_manual_sdpa_investigation_report(investigation: dict) -> dict:
    """Print and return structured fused-vs-manual L0 decode SDPA investigation metrics."""
    manual = investigation["manual_capture"]
    fused = investigation["fused_capture"]
    meta = manual["meta"]
    decode_step = investigation["decode_step"]
    position = investigation["position"]
    n_heads = meta["n_heads"]
    head_dim = meta["head_dim"]
    valid_len = meta["valid_len"]

    print("\n" + "=" * 88)
    print(f"[fused vs manual fp32 SDPA investigation] decode_step={decode_step}  position={position}")
    print(f"valid_len={valid_len}  n_heads={n_heads}  head_dim={head_dim}  scale={meta['scale']:.6g}")
    print(f"input embed PCC (twin contexts): {investigation['input_embed_pcc']:.8f}")
    print("=" * 88)

    stage_metrics: dict[str, dict] = {}
    comparable_pccs: list[tuple[str, float]] = []

    for stage_name in L0_DECODE_PATH_COMPARE_STAGES:
        manual_stage = manual["stages"].get(stage_name)
        fused_stage = fused["stages"].get(stage_name)

        print(f"\n--- Stage: {stage_name} ---")
        if stage_name in FUSED_OPAQUE_SDPA_STAGES:
            print("  Fused path: opaque (kernel does not expose intermediate)")

        if manual_stage is not None:
            print(f"  Manual shape: {manual_stage['shape']}")
            _print_stage_health("Manual health", manual_stage)
        else:
            print("  Manual shape: N/A")

        if fused_stage is not None:
            print(f"  Fused shape:  {fused_stage['shape']}")
            _print_stage_health("Fused health", fused_stage)
        elif stage_name not in FUSED_OPAQUE_SDPA_STAGES:
            print("  Fused shape:  N/A")

        metrics = _compare_stage_tensors(manual_stage, fused_stage)
        stage_metrics[stage_name] = metrics

        if metrics.get("comparable"):
            comparable_pccs.append((stage_name, metrics["pcc"]))
            print(
                f"  Compare manual vs fused: PCC={metrics['pcc']:.6f}  "
                f"max_abs={metrics['max_abs_error']:.6g}  mean_abs={metrics['mean_abs_error']:.6g}  "
                f"rms={metrics['rms_error']:.6g}  cosine={metrics['cosine_similarity']:.6f}"
            )
            dist = _error_distribution_stats(manual_stage["flat"], fused_stage["flat"])
            print(
                f"  Error distribution: {dist['character']}  "
                f"top1%_error_fraction={dist['top1pct_error_fraction']:.4f}"
            )
            print(f"  Top-10 abs errors: {[f'{e:.6g}' for e in dist['top10_max_abs_errors']]}")
        else:
            print(f"  Compare manual vs fused: {metrics.get('reason', 'not comparable')}")

        if stage_name == "softmax_probs" and manual_stage is not None:
            probs_2d = manual_stage["tensor"].reshape(n_heads, valid_len)
            sm = _softmax_distribution_stats(probs_2d)
            uniform_entropy = math.log(valid_len)
            print(
                f"  Manual softmax: entropy_mean={sm['entropy_mean']:.4f}  "
                f"(uniform baseline={uniform_entropy:.4f})  "
                f"max_prob_mean={sm['max_prob_mean']:.4f}  max_prob_max={sm['max_prob_max']:.4f}"
            )
            peakedness = "peaked" if sm["entropy_mean"] < uniform_entropy * 0.5 else "moderate"
            print(f"  Manual distribution shape: {peakedness} (lower entropy => more peaked)")
            print(f"  Manual top-k prob means (k=1..5): {[f'{v:.4f}' for v in sm['topk_prob_means']]}")

            scaled_stage = manual["stages"].get("scaled_scores")
            if scaled_stage is not None:
                scaled_2d = scaled_stage["tensor"].reshape(n_heads, valid_len)
                bf16_probs = _torch_bf16_softmax_from_scaled(scaled_2d)
                bf16_sm = _softmax_distribution_stats(bf16_probs)
                _, bf16_pcc = comp_pcc(probs_2d.reshape(-1), bf16_probs.reshape(-1), pcc=0.0)
                print(
                    f"  Torch bf16-sim softmax vs manual fp32 softmax: PCC={bf16_pcc:.6f}  "
                    f"entropy_delta={bf16_sm['entropy_mean'] - sm['entropy_mean']:+.4f}  "
                    f"max_prob_delta={bf16_sm['max_prob_mean'] - sm['max_prob_mean']:+.4f}"
                )

        if stage_name == "sdpa_out" and manual_stage is not None and fused_stage is not None:
            ref_2d = manual_stage["tensor"].reshape(n_heads, head_dim)
            cmp_2d = fused_stage["tensor"].reshape(n_heads, head_dim)
            per_head = _per_head_pcc(ref_2d, cmp_2d)
            if per_head:
                worst_h = min(range(len(per_head)), key=lambda i: per_head[i])
                best_h = max(range(len(per_head)), key=lambda i: per_head[i])
                print(
                    f"  Per-head sdpa_out PCC: min={min(per_head):.6f} (h{worst_h})  "
                    f"max={max(per_head):.6f} (h{best_h})  mean={sum(per_head)/len(per_head):.6f}"
                )
                low_heads = sorted(range(len(per_head)), key=lambda i: per_head[i])[:5]
                print("  Lowest-5 head PCCs: " + ", ".join(f"h{h}={per_head[h]:.5f}" for h in low_heads))

    print("\n" + "=" * 88)
    print("[Investigation report]")
    print("=" * 88)

    first_sig = next(
        (name for name, pcc in comparable_pccs if pcc < FUSED_VS_MANUAL_PCC_SIGNIFICANT_DROP),
        comparable_pccs[0][0] if comparable_pccs else "none",
    )
    print(f"\n1. First stage with significant manual-vs-fused difference: {first_sig}")
    if comparable_pccs:
        print("   Stage-wise PCC (manual as reference, fused as candidate):")
        prev_pcc = 1.0
        for name, pcc in comparable_pccs:
            delta = prev_pcc - pcc
            growth = "sudden" if delta > 0.003 else "gradual"
            sig = " <-- first significant drop" if name == first_sig else ""
            print(f"     {name:24s}  PCC={pcc:.6f}  additional_loss={delta:+.6f}  ({growth}){sig}")
            prev_pcc = pcc

    sdpa_metrics = stage_metrics.get("sdpa_out", {})
    hidden_metrics = stage_metrics.get("hidden_after_residual", {})
    oproj_metrics = stage_metrics.get("o_proj_out", {})
    if sdpa_metrics.get("comparable") and hidden_metrics.get("comparable"):
        sdpa_pcc = sdpa_metrics["pcc"]
        hidden_pcc = hidden_metrics["pcc"]
        total_loss = 1.0 - hidden_pcc
        sdpa_loss = 1.0 - sdpa_pcc
        frac_at_sdpa = sdpa_loss / total_loss if total_loss > 1e-9 else float("nan")
        print(f"\n2. Error introduced per subsequent stage (PCC loss = 1 - PCC vs manual):")
        if oproj_metrics.get("comparable"):
            oproj_pcc = oproj_metrics["pcc"]
            print(f"     sdpa_out -> o_proj_out:           {(1-oproj_pcc) - sdpa_loss:+.6f}")
            print(f"     o_proj_out -> hidden_after_res:   {(1-hidden_pcc) - (1-oproj_pcc):+.6f}")
        print(
            f"\n3. Majority of final PCC loss already at sdpa_out? "
            f"{frac_at_sdpa*100:.1f}% of total hidden loss is present immediately after SDPA "
            f"(sdpa PCC={sdpa_pcc:.6f}, hidden PCC={hidden_pcc:.6f})."
        )

    if sdpa_metrics.get("comparable") and oproj_metrics.get("comparable"):
        sdpa_pcc = sdpa_metrics["pcc"]
        oproj_pcc = oproj_metrics["pcc"]
        oproj_delta = sdpa_pcc - oproj_pcc
        behavior = "amplifies" if oproj_delta > 0.001 else "propagates"
        print(
            f"\n4. Output projection {behavior} SDPA error "
            f"(sdpa PCC={sdpa_pcc:.6f}, o_proj PCC={oproj_pcc:.6f}, delta={oproj_delta:+.6f})."
        )

    scaled_stage = manual["stages"].get("scaled_scores")
    softmax_stage = manual["stages"].get("softmax_probs")
    sdpa_stage_m = manual["stages"].get("sdpa_out")
    sdpa_stage_f = fused["stages"].get("sdpa_out")
    conclusion_parts = []
    if scaled_stage is not None and softmax_stage is not None:
        scaled_2d = scaled_stage["tensor"].reshape(n_heads, valid_len)
        bf16_probs = _torch_bf16_softmax_from_scaled(scaled_2d)
        _, bf16_sm_pcc = comp_pcc(softmax_stage["flat"], bf16_probs.reshape(-1), pcc=0.0)
        conclusion_parts.append(f"manual fp32 softmax vs torch-bf16-sim softmax PCC={bf16_sm_pcc:.6f}")
    if sdpa_stage_m is not None and sdpa_stage_f is not None:
        _, sdpa_cmp_pcc = comp_pcc(sdpa_stage_m["flat"], sdpa_stage_f["flat"], pcc=0.0)
        ref_2d = sdpa_stage_m["tensor"].reshape(n_heads, head_dim)
        cmp_2d = sdpa_stage_f["tensor"].reshape(n_heads, head_dim)
        per_head = _per_head_pcc(ref_2d, cmp_2d)
        head_spread = max(per_head) - min(per_head) if per_head else 0.0
        if sdpa_cmp_pcc >= 0.995:
            conclusion_parts.append(
                "fused-vs-manual sdpa_out difference is small; drift may be elsewhere or masked at L0-only scope"
            )
        elif sdpa_cmp_pcc >= 0.985 and head_spread < 0.05:
            conclusion_parts.append(
                "fused-vs-manual sdpa_out gap is moderate and uniform across heads; "
                "consistent with accumulated bf16/softmax numerics"
            )
        elif head_spread >= 0.05:
            worst_h = min(range(len(per_head)), key=lambda i: per_head[i])
            conclusion_parts.append(
                f"fused-vs-manual sdpa_out shows head-specific outlier (h{worst_h} PCC={per_head[worst_h]:.4f}, "
                f"spread={head_spread:.4f}); this pattern suggests a possible fused-kernel head/indexing issue "
                "rather than uniform bf16 softmax noise"
            )
        else:
            conclusion_parts.append(
                "fused-vs-manual sdpa_out gap is large relative to typical bf16 softmax noise; "
                "suggests possible fused-kernel implementation or reduction-order issue"
            )

    print("\n5. Numerics vs implementation hypothesis:")
    for part in conclusion_parts:
        print(f"   - {part}")
    print(
        "   - Fused kernel intermediates (qk/scaled/softmax/weighted_v) are opaque; divergence is first "
        "observable at sdpa_out when comparing against the decomposed manual fp32 path."
    )

    print("\n" + "=" * 88)
    return {
        "decode_step": decode_step,
        "position": position,
        "stage_metrics": stage_metrics,
        "comparable_pccs": comparable_pccs,
        "first_significant_stage": first_sig,
        "input_embed_pcc": investigation["input_embed_pcc"],
    }


def compare_layerwise_decode_pcc(
    ref_layers: list[torch.Tensor],
    ref_final: torch.Tensor,
    tt_layers: list[torch.Tensor],
    tt_final: torch.Tensor,
) -> list[tuple[str, float]]:
    """Compare HF vs TT hidden states layer-by-layer; returns [(label, pcc), ...]."""
    if len(ref_layers) != len(tt_layers):
        raise AssertionError(f"Layer count mismatch: HF={len(ref_layers)} TT={len(tt_layers)}")

    results: list[tuple[str, float]] = []
    for idx, (ref_h, tt_h) in enumerate(zip(ref_layers, tt_layers)):
        label = "embed" if idx == 0 else f"L{idx - 1}"
        _, pcc = comp_pcc(ref_h, tt_h, pcc=0.0)
        results.append((label, pcc))

    _, final_pcc = comp_pcc(ref_final, tt_final, pcc=0.0)
    results.append(("final", final_pcc))
    return results


def print_layerwise_decode_pcc_table(decode_step: int, position: int, layer_pccs: list[tuple[str, float]]) -> None:
    """Print layer-wise PCC for one decode step."""
    print(f"\n[layerwise decode] step={decode_step}  position={position}")
    print("Layer  | PCC")
    print("-------|--------")
    for label, pcc in layer_pccs:
        print(f"{label:6s} | {pcc:.5f}")
    pccs = [p for _, p in layer_pccs]
    print(f"min={min(pccs):.5f}  first_below_0.99={next((l for l, p in layer_pccs if p < PCC_THRESHOLD), 'none')}")


def build_tt_lm(lm_state: dict, mesh_device, cfg) -> TTVibeVoiceLM:
    weights = preprocess_lm_weights(lm_state, mesh_device, cfg)
    return TTVibeVoiceLM(weights, mesh_device)


def tt_prefill_hidden(lm_tt: TTVibeVoiceLM, input_ids: torch.Tensor, kv_cache) -> torch.Tensor:
    """Run TT prefill and return full-sequence hidden states [B, S, hidden].

    ``prefill()`` / ``prefill_embeds()`` only return the final chunk's hidden states when
    S > PREFILL_CHUNK_SIZE.  For ISL-sweep PCC we mirror the model's chunk loop here and
    concatenate per-chunk hiddens so the reference comparison spans all positions.
    """
    seq_len = input_ids.shape[1]
    if seq_len <= PREFILL_CHUNK_SIZE:
        _, tt_hidden = lm_tt.prefill(input_ids, kv_cache=kv_cache, return_last_hidden=True)
        return ttnn.to_torch(tt_hidden).to(torch.float32).squeeze(1)

    inputs_embeds = lm_tt._embed(input_ids)
    hidden_dim = inputs_embeds.shape[-1]
    hidden_parts = []
    for start in range(0, seq_len, PREFILL_CHUNK_SIZE):
        end = min(start + PREFILL_CHUNK_SIZE, seq_len)
        chunk = ttnn.slice(
            inputs_embeds,
            [0, 0, start, 0],
            [1, 1, end, hidden_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _, tt_hidden = lm_tt.forward(
            chunk,
            start_pos=start,
            kv_cache=kv_cache,
            return_last_hidden=True,
        )
        hidden_parts.append(ttnn.to_torch(tt_hidden).to(torch.float32).squeeze(1))
    return torch.cat(hidden_parts, dim=1)


def tt_decode_hidden(lm_tt: TTVibeVoiceLM, input_id: torch.Tensor, start_pos: int, kv_cache) -> torch.Tensor:
    _, tt_dec_hidden = lm_tt.decode_step(input_id, start_pos, kv_cache, return_last_hidden=True)
    return ttnn.to_torch(tt_dec_hidden).to(torch.float32).squeeze(1)  # [B, 1, hidden]


def hf_cache_seq_length(past_key_values) -> int:
    """Return the number of tokens stored in HF ``past_key_values``."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    if hasattr(past_key_values, "key_cache") and past_key_values.key_cache:
        return past_key_values.key_cache[0].shape[-2]
    return past_key_values[0][0].shape[2]


def expected_decode_start_pos(prefill_len: int, step: int) -> int:
    """Absolute RoPE/KV index for decode step ``step`` after a ``prefill_len``-token prefill."""
    return prefill_len + step


def verify_decode_cache_positions(
    step: int,
    prefill_len: int,
    start_pos: int,
    hf_cache,
    tt_positions: list[int],
) -> None:
    """Assert decode ``start_pos`` and HF cache length advance by one token per step."""
    expected_pos = expected_decode_start_pos(prefill_len, step)
    expected_hf_len = prefill_len + step
    if start_pos != expected_pos:
        raise AssertionError(
            f"decode step {step}: TT start_pos={start_pos}, expected {expected_pos} " f"(prefill_len={prefill_len})"
        )
    hf_len = hf_cache_seq_length(hf_cache)
    if hf_len != expected_hf_len:
        raise AssertionError(
            f"decode step {step}: HF cache length={hf_len}, expected {expected_hf_len} " f"(prefill_len={prefill_len})"
        )
    tt_positions.append(start_pos)


def compare_decode_hidden_pcc(ref_decode: torch.Tensor, tt_decode: torch.Tensor):
    """Compare single-step decode hidden states; returns (passed, pcc)."""
    ref_f = ref_decode.to(torch.float32)
    tt_f = tt_decode.to(torch.float32)
    if ref_f.shape != tt_f.shape:
        raise AssertionError(f"Decode hidden shape mismatch: ref={tuple(ref_f.shape)} tt={tuple(tt_f.shape)}")
    return comp_pcc(ref_f, tt_f, pcc=PCC_THRESHOLD)


def print_decode_pcc_summary(step_pccs: list[float]) -> None:
    """Print Step | PCC table for a multi-step decode sweep."""
    print("\nDecode PCC summary:")
    print("Step | PCC")
    print("-----|--------")
    for step, pcc in enumerate(step_pccs):
        print(f"{step:4d} | {pcc:.5f}")
    print(f"min={min(step_pccs):.5f}  mean={sum(step_pccs) / len(step_pccs):.5f}")


@contextlib.contextmanager
def force_manual_fp32_decode_sdpa():
    """Test-only monkeypatch: route decode through existing manual fp32 SDPA in ``_attention_layer``.

    Forces ``_fused_sdpa_decode_safe`` to return False so production code takes the existing
    ``else`` branch (fp32 QKᵀ → scale → ``ttnn.softmax`` → fp32 @ V). Does not alter that branch.
    """
    import models.experimental.vibevoice.tt.ttnn_vibevoice_lm as lm_mod

    original = lm_mod._fused_sdpa_decode_safe

    def _always_manual(_valid_len: int, _k_chunk: int) -> bool:
        return False

    lm_mod._fused_sdpa_decode_safe = _always_manual
    try:
        yield
    finally:
        lm_mod._fused_sdpa_decode_safe = original


def run_multi_step_decode_pcc_sweep(mesh_device, lm_state, vv_config, *, label: str) -> list[float]:
    """Run 32-token prefill + 10 decode steps (seed=0); return per-step PCC list."""
    torch.manual_seed(0)
    cfg = vv_config.decoder

    prompt = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), dtype=torch.long)
    decode_tokens = torch.randint(0, cfg.vocab_size, (1, DECODE_GENERATION_LENGTH), dtype=torch.long)

    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    kv_cache = lm_tt.alloc_kv_cache(SEQ_LEN + DECODE_GENERATION_LENGTH + 8)
    tt_prefill_hidden(lm_tt, prompt, kv_cache)

    hf_cache = reference_lm_prefill_cache(lm_state, prompt, vv_config)
    prefill_hf_len = hf_cache_seq_length(hf_cache)
    if prefill_hf_len != SEQ_LEN:
        raise AssertionError(f"HF cache length after prefill={prefill_hf_len}, expected {SEQ_LEN}")

    step_pccs: list[float] = []
    tt_positions: list[int] = []
    print(f"\n[{label}] multi-step decode after prefill (seed=0, threshold={PCC_THRESHOLD})")

    for step in range(DECODE_GENERATION_LENGTH):
        token = decode_tokens[:, step : step + 1]
        position = SEQ_LEN + step

        verify_decode_cache_positions(step, SEQ_LEN, position, hf_cache, tt_positions)

        ref_decode, hf_cache = reference_lm_decode_hidden(lm_state, token, vv_config, hf_cache)
        tt_decode = tt_decode_hidden(lm_tt, token, position, kv_cache)
        _, pcc_d = compare_decode_hidden_pcc(ref_decode, tt_decode)
        step_pccs.append(pcc_d)
        print(f"Decode step {step}  PCC={pcc_d:.5f}")

    assert_tt_decode_positions_monotonic(tt_positions, SEQ_LEN, DECODE_GENERATION_LENGTH)
    print_decode_pcc_summary(step_pccs)
    return step_pccs


def print_manual_fp32_vs_fused_sdpa_comparison(
    manual_pccs: list[float],
    fused_pccs: list[float] | None = None,
) -> None:
    """Print manual fp32 SDPA vs fused-kernel baseline comparison for the decode drift experiment."""
    fused = list(fused_pccs or FUSED_MULTI_STEP_DECODE_PCC_BASELINE)
    if len(manual_pccs) != len(fused):
        raise AssertionError(f"PCC length mismatch: manual={len(manual_pccs)} fused={len(fused)}")

    print("\n[manual fp32 SDPA vs fused SDPA decode] comparison (seed=0)")
    print("Step | Manual PCC | Fused PCC | Delta   | Manual pass")
    print("-----|------------|-----------|---------|------------")
    manual_failures = []
    fused_failures = []
    for step, (manual, fused_pcc) in enumerate(zip(manual_pccs, fused)):
        delta = manual - fused_pcc
        manual_pass = manual >= PCC_THRESHOLD
        if not manual_pass:
            manual_failures.append(step)
        if fused_pcc < PCC_THRESHOLD:
            fused_failures.append(step)
        print(
            f"{step:4d} | {manual:.5f}     | {fused_pcc:.5f}    | {delta:+.5f} | "
            f"{'PASS' if manual_pass else 'FAIL'}"
        )

    manual_all_pass = len(manual_failures) == 0
    fused_all_pass = len(fused_failures) == 0
    print(f"\nManual fp32 SDPA: all steps >= {PCC_THRESHOLD}? {manual_all_pass}")
    if manual_failures:
        print(f"  Manual failing steps: {manual_failures}")
    print(f"Fused SDPA baseline: all steps >= {PCC_THRESHOLD}? {fused_all_pass}")
    if fused_failures:
        print(f"  Fused failing steps: {fused_failures}")

    if manual_all_pass and not fused_all_pass:
        print(
            "\nConclusion: Forcing manual fp32 SDPA passes all steps while fused SDPA did not. "
            "This strongly indicates the fused decode SDPA kernel is the primary source of drift."
        )
    elif manual_all_pass and fused_all_pass:
        print("\nConclusion: Both paths pass; fused-kernel drift is not reproduced on this run.")
    elif not manual_all_pass and not fused_all_pass:
        improved = [s for s in range(len(manual_pccs)) if manual_pccs[s] >= PCC_THRESHOLD and fused[s] < PCC_THRESHOLD]
        if improved:
            print(
                f"\nConclusion: Manual fp32 SDPA improved steps {improved} above threshold, but drift remains elsewhere "
                "(KV cache accumulation, non-SDPA ops, or manual softmax numerics)."
            )
        else:
            print(
                "\nConclusion: Manual fp32 SDPA did not fix failing steps. Drift is likely NOT caused solely by "
                "the fused decode SDPA kernel."
            )
    else:
        print("\nConclusion: Unexpected pattern — review per-step deltas above.")


@contextlib.contextmanager
def hf_eager_bf16_softmax_attention():
    """Test-only: HF eager attention keeping softmax in activation dtype (bf16), not fp32."""
    from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv

    orig_eager = qwen2_mod.eager_attention_forward

    def _patched_eager(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    qwen2_mod.eager_attention_forward = _patched_eager
    try:
        yield
    finally:
        qwen2_mod.eager_attention_forward = orig_eager


def run_multi_step_decode_pcc_sweep_with_hf_attn(
    mesh_device,
    lm_state,
    vv_config,
    *,
    hf_attn_implementation: str,
    label: str,
    hf_softmax_patch: contextlib.AbstractContextManager | None = None,
) -> list[float]:
    """Same seed=0 multi-step sweep, but HF prefill+decode use ``hf_attn_implementation``."""
    patch = hf_softmax_patch if hf_softmax_patch is not None else contextlib.nullcontext()
    with patch:
        torch.manual_seed(0)
        cfg = vv_config.decoder

        prompt = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), dtype=torch.long)
        decode_tokens = torch.randint(0, cfg.vocab_size, (1, DECODE_GENERATION_LENGTH), dtype=torch.long)

        lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
        kv_cache = lm_tt.alloc_kv_cache(SEQ_LEN + DECODE_GENERATION_LENGTH + 8)
        tt_prefill_hidden(lm_tt, prompt, kv_cache)

        hf_cache = reference_lm_prefill_cache(lm_state, prompt, vv_config, attn_implementation=hf_attn_implementation)
        prefill_hf_len = hf_cache_seq_length(hf_cache)
        if prefill_hf_len != SEQ_LEN:
            raise AssertionError(f"HF cache length after prefill={prefill_hf_len}, expected {SEQ_LEN}")

        step_pccs: list[float] = []
        tt_positions: list[int] = []
        print(f"\n[{label}] multi-step decode (seed=0, HF attn={hf_attn_implementation})")

        for step in range(DECODE_GENERATION_LENGTH):
            token = decode_tokens[:, step : step + 1]
            position = SEQ_LEN + step

            verify_decode_cache_positions(step, SEQ_LEN, position, hf_cache, tt_positions)

            ref_decode, hf_cache = reference_lm_decode_hidden(
                lm_state,
                token,
                vv_config,
                hf_cache,
                attn_implementation=hf_attn_implementation,
            )
            tt_decode = tt_decode_hidden(lm_tt, token, position, kv_cache)
            _, pcc_d = compare_decode_hidden_pcc(ref_decode, tt_decode)
            step_pccs.append(pcc_d)
            print(f"Decode step {step}  PCC={pcc_d:.5f}")

        assert_tt_decode_positions_monotonic(tt_positions, SEQ_LEN, DECODE_GENERATION_LENGTH)
        print_decode_pcc_summary(step_pccs)
        return step_pccs


def print_hf_decode_reference_comparison(reference_runs: dict[str, list[float]]) -> None:
    """Print per-step PCC for multiple HF decode reference modes vs the same TT fused run."""
    names = list(reference_runs.keys())
    n = len(next(iter(reference_runs.values())))
    print("\n[HF decode reference comparison vs TT fused SDPA] seed=0")
    header = "Step | " + " | ".join(f"{name:>12s}" for name in names) + " | best"
    print(header)
    print("-----|" + "|".join("-" * 14 for _ in names) + "|--------")
    for step in range(n):
        row_pccs = {name: reference_runs[name][step] for name in names}
        best = max(row_pccs, key=row_pccs.get)
        vals = " | ".join(f"{row_pccs[name]:12.5f}" for name in names)
        print(f"{step:4d} | {vals} | {best}")

    print("\nSummary (all steps >= 0.99?):")
    for name in names:
        pccs = reference_runs[name]
        fails = [i for i, p in enumerate(pccs) if p < PCC_THRESHOLD]
        print(
            f"  {name:12s}: min={min(pccs):.5f} mean={sum(pccs)/len(pccs):.5f}  pass={not fails}  fails={fails or 'none'}"
        )


def assert_tt_decode_positions_monotonic(tt_positions: list[int], prefill_len: int, num_steps: int) -> None:
    """Verify TT decode positions increase by 1 each step, starting at ``prefill_len``."""
    expected = [expected_decode_start_pos(prefill_len, step) for step in range(num_steps)]
    if tt_positions != expected:
        raise AssertionError(f"TT decode positions {tt_positions} != expected {expected}")


def compare_prefill_hidden_pcc(
    ref_prefill: torch.Tensor,
    tt_prefill: torch.Tensor,
    seq_len: int,
    *,
    per_token: bool = True,
):
    """Compare prefill hidden states; returns (passed, overall_pcc, per_position_pcc).

    Set ``per_token=False`` for long ISL sweeps — per-position ``comp_pcc`` in a Python loop
    scales linearly with sequence length and makes 2k+ runs impractically slow.
    """
    ref_f = ref_prefill.to(torch.float32)
    tt_f = tt_prefill.to(torch.float32)
    if ref_f.shape != tt_f.shape:
        raise AssertionError(
            f"Prefill hidden shape mismatch for seq_len={seq_len}: ref={tuple(ref_f.shape)} tt={tuple(tt_f.shape)}"
        )
    passed_p, pcc_p = comp_pcc(ref_f, tt_f, pcc=PCC_THRESHOLD)
    if not per_token:
        return passed_p, pcc_p, []
    per_pos = [comp_pcc(ref_f[:, p], tt_f[:, p], pcc=PCC_THRESHOLD)[1] for p in range(seq_len)]
    return passed_p, pcc_p, per_pos


def prefill_isl_sweep_effective_lengths(vv_config, isl_lengths=None) -> tuple[list[int], int]:
    """Return ISL list capped by ``decoder.max_position_embeddings`` and the model limit."""
    lengths = list(isl_lengths or PREFILL_ISL_EXTENDED_SWEEP_LENGTHS)
    max_pos = vv_config.decoder.max_position_embeddings
    effective = [n for n in lengths if n <= max_pos]
    return effective, max_pos


def run_prefill_isl_sweep_timed(
    mesh_device,
    lm_state,
    vv_config,
    isl_lengths=None,
    *,
    verbose_debug_max: int = 0,
    per_token_pcc_max: int = 1024,
) -> list[dict]:
    """Run prefill PCC sweep with HF/TT wall times per input sequence length."""
    cfg = vv_config.decoder
    effective_lengths, max_pos = prefill_isl_sweep_effective_lengths(vv_config, isl_lengths)
    skipped = [n for n in (isl_lengths or PREFILL_ISL_EXTENDED_SWEEP_LENGTHS) if n > max_pos]

    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    results: list[dict] = []

    if skipped:
        print(
            f"[prefill ISL sweep] skipping lengths > max_position_embeddings={max_pos}: "
            + ", ".join(str(n) for n in skipped),
            flush=True,
        )

    for seq_len in effective_lengths:
        print(f"[prefill ISL sweep] ISL={seq_len} starting...", flush=True)
        torch.manual_seed(0)
        input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len), dtype=torch.long)
        row: dict = {"seq_len": seq_len, "max_position_embeddings": max_pos}

        try:
            t0 = time.perf_counter()
            ref_prefill = reference_lm_forward(lm_state, input_ids, vv_config)
            row["hf_sec"] = time.perf_counter() - t0
            print(f"[prefill ISL sweep] ISL={seq_len} HF done in {row['hf_sec']:.1f}s", flush=True)

            kv_cache = lm_tt.alloc_kv_cache(seq_len + 8)
            row["kv_cache_requested"] = seq_len + 8
            row["kv_cache_aligned"] = kv_cache.max_seq

            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            tt_prefill = tt_prefill_hidden(lm_tt, input_ids, kv_cache)
            ttnn.synchronize_device(mesh_device)
            row["tt_sec"] = time.perf_counter() - t0
            print(f"[prefill ISL sweep] ISL={seq_len} TT done in {row['tt_sec']:.1f}s", flush=True)

            if verbose_debug_max and seq_len <= verbose_debug_max:
                print_prefill_pcc_isl_debug(ref_prefill, tt_prefill, seq_len)

            per_token = seq_len <= per_token_pcc_max
            passed_p, pcc_p, per_pos = compare_prefill_hidden_pcc(ref_prefill, tt_prefill, seq_len, per_token=per_token)
            row.update(
                {
                    "status": "ok",
                    "overall_pcc": pcc_p,
                    "min_pcc": min(per_pos) if per_pos else pcc_p,
                    "median_pcc": sorted(per_pos)[seq_len // 2] if per_pos else pcc_p,
                    "last_pcc": per_pos[-1] if per_pos else pcc_p,
                    "pcc_pass": passed_p,
                    "total_sec": row["hf_sec"] + row["tt_sec"],
                }
            )
            print(
                f"[prefill ISL sweep] ISL={seq_len} overall_PCC={pcc_p:.5f} "
                f"pass={'yes' if passed_p else 'no'} total={row['total_sec']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            row.update({"status": "error", "error": str(exc)})
            print(f"[prefill ISL sweep] ISL={seq_len} ERROR: {exc}", flush=True)

        results.append(row)

    return results


def print_prefill_isl_sweep_timing_table(results: list[dict]) -> None:
    """Print ISL sweep summary: timing + PCC per sequence length."""
    print("\n[prefill ISL sweep] timing + PCC summary")
    print("ISL    | HF(s)  | TT(s)  | Total(s) | KV aligned | Overall PCC | Min PCC | Pass | Status")
    print("-------|--------|--------|----------|------------|-------------|---------|------|-------")
    for row in results:
        if row.get("status") != "ok":
            print(
                f"{row['seq_len']:6d} |   —    |   —    |     —    |     —      |      —      |    —    |  —   | "
                f"ERROR: {row.get('error', 'unknown')}"
            )
            continue
        print(
            f"{row['seq_len']:6d} | {row['hf_sec']:6.2f} | {row['tt_sec']:6.2f} | {row['total_sec']:8.2f} | "
            f"{row['kv_cache_aligned']:10d} | {row['overall_pcc']:11.5f} | {row['min_pcc']:7.5f} | "
            f"{'yes' if row['pcc_pass'] else 'no':4s} | ok"
        )


def print_prefill_pcc_isl_debug(
    ref_prefill: torch.Tensor,
    tt_prefill: torch.Tensor,
    seq_len: int,
    *,
    chunk_size: int = PREFILL_CHUNK_SIZE,
) -> None:
    """Temporary ISL-sweep diagnostics: per-token PCC, chunk boundaries, outlier stats."""
    ref_f = ref_prefill.to(torch.float32)
    tt_f = tt_prefill.to(torch.float32)

    print(
        f"[prefill debug] shape check: HF={tuple(ref_f.shape)} TT={tuple(tt_f.shape)} match={ref_f.shape == tt_f.shape}"
    )
    if ref_f.shape != tt_f.shape:
        print("[prefill debug] ABORT — hidden-state shapes differ; remaining diagnostics skipped.")
        return

    per_pos = [comp_pcc(ref_f[:, p], tt_f[:, p], pcc=0.0)[1] for p in range(seq_len)]

    print(f"[prefill debug] per-token PCC ({seq_len} tokens):")
    for p in range(seq_len):
        print(f"  p{p:4d}  PCC={per_pos[p]:.6f}")

    lows = sorted(range(seq_len), key=lambda i: per_pos[i])[:10]
    print("[prefill debug] 10 lowest-PCC positions:")
    print("  " + ", ".join(f"p{i}={per_pos[i]:.6f}" for i in lows))

    chunk_boundaries = list(range(chunk_size, seq_len, chunk_size))
    if chunk_boundaries:
        print(f"[prefill debug] PCC around chunk boundaries (chunk_size={chunk_size}):")
        for boundary in chunk_boundaries:
            lo = max(0, boundary - 6)
            hi = min(seq_len - 1, boundary + 5)
            print(f"  boundary@{boundary} (positions {lo}–{hi}):")
            for p in range(lo, hi + 1):
                marker = ""
                if p == boundary:
                    marker = " <-- chunk start"
                elif p == boundary - 1:
                    marker = " <-- prev chunk last"
                print(f"    p{p:4d}  PCC={per_pos[p]:.6f}{marker}")
    else:
        print(f"[prefill debug] no chunk boundaries within seq_len={seq_len} (chunk_size={chunk_size})")

    worst_p = lows[0]
    ref_worst = ref_f[:, worst_p]
    tt_worst = tt_f[:, worst_p]
    abs_err = (ref_worst - tt_worst).abs()
    print(f"[prefill debug] lowest-PCC token p{worst_p} (PCC={per_pos[worst_p]:.6f}):")
    print(f"  HF shape={tuple(ref_worst.shape)}  TT shape={tuple(tt_worst.shape)}")
    print(f"  max_abs_error={abs_err.max().item():.6f}  mean_abs_error={abs_err.mean().item():.6f}")

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    print(f"[prefill debug] per-chunk overall PCC ({num_chunks} chunk(s), chunk_size={chunk_size}):")
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, seq_len)
        ref_chunk = ref_f[:, start:end]
        tt_chunk = tt_f[:, start:end]
        _, chunk_pcc = comp_pcc(ref_chunk, tt_chunk, pcc=0.0)
        chunk_per_pos = [comp_pcc(ref_f[:, p], tt_f[:, p], pcc=0.0)[1] for p in range(start, end)]
        print(
            f"  chunk {chunk_idx}  positions [{start}:{end})  "
            f"overall_PCC={chunk_pcc:.6f}  min={min(chunk_per_pos):.6f}  "
            f"median={sorted(chunk_per_pos)[len(chunk_per_pos) // 2]:.6f}  last={chunk_per_pos[-1]:.6f}"
        )
