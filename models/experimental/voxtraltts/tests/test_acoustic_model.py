# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-backed tests for ``VoxtralTTAcousticModel`` / ``FlowMatchingAudioTransformerRef``.

**Velocity trunk**
    ``predict_velocity`` vs ``_predict_velocity`` (projections → blocks → ``acoustic_codebook_output``).
    On failure, the assertion includes a **per-op PCC table** (forward order). Set
    ``VOXTRAL_ACOUSTIC_DEBUG_PCC=1`` to print that table when the test passes.

**Semantic head**
    ``semantic_codebook_output`` vs ``ttnn.linear(..., w_semantic)``.

**End-to-end ``forward()``**
    Same RNG seed before reference vs TT decode; **semantic token** must match exactly; **acoustic**
    discrete codes use a **minimum agreement rate** (residual rounding vs CPU; TT acoustic stack uses
    HiFi4 + FP32 dest acc for matmuls / norms to tighten agreement).
**Layer components**
    Single-layer attention and SwiGLU MLP PCC vs CPU (layer index parametrized).
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.cpu_flow_matching_acoustic import (
    FlowMatchingAudioTransformerRef,
    build_audio_model_args_from_voxtral_config,
)
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.acoustic_model import VoxtralTTAcousticModel
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict
from models.tt_transformers.tt.common import Mode


def _load_reference_model(model_name_or_path: str):
    cfg = load_voxtral_config(model_name_or_path)
    args_d = build_audio_model_args_from_voxtral_config(cfg)
    ref = FlowMatchingAudioTransformerRef(args_d)
    full_sd = _load_safetensors_state_dict(model_name_or_path)
    for k, v in full_sd.items():
        if k.startswith("acoustic_transformer."):
            ref.load_weight((k[len("acoustic_transformer.") :], v))
    ref = ref.to(torch.bfloat16).eval()
    return ref, cfg


def _load_tt(mesh_device, model_name_or_path: str) -> VoxtralTTAcousticModel:
    return VoxtralTTAcousticModel.create_from_model_name(
        mesh_device,
        model_name_or_path=model_name_or_path,
        dtype=ttnn.bfloat16,
    )


def _align_to_ref_shape(ref_t: torch.Tensor, tt_t: torch.Tensor) -> torch.Tensor:
    """Trim TT tensor logical extents to reference (ttnn may pad)."""
    out = tt_t
    for dim, size in enumerate(ref_t.shape):
        if dim < out.dim() and out.shape[dim] > size:
            sl = [slice(None)] * out.dim()
            sl[dim] = slice(0, size)
            out = out[tuple(sl)]
    return out.reshape(ref_t.shape)


def _reference_predict_velocity_debug(
    ref: FlowMatchingAudioTransformerRef,
    x_t: torch.Tensor,
    llm_h: torch.Tensor,
    t_emb: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Same tensor stepping as ``FlowMatchingAudioTransformerRef._predict_velocity`` + intermediates."""
    debug: dict[str, torch.Tensor] = {}
    time_dtype = ref.time_projection.weight.dtype
    llm_dtype = ref.llm_projection.weight.dtype
    input_dtype = ref.input_projection.weight.dtype

    t_emb_p = ref.time_projection(t_emb.to(dtype=time_dtype)).to(dtype=input_dtype)
    llm_p = ref.llm_projection(llm_h.to(dtype=llm_dtype)).to(dtype=input_dtype)
    x_t_p = x_t.to(dtype=input_dtype)
    p0 = ref.input_projection(x_t_p.unsqueeze(1))
    debug["proj_input"] = p0.float()
    debug["proj_time"] = t_emb_p.unsqueeze(1).float()
    debug["proj_llm"] = llm_p.unsqueeze(1).float()

    h = torch.cat([p0, t_emb_p.unsqueeze(1), llm_p.unsqueeze(1)], dim=1)
    debug["concat_input"] = h.unsqueeze(1).float()

    for i in ref.layers_ids:
        layer = ref.layers[str(i)]
        n1 = layer.attention_norm(h)
        debug[f"layer{i}.attn_norm"] = n1.unsqueeze(1).float()
        a = layer.attention(n1)
        debug[f"layer{i}.attn_out"] = a.unsqueeze(1).float()
        h = h + a
        debug[f"layer{i}.post_attn"] = h.unsqueeze(1).float()
        n2 = layer.ffn_norm(h)
        debug[f"layer{i}.ffn_norm"] = n2.unsqueeze(1).float()
        f = layer.feed_forward(n2)
        debug[f"layer{i}.ffn_out"] = f.unsqueeze(1).float()
        h = h + f
        debug[f"layer{i}.post_ffn"] = h.unsqueeze(1).float()

    h = ref.norm(h)
    debug["final_norm"] = h.unsqueeze(1).float()
    v = ref.acoustic_codebook_output(h[:, 0, :])
    debug["velocity"] = v.unsqueeze(1).unsqueeze(1).float()
    return v.float(), debug


def _acoustic_debug_pcc_enabled() -> bool:
    return os.environ.get("VOXTRAL_ACOUSTIC_DEBUG_PCC", "").lower() in ("1", "true", "yes", "on")


def _forward_e2e_debug(msg: str) -> None:
    if os.environ.get("VOXTRAL_ACOUSTIC_FORWARD_E2E_DEBUG", "").lower() in ("1", "true", "yes", "on"):
        print(f"[test_acoustic_forward_e2e_matches_reference] {msg}")


def _ordered_debug_keys(keys: list[str]) -> list[str]:
    """Pipeline order for PCC rows so the first ``LOW`` matches the first diverging stage."""
    keys_set = set(keys)
    layer_ids: list[int] = []
    for k in keys_set:
        if k.startswith("layer") and k.endswith(".attn_norm"):
            try:
                layer_ids.append(int(k.split(".")[0][len("layer") :]))
            except ValueError:
                pass
    layer_ids = sorted(set(layer_ids))
    stage_suffixes = ("attn_norm", "attn_out", "post_attn", "ffn_norm", "ffn_out", "post_ffn")
    ordered: list[str] = []
    for head in ("proj_input", "proj_time", "proj_llm", "concat_input"):
        if head in keys_set:
            ordered.append(head)
    for i in layer_ids:
        for suf in stage_suffixes:
            k = f"layer{i}.{suf}"
            if k in keys_set:
                ordered.append(k)
    for tail in ("final_norm", "velocity"):
        if tail in keys_set:
            ordered.append(tail)
    rest = sorted(keys_set.difference(ordered))
    return ordered + rest


def _per_op_pcc_report(
    ref: FlowMatchingAudioTransformerRef,
    tt_model: VoxtralTTAcousticModel,
    x_t: torch.Tensor,
    llm_h: torch.Tensor,
    t_emb: torch.Tensor,
    *,
    b: int,
) -> str:
    """Second TT/CPU forward with intermediates; returns multiline string for pytest failure message."""
    _, ref_dbg = _reference_predict_velocity_debug(ref, x_t, llm_h, t_emb)
    tt_vel_tt, tt_dbg = tt_model.predict_velocity_debug(x_t, llm_h, t_emb)
    ttnn.deallocate(tt_vel_tt)
    lines = []
    for k in _ordered_debug_keys(list(ref_dbg.keys())):
        tt_k = _align_to_ref_shape(ref_dbg[k], tt_dbg[k])
        passing_k, pcc_k = comp_pcc(ref_dbg[k], tt_k, pcc=0.99)
        status = "OK" if passing_k else "LOW"
        lines.append(f"  {k:24s} PCC={float(pcc_k):.6f}  {status}")
    return "\n".join(lines)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@torch.no_grad()
def test_acoustic_predict_velocity_pcc(mesh_device, reset_seeds):
    """TT FM velocity matches CPU reference (Pearson correlation >= 0.99)."""
    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        ref, cfg = _load_reference_model(model_name_or_path)
        tt_model = _load_tt(mesh_device, model_name_or_path)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Acoustic load failed: {exc}")

    torch.manual_seed(0)
    b = 1
    n_acoustic = cfg.audio_model_args.n_acoustic_codebook
    d_llm = cfg.audio_model_args.acoustic_transformer_args.input_dim

    x_t = torch.randn(b, n_acoustic, dtype=torch.bfloat16)
    llm_h = torch.randn(b, d_llm, dtype=torch.bfloat16)
    t_scalar = torch.tensor([[0.25]], dtype=torch.bfloat16).expand(b, 1)
    t_emb = ref.time_embedding(t_scalar).to(torch.bfloat16)

    reference_velocity = ref._predict_velocity(x_t, llm_h, t_emb).float()
    tt_out = tt_model.predict_velocity(x_t, llm_h, t_emb)
    tt_velocity = ttnn.to_torch(tt_out).float().reshape(b, -1)
    ttnn.deallocate(tt_out)

    passing, pcc_val = comp_pcc(reference_velocity, tt_velocity, pcc=0.99)

    extra = ""
    if not passing or _acoustic_debug_pcc_enabled():
        table = _per_op_pcc_report(ref, tt_model, x_t, llm_h, t_emb, b=b)
        extra = (
            "\n\nPer-op PCC vs CPU reference (rows in **forward order**; first LOW pinpoints the stage):\n"
            + table
            + "\n\nTip: full message appears above without pytest -s when the assertion fails."
        )
        if _acoustic_debug_pcc_enabled():
            print(extra)

    assert passing, f"Acoustic predict_velocity PCC failed: PCC={float(pcc_val):.6f} (required >= 0.99)." + extra


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@torch.no_grad()
def test_acoustic_semantic_logits_pcc(mesh_device, reset_seeds):
    """``semantic_codebook_output`` matches TT linear (Pearson >= 0.99)."""
    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        ref, cfg = _load_reference_model(model_name_or_path)
        tt_model = _load_tt(mesh_device, model_name_or_path)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Acoustic load failed: {exc}")

    torch.manual_seed(0)
    bsz = 1
    d_llm = cfg.audio_model_args.acoustic_transformer_args.input_dim
    llm_h = torch.randn(bsz, d_llm, dtype=torch.bfloat16)

    w_dtype = ref.semantic_codebook_output.weight.dtype
    ref_logits = ref.semantic_codebook_output(llm_h.to(dtype=w_dtype)).float()

    tt_llm = ttnn.from_torch(
        llm_h.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sem_tt = ttnn.linear(tt_llm, tt_model.w_semantic, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tt_llm)
    tt_logits = ttnn.to_torch(sem_tt).float().reshape(bsz, -1)
    ttnn.deallocate(sem_tt)

    tt_aligned = _align_to_ref_shape(ref_logits, tt_logits)
    passing, pcc_val = comp_pcc(ref_logits, tt_aligned, pcc=0.99)
    assert passing, (
        f"Semantic logits PCC failed: PCC={float(pcc_val):.6f} (required >= 0.99). "
        f"ref.shape={tuple(ref_logits.shape)} tt.shape={tuple(tt_logits.shape)}."
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@torch.no_grad()
def test_acoustic_forward_e2e_matches_reference(mesh_device, reset_seeds):
    """Full ``forward()`` integration: RNG synced; semantic token exact; acoustic codes mostly match.

    TT runs FM decode on device in BF16; CPU reference uses PyTorch matmul—small trajectory drift can
    flip ``round()`` on a few acoustic bins. Semantic argmax uses the same masked logits shape as ref.
    """
    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        ref, cfg = _load_reference_model(model_name_or_path)
        tt_model = _load_tt(mesh_device, model_name_or_path)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Acoustic load failed: {exc}")

    torch.manual_seed(42)
    bsz = 1
    d_llm = cfg.audio_model_args.acoustic_transformer_args.input_dim
    llm_h = torch.randn(bsz, d_llm, dtype=torch.bfloat16)
    cfg_alpha = torch.tensor(0.73, dtype=torch.bfloat16)

    torch.manual_seed(12345)
    ref_out = ref.forward(llm_h, cfg_alpha)

    torch.manual_seed(12345)
    tt_out = tt_model.forward(llm_h, cfg_alpha)

    _forward_e2e_debug(f"ref_out shape={tuple(ref_out.shape)} dtype={ref_out.dtype} device={ref_out.device}")
    _forward_e2e_debug(f"tt_out  shape={tuple(tt_out.shape)} dtype={tt_out.dtype} device={tt_out.device}")

    assert (
        ref_out.shape == tt_out.shape
    ), f"[shape] forward mismatch: ref={tuple(ref_out.shape)} tt={tuple(tt_out.shape)}"
    assert ref_out.dtype == tt_out.dtype, f"[dtype] forward mismatch: ref={ref_out.dtype} tt={tt_out.dtype}"
    tt_cmp = tt_out.to(ref_out.device) if tt_out.device != ref_out.device else tt_out

    sem_ok = torch.equal(ref_out[:, :1], tt_cmp[:, :1])
    _forward_e2e_debug(f"semantic ref={ref_out[:, :1].tolist()} tt={tt_cmp[:, :1].tolist()} equal={sem_ok}")
    assert sem_ok, "[semantic] forward() semantic token differs from reference (TT semantic linear / mask / argmax)."

    n_acoustic = ref_out.shape[1] - 1
    if n_acoustic > 0:
        acoustic_ok = ref_out[:, 1:] == tt_cmp[:, 1:]
        match_frac = float(acoustic_ok.float().mean().item())
        n_bad = int((~acoustic_ok).sum().item())
        _forward_e2e_debug(f"acoustic match_frac={match_frac:.4f} mismatches={n_bad}/{acoustic_ok.numel()}")

        min_frac = 0.94
        if match_frac < min_frac:
            bad_idx = torch.nonzero(ref_out[:, 1:] != tt_cmp[:, 1:], as_tuple=False)
            pytest.fail(
                "[acoustic] "
                f"forward() acoustic codes agree in only {match_frac:.4f} of positions "
                f"(required >= {min_frac}); BF16 FM vs CPU drift at quantization. "
                f"First differing flat indices (batch,col): {bad_idx[:16].tolist()}"
                + (" ..." if bad_idx.shape[0] > 16 else "")
            )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("layer_idx", [0])
@torch.no_grad()
def test_acoustic_layer_attention_pcc(mesh_device, reset_seeds, layer_idx):
    """Single-layer bidirectional attention vs CPU ``BidirectionalAttention`` (PCC >= 0.99)."""
    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        ref, cfg = _load_reference_model(model_name_or_path)
        tt_model = _load_tt(mesh_device, model_name_or_path)
    except Exception as exc:
        pytest.skip(f"Acoustic load failed: {exc}")

    dim = cfg.audio_model_args.acoustic_transformer_args.dim
    torch.manual_seed(1)
    bsz, seq = 1, 3
    x = torch.randn(bsz, seq, dim, dtype=torch.bfloat16)

    layer = ref.layers[str(layer_idx)]
    n = layer.attention_norm(x)
    ref_out = layer.attention(n)

    x_tt = ttnn.from_torch(
        x.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attn_norm = tt_model.attn_norms[layer_idx](x_tt, mode=Mode.DECODE)
    cos = tt_model._cos_identity
    sin = tt_model._sin_identity
    out_tt = tt_model.attentions[layer_idx](attn_norm, cos, sin, attention_mask=None)
    ttnn.deallocate(attn_norm)
    tt_torch = ttnn.to_torch(out_tt).float().reshape(bsz, seq, dim)
    ttnn.deallocate(out_tt)

    passing, pcc_val = comp_pcc(ref_out.float(), tt_torch, pcc=0.99)
    assert passing, f"Layer {layer_idx} attention PCC failed: PCC={float(pcc_val):.6f} (required >= 0.99)."


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("layer_idx", [0])
@torch.no_grad()
def test_acoustic_layer_mlp_pcc(mesh_device, reset_seeds, layer_idx):
    """Single-layer SwiGLU FFN vs CPU ``FeedForward`` after ``ffn_norm`` (PCC >= 0.99)."""
    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        ref, cfg = _load_reference_model(model_name_or_path)
        tt_model = _load_tt(mesh_device, model_name_or_path)
    except Exception as exc:
        pytest.skip(f"Acoustic load failed: {exc}")

    dim = cfg.audio_model_args.acoustic_transformer_args.dim
    torch.manual_seed(2)
    bsz, seq = 1, 3
    x = torch.randn(bsz, seq, dim, dtype=torch.bfloat16)

    layer = ref.layers[str(layer_idx)]
    n = layer.ffn_norm(x)
    ref_out = layer.feed_forward(n)

    x_tt = ttnn.from_torch(
        x.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    normed_tt = tt_model.ffn_norms[layer_idx](x_tt, mode=Mode.DECODE)
    out_tt = tt_model.mlps[layer_idx](normed_tt)
    ttnn.deallocate(normed_tt)
    tt_torch = ttnn.to_torch(out_tt).float().reshape(bsz, seq, dim)
    ttnn.deallocate(out_tt)

    passing, pcc_val = comp_pcc(ref_out.float(), tt_torch, pcc=0.99)
    assert passing, f"Layer {layer_idx} MLP PCC failed: PCC={float(pcc_val):.6f} (required >= 0.99)."
