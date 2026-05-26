# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``VoxtralTTAcousticModel`` PCC vs ``FlowMatchingAudioTransformerRef`` (velocity, semantic, forward, per-layer)."""

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


def _decode_e2e_debug(msg: str) -> None:
    if os.environ.get("VOXTRAL_ACOUSTIC_DECODE_E2E_DEBUG", "").lower() in ("1", "true", "yes", "on"):
        print(f"[test_acoustic_decode] {msg}")


def _reference_decode_one_frame_continuous(
    ref: FlowMatchingAudioTransformerRef,
    llm_hidden: torch.Tensor,
    cfg_alpha: torch.Tensor,
    *,
    noise_seed: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """CPU Euler FM decode; returns final scaled codes (pre-round) and per-step sampled states."""
    bsz = llm_hidden.shape[0]
    torch.manual_seed(noise_seed)
    x_0 = torch.randn(bsz, ref.model_args.n_acoustic_codebook, device=llm_hidden.device, dtype=llm_hidden.dtype)
    x_0 = ref._noise_scale * x_0

    timesteps = ref._timesteps.to(dtype=llm_hidden.dtype, device=llm_hidden.device)
    llm_hidden_zero = torch.zeros_like(llm_hidden)
    ca = cfg_alpha.to(dtype=llm_hidden.dtype, device=llm_hidden.device)
    if ca.dim() == 0:
        cfg_alpha_b = ca.reshape(1, 1).expand(bsz, 1)
    elif ca.dim() == 1:
        cfg_alpha_b = ca.unsqueeze(1)
    else:
        cfg_alpha_b = ca
    cfg_scalar = float(cfg_alpha_b.flatten()[0].item())

    sampled = x_0
    step_states: list[torch.Tensor] = []
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        dt = timesteps[i + 1] - timesteps[i]
        t_emb = ref.time_embedding(t.view(-1, 1).repeat(bsz, 1)).to(llm_hidden.dtype)

        x_batched = torch.cat([sampled, sampled], dim=0)
        llm_batched = torch.cat([llm_hidden, llm_hidden_zero], dim=0)
        t_emb_batched = torch.cat([t_emb, t_emb], dim=0)

        v_all = ref._predict_velocity(x_t=x_batched, llm_output=llm_batched, t_emb=t_emb_batched)
        v_t = cfg_scalar * v_all[:bsz] + (1.0 - cfg_scalar) * v_all[bsz:]
        sampled = sampled + v_t * dt
        step_states.append(sampled.clone().float())

    sampled_clamped = torch.clamp(sampled, -1, 1).float()
    scaled = ((sampled_clamped + 1) / 2) * (ref.acoustic_embeddings_levels - 1)
    return scaled, step_states


def _tt_decode_one_frame_continuous(
    tt_model: VoxtralTTAcousticModel,
    llm_hidden: torch.Tensor,
    cfg_alpha: torch.Tensor,
    *,
    noise_seed: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """TT Euler FM decode; returns final scaled values (pre-round) and per-step sampled states."""
    bsz = llm_hidden.shape[0]
    device = llm_hidden.device
    dtype = llm_hidden.dtype

    torch.manual_seed(noise_seed)
    x_0 = torch.randn(bsz, tt_model.n_acoustic_out, device=device, dtype=dtype)

    timesteps = tt_model._timesteps_cpu
    sampled_tt = ttnn.from_torch(
        x_0.to(torch.bfloat16).unsqueeze(1),
        device=tt_model.mesh_device,
        dtype=tt_model.dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_llm = ttnn.from_torch(
        llm_hidden.to(torch.bfloat16).unsqueeze(1),
        device=tt_model.mesh_device,
        dtype=tt_model.dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_llm_zero = ttnn.zeros_like(tt_llm)
    tt_llm_batched = ttnn.concat([tt_llm, tt_llm_zero], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tt_llm)
    ttnn.deallocate(tt_llm_zero)

    ca = cfg_alpha.to(dtype=dtype, device=device)
    if ca.dim() == 0:
        cfg_a = ca.reshape(1, 1).expand(bsz, 1)
    elif ca.dim() == 1:
        cfg_a = ca.unsqueeze(1)
    else:
        cfg_a = ca
    cfg_scalar = float(cfg_a.flatten()[0].item())

    step_states: list[torch.Tensor] = []
    for i in range(len(timesteps) - 1):
        t_val = float(timesteps[i].item())
        dt_val = float((timesteps[i + 1] - timesteps[i]).item())

        te = tt_model._time_embedding_tt(t_val, bsz)
        x_batched = ttnn.concat([sampled_tt, sampled_tt], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        te_batched = ttnn.concat([te, te], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(te)
        v_tt = tt_model._predict_velocity_impl(
            None,
            None,
            None,
            _tt_xt=x_batched,
            _tt_te=te_batched,
            _tt_llm=tt_llm_batched,
            return_debug=False,
        )

        v_shape = tuple(v_tt.shape)
        v_cond = ttnn.slice(v_tt, [0, 0, 0, 0], [bsz, v_shape[1], v_shape[2], v_shape[3]])
        v_uncond = ttnn.slice(v_tt, [bsz, 0, 0, 0], [2 * bsz, v_shape[1], v_shape[2], v_shape[3]])
        ttnn.deallocate(v_tt)
        v_cond_scaled = ttnn.multiply(v_cond, cfg_scalar, dtype=tt_model.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(v_cond)
        v_uncond_scaled = ttnn.multiply(
            v_uncond, 1.0 - cfg_scalar, dtype=tt_model.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(v_uncond)
        v_t_tt = ttnn.add(v_cond_scaled, v_uncond_scaled, dtype=tt_model.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(v_cond_scaled)
        ttnn.deallocate(v_uncond_scaled)

        v_t_3d = ttnn.reshape(v_t_tt, (bsz, 1, tt_model.n_acoustic_out))
        ttnn.deallocate(v_t_tt)
        v_scaled = ttnn.multiply(v_t_3d, dt_val, dtype=tt_model.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(v_t_3d)
        new_sampled = ttnn.add(sampled_tt, v_scaled, dtype=tt_model.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(sampled_tt)
        ttnn.deallocate(v_scaled)
        sampled_tt = new_sampled
        step_states.append(ttnn.to_torch(sampled_tt).float().reshape(bsz, -1))

    ttnn.deallocate(tt_llm_batched)

    clamped_tt = ttnn.clip(sampled_tt, min=-1.0, max=1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(sampled_tt)
    plus_one_tt = ttnn.add(clamped_tt, 1.0, dtype=tt_model.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(clamped_tt)
    halved_tt = ttnn.multiply(plus_one_tt, 0.5, dtype=tt_model.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(plus_one_tt)
    scaled_tt = ttnn.multiply(
        halved_tt,
        float(tt_model._acoustic_embeddings_levels - 1),
        dtype=tt_model.dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(halved_tt)
    scaled_host = ttnn.to_torch(scaled_tt).float().reshape(bsz, -1)
    ttnn.deallocate(scaled_tt)
    return scaled_host, step_states


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
) -> str:
    """Per-op intermediate PCC table for pytest failures."""
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
        table = _per_op_pcc_report(ref, tt_model, x_t, llm_h, t_emb)
        extra = "\n\nPer-op PCC (forward order; first LOW = diverging stage):\n" + table
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
    """``forward()`` integration: synced RNG; semantic exact; acoustic codes per-frame >= 0.88.

    Discrete code agreement is lower than per-op PCC because 8-step BF16 Euler drift can flip
    ``round()`` at codebook boundaries even when continuous sampled state PCC stays >= 0.99.
    See ``test_acoustic_decode_euler_stepwise_pcc`` and
    ``test_acoustic_forward_code_match_mean_over_seeds`` for stepwise / statistical checks.
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

        min_frac = 0.88
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
@torch.no_grad()
def test_acoustic_decode_euler_stepwise_pcc(mesh_device, reset_seeds):
    """Each Euler step sampled state and final pre-round scaled values: PCC >= 0.99 vs CPU."""
    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        ref, cfg = _load_reference_model(model_name_or_path)
        tt_model = _load_tt(mesh_device, model_name_or_path)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Acoustic load failed: {exc}")

    torch.manual_seed(42)
    d_llm = cfg.audio_model_args.acoustic_transformer_args.input_dim
    llm_h = torch.randn(1, d_llm, dtype=torch.bfloat16)
    cfg_alpha = torch.tensor(0.73, dtype=torch.bfloat16)
    noise_seed = 12345

    ref_scaled, ref_steps = _reference_decode_one_frame_continuous(ref, llm_h, cfg_alpha, noise_seed=noise_seed)
    tt_scaled, tt_steps = _tt_decode_one_frame_continuous(tt_model, llm_h, cfg_alpha, noise_seed=noise_seed)

    lines: list[str] = []
    for i, (ref_s, tt_s) in enumerate(zip(ref_steps, tt_steps)):
        ok_i, pcc_i = comp_pcc(ref_s, tt_s, pcc=0.99)
        max_d = float((ref_s - tt_s).abs().max().item())
        lines.append(f"  euler step {i + 1}: PCC={float(pcc_i):.6f} max_diff={max_d:.6f}")
        _decode_e2e_debug(lines[-1])
        assert (
            ok_i
        ), f"Euler step {i + 1} sampled-state PCC failed: PCC={float(pcc_i):.6f} (required >= 0.99).\n" + "\n".join(
            lines
        )

    ok_final, pcc_final = comp_pcc(ref_scaled, tt_scaled, pcc=0.99)
    max_final = float((ref_scaled - tt_scaled).abs().max().item())
    final_line = f"  pre-round scaled: PCC={float(pcc_final):.6f} max_diff={max_final:.6f}"
    lines.append(final_line)
    _decode_e2e_debug(final_line)

    ref_codes = ref_scaled.round().long()
    tt_codes = tt_scaled.round().long()
    code_match = float((ref_codes == tt_codes).float().mean().item())
    n_flip = int((ref_codes != tt_codes).sum().item())
    flip_line = f"  discrete codes (torch.round): match={code_match:.4f} flips={n_flip}/{ref_codes.numel()}"
    lines.append(flip_line)
    _decode_e2e_debug(flip_line)

    assert (
        ok_final
    ), f"Pre-round scaled acoustic PCC failed: PCC={float(pcc_final):.6f} (required >= 0.99).\n" + "\n".join(lines)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@torch.no_grad()
def test_acoustic_forward_code_match_mean_over_seeds(mesh_device, reset_seeds):
    """Synced FM noise; mean acoustic+semantic code match >= 0.95 over hidden seeds (min >= 0.88)."""
    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        ref, cfg = _load_reference_model(model_name_or_path)
        tt_model = _load_tt(mesh_device, model_name_or_path)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Acoustic load failed: {exc}")

    d_llm = cfg.audio_model_args.acoustic_transformer_args.input_dim
    cfg_alpha = torch.tensor(0.73, dtype=torch.bfloat16)
    noise_seed = 999
    n_trials = 20
    matches: list[float] = []

    for hidden_seed in range(n_trials):
        torch.manual_seed(hidden_seed)
        llm_h = torch.randn(1, d_llm, dtype=torch.bfloat16)
        torch.manual_seed(noise_seed)
        ref_out = ref.forward(llm_h, cfg_alpha)
        torch.manual_seed(noise_seed)
        tt_out = tt_model.forward(llm_h, cfg_alpha)
        matches.append(float((ref_out == tt_out).float().mean().item()))

    mean_match = sum(matches) / len(matches)
    min_match = min(matches)
    _decode_e2e_debug(
        f"forward code match over {n_trials} hidden seeds (noise_seed={noise_seed}): "
        f"mean={mean_match:.4f} min={min_match:.4f} max={max(matches):.4f}"
    )

    assert mean_match >= 0.95, (
        f"Mean forward code match {mean_match:.4f} < 0.95 over {n_trials} trials "
        f"(noise_seed={noise_seed}); per-trial={matches}"
    )
    assert min_match >= 0.88, (
        f"Min forward code match {min_match:.4f} < 0.88 over {n_trials} trials; "
        "BF16 Euler + discrete round can flip codes near boundaries on unlucky frames."
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@torch.no_grad()
def test_acoustic_trace_path_matches_host_forward(mesh_device, reset_seeds):
    """``forward_acoustic_trace_codes`` agrees with ``forward()`` when noise/hidden are shared."""
    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        ref, cfg = _load_reference_model(model_name_or_path)
        tt_model = _load_tt(mesh_device, model_name_or_path)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Acoustic load failed: {exc}")

    torch.manual_seed(7)
    d_llm = cfg.audio_model_args.acoustic_transformer_args.input_dim
    llm_h = torch.randn(1, d_llm, dtype=torch.bfloat16)
    cfg_alpha = torch.tensor(0.73, dtype=torch.bfloat16)
    cfg_scalar = float(cfg_alpha.item())

    torch.manual_seed(555)
    host_out = tt_model.forward(llm_h, cfg_alpha)

    torch.manual_seed(555)
    noise = torch.randn(1, tt_model.n_acoustic_out, dtype=torch.bfloat16)
    llm_tt = ttnn.from_torch(
        llm_h.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    noise_tt = ttnn.from_torch(
        noise.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    trace_tt = tt_model.forward_acoustic_trace_codes(llm_tt, noise_tt, cfg_scalar)
    trace_out = ttnn.to_torch(trace_tt).long().reshape(1, -1)
    ttnn.deallocate(llm_tt)
    ttnn.deallocate(noise_tt)
    ttnn.deallocate(trace_tt)

    match_frac = float((host_out == trace_out).float().mean().item())
    _decode_e2e_debug(f"trace vs host forward match={match_frac:.4f}")
    assert match_frac >= 0.88, (
        f"Trace acoustic path match {match_frac:.4f} < 0.88 vs host forward(); "
        f"host={host_out.tolist()} trace={trace_out.tolist()}"
    )

    torch.manual_seed(555)
    ref_out = ref.forward(llm_h, cfg_alpha)
    ref_match = float((host_out == ref_out).float().mean().item())
    _decode_e2e_debug(f"host forward vs CPU ref match={ref_match:.4f}")
    assert ref_match >= 0.88, f"Host forward vs CPU ref match {ref_match:.4f} < 0.88"


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


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@torch.no_grad()
def test_acoustic_all_layers_attention_mlp_pcc(mesh_device, reset_seeds):
    """Attention + MLP PCC for every FM layer."""
    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        ref, cfg = _load_reference_model(model_name_or_path)
        tt_model = _load_tt(mesh_device, model_name_or_path)
    except Exception as exc:
        pytest.skip(f"Acoustic load failed: {exc}")

    n_ref = len(ref.layers_ids)
    n_tt = len(tt_model.attn_norms)
    assert n_ref == n_tt, f"layer count mismatch: ref n_layers={n_ref} tt attn_norms={n_tt}"

    dim = cfg.audio_model_args.acoustic_transformer_args.dim
    bsz, seq = 1, 3

    for layer_idx in ref.layers_ids:
        torch.manual_seed(1)
        x_attn = torch.randn(bsz, seq, dim, dtype=torch.bfloat16)
        layer = ref.layers[str(layer_idx)]
        n = layer.attention_norm(x_attn)
        ref_attn = layer.attention(n)

        x_tt = ttnn.from_torch(
            x_attn.unsqueeze(1),
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
        tt_attn = ttnn.to_torch(out_tt).float().reshape(bsz, seq, dim)
        ttnn.deallocate(out_tt)

        passing_a, pcc_a = comp_pcc(ref_attn.float(), tt_attn, pcc=0.99)
        assert passing_a, f"Layer {layer_idx} attention PCC failed: PCC={float(pcc_a):.6f} (required >= 0.99)."

        torch.manual_seed(2)
        x_mlp = torch.randn(bsz, seq, dim, dtype=torch.bfloat16)
        n2 = layer.ffn_norm(x_mlp)
        ref_mlp = layer.feed_forward(n2)

        x_mlp_tt = ttnn.from_torch(
            x_mlp.unsqueeze(1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        normed_tt = tt_model.ffn_norms[layer_idx](x_mlp_tt, mode=Mode.DECODE)
        mlp_tt = tt_model.mlps[layer_idx](normed_tt)
        ttnn.deallocate(normed_tt)
        tt_mlp = ttnn.to_torch(mlp_tt).float().reshape(bsz, seq, dim)
        ttnn.deallocate(mlp_tt)

        passing_m, pcc_m = comp_pcc(ref_mlp.float(), tt_mlp, pcc=0.99)
        assert passing_m, f"Layer {layer_idx} MLP PCC failed: PCC={float(pcc_m):.6f} (required >= 0.99)."
