# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``VoxtralTTAcousticModel`` component tests vs ``FlowMatchingAudioTransformerRef``.

Continuous tensors: PCC (``comp_pcc``).  Discrete forward codes: exact match fraction vs CPU.
FM noise is synchronized (``torch.randn`` uploaded to TT) whenever CPU and TT are compared.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.cpu_flow_matching_acoustic import (
    AudioSpecialTokens,
    FlowMatchingAudioTransformerRef,
    build_audio_model_args_from_voxtral_config,
)
from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.acoustic_model import VoxtralTTAcousticModel
from models.experimental.voxtraltts.tt.voxtral_tts import ACOUSTIC_CFG_ALPHA_DEFAULT
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict
from models.experimental.voxtraltts.utils.rng import acoustic_fm_noise_seed
from models.tt_transformers.tt.common import Mode

_E2E_DEMO_TEXT = (
    "Voxtral is a four billion parameter open weight text to speech model "
    "released by Mistral AI in two thousand twenty six, designed for low "
    "latency multilingual voice generation across English, Spanish, French, "
    "Portuguese, Hindi, German, Dutch, and Italian. It builds on the "
    "Ministral three billion language backbone with a flow matching acoustic "
    "decoder and produces audio at twelve point five hertz with high quality, "
    "suitable for streaming voice applications and real time agent deployments."
)
_E2E_DEMO_VOICE = "casual_male"

ACOUSTIC_VELOCITY_PCC = 0.99
ACOUSTIC_SEMANTIC_LOGITS_PCC = 0.99
ACOUSTIC_EULER_STATE_PCC = 0.99
# Discrete acoustic code match (cols 1–36). TT FM Euler can flip ``round()`` at boundaries (~32/36 today).
ACOUSTIC_FORWARD_ACOUSTIC_MATCH_FRAC = 0.88


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


def _forward_debug(msg: str) -> None:
    if os.environ.get("VOXTRAL_ACOUSTIC_FORWARD_E2E_DEBUG", "").lower() in ("1", "true", "yes", "on"):
        print(f"[test_acoustic_forward_matches_cpu_reference] {msg}")


def _decode_e2e_debug(msg: str) -> None:
    if os.environ.get("VOXTRAL_ACOUSTIC_DECODE_E2E_DEBUG", "").lower() in ("1", "true", "yes", "on"):
        print(f"[test_acoustic_decode] {msg}")


def _synced_fm_noise_cpu(
    bsz: int,
    n_acoustic: int,
    noise_seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    torch.manual_seed(noise_seed)
    return torch.randn(bsz, n_acoustic, device=device, dtype=dtype)


def _reference_forward_synced(
    ref: FlowMatchingAudioTransformerRef,
    llm_h: torch.Tensor,
    cfg_alpha: torch.Tensor,
    *,
    noise_seed: int,
) -> torch.Tensor:
    """CPU ``forward()`` with deterministic FM noise (``torch.manual_seed`` before ``decode_one_frame``)."""
    torch.manual_seed(noise_seed)
    return ref.forward(llm_h, cfg_alpha).long()


def _tt_forward_synced(
    tt_model: VoxtralTTAcousticModel,
    mesh_device: ttnn.MeshDevice,
    llm_h: torch.Tensor,
    cfg_alpha: torch.Tensor,
    *,
    noise_seed: int,
) -> torch.Tensor:
    """TT ``forward()`` with the same ``torch.randn`` noise tensor uploaded to device."""
    bsz = llm_h.shape[0]
    cfg_scalar = float(cfg_alpha.item())
    x_0 = _synced_fm_noise_cpu(
        bsz,
        tt_model.n_acoustic_out,
        noise_seed,
        device=llm_h.device,
        dtype=llm_h.dtype,
    )
    llm_tt = ttnn.from_torch(
        llm_h.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    noise_tt = ttnn.from_torch(
        x_0.unsqueeze(1).contiguous(),
        device=mesh_device,
        dtype=tt_model.dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    codes_tt = tt_model.forward(llm_tt, noise_tt, cfg_scalar)
    ttnn.deallocate(llm_tt)
    ttnn.deallocate(noise_tt)
    codes = ttnn.to_torch(codes_tt).long().reshape(bsz, -1)
    ttnn.deallocate(codes_tt)
    return codes


def _load_step0_hidden_bf16(model_name_or_path: str) -> torch.Tensor:
    cpu = VoxtralCPUReference(model_name_or_path=model_name_or_path, dtype="bfloat16", device="cpu")
    _, _, cpu_trace = cpu.generate(
        text=_E2E_DEMO_TEXT,
        voice=_E2E_DEMO_VOICE,
        max_tokens=8,
        seed=0,
        return_tokenizer_codes=True,
        return_debug=True,
    )
    hidden_t = cpu_trace.get("step.0.text.hidden_in")
    if hidden_t is None:
        raise RuntimeError("CPU generate debug trace missing step.0.text.hidden_in")
    return hidden_t.to(dtype=torch.bfloat16).reshape(1, -1)


def _assert_forward_matches_cpu(
    ref_out: torch.Tensor,
    tt_out: torch.Tensor,
    *,
    min_acoustic_match_frac: float,
    label: str,
) -> None:
    """Semantic exact; acoustic + full code match fractions vs CPU (discrete, not PCC)."""
    tt_cmp = tt_out.to(ref_out.device) if tt_out.device != ref_out.device else tt_out
    assert ref_out.shape == tt_cmp.shape, f"{label}: shape ref={tuple(ref_out.shape)} tt={tuple(tt_cmp.shape)}"
    assert torch.equal(
        ref_out[:, :1], tt_cmp[:, :1]
    ), f"{label}: semantic mismatch ref={ref_out[:, :1].tolist()} tt={tt_cmp[:, :1].tolist()}"

    n_acoustic = ref_out.shape[1] - 1
    if n_acoustic == 0:
        return

    n_match = int((ref_out[:, 1:] == tt_cmp[:, 1:]).sum().item())
    acoustic_match = n_match / n_acoustic
    full_match = float((ref_out == tt_cmp).float().mean().item())
    logger.info(
        f"  {label}: acoustic code match={acoustic_match:.4f} ({n_match}/{n_acoustic}) " f"full={full_match:.4f}"
    )
    _forward_debug(
        f"{label}: acoustic={acoustic_match:.4f} full={full_match:.4f} " f"ref={ref_out.tolist()} tt={tt_cmp.tolist()}"
    )
    assert acoustic_match >= min_acoustic_match_frac, (
        f"{label}: acoustic code match {n_match}/{n_acoustic} ({acoustic_match:.4f}) "
        f"below {min_acoustic_match_frac}"
    )


def _tt_upload_velocity_inputs(
    tt_model: VoxtralTTAcousticModel,
    mesh_device: ttnn.MeshDevice,
    x_t: torch.Tensor,
    llm_h: torch.Tensor,
    t_emb: torch.Tensor,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    mem = tt_model._matmul_act_mem_config
    tt_xt = ttnn.from_torch(
        x_t.unsqueeze(1).to(torch.bfloat16),
        device=mesh_device,
        dtype=tt_model.dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem,
    )
    tt_te = ttnn.from_torch(
        t_emb.unsqueeze(1).to(torch.bfloat16),
        device=mesh_device,
        dtype=tt_model.dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem,
    )
    tt_llm = ttnn.from_torch(
        llm_h.unsqueeze(1).to(torch.bfloat16),
        device=mesh_device,
        dtype=tt_model.dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem,
    )
    return tt_xt, tt_te, tt_llm


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

    timesteps = list(range(len(tt_model._euler_t_vals)))
    sampled_tt = ttnn.from_torch(
        x_0.to(torch.bfloat16).unsqueeze(1),
        device=tt_model.mesh_device,
        dtype=tt_model.dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sampled_tt = ttnn.typecast(sampled_tt, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
    for i in timesteps:
        t_val = tt_model._euler_t_vals[i]
        dt_val = tt_model._euler_dt_vals[i]

        te = tt_model._time_embedding_tt(t_val, bsz)
        x_in = tt_model._sampled_tt_for_velocity(sampled_tt)
        x_batched = ttnn.concat([x_in, x_in], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if x_in is not sampled_tt and x_in.is_allocated():
            ttnn.deallocate(x_in)
        te_batched = ttnn.concat([te, te], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(te)
        v_tt = tt_model.predict_velocity_tt(x_batched, te_batched, tt_llm_batched, borrow_llm=True)

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
        sampled_tt = tt_model._euler_integrate_sampled(sampled_tt, v_t_3d, dt_val)
        step_states.append(ttnn.to_torch(sampled_tt).float().reshape(bsz, -1))

    ttnn.deallocate(tt_llm_batched)

    scaled_tt = tt_model.fm_pre_round_scaled_tt(sampled_tt)
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
    x_t: torch.Tensor,
    llm_h: torch.Tensor,
    t_emb: torch.Tensor,
) -> str:
    """CPU reference intermediate keys for pytest failure hints."""
    _, ref_dbg = _reference_predict_velocity_debug(ref, x_t, llm_h, t_emb)
    lines = ["  CPU reference velocity stages (TT per-op debug not in model):"]
    for k in _ordered_debug_keys(list(ref_dbg.keys())):
        lines.append(f"  {k}")
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
    tt_xt, tt_te, tt_llm = _tt_upload_velocity_inputs(tt_model, mesh_device, x_t, llm_h, t_emb)
    tt_out = tt_model.predict_velocity_tt(tt_xt, tt_te, tt_llm)
    tt_velocity = ttnn.to_torch(tt_out).float().reshape(b, -1)
    ttnn.deallocate(tt_out)

    passing, pcc_val = comp_pcc(reference_velocity, tt_velocity, pcc=ACOUSTIC_VELOCITY_PCC)

    extra = ""
    if not passing or _acoustic_debug_pcc_enabled():
        table = _per_op_pcc_report(ref, x_t, llm_h, t_emb)
        extra = "\n\nPer-op PCC (forward order; first LOW = diverging stage):\n" + table
        if _acoustic_debug_pcc_enabled():
            print(extra)

    assert passing, (
        f"Acoustic predict_velocity PCC failed: PCC={float(pcc_val):.6f} "
        f"(required >= {ACOUSTIC_VELOCITY_PCC})." + extra
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@torch.no_grad()
def test_acoustic_semantic_logits_pcc(mesh_device, reset_seeds):
    """TT semantic linear: global + ref-top-k + watch {855,6114,6286} vs CPU reference."""
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
    ref_logits[:, ref._empty_audio_token_id] = float("-inf")
    tail = len(AudioSpecialTokens.all_special_tokens()) + ref.model_args.semantic_codebook_size
    ref_logits[:, tail:] = float("-inf")

    tt_llm = ttnn.from_torch(
        llm_h.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    sem_logits_tt = tt_model.semantic_logits_tt(tt_llm)
    ttnn.deallocate(tt_llm)
    tt_logits = ttnn.to_torch(sem_logits_tt).float()
    ttnn.deallocate(sem_logits_tt)
    if tt_logits.dim() == 4:
        tt_logits = tt_logits.squeeze(1)
    if tt_logits.dim() == 3:
        tt_logits = tt_logits.squeeze(1)
    tt_logits = tt_logits[..., : tt_model._sem_vocab_size]
    tt_logits = _align_to_ref_shape(ref_logits, tt_logits)

    passing, pcc_val = comp_pcc(ref_logits, tt_logits, pcc=ACOUSTIC_SEMANTIC_LOGITS_PCC)
    assert passing, (
        f"Semantic logits PCC failed: PCC={float(pcc_val):.6f} (required >= {ACOUSTIC_SEMANTIC_LOGITS_PCC}). "
        f"ref.shape={tuple(ref_logits.shape)} tt.shape={tuple(tt_logits.shape)}."
    )
    assert torch.equal(ref_logits.argmax(dim=-1), tt_logits.argmax(dim=-1)), (
        f"Semantic argmax mismatch: ref={ref_logits.argmax(dim=-1).tolist()} " f"tt={tt_logits.argmax(dim=-1).tolist()}"
    )

    watch = (855, 6114, 6286)
    watch_ok = [i for i in watch if i < ref_logits.shape[-1]]
    if watch_ok:
        ref_watch = ref_logits[:, watch_ok]
        tt_watch = tt_logits[:, watch_ok]
        watch_pass, watch_pcc = comp_pcc(ref_watch, tt_watch, pcc=0.999)
        watch_max_diff = float((tt_watch - ref_watch).abs().max().item())
        assert watch_pass, (
            f"Watch-index logits PCC failed at {watch_ok}: PCC={float(watch_pcc):.6f} "
            f"max|Δ|={watch_max_diff:.4f} ref={ref_watch.tolist()} tt={tt_watch.tolist()}"
        )

    top_k = min(10, ref_logits.shape[-1])
    ref_top_idx = torch.topk(ref_logits[0], k=top_k).indices
    ref_topk = ref_logits[:, ref_top_idx]
    tt_topk = tt_logits[:, ref_top_idx]
    topk_pass, topk_pcc = comp_pcc(ref_topk, tt_topk, pcc=0.999)
    assert topk_pass, f"Ref-top-{top_k} logits PCC failed: PCC={float(topk_pcc):.6f} " f"idx={ref_top_idx.tolist()}"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize(
    "hidden_case,cfg_value,noise_seed",
    [
        ("random", 0.73, 12345),
        ("step0", ACOUSTIC_CFG_ALPHA_DEFAULT, acoustic_fm_noise_seed(0, 0)),
    ],
    ids=["random_hidden", "step0_hidden"],
)
@torch.no_grad()
def test_acoustic_forward_matches_cpu_reference(mesh_device, reset_seeds, hidden_case, cfg_value, noise_seed):
    """Full ``forward()`` vs CPU with synced ``torch.randn`` FM noise.

    Semantic column: exact match.  Acoustic columns: match fraction (discrete codes, not PCC).
    FM pre-round scaled state: PCC >= ``ACOUSTIC_EULER_STATE_PCC``.
    """
    model_name_or_path = resolve_voxtral_model_name_or_skip()
    try:
        ref, cfg = _load_reference_model(model_name_or_path)
        tt_model = _load_tt(mesh_device, model_name_or_path)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Acoustic load failed: {exc}")

    d_llm = cfg.audio_model_args.acoustic_transformer_args.input_dim
    cfg_alpha = torch.tensor(cfg_value, dtype=torch.bfloat16)

    if hidden_case == "random":
        torch.manual_seed(42)
        llm_h = torch.randn(1, d_llm, dtype=torch.bfloat16)
    else:
        try:
            llm_h = _load_step0_hidden_bf16(model_name_or_path)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"Step-0 hidden load failed: {exc}")

    ref_out = _reference_forward_synced(ref, llm_h, cfg_alpha, noise_seed=noise_seed)
    tt_out = _tt_forward_synced(tt_model, mesh_device, llm_h, cfg_alpha, noise_seed=noise_seed)

    ref_scaled, _ = _reference_decode_one_frame_continuous(ref, llm_h, cfg_alpha, noise_seed=noise_seed)
    tt_scaled, _ = _tt_decode_one_frame_continuous(tt_model, llm_h, cfg_alpha, noise_seed=noise_seed)
    ok_pcc, pcc_val = comp_pcc(ref_scaled, tt_scaled, pcc=ACOUSTIC_EULER_STATE_PCC)
    logger.info(
        f"  {hidden_case}: FM pre-round scaled PCC={float(pcc_val):.6f} "
        f"max|Δ|={float((ref_scaled - tt_scaled).abs().max()):.6f}"
    )
    assert ok_pcc, f"{hidden_case}: FM pre-round scaled PCC {float(pcc_val):.6f} " f"below {ACOUSTIC_EULER_STATE_PCC}"

    _assert_forward_matches_cpu(
        ref_out,
        tt_out,
        min_acoustic_match_frac=ACOUSTIC_FORWARD_ACOUSTIC_MATCH_FRAC,
        label=hidden_case,
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
        ok_i, pcc_i = comp_pcc(ref_s, tt_s, pcc=ACOUSTIC_EULER_STATE_PCC)
        max_d = float((ref_s - tt_s).abs().max().item())
        lines.append(f"  euler step {i + 1}: PCC={float(pcc_i):.6f} max_diff={max_d:.6f}")
        _decode_e2e_debug(lines[-1])
        assert ok_i, (
            f"Euler step {i + 1} sampled-state PCC failed: PCC={float(pcc_i):.6f} "
            f"(required >= {ACOUSTIC_EULER_STATE_PCC}).\n" + "\n".join(lines)
        )

    ok_final, pcc_final = comp_pcc(ref_scaled, tt_scaled, pcc=ACOUSTIC_EULER_STATE_PCC)
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

    assert ok_final, (
        f"Pre-round scaled acoustic PCC failed: PCC={float(pcc_final):.6f} "
        f"(required >= {ACOUSTIC_EULER_STATE_PCC}).\n" + "\n".join(lines)
    )
    assert (
        code_match >= ACOUSTIC_FORWARD_ACOUSTIC_MATCH_FRAC
    ), f"Euler discrete code match {code_match:.4f} below {ACOUSTIC_FORWARD_ACOUSTIC_MATCH_FRAC}\n" + "\n".join(lines)


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
        out_tt = tt_model.attentions[layer_idx](attn_norm, None, None, attention_mask=None)
        ttnn.deallocate(attn_norm)
        tt_attn = ttnn.to_torch(out_tt).float().reshape(bsz, seq, dim)
        ttnn.deallocate(out_tt)

        passing_a, pcc_a = comp_pcc(ref_attn.float(), tt_attn, pcc=ACOUSTIC_VELOCITY_PCC)
        assert passing_a, (
            f"Layer {layer_idx} attention PCC failed: PCC={float(pcc_a):.6f} " f"(required >= {ACOUSTIC_VELOCITY_PCC})."
        )

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

        passing_m, pcc_m = comp_pcc(ref_mlp.float(), tt_mlp, pcc=ACOUSTIC_VELOCITY_PCC)
        assert passing_m, (
            f"Layer {layer_idx} MLP PCC failed: PCC={float(pcc_m):.6f} " f"(required >= {ACOUSTIC_VELOCITY_PCC})."
        )
