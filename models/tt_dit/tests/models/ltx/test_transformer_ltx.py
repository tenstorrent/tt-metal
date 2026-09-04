# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 transformer unit tests; structure mirrors test_transformer_wan.py.

Torch reference is diffusers' LTX2VideoTransformer{Block,3DModel}. The TT runtime fuses RoPE
into rotary_embedding_llama (INTERLEAVED + trans_mat) with a load-time Q/K permute, so every
TT freq feed is INTERLEAVED. Modality axis selects video vs av; set LTX_SKIP_PCC=1 to assert
shape/finiteness only.
"""

import os
import time

import pytest
import torch
from loguru import logger
from safetensors.torch import load_file
from tracy import signpost

import ttnn
from models.tt_dit.models.transformers.ltx import attention_ltx, transformer_ltx
from models.tt_dit.models.transformers.ltx.rope_ltx import LTXRopeType, precompute_freqs_cis
from models.tt_dit.models.transformers.ltx.transformer_ltx import (
    LTXTransformerBlock,
    LTXTransformerModel,
    build_audio_masks,
    build_video_pad_mask,
)
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.patchifiers import AudioLatentShape, VideoPixelShape
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard
from models.tt_dit.utils.test import line_params, ring_params

# ---------------------------------------------------------------------------
# LTX-2.3-22B distilled transformer configuration
# ---------------------------------------------------------------------------
DIM = 4096
NUM_HEADS = 32
HEAD_DIM = DIM // NUM_HEADS  # 128
IN_CHANNELS = 128
OUT_CHANNELS = 128
CTX_DIM = 4096
AUDIO_DIM = 2048
AUDIO_HEAD_DIM = AUDIO_DIM // NUM_HEADS  # 64
AUDIO_IN_CHANNELS = 128
AUDIO_CTX_DIM = 2048
EPS = 1e-6
PROMPT_LEN = 32

# RoPE
ROPE_THETA = 10000.0
VIDEO_ROPE_MAX_POS = [20, 2048, 2048]
AUDIO_ROPE_MAX_POS = [20]
CROSS_PE_MAX_POS = [20]

# AV-mode auxiliary
AUDIO_N = 256  # sp-aligned for sp ∈ {1, 2, 4, 8}
# 22B checkpoint variants (AV mode only): "fast" = distilled-1.1 (production), "dev" = base 22B.
_CHECKPOINT_22B_VARIANTS = {
    "fast": os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors"),
    "dev": os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors"),
}


# Fallback: the same checkpoints as cached by huggingface_hub (Lightricks/LTX-2.3 repo).
_CHECKPOINT_22B_HF_FILENAMES = {
    "fast": "ltx-2.3-22b-distilled-1.1.safetensors",
    "dev": "ltx-2.3-22b-dev.safetensors",
}


def _resolve_checkpoint_22b(variant: str) -> str:
    """Path for the 22B variant: LTX_CHECKPOINT env, then local cache, then huggingface_hub snapshot."""
    env = os.environ.get("LTX_CHECKPOINT")
    if env:
        return env
    local = _CHECKPOINT_22B_VARIANTS[variant]
    if os.path.exists(local):
        return local
    snap_root = os.path.expanduser("~/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots")
    if os.path.isdir(snap_root):
        for snap in sorted(os.listdir(snap_root)):
            cand = os.path.join(snap_root, snap, _CHECKPOINT_22B_HF_FILENAMES[variant])
            if os.path.exists(cand):
                return cand
    return local


# Block-test reference uses small scaled random weights; default init blows up the adaln chains.
WEIGHT_SEED = 1234
INPUT_SEED = 42
TIMESTEP_VAL = 0.01

# Toggle PCC verification via env (mirrors Wan's `dit_unit_test`). Default ON.
_RUN_PCC_DEFAULT = {"1": False, "0": True}.get(os.environ.get("LTX_SKIP_PCC"), True)

# Fused-vs-unfused agreement bounds for LTX_FOLD_GATED_RESIDUAL. Folding the gated add into the
# to_out matmul epilogue re-associates the bf16 accumulation, so the two paths agree to rounding,
# not to the bit. CONTROL is the device's own run-to-run noise, measured by repeating a single
# path: the equivalence bound is only meaningful while the floor sits well inside it, so the
# control is asserted first and a regression there invalidates the gate rather than the fold.
# The device is bit-deterministic, so the floor is exact and the whole fused-vs-unfused delta is
# the fold's.
#
# Bounds are set from the separation between a correct fold and a wrong one. Feeding a fold the
# self-attention gate in place of the cross-attention gate (same shape, so it fails silently)
# moves audio to PCC 99.85 / RMSE 6.0%, against 99.9996 / 0.3% when correct. Note that wrong fold
# still clears the diffusers oracle below (pcc=0.992), which is why that oracle cannot gate this
# and these bounds sit an order of magnitude tighter. RMSE/σ carries the gate: PCC is shift- and
# scale-invariant, so a fold that dropped the gate multiply outright would still score high PCC
# wherever the gate is near-constant.
_FOLD_CONTROL_PCC = 0.999999
_FOLD_CONTROL_RMSE = 0.002
_FOLD_EQUIV_PCC = 0.9995
_FOLD_EQUIV_RMSE = 0.015

# Bounds for the LTX_PROBE_ADDCMUL_SPLIT control. The probe is not a shipping path: it swaps
# addcmul(t, t1, t2) for the algebraically identical add(t, multiply(t1, t2)) at the same three
# residuals the fold touches, perturbing nothing but bf16 rounding. Its whole job is to be the
# yardstick the fold's 48-layer drift is read against, and it is only a yardstick while the kick it
# injects per layer is no bigger than the fold's — so the bounds here are deliberately the fold's
# own. Red means the two perturbations are NOT magnitude-matched, and the 48-layer comparison is
# unnormalized: a control that kicks harder is expected to drift further, which would prove nothing
# about the fold.
_PROBE_CONTROL_PCC = _FOLD_CONTROL_PCC
_PROBE_CONTROL_RMSE = _FOLD_CONTROL_RMSE
_PROBE_EQUIV_PCC = _FOLD_EQUIV_PCC
_PROBE_EQUIV_RMSE = _FOLD_EQUIV_RMSE

# Agreement bounds for LTX_DEDUP_GATE_GATHER. Both paths gather the same activation and run the
# same two projections on it; only the kernel differs (fused all_gather_minimal_matmul_async vs
# a standalone gather + minimal_matmul), so they agree to bf16 rounding of the matmul, not to the
# bit. Tighter than the fold's bounds because no accumulation is re-associated across ops here.
# The control (same path twice) pins the device noise floor and is asserted first: a floor as wide
# as the bound would pass the dedup without having tested it. Separation is set by the mutant:
# LTX_DEDUP_GATE_MUTANT feeds the gate a corrupted copy of the gathered activation — the silent
# miswiring this cut could introduce — and must drive these bounds red.
_DEDUP_CONTROL_PCC = 0.999999
_DEDUP_CONTROL_RMSE = 0.002
_DEDUP_EQUIV_PCC = 0.9999
_DEDUP_EQUIV_RMSE = 0.008


# ---------------------------------------------------------------------------
# Parametrize lists
# ---------------------------------------------------------------------------
_LTX_TRANSFORMER_MESH_PARAMS = [
    # No 1x1 (sp=1) config: real-grid shapes require SP padding, and video self-attention only
    # masks padded keys via ring SDPA's logical_n (sp>1). Plain SDPA (sp=1) would attend padded
    # keys and corrupt real outputs. Production never runs sp=1.
    # 2x4sp0tp1 keeps is_fsdp=True for FSDP-path coverage on a 2D mesh.
    pytest.param((2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp0tp1"),
    # 2x4sp1tp0 mirrors production BH 2x4: is_fsdp=False (production loads via dynamic_load, not
    # per-layer FSDP gathers) or the profile shows phantom weight-gather collectives.
    pytest.param((2, 4), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="2x4sp1tp0"),
    pytest.param((4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring, True, id="wh_4x8sp1tp0"),
    pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="ring_bh_4x8sp1tp0"),
    pytest.param((4, 8), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="line_bh_4x8sp1tp0"),
]

# 1080p fast-pipeline real latent grids (F, H_lat, W_lat) — the exact (latent_frames, h//32,
# w//32) production feeds RoPE. These are NOT tile/SP-aligned, so the test SP-pads the sequence
# (round F·H·W up to TILE * sp_factor) exactly like LTXPipeline._sp_pad_len, then crops back.
_LTX_TRANSFORMER_SHAPE_PARAMS = [
    pytest.param(19, 17, 30, id="stage_1"),  # F·H·W = 9690 (real); SP-padded on device
    pytest.param(19, 34, 60, id="stage_2"),  # F·H·W = 38760 (real); SP-padded on device
]

_LTX_TRANSFORMER_MODALITY_PARAMS = [
    pytest.param(False, id="video"),
    pytest.param(True, id="av"),
]

_LTX_TRANSFORMER_RUN_PCC_PARAMS = [pytest.param(_RUN_PCC_DEFAULT, id="pcc" if _RUN_PCC_DEFAULT else "nopcc")]

# Checkpoint variant for AV mode ("fast" = production default); ignored in video mode.
_LTX_TRANSFORMER_CKPT_PARAMS = [
    pytest.param("fast", id="ckpt_fast"),
    pytest.param("dev", id="ckpt_dev"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interleaved_to_bhnd(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(B, N, dim) INTERLEAVED freqs → (B, num_heads, N, head_dim) for rotary_embedding_llama."""
    B, N, dim = t.shape
    head_dim = dim // num_heads
    return t.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def _sp_pad_len(n_real: int, sp_factor: int) -> int:
    """Round a seq len up to ttnn.TILE_SIZE * sp_factor (mirrors LTXPipeline._sp_pad_len)."""
    divisor = ttnn.TILE_SIZE * sp_factor
    return ((n_real + divisor - 1) // divisor) * divisor


def _pad_seq_dim(t: torch.Tensor, n_pad: int, dim: int = -2) -> torch.Tensor:
    """Zero-pad a torch tensor's sequence dim up to n_pad (no-op if already ≥ n_pad)."""
    n = t.shape[dim]
    if n_pad <= n:
        return t
    shape = list(t.shape)
    shape[dim] = n_pad
    out = torch.zeros(shape, dtype=t.dtype)
    idx = [slice(None)] * t.ndim
    idx[dim] = slice(0, n)
    out[tuple(idx)] = t
    return out


def _pad_rope_bhnd(cos_i: torch.Tensor, sin_i: torch.Tensor, pad_to: int | None):
    """Right-pad BHND INTERLEAVED freqs on the seq dim (2) with identity rotation (cos=1, sin=0).

    Same convention as rope_ltx.pad_video_rope_sp: padded slots are no-op rotations; SDPA
    still masks them out via logical_n / padding masks.
    """
    if pad_to is None or pad_to <= cos_i.shape[2]:
        return cos_i, sin_i
    pad = pad_to - cos_i.shape[2]
    cos_i = torch.nn.functional.pad(cos_i, (0, 0, 0, pad), value=1.0)
    sin_i = torch.nn.functional.pad(sin_i, (0, 0, 0, pad), value=0.0)
    return cos_i, sin_i


def _diffusers_qk_to_split(t: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """diffusers (interleaved-rotation) Q/K → Lightricks SPLIT convention the TT loader expects."""
    D = head_dim
    D_half = D // 2
    inv = torch.empty(D, dtype=torch.long)
    inv[:D_half] = torch.arange(0, D, 2)
    inv[D_half:] = torch.arange(1, D, 2)
    rest = t.shape[1:]
    return t.reshape(num_heads, D, *rest).index_select(1, inv).reshape(num_heads * D, *rest)


def _convert_diffusers_video_block_to_tt(state: dict, *, num_heads: int, head_dim: int) -> dict:
    """diffusers LTX2VideoTransformerBlock state_dict → TT video block loader input (video subset, split Q/K)."""
    keep_exact = {"scale_shift_table", "prompt_scale_shift_table"}
    out = {k: v for k, v in state.items() if k in keep_exact or k.startswith(("attn1.", "attn2.", "ff."))}
    for base in ("attn1", "attn2"):
        for proj in ("to_q", "to_k"):
            for suf in ("weight", "bias"):
                kk = f"{base}.{proj}.{suf}"
                if kk in out:
                    out[kk] = _diffusers_qk_to_split(out[kk], num_heads, head_dim)
        for nk in ("norm_q.weight", "norm_k.weight"):
            kk = f"{base}.{nk}"
            if kk in out:
                out[kk] = _diffusers_qk_to_split(out[kk], num_heads, head_dim)
    return out


def _make_diffusers_video_block():
    """Build the diffusers LTX-2 block in the video config used as the PCC reference."""
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformerBlock

    return LTX2VideoTransformerBlock(
        dim=DIM,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        cross_attention_dim=CTX_DIM,
        audio_dim=AUDIO_DIM,
        audio_num_attention_heads=NUM_HEADS,
        audio_attention_head_dim=AUDIO_HEAD_DIM,
        audio_cross_attention_dim=AUDIO_CTX_DIM,
        video_gated_attn=False,
        video_cross_attn_adaln=True,
        audio_gated_attn=False,
        audio_cross_attn_adaln=False,
        eps=EPS,
        rope_type="interleaved",
    )


def _diffusers_video_block_ref(block, *, x, context, temb, prompt_temb, cos_i, sin_i):
    """Run the diffusers block video-only (a2v/v2a disabled) and return the video output."""
    B, N, _ = x.shape
    a_N = 32
    audio_hidden = torch.zeros(B, a_N, AUDIO_DIM)
    audio_enc = torch.zeros(B, context.shape[1], AUDIO_CTX_DIM)
    temb_audio = torch.zeros(B, 1, 6 * AUDIO_DIM)  # audio_cross_attn_adaln=False → 6 mod params
    temb_prompt_audio = torch.zeros(B, 1, 2 * AUDIO_DIM)
    with torch.no_grad():
        out_v, _ = block(
            hidden_states=x,
            audio_hidden_states=audio_hidden,
            encoder_hidden_states=context,
            audio_encoder_hidden_states=audio_enc,
            temb=temb,
            temb_audio=temb_audio,
            temb_ca_scale_shift=None,
            temb_ca_audio_scale_shift=None,
            temb_ca_gate=None,
            temb_ca_audio_gate=None,
            temb_prompt=prompt_temb,
            temb_prompt_audio=temb_prompt_audio,
            video_rotary_emb=(cos_i, sin_i),
            audio_rotary_emb=None,
            use_a2v_cross_attention=False,
            use_v2a_cross_attention=False,
        )
    return out_v


def _make_diffusers_video_model(num_layers: int = 1):
    """diffusers LTX2VideoTransformer3DModel configured to match the TT video model."""
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    return LTX2VideoTransformer3DModel(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        cross_attention_dim=CTX_DIM,
        audio_in_channels=AUDIO_IN_CHANNELS,
        audio_out_channels=AUDIO_IN_CHANNELS,
        audio_num_attention_heads=NUM_HEADS,
        audio_attention_head_dim=AUDIO_HEAD_DIM,
        audio_cross_attention_dim=AUDIO_CTX_DIM,
        num_layers=num_layers,
        cross_attn_mod=True,
        audio_cross_attn_mod=False,
        norm_eps=EPS,
        rope_theta=10000.0,
        rope_double_precision=False,
        use_prompt_embeddings=False,
        rope_type="interleaved",
    )


def _convert_diffusers_video_model_to_tt(state: dict, *, num_heads: int, head_dim: int) -> dict:
    """diffusers LTX2VideoTransformer3DModel state_dict → TT LTXTransformerModel loader input."""
    drop_top = ("audio_", "av_cross_attn_", "caption_projection")
    rename_top = {"proj_in.": "patchify_proj.", "time_embed.": "adaln_single.", "prompt_adaln.": "prompt_adaln_single."}
    out: dict = {}
    for k, v in state.items():
        if k.startswith("transformer_blocks."):
            _, _idx, rest = k.split(".", 2)
            if rest in ("scale_shift_table", "prompt_scale_shift_table") or rest.startswith(
                ("attn1.", "attn2.", "ff.")
            ):
                out[k] = v
            continue
        if k.startswith(drop_top):
            continue
        nk = k
        for a, b in rename_top.items():
            if nk.startswith(a):
                nk = b + nk[len(a) :]
                break
        out[nk] = v
    qk_suffixes = ("to_q.weight", "to_q.bias", "to_k.weight", "to_k.bias", "norm_q.weight", "norm_k.weight")
    for k in list(out.keys()):
        if (".attn1." in k or ".attn2." in k) and k.endswith(qk_suffixes):
            out[k] = _diffusers_qk_to_split(out[k], num_heads, head_dim)
    return out


def _diffusers_video_model_ref(model, *, video_lat, video_prompt, sigma_val, F, H, W):
    """Run the diffusers 3D model video-only (isolate_modalities=True) and return video output."""
    B, N, _ = video_lat.shape
    gt, gh, gw, _ = _video_grid(F, H, W)
    video_coords = torch.stack([gt, gh, gw], dim=0).float().unsqueeze(0)  # (1, 3, N)
    a_N = 64
    audio_lat = torch.zeros(B, a_N, AUDIO_IN_CHANNELS)
    audio_prompt = torch.zeros(B, video_prompt.shape[1], AUDIO_CTX_DIM)
    audio_coords = torch.arange(a_N).reshape(1, 1, a_N).float()
    ts = torch.tensor([sigma_val * 1000.0])
    with torch.no_grad():
        out = model(
            hidden_states=video_lat,
            audio_hidden_states=audio_lat,
            encoder_hidden_states=video_prompt,
            audio_encoder_hidden_states=audio_prompt,
            timestep=ts,
            audio_timestep=ts,
            sigma=ts,
            audio_sigma=ts,
            video_coords=video_coords,
            audio_coords=audio_coords,
            isolate_modalities=True,
            return_dict=False,
        )
    return out[0] if isinstance(out, (tuple, list)) else out


def _diffusers_video_model_ref_pertoken(model, *, video_lat, video_prompt, video_ts_real, sigma_scalar, F, H, W):
    """Per-token oracle: same as ``_diffusers_video_model_ref`` but ``timestep`` is ``(1, N)``.

    The diffusers LTX2 model flattens ``timestep`` through ``time_embed`` then reshapes to
    ``(B, N, ...)`` modulation, so a per-token timestep yields genuine per-token modulation.
    ``sigma`` (prompt/cross modulation) stays scalar, matching the TT scalar-timestep path.
    """
    B, N, _ = video_lat.shape
    gt, gh, gw, _ = _video_grid(F, H, W)
    video_coords = torch.stack([gt, gh, gw], dim=0).float().unsqueeze(0)  # (1, 3, N)
    a_N = 64
    audio_lat = torch.zeros(B, a_N, AUDIO_IN_CHANNELS)
    audio_prompt = torch.zeros(B, video_prompt.shape[1], AUDIO_CTX_DIM)
    audio_coords = torch.arange(a_N).reshape(1, 1, a_N).float()
    ts_pertoken = video_ts_real.reshape(1, -1) * 1000.0  # (1, N)
    ts_scalar = torch.tensor([sigma_scalar * 1000.0])
    with torch.no_grad():
        out = model(
            hidden_states=video_lat,
            audio_hidden_states=audio_lat,
            encoder_hidden_states=video_prompt,
            audio_encoder_hidden_states=audio_prompt,
            timestep=ts_pertoken,
            audio_timestep=ts_scalar,
            sigma=ts_scalar,
            audio_sigma=ts_scalar,
            video_coords=video_coords,
            audio_coords=audio_coords,
            isolate_modalities=True,
            return_dict=False,
        )
    return out[0] if isinstance(out, (tuple, list)) else out


def _lightricks_qk_to_interleaved(t: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Forward per-head SPLIT→INTERLEAVED Q/K permute (same as LTXAttention._permute_qk)."""
    D = head_dim
    D_half = D // 2
    perm = torch.empty(D, dtype=torch.long)
    perm[0::2] = torch.arange(D_half)
    perm[1::2] = torch.arange(D_half, D)
    rest = t.shape[1:]
    return t.reshape(num_heads, D, *rest).index_select(1, perm).reshape(num_heads * D, *rest)


# Per-attention head_dim for the AV block (video self/text-cross use 128; audio + cross-modal use 64).
_AV_ATTN_HEAD_DIM = {
    "attn1": HEAD_DIM,
    "attn2": HEAD_DIM,
    "audio_attn1": AUDIO_HEAD_DIM,
    "audio_attn2": AUDIO_HEAD_DIM,
    "audio_to_video_attn": AUDIO_HEAD_DIM,
    "video_to_audio_attn": AUDIO_HEAD_DIM,
}


def _convert_lightricks_av_to_diffusers(state: dict, *, num_heads: int) -> dict:
    """Raw Lightricks/ltx_core AV state_dict → diffusers LTX2VideoTransformer3DModel (AV) names."""
    rename_top = {
        "patchify_proj.": "proj_in.",
        "adaln_single.": "time_embed.",
        "prompt_adaln_single.": "prompt_adaln.",
        "audio_patchify_proj.": "audio_proj_in.",
        "audio_adaln_single.": "audio_time_embed.",
        "audio_prompt_adaln_single.": "audio_prompt_adaln.",
        "av_ca_video_scale_shift_adaln_single.": "av_cross_attn_video_scale_shift.",
        "av_ca_audio_scale_shift_adaln_single.": "av_cross_attn_audio_scale_shift.",
        "av_ca_a2v_gate_adaln_single.": "av_cross_attn_video_a2v_gate.",
        "av_ca_v2a_gate_adaln_single.": "av_cross_attn_audio_v2a_gate.",
    }
    block_table_rename = {
        "scale_shift_table_a2v_ca_video": "video_a2v_cross_attn_scale_shift_table",
        "scale_shift_table_a2v_ca_audio": "audio_a2v_cross_attn_scale_shift_table",
    }
    out: dict = {}
    for k, v in state.items():
        if k.startswith("transformer_blocks."):
            prefix, idx, rest = k.split(".", 2)
            if rest in block_table_rename:
                rest = block_table_rename[rest]
            rest = rest.replace("q_norm", "norm_q").replace("k_norm", "norm_k")
            out[f"{prefix}.{idx}.{rest}"] = v
        else:
            nk = k
            for a, b in rename_top.items():
                if nk.startswith(a):
                    nk = b + nk[len(a) :]
                    break
            out[nk] = v
    qk_suffixes = ("to_q.weight", "to_q.bias", "to_k.weight", "to_k.bias", "norm_q.weight", "norm_k.weight")
    for k in list(out.keys()):
        if not k.startswith("transformer_blocks."):
            continue
        attn = k.split(".")[2]
        if attn in _AV_ATTN_HEAD_DIM and k.endswith(qk_suffixes):
            out[k] = _lightricks_qk_to_interleaved(out[k], num_heads, _AV_ATTN_HEAD_DIM[attn])
    return out


def _make_diffusers_av_model(num_layers: int = 1):
    """diffusers ``LTX2VideoTransformer3DModel`` configured to match the TT AV model (gated
    attention, 9-param video+audio modulation, cross-modal a2v/v2a, no caption projection)."""
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    return LTX2VideoTransformer3DModel(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        cross_attention_dim=CTX_DIM,
        audio_in_channels=AUDIO_IN_CHANNELS,
        audio_out_channels=AUDIO_IN_CHANNELS,
        audio_num_attention_heads=NUM_HEADS,
        audio_attention_head_dim=AUDIO_HEAD_DIM,
        audio_cross_attention_dim=AUDIO_CTX_DIM,
        num_layers=num_layers,
        gated_attn=True,
        audio_gated_attn=True,
        cross_attn_mod=True,
        audio_cross_attn_mod=True,
        norm_eps=EPS,
        rope_theta=ROPE_THETA,
        rope_double_precision=False,
        use_prompt_embeddings=False,
        rope_type="interleaved",
        # =1 matches ltx_core's av_ca_timestep_scale_multiplier (a2v/v2a gate adaln sees raw sigma).
        cross_attn_timestep_scale_multiplier=1,
    )


def _diffusers_av_model_ref(model, *, video_lat, video_prompt, audio_lat, audio_prompt, sigma_val, F, H, W, audio_N):
    """Run the diffusers AV model (cross-modal enabled) and return (video, audio) outputs."""
    gt, gh, gw, _ = _video_grid(F, H, W)
    video_coords = torch.stack([gt, gh, gw], dim=0).float().unsqueeze(0)  # (1, 3, N)
    audio_coords = torch.arange(audio_N).reshape(1, 1, audio_N).float()
    ts = torch.tensor([sigma_val * 1000.0])
    with torch.no_grad():
        out = model(
            hidden_states=video_lat,
            audio_hidden_states=audio_lat,
            encoder_hidden_states=video_prompt,
            audio_encoder_hidden_states=audio_prompt,
            timestep=ts,
            audio_timestep=ts,
            sigma=ts,
            audio_sigma=ts,
            video_coords=video_coords,
            audio_coords=audio_coords,
            isolate_modalities=False,
            return_dict=False,
        )
    return out[0], out[1]


def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )


def _make_ccl_manager(mesh_device, num_links, topology):
    # LTX_NUM_LINKS overrides the param default to A/B fabric link count (BH prod = 2, WH = 4).
    num_links = int(os.environ.get("LTX_NUM_LINKS", str(num_links)))
    return CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)


def _video_grid(F, H, W):
    """Return (grid_t, grid_h, grid_w) flat tensors and stacked 3D positions (1, N, 3)."""
    t_ids, h_ids, w_ids = torch.arange(F), torch.arange(H), torch.arange(W)
    gt, gh, gw = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    gt_f, gh_f, gw_f = gt.flatten(), gh.flatten(), gw.flatten()
    positions = torch.stack([gt_f, gh_f, gw_f], dim=-1).float().unsqueeze(0)
    return gt_f, gh_f, gw_f, positions


def _video_rope_freqs(F, H, W, rope_type=LTXRopeType.SPLIT):
    _, _, _, positions = _video_grid(F, H, W)
    return precompute_freqs_cis(
        positions,
        dim=DIM,
        out_dtype=torch.float32,
        theta=ROPE_THETA,
        max_pos=VIDEO_ROPE_MAX_POS,
        num_attention_heads=NUM_HEADS,
        rope_type=rope_type,
    )


def _video_cross_pe_freqs(F, H, W, rope_type=LTXRopeType.SPLIT):
    """T-only cross-PE for V↔A, dim matches audio (AUDIO_DIM)."""
    _, _, _, positions = _video_grid(F, H, W)
    return precompute_freqs_cis(
        positions[..., 0:1],
        dim=AUDIO_DIM,
        out_dtype=torch.float32,
        theta=ROPE_THETA,
        max_pos=CROSS_PE_MAX_POS,
        num_attention_heads=NUM_HEADS,
        rope_type=rope_type,
    )


def _audio_rope_freqs(audio_N, rope_type=LTXRopeType.SPLIT):
    a_pos = torch.arange(audio_N).float().unsqueeze(0).unsqueeze(-1)
    return precompute_freqs_cis(
        a_pos,
        dim=AUDIO_DIM,
        out_dtype=torch.float32,
        theta=ROPE_THETA,
        max_pos=AUDIO_ROPE_MAX_POS,
        num_attention_heads=NUM_HEADS,
        rope_type=rope_type,
    )


def _audio_cross_pe_freqs(audio_N, rope_type=LTXRopeType.SPLIT):
    a_pos = torch.arange(audio_N).float().unsqueeze(0).unsqueeze(-1)
    return precompute_freqs_cis(
        a_pos,
        dim=AUDIO_DIM,
        out_dtype=torch.float32,
        theta=ROPE_THETA,
        max_pos=CROSS_PE_MAX_POS,
        num_attention_heads=NUM_HEADS,
        rope_type=rope_type,
    )


def _audio_seq_lens(F: int, sp_factor: int) -> tuple[int, int]:
    """Fast-pipeline audio seq lengths from the latent frame count (audio_N_real, padded to 32*sp)."""
    num_frames = (F - 1) * 8 + 1
    vps = VideoPixelShape(batch=1, frames=num_frames, height=64, width=64, fps=24)
    audio_N_real = AudioLatentShape.from_video_pixel_shape(vps).frames
    audio_N = ((audio_N_real + 32 * sp_factor - 1) // (32 * sp_factor)) * (32 * sp_factor)
    return audio_N, audio_N_real


def _tt_rope(freqs_fn, *args, mesh_device, sp_axis, tp_axis, pad_to=None):
    """Build INTERLEAVED freqs for the TT runtime: precompute → BHND reshape → SP-pad → 2D shard."""
    cos_i, sin_i = freqs_fn(*args, rope_type=LTXRopeType.INTERLEAVED)
    cos_i = _interleaved_to_bhnd(cos_i, NUM_HEADS)
    sin_i = _interleaved_to_bhnd(sin_i, NUM_HEADS)
    cos_i, sin_i = _pad_rope_bhnd(cos_i, sin_i, pad_to)
    tt_cos = bf16_tensor_2dshard(cos_i, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_i, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    return tt_cos, tt_sin


def _tt_rope_full(freqs_fn, *args, mesh_device, tp_axis, pad_to=None):
    """INTERLEAVED freqs replicated full-seq (cross-attn K side): BHND reshape, SP-pad, no SP shard."""
    cos_i, sin_i = freqs_fn(*args, rope_type=LTXRopeType.INTERLEAVED)
    cos_i = _interleaved_to_bhnd(cos_i, NUM_HEADS)
    sin_i = _interleaved_to_bhnd(sin_i, NUM_HEADS)
    cos_i, sin_i = _pad_rope_bhnd(cos_i, sin_i, pad_to)
    tt_cos = bf16_tensor(cos_i, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)
    tt_sin = bf16_tensor(sin_i, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)
    return tt_cos, tt_sin


def _make_tt_block(*, mesh_device, ccl_manager, parallel_config, is_fsdp, has_audio):
    return LTXTransformerBlock(
        video_dim=DIM,
        video_ffn_dim=DIM * 4,
        video_num_heads=NUM_HEADS,
        video_cross_attention_dim=CTX_DIM,
        audio_dim=AUDIO_DIM,
        audio_ffn_dim=AUDIO_DIM * 4,
        audio_num_heads=NUM_HEADS,
        audio_cross_attention_dim=AUDIO_CTX_DIM,
        eps=EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=has_audio,
        apply_gated_attention=has_audio,
        cross_attention_adaln=True,
    )


def _make_tt_model(
    *, mesh_device, ccl_manager, parallel_config, is_fsdp, has_audio, num_layers, image_conditioning=False
):
    return LTXTransformerModel(
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_layers=num_layers,
        cross_attention_dim=CTX_DIM,
        audio_num_attention_heads=NUM_HEADS,
        audio_attention_head_dim=AUDIO_HEAD_DIM,
        audio_in_channels=AUDIO_IN_CHANNELS,
        audio_out_channels=AUDIO_IN_CHANNELS,
        audio_cross_attention_dim=AUDIO_CTX_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=has_audio,
        apply_gated_attention=has_audio,
        cross_attention_adaln=True,
        image_conditioning=image_conditioning,
    )


def _load_22b_state_dict(num_layers: int, checkpoint_path: str) -> dict | None:
    """Load LTX-2.3-22B weights filtered to the first num_layers blocks; None if checkpoint missing."""
    if not os.path.exists(checkpoint_path):
        return None
    raw = load_file(checkpoint_path)
    prefix = "model.diffusion_model."
    sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    return {k: v for k, v in sd.items() if not k.startswith("transformer_blocks.") or int(k.split(".")[1]) < num_layers}


def _scale_init_(module: torch.nn.Module, seed: int = WEIGHT_SEED) -> None:
    """Deterministic N(0, 0.1²) reinit in-place. See block test comment for why."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, dtype=p.dtype) * 0.1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_TRANSFORMER_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("F", "H", "W"), _LTX_TRANSFORMER_SHAPE_PARAMS)
@pytest.mark.parametrize("has_audio", _LTX_TRANSFORMER_MODALITY_PARAMS)
@pytest.mark.parametrize("run_pcc", _LTX_TRANSFORMER_RUN_PCC_PARAMS)
@pytest.mark.parametrize("checkpoint_variant", _LTX_TRANSFORMER_CKPT_PARAMS)
def test_ltx_transformer_block(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    F: int,
    H: int,
    W: int,
    has_audio: bool,
    run_pcc: bool,
    checkpoint_variant: str,
    reset_seeds,
) -> None:
    """Test LTXTransformerBlock: TT forward, with optional PCC vs the diffusers LTX-2 block."""
    # Checkpoint variant only affects AV weight loading; skip the redundant copy in video mode.
    if not has_audio and checkpoint_variant != "fast":
        pytest.skip("checkpoint_variant only affects AV mode (video uses random scaled weights)")
    checkpoint_22b = _resolve_checkpoint_22b(checkpoint_variant)
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    # Real latent grid → SP-padded sequence (mirrors LTXPipeline). video_N is the padded device
    # length; video_N_real is the logical token count fed to SDPA's logical_n / padding masks.
    video_N_real = F * H * W
    video_N = _sp_pad_len(video_N_real, sp_factor)
    # Audio seq length mirrors the fast pipeline so op sizes / padding match production.
    audio_N, audio_N_real = _audio_seq_lens(F, sp_factor)
    assert video_N % (32 * sp_factor) == 0, f"video_N={video_N} not sp/tile-aligned for sp={sp_factor}"
    assert audio_N % (32 * sp_factor) == 0, f"audio_N={audio_N} not sp/tile-aligned for sp={sp_factor}"

    # AV mode lacks the LTXModel-level cross-PE plumbing, so it asserts shape/finiteness only.
    do_pcc = run_pcc and not has_audio
    torch_block = None
    torch_out = None
    if do_pcc:
        # Reference is the diffusers LTX-2 block run video-only (audio + a2v/v2a disabled).
        torch_block = _make_diffusers_video_block()
        torch_block.eval()
        _scale_init_(torch_block)

    # TT block
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    tt_block = _make_tt_block(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=has_audio,
    )

    if do_pcc:
        # diffusers→TT: keep video subset, convert attn Q/K to split. Clone to avoid aliasing.
        conv = _convert_diffusers_video_block_to_tt(torch_block.state_dict(), num_heads=NUM_HEADS, head_dim=HEAD_DIM)
        tt_block.load_torch_state_dict({k: v.detach().clone() for k, v in conv.items()})
    elif has_audio:
        sd = _load_22b_state_dict(num_layers=1, checkpoint_path=checkpoint_22b)
        if sd is None:
            pytest.skip(f"22B checkpoint not found at {checkpoint_22b}")
        block_sd = {
            k[len("transformer_blocks.0.") :]: v for k, v in sd.items() if k.startswith("transformer_blocks.0.")
        }
        tt_block.load_torch_state_dict(block_sd, strict=False)
    else:
        # video mode, PCC disabled — harvest a shape-matching state dict from a throwaway block.
        dummy = _make_diffusers_video_block()
        _scale_init_(dummy)
        conv = _convert_diffusers_video_block_to_tt(dummy.state_dict(), num_heads=NUM_HEADS, head_dim=HEAD_DIM)
        tt_block.load_torch_state_dict({k: v.detach().clone() for k, v in conv.items()})
        del dummy

    # Inputs (real-grid token count; the TT side gets a zero-padded copy below).
    torch.manual_seed(INPUT_SEED)
    x = torch.randn(1, video_N_real, DIM, dtype=torch.float32)
    context = torch.randn(1, PROMPT_LEN, CTX_DIM, dtype=torch.float32)
    temb = torch.randn(1, 1, 9 * DIM, dtype=torch.float32)  # 9 adaln params
    prompt_temb = torch.randn(1, 1, 2 * DIM, dtype=torch.float32)  # 2 adaln params for prompt

    # Torch forward (video-only branch): diffusers block consumes INTERLEAVED cos/sin, same
    # convention as the TT runtime (tt_dit interleaved freqs ≈ diffusers rope, PCC ~1.0).
    if do_pcc:
        cos_int, sin_int = _video_rope_freqs(F, H, W, rope_type=LTXRopeType.INTERLEAVED)
        torch_out = _diffusers_video_block_ref(
            torch_block, x=x, context=context, temb=temb, prompt_temb=prompt_temb, cos_i=cos_int, sin_i=sin_int
        )
        logger.info(f"torch block output {tuple(torch_out.shape)}")

    # TT video tensors. Runtime rope is INTERLEAVED + trans_mat. Sequence is SP-padded to video_N;
    # rope pads with identity rotation; video_N (logical) goes to the block for SDPA masking.
    spatial = _pad_seq_dim(x, video_N, dim=1).unsqueeze(0)
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_prompt = bf16_tensor(context.unsqueeze(0), device=mesh_device)
    # Outer-param layout (coeff, B=1, 1, D) so the block's chunk(dim=0) is a free tile-aligned slice.
    tt_temb = bf16_tensor(
        temb.reshape(9, DIM).unsqueeze(1).unsqueeze(1), device=mesh_device, mesh_axis=tp_axis, shard_dim=3
    )
    tt_prompt_temb = bf16_tensor(prompt_temb.reshape(2, DIM).unsqueeze(1).unsqueeze(1), device=mesh_device)
    tt_cos, tt_sin = _tt_rope(
        _video_rope_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, pad_to=video_N
    )
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    forward_kwargs = dict(
        video_1BND=tt_spatial,
        video_prompt=tt_prompt,
        video_temb=tt_temb,
        video_N=video_N_real,
        video_rope_cos=tt_cos,
        video_rope_sin=tt_sin,
        trans_mat=tt_trans_mat,
        video_prompt_temb=tt_prompt_temb,
    )
    if has_audio:
        # Real tokens in [:audio_N_real], zeros in the padded tail (matches the padded audio latent).
        a_x = torch.zeros(1, audio_N, AUDIO_DIM, dtype=torch.float32)
        a_x[:, :audio_N_real, :] = torch.randn(1, audio_N_real, AUDIO_DIM, dtype=torch.float32)
        a_ctx = torch.randn(1, PROMPT_LEN, AUDIO_CTX_DIM, dtype=torch.float32)
        a_temb = torch.randn(1, 1, 9 * AUDIO_DIM, dtype=torch.float32)
        a_prompt_temb = torch.randn(1, 1, 2 * AUDIO_DIM, dtype=torch.float32)
        av_ca_v = torch.randn(1, 1, 5 * DIM, dtype=torch.float32)  # 4 scale-shift + 1 gate
        av_ca_a = torch.randn(1, 1, 5 * AUDIO_DIM, dtype=torch.float32)

        a_cos, a_sin = _tt_rope(_audio_rope_freqs, audio_N, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis)
        vx_cos, vx_sin = _tt_rope(
            _video_cross_pe_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, pad_to=video_N
        )
        ax_cos, ax_sin = _tt_rope(
            _audio_cross_pe_freqs, audio_N, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis
        )
        ax_cos_full, ax_sin_full = _tt_rope_full(
            _audio_cross_pe_freqs, audio_N, mesh_device=mesh_device, tp_axis=tp_axis
        )

        # Padding masks, same construction as the fast pipeline (audio padded, video aligned).
        a_attn_mask, a_pad_sp, a_pad_full = build_audio_masks(
            audio_N, audio_N_real, mesh_device=mesh_device, sp_axis=sp_axis
        )
        v_pad_sp = build_video_pad_mask(video_N, video_N_real, mesh_device=mesh_device, sp_axis=sp_axis)

        forward_kwargs.update(
            audio_1BND=bf16_tensor_2dshard(
                a_x.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
            ),
            audio_prompt=bf16_tensor(a_ctx.unsqueeze(0), device=mesh_device),
            audio_temb=bf16_tensor(
                a_temb.reshape(9, AUDIO_DIM).unsqueeze(1).unsqueeze(1),
                device=mesh_device,
                mesh_axis=tp_axis,
                shard_dim=3,
            ),
            audio_prompt_temb=bf16_tensor(
                a_prompt_temb.reshape(2, AUDIO_DIM).unsqueeze(1).unsqueeze(1), device=mesh_device
            ),
            av_ca_temb=bf16_tensor(
                av_ca_v.reshape(5, DIM).unsqueeze(1).unsqueeze(1), device=mesh_device, mesh_axis=tp_axis, shard_dim=3
            ),
            av_ca_audio_temb=bf16_tensor(
                av_ca_a.reshape(5, AUDIO_DIM).unsqueeze(1).unsqueeze(1),
                device=mesh_device,
                mesh_axis=tp_axis,
                shard_dim=3,
            ),
            audio_N=audio_N,
            audio_rope_cos=a_cos,
            audio_rope_sin=a_sin,
            video_cross_pe_cos=vx_cos,
            video_cross_pe_sin=vx_sin,
            audio_cross_pe_cos=ax_cos,
            audio_cross_pe_sin=ax_sin,
            audio_cross_pe_cos_full=ax_cos_full,
            audio_cross_pe_sin_full=ax_sin_full,
            audio_attn_mask=a_attn_mask,
            audio_padding_mask=a_pad_sp,
            audio_padding_mask_full=a_pad_full,
            video_padding_mask=v_pad_sp,
        )
        # LTX_SKIP_CROSS_ATTN ablates the a2v/v2a cross-modal block to split the audio-path floor:
        # (av) − (av skip-cross) isolates the cross-modal cost — the target of the A→V collective
        # fold — and the residual is audio-self attention. The av path is WARM_FWD-only (no PCC),
        # so dropping the cross-modal residual does not disturb any quality gate.
        if os.environ.get("LTX_SKIP_CROSS_ATTN", "0") in ("1", "true", "True"):
            forward_kwargs["skip_cross_attn"] = True

    # L1 quality gate: LTX_QUANT names a QuantConfig preset (e.g. all_bf8_lofi) to apply the
    # exact pipeline quant path (weight typecast + compute configs) to this block, then PCC it
    # against the bf16 diffusers oracle. Off by default — baseline runs bf16/HiFi2.
    _quant_preset = os.environ.get("LTX_QUANT", "").strip()
    if _quant_preset:
        from models.tt_dit.pipelines.ltx.quant_config import QuantConfig, apply_quant_config_to_block

        _factory = getattr(QuantConfig, _quant_preset, None)
        assert callable(_factory), f"LTX_QUANT='{_quant_preset}' is not a QuantConfig preset"
        logger.info(f"LTX_QUANT='{_quant_preset}': quantizing block for PCC gate")
        apply_quant_config_to_block(tt_block, _factory(), mesh_device.arch(), has_audio)

    # LTX_PROFILE_ITERS>1 re-runs the same forward so warm (program-cache-hit) iterations exist for a
    # steady-state profile; the block is functional (no input mutation) so every iteration is identical.
    # Drain each lap so warm markers reach profile_log_device.csv even if teardown is cut short.
    _prof_iters = int(os.environ.get("LTX_PROFILE_ITERS", "1"))
    _warm_ms = []
    for _i in range(_prof_iters):
        _t0 = time.perf_counter()
        tt_out = tt_block(**forward_kwargs)
        ttnn.synchronize_device(mesh_device)
        if _i > 0:  # first lap is cold-compile; time only warm laps
            _warm_ms.append((time.perf_counter() - _t0) * 1000)
        if _prof_iters > 1:
            ttnn.ReadDeviceProfiler(mesh_device)
    if _warm_ms:
        logger.info(
            f"WARM_FWD_MS={sum(_warm_ms) / len(_warm_ms):.2f} "
            f"num_links={os.environ.get('LTX_NUM_LINKS', 'param')} iters={len(_warm_ms)}"
        )
    if os.environ.get("LTX_DEDUP_PERF_AB", "0") in ("1", "true", "True"):
        _dedup_perf_ab(tt_block, forward_kwargs, mesh_device, iters=max(_prof_iters, 5))

    # Signpost-bracket one warm forward for tt-perf-report --start/end-signpost. The loop above
    # leaves the block warm, so the bracketed forward is a program-cache hit.
    tt_out = tt_block(**forward_kwargs)
    ttnn.synchronize_device(mesh_device)
    signpost("start")
    tt_out = tt_block(**forward_kwargs)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")

    if has_audio:
        tt_v, tt_a = tt_out
    else:
        tt_v = tt_out

    # Gather video output
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3
    tt_v_torch = ttnn.to_torch(
        tt_v,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    ).squeeze(0)

    # Crop the SP-padding tail off the gathered video before checking/comparing real tokens.
    tt_v_torch = tt_v_torch[:, :video_N_real, :]
    assert tt_v_torch.shape == (1, video_N_real, DIM), f"video shape {tt_v_torch.shape}"
    assert torch.isfinite(tt_v_torch).all(), "video output NaN/Inf"

    if has_audio:
        tt_a_torch = ttnn.to_torch(
            tt_a,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
        ).squeeze(0)[:, :audio_N_real, :]
        assert tt_a_torch.shape == (1, audio_N_real, AUDIO_DIM), f"audio shape {tt_a_torch.shape}"
        assert torch.isfinite(tt_a_torch).all(), "audio output NaN/Inf"

    if do_pcc:
        # 4x8 mesh has 8-way SP ring all-gathers — looser tolerance.
        pcc = 0.988 if mesh_device.get_num_devices() > 8 else 0.999
        rmse = 0.10 if mesh_device.get_num_devices() > 8 else 0.032
        assert_quality(torch_out, tt_v_torch, pcc=pcc, relative_rmse=rmse)
        logger.info(f"PASSED block PCC: video {tuple(tt_v_torch.shape)}")
    else:
        logger.info(f"PASSED block (no PCC): video {tuple(tt_v_torch.shape)}")


def _assert_fold_equivalence(forward_to_host, *, fused_video, fused_audio) -> None:
    """Gate LTX_FOLD_GATED_RESIDUAL: the fused path must match the unfused one to within bf16 rounding.

    The caller has already run the fused path. Re-running with the module flag cleared exercises the
    standalone-addcmul path on the same weights, inputs, device and process, so a difference between
    the two is attributable to the fold and nothing else. A second unfused run pins the device's
    run-to-run noise floor, which must land well inside the equivalence bound for that bound to mean
    anything — a floor as wide as the bound would pass the fold without having tested it.
    """
    assert transformer_ltx.LTX_FOLD_GATED_RESIDUAL, "the fused path must be the one already run"
    assert fused_audio is not None, "the gated residuals under test exist only on the AV path"

    transformer_ltx.LTX_FOLD_GATED_RESIDUAL = False
    try:
        unfused_video, unfused_audio = forward_to_host()
        control_video, control_audio = forward_to_host()
    finally:
        transformer_ltx.LTX_FOLD_GATED_RESIDUAL = True

    logger.info("fold A/B — noise floor, unfused vs unfused (video):")
    assert_quality(unfused_video, control_video, pcc=_FOLD_CONTROL_PCC, relative_rmse=_FOLD_CONTROL_RMSE)
    logger.info("fold A/B — noise floor, unfused vs unfused (audio):")
    assert_quality(unfused_audio, control_audio, pcc=_FOLD_CONTROL_PCC, relative_rmse=_FOLD_CONTROL_RMSE)

    logger.info("fold A/B — fused vs unfused (video):")
    assert_quality(unfused_video, fused_video, pcc=_FOLD_EQUIV_PCC, relative_rmse=_FOLD_EQUIV_RMSE)
    logger.info("fold A/B — fused vs unfused (audio):")
    assert_quality(unfused_audio, fused_audio, pcc=_FOLD_EQUIV_PCC, relative_rmse=_FOLD_EQUIV_RMSE)
    logger.info("PASSED fold equivalence")


def _assert_probe_split_magnitude(forward_to_host, *, addcmul_video, addcmul_audio) -> None:
    """Size LTX_PROBE_ADDCMUL_SPLIT's per-layer kick against the fold's, on the unfolded path.

    The probe reaches the three gated residuals only through _gated_residual(), which the fold
    compiles out, so it perturbs anything at all only under LTX_FOLD_GATED_RESIDUAL=0 — which is
    also the baseline the 48-layer A/B reads both the fold and this control against. That makes the
    two perturbations single-flag deltas off one common baseline, hitting the same three sites.

    The caller has already run the plain-addcmul path. Setting the module flag re-runs the same
    weights, inputs, device and process through the split, so the difference is the rounding change
    and nothing else. Running the split twice pins the device's run-to-run floor, which must sit
    well inside the bound for the bound to mean anything.
    """
    assert not transformer_ltx.LTX_FOLD_GATED_RESIDUAL, (
        "the fold absorbs the three addcmuls into the to_out epilogue, so _gated_residual — the "
        "probe's only consumer — never runs and the probe measures nothing"
    )
    assert not transformer_ltx.LTX_PROBE_ADDCMUL_SPLIT, "the plain-addcmul path must be the one already run"
    assert addcmul_audio is not None, "the gated residuals this probe perturbs exist only on the AV path"

    transformer_ltx.LTX_PROBE_ADDCMUL_SPLIT = True
    try:
        split_video, split_audio = forward_to_host()
        control_video, control_audio = forward_to_host()
    finally:
        transformer_ltx.LTX_PROBE_ADDCMUL_SPLIT = False

    logger.info("probe A/B — noise floor, split vs split (video):")
    assert_quality(split_video, control_video, pcc=_PROBE_CONTROL_PCC, relative_rmse=_PROBE_CONTROL_RMSE)
    logger.info("probe A/B — noise floor, split vs split (audio):")
    assert_quality(split_audio, control_audio, pcc=_PROBE_CONTROL_PCC, relative_rmse=_PROBE_CONTROL_RMSE)

    logger.info("probe A/B — split vs addcmul (video):")
    assert_quality(addcmul_video, split_video, pcc=_PROBE_EQUIV_PCC, relative_rmse=_PROBE_EQUIV_RMSE)
    logger.info("probe A/B — split vs addcmul (audio):")
    assert_quality(addcmul_audio, split_audio, pcc=_PROBE_EQUIV_PCC, relative_rmse=_PROBE_EQUIV_RMSE)
    logger.info("PASSED probe magnitude")


def _assert_dedup_equivalence(forward_to_host, *, dedup_video, dedup_audio) -> None:
    """Gate LTX_DEDUP_GATE_GATHER: one hoisted activation gather feeding the gate and Q/QKV must
    match the two fused gathers it replaces, to within bf16 rounding.

    The caller has already run the dedup path. Clearing the module flag restores the double-gather
    on the same weights, inputs, device and process, so any difference is the dedup's. A second
    double-gather run pins the device's run-to-run noise floor, which must land well inside the
    equivalence bound for that bound to mean anything.
    """
    assert attention_ltx.LTX_DEDUP_GATE_GATHER, "the dedup path must be the one already run"
    assert dedup_audio is not None, "the gate whose gather this dedups exists only on the AV path"

    attention_ltx.LTX_DEDUP_GATE_GATHER = False
    try:
        base_video, base_audio = forward_to_host()
        control_video, control_audio = forward_to_host()
    finally:
        attention_ltx.LTX_DEDUP_GATE_GATHER = True

    logger.info("dedup A/B — noise floor, double-gather vs double-gather (video):")
    assert_quality(base_video, control_video, pcc=_DEDUP_CONTROL_PCC, relative_rmse=_DEDUP_CONTROL_RMSE)
    logger.info("dedup A/B — noise floor, double-gather vs double-gather (audio):")
    assert_quality(base_audio, control_audio, pcc=_DEDUP_CONTROL_PCC, relative_rmse=_DEDUP_CONTROL_RMSE)

    logger.info("dedup A/B — dedup vs double-gather (video):")
    assert_quality(base_video, dedup_video, pcc=_DEDUP_EQUIV_PCC, relative_rmse=_DEDUP_EQUIV_RMSE)
    logger.info("dedup A/B — dedup vs double-gather (audio):")
    assert_quality(base_audio, dedup_audio, pcc=_DEDUP_EQUIV_PCC, relative_rmse=_DEDUP_EQUIV_RMSE)
    logger.info("PASSED dedup equivalence")


def _dedup_perf_ab(tt_block, forward_kwargs, mesh_device, *, iters: int) -> None:
    """Time the block with LTX_DEDUP_GATE_GATHER on vs off, on one built block in one process.

    The flag is a module global read inside attention forward(), so flipping it between passes swaps
    the gather path with nothing rebuilt. Separate processes would fold device-open and 22B weight-load
    variance into the delta; here the flag is the only thing that differs between passes.

    Each pass drops its own first lap: a flip selects op variants the program cache has not seen, so
    the lap right after it is a cold compile, not a steady-state forward. The third pass repeats the
    first pass's setting — if those two disagree by more than the delta, the device drifted under the
    measurement and the delta means nothing.
    """
    was_enabled = attention_ltx.LTX_DEDUP_GATE_GATHER

    def _pass(enabled: bool) -> float:
        attention_ltx.LTX_DEDUP_GATE_GATHER = enabled
        laps = []
        for _ in range(iters):
            t0 = time.perf_counter()
            tt_block(**forward_kwargs)
            ttnn.synchronize_device(mesh_device)
            laps.append((time.perf_counter() - t0) * 1000)
        warm = laps[1:]
        logger.info(
            f"DEDUP_PERF pass dedup={int(enabled)} dropped_lap={laps[0]:.2f} " f"warm={[f'{lap:.2f}' for lap in warm]}"
        )
        return sum(warm) / len(warm)

    try:
        on_1 = _pass(True)
        off = _pass(False)
        on_2 = _pass(True)
    finally:
        attention_ltx.LTX_DEDUP_GATE_GATHER = was_enabled

    on = (on_1 + on_2) / 2
    logger.info(
        f"DEDUP_PERF_MS on={on_1:.2f} off={off:.2f} on_repeat={on_2:.2f} "
        f"delta={on - off:+.2f} drift={abs(on_2 - on_1):.2f} warm_iters={iters - 1}"
    )


def _run_inner_step(
    *,
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    F,
    H,
    W,
    has_audio,
    run_pcc,
    checkpoint_variant: str,
    use_forward_alias: bool,
    fold_ab: bool = False,
    probe_ab: bool = False,
    dedup_ab: bool = False,
):
    """Shared body for the model forward tests; use_forward_alias picks tt_model(...) vs .forward(...)."""
    # Checkpoint variant only affects AV weight loading; skip the redundant copy in video mode.
    if not has_audio and checkpoint_variant != "fast":
        pytest.skip("checkpoint_variant only affects AV mode (video uses random scaled weights)")
    checkpoint_22b = _resolve_checkpoint_22b(checkpoint_variant)
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    # Real latent grid → SP-padded sequence (mirrors LTXPipeline). video_N_real is the logical
    # token count (SDPA logical_n / pad masks); video_N is the padded on-device length.
    video_N_real = F * H * W
    video_N = _sp_pad_len(video_N_real, sp_factor)
    audio_N = AUDIO_N
    assert video_N % (32 * sp_factor) == 0
    if has_audio:
        assert audio_N % (32 * sp_factor) == 0

    # === Reference model ===
    do_pcc = run_pcc
    ref_video = None
    ref_audio = None
    state_dict = None

    if has_audio:
        # AV PCC path: 22B with strict=False.
        state_dict = _load_22b_state_dict(num_layers=1, checkpoint_path=checkpoint_22b)
        if state_dict is None:
            pytest.skip(f"22B checkpoint not found at {checkpoint_22b}")
    else:
        # Video PCC path: random scaled weights (1 layer) from the diffusers 3D model.
        torch_model = _make_diffusers_video_model(num_layers=1)
        torch_model.eval()
        _scale_init_(torch_model)
        state_dict = _convert_diffusers_video_model_to_tt(
            torch_model.state_dict(), num_heads=NUM_HEADS, head_dim=HEAD_DIM
        )
        state_dict = {k: v.detach().clone() for k, v in state_dict.items()}

    # === Inputs ===
    # Real-grid latent for the torch reference; the TT side gets a zero-padded copy (video_N).
    torch.manual_seed(INPUT_SEED)
    video_lat_real = torch.randn(1, video_N_real, IN_CHANNELS, dtype=torch.float32)
    video_lat = _pad_seq_dim(video_lat_real, video_N, dim=1)
    video_prompt = torch.randn(1, PROMPT_LEN, CTX_DIM, dtype=torch.float32)
    sigma_val = TIMESTEP_VAL if not has_audio else 0.5  # AV path uses production sigma=0.5
    timestep_torch = torch.tensor([sigma_val])

    audio_lat = None
    audio_prompt = None
    if has_audio:
        audio_lat = torch.randn(1, audio_N, AUDIO_IN_CHANNELS, dtype=torch.float32)
        audio_prompt = torch.randn(1, PROMPT_LEN, AUDIO_CTX_DIM, dtype=torch.float32)

    # === Reference forward (before TT to avoid weight aliasing) ===
    # The torch LTXModel computes rope internally from positions; only the TT side builds freqs.
    if do_pcc and not has_audio:
        ref_video = _diffusers_video_model_ref(
            torch_model, video_lat=video_lat_real, video_prompt=video_prompt, sigma_val=sigma_val, F=F, H=H, W=W
        )
        logger.info(f"ref video {tuple(ref_video.shape)} range=[{ref_video.min():.3f}, {ref_video.max():.3f}]")
        del torch_model

    if do_pcc and has_audio:
        # AV reference: load the raw 22B into the diffusers AV model and run the cross-modal forward.
        ref_model = _make_diffusers_av_model(num_layers=1)
        ref_model.load_state_dict(_convert_lightricks_av_to_diffusers(state_dict, num_heads=NUM_HEADS), strict=False)
        ref_model.eval()
        t0 = time.time()
        ref_video, ref_audio = _diffusers_av_model_ref(
            ref_model,
            video_lat=video_lat_real,
            video_prompt=video_prompt,
            audio_lat=audio_lat,
            audio_prompt=audio_prompt,
            sigma_val=sigma_val,
            F=F,
            H=H,
            W=W,
            audio_N=audio_N,
        )
        logger.info(
            f"AV ref forward: {time.time() - t0:.1f}s — video {tuple(ref_video.shape)} audio {tuple(ref_audio.shape)}"
        )
        del ref_model
        import gc

        gc.collect()

    # === TT model ===
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    tt_model = _make_tt_model(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=has_audio,
        num_layers=1,
    )
    t0 = time.time()
    tt_model.load_torch_state_dict(state_dict, strict=not has_audio)
    logger.info(f"state-dict load: {time.time() - t0:.1f}s")

    # Quant is the one lever that genuinely changes the math, so it has to face the diffusers AV oracle
    # below (pcc 0.992) — a tighter bar than the block test's 0.988, and over the whole model wrapper.
    # The pipeline reads the same var through the same QuantConfig factory, so a preset that clears this
    # gate is precisely the one the pipeline runs.
    quant_preset = os.environ.get("LTX_QUANT", "").strip()
    if quant_preset:
        from models.tt_dit.pipelines.ltx.quant_config import QuantConfig, apply_quant_config

        factory = getattr(QuantConfig, quant_preset, None)
        assert callable(factory), f"LTX_QUANT='{quant_preset}' is not a QuantConfig preset"
        logger.info(f"LTX_QUANT='{quant_preset}': quantizing model for the AV PCC gate")
        apply_quant_config(tt_model, factory())

    # === Build TT-side RoPE / cross-PE / prompt tensors (INTERLEAVED + trans_mat) ===
    tt_video_prompt = bf16_tensor(video_prompt.unsqueeze(0), device=mesh_device)
    tt_vc, tt_vs = _tt_rope(
        _video_rope_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, pad_to=video_N
    )
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    call_kwargs = dict(
        video_1BNI_torch=video_lat.unsqueeze(0),
        video_prompt_1BLP=tt_video_prompt,
        video_rope_cos=tt_vc,
        video_rope_sin=tt_vs,
        video_N=video_N_real,
        trans_mat=tt_trans_mat,
        timestep_torch=timestep_torch,
    )
    if has_audio:
        a_cos, a_sin = _tt_rope(_audio_rope_freqs, audio_N, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis)
        vx_cos, vx_sin = _tt_rope(
            _video_cross_pe_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, pad_to=video_N
        )
        ax_cos, ax_sin = _tt_rope(
            _audio_cross_pe_freqs, audio_N, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis
        )
        ax_cos_full, ax_sin_full = _tt_rope_full(
            _audio_cross_pe_freqs, audio_N, mesh_device=mesh_device, tp_axis=tp_axis
        )
        # Zero padded video tokens as V→A cross-attention keys (matches build_video_pad_mask).
        v_pad_sp = build_video_pad_mask(video_N, video_N_real, mesh_device=mesh_device, sp_axis=sp_axis)
        call_kwargs.update(
            audio_1BNI_torch=audio_lat.unsqueeze(0),
            audio_prompt_1BLP=bf16_tensor(audio_prompt.unsqueeze(0), device=mesh_device),
            audio_rope_cos=a_cos,
            audio_rope_sin=a_sin,
            audio_N=audio_N,
            video_cross_pe_cos=vx_cos,
            video_cross_pe_sin=vx_sin,
            audio_cross_pe_cos=ax_cos,
            audio_cross_pe_sin=ax_sin,
            audio_cross_pe_cos_full=ax_cos_full,
            audio_cross_pe_sin_full=ax_sin_full,
            video_padding_mask=v_pad_sp,
        )

    # === Forward ===
    # The model is functional (no input mutation), so this may be re-invoked to compare code paths
    # on identical weights and inputs — see the fold A/B below.
    def _forward_to_host():
        t0 = time.time()
        result = tt_model(**call_kwargs) if use_forward_alias else tt_model.forward(**call_kwargs)
        logger.info(f"TT forward: {time.time() - t0:.1f}s")

        # Crop the SP-padding tail off the gathered video output before checking/comparing real tokens.
        if has_audio:
            tt_v_dev, tt_a_dev = result
            video = LTXTransformerModel.device_to_host(tt_v_dev).squeeze(0)[:, :video_N_real, :]
            audio = LTXTransformerModel.device_to_host(tt_a_dev).squeeze(0)
            assert video.shape == (1, video_N_real, OUT_CHANNELS), f"video shape {video.shape}"
            assert audio.shape == (1, audio_N, AUDIO_IN_CHANNELS), f"audio shape {audio.shape}"
            assert torch.isfinite(video).all() and torch.isfinite(audio).all(), "NaN/Inf in TT output"
            return video, audio

        video = LTXTransformerModel.device_to_host(result).squeeze(0)[:, :video_N_real, :]
        assert video.shape == (1, video_N_real, OUT_CHANNELS), f"video shape {video.shape}"
        assert torch.isfinite(video).all(), "NaN/Inf in TT output"
        return video, None

    tt_video, tt_audio = _forward_to_host()

    if fold_ab:
        _assert_fold_equivalence(_forward_to_host, fused_video=tt_video, fused_audio=tt_audio)

    if probe_ab:
        _assert_probe_split_magnitude(_forward_to_host, addcmul_video=tt_video, addcmul_audio=tt_audio)

    if dedup_ab:
        _assert_dedup_equivalence(_forward_to_host, dedup_video=tt_video, dedup_audio=tt_audio)

    del tt_model

    if do_pcc:
        assert_quality(ref_video, tt_video, pcc=0.992, relative_rmse=0.15)
        if has_audio:
            assert_quality(ref_audio, tt_audio, pcc=0.992, relative_rmse=0.15)
        logger.info("PASSED PCC")
    else:
        logger.info("PASSED (no PCC)")


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_TRANSFORMER_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("F", "H", "W"), _LTX_TRANSFORMER_SHAPE_PARAMS)
@pytest.mark.parametrize("has_audio", _LTX_TRANSFORMER_MODALITY_PARAMS)
@pytest.mark.parametrize("run_pcc", _LTX_TRANSFORMER_RUN_PCC_PARAMS)
@pytest.mark.parametrize("checkpoint_variant", _LTX_TRANSFORMER_CKPT_PARAMS)
def test_ltx_transformer_model(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    F,
    H,
    W,
    has_audio,
    run_pcc,
    checkpoint_variant,
    reset_seeds,
) -> None:
    """Full LTXTransformerModel forward via ``__call__``."""
    _run_inner_step(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        topology=topology,
        is_fsdp=is_fsdp,
        F=F,
        H=H,
        W=W,
        has_audio=has_audio,
        run_pcc=run_pcc,
        checkpoint_variant=checkpoint_variant,
        use_forward_alias=True,
    )


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_TRANSFORMER_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("F", "H", "W"), _LTX_TRANSFORMER_SHAPE_PARAMS)
@pytest.mark.parametrize("checkpoint_variant", _LTX_TRANSFORMER_CKPT_PARAMS)
def test_ltx_fold_gated_residual(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    F,
    H,
    W,
    checkpoint_variant,
    reset_seeds,
) -> None:
    """LTX_FOLD_GATED_RESIDUAL must not move the AV output: fused vs unfused, plus the diffusers oracle.

    AV-only, because the three residuals it folds (audio cross-attn, A→V, V→A) exist only there.
    """
    if not transformer_ltx.LTX_FOLD_GATED_RESIDUAL:
        pytest.skip("LTX_FOLD_GATED_RESIDUAL=0: the fused path this gates is compiled out")
    _run_inner_step(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        topology=topology,
        is_fsdp=is_fsdp,
        F=F,
        H=H,
        W=W,
        has_audio=True,
        run_pcc=True,
        checkpoint_variant=checkpoint_variant,
        use_forward_alias=True,
        fold_ab=True,
    )


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_TRANSFORMER_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("F", "H", "W"), _LTX_TRANSFORMER_SHAPE_PARAMS)
@pytest.mark.parametrize("checkpoint_variant", _LTX_TRANSFORMER_CKPT_PARAMS)
def test_ltx_probe_addcmul_split(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    F,
    H,
    W,
    checkpoint_variant,
    reset_seeds,
) -> None:
    """Size the LTX_PROBE_ADDCMUL_SPLIT control's per-layer perturbation against the fold's.

    The 48-layer A/B reads the fold's final-latent drift against this control's, and that only
    licenses a conclusion about the fold while the two kick the sampler comparably hard per layer.
    Nothing measured the control's kick; this does, on the same block, weights, seed and bounds the
    fold's own 1-layer gate uses, so the two numbers are directly comparable.

    AV-only and unfolded-only: the three residuals the probe perturbs exist only on the AV path,
    and the fold compiles out _gated_residual, the probe's only consumer.
    """
    if transformer_ltx.LTX_FOLD_GATED_RESIDUAL:
        pytest.skip("LTX_FOLD_GATED_RESIDUAL=1: the fold compiles out _gated_residual, the probe's only consumer")
    if transformer_ltx.LTX_PROBE_ADDCMUL_SPLIT:
        pytest.skip("LTX_PROBE_ADDCMUL_SPLIT=1: the plain-addcmul baseline this measures against is compiled out")
    _run_inner_step(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        topology=topology,
        is_fsdp=is_fsdp,
        F=F,
        H=H,
        W=W,
        has_audio=True,
        run_pcc=True,
        checkpoint_variant=checkpoint_variant,
        use_forward_alias=True,
        probe_ab=True,
    )


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_TRANSFORMER_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("F", "H", "W"), _LTX_TRANSFORMER_SHAPE_PARAMS)
@pytest.mark.parametrize("checkpoint_variant", _LTX_TRANSFORMER_CKPT_PARAMS)
def test_ltx_dedup_gate_gather(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    F,
    H,
    W,
    checkpoint_variant,
    reset_seeds,
) -> None:
    """LTX_DEDUP_GATE_GATHER must not move the AV output: one hoisted gather vs two fused ones.

    AV-only, because the per-head gate whose gather this dedups is only built with audio
    (apply_gated_attention=has_audio).
    """
    if not attention_ltx.LTX_DEDUP_GATE_GATHER:
        pytest.skip("LTX_DEDUP_GATE_GATHER=0: the hoisted-gather path this gates is compiled out")
    if topology == ttnn.Topology.Linear and tuple(mesh_device.shape)[tp_axis] > 1:
        pytest.skip("Linear topology already hoists the gather (use_nonfused_agmm): dedup is a no-op")
    _run_inner_step(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        topology=topology,
        is_fsdp=is_fsdp,
        F=F,
        H=H,
        W=W,
        has_audio=True,
        run_pcc=True,
        checkpoint_variant=checkpoint_variant,
        use_forward_alias=True,
        dedup_ab=True,
    )


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_TRANSFORMER_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("F", "H", "W"), _LTX_TRANSFORMER_SHAPE_PARAMS)
@pytest.mark.parametrize("has_audio", _LTX_TRANSFORMER_MODALITY_PARAMS)
@pytest.mark.parametrize("run_pcc", _LTX_TRANSFORMER_RUN_PCC_PARAMS)
@pytest.mark.parametrize("checkpoint_variant", _LTX_TRANSFORMER_CKPT_PARAMS)
def test_ltx_transformer_inner_step(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    F,
    H,
    W,
    has_audio,
    run_pcc,
    checkpoint_variant,
    reset_seeds,
) -> None:
    """LTXTransformerModel.forward — explicit call, denoising-loop path."""
    _run_inner_step(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        topology=topology,
        is_fsdp=is_fsdp,
        F=F,
        H=H,
        W=W,
        has_audio=has_audio,
        run_pcc=run_pcc,
        checkpoint_variant=checkpoint_variant,
        use_forward_alias=False,
    )


# ---------------------------------------------------------------------------
# Per-token video timestep (I2V) — equivalence to the scalar path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [pytest.param((2, 4), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="2x4sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("F", "H", "W"), [pytest.param(19, 17, 30, id="stage_1")])
def test_ltx_per_token_timestep_equivalence(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    F,
    H,
    W,
    reset_seeds,
) -> None:
    """A uniform per-token video timestep must reproduce the scalar-timestep path.

    This is the weight-agnostic invariant behind I2V: with ``image_conditioning=True`` and a
    per-token ``video_timestep = sigma`` for every token, the model output must match the
    ``image_conditioning=False`` scalar path (T2V/AV stays bit-similar). The frame-0 pinning
    behaviour itself is covered end-to-end by the pipeline; here we isolate the DiT plumbing.
    """
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    video_N_real = F * H * W
    video_N = _sp_pad_len(video_N_real, sp_factor)

    # Shared random-scaled weights (1 layer) from the diffusers 3D video model.
    torch_model = _make_diffusers_video_model(num_layers=1)
    torch_model.eval()
    _scale_init_(torch_model)
    state_dict = _convert_diffusers_video_model_to_tt(torch_model.state_dict(), num_heads=NUM_HEADS, head_dim=HEAD_DIM)
    state_dict = {k: v.detach().clone() for k, v in state_dict.items()}
    del torch_model

    torch.manual_seed(INPUT_SEED)
    video_lat_real = torch.randn(1, video_N_real, IN_CHANNELS, dtype=torch.float32)
    video_lat = _pad_seq_dim(video_lat_real, video_N, dim=1)
    video_prompt = torch.randn(1, PROMPT_LEN, CTX_DIM, dtype=torch.float32)
    sigma_val = TIMESTEP_VAL
    timestep_torch = torch.tensor([sigma_val])

    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)

    # Shared TT-side RoPE / prompt tensors.
    tt_video_prompt = bf16_tensor(video_prompt.unsqueeze(0), device=mesh_device)
    tt_vc, tt_vs = _tt_rope(
        _video_rope_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, pad_to=video_N
    )
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)
    base_kwargs = dict(
        video_1BNI_torch=video_lat.unsqueeze(0),
        video_prompt_1BLP=tt_video_prompt,
        video_rope_cos=tt_vc,
        video_rope_sin=tt_vs,
        video_N=video_N_real,
        trans_mat=tt_trans_mat,
        timestep_torch=timestep_torch,
    )

    # Scalar path (image_conditioning=False).
    scalar_model = _make_tt_model(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=False,
        num_layers=1,
        image_conditioning=False,
    )
    scalar_model.load_torch_state_dict(state_dict, strict=True)
    out_scalar = LTXTransformerModel.device_to_host(scalar_model.forward(**base_kwargs)).squeeze(0)[:, :video_N_real, :]
    del scalar_model

    # Per-token path (image_conditioning=True) with a uniform timestep = sigma over the padded grid.
    pertoken_model = _make_tt_model(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=False,
        num_layers=1,
        image_conditioning=True,
    )
    pertoken_model.load_torch_state_dict(state_dict, strict=True)
    video_timestep_torch = torch.full((video_N,), sigma_val, dtype=torch.float32)
    out_pertoken = LTXTransformerModel.device_to_host(
        pertoken_model.forward(video_timestep_torch=video_timestep_torch, **base_kwargs)
    ).squeeze(0)[:, :video_N_real, :]
    del pertoken_model

    assert out_pertoken.shape == out_scalar.shape, f"{out_pertoken.shape} vs {out_scalar.shape}"
    assert torch.isfinite(out_pertoken).all(), "NaN/Inf in per-token output"
    # Same weights + uniform sigma: the per-token MLP collapses to the scalar broadcast (bf16-equal).
    assert_quality(out_scalar, out_pertoken, pcc=0.999, relative_rmse=0.02)
    logger.info("PASSED: uniform per-token timestep reproduces the scalar path")


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [pytest.param((2, 4), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="2x4sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("F", "H", "W"), [pytest.param(19, 17, 30, id="stage_1")])
def test_ltx_per_token_timestep_nonuniform(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    is_fsdp,
    F,
    H,
    W,
    reset_seeds,
) -> None:
    """NON-uniform per-token timestep (the real I2V case) must match the diffusers per-token oracle.

    This is the case the existing equivalence test does NOT cover: frame-0 tokens at sigma=0
    (the pinned image anchor) and every other token at a high sigma. If the per-token gate does
    not reach the fused attn/FFN epilogues, non-frame-0 tokens get the wrong (frame-0) gate and
    the output diverges from the oracle — the DiT-level signature of "frame 0 ok, rest collapses".
    """
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    video_N_real = F * H * W
    video_N = _sp_pad_len(video_N_real, sp_factor)
    frame0_tokens = H * W  # token order is f*H*W + h*W + w, so frame 0 == first H*W tokens
    sigma_high = float(os.environ.get("LTX_TEST_SIGMA", "0.7"))

    # Shared random-scaled weights (1 layer) from the diffusers 3D video model.
    torch_model = _make_diffusers_video_model(num_layers=1)
    torch_model.eval()
    _scale_init_(torch_model)
    state_dict = _convert_diffusers_video_model_to_tt(torch_model.state_dict(), num_heads=NUM_HEADS, head_dim=HEAD_DIM)
    state_dict = {k: v.detach().clone() for k, v in state_dict.items()}

    torch.manual_seed(INPUT_SEED)
    video_lat_real = torch.randn(1, video_N_real, IN_CHANNELS, dtype=torch.float32)
    video_lat = _pad_seq_dim(video_lat_real, video_N, dim=1)
    video_prompt = torch.randn(1, PROMPT_LEN, CTX_DIM, dtype=torch.float32)

    # Per-token timestep: frame-0 tokens pinned at sigma=0, everything else (incl. SP padding) at sigma_high.
    video_ts_real = torch.full((video_N_real,), sigma_high, dtype=torch.float32)
    video_ts_real[:frame0_tokens] = 0.0
    video_timestep_torch = torch.full((video_N,), sigma_high, dtype=torch.float32)
    video_timestep_torch[:frame0_tokens] = 0.0

    # === Reference (per-token oracle) — before TT to avoid weight aliasing ===
    ref_video = _diffusers_video_model_ref_pertoken(
        torch_model,
        video_lat=video_lat_real,
        video_prompt=video_prompt,
        video_ts_real=video_ts_real,
        sigma_scalar=sigma_high,
        F=F,
        H=H,
        W=W,
    )
    del torch_model

    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    tt_video_prompt = bf16_tensor(video_prompt.unsqueeze(0), device=mesh_device)
    tt_vc, tt_vs = _tt_rope(
        _video_rope_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, pad_to=video_N
    )
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    model = _make_tt_model(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=False,
        num_layers=1,
        image_conditioning=True,
    )
    model.load_torch_state_dict(state_dict, strict=True)
    out_tt = LTXTransformerModel.device_to_host(
        model.forward(
            video_1BNI_torch=video_lat.unsqueeze(0),
            video_prompt_1BLP=tt_video_prompt,
            video_rope_cos=tt_vc,
            video_rope_sin=tt_vs,
            video_N=video_N_real,
            trans_mat=tt_trans_mat,
            timestep_torch=torch.tensor([sigma_high]),
            video_timestep_torch=video_timestep_torch,
        )
    ).squeeze(0)[:, :video_N_real, :]
    del model

    assert out_tt.shape == ref_video.shape, f"{out_tt.shape} vs {ref_video.shape}"
    assert torch.isfinite(out_tt).all(), "NaN/Inf in per-token output"

    # Localize frame-0 (pinned, sigma=0) vs the rest (sigma_high): a per-token gate that doesn't
    # reach the fused epilogues makes the "rest" track the oracle far worse than frame 0.
    def _pcc(a, b):
        a32, b32 = a.float().flatten(), b.float().flatten()
        return torch.corrcoef(torch.stack([a32, b32]))[0, 1].item()

    pcc_all = _pcc(ref_video, out_tt)
    pcc_f0 = _pcc(ref_video[:, :frame0_tokens, :], out_tt[:, :frame0_tokens, :])
    pcc_rest = _pcc(ref_video[:, frame0_tokens:, :], out_tt[:, frame0_tokens:, :])
    logger.info(f"non-uniform per-token (TT vs oracle): all={pcc_all:.5f} frame0={pcc_f0:.5f} rest={pcc_rest:.5f}")

    assert_quality(ref_video, out_tt, pcc=0.99, relative_rmse=0.03)
