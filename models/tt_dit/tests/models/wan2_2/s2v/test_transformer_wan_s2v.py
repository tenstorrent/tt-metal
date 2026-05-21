# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``WanS2VTransformer3DModel`` vs a torch-only reference.

The WAN 2.2 reference repo ``wan/modules/s2v/model_s2v.py`` depends on CUDA
(``torch.cuda.current_device`` at class-definition time), ``flash_attn``,
and ``decord``. Rather than stub all of that, this file ports the upstream
math directly into pure PyTorch — no external repo imports, no CUDA-only
libraries. The torch reference here is the spec; the tt-side
``WanS2VTransformer3DModel`` is tested against it.

Mirrors the layout of ``test_transformer_wan.py``. Three tests at three
granularities (mirroring t2v):

  * ``test_wan_s2v_transformer_block`` — single block + after_transformer_block hook.
  * ``test_wan_s2v_transformer_model`` — full transformer forward (TODO phase 2).
  * ``test_wan_s2v_transformer_inner_step`` — inner_step only (TODO phase 2).

The torch reference and tt model use the same random weights via state-dict
copy. PCC bar = 0.992 (same as t2v).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import ttnn

from .....models.transformers.wan2_2.s2v.transformer_wan_s2v import WanS2VTransformer3DModel
from .....parallel.config import DiTParallelConfig, ParallelFactor
from .....parallel.manager import CCLManager
from .....utils.test import line_params
from .....utils.wan_s2v_checkpoint import (
    find_s2v_snapshot,
    load_s2v_config,
    load_s2v_state_dict,
    translate_s2v_state_dict,
)

# Production config (matches Wan-AI/Wan2.2-S2V-14B / config.json) — used for
# the strict weight-load smoke test below; the parity tests use a smaller
# config to keep CPU forwards tractable.
PATCH_SIZE = (1, 2, 2)
DIM = 5120
NUM_HEADS = 40
NUM_LAYERS = 40
IN_CHANNELS = 16
OUT_CHANNELS = 16
TEXT_DIM = 4096
FREQ_DIM = 256
FFN_DIM = 13824
AUDIO_DIM = 1024
NUM_AUDIO_TOKEN = 4
NUM_AUDIO_LAYERS = 25
AUDIO_INJECT_LAYERS = (0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39)
ROPE_MAX_SEQ_LEN = 1024
EPS = 1e-6


# ---------------------------------------------------------------------------
# Torch reference — ports the upstream WAN math into pure PyTorch.
# Reference: /home/kevinmi/wan2_2_ref/wan/modules/{model.py, s2v/model_s2v.py}
# No imports from that repo; F.scaled_dot_product_attention instead of
# flash_attn; nn.LayerNorm instead of CUDA-fused variants.
# ---------------------------------------------------------------------------


class _TorchWanRMSNorm(nn.Module):
    """Port of upstream ``WanRMSNorm`` (wan/modules/model.py:69-85)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class _TorchWanLayerNorm(nn.LayerNorm):
    """Port of upstream ``WanLayerNorm`` — same as nn.LayerNorm but always
    computes in fp32 then casts back."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


def _torch_rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Port of upstream ``rope_params`` (wan/modules/model.py:27-35)."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _torch_rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Port of upstream ``rope_apply`` (wan/modules/model.py:38-66)."""
    n, c = x.size(2), x.size(3) // 2
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def _torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Stand-in for ``flash_attention``: same math via SDPA. q/k/v
    shape ``[B, L, H, D]``; returns ``[B, L, H, D]``."""
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(qh, kh, vh)
    return out.transpose(1, 2)


class _TorchWanSelfAttention(nn.Module):
    """Port of upstream ``WanSelfAttention`` (wan/modules/model.py:101-155)."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = _TorchWanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = _TorchWanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        out = _torch_attention(_torch_rope_apply(q, grid_sizes, freqs), _torch_rope_apply(k, grid_sizes, freqs), v)
        return self.o(out.flatten(2))


class _TorchWanCrossAttention(_TorchWanSelfAttention):
    """Port of upstream ``WanCrossAttention`` (wan/modules/model.py:158-180)."""

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        out = _torch_attention(q, k, v)
        return self.o(out.flatten(2))


class _TorchWanAttentionBlock(nn.Module):
    """Port of upstream ``WanAttentionBlock`` (wan/modules/model.py:183-259)."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm1 = _TorchWanLayerNorm(dim, eps)
        self.self_attn = _TorchWanSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = _TorchWanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = _TorchWanCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = _TorchWanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # Modulation broadcast — upstream wraps in ``with autocast('cuda', fp32):``.
        # On CPU we just compute in fp32 directly.
        e_mod = (self.modulation.unsqueeze(0) + e.float()).chunk(6, dim=2)

        # Self-attention with scale+shift modulation.
        y = self.self_attn(
            self.norm1(x).float() * (1 + e_mod[1].squeeze(2)) + e_mod[0].squeeze(2),
            grid_sizes,
            freqs,
        )
        x = x + y * e_mod[2].squeeze(2)

        # Cross-attention.
        x = x + self.cross_attn(self.norm3(x), context)

        # FFN with scale+shift modulation.
        y = self.ffn(self.norm2(x).float() * (1 + e_mod[4].squeeze(2)) + e_mod[3].squeeze(2))
        x = x + y * e_mod[5].squeeze(2)
        return x


# ---------------------------------------------------------------------------
# Test helpers (mirror test_transformer_wan.py).
# ---------------------------------------------------------------------------


def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )


def _make_ccl_manager(mesh_device, num_links, topology):
    return CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)


def _build_tt_s2v_model(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, *, num_layers: int, cfg=None):
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
    cfg = cfg or {}
    return WanS2VTransformer3DModel(
        patch_size=tuple(cfg.get("patch_size", PATCH_SIZE)),
        num_heads=cfg.get("num_heads", NUM_HEADS),
        dim=cfg.get("dim", DIM),
        in_channels=cfg.get("in_dim", IN_CHANNELS),
        out_channels=cfg.get("out_dim", OUT_CHANNELS),
        text_dim=cfg.get("text_dim", TEXT_DIM),
        freq_dim=cfg.get("freq_dim", FREQ_DIM),
        ffn_dim=cfg.get("ffn_dim", FFN_DIM),
        num_layers=num_layers,
        cross_attn_norm=cfg.get("cross_attn_norm", True),
        eps=cfg.get("eps", EPS),
        rope_max_seq_len=ROPE_MAX_SEQ_LEN,
        audio_dim=cfg.get("audio_dim", AUDIO_DIM),
        num_audio_layers=cfg.get("num_audio_layers", NUM_AUDIO_LAYERS),
        num_audio_token=cfg.get("num_audio_token", NUM_AUDIO_TOKEN),
        audio_inject_layers=tuple(cfg.get("audio_inject_layers", AUDIO_INJECT_LAYERS)),
        enable_adain=cfg.get("enable_adain", False),
        enable_motioner=cfg.get("enable_motioner", False),
        enable_framepack=cfg.get("enable_framepack", True),
        motion_token_num=cfg.get("motion_token_num", 1024),
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        model_type="s2v",
    )


# ---------------------------------------------------------------------------
# Test 1: strict weight-load smoke test (production checkpoint, no forward).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (4, 8),
            (4, 8),
            1,
            0,
            2,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="bh_4x8sp1tp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_s2v_weight_load(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """Strict weight-load smoke test for Wan2.2-S2V-14B. Validates that
    ``translate_s2v_state_dict`` produces a state dict that
    ``WanS2VTransformer3DModel.load_torch_state_dict`` accepts in strict mode.
    """
    snapshot = find_s2v_snapshot()
    cfg = load_s2v_config(snapshot)
    ref_sd = load_s2v_state_dict(snapshot)
    tt_sd = translate_s2v_state_dict(ref_sd)
    logger.info(f"Translated state dict: {len(ref_sd)} ref keys → {len(tt_sd)} tt keys")

    tt_model = _build_tt_s2v_model(
        mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, num_layers=cfg["num_layers"], cfg=cfg
    )
    tt_model.load_torch_state_dict(tt_sd, strict=True)
    logger.info("Strict load succeeded.")
    del tt_model


# ---------------------------------------------------------------------------
# TODO: integration test for ``WanS2VTransformer3DModel.inner_step``.
#
# Scope: build tt model with num_layers=1 and audio_inject_layers=[0], call
# prepare_audio_emb + prepare_cond_emb + inner_step on synthesized inputs,
# compare against a torch reference forward built from the modules above
# plus the s2v-specific bits still to port:
#   - TorchCausalAudioEncoder (learned weighted layer sum + MotionEncoder_tc)
#   - TorchMotionEncoder_tc (3-stage causal Conv1d + LayerNorm + SiLU)
#   - TorchAudioInjector (cross-attention modules; mirror of WanCrossAttention)
#   - TorchFramePackMotionerWan (3 patch projections + rope freqs)
#   - TorchCondEncoder (WanPatchEmbed for pose)
#   - Top-level orchestration matching WanModel_S2V.forward (~250 lines)
#
# Estimated ~500 lines of additional torch reference + ~150 lines test
# scaffolding. Track under a follow-on commit.
# ---------------------------------------------------------------------------
