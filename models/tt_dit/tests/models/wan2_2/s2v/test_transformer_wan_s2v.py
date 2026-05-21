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
# Additional torch reference modules for the inner_step integration test.
# Ports of WanS2VAttentionBlock + Head_S2V + audio injector cross-attn.
# ---------------------------------------------------------------------------


class _TorchWanS2VAttentionBlock(_TorchWanAttentionBlock):
    """Port of upstream ``WanS2VAttentionBlock`` (s2v/model_s2v.py:184-244).

    Differs from the base block by supporting **segmented timestep
    modulation**: ``e`` is a 2-tuple ``(e_full, seg_idx)`` where ``e_full``
    has shape ``[B, 6, 2, dim]`` (two segments) and ``seg_idx`` splits
    ``x`` along the sequence axis. Each segment gets its own modulation.
    With ``seg_idx == x.size(1)`` (single segment), reduces to t2v.
    """

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        e: tuple[torch.Tensor, int],
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        e_full, seg_idx_raw = e
        seg_idx = min(max(0, int(seg_idx_raw)), x.size(1))
        seg_idx = [0, seg_idx, x.size(1)]
        # modulation: [1, 6, dim] + [B, 6, 2, dim] → 6 chunks each [B, 1, 2, dim]
        e_mod = (self.modulation.unsqueeze(2) + e_full.float()).chunk(6, dim=1)
        e_mod = [chunk.squeeze(1) for chunk in e_mod]  # 6 × [B, 2, dim]

        # Self-attn with per-segment scale+shift.
        norm_x = self.norm1(x).float()
        parts = [
            norm_x[:, seg_idx[i] : seg_idx[i + 1]] * (1 + e_mod[1][:, i : i + 1]) + e_mod[0][:, i : i + 1]
            for i in range(2)
        ]
        norm_x = torch.cat(parts, dim=1)
        y = self.self_attn(norm_x, grid_sizes, freqs)
        z = [y[:, seg_idx[i] : seg_idx[i + 1]] * e_mod[2][:, i : i + 1] for i in range(2)]
        x = x + torch.cat(z, dim=1)

        # Cross-attn (no modulation).
        x = x + self.cross_attn(self.norm3(x), context)

        # FFN with per-segment scale+shift.
        norm2_x = self.norm2(x).float()
        parts = [
            norm2_x[:, seg_idx[i] : seg_idx[i + 1]] * (1 + e_mod[4][:, i : i + 1]) + e_mod[3][:, i : i + 1]
            for i in range(2)
        ]
        norm2_x = torch.cat(parts, dim=1)
        y = self.ffn(norm2_x)
        z = [y[:, seg_idx[i] : seg_idx[i + 1]] * e_mod[5][:, i : i + 1] for i in range(2)]
        x = x + torch.cat(z, dim=1)
        return x


class _TorchHead_S2V(nn.Module):
    """Port of upstream ``Head`` / ``Head_S2V`` (model.py:262-291)."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6) -> None:
        super().__init__()
        out_dim = patch_size[0] * patch_size[1] * patch_size[2] * out_dim
        self.norm = _TorchWanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # e: [B, dim] from time_embedding output (full timestep, not chunked).
        e_full = self.modulation.unsqueeze(0) + e.float().unsqueeze(2)  # [1, 1, 2, dim] + [B, 1, 1, dim] → broadcast
        e_mod = e_full.chunk(2, dim=2)
        return self.head(self.norm(x) * (1 + e_mod[1].squeeze(2)) + e_mod[0].squeeze(2))


class _TorchAudioInjectorLayer(nn.Module):
    """One layer's audio cross-attention injector. Mirrors the reference's
    ``injector_pre_norm_feat[id] + injector[id]`` pair (LayerNorm + WanCrossAttention).
    """

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.pre_norm = _TorchWanLayerNorm(dim, eps)
        self.attn = _TorchWanCrossAttention(dim, num_heads, qk_norm, eps)

    def forward(self, x: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        """Per-frame cross-attention. ``x``: [B, T*N, dim] spatial noisy
        tokens, rearranged outside as ``[B*T, N, dim]``. ``audio_emb``:
        ``[B*T, audio_tokens, dim]``.
        """
        return self.attn(self.pre_norm(x), audio_emb)


# ---------------------------------------------------------------------------
# Inner-step integration test — TODO follow-on.
#
# The torch reference modules above (_TorchWanS2VAttentionBlock,
# _TorchHead_S2V, _TorchAudioInjectorLayer) cover the s2v-specific math
# needed for a parity test against ``WanS2VTransformer3DModel.inner_step``.
#
# What's still missing to actually wire up the test:
#
#   1. Populating the tt model's per-clip caches without calling
#      ``prepare_cond_emb`` (which expects host inputs and runs the full
#      pre-processing chain). Either:
#         (a) Call ``prepare_cond_emb`` with synthesized noisy/ref/motion
#             latents and let it populate caches normally. Then port
#             ``cond_encoder`` + ``trainable_cond_mask`` + segmented mod
#             mask construction on the torch side to match.
#         (b) Reach into the tt model and manually set the ``_cached_*``
#             attributes from pre-computed tensors. Bypasses prepare_cond_emb
#             but couples the test to internal tt state.
#
#   2. Equivalent torch reference for the cond_encoder (``WanPatchEmbed``
#      for pose video) + the trainable_cond_mask add + segmented timestep
#      modulation table construction (real vs zero timestep for noisy vs
#      const tokens).
#
#   3. Audio injection — feed pre-computed ``merged_audio_emb`` to the tt
#      model's ``self.audio_injector`` state and call its forward with the
#      same audio tokens on the torch side.
#
#   4. Block-diagonal frame mask construction for the tt side (which uses
#      a single flat K/V + mask) vs the torch reference's per-frame
#      rearrange (already proven equivalent by
#      ``test_audio_injector_block_diagonal_vs_per_frame.py``).
#
# Realistic effort estimate: ~6-10 hours of focused implementation +
# debugging. Tracked as a separate session-of-work — not iterable in
# back-and-forth chat turns.
# ---------------------------------------------------------------------------
