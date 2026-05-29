# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the full dots.ocr DotsVisionTransformer (vision tower).

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`vision_tower_forward`

DotsVisionTransformer.forward:

    patch_embed -> N x vision_block -> [post_trunk_norm] -> patch_merger

This is the FINAL vision-side assembly. It does NOT re-implement any block maths --
it imports and composes the already-verified modules:

    TtVisionBlock        (x N, pre-norm residual)  -- tt/vision_block.py
    TtVisionRMSNorm      (post_trunk norm, eps 1e-5) -- tt/vision_rmsnorm.py
    TtVisionPatchMerger  (LayerNorm + GELU MLP, merge 2x2) -- tt/vision_patch_merger.py

Host-resident boundary (patch_embed):
    DotsPatchEmbed is a single Conv2d(3,1536,k=14,s=14) over packed/patchified
    pixels followed by an RMSNorm. Per the qwen25_vl TTNN demos, the Conv2d
    patchify (a one-shot im2col + matmul) runs on the host and the resulting
    patch tokens are uploaded to the device. So this tower's DEVICE input is the
    post-patch-embed hidden states [num_patches, embed_dim]; the patch_embed is
    computed on the host via the verified reference (see ``host_patch_embed``)
    and is the only documented host step. Everything from there on -- 2D RoPE,
    the N transformer blocks, the post-trunk norm, and the patch merger -- runs
    on device with ttnn ops (no host-side matmul / softmax / activation in the
    device path).

2D vision RoPE (theta 1e4) and the block-diagonal cu_seqlens mask are precomputed
on host at construction time (exactly as the standalone block did) and threaded
into every TtVisionBlock; they are position/grid metadata, not learned activations.

embed_dim 1536, num_heads 12, head_dim 128, spatial_merge_size 2,
rms_norm_eps 1e-5 (RMSNorm blocks + post_trunk), ln_eps 1e-6 (merger LayerNorm),
use_bias False.

The model dir name (rednote_hilab_dots.ocr) contains a dot, so the sibling
modules cannot be imported via the normal dotted package path -- they are loaded
by file path with importlib (the same convention the tests use).

Reference TTNN impl this follows: models/demos/qwen25_vl/tt/model.py
"""
import importlib.util
import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

_TT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(module_name: str, file_name: str, symbol: str):
    """Import a sibling module by file path (dir name has a dot)."""
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_TT_DIR, file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, symbol)


TtVisionBlock = _load_sibling("dots_tt_vision_block", "vision_block.py", "TtVisionBlock")
TtVisionRMSNorm = _load_sibling("dots_tt_vision_rmsnorm", "vision_rmsnorm.py", "TtVisionRMSNorm")
TtVisionPatchMerger = _load_sibling("dots_tt_vision_patch_merger", "vision_patch_merger.py", "TtVisionPatchMerger")


def _vision_rot_pos_emb(grid_thw, spatial_merge_size: int, head_dim: int, theta: float = 1e4) -> torch.Tensor:
    """Build the 2D (h,w) vision RoPE freqs table for the patch grid.

    Mirrors DotsVisionTransformer.rot_pos_emb (host-side position metadata, not a
    learned activation): per grid build (h,w) pos ids, gather from a
    VisionRotaryEmbedding(head_dim//2) table, flatten -> [seq, head_dim//2].
    """
    pos_ids = []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = int(grid_thw[:, 1:].max())

    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    seq = torch.arange(max_grid_size, dtype=torch.float)
    rotary_pos_emb_full = torch.outer(seq, inv_freq)  # [max_grid_size, head_dim//4]
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)  # [seq, head_dim//2]
    return rotary_pos_emb


def _cu_seqlens(grid_thw) -> torch.Tensor:
    """Cumulative seqlens (block-diagonal attention mask), padded with a leading 0."""
    cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(0, dtype=torch.int32)
    cu = torch.nn.functional.pad(cu, (1, 0), value=0)
    return cu


def host_patch_embed(
    pixel_values: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    num_channels: int = 3,
    temporal_patch_size: int = 1,
    patch_size: int = 14,
    embed_dim: int = 1536,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Host-resident patch embed (the only documented host step): Conv2d patchify +
    RMSNorm, producing the [num_patches, embed_dim] device input.

    This mirrors reference ``vision_patch_embed_forward`` (Conv2d(3,1536,k=14,s=14)
    over packed pixels, then RMSNorm). Per the qwen25_vl pattern the patchify runs
    on host once per image; its output tokens are uploaded to the device tower.
    """
    x = pixel_values.view(-1, num_channels, temporal_patch_size, patch_size, patch_size)[:, :, 0]
    x = torch.nn.functional.conv2d(
        x.to(torch.float32),
        proj_weight.to(torch.float32),
        proj_bias.to(torch.float32),
        stride=(patch_size, patch_size),
    ).view(-1, embed_dim)
    # RMSNorm (fp32) matching vision_rmsnorm_forward.
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * norm_weight.to(torch.float32)
    return x


class TtVisionTower(LightweightModule):
    """dots.ocr DotsVisionTransformer (full vision tower assembly).

    Composes N verified TtVisionBlock + a post-trunk TtVisionRMSNorm + a
    TtVisionPatchMerger. The patch_embed Conv2d+RMSNorm is run on the host
    (the documented host-resident boundary); call :meth:`patch_embed` (or the
    module-level :func:`host_patch_embed`) to produce the device input.

    The 2D vision RoPE table (theta 1e4) and the cu_seqlens block-diagonal mask
    are derived from ``grid_thw`` on host and threaded into every block.

    Args:
        device: ttnn Device or MeshDevice.
        state_dict: flat reference state_dict with keys
            'blocks.{i}.norm1.weight', 'blocks.{i}.attn.qkv.weight',
            'blocks.{i}.attn.proj.weight', 'blocks.{i}.norm2.weight',
            'blocks.{i}.mlp.fc{1,2,3}.weight', 'post_trunk_norm.weight',
            'merger.ln_q.weight/bias', 'merger.mlp.0.weight/bias',
            'merger.mlp.2.weight/bias', 'patch_embed.proj.weight/bias',
            'patch_embed.norm.weight'.
        grid_thw: torch.Tensor [num_images, 3] (t, h, w) patch grids.
        num_layers: number of transformer blocks (the golden runs reduced, e.g. 2).
        embed_dim: 1536.
        num_heads: 12.
        spatial_merge_size: 2.
        rms_norm_eps: RMSNorm epsilon (1e-5) for blocks + post_trunk norm.
        ln_eps: LayerNorm epsilon (1e-6) for the merger.
        post_norm: apply the post-trunk RMSNorm (True).
        rope_theta: vision RoPE base (1e4).
        dtype: activation/weight dtype (bf16).
    """

    def __init__(
        self,
        device,
        state_dict,
        grid_thw,
        num_layers: int,
        embed_dim: int = 1536,
        num_heads: int = 12,
        num_channels: int = 3,
        temporal_patch_size: int = 1,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        rms_norm_eps: float = 1e-5,
        ln_eps: float = 1e-6,
        post_norm: bool = True,
        rope_theta: float = 1e4,
        dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.rms_norm_eps = rms_norm_eps
        self.post_norm = post_norm
        head_dim = embed_dim // num_heads

        # Keep the patch_embed weights on host (host-resident boundary).
        self._pe_proj_weight = state_dict["patch_embed.proj.weight"].to(torch.float32)
        self._pe_proj_bias = state_dict["patch_embed.proj.bias"].to(torch.float32)
        self._pe_norm_weight = state_dict["patch_embed.norm.weight"].to(torch.float32)

        # 2D RoPE + cu_seqlens for this grid (host-side position/grid metadata).
        grid_thw = torch.as_tensor(grid_thw)
        rotary_pos_emb = _vision_rot_pos_emb(grid_thw, spatial_merge_size, head_dim, theta=rope_theta)
        cu_seqlens = _cu_seqlens(grid_thw)
        seq_length = int(grid_thw[:, 0].mul(grid_thw[:, 1]).mul(grid_thw[:, 2]).sum())

        self.blocks = [
            TtVisionBlock(
                device=device,
                norm1_weight=state_dict[f"blocks.{i}.norm1.weight"].to(torch.float32),
                qkv_weight=state_dict[f"blocks.{i}.attn.qkv.weight"].to(torch.float32),
                proj_weight=state_dict[f"blocks.{i}.attn.proj.weight"].to(torch.float32),
                norm2_weight=state_dict[f"blocks.{i}.norm2.weight"].to(torch.float32),
                fc1_weight=state_dict[f"blocks.{i}.mlp.fc1.weight"].to(torch.float32),
                fc3_weight=state_dict[f"blocks.{i}.mlp.fc3.weight"].to(torch.float32),
                fc2_weight=state_dict[f"blocks.{i}.mlp.fc2.weight"].to(torch.float32),
                rotary_pos_emb=rotary_pos_emb,
                cu_seqlens=cu_seqlens,
                seq_length=seq_length,
                num_heads=num_heads,
                head_dim=head_dim,
                eps=rms_norm_eps,
                dtype=dtype,
                weight_memory_config=weight_memory_config,
            )
            for i in range(num_layers)
        ]

        if post_norm:
            self.post_trunk_norm = TtVisionRMSNorm(
                device=device,
                dim=embed_dim,
                weight=state_dict["post_trunk_norm.weight"].to(torch.float32),
                eps=rms_norm_eps,
                weight_dtype=dtype,
                weight_memory_config=weight_memory_config,
            )

        self.merger = TtVisionPatchMerger(
            device=device,
            ln_weight=state_dict["merger.ln_q.weight"].to(torch.float32),
            ln_bias=state_dict["merger.ln_q.bias"].to(torch.float32),
            fc1_weight=state_dict["merger.mlp.0.weight"].to(torch.float32),
            fc1_bias=state_dict["merger.mlp.0.bias"].to(torch.float32),
            fc2_weight=state_dict["merger.mlp.2.weight"].to(torch.float32),
            fc2_bias=state_dict["merger.mlp.2.bias"].to(torch.float32),
            context_dim=embed_dim,
            spatial_merge_size=spatial_merge_size,
            ln_eps=ln_eps,
            dtype=dtype,
            weight_memory_config=weight_memory_config,
        )

    def patch_embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Host-resident patch embed -> [num_patches, embed_dim] (torch fp32)."""
        return host_patch_embed(
            pixel_values,
            self._pe_proj_weight,
            self._pe_proj_bias,
            self._pe_norm_weight,
            num_channels=self.num_channels,
            temporal_patch_size=self.temporal_patch_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            eps=self.rms_norm_eps,
        )

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """hidden_states: [num_patches, embed_dim] (TILE layout, post patch_embed)
        -> merged visual tokens [num_patches // merge**2, hidden_size].
        """
        for block in self.blocks:
            hidden_states = block(hidden_states)

        if self.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.merger(hidden_states)
        return hidden_states
