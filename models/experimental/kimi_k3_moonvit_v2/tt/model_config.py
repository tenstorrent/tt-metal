# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Configuration for the MoonViT-V2 encoder, patch merger and projector.

`MoonViTConfig` holds the architecture; its defaults are the Kimi-K3 `vision_config`, and
`from_hf_config` reads the same fields from a checkpoint's `config.json`. `VisionParallel` is the
mesh layout the TT modules receive, resolved from a sequence-parallel `ParallelFactor` and the
`TT_CCL` that owns the ring resources (`ccl.py`). This module imports no ttnn at import time, so the CPU reference can use it.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import torch

    import ttnn
    from .ccl import ParallelFactor, TT_CCL


@dataclass(frozen=True)
class MoonViTConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 27
    num_attention_heads: int = 12
    qkv_hidden_size: int = 1536
    in_channels: int = 3
    patch_size: int = 14
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    init_pos_emb_time: int = 4
    pos_emb_interpolation_mode: str = "bilinear"
    attn_bias: bool = False
    linear_bias: bool = False
    patch_embed_proj_bias: bool = False
    rope_theta: float = 10000.0
    # Upstream norms use nn.RMSNorm(eps=None), i.e. finfo(bf16).eps for bf16 activations.
    norm_eps: float = 2.0**-7
    # Processor caps: `patch_limit_on_one_side` (also the RoPE table size) and `in_patch_limit`.
    # Every image is padded to `max_patches` rows on device; ring attention masks the padding.
    max_side_patches: int = 512
    max_patches: int = 65536
    # Patch merger (`sd2_tpool`): each 2x2 window of patches becomes one token.
    merge_kernel_size: tuple[int, int] = (2, 2)
    # Projector (`patchmergerv2`): merged token -> language-model width, then RMSNorm.
    text_hidden_size: int = 7168
    projector_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.qkv_hidden_size // self.num_attention_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch_size * self.patch_size

    @property
    def merge_area(self) -> int:
        return self.merge_kernel_size[0] * self.merge_kernel_size[1]

    @property
    def merged_dim(self) -> int:
        """Width of a merged token: the window's patches concatenated."""
        return self.hidden_size * self.merge_area

    def with_layers(self, num_hidden_layers: int) -> MoonViTConfig:
        return dataclasses.replace(self, num_hidden_layers=num_hidden_layers)

    @classmethod
    def from_hf_config(cls, config: dict | str | Path) -> MoonViTConfig:
        """Read the `vision_config` block of a Kimi-K3 `config.json` (a dict, the file, or its folder)."""
        if not isinstance(config, dict):
            path = Path(config)
            if path.is_dir():
                path = path / "config.json"
            config = json.loads(path.read_text())
        vc = config.get("vision_config", config)
        unsupported = {
            k: vc[k]
            for k, want in (
                ("norm_type", "rmsnorm"),
                ("mlp_type", "mlp2"),
                ("pos_emb_type", "divided_fixed"),
                ("activation_func", "gelu_pytorch_tanh"),
                ("merge_type", "sd2_tpool"),
                ("mm_projector_type", "patchmergerv2"),
                ("projector_hidden_act", "gelu"),
            )
            if k in vc and vc[k] != want
        }
        if unsupported:
            raise ValueError(f"vision_config differs from MoonViT-V2 in {unsupported}")
        if vc.get("mm_hidden_size", vc["vt_hidden_size"]) != vc["vt_hidden_size"]:
            raise ValueError(f"mm_hidden_size {vc['mm_hidden_size']} != vt_hidden_size {vc['vt_hidden_size']}")
        return cls(
            hidden_size=vc["vt_hidden_size"],
            intermediate_size=vc["vt_intermediate_size"],
            num_hidden_layers=vc["vt_num_hidden_layers"],
            num_attention_heads=vc["vt_num_attention_heads"],
            qkv_hidden_size=vc.get("qkv_hidden_size") or vc["vt_hidden_size"],
            patch_size=vc["patch_size"],
            init_pos_emb_height=vc["init_pos_emb_height"],
            init_pos_emb_width=vc["init_pos_emb_width"],
            init_pos_emb_time=vc["init_pos_emb_time"],
            pos_emb_interpolation_mode=vc.get("pos_emb_interpolation_mode", "bilinear"),
            attn_bias=vc.get("attn_bias", True),
            linear_bias=vc.get("linear_bias", True),
            patch_embed_proj_bias=vc.get("patch_embed_proj_bias", True),
            merge_kernel_size=tuple(vc["merge_kernel_size"]),
            text_hidden_size=vc["text_hidden_size"],
            projector_norm_eps=vc["projector_ln_eps"],
        )


def check_grid(grid_thws: torch.Tensor, config: MoonViTConfig) -> tuple[int, int]:
    """Validate a `grid_thws` of one still image and return its (h, w) in patches.

    One image per call; no packing and no video. The processor pads images to a multiple of
    `merge_kernel_size * patch_size` (28 px), so h and w are multiples of the merge kernel.
    """
    if tuple(grid_thws.shape) != (1, 3):
        raise ValueError(f"one image per call, got grid_thws {grid_thws.tolist()}")
    t, h, w = grid_thws[0].tolist()
    if t != 1:
        raise ValueError(f"video is not supported, got t={t}")
    kh, kw = config.merge_kernel_size
    if h % kh or w % kw:
        raise ValueError(f"grid {h}x{w} is not a multiple of the {kh}x{kw} merge kernel")
    if h > config.max_side_patches or w > config.max_side_patches or h * w > config.max_patches:
        raise ValueError(f"grid {h}x{w} exceeds {config.max_side_patches} per side or {config.max_patches} patches")
    return h, w


class VisionParallel(NamedTuple):
    """Sequence parallelism only: weights are replicated, the sequence is split along `sp_axis`."""

    sp_axis: int | None
    sp_factor: int
    ccl: TT_CCL | None

    @property
    def sp(self) -> bool:
        return self.sp_factor > 1


def resolve_parallel(
    mesh_device: ttnn.MeshDevice,
    sequence_parallel: ParallelFactor | None,
    ccl: TT_CCL | None,
    config: MoonViTConfig,
) -> VisionParallel:
    sp = sequence_parallel
    if sp is None or sp.factor == 1:
        return VisionParallel(sp_axis=None, sp_factor=1, ccl=ccl)
    if ccl is None:
        raise ValueError("sequence parallelism needs a TT_CCL")
    if tuple(mesh_device.shape)[sp.mesh_axis] != sp.factor:
        raise ValueError(f"sp factor {sp.factor} != mesh axis {sp.mesh_axis} size {tuple(mesh_device.shape)}")
    if config.max_patches % (sp.factor * config.merge_area * 32):
        raise ValueError(
            f"max_patches={config.max_patches} does not split into tiles over {sp.factor} devices after merging"
        )
    return VisionParallel(sp_axis=sp.mesh_axis, sp_factor=sp.factor, ccl=ccl)


def compute_kernel_config(mesh_device: ttnn.MeshDevice):
    """HiFi4 with fp32 accumulation for every matmul: highest fidelity, no quantization."""
    import ttnn

    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
