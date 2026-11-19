# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""SigLIP2 vision tower and patch-merger projector on device, composed from tt_dit vision modules.

The blocks are tt_dit `Qwen3VlVisionBlock`s. SigLIP2 has no rotary embedding, so they run with an identity RoPE
(cos=1, sin=0): the rotation becomes a no-op, and the q/k channel permutation the block applies at load
(`rope_permute_qk`) is the same on q and k, so q.k is unchanged.
"""

from collections.abc import Sequence

import torch

import ttnn
from models.experimental.alpamayo2_edge.reference.config import ProjectorConfig, VisionConfig
from models.experimental.alpamayo2_edge.reference.vision import vision_position_embeddings
from models.tt_dit.encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionBlock
from models.tt_dit.encoders.vision import VisionPatchEmbed
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.layers.normalization import LayerNorm
from models.tt_dit.utils.tensor import bf16_tensor

_TILE = 32
_HIDDEN_ACT = "gelu_pytorch_tanh"
# Hard-coded in the upstream patch merger rather than read from the config.
_PROJECTOR_NORM_EPS = 1e-6
_POS_EMBED_KEY = "embeddings.position_embedding.weight"


def identity_rope(
    n: int, *, mesh_device: ttnn.MeshDevice, head_dim: int = VisionConfig.hidden // VisionConfig.heads
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """`(cos, sin)` = (ones, zeros), shaped like `vision_rope_tensors` for `n` rows: `(n, padded_head_dim)`."""
    padded_head_dim = _TILE * -(-head_dim // _TILE)
    cos = bf16_tensor(torch.ones(n, padded_head_dim), device=mesh_device)
    sin = bf16_tensor(torch.zeros(n, padded_head_dim), device=mesh_device)
    return cos, sin


class PosEmbedCache:
    """Device position embeddings per image-grid layout.

    The camera resolutions are fixed, so each layout is interpolated on the host and uploaded once.
    """

    def __init__(self, pos_weight: torch.Tensor, *, merge: int, mesh_device: ttnn.MeshDevice) -> None:
        self._pos_weight = pos_weight
        self._merge = merge
        self._mesh_device = mesh_device
        self._cache: dict[tuple, ttnn.Tensor] = {}

    def get(self, grid_thw: torch.Tensor) -> ttnn.Tensor:
        key = tuple(map(tuple, grid_thw.tolist()))
        if key not in self._cache:
            pos = vision_position_embeddings(self._pos_weight.float(), grid_thw, self._merge)
            self._cache[key] = bf16_tensor(pos, device=self._mesh_device)
        return self._cache[key]


def _merge_rows(x: ttnn.Tensor, width: int) -> ttnn.Tensor:
    """Fold consecutive rows into `width`-wide rows through row-major, so tile padding never enters a group."""
    rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.to_layout(ttnn.reshape(rm, (-1, width)), ttnn.TILE_LAYOUT)
    ttnn.deallocate(rm)
    return out


class TtProjector(Module):
    """Pre-shuffle LayerNorm, then each `merge**2` consecutive rows -> one token through fc1, exact GELU, fc2."""

    def __init__(self, vcfg: VisionConfig, pcfg: ProjectorConfig, *, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__()
        self.group = vcfg.merge**2
        self.merged_size = vcfg.hidden * self.group
        self.norm = LayerNorm(vcfg.hidden, norm_eps=_PROJECTOR_NORM_EPS, mesh_device=mesh_device)
        self.linear_fc1 = Linear(
            self.merged_size, pcfg.hidden, bias=True, activation_fn="gelu", mesh_device=mesh_device
        )
        self.linear_fc2 = Linear(pcfg.hidden, pcfg.out, bias=True, mesh_device=mesh_device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """`[N, hidden]` -> `[N // merge**2, out]`."""
        if x.shape[-2] % self.group != 0:
            raise ValueError(f"{x.shape[-2]} rows are not a whole number of {self.group}-row merge groups")
        x = _merge_rows(self.norm.forward(x), self.merged_size)
        return self.linear_fc2.forward(self.linear_fc1.forward(x))


class TtVisionTower(Module):
    """Patch embedding + position embedding, `vcfg.layers` SigLIP2 blocks, `post_layernorm` and the projector."""

    def __init__(self, vcfg: VisionConfig, pcfg: ProjectorConfig, *, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.head_dim = vcfg.hidden // vcfg.heads
        self.patch_embed = VisionPatchEmbed(
            in_channels=3,
            hidden_size=vcfg.hidden,
            patch_size=vcfg.patch,
            temporal_patch_size=1,
            bias=True,
            mesh_device=mesh_device,
        )
        self.blocks = ModuleList(
            Qwen3VlVisionBlock(
                hidden_size=vcfg.hidden,
                num_heads=vcfg.heads,
                intermediate_size=vcfg.mlp,
                hidden_act=_HIDDEN_ACT,
                norm_eps=vcfg.eps,
                mesh_device=mesh_device,
            )
            for _ in range(vcfg.layers)
        )
        self.post_layernorm = LayerNorm(vcfg.hidden, norm_eps=vcfg.eps, mesh_device=mesh_device)
        self.projector = TtProjector(vcfg, pcfg, mesh_device=mesh_device)
        self._identity_rope: dict[int, tuple[ttnn.Tensor, ttnn.Tensor]] = {}

    @staticmethod
    def convert_state(visual: dict, projector: dict) -> dict:
        """Checkpoint `model.visual.*` / `model.projector.*` tensors -> this module's keys.

        The position table is not a device parameter; it feeds `PosEmbedCache` on the host and is skipped here.
        """
        out = {f"projector.{k}": v for k, v in projector.items()}
        renames = {
            "layer_norm1.": "norm1.",
            "layer_norm2.": "norm2.",
            "self_attn.out_proj.": "attn.proj.",
            "mlp.fc1.": "mlp.linear_fc1.",
            "mlp.fc2.": "mlp.linear_fc2.",
        }
        for key, value in visual.items():
            if key == _POS_EMBED_KEY:
                continue
            if key.startswith("embeddings.patch_embedding."):
                out["patch_embed.proj." + key.removeprefix("embeddings.patch_embedding.")] = value
            elif key.startswith("post_layernorm."):
                out[key] = value
            elif key.startswith("encoder.layers."):
                index, rest = key.removeprefix("encoder.layers.").split(".", 1)
                if rest.startswith(("self_attn.q_proj.", "self_attn.k_proj.", "self_attn.v_proj.")):
                    continue
                for old, new in renames.items():
                    if rest.startswith(old):
                        out[f"blocks.{index}.{new}{rest.removeprefix(old)}"] = value
                        break
                else:
                    raise KeyError(f"unhandled vision checkpoint key {key!r}")
            else:
                raise KeyError(f"unhandled vision checkpoint key {key!r}")

        layers = {key.split(".")[2] for key in visual if key.startswith("encoder.layers.")}
        for index in layers:
            prefix = f"encoder.layers.{index}.self_attn."
            for kind in ("weight", "bias"):
                out[f"blocks.{index}.attn.qkv.{kind}"] = torch.cat(
                    [visual[f"{prefix}{proj}_proj.{kind}"] for proj in ("q", "k", "v")], dim=0
                )
        return out

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # A depth override builds fewer blocks than the checkpoint holds; the deeper blocks are not loaded.
        for key in [k for k in state if k.startswith("blocks.")]:
            if int(key.split(".")[1]) >= len(self.blocks):
                del state[key]

    def identity_rope(self, n: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if n not in self._identity_rope:
            self._identity_rope[n] = identity_rope(n, mesh_device=self.mesh_device, head_dim=self.head_dim)
        return self._identity_rope[n]

    def embed(self, patches: ttnn.Tensor, pos_embeds: ttnn.Tensor) -> ttnn.Tensor:
        """`[N, 768]` patches -> `[N, hidden]` block input."""
        return ttnn.add(self.patch_embed.forward(patches), pos_embeds)

    def forward(self, patches: ttnn.Tensor, pos_embeds: ttnn.Tensor, cu_seqlens: Sequence[int]) -> ttnn.Tensor:
        """`[N, 768]` patches and `[N, hidden]` position embeddings -> `[N // merge**2, out]` tokens.

        `cu_seqlens` bounds one image per attention window.
        """
        x = self.embed(patches, pos_embeds)
        rope = self.identity_rope(x.shape[-2])
        for block in self.blocks:
            x = block.forward(x, pos_embeds=rope, cu_seqlens=cu_seqlens)
        return self.projector.forward(self.post_layernorm.forward(x))
