# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch FP32 reference of the Qwen3.8-Flash-Next vision tower (``model.visual.*``).

The tower turns one image's pixel patches into the merged visual features the language model splices in place of
its ``<|image_pad|>`` tokens: patch embedding (a Conv3d whose kernel equals its stride, i.e. one matmul over the
unfolded patches) plus a learned position table resampled to the image grid, 27 pre-LayerNorm blocks (full
non-causal attention with a two-axis rotary embedding, GELU-tanh MLP), and a merger (LayerNorm, 2x2 spatial group,
linear, GELU, linear) to ``out_hidden_size`` features, one per four patches.

Everything here is FP32 torch with no TTNN import; a development-only check tool (``vision_reference_check.py``) proves it
equal to the pinned Transformers ``Qwen4ExpVisionModel`` (and this module's preprocessing equal to its image processor)
on the same pixel input.  The device tower (``ttnn/vision.py``) is measured against this module block by block.  The functions
that fix the device side's contracts are the preprocessing (patch order), :func:`vision_position_ids`,
:func:`rotary_cos_sin`, :func:`interpolated_pos_embed` and :func:`attention_segments`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

VISION_TENSOR_PREFIX = "model.visual."
VISION_TENSOR_COUNT = 333
VISION_TENSOR_BYTES = 897_862_112


@dataclass(frozen=True)
class VisionTowerConfig:
    """The checkpoint's ``vision_config`` and ``preprocessor_config`` values the tower depends on (fail-closed)."""

    depth: int = 27
    hidden_size: int = 1152
    num_heads: int = 16
    intermediate_size: int = 4304
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2560
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    image_mean: float = 0.5
    image_std: float = 0.5
    min_pixels: int = 65_536  # preprocessor_config.json size.shortest_edge
    max_pixels: int = 16_777_216  # preprocessor_config.json size.longest_edge

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size

    @property
    def grid_side(self) -> int:
        return math.isqrt(self.num_position_embeddings)

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge_size**2

    @property
    def merged_hidden_size(self) -> int:
        return self.hidden_size * self.merge_unit

    @property
    def resize_factor(self) -> int:
        return self.patch_size * self.spatial_merge_size

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be a multiple of num_heads")
        if self.head_dim % 4:
            raise ValueError("head_dim must be a multiple of four (two rotary axes, rotate-half pairs)")
        if self.grid_side**2 != self.num_position_embeddings:
            raise ValueError("num_position_embeddings must be a square")

    @classmethod
    def from_checkpoint(cls, root: str | Path) -> "VisionTowerConfig":
        """Read ``config.json``'s ``vision_config`` and ``preprocessor_config.json``; refuse any unexpected value."""

        root = Path(root)
        document = json.loads((root / "config.json").read_text())
        vision = document.get("vision_config")
        if not isinstance(vision, dict):
            raise ValueError("config.json has no vision_config object")
        expected_fixed = {
            "model_type": "qwen4_exp",
            "hidden_act": "gelu_pytorch_tanh",
            "deepstack_visual_indexes": [],
        }
        for key, value in expected_fixed.items():
            if vision.get(key) != value:
                raise ValueError(f"vision_config.{key} must be {value!r}, got {vision.get(key)!r}")
        integer_keys = {
            "depth": "depth",
            "hidden_size": "hidden_size",
            "num_heads": "num_heads",
            "intermediate_size": "intermediate_size",
            "patch_size": "patch_size",
            "temporal_patch_size": "temporal_patch_size",
            "spatial_merge_size": "spatial_merge_size",
            "in_channels": "in_channels",
            "num_position_embeddings": "num_position_embeddings",
            "out_hidden_size": "out_hidden_size",
        }
        values: dict[str, Any] = {}
        for key, field in integer_keys.items():
            value = vision.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"vision_config.{key} must be a positive integer, got {value!r}")
            values[field] = value
        preprocessor = json.loads((root / "preprocessor_config.json").read_text())
        size = preprocessor.get("size")
        if not isinstance(size, dict):
            raise ValueError("preprocessor_config.json has no size object")
        for key, field in (("shortest_edge", "min_pixels"), ("longest_edge", "max_pixels")):
            value = size.get(key)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"preprocessor size.{key} must be a positive integer, got {value!r}")
            values[field] = value
        for key, expected in (
            ("patch_size", values["patch_size"]),
            ("temporal_patch_size", values["temporal_patch_size"]),
            ("merge_size", values["spatial_merge_size"]),
        ):
            if preprocessor.get(key) != expected:
                raise ValueError(f"preprocessor {key} {preprocessor.get(key)!r} differs from vision_config {expected}")
        mean = preprocessor.get("image_mean")
        std = preprocessor.get("image_std")
        if mean != [0.5, 0.5, 0.5] or std != [0.5, 0.5, 0.5]:
            raise ValueError(f"preprocessor image_mean/image_std must be 0.5 per channel, got {mean} / {std}")
        config = cls(**values)
        if config != cls():
            raise ValueError(f"vision configuration differs from the pinned release: {config}")
        return config


def vision_tensor_names(config: VisionTowerConfig = VisionTowerConfig()) -> tuple[str, ...]:
    """Every ``model.visual.*`` tensor of the checkpoint (333 names), sorted."""

    names = [
        f"{VISION_TENSOR_PREFIX}patch_embed.proj.weight",
        f"{VISION_TENSOR_PREFIX}patch_embed.proj.bias",
        f"{VISION_TENSOR_PREFIX}pos_embed.weight",
        f"{VISION_TENSOR_PREFIX}merger.norm.weight",
        f"{VISION_TENSOR_PREFIX}merger.norm.bias",
        f"{VISION_TENSOR_PREFIX}merger.linear_fc1.weight",
        f"{VISION_TENSOR_PREFIX}merger.linear_fc1.bias",
        f"{VISION_TENSOR_PREFIX}merger.linear_fc2.weight",
        f"{VISION_TENSOR_PREFIX}merger.linear_fc2.bias",
    ]
    for index in range(config.depth):
        for module in ("norm1", "norm2", "attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"):
            for parameter in ("weight", "bias"):
                names.append(f"{VISION_TENSOR_PREFIX}blocks.{index}.{module}.{parameter}")
    return tuple(sorted(names))


def vision_tensor_shapes(config: VisionTowerConfig = VisionTowerConfig()) -> dict[str, tuple[int, ...]]:
    """The checkpoint shape of every tower tensor, keyed without the ``model.visual.`` prefix."""

    hidden, inter, merged = config.hidden_size, config.intermediate_size, config.merged_hidden_size
    shapes: dict[str, tuple[int, ...]] = {
        "patch_embed.proj.weight": (
            hidden,
            config.in_channels,
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        ),
        "patch_embed.proj.bias": (hidden,),
        "pos_embed.weight": (config.num_position_embeddings, hidden),
        "merger.norm.weight": (hidden,),
        "merger.norm.bias": (hidden,),
        "merger.linear_fc1.weight": (merged, merged),
        "merger.linear_fc1.bias": (merged,),
        "merger.linear_fc2.weight": (config.out_hidden_size, merged),
        "merger.linear_fc2.bias": (config.out_hidden_size,),
    }
    for index in range(config.depth):
        prefix = f"blocks.{index}."
        shapes[prefix + "norm1.weight"] = (hidden,)
        shapes[prefix + "norm1.bias"] = (hidden,)
        shapes[prefix + "norm2.weight"] = (hidden,)
        shapes[prefix + "norm2.bias"] = (hidden,)
        shapes[prefix + "attn.qkv.weight"] = (3 * hidden, hidden)
        shapes[prefix + "attn.qkv.bias"] = (3 * hidden,)
        shapes[prefix + "attn.proj.weight"] = (hidden, hidden)
        shapes[prefix + "attn.proj.bias"] = (hidden,)
        shapes[prefix + "mlp.linear_fc1.weight"] = (inter, hidden)
        shapes[prefix + "mlp.linear_fc1.bias"] = (inter,)
        shapes[prefix + "mlp.linear_fc2.weight"] = (hidden, inter)
        shapes[prefix + "mlp.linear_fc2.bias"] = (hidden,)
    return shapes


# ----------------------------------------------------------------------------------------------------------------------
# Preprocessing: the image processor's resize rule, normalization and patch order.
# ----------------------------------------------------------------------------------------------------------------------


def smart_resize(height: int, width: int, *, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """The processor's target size: both sides multiples of ``factor``, area within the pixel bounds."""

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def normalize_image(image_uint8: torch.Tensor, config: VisionTowerConfig) -> torch.Tensor:
    """``[3, H, W]`` uint8 -> FP32 ``(x / 255 - mean) / std`` (the processor's rescale and normalize)."""

    if image_uint8.dtype != torch.uint8 or image_uint8.ndim != 3 or image_uint8.shape[0] != config.in_channels:
        raise ValueError(
            f"expected a [{config.in_channels}, H, W] uint8 image, got {image_uint8.dtype} {tuple(image_uint8.shape)}"
        )
    return (image_uint8.to(torch.float32) * (1.0 / 255.0) - config.image_mean) / config.image_std


def patchify(image: torch.Tensor, config: VisionTowerConfig) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """``[3, H, W]`` FP32 (H, W multiples of the resize factor) -> ``[N, patch_dim]`` patches in the processor's order.

    Patch i is the (block_row, block_col, in_row, in_col) raster over 2x2 merge blocks; its values are flattened
    (channel, temporal, patch_row, patch_col) with the single frame repeated over the temporal axis.  The second
    return value is ``grid_thw = (1, H / patch, W / patch)``.
    """

    channels, height, width = image.shape
    patch, merge, temporal = config.patch_size, config.spatial_merge_size, config.temporal_patch_size
    if channels != config.in_channels or height % config.resize_factor or width % config.resize_factor:
        raise ValueError(
            f"image [{channels}, {height}, {width}] is not [{config.in_channels}, k*{config.resize_factor}, k*{config.resize_factor}]"
        )
    grid_h, grid_w = height // patch, width // patch
    patches = image.reshape(channels, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    # [gh/m, gw/m, m, m, channel, patch, patch]
    patches = patches.permute(1, 4, 2, 5, 0, 3, 6)
    patches = patches.unsqueeze(5).expand(-1, -1, -1, -1, -1, temporal, -1, -1)
    return patches.reshape(grid_h * grid_w, config.patch_dim).contiguous(), (1, grid_h, grid_w)


def preprocess_image(image_uint8: torch.Tensor, config: VisionTowerConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 ``[3, H, W]`` -> (pixel patches ``[N, patch_dim]`` FP32, ``grid_thw`` ``[1, 3]`` int64).

    Images whose sides are already multiples of the resize factor inside the pixel bounds are patchified as they are
    (the test fixtures); other sizes are resized with bicubic antialiased interpolation as the processor does.
    """

    height, width = int(image_uint8.shape[-2]), int(image_uint8.shape[-1])
    target = smart_resize(
        height, width, factor=config.resize_factor, min_pixels=config.min_pixels, max_pixels=config.max_pixels
    )
    image = image_uint8
    if target != (height, width):
        resized = F.interpolate(
            image_uint8.unsqueeze(0).to(torch.float32), size=target, mode="bicubic", antialias=True, align_corners=False
        )
        image = resized.squeeze(0).round().clamp(0, 255).to(torch.uint8)
    patches, grid = patchify(normalize_image(image, config), config)
    return patches, torch.tensor([grid], dtype=torch.long)


# ----------------------------------------------------------------------------------------------------------------------
# Positions: rotary position ids, the learned table's resampling, attention segments.
# ----------------------------------------------------------------------------------------------------------------------


def _grid_rows(grid_thw: torch.Tensor) -> list[tuple[int, int, int]]:
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError(f"grid_thw must be [images, 3], got {tuple(grid_thw.shape)}")
    rows = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
    for t, h, w in rows:
        if t <= 0 or h <= 0 or w <= 0:
            raise ValueError(f"grid_thw entries must be positive, got {(t, h, w)}")
    return rows


def vision_position_ids(grid_thw: torch.Tensor, config: VisionTowerConfig) -> torch.Tensor:
    """``[N, 2]`` (row, col) patch positions in the processor's merge-block order, repeated over frames."""

    merge = config.spatial_merge_size
    pieces = []
    for t, h, w in _grid_rows(grid_thw):
        if h % merge or w % merge:
            raise ValueError(f"grid {(h, w)} is not a multiple of the merge size {merge}")
        rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        block = (h // merge, merge, w // merge, merge)
        rows = rows.reshape(block).transpose(1, 2).flatten()
        cols = cols.reshape(block).transpose(1, 2).flatten()
        pieces.append(torch.stack([rows, cols], dim=-1).repeat(t, 1))
    return torch.cat(pieces, dim=0)


def rotary_inv_freq(config: VisionTowerConfig) -> torch.Tensor:
    """``[head_dim / 4]`` inverse frequencies: ``Qwen4ExpVisionRotaryEmbedding(head_dim // 2)``."""

    dim = config.head_dim // 2
    return 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def rotary_cos_sin(position_ids: torch.Tensor, config: VisionTowerConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """``[N, head_dim]`` cos and sin in the rotate-half layout: ``cat(freqs(row), freqs(col))`` duplicated."""

    inv_freq = rotary_inv_freq(config)
    freqs = (position_ids.to(torch.float32).unsqueeze(-1) * inv_freq).flatten(1)  # [N, head_dim / 2]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def pos_embed_interpolation(grid_thw: torch.Tensor, config: VisionTowerConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear (align_corners=True) resampling of the square learned table to each grid, in the processor's
    patch order: ``[N, 4]`` table row indices and ``[N, 4]`` weights per patch."""

    side = config.grid_side
    indices, weights = [], []
    for t, h, w in _grid_rows(grid_thw):
        positions = vision_position_ids(torch.tensor([[1, h, w]]), config)
        rows, cols = positions[:, 0].to(torch.float32), positions[:, 1].to(torch.float32)
        row_src = rows * (side - 1) / max(h - 1, 1)
        col_src = cols * (side - 1) / max(w - 1, 1)
        row_floor, col_floor = torch.floor(row_src), torch.floor(col_src)
        offsets = torch.arange(0, 2, dtype=torch.float32)
        row_taps = (row_floor[:, None] + offsets).clamp(0, side - 1).long()
        col_taps = (col_floor[:, None] + offsets).clamp(0, side - 1).long()
        row_weights = (1 - (row_src[:, None] - row_floor[:, None] - offsets).abs()).clamp(min=0)
        col_weights = (1 - (col_src[:, None] - col_floor[:, None] - offsets).abs()).clamp(min=0)
        index = (row_taps[:, :, None] * side + col_taps[:, None, :]).reshape(-1, 4)
        weight = (row_weights[:, :, None] * col_weights[:, None, :]).reshape(-1, 4)
        indices.append(index.repeat(t, 1))
        weights.append(weight.repeat(t, 1))
    return torch.cat(indices, dim=0), torch.cat(weights, dim=0)


def interpolated_pos_embed(table: torch.Tensor, grid_thw: torch.Tensor, config: VisionTowerConfig) -> torch.Tensor:
    """``[N, hidden]`` FP32 position embedding for the grid from the ``[num_positions, hidden]`` table."""

    indices, weights = pos_embed_interpolation(grid_thw, config)
    return (table.to(torch.float32)[indices] * weights[:, :, None]).sum(1)


def attention_segments(grid_thw: torch.Tensor) -> tuple[tuple[int, int], ...]:
    """Half-open patch ranges that attend among themselves: one per frame (``h * w`` patches each)."""

    segments, start = [], 0
    for t, h, w in _grid_rows(grid_thw):
        for _ in range(t):
            segments.append((start, start + h * w))
            start += h * w
    return tuple(segments)


def merged_token_count(grid_thw: torch.Tensor, config: VisionTowerConfig) -> int:
    """The number of feature rows (= ``<|image_pad|>`` tokens) the grid produces."""

    return sum(t * h * w for t, h, w in _grid_rows(grid_thw)) // config.merge_unit


# ----------------------------------------------------------------------------------------------------------------------
# The tower.
# ----------------------------------------------------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``[N, heads, head_dim]`` with ``[N, head_dim]`` cos / sin (broadcast over heads), FP32."""

    return x * cos.unsqueeze(-2) + _rotate_half(x) * sin.unsqueeze(-2)


@dataclass
class VisionTowerTrace:
    """Every stage of one forward, for the block-by-block device comparison."""

    embed: torch.Tensor  # [N, hidden] after patch embed + position embedding
    blocks: list[torch.Tensor]  # [N, hidden] after each block
    features: torch.Tensor  # [N / 4, out_hidden]


class VisionTowerReference:
    """FP32 tower over a ``{name: tensor}`` state dict keyed without the ``model.visual.`` prefix."""

    def __init__(self, state_dict: Mapping[str, torch.Tensor], config: VisionTowerConfig = VisionTowerConfig()):
        self.config = config
        expected = vision_tensor_shapes(config)
        names = set(state_dict)
        if names != set(expected):
            missing, extra = sorted(set(expected) - names)[:4], sorted(names - set(expected))[:4]
            raise ValueError(f"vision state dict names differ: missing {missing} extra {extra}")
        self.weights: dict[str, torch.Tensor] = {}
        for name, shape in expected.items():
            tensor = state_dict[name]
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {tuple(tensor.shape)}")
            self.weights[name] = tensor.detach().to(torch.float32).contiguous()
        self.patch_matrix = self.weights["patch_embed.proj.weight"].reshape(config.hidden_size, config.patch_dim)

    @classmethod
    def from_checkpoint(cls, checkpoint) -> "VisionTowerReference":
        """From ``Qwen38Checkpoint`` (its ``vision_state_dict`` reads the 333 BF16 tensors from shard 1)."""

        return cls(checkpoint.vision_state_dict(), VisionTowerConfig.from_checkpoint(checkpoint.root))

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return F.linear(x, self.weights[name + ".weight"], self.weights[name + ".bias"])

    def _norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return F.layer_norm(
            x, (x.shape[-1],), self.weights[name + ".weight"], self.weights[name + ".bias"], self.config.layer_norm_eps
        )

    def block_norm(self, index: int, position: int, hidden: torch.Tensor) -> torch.Tensor:
        """Block ``index``'s first (``position`` 1, before attention) or second (2, before the MLP) LayerNorm."""

        if position not in (1, 2):
            raise ValueError("position is 1 (norm1) or 2 (norm2)")
        return self._norm(hidden, f"blocks.{index}.norm{position}")

    def patch_embed(self, pixel_patches: torch.Tensor) -> torch.Tensor:
        """``[N, patch_dim]`` -> ``[N, hidden]``: the Conv3d with kernel == stride as one linear."""

        if pixel_patches.ndim != 2 or pixel_patches.shape[1] != self.config.patch_dim:
            raise ValueError(f"pixel patches must be [N, {self.config.patch_dim}], got {tuple(pixel_patches.shape)}")
        return F.linear(pixel_patches.to(torch.float32), self.patch_matrix, self.weights["patch_embed.proj.bias"])

    def embed(self, pixel_patches: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Patch embedding plus the resampled position table: the input of block 0."""

        hidden = self.patch_embed(pixel_patches)
        if hidden.shape[0] != sum(t * h * w for t, h, w in _grid_rows(grid_thw)):
            raise ValueError(f"{hidden.shape[0]} patches do not match grid_thw {grid_thw.tolist()}")
        return hidden + interpolated_pos_embed(self.weights["pos_embed.weight"], grid_thw, self.config)

    def positions(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]:
        cos, sin = rotary_cos_sin(vision_position_ids(grid_thw, self.config), self.config)
        return cos, sin, attention_segments(grid_thw)

    def attention(
        self,
        index: int,
        normed: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        segments: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        """Block ``index``'s attention on the LayerNorm output: full attention inside each segment."""

        config = self.config
        prefix = f"blocks.{index}.attn."
        n = normed.shape[0]
        qkv = self._linear(normed, prefix + "qkv").reshape(n, 3, config.num_heads, config.head_dim)
        q, k, v = qkv.unbind(1)  # [N, heads, head_dim]
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        q, k, v = (x.transpose(0, 1) for x in (q, k, v))  # [heads, N, head_dim]
        output = torch.empty_like(q)
        scale = config.head_dim**-0.5
        for start, end in segments:
            scores = torch.matmul(q[:, start:end], k[:, start:end].transpose(-1, -2)) * scale
            output[:, start:end] = torch.matmul(torch.softmax(scores, dim=-1), v[:, start:end])
        return self._linear(output.transpose(0, 1).reshape(n, config.hidden_size), prefix + "proj")

    def mlp(self, index: int, normed: torch.Tensor) -> torch.Tensor:
        prefix = f"blocks.{index}.mlp."
        return self._linear(
            F.gelu(self._linear(normed, prefix + "linear_fc1"), approximate="tanh"), prefix + "linear_fc2"
        )

    def block(
        self,
        index: int,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        segments: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        prefix = f"blocks.{index}."
        hidden = hidden + self.attention(index, self._norm(hidden, prefix + "norm1"), cos, sin, segments)
        return hidden + self.mlp(index, self._norm(hidden, prefix + "norm2"))

    def merger(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[N, hidden]`` -> ``[N / 4, out_hidden]``: LayerNorm, 2x2 group (four consecutive patches), fc1, GELU, fc2."""

        config = self.config
        if hidden.shape[0] % config.merge_unit:
            raise ValueError(f"{hidden.shape[0]} patches are not a multiple of the merge unit {config.merge_unit}")
        grouped = self._norm(hidden, "merger.norm").reshape(-1, config.merged_hidden_size)
        return self._linear(F.gelu(self._linear(grouped, "merger.linear_fc1")), "merger.linear_fc2")

    def forward(self, pixel_patches: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        return self.forward_trace(pixel_patches, grid_thw).features

    def forward_trace(self, pixel_patches: torch.Tensor, grid_thw: torch.Tensor) -> VisionTowerTrace:
        with torch.no_grad():
            embed = self.embed(pixel_patches, grid_thw)
            cos, sin, segments = self.positions(grid_thw)
            hidden, blocks = embed, []
            for index in range(self.config.depth):
                hidden = self.block(index, hidden, cos, sin, segments)
                blocks.append(hidden)
            return VisionTowerTrace(embed, blocks, self.merger(hidden))
