# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Full MiniMax-M3-VL vision tower (+ optional projector).

Forward pipeline (matches `MiniMaxM3VLVisionModel.forward` then
`MiniMaxM3VLMultiModalProjector`):

    pixel_values (L, 1176) host
        -> patch_embed       (Conv3d-as-Linear 1176 -> 1280)          device
        -> pre_layrnorm
        -> encoder blocks, one per layer of the config (32 in the checkpoint)
              (LN1 -> attn(3D RoPE, full SDPA) -> residual; LN2 -> MLP -> residual)
        -> [tower output (L, 1280)]
        -> projector         (per-patch MLP -> merge 4 patches -> merge MLP)
        -> vision tokens (L // 4, 6144)

There is NO post-encoder LayerNorm and NO attention windowing: the tower
attends fully. The 3D-RoPE cos/sin depend only on `image_grid_thw`: they are
computed on host and pushed to DRAM with the first image of a grid, then
kept for the next images of that grid; each forward copies them to L1 once
for all blocks. from_state_dict builds the model from weights under
transformers 5.12's MiniMaxM3VLModel names, as loader.load_vision_state_dict
reads the checkpoint's and the unit tests draw seeded random ones.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.tensor_utils import nearest_32
from models.demos.minimax_m3.tt.vision.attention import M3VLAttention
from models.demos.minimax_m3.tt.vision.block import M3VLBlock
from models.demos.minimax_m3.tt.vision.common import to_device
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.image_input import MERGE_SIZE, check_image_grid, check_image_input
from models.demos.minimax_m3.tt.vision.layernorm import M3VLLayerNorm
from models.demos.minimax_m3.tt.vision.mlp import M3VLMLP
from models.demos.minimax_m3.tt.vision.patch_embed import M3VLPatchEmbed
from models.demos.minimax_m3.tt.vision.projector import M3VLProjector
from models.demos.minimax_m3.tt.vision.rope import rope_tensors

# The RoPE tables of the ROPE_CACHE_GRIDS (at least 1) image grids used last stay in DRAM. 16 is not tuned to any
# mix of image sizes. It is cheap: cos and sin of a 672x672 image, the largest that the image processor makes (2304
# patches), take 0.88 MB of DRAM, so 14 MB for 16 such grids.
ROPE_CACHE_GRIDS = 16

# Weight names: transformers 5.12's MiniMaxM3VLModel names, as loader.load_vision_state_dict gives them.
_TOWER = "vision_tower."
_PROJECTOR = "multi_modal_projector."
_LAYERS = _TOWER + "layers."
_PATCH_EMBED = _TOWER + "embeddings.proj.weight"
_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")


def _weight_shapes(config: MiniMaxM3VisionConfig, with_projector: bool) -> Dict[str, Tuple[int, ...]]:
    """{name: shape} of the weights that from_state_dict reads: the tower's patch embedding (a Conv3d without bias),
    pre_layrnorm and each layer's LayerNorms, attention projections and MLP, and with_projector the projector's two
    MLPs. Every other weight has a bias; the linears' weights are [out, in]."""
    c = config
    shapes = {_PATCH_EMBED: (c.hidden_size, c.num_channels, c.temporal_patch_size, c.patch_size, c.patch_size)}

    def add(name: str, *weight_shape: int):
        shapes[f"{name}.weight"] = weight_shape
        shapes[f"{name}.bias"] = weight_shape[:1]

    add(_TOWER + "pre_layrnorm", c.hidden_size)
    for i in range(c.num_hidden_layers):
        layer = f"{_LAYERS}{i}."
        add(layer + "layer_norm1", c.hidden_size)
        for projection in _ATTENTION_PROJECTIONS:
            add(f"{layer}self_attn.{projection}", c.hidden_size, c.hidden_size)
        add(layer + "layer_norm2", c.hidden_size)
        add(layer + "mlp.fc1", c.intermediate_size, c.hidden_size)
        add(layer + "mlp.fc2", c.hidden_size, c.intermediate_size)
    if with_projector:
        add(_PROJECTOR + "linear_1", c.projector_hidden_size, c.hidden_size)
        add(_PROJECTOR + "linear_2", c.text_hidden_size, c.projector_hidden_size)
        add(_PROJECTOR + "merge_linear_1", c.projector_hidden_size, c.merged_hidden_size)
        add(_PROJECTOR + "merge_linear_2", c.text_hidden_size, c.projector_hidden_size)
    return shapes


def _checked_weights(
    state_dict: Mapping[str, torch.Tensor], config: MiniMaxM3VisionConfig, with_projector: bool
) -> Dict[str, torch.Tensor]:
    """The weights of _weight_shapes(config, with_projector) from state_dict, each value read once, a patch embedding
    flattened to [hidden, patch_dim] in its Conv3d layout. No other value is read.

    Raises ValueError, before it reads a value, for a key under vision_tower., or with_projector under
    multi_modal_projector., that the model has no place for (the keys of the layers past num_hidden_layers are skipped)
    and for a weight that state_dict lacks; then for a weight of another shape than config's.
    """
    shapes = _weight_shapes(config, with_projector)
    keys = list(state_dict)  # the keys only: `in` on a Mapping reads the value
    vision = (_TOWER, _PROJECTOR) if with_projector else (_TOWER,)

    def skipped(key: str) -> bool:
        index = key[len(_LAYERS) :].split(".", 1)[0] if key.startswith(_LAYERS) else ""
        return index.isdigit() and int(index) >= config.num_hidden_layers

    unknown = sorted(key for key in keys if key.startswith(vision) and key not in shapes and not skipped(key))
    if unknown:
        raise ValueError(f"state_dict has vision weights that the model has no place for: {', '.join(unknown)}")
    missing = sorted(shapes.keys() - set(keys))
    if missing:
        raise ValueError(f"state_dict lacks weights that the model needs: {', '.join(missing)}")
    weights = {name: state_dict[name] for name in shapes}
    if tuple(weights[_PATCH_EMBED].shape) == (config.hidden_size, config.patch_dim):
        weights[_PATCH_EMBED] = weights[_PATCH_EMBED].reshape(shapes[_PATCH_EMBED])
    wrong = [
        f"{name} {tuple(value.shape)}, not {shapes[name]}"
        for name, value in weights.items()
        if tuple(value.shape) != shapes[name]
    ]
    if wrong:
        raise ValueError(f"state_dict has weights of other shapes than the config's: {', '.join(wrong)}")
    return weights


def _check_merge_size(config: MiniMaxM3VisionConfig) -> None:
    """Raises ValueError for a config whose spatial_merge_size is not image_input.MERGE_SIZE: the input checks and
    num_tokens count tokens with MERGE_SIZE, while the RoPE positions and the projector's fold use the config's."""
    if config.spatial_merge_size != MERGE_SIZE:
        raise ValueError(
            f"config spatial_merge_size must be {MERGE_SIZE}, as image_input counts tokens with it,"
            f" got {config.spatial_merge_size}"
        )


class M3VLVisionModel(LightweightModule):
    """Full M3-VL vision tower + optional multimodal projector.

    from_state_dict's dtype is the dtype of the weights on the device; only bfloat16 weights have run on a device. The
    pixel values go to the device as bf16 whatever it is, so the activations are bf16, the only dtype the RoPE op
    (rotary_embedding_llama) takes.
    """

    def __init__(
        self,
        mesh_device,
        config: MiniMaxM3VisionConfig,
        padded_head_dim: int,
        patch_embed: M3VLPatchEmbed,
        pre_layrnorm: M3VLLayerNorm,
        blocks: List[M3VLBlock],
        projector: Optional[M3VLProjector] = None,
    ):
        super().__init__()
        _check_merge_size(config)
        self.device = mesh_device
        self.config = config
        self.padded_head_dim = padded_head_dim
        self.patch_embed = patch_embed
        self.pre_layrnorm = pre_layrnorm
        self.blocks = blocks
        self.projector = projector
        self._rope_cache = OrderedDict()  # (t, h, w) -> (cos, sin) on device, the least recently used first

    @classmethod
    def from_state_dict(
        cls,
        mesh_device,
        state_dict: Mapping[str, torch.Tensor],
        config: MiniMaxM3VisionConfig,
        with_projector: bool = True,
        dtype=ttnn.bfloat16,
    ) -> "M3VLVisionModel":
        """The model of config with the weights of state_dict, under transformers 5.12's MiniMaxM3VLModel names
        (vision_tower.*, multi_modal_projector.*) as loader.load_vision_state_dict reads the checkpoint's, in any float
        dtype: each goes to the device as dtype, rounded to bf16 on the host first (common.to_device). It builds no
        torch module.

        state_dict may hold other weights, a language model's for example, and may be a lazy mapping: only the values of
        the model's weights are read, once each, so neither those of the layers past config.num_hidden_layers nor,
        without the projector, the projector's. Raises ValueError, before any upload, for a config whose
        spatial_merge_size is not image_input.MERGE_SIZE, for a key under vision_tower., or with the projector under
        multi_modal_projector., that the model has no place for, for a missing weight and for a weight of another shape
        than config's; a patch embedding flattened to [hidden, patch_dim] is taken as the Conv3d's.
        """
        _check_merge_size(config)
        weights = _checked_weights(state_dict, config, with_projector)
        # q, k and v heads padded to whole tiles on device (80 -> 96), in the weights and the RoPE tables.
        padded_head_dim = nearest_32(config.head_dim)

        def weight_bias(name: str) -> Tuple[torch.Tensor, torch.Tensor]:
            return weights[f"{name}.weight"], weights[f"{name}.bias"]

        def layer_norm(name: str) -> M3VLLayerNorm:
            return M3VLLayerNorm(
                mesh_device, config.hidden_size, *weight_bias(name), dtype=dtype, eps=config.layer_norm_eps
            )

        def mlp(fc1: str, fc2: str) -> M3VLMLP:
            return M3VLMLP(mesh_device, *weight_bias(fc1), *weight_bias(fc2), dtype=dtype)

        def block(layer: str) -> M3VLBlock:
            projections = [weight_bias(f"{layer}self_attn.{projection}") for projection in _ATTENTION_PROJECTIONS]
            return M3VLBlock(
                layer_norm1=layer_norm(layer + "layer_norm1"),
                attention=M3VLAttention.from_projections(
                    mesh_device, projections, config.head_dim, padded_head_dim, dtype=dtype
                ),
                layer_norm2=layer_norm(layer + "layer_norm2"),
                mlp=mlp(layer + "mlp.fc1", layer + "mlp.fc2"),
            )

        patch_embed = M3VLPatchEmbed(mesh_device, weights[_PATCH_EMBED], dtype=dtype)
        pre_layrnorm = layer_norm(_TOWER + "pre_layrnorm")
        blocks = [block(f"{_LAYERS}{i}.") for i in range(config.num_hidden_layers)]
        projector = None
        if with_projector:
            projector = M3VLProjector(
                mlp=mlp(_PROJECTOR + "linear_1", _PROJECTOR + "linear_2"),
                merge_mlp=mlp(_PROJECTOR + "merge_linear_1", _PROJECTOR + "merge_linear_2"),
                spatial_merge_size=config.spatial_merge_size,
            )
        return cls(
            mesh_device=mesh_device,
            config=config,
            padded_head_dim=padded_head_dim,
            patch_embed=patch_embed,
            pre_layrnorm=pre_layrnorm,
            blocks=blocks,
            projector=projector,
        )

    # ------------------------------------------------------------------
    def push_pixel_values(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Check one image's input and push its pixel values to device as forward_device takes them.

        Takes one image, pixel_values (L, 1176) and image_grid_thw (1, 3), and raises ValueError before any device work
        unless they pass check_image_input. On the device, rows past the grid's patches would get zero q and k from the
        RoPE op (rotary_embedding_llama) without an error, one NaN or Inf value would reach every patch through
        attention, and the patches of several images would attend to each other, as in Hugging Face 5.12.1; vLLM and
        SGLang keep attention inside each image. Returns the pixel values on device as bf16 (1, 1, L, 1176), whatever
        the weights' dtype.
        """
        check_image_input(pixel_values, image_grid_thw, self.config.patch_dim)
        return to_device(pixel_values.reshape(1, 1, *pixel_values.shape), self.device)

    def _stage_rope(self, image_grid_thw: torch.Tensor):
        """New cos and sin tables of the grid on device (rope_tensors), which the caller owns."""
        return rope_tensors(self.device, image_grid_thw, self.config, self.padded_head_dim)

    def _cached_rope(self, image_grid_thw: torch.Tensor):
        """The cos and sin tables of the grid, staged with its first image and kept for the next images of that grid.

        Holds the tables of the ROPE_CACHE_GRIDS grids used last and frees the two tables of the grid it drops. Only
        _tower_device uses them, to copy them to L1 for the blocks, so no caller holds a table that this frees.
        """
        key = tuple(image_grid_thw[0].tolist())
        if key in self._rope_cache:
            self._rope_cache.move_to_end(key)
            return self._rope_cache[key]
        if len(self._rope_cache) >= ROPE_CACHE_GRIDS:
            for table in self._rope_cache.popitem(last=False)[1]:
                ttnn.deallocate(table)
        self._rope_cache[key] = self._stage_rope(image_grid_thw)
        return self._rope_cache[key]

    def _rope_to_l1(self, cos: ttnn.Tensor, sin: ttnn.Tensor):
        """L1 copies of the cached cos and sin for one forward, which the caller frees.

        The RoPE op reads the cos and sin of each sequence tile again for each head, so the blocks read them from L1.
        The cache stays in DRAM: in L1 it would hold the tables of up to ROPE_CACHE_GRIDS grids at all times, in room
        that circular buffers need.
        """
        return tuple(ttnn.to_memory_config(table, ttnn.L1_MEMORY_CONFIG) for table in (cos, sin))

    # ------------------------------------------------------------------
    def forward_tower(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Run patch_embed -> pre_layrnorm -> the blocks on one image (push_pixel_values). Returns (1, 1, L, 1280)."""
        return self._tower_device(self.push_pixel_values(pixel_values, image_grid_thw), image_grid_thw)

    def _tower_device(self, x_tt: ttnn.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """forward_tower's device ops, on the pixel values that push_pixel_values gave, which stay allocated."""
        x_tt = self.patch_embed(x_tt)
        x_tt = self.pre_layrnorm(x_tt)
        cos_tt, sin_tt = self._rope_to_l1(*self._cached_rope(image_grid_thw))
        try:
            for blk in self.blocks:
                x_tt = blk(x_tt, cos_tt, sin_tt)
        finally:
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)
        return x_tt

    def forward_device(self, x_tt: ttnn.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Every device op of forward, on the pixel values that push_pixel_values gave for image_grid_thw. They stay
        allocated, so one upload serves several calls. With projector: (1, 1, L//4, 6144); else the tower out (1, 1, L,
        1280).

        Raises ValueError before any device work unless image_grid_thw passes check_image_grid and x_tt has the shape
        that push_pixel_values gives for it, (1, 1, h * w, patch_dim), which it reads from the tensor's host metadata.
        Pixel values pushed for another grid of the same patch count, such as 32 x 72 for 48 x 48 or 46 x 34 for
        34 x 46, pass and give wrong tokens without an error.
        """
        expected_shape = (1, 1, check_image_grid(image_grid_thw), self.config.patch_dim)
        if tuple(x_tt.shape) != expected_shape:
            raise ValueError(
                f"x_tt must be the pixel values that push_pixel_values gives for grid_thw {image_grid_thw.tolist()},"
                f" of shape {expected_shape}, got {tuple(x_tt.shape)}"
            )
        x_tt = self._tower_device(x_tt, image_grid_thw)
        if self.projector is None:
            return x_tt
        out_tt = self.projector(x_tt)
        ttnn.deallocate(x_tt)
        return out_tt

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Full forward of one image: push_pixel_values, then forward_device."""
        return self.forward_device(self.push_pixel_values(pixel_values, image_grid_thw), image_grid_thw)
