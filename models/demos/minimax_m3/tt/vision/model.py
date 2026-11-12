# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3 vision tower and projector on the chip.

It follows Hugging Face's MiniMaxM3VLVisionModel and MiniMaxM3VLMultiModalProjector:

    pixel_values (L, 1176), on the host
        -> patch_embed (1176 -> 1280)
        -> pre_layrnorm
        -> one block per layer: LayerNorm, attention with 3D RoPE, residual; LayerNorm, MLP, residual
        -> tower output (L, 1280)
        -> projector: MLP per patch, merge each 2 x 2 block of patches, MLP
        -> vision tokens (L / 4, 6144)

There is no final LayerNorm and no attention window. The RoPE tables depend only on the image grid, so the model
keeps them for the next images of that grid. The blocks keep their activations in L1 where common.block_placement
proves they fit. The tower's output is in DRAM.
"""
from __future__ import annotations

import re
import traceback
from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Tuple

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.tensor_utils import nearest_32
from models.demos.minimax_m3.tt.vision.attention import M3VLAttention
from models.demos.minimax_m3.tt.vision.block import M3VLBlock
from models.demos.minimax_m3.tt.vision.common import (
    DRAM_PLACEMENT,
    BlockPlacement,
    block_placement,
    l1_free_for_activations,
    to_device,
)
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.image_input import MERGE_SIZE, check_image_grid, check_image_input
from models.demos.minimax_m3.tt.vision.layernorm import M3VLLayerNorm
from models.demos.minimax_m3.tt.vision.mlp import M3VLMLP
from models.demos.minimax_m3.tt.vision.patch_embed import M3VLPatchEmbed
from models.demos.minimax_m3.tt.vision.projector import M3VLProjector
from models.demos.minimax_m3.tt.vision.rope import rope_tensors

# How many recent image grids keep their RoPE tables in DRAM, at least 1. Not tuned, as the tables are small.
ROPE_CACHE_GRIDS = 16
# tt-metal's error for an op whose circular buffers would overlap a tensor in L1 (ProgramImpl's check of the circular
# buffer region). It reaches Python as a RuntimeError with this line in its message. The groups leave out the program
# id, which is new for each program, so a clash that lasts gives the same reason for every image.
L1_CLASH = re.compile(r"(Statically allocated circular buffers) in program \d+ (clash with L1 buffers[^\n]*)")
# Every DRAM_FALLBACK_WARN_EVERY-th fallback to DRAM_PLACEMENT is a warning, even with the last warning's reason.
DRAM_FALLBACK_WARN_EVERY = 100

# Weight names: transformers 5.12's MiniMaxM3VLModel names, as loader.load_vision_state_dict gives them.
_TOWER = "vision_tower."
_PROJECTOR = "multi_modal_projector."
_LAYERS = _TOWER + "layers."
_PATCH_EMBED = _TOWER + "embeddings.proj.weight"
_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")


def _weight_shapes(config: MiniMaxM3VisionConfig, with_projector: bool) -> Dict[str, Tuple[int, ...]]:
    """Return {name: shape} of every weight that from_state_dict reads.

    The projector's weights are included only with_projector. Linear weights are [out, in]. Every layer but the patch
    embedding, a Conv3d, also has a bias.
    """
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
    """Pick the model's weights out of state_dict and check them.

    Reads each of their values once, and no other value. A patch embedding flattened to [hidden, patch_dim] is
    reshaped to the Conv3d's shape. The layers past num_hidden_layers are skipped.

    Raises ValueError for a missing weight or an unknown vision key, before reading any value. Raises it for a weight
    of the wrong shape after reading. A vision key is one under vision_tower., or under multi_modal_projector.
    with_projector.
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
    """Raise ValueError unless config.spatial_merge_size is image_input.MERGE_SIZE.

    The input checks count tokens with MERGE_SIZE, while the RoPE positions and the projector use the config's.
    """
    if config.spatial_merge_size != MERGE_SIZE:
        raise ValueError(
            f"config spatial_merge_size must be {MERGE_SIZE}, as image_input counts tokens with it,"
            f" got {config.spatial_merge_size}"
        )


class M3VLVisionModel(LightweightModule):
    """MiniMax-M3 vision tower on the chip, with or without the projector.

    Build it with from_state_dict. Activations are bf16 whatever the weights' dtype, as the RoPE op takes only bf16.
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
        # Where the blocks keep their activations, one of common.ACTIVATIONS, read at each forward. "l1" is the
        # default; "l1_interleaved" and "dram" are for comparison.
        self.activations = "l1"
        self._dram_fallbacks = 0  # fallbacks to DRAM_PLACEMENT so far
        self._warned_fallback = None  # the reason of the last fallback logged as a warning

    @classmethod
    def from_state_dict(
        cls,
        mesh_device,
        state_dict: Mapping[str, torch.Tensor],
        config: MiniMaxM3VisionConfig,
        with_projector: bool = True,
        dtype=ttnn.bfloat16,
    ) -> "M3VLVisionModel":
        """Build the model and upload the weights of state_dict, without building a torch module.

        Args:
            state_dict: weights under transformers 5.12's names (vision_tower.*, multi_modal_projector.*), as
                loader.load_vision_state_dict returns them, in any float dtype. The patch embedding may also be
                flattened to [hidden, patch_dim]. Only the model's weights are read, once each, so this may be a lazy
                mapping over a whole checkpoint. Other keys are ignored, such as a language model's, those of layers
                past config.num_hidden_layers and, without the projector, multi_modal_projector.*.
            with_projector: also build the projector.
            dtype: the weights' dtype on the chip. Each weight is rounded to bf16 on the host first.

        Raises:
            ValueError: before any upload, for a spatial_merge_size other than image_input.MERGE_SIZE, a vision
                weight that the model has no place for, a missing weight or a weight of the wrong shape.
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
        """Check one image's input and upload its pixel values for forward_device.

        pixel_values: (L, 1176) float patch rows. image_grid_thw: the (1, 3) grid [[1, h, w]], with L = h * w.
        Returns bf16 (1, 1, L, 1176) on the chip, whatever the weights' dtype.
        Raises ValueError before any device work unless the input passes check_image_input. On the chip, such input
        would give wrong tokens without an error.
        """
        check_image_input(pixel_values, image_grid_thw, self.config.patch_dim)
        return to_device(pixel_values.reshape(1, 1, *pixel_values.shape), self.device)

    def _stage_rope(self, image_grid_thw: torch.Tensor):
        """Upload new cos and sin tables for the grid (rope_tensors), which the caller owns."""
        return rope_tensors(self.device, image_grid_thw, self.config, self.padded_head_dim)

    def _cached_rope(self, image_grid_thw: torch.Tensor):
        """Return the grid's cos and sin tables, uploaded with its first image and kept for its later images.

        Keeps the ROPE_CACHE_GRIDS most recently used grids and frees the tables of the grid it drops. Only
        _tower_device uses them, so no caller holds a table that this frees.
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
        """Copy the cached cos and sin to L1 for one forward; the caller frees the copies.

        The RoPE op rereads the tables for every head, so the blocks read them from L1. The cache stays in DRAM to
        leave L1 to the ops' circular buffers.
        """
        return tuple(ttnn.to_memory_config(table, ttnn.L1_MEMORY_CONFIG) for table in (cos, sin))

    # ------------------------------------------------------------------
    def forward_tower(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Run the tower on one image: push_pixel_values, then patch_embed, pre_layrnorm and the blocks.

        Returns (1, 1, L, 1280).
        """
        return self._tower_device(self.push_pixel_values(pixel_values, image_grid_thw), image_grid_thw)

    def placement(self, x_tt: ttnn.Tensor) -> BlockPlacement:
        """Return the BlockPlacement that a forward gives the blocks for pixel values x_tt from push_pixel_values.

        That is block_placement's, or DRAM_PLACEMENT with activations "dram". An L1 placement also needs
        l1_free_for_activations to pass. Otherwise it is DRAM_PLACEMENT, and _log_dram_fallback logs why.
        """
        if self.activations == "dram" or not self.blocks:
            return DRAM_PLACEMENT
        placement = block_placement(self.blocks[0], x_tt.padded_shape[-2], self.activations)
        if placement is DRAM_PLACEMENT:
            return placement
        free, why_not = l1_free_for_activations(self.device)
        if free:
            return placement
        self._log_dram_fallback(why_not)
        return DRAM_PLACEMENT

    def _log_dram_fallback(self, reason: str) -> None:
        """Log a fallback to DRAM_PLACEMENT with its reason and the model's running count of fallbacks.

        It is a warning when the reason differs from the last warning's, and at every DRAM_FALLBACK_WARN_EVERY-th
        fallback. Other repeats are logged at debug level.
        """
        self._dram_fallbacks += 1
        message = f"the vision blocks keep their activations in DRAM (fallback {self._dram_fallbacks}): {reason}"
        if reason != self._warned_fallback or self._dram_fallbacks % DRAM_FALLBACK_WARN_EVERY == 0:
            self._warned_fallback = reason
            logger.warning(message)
        else:
            logger.debug(message)

    def _tower_device(self, x_tt: ttnn.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Run forward_tower's device ops on pixel values from push_pixel_values, leaving x_tt allocated.

        The blocks get placement(x_tt). The last block writes the tower's output to DRAM. If an op's circular buffers
        clash with the blocks' L1 tensors, the image runs again with DRAM_PLACEMENT, once.
        """
        placement = self.placement(x_tt)
        if placement is DRAM_PLACEMENT:
            return self._run_tower(x_tt, image_grid_thw, placement)
        try:
            return self._run_tower(x_tt, image_grid_thw, placement)
        except RuntimeError as error:
            clash = L1_CLASH.search(str(error))
            if clash is None:
                raise
            # The failed run's frames hold the tensors it made: clearing them frees those tensors before the rerun.
            traceback.clear_frames(error.__traceback__)
            line = " ".join(clash.groups())  # tt-metal's line without the program id
            reason = f"placement {placement.name} at {x_tt.padded_shape[-2]} rows failed: {line}"
        self._log_dram_fallback(reason)
        return self._run_tower(x_tt, image_grid_thw, DRAM_PLACEMENT)

    def _run_tower(self, x_tt: ttnn.Tensor, image_grid_thw: torch.Tensor, placement: BlockPlacement) -> ttnn.Tensor:
        """Run patch_embed, pre_layrnorm and the blocks with placement; the last block writes to DRAM."""
        x_tt = self.patch_embed(x_tt)
        x_tt = self.pre_layrnorm(x_tt, memory_config=placement.block_input)
        cos_tt, sin_tt = self._rope_to_l1(*self._cached_rope(image_grid_thw))
        try:
            for i, blk in enumerate(self.blocks):
                last = i == len(self.blocks) - 1
                out_memory_config = ttnn.DRAM_MEMORY_CONFIG if last else None
                x_tt = blk(x_tt, cos_tt, sin_tt, placement, out_memory_config=out_memory_config)
        finally:
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)
        return x_tt

    def forward_device(self, x_tt: ttnn.Tensor, image_grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Run forward's device ops on pixel values that push_pixel_values uploaded for image_grid_thw.

        Leaves x_tt allocated, so one upload can serve several calls. Returns the vision tokens (1, 1, L / 4, 6144),
        or the tower output (1, 1, L, 1280) without the projector.

        Raises ValueError before any device work unless image_grid_thw passes check_image_grid and x_tt has the shape
        (1, 1, h * w, patch_dim). Pixel values of another grid with the same patch count pass and give wrong tokens.
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
