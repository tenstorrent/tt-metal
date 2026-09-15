# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from collections.abc import Mapping

import torch
import transformers

import ttnn
from models.tt_dit.blocks.rope import RopeConfig
from models.tt_dit.encoders.transformer import (
    WEIGHT_CACHE_DTYPE,
    StateConversion,
    TransformerEncoder,
    TransformerEncoderConfig,
)
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import cache


class Qwen3VlEncoder(TransformerEncoder):
    @staticmethod
    def config_from_hf(hf_config: transformers.PretrainedConfig) -> TransformerEncoderConfig:
        """Takes a `Qwen3VLConfig`, or the `Qwen3VLTextConfig` under its `text_config`."""
        text_config = getattr(hf_config, "text_config", hf_config)
        rope = text_config.rope_parameters

        return TransformerEncoderConfig(
            vocab_size=text_config.vocab_size,
            head_size=text_config.head_dim,
            embed_size=text_config.hidden_size,
            ff_size=text_config.intermediate_size,
            num_layers=text_config.num_hidden_layers,
            num_heads=text_config.num_attention_heads,
            num_kv_heads=text_config.num_key_value_heads,
            norm_eps=text_config.rms_norm_eps,
            attn_qkv_bias=text_config.attention_bias,
            attn_out_bias=False,
            # Qwen3 applies a per-head RMSNorm to the queries and keys.
            attn_qk_norm=True,
            rope_config=RopeConfig(
                theta=rope["rope_theta"],
                mrope_section=list(rope["mrope_section"]),
                mrope_interleaved=True,
            ),
        )

    @staticmethod
    def convert_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return STATE_CONVERSION.convert(state_dict)


# The keys of `Qwen3VLForConditionalGeneration`; the `model.` prefix is optional so a checkpoint
# of the bare `Qwen3VLModel` converts too.
STATE_CONVERSION = StateConversion(
    rename=[
        (r"^(?:model\.)?language_model\.embed_tokens", r"token_embedding"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.self_attn\.([qkvo])_proj", r"layers.\1.attn.\2_proj"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.self_attn\.([qk])_norm", r"layers.\1.attn.\2_norm"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.mlp\.gate_proj", r"layers.\1.ff.gate"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.mlp\.up_proj", r"layers.\1.ff.linear_in"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.mlp\.down_proj", r"layers.\1.ff.linear_out"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.post_attention_layernorm", r"layers.\1.ff_norm"),
        (r"^(?:model\.)?language_model\.layers\.([0-9]+)\.input_layernorm", r"layers.\1.attn_norm"),
        (r"^(?:model\.)?language_model\.norm\.weight", r"final_norm.weight"),
        (r"^lm_head\.weight", r"final_linear.weight"),
    ],
    remove=[r"^(?:model\.)?visual"],
)


class Qwen3VlCheckpoint:
    """A Qwen3-VL checkpoint: fetches weights and builds a loaded ``Qwen3VlEncoder``.

    Reads only ``config.json`` in ``__init__``; the actual torch weights are loaded lazily, on
    ``build()`` cache-miss only. Only the language model is kept, so the vision tower is never
    instantiated.
    """

    def __init__(self, name: str, *, subfolder: str = "") -> None:
        hf_config = transformers.AutoConfig.from_pretrained(name, subfolder=subfolder)

        self._name = name
        self._subfolder = subfolder
        self.config = Qwen3VlEncoder.config_from_hf(hf_config)

    def build(
        self,
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig,
        ccl_manager: CCLManager | None = None,
    ) -> Qwen3VlEncoder:
        """Construct a ``Qwen3VlEncoder`` for this checkpoint and load its weights."""
        model = Qwen3VlEncoder(
            self.config,
            device=device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        self.load_weights(model, device=device, parallel_config=parallel_config)
        return model

    def load_weights(
        self,
        model: Qwen3VlEncoder,
        *,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig,
    ) -> None:
        """Load the weights of a model built by ``build``, e.g. again after ``deallocate_weights``."""
        cache.load_model(
            model,
            get_torch_state_dict=self._load_state_dict,
            model_name=self._name,
            subfolder=self._subfolder,
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
            mesh_device=device,
            dtype=WEIGHT_CACHE_DTYPE,
        )

    def _load_state_dict(self) -> dict[str, torch.Tensor]:
        torch_model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            self._name,
            subfolder=self._subfolder,
            torch_dtype=torch.bfloat16,
        )
        return Qwen3VlEncoder.convert_state(torch_model.state_dict())


def mrope_position_ids(
    mm_token_type_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    spatial_merge_size: int,
) -> torch.Tensor:
    """The (3, batch, sequence) multimodal rope positions of a prompt without padding.

    `mm_token_type_ids` marks every token as text (0), image (1) or video (2), and the grids give
    the temporal, height and width extent of every image and video in the order their runs appear.
    A text run counts up on all three axes. A vision run takes its grid coordinates, offset by the
    position after the preceding text, and the text after it resumes at that offset plus the
    run's largest spatial extent. Follows `Qwen3VLModel.get_rope_index` of transformers, where a
    video's frames are separated by text and each is its own run.
    """
    grids = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(_video_frame_grids(video_grid_thw)) if video_grid_thw is not None else None,
    }

    batch_size, sequence_length = mm_token_type_ids.shape
    position_ids = torch.zeros([3, batch_size, sequence_length], dtype=torch.long)

    for batch_idx, token_types in enumerate(mm_token_type_ids.tolist()):
        current_pos = 0
        parts = []
        for token_type, group in itertools.groupby(token_types):
            length = len(list(group))

            if token_type == 0:
                parts.append(torch.arange(length).expand(3, -1) + current_pos)
                current_pos += length
                continue

            grid_iter = grids.get(token_type)
            grid = next(grid_iter, None) if grid_iter is not None else None
            if grid is None:
                msg = f"no matching grid for the vision run of {length} tokens of type {token_type}"
                raise ValueError(msg)

            block = vision_position_ids(current_pos, grid, spatial_merge_size=spatial_merge_size)
            if block.shape[1] != length:
                msg = f"a vision run of {length} tokens does not match the {block.shape[1]} tokens of grid {grid.tolist()}"
                raise ValueError(msg)
            parts.append(block)
            current_pos += max(int(grid[1]), int(grid[2])) // spatial_merge_size

        position_ids[:, batch_idx] = torch.cat(parts, dim=1)

    return position_ids


def vision_position_ids(start_position: int, grid_thw: torch.Tensor, *, spatial_merge_size: int) -> torch.Tensor:
    """The (3, tokens) positions of one image or video frame block, offset by `start_position`.

    Follows `Qwen3VLModel.get_vision_position_ids` of transformers.
    """
    t = int(grid_thw[0])
    h = int(grid_thw[1]) // spatial_merge_size
    w = int(grid_thw[2]) // spatial_merge_size

    temporal = torch.arange(t) + start_position
    height = torch.arange(h) + start_position
    width = torch.arange(w) + start_position

    grid = torch.meshgrid(temporal, height, width, indexing="ij")
    return torch.stack(grid, dim=0).reshape(3, -1)


def _video_frame_grids(video_grid_thw: torch.Tensor) -> torch.Tensor:
    """One (1, h, w) grid per frame: timestamps separate a video's frames into separate runs."""
    frames = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    frames[:, 0] = 1
    return frames
