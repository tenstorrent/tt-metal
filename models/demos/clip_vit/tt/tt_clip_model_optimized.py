# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import ttnn
from models.demos.clip_vit.tt.tt_clip_text_optimized import TtCLIPTextModel as TtCLIPTextModelOptimized
from models.demos.clip_vit.tt.tt_clip_text_optimized import build_text_encoder_configs
from models.demos.clip_vit.tt.tt_clip_vision_optimized import TtCLIPVisionModel, build_vision_encoder_configs


def l2_normalize(tensor: ttnn.Tensor, dim: int = -1, epsilon: float = 1e-12):
    squared = ttnn.mul(tensor, tensor)
    sum_squared = ttnn.sum(squared, dim=dim, keepdim=True)
    sum_squared = ttnn.add(sum_squared, epsilon)
    l2_norm = ttnn.sqrt(sum_squared)
    normalized = ttnn.div(tensor, l2_norm)
    return normalized


class TtCLIPModelOptimized:
    def __init__(
        self,
        config,
        parameters,
        device,
        vision_batch=35,
        text_batch=35,
        dtype=ttnn.bfloat8_b,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.vision_batch = vision_batch
        self.text_batch = text_batch

        text_config = config.text_config
        vision_config = config.vision_config

        vision_memory_configs, vision_program_configs = build_vision_encoder_configs(
            vision_config,
            device,
            vision_batch,
        )

        text_memory_configs, text_program_configs = build_text_encoder_configs(
            text_config,
            device,
            text_batch,
        )

        self.vision_model = TtCLIPVisionModel(
            vision_config,
            parameters.vision_model,
            device,
            vision_memory_configs,
            vision_program_configs,
            dtype=dtype,
        )

        self.text_model = TtCLIPTextModelOptimized(
            text_config,
            parameters.text_model,
            device,
            text_memory_configs,
            text_program_configs,
            dtype=dtype,
        )

        self.visual_projection_weight = ttnn.from_torch(
            parameters.visual_projection.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.text_projection_weight = ttnn.from_torch(
            parameters.text_projection.weight.T,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        logit_scale = ttnn.from_torch(
            parameters.logit_scale.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.logit_scale = ttnn.exp(logit_scale)

    def _get_text_features_single_batch(
        self,
        input_ids: ttnn.Tensor,
        position_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Process a single batch through text encoder + projection."""

        text_outputs = self.text_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        batch_size, seq_len, hidden_size = text_outputs.shape

        # Find EOS token positions (highest token ID = EOS)
        input_ids_float = ttnn.typecast(input_ids, ttnn.float32)
        eos_positions = ttnn.argmax(input_ids_float, dim=-1)
        eos_positions = ttnn.reshape(eos_positions, (batch_size, 1, 1))
        eos_positions = ttnn.to_layout(eos_positions, ttnn.TILE_LAYOUT)
        eos_positions = ttnn.repeat(eos_positions, ttnn.Shape([1, 1, hidden_size]))

        pooled_output = ttnn.gather(text_outputs, dim=1, index=eos_positions)
        ttnn.deallocate(text_outputs)
        pooled_output = ttnn.reshape(pooled_output, (batch_size, hidden_size))

        text_features = ttnn.matmul(pooled_output, self.text_projection_weight)
        ttnn.deallocate(pooled_output)

        return text_features

    def _get_image_features_single_batch(
        self,
        pixel_values: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Process a single batch through vision encoder + projection."""

        pooled_output = self.vision_model(pixel_values=pixel_values)

        image_features = ttnn.matmul(pooled_output, self.visual_projection_weight)
        ttnn.deallocate(pooled_output)

        return image_features

    def _process_all_text(
        self,
        input_ids: ttnn.Tensor,
        position_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Process all text inputs in chunks of self.batch.

        Returns:
            text_embeds: [N, projection_dim] accumulated in DRAM
        """
        total = input_ids.shape[0]
        num_batches = (total + self.text_batch - 1) // self.text_batch
        all_text_embeds = []

        for i in range(num_batches):
            start = i * self.text_batch
            end = min(start + self.text_batch, total)

            batch_ids = ttnn.slice(input_ids, [start, 0], [end, input_ids.shape[1]])

            batch_pos_ids = None
            if position_ids is not None:
                batch_pos_ids = ttnn.slice(position_ids, [start, 0], [end, position_ids.shape[1]])

            batch_mask = None
            if attention_mask is not None:
                batch_mask = ttnn.slice(attention_mask, [start, 0], [end, attention_mask.shape[1]])

            text_features = self._get_text_features_single_batch(
                batch_ids,
                batch_pos_ids,
                batch_mask,
            )
            all_text_embeds.append(text_features)

        return ttnn.concat(all_text_embeds, dim=0)

    def _process_all_vision(
        self,
        pixel_values: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Process all images in chunks of self.batch.

        Returns:
            image_embeds: [N, projection_dim] accumulated in DRAM
        """
        total = pixel_values.shape[0]
        num_batches = (total + self.vision_batch - 1) // self.vision_batch
        all_image_embeds = []

        for i in range(num_batches):
            start = i * self.vision_batch
            end = min(start + self.vision_batch, total)

            batch_pixels = ttnn.slice(
                pixel_values,
                [start, 0, 0, 0],
                [end, 3, 224, 224],
            )

            image_features = self._get_image_features_single_batch(batch_pixels)
            all_image_embeds.append(image_features)

        return ttnn.concat(all_image_embeds, dim=0)

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        pixel_values: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            input_ids: [N, seq_len] text token IDs
            pixel_values: [N, 3, 224, 224] images
            attention_mask: optional [N, seq_len]
            position_ids: optional [N, seq_len]

        Returns:
            logits_per_image: [N, N] similarity scores (image → text)
            logits_per_text: [N, N] similarity scores (text → image)
        """

        text_features = self._process_all_text(
            input_ids,
            position_ids,
            attention_mask,
        )

        image_features = self._process_all_vision(pixel_values)

        text_embeds = l2_normalize(text_features, dim=-1)
        image_embeds = l2_normalize(image_features, dim=-1)

        ttnn.deallocate(text_features)
        ttnn.deallocate(image_features)

        text_embeds_t = ttnn.permute(text_embeds, (1, 0))
        logits_per_image = ttnn.matmul(image_embeds, text_embeds_t)
        logits_per_image = ttnn.mul(logits_per_image, self.logit_scale)
        logits_per_text = ttnn.permute(logits_per_image, (1, 0))

        ttnn.deallocate(text_embeds)
        ttnn.deallocate(image_embeds)
        ttnn.deallocate(text_embeds_t)

        return logits_per_image, logits_per_text
