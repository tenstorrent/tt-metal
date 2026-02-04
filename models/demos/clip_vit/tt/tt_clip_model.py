from typing import Optional, Tuple

import ttnn
from models.demos.clip_vit.tt.tt_clip_text import TtCLIPTextModel
from models.demos.clip_vit.tt.tt_clip_vision import TtCLIPVisionModel


class TtCLIPModel:
    def __init__(self, config, parameters, device):
        self.config = config
        self.device = device

        text_config = config.text_config
        vision_config = config.vision_config

        self.text_model = TtCLIPTextModel(text_config, parameters.text_model, device)
        self.vision_model = TtCLIPVisionModel(vision_config, parameters.vision_model, device)

        self.projection_dim = config.projection_dim
        self.logit_scale_init_value = config.logit_scale_init_value
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.visual_projection_weight = ttnn.from_torch(
            parameters.visual_projection.weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        self.text_projection_weight = ttnn.from_torch(
            parameters.text_projection.weight.T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        logit_scale = ttnn.from_torch(
            parameters.logit_scale.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.logit_scale = ttnn.exp(logit_scale)

    def get_text_features(
        self,
        input_ids: ttnn.Tensor,
        position_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            position_ids: Position IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
        Returns:
            text_features: shape (batch_size, projection_dim)
        """

        text_outputs = self.text_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        batch_size, seq_len, hidden_size = text_outputs.shape

        # Find EOS token positions
        input_ids_float = ttnn.typecast(input_ids, ttnn.float32)
        eos_positions = ttnn.argmax(input_ids_float, dim=-1)
        eos_positions = ttnn.reshape(eos_positions, (batch_size, 1, 1))
        eos_positions = ttnn.to_layout(eos_positions, ttnn.TILE_LAYOUT)
        eos_positions = ttnn.repeat(eos_positions, ttnn.Shape([1, 1, hidden_size]))

        # Get hidden state at EOS positions
        pooled_output = ttnn.gather(text_outputs, dim=1, index=eos_positions)
        pooled_output = ttnn.reshape(pooled_output, (batch_size, hidden_size))

        text_features = ttnn.matmul(pooled_output, self.text_projection_weight)

        return text_features

    def get_image_features(
        self,
        pixel_values: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Args:
            pixel_values: Input images, shape (batch_size, num_channels, height, width)
        Returns:
            image_features: shape (batch_size, projection_dim)
        """
        pooled_output = self.vision_model(pixel_values=pixel_values)

        image_features = ttnn.matmul(pooled_output, self.visual_projection_weight)

        return image_features

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        pixel_values: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        return_loss: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        text_embeds = ttnn.layer_norm(text_features, epsilon=1e-12)
        image_embeds = ttnn.layer_norm(image_features, epsilon=1e-12)

        text_embeds_t = ttnn.permute(text_embeds, (1, 0))
        logits_per_image = ttnn.matmul(image_embeds, text_embeds_t)
        logits_per_image = ttnn.mul(logits_per_image, self.logit_scale)
        logits_per_text = ttnn.permute(logits_per_image, (1, 0))

        return logits_per_image, logits_per_text
