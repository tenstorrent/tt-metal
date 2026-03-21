# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, List
import json


class TinyTimeMixerConfig:
    r"""
    Configuration class for TinyTimeMixer model.

    Args:
        context_length (int): The context/history length for the input sequence.
        patch_length (int): The patch length for the input sequence.
        num_input_channels (int): Number of input variates.
        patch_stride (int): Amount of points to stride.
        d_model (int): Hidden feature size of the model.
        prediction_length (int): Number of time steps to forecast.
        expansion_factor (int): Expansion factor for MLP.
        num_layers (int): Number of layers.
        dropout (float): Dropout probability.
        mode (str): Mixer mode ("common_channel" or "mix_channel").
        gated_attn (bool): Enable gated attention.
        norm_mlp (str): Normalization type.
        self_attn (bool): Enable self-attention.
        self_attn_heads (int): Number of self-attention heads.
        use_positional_encoding (bool): Use positional encoding.
        positional_encoding_type (str): Type of positional encoding.
        scaling (str): Scaling method.
        loss (str): Loss function.
        init_std (float): Init std.
        norm_eps (float): Norm epsilon.
        adaptive_patching_levels (int): Adaptive patching levels.
        head_dropout (float): Head dropout.
        prediction_channel_indices (List[int]): Prediction channel indices.
        use_decoder (bool): Use decoder.
        decoder_num_layers (int): Decoder layers.
        decoder_d_model (int): Decoder d_model.
        enable_forecast_channel_mixing (bool): Enable forecast channel mixing.
    """

    def __init__(
        self,
        context_length: int = 512,
        patch_length: int = 16,
        num_input_channels: int = 7,
        patch_stride: int = 8,
        d_model: int = 64,
        prediction_length: int = 96,
        expansion_factor: int = 2,
        num_layers: int = 6,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = True,
        norm_mlp: str = "LayerNorm",
        self_attn: bool = False,
        self_attn_heads: int = 1,
        use_positional_encoding: bool = False,
        positional_encoding_type: str = "sincos",
        scaling: str = "std",
        loss: str = "mse",
        init_std: float = 0.02,
        norm_eps: float = 1e-5,
        adaptive_patching_levels: int = 0,
        head_dropout: float = 0.2,
        prediction_channel_indices: Optional[List[int]] = None,
        use_decoder: bool = False,
        decoder_num_layers: int = 8,
        decoder_d_model: int = 64,
        enable_forecast_channel_mixing: bool = False,
        **kwargs,
    ):
        self.context_length = context_length
        self.patch_length = patch_length
        self.num_input_channels = num_input_channels
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.prediction_length = prediction_length
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.gated_attn = gated_attn
        self.norm_mlp = norm_mlp
        self.self_attn = self_attn
        self.self_attn_heads = self_attn_heads
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_type = positional_encoding_type
        self.scaling = scaling
        self.loss = loss
        self.init_std = init_std
        self.norm_eps = norm_eps
        self.adaptive_patching_levels = adaptive_patching_levels
        self.head_dropout = head_dropout
        self.prediction_channel_indices = prediction_channel_indices
        self.use_decoder = use_decoder
        self.decoder_num_layers = decoder_num_layers
        self.decoder_d_model = decoder_d_model
        self.enable_forecast_channel_mixing = enable_forecast_channel_mixing

        # Compute num_patches
        self.num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1

    def to_json_string(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json_string(cls, json_string):
        config_dict = json.loads(json_string)
        return cls(**config_dict)</content>
<parameter name="filePath">/home/mahmudsudo/tt-metal/models/tt_tinytimemixer/configuration_tinytimemixer.py