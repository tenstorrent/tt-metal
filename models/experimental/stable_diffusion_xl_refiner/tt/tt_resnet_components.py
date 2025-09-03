import ttnn
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_linear_params,
)
from models.experimental.stable_diffusion_xl_refiner.tt.components.convolution_layer import ConvolutionLayer

# TODO: Remove hardcoded values


class ResNetTimeEmbedding:
    def __init__(self, device, weights, bias, conv_w_dtype=ttnn.bfloat16):
        self.device = device

        self.temb_weights, self.temb_bias = prepare_linear_params(device, weights, bias, conv_w_dtype)

    def apply(self, hidden_states, temb):
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        temb = ttnn.silu(temb)
        temb = ttnn.linear(
            temb,
            self.temb_weights,
            bias=self.temb_bias,
        )
        return ttnn.add(hidden_states, temb, use_legacy=True)


class ResNetShortcutConnection:
    def __init__(self, device, use_conv_shortcut=False, conv_weights=None, conv_bias=None, conv_config=None):
        self.device = device
        self.use_conv_shortcut = use_conv_shortcut

        if self.use_conv_shortcut and conv_weights is not None:
            self.shortcut_conv = ConvolutionLayer(device, conv_weights, conv_bias, conv_config)
        else:
            self.shortcut_conv = None

    def apply(self, input_tensor, hidden_states, input_shape):
        B, C, H, W = input_shape

        shortcut = input_tensor

        if self.shortcut_conv is not None:
            shortcut, [C, H, W] = self.shortcut_conv.apply(shortcut, B, C, H, W)

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        shortcut = ttnn.to_layout(shortcut, ttnn.TILE_LAYOUT)
        result = ttnn.add(hidden_states, shortcut, use_legacy=True)
        return ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG), [C, H, W]
