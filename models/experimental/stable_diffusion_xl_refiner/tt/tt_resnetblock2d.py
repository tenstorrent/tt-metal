import torch.nn as nn
import ttnn
from .tt_config import get_resnet_config
from .tt_resnet_components import (
    ResNetWeightLoader,
    ResNetNormalizationLayer,
    ResNetConvolutionLayer,
    ResNetTimeEmbedding,
    ResNetShortcutConnection,
)


class TtResnetBlock2D(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        use_conv_shortcut: bool = False,
    ):
        super().__init__()

        self.device = device
        self.module_path = module_path
        self.use_conv_shortcut = use_conv_shortcut

        # Configuration setup
        self.block_config = get_resnet_config(module_path)

        # Initialize components
        self._initialize_components(state_dict)

    def _initialize_components(self, state_dict):
        # Order of layers:
        # 1. norm_layer_1
        # 2. conv_layer_1
        # 3. time_embedding
        # 4. norm_layer_2
        # 5. conv_layer_2
        # 6. shortcut (if use_conv_shortcut is True)

        # Load weights
        self.weight_loader = ResNetWeightLoader(state_dict, self.module_path, self.use_conv_shortcut)

        self.norm_layer_1 = ResNetNormalizationLayer(
            self.device,
            self.weight_loader.norm_weights_1,
            self.weight_loader.norm_bias_1,
            self.block_config.norm1,
        )

        self.norm_layer_2 = ResNetNormalizationLayer(
            self.device,
            self.weight_loader.norm_weights_2,
            self.weight_loader.norm_bias_2,
            self.block_config.norm2,
        )

        self.conv_layer_1 = ResNetConvolutionLayer(
            self.device,
            self.weight_loader.conv_weights_1,
            self.weight_loader.conv_bias_1,
            self.block_config.conv1,
        )

        self.conv_layer_2 = ResNetConvolutionLayer(
            self.device,
            self.weight_loader.conv_weights_2,
            self.weight_loader.conv_bias_2,
            self.block_config.conv2,
        )

        self.time_embedding = ResNetTimeEmbedding(
            self.device,
            self.weight_loader.time_emb_weights,
            self.weight_loader.time_emb_bias,
        )

        self.shortcut_connection = ResNetShortcutConnection(
            self.device,
            use_conv_shortcut=self.use_conv_shortcut,  # Use actual presence of weights
            conv_weights=self.weight_loader.conv_weights_3,
            conv_bias=self.weight_loader.conv_bias_3,
            conv_config=self.block_config.conv_shortcut if self.use_conv_shortcut else None,
        )

    def forward(self, input_tensor, temb, input_shape):
        B, C, H, W = input_shape
        hidden_states = input_tensor

        # First normalization + activation
        hidden_states = self.norm_layer_1.apply(hidden_states)
        hidden_states = ttnn.silu(hidden_states)

        # First convolution
        hidden_states, [C, H, W] = self.conv_layer_1.apply(hidden_states, B, C, H, W)

        # Time embedding
        temb = self.time_embedding.apply(temb)

        # Add time embedding
        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        hidden_states = ttnn.add(hidden_states, temb, use_legacy=True)

        # Second normalization + activation
        hidden_states = self.norm_layer_2.apply(hidden_states)
        hidden_states = ttnn.silu(hidden_states)

        # Second convolution
        hidden_states, [C, H, W] = self.conv_layer_2.apply(hidden_states, B, C, H, W)

        # Apply shortcut connection
        hidden_states, [C, H, W] = self.shortcut_connection.apply(input_tensor, hidden_states, input_shape)

        # hidden_states = ttnn.add(hidden_states, input_tensor, use_legacy=True)
        # hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)

        return hidden_states, [C, H, W]
