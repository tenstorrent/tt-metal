"""
Components for TT ResNet blocks - separated for better modularity
"""
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_gn_mask,
    prepare_gn_beta_gamma,
    prepare_conv_params,
    prepare_split_conv_params,
    split_conv2d,
    prepare_linear_params,
)


class ResNetWeightLoader:
    def __init__(self, state_dict, module_path, use_conv_shortcut=False):
        self.module_path = module_path
        self.use_conv_shortcut = use_conv_shortcut
        self._load_all_weights(state_dict)

    def _load_all_weights(self, state_dict):
        # Norm weights
        self.norm_weights_1 = state_dict[f"{self.module_path}.norm1.weight"]
        self.norm_bias_1 = state_dict[f"{self.module_path}.norm1.bias"]
        self.norm_weights_2 = state_dict[f"{self.module_path}.norm2.weight"]
        self.norm_bias_2 = state_dict[f"{self.module_path}.norm2.bias"]

        # Conv weights
        self.conv_weights_1 = state_dict[f"{self.module_path}.conv1.weight"]
        self.conv_bias_1 = state_dict[f"{self.module_path}.conv1.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.conv_weights_2 = state_dict[f"{self.module_path}.conv2.weight"]
        self.conv_bias_2 = state_dict[f"{self.module_path}.conv2.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Time embedding weights
        self.time_emb_weights = state_dict[f"{self.module_path}.time_emb_proj.weight"]
        self.time_emb_bias = state_dict[f"{self.module_path}.time_emb_proj.bias"]

        # Shortcut weights - always check if they exist in state_dict
        # regardless of use_conv_shortcut parameter
        if f"{self.module_path}.conv_shortcut.weight" in state_dict:
            self.conv_weights_3 = state_dict[f"{self.module_path}.conv_shortcut.weight"]
            self.conv_bias_3 = (
                state_dict[f"{self.module_path}.conv_shortcut.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            self.has_conv_shortcut = True
        else:
            self.conv_weights_3 = None
            self.conv_bias_3 = None
            self.has_conv_shortcut = False


class ResNetNormalizationLayer:
    def __init__(self, device, weights, bias, norm_config):
        self.device = device

        # From norm_config
        self.norm_groups = norm_config.num_groups
        self.norm_eps = norm_config.eps
        self.core_grid = norm_config.core_grid
        self.num_out_blocks = norm_config.num_out_blocks

        # Prepare normalization parameters
        self.gamma_t, self.beta_t = prepare_gn_beta_gamma(  # Copy paste from group_norm_DRAM tests
            device, weights, bias, self.core_grid.y
        )
        self.input_mask = prepare_gn_mask(  # Copy paste from group_norm_DRAM tests
            device, weights.shape[0], self.norm_groups, self.core_grid.y
        )

    def apply(self, hidden_states):
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=self.norm_groups,
            input_mask=self.input_mask,
            weight=self.gamma_t,
            bias=self.beta_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Copy paste from group_norm_DRAM tests
            core_grid=self.core_grid,
            epsilon=self.norm_eps,
            inplace=False,
            num_out_blocks=self.num_out_blocks,
        )
        return ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)


class ResNetConvolutionLayer:
    def __init__(self, device, weights, bias, conv_config):
        self.device = device
        self.conv_config = conv_config
        self.split_in = conv_config.split_in
        self.split_out = conv_config.split_out

        self.split_conv = self.split_in > 1 or self.split_out > 1

        # Prepare conv parameters
        if self.split_conv:
            self.tt_weights, self.tt_bias, self.conv_params = prepare_split_conv_params(
                weights, bias, ttnn.bfloat16, self.split_in, self.split_out
            )
        else:
            self.tt_weights, self.tt_bias, self.conv_params = prepare_conv_params(weights, bias, ttnn.bfloat16)

    def apply(self, hidden_states, B, C, H, W):
        if self.split_conv:
            return self._apply_split_conv(hidden_states, B, C, H, W)
        else:
            return self._apply_regular_conv(hidden_states, B, C, H, W)

    def _apply_split_conv(self, hidden_states, B, C, H, W):
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states, [C, H, W], [self.tt_weights, self.tt_bias] = split_conv2d(
            device=self.device,
            hidden_states=hidden_states,
            input_shape=[B, C, H, W],
            conv_weights=self.tt_weights,
            conv_bias=self.tt_bias,
            split_in=self.split_in,
            split_out=self.split_out,
            compute_config=self.conv_config.compute_config,
            conv_config=self.conv_config.conv2d_config,
            conv_params=self.conv_params,
            conv_dtype=ttnn.bfloat16,
            stride=self.conv_config.stride,
            padding=self.conv_config.padding,
            dilation=self.conv_config.dilation,
            groups=self.conv_config.groups,
        )
        return hidden_states, [C, H, W]

    def _apply_regular_conv(self, hidden_states, B, C, H, W):
        [hidden_states, [H, W], [self.tt_weights, self.tt_bias]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_weights,
            in_channels=self.conv_params["input_channels"],
            out_channels=self.conv_params["output_channels"],
            device=self.device,
            bias_tensor=self.tt_bias,
            kernel_size=self.conv_params["kernel_size"],
            stride=self.conv_config.stride,
            padding=self.conv_config.padding,
            dilation=self.conv_config.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv_config.conv2d_config,
            compute_config=self.conv_config.compute_config,
            groups=self.conv_config.groups,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        C = self.conv_params["output_channels"]
        return hidden_states, [C, H, W]


class ResNetTimeEmbedding:
    def __init__(self, device, weights, bias, conv_w_dtype=ttnn.bfloat16):
        self.device = device

        self.tt_weights, self.tt_bias = prepare_linear_params(device, weights, bias, conv_w_dtype)

    def apply(self, temb):
        temb = ttnn.silu(temb)
        return ttnn.linear(
            temb,
            self.tt_weights,
            bias=self.tt_bias,
        )


class ResNetShortcutConnection:
    def __init__(self, device, use_conv_shortcut=False, conv_weights=None, conv_bias=None, conv_config=None):
        self.device = device
        self.use_conv_shortcut = use_conv_shortcut

        if self.use_conv_shortcut and conv_weights is not None:
            self.shortcut_conv = ResNetConvolutionLayer(device, conv_weights, conv_bias, conv_config)
        else:
            self.shortcut_conv = None

    def apply(self, input_tensor, hidden_states, input_shape):
        B, C, H, W = input_shape

        shortcut = input_tensor

        if self.shortcut_conv is not None:
            shortcut, [C, H, W] = self.shortcut_conv.apply(shortcut, B, C, H, W)

        result = ttnn.add(hidden_states, shortcut, use_legacy=True)
        return ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG), [C, H, W]
