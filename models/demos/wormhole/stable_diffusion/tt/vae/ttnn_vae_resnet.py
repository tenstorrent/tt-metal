import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_utils import (
    get_default_compute_config,
    get_default_conv_config,
    prepare_split_conv_weights_bias,
    split_conv_and_run,
    prepare_group_norm,
)


class ResnetBlock:
    def __init__(
        self,
        torch_resnet,
        device,
        in_channels,
        input_height,
        input_width,
        out_channels,
        norm1_num_blocks,
        norm2_num_blocks,
        conv1_channel_split_factors,
        conv2_channel_split_factors,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_height = input_height
        self.input_width = input_width
        self.conv1_in_channel_split_factor = conv1_channel_split_factors[0]
        self.conv1_out_channel_split_factor = conv1_channel_split_factors[1]
        self.conv2_in_channel_split_factor = conv2_channel_split_factors[0]
        self.conv2_out_channel_split_factor = conv2_channel_split_factors[1]

        self.compute_config = get_default_compute_config(device)
        self.conv_config = get_default_conv_config()

        # groupnorm 1
        self.norm1_num_blocks = norm1_num_blocks
        self.norm1_grid_core = ttnn.CoreGrid(y=4, x=8) if in_channels == 128 else ttnn.CoreGrid(y=8, x=8)
        (
            self.norm1_input_mask,
            self.norm1_weights,
            self.norm1_bias,
        ) = prepare_group_norm(
            self.device,
            in_channels,
            self.norm1_grid_core,
            torch_resnet.norm1.weight,
            torch_resnet.norm1.bias,
        )

        # conv 1
        self.conv1_weights, self.conv1_bias = prepare_split_conv_weights_bias(
            in_channels,
            out_channels,
            self.conv1_in_channel_split_factor,
            self.conv1_out_channel_split_factor,
            torch_resnet.conv1.weight,
            torch_resnet.conv1.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        )

        # groupnorm 2
        self.norm2_num_blocks = norm2_num_blocks
        self.norm2_grid_core = ttnn.CoreGrid(y=4, x=8) if out_channels == 128 else ttnn.CoreGrid(y=8, x=8)
        (
            self.norm2_input_mask,
            self.norm2_weights,
            self.norm2_bias,
        ) = prepare_group_norm(
            self.device,
            out_channels,
            self.norm2_grid_core,
            torch_resnet.norm2.weight,
            torch_resnet.norm2.bias,
        )

        # conv 2
        self.conv2_weights, self.conv2_bias = prepare_split_conv_weights_bias(
            out_channels,
            out_channels,
            self.conv2_in_channel_split_factor,
            self.conv2_out_channel_split_factor,
            torch_resnet.conv2.weight,
            torch_resnet.conv2.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        )

        # conv shortcut
        self.conv_shortcut = False
        if torch_resnet.conv_shortcut:
            self.conv_shortcut_weights, self.conv_shortcut_bias = prepare_split_conv_weights_bias(
                in_channels,
                out_channels,
                self.conv1_in_channel_split_factor,
                self.conv1_out_channel_split_factor,
                torch_resnet.conv_shortcut.weight,
                torch_resnet.conv_shortcut.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            )
            self.conv_shortcut = True

    def __call__(self, input_tensor, groups=32, eps=1e-5):
        hidden_states = input_tensor
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, [1, 1, self.input_height * self.input_width, self.in_channels])
        hidden_states = ttnn.tilize_with_zero_padding(hidden_states, use_multicore=True)

        # groupnorm 1
        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=groups,
            input_mask=self.norm1_input_mask,
            weight=self.norm1_weights,
            bias=self.norm1_bias,
            epsilon=eps,
            core_grid=self.norm1_grid_core,
            dtype=ttnn.bfloat8_b,
            inplace=False,
            num_out_blocks=self.norm1_num_blocks,
        )

        # silu 1
        hidden_states = ttnn.silu(hidden_states)

        # conv 1
        hidden_states = split_conv_and_run(
            hidden_states,
            self.conv1_weights,
            self.conv1_bias,
            self.device,
            self.in_channels,
            self.input_height,
            self.input_width,
            self.out_channels,
            self.conv1_in_channel_split_factor,
            self.conv1_out_channel_split_factor,
            self.compute_config,
            self.conv_config,
        )

        # groupnorm 2
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=groups,
            input_mask=self.norm2_input_mask,
            weight=self.norm2_weights,
            bias=self.norm2_bias,
            epsilon=eps,
            core_grid=self.norm2_grid_core,
            dtype=ttnn.bfloat8_b,
            inplace=False,
            num_out_blocks=self.norm2_num_blocks,
        )

        # silu 2
        hidden_states = ttnn.silu(hidden_states)

        # conv 2
        hidden_states = split_conv_and_run(
            hidden_states,
            self.conv2_weights,
            self.conv2_bias,
            self.device,
            self.out_channels,
            self.input_height,
            self.input_width,
            self.out_channels,
            self.conv2_in_channel_split_factor,
            self.conv2_out_channel_split_factor,
            self.compute_config,
            self.conv_config,
        )

        if self.conv_shortcut:
            input_tensor = split_conv_and_run(
                input_tensor,
                self.conv_shortcut_weights,
                self.conv_shortcut_bias,
                self.device,
                self.in_channels,
                self.input_height,
                self.input_width,
                self.out_channels,
                self.conv1_in_channel_split_factor,
                self.conv1_out_channel_split_factor,
                self.compute_config,
                self.conv_config,
                kernel_size=1,
                padding=0,
            )

        if hidden_states.shape != input_tensor.shape:
            hidden_states = ttnn.reshape(hidden_states, input_tensor.shape)

        hidden_states = ttnn.add(hidden_states, input_tensor, output_tensor=hidden_states)
        input_tensor.deallocate(True)

        return hidden_states
