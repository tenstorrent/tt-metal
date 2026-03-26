import ttnn

from models.experimental.tt_symbiote.core.utils import safe_permute


class TtConv1d:
    def __init__(self, device, parameters, stride=1, dilation=1, groups=1):
        self.device = device
        self.weight = parameters["conv"]["weight"]
        self.bias = parameters["conv"].get("bias", None)

        self.out_channels = self.weight.shape[0]
        self.in_channels = self.weight.shape[1]
        self.kernel_size = self.weight.shape[2]

        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.effective_kernel = (self.kernel_size - 1) * self.dilation + 1
        self.left_padding = self.effective_kernel - self.stride

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.float32,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=False,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            config_tensors_in_dram=True,
            reshard_if_not_optimal=False,
            enable_kernel_stride_folding=False,
            transpose_shards=False,
            enable_activation_reuse=False,
            full_inner_dim=False,
        )

    # -----------------------------
    # PURE TT dynamic padding
    # -----------------------------
    def _get_extra_padding(self, length):
        n_frames = (length - self.effective_kernel + self.left_padding) / self.stride + 1
        ideal_length = (int(n_frames + 0.9999) - 1) * self.stride + (self.effective_kernel - self.left_padding)
        return int(ideal_length - length)

    # -----------------------------
    # PURE TT padding (NO TORCH)
    # -----------------------------
    def _pad_input_tt(self, x, batch_size, input_length):
        extra_padding = self._get_extra_padding(input_length)

        new_length = input_length + self.left_padding + extra_padding

        # Step 1: Create zero tensor in TT
        padded = ttnn.zeros(
            [batch_size, self.in_channels, new_length],
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # Step 2: Copy original data into padded tensor (shifted right)
        padded = ttnn.slice_scatter(
            padded,
            x,
            starts=[0, 0, self.left_padding],
        )

        return padded, new_length

    # -----------------------------
    # Depthwise/group conv
    # -----------------------------
    def _apply_group_conv(self, x, batch_size, input_length):
        if self.groups == 1:
            return self._conv2d_call(x, batch_size, input_length)

        outputs = []
        split_size = self.in_channels // self.groups

        for g in range(self.groups):
            x_g = x[:, :, :, g * split_size : (g + 1) * split_size]

            w_g = self.weight[g * (self.out_channels // self.groups) : (g + 1) * (self.out_channels // self.groups)]

            b_g = None
            if self.bias is not None:
                b_g = self.bias[g * (self.out_channels // self.groups) : (g + 1) * (self.out_channels // self.groups)]

            out_g, _, _ = ttnn.conv2d(
                input_tensor=x_g,
                weight_tensor=w_g,
                device=self.device,
                in_channels=split_size,
                out_channels=(self.out_channels // self.groups),
                batch_size=batch_size,
                input_height=input_length,
                input_width=1,
                kernel_size=(self.kernel_size, 1),
                stride=(self.stride, 1),
                padding=(0, 0),
                bias_tensor=b_g,
                conv_config=self.conv_config,
            )

            outputs.append(out_g)

        return ttnn.concat(outputs, dim=3)

    # -----------------------------
    # Conv core
    # -----------------------------
    def _conv2d_call(self, x, batch_size, input_length):
        result, _, _ = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_length,
            input_width=1,
            kernel_size=(self.kernel_size, 1),
            stride=(self.stride, 1),
            padding=(0, 0),
            bias_tensor=self.bias,
            conv_config=self.conv_config,
        )
        return result

    # -----------------------------
    # Forward
    # -----------------------------
    def __call__(self, x, batch_size, input_length):
        """
        Input:  [B, C, L]
        Output: [B, C_out, L]
        """

        # -----------------
        # Step 1: PURE TT padding
        # -----------------
        x, new_length = self._pad_input_tt(x, batch_size, input_length)

        # -----------------
        # Step 2: reshape → Conv2D
        # -----------------
        x = safe_permute(x, [0, 2, 1])
        x = ttnn.reshape(x, [batch_size, 1, new_length, self.in_channels])

        # -----------------
        # Step 3: Conv
        # -----------------
        result = self._apply_group_conv(x, batch_size, new_length)

        # -----------------
        # Step 4: reshape back
        # -----------------
        result = ttnn.reshape(result, [batch_size, result.shape[2], self.out_channels])
        result = safe_permute(result, [0, 2, 1])

        return result
