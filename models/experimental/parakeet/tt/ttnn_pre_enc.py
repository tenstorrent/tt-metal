import ttnn
import torch
import math


class TTNNConvSubsampling:
    """
    TTNN implementation of ConvSubsampling with depthwise separable convolutions and masking.
    Matches PyTorch reference (pytorch_conf_enc.py):
    - Conv2d(1, 256, 3x3, stride=2, padding=1) + ReLU
    - Conv2d(256, 256, 3x3, stride=2, padding=1, groups=256)
    - Conv2d(256, 256, 1x1) + ReLU
    - Conv2d(256, 256, 3x3, stride=2, padding=1, groups=256)
    - Conv2d(256, 256, 1x1) + ReLU
    - Linear(conv_out_features, feat_out) where conv_out_features = conv_channels * (freq after 3 striding convs)
    """

    def __init__(self, device, feat_in=80, feat_out=1024, conv_channels=256):
        self.device = device
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.conv_channels = conv_channels
        self.subsampling_factor = 8
        self._sampling_num = int(math.log(self.subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = 3
        self._left_padding = (self._kernel_size - 1) // 2
        self._right_padding = (self._kernel_size - 1) // 2

        # Initialize compute config
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Initialize conv configs for each layer
        self._init_conv_configs()

        # Initialize weights
        self._init_weights()

    def _init_conv_configs(self):
        """Initialize Conv2dConfig for each convolution layer."""
        # Common config for all layers
        base_config = {
            "weights_dtype": ttnn.bfloat16,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        }

        # Layer 0: Initial conv with ReLU
        self.conv0_config = ttnn.Conv2dConfig(
            **base_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        )

        # Layer 1: Depthwise conv (groups=conv_channels)
        self.conv1_config = ttnn.Conv2dConfig(**base_config)

        # Layer 2: Pointwise conv (1x1) with ReLU
        self.conv2_config = ttnn.Conv2dConfig(
            **base_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        )

        # Layer 3: Depthwise conv
        self.conv3_config = ttnn.Conv2dConfig(**base_config)

        # Layer 4: Pointwise conv with ReLU
        self.conv4_config = ttnn.Conv2dConfig(
            **base_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        )

    def _init_weights(self):
        """Initialize weights following NeMo's depthwise conv initialization."""
        # Layer 0: Conv2d(1, 256, 3x3, stride=2, padding=1)
        self.conv0_weight = torch.randn(256, 1, 3, 3, dtype=torch.bfloat16)
        self.conv0_bias = torch.randn(256, dtype=torch.bfloat16)
        scale = 1.0 / self._kernel_size
        self.conv0_weight.uniform_(-scale, scale)
        self.conv0_bias.uniform_(-scale, scale)

        # Layer 1: Depthwise Conv2d(256, 256, 3x3, stride=2, groups=256)
        self.conv1_weight = torch.randn(256, 1, 3, 3, dtype=torch.bfloat16)
        self.conv1_bias = torch.randn(256, dtype=torch.bfloat16)
        dw_max = (self._kernel_size**2) ** -0.5
        self.conv1_weight.uniform_(-dw_max, dw_max)
        self.conv1_bias.uniform_(-dw_max, dw_max)

        # Layer 2: Pointwise Conv2d(256, 256, 1x1)
        self.conv2_weight = torch.randn(256, 256, 1, 1, dtype=torch.bfloat16)
        self.conv2_bias = torch.randn(256, dtype=torch.bfloat16)
        pw_max = self.conv_channels**-0.5
        self.conv2_weight.uniform_(-pw_max, pw_max)
        self.conv2_bias.uniform_(-pw_max, pw_max)

        # Layer 3: Depthwise Conv2d(256, 256, 3x3, stride=2, groups=256)
        self.conv3_weight = torch.randn(256, 1, 3, 3, dtype=torch.bfloat16)
        self.conv3_bias = torch.randn(256, dtype=torch.bfloat16)
        self.conv3_weight.uniform_(-dw_max, dw_max)
        self.conv3_bias.uniform_(-dw_max, dw_max)

        # Layer 4: Pointwise Conv2d(256, 256, 1x1)
        self.conv4_weight = torch.randn(256, 256, 1, 1, dtype=torch.bfloat16)
        self.conv4_bias = torch.randn(256, dtype=torch.bfloat16)
        self.conv4_weight.uniform_(-pw_max, pw_max)
        self.conv4_bias.uniform_(-pw_max, pw_max)

        # Linear layer weights
        self._calculate_conv_out_features()
        self.linear_weight = torch.randn(self.conv_out_features, self.feat_out, dtype=torch.bfloat16)
        self.linear_bias = torch.randn(self.feat_out, dtype=torch.bfloat16)
        fc_scale = (self.feat_out * self.feat_in / self._sampling_num) ** -0.5
        self.linear_weight.uniform_(-fc_scale, fc_scale)
        self.linear_bias.uniform_(-fc_scale, fc_scale)

        # Convert weights to TTNN tensors
        self._convert_weights_to_ttnn()

    def _calculate_conv_out_features(self):
        """Calculate output features after convolutions."""
        in_length = torch.tensor(self.feat_in, dtype=torch.float)
        out_length = self._calc_length(
            lengths=in_length,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=False,
            repeat_num=self._sampling_num,
        )
        self.conv_out_features = self.conv_channels * int(out_length)

    def _calc_length(self, lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
        """Calculate output length after convolutions."""
        add_pad = all_paddings - kernel_size
        one = 1.0
        for _ in range(repeat_num):
            lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
            if ceil_mode:
                lengths = torch.ceil(lengths)
            else:
                lengths = torch.floor(lengths)
        return lengths.to(dtype=torch.int)

    def _convert_weights_to_ttnn(self):
        """Convert PyTorch weights to TTNN tensors on device."""
        # Conv weights (OIHW format for TTNN)
        self.tt_conv0_weight = ttnn.from_torch(self.conv0_weight, dtype=ttnn.bfloat16, device=self.device)
        self.tt_conv0_bias = ttnn.from_torch(
            self.conv0_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, device=self.device
        )
        self.tt_conv1_weight = ttnn.from_torch(self.conv1_weight, dtype=ttnn.bfloat16, device=self.device)
        self.tt_conv1_bias = ttnn.from_torch(
            self.conv1_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, device=self.device
        )
        self.tt_conv2_weight = ttnn.from_torch(self.conv2_weight, dtype=ttnn.bfloat16, device=self.device)
        self.tt_conv2_bias = ttnn.from_torch(
            self.conv2_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, device=self.device
        )
        self.tt_conv3_weight = ttnn.from_torch(self.conv3_weight, dtype=ttnn.bfloat16, device=self.device)
        self.tt_conv3_bias = ttnn.from_torch(
            self.conv3_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, device=self.device
        )
        self.tt_conv4_weight = ttnn.from_torch(self.conv4_weight, dtype=ttnn.bfloat16, device=self.device)
        self.tt_conv4_bias = ttnn.from_torch(
            self.conv4_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, device=self.device
        )
        # Linear weights (TILE_LAYOUT required for ttnn.linear/matmul)
        self.tt_linear_weight = ttnn.from_torch(
            self.linear_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.tt_linear_bias = ttnn.from_torch(
            self.linear_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    def set_weights_from_reference(self, ref_module):
        """Copy weights from reference ConvSubsampling (PyTorch) so outputs match for PCC."""
        # ref.conv: [0]=conv0, [2]=dw1, [3]=pw2, [5]=dw3, [6]=pw4
        self.conv0_weight.copy_(ref_module.conv[0].weight.detach().to(torch.bfloat16))
        self.conv0_bias.copy_(ref_module.conv[0].bias.detach().to(torch.bfloat16))
        self.conv1_weight.copy_(ref_module.conv[2].weight.detach().to(torch.bfloat16))
        self.conv1_bias.copy_(ref_module.conv[2].bias.detach().to(torch.bfloat16))
        self.conv2_weight.copy_(ref_module.conv[3].weight.detach().to(torch.bfloat16))
        self.conv2_bias.copy_(ref_module.conv[3].bias.detach().to(torch.bfloat16))
        self.conv3_weight.copy_(ref_module.conv[5].weight.detach().to(torch.bfloat16))
        self.conv3_bias.copy_(ref_module.conv[5].bias.detach().to(torch.bfloat16))
        self.conv4_weight.copy_(ref_module.conv[6].weight.detach().to(torch.bfloat16))
        self.conv4_bias.copy_(ref_module.conv[6].bias.detach().to(torch.bfloat16))
        # ref.out: weight (feat_out, conv_out_features) -> we need (conv_out_features, feat_out)
        self.linear_weight.copy_(ref_module.out.weight.T.detach().to(torch.bfloat16))
        self.linear_bias.copy_(ref_module.out.bias.detach().to(torch.bfloat16))
        self._convert_weights_to_ttnn()

    def _apply_mask(self, x, lengths, batch_size_logical=None, out_h=None, out_w=None):
        """Apply mask to tensor based on lengths. Build on host to avoid torch/ttnn mix, then upload.

        When TTNN conv returns merged layout (1, 1, batch*out_h*out_w, C), pass batch_size_logical, out_h, out_w
        so we build a mask that zeroes positions where time_in_batch >= lengths[batch_id] for each batch.
        """
        batch_size, channels, time, features = x.shape
        if isinstance(lengths, torch.Tensor):
            lengths_np = lengths.cpu()
        else:
            lengths_np = torch.tensor(lengths, dtype=torch.long)

        # Merged layout: x is (1, 1, batch_size*out_h*out_w, C); mask per (batch, time)
        if (
            batch_size_logical is not None
            and out_h is not None
            and out_w is not None
            and batch_size == 1
            and channels == 1
            and time == batch_size_logical * out_h * out_w
        ):
            # For each position i: batch_id = i // (out_h*out_w), time_in_batch = (i % (out_h*out_w)) // out_w
            total = time
            batch_steps = out_h * out_w
            lengths_batch = lengths_np[:batch_size_logical]
            mask_1d = torch.ones(total, dtype=torch.bfloat16, device=lengths_np.device)
            for i in range(total):
                b = i // batch_steps
                t = (i % batch_steps) // out_w
                if b < lengths_batch.shape[0] and t >= lengths_batch[b].item():
                    mask_1d[i] = 0
            mask = mask_1d.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(1, 1, total, features)
        else:
            # Standard (batch, channels, time, features) layout
            lengths_batch = lengths_np[:batch_size]
            time_idx = torch.arange(time, device=lengths_np.device).unsqueeze(0)
            time_mask = time_idx < lengths_batch.unsqueeze(1)
            mask = (
                time_mask.unsqueeze(-1)
                .expand(batch_size, time, features)
                .unsqueeze(1)
                .expand(batch_size, channels, time, features)
                .to(torch.bfloat16)
            )

        if mask.dtype != torch.bfloat16:
            mask = mask.to(torch.bfloat16)
        mask_tt = ttnn.from_torch(mask, dtype=ttnn.bfloat16, device=self.device)
        return ttnn.multiply(x, mask_tt)

    def _calculate_conv_output_size(self, input_size, kernel_size, stride, padding):
        """Calculate exact output size after convolution."""
        return (input_size + padding[0] + padding[1] - kernel_size) // stride + 1

    def forward(self, x, lengths):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, time, feat_in)
            lengths: Length tensor of shape (batch,)

        Returns:
            x: Output tensor of shape (batch, time//8, feat_out)
            lengths: Updated lengths tensor of shape (batch,)
        """
        batch_size, time, feat_in = x.shape

        # Convert input to conv format (batch, 1, time, feat_in)
        x = ttnn.unsqueeze(x, 1)
        current_lengths = lengths.clone()

        # Apply initial mask
        x = self._apply_mask(x, current_lengths)

        # Layer 0: Conv2d(1, 256, 3x3, stride=2, padding=1) + ReLU
        x, [out_h, out_w], [self.tt_conv0_weight, self.tt_conv0_bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.tt_conv0_weight,
            bias_tensor=self.tt_conv0_bias,
            in_channels=1,
            out_channels=self.conv_channels,
            batch_size=batch_size,
            input_height=time,
            input_width=feat_in,
            kernel_size=(self._kernel_size, self._kernel_size),
            stride=(self._stride, self._stride),
            padding=(self._left_padding, self._right_padding),
            device=self.device,
            conv_config=self.conv0_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        # Update lengths
        current_lengths = self._calculate_conv_output_size(
            current_lengths, self._kernel_size, self._stride, (self._left_padding, self._right_padding)
        )
        x = self._apply_mask(x, current_lengths, batch_size, out_h, out_w)

        # Layer 1: Depthwise Conv2d(256, 256, 3x3, stride=2, groups=256)
        x, [out_h, out_w], [self.tt_conv1_weight, self.tt_conv1_bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.tt_conv1_weight,
            bias_tensor=self.tt_conv1_bias,
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            batch_size=batch_size,
            input_height=out_h,
            input_width=out_w,
            kernel_size=(self._kernel_size, self._kernel_size),
            stride=(self._stride, self._stride),
            padding=(self._left_padding, self._right_padding),
            groups=self.conv_channels,
            device=self.device,
            conv_config=self.conv1_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        # Update lengths
        current_lengths = self._calculate_conv_output_size(
            current_lengths, self._kernel_size, self._stride, (self._left_padding, self._right_padding)
        )
        x = self._apply_mask(x, current_lengths, batch_size, out_h, out_w)

        # Layer 2: Pointwise Conv2d(256, 256, 1x1) + ReLU
        x, [out_h, out_w], [self.tt_conv2_weight, self.tt_conv2_bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.tt_conv2_weight,
            bias_tensor=self.tt_conv2_bias,
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            batch_size=batch_size,
            input_height=out_h,
            input_width=out_w,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            device=self.device,
            conv_config=self.conv2_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        x = self._apply_mask(x, current_lengths, batch_size, out_h, out_w)

        # Layer 3: Depthwise Conv2d(256, 256, 3x3, stride=2, groups=256)
        x, [out_h, out_w], [self.tt_conv3_weight, self.tt_conv3_bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.tt_conv3_weight,
            bias_tensor=self.tt_conv3_bias,
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            batch_size=batch_size,
            input_height=out_h,
            input_width=out_w,
            kernel_size=(self._kernel_size, self._kernel_size),
            stride=(self._stride, self._stride),
            padding=(self._left_padding, self._right_padding),
            groups=self.conv_channels,
            device=self.device,
            conv_config=self.conv3_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        # Update lengths
        current_lengths = self._calculate_conv_output_size(
            current_lengths, self._kernel_size, self._stride, (self._left_padding, self._right_padding)
        )
        x = self._apply_mask(x, current_lengths, batch_size, out_h, out_w)

        # Layer 4: Pointwise Conv2d(256, 256, 1x1) + ReLU
        x, [out_h, out_w], [self.tt_conv4_weight, self.tt_conv4_bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.tt_conv4_weight,
            bias_tensor=self.tt_conv4_bias,
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            batch_size=batch_size,
            input_height=out_h,
            input_width=out_w,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            device=self.device,
            conv_config=self.conv4_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        x = self._apply_mask(x, current_lengths, batch_size, out_h, out_w)

        # Flatten: TTNN conv returns (1, 1, batch*out_h*out_w, channels); reshape to (batch, out_h, conv_channels*out_w)
        # to match reference (B, T, C, F) -> (B, T, C*F) i.e. channel-major last dim
        x_torch = ttnn.to_torch(x)
        # (1, 1, batch_size*out_h*out_w, conv_channels) -> (batch_size, out_h, out_w, conv_channels)
        x_torch = x_torch.reshape(batch_size, out_h, out_w, self.conv_channels)
        # (batch_size, out_h, conv_channels, out_w) -> (batch_size, out_h, conv_channels*out_w)
        x_torch = x_torch.permute(0, 1, 3, 2).reshape(batch_size, out_h, self.conv_channels * out_w)
        x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # Apply final linear projection
        x = ttnn.linear(
            x,
            self.tt_linear_weight,
            bias=self.tt_linear_bias,
            compute_kernel_config=self.compute_config,
        )

        return x, current_lengths


# Example usage
if __name__ == "__main__":
    # Initialize device
    dev = ttnn.open_device(0)

    # Create model
    model = TTNNConvSubsampling(dev, feat_in=128, feat_out=1024, conv_channels=256)

    # Test input (batch, time, feat_in)
    x_test = torch.randn(1, 744, 128)
    lengths_test = torch.tensor([743], dtype=torch.long)

    # Convert input to TTNN
    tt_x = ttnn.from_torch(x_test, dtype=ttnn.bfloat16, device=dev)

    # Forward
    tt_out, tt_lengths = model.forward(tt_x, lengths_test)

    # Convert back to torch for printing
    out_torch = ttnn.to_torch(tt_out)
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {out_torch.shape}")
    print(f"Input lengths: {lengths_test.tolist()}")
    print(f"Output lengths: {tt_lengths.tolist() if hasattr(tt_lengths, 'tolist') else tt_lengths}")

    ttnn.close_device(dev)
