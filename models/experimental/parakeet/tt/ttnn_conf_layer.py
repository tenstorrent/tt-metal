import ttnn


# ============================================================
# FeedForward (Macaron)
# ============================================================


class TtConformerFeedForward:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x, parameters):
        x = ttnn.linear(x, parameters.linear1.weight, compute_kernel_config=self.compute_config)
        x = ttnn.silu(x)
        x = ttnn.linear(x, parameters.linear2.weight, compute_kernel_config=self.compute_config)
        return x


# ============================================================
# Convolution Module
# ============================================================
class TtConformerConvolution:
    def __init__(self, d_model, kernel_size, device, dtype):
        self.d_model = d_model
        self.padding = (kernel_size - 1) // 2
        self.device = device
        self.dtype = dtype

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,
            deallocate_activation=True,
            act_block_h_override=32,  # minimal to reduce L1 use (depthwise with kernel_size=31 can OOM)
        )

    def __call__(self, x, pad_mask, parameters):
        B, T, D = x.shape

        # conv1d reshapes input to (B, 1, input_length, in_channels) -> expect (B, T, C) layout
        # =====================================================
        # Pointwise Conv1: x (B, T, D) -> (B, T, 2D)
        # =====================================================

        x, out_length, _ = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=parameters.pointwise1.weight,
            bias_tensor=getattr(parameters.pointwise1, "bias", None),
            in_channels=D,
            out_channels=parameters.pointwise1.weight.shape[0],
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            batch_size=B,
            input_length=T,
            device=self.device,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (B, out_length, parameters.pointwise1.weight.shape[0]))

        # =====================================================
        # GLU: (B, T, 2D) -> (B, T, D)
        # =====================================================

        two_d = parameters.pointwise1.weight.shape[0]
        d_glu = two_d // 2
        x = ttnn.unsqueeze(x, -2)  # (B, T, 1, 2D)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        # Manual GLU: split on last dim, first_half * sigmoid(second_half)
        # (slice uses padded shape; ensure we split logical channel dim)
        s_a = (0, 0, 0, 0)
        e_a = (B, out_length, 1, d_glu)
        s_b = (0, 0, 0, d_glu)
        e_b = (B, out_length, 1, two_d)
        step = (1, 1, 1, 1)
        half_a = ttnn.slice(x, s_a, e_a, step)
        half_b = ttnn.slice(x, s_b, e_b, step)
        x = ttnn.multiply(half_a, ttnn.sigmoid(half_b))
        x = ttnn.permute(x, (0, 3, 1, 2))  # (B, D, T, 1)
        x = ttnn.squeeze(x, -1)  # (B, D, T) or may stay 4D
        channels = x.shape[1]

        # =====================================================
        # Depthwise Conv: conv1d expects (B, T, C) -> permute to (B, T, D)
        # =====================================================

        if len(x.shape) == 4:
            x = ttnn.permute(x, (0, 2, 1, 3))  # (B, T, D, 1)
            x = ttnn.squeeze(x, -1)  # (B, T, D)
        else:
            x = ttnn.permute(x, (0, 2, 1))  # (B, D, T) -> (B, T, D)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x, out_length, _ = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=parameters.depthwise.weight,
            bias_tensor=getattr(parameters.depthwise, "bias", None),
            in_channels=channels,
            out_channels=channels,
            kernel_size=2 * self.padding + 1,
            stride=1,
            padding=self.padding,
            groups=channels,
            batch_size=B,
            input_length=out_length,
            device=self.device,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (B, out_length, channels))

        # Apply mask
        if pad_mask is not None:
            mask = ttnn.unsqueeze(pad_mask, -1)  # (B, T, 1)
            mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.where(mask, 0, x)

        # (B, T, C) -> (B, C, T) for BatchNorm
        x = ttnn.permute(x, (0, 2, 1))

        # =====================================================
        # BatchNorm (expects 4D tilized stats: (1, C, 1, 1))
        # =====================================================

        bn_run_mean = ttnn.reshape(parameters.bn.running_mean, (1, channels, 1, 1))
        bn_run_var = ttnn.reshape(parameters.bn.running_var, (1, channels, 1, 1))
        bn_weight = ttnn.reshape(parameters.bn.weight, (1, channels, 1, 1))
        bn_bias = ttnn.reshape(parameters.bn.bias, (1, channels, 1, 1))
        bn_run_mean = ttnn.to_layout(bn_run_mean, ttnn.TILE_LAYOUT)
        bn_run_var = ttnn.to_layout(bn_run_var, ttnn.TILE_LAYOUT)
        bn_weight = ttnn.to_layout(bn_weight, ttnn.TILE_LAYOUT)
        bn_bias = ttnn.to_layout(bn_bias, ttnn.TILE_LAYOUT)

        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.batch_norm(
            input=ttnn.unsqueeze(x, -1),
            running_mean=bn_run_mean,
            running_var=bn_run_var,
            weight=bn_weight,
            bias=bn_bias,
            training=False,
            eps=1e-5,
        )

        x = ttnn.squeeze(x, -1)

        x = ttnn.silu(x)

        # =====================================================
        # Pointwise Conv2: conv1d expects (B, T, C) -> permute (B, D, T) to (B, T, D)
        # =====================================================

        x = ttnn.permute(x, (0, 2, 1))  # (B, D, T) -> (B, T, D)
        x, out_length, _ = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=parameters.pointwise2.weight,
            bias_tensor=getattr(parameters.pointwise2, "bias", None),
            in_channels=channels,
            out_channels=parameters.pointwise2.weight.shape[0],
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            batch_size=B,
            input_length=x.shape[1],
            device=self.device,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # convert to logical (B,T,D)

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        x = ttnn.reshape(x, (B, out_length, D))

        return x


# Conformer Layer (FULL PRODUCTION)
# ============================================================


# ============================================================
# Conformer Encoder (STACK OF LAYERS)
# ============================================================
