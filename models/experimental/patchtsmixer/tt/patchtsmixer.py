import ttnn

from .patchtsmixer_utils import apply_linear_height_sharded, make_1d_mcast_prog_config_for_height_sharded


class TtPatchTSMixerGatedAttention:
    """
    Stage 2 Optimization: L1 memory for weights and outputs
    """

    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        *,
        use_height_sharding: bool = True,
        min_k_tiles: int = 1,
    ):
        self.device = device
        self.base_address = base_address
        self.use_height_sharding = use_height_sharding
        self.min_k_tiles = min_k_tiles

        # Keep weights in DRAM (hardware multicasts to cores)
        self.weight = parameters[f"{base_address}.attn_layer.weight"]
        self.bias = parameters[f"{base_address}.attn_layer.bias"]

        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        # x shape: (B, C, Np, D)
        B = x.shape[0]
        C = x.shape[1]

        if self.use_height_sharding:
            y = apply_linear_height_sharded(
                x,
                self.weight,
                self.bias,
                M=B * C * x.shape[2],
                K=x.shape[3],
                out_shape=(B, C, x.shape[2], x.shape[3]),
                use_sharding=True,
                compute_config=self.compute_config,
                min_K_tiles=self.min_k_tiles,
            )
        else:
            y = ttnn.linear(
                x,
                self.weight,
                bias=self.bias,
                core_grid=ttnn.CoreGrid(y=min(B * C, 8), x=8),
                compute_kernel_config=self.compute_config,
            )
        # Softmax (let TTNN manage memory)
        w = ttnn.softmax(y, dim=-1)
        ttnn.deallocate(y)
        # Multiply (let TTNN manage memory)
        out = ttnn.multiply(x, w)
        ttnn.deallocate(w)
        return out


class TtPatchTSMixerPositionalEncoding:
    """
    TTNN equivalent of PatchTSMixerPositionalEncoding.

    Expects:
        x: (B, C, N_p, D) as TTNN tensor
        pe: stored as (1, 1, N_p, D) TTNN tensor for broadcast

    """

    def __init__(self, device, base_address: str, parameters: dict, *, num_patches: int, d_model: int):
        self.device = device
        self.base = base_address

        # Store PE in L1 (small tensor, fast broadcast)
        self.pe = ttnn.to_memory_config(parameters[f"{self.base}.pe"], ttnn.L1_MEMORY_CONFIG)

    def __call__(self, x):
        # Addition with input memory config preserved
        return ttnn.add(x, self.pe)


class TtPatchTSMixerBatchNorm:
    """
    Bn1d(d_model=d) applies to x reshaped as (B*C, D, N_p),
    then reshaped back to (B, C, N_p, D),

    """

    def __init__(self, device, base_address: str, parameters: dict, eps=1e-5):
        self.device = device
        self.base = base_address
        self.eps = eps

        # shapes: [1, D, 1, 1] as required by ttnn.bach_norm
        self.weight = parameters[f"{self.base}.norm.weight"]  # (1, D, 1, 1)
        self.bias = parameters[f"{self.base}.norm.bias"]  # (1, D, 1, 1)
        self.mean = parameters[f"{self.base}.norm.running_mean"]  # (1, D, 1, 1)
        self.var = parameters[f"{self.base}.norm.running_var"]  # (1, D, 1, 1)

    def __call__(self, x):
        B, C, N_p, D = x.shape
        # x is a TTNN tensor representing (B, C, N_p, D)
        # ttnn.batch_norm requires rank-4 TILE on device.

        # (B, C, N_p, D) -> (B*C, D, N_p, 1)
        y = ttnn.reshape(x, (B * C, N_p, D))  # (B * C, N_p, D)

        # (B*C, N_p, D) -> (B*C, D, N_p)
        y = ttnn.permute(y, (0, 2, 1))

        # Rank-4 requirement: (B*C, D, N_p) -> (B*C, D, N_p, 1)
        y = ttnn.unsqueeze(y, -1)

        # Apply batchNorm
        y = ttnn.batch_norm(
            y,
            running_mean=self.mean,
            running_var=self.var,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
            training=False,
        )

        # (B*C, D, N_p, 1) -> (B, C, N_p, D)
        y = ttnn.squeeze(y, -1)  # (B*C, D, N_p)
        y = ttnn.permute(y, (0, 2, 1))  # (B*C, N_p, D)
        y = ttnn.reshape(y, (B, C, N_p, D))
        return y


class TtPatchTSMixerLayerNorm:
    def __init__(self, device, base_address: str, parameters: dict, eps=1e-5):
        self.device = device
        self.base = base_address
        self.eps = eps

        # Store params in L1 (tiny, safe)
        self.gamma = ttnn.to_memory_config(parameters[f"{self.base}.norm.weight"], ttnn.L1_MEMORY_CONFIG)
        self.beta = ttnn.to_memory_config(parameters[f"{self.base}.norm.bias"], ttnn.L1_MEMORY_CONFIG)

    def __call__(self, x):
        # Layer norm - let TTNN manage output memory
        return ttnn.layer_norm(x, weight=self.gamma, bias=self.beta, epsilon=self.eps)


class TtPatchTSMixerLayerNormDispatcher:
    """
    dispatcher that reuses the already-implemented LN/BN TT modules.
    """

    def __init__(self, device, base_address: str, parameters: dict, norm_type: str = "LayerNorm", eps: float = 1e-5):
        norm_type = (norm_type or "layernorm").lower()
        self.is_batch = "batch" in norm_type

        if self.is_batch:
            # BN implementation
            self.impl = TtPatchTSMixerBatchNorm(
                device=device,
                base_address=base_address,
                parameters=parameters,
                eps=eps,
            )
        else:
            # LN implementation
            self.impl = TtPatchTSMixerLayerNorm(
                device=device,
                base_address=base_address,
                parameters=parameters,
                eps=eps,
            )

    def __call__(self, x, **kwargs):
        return self.impl(x, **kwargs)


class TtPatchTSMixerMLP:
    """
    TTNN equivalent of PatchTSMixerMLP in inference mode (dropout ignored for now).

    Expects input x to be rank-4 TILE on device.
    Applies:
        x = gelu(x @ W1 + b1)
        x = x @ W2 + b2

    Supports HEIGHT sharding optimization for improved parallelization.
    """

    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        eps: float = 0.0,
        use_height_sharding: bool = True,
        min_k_tiles_fc1: int = 1,
        min_k_tiles_fc2: int = 2,
    ):
        self.device = device
        self.base = base_address
        self.use_height_sharding = use_height_sharding
        self.min_k_tiles_fc1 = min_k_tiles_fc1
        self.min_k_tiles_fc2 = min_k_tiles_fc2

        # Keep weights in DRAM (hardware multicasts to cores during matmul)
        self.w1 = parameters[f"{self.base}.fc1.weight"]
        self.b1 = parameters[f"{self.base}.fc1.bias"]
        self.w2 = parameters[f"{self.base}.fc2.weight"]
        self.b2 = parameters[f"{self.base}.fc2.bias"]

        # Compute kernel configuration
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        # x shape: (B, C, Np, D) or (B, C, D, Np) depending on mixer
        B, C, dim3, K = x.shape

        # Calculate effective M dimension for matmul parallelization
        # For patch mixing: (B,C,D,Np) → M_eff = B*C*D
        # For feature mixing: (B,C,Np,D) → M_eff = B*C*Np
        M = B * C * dim3

        if not self.use_height_sharding:
            # Non-sharded path: use simple linear with activation fusion
            x1 = ttnn.linear(
                x,
                self.w1,
                bias=self.b1,
                activation="gelu",
                compute_kernel_config=self.compute_config,
            )
            x2 = ttnn.linear(
                x1,
                self.w2,
                bias=self.b2,
                compute_kernel_config=self.compute_config,
            )
            ttnn.deallocate(x1)
            return x2

        def _infer_out_features(weight, in_features: int) -> int:
            shape = [int(x) for x in weight.shape]
            if len(shape) == 2:
                out_f, in_f = shape
            elif len(shape) == 4:
                _, _, in_f, out_f = shape
            else:
                raise RuntimeError(f"Unsupported weight rank: weight={weight.shape}")

            if in_f == in_features:
                return out_f
            if out_f == in_features:
                return in_f
            raise RuntimeError(f"Can't infer out_features: weight={weight.shape}, in_features={in_features}")

        # HEIGHT-sharded path: shard once, keep sharded through MLP
        expansion_K = _infer_out_features(self.w1, K)
        out_K = _infer_out_features(self.w2, expansion_K)

        # FC1 sharded (keep sharded)
        y = apply_linear_height_sharded(
            x,
            self.w1,
            self.b1,
            M=M,
            K=K,
            out_shape=(1, 1, M, expansion_K),  # Keep in 2D for sharded ops
            use_sharding=True,
            compute_config=self.compute_config,
            min_K_tiles=self.min_k_tiles_fc1,
            return_sharded=True,  # Keep sharded for GELU
            activation="gelu",
            program_config_factory=lambda **kwargs: make_1d_mcast_prog_config_for_height_sharded(
                nc=kwargs["nc"],
                shard_shape=kwargs["shard_shape"],
                out_k=kwargs["out_k"],
                out_subblock_h=1,
            ),
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU),
        )

        # y_gelu = ttnn.gelu(y)
        # ttnn.deallocate(y)
        # y = y_gelu

        y2 = apply_linear_height_sharded(
            y,
            self.w2,
            self.b2,
            M=M,
            K=expansion_K,
            out_shape=(1, 1, M, out_K),
            use_sharding=True,
            compute_config=self.compute_config,
            min_K_tiles=self.min_k_tiles_fc2,
            return_sharded=False,
        )
        ttnn.deallocate(y)

        x_out = ttnn.reshape(y2, (B, C, dim3, out_K))
        return x_out


class TtFeatureMixerBlock:
    """
    TTNN equivalent of FeatureMixerBlock.

    Expected x shape: (B, C, N_p, D) in TILE layout on device.
    Uses HEIGHT_SHARDED to parallelize along B*C dimension for independent sequence processing.
    """

    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        d_model: int,
        norm_type: str = "LayerNorm",
        use_gated_attn: bool = False,
        eps: float = 1e-5,
    ):
        self.device = device
        self.base = base_address
        self.use_gated_attn = use_gated_attn
        self.d_model = d_model

        # Submodules use base-addressed paths
        self.norm = TtPatchTSMixerLayerNormDispatcher(
            device=device,
            base_address=f"{self.base}.norm",
            parameters=parameters,
            norm_type="layer_norm",
            eps=eps,
        )

        self.mlp = TtPatchTSMixerMLP(
            device=device,
            base_address=f"{self.base}.mlp",
            parameters=parameters,
        )

        if use_gated_attn:
            self.gate = TtPatchTSMixerGatedAttention(
                device=device, base_address=f"{self.base}.gate", parameters=parameters
            )

    def __call__(self, x):
        # Input: (B, C, Np, D)

        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        if self.use_gated_attn:
            x = self.gate(x)
        out = ttnn.add(x, residual)
        ttnn.deallocate(x)
        return out


class TtPatchMixerBlock:
    """
    TTNN equivalent of PatchMixerBlock.

    Input shape: (B, C, N_p, D)
    Mixes over patch dimension N_p by moving it to the last dim.
    """

    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        *,
        norm_type: str = "LayerNorm",
        use_gated_attn: bool = False,
        eps: float = 1e-5,
    ):
        self.device = device
        self.base = base_address
        self.use_gated_attn = use_gated_attn

        self.norm = TtPatchTSMixerLayerNormDispatcher(
            device=device,
            base_address=f"{self.base}.norm",
            parameters=parameters,
            norm_type=norm_type,
            eps=eps,
        )

        self.mlp = TtPatchTSMixerMLP(
            device=device,
            base_address=f"{self.base}.mlp",
            parameters=parameters,
            eps=eps,
        )

        if use_gated_attn:
            self.gate = TtPatchTSMixerGatedAttention(
                device=device,
                base_address=f"{self.base}.gate",
                parameters=parameters,
            )

    def __call__(self, x):
        residual = x

        # (B, C, N_p, D)
        x_norm = self.norm(x)

        # Move patches to last dim so MLP mixes patches:
        # (B, C, N_p, D) -> (B, C, D, N_p)
        x = ttnn.permute(x_norm, (0, 1, 3, 2))
        ttnn.deallocate(x_norm)

        # MLP over last dim (N_p)
        x = self.mlp(x)

        # Optional gated attention over patches (last dim = N_p)
        if self.use_gated_attn:
            x = self.gate(x)

        # Back (B, C, D, N_p) -> (B, C, Np, D)
        x_perm = ttnn.permute(x, (0, 1, 3, 2))
        ttnn.deallocate(x)

        out = ttnn.add(x_perm, residual)
        ttnn.deallocate(x_perm)
        return out


class TtPatchTSMixerChannelFeatureMixerBlock:
    """
    TTNN equivalent of PatchTSMixerChannelFeatureMixerBlock

    input shape: (B, C, N_p, D)
    Mixes over channel dim C by permuting to (B, D, N_p, C) and applying over last dim
    """

    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        *,
        d_model: int,
        num_channels: str,
        expansion: int,
        norm_type: str = "LayerNorm",
        use_gated_attn: bool = False,
        eps: float = 1e-5,
    ):
        self.device = device
        self.base = base_address
        self.use_gated_attn = use_gated_attn

        self.norm = TtPatchTSMixerLayerNormDispatcher(
            device=device,
            base_address=f"{self.base}.norm",
            parameters=parameters,
            norm_type=norm_type,
            eps=eps,
        )

        self.mlp = TtPatchTSMixerMLP(
            device=device,
            base_address=f"{self.base}.mlp",
            parameters=parameters,
            eps=eps,
            min_k_tiles_fc2=1,
        )

        if use_gated_attn:
            self.gate = TtPatchTSMixerGatedAttention(
                device=device,
                base_address=f"{self.base}.gate",
                parameters=parameters,
            )

    def __call__(self, x):
        residual = x

        x_norm = self.norm(x)

        # Move channel to last dim (B, C, N_p, D) -> (B, D, N_p, C)
        x = ttnn.permute(x_norm, (0, 3, 2, 1))
        ttnn.deallocate(x_norm)

        if self.use_gated_attn:
            x = self.gate(x)  # gate over channels

        x = self.mlp(x)  # MLP over channels (last dim)

        # Back: (B, D, N_p, C) -> (B, C, N_p, D)
        x_perm = ttnn.permute(x, (0, 3, 2, 1))
        ttnn.deallocate(x)

        out = ttnn.add(x_perm, residual)
        ttnn.deallocate(x_perm)
        return out


class TtPatchTSMixerLayer:
    """
    TTNN equivalent of PatchTSMixerLayer.
    Input/Output shape: (B, C, Np, D)
    """

    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        *,
        num_patches: int,
        d_model: int,
        num_channels: int,
        mode: str = "common_channel",
        norm_type: str = "LayerNorm",
        expansion: int = 2,
        use_gated_attn: bool = False,
        eps: float = 1e-5,
    ):
        self.device = device
        self.base = base_address
        self.mode = mode

        # Optional channel mixer (only when mode == "mix_channel")
        self.channel_mixer = None
        if mode == "mix_channel":
            self.channel_mixer = TtPatchTSMixerChannelFeatureMixerBlock(
                device=device,
                base_address=f"{self.base}.channel_mixer",
                parameters=parameters,
                num_channels=num_channels,
                d_model=num_channels,
                norm_type=norm_type,
                expansion=expansion,
                use_gated_attn=use_gated_attn,
                eps=eps,
            )

        # Patch mixer
        self.patch_mixer = TtPatchMixerBlock(
            device=device,
            base_address=f"{self.base}.patch_mixer",
            parameters=parameters,
            norm_type=norm_type,
            use_gated_attn=use_gated_attn,
            eps=eps,
        )

        # Feature mixer
        self.feature_mixer = TtFeatureMixerBlock(
            device=device,
            base_address=f"{self.base}.feature_mixer",
            parameters=parameters,
            d_model=d_model,
            norm_type=norm_type,
            use_gated_attn=use_gated_attn,
            eps=eps,
        )

    def __call__(self, x):
        if self.channel_mixer is not None:
            x = self.channel_mixer(x)
        x = self.patch_mixer(x)
        x = self.feature_mixer(x)
        return x


class TtPatchTSMixerBlock:
    """
    TTNN equivalent of PatchTSMixerBlock:
      - runs a list of TtPatchTSMixerLayer modules
      - optionally collects hidden states
    """

    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        *,
        num_layers: int,
        layer_kwargs: dict,
        norm_type: str = "LayerNorm",
    ):
        self.device = device
        self.base = base_address
        self.num_layers = num_layers

        self.layers = []
        for i in range(num_layers):
            layer_base = f"{self.base}.layers.{i}"
            self.layers.append(
                TtPatchTSMixerLayer(
                    device=device,
                    base_address=layer_base,
                    parameters=parameters,
                    norm_type=norm_type,
                    **layer_kwargs,
                )
            )

    def __call__(self, hidden, *, output_hidden_states: bool = False):
        all_hidden_states = [] if output_hidden_states else None
        x = hidden
        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                all_hidden_states.append(x)
        return x, all_hidden_states


import ttnn


class TtPatchTSMixerForecastHead:
    """
    TTNN equivalent of PatchTSMixerForecastHead.

    Input:  (B, C, Np, D)  rank-4
    Output: (B, H, C).

    """

    def __init__(self, device, base_address: str, parameters: dict, *, prediction_length: int):
        self.device = device
        self.base = base_address
        self.H = prediction_length

        # Keep weights in DRAM (can be large for forecasting)
        self.weight = parameters[f"{self.base}.proj.weight"]
        self.bias = parameters.get(f"{self.base}.proj.bias", None)

    def __call__(self, x):
        B, C, Np, D = x.shape

        # Flatten - let TTNN manage (can materialize, can be large)
        x_flat = ttnn.reshape(x, (B, C, 1, Np * D))
        ttnn.deallocate(x)

        # Linear - let TTNN manage
        y = ttnn.linear(x_flat, self.weight, bias=self.bias)
        ttnn.deallocate(x_flat)

        # Permute - let TTNN manage
        out = ttnn.permute(y, (0, 3, 2, 1))
        ttnn.deallocate(y)
        return out


class TtPatchTSMixerLinearHead:
    """
    TTNN equivalent of PatchTSMixerLinearHead for Classification and Regression.

    Input:  (B, C, Np, D)
    Output: (B, num_targets)
    """

    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        *,
        num_targets: int,
        head_aggregation: str = None,  # None, "use_last", "max_pool", "avg_pool"
        output_range: tuple = None,  # (min, max) for sigmoid scaling
    ):
        self.device = device
        self.base = base_address
        self.head_aggregation = head_aggregation
        self.output_range = output_range
        self.num_targets = num_targets

        # Projection weight/bias
        # Shape depends on aggregation:
        # - None: (num_targets, C*D*Np)
        # - aggregation: (num_targets, C*D)
        self.projection_weight = parameters[f"{self.base}.projection.weight"]
        self.projection_bias = parameters[f"{self.base}.projection.bias"]

    def __call__(self, x):
        """
        Args:
            x: TTNN tensor (B, C, Np, D)

        Returns:
            ttnn.Tensor (B, num_targets)
        """
        B, C, Np, D = x.shape

        # Transpose: (B, C, Np, D) -> (B, C, D, Np)
        x_perm = ttnn.permute(x, (0, 1, 3, 2))
        ttnn.deallocate(x)

        # Apply aggregation over patch dimension (last dim)
        if self.head_aggregation == "use_last":
            # Take last patch: (B, C, D, Np) -> (B, C, D, 1)
            # TTNN slice syntax: tensor[start:stop] along dim
            x_agg = x_perm[:, :, :, Np - 1 : Np]  # (B, C, D, 1)
            ttnn.deallocate(x_perm)
            x = ttnn.squeeze(x_agg, -1)  # (B, C, D)
            ttnn.deallocate(x_agg)

        elif self.head_aggregation == "max_pool":
            # Max pool over patches: (B, C, D, Np) -> (B, C, D)
            x = ttnn.max(x_perm, dim=-1)
            ttnn.deallocate(x_perm)

        elif self.head_aggregation == "avg_pool":
            # Average pool over patches: (B, C, D, Np) -> (B, C, D)
            x = ttnn.mean(x_perm, dim=-1)
            ttnn.deallocate(x_perm)
        else:
            x = x_perm

        # else: head_aggregation is None, keep all patches (B, C, D, Np)

        # Flatten: (B, C, D, ...) -> (B, C*D*...)
        if self.head_aggregation is None:
            # (B, C, D, Np) -> (B, C*D*Np)
            x_flat = ttnn.reshape(x, (B, C * D * Np))
        else:
            # (B, C, D) -> (B, C*D)
            x_flat = ttnn.reshape(x, (B, C * D))
        ttnn.deallocate(x)

        # Linear projection: (B, features) -> (B, num_targets)
        x = ttnn.linear(x_flat, self.projection_weight, bias=self.projection_bias)
        ttnn.deallocate(x_flat)

        # Optional: Apply sigmoid + range scaling
        if self.output_range is not None:
            min_val, max_val = self.output_range
            x = ttnn.sigmoid(x, vector_mode=4, fast_and_approximate_mode=True)
            x = x * (max_val - min_val) + min_val

        return x  # (B, num_targets)


class TtPatchTSMixerPretrainHead:
    """
    TTNN equivalent of PatchTSMixerPretrainHead for self-supervised pre-training.

    Projects from d_model back to patch_length to reconstruct masked patches.

    Input:  (B, C, Np, D)
    Output: (B, C, Np, patch_length)
    """

    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        *,
        patch_length: int,
    ):
        self.device = device
        self.base = base_address
        self.patch_length = patch_length

        # Projection: d_model -> patch_length
        self.projection_weight = parameters[f"{self.base}.projection.weight"]
        self.projection_bias = parameters[f"{self.base}.projection.bias"]

    def __call__(self, x):
        """
        x: TTNN tensor (B, C, Np, D)
        returns: ttnn tensor (B, C, Np, patch_length)
        """
        # Linear projection: (B, C, Np, D) @ (D, patch_length) -> (B, C, Np, patch_length)
        return ttnn.linear(x, self.projection_weight, bias=self.projection_bias)


class TtPatchTSMixerPatchify:
    """
    Input tensor shape expected: (B, L, C)  (HF-style)
    Output TTNN tensor: (B, C, N_patches, patch_length)

    """

    def __init__(self, *, device, context_length, patch_length, patch_stride):
        self.device = device
        self.context_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride

        self.num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1
        new_len = patch_length + patch_stride * (self.num_patches - 1)
        self.sequence_start = context_length - new_len
        self.new_len = new_len

        # Cache indices in L1 with TILE_LAYOUT
        self.idx2 = self._build_idx2()

        # Cache for idx4 tensor per (B, C) shape to avoid repeated expansion
        self._idx4_cache = {}

        # Memory configs: L1 for small/hot data, DRAM for large temporary tensors
        self.l1_mem_config = ttnn.L1_MEMORY_CONFIG
        self.dram_mem_config = ttnn.DRAM_MEMORY_CONFIG

    def _build_idx2(self):
        P = self.patch_length
        N_p = self.num_patches
        S = self.patch_stride

        offsets = ttnn.arange(0, P, dtype=ttnn.uint32, device=self.device)
        patch_ids = ttnn.arange(0, N_p, dtype=ttnn.uint32, device=self.device)
        patch_starts = patch_ids * S

        idx2 = ttnn.reshape(patch_starts, (N_p, 1)) + ttnn.reshape(offsets, (1, P))

        # Store in L1 and TILE_LAYOUT
        idx2 = ttnn.to_layout(idx2, ttnn.TILE_LAYOUT)
        idx2 = ttnn.to_memory_config(idx2, ttnn.L1_MEMORY_CONFIG)
        return idx2

    def _get_or_create_idx4(self, B: int, C: int):
        """
        Get cached idx4 tensor for given (B, C) shape, or create and cache it.
        Avoids repeating expensive idx4 expansion on every forward pass.
        """
        key = (B, C)
        if key in self._idx4_cache:
            return self._idx4_cache[key]

        Np = self.num_patches
        P = self.patch_length

        # Build idx4: (1, Np, P, 1) -> (B, Np, P, C)
        # Let TTNN manage memory for idx4 (can be moderate size)
        idx4 = ttnn.reshape(self.idx2, (1, Np, P, 1))
        idx4 = ttnn.repeat(idx4, (B, 1, 1, C))

        self._idx4_cache[key] = idx4
        return idx4

    def __call__(self, x):
        B, L, C = x.shape if len(x.shape) == 3 else (x.shape[0], x.shape[2], x.shape[3])

        if len(x.shape) == 3:
            x = ttnn.reshape(x, (B, 1, L, C))

        # Slice directly (output stays in input's memory)
        x = ttnn.slice(x, (0, 0, self.sequence_start, 0), (B, 1, self.sequence_start + self.new_len, C))

        # Ensure TILE_LAYOUT (let TTNN choose memory location)
        if x.get_layout() != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        Np = self.num_patches
        P = self.patch_length

        # NOTE: TTNN gather limitation - requires matching dimensions
        # Ideally: gather with x=(B,1,L,C) and idx=(B,Np,P,C) using broadcasting
        # Reality: gather requires x=(B,Np,L,C) to match idx=(B,Np,P,C) on Np dimension
        # This creates 63x memory expansion: (B,1,L,C) -> (B,63,L,C) for typical config
        # TODO: Request TTNN enhancement to support broadcasting in gather operation

        # Large temporary expansion → DRAM (avoid L1 thrashing)
        # Example: (2,1,512,7) → (2,63,512,7) = 7KB → 441KB
        x = ttnn.repeat(x, (1, Np, 1, 1), memory_config=self.dram_mem_config)

        # Use cached idx4 to avoid repeated expansion
        idx4 = self._get_or_create_idx4(B, C)

        # Gather - let TTNN manage memory (output consumed by embedding)
        y = ttnn.gather(x, dim=2, index=idx4)
        ttnn.deallocate(x)  # Free huge repeated tensor immediately

        # Final permute - let TTNN manage memory
        y = ttnn.permute(y, (0, 3, 1, 2))
        return y


class TtPatchTSMixerEmbedding:
    """
    TTNN equivalent of PatchTSMixerEmbedding.

    Input tensor:  past_values already transposed to (B, C, L) in the model.
    Output TTNN:  (B, C, Np, d_model)

    """

    def __init__(
        self, device, base_address: str, parameters: dict, *, context_length, patch_length, patch_stride, d_model
    ):
        self.device = device
        self.base = base_address
        self.context_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.d_model = d_model

        self.patchify = TtPatchTSMixerPatchify(
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            device=device,
        )

        # Keep weight in DRAM (embedding projection can be moderate-large)
        self.weight = parameters[f"{self.base}.proj.weight"]
        self.bias = parameters.get(f"{self.base}.proj.bias", None)

    def __call__(self, x: ttnn.Tensor, *, dtype=ttnn.bfloat16):
        """
        x: (B, C, L) ttnn tensor (host)
        returns: TTNN tensor (B, C, Np, d_model)
        """

        # Permute - let TTNN manage memory
        x_lc = ttnn.permute(x, (0, 2, 1))

        # Patchify
        patches_tt = self.patchify(x_lc)
        ttnn.deallocate(x_lc)

        # Linear projection: (B, C, Np, patch_length) @ (patch_length, d_model)
        # Use multi-core parallelization for input embedding
        B, C, Np, P = patches_tt.shape
        M_eff = B * C * Np  # Effective batch dimension for parallelization
        out = ttnn.linear(
            patches_tt,
            self.weight,
            bias=self.bias,
            core_grid=ttnn.CoreGrid(y=min(M_eff, 8), x=8),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(patches_tt)
        return out


class TtPatchTSMixerModelForForecasting:
    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        *,
        context_length: int,
        prediction_length: int,
        patch_length: int,
        patch_stride: int,
        num_channels: int,
        d_model: int,
        num_layers: int,
        mode: str = "common_channel",
        expansion: int = 2,
        use_gated_attn: bool = False,
        eps: float = 1e-5,
    ):
        self.device = device
        self.base = base_address

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_channels = num_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.mode = mode
        self.expansion = expansion
        self.use_gated_attn = use_gated_attn
        self.eps = eps

        # HF-compatible num_patches
        self.num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1

        # 1) patch embedding
        self.patch_embed = TtPatchTSMixerEmbedding(
            device=device,
            base_address=f"{self.base}.patch_embed",
            parameters=parameters,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
        )

        # 2) positional encoding
        self.pos_enc = TtPatchTSMixerPositionalEncoding(
            device=device,
            base_address=f"{self.base}.pos_enc",
            parameters=parameters,  # depends on your PE design (buffer/param)
            num_patches=self.num_patches,
            d_model=d_model,
        )

        # 3) mixer stack
        layer_kwargs = dict(
            num_patches=self.num_patches,
            d_model=d_model,
            num_channels=num_channels,
            mode=mode,
            expansion=expansion,
            use_gated_attn=use_gated_attn,
            eps=eps,
        )
        self.mixer_block = TtPatchTSMixerBlock(
            device=device,
            base_address=f"{self.base}.mixer_block",
            parameters=parameters,
            num_layers=num_layers,
            layer_kwargs=layer_kwargs,
            norm_type="LayerNorm",  # or pass through if you support BN dispatch
        )

        # 4) head
        self.head = TtPatchTSMixerForecastHead(
            device=device,
            base_address=f"{self.base}.head",
            parameters=parameters,
            prediction_length=prediction_length,
        )

    def __call__(self, past_values: ttnn.Tensor, *, dtype=ttnn.bfloat16):
        """
        past_values: ttnn tensor (B, L, C)
        returns: TTNN tensor (B, H, 1, C)
        """
        B, L, C = past_values.shape
        assert L == self.context_length
        assert C == self.num_channels

        # Transpose: (B, L, C) -> (B, C, L)
        x_bcl = ttnn.permute(past_values, (0, 2, 1))

        # 1) embedding: returns TT (B, C, Np, D)
        x = self.patch_embed(x_bcl, dtype=dtype)

        # 2) PE: (B, C, Np, D)
        x = self.pos_enc(x)

        # 3) mixer block
        x, _ = self.mixer_block(x, output_hidden_states=False)

        # 4) head: returns TT (B, H, 1, C)
        y = self.head(x)

        return y


class TtPatchTSMixerForRegression:
    """
    PatchTSMixer for time series regression.

    Returns continuous values for num_targets regression targets.
    Optionally constrains outputs to a specified range using sigmoid.
    """

    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        *,
        context_length: int,
        patch_length: int,
        patch_stride: int,
        num_channels: int,
        d_model: int,
        num_layers: int,
        num_targets: int,
        output_range: tuple = None,  # (min, max)
        mode: str = "common_channel",
        expansion: int = 2,
        use_gated_attn: bool = False,
        head_aggregation: str = "avg_pool",
        eps: float = 1e-5,
    ):
        self.device = device
        self.base = base_address

        self.context_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_channels = num_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_targets = num_targets
        self.output_range = output_range
        self.mode = mode
        self.expansion = expansion
        self.use_gated_attn = use_gated_attn
        self.head_aggregation = head_aggregation
        self.eps = eps

        # HF-compatible num_patches
        self.num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1

        # 1) patch embedding
        embed_addr = f"{self.base}.embedding" if self.base else "embedding"
        self.patch_embed = TtPatchTSMixerEmbedding(
            device=device,
            base_address=embed_addr,
            parameters=parameters,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
        )

        # 2) positional encoding
        pos_addr = f"{self.base}.pos_encoder" if self.base else "pos_encoder"
        self.pos_enc = TtPatchTSMixerPositionalEncoding(
            device=device,
            base_address=pos_addr,
            parameters=parameters,
            num_patches=self.num_patches,
            d_model=d_model,
        )

        # 3) encoder (mixer block)
        layer_kwargs = dict(
            num_patches=self.num_patches,
            d_model=d_model,
            num_channels=num_channels,
            mode=mode,
            expansion=expansion,
            use_gated_attn=use_gated_attn,
            eps=eps,
        )
        encoder_addr = f"{self.base}.encoder" if self.base else "encoder"
        self.encoder = TtPatchTSMixerBlock(
            device=device,
            base_address=encoder_addr,
            parameters=parameters,
            num_layers=num_layers,
            layer_kwargs=layer_kwargs,
            norm_type="LayerNorm",
        )

        # 4) regression head
        head_addr = f"{self.base}.head" if self.base else "head"
        self.head = TtPatchTSMixerLinearHead(
            device=device,
            base_address=head_addr,
            parameters=parameters,
            num_targets=num_targets,
            head_aggregation=head_aggregation,
            output_range=output_range,
        )

    def __call__(self, past_values: ttnn.Tensor, *, dtype=ttnn.bfloat16):
        """
        past_values: ttnn tensor (B, L, C)
        returns: ttnn tensor (B, num_targets)
        """
        B, L, C = past_values.shape
        assert L == self.context_length
        assert C == self.num_channels

        # match PyTorch: (B, L, C) -> (B, C, L) for embedding
        x_bcl = ttnn.permute(past_values, (0, 2, 1))

        # 1) embedding: returns TT (B, C, Np, D)
        x = self.patch_embed(x_bcl, dtype=dtype)

        # 2) PE: (B, C, Np, D)
        x = self.pos_enc(x)

        # 3) encoder
        x, _ = self.encoder(x, output_hidden_states=False)

        # 4) head: returns ttnn Tensor (B, num_targets)
        predictions = self.head(x)

        return predictions


class TtPatchTSMixerForTimeSeriesClassification:
    """
    PatchTSMixer for time series classification.

    Returns class logits for num_classes classification targets.
    Typically uses avg_pool aggregation over patches.
    """

    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        *,
        context_length: int,
        patch_length: int,
        patch_stride: int,
        num_channels: int,
        d_model: int,
        num_layers: int,
        num_classes: int,
        mode: str = "common_channel",
        expansion: int = 2,
        use_gated_attn: bool = False,
        head_aggregation: str = "avg_pool",
        eps: float = 1e-5,
    ):
        self.device = device
        self.base = base_address

        self.context_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_channels = num_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.mode = mode
        self.expansion = expansion
        self.use_gated_attn = use_gated_attn
        self.head_aggregation = head_aggregation
        self.eps = eps

        # HF-compatible num_patches
        self.num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1

        # 1) patch embedding
        embed_addr = f"{self.base}.embedding" if self.base else "embedding"
        self.patch_embed = TtPatchTSMixerEmbedding(
            device=device,
            base_address=embed_addr,
            parameters=parameters,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
        )

        # 2) positional encoding
        pos_addr = f"{self.base}.pos_encoder" if self.base else "pos_encoder"
        self.pos_enc = TtPatchTSMixerPositionalEncoding(
            device=device,
            base_address=pos_addr,
            parameters=parameters,
            num_patches=self.num_patches,
            d_model=d_model,
        )

        # 3) encoder (mixer block)
        layer_kwargs = dict(
            num_patches=self.num_patches,
            d_model=d_model,
            num_channels=num_channels,
            mode=mode,
            expansion=expansion,
            use_gated_attn=use_gated_attn,
            eps=eps,
        )
        encoder_addr = f"{self.base}.encoder" if self.base else "encoder"
        self.encoder = TtPatchTSMixerBlock(
            device=device,
            base_address=encoder_addr,
            parameters=parameters,
            num_layers=num_layers,
            layer_kwargs=layer_kwargs,
            norm_type="LayerNorm",
        )

        # 4) classification head
        head_addr = f"{self.base}.head" if self.base else "head"
        self.head = TtPatchTSMixerLinearHead(
            device=device,
            base_address=head_addr,
            parameters=parameters,
            num_targets=num_classes,
            head_aggregation=head_aggregation,
            output_range=None,  # No output range for classification
        )

    def __call__(self, past_values: ttnn.Tensor, *, dtype=ttnn.bfloat16):
        """
        past_values: ttnn tensor (B, L, C)
        returns: ttnn tensor (B, num_classes)
        """
        B, L, C = past_values.shape
        assert L == self.context_length
        assert C == self.num_channels

        # match PyTorch: (B, L, C) -> (B, C, L) for embedding
        x_bcl = ttnn.permute(past_values, (0, 2, 1))

        # 1) embedding: returns TT (B, C, Np, D)
        x = self.patch_embed(x_bcl, dtype=dtype)

        # 2) PE: (B, C, Np, D)
        x = self.pos_enc(x)

        # 3) encoder
        x, _ = self.encoder(x, output_hidden_states=False)

        # 4) head: returns (B, num_classes)
        logits = self.head(x)

        return logits


class TtPatchTSMixerForPretraining:
    """
    PatchTSMixer for self-supervised pre-training via masked patch prediction.

    Reconstructs masked patches from their context.
    Returns reconstructed patches of shape (B, C, Np, patch_length).
    """

    def __init__(
        self,
        device,
        base_address: str,
        parameters: dict,
        *,
        context_length: int,
        patch_length: int,
        patch_stride: int,
        num_channels: int,
        d_model: int,
        num_layers: int,
        mode: str = "common_channel",
        expansion: int = 2,
        use_gated_attn: bool = False,
        eps: float = 1e-5,
    ):
        self.device = device
        self.base = base_address

        self.context_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_channels = num_channels
        self.d_model = d_model
        self.num_layers = num_layers
        self.mode = mode
        self.expansion = expansion
        self.use_gated_attn = use_gated_attn
        self.eps = eps

        # HF-compatible num_patches
        self.num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1

        # 1) patch embedding
        embed_addr = f"{self.base}.embedding" if self.base else "embedding"
        self.patch_embed = TtPatchTSMixerEmbedding(
            device=device,
            base_address=embed_addr,
            parameters=parameters,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
        )

        # 2) positional encoding
        pos_addr = f"{self.base}.pos_encoder" if self.base else "pos_encoder"
        self.pos_enc = TtPatchTSMixerPositionalEncoding(
            device=device,
            base_address=pos_addr,
            parameters=parameters,
            num_patches=self.num_patches,
            d_model=d_model,
        )

        # 3) encoder (mixer block)
        layer_kwargs = dict(
            num_patches=self.num_patches,
            d_model=d_model,
            num_channels=num_channels,
            mode=mode,
            expansion=expansion,
            use_gated_attn=use_gated_attn,
            eps=eps,
        )
        encoder_addr = f"{self.base}.encoder" if self.base else "encoder"
        self.encoder = TtPatchTSMixerBlock(
            device=device,
            base_address=encoder_addr,
            parameters=parameters,
            num_layers=num_layers,
            layer_kwargs=layer_kwargs,
            norm_type="LayerNorm",
        )

        # 4) pre-training head
        head_addr = f"{self.base}.head" if self.base else "head"
        self.head = TtPatchTSMixerPretrainHead(
            device=device,
            base_address=head_addr,
            parameters=parameters,
            patch_length=patch_length,
        )

    def __call__(self, past_values: ttnn.Tensor, *, dtype=ttnn.bfloat16):
        """
        past_values: ttnn tensor (B, L, C)
        returns: ttnn tensor (B, C, Np, patch_length)
        """
        B, L, C = past_values.shape
        assert L == self.context_length
        assert C == self.num_channels

        # match PyTorch: (B, L, C) -> (B, C, L) for embedding
        x_bcl = ttnn.permute(past_values, (0, 2, 1))

        # 1) embedding: returns TT (B, C, Np, D)
        x = self.patch_embed(x_bcl, dtype=dtype)

        # 2) PE: (B, C, Np, D)
        x = self.pos_enc(x)

        # 3) encoder
        x, _ = self.encoder(x, output_hidden_states=False)

        # 4) head: returns tensor (B, C, Np, patch_length)
        reconstructed = self.head(x)

        return reconstructed
