import torch

import ttnn


class TtPatchTSMixerGatedAttention:
    def __init__(self, device, base_address: str, parameters: dict):
        self.device = device
        self.base_address = base_address

        self.weight = parameters[f"{base_address}.attn_layer.weight"]
        self.bias = parameters[f"{base_address}.attn_layer.bias"]

    def __call__(self, x):
        # x: TTNN tensor, last dimension = d_model
        y = ttnn.linear(x, self.weight, bias=self.bias)
        w = ttnn.softmax(y, dim=-1)
        return ttnn.multiply(x, w)


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
        self.pe = parameters[f"{self.base}.pe"]

    def __call__(self, x):
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
        self.gamma = parameters[f"{self.base}.norm.weight"]  # (1, 1, 1, D)
        self.beta = parameters[f"{self.base}.norm.bias"]  # (1, 1, 1, D)

    def __call__(self, x):
        # x: (B, C, N_p, D)
        return ttnn.layer_norm(x, weight=self.gamma, bias=self.beta, epsilon=self.eps)


class TtPatchTSMixerLayerNormDispatcher:
    """
    dispatcher that reuses the already-implemented LN/BN TT modules.

    Matches Pytorch semantics:
        - decides in __init__ based on norm_type
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
    """

    def __init__(self, device, base_address: str, parameters: dict, eps: float = 0.0):
        self.device = device
        self.base = base_address

        self.w1 = parameters[f"{self.base}.fc1.weight"]  # (1, 1, in, hidden)
        self.b1 = parameters[f"{self.base}.fc1.bias"]  # (1, 1, 1, hidden)
        self.w2 = parameters[f"{self.base}.fc2.weight"]  # (1, 1, hidden, out)
        self.b2 = parameters[f"{self.base}.fc2.bias"]  # (1, 1, 1, out)

    def __call__(self, x):
        # x: (..., in_features) -> rank-4 i PatchTSMixer
        x = ttnn.linear(x, self.w1, bias=self.b1, activation="gelu")
        x = ttnn.linear(x, self.w2, bias=self.b2)
        return x


class TtFeatureMixerBlock:
    """
    TTNN equivalent of FeatureMixerBlock.

    Expected x shape: (B, C, N_p, D) in TILE layout on device.
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
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        if self.use_gated_attn:
            x = self.gate(x)
        x = ttnn.add(x, residual)
        return x


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
        x = self.norm(x)

        # Move patches to last dim so MLP mixes patches:
        # (B, C, N_p, D) -> (B, C, D, N_p)
        x = ttnn.permute(x, (0, 1, 3, 2))

        # MLP over last dim (N_p)
        x = self.mlp(x)

        # Optional gated attention over patches (last dim = N_p)
        if self.use_gated_attn:
            x = self.gate(x)

        # Back (B, C, D, N_p) -> (B, C, Np, D)
        x = ttnn.permute(x, (0, 1, 3, 2))

        return ttnn.add(x, residual)


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

        self.mlp = TtPatchTSMixerMLP(device=device, base_address=f"{self.base}.mlp", parameters=parameters, eps=eps)

        if use_gated_attn:
            self.gate = TtPatchTSMixerGatedAttention(
                device=device,
                base_address=f"{self.base}.gate",
                parameters=parameters,
            )

    def __call__(self, x):
        residual = x

        x = self.norm(x)

        # Move channel to last dim (B, C, N_p, D) -> (B, D, N_p, C)
        x = ttnn.permute(x, (0, 3, 2, 1))

        if self.use_gated_attn:
            x = self.gate(x)  # gate over channels

        x = self.mlp(x)  # MLP over channels (last dim)

        # Back: (B, D, N_p, C) -> (B, C, N_p, D)
        x = ttnn.permute(x, (0, 3, 2, 1))
        return ttnn.add(x, residual)


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
    Output: (B, H, C) we will return rank-3 torch-like after to_torch, but internally keep rank-4.
    """

    def __init__(self, device, base_address: str, parameters: dict, *, prediction_length: int):
        self.device = device
        self.base = base_address
        self.H = prediction_length

        # weight should represent [Np*D, H] or [H, Np*D] depending on transpose_b usage.
        self.weight = parameters[f"{self.base}.proj.weight"]
        self.bias = parameters.get(f"{self.base}.proj.bias", None)

    def __call__(self, x):
        # x: (B, C, Np, D)
        # flatten last two dims -> (B, C, 1, Np*D)
        B, C, Np, D = x.shape
        x = ttnn.reshape(x, (B, C, 1, Np * D))

        # linear over last dim -> (B, C, 1, H)
        # Depending on how preprocess_linear formats weights, you may need transpose_b=True/False.
        y = ttnn.linear(x, self.weight, bias=self.bias)

        # (B, C, 1, H) -> (B, H, C)
        y = ttnn.permute(y, (0, 3, 2, 1))  # (B, H, 1, C)

        return y


class TtPatchTSMixerPatchify:
    """
    Bring-up version:
      - patchify/unfold done on host (torch)
      - output moved to device as a TTNN tensor

    Later do a full port to TTNN.

    Input torch shape expected: (B, L, C)  (HF-style)
    Output TTNN tensor: (B, C, N_patches, patch_length)
    """

    def __init__(self, *, context_length, patch_length, patch_stride):
        self.context_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride

        self.num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1
        new_len = patch_length + patch_stride * (self.num_patches - 1)
        self.sequence_start = context_length - new_len

    def __call__(self, x_torch: torch.Tensor, *, device, dtype=ttnn.bfloat16):
        # x_torch: (B, L, C)
        B, L, C = x_torch.shape
        if L != self.context_length:
            raise ValueError(f"Expected sequence length {self.context_length}, got {L}")

        # crop left side if needed
        x = x_torch[:, self.sequence_start :, :]  # (B, new_len, C)

        # unfold along time
        patches = x.unfold(dimension=1, size=self.patch_length, step=self.patch_stride)
        # patches: (B, N_patches, C, patch_length)

        # transpose to (B, C, N_patches, patch_length)
        patches = patches.transpose(1, 2).contiguous()

        # move to device
        return ttnn.from_torch(patches, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)


class TtPatchTSMixerEmbedding:
    """
    TTNN equivalent of PatchTSMixerEmbedding.

    Input torch:  past_values already transposed to (B, C, L) in the model.
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
        )

        # proj weight/bias
        self.weight = parameters[f"{self.base}.proj.weight"]
        self.bias = parameters.get(f"{self.base}.proj.bias", None)

    def __call__(self, x_torch: torch.Tensor, *, dtype=ttnn.bfloat16):
        """
        x_torch: (B, C, L) torch tensor (host)
        returns: TTNN tensor (B, C, Np, d_model)
        """
        # (B,C,L) -> (B,L,C) for patchify logic
        x_lc = x_torch.transpose(1, 2).contiguous()

        # patchify on host, move to device
        patches_tt = self.patchify(x_lc, device=self.device, dtype=dtype)  # (B,C,Np,patch_len) on device

        # linear over last dim patch_len -> d_model
        # reshape to rank-4 already, so we can call ttnn.linear directly:
        # (B,C,Np,patch_len) @ (patch_len,d_model) => (B,C,Np,d_model)
        out = ttnn.linear(patches_tt, self.weight, bias=self.bias)
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

    def __call__(self, past_values: torch.Tensor, *, dtype=ttnn.bfloat16):
        """
        past_values: torch tensor (B, L, C)
        returns: TTNN tensor (B, H, 1, C)  (squeeze dim=2 in torch)
        """
        B, L, C = past_values.shape
        assert L == self.context_length
        assert C == self.num_channels

        # match PyTorch: (B, L, C) -> (B, C, L) for embedding
        x_bcl = past_values.transpose(1, 2).contiguous()

        # 1) embedding: returns TT (B, C, Np, D)
        x = self.patch_embed(x_bcl, dtype=dtype)

        # 2) PE: (B, C, Np, D)
        x = self.pos_enc(x)

        # 3) mixer block
        x, _ = self.mixer_block(x, output_hidden_states=False)

        # 4) head: returns TT (B, H, 1, C)
        y = self.head(x)

        return y
