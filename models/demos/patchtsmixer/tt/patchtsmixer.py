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

    def __init__(self, device, base_address: str, parameters: dict):
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
        norm_type = ("norm_type" or "layernorm").lower()
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
