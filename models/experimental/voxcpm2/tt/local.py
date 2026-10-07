# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident VoxCPM2 local modules, derived from OpenBMB/VoxCPM.

Reference source: f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629 (Apache-2.0).
Weights and sinusoidal frequency constants cross the host boundary at construction;
learned projections, attention, timestep trigonometry and quantization run in TTNN.
Selected BF16 cases passed real-device PCC checks in validation/RESULTS.md.
Short patch sequences use logical shapes.
"""

from copy import deepcopy
from math import log


def make_local_config(lm_config, component_config):
    """Apply the same config mutations as VoxCPM2Model for LocEnc / LocDiT.

    In particular, KV-head count, RoPE and muP scaling remain inherited from
    the base LM. They must not be silently replaced with generic defaults.
    """
    config = deepcopy(dict(lm_config))
    for source, target in (
        ("hidden_dim", "hidden_size"),
        ("ffn_dim", "intermediate_size"),
        ("num_heads", "num_attention_heads"),
        ("num_layers", "num_hidden_layers"),
    ):
        value = component_config[source]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{source} must be a positive integer")
        config[target] = value
    config["kv_channels"] = component_config.get("kv_channels")
    config["vocab_size"] = 0
    _validate_local_config(config)
    return config


def _validate_local_config(config):
    if config.get("vocab_size") != 0:
        raise ValueError("Local encoder / DiT config requires vocab_size=0")
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
        raise ValueError(
            "Local attention heads must be divisible by inherited KV heads"
        )
    if config.get("kv_channels") is None and config["hidden_size"] % heads:
        raise ValueError(
            "hidden_size must be divisible by attention heads without kv_channels"
        )


def _runtime():
    # Lazy import keeps config validation available without a TT installation.
    import ttnn

    from .minicpm import TtMiniCPMModel
    from .ops import TtLinear, upload

    return ttnn, TtMiniCPMModel, TtLinear, upload


class TtLocalEncoder:
    """[batch, chunks, patch, features] -> [batch, chunks, hidden]."""

    def __init__(self, config, state, prefix, device, dtype, input_dim=64):
        _validate_local_config(config)
        self.hidden_size = config["hidden_size"]
        self.input_dim = input_dim
        ttnn, model, linear, upload = _runtime()
        self.in_proj = linear(state, f"{prefix}.in_proj", device, dtype)
        token = state[f"{prefix}.special_token"]
        if tuple(token.shape) != (1, 1, 1, self.hidden_size):
            raise ValueError(
                "Local encoder special_token has an incompatible checkpoint shape"
            )
        self.special_token = upload(
            token.reshape(1, 1, self.hidden_size), device, dtype
        )
        self.encoder = model(config, state, f"{prefix}.encoder", device, dtype)

    def __call__(self, x):
        ttnn, _, _, _ = _runtime()
        if len(x.shape) != 4 or x.shape[-1] != self.input_dim:
            raise ValueError("Local encoder expects [batch, chunks, patch, input_dim]")
        batch, chunks, patch, channels = tuple(x.shape)
        if min(batch, chunks, patch) <= 0:
            raise ValueError("Local encoder input dimensions must be positive")
        projected = self.in_proj(ttnn.reshape(x, (batch * chunks, patch, channels)))
        tokens = ttnn.repeat(self.special_token, (batch * chunks, 1, 1))
        hidden = ttnn.concat([tokens, projected], dim=1)
        hidden = self.encoder(hidden, is_causal=False)
        cls = ttnn.slice(hidden, (0, 0, 0), (batch * chunks, 1, self.hidden_size))
        return ttnn.reshape(cls, (batch, chunks, self.hidden_size))


class _TimestepMLP:
    def __init__(self, state, prefix, device, dtype):
        _, _, linear, _ = _runtime()
        self.linear_1 = linear(state, f"{prefix}.linear_1", device, dtype)
        self.linear_2 = linear(state, f"{prefix}.linear_2", device, dtype)

    def __call__(self, x):
        ttnn, _, _, _ = _runtime()
        return self.linear_2(ttnn.silu(self.linear_1(x)))


class TtLocalDiT:
    """Local DiT v2, including independent t and delta-t embeddings.

    x / cond are [batch, feat_dim, time], mu is [batch, mu_tokens * hidden] or
    [batch, mu_tokens, hidden]; t and dt are device tensors of shape [batch].
    Return shape equals x. The token order is mu, time, condition, target.
    """

    def __init__(self, config, state, prefix, device, dtype, in_channels=64):
        _validate_local_config(config)
        self.hidden_size = config["hidden_size"]
        if self.hidden_size < 4 or self.hidden_size % 2:
            raise ValueError(
                "Local DiT sinusoidal embedding requires even hidden_size >= 4"
            )
        self.in_channels = in_channels
        self.dtype = dtype
        ttnn, model, linear, upload = _runtime()
        self.in_proj = linear(state, f"{prefix}.in_proj", device, dtype)
        self.cond_proj = linear(state, f"{prefix}.cond_proj", device, dtype)
        self.out_proj = linear(state, f"{prefix}.out_proj", device, dtype)
        self.time_mlp = _TimestepMLP(state, f"{prefix}.time_mlp", device, dtype)
        self.delta_time_mlp = _TimestepMLP(
            state, f"{prefix}.delta_time_mlp", device, dtype
        )
        self.decoder = model(config, state, f"{prefix}.decoder", device, dtype)
        # A static model constant, not a host implementation of timestep forward.
        import torch

        half = self.hidden_size // 2
        # Native SinusoidalPosEmb performs arange, the exponent multiply and exp
        # in the timestep dtype. Computing this table in FP32 then casting misses
        # BF16 rounding of both the index and exponent, especially above index 256.
        self.frequencies = {
            tt_dtype: upload(
                torch.exp(
                    torch.arange(half, dtype=torch_dtype) * (-log(10000) / (half - 1))
                ).reshape(1, half),
                device,
                tt_dtype,
            )
            for tt_dtype, torch_dtype in (
                (ttnn.bfloat16, torch.bfloat16),
                (ttnn.float32, torch.float32),
            )
        }

    def _time_embedding(self, t, batch):
        ttnn, _, _, _ = _runtime()
        if tuple(t.shape) != (batch,):
            raise ValueError("Local DiT t / dt must have shape [batch]")
        if t.dtype not in self.frequencies:
            raise ValueError("Local DiT timestep dtype must be BF16 or FP32")
        # Both multiplications have native timestep storage/rounding boundaries.
        # Promote only the rounded angles for trigonometry, then round its result
        # back to timestep dtype before the native cast to the projection dtype.
        scaled_t = ttnn.multiply(ttnn.reshape(t, (batch, 1)), 1000.0)
        angles = ttnn.multiply(scaled_t, self.frequencies[t.dtype])
        angles = ttnn.typecast(angles, ttnn.float32)
        embedding = ttnn.concat(
            [
                ttnn.typecast(ttnn.sin(angles), t.dtype),
                ttnn.typecast(ttnn.cos(angles), t.dtype),
            ],
            dim=-1,
        )
        return ttnn.reshape(
            ttnn.typecast(embedding, self.dtype), (batch, 1, self.hidden_size)
        )

    def __call__(self, x, mu, t, cond, dt):
        ttnn, _, _, _ = _runtime()
        if len(x.shape) != 3 or x.shape[1] != self.in_channels:
            raise ValueError("Local DiT x must have shape [batch, in_channels, time]")
        batch, channels, length = tuple(x.shape)
        if len(cond.shape) != 3 or tuple(cond.shape)[:2] != (batch, channels):
            raise ValueError(
                "Local DiT condition must match x batch and feature dimensions"
            )
        if len(mu.shape) not in (2, 3) or mu.shape[0] != batch:
            raise ValueError(
                "Local DiT mu must be [batch, tokens * hidden] or [batch, tokens, hidden]"
            )
        if (len(mu.shape) == 3 and mu.shape[-1] != self.hidden_size) or (
            len(mu.shape) == 2 and mu.shape[-1] % self.hidden_size
        ):
            raise ValueError("Local DiT mu must contain complete hidden-size tokens")
        if min(batch, length) <= 0:
            raise ValueError("Local DiT batch and target length must be positive")
        prefix = cond.shape[2]
        mu_tokens = (
            mu.shape[1] // self.hidden_size if len(mu.shape) == 2 else mu.shape[1]
        )
        mu = ttnn.reshape(mu, (batch, mu_tokens, self.hidden_size))
        target = self.in_proj(ttnn.permute(x, (0, 2, 1)))
        condition = self.cond_proj(ttnn.permute(cond, (0, 2, 1)))
        time = ttnn.add(
            self.time_mlp(self._time_embedding(t, batch)),
            self.delta_time_mlp(self._time_embedding(dt, batch)),
        )
        hidden = self.decoder(
            ttnn.concat([mu, time, condition, target], dim=1), is_causal=False
        )
        start = prefix + mu_tokens + 1
        hidden = ttnn.slice(
            hidden, (0, start, 0), (batch, start + length, self.hidden_size)
        )
        return ttnn.permute(self.out_proj(hidden), (0, 2, 1))


class TtScalarQuantization:
    """Inference FSQ: out_proj(round(tanh(in_proj(hidden)) * scale) / scale)."""

    def __init__(self, state, prefix, device, dtype, scale=9):
        if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
            raise ValueError("FSQ scale must be a positive integer")
        _, _, linear, _ = _runtime()
        self.scale = scale
        self.in_proj = linear(state, f"{prefix}.in_proj", device, dtype)
        self.out_proj = linear(state, f"{prefix}.out_proj", device, dtype)

    def __call__(self, hidden):
        ttnn, _, _, _ = _runtime()
        bounded = ttnn.tanh(self.in_proj(hidden))
        quantized = ttnn.round(ttnn.multiply(bounded, float(self.scale)), decimals=0)
        return self.out_proj(ttnn.multiply(quantized, 1.0 / self.scale))
