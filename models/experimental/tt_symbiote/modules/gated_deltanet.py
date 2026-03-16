# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Optional

from models.experimental.tt_symbiote.core.module import TTNNModule

try:
    from models.experimental.tt_symbiote.modules.delta_rule import fused_chunked_delta_rule_ttnn
except ImportError:
    fused_chunked_delta_rule_ttnn = None

try:
    from models.experimental.tt_symbiote.modules.recurrent_deltanet import (
        recurrent_gated_delta_rule_ttnn,
        l2_norm_ttnn as l2_norm_ttnn_recurrent,
    )
except ImportError:
    recurrent_gated_delta_rule_ttnn = None
    l2_norm_ttnn_recurrent = None


def rms_norm_gated_ttnn(x, gate, weight, eps=1e-5):
    """RMSNorm with SiLU gating using TTNN ops."""
    x_sq = ttnn.multiply(x, x, memory_config=ttnn.L1_MEMORY_CONFIG)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    inv_rms = ttnn.rsqrt(
        ttnn.add(variance, eps, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG
    )
    x_normed = ttnn.multiply(x, inv_rms, memory_config=ttnn.L1_MEMORY_CONFIG)
    x_normed = ttnn.multiply(x_normed, weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    gate_act = ttnn.silu(gate, memory_config=ttnn.L1_MEMORY_CONFIG)
    return ttnn.multiply(x_normed, gate_act, memory_config=ttnn.L1_MEMORY_CONFIG)


def rms_norm_ttnn(x, weight, eps=1e-5):
    """Standard RMSNorm using TTNN ops."""
    x_sq = ttnn.multiply(x, x, memory_config=ttnn.L1_MEMORY_CONFIG)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    inv_rms = ttnn.rsqrt(
        ttnn.add(variance, eps, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG
    )
    x_normed = ttnn.multiply(x, inv_rms, memory_config=ttnn.L1_MEMORY_CONFIG)
    return ttnn.multiply(x_normed, weight, memory_config=ttnn.L1_MEMORY_CONFIG)


def _causal_conv1d_fir(x, weight, bias, kernel_size, device):
    """
    Manual FIR decomposition of depthwise causal conv1d + SiLU.

    Used for large T where native ttnn.conv1d would OOM in L1.
    Decomposes the convolution into K element-wise multiply+accumulate
    operations on shifted slices.
    """
    B, T, D = x.shape[0], x.shape[1], x.shape[2]

    pad = ttnn.zeros(
        [B, kernel_size - 1, D],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x_padded = ttnn.concat([pad, x], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

    weight_torch = ttnn.to_torch(weight)

    out = None
    for k in range(kernel_size):
        x_slice = x_padded[:, k : k + T]
        x_slice = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT)

        w_k = weight_torch[:, 0, k].reshape(1, 1, D).contiguous()
        w_k_dev = ttnn.from_torch(
            w_k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        term = ttnn.multiply(x_slice, w_k_dev, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = term if out is None else ttnn.add(out, term, memory_config=ttnn.L1_MEMORY_CONFIG)

    if bias is not None:
        bias_torch = ttnn.to_torch(bias).reshape(1, 1, D).contiguous()
        bias_dev = ttnn.from_torch(
            bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        out = ttnn.add(out, bias_dev, memory_config=ttnn.L1_MEMORY_CONFIG)

    return ttnn.silu(out, memory_config=ttnn.L1_MEMORY_CONFIG)


def causal_conv1d_ttnn(x, weight, bias, kernel_size, device, max_conv_len=512):
    """
    Depthwise causal conv1d + SiLU using native ttnn.conv1d.

    Falls back to manual FIR decomposition for T > max_conv_len to
    avoid L1 OOM in the conv1d kernel.
    """
    B, T, D = x.shape[0], x.shape[1], x.shape[2]

    if T > max_conv_len:
        return _causal_conv1d_fir(x, weight, bias, kernel_size, device)

    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    pad_zeros = ttnn.zeros(
        [B, kernel_size - 1, D],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x_padded = ttnn.concat([pad_zeros, x_rm], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=True,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        config_tensors_in_dram=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
    )

    [out, out_length, _] = ttnn.conv1d(
        input_tensor=x_padded,
        weight_tensor=weight,
        in_channels=D,
        out_channels=D,
        device=device,
        bias_tensor=bias,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        batch_size=B,
        input_length=T + kernel_size - 1,
        groups=D,
        dtype=ttnn.bfloat16,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    out = ttnn.sharded_to_interleaved(out, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.reshape(out, [B, T, D], memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    return out


def gated_deltanet_forward_ttnn(
    hidden_states,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    a_proj_weight,
    b_proj_weight,
    o_proj_weight,
    q_conv_weight,
    k_conv_weight,
    v_conv_weight,
    q_conv_bias,
    k_conv_bias,
    v_conv_bias,
    A_log,
    dt_bias,
    o_norm_weight,
    g_proj_weight=None,
    num_heads=4,
    num_v_heads=None,
    head_k_dim=256,
    head_v_dim=512,
    conv_kernel_size=4,
    use_gate=True,
    allow_neg_eigval=False,
    norm_eps=1e-5,
    device=None,
    recurrent_state=None,
    mode="recurrent",
    chunk_size=64,
):
    """
    TTNN forward pass for the Gated DeltaNet layer.

    Supports two modes:
      - "recurrent": token-by-token, best for decode (T=1)
      - "chunk": chunked parallel, best for prefill (T>1)

    Args:
        hidden_states: ttnn.Tensor [B, T, hidden_size]
        *_proj_weight: ttnn.Tensor weight matrices in [in_features, out_features] format
        *_conv_weight: ttnn.Tensor conv1d weights (NOT transposed)
        A_log: ttnn.Tensor [num_v_heads]
        dt_bias: ttnn.Tensor [num_v_heads]
        o_norm_weight: ttnn.Tensor [head_v_dim]
        g_proj_weight: ttnn.Tensor gate projection (if use_gate)
        device: ttnn device
        mode: "recurrent" or "chunk"
        chunk_size: chunk size for chunked mode

    Returns:
        output: ttnn.Tensor [B, T, hidden_size]
    """
    if num_v_heads is None:
        num_v_heads = num_heads

    if recurrent_gated_delta_rule_ttnn is None or fused_chunked_delta_rule_ttnn is None:
        raise ImportError("Delta rule implementations not available")

    B = hidden_states.shape[0]
    T = hidden_states.shape[1]

    q = ttnn.linear(hidden_states, q_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.linear(hidden_states, k_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.linear(hidden_states, v_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)

    q = causal_conv1d_ttnn(q, q_conv_weight, q_conv_bias, conv_kernel_size, device)
    k = causal_conv1d_ttnn(k, k_conv_weight, k_conv_bias, conv_kernel_size, device)
    v = causal_conv1d_ttnn(v, v_conv_weight, v_conv_bias, conv_kernel_size, device)

    q = ttnn.reshape(q, [B, T, num_heads, head_k_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.reshape(k, [B, T, num_heads, head_k_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.reshape(v, [B, T, num_v_heads, head_v_dim], memory_config=ttnn.L1_MEMORY_CONFIG)

    if num_v_heads > num_heads:
        repeats = num_v_heads // num_heads
        q = ttnn.repeat_interleave(q, repeats, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        k = ttnn.repeat_interleave(k, repeats, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

    beta = ttnn.sigmoid(
        ttnn.linear(hidden_states, b_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    if allow_neg_eigval:
        beta = ttnn.multiply(beta, 2.0, memory_config=ttnn.L1_MEMORY_CONFIG)

    a = ttnn.linear(hidden_states, a_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    a_biased = ttnn.add(a, dt_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    sp = ttnn.softplus(a_biased, memory_config=ttnn.L1_MEMORY_CONFIG)
    A = ttnn.exp(A_log, memory_config=ttnn.L1_MEMORY_CONFIG)
    A_neg = ttnn.neg(A, memory_config=ttnn.L1_MEMORY_CONFIG)
    g = ttnn.multiply(A_neg, sp, memory_config=ttnn.L1_MEMORY_CONFIG)

    if mode == "chunk":
        o, _ = fused_chunked_delta_rule_ttnn(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
            chunk_size=chunk_size,
            initial_state=recurrent_state,
            device=device,
        )
    else:
        if l2_norm_ttnn_recurrent is None:
            raise ImportError("l2_norm_ttnn not available from recurrent_deltanet")
        q = l2_norm_ttnn_recurrent(q, dim=-1)
        k = l2_norm_ttnn_recurrent(k, dim=-1)
        o, _ = recurrent_gated_delta_rule_ttnn(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
            initial_state=recurrent_state,
            device=device,
        )

    if use_gate and g_proj_weight is not None:
        gate = ttnn.linear(hidden_states, g_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.reshape(gate, [B, T, num_v_heads, head_v_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
        o = rms_norm_gated_ttnn(o, gate, o_norm_weight, eps=norm_eps)
    else:
        o = rms_norm_ttnn(o, o_norm_weight, eps=norm_eps)

    o = ttnn.reshape(o, [B, T, num_v_heads * head_v_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
    o = ttnn.linear(o, o_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)

    return o


class TTNNGatedDeltaNet(TTNNModule):
    """TTNN-accelerated Gated DeltaNet layer that routes to chunked or recurrent mode."""

    def __init__(
        self,
        num_heads: int = 4,
        num_v_heads: Optional[int] = None,
        head_k_dim: int = 256,
        head_v_dim: int = 512,
        conv_kernel_size: int = 4,
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
        mode: str = "recurrent",
        chunk_size: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads or num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.use_gate = use_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.norm_eps = norm_eps
        self.mode = mode
        self.chunk_size = chunk_size

        # Weight attributes (will be set externally)
        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None
        self.a_proj_weight = None
        self.b_proj_weight = None
        self.o_proj_weight = None
        self.g_proj_weight = None
        self.q_conv_weight = None
        self.k_conv_weight = None
        self.v_conv_weight = None
        self.q_conv_bias = None
        self.k_conv_bias = None
        self.v_conv_bias = None
        self.A_log = None
        self.dt_bias = None
        self.o_norm_weight = None

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        recurrent_state: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass through Gated DeltaNet.

        Args:
            hidden_states: [B, T, hidden_size] input tensor
            recurrent_state: [B, H, K, V] optional initial state

        Returns:
            output: [B, T, hidden_size] output tensor
        """
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        output = gated_deltanet_forward_ttnn(
            hidden_states=hidden_states,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            a_proj_weight=self.a_proj_weight,
            b_proj_weight=self.b_proj_weight,
            o_proj_weight=self.o_proj_weight,
            q_conv_weight=self.q_conv_weight,
            k_conv_weight=self.k_conv_weight,
            v_conv_weight=self.v_conv_weight,
            q_conv_bias=self.q_conv_bias,
            k_conv_bias=self.k_conv_bias,
            v_conv_bias=self.v_conv_bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            o_norm_weight=self.o_norm_weight,
            g_proj_weight=self.g_proj_weight,
            num_heads=self.num_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_kernel_size=self.conv_kernel_size,
            use_gate=self.use_gate,
            allow_neg_eigval=self.allow_neg_eigval,
            norm_eps=self.norm_eps,
            device=self.device,
            recurrent_state=recurrent_state,
            mode=self.mode,
            chunk_size=self.chunk_size,
        )

        return output
