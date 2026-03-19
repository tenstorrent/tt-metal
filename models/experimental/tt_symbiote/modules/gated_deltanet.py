# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

try:
    from models.experimental.tt_symbiote.modules.chunked_deltanet import fused_chunked_delta_rule_ttnn
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
    return_conv_state=False,
):
    """
    TTNN forward pass for the Gated DeltaNet layer.

    Parity checklist vs torch (core/torch_gated_deltanet.py):
    [x] 1. Linear projections q,k,v (ttnn.linear)
    [x] 2. Causal conv1d + SiLU on q,k,v (causal_conv1d_ttnn) - NO conv_state for T==1; use torch fallback
    [x] 3. Reshape to [B,T,H,D] and GVA repeat_interleave for q,k when num_v_heads > num_heads
    [x] 4. beta = sigmoid(b_proj) with allow_neg_eigval -> beta*2
    [x] 5. g = -exp(A_log) * softplus(a + dt_bias)
    [x] 6. Delta rule: chunk or recurrent with use_qk_l2norm (l2_norm on q,k), scale=1/sqrt(K)
    [x] 7. Output norm: rms_norm_gated(o, gate, weight) or rms_norm(o, weight)
    [x] 8. Output projection o_proj
    Supports two modes:
      - "recurrent": token-by-token, for decode (T=1)
      - "chunk": chunked parallel, for prefill (T>1)

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
        new_recurrent_state: ttnn.Tensor [B, num_v_heads, head_k_dim, head_v_dim] for cache update
    """
    if num_v_heads is None:
        num_v_heads = num_heads

    if recurrent_gated_delta_rule_ttnn is None or fused_chunked_delta_rule_ttnn is None:
        raise ImportError("Delta rule implementations not available")

    B = hidden_states.shape[0]
    T = hidden_states.shape[1]
    key_dim = num_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    q = ttnn.linear(hidden_states, q_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.linear(hidden_states, k_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.linear(hidden_states, v_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Capture pre-conv q,k,v for conv_state (HF format: [B, key_dim*2+value_dim, K-1])
    conv_state = None
    if return_conv_state and T >= conv_kernel_size - 1:
        n = conv_kernel_size - 1
        q_slice = ttnn.slice(q, [0, T - n, 0], [B, T, key_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
        k_slice = ttnn.slice(k, [0, T - n, 0], [B, T, key_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
        v_slice = ttnn.slice(v, [0, T - n, 0], [B, T, value_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
        mixed = ttnn.concat([q_slice, k_slice, v_slice], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        conv_state = ttnn.permute(mixed, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)

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
        o, new_recurrent_state = fused_chunked_delta_rule_ttnn(
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
        o, new_recurrent_state = recurrent_gated_delta_rule_ttnn(
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

    return o, new_recurrent_state, conv_state


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
        layer_idx: Optional[int] = None,
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
        self.layer_idx = layer_idx
        self.hidden_size = None  # set from torch module in from_torch for tensor-parallel

        # Weight attributes (will be set externally or by preprocess_weights_impl)
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

    @classmethod
    def from_torch(cls, torch_module):
        """
        Build TTNNGatedDeltaNet from HuggingFace Qwen3NextGatedDeltaNet.

        Slices in_proj_qkvz (q,k,v,z) and in_proj_ba (b,a), splits conv1d into q/k/v conv,
        and copies A_log, dt_bias, norm.weight, out_proj.
        """
        num_heads = torch_module.num_k_heads
        num_v_heads = torch_module.num_v_heads
        key_dim = torch_module.key_dim
        value_dim = torch_module.value_dim
        conv_dim = torch_module.conv_dim
        new_module = cls(
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            head_k_dim=torch_module.head_k_dim,
            head_v_dim=torch_module.head_v_dim,
            conv_kernel_size=torch_module.conv_kernel_size,
            use_gate=True,
            allow_neg_eigval=False,
            norm_eps=torch_module.layer_norm_epsilon,
            mode="recurrent",
            chunk_size=64,
            layer_idx=getattr(torch_module, "layer_idx", None),
        )
        new_module._fallback_torch_layer = torch_module
        new_module.hidden_size = torch_module.hidden_size
        return new_module

    def preprocess_weights_impl(self):
        t = self._fallback_torch_layer
        if t is None:
            return

        key_dim = t.key_dim
        value_dim = t.value_dim
        num_v_heads = t.num_v_heads
        hidden_size = t.hidden_size

        # in_proj_qkvz: [projection_size_qkvz, hidden_size] = [key_dim*2+value_dim*2, hidden_size]
        # Split into q, k, v, z (gate): each (key_dim, hidden), (key_dim, hidden), (value_dim, hidden), (value_dim, hidden)
        w_qkvz = t.in_proj_qkvz.weight.to(torch.bfloat16)
        q_proj = w_qkvz[0:key_dim].T.contiguous()
        k_proj = w_qkvz[key_dim : 2 * key_dim].T.contiguous()
        v_proj = w_qkvz[2 * key_dim : 2 * key_dim + value_dim].T.contiguous()
        g_proj = w_qkvz[2 * key_dim + value_dim : 2 * key_dim + 2 * value_dim].T.contiguous()

        self.q_proj_weight = ttnn.from_torch(q_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.k_proj_weight = ttnn.from_torch(k_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.v_proj_weight = ttnn.from_torch(v_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.g_proj_weight = ttnn.from_torch(g_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # in_proj_ba: [num_v_heads*2, hidden_size] -> b_proj, a_proj
        w_ba = t.in_proj_ba.weight.to(torch.bfloat16)
        b_proj = w_ba[0:num_v_heads].T.contiguous()
        a_proj = w_ba[num_v_heads : 2 * num_v_heads].T.contiguous()
        self.b_proj_weight = ttnn.from_torch(b_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.a_proj_weight = ttnn.from_torch(a_proj, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # conv1d: weights must be host-side ROW_MAJOR for ttnn.conv1d (it prepares and moves to device)
        # Weight shape 3D (out_ch, 1, kernel_size) -> conv2d path reshapes to 4D
        cw = t.conv1d.weight.squeeze(1).to(torch.bfloat16)
        q_conv = cw[0:key_dim].unsqueeze(1).contiguous()
        k_conv = cw[key_dim : 2 * key_dim].unsqueeze(1).contiguous()
        v_conv = cw[2 * key_dim :].unsqueeze(1).contiguous()
        self.q_conv_weight = ttnn.from_torch(q_conv, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.k_conv_weight = ttnn.from_torch(k_conv, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.v_conv_weight = ttnn.from_torch(v_conv, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        self.q_conv_bias = None
        self.k_conv_bias = None
        self.v_conv_bias = None
        if t.conv1d.bias is not None:
            b = t.conv1d.bias.to(torch.bfloat16)
            # Host conv bias must be 4D (1,1,1,C) and ROW_MAJOR for ttnn.conv1d
            self.q_conv_bias = ttnn.from_torch(
                b[0:key_dim].reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            self.k_conv_bias = ttnn.from_torch(
                b[key_dim : 2 * key_dim].reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            self.v_conv_bias = ttnn.from_torch(
                b[2 * key_dim :].reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        self.A_log = ttnn.from_torch(
            t.A_log.unsqueeze(0).unsqueeze(0).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.dt_bias = ttnn.from_torch(
            t.dt_bias.unsqueeze(0).unsqueeze(0).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.o_norm_weight = ttnn.from_torch(
            t.norm.weight.unsqueeze(0).unsqueeze(0).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.o_proj_weight = ttnn.from_torch(
            t.out_proj.weight.T.contiguous().to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def move_weights_to_device_impl(self):
        if self.device is None:
            return
        # Linear/norm weights: move to device. Conv weights stay on host in ROW_MAJOR for ttnn.conv1d.
        for name in (
            "q_proj_weight",
            "k_proj_weight",
            "v_proj_weight",
            "a_proj_weight",
            "b_proj_weight",
            "o_proj_weight",
            "g_proj_weight",
            "A_log",
            "dt_bias",
            "o_norm_weight",
        ):
            w = getattr(self, name, None)
            if w is not None:
                setattr(self, name, ttnn.to_device(w, self.device))
        # Conv weights/biases stay on host (ROW_MAJOR); ttnn.conv1d moves them to device internally

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        recurrent_state: Optional[ttnn.Tensor] = None,
        cache_params=None,
        attention_mask=None,
        **kwargs,
    ) -> ttnn.Tensor:
        """
        Forward pass through Gated DeltaNet.

        Compatible with HuggingFace Qwen3NextDecoderLayer.linear_attn call:
        linear_attn(hidden_states=..., cache_params=past_key_values, attention_mask=...).
        """
        # Decode (T==1): TTNN lacks conv_state for causal conv. Fall back to torch layer which uses
        # cache_params.conv_states and recurrent_states correctly. Only on single device.
        T = hidden_states.shape[1]
        num_dev = self.device.get_num_devices() if self.device else 0
        if T == 1 and self.layer_idx is not None and self.layer_idx < 3:
            print(
                f"[DEBUG TTNNGatedDeltaNet] T=1 decode layer_idx={self.layer_idx} num_devices={num_dev} "
                f"fallback_avail={getattr(self, '_fallback_torch_layer', None) is not None} "
                f"will_fallback={num_dev == 1 and getattr(self, '_fallback_torch_layer', None) is not None}"
            )
        if (
            T == 1
            and getattr(self, "_fallback_torch_layer", None) is not None
            and self.device is not None
            and self.device.get_num_devices() == 1
        ):
            has_prev = cache_params is not None and getattr(cache_params, "has_previous_state", False)
            conv_ok = (
                cache_params is not None
                and getattr(cache_params, "conv_states", None) is not None
                and self.layer_idx is not None
                and self.layer_idx < len(cache_params.conv_states)
                and cache_params.conv_states[self.layer_idx] is not None
            )
            print(
                f"[DEBUG TTNNGatedDeltaNet] T=1 fallback to torch layer_idx={self.layer_idx} "
                f"has_previous_state={has_prev} conv_states[{self.layer_idx}]={'set' if conv_ok else 'None'}"
            )
            if hasattr(hidden_states, "to_torch"):
                hs_torch = hidden_states.to_torch.to(torch.bfloat16)
            else:
                hs_torch = ttnn.to_torch(hidden_states).to(torch.bfloat16)
            out = self._fallback_torch_layer(
                hidden_states=hs_torch,
                cache_params=cache_params,
                cache_position=kwargs.get("cache_position"),
                attention_mask=attention_mask,
            )
            return TorchTTNNTensor(out)

        if cache_params is not None and self.layer_idx is not None:
            recurrent_states = getattr(cache_params, "recurrent_states", None)
            if recurrent_states is not None and self.layer_idx < len(recurrent_states):
                rs = recurrent_states[self.layer_idx]
                if rs is None:
                    recurrent_state = None
                elif hasattr(rs, "to_ttnn"):
                    recurrent_state = rs.to_ttnn
                elif isinstance(rs, torch.Tensor):
                    from_torch_kw = dict(
                        device=self.device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                    )
                    if self.device.get_num_devices() > 1:
                        from_torch_kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
                    recurrent_state = ttnn.from_torch(rs.to(torch.bfloat16), **from_torch_kw)
                else:
                    recurrent_state = rs
            else:
                recurrent_state = None

        if hasattr(hidden_states, "to_ttnn"):
            hidden_states = hidden_states.to_ttnn
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        # All-gather hidden_states when tensor parallel (sharded hidden dim)
        need_reduce_scatter = (
            self.device_state is not None
            and self.device.get_num_devices() > 1
            and self.hidden_size is not None
            and hidden_states.shape[-1] != self.hidden_size
        )
        if need_reduce_scatter:
            hidden_states = ttnn.experimental.all_gather_async(
                hidden_states,
                dim=-1,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
                num_links=1,
                topology=ttnn.Topology.Linear,
            )

        # Match torch gated_deltanet_forward: T <= 64 uses recurrent, T > 64 uses chunk
        # (torch: effective_mode = "fused_recurrent" if T <= 64 else mode)
        T = hidden_states.shape[1]
        use_mode = "chunk" if T > 64 else "recurrent"
        need_conv_state = cache_params is not None and self.layer_idx is not None and T > 1
        if T > 1 and self.layer_idx is not None and self.layer_idx < 3:
            print(
                f"[DEBUG TTNNGatedDeltaNet] Prefill T={T} layer_idx={self.layer_idx} "
                f"need_conv_state={need_conv_state} cache_params={'yes' if cache_params else 'no'}"
            )

        output, new_recurrent_state, conv_state = gated_deltanet_forward_ttnn(
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
            mode=use_mode,
            chunk_size=self.chunk_size,
            return_conv_state=need_conv_state,
        )

        # Update cache: conv_state for torch decode fallback, recurrent_state for TTNN decode
        if need_conv_state and conv_state is not None:
            conv_states = getattr(cache_params, "conv_states", None)
            if conv_states is not None and self.layer_idx < len(conv_states):
                if self.device.get_num_devices() > 1:
                    conv_torch = ttnn.to_torch(ttnn.get_device_tensors(conv_state)[0])
                else:
                    conv_torch = ttnn.to_torch(conv_state)
                conv_states[self.layer_idx] = conv_torch.to(torch.bfloat16)
                if self.layer_idx == 0:
                    print(
                        f"[DEBUG TTNNGatedDeltaNet] Prefill wrote conv_state layer_idx={self.layer_idx} "
                        f"T={T} shape={tuple(conv_states[self.layer_idx].shape)}"
                    )

        # Update cache for decode (recurrent state must be written back so next token sees it)
        # Keep torch and TTNN in same dtype (bfloat16): recurrent kernel may return state in float32.
        # State is replicated across devices (each has full [B,H,K,V]); use one device's copy to avoid
        # mesh_composer concatenating 8 copies -> wrong shape [1,32,128,1024] instead of [1,32,128,128].
        if cache_params is not None and self.layer_idx is not None and new_recurrent_state is not None:
            recurrent_states = getattr(cache_params, "recurrent_states", None)
            if recurrent_states is not None and self.layer_idx < len(recurrent_states):
                state_bf16 = ttnn.typecast(new_recurrent_state, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
                if self.device.get_num_devices() > 1:
                    state_torch = ttnn.to_torch(ttnn.get_device_tensors(state_bf16)[0])
                else:
                    state_torch = ttnn.to_torch(state_bf16)
                recurrent_states[self.layer_idx] = state_torch.to(torch.bfloat16)

        if need_reduce_scatter:
            # Reduce-scatter output to match sharded residual for residual add
            output = ttnn.reshape(output, (output.shape[0], 1, output.shape[1], output.shape[2]))
            output = ttnn.experimental.reduce_scatter_minimal_async(
                output,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
                barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            )
            output = ttnn.div(output, float(self.device.get_num_devices()))
            output = ttnn.squeeze(output, 1)

        return output
