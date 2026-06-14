# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
TT-NN Gated DeltaNet module for Qwen3.6-27B.

Implements both decode (single-token recurrent) and prefill (chunk-wise parallel)
using TT-NN ops. Structured for eventual replacement with fused device kernels.

DeltaNet recurrence (per head, per token):
    g = exp(-A_log.exp() * softplus(a + dt_bias))
    beta = sigmoid(b)
    q, k = l2norm(q), l2norm(k)
    S = S * g + outer(k, (v - S^T @ k) * beta)
    output = S^T @ q
"""

import math

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


class TtDeltaNetState:
    """Manages DeltaNet recurrent state and conv1d state for all layers."""

    def __init__(self, num_layers, layer_types, device, config):
        self.device = device
        self.conv_states = {}
        self.recurrent_states = {}
        # When True (during trace capture/replay), the decode step writes the new
        # recurrent/conv state in-place into the persistent buffers (via ttnn.copy)
        # instead of rebinding to fresh tensors, so the state advances across
        # repeated executions of a captured trace.
        self.trace_mode = False

        for i in range(num_layers):
            if layer_types[i] == "linear_attention":
                num_v_heads = config.linear_num_value_heads
                k_dim = config.linear_key_head_dim
                v_dim = config.linear_value_head_dim
                conv_dim = config.linear_key_head_dim * config.linear_num_key_heads * 2 + v_dim * num_v_heads
                conv_k = config.linear_conv_kernel_dim

                self.recurrent_states[i] = ttnn.zeros(
                    [1, num_v_heads, k_dim, v_dim],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                self.conv_states[i] = ttnn.zeros(
                    [1, 1, conv_dim, 32],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

    def get_recurrent_state(self, layer_idx):
        return self.recurrent_states.get(layer_idx)

    def set_recurrent_state(self, layer_idx, state):
        self.recurrent_states[layer_idx] = state

    def get_conv_state(self, layer_idx):
        return self.conv_states.get(layer_idx)

    def set_conv_state(self, layer_idx, state):
        self.conv_states[layer_idx] = state

    def get_conv_state_cpu(self, layer_idx, conv_k=4):
        """Get conv_state as a CPU tensor [1, conv_dim, conv_k] for prefill/fallback paths."""
        cs = self.conv_states.get(layer_idx)
        if cs is None:
            return None
        cs_cpu = ttnn.to_torch(cs)  # [1, 1, conv_dim, 32]
        return cs_cpu[0, :, :, :conv_k]  # [1, conv_dim, conv_k]

    def set_conv_state_from_cpu(self, layer_idx, cpu_state):
        """Set conv_state from CPU tensor [1, conv_dim, conv_k] → device [1,1,conv_dim,32]."""
        conv_dim = cpu_state.shape[1]
        conv_k = cpu_state.shape[2]
        padded = torch.nn.functional.pad(cpu_state, (0, 32 - conv_k))  # [1, conv_dim, 32]
        padded = padded.unsqueeze(0).to(torch.bfloat16)  # [1, 1, conv_dim, 32]
        self.conv_states[layer_idx] = ttnn.from_torch(
            padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )


def tt_l2norm(x, dim=-1, eps=1e-6):
    sq = ttnn.mul(x, x)
    sq_sum = ttnn.sum(sq, dim=dim, keepdim=True)
    sq_sum_eps = ttnn.add(sq_sum, eps)
    inv_norm = ttnn.rsqrt(sq_sum_eps)
    return ttnn.mul(x, inv_norm)


def tt_softplus(x):
    """softplus(x) = log(1 + exp(x))"""
    exp_x = ttnn.exp(x)
    one_plus = ttnn.add(exp_x, 1.0)
    return ttnn.log(one_plus)


def tt_rms_norm_per_head(x, weight, dv, eps=1e-6):
    """Per-head RMSNorm on device. x: [1, H, 1, Dv], weight: [1, 1, 1, Dv]."""
    x_sq = ttnn.mul(x, x)
    x_sq_sum = ttnn.sum(x_sq, dim=-1, keepdim=True)
    mean = ttnn.mul(x_sq_sum, 1.0 / dv)
    inv_rms = ttnn.rsqrt(ttnn.add(mean, eps))
    normed = ttnn.mul(x, inv_rms)
    return ttnn.mul(normed, weight)


try:
    _deltanet_decode_op = ttnn.experimental.deltanet_decode
    USE_FUSED_KERNEL = True
except AttributeError:
    USE_FUSED_KERNEL = False

try:
    _deltanet_decode_full_op = ttnn.experimental.deltanet_decode_full
    USE_FULL_FUSED_KERNEL = True
except AttributeError:
    USE_FULL_FUSED_KERNEL = False

try:
    _deltanet_prefill_full_op = ttnn.experimental.deltanet_prefill_full
    USE_PREFILL_FUSED_KERNEL = True
except AttributeError:
    USE_PREFILL_FUSED_KERNEL = False

import os
if os.environ.get("DISABLE_FUSED_KERNEL"):
    USE_FUSED_KERNEL = False
    USE_FULL_FUSED_KERNEL = False
    USE_PREFILL_FUSED_KERNEL = False
    _deltanet_decode_op = None
    _deltanet_decode_full_op = None
    _deltanet_prefill_full_op = None
elif os.environ.get("DISABLE_FULL_FUSED_KERNEL"):
    USE_FULL_FUSED_KERNEL = False
    USE_PREFILL_FUSED_KERNEL = False
    _deltanet_decode_full_op = None
    _deltanet_prefill_full_op = None


class TtGatedDeltaNet(LightweightModule):
    """
    Gated DeltaNet layer using TT-NN op composition.

    Decode path (S=1): recurrent step using individual ops.
    Prefill path: chunk-wise parallel algorithm.
    """

    def __init__(self, device, state_dict, layer_idx, config, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.layer_idx = layer_idx
        self.dtype = dtype
        proj_dtype = getattr(config, "weights_dtype", dtype)

        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.head_expand_ratio = self.num_v_heads // self.num_k_heads
        self.scale = self.head_k_dim**-0.5

        prefix = f"model.layers.{layer_idx}"

        self.in_proj_qkv_w = self._load_weight(
            state_dict, f"{prefix}.linear_attn.in_proj_qkv.weight", proj_dtype, transpose=True
        )
        self.in_proj_z_w = self._load_weight(
            state_dict, f"{prefix}.linear_attn.in_proj_z.weight", proj_dtype, transpose=True
        )
        self.in_proj_b_w = self._load_weight(
            state_dict, f"{prefix}.linear_attn.in_proj_b.weight", proj_dtype, transpose=True
        )
        self.in_proj_a_w = self._load_weight(
            state_dict, f"{prefix}.linear_attn.in_proj_a.weight", proj_dtype, transpose=True
        )
        self.out_proj_w = self._load_weight(
            state_dict, f"{prefix}.linear_attn.out_proj.weight", proj_dtype, transpose=True
        )
        self.norm_weight = self._load_weight(
            state_dict, f"{prefix}.linear_attn.norm.weight", dtype
        )

        self.A_log = self._load_weight(
            state_dict, f"{prefix}.linear_attn.A_log", ttnn.float32
        )
        self.dt_bias = self._load_weight(
            state_dict, f"{prefix}.linear_attn.dt_bias", ttnn.float32
        )
        self.A_log_bf16 = self._load_weight(
            state_dict, f"{prefix}.linear_attn.A_log", dtype
        )
        self.dt_bias_bf16 = self._load_weight(
            state_dict, f"{prefix}.linear_attn.dt_bias", dtype
        )

        a_log_key = f"{prefix}.linear_attn.A_log"
        dt_bias_key = f"{prefix}.linear_attn.dt_bias"
        self.A_log_cpu = state_dict[a_log_key][:self.num_v_heads].float() if a_log_key in state_dict else None
        self.dt_bias_cpu = state_dict[dt_bias_key][:self.num_v_heads].float() if dt_bias_key in state_dict else None
        norm_key = f"{prefix}.linear_attn.norm.weight"
        self.norm_weight_cpu = state_dict[norm_key][:self.head_v_dim].float() if norm_key in state_dict else None

        conv_w_key = f"{prefix}.linear_attn.conv1d.weight"
        if conv_w_key in state_dict:
            self.conv1d_weight = state_dict[conv_w_key].squeeze(1)  # [conv_dim, kernel_size]
            w_padded = torch.nn.functional.pad(self.conv1d_weight, (0, 28))  # [conv_dim, 32]
            self.conv1d_weight_tt = ttnn.from_torch(
                w_padded.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
            )
        else:
            self.conv1d_weight = None
            self.conv1d_weight_tt = None

    def _load_weight(self, state_dict, key, dtype, transpose=False):
        if key not in state_dict:
            return None
        w = state_dict[key]
        if isinstance(w, torch.Tensor):
            if w.dim() == 1:
                w = w.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif w.dim() == 2:
                if transpose:
                    w = w.T.contiguous()
                w = w.unsqueeze(0).unsqueeze(0)
            return ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        return w

    def _decode_step(self, hidden_states, deltanet_state):
        """
        Single-token decode: recurrent DeltaNet step.

        hidden_states: [1, 1, 1, hidden_size] on device (4D TILE_LAYOUT)
        Returns: [1, 1, 1, hidden_size] on device
        """
        B = 1

        qkv = ttnn.linear(hidden_states, self.in_proj_qkv_w)
        z = ttnn.linear(hidden_states, self.in_proj_z_w)
        b_proj = ttnn.linear(hidden_states, self.in_proj_b_w)
        a_proj = ttnn.linear(hidden_states, self.in_proj_a_w)

        conv_state_cpu = deltanet_state.get_conv_state_cpu(self.layer_idx, self.conv_kernel_size)
        qkv_cpu = ttnn.to_torch(qkv).flatten()  # [conv_dim]

        if conv_state_cpu is not None:
            conv_state_np = conv_state_cpu.squeeze(0)  # [conv_dim, kernel_size]
            conv_state_np = torch.roll(conv_state_np, shifts=-1, dims=-1)
            conv_state_np[:, -1] = qkv_cpu
            deltanet_state.set_conv_state_from_cpu(self.layer_idx, conv_state_np.unsqueeze(0))
            qkv_conv = (conv_state_np * self.conv1d_weight).sum(dim=-1)  # [conv_dim]
            qkv_conv = torch.nn.functional.silu(qkv_conv)
        else:
            deltanet_state.set_conv_state_from_cpu(
                self.layer_idx,
                qkv_cpu.unsqueeze(0).unsqueeze(-1).expand(-1, -1, self.conv_kernel_size).clone()
            )
            qkv_conv = torch.nn.functional.silu(qkv_cpu)

        query_t, key_t, value_t = torch.split(
            qkv_conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )

        query_t = query_t.reshape(B, self.num_k_heads, self.head_k_dim)
        key_t = key_t.reshape(B, self.num_k_heads, self.head_k_dim)
        value_t = value_t.reshape(B, self.num_v_heads, self.head_v_dim)

        b_cpu = ttnn.to_torch(b_proj).flatten()[:self.num_v_heads]
        a_cpu = ttnn.to_torch(a_proj).flatten()[:self.num_v_heads]

        beta = torch.sigmoid(b_cpu.float())  # [num_v_heads]

        g = -self.A_log_cpu.exp() * torch.nn.functional.softplus(a_cpu.float() + self.dt_bias_cpu)  # [num_v_heads]

        if self.head_expand_ratio > 1:
            query_t = query_t.repeat_interleave(self.head_expand_ratio, dim=1)
            key_t = key_t.repeat_interleave(self.head_expand_ratio, dim=1)

        query_t = self._l2norm_cpu(query_t, dim=-1).float()  # [B, H, Dk]
        key_t = self._l2norm_cpu(key_t, dim=-1).float()
        value_t = value_t.float()  # [B, H, Dv]

        scale = self.head_k_dim**-0.5
        query_t = query_t * scale

        state_cpu = ttnn.to_torch(deltanet_state.get_recurrent_state(self.layer_idx))
        # state_cpu: [1, num_v_heads, k_dim, v_dim]

        q_t = query_t[0]  # [H, Dk]
        k_t = key_t[0]    # [H, Dk]
        v_t = value_t[0]  # [H, Dv]
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)  # [H, 1, 1]
        beta_t = beta.unsqueeze(-1)  # [H, 1]

        S = state_cpu[0]  # [H, Dk, Dv]
        S = S * g_t
        kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)  # [H, Dv]
        delta = (v_t - kv_mem) * beta_t  # [H, Dv]
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # [H, Dk, Dv]
        output_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)  # [H, Dv]

        deltanet_state.set_recurrent_state(
            self.layer_idx,
            ttnn.from_torch(S.unsqueeze(0).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device),
        )

        z_cpu = ttnn.to_torch(z).flatten()[:self.num_v_heads * self.head_v_dim]
        z_cpu = z_cpu.reshape(self.num_v_heads, self.head_v_dim).float()
        variance = output_t.pow(2).mean(-1, keepdim=True)
        out_normed = output_t * torch.rsqrt(variance + 1e-6)
        out_normed = self.norm_weight_cpu * out_normed
        out_gated = out_normed * torch.nn.functional.silu(z_cpu)

        out_gated = out_gated.reshape(1, 1, 1, -1).to(torch.bfloat16)  # [1, 1, 1, H_out]

        output = ttnn.linear(
            ttnn.from_torch(out_gated, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device),
            self.out_proj_w,
        )
        return output

    def _decode_step_fused(self, hidden_states, deltanet_state):
        """
        Single-token decode using the fused DeltaNet device kernel.
        Replaces the CPU recurrence with a single ttnn.experimental.deltanet_decode call.
        """
        B = 1

        qkv = ttnn.linear(hidden_states, self.in_proj_qkv_w)
        z = ttnn.linear(hidden_states, self.in_proj_z_w)
        b_proj = ttnn.linear(hidden_states, self.in_proj_b_w)
        a_proj = ttnn.linear(hidden_states, self.in_proj_a_w)

        conv_state_cpu = deltanet_state.get_conv_state_cpu(self.layer_idx, self.conv_kernel_size)
        qkv_cpu = ttnn.to_torch(qkv).flatten()

        if conv_state_cpu is not None:
            conv_state_np = conv_state_cpu.squeeze(0)
            conv_state_np = torch.roll(conv_state_np, shifts=-1, dims=-1)
            conv_state_np[:, -1] = qkv_cpu
            deltanet_state.set_conv_state_from_cpu(self.layer_idx, conv_state_np.unsqueeze(0))
            qkv_conv = (conv_state_np * self.conv1d_weight).sum(dim=-1)
            qkv_conv = torch.nn.functional.silu(qkv_conv)
        else:
            deltanet_state.set_conv_state_from_cpu(
                self.layer_idx,
                qkv_cpu.unsqueeze(0).unsqueeze(-1).expand(-1, -1, self.conv_kernel_size).clone()
            )
            qkv_conv = torch.nn.functional.silu(qkv_cpu)

        query_t, key_t, value_t = torch.split(
            qkv_conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )

        query_t = query_t.reshape(B, self.num_k_heads, self.head_k_dim)
        key_t = key_t.reshape(B, self.num_k_heads, self.head_k_dim)
        value_t = value_t.reshape(B, self.num_v_heads, self.head_v_dim)

        b_cpu = ttnn.to_torch(b_proj).flatten()[:self.num_v_heads]
        a_cpu = ttnn.to_torch(a_proj).flatten()[:self.num_v_heads]

        beta = torch.sigmoid(b_cpu.float())
        g = -self.A_log_cpu.exp() * torch.nn.functional.softplus(a_cpu.float() + self.dt_bias_cpu)
        decay = g.exp()

        if self.head_expand_ratio > 1:
            query_t = query_t.repeat_interleave(self.head_expand_ratio, dim=1)
            key_t = key_t.repeat_interleave(self.head_expand_ratio, dim=1)

        query_t = self._l2norm_cpu(query_t, dim=-1).float() * self.scale
        key_t = self._l2norm_cpu(key_t, dim=-1).float()
        value_t = value_t.float()

        H = self.num_v_heads
        Dk = self.head_k_dim
        Dv = self.head_v_dim

        q_tt = ttnn.from_torch(
            query_t.reshape(1, H, 1, Dk).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        k_tt = ttnn.from_torch(
            key_t.reshape(1, H, 1, Dk).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        v_tt = ttnn.from_torch(
            value_t.reshape(1, H, 1, Dv).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        decay_tt = ttnn.from_torch(
            decay.reshape(1, H, 1, 1).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        beta_tt = ttnn.from_torch(
            beta.reshape(1, H, 1, 1).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )

        state = deltanet_state.get_recurrent_state(self.layer_idx)
        # State is always bf16 now — no typecast needed

        output_tt, new_state = _deltanet_decode_op(
            q_tt, k_tt, v_tt, decay_tt, beta_tt, state,
            num_heads=H, k_head_dim=Dk, v_head_dim=Dv,
        )

        deltanet_state.set_recurrent_state(self.layer_idx, new_state)

        out_normed = tt_rms_norm_per_head(output_tt, self.norm_weight, Dv)

        out_rm = ttnn.to_layout(out_normed, ttnn.ROW_MAJOR_LAYOUT)
        out_flat = ttnn.to_layout(ttnn.reshape(out_rm, [1, 1, 1, H * Dv]), ttnn.TILE_LAYOUT)
        out_gated = ttnn.mul(out_flat, ttnn.silu(z))
        output = ttnn.linear(out_gated, self.out_proj_w)
        return output

    def _prefill(self, hidden_states, deltanet_state):
        """
        Multi-token prefill with batched projections.

        Reads each weight matrix once from DRAM for all S tokens (vs S times
        in the token-by-token path). Recurrence is still sequential per token.

        hidden_states: [1, 1, S, hidden_size] on device
        Returns: [1, 1, S, hidden_size] on device
        """
        B = 1
        S = hidden_states.shape[2]
        H = self.num_v_heads
        Dk = self.head_k_dim
        Dv = self.head_v_dim

        qkv_all = ttnn.linear(hidden_states, self.in_proj_qkv_w)
        z_all = ttnn.linear(hidden_states, self.in_proj_z_w)
        b_all = ttnn.linear(hidden_states, self.in_proj_b_w)
        a_all = ttnn.linear(hidden_states, self.in_proj_a_w)

        qkv_cpu = ttnn.to_torch(qkv_all).reshape(S, -1)
        z_cpu = ttnn.to_torch(z_all).reshape(S, -1)
        b_cpu = ttnn.to_torch(b_all).reshape(S, -1)
        a_cpu = ttnn.to_torch(a_all).reshape(S, -1)

        ttnn.deallocate(qkv_all)
        ttnn.deallocate(z_all)
        ttnn.deallocate(b_all)
        ttnn.deallocate(a_all)

        state_cpu = ttnn.to_torch(deltanet_state.get_recurrent_state(self.layer_idx))
        S_state = state_cpu[0].float()

        conv_state_cpu = deltanet_state.get_conv_state_cpu(self.layer_idx, self.conv_kernel_size)

        outputs = []
        for t in range(S):
            qkv_t = qkv_cpu[t]
            if conv_state_cpu is not None:
                conv_state_sq = conv_state_cpu.squeeze(0)
                conv_state_sq = torch.roll(conv_state_sq, shifts=-1, dims=-1)
                conv_state_sq[:, -1] = qkv_t
                conv_state_cpu = conv_state_sq.unsqueeze(0)
                qkv_conv = (conv_state_sq * self.conv1d_weight).sum(dim=-1)
                qkv_conv = torch.nn.functional.silu(qkv_conv)
            else:
                conv_state_cpu = qkv_t.unsqueeze(0).unsqueeze(-1).expand(-1, -1, self.conv_kernel_size).clone()
                qkv_conv = torch.nn.functional.silu(qkv_t)

            query_t, key_t, value_t = torch.split(
                qkv_conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
            )
            query_t = query_t.reshape(B, self.num_k_heads, Dk)
            key_t = key_t.reshape(B, self.num_k_heads, Dk)
            value_t = value_t.reshape(B, H, Dv)

            b_t = b_cpu[t][:H]
            a_t = a_cpu[t][:H]
            beta = torch.sigmoid(b_t.float())
            g = -self.A_log_cpu.exp() * torch.nn.functional.softplus(a_t.float() + self.dt_bias_cpu)

            if self.head_expand_ratio > 1:
                query_t = query_t.repeat_interleave(self.head_expand_ratio, dim=1)
                key_t = key_t.repeat_interleave(self.head_expand_ratio, dim=1)

            query_t = self._l2norm_cpu(query_t, dim=-1).float() * self.scale
            key_t = self._l2norm_cpu(key_t, dim=-1).float()
            value_t = value_t.float()

            q = query_t[0]
            k = key_t[0]
            v = value_t[0]
            g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta.unsqueeze(-1)

            S_state = S_state * g_t
            kv_mem = (S_state * k.unsqueeze(-1)).sum(dim=-2)
            delta = (v - kv_mem) * beta_t
            S_state = S_state + k.unsqueeze(-1) * delta.unsqueeze(-2)
            output_t = (S_state * q.unsqueeze(-1)).sum(dim=-2)

            z_t = z_cpu[t][:H * Dv].reshape(H, Dv).float()
            variance = output_t.pow(2).mean(-1, keepdim=True)
            out_normed = output_t * torch.rsqrt(variance + 1e-6)
            out_normed = self.norm_weight_cpu * out_normed
            out_gated = out_normed * torch.nn.functional.silu(z_t)
            outputs.append(out_gated.reshape(-1))

        deltanet_state.set_recurrent_state(
            self.layer_idx,
            ttnn.from_torch(
                S_state.unsqueeze(0).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            ),
        )
        deltanet_state.set_conv_state_from_cpu(self.layer_idx, conv_state_cpu)

        out_stacked = torch.stack(outputs, dim=0).to(torch.bfloat16)
        out_tt = ttnn.from_torch(
            out_stacked.reshape(1, 1, S, H * Dv),
            dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        return ttnn.linear(out_tt, self.out_proj_w)

    def _decode_step_full_fused(self, hidden_states, deltanet_state):
        """
        Single-token decode using the fully fused DeltaNet kernel (phase B2).
        Conv1d, recurrence, decay/beta, norm, gate — all on device.
        Zero host↔device transfers in the DeltaNet layer.
        """
        qkv = ttnn.linear(hidden_states, self.in_proj_qkv_w)
        z = ttnn.linear(hidden_states, self.in_proj_z_w)
        b_proj = ttnn.linear(hidden_states, self.in_proj_b_w)
        a_proj = ttnn.linear(hidden_states, self.in_proj_a_w)

        H = self.num_v_heads
        Dk = self.head_k_dim
        Dv = self.head_v_dim

        conv_state = deltanet_state.get_conv_state(self.layer_idx)
        state = deltanet_state.get_recurrent_state(self.layer_idx)

        output_tt, new_state, new_conv_state = _deltanet_decode_full_op(
            qkv, z, b_proj, a_proj,
            conv_state, state,
            self.conv1d_weight_tt, self.A_log_bf16,
            self.dt_bias_bf16, self.norm_weight,
            num_heads=H, num_k_heads=self.num_k_heads,
            k_head_dim=Dk, v_head_dim=Dv,
            conv_dim=self.key_dim * 2 + self.value_dim,
            conv_kernel_size=self.conv_kernel_size,
            head_expand_ratio=self.head_expand_ratio,
        )

        if getattr(deltanet_state, "trace_mode", False):
            # Advance state in-place so a captured trace keeps using the same buffers.
            ttnn.copy(new_state, state)
            ttnn.copy(new_conv_state, conv_state)
        else:
            deltanet_state.set_recurrent_state(self.layer_idx, new_state)
            deltanet_state.set_conv_state(self.layer_idx, new_conv_state)

        return ttnn.linear(output_tt, self.out_proj_w)

    def _prefill_fused(self, hidden_states, deltanet_state):
        """
        Multi-token prefill using the fused device kernel.
        All S tokens processed on-device with state ping-ponging in L1.
        """
        S = hidden_states.shape[2]
        H = self.num_v_heads
        Dk = self.head_k_dim
        Dv = self.head_v_dim

        qkv_all = ttnn.linear(hidden_states, self.in_proj_qkv_w)
        z_all = ttnn.linear(hidden_states, self.in_proj_z_w)
        b_all = ttnn.linear(hidden_states, self.in_proj_b_w)
        a_all = ttnn.linear(hidden_states, self.in_proj_a_w)

        conv_state = deltanet_state.get_conv_state(self.layer_idx)
        state = deltanet_state.get_recurrent_state(self.layer_idx)

        results = _deltanet_prefill_full_op(
            qkv_all, z_all, b_all, a_all,
            conv_state, state,
            self.conv1d_weight_tt, self.A_log_bf16,
            self.dt_bias_bf16, self.norm_weight,
            num_heads=H, num_k_heads=self.num_k_heads,
            k_head_dim=Dk, v_head_dim=Dv,
            conv_dim=self.key_dim * 2 + self.value_dim,
            conv_kernel_size=self.conv_kernel_size,
            head_expand_ratio=self.head_expand_ratio,
            seq_len=S,
        )
        output_tt, new_state, new_conv_state = results[0], results[1], results[2]

        ttnn.deallocate(qkv_all)
        ttnn.deallocate(z_all)
        ttnn.deallocate(b_all)
        ttnn.deallocate(a_all)

        deltanet_state.set_recurrent_state(self.layer_idx, new_state)
        deltanet_state.set_conv_state(self.layer_idx, new_conv_state)

        # output_tt is [S*H, 1, 1, Dv], reshape to [1, 1, S, H*Dv] for out_proj
        out_rm = ttnn.to_layout(output_tt, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(output_tt)
        out_reshaped = ttnn.reshape(out_rm, [1, 1, S, H * Dv])
        ttnn.deallocate(out_rm)
        out_tile = ttnn.to_layout(out_reshaped, ttnn.TILE_LAYOUT)
        ttnn.deallocate(out_reshaped)

        return ttnn.linear(out_tile, self.out_proj_w)

    def forward(self, hidden_states, deltanet_state, mode="decode"):
        if mode == "decode":
            if USE_FULL_FUSED_KERNEL:
                return self._decode_step_full_fused(hidden_states, deltanet_state)
            if USE_FUSED_KERNEL:
                return self._decode_step_fused(hidden_states, deltanet_state)
            return self._decode_step(hidden_states, deltanet_state)
        else:
            if USE_PREFILL_FUSED_KERNEL:
                return self._prefill_fused(hidden_states, deltanet_state)
            return self._prefill(hidden_states, deltanet_state)

    @staticmethod
    def _l2norm_cpu(x, dim=-1, eps=1e-6):
        return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
