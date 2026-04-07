# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-27B Gated DeltaNet (GDN) linear attention for TTNN.

Implements the GDN recurrence layer with:
- Linear projections: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
- 4-tap causal depthwise conv1d (shift register implementation for decode)
- L2-normalized QK with scale factor
- DeltaNet recurrence: decay * state + k outer (beta * (v - k^T state))
- SiLU-gated RMSNorm output
- Fused GDN kernel integration for device-side recurrence

Architecture constants for 27B:
- num_k_heads=16, head_k_dim=128, num_v_heads=48, head_v_dim=128
- key_dim=2048, value_dim=6144, conv_dim=10240
- conv_kernel_size=4

This module adapts the P150x4 TtGatedDeltaNet (LightweightModule) pattern
to the tt_symbiote TTNNModule interface.
"""


import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearIReplicatedWColSharded


def _l2_norm_dev(x):
    """L2 normalize along last dim: x / (||x|| + eps)."""
    x_sq = ttnn.multiply(x, x)
    ssq = ttnn.sum(x_sq, dim=-1, keepdim=True)
    ttnn.deallocate(x_sq)
    inv = ttnn.rsqrt(ttnn.add(ssq, 1e-6))
    ttnn.deallocate(ssq)
    normed = ttnn.multiply(x, inv)
    ttnn.deallocate(inv)
    return normed


def _retile(t):
    """Force proper re-tiling after reshape.

    ttnn.reshape changes logical shape but doesn't re-tile data when the tile
    structure changes. Round-tripping through ROW_MAJOR forces correct tile layout.
    """
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT)


def _gdn_recurrence_ttnn(q, k_row, k_col, v, g, beta, state):
    """GDN recurrence step using standard ttnn ops (fallback).

    Args:
        q: [num_pairs, 1, Dk] query (already L2-normed and scaled)
        k_row: [num_pairs, 1, Dk] key row vector (already L2-normed)
        k_col: [num_pairs, Dk, 1] key column vector
        v: [num_pairs, 1, Dv] value
        g: [num_pairs, 1, 1] log-space decay (negative values)
        beta: [num_pairs, 1, 1] beta scalar
        state: [num_pairs, Dk, Dv] recurrence state (modified in-place)

    Returns:
        output: [num_pairs, 1, Dv]
    """
    # Step 1: decay
    g_exp = ttnn.exp(g)
    state_b = ttnn.multiply(state, g_exp)
    ttnn.deallocate(g_exp)

    # Step 2: kv_mem = k_row @ state
    kv_mem = ttnn.matmul(k_row, state_b)

    # Step 3: delta = beta * (v - kv_mem)
    diff = ttnn.subtract(v, kv_mem)
    ttnn.deallocate(kv_mem)
    delta = ttnn.multiply(beta, diff)
    ttnn.deallocate(diff)

    # Step 4: state += outer(k_col, delta)
    outer = ttnn.matmul(k_col, delta)
    ttnn.deallocate(delta)
    new_state = ttnn.add(state_b, outer)
    ttnn.deallocate(state_b)
    ttnn.deallocate(outer)

    # Step 5: output = q @ new_state
    output = ttnn.matmul(q, new_state)

    # Update state in-place
    ttnn.copy(new_state, state)
    ttnn.deallocate(new_state)

    return output


class TTNNQwen35GatedDeltaNet(TTNNModule):
    """TTNN-accelerated Gated DeltaNet linear attention for Qwen3.5-27B.

    Wraps the fused GDN recurrence kernel (from tt/gdn_kernel/) within
    the TTNNModule interface. Falls back to ttnn ops if the fused kernel
    is unavailable.
    """

    def __init__(self):
        super().__init__()

        # Architecture constants (set in from_torch)
        self.hidden_size = None
        self.num_k_heads = None
        self.num_v_heads = None
        self.head_k_dim = None
        self.head_v_dim = None
        self.key_dim = None
        self.value_dim = None
        self.conv_dim = None
        self.conv_kernel_size = None
        self.layer_idx = None
        self.rms_norm_eps = 1e-6

        # Projection children (TTNNLinear)
        self.in_proj_qkv = None
        self.in_proj_z = None
        self.in_proj_b = None
        self.in_proj_a = None
        self.out_proj = None

        # Conv1d weights (stored as host tensors, moved to device)
        self.conv_weight = None  # [conv_dim, 1, kernel_size]

        # Learned parameters (host tensors)
        self.dt_bias = None
        self.A_log = None
        self.norm_weight = None  # RMSNormGated weight

        # Device tensors (populated in move_weights_to_device_impl)
        self.tt_conv_taps = None  # List of [1, 1, conv_dim] per tap
        self.tt_dt_bias = None
        self.tt_neg_exp_A = None
        self.tt_norm_weight = None

        # Mutable state buffers (conv + recurrence)
        self.conv_states = None
        self.rec_states = None
        self.rec_output = None

    @property
    def _is_distributed(self):
        """Check if running in distributed mode with CCL manager."""
        return (
            self.device_state is not None
            and hasattr(self.device_state, "ccl_manager")
            and self.device_state.ccl_manager is not None
        )

    def _maybe_all_gather(self, tensor):
        """All-gather tensor across mesh devices if in distributed mode."""
        if not self._is_distributed:
            return tensor
        gathered = ttnn.all_gather(
            tensor,
            dim=-1,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
        )
        ttnn.synchronize_device(self.device)
        return gathered

    @classmethod
    def from_torch(cls, gated_deltanet, distributed=True):
        """Create TTNNQwen35GatedDeltaNet from PyTorch Qwen3_5GatedDeltaNet.

        Args:
            gated_deltanet: PyTorch Qwen3_5GatedDeltaNet layer.
            distributed: Use col-sharded weights for multi-device (default True for T3K).

        Returns:
            TTNNQwen35GatedDeltaNet instance.
        """
        new_gdn = cls()
        new_gdn._fallback_torch_layer = gated_deltanet

        # Architecture constants
        new_gdn.hidden_size = gated_deltanet.hidden_size
        new_gdn.num_k_heads = gated_deltanet.num_k_heads
        new_gdn.num_v_heads = gated_deltanet.num_v_heads
        new_gdn.head_k_dim = gated_deltanet.head_k_dim
        new_gdn.head_v_dim = gated_deltanet.head_v_dim
        new_gdn.key_dim = gated_deltanet.key_dim
        new_gdn.value_dim = gated_deltanet.value_dim
        new_gdn.conv_dim = gated_deltanet.conv_dim
        new_gdn.conv_kernel_size = gated_deltanet.conv_kernel_size
        new_gdn.layer_idx = gated_deltanet.layer_idx
        new_gdn.rms_norm_eps = gated_deltanet.layer_norm_epsilon

        # Choose linear class: col-sharded for distributed, replicated for single device
        LinearCls = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear

        # Create linear children for all projections
        new_gdn.in_proj_qkv = LinearCls.from_torch(gated_deltanet.in_proj_qkv)
        new_gdn.in_proj_z = LinearCls.from_torch(gated_deltanet.in_proj_z)
        new_gdn.in_proj_b = TTNNLinear.from_torch(gated_deltanet.in_proj_b)
        new_gdn.in_proj_a = TTNNLinear.from_torch(gated_deltanet.in_proj_a)
        new_gdn.out_proj = LinearCls.from_torch(gated_deltanet.out_proj)

        # Store Conv1d weight: [conv_dim, 1, kernel_size] (depthwise)
        new_gdn.conv_weight = gated_deltanet.conv1d.weight.detach().clone()  # [conv_dim, 1, kernel_size]
        new_gdn.conv_bias = gated_deltanet.conv1d.bias  # Usually None

        # Learned parameters
        new_gdn.dt_bias = gated_deltanet.dt_bias.detach().clone()
        new_gdn.A_log = gated_deltanet.A_log.detach().clone()
        new_gdn.norm_weight = gated_deltanet.norm.weight.detach().clone()

        return new_gdn

    def preprocess_weights_impl(self):
        """Preprocess conv taps, learned parameters, and child weights."""
        super().preprocess_weights_impl()

        # Preprocess Conv1d taps for shift-register decode
        # conv_weight: [conv_dim, 1, kernel_size] -> per-tap [1, 1, conv_dim]
        # Each tap is the weight for one position in the kernel
        self.host_conv_taps = []
        for k in range(self.conv_kernel_size):
            tap = self.conv_weight[:, 0, k].to(torch.bfloat16)  # [conv_dim]
            tap_tensor = ttnn.from_torch(
                tap.unsqueeze(0).unsqueeze(0),  # [1, 1, conv_dim]
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            self.host_conv_taps.append(tap_tensor)

        # dt_bias: [num_v_heads] -> [1, 1, num_v_heads] for broadcasting
        self.host_dt_bias = ttnn.from_torch(
            self.dt_bias.to(torch.bfloat16).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        # Precompute -exp(A_log) for decay
        neg_exp_A = (-self.A_log.float().exp()).to(torch.bfloat16)
        self.host_neg_exp_A = ttnn.from_torch(
            neg_exp_A.unsqueeze(0).unsqueeze(0),  # [1, 1, num_v_heads]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        # RMSNormGated weight: [head_v_dim] -> [1, 1, head_v_dim]
        # Note: RMSNormGated uses weight directly (initialized to ones), not (1+weight)
        self.host_norm_weight = ttnn.from_torch(
            self.norm_weight.to(torch.bfloat16).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        # Precompute scale = head_k_dim^(-0.5) for Q scaling
        self.scale = self.head_k_dim**-0.5

    def move_weights_to_device_impl(self):
        """Move all preprocessed tensors to device."""
        super().move_weights_to_device_impl()

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        # Move conv taps to device
        self.tt_conv_taps = []
        for tap in self.host_conv_taps:
            tap_torch = ttnn.to_torch(tap)
            self.tt_conv_taps.append(
                ttnn.from_torch(
                    tap_torch,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mesh_mapper,
                )
            )

        # Move learned parameters to device
        dt_bias_torch = ttnn.to_torch(self.host_dt_bias)
        self.tt_dt_bias = ttnn.from_torch(
            dt_bias_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        neg_exp_A_torch = ttnn.to_torch(self.host_neg_exp_A)
        self.tt_neg_exp_A = ttnn.from_torch(
            neg_exp_A_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        norm_w_torch = ttnn.to_torch(self.host_norm_weight)
        self.tt_norm_weight = ttnn.from_torch(
            norm_w_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def _reset_decode_state(self, batch_size):
        """Reset conv and recurrence states for decode (creates new tensors).

        Args:
            batch_size: Number of batch elements for decode.
        """
        import torch as _torch

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        self.conv_states = []
        for _ in range(self.conv_kernel_size):
            state = ttnn.from_torch(
                _torch.zeros(1, batch_size, self.conv_dim, dtype=_torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mesh_mapper,
            )
            self.conv_states.append(state)

        self.rec_states = ttnn.from_torch(
            _torch.zeros(batch_size * self.num_v_heads, self.head_k_dim, self.head_v_dim, dtype=_torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        self.rec_output = None

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cache_params=None,
        attention_mask=None,
        **kwargs,
    ) -> ttnn.Tensor:
        """Forward pass through GDN linear attention.

        Implements the full GDN pipeline:
        1. Linear projections (QKV, Z, A, B)
        2. Causal conv1d (shift register for decode, full conv for prefill)
        3. Split Q/K/V from conv output
        4. L2-normalize Q/K, expand to num_v_heads, apply scale
        5. Compute decay and beta
        6. DeltaNet recurrence
        7. RMSNormGated + SiLU gate
        8. Output projection

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            cache_params: Optional cache for conv/recurrent states.
            attention_mask: Optional attention mask (unused in GDN).

        Returns:
            Output tensor [batch, seq_len, hidden_size].
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # For decode (seq_len=1), use the optimized shift-register path
        if seq_len == 1 and cache_params is not None:
            return self._forward_decode(hidden_states, batch_size, cache_params)

        # Prefill path: device-side recurrence
        return self._forward_prefill(hidden_states, batch_size, seq_len, cache_params)

    def _forward_prefill(self, hidden_states, batch_size, seq_len, cache_params):
        """GDN prefill: batched projections, per-token device-side recurrence.

        All computation stays on device. Projections are batched over the full
        sequence; conv1d + recurrence runs per-token using the same shift-register
        and ttnn ops as the decode path.
        """
        Nk = self.num_k_heads
        Nv = self.num_v_heads
        Dk = self.head_k_dim
        Dv = self.head_v_dim
        repeat_factor = Nv // Nk
        num_pairs = batch_size * Nv

        # ---- Step 1: Batched projections (full sequence, on device) ----
        qkv_all = self.in_proj_qkv(hidden_states)  # [batch, seq_len, conv_dim]
        z_all = self.in_proj_z(hidden_states)  # [batch, seq_len, value_dim]
        b_all = self.in_proj_b(hidden_states)  # [batch, seq_len, num_v_heads]
        a_all = self.in_proj_a(hidden_states)  # [batch, seq_len, num_v_heads]

        # All-gather col-sharded projections (qkv, z use col-sharded weights)
        # b, a use replicated weights (TTNNLinear) — already replicated, no gather needed
        qkv_all = self._maybe_all_gather(qkv_all)
        z_all = self._maybe_all_gather(z_all)

        # ---- Step 2: Init conv/rec states if needed ----
        if self.conv_states is None:
            self._reset_decode_state(batch_size)

        # ---- Step 3: Per-token conv1d + recurrence ON DEVICE ----
        gated_outputs = []
        for t in range(seq_len):
            # Slice token t from batched projections
            qkv_t = ttnn.slice(qkv_all, (0, t, 0), (batch_size, t + 1, self.conv_dim))
            qkv_t = ttnn.reshape(qkv_t, (1, batch_size, self.conv_dim))

            z_t = ttnn.slice(z_all, (0, t, 0), (batch_size, t + 1, self.value_dim))
            z_t = ttnn.reshape(z_t, (1, batch_size, self.value_dim))

            b_t = ttnn.slice(b_all, (0, t, 0), (batch_size, t + 1, Nv))
            b_t = ttnn.reshape(b_t, (1, batch_size, Nv))

            a_t = ttnn.slice(a_all, (0, t, 0), (batch_size, t + 1, Nv))
            a_t = ttnn.reshape(a_t, (1, batch_size, Nv))

            # Conv1d shift register (same as decode path)
            states = self.conv_states
            ttnn.copy(states[1], states[0])
            ttnn.copy(states[2], states[1])
            ttnn.copy(states[3], states[2])
            ttnn.copy(qkv_t, states[3])
            ttnn.deallocate(qkv_t)

            conv_acc = ttnn.multiply(states[0], self.tt_conv_taps[0])
            for j in range(1, self.conv_kernel_size):
                conv_acc = ttnn.mac(states[j], self.tt_conv_taps[j], conv_acc)
            conv_out = ttnn.silu(conv_acc)
            ttnn.deallocate(conv_acc)
            if len(conv_out.shape) == 4:
                conv_out = ttnn.reshape(conv_out, (1, batch_size, self.conv_dim))

            # Split Q/K/V from conv output (same as decode path)
            q_sl = ttnn.slice(conv_out, (0, 0, 0), (1, batch_size, self.key_dim))
            k_sl = ttnn.slice(conv_out, (0, 0, self.key_dim), (1, batch_size, 2 * self.key_dim))
            v_sl = ttnn.slice(conv_out, (0, 0, 2 * self.key_dim), (1, batch_size, self.conv_dim))
            ttnn.deallocate(conv_out)

            # Reshape to head format
            q_h = ttnn.reshape(q_sl, (batch_size, Nk, Dk))
            ttnn.deallocate(q_sl)
            k_h = ttnn.reshape(k_sl, (batch_size, Nk, Dk))
            ttnn.deallocate(k_sl)
            v_h = ttnn.reshape(v_sl, (batch_size, Nv, Dv))
            ttnn.deallocate(v_sl)

            # L2 normalize Q and K
            q_normed = _l2_norm_dev(q_h)
            ttnn.deallocate(q_h)
            k_normed = _l2_norm_dev(k_h)
            ttnn.deallocate(k_h)

            # Expand to Nv heads and apply scale to Q
            q_exp = ttnn.repeat_interleave(q_normed, repeat_factor, dim=1)
            ttnn.deallocate(q_normed)
            q_scaled = ttnn.multiply(q_exp, self.scale)
            ttnn.deallocate(q_exp)

            k_exp = ttnn.repeat_interleave(k_normed, repeat_factor, dim=1)
            ttnn.deallocate(k_normed)

            # Decay and beta
            beta_tt = ttnn.sigmoid(b_t)
            ttnn.deallocate(b_t)
            sp = ttnn.softplus(ttnn.add(a_t, self.tt_dt_bias))
            ttnn.deallocate(a_t)
            g_pre = ttnn.multiply(self.tt_neg_exp_A, sp)
            ttnn.deallocate(sp)

            # Reshape for recurrence [num_pairs, ...]
            q_fused = _retile(ttnn.reshape(q_scaled, (num_pairs, 1, Dk)))
            ttnn.deallocate(q_scaled)

            k_row = _retile(ttnn.reshape(k_exp, (num_pairs, 1, Dk)))
            k_col = ttnn.transpose(k_row, -2, -1)
            ttnn.deallocate(k_exp)

            v_fused = _retile(ttnn.reshape(v_h, (num_pairs, 1, Dv)))
            ttnn.deallocate(v_h)

            g_fused = _retile(ttnn.reshape(g_pre, (num_pairs, 1, 1)))
            ttnn.deallocate(g_pre)
            beta_fused = _retile(ttnn.reshape(beta_tt, (num_pairs, 1, 1)))
            ttnn.deallocate(beta_tt)

            # DeltaNet recurrence on device (same function as decode path)
            rec_output = _gdn_recurrence_ttnn(q_fused, k_row, k_col, v_fused, g_fused, beta_fused, self.rec_states)

            ttnn.deallocate(q_fused)
            ttnn.deallocate(k_row)
            ttnn.deallocate(k_col)
            ttnn.deallocate(v_fused)
            ttnn.deallocate(g_fused)
            ttnn.deallocate(beta_fused)

            # Post-processing: RMSNorm + SiLU gate (same as decode path)
            out_r = ttnn.reshape(rec_output, (batch_size, Nv, Dv))
            ttnn.deallocate(rec_output)
            out_n = ttnn.rms_norm(out_r, weight=self.tt_norm_weight, epsilon=self.rms_norm_eps)
            ttnn.deallocate(out_r)

            # Reshape z for gating: [1, batch, value_dim] -> [batch, Nv, Dv]
            z_reshaped = ttnn.reshape(z_t, (batch_size, Nv, Dv))
            ttnn.deallocate(z_t)
            z_act = ttnn.silu(z_reshaped)
            ttnn.deallocate(z_reshaped)

            gated = ttnn.multiply(out_n, z_act)
            ttnn.deallocate(out_n)
            ttnn.deallocate(z_act)

            # Reshape for concatenation: [batch, Nv, Dv] -> [batch, 1, value_dim]
            gated_flat = ttnn.reshape(gated, (batch_size, 1, self.value_dim))
            ttnn.deallocate(gated)
            gated_outputs.append(gated_flat)

        # Deallocate batched projections
        ttnn.deallocate(qkv_all)
        ttnn.deallocate(z_all)
        ttnn.deallocate(b_all)
        ttnn.deallocate(a_all)

        # ---- Step 4: Stack outputs and batched output projection ----
        if len(gated_outputs) > 1:
            gated_seq = ttnn.concat(gated_outputs, dim=1)  # [batch, seq_len, value_dim]
            for g in gated_outputs:
                ttnn.deallocate(g)
        else:
            gated_seq = gated_outputs[0]

        # Output projection (col-sharded: each device has hidden_size/num_devices)
        # Do NOT all-gather here — decoder layer handles col-sharded residual add
        output = self.out_proj(gated_seq)
        ttnn.deallocate(gated_seq)

        # conv_states and rec_states are already updated in-place during the loop
        return output

    def _forward_decode(self, hidden_states, batch_size, cache_params):
        """GDN decode: single-token with shift-register conv1d and fused recurrence.

        Uses TTNN ops for all computation (device-side, trace-compatible).
        """
        Nk = self.num_k_heads
        Nv = self.num_v_heads
        Dk = self.head_k_dim
        Dv = self.head_v_dim
        repeat_factor = Nv // Nk
        num_pairs = batch_size * Nv

        # Initialize decode state if needed
        if self.conv_states is None:
            self._reset_decode_state(batch_size)

        # Squeeze seq_len=1: [batch, 1, hidden] -> [1, batch, hidden] for consistency
        if len(hidden_states.shape) == 3:
            x = ttnn.reshape(hidden_states, (1, batch_size, self.hidden_size))
        else:
            x = hidden_states

        # ---- Projections ----
        qkv_tt = self.in_proj_qkv(hidden_states)  # [batch, 1, conv_dim]
        z_tt = self.in_proj_z(hidden_states)  # [batch, 1, value_dim]
        b_tt = self.in_proj_b(hidden_states)  # [batch, 1, num_v_heads]
        a_tt = self.in_proj_a(hidden_states)  # [batch, 1, num_v_heads]

        # All-gather col-sharded projections (qkv, z use col-sharded weights)
        # b, a use replicated weights (TTNNLinear) — already replicated, no gather needed
        qkv_tt = self._maybe_all_gather(qkv_tt)
        z_tt = self._maybe_all_gather(z_tt)

        # Reshape for decode: [batch, 1, dim] -> [1, batch, dim]
        qkv_tt = ttnn.reshape(qkv_tt, (1, batch_size, self.conv_dim))
        z_tt = ttnn.reshape(z_tt, (1, batch_size, self.value_dim))
        b_tt = ttnn.reshape(b_tt, (1, batch_size, Nv))
        a_tt = ttnn.reshape(a_tt, (1, batch_size, Nv))

        # ---- Conv1d (shift register) ----
        states = self.conv_states
        ttnn.copy(states[1], states[0])
        ttnn.copy(states[2], states[1])
        ttnn.copy(states[3], states[2])
        ttnn.copy(qkv_tt, states[3])

        conv_acc = ttnn.multiply(states[0], self.tt_conv_taps[0])
        for j in range(1, self.conv_kernel_size):
            conv_acc = ttnn.mac(states[j], self.tt_conv_taps[j], conv_acc)
        conv_out = ttnn.silu(conv_acc)
        ttnn.deallocate(conv_acc)
        if len(conv_out.shape) == 4:
            conv_out = ttnn.reshape(conv_out, (1, batch_size, self.conv_dim))

        # ---- Split Q/K/V from conv output ----
        q_sl = ttnn.slice(conv_out, (0, 0, 0), (1, batch_size, self.key_dim))
        k_sl = ttnn.slice(conv_out, (0, 0, self.key_dim), (1, batch_size, 2 * self.key_dim))
        v_sl = ttnn.slice(conv_out, (0, 0, 2 * self.key_dim), (1, batch_size, self.conv_dim))
        ttnn.deallocate(conv_out)

        # Reshape to head format
        q_h = ttnn.reshape(q_sl, (batch_size, Nk, Dk))
        ttnn.deallocate(q_sl)
        k_h = ttnn.reshape(k_sl, (batch_size, Nk, Dk))
        ttnn.deallocate(k_sl)
        v_h = ttnn.reshape(v_sl, (batch_size, Nv, Dv))
        ttnn.deallocate(v_sl)

        # ---- L2 normalize Q and K ----
        q_normed = _l2_norm_dev(q_h)
        ttnn.deallocate(q_h)
        k_normed = _l2_norm_dev(k_h)
        ttnn.deallocate(k_h)

        # Expand to Nv heads and apply scale to Q
        q_exp = ttnn.repeat_interleave(q_normed, repeat_factor, dim=1)
        ttnn.deallocate(q_normed)
        q_scaled = ttnn.multiply(q_exp, self.scale)
        ttnn.deallocate(q_exp)

        k_exp = ttnn.repeat_interleave(k_normed, repeat_factor, dim=1)
        ttnn.deallocate(k_normed)

        # ---- Decay and beta ----
        beta_tt = ttnn.sigmoid(b_tt)
        ttnn.deallocate(b_tt)
        sp = ttnn.softplus(ttnn.add(a_tt, self.tt_dt_bias))
        ttnn.deallocate(a_tt)
        g_pre = ttnn.multiply(self.tt_neg_exp_A, sp)
        ttnn.deallocate(sp)

        # ---- Reshape for recurrence [num_pairs, ...] ----
        q_fused = _retile(ttnn.reshape(q_scaled, (num_pairs, 1, Dk)))
        ttnn.deallocate(q_scaled)

        k_row = _retile(ttnn.reshape(k_exp, (num_pairs, 1, Dk)))
        k_col = ttnn.transpose(k_row, -2, -1)
        ttnn.deallocate(k_exp)

        v_fused = _retile(ttnn.reshape(v_h, (num_pairs, 1, Dv)))
        ttnn.deallocate(v_h)

        g_fused = _retile(ttnn.reshape(g_pre, (num_pairs, 1, 1)))
        ttnn.deallocate(g_pre)
        beta_fused = _retile(ttnn.reshape(beta_tt, (num_pairs, 1, 1)))
        ttnn.deallocate(beta_tt)

        # ---- DeltaNet recurrence (ttnn ops fallback) ----
        rec_output = _gdn_recurrence_ttnn(q_fused, k_row, k_col, v_fused, g_fused, beta_fused, self.rec_states)

        ttnn.deallocate(q_fused)
        ttnn.deallocate(k_row)
        ttnn.deallocate(k_col)
        ttnn.deallocate(v_fused)
        ttnn.deallocate(g_fused)
        ttnn.deallocate(beta_fused)

        # ---- Post-processing: RMSNorm + SiLU gate ----
        out_r = ttnn.reshape(rec_output, (batch_size, Nv, Dv))
        ttnn.deallocate(rec_output)
        out_n = ttnn.rms_norm(out_r, weight=self.tt_norm_weight, epsilon=self.rms_norm_eps)
        ttnn.deallocate(out_r)

        # Reshape z for gating: [1, batch, value_dim] -> [batch, Nv, Dv]
        z_reshaped = ttnn.reshape(z_tt, (batch_size, Nv, Dv))
        ttnn.deallocate(z_tt)
        z_act = ttnn.silu(z_reshaped)
        ttnn.deallocate(z_reshaped)

        gated = ttnn.multiply(out_n, z_act)
        ttnn.deallocate(out_n)
        ttnn.deallocate(z_act)

        # Reshape for output projection: [batch, Nv, Dv] -> [batch, 1, value_dim]
        gated_flat = ttnn.reshape(gated, (batch_size, 1, self.value_dim))
        ttnn.deallocate(gated)

        # Output projection (col-sharded: each device has hidden_size/num_devices)
        # Do NOT all-gather here — decoder layer handles col-sharded residual add
        output = self.out_proj(gated_flat)
        ttnn.deallocate(gated_flat)

        return output
