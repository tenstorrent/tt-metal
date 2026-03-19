# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Gated DeltaNet (linear attention) module for Qwen3.5.

Implements the Gated Delta Rule recurrence for single-token decode on
Tenstorrent hardware. The algorithm processes one token per user per step,
updating a per-layer recurrent state instead of using a KV cache:

    state *= exp(gate)                          # global decay
    kv_mem = einsum('hkv,hk->hv', state, k)    # retrieve from state
    delta  = (v - kv_mem) * beta                # compute correction
    state += einsum('hk,hv->hkv', k, delta)    # write to state
    output = einsum('hkv,hk->hv', state, q)    # read from state

The recurrence runs on host (CPU) in float32 because it is extremely
sensitive to compound quantization error: bfp8 projections in upstream
layers produce slightly wrong Q/K/V, and the persistent recurrent state
amplifies these errors across tokens. Projections, conv1d, norms, and
output projection stay on device. The host roundtrip is ~50 KB per
layer per token (negligible at current decode speeds).

Reference: "Gated Delta Networks with Softmax Attention" (Yang et al., 2025)
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class GatedDeltaNet(LightweightModule):
    """Gated DeltaNet linear attention layer for Qwen3.5.

    This implements the decode (single-token) path only.
    Prefill is not yet supported and should fall back to token-by-token decode.

    Args:
        mesh_device: TT device or mesh device to place tensors on.
        args: ModelArgs with Qwen3.5 config (linear_num_value_heads, etc.).
        state_dict: Model state dict with weights for this layer.
        weight_cache_path: Path for caching converted weights on disk.
        layer_num: Layer index (0-based) in the full model.
        dtype: Weight data type (e.g. ttnn.bfloat8_b for BFP8 weights).
    """

    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.layer_num = layer_num

        # =====================================================================
        # DeltaNet architecture parameters (from Qwen3.5 config)
        # =====================================================================
        self.hidden_size = args.dim  # 5120
        self.num_v_heads = args.linear_num_value_heads  # 48 (value/output heads)
        self.num_k_heads = args.linear_num_key_heads  # 16 (key/query heads, GQA)
        self.head_k_dim = args.linear_key_head_dim  # 128
        self.head_v_dim = args.linear_value_head_dim  # 128
        self.key_dim = self.head_k_dim * self.num_k_heads  # 2048
        self.value_dim = self.head_v_dim * self.num_v_heads  # 6144
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 10240 (Q+K+V)
        self.conv_kernel_size = args.linear_conv_kernel_dim  # 4
        self.gqa_ratio = self.num_v_heads // self.num_k_heads  # 3
        self.scale = 1.0 / math.sqrt(self.head_k_dim)

        # HiFi2 for projection matmuls: bfp8 weights default to LoFi which
        # drops too much precision over 48 DeltaNet layers. HiFi2 matches MLP.
        self.proj_compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # HiFi4 + fp32 accumulation for recurrence matmuls (state is fp32)
        self.recurrence_compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Weight key prefix: layers.{layer_num}.linear_attn
        layer_prefix = args.get_state_dict_prefix("GatedDeltaNet", layer_num)

        if args.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{layer_prefix}.{name}"

        def load_weight(name, transpose=True):
            """Load a 2D weight, optionally transposed for ttnn.linear."""
            key = f"{layer_prefix}.{name}.weight"
            w = state_dict[key]
            if transpose and w.dim() == 2:
                w = w.transpose(-2, -1)  # (out, in) -> (in, out) for ttnn.linear
            return w

        def load_param(name):
            """Load a non-weight parameter (e.g. dt_bias, A_log)."""
            return state_dict[f"{layer_prefix}.{name}"]

        # =====================================================================
        # Projection weights (on device, DRAM)
        # Fused: QKV+Z into one bfp8 weight, B+A into one bf16 weight.
        # 2 matmuls + 2 to_torch instead of 4 + 4 (saves ~1 ms/layer).
        # =====================================================================
        proj_dtype = dtype
        # All projections fused into one bfp8 weight: QKV + Z + B + A
        # Gate projections (B, A) were bf16 before, but with host float32
        # recurrence the gate precision is handled on host. bfp8 is OK now.
        w_all = torch.cat(
            [
                load_weight("in_proj_qkv"),
                load_weight("in_proj_z"),
                load_weight("in_proj_b"),
                load_weight("in_proj_a"),
            ],
            dim=-1,
        )
        self._proj_splits = [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads]
        self.in_proj_all = ttnn.as_tensor(
            w_all.unsqueeze(0).unsqueeze(0),
            dtype=proj_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("in_proj_all_fused"),
        )
        self.out_proj = ttnn.as_tensor(
            load_weight("out_proj").unsqueeze(0).unsqueeze(0),
            dtype=proj_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("out_proj"),
        )

        # =====================================================================
        # Conv, gate, norm weights on HOST (these ops run on host to reduce
        # device kernel launches from ~50 to ~6 per layer)
        # =====================================================================
        conv_weight_raw = state_dict[f"{layer_prefix}.conv1d.weight"].float().squeeze(1)  # (conv_dim, K)
        self._conv_w = conv_weight_raw.T  # (K, conv_dim)
        self._dt_bias = load_param("dt_bias").float()
        self._A_exp = load_param("A_log").float().exp()
        self._norm_w = state_dict[f"{layer_prefix}.norm.weight"].float()

    # =========================================================================
    # KV cache compatibility stubs
    # =========================================================================
    @property
    def layer_past(self):
        """DeltaNet uses recurrent state, not a KV cache."""
        return None

    @layer_past.setter
    def layer_past(self, value):
        pass

    # =========================================================================
    # State management
    # =========================================================================
    def initialize_states(self, batch_size=1):
        """Initialize host-side recurrent state and conv buffer."""
        self._host_state = torch.zeros(self.num_v_heads, self.head_k_dim, self.head_v_dim)
        self._conv_state = torch.zeros(self.conv_kernel_size, self.conv_dim)

    # =========================================================================
    # Forward pass
    # =========================================================================
    def forward(self, x):
        """Single-token decode forward pass.

        Projections and conv1d run on device; recurrence runs on host (float32)
        to avoid compound quantization error in the persistent state.

        Args:
            x: (1, 1, B_pad, hidden_size) -- norm output from decoder block

        Returns:
            output: (1, 1, B_pad, hidden_size) -- ready for residual add
        """
        B_pad = x.shape[2]  # 32

        # =================================================================
        # 1. Linear projections  [on device, 4D]
        #
        # These are the expensive operations (5120 x 10240 etc.).
        # ttnn.linear handles 4D naturally: matmul over last dim.
        # =================================================================
        # 1 fused projection (QKV+Z+B+A) -> 1 matmul + 1 to_torch
        all_proj = ttnn.linear(x, self.in_proj_all, compute_kernel_config=self.proj_compute_config)

        F = torch.nn.functional
        all_h = ttnn.to_torch(all_proj).float()[0, 0, 0, :]
        ttnn.deallocate(all_proj)
        s = self._proj_splits
        qkv_h = all_h[: s[0]]
        z_h = all_h[s[0] : s[0] + s[1]]
        b_h = all_h[s[0] + s[1] : s[0] + s[1] + s[2]]
        a_h = all_h[s[0] + s[1] + s[2] :]

        # Conv1d (host ring buffer, precomputed weights)
        self._conv_state[:-1] = self._conv_state[1:].clone()
        self._conv_state[-1] = qkv_h
        qkv_h = F.silu((self._conv_state * self._conv_w).sum(dim=0))

        # Split Q, K, V + GQA expand + L2 normalize
        q_h = qkv_h[: self.key_dim].reshape(self.num_k_heads, self.head_k_dim)
        k_h = qkv_h[self.key_dim : 2 * self.key_dim].reshape(self.num_k_heads, self.head_k_dim)
        v_h = qkv_h[2 * self.key_dim :].reshape(self.num_v_heads, self.head_v_dim)
        if self.gqa_ratio > 1:
            q_h = q_h.repeat_interleave(self.gqa_ratio, dim=0)
            k_h = k_h.repeat_interleave(self.gqa_ratio, dim=0)
        q_h = F.normalize(q_h, dim=-1) * self.scale
        k_h = F.normalize(k_h, dim=-1)

        # Gates
        beta = b_h.sigmoid()
        decay = (-self._A_exp * F.softplus(a_h + self._dt_bias)).exp()

        # Recurrence (float32, state on host, bmm for speed)
        self._host_state *= decay.unsqueeze(-1).unsqueeze(-1)
        kv_mem = torch.bmm(k_h.unsqueeze(1), self._host_state).squeeze(1)  # (H, V)
        delta = (v_h - kv_mem) * beta.unsqueeze(-1)
        self._host_state += torch.bmm(k_h.unsqueeze(2), delta.unsqueeze(1))  # rank-1 update
        output_h = torch.bmm(q_h.unsqueeze(1), self._host_state).squeeze(1)  # (H, V)

        # Gated RMSNorm + head merge (host)
        z_heads = z_h.reshape(self.num_v_heads, self.head_v_dim)
        variance = output_h.pow(2).mean(-1, keepdim=True)
        output_normed = output_h * torch.rsqrt(variance + self.args.norm_eps)
        output_normed = output_normed * self._norm_w
        output_gated = output_normed * F.silu(z_heads)
        output_flat = output_gated.reshape(1, self.value_dim)

        # =================================================================
        # 9. Output projection  [on device]
        # =================================================================
        out_pad = torch.zeros(1, 1, B_pad, self.value_dim)
        out_pad[0, 0, 0, :] = output_flat
        output = ttnn.from_torch(
            out_pad,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output = ttnn.linear(output, self.out_proj, compute_kernel_config=self.proj_compute_config)

        return output  # (1, 1, B_pad, hidden_size)
