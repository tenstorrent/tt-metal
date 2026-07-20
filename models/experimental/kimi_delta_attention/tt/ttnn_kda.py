# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# KDA (Kimi Delta Attention) ttnn layer — end-to-end, layered on the recurrent KDA op.
# Mirrors torch_functional/kda_layer.py::KimiDeltaAttentionRef param-for-param.
#
# On device: q/k/v/f/b/g projections, L2-norm, KDA gate, diagonal-gate recurrence, gated-RMSNorm,
#            output projection.
# CPU fallback (allowed at the Phase-6 gate; logged for Phase 7): the depthwise causal short-conv
#            + SiLU. The in-tree GDN reference (ttnn_gated_deltanet.py causal_conv1d_ttnn) shows the
#            on-device port; deferred so Phase 6 proves the *algorithm*.

from __future__ import annotations

import torch
import ttnn
from einops import rearrange

from ..torch_functional.kda_layer import causal_short_conv
from .ttnn_kda_ops import kda_gate_ttnn, l2norm_ttnn, recurrent_kda_ttnn

_MM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _lin(name, w):  # store weight transposed to [in, out] for ttnn.linear
    return w


class TtKimiDeltaAttention:
    """ttnn KDA layer. Weights sourced from a torch KimiDeltaAttentionRef state_dict."""

    def __init__(self, ref, device):
        self.device = device
        self.H = ref.num_heads
        self.HV = ref.num_v_heads
        self.K = ref.head_k_dim
        self.V = ref.head_v_dim
        self.conv_size = ref.conv_size
        self.use_short_conv = ref.use_short_conv
        self.allow_neg_eigval = ref.allow_neg_eigval
        self.lower_bound = ref.lower_bound
        self.norm_eps = ref.norm_eps
        sd = ref.state_dict()

        def up(t, tile=True):
            return ttnn.from_torch(
                t.contiguous().to(torch.float32), dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT if tile else ttnn.ROW_MAJOR_LAYOUT, device=device,
            )

        # projections stored transposed -> [in, out]
        self.w_q = up(sd["q_proj.weight"].T)
        self.w_k = up(sd["k_proj.weight"].T)
        self.w_v = up(sd["v_proj.weight"].T)
        self.w_f0 = up(sd["f_proj.0.weight"].T)
        self.w_f1 = up(sd["f_proj.1.weight"].T)
        self.w_b = up(sd["b_proj.weight"].T)
        self.w_g0 = up(sd["g_proj.0.weight"].T)
        self.w_g1 = up(sd["g_proj.1.weight"].T)
        self.b_g1 = up(sd["g_proj.1.bias"].reshape(1, 1, -1))
        self.w_o = up(sd["o_proj.weight"].T)
        # gate params reshaped to [HV,K] / [HV,1]
        self.A_log = up(sd["A_log"].reshape(self.HV, 1))
        self.dt_bias = up(sd["dt_bias"].reshape(self.HV, self.K))
        self.o_norm_w = up(sd["o_norm_weight"].reshape(1, 1, 1, self.V))
        # conv weights kept on host (CPU-fallback conv)
        if self.use_short_conv:
            self.q_conv = sd["q_conv"]
            self.k_conv = sd["k_conv"]
            self.v_conv = sd["v_conv"]

    def _proj(self, x, w):
        return ttnn.linear(x, w, compute_kernel_config=_MM)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        x = ttnn.from_torch(hidden_states.to(torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)

        q = self._proj(x, self.w_q)
        k = self._proj(x, self.w_k)
        v = self._proj(x, self.w_v)

        # --- CPU-fallback short conv + SiLU (Phase 7: port to ttnn.conv1d) ---
        if self.use_short_conv:
            q = ttnn.from_torch(causal_short_conv(ttnn.to_torch(q), self.q_conv), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
            k = ttnn.from_torch(causal_short_conv(ttnn.to_torch(k), self.k_conv), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
            v = ttnn.from_torch(causal_short_conv(ttnn.to_torch(v), self.v_conv), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        else:
            q, k, v = ttnn.silu(q), ttnn.silu(k), ttnn.silu(v)

        # gate g and beta
        g = self._proj(self._proj(x, self.w_f0), self.w_f1)          # [B,T,HV*K]
        beta = ttnn.sigmoid(self._proj(x, self.w_b))                  # [B,T,HV]
        if self.allow_neg_eigval:
            beta = ttnn.multiply(beta, 2.0)

        # reshape to heads
        q = ttnn.reshape(q, [B, T, self.H, self.K])
        k = ttnn.reshape(k, [B, T, self.H, self.K])
        v = ttnn.reshape(v, [B, T, self.HV, self.V])
        g = ttnn.reshape(g, [B, T, self.HV, self.K])

        q = l2norm_ttnn(q)
        k = l2norm_ttnn(k)
        g = kda_gate_ttnn(g, self.A_log, self.dt_bias, self.lower_bound)

        o, _ = recurrent_kda_ttnn(q, k, v, g, beta, device=self.device)   # [B,T,HV,V]

        # gated RMSNorm (norm before gate) then o_proj
        gate = ttnn.reshape(ttnn.add(self._proj(self._proj(x, self.w_g0), self.w_g1), self.b_g1), [B, T, self.HV, self.V])
        o_norm = ttnn.rms_norm(o, epsilon=self.norm_eps, weight=self.o_norm_w)
        o = ttnn.multiply(o_norm, ttnn.sigmoid(gate))
        o = ttnn.reshape(o, [B, T, self.HV * self.V])
        o = self._proj(o, self.w_o)
        return ttnn.to_torch(o)
