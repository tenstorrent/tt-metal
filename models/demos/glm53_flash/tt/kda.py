# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA (Kimi Delta Attention) layer for GLM-5.3-Flash on one device.

Prefill reuses ``models.demos.deepseek_v3_d_p.tt.kda.ttKDA`` (chunked delta-rule kernels, proven
against the Kimi-K3 torch reference, which ``reference/check_kda_reference.py`` shows is the same
function as HF's ``Glm5NextTextLinearAttention`` with GLM weights at PCC 0.99999).

Decode (T=1, B users) is the design's v1 composite on the same device-owned state:

    S *= exp(g)[.., :, None]       (per-channel decay, g = -5 sigmoid(exp(A_log) (f + dt_bias)))
    kv = k^T S ; delta = beta (v - kv) ; S += k^T delta ; o = q S

with q, k L2-normalized per head and q additionally scaled by K^-0.5 (FLA semantics), followed by
the gated RMSNorm (sigmoid(g_b(g_a(x)))) and o_proj. Every elementwise step is a broadcast
multiply plus a reduction so no tile padding ever enters a contraction.

State contract (``ttKDA.KdaState``): ``recurrent [B, H, K, V]`` fp32 TILE, ``convolution
[B, kernel-1, Q+K+V]`` bf16 ROW_MAJOR, both DRAM-interleaved; the same tensors carry from prefill
into decode.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.kda import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights, load_kda_weights
from models.demos.glm53_flash.tt.config import GLM53FlashConfig

L2_NORM_EPS = 1e-6  # FLA l2norm: x / sqrt(sum(x^2) + 1e-6)


def kda_config(cfg: GLM53FlashConfig) -> KDAConfig:
    return KDAConfig(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.kda_num_heads,
        head_k_dim=cfg.kda_head_dim,
        head_v_dim=cfg.kda_head_dim,
        conv_kernel_size=cfg.kda_conv_kernel,
        norm_eps=cfg.rms_norm_eps,
        use_full_rank_gate=False,
        gate_lower_bound=cfg.kda_gate_lower_bound,
    )


def device_scalar_u32(device, value: int) -> ttnn.Tensor:
    """Replicated UINT32 row-major [1,1,1,1] device scalar (ttKDA's actual_start / actual_end)."""
    return ttnn.from_torch(
        torch.tensor([[[[int(value)]]]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _fp32_compute(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _hifi2_compute(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _upload(device, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


class TtKDA(LightweightModule):
    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        attn_state_dict: dict[str, torch.Tensor],
        *,
        layer_idx: int,
        weight_cache_path: Path | None = None,
        weight_dtype=ttnn.bfloat8_b,
    ):
        """``attn_state_dict``: the raw checkpoint tensors of ``self_attn.`` (prefix stripped)."""
        self.device = device
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.kcfg = kda_config(cfg)
        self.H, self.K, self.V = self.kcfg.num_heads, self.kcfg.head_k_dim, self.kcfg.head_v_dim
        self.qkv_dim = self.kcfg.q_dim
        self.conv_width = self.kcfg.q_dim + self.kcfg.k_dim + self.kcfg.v_dim
        raw = normalize_kda_state_dict(attn_state_dict, self.kcfg)
        self.raw = raw
        self.weight_cache_path = weight_cache_path
        # ---- prefill (ttKDA) weights, materialized once and shared across sequence lengths
        self.prefill_weights: KDAWeights = load_kda_weights(
            device,
            self.kcfg,
            raw,
            weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.kda",
            tensor_parallel_axis=1,
        )
        self._prefill_layers: dict[int, ttKDA] = {}
        # ---- decode weights
        self.w_qkv = _upload(
            device,
            torch.cat([raw["q_proj.weight"], raw["k_proj.weight"], raw["v_proj.weight"]], dim=0).t(),
            weight_dtype,
        )
        taps = [
            torch.cat(
                [raw["q_conv1d.weight"][:, 0, j], raw["k_conv1d.weight"][:, 0, j], raw["v_conv1d.weight"][:, 0, j]]
            ).reshape(1, 1, 1, -1)
            for j in range(self.kcfg.conv_kernel_size)
        ]
        self.taps = [_upload(device, t.float(), ttnn.bfloat16) for t in taps]
        self.w_fa = _upload(device, raw["f_a_proj.weight"].t(), ttnn.bfloat16)
        self.w_fb = _upload(device, raw["f_b_proj.weight"].t(), ttnn.bfloat16)
        self.w_ga = _upload(device, raw["g_a_proj.weight"].t(), ttnn.bfloat16)
        self.w_gb = _upload(device, raw["g_b_proj.weight"].t(), ttnn.bfloat16)
        self.w_b = _upload(device, raw["b_proj.weight"].t(), ttnn.bfloat16)
        a_exp = raw["A_log"].float().reshape(self.H, 1).exp().expand(self.H, self.K).reshape(1, 1, 1, -1)
        self.a_exp = _upload(device, a_exp, ttnn.float32)
        self.dt_bias = _upload(device, raw["dt_bias"].float().reshape(1, 1, 1, -1), ttnn.float32)
        self.o_norm_w = _upload(device, raw["o_norm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.w_o = _upload(device, raw["o_proj.weight"].t(), weight_dtype)
        self.fp32 = _fp32_compute(device)
        self.hifi2 = _hifi2_compute(device)
        self.gate_lower_bound = float(self.kcfg.gate_lower_bound)

    # ------------------------------------------------------------------ state
    def allocate_state(self, batch_size: int = 1) -> KdaState:
        """Zero state for ``batch_size`` users (ttKDA layout)."""
        return KdaState(
            recurrent=ttnn.zeros(
                (batch_size, self.H, self.K, self.V),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            convolution=ttnn.zeros(
                (batch_size, self.kcfg.conv_kernel_size - 1, self.conv_width),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )

    # ---------------------------------------------------------------- prefill
    def prefill_layer(self, active_seq_len: int) -> ttKDA:
        """ttKDA instance for one physical (32-aligned) chunk length; weights are shared."""
        if active_seq_len % ttnn.TILE_SIZE:
            raise ValueError(f"active_seq_len must be a multiple of {ttnn.TILE_SIZE}, got {active_seq_len}")
        layer = self._prefill_layers.get(active_seq_len)
        if layer is None:
            program_config = KDAProgramConfig(
                gated_rms_output_dtype=ttnn.bfloat16,
                input_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
                output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
            )
            layer = ttKDA(
                self.device,
                self.kcfg,
                None,
                layer_idx=self.layer_idx,
                weights=self.prefill_weights,
                sp_axis=0,
                tp_axis=1,
                program_config=program_config,
                active_seq_len=active_seq_len,
            )
            self._prefill_layers[active_seq_len] = layer
        return layer

    def prefill_forward(
        self, x: ttnn.Tensor, state: KdaState, actual_start: ttnn.Tensor, actual_end: ttnn.Tensor | None = None
    ) -> tuple[ttnn.Tensor, KdaState]:
        """x: [1, T_phys, hidden] bf16 TILE (T_phys % 32 == 0; rows past ``actual_end`` are padding).
        Returns y [1, T_phys, hidden] bf16 and the new state (the old one is only read)."""
        layer = self.prefill_layer(x.shape[1])
        return layer.forward(x, state, actual_start, actual_end)

    # ----------------------------------------------------------------- decode
    def decode_forward(self, x: ttnn.Tensor, state: KdaState) -> tuple[ttnn.Tensor, KdaState]:
        """x: [1, 1, B, hidden] bf16 TILE, one token per user. Returns y [1, 1, B, hidden] and the
        new state. Composite v1 (section 5.3 of the design)."""
        B = x.shape[-2]
        H, K, V = self.H, self.K, self.V
        BH = B * H
        if state.recurrent.shape[0] != B:
            raise ValueError(f"state batch {state.recurrent.shape[0]} != input batch {B}")

        # --- projections + causal conv (kernel 4) over the 3-row history + current token
        qkv = ttnn.linear(x, self.w_qkv, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,3*QKV]
        hist = ttnn.to_layout(state.convolution, ttnn.TILE_LAYOUT)  # [B, 3, W]
        hist4 = ttnn.reshape(hist, [1, B, self.kcfg.conv_kernel_size - 1, self.conv_width])
        conv = ttnn.multiply(qkv, self.taps[3])
        for j in range(self.kcfg.conv_kernel_size - 1):
            row = ttnn.slice(hist4, [0, 0, j, 0], [1, B, j + 1, self.conv_width])  # [1,B,1,W]
            row = ttnn.reshape(row, [1, 1, B, self.conv_width])
            conv = ttnn.add(conv, ttnn.multiply(row, self.taps[j]), output_tensor=conv)
        conv = ttnn.silu(conv, output_tensor=conv)  # [1,1,B,3*QKV] bf16
        # new history = rows 1..2 of the old history + the raw (pre-conv) projection of this token
        qkv_rm = ttnn.reshape(ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT), [B, 1, self.conv_width])
        tail = ttnn.slice(state.convolution, [0, 1, 0], [B, self.kcfg.conv_kernel_size - 1, self.conv_width])
        new_conv = ttnn.concat([tail, qkv_rm], dim=1)  # [B, 3, W] row-major bf16
        ttnn.deallocate(qkv)
        ttnn.deallocate(hist)

        q = ttnn.slice(conv, [0, 0, 0, 0], [1, 1, B, self.qkv_dim])
        k = ttnn.slice(conv, [0, 0, 0, self.qkv_dim], [1, 1, B, 2 * self.qkv_dim])
        v = ttnn.slice(conv, [0, 0, 0, 2 * self.qkv_dim], [1, 1, B, 3 * self.qkv_dim])
        ttnn.deallocate(conv)

        # --- gates (fp32, as HF)
        f = ttnn.linear(x, self.w_fa, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        f = ttnn.linear(f, self.w_fb, dtype=ttnn.float32, compute_kernel_config=self.fp32)  # [1,1,B,H*K]
        f = ttnn.add(f, self.dt_bias, output_tensor=f)
        f = ttnn.multiply(f, self.a_exp, output_tensor=f)
        g = ttnn.sigmoid(f, output_tensor=f)
        g = ttnn.multiply(g, self.gate_lower_bound, output_tensor=g)
        decay = ttnn.exp(g, output_tensor=g)  # [1,1,B,H*K] fp32
        beta = ttnn.sigmoid(ttnn.linear(x, self.w_b, dtype=ttnn.float32, compute_kernel_config=self.fp32))  # [1,1,B,H]
        og = ttnn.linear(x, self.w_ga, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        og = ttnn.linear(og, self.w_gb, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,H*V]

        # --- per-head L2 norm of q, k (FLA: x / sqrt(sum x^2 + 1e-6)); q also scaled by K^-0.5
        qh = ttnn.reshape(ttnn.typecast(q, ttnn.float32), [1, B, H, K])
        kh = ttnn.reshape(ttnn.typecast(k, ttnn.float32), [1, B, H, K])
        vh = ttnn.reshape(ttnn.typecast(v, ttnn.float32), [1, B, H, V])
        qh = ttnn.multiply(ttnn.rms_norm(qh, epsilon=L2_NORM_EPS / K, compute_kernel_config=self.fp32), 1.0 / K)
        kh = ttnn.multiply(ttnn.rms_norm(kh, epsilon=L2_NORM_EPS / K, compute_kernel_config=self.fp32), K**-0.5)

        # --- recurrence on S [1, B*H, K, V] fp32
        S = ttnn.reshape(state.recurrent, [1, BH, K, V])
        decay_col = ttnn.transpose(ttnn.reshape(decay, [1, BH, 1, K]), -2, -1)  # [1,BH,K,1]
        S = ttnn.multiply(S, decay_col)  # new tensor (the input state is only read)
        k_row = ttnn.reshape(kh, [1, BH, 1, K])
        k_col = ttnn.transpose(k_row, -2, -1)  # [1,BH,K,1]
        kv = ttnn.sum(ttnn.multiply(S, k_col), dim=-2, keepdim=True)  # [1,BH,1,V]
        v_row = ttnn.reshape(vh, [1, BH, 1, V])
        beta_col = ttnn.reshape(beta, [1, BH, 1, 1])
        delta = ttnn.multiply(ttnn.subtract(v_row, kv), beta_col)  # [1,BH,1,V]
        S = ttnn.add(S, ttnn.multiply(k_col, delta), output_tensor=S)  # outer product k^T delta
        q_col = ttnn.transpose(ttnn.reshape(qh, [1, BH, 1, K]), -2, -1)
        o = ttnn.sum(ttnn.multiply(S, q_col), dim=-2, keepdim=True)  # [1,BH,1,V]
        new_recurrent = ttnn.reshape(S, [B, H, K, V])

        # --- gated RMSNorm per head: rmsnorm(o) * w * sigmoid(gate)
        oh = ttnn.reshape(o, [1, B, H, V])
        oh = ttnn.rms_norm(oh, epsilon=self.kcfg.norm_eps, weight=self.o_norm_w, compute_kernel_config=self.fp32)
        gate = ttnn.sigmoid(ttnn.reshape(ttnn.typecast(og, ttnn.float32), [1, B, H, V]))
        oh = ttnn.multiply(oh, gate, output_tensor=oh)
        o_flat = ttnn.typecast(ttnn.reshape(oh, [1, 1, B, H * V]), ttnn.bfloat16)
        y = ttnn.linear(o_flat, self.w_o, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,hidden]
        return y, KdaState(recurrent=new_recurrent, convolution=new_conv)
