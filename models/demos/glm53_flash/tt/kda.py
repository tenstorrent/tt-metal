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

``GLM53_KDA_FUSED=1`` (design 8.1) runs the whole decode block as 6 ops: the packed qkv matmul,
two gate matmuls (``[f_a | g_a | b]`` then the block-diagonal ``[f_b, g_b, I]``), ONE
``ttnn.transformer.kda_decode`` program (conv + SiLU from the device-owned history, L2 norms,
gates, the per-key-channel delta-rule recurrence, the gated RMSNorm; state and conv history
updated in place; row-major output so the users of a head can be split over cores), its
tilize, then o_proj. ``GLM53_KDA_FUSED=2`` keeps the composite's conv/gates and
replaces only the recurrence with ``fused_recurrent_gated_delta_rule`` (its rank-4 per-key ``g``,
ring mode in place on ``state.recurrent``, q/k scales folded into the L2-norm weights).

State contract (``ttKDA.KdaState``): ``recurrent [B, H, K, V]`` fp32 TILE, ``convolution
[B, kernel-1, Q+K+V]`` bf16 ROW_MAJOR, both DRAM-interleaved; the same tensors carry from prefill
into decode.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.kda import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights, load_kda_weights
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.matmul_configs import DecodeLinear
from models.demos.glm53_flash.tt.mesh import GLMMesh, interleave_blocks, single_chip

L2_NORM_EPS = 1e-6  # FLA l2norm: x / sqrt(sum(x^2) + 1e-6)
# decode path: "composite" (default) | "block" (GLM53_KDA_FUSED=1: one kda_decode program for the whole block) |
# "recurrence" (GLM53_KDA_FUSED=2: composite conv/gates + the fused recurrence op)
_FUSED_MODES = {
    "0": "composite",
    "1": "block",
    "2": "recurrence",
    "composite": "composite",
    "block": "block",
    "recurrence": "recurrence",
}
KDA_FUSED_DEFAULT = _FUSED_MODES.get(os.environ.get("GLM53_KDA_FUSED", "0"), "composite")


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
        mesh: GLMMesh | None = None,
        fused_decode: str | bool | None = None,
    ):
        """``attn_state_dict``: the raw checkpoint tensors of ``self_attn.`` (prefix stripped).
        ``fused_decode``: "composite" | "block" | "recurrence" (default: the ``GLM53_KDA_FUSED``
        environment knob, composite); ``True`` means "block".

        On a mesh the heads are split over the columns (``mesh.tp``): every projection that is
        per head is column-sharded head-major, the state holds the local heads, the decode output
        projection is row-parallel and all-reduced over the columns (ttKDA's prefill weights use
        the same placement through ``tensor_parallel_axis=1``)."""
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        tp = self.mesh.tp
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.kcfg = kda_config(cfg)
        assert self.kcfg.num_heads % tp == 0, (self.kcfg.num_heads, tp)
        self.H_total = self.kcfg.num_heads
        self.H, self.K, self.V = self.kcfg.num_heads // tp, self.kcfg.head_k_dim, self.kcfg.head_v_dim
        self.qkv_dim = self.kcfg.q_dim // tp  # local heads x K
        self.conv_width = (self.kcfg.q_dim + self.kcfg.k_dim + self.kcfg.v_dim) // tp  # [q_c | k_c | v_c]
        raw = normalize_kda_state_dict(attn_state_dict, self.kcfg)
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
        # ---- decode weights (per-head tensors column-sharded head-major: [q_c | k_c | v_c] per chip)
        up = self.mesh.upload
        self.w_qkv = up(
            interleave_blocks([raw["q_proj.weight"].t(), raw["k_proj.weight"].t(), raw["v_proj.weight"].t()], tp),
            weight_dtype,
            cols=-1,
        )
        taps = [
            interleave_blocks(
                [raw["q_conv1d.weight"][:, 0, j], raw["k_conv1d.weight"][:, 0, j], raw["v_conv1d.weight"][:, 0, j]], tp
            ).reshape(1, 1, 1, -1)
            for j in range(self.kcfg.conv_kernel_size)
        ]
        self.taps = [up(t.float(), ttnn.bfloat16, cols=-1) for t in taps]
        self.w_fa = up(raw["f_a_proj.weight"].t(), ttnn.bfloat16)
        self.w_fb = up(raw["f_b_proj.weight"].t(), ttnn.bfloat16, cols=-1)
        self.w_ga = up(raw["g_a_proj.weight"].t(), ttnn.bfloat16)
        self.w_gb = up(raw["g_b_proj.weight"].t(), ttnn.bfloat16, cols=-1)
        self.w_b = up(raw["b_proj.weight"].t(), ttnn.bfloat16, cols=-1)
        a_exp = raw["A_log"].float().reshape(self.H_total, 1).exp().expand(self.H_total, self.K).reshape(1, 1, 1, -1)
        self.a_exp = up(a_exp, ttnn.float32, cols=-1)
        self.dt_bias = up(raw["dt_bias"].float().reshape(1, 1, 1, -1), ttnn.float32, cols=-1)
        self.o_norm_w = up(raw["o_norm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.w_o = up(raw["o_proj.weight"].t(), weight_dtype, cols=-2)
        self.fp32 = _fp32_compute(device)
        self.hifi2 = _hifi2_compute(device)
        # decode-shaped (M <= 32) projections use the 1D mcast configs of tt/matmul_configs.py
        self.lin_qkv = DecodeLinear(device, self.w_qkv)
        self.lin_o = DecodeLinear(device, self.w_o)
        self.lin_fa = DecodeLinear(device, self.w_fa)
        self.lin_fb = DecodeLinear(device, self.w_fb)
        self.lin_ga = DecodeLinear(device, self.w_ga)
        self.lin_gb = DecodeLinear(device, self.w_gb)
        self.lin_b = DecodeLinear(device, self.w_b)
        self.gate_lower_bound = float(self.kcfg.gate_lower_bound)
        self._scalars: dict[int, ttnn.Tensor] = {}
        if fused_decode is None:
            self.fused_decode = KDA_FUSED_DEFAULT
        elif isinstance(fused_decode, bool):
            self.fused_decode = "block" if fused_decode else "composite"
        else:
            self.fused_decode = _FUSED_MODES[str(fused_decode)]
        K, V, H = self.K, self.V, self.H
        if self.fused_decode == "recurrence":
            # rms_norm(x, eps/K) = sqrt(K) l2norm(x): the weights fold l2norm(q) K^-0.5 (FLA's query scale, so
            # the op runs with scale 1) and l2norm(k) into the norms
            self.q_norm_w = up(torch.full((1, 1, 1, K), 1.0 / K), ttnn.float32)
            self.k_norm_w = up(torch.full((1, 1, 1, K), K**-0.5), ttnn.float32)
            self._ring_idx: dict[int, ttnn.Tensor] = {}
        if self.fused_decode == "block":
            # the kda_decode program's constants (its tensor contract is in the op's docstring)
            self.taps_blk = up(
                torch.stack([t.reshape(-1).float() for t in taps]).reshape(1, 1, 4, -1), ttnn.float32, cols=-1
            )
            self.chan_consts = up(
                torch.stack([raw["dt_bias"].float().reshape(-1), a_exp.reshape(-1)]).reshape(1, 1, 2, -1),
                ttnn.float32,
                cols=-1,
            )
            self.o_norm_w32 = up(raw["o_norm.weight"].float().reshape(1, 1, 1, -1), ttnn.float32)
            sel = torch.zeros(33, 32, 32)
            sel[torch.arange(32), 0, torch.arange(32)] = 1.0  # tile r: 1 at (0, r)
            sel[32] = 1.0  # all ones: row sums by matmul
            self.selectors = up(sel.reshape(33 * 32, 32), ttnn.bfloat16)
            # gates in two matmuls: x -> [f_a | g_a | b_logits] (bf16, as HF), then the block-diagonal
            # [f_b, g_b, I] -> [f | og | b_logits] (fp32 accumulate); the identity block carries the logits
            Hb = -(-H // 32) * 32
            # per chip: [f_a | g_a | b columns of its heads | zero pad to Hb] (an even column split over the TP
            # columns hands every chip its own block)
            w_fa_t, w_ga_t, w_b_t = (raw[n].t() for n in ("f_a_proj.weight", "g_a_proj.weight", "b_proj.weight"))
            parts = [
                torch.cat([w_fa_t, w_ga_t, w_b_t[:, c * H : (c + 1) * H], torch.zeros(w_b_t.shape[0], Hb - H)], dim=1)
                for c in range(tp)
            ]
            self.w_g1 = up(torch.cat(parts, dim=1), ttnn.bfloat16, cols=-1)
            d_a = raw["f_a_proj.weight"].shape[0]
            # per TP column c: [f_b | g_b] columns of ITS heads (head-major, heads c*H .. (c+1)*H) and the identity
            # on its b logits; concatenated so the even column split hands every chip its own block
            fb_t = raw["f_b_proj.weight"].t().float()  # [d_a, H_total*K]
            gb_t = raw["g_b_proj.weight"].t().float()  # [d_a, H_total*V]
            blocks = []
            for c in range(tp):
                w_g2 = torch.zeros(2 * d_a + Hb, H * K + H * V + Hb)
                w_g2[:d_a, : H * K] = fb_t[:, c * H * K : (c + 1) * H * K]
                w_g2[d_a : 2 * d_a, H * K : H * K + H * V] = gb_t[:, c * H * V : (c + 1) * H * V]
                w_g2[2 * d_a : 2 * d_a + H, H * K + H * V : H * K + H * V + H] = torch.eye(H)
                blocks.append(w_g2)
            self.w_g2 = up(torch.cat(blocks, dim=1).to(torch.bfloat16), ttnn.bfloat16, cols=-1)
            # decode-shaped (M <= 32) gate matmuls use the 1D mcast configs of tt/matmul_configs.py
            self.lin_g1 = DecodeLinear(device, self.w_g1)
            self.lin_g2 = DecodeLinear(device, self.w_g2)

    def scalar(self, value: int) -> ttnn.Tensor:
        """Cached replicated UINT32 device scalar (ttKDA's actual_start / actual_end)."""
        if value not in self._scalars:
            self._scalars[value] = self.mesh.upload(
                torch.tensor([[[[int(value)]]]], dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        return self._scalars[value]

    def ring_idx(self, batch_size: int) -> ttnn.Tensor:
        """Ring-mode block index idx[h] = h for B*H local heads (the fused op then reads and writes every
        head's state in place in the caller's [B*H, K, V] buffer); cached per batch size."""
        if batch_size not in self._ring_idx:
            idx = torch.arange(batch_size * self.H, dtype=torch.int32).reshape(1, -1)
            self._ring_idx[batch_size] = self.mesh.upload(idx, ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        return self._ring_idx[batch_size]

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
            kw = {}
            if self.mesh.rows > 1:
                # sequence-parallel over the rows: the grouped scan needs summary_group_chunks to divide the
                # LOCAL chunk count (active_seq_len / rows / 32); largest divisor <= the default 20
                local_chunks = active_seq_len // self.mesh.rows // ttnn.TILE_SIZE
                g = max(d for d in range(1, min(20, local_chunks) + 1) if local_chunks % d == 0)
                kw["recurrence"] = KDARecurrenceProgramConfig(summary_group_chunks=g)
            program_config = KDAProgramConfig(
                gated_rms_output_dtype=ttnn.bfloat16,
                input_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
                output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
                **kw,
            )
            tt_ccl = None
            if self.mesh.tp > 1:
                from models.common.modules.tt_ccl import get_tt_ccl

                tt_ccl = get_tt_ccl(self.device)  # ttKDA's TP reduce-scatter semaphores (one set per mesh)
            layer = ttKDA(
                self.device,
                self.kcfg,
                None,
                layer_idx=self.layer_idx,
                weights=self.prefill_weights,
                tt_ccl=tt_ccl,
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
        Returns y [1, T_phys, hidden] bf16 and the new state (the old one is only read).
        On a multi-row mesh ttKDA is sequence-parallel over the rows: ``x`` is this row's slab
        (row r holds rows [r*T/rows, (r+1)*T/rows) of the chunk, T = rows * T_phys_local) and the
        returned carries are replicated over the rows (TP-sharded heads as in decode)."""
        layer = self.prefill_layer(x.shape[1] * self.mesh.rows)
        y, new_state = layer.forward(x, state, actual_start, actual_end)
        if self.mesh.tp > 1:
            # ttKDA reduce-scatters the TP partials on the hidden dim; make the output replicated
            y = self.mesh.all_gather_cols(y, dim=-1)
        return y, new_state

    # ----------------------------------------------------------------- decode
    def decode_forward(self, x: ttnn.Tensor, state: KdaState, *, inplace: bool = True) -> tuple[ttnn.Tensor, KdaState]:
        """x: [1, 1, B, hidden] bf16 TILE, one token per user. Returns y [1, 1, B, hidden] and the
        state after the token: ``inplace`` (default) updates ``state``'s tensors in place and returns
        the same object (trace-safe); otherwise returns a fresh ``KdaState``. Composite v1 (design 5.3)."""
        B = x.shape[-2]
        H, K, V = self.H, self.K, self.V
        BH = B * H
        if state.recurrent.shape[0] != B:
            raise ValueError(f"state batch {state.recurrent.shape[0]} != input batch {B}")
        if self.fused_decode == "block":
            return self._block_decode(x, state, inplace)

        # --- projections + causal conv (kernel 4) over the 3-row history + current token
        qkv = self.lin_qkv(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,3*QKV]
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
        f = self.lin_fa(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        f = self.lin_fb(f, dtype=ttnn.float32, compute_kernel_config=self.fp32)  # [1,1,B,H*K]
        f = ttnn.add(f, self.dt_bias, output_tensor=f)
        f = ttnn.multiply(f, self.a_exp, output_tensor=f)
        g = ttnn.sigmoid(f, output_tensor=f)
        g = ttnn.multiply(g, self.gate_lower_bound, output_tensor=g)  # [1,1,B,H*K] fp32 log decay
        if self.fused_decode == "composite":
            decay = ttnn.exp(g, output_tensor=g)  # [1,1,B,H*K] fp32
        beta = ttnn.sigmoid(self.lin_b(x, dtype=ttnn.float32, compute_kernel_config=self.fp32))  # [1,1,B,H]
        og = self.lin_ga(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        og = self.lin_gb(og, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,H*V]

        # --- per-head L2 norm of q, k (FLA: x / sqrt(sum x^2 + 1e-6)); q also scaled by K^-0.5
        qh = ttnn.reshape(ttnn.typecast(q, ttnn.float32), [1, B, H, K])
        kh = ttnn.reshape(ttnn.typecast(k, ttnn.float32), [1, B, H, K])
        vh = ttnn.reshape(ttnn.typecast(v, ttnn.float32), [1, B, H, V])
        if self.fused_decode == "recurrence":
            qh = ttnn.rms_norm(qh, epsilon=L2_NORM_EPS / K, weight=self.q_norm_w, compute_kernel_config=self.fp32)
            kh = ttnn.rms_norm(kh, epsilon=L2_NORM_EPS / K, weight=self.k_norm_w, compute_kernel_config=self.fp32)
            o, new_recurrent = self._fused_recurrence(qh, kh, vh, g, beta, state, inplace)
            return self._output(x, o, og, new_conv, state, new_recurrent, inplace)
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
        return self._output(x, ttnn.reshape(o, [1, B, H, V]), og, new_conv, state, new_recurrent, inplace)

    def _block_decode(self, x: ttnn.Tensor, state: KdaState, inplace: bool) -> tuple[ttnn.Tensor, KdaState]:
        """The 6-op decode block: qkv matmul, two gate matmuls, ONE ``kda_decode`` program (conv, norms,
        gates, recurrence, gated RMSNorm; state + conv history in place), its tilize, o_proj. ``inplace=False``
        runs on copies of the state tensors (the program always updates its inputs in place)."""
        rec, conv = state.recurrent, state.convolution
        if not inplace:
            rec, conv = ttnn.clone(rec), ttnn.clone(conv)
        qkv = self.lin_qkv(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,3*H*K]
        g1 = self.lin_g1(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [f_a | g_a | b]
        g2 = self.lin_g2(g1, dtype=ttnn.float32, compute_kernel_config=self.fp32)  # [f | og | b]
        o = ttnn.transformer.kda_decode(
            qkv,
            g2,
            conv,
            rec,
            self.taps_blk,
            self.chan_consts,
            self.o_norm_w32,
            self.selectors,
            self.H,
            gate_lower_bound=self.gate_lower_bound,
            l2_eps=L2_NORM_EPS,
            norm_eps=self.kcfg.norm_eps,
        )  # [1,1,B,H*V] bf16 ROW_MAJOR (one row per user), normed and gated
        ttnn.deallocate(qkv)
        ttnn.deallocate(g1)
        ttnn.deallocate(g2)
        o_t = ttnn.to_layout(o, ttnn.TILE_LAYOUT)
        ttnn.deallocate(o)
        y = self.lin_o(o_t, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,hidden]
        ttnn.deallocate(o_t)
        y = self.mesh.all_reduce_cols(y)
        if inplace:
            return y, state
        return y, KdaState(recurrent=rec, convolution=conv)

    def _fused_recurrence(self, qh, kh, vh, g, beta, state: KdaState, inplace: bool):
        """One ``fused_recurrent_gated_delta_rule`` call for all B*H local heads: q [1,B,H,K] (l2-normed and
        K^-0.5-scaled), k [1,B,H,K] (l2-normed), v [1,B,H,V] fp32; g [1,1,B,H*K] per-key-channel log decay;
        beta [1,1,B,H]. Returns o [1,B,H,V] fp32 and the new recurrent state [B,H,K,V]: ``inplace`` runs the
        op's ring mode on a view of ``state.recurrent`` (read once, written once, no copy)."""
        B, H, K, V = state.recurrent.shape[0], self.H, self.K, self.V
        q4 = ttnn.reshape(qh, [B, 1, H, K])  # views: the tiled dims are unchanged
        k4 = ttnn.reshape(kh, [B, 1, H, K])
        v4 = ttnn.reshape(vh, [B, 1, H, V])
        g4 = ttnn.reshape(g, [B, 1, H, K])  # [1,1,B,H*K] -> head rows (same relayout as the composite's)
        b3 = ttnn.reshape(beta, [B, 1, H])
        if inplace:
            ring = ttnn.reshape(state.recurrent, [B * H, K, V])  # 0-cost view of the persistent state
            o, _ = ttnn.transformer.fused_recurrent_gated_delta_rule(
                q4,
                k4,
                v4,
                g4,
                b3,
                scale=1.0,
                initial_state=ring,
                initial_state_block_idx=self.ring_idx(B),
                output_per_token_state=True,
            )
            new_recurrent = state.recurrent
        else:
            o, new_recurrent = ttnn.transformer.fused_recurrent_gated_delta_rule(
                q4, k4, v4, g4, b3, scale=1.0, initial_state=state.recurrent, output_final_state=True
            )
        return ttnn.reshape(o, [1, B, H, V]), new_recurrent

    def _output(self, x, oh, og, new_conv, state: KdaState, new_recurrent, inplace: bool):
        """Gated RMSNorm per head (rmsnorm(o) * w * sigmoid(gate)), o_proj, TP all-reduce and the state
        hand-back. ``oh`` [1,B,H,V] fp32."""
        B, H, V = oh.shape[1], self.H, self.V
        oh = ttnn.rms_norm(oh, epsilon=self.kcfg.norm_eps, weight=self.o_norm_w, compute_kernel_config=self.fp32)
        gate = ttnn.sigmoid(ttnn.reshape(ttnn.typecast(og, ttnn.float32), [1, B, H, V]))
        oh = ttnn.multiply(oh, gate, output_tensor=oh)
        o_flat = ttnn.typecast(ttnn.reshape(oh, [1, 1, B, H * V]), ttnn.bfloat16)
        y = self.lin_o(o_flat, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,hidden]
        y = self.mesh.all_reduce_cols(y)  # TP partial -> replicated over the row's columns
        if inplace:
            # write the new carries back into the caller's persistent tensors so a captured decode
            # trace reads the updated state on the next replay (device-owned autoregressive state);
            # the fused path already updated the recurrent state in place
            if new_recurrent is not state.recurrent:
                ttnn.copy(new_recurrent, state.recurrent)
                ttnn.deallocate(new_recurrent)
            ttnn.copy(new_conv, state.convolution)
            ttnn.deallocate(new_conv)
            return y, state
        return y, KdaState(recurrent=new_recurrent, convolution=new_conv)
