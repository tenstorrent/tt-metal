# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention layer for a 1xN tensor-parallel Blackhole mesh: prefill + decode sharing one weight set.

Prefill reuses the DeepSeek-prefill demo's ``ttKDA`` (fused head-major projection, ``ttnn.experimental.kda`` chunked
recurrence, sigmoid-gated RMSNorm) with three changes: (1) a ``valid_len`` <= T_pad so requests whose length is not a
multiple of 32 run on a padded chunk — pad rows get beta = 0 and log-decay 0, so they neither write nor decay the
state, and the convolution carry is taken at ``valid_len``; (2) the output projection ends in an all-reduce (replicated
residual) instead of ttKDA's reduce-scatter; (3) chunked long prompts carry ``KdaState`` between chunks.
Decode runs the same projections on [1,B,hidden], a 4-tap shift-register convolution and the per-channel recurrent step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.ccl import KimiCCL
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.kda.decode_step import recurrent_kda_decode_ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.weights import as_device_tensor
from models.demos.deepseek_v3_d_p.reference.kda.config import KDA_SOFTPLUS_BETA, KDA_SOFTPLUS_THRESHOLD, KDAConfig
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_CHUNK_SIZE, KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA

TILE = ttnn.TILE_SIZE


@dataclass
class KDADecodeState:
    """Batched decode carries (fixed device addresses; updated in place)."""

    recurrent: ttnn.Tensor  # [B, H_local, K, V] fp32 TILE
    conv_history: list[ttnn.Tensor]  # 3 x [1, B, C_local] bf16 TILE, oldest first


def ceil32(n: int) -> int:
    return -(-n // TILE) * TILE


class KimiKDA:
    def __init__(
        self,
        mesh_device,
        config: KDAConfig,
        state_dict: Mapping[str, torch.Tensor] | None,
        *,
        layer_idx: int,
        ccl: KimiCCL,
        weight_cache_path: Path | None = None,
        long_prefill_chunks: int = 2048,
        gate_clamp_min: float | None = -5.0,
        exact_tail: int | None = 32,
    ):
        self.mesh_device = mesh_device
        self.ccl = ccl
        self.layer_idx = layer_idx
        self.full_config = config
        program_config = KDAProgramConfig(
            recurrence=KDARecurrenceProgramConfig(local_scan_strategy="direct"),
            qkv_channel_chunk_size=768,
            tp_ccl_topology=ccl.topology,
            gated_rms_output_dtype=ttnn.bfloat16,
            output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        )
        self.kda = ttKDA(
            mesh_device,
            config,
            state_dict,
            layer_idx=layer_idx,
            weight_cache_path=weight_cache_path,
            tt_ccl=ccl.tt_ccl,
            sp_axis=0,
            tp_axis=1,
            program_config=program_config,
        )
        self.config = self.kda.config  # TP-local head count
        self.weights = self.kda.weights
        self.tp = self.kda.tensor_parallel_size
        self.conv_width = self.kda._convolution_width  # Q_local + K_local + V_local
        self.scale = config.head_k_dim**-0.5
        self.long_prefill_chunks = long_prefill_chunks
        # The chunked ttnn.experimental.kda recurrence is exact for log-decays >= ~-8 and breaks down (NaN/garbage state)
        # for the very large negative gates this checkpoint produces (exp(A_log) up to ~200 -> g down to -228). Channels with
        # g < -8 forget everything within one token either way (exp(-8) = 3e-4), so clamping is numerically neutral (verified
        # against the fp32 oracle) and keeps the kernel in its valid range. Decode uses the fp32 recurrent step and needs no clamp.
        self.gate_clamp_min = gate_clamp_min
        self.exact_tail = (
            exact_tail  # None: kernel only (masked padding); N: last N..N+31 valid tokens via the exact fp32 step
        )
        # fp32 per-channel decay scale (-exp(A_log), repeated over K) for the exact decode step: the kernel's bf16 copy rounds
        # exp(A_log) (up to 201) to ~0.4 %, which is a visible per-step decay error for medium-decay heads.
        if state_dict is not None:
            a_log = state_dict["A_log"].float().reshape(-1)
            scale = -a_log.exp() if config.gate_lower_bound is None else a_log.exp()
            host = (
                scale.reshape(1, 1, config.num_heads, 1)
                .expand(-1, -1, -1, config.head_k_dim)
                .reshape(1, 1, config.q_dim)
                .contiguous()
            )
        else:
            host = None
        self.decay_scale_f32 = as_device_tensor(
            mesh_device,
            host,
            name=f"layer_{layer_idx}.kda.decay_scale_f32",
            dtype=ttnn.float32,
            shard_dim=-1,
            cache_path=weight_cache_path,
        )
        self._masks: dict[tuple[int, int], tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        # long-lived device tensors must exist BEFORE the first trace capture (a later allocation can land in a trace's
        # scratch region and be overwritten by every replay): allocate the exact-tail decode scratch now.
        self._tail_ds = self.allocate_decode_state(batch=1) if self.exact_tail is not None else None

    # ---- state -------------------------------------------------------------------------------
    def allocate_prefill_state(self) -> KdaState:
        return self.kda.allocate_state(batch_size=1)

    def allocate_decode_state(self, batch: int) -> KDADecodeState:
        c = self.config
        rec = ttnn.zeros(
            (batch, c.num_heads, c.head_k_dim, c.head_v_dim),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hist = [
            ttnn.zeros(
                (1, batch, self.conv_width),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(c.conv_kernel_size - 1)
        ]
        return KDADecodeState(recurrent=rec, conv_history=hist)

    def prefill_state_to_decode(self, ps: KdaState, ds: KDADecodeState, slot: int = 0) -> None:
        """Copy one request's prefill carries into decode slot ``slot`` (in place, address-stable)."""
        c = self.config
        B = ds.recurrent.shape[0]
        if B == 1:
            ttnn.copy(ps.recurrent, ds.recurrent)
        else:
            ttnn.experimental.slice_write(
                ps.recurrent,
                ds.recurrent,
                [slot, 0, 0, 0],
                [slot + 1, c.num_heads, c.head_k_dim, c.head_v_dim],
                [1, 1, 1, 1],
            )
        conv = ttnn.to_layout(ps.convolution, ttnn.TILE_LAYOUT)  # [1, 3, C]
        for j in range(c.conv_kernel_size - 1):
            row = ttnn.slice(conv, (0, j, 0), (1, j + 1, self.conv_width))  # [1,1,C]
            if B == 1:
                ttnn.copy(row, ds.conv_history[j])
            else:
                ttnn.experimental.slice_write(
                    row, ds.conv_history[j], [0, slot, 0], [1, slot + 1, self.conv_width], [1, 1, 1]
                )
            ttnn.deallocate(row)
        ttnn.deallocate(conv)

    def decode_state_to_prefill(self, ds: KDADecodeState, slot: int = 0) -> KdaState:
        """Extract slot ``slot`` as a prefill-style state (for tests / continued prefill)."""
        # NB: a full-range ttnn.slice returns the INPUT tensor itself (B == 1) -> clone so callers may free the result
        # without destroying the decode-state buffers.
        B = ds.recurrent.shape[0]
        if B == 1:
            rec = ttnn.clone(ds.recurrent, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            rows = list(ds.conv_history)  # aliases: never deallocate
        else:
            rec = ttnn.slice(ds.recurrent, (slot, 0, 0, 0), (slot + 1,) + tuple(ds.recurrent.shape)[1:])
            rows = [ttnn.slice(h, (0, slot, 0), (1, slot + 1, self.conv_width)) for h in ds.conv_history]
        cat = ttnn.concat(rows, dim=1)  # fresh tensor
        conv = ttnn.to_layout(cat, ttnn.ROW_MAJOR_LAYOUT)
        if conv is not cat:
            ttnn.deallocate(cat)
        if B != 1:
            for r in rows:
                ttnn.deallocate(r)
        return KdaState(recurrent=rec, convolution=conv)

    # ---- prefill -----------------------------------------------------------------------------
    def _valid_masks(self, t_pad: int, valid_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        key = (t_pad, valid_len)
        if key not in self._masks:
            m = torch.zeros(1, t_pad, 1)
            m[:, :valid_len] = 1.0
            mesh = ttnn.ReplicateTensorToMesh(self.mesh_device) if self.tp > 1 else None
            bf = ttnn.from_torch(
                m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, mesh_mapper=mesh
            )
            f32 = ttnn.from_torch(
                m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, mesh_mapper=mesh
            )
            self._masks[key] = (bf, f32)
        return self._masks[key]

    def _convolve(self, qkv: ttnn.Tensor, conv_state: ttnn.Tensor, valid_len: int):
        """Depthwise causal conv + SiLU; the new carry is the 3 rows ending at ``valid_len``."""
        c = self.config
        T = qkv.shape[1]
        qkv_rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        state_rm = ttnn.to_layout(conv_state, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = c.conv_kernel_size - 1
        if valid_len >= k:
            new_state = ttnn.slice(qkv_rm, (0, valid_len - k, 0), (1, valid_len, self.conv_width))
        else:  # fewer than 3 valid tokens: the carry still contains rows of the previous carry
            window = ttnn.concat([state_rm, qkv_rm], dim=1)
            new_state = ttnn.slice(window, (0, valid_len, 0), (1, valid_len + k, self.conv_width))
            ttnn.deallocate(window)
        q, k_, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            qkv_rm,
            state_rm,
            *self.weights.convolution_taps,
            c.q_dim,
            c.k_dim,
            c.v_dim,
            program_config=self.kda.qkv_convolution_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return q, k_, v, new_state

    def _prefill_kernel(
        self, hidden: ttnn.Tensor, state: KdaState, *, valid_len: int | None = None
    ) -> tuple[ttnn.Tensor, KdaState]:
        """Chunked-kernel path. hidden [1, T_pad, hidden] (T_pad % 32 == 0, replicated on every chip) -> ([1, 1, T_pad, hidden], new state).
        Rows >= valid_len are padding; they leave the carried state unchanged and their outputs are garbage."""
        if len(hidden.shape) == 4:
            hidden = ttnn.reshape(hidden, (hidden.shape[-3], hidden.shape[-2], hidden.shape[-1]))
        T = hidden.shape[1]
        assert T % KDA_CHUNK_SIZE == 0 and hidden.shape[0] == 1, hidden.shape
        valid_len = T if valid_len is None else valid_len
        kda = self.kda
        projected = kda._project_inputs(hidden)
        q, k, v, new_conv = self._convolve(projected.qkv, state.convolution, valid_len)
        gate, beta = kda._compute_gates(beta=projected.beta, decay_rank=projected.decay_rank)
        if self.gate_clamp_min is not None:
            gate = ttnn.clamp(gate, min=self.gate_clamp_min, max=0.0)
        if valid_len < T:
            mask_bf, mask_f32 = self._valid_masks(T, valid_len)
            gate = ttnn.multiply(
                gate, mask_bf, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )  # exp(0) = 1: no decay on pad rows
            beta = ttnn.multiply(beta, mask_f32, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # beta 0: no write on pad rows
        new_rec, out = kda.recurrence(q=q, k=k, v=v, gate=gate, beta=beta, initial_state=state.recurrent)
        out = kda._kda_rms_norm(out, projected.output_gate)
        out = ttnn.linear(
            out,
            self.weights.output_projection,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=kda.output_projection_compute_config,
        )
        out = ttnn.reshape(out, (1, 1, T, out.shape[-1]))
        out = self.ccl.all_reduce(out)
        return out, KdaState(recurrent=new_rec, convolution=new_conv)

    def forward_prefill(
        self, hidden: ttnn.Tensor, state: KdaState, *, valid_len: int | None = None
    ) -> tuple[ttnn.Tensor, KdaState]:
        """hidden [1, T_pad, hidden] (T_pad % 32 == 0, replicated) -> ([1, 1, T_pad, hidden], new state).

        The chunked ttnn.experimental.kda kernel takes a bf16 log-decay; with this checkpoint's large decays the bf16 rounding
        accumulates inside each 32-token chunk and leaves the carried state of the fast-decay heads inaccurate (PCC ~0.77 at
        T=128) although the outputs are fine (PCC ~0.997). So the leading tokens run through the kernel and the last
        ``exact_tail`` (32..63) valid tokens run through the exact fp32 one-token recurrence: fast heads forget the kernel's
        state error within a few tokens (exp(-5) per token) and slow heads were accurate to begin with. The final state
        handed to decode is therefore exact. Rows >= valid_len are padding (outputs zero)."""
        if len(hidden.shape) == 4:
            hidden = ttnn.reshape(hidden, (hidden.shape[-3], hidden.shape[-2], hidden.shape[-1]))
        T = hidden.shape[1]
        assert T % KDA_CHUNK_SIZE == 0 and hidden.shape[0] == 1, hidden.shape
        valid_len = T if valid_len is None else valid_len
        if self.exact_tail is None:
            if T > self.long_prefill_chunks:
                return self._forward_prefill_chunked(hidden, state, valid_len)
            return self._prefill_kernel(hidden, state, valid_len=valid_len)
        T_k = max(0, ((valid_len - self.exact_tail) // KDA_CHUNK_SIZE) * KDA_CHUNK_SIZE)
        outs = []
        if T_k > 0:
            head = ttnn.slice(hidden, (0, 0, 0), (1, T_k, hidden.shape[-1])) if T_k < T else hidden
            if T_k > self.long_prefill_chunks:
                out_k, state = self._forward_prefill_chunked(head, state, T_k)
            else:
                out_k, state = self._prefill_kernel(head, state, valid_len=T_k)
            if head is not hidden:
                ttnn.deallocate(head)
            outs.append(out_k)
        ds = self._tail_state()
        self.prefill_state_to_decode(state, ds)
        for t in range(T_k, valid_len):
            x_t = ttnn.slice(hidden, (0, t, 0), (1, t + 1, hidden.shape[-1]))
            outs.append(self.forward_decode(ttnn.reshape(x_t, (1, 1, 1, hidden.shape[-1])), ds))
            ttnn.deallocate(x_t)
        new_state = self.decode_state_to_prefill(ds)
        out = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)
        if len(outs) > 1:
            for o in outs:
                ttnn.deallocate(o)
        if out.shape[2] < T:
            out = ttnn.pad(out, [(0, 0), (0, 0), (0, T - out.shape[2]), (0, 0)], value=0.0)
        return out, new_state

    def _tail_state(self) -> KDADecodeState:
        if self._tail_ds is None:
            self._tail_ds = self.allocate_decode_state(batch=1)
        return self._tail_ds

    def _forward_prefill_chunked(self, hidden, state, valid_len):
        T = hidden.shape[1]
        outs = []
        for start in range(0, T, self.long_prefill_chunks):
            end = min(T, start + self.long_prefill_chunks)
            if start >= valid_len:  # fully padded chunk: nothing to carry; skip compute, emit zeros later
                break
            chunk = ttnn.slice(hidden, (0, start, 0), (1, end, hidden.shape[-1]))
            out, state = self._prefill_kernel(chunk, state, valid_len=min(end, valid_len) - start)
            ttnn.deallocate(chunk)
            outs.append(out)
        out = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)
        if out.shape[2] < T:
            out = ttnn.pad(out, [(0, 0), (0, 0), (0, T - out.shape[2]), (0, 0)], value=0.0)
        return out, state

    # ---- decode ------------------------------------------------------------------------------
    def _decode_gates_fp32(self, beta_raw: ttnn.Tensor, decay_rank: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Decode gates in fp32: g = decay_scale * softplus(f_b(decay_rank) + dt_bias), beta = sigmoid(b). The chunked prefill
        kernel forces a bf16 gate; the exact recurrent step must not inherit that rounding (|g| up to ~200 here)."""
        w = self.weights
        beta = ttnn.sigmoid(ttnn.typecast(beta_raw, ttnn.float32))
        raw = ttnn.linear(
            decay_rank,
            w.decay_output_projection,
            bias=w.decay_bias_flat,
            compute_kernel_config=self.kda.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.float32,
        )
        sp = ttnn.softplus(raw, beta=KDA_SOFTPLUS_BETA, threshold=KDA_SOFTPLUS_THRESHOLD)
        ttnn.deallocate(raw)
        if self.full_config.gate_lower_bound is None:
            gate = ttnn.multiply(sp, self.decay_scale_f32)  # -exp(A_log) in fp32
        else:
            gate = ttnn.multiply(
                ttnn.sigmoid(ttnn.multiply(sp, self.decay_scale_f32)), self.full_config.gate_lower_bound
            )
        ttnn.deallocate(sp)
        return gate, beta

    def forward_decode(self, hidden: ttnn.Tensor, ds: KDADecodeState) -> ttnn.Tensor:
        """hidden [1, 1, B, hidden] replicated -> [1, 1, B, hidden] replicated; updates ``ds`` in place."""
        c, kda, w = self.config, self.kda, self.weights
        B = hidden.shape[-2]
        Bmax = ds.recurrent.shape[0]
        assert B == Bmax, f"decode width {B} != allocated slots {Bmax} (bucketing comes later)"
        x = ttnn.reshape(hidden, (1, B, hidden.shape[-1]))
        p = kda._project_inputs(x)  # qkv [1,B,C], decay_rank [1,B,128], output_gate [1,B,V_loc], beta [1,B,H_loc]
        # shift-register convolution: history[0] oldest ... history[2] newest, taps[0..2] on history, taps[3] on current
        hist = ds.conv_history
        conv = ttnn.multiply(hist[0], w.convolution_taps[0], memory_config=_L1())
        for j in range(1, len(hist)):
            conv = ttnn.mac(hist[j], w.convolution_taps[j], conv)
        conv = ttnn.mac(p.qkv, w.convolution_taps[len(hist)], conv)
        conv = ttnn.silu(conv, memory_config=_L1())
        for j in range(len(hist) - 1):
            ttnn.copy(hist[j + 1], hist[j])
        ttnn.copy(p.qkv, hist[-1])
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, c.q_dim)), (B, 1, c.num_heads, c.head_k_dim))
        k = ttnn.reshape(
            ttnn.slice(conv, (0, 0, c.q_dim), (1, B, c.q_dim + c.k_dim)), (B, 1, c.num_heads, c.head_k_dim)
        )
        v = ttnn.reshape(
            ttnn.slice(conv, (0, 0, c.q_dim + c.k_dim), (1, B, self.conv_width)), (B, 1, c.num_heads, c.head_v_dim)
        )
        ttnn.deallocate(conv)
        gate, beta = self._decode_gates_fp32(p.beta, p.decay_rank)  # gate [1,B,H_loc*K] fp32 (log decay), beta fp32
        g = ttnn.reshape(gate, (B, 1, c.num_heads, c.head_k_dim))
        beta = ttnn.reshape(beta, (B, 1, c.num_heads))
        o, new_rec = recurrent_kda_decode_ttnn(
            q, k, v, beta, g, ds.recurrent, scale=self.scale, device=self.mesh_device
        )
        ttnn.copy(new_rec, ds.recurrent)  # in place: keeps the decode trace's state address stable
        ttnn.deallocate(new_rec)
        # gated RMSNorm per head: norm(o) * w * sigmoid(gate)
        o = ttnn.reshape(o, (B, c.num_heads, c.head_v_dim))
        o = ttnn.rms_norm(o, epsilon=self.full_config.norm_eps, weight=w.norm)
        og = ttnn.sigmoid(ttnn.reshape(p.output_gate, (B, c.num_heads, c.head_v_dim)))
        o = ttnn.multiply(o, og)
        o = ttnn.reshape(o, (1, B, c.num_heads * c.head_v_dim))
        out = ttnn.linear(
            o,
            w.output_projection,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=kda.output_projection_compute_config,
            dtype=ttnn.bfloat16,
        )
        out = ttnn.reshape(out, (1, 1, B, out.shape[-1]))
        return self.ccl.all_reduce(out)


def _L1():
    return ttnn.L1_MEMORY_CONFIG
