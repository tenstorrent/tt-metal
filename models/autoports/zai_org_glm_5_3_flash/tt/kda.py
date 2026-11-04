# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention (KDA) block of GLM-5.3-Flash (modeling_glm5_next.py:617-769).

    u     = [x Wq | x Wk | x Wv]                               (bfloat8_b fused [4096, 24576])
    q,k,v = SiLU(causal depthwise conv_4(u))                   (3-token pre-conv history per user)
    g     = -5 sigmoid(exp(A_log[h]) (f_b(f_a(x)) + dt_bias))  [T, 64, 128]  per head AND key channel
    beta  = sigmoid(x Wb)                                      [T, 64]
    S    <- diag(exp g_t) S;  S += k̂_t (beta_t (v_t - S^T k̂_t))^T;  o_t = S^T q̂_t / sqrt(128)   (q̂, k̂ l2-normalised)
    out   = o_proj( RMSNorm_128(o) * w * sigmoid(g_b(g_a(x))) )

Prefill (one user, any logical length, physical chunk C = multiple of 32): `ttnn.experimental.kda`
qkv_causal_conv1d_silu + prepare_chunk_recurrence, then an exact-fp32-carry scan over the 32-token chunks
(`_chunk_recurrence`), with `actual_end` turning padded rows into identity steps so padding never reaches the state or the
conv history. A 32-row chunk with NO valid row is skipped by prepare_chunk_recurrence (its terms stay unwritten); the
scan keeps the state across such a chunk (`keep_if_dead`), so a physical chunk may carry any amount of padding. KDA has no positions, so on one device every chunk uses actual_start = 0 and the carried state (S, conv
history) provides continuity across chunks and requests.

Decode (B users, one token each): the token recurrence with per-key-channel decay as fp32 batched matmuls
(`ttnn.transformer.fused_recurrent_gated_delta_rule` has a per-head scalar decay and is not exact for KDA).

Per-user state (`KDAStateCache`): per slot, recurrent S [1, 64, 128, 128] fp32 TILE and the last 3 pre-conv rows
[1, 3, 24576] bf16 ROW_MAJOR — persistent device buffers that are only ever updated with in-place `ttnn.copy`, so their
addresses stay valid for a captured decode trace while prefills of other requests run (decode stacks the slots inside
the trace and copies the results back).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.precision import KDA_STATE_DTYPE, hifi2_fp32, hifi4_fp32

try:  # prefill-only (chunked selections); absent from tenstorrent/tt-metal: the decode stack never reaches it
    from models.demos.deepseek_v3_d_p.tt.kda.chronological_selections import ChronologicalSelections
except ModuleNotFoundError:  # pragma: no cover
    ChronologicalSelections = None

DRAM = ttnn.DRAM_MEMORY_CONFIG
CONV_K = 4
CHUNK = 32  # KDA chunk of prepare_chunk_recurrence


def chunk_live_masks(rows: ttnn.Tensor, valid_end_f: ttnn.Tensor, n_chunks: int) -> list:
    """Per 32-row recurrence chunk n: [1, 1, 1] fp32 1.0 when the chunk holds at least one valid row (n * CHUNK <
    valid_end), else 0.0. Device values (trace-safe: the valid end is a device tensor written per prefill chunk)."""
    return [
        ttnn.reshape(ttnn.lt(ttnn.slice(rows, (0, 0, n * CHUNK, 0), (1, 1, n * CHUNK + 1, 1)), valid_end_f), (1, 1, 1))
        for n in range(n_chunks)
    ]


def keep_if_dead(live: ttnn.Tensor, new_state: ttnn.Tensor, state: ttnn.Tensor) -> ttnn.Tensor:
    """The scan's state after chunk n: `new_state` for a chunk with valid rows, else the unchanged `state`.
    prepare_chunk_recurrence skips every chunk past actual_end (its reader, compute and writer kernels `continue` for
    chunk >= valid_chunks), so such a chunk's t_inv / v_beta / kd / intra / q_decay / k_dec_t are never written and hold
    stale DRAM; its decay is masked to 1, but the update k_dec^T U would still add those terms to the state (measured:
    a wrong KDA state after a prefill chunk with >= 32 padded rows, doc/ttnn/04-multichip-decoder §3c). ttnn.where is a
    lane select, so stale NaN / Inf in `new_state` cannot leak into the kept state."""
    return ttnn.where(ttnn.add(ttnn.zeros_like(new_state), live), new_state, state)


@dataclass
class KDAStateCache:
    """Persistent per-user KDA state for `slots` users (one decoder layer)."""

    recurrent: list[ttnn.Tensor]  # per slot [1, H, K, V] fp32 TILE
    conv: list[ttnn.Tensor]  # per slot [1, 3, 3*H*K] bf16 ROW_MAJOR (pre-conv q|k|v rows t-3, t-2, t-1)

    @property
    def slots(self) -> int:
        return len(self.recurrent)


class KDA:
    def __init__(
        self,
        mesh_device,
        *,
        w_qkv,
        w_lowrank,
        w_fb,
        w_gb,
        w_o,
        taps,
        o_norm,
        dt_bias: torch.Tensor,
        a_log: torch.Tensor,
        num_heads: int,
        head_dim: int,
        hidden: int,
        lower_bound: float,
        eps: float,
    ):
        """Weights in cache layout ([in, out]); w_lowrank = [f_a | g_a | b] [4096, 128 + 128 + 64] bf16;
        taps [4, 3*H*K] bf16 (row j multiplies input row t-3+j); dt_bias [H*K] and a_log [H] fp32 host tensors."""
        self.device = mesh_device
        self.h = num_heads
        self.d = head_dim
        self.width = num_heads * head_dim  # 8192
        self.hidden = hidden
        self.lower_bound = lower_bound
        self.eps = eps
        self.w_qkv, self.w_lowrank, self.w_fb, self.w_gb, self.w_o = w_qkv, w_lowrank, w_fb, w_gb, w_o
        self.o_norm = o_norm  # [128] bf16
        self.o_norm_row = ttnn.reshape(ttnn.typecast(o_norm, ttnn.float32), (1, 1, 1, head_dim))
        # per-channel decay rate exp(A_log[h]) and dt_bias, fp32 [1, 1, 1, H*K]
        rate = torch.exp(a_log.float()).repeat_interleave(head_dim)
        self.decay_rate = ttnn.from_torch(
            rate.reshape(1, 1, 1, -1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
        )
        self.dt_bias = ttnn.from_torch(
            dt_bias.float().reshape(1, 1, 1, -1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
        )
        # conv taps: prefill op wants 4 separate [1,1,3*H*K] bf16 tiles; decode uses fp32 rows [1,1,1,3*H*K]
        self.taps_prefill = []
        self.taps_decode = []
        conv_width = 3 * self.width
        for j in range(CONV_K):
            row = ttnn.slice(taps, (j, 0), (j + 1, conv_width))
            self.taps_prefill.append(ttnn.reshape(row, (1, 1, conv_width)))
            self.taps_decode.append(ttnn.reshape(ttnn.typecast(row, ttnn.float32), (1, 1, 1, conv_width)))
        self.ckc_proj = hifi2_fp32(mesh_device)
        self.ckc_fp32 = hifi4_fp32(mesh_device)
        self.ckc_kda = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.conv_program = ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=768)
        self.row_ids = ttnn.from_torch(
            torch.arange(2048, dtype=torch.float32).reshape(1, 1, -1, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
        )
        self.zero_u32 = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
        )

    # ---------------------------------------------------------------------------------------------------------------
    def allocate_state(self, slots: int) -> KDAStateCache:
        return KDAStateCache(
            recurrent=[
                ttnn.zeros(
                    (1, self.h, self.d, self.d),
                    dtype=KDA_STATE_DTYPE,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=DRAM,
                )
                for _ in range(slots)
            ],
            conv=[
                ttnn.zeros(
                    (1, CONV_K - 1, 3 * self.width),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=DRAM,
                )
                for _ in range(slots)
            ],
        )

    def _mm(self, a, b):
        return ttnn.matmul(a, b, compute_kernel_config=self.ckc_fp32, dtype=ttnn.float32)

    def _chunk_recurrence(self, q, k, v, g, beta, s0, actual_end, g32, valid_end_f):
        """Chunked delta rule over one physical chunk: `prepare_chunk_recurrence` builds the per-32-token chunk terms
        (l2-normalised q/k with the D^-1/2 scale, cumulative per-channel decays, the triangular inverse; rows past
        `actual_end` become identity steps), then the chunks are scanned in order:
            U_n = T_n^-1 (v_beta_n - kd_n S_n);  Y_n = q_decay_n S_n + intra_n U_n;  S_n+1 = decay_n * S_n + k_dec_n^T U_n
        The per-chunk state decay exp(sum_chunk g) is computed here from the fp32 gate `g32` (padded rows masked with
        `valid_end_f`): the op's `final_decay` output is ~2.5e-4 too small for slow channels, a bias that multiplies the
        state every chunk and compounds with length. The carried state only goes through exact fp32 eltwise ops.
        q/k/v/g [1, C, H*D] bf16, beta [1, C, H] fp32, s0 [1, H, D, D] fp32, g32 [1, 1, C, H*D] fp32,
        valid_end_f [1, 1, 1, 1] fp32 -> (Y [H, C, D] fp32, S [1, H, D, D] fp32)."""
        c, h, d = q.shape[-2], self.h, self.d
        n_chunks = c // CHUNK
        beta_by_chunk = ttnn.reshape(ttnn.permute(beta, (0, 2, 1)), (h, n_chunks, CHUNK, 1))
        v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv = ttnn.experimental.kda.prepare_chunk_recurrence(
            q,
            k,
            v,
            g,
            beta_by_chunk,
            h,
            memory_config=DRAM,
            compute_kernel_config=self.ckc_kda,
            output_bf16_mask=0,
            actual_start=self.zero_u32,
            actual_end=actual_end,
            sequence_parallel_axis=0,
        )
        rows = ttnn.slice(self.row_ids, (0, 0, 0, 0), (1, 1, c, 1))
        g_valid = ttnn.multiply(g32, ttnn.lt(rows, valid_end_f))  # padded rows are identity steps
        g_sum = ttnn.sum(ttnn.reshape(g_valid, (n_chunks, CHUNK, h * d)), dim=1, keepdim=True)  # [N, 1, HD]
        chunk_decay = ttnn.exp(
            ttnn.reshape(ttnn.permute(ttnn.reshape(g_sum, (n_chunks, h, d)), (1, 0, 2)), (h, n_chunks, d, 1))
        )
        mm = self._mm
        live = chunk_live_masks(rows, valid_end_f, n_chunks)
        state = ttnn.reshape(s0, (h, d, d))
        outs = []
        for n in range(n_chunks):
            sl = lambda t, rows, cols: ttnn.reshape(
                ttnn.slice(t, (0, n, 0, 0), (h, n + 1, rows, cols)), (h, rows, cols)
            )
            u = mm(sl(t_inv, CHUNK, CHUNK), ttnn.subtract(sl(v_beta, CHUNK, d), mm(sl(kd, CHUNK, d), state)))
            outs.append(ttnn.add(mm(sl(q_decay, CHUNK, d), state), mm(sl(intra, CHUNK, CHUNK), u)))
            new_state = ttnn.add(ttnn.multiply(state, sl(chunk_decay, d, 1)), mm(sl(k_dec_t, d, CHUNK), u))
            state = keep_if_dead(live[n], new_state, state)
        y = outs[0] if n_chunks == 1 else ttnn.concat(outs, dim=1)
        return y, ttnn.reshape(state, (1, h, d, d))

    def _lowrank(self, x):
        lr = ttnn.linear(
            x, self.w_lowrank, compute_kernel_config=self.ckc_fp32, memory_config=DRAM, dtype=ttnn.bfloat16
        )
        t = x.shape[-2]
        f_a = ttnn.slice(lr, (0, 0, 0, 0), (1, 1, t, self.d))
        g_a = ttnn.slice(lr, (0, 0, 0, self.d), (1, 1, t, 2 * self.d))
        b = ttnn.slice(lr, (0, 0, 0, 2 * self.d), (1, 1, t, 2 * self.d + self.h))
        return f_a, g_a, b

    def _decay(self, f_a):
        """log-decay g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a) + dt_bias)) [1,1,T,H*K] fp32."""
        f = ttnn.linear(f_a, self.w_fb, compute_kernel_config=self.ckc_fp32, memory_config=DRAM, dtype=ttnn.float32)
        f = ttnn.add(f, self.dt_bias)
        z = ttnn.multiply(f, self.decay_rate)
        return ttnn.multiply(ttnn.sigmoid(z), self.lower_bound)

    # ---------------------------------------------------------------------------------------------------------------
    def prefill(
        self, x: ttnn.Tensor, state: KDAStateCache, slot: int, actual_end: ttnn.Tensor, valid_end_f: ttnn.Tensor
    ) -> ttnn.Tensor:
        """One user's chunk. x [1,1,C,4096] bf16 (C multiple of 32; rows >= actual_end are padding, their outputs are
        unspecified). Reads and replaces the user's state in `slot`. Returns [1,1,C,4096] bf16."""
        c = x.shape[-2]
        conv_width = 3 * self.width
        qkv = ttnn.linear(x, self.w_qkv, compute_kernel_config=self.ckc_proj, memory_config=DRAM, dtype=ttnn.bfloat16)
        f_a, g_a, b = self._lowrank(x)
        gate_out = ttnn.linear(
            g_a, self.w_gb, compute_kernel_config=self.ckc_fp32, memory_config=DRAM, dtype=ttnn.bfloat16
        )
        g32 = self._decay(f_a)
        g = ttnn.typecast(g32, ttnn.bfloat16)  # prepare_chunk_recurrence takes a BF16 gate
        beta = ttnn.sigmoid(ttnn.typecast(b, ttnn.float32))

        carry, s0 = self.read_slot(state, slot)
        selections = ChronologicalSelections(
            ttnn.experimental.kda.chronological_selections(
                self.zero_u32, 0, c, self.h, self.d, self.d, actual_end=actual_end
            )
        )
        qkv_rm = ttnn.to_layout(ttnn.reshape(qkv, (1, c, conv_width)), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(qkv)
        new_carry = selections.select_local_final_history(qkv_rm, carry)
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            qkv_rm,
            carry,
            *self.taps_prefill,
            self.width,
            self.width,
            self.width,
            program_config=self.conv_program,
            actual_start=self.zero_u32,
            sequence_parallel_axis=0,
            predecessor_carry=carry,
            memory_config=DRAM,
        )
        ttnn.deallocate(qkv_rm)
        y, final_state = self._chunk_recurrence(
            q,
            k,
            v,
            ttnn.reshape(g, (1, c, self.width)),
            ttnn.reshape(beta, (1, c, self.h)),
            s0,
            actual_end,
            g32,
            valid_end_f,
        )
        o = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            y,
            ttnn.reshape(gate_out, (1, c, self.width)),
            self.o_norm,
            self.h,
            epsilon=self.eps,
            memory_config=DRAM,
            compute_kernel_config=self.ckc_kda,
            output_dtype=ttnn.bfloat16,
        )
        out = ttnn.linear(
            ttnn.reshape(o, (1, 1, c, self.width)),
            self.w_o,
            compute_kernel_config=self.ckc_proj,
            memory_config=DRAM,
            dtype=ttnn.bfloat16,
        )
        self.write_slot(state, slot, new_carry, final_state)
        return out

    def read_slot(self, state: KDAStateCache, slot: int):
        """(conv history [1, 3, 3HK] bf16 ROW_MAJOR, recurrent state [1, H, K, V] fp32) of one user slot (the
        persistent buffers themselves; the prefill ops only read them)."""
        return state.conv[slot], state.recurrent[slot]

    def write_slot(self, state: KDAStateCache, slot: int, carry: ttnn.Tensor, recurrent: ttnn.Tensor):
        """Store one user's carries in place (the persistent buffers keep their addresses)."""
        ttnn.copy(carry, state.conv[slot])
        ttnn.copy(recurrent, state.recurrent[slot])

    # ---------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _keep_inactive(active: ttnn.Tensor, new: ttnn.Tensor, old: ttnn.Tensor) -> ttnn.Tensor:
        """Per user slot (dim 0): `new` where active, `old` where inactive. Exact selection, so an inactive slot's carry
        is written back bit-identical whatever its (don't-care) input row held: the condition is built from `old`, the
        slot's own stored carry (zero-initialised, finite), never from `new`, which may be NaN/Inf for an inactive row.
        active: [B,1,1,1] fp32 1.0 / 0.0."""
        mask = ttnn.reshape(active, (active.shape[0],) + (1,) * (len(new.shape) - 1))
        mask = ttnn.add(ttnn.multiply(old, 0.0), ttnn.typecast(mask, old.dtype))  # 0/1 broadcast to the carry's shape
        return ttnn.where(mask, new, old)

    def decode(self, x: ttnn.Tensor, state: KDAStateCache, active: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """x [1,1,B,4096] bf16, one token per user slot 0..B-1 (B == state.slots). Updates state in place.
        active: optional [B,1,1,1] fp32 device mask (1 = decode this slot, 0 = inactive): inactive slots keep their
        recurrent state and conv history bit-identical (their output rows are don't-care)."""
        bsz = x.shape[-2]
        h, d, w = self.h, self.d, self.width
        conv_width = 3 * w
        qkv = ttnn.linear(x, self.w_qkv, compute_kernel_config=self.ckc_proj, memory_config=DRAM, dtype=ttnn.bfloat16)
        f_a, g_a, b = self._lowrank(x)
        gate_out = ttnn.linear(
            g_a, self.w_gb, compute_kernel_config=self.ckc_fp32, memory_config=DRAM, dtype=ttnn.float32
        )
        g = self._decay(f_a)  # [1,1,B,H*K] fp32
        beta = ttnn.sigmoid(ttnn.typecast(b, ttnn.float32))  # [1,1,B,H]

        # causal conv over [hist(t-3), hist(t-2), hist(t-1), u_t] in fp32, then SiLU
        qkv_rows = ttnn.reshape(qkv, (bsz, 1, conv_width))  # [B,1,3W] bf16 TILE
        conv_rm = state.conv[0] if bsz == 1 else ttnn.concat(state.conv, dim=0)  # [B,3,3W] ROW_MAJOR
        hist = ttnn.to_layout(conv_rm, ttnn.TILE_LAYOUT)
        acc = ttnn.multiply(
            ttnn.typecast(qkv_rows, ttnn.float32), ttnn.reshape(self.taps_decode[3], (1, 1, conv_width))
        )
        hist_rows = []
        for j in range(CONV_K - 1):
            hj = ttnn.slice(hist, (0, j, 0), (bsz, j + 1, conv_width))
            hist_rows.append(hj)
            acc = ttnn.addcmul(
                acc, ttnn.typecast(hj, ttnn.float32), ttnn.reshape(self.taps_decode[j], (1, 1, conv_width))
            )
        conv = ttnn.silu(acc)  # [B,1,3W] fp32
        new_hist = ttnn.concat([hist_rows[1], hist_rows[2], qkv_rows], dim=1)  # [B,3,3W] bf16
        if active is not None:
            new_hist = self._keep_inactive(active, new_hist, hist)
        new_hist_rm = ttnn.to_layout(new_hist, ttnn.ROW_MAJOR_LAYOUT)
        for u in range(bsz):
            ttnn.copy(
                new_hist_rm if bsz == 1 else ttnn.slice(new_hist_rm, (u, 0, 0), (u + 1, CONV_K - 1, conv_width)),
                state.conv[u],
            )

        def heads(t):  # [B,1,W] -> [B*H, 1, D]
            return ttnn.reshape(t, (bsz * h, 1, d))

        q = heads(ttnn.slice(conv, (0, 0, 0), (bsz, 1, w)))
        k = heads(ttnn.slice(conv, (0, 0, w), (bsz, 1, 2 * w)))
        v = heads(ttnn.slice(conv, (0, 0, 2 * w), (bsz, 1, 3 * w)))
        # l2norm over D (x / sqrt(sum x^2 + 1e-6)), q additionally scaled by D^-0.5
        q = ttnn.multiply(q, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(q, q), dim=-1, keepdim=True), 1e-6)))
        q = ttnn.multiply(q, d**-0.5)
        k = ttnn.multiply(k, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(k, k), dim=-1, keepdim=True), 1e-6)))

        s_old = state.recurrent[0] if bsz == 1 else ttnn.concat(state.recurrent, dim=0)  # [B,H,D,D] fp32
        s = ttnn.reshape(s_old, (bsz * h, d, d))
        decay = ttnn.exp(ttnn.reshape(g, (bsz * h, d, 1)))  # per key channel (row of S)
        s = ttnn.multiply(s, decay)
        kv_mem = ttnn.matmul(k, s, compute_kernel_config=self.ckc_fp32, dtype=ttnn.float32)  # [BH,1,D]
        beta_h = ttnn.reshape(beta, (bsz * h, 1, 1))
        delta = ttnn.multiply(ttnn.subtract(v, kv_mem), beta_h)
        k_col = ttnn.transpose(k, -2, -1)  # [BH, D, 1]
        s = ttnn.add(s, ttnn.multiply(k_col, delta))  # rank-1 update k delta^T via broadcast
        o = ttnn.matmul(q, s, compute_kernel_config=self.ckc_fp32, dtype=ttnn.float32)  # [BH,1,D]
        s = ttnn.reshape(s, (bsz, h, d, d))
        if active is not None:
            s = self._keep_inactive(active, s, s_old)
        for u in range(bsz):
            ttnn.copy(s if bsz == 1 else ttnn.slice(s, (u, 0, 0, 0), (u + 1, h, d, d)), state.recurrent[u])

        # gated RMSNorm over D in fp32, weight o_norm, gate sigmoid(g_b(g_a(x)))
        o = ttnn.reshape(o, (1, bsz, h, d))
        var = ttnn.mean(ttnn.multiply(o, o), dim=-1, keepdim=True)
        o = ttnn.multiply(o, ttnn.rsqrt(ttnn.add(var, self.eps)))
        o = ttnn.multiply(o, self.o_norm_row)
        gate = ttnn.sigmoid(ttnn.reshape(gate_out, (1, bsz, h, d)))
        o = ttnn.multiply(o, gate)
        o = ttnn.typecast(ttnn.reshape(o, (1, 1, bsz, w)), ttnn.bfloat16)
        return ttnn.linear(o, self.w_o, compute_kernel_config=self.ckc_proj, memory_config=DRAM, dtype=ttnn.bfloat16)
