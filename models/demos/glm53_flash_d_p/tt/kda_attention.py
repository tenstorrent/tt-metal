# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 KDA attention (kda_dense / kda_moe attention step) on the 2x2 mesh, from DeepSeek's ttKDA.

x [1, 1, S, 4096] replicated -> ttnn.mesh_partition (dim -2, cluster_axis 0): mesh row r holds rows [r S/2, (r+1) S/2)
(SP = 2 on axis 0) -> ttKDA (TP = 2 on axis 1: column c holds heads 32c..32c+31; fused input projection, q/k/v conv
with the SP halo, bounded decay -5 sigmoid(exp(A_log) (f_b f_a x + dt_bias)), fp32 beta, chunked recurrence with the
grouped SP scan, gated RMSNorm, row-parallel o_proj + reduce-scatter on axis 1) -> all_gather (dim -1, axis 1) +
all_gather (dim -2, axis 0) -> [1, 1, S, 4096] replicated bf16.

Every matmul-like program is HiFi4 (KDA's defaults put the recurrence affine prefix and scan at HiFi2). The grouped-scan
group count comes from the device grid. actual_start is a device uint32 scalar sliced from a table of every
64-aligned start up to max_seq built at load. For a chunk-aligned start (start a multiple of S), ttKDA's chronology
puts SP rank 0 first (first_rank = (start / (S/2)) % 2 = 0, no split): the contiguous halves above.
The carries live in address-stable buffers (kimi_k3/kda_state.py pattern); a chunk at start 0 reads a zeroed state.
Precision departures from ttKDA: the decay gate in fp32, and prepare_chunk_recurrence's k_dec_t recomputed without
its TF32 cancellation (_PreciseDecayRecurrence; GLM_KDA_DECAY=kernel keeps the kernel's own term for comparison).
"""

from __future__ import annotations

import os
from dataclasses import replace

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tt.kda.config import (
    KDA_RECURRENT_STATE_DTYPE,
    KDAProgramConfig,
    KDARecurrenceProgramConfig,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import KDARecurrence
from models.demos.deepseek_v3_d_p.tt.kda.weights import load_kda_weights
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

SP_AXIS, TP_AXIS = 0, 1
PRECISE_DECAY = os.environ.get("GLM_KDA_DECAY", "precise") == "precise"  # "kernel": prepare's own k_dec_t
START_ALIGN = 64
_KDA_NAMES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "A_log",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "dt_bias",
    "b_proj.weight",
    "g_a_proj.weight",
    "g_b_proj.weight",
    "o_norm.weight",
    "o_proj.weight",
)


def kda_config(cfg) -> KDAConfig:
    return KDAConfig(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.linear_num_heads,
        head_k_dim=cfg.linear_head_dim,
        head_v_dim=cfg.linear_head_dim,
        conv_kernel_size=cfg.linear_conv_kernel,
        norm_eps=cfg.rms_norm_eps,
        use_full_rank_gate=False,
        gate_lower_bound=cfg.linear_lower_bound,
    )


def _replicated(mesh, t: torch.Tensor, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


class _PreciseDecayRecurrence(KDARecurrence):
    """KDARecurrence with prepare_chunk_recurrence's k_dec_t recomputed without cancellation.

    The kernel forms k_dec_t = k * exp(G_last - G_j) as exp(a - G_j) * exp(G_last - a) with G = cumsum(g) and
    a = G_last / 2, subtracting on the FPU, which reads fp32 operands at TF32 precision (10-bit mantissa). Near
    the -5 gate bound |G| reaches ~150 per 32-token chunk (TF32 step 0.125), so the factor is off by up to e^0.1 and
    biased low (GLM layer 0: the last token's factor, exactly 1, reads 0.967), and the fast-decay rows of the carried
    state come out ~5% small. Here: G_last - G_j = sum_{i > j} g_i as a bf16 matmul with a constant 0/1 mask (exact
    products, fp32 accumulation), exp and the l2-normalized k in fp32 (SFPU), then the kernel's layout [H, N, K, 32].
    """

    def __init__(self, device, program_config, *, heads, key_dim, compute_config, **kwargs):
        super().__init__(device, program_config, heads=heads, key_dim=key_dim, **kwargs)
        c = ttnn.TILE_SIZE
        self._ckc = compute_config
        i = torch.arange(c)
        self._suffix = _replicated(device, (i.view(c, 1) > i.view(1, c)).float(), ttnn.bfloat16)  # M[i, j] = i > j
        cols = torch.arange(heads * key_dim) // key_dim
        heads_of = (cols.view(-1, 1) == torch.arange(heads).view(1, -1)).float()  # [H*K, H]
        self._head_sum = _replicated(device, heads_of, ttnn.bfloat16)
        self._head_bcast = _replicated(device, heads_of.t(), ttnn.bfloat16)

    def _k_dec_t(self, k: ttnn.Tensor, gate: ttnn.Tensor) -> ttnn.Tensor:
        geo = self._geometry
        n, c, h, kd = geo.num_chunks, geo.chunk_size, geo.heads, geo.key_dim
        mm = dict(compute_kernel_config=self._ckc, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dram = ttnn.DRAM_MEMORY_CONFIG
        # l2 norm over each head's K channels (eps 1e-6 inside the sqrt, as the kernel and FLA)
        k32 = ttnn.typecast(k, ttnn.float32, memory_config=dram)
        sq = ttnn.multiply(k32, k32, memory_config=dram)
        ss = ttnn.matmul(sq, self._head_sum, **mm)  # [1, T, H]
        ttnn.deallocate(sq)
        inv = ttnn.rsqrt(ttnn.add(ss, 1e-6, memory_config=dram), memory_config=dram)
        ttnn.deallocate(ss)
        invb = ttnn.matmul(inv, self._head_bcast, **mm)  # [1, T, H*K]
        ttnn.deallocate(inv)
        khat = ttnn.multiply(k32, invb, memory_config=dram)
        ttnn.deallocate(k32)
        ttnn.deallocate(invb)
        kt = ttnn.transpose(ttnn.reshape(khat, (n, c, h * kd)), -2, -1, memory_config=dram)  # [N, H*K, C]
        ttnn.deallocate(khat)
        gt = ttnn.transpose(ttnn.reshape(gate, (n, c, h * kd)), -2, -1, memory_config=dram)
        suf = ttnn.matmul(gt, self._suffix, **mm)  # [N, H*K, C]: sum_{i > j} g_i
        ttnn.deallocate(gt)
        dec = ttnn.exp(suf, memory_config=dram)
        ttnn.deallocate(suf)
        out = ttnn.multiply(kt, dec, memory_config=dram)
        ttnn.deallocate(kt)
        ttnn.deallocate(dec)
        res = ttnn.permute(ttnn.reshape(out, (n, h, kd, c)), (1, 0, 2, 3), memory_config=dram)  # [H, N, K, C]
        ttnn.deallocate(out)
        return res

    def _prepare(self, *, k, gate, **kwargs):
        prepared, state, geometry = super()._prepare(k=k, gate=gate, **kwargs)
        ttnn.deallocate(prepared.k_dec_t)
        return replace(prepared, k_dec_t=self._k_dec_t(k, gate)), state, geometry


class _GlmKDA(ttKDA):
    """ttKDA with the precise-decay recurrence, and the bounded decay computed in fp32 (f_b output, dt_bias,
    exp(A_log), sigmoid, x -5) and rounded to bf16 once for prepare_chunk_recurrence (ttKDA keeps exp(A_log) and
    dt_bias in bf16)."""

    def __init__(self, *args, decay_scale32, decay_bias32, program_config, **kwargs):
        super().__init__(*args, program_config=program_config, **kwargs)
        self.decay_scale32, self.decay_bias32 = decay_scale32, decay_bias32
        if PRECISE_DECAY:
            self.recurrence = _PreciseDecayRecurrence(
                self.device,
                program_config.recurrence,
                sequence_parallel_axis=self.sequence_parallel_axis,
                local_rows=self.active_seq_len_local,
                heads=self.config.num_heads,
                key_dim=self.config.head_k_dim,
                value_dim=self.config.head_v_dim,
                compute_config=self.kda_compute_config,
            )

    def _compute_gates(self, *, beta, decay_rank):
        beta32 = ttnn.sigmoid(
            ttnn.typecast(beta, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        raw = ttnn.linear(
            decay_rank,
            self.weights.decay_output_projection,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        z = ttnn.add(raw, self.decay_bias32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(raw)
        zs = ttnn.multiply(z, self.decay_scale32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(z)
        sg = ttnn.sigmoid(zs, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(zs)
        gate = ttnn.multiply(
            sg, self.config.gate_lower_bound, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(sg)
        return gate, beta32


def _scan_groups(mesh, local_chunks: int, local_heads: int) -> int:
    """Largest group count that divides the local chunk count and fits one summary owner per (head, group)."""
    grid = mesh.compute_with_storage_grid_size()
    cap = grid.x * grid.y
    return max(g for g in range(1, local_chunks + 1) if local_chunks % g == 0 and local_heads * g <= cap)


class TtKdaAttention(LightweightModule):
    def __init__(self, mesh, state_dict: dict, cfg, max_seq: int, layer: int = 0):
        self.mesh = mesh
        self.layer = layer
        self.kcfg = kda_config(cfg)
        self.hidden = cfg.hidden_size
        self.sp = tuple(mesh.shape)[SP_AXIS]
        self.tp = tuple(mesh.shape)[TP_AXIS]
        self.local_heads = self.kcfg.num_heads // self.tp
        self.conv_width = 3 * self.local_heads * self.kcfg.head_k_dim  # [q_c | k_c | v_c] per TP rank
        self.tt_ccl = get_tt_ccl(mesh)
        self.weights = load_kda_weights(
            mesh,
            self.kcfg,
            state_dict,
            None,
            cache_name_prefix=f"glm53.layer_{layer}.kda",
            tensor_parallel_axis=TP_AXIS,
        )
        h, d = self.kcfg.num_heads, self.kcfg.head_k_dim
        scale = state_dict["A_log"].float().reshape(h, 1).exp().expand(h, d).reshape(1, 1, h * d)
        bias = state_dict["dt_bias"].float().reshape(1, 1, h * d)
        tp_mapper = ttnn.ShardTensor2dMesh(mesh, dims=(None, -1), mesh_shape=tuple(mesh.shape))
        self.decay_scale32, self.decay_bias32 = (
            ttnn.from_torch(
                t.contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=tp_mapper,
            )
            for t in (scale, bias)
        )
        self._layers: dict[int, ttKDA] = {}
        n = max_seq // START_ALIGN + 1
        starts = torch.arange(n, dtype=torch.int64).mul(START_ALIGN).reshape(n, 1)
        self.starts = ttnn.from_torch(
            starts,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.state = self._zeros()
        self.zero_state = self._zeros()

    def _zeros(self) -> KdaState:
        k = self.kcfg
        return KdaState(
            recurrent=ttnn.zeros(
                (1, self.local_heads, k.head_k_dim, k.head_v_dim),
                dtype=KDA_RECURRENT_STATE_DTYPE,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            convolution=ttnn.zeros(
                (1, k.conv_kernel_size - 1, self.conv_width),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )

    def _program_config(self, local_rows: int) -> KDAProgramConfig:
        chunks = local_rows // ttnn.TILE_SIZE
        groups = _scan_groups(self.mesh, chunks, self.local_heads)
        return KDAProgramConfig(
            recurrence=KDARecurrenceProgramConfig(
                local_scan_strategy="grouped",
                summary_group_chunks=chunks // groups,
                affine_prefix_math_fidelity=ttnn.MathFidelity.HiFi4,
                scan_math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
            tp_ccl_topology=ttnn.Topology.Linear,
            gated_rms_output_dtype=ttnn.float32,
            output_projection_math_fidelity=ttnn.MathFidelity.HiFi4,
        )

    def _kda(self, s: int) -> ttKDA:
        """One ttKDA per chunk length (its graph is fixed at construction); weights shared."""
        if s not in self._layers:
            self._layers[s] = _GlmKDA(
                self.mesh,
                self.kcfg,
                weights=self.weights,
                layer_idx=self.layer,
                tt_ccl=self.tt_ccl,
                sp_axis=SP_AXIS,
                tp_axis=TP_AXIS,
                program_config=self._program_config(s // self.sp),
                active_seq_len=s,
                decay_scale32=self.decay_scale32,
                decay_bias32=self.decay_bias32,
            )
        return self._layers[s]

    def __call__(self, x: ttnn.Tensor, start: int) -> ttnn.Tensor:
        """x [1, 1, S, H] replicated bf16, start the chunk's absolute position (a multiple of S) ->
        attn_out [1, 1, S, H] replicated bf16. Advances the carried state."""
        s = x.shape[-2]
        assert start % s == 0 and start % START_ALIGN == 0, f"chunk start {start} must be a multiple of S={s}"
        kda = self._kda(s)
        xs = ttnn.mesh_partition(x, dim=-2, cluster_axis=SP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = ttnn.reshape(xs, (1, s // self.sp, self.hidden))
        i = start // START_ALIGN
        actual_start = ttnn.slice(self.starts, (i, 0), (i + 1, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        state = self.zero_state if start == 0 else self.state
        out, new = kda.forward(h, state, actual_start=actual_start)
        ttnn.deallocate(actual_start)
        ttnn.deallocate(xs)
        ttnn.copy(new.recurrent, self.state.recurrent)
        ttnn.copy(new.convolution, self.state.convolution)
        ttnn.deallocate(new.recurrent)
        ttnn.deallocate(new.convolution)
        # out [1, S/2, H/2]: rows of this SP rank, hidden reduce-scattered over TP.
        o4 = ttnn.reshape(out, (1, 1, s // self.sp, self.hidden // self.tp))
        g1 = ttnn.all_gather(o4, dim=-1, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        g2 = ttnn.all_gather(g1, dim=-2, cluster_axis=SP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(g1)
        if g2.dtype != ttnn.bfloat16:
            y = ttnn.typecast(g2, ttnn.bfloat16)
            ttnn.deallocate(g2)
            return y
        return g2

    # ---- state at the harness boundary (prefix load / read-back; never per chunk inside the forward)
    def _conv_to_device_order(self, conv: torch.Tensor) -> torch.Tensor:
        """Reference [3, q | k | v] (each H*D) -> [3, tp * (q_c | k_c | v_c)]."""
        rows = conv.shape[0]
        return conv.reshape(rows, 3, self.tp, -1).transpose(1, 2).reshape(rows, -1)

    def _conv_to_ref_order(self, conv: torch.Tensor) -> torch.Tensor:
        rows = conv.shape[0]
        return conv.reshape(rows, self.tp, 3, -1).transpose(1, 2).reshape(rows, -1)

    def load_state(self, tensors: dict) -> None:
        """Host prefix state (reference layout) -> the carried device state (copied into the stable buffers)."""
        mapper = lambda d: ttnn.ShardTensor2dMesh(self.mesh, dims=(None, d), mesh_shape=tuple(self.mesh.shape))  # noqa
        rec = tensors["kda_recurrent"].float().reshape(1, self.kcfg.num_heads, self.kcfg.head_k_dim, -1)
        conv = self._conv_to_device_order(tensors["kda_conv"].float()).unsqueeze(0)
        rd = ttnn.from_torch(
            rec.contiguous(),
            dtype=KDA_RECURRENT_STATE_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper(1),
        )
        cd = ttnn.from_torch(
            conv.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper(2),
        )
        ttnn.copy(rd, self.state.recurrent)
        ttnn.copy(cd, self.state.convolution)
        ttnn.deallocate(rd)
        ttnn.deallocate(cd)

    def state_torch(self) -> dict:
        """The carried state in the reference layout (mesh row 0's copy; rows are replicated)."""
        cols = tuple(self.mesh.shape)[1]
        rec = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(self.state.recurrent)[:cols]]
        conv = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(self.state.convolution)[:cols]]
        rec = torch.cat(rec, dim=1).reshape(self.kcfg.num_heads, self.kcfg.head_k_dim, -1).float()
        conv = self._conv_to_ref_order(torch.cat(conv, dim=-1).reshape(self.kcfg.conv_kernel_size - 1, -1).float())
        return {"kda_recurrent": rec, "kda_conv": conv}


def build_kda_attention(mesh, loader, cfg, layer: int, max_seq: int) -> TtKdaAttention:
    sd = {n: loader.layer(layer, f"self_attn.{n}") for n in _KDA_NAMES}
    return TtKdaAttention(mesh, sd, cfg, max_seq, layer)
