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
An optional actual_end (exclusive valid end, 32-aligned; the serving contract pads the last chunk) is sliced from a
second load-time table (every 32-aligned end up to max_seq) and passed to ttKDA, so the carries stop at the valid end.
bind_state points the module at another address-stable state (one per serving slot, tt/runners).
The gate, beta and bounded-decay paths are ttKDA's fused ones (PR #59918: the decay projection reads its rank columns
in place with the per-head scale folded into its weights and the sigmoid in its pack; chunk preparation applies -5 and
beta's sigmoid; the gated norm reads its gate in place). GLM's own fp32 gate path and recomputed k_dec_t
(_PreciseDecayRecurrence) were dropped with it.
"""

from __future__ import annotations

import os

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
from models.demos.deepseek_v3_d_p.tt.kda.weights import load_kda_weights
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.glm53_flash_d_p.tt import mm_configs

SP_AXIS, TP_AXIS = 0, 1


def kda_fidelity():
    """GLM_KDA_FIDELITY, default HiFi4 (not the attention fidelity). At HiFi2 the KDA attention output is ~3.4% low on
    every row (L0 component test rel L2 0.0345, norm ratio 0.960..0.979; HiFi4 0.0071, 0.990..0.998). Full model:
    s4096 KV PCC mean kv_latent 0.98405 -> 0.98509, index_key 0.99491 -> 0.99546; 56k prefill 8.47 -> 8.62 s."""
    return getattr(ttnn.MathFidelity, os.environ.get("GLM_KDA_FIDELITY", "HiFi4"))


KDA_OUT_RS = os.environ.get("GLM_KDA_OUT_RS", "fabric")
KDA_LINKS = int(os.environ.get("GLM_MOE_LINKS", "2"))
START_ALIGN = 64
END_ALIGN = 32
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


class _GlmKDA(ttKDA):
    """ttKDA with GLM's projection matmul configs, its fidelity override and the row-partial output path."""

    def __init__(self, *args, program_config, **kwargs):
        super().__init__(*args, program_config=program_config, **kwargs)
        # explicit projection schedules (tt/mm_configs.py; tests/test_matmul_tune.py at 2560 rows per chip): input
        # projection minimal_matmul M4 K8 N4 (1.511 -> 1.260 ms, 91% of HiFi4), o_proj 2D multicast (0.504 -> 0.450 ms)
        if mm_configs.ENABLED:
            k_o, n_o = tuple(self.weights.output_projection.shape)[-2:]
            self.input_projection_minimal_matmul_config = mm_configs.minimal_config(self.device, 4, 8, 4, 1, 4)
            self.output_projection_program_config = mm_configs.mm2d(
                self.device, self.active_seq_len_local, k_o, n_o, ttnn.float32, ttnn.bfloat16, ttnn.float32
            )
        fid = kda_fidelity()
        if fid != ttnn.MathFidelity.HiFi4:  # ttKDA hard-codes HiFi4 for its projections and KDA ops
            arch = self.device.arch()
            self.compute_config = ttnn.init_device_compute_kernel_config(
                arch, math_fidelity=fid, fp32_dest_acc_en=True, packer_l1_acc=True
            )
            self.kda_compute_config = ttnn.init_device_compute_kernel_config(
                arch, math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
            )

    row_partial = False  # set per call by TtKdaAttention: skip the TP reduction, return o_proj's bf16 partial

    def _project_output(self, output):
        """ttKDA's o_proj + reduce-scatter on the hidden dim, or (row_partial) o_proj only: [1, S/sp, H] partial sums
        over the TP group in bf16, reduced over rows by the caller (fabric_reduce_scatter)."""
        if not self.row_partial:
            return super()._project_output(output)
        o = ttnn.linear(
            output,
            self.weights.output_projection,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.output_projection_program_config,
            compute_kernel_config=self.output_projection_compute_config,
        )
        return o


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
        m = max_seq // END_ALIGN + 1
        ends = torch.arange(m, dtype=torch.int64).mul(END_ALIGN).reshape(m, 1)
        self.ends = ttnn.from_torch(
            ends,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.state = self._zeros()
        self.zero_state = self._zeros()

    def _zeros(self) -> KdaState:
        return kda_state_zeros(self.mesh, self.kcfg)

    def bind_state(self, state: KdaState) -> None:
        """Carry the state in another address-stable buffer pair (kda_state_zeros); the forward copies into it."""
        self.state = state

    def _program_config(self, local_rows: int) -> KDAProgramConfig:
        chunks = local_rows // ttnn.TILE_SIZE
        groups = _scan_groups(self.mesh, chunks, self.local_heads)
        return KDAProgramConfig(
            recurrence=KDARecurrenceProgramConfig(
                local_scan_strategy="grouped",
                summary_group_chunks=chunks // groups,
                affine_prefix_math_fidelity=kda_fidelity(),
                scan_math_fidelity=kda_fidelity(),
            ),
            tp_ccl_topology=ttnn.Topology.Linear,
            gated_rms_output_dtype=ttnn.float32,
            output_projection_math_fidelity=kda_fidelity(),
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
            )
        return self._layers[s]

    def __call__(self, x: ttnn.Tensor, start: int, end: int | None = None, split: bool = False) -> ttnn.Tensor:
        """x [1, 1, S, H] replicated bf16, start the chunk's absolute position (a multiple of S) ->
        attn_out [1, 1, S, H] replicated bf16. Advances the carried state. end (optional): the exclusive valid end,
        32-aligned, in (start, start + S]; rows past it are pad (their output is unspecified, the state skips them).
        split: x and attn_out are this chip's [1, 1, S/4, H] (tt/common.py split layout): the input is gathered on
        axis 1 to the SP half, the output all_gathered on axis 1 (hidden) and cut to the chip's quarter."""
        s = x.shape[-2] * (self.sp * self.tp if split else 1)
        assert start % s == 0 and start % START_ALIGN == 0, f"chunk start {start} must be a multiple of S={s}"
        if end is not None:
            assert start < end <= start + s and end % END_ALIGN == 0, f"actual_end {end} for chunk [{start}, +{s})"
        kda = self._kda(s)
        if split:  # rows of mesh row r = its two chips' quarters
            from models.demos.glm53_flash_d_p.tt.common import gather_axis

            xs = gather_axis(x, TP_AXIS)
        else:
            xs = ttnn.mesh_partition(x, dim=-2, cluster_axis=SP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = ttnn.reshape(xs, (1, s // self.sp, self.hidden))
        i = start // START_ALIGN
        actual_start = ttnn.slice(self.starts, (i, 0), (i + 1, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        actual_end = None
        if end is not None:
            j = end // END_ALIGN
            actual_end = ttnn.slice(self.ends, (j, 0), (j + 1, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        state = self.zero_state if start == 0 else self.state
        # split: o_proj partials reduce-scattered over rows on axis 1 by MiMo's fabric_reduce_scatter (bf16), straight
        # to the chip's quarter (GLM_KDA_OUT_RS=fabric, default); "ttnn": ttKDA's fp32 reduce-scatter on the hidden
        # dim + all_gather (hidden) + mesh_partition
        kda.row_partial = split and KDA_OUT_RS == "fabric"
        out, new = kda.forward(h, state, actual_start=actual_start, actual_end=actual_end)
        ttnn.deallocate(actual_start)
        if actual_end is not None:
            ttnn.deallocate(actual_end)
        ttnn.deallocate(xs)
        ttnn.copy(new.recurrent, self.state.recurrent)
        ttnn.copy(new.convolution, self.state.convolution)
        ttnn.deallocate(new.recurrent)
        ttnn.deallocate(new.convolution)
        if kda.row_partial:  # out [1, S/2, H] bf16 partial -> the chip's [1, 1, S/4, H] quarter of the sum
            o4 = ttnn.reshape(out, (1, 1, s // self.sp, self.hidden))
            y = ttnn.bringup.fabric_reduce_scatter(o4, cluster_axis=TP_AXIS, num_links=KDA_LINKS)
            ttnn.deallocate(out)
            return y
        # out [1, S/2, H/2]: rows of this SP rank, hidden reduce-scattered over TP.
        o4 = ttnn.reshape(out, (1, 1, s // self.sp, self.hidden // self.tp))
        g1 = ttnn.all_gather(o4, dim=-1, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        if split:
            g2 = ttnn.mesh_partition(g1, dim=-2, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
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

    def state_torch(self, state: KdaState | None = None) -> dict:
        """The carried state (or ``state``, e.g. a serving slot's) in the reference layout (mesh row 0's copy; rows
        are replicated)."""
        state = self.state if state is None else state
        cols = tuple(self.mesh.shape)[1]
        rec = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(state.recurrent)[:cols]]
        conv = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(state.convolution)[:cols]]
        rec = torch.cat(rec, dim=1).reshape(self.kcfg.num_heads, self.kcfg.head_k_dim, -1).float()
        conv = self._conv_to_ref_order(torch.cat(conv, dim=-1).reshape(self.kcfg.conv_kernel_size - 1, -1).float())
        return {"kda_recurrent": rec, "kda_conv": conv}


def kda_state_zeros(mesh, kcfg: KDAConfig) -> KdaState:
    """A zeroed per-chip carry pair: recurrent [1, H / tp, K, V] (TP shard of the heads), conv [1, 3, 3 H / tp K]."""
    tp = tuple(mesh.shape)[TP_AXIS]
    local_heads = kcfg.num_heads // tp
    return KdaState(
        recurrent=ttnn.zeros(
            (1, local_heads, kcfg.head_k_dim, kcfg.head_v_dim),
            dtype=KDA_RECURRENT_STATE_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        convolution=ttnn.zeros(
            (1, kcfg.conv_kernel_size - 1, 3 * local_heads * kcfg.head_k_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )


def build_kda_attention(mesh, loader, cfg, layer: int, max_seq: int) -> TtKdaAttention:
    sd = {n: loader.layer(layer, f"self_attn.{n}") for n in _KDA_NAMES}
    return TtKdaAttention(mesh, sd, cfg, max_seq, layer)
