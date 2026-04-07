# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5 Gated DeltaNet (GDN) linear attention layer.

Implements the DeltaNet recurrence with:
- 4-tap causal conv1d with trace-compatible shift register
- L2-normalized QK with scale factor
- DeltaNet recurrence: decay * state + k ⊗ (beta * (v - k^T state))
- SiLU-gated output with Z projection
- All operations on device (trace-compatible)
"""

import math as _math
import os as _os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_full_fused_inplace, gdn_recurrence_fused_inplace
from models.demos.qwen35_27b.tt.model_config import create_prefill_matmul_program_config
from models.tt_transformers.tt.ccl import tt_all_reduce


def _shard_linear(x_tt, weight, act_shard_cfg, prog_cfg, compute_cfg):
    x_sharded = ttnn.to_memory_config(x_tt, act_shard_cfg)
    return ttnn.linear(
        x_sharded,
        weight,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=prog_cfg,
        compute_kernel_config=compute_cfg,
    )


def _unshard(t):
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return t


def _retile(t):
    """Force proper re-tiling after reshape.

    ttnn.reshape changes logical shape but doesn't re-tile data when the tile
    structure changes (e.g. [B,H,D] -> [B*H,1,D]). Round-tripping through
    ROW_MAJOR forces correct tile padding/layout for raw tile-index access.
    """
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT)


def _l2_norm_dev(x):
    """L2 normalize along last dim: x / (||x|| + eps)"""
    x_sq = ttnn.multiply(x, x)
    ssq = ttnn.sum(x_sq, dim=-1, keepdim=True)
    ttnn.deallocate(x_sq)
    inv = ttnn.rsqrt(ttnn.add(ssq, 1e-6))
    ttnn.deallocate(ssq)
    normed = ttnn.multiply(x, inv)
    ttnn.deallocate(inv)
    return normed


class TtGatedDeltaNet(LightweightModule):
    """Gated DeltaNet (GDN) linear attention for Qwen3.5.

    Matches the Attention constructor signature so TransformerBlock can instantiate it.
    """

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype

        # No KV cache for GDN layers
        self.layer_past = None
        self.is_sliding = False

        self.batch_size = args.max_batch_size
        self.scale = args.gdn_dk**-0.5

        # Architecture constants from args
        self.Nk_TP = args.gdn_nk_tp
        self.Nv_TP = args.gdn_nv_tp
        self.Dk = args.gdn_dk
        self.Dv = args.gdn_dv
        self.qkv_dim_tp = args.gdn_qkv_dim_tp
        self.qkvz_dim_tp = args.gdn_qkvz_dim_tp
        self.value_dim_tp = args.gdn_value_dim_tp
        self.key_dim_tp = args.gdn_key_dim_tp
        self.conv_kernel_size = args.gdn_conv_kernel_size

        self.compute_cfg = args.compute_kernel_config_hifi2

        self.tw = self._load_weights(state_dict, layer_num, mesh_device, weight_cache_path)

        # Mutable state buffers (conv + recurrence)
        self.conv_states = None
        self.rec_states = None
        self.rec_output = None  # Pre-allocated output for fused kernel

        # Precomputed constants (set in set_weights or _precompute_constants)
        self.neg_exp_A = None
        self.scale_tt = None  # Q scale as device tile
        self.rms_scale_tt = None  # sqrt(Dv) as device tile
        self.fused_output = None  # Pre-allocated [1, B, value_dim_tp] for full fused

        self._use_full_fused = not _os.environ.get("GDN_DISABLE_FULL_FUSED", "")

    def _load_weights(self, state_dict, layer_num, mesh_device, weight_cache_path):
        if isinstance(state_dict, dict) and "qkvz" in state_dict:
            return state_dict
        return {}

    def set_weights(self, layer_weights):
        """Set pre-loaded mesh tensor weights from Qwen35 weight loading."""
        self.tw = layer_weights
        self._precompute_constants()

    def _precompute_constants(self):
        """Precompute constants: neg_exp_A = -exp(A_log), scale_tt, rms_scale_tt."""
        if "A_log" in self.tw:
            exp_A = ttnn.exp(self.tw["A_log"])
            self.neg_exp_A = ttnn.neg(exp_A)
            ttnn.deallocate(exp_A)

        # Precompute scalar tiles for full fused kernel
        mesh = self.mesh_device

        def _scalar_to_mesh(val):
            t = torch.full((1, 1, 1), val, dtype=torch.bfloat16)
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        self.scale_tt = _scalar_to_mesh(self.scale)
        self.rms_scale_tt = _scalar_to_mesh(_math.sqrt(self.Dv))
        self.rms_eps_tt = _scalar_to_mesh(self.Dv * 1e-6)

    def reset_state(self):
        """Reset conv and recurrence states to zero (creates new tensors)."""
        B = self.batch_size
        mesh = self.mesh_device

        def _to_mesh(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        self.conv_states = [
            _to_mesh(torch.zeros(1, B, self.qkv_dim_tp, dtype=torch.bfloat16)) for _ in range(self.conv_kernel_size)
        ]
        self.rec_states = _to_mesh(torch.zeros(B * self.Nv_TP, self.Dk, self.Dv, dtype=torch.bfloat16))
        self.rec_output = None  # Will be allocated on first forward

    def reset_state_inplace(self):
        """Zero states in-place, preserving tensor IDs (trace-compatible)."""
        if self.conv_states is None:
            self.reset_state()
            return

        B = self.batch_size
        mesh = self.mesh_device

        def _to_mesh(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        zeros_conv = _to_mesh(torch.zeros(1, B, self.qkv_dim_tp, dtype=torch.bfloat16))
        for cs in self.conv_states:
            ttnn.copy(zeros_conv, cs)
        ttnn.deallocate(zeros_conv)

        zeros_rec = _to_mesh(torch.zeros(B * self.Nv_TP, self.Dk, self.Dv, dtype=torch.bfloat16))
        ttnn.copy(zeros_rec, self.rec_states)
        ttnn.deallocate(zeros_rec)

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        if mode == "prefill" or (
            hasattr(mode, "value") and mode.value == "prefill"
        ):  # Mode is always decode; even for prefill
            return self.forward_prefill(x, current_pos)
        return self.forward_decode(x)

    def forward_decode(self, x):
        """GDN decode: dispatches to fused or unfused path."""
        if self._use_full_fused:
            try:
                return self._forward_decode_fused(x)  # e2e traced uses fused path
            except Exception:
                from loguru import logger

                logger.warning("Full fused GDN failed, falling back to unfused path")
                self._use_full_fused = False
        return self._forward_decode_unfused(x)

    def _forward_decode_fused(self, x):
        """GDN decode with full fused kernel (Phase A).

        Passes conv_out directly to the kernel — the reader extracts Q/K/V
        per-pair via sub-tile row reads. No Python-side slice/reshape/retile
        for Q/K/V/z. RMS norm + SiLU gate done via ttnn.
        """
        tw = self.tw
        B = self.batch_size
        Nk_TP = self.Nk_TP
        Nv_TP = self.Nv_TP
        Dk = self.Dk
        Dv = self.Dv
        qkv_dim_tp = self.qkv_dim_tp
        qkvz_dim_tp = self.qkvz_dim_tp
        key_dim_tp = self.key_dim_tp
        act_shard = self.args.act_shard_hidden
        num_pairs = B * Nv_TP
        repeat_factor = Nv_TP // Nk_TP

        if self.conv_states is None:
            self.reset_state()

        # Framework passes 4D [1, 1, B, H]; flatten to [1, B, H]
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))

        # ---- Projections ----
        qkvz_tt = _unshard(_shard_linear(x, tw["qkvz"], act_shard, self.args.gdn_qkvz_progcfg, self.compute_cfg))
        qkv_tt = ttnn.slice(qkvz_tt, (0, 0, 0), (1, B, qkv_dim_tp))
        z_tt = ttnn.slice(qkvz_tt, (0, 0, qkv_dim_tp), (1, B, qkvz_dim_tp))
        ttnn.deallocate(qkvz_tt)
        ab_tt = ttnn.linear(x, tw["ab"])

        if len(ab_tt.shape) == 4:
            ab_tt = ttnn.reshape(ab_tt, (1, B, Nv_TP * 2))
        a_tt = ttnn.slice(ab_tt, (0, 0, 0), (1, B, Nv_TP))
        b_tt = ttnn.slice(ab_tt, (0, 0, Nv_TP), (1, B, Nv_TP * 2))
        ttnn.deallocate(ab_tt)

        # ---- Conv1d ----
        if len(qkv_tt.shape) == 4:
            qkv_tt = ttnn.reshape(qkv_tt, (1, B, qkv_dim_tp))
        states = self.conv_states
        ttnn.copy(states[1], states[0])
        ttnn.copy(states[2], states[1])
        ttnn.copy(states[3], states[2])
        ttnn.copy(qkv_tt, states[3])

        conv_acc = ttnn.multiply(states[0], tw["conv_taps"][0])
        for j in range(1, self.conv_kernel_size):
            conv_acc = ttnn.mac(states[j], tw["conv_taps"][j], conv_acc)
        conv_out = ttnn.silu(conv_acc)
        ttnn.deallocate(conv_acc)
        if len(conv_out.shape) == 4:
            conv_out = ttnn.reshape(conv_out, (1, B, qkv_dim_tp))

        # ---- PREP: scalars extracted by reader kernel from original shapes ----
        # Q/K/V read from conv_out by reader. a/b/neg_exp_A/dt_bias scalars
        # extracted per-pair in reader via sub-tile scalar reads.
        # z is handled in POST via ttnn.silu(z_tt).
        a_tt = _unshard(a_tt)
        b_tt = _unshard(b_tt)

        # ---- Pre-allocate output buffer [num_pairs, 1, Dv] ----
        if self.fused_output is None:
            self.fused_output = ttnn.from_torch(
                torch.zeros(num_pairs, 1, Dv, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        # ---- Full fused kernel: L2 norm + gates + recurrence (1 dispatch) ----
        # conv_out [1, B, qkv_dim_tp] passed directly — reader extracts Q/K/V
        conv_out = _unshard(conv_out)

        gdn_full_fused_inplace(
            conv_out,
            a_tt,
            b_tt,
            self.neg_exp_A,
            tw["dt_bias"],
            tw["norm_w"],
            self.scale_tt,
            self.rms_scale_tt,
            self.rms_eps_tt,
            self.rec_states,
            self.fused_output,
            num_pairs=num_pairs,
            num_cores=min(96, num_pairs),
            Nv_TP=Nv_TP,
            Nk_TP=Nk_TP,
            repeat_factor=repeat_factor,
            key_dim_tp=key_dim_tp,
        )

        ttnn.deallocate(conv_out)
        ttnn.deallocate(a_tt)
        ttnn.deallocate(b_tt)

        # ---- Post-kernel: RMS norm + SiLU gate via ttnn (FP32 precision) ----
        out_r = ttnn.reshape(self.fused_output, (B, Nv_TP, Dv))
        out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6)
        ttnn.deallocate(out_r)
        out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
        ttnn.deallocate(out_n)
        z_act = ttnn.silu(z_tt)
        ttnn.deallocate(z_tt)
        out_f = _unshard(out_f)
        gated = ttnn.multiply(out_f, z_act)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z_act)

        # ---- Output projection + all-reduce ----
        act_shard_out = self.args.act_shard_gdn_value
        out_partial = _unshard(
            _shard_linear(gated, tw["out"], act_shard_out, self.args.gdn_out_progcfg, self.compute_cfg)
        )

        out_partial = ttnn.reshape(out_partial, (1, 1, B, out_partial.shape[-1]))
        return self._all_reduce(out_partial)

    def _forward_decode_unfused(self, x):
        """GDN decode: original unfused path (fallback)."""
        tw = self.tw
        B = self.batch_size
        Nk_TP = self.Nk_TP
        Nv_TP = self.Nv_TP
        Dk = self.Dk
        Dv = self.Dv
        qkv_dim_tp = self.qkv_dim_tp
        qkvz_dim_tp = self.qkvz_dim_tp
        act_shard = self.args.act_shard_hidden

        if self.conv_states is None:
            self.reset_state()

        # Framework passes 4D [1, 1, B, H]; flatten to [1, B, H]
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))

        # ---- Projections ----
        qkvz_tt = _unshard(_shard_linear(x, tw["qkvz"], act_shard, self.args.gdn_qkvz_progcfg, self.compute_cfg))
        qkv_tt = ttnn.slice(qkvz_tt, (0, 0, 0), (1, B, qkv_dim_tp))
        z_tt = ttnn.slice(qkvz_tt, (0, 0, qkv_dim_tp), (1, B, qkvz_dim_tp))
        ttnn.deallocate(qkvz_tt)
        ab_tt = ttnn.linear(x, tw["ab"])

        if len(ab_tt.shape) == 4:
            ab_tt = ttnn.reshape(ab_tt, (1, B, Nv_TP * 2))
        a_tt = ttnn.slice(ab_tt, (0, 0, 0), (1, B, Nv_TP))
        b_tt = ttnn.slice(ab_tt, (0, 0, Nv_TP), (1, B, Nv_TP * 2))
        ttnn.deallocate(ab_tt)

        # ---- Conv1d (shift register + weighted sum) ----
        if len(qkv_tt.shape) == 4:
            qkv_tt = ttnn.reshape(qkv_tt, (1, B, qkv_dim_tp))
        states = self.conv_states
        ttnn.copy(states[1], states[0])
        ttnn.copy(states[2], states[1])
        ttnn.copy(states[3], states[2])
        ttnn.copy(qkv_tt, states[3])

        conv_acc = ttnn.multiply(states[0], tw["conv_taps"][0])
        for j in range(1, self.conv_kernel_size):
            conv_acc = ttnn.mac(states[j], tw["conv_taps"][j], conv_acc)
        conv_out = ttnn.silu(conv_acc)
        ttnn.deallocate(conv_acc)
        if len(conv_out.shape) == 4:
            conv_out = ttnn.reshape(conv_out, (1, B, qkv_dim_tp))

        # ---- Split Q/K/V from conv output ----
        key_dim_tp = self.key_dim_tp
        q_sl = ttnn.slice(conv_out, (0, 0, 0), (1, B, key_dim_tp))
        k_sl = ttnn.slice(conv_out, (0, 0, key_dim_tp), (1, B, 2 * key_dim_tp))
        v_sl = ttnn.slice(conv_out, (0, 0, 2 * key_dim_tp), (1, B, qkv_dim_tp))
        ttnn.deallocate(conv_out)

        # Reshape to head format
        q_h = ttnn.reshape(q_sl, (B, Nk_TP, Dk))
        ttnn.deallocate(q_sl)
        k_h = ttnn.reshape(k_sl, (B, Nk_TP, Dk))
        ttnn.deallocate(k_sl)
        v_h = ttnn.reshape(v_sl, (B, Nv_TP, Dv))
        ttnn.deallocate(v_sl)

        # ---- L2 normalize BEFORE repeat_interleave (3x less compute) ----
        q_normed = _l2_norm_dev(q_h)
        ttnn.deallocate(q_h)
        k_normed = _l2_norm_dev(k_h)
        ttnn.deallocate(k_h)

        # Expand to Nv_TP heads and apply scale to Q
        repeat_factor = Nv_TP // Nk_TP
        q_exp = ttnn.repeat_interleave(q_normed, repeat_factor, dim=1)
        ttnn.deallocate(q_normed)
        q_ns = ttnn.multiply(q_exp, self.scale)
        ttnn.deallocate(q_exp)

        k_exp = ttnn.repeat_interleave(k_normed, repeat_factor, dim=1)
        ttnn.deallocate(k_normed)

        # ---- Decay and beta ----
        beta_tt = ttnn.sigmoid(b_tt)
        ttnn.deallocate(b_tt)
        sp = ttnn.softplus(ttnn.add(a_tt, tw["dt_bias"]))
        ttnn.deallocate(a_tt)
        g_pre = ttnn.multiply(self.neg_exp_A, sp)
        ttnn.deallocate(sp)

        # ---- Reshape for fused kernel [num_pairs, ...] ----
        num_pairs = B * Nv_TP

        q_fused = _retile(_unshard(ttnn.reshape(q_ns, (num_pairs, 1, Dk))))
        ttnn.deallocate(q_ns)

        k_row = _retile(_unshard(ttnn.reshape(k_exp, (num_pairs, 1, Dk))))
        k_col = _unshard(ttnn.transpose(k_row, -2, -1))
        ttnn.deallocate(k_exp)

        v_fused = _retile(_unshard(ttnn.reshape(v_h, (num_pairs, 1, Dv))))
        ttnn.deallocate(v_h)

        g_fused = _retile(_unshard(ttnn.reshape(g_pre, (num_pairs, 1, 1))))
        ttnn.deallocate(g_pre)
        beta_fused = _retile(_unshard(ttnn.reshape(beta_tt, (num_pairs, 1, 1))))
        ttnn.deallocate(beta_tt)

        # ---- Fused DeltaNet recurrence ----
        if self.rec_output is None:
            self.rec_output = _unshard(ttnn.zeros_like(q_fused))

        gdn_recurrence_fused_inplace(
            q_fused,
            k_row,
            k_col,
            v_fused,
            g_fused,
            beta_fused,
            self.rec_states,
            self.rec_output,
            num_cores=10,
        )
        ttnn.deallocate(q_fused)
        ttnn.deallocate(k_row)
        ttnn.deallocate(k_col)
        ttnn.deallocate(v_fused)
        ttnn.deallocate(g_fused)
        ttnn.deallocate(beta_fused)

        # ---- Post-processing ----
        out_r = ttnn.reshape(self.rec_output, (B, Nv_TP, Dv))
        out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6)
        ttnn.deallocate(out_r)
        out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
        ttnn.deallocate(out_n)

        # Gated output with Z (SiLU)
        z_act = ttnn.silu(z_tt)
        ttnn.deallocate(z_tt)
        out_f = _unshard(out_f)
        gated = ttnn.multiply(out_f, z_act)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z_act)

        # Output projection + all-reduce
        act_shard_out = self.args.act_shard_gdn_value
        out_partial = _unshard(
            _shard_linear(gated, tw["out"], act_shard_out, self.args.gdn_out_progcfg, self.compute_cfg)
        )
        ttnn.deallocate(gated)

        # Reshape to 4D [1, 1, B, H] for framework
        out_partial = ttnn.reshape(out_partial, (1, 1, B, out_partial.shape[-1]))
        return self._all_reduce(out_partial)

    def _all_reduce(self, partial_mesh):
        return tt_all_reduce(
            partial_mesh,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _init_prefill_states(self):
        """Create B=1 conv/rec states for prefill (separate from B=32 decode states)."""
        mesh = self.mesh_device

        def _to_mesh(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        self._prefill_conv_states = [
            _to_mesh(torch.zeros(1, 1, self.qkv_dim_tp, dtype=torch.bfloat16)) for _ in range(self.conv_kernel_size)
        ]
        self._prefill_rec_states = _to_mesh(torch.zeros(1 * self.Nv_TP, self.Dk, self.Dv, dtype=torch.bfloat16))
        self._prefill_fused_output = _to_mesh(torch.zeros(1 * self.Nv_TP, 1, self.Dv, dtype=torch.bfloat16))

    def replicate_prefill_state_to_batch(self):
        """Copy B=1 prefill GDN states to all B=32 decode slots.

        After prefilling 1 user, this replicates user 0's conv and recurrence
        states across all batch_size slots so batched decode starts correctly.

        Each device holds its own TP shard, so we replicate per-device.
        """
        B = self.batch_size
        mesh = self.mesh_device

        if not hasattr(self, "_prefill_conv_states") or self._prefill_conv_states is None:
            return

        if self.conv_states is None:
            self.reset_state()

        # Conv states: [1, 1, qkv_dim_tp] per device -> [1, B, qkv_dim_tp] per device
        for i in range(self.conv_kernel_size):
            per_dev = ttnn.get_device_tensors(self._prefill_conv_states[i])
            batched_parts = []
            for dev_t in per_dev:
                cpu = ttnn.to_torch(dev_t)  # [1, 1, qkv_dim_tp]
                batched_parts.append(cpu.expand(1, B, -1).contiguous())
            combined = torch.cat(batched_parts, dim=0)  # [num_devices, B, qkv_dim_tp]
            # ShardTensorToMesh on dim=0 splits back to per-device
            new_state = ttnn.from_torch(
                combined,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            ttnn.copy(new_state, self.conv_states[i])
            ttnn.deallocate(new_state)

        # Rec states: [Nv_TP, Dk, Dv] per device -> [B*Nv_TP, Dk, Dv] per device
        per_dev_rec = ttnn.get_device_tensors(self._prefill_rec_states)
        batched_rec_parts = []
        for dev_t in per_dev_rec:
            cpu = ttnn.to_torch(dev_t)  # [Nv_TP, Dk, Dv]
            batched_rec_parts.append(cpu.repeat(B, 1, 1))  # [B*Nv_TP, Dk, Dv]
        combined_rec = torch.cat(batched_rec_parts, dim=0)  # [num_devices*B*Nv_TP, Dk, Dv]
        new_rec = ttnn.from_torch(
            combined_rec,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        ttnn.copy(new_rec, self.rec_states)
        ttnn.deallocate(new_rec)

        # Clean up prefill states
        for s in self._prefill_conv_states:
            ttnn.deallocate(s)
        ttnn.deallocate(self._prefill_rec_states)
        ttnn.deallocate(self._prefill_fused_output)
        self._prefill_conv_states = None
        self._prefill_rec_states = None
        self._prefill_fused_output = None

    def forward_prefill(self, x, current_pos):
        """GDN prefill: batched projections + conv1d, sequential recurrence with B=1.

        Input: [1, 1, seq_len, dim] — 1 user, full prompt sequence
        Output: [1, 1, seq_len, dim] — full sequence output for residual stream

        Projections (QKVZ, AB) and causal conv1d are computed once for the full
        sequence using batched ops. Only the recurrence loop runs per-token.
        Output projection is batched for the full sequence.
        """
        tw = self.tw
        Nk_TP = self.Nk_TP
        Nv_TP = self.Nv_TP
        Dk = self.Dk
        Dv = self.Dv
        qkv_dim_tp = self.qkv_dim_tp
        qkvz_dim_tp = self.qkvz_dim_tp
        key_dim_tp = self.key_dim_tp
        dim = self.args.dim
        B_pf = 1  # Prefill processes 1 user

        # Ensure 4D: [1, 1, seq_len, dim]
        if len(x.shape) == 3:
            x = ttnn.reshape(x, (1, 1, x.shape[1], x.shape[2]))
        seq_len = x.shape[2]

        # Init B=1 prefill states
        if not hasattr(self, "_prefill_conv_states") or self._prefill_conv_states is None:
            self._init_prefill_states()

        # ---- Batch projections (2D matmul, full sequence) ----
        x_dram = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        qkvz_progcfg = create_prefill_matmul_program_config(seq_len, dim, qkvz_dim_tp)
        qkvz_all = ttnn.linear(
            x_dram,
            tw["qkvz"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=qkvz_progcfg,
            compute_kernel_config=self.compute_cfg,
        )
        # qkvz_all: [1, 1, seq_len, qkvz_dim_tp]

        ab_progcfg = create_prefill_matmul_program_config(seq_len, dim, Nv_TP * 2)
        ab_all = ttnn.linear(
            x_dram,
            tw["ab"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=ab_progcfg,
            compute_kernel_config=self.compute_cfg,
        )
        # ab_all: [1, 1, seq_len, Nv_TP*2]
        ttnn.deallocate(x_dram)

        # ---- Split projections and pre-compute batched conv1d ----
        # Split qkv and z from qkvz_all: [1, 1, seq_len, qkvz_dim_tp]
        qkv_all = ttnn.slice(qkvz_all, (0, 0, 0, 0), (1, 1, seq_len, qkv_dim_tp))
        z_all = ttnn.slice(qkvz_all, (0, 0, 0, qkv_dim_tp), (1, 1, seq_len, qkvz_dim_tp))
        ttnn.deallocate(qkvz_all)

        # Split a and b from ab_all: [1, 1, seq_len, Nv_TP*2]
        a_all = ttnn.slice(ab_all, (0, 0, 0, 0), (1, 1, seq_len, Nv_TP))
        b_all = ttnn.slice(ab_all, (0, 0, 0, Nv_TP), (1, 1, seq_len, Nv_TP * 2))
        ttnn.deallocate(ab_all)

        # Batched causal conv1d over full sequence
        # conv_out[t] = silu(tap[0]*qkv[t-3] + tap[1]*qkv[t-2] + tap[2]*qkv[t-1] + tap[3]*qkv[t])
        # where qkv[t<0] = 0 (causal zero-padding)
        K = self.conv_kernel_size  # 4

        # Build shifted views: for tap[j], we need qkv shifted right by (K-1-j) positions
        # Pad qkv_all with (K-1)=3 zero rows at the start of the sequence dim
        pad_shape = [1, 1, K - 1, qkv_dim_tp]
        zero_pad = ttnn.from_torch(
            torch.zeros(pad_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        padded = ttnn.concat([zero_pad, qkv_all], dim=2)  # [1, 1, seq_len + K-1, qkv_dim_tp]
        ttnn.deallocate(zero_pad)

        # Compute weighted sum of shifted versions
        # tap[0] applies to the oldest input (shift right by K-1), tap[K-1] to current (no shift)
        # Reshape taps from [1, 1, qkv_dim_tp] to [1, 1, 1, qkv_dim_tp] for 4D broadcast
        conv_out_all = None
        for j in range(K):
            shift = K - 1 - j  # tap[0] -> shift 3, tap[3] -> shift 0
            shifted = ttnn.slice(padded, (0, 0, shift, 0), (1, 1, shift + seq_len, qkv_dim_tp))
            tap_4d = ttnn.reshape(tw["conv_taps"][j], (1, 1, 1, qkv_dim_tp))
            if conv_out_all is None:
                conv_out_all = ttnn.multiply(shifted, tap_4d)
            else:
                conv_out_all = ttnn.mac(shifted, tap_4d, conv_out_all)
            ttnn.deallocate(shifted)
        ttnn.deallocate(padded)

        conv_out_all = ttnn.silu(conv_out_all)
        # conv_out_all: [1, 1, seq_len, qkv_dim_tp]

        # Save last (K-1) tokens of qkv input into prefill conv states for decode
        states = self._prefill_conv_states
        for j in range(K):
            # states[0] = qkv[seq_len - K], ..., states[K-1] = qkv[seq_len - 1]
            t_idx = seq_len - K + j
            if t_idx >= 0:
                qkv_t = ttnn.slice(qkv_all, (0, 0, t_idx, 0), (1, 1, t_idx + 1, qkv_dim_tp))
                qkv_t = ttnn.reshape(qkv_t, (1, B_pf, qkv_dim_tp))
                ttnn.copy(qkv_t, states[j])
                ttnn.deallocate(qkv_t)
        ttnn.deallocate(qkv_all)

        # ---- Sequential per-token recurrence (conv1d already computed) ----
        gated_outputs = []
        num_pairs_pf = B_pf * Nv_TP
        repeat_factor = Nv_TP // Nk_TP

        for t in range(seq_len):
            # Slice pre-computed conv_out, z, a, b for token t
            conv_out_t = ttnn.slice(conv_out_all, (0, 0, t, 0), (1, 1, t + 1, qkv_dim_tp))
            conv_out_t = ttnn.reshape(conv_out_t, (1, B_pf, qkv_dim_tp))

            a_tt = ttnn.slice(a_all, (0, 0, t, 0), (1, 1, t + 1, Nv_TP))
            a_tt = ttnn.reshape(a_tt, (1, B_pf, Nv_TP))

            b_tt = ttnn.slice(b_all, (0, 0, t, 0), (1, 1, t + 1, Nv_TP))
            b_tt = ttnn.reshape(b_tt, (1, B_pf, Nv_TP))

            z_tt = ttnn.slice(z_all, (0, 0, t, 0), (1, 1, t + 1, qkvz_dim_tp - qkv_dim_tp))
            z_tt = ttnn.reshape(z_tt, (1, B_pf, qkvz_dim_tp - qkv_dim_tp))

            # Fused kernel with B=1
            a_tt = _unshard(a_tt)
            b_tt = _unshard(b_tt)
            conv_out_t = _unshard(conv_out_t)

            gdn_full_fused_inplace(
                conv_out_t,
                a_tt,
                b_tt,
                self.neg_exp_A,
                tw["dt_bias"],
                tw["norm_w"],
                self.scale_tt,
                self.rms_scale_tt,
                self.rms_eps_tt,
                self._prefill_rec_states,
                self._prefill_fused_output,
                num_pairs=num_pairs_pf,
                num_cores=min(96, num_pairs_pf),
                Nv_TP=Nv_TP,
                Nk_TP=Nk_TP,
                repeat_factor=repeat_factor,
                key_dim_tp=key_dim_tp,
            )

            ttnn.deallocate(conv_out_t)
            ttnn.deallocate(a_tt)
            ttnn.deallocate(b_tt)

            # Post-kernel: RMS norm + SiLU gate
            out_r = ttnn.reshape(self._prefill_fused_output, (B_pf, Nv_TP, Dv))
            out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6)
            ttnn.deallocate(out_r)
            out_f = ttnn.reshape(out_n, (1, B_pf, self.value_dim_tp))
            ttnn.deallocate(out_n)

            z_act = ttnn.silu(z_tt)
            ttnn.deallocate(z_tt)
            out_f = _unshard(out_f)
            gated = ttnn.multiply(out_f, z_act)
            ttnn.deallocate(out_f)
            ttnn.deallocate(z_act)

            gated_outputs.append(gated)  # [1, 1, value_dim_tp]

        ttnn.deallocate(conv_out_all)
        ttnn.deallocate(z_all)
        ttnn.deallocate(a_all)
        ttnn.deallocate(b_all)

        # ---- Stack outputs and batch output projection ----
        # list of [1, 1, value_dim_tp] -> [1, 1, seq_len, value_dim_tp]
        if len(gated_outputs) > 1:
            gated_seq = ttnn.concat(gated_outputs, dim=1)  # [1, seq_len, value_dim_tp]
            for g in gated_outputs:
                ttnn.deallocate(g)
            gated_seq = ttnn.reshape(gated_seq, (1, 1, seq_len, self.value_dim_tp))
        else:
            gated_seq = ttnn.reshape(gated_outputs[0], (1, 1, 1, self.value_dim_tp))

        # Output projection with 2D matmul (full sequence)
        out_progcfg = create_prefill_matmul_program_config(seq_len, self.value_dim_tp, dim)
        out_partial = ttnn.linear(
            gated_seq,
            tw["out"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=out_progcfg,
            compute_kernel_config=self.compute_cfg,
        )
        ttnn.deallocate(gated_seq)

        # All-reduce — output: [1, 1, seq_len, dim]
        return self._all_reduce(out_partial)
