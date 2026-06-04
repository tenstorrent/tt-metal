# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel (TP>1) Gated DeltaNet decode for Qwen3.5.

The delta-rule recurrence is per-value-head, so TP shards by value heads with NO
cross-device comms inside the recurrence — only an all-reduce after the
row-parallel output projection. The recurrence itself reuses the validated
composed-op `recurrent_gated_delta_rule_decode_ttnn` from the experimental
backend (it does L2-norm + scale + the delta step), driven on each device's
local heads. Projections/conv are sharded; weights kept interleaved (auto matmul).

Sharding mirrors models/demos/qwen35_27b. GDN output uses the *gated* RMSNorm
(weight, NO +1) followed by a SiLU(z) gate — distinct from the +1 QK/layer norms.
"""
import torch
from tt.ttnn_delta_rule_ops import recurrent_gated_delta_rule_decode_ttnn
from tt.ttnn_delta_rule_seq import chunk_gated_delta_rule_seq_adapter
from tt.ttnn_gated_deltanet import _causal_conv1d_fir

import models.demos.blackhole.qwen3_5_9b.tt.gdn._experimental_path  # noqa: F401  (puts experimental backend on sys.path)
import ttnn
from models.demos.blackhole.qwen3_5_9b.tt import tp_common as tpc
from models.tt_transformers.tt.ccl import tt_all_reduce


def load_gdn_weights_tp(mesh, sd, args, cache_dir=None):
    """Shard one GDN layer's RAW linear_attn.* weights across the mesh.

    sd keys (raw, from FP8 loader): linear_attn.in_proj_qkv/in_proj_z/in_proj_a/
    in_proj_b/out_proj/conv1d/A_log/dt_bias/norm.weight.
    """
    tp = args.num_devices
    nk, dk, nv, dv = args.gdn_nk, args.gdn_dk, args.gdn_nv, args.gdn_dv
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    qkv_per = args.gdn_qkv_dim_tp
    z_per = args.gdn_z_dim_tp
    nv_per = args.gdn_nv_tp

    if cache_dir is not None:
        import os

        os.makedirs(cache_dir, exist_ok=True)

    def c(n):
        return str(cache_dir / n) if cache_dir is not None else None

    # Keys may arrive raw (`linear_attn.in_proj_qkv.weight`) or already stripped
    # by substate() in the integrated model path (`in_proj_qkv.weight`).
    P = "linear_attn." if ("linear_attn.in_proj_qkv.weight" in sd) else ""
    # ---- fused QKV+Z (column-parallel) ----
    qkv_re = tpc.prepare_gdn_qkv(sd[P + "in_proj_qkv.weight"], key_dim, value_dim, nk, dk, nv, dv, tp)
    z_w = sd[P + "in_proj_z.weight"]
    fused = torch.cat(
        [
            torch.cat([qkv_re[d * qkv_per : (d + 1) * qkv_per], z_w[d * z_per : (d + 1) * z_per]], dim=0)
            for d in range(tp)
        ],
        dim=0,
    )
    tw = {}
    tw["qkvz"] = tpc.shard_w(
        fused, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=c("qkvz"), dtype=ttnn.bfloat8_b
    )
    # ---- fused A+B (column-parallel), per-device [a(nv_per), b(nv_per)] ----
    a_w, b_w = sd[P + "in_proj_a.weight"], sd[P + "in_proj_b.weight"]
    ab = torch.cat(
        [torch.cat([a_w[d * nv_per : (d + 1) * nv_per], b_w[d * nv_per : (d + 1) * nv_per]], dim=0) for d in range(tp)],
        dim=0,
    )
    tw["ab"] = tpc.shard_w(
        ab, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=c("ab"), dtype=ttnn.bfloat8_b
    )
    # ---- out projection (row-parallel) ----
    tw["out"] = tpc.shard_w(
        sd[P + "out_proj.weight"],
        mesh,
        dim=0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("out"),
        dtype=ttnn.bfloat8_b,
    )
    # ---- per-head params ----
    tw["dt_bias"] = tpc.shard_small(sd[P + "dt_bias"].float(), mesh, c("dt_bias"))
    A_log = tpc.shard_small(sd[P + "A_log"].float(), mesh, c("A_log"))
    tw["neg_exp_A"] = ttnn.neg(ttnn.exp(A_log))
    tw["norm_w"] = tpc.replicate(sd[P + "norm.weight"].float(), mesh, c("norm_w"))
    # ---- conv taps (4), sharded per Q/K/V head grouping ----
    taps = tpc.prepare_conv_taps(sd[P + "conv1d.weight"], key_dim, nk, dk, nv, dv, args.gdn_conv_kernel_size, tp)
    tw["conv_taps"] = [tpc.shard_small(taps[j], mesh, c(f"tap{j}")) for j in range(args.gdn_conv_kernel_size)]
    return tw


class TPGatedDeltaNet:
    """Standalone TP GDN decode (per-device value-head recurrence + all-reduce)."""

    def __init__(self, mesh, args, tw, tt_ccl):
        self.mesh = mesh
        self.args = args
        self.tw = tw
        self.tt_ccl = tt_ccl
        self.B = args.max_batch_size
        self.Nk = args.gdn_nk_tp
        self.Nv = args.gdn_nv_tp
        self.Dk = args.gdn_dk
        self.Dv = args.gdn_dv
        self.qkv_dim_tp = args.gdn_qkv_dim_tp
        self.qkvz_dim_tp = args.gdn_qkvz_dim_tp
        self.key_dim_tp = args.gdn_key_dim_tp
        self.value_dim_tp = args.gdn_value_dim_tp
        self.K = args.gdn_conv_kernel_size
        self.scale = self.Dk**-0.5
        self.cfg = tpc.COMPUTE_HIFI2
        self.conv_states = None
        self.rec_state = None

    def reset_state(self):
        def z(shape):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        self.conv_states = [z((1, self.B, self.qkv_dim_tp)) for _ in range(self.K)]
        self.rec_state = z((self.B, self.Nv, self.Dk, self.Dv))

    def forward_prefill(self, x, chunk_size=128, valid_len=None, capture_state=False):
        """Causal chunk-prefill over a sequence (from scratch, zero init state).

        x: [1,1,T,dim] replicated. Reuses the shared gated_delta_attn_seq kernel
        (via chunk_gated_delta_rule_seq_adapter) and the FIR conv on per-device
        value heads. Output fractured along dim=3.

        valid_len: number of REAL tokens (< T means the rest is right-padding);
        the recurrence + conv state are masked to reflect exactly valid_len tokens.
        capture_state: when True, store the final recurrent state + the last K-1
        real conv inputs into self.rec_state / self.conv_states so decode continues.
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))  # [1,T,dim]
        T = x.shape[1]
        vlen = valid_len or T

        qkvz = ttnn.linear(x, tw["qkvz"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, T, self.qkv_dim_tp))
        z = ttnn.slice(qkvz, (0, 0, self.qkv_dim_tp), (1, T, self.qkvz_dim_tp))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, tw["ab"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (1, T, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, T, 2 * Nv))
        ttnn.deallocate(ab)

        # FIR causal conv1d + SiLU over the sequence (zero init conv state).
        # valid_len makes new_state capture the last K-1 REAL tokens (not padding).
        conv, conv_new_state = _causal_conv1d_fir(
            qkv,
            None,
            None,
            self.K,
            self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            conv_state=None,
            weight_taps=tw["conv_taps"],
            bias_dev=None,
            valid_len=vlen,
        )
        ttnn.deallocate(qkv)

        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd)), (1, T, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp)), (1, T, Nv, Dv))
        ttnn.deallocate(conv)
        rf = Nv // Nk
        q = ttnn.repeat_interleave(q, rf, dim=2)
        k = ttnn.repeat_interleave(k, rf, dim=2)

        beta = ttnn.reshape(ttnn.sigmoid(b), (1, T, Nv))
        ttnn.deallocate(b)
        g = ttnn.reshape(ttnn.multiply(tw["neg_exp_A"], ttnn.softplus(ttnn.add(a, tw["dt_bias"]))), (1, T, Nv))
        ttnn.deallocate(a)

        o, final_state = chunk_gated_delta_rule_seq_adapter(
            q,
            k,
            v,
            beta,
            g,
            chunk_size=chunk_size,
            scale=self.scale,
            initial_state=None,
            device=self.mesh,
            valid_len=vlen,
        )
        if capture_state:
            # Carry recurrent + conv state into decode. conv_states[1..K-1] = last
            # K-1 real conv inputs (conv_new_state [B,K-1,D]); [0] stays zero.
            self.rec_state = final_state
            B, D = 1, self.qkv_dim_tp
            if self.conv_states is None:
                self.reset_state()
            zero = ttnn.from_torch(
                torch.zeros(1, B, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            ttnn.copy(zero, self.conv_states[0])
            for j in range(self.K - 1):
                src = ttnn.reshape(ttnn.slice(conv_new_state, (0, j, 0), (1, j + 1, D)), (1, B, D))
                ttnn.copy(src, self.conv_states[j + 1])
            ttnn.deallocate(zero)
        # o: [1, T, Nv, Dv] -> gated RMSNorm over Dv + SiLU(z) gate
        out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6)
        ttnn.deallocate(o)
        out_f = ttnn.reshape(out_n, (1, T, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = ttnn.multiply(out_f, ttnn.silu(z))
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        partial = ttnn.linear(gated, tw["out"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_decode(self, x):
        tw, B, Nk, Nv, Dk, Dv = self.tw, self.B, self.Nk, self.Nv, self.Dk, self.Dv
        if self.conv_states is None:
            self.reset_state()
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))  # [1,B,dim]

        qkvz = ttnn.linear(x, tw["qkvz"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, B, self.qkv_dim_tp))
        z = ttnn.slice(qkvz, (0, 0, self.qkv_dim_tp), (1, B, self.qkvz_dim_tp))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, tw["ab"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (1, B, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, B, 2 * Nv))
        ttnn.deallocate(ab)

        # conv1d shift-register (copy(src, dst): dst is 2nd arg) + weighted sum + silu
        st = self.conv_states
        for j in range(self.K - 1):
            ttnn.copy(st[j + 1], st[j])
        ttnn.copy(qkv, st[self.K - 1])
        ttnn.deallocate(qkv)
        conv = ttnn.multiply(st[0], tw["conv_taps"][0])
        for j in range(1, self.K):
            conv = ttnn.mac(st[j], tw["conv_taps"][j], conv)
        conv = ttnn.silu(conv)

        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, B, 2 * kd)), (B, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, Nv, Dv))
        ttnn.deallocate(conv)

        # expand Q/K from Nk to Nv heads (GQA); recurrence fn L2-norms + scales internally
        rf = Nv // Nk
        q = ttnn.repeat_interleave(q, rf, dim=1)
        k = ttnn.repeat_interleave(k, rf, dim=1)
        q = ttnn.reshape(q, (B, 1, Nv, Dk))
        k = ttnn.reshape(k, (B, 1, Nv, Dk))
        v = ttnn.reshape(v, (B, 1, Nv, Dv))

        beta = ttnn.reshape(ttnn.sigmoid(b), (B, 1, Nv))
        ttnn.deallocate(b)
        g = ttnn.multiply(tw["neg_exp_A"], ttnn.softplus(ttnn.add(a, tw["dt_bias"])))
        ttnn.deallocate(a)
        g = ttnn.reshape(g, (B, 1, Nv))

        o, self.rec_state = recurrent_gated_delta_rule_decode_ttnn(
            q, k, v, beta, g, scale=self.scale, initial_state=self.rec_state, device=self.mesh
        )

        out_r = ttnn.reshape(o, (B, Nv, Dv))
        out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6)  # gated norm: weight only (no +1)
        ttnn.deallocate(out_r)
        out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = ttnn.multiply(out_f, ttnn.silu(z))
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)

        partial = ttnn.linear(gated, tw["out"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, B, partial.shape[-1]))
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
