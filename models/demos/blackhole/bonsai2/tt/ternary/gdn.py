# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bonsai's Gated DeltaNet projections in their own ternary values for one-token decode.

in_proj: qkv (10240) and z (6144) in one ternary decode linear (512 output tile columns over 130 cores) whose writer
splits them into two tensors; a and b (48 each, BF16 in the GGUF, not ternary) stay dense. out_proj: ternary on the
rotated activation. The GGUF orders the v heads grouped by key head ("gdn_v_grouped"), HF by value head: the qkv v
rows and the z rows are reordered on the host, as Alejandro's converter does. out_proj needs no reorder: the
converter kept ssm_out as is (its column-reorder candidate scored lower against the base model; conversion report)."""

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.gdn_step.gdn_step import GDNStep
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import TernaryDecodeLinear3, pack_records_inline
from models.demos.blackhole.bonsai2.tt.ternary.rotate import DecodeRotation


def v_to_hf(nv: int, nk: int, unit: int) -> np.ndarray:
    """Index g with hf = gguf[g] for the v-head axis (convert_gguf_to_hf.py)."""
    return np.arange(nv * unit).reshape(nv // nk, nk, unit).transpose(1, 0, 2).reshape(-1)


def gdn_dims(gguf):
    kv = gguf.kv
    nv, nk = int(kv["qwen35.ssm.time_step_rank"]), int(kv["qwen35.ssm.group_count"])
    hd, hk = int(kv["qwen35.ssm.inner_size"]) // nv, int(kv["qwen35.ssm.state_size"])
    return nv, nk, hd, hk


def pack_gdn(gguf, layer_num):
    """Host records for one GDN layer: {'qkvz': (rec, ppr, K, N, split), 'out': (rec, ppr, K, N)}."""
    nv, nk, hd, hk = gdn_dims(gguf)
    qk = 2 * nk * hk
    t_qkv, s_qkv = gguf.ternary(f"blk.{layer_num}.attn_qkv.weight")
    t_z, s_z = gguf.ternary(f"blk.{layer_num}.attn_gate.weight")
    g_qkv = np.concatenate([np.arange(qk), qk + v_to_hf(nv, nk, hd)])
    g_z = v_to_hf(nv, nk, hd)
    t = np.concatenate([t_qkv[:, g_qkv], t_z[:, g_z]], axis=1)
    s = np.concatenate([s_qkv[g_qkv], s_z[g_z]], axis=0)
    rec, ppr = pack_records_inline(np.ascontiguousarray(t), np.ascontiguousarray(s))
    t_o, s_o = gguf.ternary(f"blk.{layer_num}.ssm_out.weight")
    rec_o, ppr_o = pack_records_inline(t_o, s_o)
    return {
        "qkvz": (rec, ppr, t.shape[0], t.shape[1], t_qkv.shape[1]),
        "out": (rec_o, ppr_o, t_o.shape[0], t_o.shape[1]),
    }


def out_chunk_src(gguf) -> np.ndarray:
    """Rotated position chunk c (gguf column order) reads input tile chunk_src[c] (HF order): x_gguf[p[j]] = x_hf[j]."""
    nv, nk, hd, _ = gdn_dims(gguf)
    p = v_to_hf(nv, nk, hd)  # W_hf[:, j] = W_gguf[:, p[j]]
    q = np.empty_like(p)
    q[p] = np.arange(p.size)  # x_gguf[i] = x_hf[q[i]]
    chunks = q.reshape(-1, 32)
    assert np.all(chunks[:, 0] % 32 == 0) and np.all(np.diff(chunks, axis=1) == 1), "heads must move in whole chunks"
    return chunks[:, 0] // 32


class TernaryGDN:
    def __init__(self, device, packed, rot_in: DecodeRotation, rot_out: DecodeRotation, ab_w, step=None):
        rec, ppr, K, N, split = packed["qkvz"]
        self.qkvz = TernaryDecodeLinear3.from_packed(device, rec, ppr, K, N)
        self.split = split
        rec, ppr, K, N = packed["out"]
        self.out = TernaryDecodeLinear3.from_packed(device, rec, ppr, K, N)
        self.rot_in, self.rot_out = rot_in, rot_out
        self.step = step  # GDNStep (tt/gdn_step): conv + gates + state update + gated norm in one op
        # a and b (48 each, dense) in one 5120 x 96 matmul: the default config ran each on 2 cores at 113 us;
        # 3 cores with in0_block_w=8 take 21 us (tests/test_ab_bench.py).
        self.ab_w = ab_w
        self.na = ab_w.shape[-1] // 2
        self.ab_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(ab_w.shape[-1] // 32, 1),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def in_proj(self, x, compute_kernel_config=None):
        """x [1, 1, 5120] -> (qkv [1, 1, 10240], a [1, 1, 48], b [1, 1, 48], z [1, 1, 6144])."""
        lead = tuple(list(x.shape)[:-1])
        xr = self.rot_in(x)
        qkv, z = self.qkvz(xr, lead=lead, split=self.split)
        ttnn.deallocate(xr)
        ab = ttnn.linear(x, self.ab_w, compute_kernel_config=compute_kernel_config, program_config=self.ab_pc)
        a = ttnn.slice(ab, [0] * (len(lead)) + [0], list(lead) + [self.na])
        b = ttnn.slice(ab, [0] * (len(lead)) + [self.na], list(lead) + [2 * self.na])
        ttnn.deallocate(ab)
        return qkv, a, b, z

    def decode(self, x, gdn, compute_kernel_config=None):
        """The whole one-token GDN layer: in_proj, the fused step (conv, gates, state update, gated norm; updates
        gdn's conv and recurrent state in place), out_proj. x [1, 1, 5120] -> [1, 1, 5120]."""
        lead = tuple(list(x.shape)[:-1])
        xr = self.rot_in(x)
        qkv, z = self.qkvz(xr, lead=lead, split=self.split)
        ttnn.deallocate(xr)
        ab = ttnn.linear(x, self.ab_w, compute_kernel_config=compute_kernel_config, program_config=self.ab_pc)
        o = self.step(qkv, ab, z, gdn.fused_conv_state, gdn.recurrent_state)
        for t in (qkv, z, ab):
            ttnn.deallocate(t)
        y = self.out_proj(o)
        ttnn.deallocate(o)
        return y

    def out_proj(self, o):
        lead = tuple(list(o.shape)[:-1])
        orr = self.rot_out(o)
        y = self.out(orr, lead=lead)
        ttnn.deallocate(orr)
        return y


class MegaSplit:
    """Stands in for the fused [qkv | a | b | z] in_proj output: gated_deltanet_forward_ttnn slices it into the four
    parts, which are already separate tensors here (no concat, no slice)."""

    def __init__(self, parts, dims):
        self.parts, self.starts = parts, np.cumsum([0] + list(dims[:-1])).tolist()

    def __getitem__(self, key):
        start = key[-1].start or 0
        return self.parts[self.starts.index(start)]


def make_gdn_step(gdn, nk, hk) -> GDNStep:
    """The layer's fused one-token step, from the weights the dense path already holds on the device."""
    w = gdn.weights
    assert w.fused_conv_bias_dev is None, "GDN conv bias is not supported by the fused step"
    taps = torch.cat([ttnn.to_torch(t).float().reshape(1, -1) for t in w.fused_conv_weight_taps], 0)
    nv = gdn.num_v_heads
    dt_bias = ttnn.to_torch(w.dt_bias).float().reshape(-1)[:nv]
    a_neg = ttnn.to_torch(w.A_neg).float().reshape(-1)[:nv]
    norm_w = ttnn.to_torch(w.o_norm_weight).float().reshape(-1)[: gdn.head_v_dim]
    return GDNStep(gdn.device, taps, dt_bias, a_neg, norm_w, nk, hk, gdn.head_v_dim, eps_o=gdn.norm_eps)


def swap_in_ternary_gdn(model, gguf, cache_dir=None) -> None:
    """Free each GDN layer's dense in_proj/out_proj weights and give it a TernaryGDN (gdn/decode.py uses it)."""
    from pathlib import Path

    nv, nk, hd, hk = gdn_dims(gguf)
    device = model.mesh_device
    rot_in = DecodeRotation(device, gguf.signs[5120])
    rot_out = DecodeRotation(device, gguf.signs[nv * hd])
    for layer in model.layers:
        if layer.is_full_attention:
            continue
        gdn = layer.attention
        w = gdn.weights
        f = Path(cache_dir) / f"blk.{layer.layer_num}.gdn.decode_inline.npz" if cache_dir else None
        if f is not None and f.exists():
            z = np.load(f)
            packed = {
                "qkvz": (z["rq"], int(z["pq"]), int(z["kq"]), int(z["nq"]), int(z["sq"])),
                "out": (z["ro"], int(z["po"]), int(z["ko"]), int(z["no"])),
            }
        else:
            packed = pack_gdn(gguf, layer.layer_num)
            if f is not None:
                q, o = packed["qkvz"], packed["out"]
                np.savez(f, rq=q[0], pq=q[1], kq=q[2], nq=q[3], sq=q[4], ro=o[0], po=o[1], ko=o[2], no=o[3])
        for t in (w.mega_fused_weight, w.o_proj_weight, w.qkv_proj_weight, w.g_proj_weight):
            if t is not None:
                ttnn.deallocate(t)
        gdn.ternary = TernaryGDN(device, packed, rot_in, rot_out, w.ab_proj_weight, step=make_gdn_step(gdn, nk, hk))
