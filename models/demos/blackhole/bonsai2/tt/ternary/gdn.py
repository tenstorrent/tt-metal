# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bonsai's Gated DeltaNet projections in their own ternary values for one-token decode.

in_proj: qkv (10240) and z (6144) in one ternary decode linear (512 output tile columns over 130 cores) whose writer
splits them into two tensors; a and b (48 each, BF16 in the GGUF, not ternary) stay dense, in their exact bf16
values (exact_ab; qwen36 keeps them as bfloat8_b). out_proj: ternary on the
rotated activation. The GGUF orders the v heads grouped by key head ("gdn_v_grouped"), HF by value head: the qkv v
rows and the z rows are reordered on the host, as Alejandro's converter does. out_proj needs no reorder: the
converter kept ssm_out as is (its column-reorder candidate scored lower against the base model; conversion report)."""

import os

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.gdn_step.gdn_step import GDNStep
from models.demos.blackhole.bonsai2.tt.ternary import prerot
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


def exact_ab(gguf, layer_num) -> torch.Tensor:
    """[5120, 96] bf16: [a | b] = ssm_alpha, ssm_beta in their GGUF bf16 values, v heads in HF order (the HF in_proj_a /
    in_proj_b are exactly gguf[v_to_hf]; neither is Hadamard-folded)."""
    nv, nk, _, _ = gdn_dims(gguf)
    p = v_to_hf(nv, nk, 1)
    parts = []
    for name in ("ssm_alpha", "ssm_beta"):
        dims, tt, raw = gguf.tensors[f"blk.{layer_num}.{name}.weight"]
        assert tt == 30 and f"blk.{layer_num}.{name}.weight" not in gguf.folded, (name, tt)
        w = torch.from_numpy(np.asarray(raw).view(np.int16).copy()).view(torch.bfloat16)
        parts.append(w.reshape(int(dims[1]), int(dims[0]))[p].T)
    return torch.cat(parts, dim=1).contiguous()


def ab_prefill_config(T):
    """a / b projection at T tokens: rows over an 8 x 8 grid, the weights multicast (bf16, HiFi4: T = 2048 142 ->
    100 us vs the default config)."""
    mt = -(-T // 32)
    pm = -(-mt // 64)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=3,
        out_block_h=pm,
        out_block_w=3,
        per_core_M=pm,
        per_core_N=3,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )


class TernaryGDN:
    def __init__(
        self, device, packed, rot_in: DecodeRotation, rot_out: DecodeRotation, ab_w, step=None, prefill_rotations=None
    ):
        rec, ppr, K, N, split = packed["qkvz"]
        self.qkvz = TernaryDecodeLinear3.from_packed(device, rec, ppr, K, N)
        self.split = split
        rec, ppr, K, N = packed["out"]
        self.out = TernaryDecodeLinear3.from_packed(device, rec, ppr, K, N)
        self.rot_in, self.rot_out = rot_in, rot_out
        self.step = step  # GDNStep (tt/gdn_step): conv + gates + state update + gated norm in one op
        # a and b (48 each, dense) in one 5120 x 96 matmul: the default config ran each on 2 cores at 113 us;
        # 3 cores with in0_block_w=40 take 16 us (in0_block_w=8: 21 us).
        self.ab_w = ab_w
        self.na = ab_w.shape[-1] // 2
        # a and b are bf16 weights (not ternary): HiFi4 uses all of their mantissa
        self.ab_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.pre_qkvz = self.pre_out = None
        if prefill_rotations is not None:
            # exact ternary prefill on the same records (bonsai2/tt/tprefill, gdn_prefill/forward.py)
            from models.demos.blackhole.bonsai2.tt.tprefill.linear import prefill_linear

            self.pre_qkvz = prefill_linear(self.qkvz, packed["qkvz"][0])
            self.pre_out = prefill_linear(self.out, packed["out"][0])
            self.prot_in, self.prot_out = prefill_rotations[self.qkvz.K], prefill_rotations[self.out.K]
        self.ab_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(ab_w.shape[-1] // 32, 1),
            in0_block_w=40,
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
        ab = ttnn.linear(x, self.ab_w, compute_kernel_config=self.ab_ckc, program_config=self.ab_pc)
        a = ttnn.slice(ab, [0] * (len(lead)) + [0], list(lead) + [self.na])
        b = ttnn.slice(ab, [0] * (len(lead)) + [self.na], list(lead) + [2 * self.na])
        ttnn.deallocate(ab)
        return qkv, a, b, z

    def takes_norm(self):
        """Whether decode() can run the layer's norm + rotation inside its qkvz op (prerot.put_norm): only when the
        step computes a / b from the tiled blocks, since that form never makes the normalised x (BONSAI_GDN_NORM_FUSE=0:
        off)."""
        return (
            getattr(self, "wab", None) is not None
            and os.environ.get("BONSAI_STEP_AB", "1") == "1"
            and os.environ.get("BONSAI_GDN_NORM_FUSE", "1") == "1"
        )

    def decode(self, x, gdn, compute_kernel_config=None):
        """The whole one-token GDN layer: in_proj, the fused step (conv, gates, state update, gated norm; updates
        gdn's conv and recurrent state in place), out_proj. x [1, 1, 5120] -> [1, 1, 5120]."""
        lead = tuple(list(x.shape)[:-1])
        nrot = prerot.take_norm(x)
        if nrot is not None:
            # the layer's norm + rotation inside the qkvz op (x is the un-normalised residual stream); it also writes
            # the normalised blocks tiled (xnt) for the step's a / b
            xnt = ttnn.allocate_tensor_on_device(
                ttnn.Shape([nrot.B * 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, x.device(), ttnn.L1_MEMORY_CONFIG
            )
            from models.demos.blackhole.bonsai2.tt.stream import decode as stream_decode

            # weight stream: qkv / z in L1 (the step reads them; DRAM reads queue behind the stream)
            qkv, z = self.qkvz.rotated(nrot, x, lead=lead, u=xnt, split=self.split, out_l1=stream_decode.active())
        else:
            xr = prerot.take(x)
            if xr is None:
                xr = self.rot_in(x)
            qkv, z = self.qkvz(xr, lead=lead, split=self.split)
            ttnn.deallocate(xr)
            xnt = prerot.take_xnt(x) if getattr(self, "wab", None) is not None else None
        stage, extra = self._stage_out(x.device())
        from models.demos.blackhole.bonsai2.tt.stream import decode as stream_decode

        if xnt is not None and os.environ.get("BONSAI_STEP_AB", "1") == "1":
            # a / b inside the step: dot products of the norm op's tiled blocks with this head's weight blocks; under
            # the weight stream its small inputs come prefetched in L1 (stream/decode.py step_pf)
            pf = stream_decode.STATE.step_pf(self) if stream_decode.active() else None
            o = self.step(
                qkv, None, z, gdn.fused_conv_state, gdn.recurrent_state, xnt=xnt, wab=self.wab, extra=extra, pf=pf
            )
            if pf is not None:
                stream_decode.STATE.after_step(self)
            for t in (qkv, z):
                ttnn.deallocate(t)
        else:
            ab = ttnn.linear(x, self.ab_w, compute_kernel_config=self.ab_ckc, program_config=self.ab_pc)
            o = self.step(qkv, ab, z, gdn.fused_conv_state, gdn.recurrent_state, extra=extra)
            for t in (qkv, z, ab):
                ttnn.deallocate(t)
        y = self.out_proj(o, stage=stage)
        ttnn.deallocate(o)
        if stage is not None:
            ttnn.deallocate(stage)
        return y

    def _stage_out(self, device):
        """(stage, extra): the out projection's weights copied into L1 during the step, on the cores it leaves free
        (the step moves to the last cores), so the out projection reads no DRAM. Off by default (BONSAI_STAGE_OUT=1:
        on): the staging reads delay the step's own (step + out projection 39.5 + 16.8 us vs 30.8 + 24, eager)."""
        if os.environ.get("BONSAI_STAGE_OUT", "0") != "1" or os.environ.get("BONSAI_ROT_FUSE", "1") != "1":
            self.step.first_core = 0
            return None, None
        grid = device.compute_with_storage_grid_size()
        cores, _ = self.out._cores(device)
        if cores + self.step.nv > grid.x * grid.y:
            self.step.first_core = 0
            return None, None
        self.step.first_core = grid.x * grid.y - self.step.nv
        stage = self.out.alloc_stage(device)
        _, kernels = self.out.stage_kernels(device, stage)
        return stage, (kernels, [self.out.w, stage])

    def out_proj(self, o, stage=None):
        lead = tuple(list(o.shape)[:-1])
        if os.environ.get("BONSAI_ROT_FUSE", "1") == "1":  # the rotation inside the out projection's op
            return self.out.rotated(
                self.rot_out, o, lead=lead, residual=prerot.take_residual(), out_l1=True, stage=stage
            )
        orr = self.rot_out(o)
        y = self.out(orr, lead=lead, residual=prerot.take_residual())  # the layer's residual add, if registered
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


def swap_in_ternary_gdn(model, gguf, cache_dir=None, prefill_rotations=None) -> None:
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
        # a / b in their exact bf16 values (the dense path's copy is bfloat8_b)
        ab_kn = exact_ab(gguf, layer.layer_num)
        ab = ttnn.from_torch(
            ab_kn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # fmt: skip
        ttnn.deallocate(w.ab_proj_weight)
        w.ab_proj_weight = ab
        gdn.ternary = TernaryGDN(
            device, packed, rot_in, rot_out, w.ab_proj_weight, step=make_gdn_step(gdn, nk, hk),
            prefill_rotations=prefill_rotations,
        )  # fmt: skip
        # per-head [a | b] weight blocks for the step's own a / b (gdn_step.pack_ab_heads)
        from models.demos.blackhole.bonsai2.tt.gdn_step.gdn_step import pack_ab_heads

        gdn.ternary.wab = ttnn.from_torch(
            pack_ab_heads(ab_kn), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # fmt: skip
