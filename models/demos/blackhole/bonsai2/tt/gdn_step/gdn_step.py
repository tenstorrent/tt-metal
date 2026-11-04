# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-token GatedDeltaNet step in one op (kernels/step_*.cpp): conv1d + SiLU, q/k L2 norm, decay/beta gates, the
delta-rule state update and readout, and the gated output RMSNorm. One core per value head (48 cores).

In: qkv [.., 10240], z [.., 6144], ab [.., 2 * nv] (the in_proj outputs, bf16 TILE, row 0 = the token), the layer's
conv state [1, 3, 10240] and recurrent state [1, nv, 128, 128] (bf16 TILE, updated in place). Out: [.., 6144] bf16
TILE, the input of out_proj."""

import struct

import torch

import ttnn

KERNELS = "models/demos/blackhole/bonsai2/tt/gdn_step/kernels/"
TILE = 32


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class GDNStep:
    def __init__(self, device, taps, dt_bias, a_neg, norm_w, nk, hk, hd, eps_o, eps_qk=1e-6):
        """taps [4, qk + qk + nv * hd] (tap 3 multiplies the new token), dt_bias / a_neg [nv], norm_w [hd]."""
        taps = torch.as_tensor(taps, dtype=torch.float32)
        self.nv = int(dt_bias.shape[0])
        self.nk, self.hk, self.hd = nk, hk, hd
        assert hk == hd == 128 and self.nv % nk == 0, (hk, hd, self.nv, nk)
        self.group = self.nv // nk
        qk = nk * hk
        ht = hd // TILE
        nch = 3 * ht
        tiles = []
        for j in range(self.nv):
            kh = j // self.group
            for c in range(nch):
                if c < ht:
                    ch = kh * hk + TILE * c
                elif c < 2 * ht:
                    ch = qk + kh * hk + TILE * (c - ht)
                else:
                    ch = 2 * qk + j * hd + TILE * (c - 2 * ht)
                t = torch.zeros(TILE, TILE)
                t[:4] = taps[:, ch : ch + TILE]
                tiles.append(t)
        for i in range(ht):
            t = torch.zeros(TILE, TILE)
            t[0] = torch.as_tensor(norm_w, dtype=torch.float32).reshape(-1)[TILE * i : TILE * i + TILE]
            tiles.append(t)
        e = torch.zeros(TILE, TILE)
        e[0, :4] = 1.0
        tiles += [e, torch.ones(TILE, TILE)]
        self.consts = ttnn.from_torch(
            torch.cat(tiles, 0).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.dt_bias = [float(v) for v in torch.as_tensor(dt_bias, dtype=torch.float32).reshape(-1)]
        self.a_neg = [float(v) for v in torch.as_tensor(a_neg, dtype=torch.float32).reshape(-1)]
        self.consts_scalar = [_f32_bits(eps_qk), _f32_bits(hk**-0.5), _f32_bits(1.0 / hd), _f32_bits(eps_o)]
        self.qkv_tiles, self.ht, self.nch = qk // TILE, ht, nch
        self.device = device
        self.compute_defines = ()  # e.g. (("STEP_ZONES", "1"),): profiler zones per phase

    def __call__(self, qkv, ab, z, conv_state, rec_state):
        device, nv, ht = self.device, self.nv, self.ht
        lead = list(z.shape)[:-1]
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(lead + [nv * self.hd]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        grid = device.compute_with_storage_grid_size()
        coords = [ttnn.CoreCoord(j % grid.x, j // grid.x) for j in range(nv)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for j, c in enumerate(coords):
            lj = self.group * (j // self.group)
            lead_core = device.worker_core_from_logical_core(coords[lj])
            rd[c.x][c.y] = [
                qkv.buffer_address(),
                z.buffer_address(),
                ab.buffer_address(),
                conv_state.buffer_address(),
                rec_state.buffer_address(),
                self.consts.buffer_address(),
                j,
                lead_core.x,
                lead_core.y,
                1 if j == lj else 0,
            ]
            wr[c.x][c.y] = [rec_state.buffer_address(), out.buffer_address(), j]
            cp[c.x][c.y] = [_f32_bits(self.dt_bias[j]), _f32_bits(self.a_neg[j])] + self.consts_scalar
        rd_ct = [nv, self.qkv_tiles, ht, self.group]
        for t in (qkv, z, ab, conv_state, rec_state, self.consts):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = [ht] + ttnn.TensorAccessorArgs(rec_state).get_compile_time_args()
        wr_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()

        def cb(index, fmt, pages):
            page = 4096 if fmt == ttnn.float32 else 2048
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        bf, f32, nch = ttnn.bfloat16, ttnn.float32, self.nch
        cbs = [
            cb(0, bf, nch),  # X: conv inputs
            cb(1, bf, nch),  # taps
            cb(2, bf, ht),  # z
            cb(3, bf, 1),  # a (row 0 = a_j)
            cb(4, bf, 1),  # b
            cb(5, bf, ht * ht),  # S
            cb(6, bf, ht),  # norm weight
            cb(7, bf, 1),  # E
            cb(8, bf, 1),  # ones
            cb(9, bf, 2),  # reader scratch
            cb(10, bf, 2),  # new conv state rows
            cb(11, f32, nch),  # X * taps
            cb(12, f32, nch),  # conv out
            cb(13, f32, 2 * ht),  # squares
            cb(14, f32, 2),  # q, k inverse norms
            cb(15, f32, 2 * ht),  # q, k normalized
            cb(16, bf, ht),  # out
            cb(17, f32, 1),  # decay
            cb(18, f32, 1),  # beta
            cb(19, f32, ht),  # k columns
            cb(20, f32, ht),  # u
            cb(21, bf, ht * ht),  # S'
            cb(22, f32, ht),  # o
            cb(23, f32, ht),  # silu(z)
            cb(24, f32, 1),  # o inverse norm
        ]
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "step_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    runtime_args=rd,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "step_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=wr_ct,
                    runtime_args=wr,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "step_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[],
                    runtime_args=cp,
                    defines=list(self.compute_defines),
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=True,
                        math_approx_mode=False,
                    ),
                ),
            ],
            semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0)],
            cbs=cbs,
        )
        return ttnn.generic_op([qkv, z, ab, conv_state, rec_state, self.consts, out], program)


def reference_step(qkv, ab, z, conv, S, taps, dt_bias, a_neg, norm_w, nk, hk, hd, eps_o, eps_qk=1e-6):
    """fp32 torch reference: qkv [Dqkv], ab [2 nv], z [nv hd], conv [3, Dqkv], S [nv, hk, hd] -> (out, S', conv')."""
    nv = dt_bias.shape[0]
    qk = nk * hk
    xs = torch.cat([conv, qkv[None]], 0)
    c = torch.nn.functional.silu((xs * taps).sum(0))
    q = c[:qk].view(nk, hk)
    k = c[qk : 2 * qk].view(nk, hk)
    v = c[2 * qk :].view(nv, hd)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + eps_qk) * hk**-0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + eps_qk)
    q = q.repeat_interleave(nv // nk, 0)
    k = k.repeat_interleave(nv // nk, 0)
    a, b = ab[:nv], ab[nv:]
    decay = torch.exp(a_neg * torch.nn.functional.softplus(a + dt_bias))
    beta = torch.sigmoid(b)
    sd = decay[:, None, None] * S
    vread = torch.einsum("hk,hkv->hv", k, sd)
    u = beta[:, None] * (v - vread)
    s_new = sd + k[:, :, None] * u[:, None, :]
    o = torch.einsum("hk,hkv->hv", q, s_new)
    on = o / torch.sqrt((o * o).mean(-1, keepdim=True) + eps_o) * norm_w * torch.nn.functional.silu(z.view(nv, hd))
    return on.reshape(-1), s_new, xs[1:]
