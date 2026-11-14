# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One GatedDeltaNet layer's chunked prefill (whole 128-token chunks, no padding) with the q/k/v head relayouts
removed: the conv output [1, T, q | k | v] goes straight into the fused chunk preprocessing (chunk_prep.ChunkPrep,
token-major mode), which L2-normalises q and k and maps each value head to its key head itself. The shared path
(gated_deltanet_forward_ttnn + chunk_gated_delta_rule_seq_adapter) untilizes, permutes, repeat-interleaves and
re-tilizes q, k and v first. Everything else follows the shared mega-fused path op for op."""

import os
import struct

import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.core_ranges import row_major_crs
from models.demos.blackhole.bonsai2.tt.gdn_prefill import chunk_prep

C = 128
KERNELS = "models/demos/blackhole/bonsai2/tt/gdn_prefill/kernels/"


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class OutputFinish:
    """Scan output o [H, T, 128] fp32 and z [1, T, H * 128] bf16 -> o / rms(o) * w * silu(z) as [1, T, H * 128] bf16,
    token-major (kernels/fin_*.cpp): replaces the shared path's untilize, permute, re-tilize, gated RMSNorm and
    reshapes. (Its +-1e4 clips are left out: they only act on overflowing values.)"""

    def __init__(self, device, norm_w, eps):
        w = torch.as_tensor(norm_w, dtype=torch.float32).reshape(-1)
        tiles = []
        for d in range(4):
            t = torch.zeros(32, 32)
            t[0] = w[32 * d : 32 * d + 32]
            tiles.append(t)
        tiles.append(torch.ones(32, 32))
        self.consts = ttnn.from_torch(torch.cat(tiles, 0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.eps, self.device = eps, device

    def __call__(self, o_bh, z, T, z_off=0):
        """o_bh: the scan output, [H, T / 128, 128, 128] (tiles in the same order as [H, T, 128]). z: [1, T, Wz] with
        the gate at tile columns z_off .. z_off + 4H (the in_proj output itself: no slice)."""
        device = self.device
        H = list(o_bh.shape)[0]
        Tt, Wt, Wz = T // 32, H * 4, list(z.shape)[2] // 32
        items = H * Tt
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, T, H * 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        grid = device.compute_with_storage_grid_size()
        ncores = min(grid.x * grid.y, items)
        crs = row_major_crs(ncores, grid.x)
        span = [items, ncores, grid.x]
        rd_ct = []
        for t in (o_bh, z, self.consts):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()

        def cbd(i, fmt, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)],
            )

        f32, bf = ttnn.float32, ttnn.bfloat16
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "fin_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    common_runtime_args=[o_bh.buffer_address(), z.buffer_address(), self.consts.buffer_address()]
                    + span
                    + [Tt, Wz, z_off],
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "fin_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    common_runtime_args=[out.buffer_address()] + span + [Tt, Wt],
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "fin_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[],
                    common_runtime_args=span + [_f32_bits(1.0 / 128), _f32_bits(self.eps)],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
                    ),
                ),
            ],
            semaphores=[],
            cbs=[
                cbd(0, f32, 4096, 8),
                cbd(1, bf, 2048, 8),
                cbd(2, f32, 4096, 5),
                cbd(3, f32, 4096, 4),
                cbd(4, f32, 4096, 1),
                cbd(5, f32, 4096, 4),
                cbd(16, bf, 2048, 8),
            ],
        )
        ttnn.generic_op([o_bh, z, self.consts, out], program)
        return out


class ConvPrefill:
    """Causal 4-tap depthwise conv + SiLU over the q | k | v channels of the in_proj output (kernels/conv_*.cpp):
    one op instead of the shared FIR's slice, concat, 4 shifted slices, re-tilizes, multiply/addcmul and SiLU. Reads
    the in_proj output [1, T, W] directly (its first D channels) and updates the layer's conv state in place."""

    def __init__(self, device, taps):
        """taps [4, D] (tap 3 multiplies the current token)."""
        taps = torch.as_tensor(taps, dtype=torch.float32)
        self.D = taps.shape[1]
        rows = torch.zeros(4 * 32, self.D)
        for k in range(4):
            rows[32 * k] = taps[k]
        self.taps = ttnn.from_torch(rows.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        m = [torch.eye(32)]
        for s_ in (1, 2, 3):  # S_s: rows shifted down by s
            t = torch.zeros(32, 32)
            for r in range(s_, 32):
                t[r, r - s_] = 1
            m.append(t)
        for s_ in (1, 2, 3):  # P_s: the previous tile's last s rows
            t = torch.zeros(32, 32)
            for r in range(s_):
                t[r, 32 - s_ + r] = 1
            m.append(t)
        for s_ in (1, 2, 3):  # Q_s: state rows 0-2 (tokens -3, -2, -1)
            t = torch.zeros(32, 32)
            for r in range(s_):
                t[r, 3 - s_ + r] = 1
            m.append(t)
        rr = torch.zeros(32, 32)  # R: new state rows 0-2 = the last tile's rows 29-31
        for j in range(3):
            rr[j, 29 + j] = 1
        m.append(rr)
        self.consts = ttnn.from_torch(
            torch.cat(m, 0).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        # conv2 (BONSAI_CONV2, default on): per channel tile-column one tile of the 4 taps' SFPU lane vectors
        # (tap k, vector v = 2 fc + par at SFPU row 4 k + v; lane (s, i) = column 16 fc + 2 i + par)
        tb = taps.bfloat16().float()
        Dt = self.D // 32
        vt = torch.zeros(Dt, 32, 32)
        for k in range(4):
            for v in range(4):
                fc, par = v >> 1, v & 1
                q = 4 * k + v
                f, kk = q // 8, q % 8
                cols = 16 * fc + 2 * torch.arange(8) + par
                for s in range(4):
                    vt[:, 16 * (f // 2) + 4 * (kk >> 1) + s, 16 * (f % 2) + 2 * torch.arange(8) + (kk & 1)] = tb[
                        k
                    ].reshape(Dt, 32)[:, cols]
        self.tap_vecs = ttnn.from_torch(
            vt.reshape(Dt * 32, 32).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.v2 = os.environ.get("BONSAI_CONV2", "1") == "1"
        self.device = device

    def __call__(self, x, state, split=None):
        """split = (w0, w1) channels (conv2 only): three outputs [1, T, w0], [1, T, w1], [1, T, D - w0 - w1] (q, k, v)
        instead of one [1, T, D]."""
        device, D = self.device, self.D
        T, W = list(x.shape)[1], list(x.shape)[2]
        Dt, Tt = D // 32, T // 32
        v2 = self.v2
        assert split is None or v2
        widths = [D] if split is None else [split[0], split[1], D - split[0] - split[1]]
        outs = [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, T, w]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
            for w in widths
        ]
        out = outs[0]
        ob, oc = (outs[1], outs[2]) if split is not None else (out, out)
        W0, W1 = (widths[0] // 32, widths[1] // 32) if split is not None else (Dt, 0)
        grid = device.compute_with_storage_grid_size()
        # conv2: work units (column, time tile) split evenly; conv: whole columns per core
        ncores = min(grid.x * grid.y, Dt * Tt if v2 else Dt)
        crs = row_major_crs(ncores, grid.x)
        taps = self.tap_vecs if v2 else self.taps
        rd_ct = []
        for t in (x, state, taps, self.consts):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = (
            ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + ttnn.TensorAccessorArgs(state).get_compile_time_args()
        )
        if v2:
            wr_ct += (
                ttnn.TensorAccessorArgs(ob).get_compile_time_args()
                + ttnn.TensorAccessorArgs(oc).get_compile_time_args()
            )

        def cbd(i, fmt, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)],
            )

        bf, f32 = ttnn.bfloat16, ttnn.float32
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + ("conv2_reader.cpp" if v2 else "conv_reader.cpp"),
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    common_runtime_args=[t.buffer_address() for t in (x, state, taps, self.consts)]
                    + [Dt, ncores, grid.x, Tt, W // 32],
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + ("conv2_writer.cpp" if v2 else "conv_writer.cpp"),
                    core_ranges=crs,
                    compile_time_args=wr_ct,
                    common_runtime_args=[out.buffer_address(), state.buffer_address(), Dt, ncores, grid.x, Tt]
                    + ([ob.buffer_address(), oc.buffer_address(), W0, W1] if v2 else []),
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + ("conv2_compute.cpp" if v2 else "conv_compute.cpp"),
                    core_ranges=crs,
                    compile_time_args=[],
                    common_runtime_args=[Dt, ncores, grid.x, Tt],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi2 if v2 else ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=True,
                    ),
                ),
            ],
            semaphores=[],
            cbs=[
                cbd(0, bf, 2048, 8 if v2 else 2),
                cbd(1, bf, 2048, 1),
                cbd(2, bf, 2048, 4),
                cbd(3, bf, 2048, 11),
                cbd(4, f32, 4096, 4),
                cbd(5, f32, 4096, 4),
                cbd(16, bf, 2048, 2),
                cbd(17, bf, 2048, 1),
            ]
            + ([cbd(6, bf, 2048, 2), cbd(7, bf, 2048, 1)] if v2 else []),
        )
        ttnn.generic_op([x, state, taps, self.consts] + outs[1:] + [out], program)  # returns the last: out
        return out if split is None else tuple(outs)


class GateTiles:
    """The a / b projection output ab [1, T, 2H] (H = 48) -> beta_bh [H, T, 1] = sigmoid(b) and
    g_bh [H, T] = A_neg * softplus(a + dt_bias), fp32 TILE, in one op (kernels/abp_*.cpp): replaces two slices, the
    sigmoid / softplus / add / multiply chain (each rounded to bf16) and the untilize / permute / tilize / reshape
    relayouts of both."""

    def __init__(self, device, dt_bias, a_neg):
        dt = torch.as_tensor(dt_bias, dtype=torch.float32).reshape(-1)
        an = torch.as_tensor(a_neg, dtype=torch.float32).reshape(-1)
        H = dt.numel()
        assert H == 48 and an.numel() == 48, (H, an.numel())
        consts = []
        for v in (dt, an):
            for i in range(2):
                t = torch.zeros(32, 32)
                cols = v[32 * i : 32 * i + 32]  # a-tile i: heads 32 i .. (tile 1: 16 heads, then the b columns)
                t[:, : cols.numel()] = cols
                consts.append(t)
        # order DTB0, DTB1, AN0, AN1
        self.consts = ttnn.from_torch(
            torch.cat([consts[0], consts[1], consts[2], consts[3]], 0),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        sel = torch.zeros(32, 32, 32)
        for j in range(32):
            sel[j, j, 0] = 1.0
        self.sel = ttnn.from_torch(
            sel.reshape(32 * 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.H, self.device = H, device

    def __call__(self, ab, T):
        device, H = self.device, self.H
        Mt = T // 32
        assert list(ab.shape)[-1] == 2 * H and list(ab.shape)[-2] == T, ab.shape

        def alloc(shape):
            return ttnn.allocate_tensor_on_device(
                ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )

        g_bh, beta_bh = alloc([H, T]), alloc([H, T, 1])
        grid = device.compute_with_storage_grid_size()
        assert Mt <= grid.x * grid.y, Mt
        coords = [ttnn.CoreCoord(c % grid.x, c // grid.x) for c in range(Mt)]
        crs = row_major_crs(len(coords), grid.x)
        rt = ttnn.RuntimeArgs()
        for i, c in enumerate(coords):
            rt[c.x][c.y] = [i]

        def cbd(i, fmt, pages):
            page = 4096 if fmt == ttnn.float32 else 2048
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)],
            )

        f32, bf = ttnn.float32, ttnn.bfloat16
        rd_ct = []
        for t in (ab, self.consts, self.sel):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "abp_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    runtime_args=rt,
                    common_runtime_args=[
                        ab.buffer_address(),
                        self.consts.buffer_address(),
                        self.sel.buffer_address(),
                        Mt,
                    ],
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "abp_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=ttnn.TensorAccessorArgs(g_bh).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(beta_bh).get_compile_time_args(),
                    runtime_args=rt,
                    common_runtime_args=[g_bh.buffer_address(), beta_bh.buffer_address(), Mt],
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "abp_compute.cpp",
                    core_ranges=crs,
                    runtime_args=rt,
                    common_runtime_args=[Mt, _f32_bits(1.0), _f32_bits(1.0), _f32_bits(20.0)],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                    ),
                ),
            ],
            semaphores=[],
            cbs=[
                cbd(0, bf, 3),
                cbd(1, f32, 4),
                cbd(2, bf, 32),
                cbd(3, f32, 2),
                cbd(4, f32, 2),
                cbd(16, f32, 2),
                cbd(17, f32, 4),
            ],
        )
        ttnn.generic_op([ab, self.consts, self.sel, g_bh, beta_bh], program)
        return beta_bh, g_bh


def supported(gdn, x, valid_len):
    T = list(x.shape)[1]
    return (
        list(x.shape)[0] == 1
        and T % C == 0
        and (valid_len is None or valid_len >= T)
        and (gdn.weights.mega_fused_weight is not None or _ternary_prefill(gdn))
        and gdn.weights.fused_conv_bias_dev is None
        and gdn.head_k_dim == 128
        and gdn.head_v_dim == 128
        and os.environ.get("QWEN_GDN_INV_DOUBLING", "0") == "0"
    )


def _ternary_prefill(gdn):
    return getattr(getattr(gdn, "ternary", None), "pre_qkvz", None) is not None


def _to_bht(t, B, T, H):
    """[B, T, H] -> [B * H, T] fp32 TILE (as the shared adapter's _to_bht)."""
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    t = ttnn.reshape(t, [B, T, H])
    t = ttnn.permute(t, (0, 2, 1))
    t = ttnn.reshape(t, [B * H, T])
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# ttnn.transformer.chunk_gated_delta_rule (fused path: per-chunk prep on producer cores handed to the scan cores over
# the NoC, no DRAM intermediates; flat token-major q / k / v with the q / k L2 norm and scale in-kernel). T = 2048,
# 16 / 48 heads: 0.71 ms vs 2.87 ms for ChunkPrep + gated_delta_attn_seq, same error vs an fp64 recurrence.
# BONSAI_GDN_CHUNK_OP=0: the ChunkPrep path.
_CGDN = {}
CGDN_C = 32  # the op's flat (token-major) path takes 32-token chunks


def _chunk_op_on(gdn, B):
    return os.environ.get("BONSAI_GDN_CHUNK_OP", "1") == "1" and B == 1 and id(gdn) in _CGDN.get("gates", {})


def _cgdn_consts(device):
    """The op's constant tiles (eye / tril / ones [1, 1, C, C], quadrant masks [1, 1, 32, 96]), built outside any trace
    capture (the op would upload them on every call otherwise)."""
    if "consts" not in _CGDN:
        C = CGDN_C
        m = torch.zeros(32, 96)
        m[:16, 0:16] = 1
        m[16:, 48:64] = 1
        m[16:, 64:80] = 1

        def up(t):
            return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        _CGDN["consts"] = dict(
            eye=up(torch.eye(C).reshape(1, 1, C, C)),
            tril=up(torch.tril(torch.ones(C, C)).reshape(1, 1, C, C)),
            ones=up(torch.ones(1, 1, C, C)),
            masks=up(m.reshape(1, 1, 32, 96)),
        )
    return _CGDN["consts"]


def _chunk_op_forward(gdn, qkv, ab, T, Hq, H, K, V, na):
    """conv outputs qkv = (q, k, v) [1, T, *] bf16, ab [1, T, a | b] -> (o [H, T, V] fp32, final state [1, H, K, V] fp32)."""
    dt_bias, a_neg = _CGDN["gates"][id(gdn)]
    abf = ttnn.typecast(ab, ttnn.float32)
    a = ttnn.slice(abf, [0, 0, 0], [1, T, na])
    b = ttnn.slice(abf, [0, 0, na], [1, T, 2 * na])
    ttnn.deallocate(abf)
    beta = ttnn.sigmoid(b)
    ap = ttnn.add(a, dt_bias)
    sp = ttnn.softplus(ap)
    g = ttnn.multiply(sp, a_neg)
    for t in (a, b, ap, sp):
        ttnn.deallocate(t)
    q, k, v = qkv
    s0 = ttnn.typecast(gdn.recurrent_state, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    o, fs = ttnn.transformer.chunk_gated_delta_rule(
        q, k, v, g, beta,
        scale=K**-0.5,
        initial_state=ttnn.reshape(s0, [1, H, K, V]),
        output_final_state=True,
        chunk_size=CGDN_C,
        output_head_major=True,
        program_config=ttnn.ChunkGdnFusedProgramConfig(),
        num_key_heads=Hq,
        **_cgdn_consts(gdn.device),
    )  # fmt: skip
    for t in (q, k, v, g, beta, s0):
        ttnn.deallocate(t)
    return o, fs


def _prep_scan(gdn, qkv, beta_bh, g_bh, B, T, Hq, H, K, V, dq, device):
    """ChunkPrep + gated_delta_attn_seq (BONSAI_GDN_CHUNK_OP=0)."""
    if "device" not in chunk_prep._PREP:
        chunk_prep._PREP["device"] = chunk_prep.ChunkPrep(device)
    tm = dict(
        Wt=dq // 32,
        rf=H // Hq,
        q_off=0,
        k_off=Hq * K // 32,
        v_off=2 * Hq * K // 32,
        eps=1e-6,
        scale=K**-0.5,
    )
    alpha = float(os.environ.get("QWEN_GDN_DIAG_ALPHA", "0.25"))
    prep = chunk_prep._PREP["device"](qkv, qkv, qkv, beta_bh, g_bh, alpha, token_major=tm)
    ttnn.deallocate(qkv)
    s0 = ttnn.typecast(
        ttnn.reshape(gdn.recurrent_state, [B * H, K, V]), ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    o_bh, final_state = ttnn.transformer.gated_delta_attn_seq(
        *prep, initial_state=s0, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for t in prep:
        ttnn.deallocate(t)
    return o_bh, final_state


def gdn_chunk_forward(gdn, x, linear_shim):
    """x [1, T, dim] -> the layer output; updates gdn.recurrent_state and gdn.fused_conv_state in place.
    linear_shim: the qwen36 prefill ttnn shim (its linear() adds the swept in_proj / out_proj program configs)."""

    w = gdn.weights
    B, T = list(x.shape)[0], list(x.shape)[1]
    Hq, H, K, V = gdn.num_heads, gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim
    dq, da, db = w.mega_qkv_dim, w.mega_a_dim, w.mega_b_dim
    ckc = gdn.compute_kernel_config
    device = gdn.device

    tern = gdn.ternary if _ternary_prefill(gdn) else None
    use_op = tern is not None and _chunk_op_on(gdn, B)
    if tern is not None:
        # exact ternary in_proj: [qkv | z] on the rotated input; a and b (bf16 weights) at HiFi4
        from models.demos.blackhole.bonsai2.tt.ternary import prerot

        xr = prerot.take(x)  # the layer's fused norm + rotation (tprefill/norm_rot.py), else rotate here
        if xr is None:
            xr = tern.prot_in(x)
        mega = tern.pre_qkvz(xr)  # [1, T, qkv | z]
        ttnn.deallocate(xr)
        from models.demos.blackhole.bonsai2.tt.ternary.gdn import ab_prefill_config

        ab = ttnn.linear(x, tern.ab_w, compute_kernel_config=tern.ab_ckc, program_config=ab_prefill_config(T))
        na = tern.na
        gates = getattr(gdn, "_bonsai_gates", None)
        if use_op:
            a_raw = b_raw = None  # ab goes to the chunk op's gates (_chunk_op_forward)
        elif gates is not None and B == 1:
            beta_bh, g_bh = gates(ab, T)  # sigmoid / softplus gates straight into prep's layouts (GateTiles)
            a_raw = b_raw = None
        else:
            a_raw = ttnn.slice(ab, [0, 0, 0], [B, T, na])
            b_raw = ttnn.slice(ab, [0, 0, na], [B, T, 2 * na])
        if not use_op:
            ttnn.deallocate(ab)
        z_off = tern.split // 32
    else:
        mega = linear_shim.linear(x, w.mega_fused_weight, memory_config=None, compute_kernel_config=ckc)
        a_raw = ttnn.to_layout(mega[:, :, dq : dq + da], ttnn.TILE_LAYOUT)
        b_raw = ttnn.to_layout(mega[:, :, dq + da : dq + da + db], ttnn.TILE_LAYOUT)
        z_off = (dq + da + db) // 32
        assert (dq + da + db) % 32 == 0, "z must start on a tile column of the in_proj output"

    if use_op:  # conv + SiLU straight into separate q, k, v (the chunk op's inputs)
        qkv = gdn._bonsai_conv(mega, gdn.fused_conv_state, split=(Hq * K, Hq * K))
    else:
        qkv = gdn._bonsai_conv(mega, gdn.fused_conv_state)  # conv + SiLU of q | k | v; conv state updated in place

    if a_raw is not None:
        beta = ttnn.sigmoid(b_raw)
        g = ttnn.multiply(w.A_neg, ttnn.softplus(ttnn.add(a_raw, w.dt_bias)))
        for t in (a_raw, b_raw):
            ttnn.deallocate(t)
        beta_bh = ttnn.reshape(_to_bht(beta, B, T, H), [B * H, T, 1])
        g_bh = _to_bht(g, B, T, H)

    if use_op:
        o_bh, final_state = _chunk_op_forward(gdn, qkv, ab, T, Hq, H, K, V, tern.na)
        ttnn.deallocate(ab)
    else:
        o_bh, final_state = _prep_scan(gdn, qkv, beta_bh, g_bh, B, T, Hq, H, K, V, dq, device)

    # o [BH, T, V] -> o / rms(o) * w * silu(z), token-major [1, T, H * V]
    fin = getattr(gdn, "_bonsai_finish", None)
    if fin is None:
        norm_w = ttnn.to_torch(w.o_norm_weight).float().reshape(-1)[:V]
        fin = gdn._bonsai_finish = OutputFinish(device, norm_w, gdn.norm_eps)
    o = fin(o_bh, mega, T, z_off)  # z read straight from the in_proj output
    ttnn.deallocate(o_bh)
    ttnn.deallocate(mega)
    if tern is not None:
        orr = tern.prot_out(o)
        ttnn.deallocate(o)
        out = tern.pre_out(orr)
        ttnn.deallocate(orr)
    else:
        out = linear_shim.linear(o, w.o_proj_weight, memory_config=None, compute_kernel_config=ckc)

    new_state = ttnn.typecast(ttnn.reshape(final_state, [B, H, K, V]), ttnn.bfloat16)
    ttnn.copy(new_state, gdn.recurrent_state)
    ttnn.deallocate(new_state)
    return out


def prepare(model):
    """Build every GDN layer's OutputFinish and the chunk preprocessing now: their constant tiles are host writes,
    which a trace capture would not replay."""
    device = model.mesh_device
    if "device" not in chunk_prep._PREP:
        chunk_prep._PREP["device"] = chunk_prep.ChunkPrep(device)
    for layer in model.layers:
        if layer.is_full_attention:
            continue
        gdn = layer.attention
        norm_w = ttnn.to_torch(gdn.weights.o_norm_weight).float().reshape(-1)[: gdn.head_v_dim]
        gdn._bonsai_finish = OutputFinish(device, norm_w, gdn.norm_eps)
        taps = torch.cat([ttnn.to_torch(t).float().reshape(1, -1) for t in gdn.weights.fused_conv_weight_taps], 0)
        gdn._bonsai_conv = ConvPrefill(device, taps)
        f32 = lambda t: ttnn.from_torch(  # noqa: E731
            ttnn.to_torch(t).float().reshape(-1)[: gdn.num_v_heads].reshape(1, 1, -1),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
        )  # fmt: skip
        _CGDN.setdefault("gates", {})[id(gdn)] = (f32(gdn.weights.dt_bias), f32(gdn.weights.A_neg))
        _cgdn_consts(device)
        if os.environ.get("BONSAI_GDN_GATES", "1") == "1" and gdn.num_v_heads == 48:
            gdn._bonsai_gates = GateTiles(
                device,
                ttnn.to_torch(gdn.weights.dt_bias).float().reshape(-1)[:48],
                ttnn.to_torch(gdn.weights.A_neg).float().reshape(-1)[:48],
            )
