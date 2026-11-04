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
        crs = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(c % grid.x, c // grid.x), ttnn.CoreCoord(c % grid.x, c // grid.x))
                for c in range(ncores)
            ]
        )
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
        self.device = device

    def __call__(self, x, state):
        device, D = self.device, self.D
        T, W = list(x.shape)[1], list(x.shape)[2]
        Dt, Tt = D // 32, T // 32
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, T, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        grid = device.compute_with_storage_grid_size()
        ncores = min(grid.x * grid.y, Dt)
        crs = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(c % grid.x, c // grid.x), ttnn.CoreCoord(c % grid.x, c // grid.x))
                for c in range(ncores)
            ]
        )
        rd_ct = []
        for t in (x, state, self.taps, self.consts):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = (
            ttnn.TensorAccessorArgs(out).get_compile_time_args()
            + ttnn.TensorAccessorArgs(state).get_compile_time_args()
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
                    kernel_source=KERNELS + "conv_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    common_runtime_args=[t.buffer_address() for t in (x, state, self.taps, self.consts)]
                    + [Dt, ncores, grid.x, Tt, W // 32],
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "conv_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=wr_ct,
                    common_runtime_args=[out.buffer_address(), state.buffer_address(), Dt, ncores, grid.x, Tt],
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "conv_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[],
                    common_runtime_args=[Dt, ncores, grid.x, Tt],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
                    ),
                ),
            ],
            semaphores=[],
            cbs=[
                cbd(0, bf, 2048, 2),
                cbd(1, bf, 2048, 1),
                cbd(2, bf, 2048, 4),
                cbd(3, bf, 2048, 11),
                cbd(4, f32, 4096, 4),
                cbd(5, f32, 4096, 4),
                cbd(16, bf, 2048, 2),
                cbd(17, bf, 2048, 1),
            ],
        )
        ttnn.generic_op([x, state, self.taps, self.consts, out], program)
        return out


def supported(gdn, x, valid_len):
    T = list(x.shape)[1]
    return (
        list(x.shape)[0] == 1
        and T % C == 0
        and (valid_len is None or valid_len >= T)
        and gdn.weights.mega_fused_weight is not None
        and gdn.weights.fused_conv_bias_dev is None
        and gdn.head_k_dim == 128
        and gdn.head_v_dim == 128
        and os.environ.get("QWEN_GDN_INV_DOUBLING", "0") == "0"
    )


def _to_bht(t, B, T, H):
    """[B, T, H] -> [B * H, T] fp32 TILE (as the shared adapter's _to_bht)."""
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    t = ttnn.reshape(t, [B, T, H])
    t = ttnn.permute(t, (0, 2, 1))
    t = ttnn.reshape(t, [B * H, T])
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def gdn_chunk_forward(gdn, x, linear_shim):
    """x [1, T, dim] -> the layer output; updates gdn.recurrent_state and gdn.fused_conv_state in place.
    linear_shim: the qwen36 prefill ttnn shim (its linear() adds the swept in_proj / out_proj program configs)."""

    w = gdn.weights
    B, T = list(x.shape)[0], list(x.shape)[1]
    Hq, H, K, V = gdn.num_heads, gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim
    dq, da, db = w.mega_qkv_dim, w.mega_a_dim, w.mega_b_dim
    ckc = gdn.compute_kernel_config
    device = gdn.device

    mega = linear_shim.linear(x, w.mega_fused_weight, memory_config=None, compute_kernel_config=ckc)
    a_raw = ttnn.to_layout(mega[:, :, dq : dq + da], ttnn.TILE_LAYOUT)
    b_raw = ttnn.to_layout(mega[:, :, dq + da : dq + da + db], ttnn.TILE_LAYOUT)

    qkv = gdn._bonsai_conv(mega, gdn.fused_conv_state)  # conv + SiLU of q | k | v; conv state updated in place

    beta = ttnn.sigmoid(b_raw)
    g = ttnn.multiply(w.A_neg, ttnn.softplus(ttnn.add(a_raw, w.dt_bias)))
    for t in (a_raw, b_raw):
        ttnn.deallocate(t)
    beta_bh = ttnn.reshape(_to_bht(beta, B, T, H), [B * H, T, 1])
    g_bh = _to_bht(g, B, T, H)

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

    # o [BH, T, V] -> o / rms(o) * w * silu(z), token-major [1, T, H * V]
    fin = getattr(gdn, "_bonsai_finish", None)
    if fin is None:
        norm_w = ttnn.to_torch(w.o_norm_weight).float().reshape(-1)[:V]
        fin = gdn._bonsai_finish = OutputFinish(device, norm_w, gdn.norm_eps)
    z_off = (dq + da + db) // 32
    assert (dq + da + db) % 32 == 0, "z must start on a tile column of the in_proj output"
    o = fin(o_bh, mega, T, z_off)  # z read straight from the in_proj output
    ttnn.deallocate(o_bh)
    ttnn.deallocate(mega)
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
