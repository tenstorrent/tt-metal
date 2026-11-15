# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""fusion-around: microbenchmarks of the ops AROUND the TP32 MoE kernel (decode), 4x8 mesh, stub kernel (partial = random per-chip tensor).
Per-op traced us (10 reps per trace, median of replays) + correctness of every tail variant against the host sum over the 32 chips.
Env: FA_T=1 (tokens per mesh row, Tt = 4 T), FA_VARIANTS=all|comma list, OUT=json path."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh, p, pcc, traced_us
from models.demos.gpt_oss.tt.ccl import CCLManager

HID = 6144
T = int(os.environ.get("FA_T", "1"))
OUT = os.environ.get("OUT", "/tmp/fusion_around.json")
VAR = os.environ.get("FA_VARIANTS", "all")

md = open_mesh(trace=300_000_000)
rows, cols = tuple(md.shape)
Tt = T * rows
res = {"T": T, "us": {}, "pcc": {}}
ckc = ttnn.init_device_compute_kernel_config(
    md.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)


def up(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, dims=(0, 1)):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=dims, mesh_shape=(rows, cols)),
    )


def chip(t, r, c):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols + c])


try:
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)

    def ag(t, dim, axis):
        return ttnn.experimental.all_gather_async(
            t,
            dim=dim,
            cluster_axis=axis,
            mesh_device=md,
            topology=ttnn.Topology.Linear if axis == 0 else ttnn.Topology.Ring,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            num_links=ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )

    def rs(t, dim, axis):
        return ttnn.experimental.reduce_scatter_minimal_async(
            t,
            dim=dim,
            multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
            num_links=ccl.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear if axis == 0 else ttnn.Topology.Ring,
            cluster_axis=axis,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )

    g = torch.Generator().manual_seed(1)
    P = torch.randn(rows, cols, Tt, HID, generator=g).to(torch.bfloat16)
    part = up(P)  # [1,1,Tt,HID] tile per chip
    tot = P.float().sum((0, 1))  # [Tt, HID]

    def sel_mask(K):  # [rows -> per row r]: picks the T tokens of row r out of K/Tt... built per variant
        pass

    # sel[r] = [T, Tt] ; sel2[r] = [T, 4*32] (picks from an AG over rows of tile-padded [32, N] blocks)
    m = torch.zeros(rows, 1, T, Tt)
    m2 = torch.zeros(rows, 1, T, rows * 32)
    for r in range(rows):
        for i in range(T):
            m[r, 0, i, r * T + i] = 1.0
            for j in range(rows):
                m2[r, 0, i, j * 32 + r * T + i] = 1.0
    sel = up(m.to(torch.bfloat16), dims=(0, None))
    sel2 = up(m2.to(torch.bfloat16), dims=(0, None))
    mk = torch.zeros(rows, 1, Tt, T)
    for r in range(rows):
        for i in range(T):
            mk[r, 0, r * T + i, i] = 1.0
    mask = up(mk.to(torch.bfloat16), dims=(0, None))
    fold = up(torch.randn(rows, 1, T, HID, generator=g).to(torch.bfloat16), dims=(0, None))
    tok = up(
        torch.randn(rows, 1, T, HID, generator=g).to(torch.bfloat16).reshape(rows * T, 1, 1, HID),
        ttnn.ROW_MAJOR_LAYOUT,
        dims=(0, None),
    )
    w8 = up(torch.rand(rows * T, 1, 1, 8, generator=g).to(torch.bfloat16), ttnn.ROW_MAJOR_LAYOUT, dims=(0, None))
    i8 = up(
        torch.randint(0, 256, (rows * T, 1, 1, 8), generator=g).to(torch.int32),
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.uint16,
        dims=(0, None),
    )
    y8 = ttnn.to_memory_config(rs(part, 3, 1), ttnn.DRAM_MEMORY_CONFIG)  # [1,1,Tt,HID/8] (a column-reduced partial)
    ttnn.synchronize_device(md)

    def T_(name, fn, **kw):
        if VAR != "all" and name not in VAR.split(","):
            return
        try:
            res["us"][name] = round(traced_us(md, fn, **kw), 1)
            p("US", name, res["us"][name])
        except Exception as e:  # noqa
            res["us"][name] = f"ERR {type(e).__name__}: {str(e)[:200]}"
            p("US", name, res["us"][name])

    # ---------------- single ops (production order)
    T_("ag_x_rows_rm", lambda: ag(tok, 0, 0))
    xall = ag(tok, 0, 0)
    T_("tilize_xall", lambda: ttnn.to_layout(ttnn.reshape(xall, [1, 1, Tt, HID]), ttnn.TILE_LAYOUT))
    T_("rs_cols", lambda: rs(part, 3, 1))
    T_("rs_rows_on_hid8", lambda: rs(y8, 3, 0))
    y2 = rs(y8, 3, 0)
    T_("ag_rows_hid32", lambda: ag(y2, 3, 0))
    y3 = ag(y2, 3, 0)
    T_("sel_mm", lambda: ttnn.matmul(sel, y3, compute_kernel_config=ckc))
    z = ttnn.matmul(sel, y3, compute_kernel_config=ckc)
    T_("ag_cols_final", lambda: ag(z, 3, 1))
    T_("fold_mask_mm", lambda: ttnn.matmul(mask, fold, compute_kernel_config=ckc))
    fp = ttnn.matmul(mask, fold, compute_kernel_config=ckc)
    T_("fold_add", lambda: ttnn.add(part, fp))

    # ---------------- route pack variants
    def pack_tile():
        idx = ttnn.to_memory_config(i8, ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.to_memory_config(w8, ttnn.DRAM_MEMORY_CONFIG)
        ids = ttnn.typecast(ttnn.to_layout(ttnn.reshape(idx, [1, 1, T, 8]), ttnn.TILE_LAYOUT), ttnn.int32)
        sc = ttnn.typecast(ttnn.to_layout(ttnn.reshape(w, [1, 1, T, 8]), ttnn.TILE_LAYOUT), ttnn.float32)
        sb = ttnn.bitwise_right_shift(ttnn.bitcast(sc, ttnn.int32), 16)
        r = ttnn.concat([ids, sb], dim=3)
        r = ttnn.to_layout(r, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(r, [1, 1, 1, 16 * T])

    def pack_u16():
        r = ttnn.concat(
            [ttnn.reshape(i8, [T, 1, 1, 8]), ttnn.bitcast(ttnn.reshape(w8, [T, 1, 1, 8]), ttnn.uint16)], dim=3
        )
        r = ttnn.typecast(r, ttnn.int32)
        return ttnn.reshape(r, [1, 1, 1, 16 * T])

    T_("pack_tile", pack_tile)
    T_("pack_u16", pack_u16)
    rt = pack_u16()
    T_("ag_route", lambda: ag(rt, 3, 0))
    T_(
        "tilize_mc_l1",
        lambda: ttnn.to_layout(
            ttnn.reshape(xall, [1, 1, Tt, HID]), ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        ),
    )
    T_("tilize_pad_op", lambda: ttnn.tilize_with_val_padding(ttnn.reshape(xall, [1, 1, Tt, HID]), [1, 1, 32, HID], 0.0))

    # ---------------- whole tail variants: part -> full-HID [1,1,T,HID] on every chip
    def tail_base():
        y = rs(part, 3, 1)
        y2_ = rs(y, 3, 0)
        y3_ = ag(y2_, 3, 0)
        z_ = ttnn.matmul(sel, y3_, compute_kernel_config=ckc)
        return ag(z_, 3, 1)

    def tail_agdim2():
        y = rs(part, 3, 1)
        yg_ = ag(y, 2, 0)
        z_ = ttnn.matmul(sel2, yg_, compute_kernel_config=ckc)
        return ag(z_, 3, 1)

    def tail_agdim2_rowsfirst_full():
        # rows stage on the FULL hidden first (gather the 4 rows' tile blocks, one matmul sums+selects), then the column all-reduce = RS + AG
        yg_ = ag(part, 2, 0)  # [1,1,128,HID]
        z_ = ttnn.matmul(sel2, yg_, compute_kernel_config=ckc)  # [1,1,T,HID] summed over rows (partial over columns)
        y = rs(z_, 3, 1)
        return ag(y, 3, 1)

    def tail_ar_rows():
        y = rs(part, 3, 1)
        y_ar = ttnn.all_reduce(y, cluster_axis=0, num_links=ccl.num_links, topology=ttnn.Topology.Linear)
        z_ = ttnn.matmul(sel, y_ar, compute_kernel_config=ckc)
        return ag(z_, 3, 1)

    def tail_rs_dim0_rm():
        y = rs(part, 3, 1)
        yr = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)  # [1,1,Tt,HID/8] RM
        yr = ttnn.reshape(yr, [Tt, 1, 1, HID // 8])
        z_ = rs(yr, 0, 0)  # [T,1,1,HID/8]
        z_ = ttnn.to_layout(ttnn.reshape(z_, [1, 1, T, HID // 8]), ttnn.TILE_LAYOUT)
        return ag(z_, 3, 1)

    def tail_a2a():
        y = rs(part, 3, 1)
        yr = ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), [Tt, 1, 1, HID // 8])
        a = ttnn.experimental.all_to_all_async_generic(
            yr, in_dim=0, out_dim=0, num_links=ccl.num_links, topology=ttnn.Topology.Linear, cluster_axis=0
        )
        a = ttnn.to_layout(ttnn.reshape(a, [1, 1, Tt, HID // 8]), ttnn.TILE_LAYOUT)
        s_ = ttnn.sum(ttnn.reshape(a, [rows, 1, T, HID // 8]), dim=0, keepdim=True)
        return ag(s_, 3, 1)

    variants = dict(
        tail_a2a=tail_a2a,
        tail_base=tail_base,
        tail_ar_rows=tail_ar_rows,
        tail_rs_dim0_rm=tail_rs_dim0_rm,
    )
    for name, fn in variants.items():
        if (VAR == "all" and name == "tail_a2a") or (VAR != "all" and name not in VAR.split(",")):
            continue  # tail_a2a (untested op, hang risk) only when named
        try:
            o = fn()
            ttnn.synchronize_device(md)
            ok = []
            for r in range(rows):
                for c in (0, 5):
                    got = chip(o, r, c).reshape(T, HID).float()
                    ok.append(pcc(got, tot[r * T : (r + 1) * T]))
            res["pcc"][name] = round(min(ok), 6)
            p("PCC", name, res["pcc"][name])
            ttnn.deallocate(o)
        except Exception as e:  # noqa
            res["pcc"][name] = f"ERR {type(e).__name__}: {str(e)[:200]}"
            p("PCC", name, res["pcc"][name])
            continue
        T_(name, fn)
finally:
    json.dump(res, open(OUT, "w"), indent=1)
    close_mesh(md)
