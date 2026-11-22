# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""K2a standalone harness: the K2a attention (head + unit cores, k2a.py) on synthetic multi-user caches for `layers`
layers, one core standing in for the streamers (kernels/k2a_harness_streamer.cpp), against a torch reference: per
layer, user, q head and row, the O row (PCC, max error), and the new K / V rows the heads prepped and the roots
spliced into the cache. The reference attention reads the cache the device left (its new rows included), so it checks
the attention and the splice; the new rows themselves are checked against torch's prep.

Run through bench/dflash2/guarded.sh:
  python -m models.demos.gemma4_31b_qb2.resident_m5_14.k2a_harness --M 1 --capacity 4096 --layers 6 --out x.json
"""
import argparse
import json
import math
import time

import torch

import ttnn
from models.demos.gemma4_31b_qb2.resident_m5_14 import PKG_DIR
from models.demos.gemma4_31b_qb2.resident_m5_14.common import EPS, streamer_cores
from models.demos.gemma4_31b_qb2.resident_m5_14.k2a import K2aAttention
from models.demos.gemma4_31b_qb2.resident_s2.run_step2 import close_mesh, open_mesh

PAGE, RING = 128, 9
KINDS = ("sliding_attention", "full_attention")


class Serve:
    pass


def rope_tables(positions):
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from models.demos.gemma4_31b_qb2.resident_m5_14.weights import Checkpoint

    cfg = AutoConfig.from_pretrained(Checkpoint(None).path, local_files_only=True).text_config
    out = {}
    for kind in KINDS:
        cos, sin = Gemma4TextRotaryEmbedding(cfg)(torch.empty(1, dtype=torch.bfloat16),
                                                  torch.arange(positions).reshape(1, -1), layer_type=kind)
        out[kind] = (cos[0].bfloat16(), sin[0].bfloat16())
    return out


def rms(x, w=None):
    x = x.float()
    y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS)
    return y * w.float() if w is not None else y


def rope_apply(x, cos, sin):
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos.float() + rot * sin.float()


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / d if d else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--capacity", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--positions", default="", help="comma list per user (-1 inert); default: a spread")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--kscale", type=float, default=1.0, help="cache values ~ N(0, kscale^2): < 1 for softer softmax")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M, L, cap = a.M, a.layers, a.capacity
    U = 32 // M
    g = torch.Generator().manual_seed(a.seed)
    if a.positions:
        pos_u = [int(x) for x in a.positions.split(",")]
    else:
        pick = [0, 5, 31, 32 - M, 100, 1022, 1023, 1024, 1100, 1150, 1151, 1152, 2047, 2048, cap - M, 777]
        pos_u = [pick[u % len(pick)] if u < len(pick) else int(torch.randint(0, cap - M + 1, (1,), generator=g))
                 for u in range(U)]
    assert len(pos_u) == U and all(-1 <= p <= cap - M for p in pos_u), pos_u
    live = [p >= 0 for p in pos_u]

    mesh = open_mesh()
    res = {"args": vars(a), "positions": pos_u}
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh)
        dram = ttnn.DRAM_MEMORY_CONFIG
        # the engine's placement around the streamers, hub and slices (model.py at R = 32: K = 4 slices)
        groups, used = streamer_cores(mesh)
        scores = [c for grp in groups for c in grp]
        xs, ys = [c.x for c in scores], [c.y for c in scores]
        free = [ttnn.CoreCoord(x, y) for y in range(min(ys), max(ys) + 1) for x in range(min(xs), max(xs) + 1)
                if (x, y) not in used]
        used_all = set(used) | {(c.x, c.y) for c in free[:5]}
        fake = scores[0]
        # pools: sliding ring of 9 pages per slot, full ceil(cap / 128) per slot; the last page of each is the scratch
        Pf = -(-cap // PAGE)
        sl_pages, fu_pages = 32 * RING + 1, 32 * Pf + 1
        layer_group = [0 if l % 6 == 5 else 1 for l in range(L)]
        tab = torch.zeros(32, 6, Pf, dtype=torch.int32)
        for s in range(32):
            tab[s, 1] = torch.tensor([s * RING + c % RING for c in range(Pf)], dtype=torch.int32)
            tab[s, 0] = torch.arange(s * Pf, (s + 1) * Pf, dtype=torch.int32)
        scr = torch.zeros(32, 1, 16, dtype=torch.int32)
        scr[:, 0, 0], scr[:, 0, 1] = fu_pages - 1, sl_pages - 1
        kv_host, kv = [], []
        for l in range(L):
            full = l % 6 == 5
            shape = (fu_pages, 1, PAGE, 512) if full else (sl_pages, 4, PAGE, 256)
            pair = [(a.kscale * torch.randn(shape, generator=g)).bfloat16() for _ in range(2)]
            kv_host.append(pair)
            kv.append(tuple(ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep,
                                            memory_config=dram) for t in pair))
        rows_pos = torch.full((32,), -1, dtype=torch.int32)
        rows_slot = torch.zeros(32, dtype=torch.int32)
        for u in range(U):
            for m in range(M):
                rows_pos[u * M + m] = pos_u[u] + m if live[u] else -1
                rows_slot[u * M + m] = u
        rope_h = rope_tables(Pf * PAGE + 64)
        serve = Serve()
        serve.positions = ttnn.from_torch(rows_pos.reshape(1, 1, 1, 32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                          device=mesh, mesh_mapper=rep, memory_config=dram)
        serve.slots = ttnn.from_torch(rows_slot.reshape(1, 1, 1, 32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                      device=mesh, mesh_mapper=rep, memory_config=dram)
        serve.tables = ttnn.from_torch(tab, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep,
                                       memory_config=dram)
        serve.scratch_t = ttnn.from_torch(scr, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                          mesh_mapper=rep, memory_config=dram)
        serve.capacity, serve.max_users, serve.layer_group = cap, 32, layer_group
        serve.pool_pages = {0: fu_pages, 1: sl_pages}
        serve.kv = kv
        serve.rope = {k: tuple(ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                               mesh_mapper=rep, memory_config=dram) for t in rope_h[k]) for k in KINDS}
        # norm weights (tile (2 l + k) * 16 + d) and the raw q|k|v rows per layer and head (rx tiles)
        wn = torch.zeros(L * 2 * 32, 512, dtype=torch.bfloat16)
        w_host = []
        rx = torch.zeros(32, L * 8 * 32 * 32, dtype=torch.bfloat16)
        raw = []
        for l in range(L):
            full = l % 6 == 5
            D = 512 if full else 256
            wq = (1 + 0.1 * torch.randn(D, generator=g)).bfloat16()
            wk = (1 + 0.1 * torch.randn(D, generator=g)).bfloat16()
            wn[(2 * l) * 32:(2 * l + 1) * 32, :D] = wq
            wn[(2 * l + 1) * 32:(2 * l + 2) * 32, :D] = wk
            w_host.append((wq, wk))
            q = torch.randn(8, 32, D, generator=g).bfloat16()  # [head, row, D]
            kvr = torch.randn(4 if not full else 1, 2, 32, D, generator=g).bfloat16()  # [kv head, k / v, row, D]
            raw.append((q, kvr))
            for h in range(8):
                base = ((l * 8 + h) * 32) * 32
                rx[:, base:base + D] = q[h]
                if full:
                    src = kvr[0, 0] if h in (0, 1) else None
                else:
                    src = kvr[h // 2, h % 2]
                if src is not None:
                    rx[:, base + 16 * 32:base + 16 * 32 + D] = src
        w_t = ttnn.from_torch(wn, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep,
                              memory_config=dram)
        rx_t = ttnn.from_torch(rx.reshape(1, 1, 32, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                               mesh_mapper=rep, memory_config=dram)
        dump_t = ttnn.from_torch(torch.zeros(1, 1, 32, L * 128 * 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=mesh, mesh_mapper=rep, memory_config=dram)

        att = K2aAttention(mesh, M, L, cap, used_all, 1, layer_group)
        res["l1"] = att.l1_bytes()
        NCRISC, BRISC = ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)  # noqa: E731

        def compute_cfg(fid, fp32, full_sync):
            c = ttnn.ComputeConfigDescriptor()
            c.math_fidelity, c.fp32_dest_acc_en, c.dst_full_sync_en = fid, fp32, full_sync
            return c

        kernel = lambda src, cr_, ct_, rt_, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=f"{PKG_DIR}/kernels/{src}", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cr_, compile_time_args=ct_, runtime_args=rt_, config=cfg)
        rope_addrs = [t.buffer_address() for k in KINDS for t in serve.rope[k]]
        kv_addrs = [(k.buffer_address(), v.buffer_address()) for k, v in kv]
        SEM_O, SEM_RXADDR = 1, 9
        kernels = att.kernels(kernel, dm, compute_cfg, [fake], serve, rope_addrs, w_t.buffer_address(), kv_addrs,
                              SEM_O, SEM_RXADDR)
        fg = ttnn.CoreRangeSet([ttnn.CoreRange(fake, fake)])
        frt = ttnn.RuntimeArgs()
        frt[fake.x][fake.y] = [rx_t.buffer_address(), dump_t.buffer_address(), SEM_O, SEM_RXADDR, L] + \
            [att.xy(c) for c in att.heads]
        kernels.append(kernel("k2a_harness_streamer.cpp", fg, [], frt, dm(BRISC, ttnn.NOC.RISCV_0_default)))
        cbs = att.cbs()
        for idx, pages in ((0, 128), (1, 32)):
            fd = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=2048)
            cbs.append(ttnn.CBDescriptor(total_size=pages * 2048, core_ranges=fg, format_descriptors=[fd]))
        sems = att.semaphores() + [ttnn.SemaphoreDescriptor(id=i, core_ranges=fg, initial_value=0) for i in range(10)]
        mesh_pd = ttnn.MeshProgramDescriptor()
        for chip in range(mesh.get_num_devices()):
            coord = ttnn.MeshCoordinate(0, chip)
            mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems,
                                                                                    cbs=cbs)
        io = [serve.positions, serve.slots, serve.tables, serve.scratch_t, w_t, rx_t, dump_t] + att.io + \
            [t for pair in kv for t in pair] + [t for k in KINDS for t in serve.rope[k]]
        print(json.dumps({"stage": "built", "l1": res["l1"]}), flush=True)
        t0 = time.perf_counter()
        ttnn.generic_op(io, mesh_pd)
        ttnn.synchronize_device(mesh)
        res["seconds"] = round(time.perf_counter() - t0, 3)
        print(json.dumps({"stage": "ran", "seconds": res["seconds"]}), flush=True)

        # ---- reference
        dump = ttnn.to_torch(ttnn.get_device_tensors(dump_t)[0]).reshape(32, L * 128 * 32).float()
        worst, rows = 1.0, []
        new_rows = []
        for l in range(L):
            full = l % 6 == 5
            D, Dt = (512, 16) if full else (256, 8)
            kind = KINDS[1 if full else 0]
            cos_t, sin_t = rope_h[kind]
            kc = ttnn.to_torch(ttnn.get_device_tensors(kv[l][0])[0]).float()
            vc = ttnn.to_torch(ttnn.get_device_tensors(kv[l][1])[0]).float()
            q, kvr = raw[l]
            wq, wk = w_host[l]
            for u in range(U):
                if not live[u]:
                    continue
                p = pos_u[u]
                s = u
                prow = [p + m for m in range(M)]
                cs, sn = cos_t[prow], sin_t[prow]
                # new K / V rows (torch prep) vs the cache's rows at p .. p + M - 1
                for gk in range(1 if full else 4):
                    src = kvr[gk]
                    rr = slice(u * M, u * M + M)
                    if full:
                        vref = rms(src[0][rr])
                        kref = rope_apply((vref.bfloat16().float() * wk.float()), cs, sn)
                    else:
                        kref = rope_apply(rms(src[0][rr], wk), cs, sn)
                        vref = rms(src[1][rr])
                    for m in range(M):
                        pos = p + m
                        pg = int(tab[s, 0 if full else 1, pos // PAGE])
                        new_rows.append({"l": l, "u": u, "g": gk, "m": m,
                                         "k_pcc": pcc(kc[pg, gk, pos % PAGE], kref[m]),
                                         "v_pcc": pcc(vc[pg, gk, pos % PAGE], vref[m])})
                for h in range(8):
                    gk = 0 if full else h // 2
                    qh = rope_apply(rms(q[h][u * M:(u + 1) * M], wq), cs, sn)  # [M, D]
                    for m in range(M):
                        pq = p + m
                        lo = 0 if full else max(0, pq - 1023)
                        ps = torch.arange(lo, pq + 1)
                        pages = tab[s, 0 if full else 1, ps // PAGE].long()
                        K_ = kc[pages, gk, ps % PAGE]
                        V_ = vc[pages, gk, ps % PAGE]
                        a_ = torch.softmax(K_ @ qh[m], dim=0)
                        oref = a_ @ V_
                        r = u * M + m
                        odev = torch.cat([dump[r, (l * 128 + h * Dt + d) * 32:(l * 128 + h * Dt + d + 1) * 32]
                                          for d in range(Dt)])
                        c_ = pcc(odev, oref)
                        worst = min(worst, c_)
                        rows.append({"l": l, "u": u, "h": h, "m": m, "pcc": round(c_, 6),
                                     "maxerr": round(float((odev - oref).abs().max()), 4)})
        res["o_min_pcc"] = worst
        res["o_rows"] = len(rows)
        res["o_worst"] = sorted(rows, key=lambda r: r["pcc"])[:8]
        res["o_all"] = [[r["l"], r["u"], r["h"], r["m"], r["pcc"], r["maxerr"]] for r in rows]
        res["kv_min_pcc"] = min([min(r["k_pcc"], r["v_pcc"]) for r in new_rows] or [1.0])
        res["kv_worst"] = sorted(new_rows, key=lambda r: min(r["k_pcc"], r["v_pcc"]))[:6]
        print(json.dumps({k: res[k] for k in ("o_min_pcc", "o_rows", "kv_min_pcc", "seconds")}), flush=True)
    finally:
        json.dump(res, open(a.out, "w"), indent=1)
        close_mesh(mesh)


if __name__ == "__main__":
    main()
