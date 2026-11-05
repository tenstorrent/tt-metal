# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip bench of the multi-core WH SDPA op (release core loop + group multicast K/V feed) through generic_op.

Usage: python wh_sdpa_op_bench.py HEADS Q_ROWS KEYS VALID_KEYS [GRID_X GRID_Y BUFS READERS REPEATS DEBUG]
Q, K, V are random; V is passed with every 32x32 tile transposed in place. Prints the op's wall time and the accuracy
of O against torch (on a sample of heads).
"""

import sys
import time

import torch

import ttnn

KDIR = "ttnn/cpp/ttnn/operations/experimental/minimax_h3_wh_sdpa/kernels/"
IDLE = 0xFFFF
MAX_READERS = 4
A_READER0_X = 19
A_LEADER_X = A_READER0_X + 2 * MAX_READERS
SCRATCH_FIXED = 4096 + 5 * 8 * 2048 + 2 * 16384  # sync, unpacker 0 region, O staging
BLOCK = 32768


def tile_transposed(v):  # [..., N, 128]: every 32x32 tile transposed in place
    *lead, n, d = v.shape
    t = v.reshape(*lead, n // 32, 32, d // 32, 32)
    return t.transpose(-3, -1).reshape(*lead, n, d)


def faces_column_major(x):  # [..., N, 128]: in every 32x32 tile swap faces (0, 1) and (1, 0) (stored f0 f2 f1 f3)
    *lead, n, d = x.shape
    t = x.reshape(*lead, n // 32, 2, 16, d // 32, 2, 16).clone()
    a = t[..., :, 0, :, :, 1, :].clone()
    t[..., :, 0, :, :, 1, :] = t[..., :, 1, :, :, 0, :]
    t[..., :, 1, :, :, 0, :] = a
    return t.reshape(*lead, n, d)


def leak_ops(dev):
    """Common ops on fixed inputs: run before and after the op, the outputs must be bit-identical (hardware state the op
    leaves behind would show up here)."""
    g = torch.Generator().manual_seed(7)
    a = torch.randn(1, 1, 256, 512, generator=g).bfloat16()
    b = torch.randn(1, 1, 512, 256, generator=g).bfloat16()
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ta, tb = ttnn.from_torch(a, **kw), ttnn.from_torch(b, **kw)
    hifi2 = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False
    )
    fp32 = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    out = {}
    first = next((x[8:] for x in sys.argv if x.startswith("--first=")), "")
    if first == "add":
        out["first_add"] = ttnn.add(ta, ta)
    elif first == "mul":
        out["first_mul"] = ttnn.multiply(ta, ta)
    elif first == "square":
        out["first_square"] = ttnn.square(ta)
    elif first == "mul2":
        out["first_mul2"] = ttnn.multiply(ta, ttnn.neg(ta))
    elif first == "muladd":
        out["first_muladd"] = ttnn.add(ttnn.multiply(ta, ta), ta)
    out["matmul"] = ttnn.matmul(ta, tb, compute_kernel_config=hifi2)
    out["matmul_fp32"] = ttnn.matmul(ta, tb, compute_kernel_config=fp32, dtype=ttnn.float32)
    out["silu"] = ttnn.silu(ta)
    out["sin"] = ttnn.sin(ta)
    out["cos"] = ttnn.cos(ta)
    out["exp"] = ttnn.exp(ta)
    out["typecast"] = ttnn.typecast(ta, ttnn.float32)
    out["mul_add"] = ttnn.add(ttnn.multiply(ta, ta), ta)
    out["mul_add2"] = ttnn.add(ttnn.multiply(ta, ta), ta)
    out["mul"] = ttnn.multiply(ta, ta)
    out["add"] = ttnn.add(ta, ta)
    out["sub"] = ttnn.subtract(ta, ttnn.neg(ta))
    out["untilize"] = ttnn.to_layout(ta, ttnn.ROW_MAJOR_LAYOUT)
    out["tilize"] = ttnn.to_layout(out["untilize"], ttnn.TILE_LAYOUT)
    out["concat"] = ttnn.concat([ta, ta], dim=3)
    out["softmax"] = ttnn.softmax(ta, dim=-1)
    out["layer_norm"] = ttnn.layer_norm(ta)
    out["clamp"] = ttnn.clamp(ta, -0.5, 0.5)
    res = {k: ttnn.to_torch(v) for k, v in out.items()}
    if "first_mul" in res and "--detail" in sys.argv:
        r = res["first_mul"].float()[0, 0, :4, :16]
        print("   first_mul rows0-3:", [[f"{v:.3g}" for v in row] for row in r.tolist()])
        print("   a^2       rows0-3:", [[f"{v:.3g}" for v in row] for row in (a.float()[0, 0, :4, :16] ** 2).tolist()])
    gold = (a.float() * a.float()).bfloat16().float() + a.float()
    for k in ("mul_add", "mul_add2"):
        e = (res[k].float() - gold).abs()
        print(f"golden {k}: max abs err {e.max():.3e}, n>0.1 {int((e > 0.1).sum())}")
        if e.max() > 0.1 and "--detail" in sys.argv:
            r = res[k].float()[0, 0, :16, :16]
            af = a.float()[0, 0, :16, :16]
            bad = e[0, 0, :16, :16] > 0.1
            print("   bad map face0 tile0:")
            for i in range(16):
                print("    ", "".join("x" if v else "." for v in bad[i].tolist()))
            idx = bad.nonzero()[:12].tolist()
            for i, j in idx:
                print(
                    f"    [{i},{j}] a={af[i,j]:.4f} got={r[i,j]:.4f} gold={gold[0,0,i,j]:.4f} got-a={r[i,j]-af[i,j]:.4f} a2={af[i,j]**2:.4f}"
                )
            # tile 2 face 0 too
            r2 = res[k].float()[0, 0, :16, 64:80]
            e2 = e[0, 0, :16, 64:80] > 0.1
            print("   bad map face0 tile2:", ["".join("x" if v else "." for v in row) for row in e2.tolist()])
    return res


def main():
    a = [int(x) for x in sys.argv[1:] if not x.startswith("--")]
    heads, q_rows, keys, valid = a[:4]
    gx, gy, bufs, readers, repeats, debug = (a[4:] + [7, 9, 16, 2, 3, 0][len(a) - 4 :])[:6]
    depth = int(next((x[8:] for x in sys.argv if x.startswith("--depth=")), "4"))
    assert q_rows % 64 == 0 and keys % 64 == 0 and valid <= keys
    torch.manual_seed(0)
    q = torch.randn(1, heads, q_rows, 128).bfloat16()
    k = torch.randn(1, heads, keys, 128).bfloat16()
    v = torch.randn(1, heads, keys, 128).bfloat16()
    k[:, :, valid:] = 0
    v[:, :, valid:] = 0

    dev = ttnn.open_device(device_id=0)
    try:
        dram = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        face1k = "--face1k" in sys.argv
        if face1k:
            debug |= 64
        kk, vt = (
            (k, tile_transposed(v)) if not face1k else (faces_column_major(k), faces_column_major(tile_transposed(v)))
        )
        if "--vtfull" in sys.argv:  # V^T as ttnn.transpose(v, -2, -1) gives it: [heads][128][keys]
            debug |= 256
            vt = v.transpose(-2, -1).contiguous()
        tq, tk, tvt = ttnn.from_torch(q, **dram), ttnn.from_torch(kk, **dram), ttnn.from_torch(vt, **dram)
        to = ttnn.from_torch(torch.zeros_like(q), **dram)
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        scratch_bytes = SCRATCH_FIXED + bufs * BLOCK
        rows = (scratch_bytes + 2047) // 2048
        spec = ttnn.ShardSpec(grid, [rows, 1024], ttnn.ShardOrientation.ROW_MAJOR)
        smc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
        tscr = ttnn.from_torch(
            torch.zeros(rows * gx * gy, 1024).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=smc,
        )
        phys = {(x, y): dev.worker_core_from_logical_core(ttnn.CoreCoord(x, y)) for x in range(gx) for y in range(gy)}

        # Feed groups: rectangles of whole columns ("c1": one column each; "c2": pairs of columns, a last single
        # column if gx is odd). Each group runs rounds (head, first query block); its cores take consecutive blocks.
        layout = next((x[3:] for x in sys.argv if x.startswith("--c")), "1")
        width = int(layout)
        groups = []
        x = 0
        while x < gx:
            w = min(width, gx - x)
            groups.append((x, x + w - 1))
            x += w
        qb = q_rows // 64
        group_rounds = [[] for _ in groups]
        sizes = [(g[1] - g[0] + 1) * gy for g in groups]
        if len(set(sizes)) == 1:
            i = 0
            for h in range(heads):
                for base in range(0, qb, sizes[0]):
                    group_rounds[i % len(groups)].append((h, base))
                    i += 1
        else:  # whole heads, greedily to the group that ends soonest
            for h in range(heads):
                gi = min(range(len(groups)), key=lambda i: len(group_rounds[i]) + -(-qb // sizes[i]))
                group_rounds[gi] += [(h, base) for base in range(0, qb, sizes[gi])]
        print("group rounds:", [len(r) for r in group_rounds], "sizes:", sizes)
        print(
            "group rects (physical):",
            [((phys[(ga, 0)].x, phys[(ga, 0)].y), (phys[(gb, gy - 1)].x, phys[(gb, gy - 1)].y)) for ga, gb in groups],
        )
        core_group = {}
        for gi, (ga, gb) in enumerate(groups):
            idx = 0
            for yy in range(gy):
                for xx in range(ga, gb + 1):
                    core_group[(xx, yy)] = (gi, idx)
                    idx += 1

        def reader_cores(gi):
            ga, gb = groups[gi]
            corners = [(ga, 0), (gb, gy - 1), (gb, 0), (ga, gy - 1), (ga, gy // 2), (gb, gy // 2 - 1)]
            out = []
            for c in corners:
                if c not in out:
                    out.append(c)
            return out[:readers]

        castio = next((x[9:] for x in sys.argv if x.startswith("--castio=")), "")

        def core_flags(x, y):
            # --castio=release: reader 0 (top-left) reads on NoC 1 and its I/O kernel casts on NoC 0; reader 1
            # (bottom-right) reads on NoC 0 and casts from its I/O kernel on NoC 1. --castio=s: reader 0's feed reads and
            # casts on NoC 0, reader 1 as in release. One caster per NoC per group, wrap-free: casts unreserved.
            if not castio:
                return 0
            ri = reader_index_of((x, y))
            if ri == IDLE:
                return 0
            if ri == 0:
                return 128 | 16 if castio == "release" else 16
            return 128 | 16

        def args(go, x, y):
            gi, idx = core_group[(x, y)]
            ga, gb = groups[gi]
            rr = group_rounds[gi]
            rounds = [(h << 16) | (base + idx if base + idx < qb else IDLE) for h, base in rr]
            rc = reader_cores(gi)
            readers_xy = []
            for c in rc:
                p = phys[c]
                readers_xy += [p.x, p.y]
            readers_xy += [0] * (2 * MAX_READERS - len(readers_xy))
            p0, p1 = phys[(ga, 0)], phys[(gb, gy - 1)]
            ri = rc.index((x, y)) if (x, y) in rc else IDLE
            base = [
                tscr.buffer_address(),
                go,
                tq.buffer_address(),
                tk.buffer_address(),
                tvt.buffer_address(),
                to.buffer_address(),
                heads,
                q_rows // 32,
                keys // 32,
                valid,
                bufs,
                len(rc),
                ri,
                sizes[gi],
                idx,
                p0.x,
                p0.y,
                p1.x,
                p1.y,
            ]
            assert len(base) == A_READER0_X
            lead = phys[rc[0]]
            return base + readers_xy + [lead.x, lead.y, 0, debug | core_flags(x, y), 0, len(rounds)] + rounds

        all_cores = [(xx, yy) for xx in range(gx) for yy in range(gy)]

        def reader_index_of(c):
            rc = reader_cores(core_group[c][0])
            return rc.index(c) if c in rc else IDLE

        def program(go):
            rt = ttnn.RuntimeArgs()
            for x in range(gx):
                for y in range(gy):
                    rt[x][y] = args(go, x, y)

            def kd(src, config, cores=None):
                if cores is None:
                    ranges, r = grid, rt
                else:
                    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
                    r = ttnn.RuntimeArgs()
                    for xx, yy in cores:
                        r[xx][yy] = args(go, xx, yy)
                return ttnn.KernelDescriptor(
                    kernel_source=KDIR + src,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=ranges,
                    compile_time_args=[],
                    defines=[("WHS_FEED_DEPTH", f"{depth}u")] if "feed" in src else [],
                    runtime_args=r,
                    config=config,
                )

            def dm(proc, noc):
                c = ttnn.DataMovementConfigDescriptor()
                c.processor, c.noc = proc, noc
                return c

            if castio == "release":
                # Reader 0 cores: feed on NoC 1 (reads), I/O on NoC 0 (casts). Everything else default.
                swap = [c for c in all_cores if reader_index_of(c) == 0]
                rest = [c for c in all_cores if c not in swap]
                P, N = ttnn.DataMovementProcessor, ttnn.NOC
                kernels = [
                    kd("wh_sdpa_io.cpp", dm(P.RISCV_0, N.NOC_1), rest),
                    kd("wh_sdpa_feed.cpp", dm(P.RISCV_1, N.NOC_0), rest),
                    kd("wh_sdpa_io.cpp", dm(P.RISCV_0, N.NOC_0), swap),
                    kd("wh_sdpa_feed.cpp", dm(P.RISCV_1, N.NOC_1), swap),
                    kd("wh_sdpa_op.cpp", ttnn.ComputeConfigDescriptor()),
                ]
            elif "--nocsplit" in sys.argv:
                # Odd reader index: the feed (reads and casts) on NoC 1, the I/O on NoC 0, so a group's two readers
                # cast over different networks.
                swap = [c for c in all_cores if reader_index_of(c) % 2 == 1 and reader_index_of(c) != IDLE]
                rest = [c for c in all_cores if c not in swap]
                P, N = ttnn.DataMovementProcessor, ttnn.NOC
                kernels = [
                    kd("wh_sdpa_io.cpp", dm(P.RISCV_0, N.NOC_1), rest),
                    kd("wh_sdpa_feed.cpp", dm(P.RISCV_1, N.NOC_0), rest),
                    kd("wh_sdpa_io.cpp", dm(P.RISCV_0, N.NOC_0), swap),
                    kd("wh_sdpa_feed.cpp", dm(P.RISCV_1, N.NOC_1), swap),
                    kd("wh_sdpa_op.cpp", ttnn.ComputeConfigDescriptor()),
                ]
            else:
                kernels = [
                    kd("wh_sdpa_io.cpp", ttnn.WriterConfigDescriptor()),
                    kd("wh_sdpa_feed.cpp", ttnn.ReaderConfigDescriptor()),
                    kd("wh_sdpa_op.cpp", ttnn.ComputeConfigDescriptor()),
                ]
            sem = ttnn.SemaphoreDescriptor(id=0, core_ranges=grid, initial_value=0)
            return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[sem], cbs=[])

        before = leak_ops(dev) if "--leak" in sys.argv else None
        times = []
        for go in range(1, repeats + 1):
            pd = program(go)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            if "--skip-op" not in sys.argv:
                ttnn.generic_op([tq, tk, tvt, tscr, to], pd)
            ttnn.synchronize_device(dev)
            times.append(time.perf_counter() - t0)
            print(f"run {go}: {times[-1] * 1e3:.2f} ms", flush=True)
        if before is not None:
            after = leak_ops(dev)
            for name in before:
                same = torch.equal(before[name], after[name])
                if same:
                    print(f"leak {name}: identical")
                else:
                    x, y = before[name].float().flatten(), after[name].float().flatten()
                    pcc = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
                    print(f"leak {name}: DIFFERS, PCC {pcc:.6f}, max abs {(x - y).abs().max():.3e}")
                    d = (before[name].float() - after[name].float()).abs().reshape(-1, before[name].shape[-1])
                    r, c = d.shape
                    t = d[: r // 32 * 32, : c // 32 * 32].reshape(r // 32, 32, c // 32, 32).amax(dim=(1, 3)) > 0
                    f = (
                        d[: r // 32 * 32, : c // 32 * 32].reshape(r // 32, 2, 16, c // 32, 2, 16).amax(dim=(0, 2, 3, 5))
                        > 0
                    )
                    print(
                        f"   tiles differing {int(t.sum())}/{t.numel()}; faces (row-half x col-half) {f.int().tolist()}; "
                        f"elements {int((d > 0).sum())}/{d.numel()}"
                    )
                    print("   tile map rows:", ["".join("x" if v else "." for v in row) for row in t.tolist()])
        o = ttnn.to_torch(to).float()
        scr = ttnn.to_torch(tscr).view(torch.int16).view(-1).view(torch.int32)  # raw words
        per_core = rows * 2048 // 4
        stats = scr.view(gx * gy, per_core)[:, (64 + 6) * 4 : (64 + 6) * 4 + 16 : 4].tolist()
        kv_wait = max(s[0] for s in stats) & 0xFFFFFFFF
        print(
            f"max TRISC0 K/V wait {kv_wait / 1e6:.2f} Mcycles; feed cycles (reader 0 of col 0) {stats[0][3] / 1e6:.2f} M"
        )
        feeds = []
        for gi in range(len(groups)):
            for ri, c in enumerate(reader_cores(gi)):
                cidx = c[1] * gx + c[0]  # shards go to cores row-major
                feeds.append(f"{c}:{(stats[cidx][3] & 0xFFFFFFFF) / 1e6:.0f}")
        print("feed Mcycles per reader:", " ".join(feeds))
        o_tt = None
        if valid == keys and "--ttnn" in sys.argv:
            pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(gx, gy), q_chunk_size=256, k_chunk_size=512, exp_approx_mode=False
            )
            ck = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False
            )
            tv = ttnn.from_torch(v, **dram)
            ttnn.deallocate(tscr)
            t0 = time.perf_counter()
            r = ttnn.transformer.scaled_dot_product_attention(
                tq, tk, tv, is_causal=False, program_config=pc, compute_kernel_config=ck
            )
            ttnn.synchronize_device(dev)
            print(f"ttnn sdpa: {(time.perf_counter() - t0) * 1e3:.2f} ms (first call)")
            o_tt = ttnn.to_torch(r).float()
        hs = sorted(set([0, heads - 1]))
        for h in hs:
            rows = torch.cat([torch.arange(0, min(128, q_rows)), torch.arange(max(0, q_rows - 128), q_rows)]).unique()
            ref = (
                torch.softmax(q[0, h, rows].float() @ k[0, h, :valid].float().T / 128**0.5, -1)
                @ v[0, h, :valid].float()
            )
            got = o[0, h, rows]
            pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
            rel = ((got - ref).norm() / ref.norm()).item()
            print(f"head {h}: PCC {pcc:.6f} rel {rel:.3e} max abs {(got - ref).abs().max():.3e}")
            if o_tt is not None:
                g2 = o_tt[0, h, rows]
                pcc2 = torch.corrcoef(torch.stack([g2.flatten(), ref.flatten()]))[0, 1].item()
                print(f"  ttnn sdpa head {h}: PCC {pcc2:.6f} rel {((g2 - ref).norm() / ref.norm()).item():.3e}")
        blocks = (valid + 63) // 64
        steps = max(len(r) for r in group_rounds) * blocks
        print(
            f"H{heads} Nq {q_rows} keys {keys} valid {valid} grid {gx}x{gy} bufs {bufs} readers {readers}: "
            f"median {sorted(times)[len(times) // 2] * 1e3:.2f} ms, {steps} block steps on the busiest core"
        )
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
