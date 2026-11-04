# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention-front POC (persistent kernel, steps 1-3): wqkv matmul -> q/kv RMSNorm -> wq_b matmul in ONE program on the 8 DRAM-bank cores of a chip.

The two weights stream from the cores' own DRAM banks through ONE L1 ring (blaze DRAMStreamingMatmul reader / compute reused), the norm and the qr
exchange run between the two matmuls while the ring prefetches the first wq_b tiles.  M = T (<= 8) tokens as 8x32 tiles.  Kernels in tt/attn_front_kernels/.

Inputs (per chip, replicated): x [T=8 tokens] as tiny 8x32 tiles in DRAM (160 pages of 512 B), wqkv / wq_b in the blaze streaming layout (bfp8 DRAM width-sharded over the
8 banks, tiles shuffled to column-major per shard), gamma std tiles [1,1,32,1792] (= qk_norm.make_gamma), outputs in std 32x32 tile tensors (rows 0..7 valid):
qr [1,1,32,1280], kvn [1,1,32,512], q [1,1,32,4096] (this chip: wq_b local columns, one head per core).
"""

import os
import struct

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _hash

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_front_kernels")
WDT = os.environ.get("AF_WDT", "bfp8")  # weight format of the streamed weights: bfp8 (production) or bfp4 (experiment)
TILE_BYTES_B8 = 1088 if WDT == "bfp8" else 576
WTYPE = ttnn.bfloat8_b if WDT == "bfp8" else ttnn.bfloat4_b
Q_LORA, HEAD_DIM, DIM = 1280, 512, 5120
NQT = Q_LORA // 32
PN1 = 7
PN2 = 16


def _page_geometry(arch_is_bh, sbk):
    """blaze get_max_page_size_and_num_pages (NoC max packet 16384 B on Blackhole)."""
    tile = TILE_BYTES_B8
    total = sbk * tile
    page = (16384 // tile) * tile
    while total % page != 0 and page >= tile:
        page -= tile
    return page, total // page


def _shuffle_tiles(tensor, num_banks, groups=None):
    """Reorder the tiles of each of ``num_banks`` N-shards of [*, K, N] from row-major to column-major (K tiles of one N column contiguous), tile = 32x32 (blaze layout)."""
    import torch

    ts = 32
    shp = tensor.shape
    K, N = shp[-2], shp[-1]
    t = tensor.reshape(-1, K, N)
    b = t.shape[0]
    kt, per_n = K // ts, N // num_banks
    pnt = per_n // ts
    nt = kt * pnt
    t = t.reshape(b, K, num_banks, per_n).permute(0, 2, 1, 3).contiguous().reshape(-1, K, per_n)
    tiles = t.reshape(-1, kt, ts, pnt, ts).permute(0, 1, 3, 2, 4).contiguous().reshape(-1, nt, ts, ts)
    if groups is None:
        i = torch.arange(nt)
        src = (i % kt) * pnt + (i // kt)
    else:
        # grouped layout: groups of ct columns, k-major inside a group, blocks of 80 tiles = 80 / ct k rows
        assert sum(groups) == pnt, (groups, pnt)
        seq, c0 = [], 0
        for ct in groups:
            for k in range(kt):
                for c in range(ct):
                    seq.append(k * pnt + c0 + c)
            c0 += ct
        src = torch.tensor(seq)
    tiles = tiles[:, src]
    tiles = tiles.reshape(-1, kt, pnt, ts, ts).permute(0, 1, 3, 2, 4).contiguous().reshape(-1, K, per_n)
    out = tiles.reshape(b, num_banks, K, per_n).permute(0, 2, 1, 3).contiguous().reshape(b, K, N)
    return out.reshape(shp)


def blaze_weight(md, w, dtype, shard_dim, mesh_cols=1, n_pad=None, groups=None):
    """Upload w [1,1,K,N] (stored [in, out]) in the blaze streaming layout: bfp8 DRAM width-sharded over the 8 banks of every chip, tiles shuffled column-major per shard.
    shard_dim None: replicated; 3: columns split over the mesh columns (each chip's slice is shuffled for its own 8 banks).
    """
    assert dtype == ttnn.bfloat8_b  # the model passes bfp8; the streamed copy uses WTYPE
    dtype = WTYPE
    nb = 8
    rows, cols = tuple(md.shape)
    K, N = int(w.shape[-2]), int(w.shape[-1])
    if n_pad is not None and n_pad > N:
        import torch

        w = torch.nn.functional.pad(w.float(), (0, n_pad - N))
        N = n_pad
    if shard_dim is None:
        n_chip, mapper = N, ttnn.ReplicateTensorToMesh(md)
        ws = _shuffle_tiles(w.float(), nb, groups)
    else:
        n_chip = N // cols
        mapper = ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(rows, cols))
        ws = _shuffle_tiles(w.float(), nb * cols, groups)
    assert n_chip % (32 * nb) == 0
    dram_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(md.dram_grid_size().x - 1, md.dram_grid_size().y - 1))}
    )
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_crs, [K, n_chip // nb], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        ws.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=md, memory_config=mem, mesh_mapper=mapper
    )


def _tiny_cb(core_set, idx, pages, rows=8):
    tile = ttnn.Tile([rows, 32])
    return ttnn.CBDescriptor(
        total_size=pages * rows * 32 * 2,
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=idx, data_format=ttnn.bfloat16, page_size=rows * 32 * 2, tile=ttnn.TileDescriptor(tile)
            )
        ],
    )


def _std_cb(core_set, idx, pages):
    return ttnn.CBDescriptor(
        total_size=pages * 2048,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=2048)],
    )


def core_layout(md, G):
    """(bank cores, all cores in order bank * G + g) of the split configuration."""
    cores = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(md, ttnn.NOC.RISCV_0_default)
    if G == 1:
        return list(cores), list(cores)
    grid = md.compute_with_storage_grid_size()
    used = {(c.x, c.y) for c in cores}
    helpers = []
    for c in cores:
        for dy in (1, -1, 2, -2, 3):
            h = (c.x, c.y + dy)
            if 0 <= h[1] < grid.y and h not in used:
                used.add(h)
                helpers.append(ttnn.CoreCoord(*h))
                break
    allc = []
    for b in range(len(cores)):
        allc += [cores[b], helpers[b]]
    return list(cores), allc


def attn_front(
    x,
    w1,
    w2,
    gamma,
    qr,
    kvn,
    q,
    stage=3,
    eps=1e-20,
    ring_tiles=240,
    use_ds_init=1,
    fp32=0,
    x_split=0,
    x_mcast=0,
    split=1,
    x1=None,
    T=None,
):
    md = x.device()
    cores = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(md, ttnn.NOC.RISCV_0_default)
    nb = len(cores)
    assert nb == 8
    G = split  # compute cores per DRAM bank (they read disjoint columns of the bank shard directly)
    assert G in (1, 2)
    PN1 = 7 if G == 1 else 8 // G  # wqkv column tiles per core (G > 1: weight padded to 8 tiles per bank)
    PN2 = 16 // G
    NKT = 56
    if G > 1:
        grid = md.compute_with_storage_grid_size()
        used = {(c.x, c.y) for c in cores}
        helpers = []
        for c in cores:
            for dy in (1, -1, 2, -2, 3):
                h = (c.x, c.y + dy)
                if 0 <= h[1] < grid.y and h not in used:
                    used.add(h)
                    helpers.append(ttnn.CoreCoord(*h))
                    break
        assert len(helpers) == nb
        allc = [None] * (nb * G)
        for b in range(nb):
            allc[b * G] = cores[b]
            allc[b * G + 1] = helpers[b]
        cores_all = allc
    else:
        cores_all = list(cores)
    nc = len(cores_all)
    cores_bank = list(cores)
    cores = cores_all
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
    phys = [md.worker_core_from_logical_core(ttnn.CoreCoord(c.x, c.y)) for c in cores]
    assert G == 1, "the grouped weight layout supports one core per bank only"
    # grouped layout (af_groups.hpp): the shard is streamed as blocks of 80 tiles in memory order
    p1_sbk, p1_nsub = 80, (PN1 * 160) // 80
    p2_sbk, p2_nsub = 80, (PN2 * 40) // 80
    p1_page, p1_np = _page_geometry(True, p1_sbk)
    p2_page, p2_np = _page_geometry(True, p2_sbk)
    assert ring_tiles % p1_sbk == 0 and ring_tiles % p2_sbk == 0
    nb1, nb2 = ring_tiles // p1_sbk, ring_tiles // p2_sbk
    x_std = tuple(x.get_tile().tile_shape) == (32, 32)
    T = int(x.shape[-2]) if T is None else T
    MT = 1 if T <= 8 else 2
    assert T <= 16 and (MT == 1 or x_std or x1 is not None), (T, x_std)
    x_l1 = (not x_std) and x.memory_config().is_sharded()
    fb = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]

    # CBs (ids)
    CB_X, CB_W, CB_Y, CB_YS, CB_SQ, CB_POUT, CB_PART, CB_SC, CB_EPS, CB_G = range(10)
    CB_EX, CB_RSQ, CB_RSK, CB_XR, CB_N, CB_IN0B, CB_Q, CB_ZERO, CB_X1, CB_IN0B1, CB_FWD = range(10, 21)
    b8 = WTYPE
    w_cb = ttnn.CBDescriptor(
        total_size=ring_tiles * TILE_BYTES_B8,
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_W,
                data_format=b8,
                page_size=TILE_BYTES_B8,
                tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
            )
        ],
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x) if x_l1 else _tiny_cb(core_set, CB_X, DIM // 32),
        w_cb,
        _tiny_cb(core_set, CB_Y, PN1 * MT),
        _std_cb(core_set, CB_YS, PN1),
        _std_cb(core_set, CB_SQ, PN1),
        _std_cb(core_set, CB_POUT, 2),
        _std_cb(core_set, CB_PART, 2 * nc),
        _std_cb(core_set, CB_SC, 1),
        _std_cb(core_set, CB_EPS, 1),
        _std_cb(core_set, CB_G, PN1),
        _std_cb(core_set, CB_EX, 1),
        _std_cb(core_set, CB_RSQ, 1),
        _std_cb(core_set, CB_RSK, 1),
        _std_cb(core_set, CB_XR, 1),
        _std_cb(core_set, CB_N, PN1),
        _tiny_cb(core_set, CB_IN0B, NQT),
        _tiny_cb(core_set, CB_Q, PN2 * MT),
        _std_cb(core_set, CB_ZERO, 1),
    ]
    if G == 2:
        assert ring_tiles == 240
        cbs.append(
            ttnn.CBDescriptor(
                total_size=3 * 40 * TILE_BYTES_B8,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_FWD,
                        data_format=b8,
                        page_size=TILE_BYTES_B8,
                        tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
                    )
                ],
            )
        )
    if MT == 2:
        cbs += [
            ttnn.cb_descriptor_from_sharded_tensor(CB_X1, x1)
            if x1 is not None
            else _tiny_cb(core_set, CB_X1, DIM // 32),
            _tiny_cb(core_set, CB_IN0B1, NQT),
        ]
    # reader (NCRISC): one kernel per core (bank id / vc are compile-time in the blaze reader)
    readers = []
    banks, vcs = [], []
    for b, c in enumerate(cores_bank):
        vc = b & 0x3
        for j in range(b):
            if cores_bank[j].y == c.y and (banks[j] & 0x3) == (b & 0x3):
                vc = (vc + 1) & 0x3
                break
        banks.append(b)
        vcs.append(vc)
    rt_r = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y))])
        if G == 2:
            partner = phys[i ^ 1]
            rt_r = ttnn.RuntimeArgs()
            rt_r[c.x][c.y] = [partner.x, partner.y]
            if i % 2 == 1:
                readers.append(
                    ttnn.KernelDescriptor(
                        kernel_source=f"{KDIR}/af_recv.cpp",
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=cs,
                        compile_time_args=[CB_W, 3, 4],
                        runtime_args=rt_r,
                        common_runtime_args=[],
                        config=ttnn.ReaderConfigDescriptor(),
                    )
                )
                continue
            readers.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/af_reader2.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cs,
                    compile_time_args=[
                        CB_W,
                        CB_FWD,
                        i // G,
                        vcs[i // G],
                        w1.buffer_address(),
                        w2.buffer_address(),
                        ring_tiles,
                        3,
                        4,
                    ],
                    runtime_args=rt_r,
                    common_runtime_args=[],
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )
            continue
        ct = [
            CB_W,
            i // G,
            (vcs[i // G] + (i % G)) & 3,
            w1.buffer_address() + (i % G) * PN1 * 160 * TILE_BYTES_B8,
            w2.buffer_address() + (i % G) * PN2 * 40 * TILE_BYTES_B8,
            p1_page,
            p1_np,
            p1_sbk,
            1,
            p1_sbk * TILE_BYTES_B8,
            p1_nsub,
            p2_page,
            p2_np,
            p2_sbk,
            1,
            p2_sbk * TILE_BYTES_B8,
            p2_nsub,
            nb1,
            nb2,
            stage,
        ]
        readers.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/af_reader.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cs,
                compile_time_args=ct,
                runtime_args=ttnn.RuntimeArgs(),
                common_runtime_args=[],
                config=ttnn.ReaderConfigDescriptor(),
            )
        )
    rt_w = ttnn.RuntimeArgs()
    rt_c = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rt_w[c.x][c.y] = [i]
        rt_c[c.x][c.y] = [i]
    common_w = [
        x.buffer_address(),
        gamma.buffer_address(),
        qr.buffer_address(),
        kvn.buffer_address(),
        q.buffer_address(),
    ]
    for p in phys:
        common_w += [p.x, p.y]
    rects = {}
    for p in phys:
        lo, hi = rects.get(p.x, (p.y, p.y))
        rects[p.x] = (min(lo, p.y), max(hi, p.y))
    gridsz = md.compute_with_storage_grid_size()
    allw = [md.worker_core_from_logical_core(ttnn.CoreCoord(i, j)) for i in range(gridsz.x) for j in range(gridsz.y)]
    common_w += [len(rects)]
    for rx, (lo, hi) in sorted(rects.items()):
        cnt = sum(1 for w_ in allw if w_.x == rx and lo <= w_.y <= hi)
        if x_mcast:
            assert cnt == hi - lo + 1, (rx, lo, hi, cnt)
        common_w += [rx, lo, hi, cnt]
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/af_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[
            CB_X,
            CB_Y,
            CB_YS,
            CB_POUT,
            CB_PART,
            CB_SC,
            CB_EPS,
            CB_G,
            CB_N,
            CB_IN0B,
            CB_Q,
            CB_ZERO,
            nc,
            PN1,
            NQT,
            DIM // 32,
            PN2,
            stage
            | (16 if x_l1 else 0)
            | (32 if x_std else 0)
            | (64 if (x_std and x_split) else 0)
            | (128 if (x_std and x_split and x_mcast) else 0),
            fb(eps),
        ]
        + _acc(x)
        + _acc(gamma)
        + _acc(qr)
        + _acc(kvn)
        + _acc(q)
        + [MT, CB_X1, CB_IN0B1, NKT, G],
        runtime_args=rt_w,
        common_runtime_args=common_w,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/af_compute.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=[
            CB_X,
            CB_W,
            CB_Y,
            CB_YS,
            CB_SQ,
            CB_POUT,
            CB_PART,
            CB_SC,
            CB_EPS,
            CB_G,
            CB_EX,
            CB_RSQ,
            CB_RSK,
            CB_XR,
            CB_N,
            CB_IN0B,
            CB_Q,
            CB_ZERO,
            nc,
            PN1,
            NQT,
            p1_sbk,
            p1_nsub,
            PN1
            if (not fp32 or PN1 <= 4)
            else 1,  # subblock_w of mm1 = per_core_n (7 <= 8 dest tiles; fp32 dest: 4 max, 7 is prime)
            p2_sbk,
            p2_nsub,
            PN2,
            8 if not fp32 else 4,  # subblock_w of mm2
            stage,
            fb(1.0 / Q_LORA),
            fb(1.0 / HEAD_DIM),
            use_ds_init,
            fb(eps),
            fp32,
            MT,
            CB_X1,
            CB_IN0B1,
            NKT,
            1 if (x_std and x_split) else 0,
        ],
        runtime_args=rt_c,
        common_runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=bool(fp32), math_approx_mode=True
        ),
    )
    prog = ttnn.ProgramDescriptor(
        kernels=readers + [writer, compute],
        semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0) for i in range(5 + 8)],
        cbs=cbs,
    )
    prog.custom_program_hash = _hash(
        0xAF1,
        G,
        2 if G == 2 else 0,
        MT,
        stage,
        fp32,
        x_l1,
        x_split,
        x_mcast,
        x_std,
        ring_tiles,
        use_ds_init,
        w1.buffer_address(),
        w2.buffer_address(),
        tuple(_acc(x)),
        tuple(_acc(q)),
    )
    ttnn.generic_op([x, w1, w2, gamma, qr, kvn, q] + ([x1] if x1 is not None else []), prog)
    return qr, kvn, q
