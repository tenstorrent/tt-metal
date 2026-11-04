"""Skinny-matmul chains on the 8 DRAM-bank cores of a chip (default off; call-time flags in tt/attention.py, tt/moe_block.py).

The weights stream from the cores' own DRAM banks through ONE L1 ring (blaze DRAMStreamingMatmul reader, 8 x 32 tiny in0 tiles, LoFi custom_mm with fp32 dest accumulation); the
intermediate of a chain is exchanged between the cores over the NoC (all-gather) and never leaves L1.  All circular buffers are transient (no persistent L1 tensor).

Chain A (``SkinnyWoAB``): attention tail  c [1,1,32,4608] (std DRAM tiles, rows 0..T-1 valid) -> wo_a (K = 4096 of 4608: the 512 leading columns of c carry the kv row, their weight rows are
zero) -> wo_b -> part [1,1,32,5120] bf16 std DRAM tiles (rows 0..7 / 0..15 valid), the input of the fuse-tail / of the all-reduce.
"""

import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attn_fused import compact_pb as AFPB

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skinny_kernels")
TB8 = 1088  # bfp8 tile bytes
NB = 8  # DRAM banks


def enabled(name):
    return os.environ.get(name, "0") == "1"


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _hash(tag, *parts):
    return (tag << 40) | (hash(parts) & ((1 << 40) - 1))


def page_geometry(sbk):
    """blaze get_max_page_size_and_num_pages (NoC max packet 16384 B on Blackhole), bfp8 tiles."""
    total = sbk * TB8
    page = (16384 // TB8) * TB8
    while total % page != 0 and page >= TB8:
        page -= TB8
    return page, total // page


def shuffle_tiles(tensor, num_banks, ct=1):
    """Reorder the tiles of each of ``num_banks`` N-shards of [*, K, N] from row-major to column-major (the K tiles of one N column contiguous), tile = 32x32 (blaze layout)."""
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
    i = torch.arange(nt)
    p_, rem_ = i // (kt * ct), i % (
        kt * ct
    )  # ct = 1: the K tiles of one column contiguous; ct = 2: groups of ct columns, k-major inside the group (tile (k, c) at k * ct + c)
    src = (rem_ // ct) * pnt + p_ * ct + (rem_ % ct)
    tiles = tiles[:, src]
    tiles = tiles.reshape(-1, kt, pnt, ts, ts).permute(0, 1, 3, 2, 4).contiguous().reshape(-1, K, per_n)
    out = tiles.reshape(b, num_banks, K, per_n).permute(0, 2, 1, 3).contiguous().reshape(b, K, N)
    return out.reshape(shp)


def _bank_mem(md, K, n_chip, dtype=ttnn.bfloat8_b):
    dram_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(md.dram_grid_size().x - 1, md.dram_grid_size().y - 1))}
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_crs, [K, n_chip // NB], ttnn.ShardOrientation.ROW_MAJOR),
    )


def bank_weight_cols(md, w, dtype=ttnn.bfloat8_b, ct=1):
    """w [K, N_total] (stored [in, out]); the columns are split over the mesh columns (N_total / cols per chip), each chip's slice goes to the 8 banks in blaze layout."""
    rows, cols = tuple(md.shape)
    K, N = int(w.shape[-2]), int(w.shape[-1])
    n_chip = N // cols
    assert n_chip % (32 * NB) == 0
    ws = shuffle_tiles(w.float().reshape(1, 1, K, N), NB * cols, ct)
    return ttnn.from_torch(
        ws.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=_bank_mem(md, K, n_chip),
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(rows, cols)),
    )


def bank_weight_rows(md, w, dtype=ttnn.bfloat8_b, ct=1):
    """w [K_total, N]; the rows are split over the mesh columns (K_total / cols per chip), each chip's [K, N] slice goes to the 8 banks (N split in 8) in blaze layout."""
    rows, cols = tuple(md.shape)
    Kt, N = int(w.shape[-2]), int(w.shape[-1])
    K = Kt // cols
    assert N % (32 * NB) == 0
    ws = shuffle_tiles(w.float().reshape(cols, K, N), NB, ct).reshape(1, 1, Kt, N)
    return ttnn.from_torch(
        ws.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=_bank_mem(md, K, N),
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 2), mesh_shape=(rows, cols)),
    )


def bank_weight_rep(md, w, dtype=ttnn.bfloat8_b):
    """w [K, N] replicated on every chip, N split over the 8 banks."""
    K, N = int(w.shape[-2]), int(w.shape[-1])
    assert N % (32 * NB) == 0
    ws = shuffle_tiles(w.float().reshape(1, 1, K, N), NB)
    return ttnn.from_torch(
        ws.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=_bank_mem(md, K, N),
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def _tiny_cb(core_set, idx, pages, rows=8):
    return ttnn.CBDescriptor(
        total_size=pages * rows * 32 * 2,
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=idx,
                data_format=ttnn.bfloat16,
                page_size=rows * 32 * 2,
                tile=ttnn.TileDescriptor(ttnn.Tile([rows, 32])),
            )
        ],
    )


def _ring_cb(core_set, idx, tiles):
    return ttnn.CBDescriptor(
        total_size=tiles * TB8,
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=idx,
                data_format=ttnn.bfloat8_b,
                page_size=TB8,
                tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
            )
        ],
    )


def bank_cores(md):
    return list(ttnn.get_optimal_dram_bank_to_logical_worker_assignment(md, ttnn.NOC.RISCV_0_default))


def _vcs(cores):
    banks, vcs = [], []
    for b, c in enumerate(cores):
        vc = b & 0x3
        for j in range(b):
            if cores[j].y == c.y and (banks[j] & 0x3) == (b & 0x3):
                vc = (vc + 1) & 0x3
                break
        banks.append(b)
        vcs.append(vc)
    return vcs


def _core_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])


def reader_ct(cb_w, bank, vc, phases):
    """phases: list of up to 3 dicts(addr, sbk, pn, nsub, nbuf)."""
    ct = [cb_w, bank, vc]
    for p in range(3):
        if p < len(phases) and phases[p]["pn"] > 0:
            ph = phases[p]
            page, npg = page_geometry(ph["sbk"])
            ct += [ph["addr"], page, npg, ph["sbk"], ph["pn"], ph["sbk"] * TB8, ph["nsub"], ph["nbuf"], 0]
        else:
            ct += [0, 0, 0, 0, 0, 0, 0, 0, 0]
    return ct


class SkinnyWoAB:
    """wo_a -> wo_b chain.  wo_a [4096, 8 * 1024] (the usual permuted layout WITHOUT the 512 zero rows), wo_b [8 * 1024, 5120] (host tensors, [in, out])."""

    KT1 = 128  # 4096 / 32 real input tiles
    KSKIP = 16  # leading tiles of c that belong to the (zero-weight) kv row
    KT2 = 32  # 1024 / 32
    N1T = 32  # wo_a output column tiles per chip
    N2T = 160  # wo_b output column tiles per chip

    def __init__(self, md, wo_a, wo_b):
        self.md = md
        self.ctd = int(
            os.environ.get("DSV41_SKINNY_CT", "1")
        )  # ct_dim: weight columns per custom_mm call (k-major column groups in DRAM; 8.1 ns vs 10.4 ns per tile product)
        self.wa = bank_weight_cols(md, wo_a, ct=self.ctd)
        self.wb = bank_weight_rows(md, wo_b, ct=self.ctd)
        self.cores = bank_cores(md)
        assert len(self.cores) == NB
        self.vcs = _vcs(self.cores)
        self.phys = [md.worker_core_from_logical_core(ttnn.CoreCoord(c.x, c.y)) for c in self.cores]
        self.ring = int(os.environ.get("DSV41_SKINNY_RING", "384"))
        self.sbk1 = int(os.environ.get("DSV41_SKINNY_SBK1", "64"))
        self.sbk2 = int(os.environ.get("DSV41_SKINNY_SBK2", "32"))
        self.fp32 = int(os.environ.get("DSV41_SKINNY_FP32", "1"))
        self.pair2 = os.environ.get("DSV41_SKINNY_PAIR2", "0") == "1"
        self.parts = {}

    def __call__(self, c, T, part=None):
        md = self.md
        assert T <= 16
        MT, G = (
            1 if T <= 8 else 2
        ), 1  # T = 9..16: two 8-row tiles (compute-bound on one core per bank, see the 2-cores-per-bank variant)
        xc = (
            c.layout == ttnn.ROW_MAJOR_LAYOUT
        )  # compact x from rope_concat(compact=True): [5 * MT, 8192] bf16 RM, tiny tiles, 160 per row tile
        KX = 160 if xc else self.KT1  # tiles per row tile in cb_x
        NC = NB * G
        PN1 = self.N1T // NC  # 4
        PN2 = self.N2T // NC  # 20
        cap = (4 if self.fp32 else 8) // MT  # dest tiles per sub-block = MT * SBW
        SBW1, SBW2 = min(PN1, cap), min(PN2, cap)
        while PN1 % SBW1:
            SBW1 -= 1
        while PN2 % SBW2:
            SBW2 -= 1
        cores = self.cores
        core_set = _core_set(cores)
        if part is None:
            part = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 32, 5120]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
            )
        sbk1, sbk2, ring = self.sbk1, self.sbk2, self.ring
        nsub1, nsub2 = self.KT1 // sbk1, self.KT2 // sbk2
        ctd = self.ctd
        pair = (
            2 if self.pair2 else 1
        ) * ctd  # reader blocks: ctd columns (k-major group) x sbk K tiles; phase 2 optionally two groups; ring / block <= 15 transaction ids
        b1 = ctd * sbk1
        assert (
            ring % b1 == 0
            and ring % (pair * sbk2) == 0
            and (PN1 * self.KT1) % (pair * sbk2) == 0
            and PN2 % pair == 0
            and PN1 % ctd == 0
        )
        nb1, nb2 = ring // b1, ring // (pair * sbk2)
        assert nb1 <= 15 and nb2 <= 15, (nb1, nb2)
        CB_X, CB_W, CB_Y, CB_YG, CB_O = range(5)
        cbs = [
            _tiny_cb(core_set, CB_X, KX * MT),
            _ring_cb(core_set, CB_W, ring),
            _tiny_cb(core_set, CB_Y, PN1 * MT),
            _tiny_cb(core_set, CB_YG, self.KT2 * MT),
            _tiny_cb(core_set, CB_O, PN2 * MT),
        ]
        readers = []
        for i, cc in enumerate(cores):
            ph = [
                dict(addr=self.wa.buffer_address(), sbk=b1, pn=PN1 // ctd, nsub=nsub1, nbuf=nb1),
                dict(addr=self.wb.buffer_address(), sbk=pair * sbk2, pn=PN2 // pair, nsub=nsub2, nbuf=nb2),
            ]
            readers.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/sk_reader.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=_core_set([cc]),
                    compile_time_args=reader_ct(CB_W, i, self.vcs[i], ph),
                    runtime_args=ttnn.RuntimeArgs(),
                    common_runtime_args=[],
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )
        rt_w = ttnn.RuntimeArgs()
        for i, cc in enumerate(cores):
            rt_w[cc.x][cc.y] = [i * PN1, i * PN2, i]
        common_w = [c.buffer_address(), part.buffer_address()]
        for p in self.phys:
            common_w += [p.x, p.y]
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/sk_woab_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[
                CB_X,
                CB_Y,
                CB_YG,
                CB_O,
                NC,
                PN1,
                KX,
                self.KSKIP,
                self.KT2,
                PN2,
                MT,
                SBW2,
                int(xc),
                SBW1,
                int(AFPB()),
            ]
            + _acc(c)
            + _acc(part),
            runtime_args=rt_w,
            common_runtime_args=common_w,
            config=ttnn.WriterConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/sk_woab_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[
                CB_X,
                CB_W,
                CB_Y,
                CB_YG,
                CB_O,
                sbk1,
                PN1,
                SBW1,
                nsub1,
                sbk2,
                PN2,
                SBW2,
                nsub2,
                MT,
                self.fp32,
                KX,
                self.KT2,
                16 if xc else 0,
                ctd,
            ],
            runtime_args=ttnn.RuntimeArgs(),
            common_runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=bool(self.fp32), math_approx_mode=True
            ),
        )
        prog = ttnn.ProgramDescriptor(
            kernels=readers + [writer, compute],
            semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=core_set, initial_value=0)],
            cbs=cbs,
        )
        prog.custom_program_hash = _hash(
            0x5A1,
            MT,
            G,
            xc,
            SBW1,
            SBW2,
            ctd,
            self.fp32,
            ring,
            sbk1,
            sbk2,
            self.wa.buffer_address(),
            self.wb.buffer_address(),
            tuple(_acc(c)),
            tuple(_acc(part)),
        )
        ttnn.generic_op([c, self.wa, self.wb, part], prog)
        return part
