"""Shared-expert chain on the 8 DRAM-bank cores (default off, call-time flag DSV41_SKINNY_SHARED=1, weights built when DSV41_SKINNY_ARM=1 or the flag is set at init).

Per mesh column c the shared expert has 288 intermediate columns (9 tiles): partial_c = (silu(min(x w0_c, L)) * clamp(x w1_c, -L, L)) w2_c   [1,1,T,5120]  (the same maths as tt/shared_expert_tp.py).
Bank core b (8 per chip) owns the gate / up column tile b (K = 160 tiles each), a K slice [20 b, 20 b + 20) of the 9th tile pair (partial sums, all-gathered; every core reduces them and computes
act_8 redundantly), and the down-projection columns [20 b, 20 b + 20) (K = 9 tiles).  Per core 360 + 180 weight tiles (590 KB) = the 4.7 MB of the shared expert per chip, no padding.
SwiGLU runs on one-face 16x16 views of the tiny-tile buffers inside the same program; act tiles and the pair-8 partials are exchanged between the cores over the NoC.
"""

import os
import struct

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import skinny as SK
from models.demos.blackhole.deepseek_v41_flash.tt.skinny import KDIR, NB, TB8, _acc, _core_set, _hash


def bank_seq_weight(md, tiles, dtype=ttnn.bfloat8_b):
    """tiles [cols, 8, T, 32, 32] float: the tile sequence of every (mesh column, bank); tile r of bank b is stored at tile position r of that bank's DRAM shard (shard [T*32, 32])."""
    cols, nb, T = tiles.shape[0], tiles.shape[1], tiles.shape[2]
    rows = md.shape[0]
    g = tiles.permute(2, 3, 0, 1, 4).reshape(T * 32, cols * nb * 32)
    dram_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(md.dram_grid_size().x - 1, md.dram_grid_size().y - 1))}
    )
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_crs, [T * 32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        g.reshape(1, 1, T * 32, cols * nb * 32).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(rows, cols)),
    )


def _fb(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _alias_cb(core_set, idx_tiny, idx16, pages):
    """one L1 buffer of `pages` 512 B pages seen as tiny 8x32 tiles (idx_tiny) and as one-face 16x16 tiles (idx16)."""
    return ttnn.CBDescriptor(
        total_size=pages * 512,
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=idx_tiny,
                data_format=ttnn.bfloat16,
                page_size=512,
                tile=ttnn.TileDescriptor(ttnn.Tile([8, 32])),
            ),
            ttnn.CBFormatDescriptor(
                buffer_index=idx16,
                data_format=ttnn.bfloat16,
                page_size=512,
                tile=ttnn.TileDescriptor(ttnn.Tile([16, 16])),
            ),
        ],
    )


class SkinnyShared:
    KT = 160  # x tiles
    NCOL = 20  # down columns per core

    def __init__(self, md, w0, w1, w2, limit=10.0):
        """w0 (gate), w1 (up): [1,1,5120,2304]; w2 (down): [1,1,2304,5120] host tensors ([in, out])."""
        self.md = md
        rows, cols = tuple(md.shape)
        assert cols == 8 and w0.shape[-1] == 2304
        self.limit = float(limit)
        ic = 288
        w0 = w0.float().reshape(5120, 2304)
        w1 = w1.float().reshape(5120, 2304)
        w2 = w2.float().reshape(2304, 5120)
        tl = lambda m, k, n: m[32 * k : 32 * k + 32, 32 * n : 32 * n + 32]  # tile (k, n) of a matrix
        w01 = torch.zeros(cols, NB, 360, 32, 32)
        w2t = torch.zeros(cols, NB, 180, 32, 32)
        for c in range(cols):
            g, u, d = w0[:, ic * c : ic * (c + 1)], w1[:, ic * c : ic * (c + 1)], w2[ic * c : ic * (c + 1)]
            for b in range(NB):
                seq = [tl(g, k, b) for k in range(160)] + [tl(u, k, b) for k in range(160)]
                seq += [tl(g, k, 8) for k in range(20 * b, 20 * b + 20)] + [
                    tl(u, k, 8) for k in range(20 * b, 20 * b + 20)
                ]
                w01[c, b] = torch.stack(seq)
                w2t[c, b] = torch.stack([tl(d, k, 20 * b + j) for j in range(20) for k in range(9)])
        self.w01 = bank_seq_weight(md, w01)
        self.w2 = bank_seq_weight(md, w2t)
        self.cores = SK.bank_cores(md)
        self.vcs = SK._vcs(self.cores)
        self.phys = [md.worker_core_from_logical_core(ttnn.CoreCoord(c.x, c.y)) for c in self.cores]
        self.fp32 = int(os.environ.get("DSV41_SKINNY_FP32", "1"))
        self.ring = 360

    def __call__(self, h, T, out=None):
        """h: std tile tensor [1,1,T,5120] bf16 (DRAM or L1) or the compact RM buffer [5, 8192]; -> out [1,1,32,5120] bf16 TILE (rows 0..T-1 valid)."""
        assert T <= 16
        MT = 1 if T <= 8 else 2
        md = self.md
        xc = h.layout == ttnn.ROW_MAJOR_LAYOUT
        cores = self.cores
        core_set = _core_set(cores)
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 32, 5120]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
            )
        ring = self.ring
        (
            CB_X,
            CB_W,
            CB_GU,
            CB_GU16,
            CB_ACTL,
            CB_ACTL16,
            CB_P,
            CB_PRG,
            CB_PRG16,
            CB_PRU,
            CB_PRU16,
            CB_ACT,
            CB_ACT8,
            CB_ACT8_16,
            CB_O,
            CB_ZERO,
            CB_ZERO16,
            CB_TMP,
            CB_TMP16,
        ) = range(19)
        cbs = [
            SK._tiny_cb(core_set, CB_X, self.KT * MT),
            SK._ring_cb(core_set, CB_W, ring),
            _alias_cb(core_set, CB_GU, CB_GU16, 2 * MT),
            _alias_cb(core_set, CB_ACTL, CB_ACTL16, MT),
            SK._tiny_cb(core_set, CB_P, 2 * MT),
            _alias_cb(core_set, CB_PRG, CB_PRG16, NB * MT),
            _alias_cb(core_set, CB_PRU, CB_PRU16, NB * MT),
            SK._tiny_cb(core_set, CB_ACT, 9 * MT),
            _alias_cb(core_set, CB_ACT8, CB_ACT8_16, MT),
            SK._tiny_cb(core_set, CB_O, self.NCOL * MT),
            _alias_cb(core_set, CB_ZERO, CB_ZERO16, 1),
            _alias_cb(core_set, CB_TMP, CB_TMP16, 2),
        ]
        readers = []
        for i, cc in enumerate(cores):
            ph = [
                dict(addr=self.w01.buffer_address(), sbk=40, pn=2, nsub=4, nbuf=ring // 40),
                dict(addr=self.w01.buffer_address() + 320 * TB8, sbk=40, pn=1, nsub=1, nbuf=ring // 40),
                dict(addr=self.w2.buffer_address(), sbk=36, pn=5, nsub=1, nbuf=ring // 36),
            ]
            readers.append(
                ttnn.KernelDescriptor(
                    kernel_source=f"{KDIR}/sk_reader.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=_core_set([cc]),
                    compile_time_args=SK.reader_ct(CB_W, i, self.vcs[i], ph),
                    runtime_args=ttnn.RuntimeArgs(),
                    common_runtime_args=[],
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )
        SBWC = (4 if self.fp32 else 5) // MT
        while self.NCOL % SBWC:
            SBWC -= 1
        rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, cc in enumerate(cores):
            rt_w[cc.x][cc.y] = [i, i * self.NCOL]
            rt_c[cc.x][cc.y] = [i]
        common_w = [h.buffer_address(), out.buffer_address()]
        for p in self.phys:
            common_w += [p.x, p.y]
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/sk_sh_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[
                CB_X,
                CB_ZERO,
                CB_ACTL,
                CB_P,
                CB_PRG,
                CB_PRU,
                CB_ACT,
                CB_ACT8,
                CB_O,
                NB,
                self.KT,
                int(xc),
                self.NCOL,
                SBWC,
                9,
                int(os.environ.get("DSV41_SKINNY_XSPLIT", "1") == "1"),
                int(SK.AFPB()),
                MT,
            ]
            + _acc(h)
            + _acc(out),
            runtime_args=rt_w,
            common_runtime_args=common_w,
            config=ttnn.WriterConfigDescriptor(),
        )
        L = self.limit
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/sk_sh_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[
                CB_X,
                CB_W,
                CB_GU,
                CB_GU16,
                CB_ACTL,
                CB_ACTL16,
                CB_P,
                CB_PRG,
                CB_PRG16,
                CB_PRU,
                CB_PRU16,
                CB_ACT,
                CB_ACT8,
                CB_ACT8_16,
                CB_O,
                CB_ZERO,
                CB_ZERO16,
                CB_TMP,
                CB_TMP16,
                self.KT,
                self.fp32,
                _fb(L),
                _fb(-L),
                _fb(-3.0e38),
                NB,
                20,
                9,
                self.NCOL,
                SBWC,
                MT,
            ],
            runtime_args=rt_c,
            common_runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=bool(self.fp32),
                math_approx_mode=os.environ.get("DSV41_SKINNY_APPROX", "1") == "1",
            ),
        )
        prog = ttnn.ProgramDescriptor(
            kernels=readers + [writer, compute],
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0) for i in range(3)],
            cbs=cbs,
        )
        prog.custom_program_hash = _hash(
            0x5B1,
            MT,
            xc,
            os.environ.get("DSV41_SKINNY_XSPLIT", "1"),
            os.environ.get("DSV41_SKINNY_APPROX", "1"),
            self.fp32,
            ring,
            self.w01.buffer_address(),
            self.w2.buffer_address(),
            tuple(_acc(h)),
            tuple(_acc(out)),
        )
        ttnn.generic_op([h, self.w01, self.w2, out], prog)
        return out
