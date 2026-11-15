# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""X40 TP32 routed experts for decode (default OFF, new, JIT only): host side of tt/tp32_kernels (layout contract: program/moe-tp32/LAYOUT.md, tt/hy4_moe_tp32_layout.py).

Tp32Experts(md, w0, w1, w2): w* torch [E=256, 6144, 2048] / [256, 2048, 6144] ([in, out]); chip rank c = mesh_row * cols + mesh_col holds intermediate columns [64 c, 64 c + 64) of every expert.
__call__(x, route, T): x [1,1,T,6144] std TILE bf16 DRAM (replicated), route [1,1,1,16 T] uint32 ROW_MAJOR DRAM (replicated; per token 8 ids + 8 bf16 score bits), T <= 8
-> out [1,1,T,6144] bf16 TILE: this chip's partial sum over its 1/32 of every selected expert (the caller reduces over the chips).
Persistent L1 per core: NONE (every buffer is a program circular buffer, see cb_bytes()); transient CB total per core ~ 0.4 MB.
"""
import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import skinny as SK
from models.demos.blackhole.deepseek_v41_flash.tt.skinny import _acc, _core_set, _hash
from models.demos.blackhole.deepseek_v41_flash.tt.skinny_shared import _alias_cb, _fb

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tp32_kernels")
HIDDEN, INTER, NEXP, TP = 6144, 2048, 256, 32
KB, ICOL = 24, 64
TB4 = 576
RING_TILES = 576  # 6 x 96 = 72 x 8
LIMIT = 10.0


def pack_chips(w0, w1, w2, chips):
    """-> W01 [len(chips), 8, 256 * 96, 32, 32], W2 [len(chips), 8, 6 * 256 * 8, 32, 32] (torch, same dtype as w*): the per-bank tile streams of LAYOUT.md."""
    E = w0.shape[0]
    g_all = w0.reshape(E, 192, 32, 64, 32).permute(0, 1, 3, 2, 4)  # [E, k, n, 32, 32]
    u_all = w1.reshape(E, 192, 32, 64, 32).permute(0, 1, 3, 2, 4)
    w01, w2s = [], []
    for c in chips:
        x4 = torch.cat(
            [g_all[:, :, 2 * c : 2 * c + 2], u_all[:, :, 2 * c : 2 * c + 2]], dim=2
        )  # [E, 192, 4, 32, 32] j = g0 g1 u0 u1
        w01.append(x4.reshape(E, 8, KB, 4, 32, 32).permute(1, 0, 2, 3, 4, 5).reshape(8, E * KB * 4, 32, 32))
        d = w2[:, ICOL * c : ICOL * (c + 1), :].reshape(E, 2, 32, 192, 32).permute(0, 1, 3, 2, 4)  # [E, k, n, 32, 32]
        w2s.append(d.reshape(E, 2, 8, 6, 4, 32, 32).permute(2, 3, 0, 1, 4, 5, 6).reshape(8, 6 * E * 8, 32, 32))
    return torch.stack(w01), torch.stack(w2s)


def bank_tiles_tensor(md, tiles, dtype=ttnn.bfloat4_b):
    """tiles [rows * cols chips (mesh order r * cols + col), 8 banks, T, 32, 32] -> one bank-sharded tensor per chip (shard [T * 32, 32] per bank)."""
    rows, cols = tuple(md.shape)
    Tn = tiles.shape[2]
    g = tiles.reshape(rows, cols, 8, Tn, 32, 32).permute(0, 3, 4, 1, 2, 5).reshape(rows * Tn * 32, cols * 8 * 32)
    dram_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(md.dram_grid_size().x - 1, md.dram_grid_size().y - 1))}
    )
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_crs, [Tn * 32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        g.reshape(1, 1, rows * Tn * 32, cols * 8 * 32).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, 3), mesh_shape=(rows, cols)),
    )


def route_tensor(md, ids, scores):
    """ids [T, 8] int, scores [T, 8] float -> replicated uint32 [1, 1, 1, 16 T] row-major DRAM tensor"""
    T = ids.shape[0]
    sbits = scores.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    r = torch.cat([ids.to(torch.int32), sbits], dim=1).reshape(1, 1, 1, 16 * T).to(torch.int32)
    return ttnn.from_torch(
        r,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=md,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


class Tp32Experts:
    def __init__(
        self, md, w0=None, w1=None, w2=None, w01_t=None, w2_t=None, limit=LIMIT, remap=None, chip0=0, layer=None
    ):
        """chip0: first chip rank of this mesh (a 1x8 slot test holds ranks chip0 .. chip0 + 7 of the 32). layer: layer id, enables the HY4_TP32_CACHE_DIR cache lookup (w0/w1/w2 may then be None)."""
        self.md = md
        self.dbg = int(os.environ.get("HY4_TP32_DBG", "0"))
        rows, cols = tuple(md.shape)
        self.limit = float(limit)
        self.from_cache = False
        if (
            w01_t is None and layer is not None
        ):  # HY4_TP32_CACHE_DIR: packed per-chip shards from the dump (keys checked in hy4_tp32_cache), no checkpoint packing
            from models.demos.blackhole.hy4_preview.tt import hy4_tp32_cache as _C

            ld = _C.load_layer(md, layer, chip0=chip0)
            if ld is not None:
                w01_t, w2_t = ld
                self.from_cache = True
        if w01_t is None:
            assert w0 is not None, "TP32 weights: no cache for this layer and no checkpoint tensors given"
            chips = [chip0 + i for i in range(rows * cols)]
            t01, t2 = pack_chips(w0, w1, w2, chips)
            self.t01, self.t2 = t01, t2  # kept for the CPU reference of the tests
            w01_t, w2_t = bank_tiles_tensor(md, t01), bank_tiles_tensor(md, t2)
        self.w01, self.w2 = w01_t, w2_t
        self.cores = SK.bank_cores(md)
        for ent in filter(None, (remap if remap is not None else os.environ.get("HY4_TP32_REMAP", "")).split("/")):
            b, xy = ent.split("@")
            x, y = xy.split(".")
            self.cores[int(b)] = ttnn.CoreCoord(int(x), int(y))
        self.vcs = SK._vcs(self.cores)
        self.phys = [md.worker_core_from_logical_core(ttnn.CoreCoord(c.x, c.y)) for c in self.cores]

    @staticmethod
    def cb_bytes(T):
        R = 2 if T <= 2 else 4 if T <= 4 else 8
        MT = 2 if T > 8 else 1
        nmax = 8 * T
        nj = (nmax + 7) // 8
        return {
            "x": 24 * 64 * R * MT,
            "w": RING_TILES * TB4,
            "p": 32 * 512 * MT,
            "pr": nj * 32 * 512 * MT,
            "sc": nj * 512 * MT,
            "actl": 4 * 512 * MT,
            "act": 2 * nmax * 64 * R * MT,
            "o": 24 * 512 * MT,
            "meta": 64 + 64 * T,
            "rt": 64 * T,
        }

    def __call__(self, h, route, T, out=None):
        assert 1 <= T <= 16
        MT = 2 if T > 8 else 1
        md = self.md
        R = 2 if T <= 2 else 4 if T <= 4 else 8
        NMAX = 8 * T
        NJMAX = (NMAX + 7) // 8
        cores = self.cores
        core_set = _core_set(cores)
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, T, HIDDEN]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
            )
        (CB_X, CB_W, CB_P, CB_PR, CB_PR16, CB_SC, CB_SC16, CB_ACTL, CB_ACTL16, CB_ACT, CB_O, CB_META, CB_RT) = range(13)
        cbs = [
            SK._tiny_cb(core_set, CB_X, KB * MT, rows=R),
            ttnn.CBDescriptor(
                total_size=RING_TILES * TB4,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_W,
                        data_format=ttnn.bfloat4_b,
                        page_size=TB4,
                        tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])),
                    )
                ],
            ),
            SK._tiny_cb(core_set, CB_P, 32 * MT),
            _alias_cb(core_set, CB_PR, CB_PR16, NJMAX * 32 * MT),
            _alias_cb(core_set, CB_SC, CB_SC16, NJMAX * MT),
            _alias_cb(core_set, CB_ACTL, CB_ACTL16, 4 * MT),
            SK._tiny_cb(core_set, CB_ACT, 2 * NMAX * MT, rows=R),
            SK._tiny_cb(core_set, CB_O, 24 * MT),
            ttnn.CBDescriptor(
                total_size=64 + 64 * T,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_META, data_format=ttnn.uint32, page_size=64 + 64 * T)
                ],
            ),
            ttnn.CBDescriptor(
                total_size=64 * T,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_RT, data_format=ttnn.uint32, page_size=64 * T)
                ],
            ),
        ]
        rt_r, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, cc in enumerate(cores):
            rt_r[cc.x][cc.y] = [i, self.vcs[i]]
            rt_w[cc.x][cc.y] = [i, KB * i]
        reader = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/tp_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[CB_W, CB_RT, T, RING_TILES, TB4, 6, self.dbg] + _acc(route),
            runtime_args=rt_r,
            common_runtime_args=[self.w01.buffer_address(), self.w2.buffer_address(), route.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        common_w = [h.buffer_address(), route.buffer_address(), out.buffer_address()]
        for q in self.phys:
            common_w += [q.x, q.y]
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/tp_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[CB_X, CB_P, CB_PR, CB_SC, CB_ACTL, CB_ACT, CB_O, CB_META, CB_RT, T, R, NMAX, self.dbg]
            + _acc(h)
            + _acc(route)
            + _acc(out),
            runtime_args=rt_w,
            common_runtime_args=common_w,
            config=ttnn.WriterConfigDescriptor(),
        )
        L = self.limit
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/tp_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[
                CB_X,
                CB_W,
                CB_P,
                CB_PR,
                CB_PR16,
                CB_SC,
                CB_SC16,
                CB_ACTL,
                CB_ACTL16,
                CB_ACT,
                CB_O,
                CB_META,
                1,
                _fb(L),
                _fb(-L),
                _fb(-3.0e38),
                self.dbg,
                MT,
                int(os.environ.get("HY4_TP32_DOWN_FP32", "0")),
            ],
            runtime_args=ttnn.RuntimeArgs(),
            common_runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=True
            ),
        )
        prog = ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute],
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0) for i in range(2)],
            cbs=cbs,
        )
        prog.custom_program_hash = _hash(
            0x7C32,
            T,
            int(os.environ.get("HY4_TP32_DOWN_FP32", "0")),
            R,
            self.dbg,
            self.w01.buffer_address(),
            self.w2.buffer_address(),
            tuple(_acc(h)),
            tuple(_acc(route)),
            tuple(_acc(out)),
        )
        ttnn.generic_op([h, route, self.w01, self.w2, out], prog)
        return out
