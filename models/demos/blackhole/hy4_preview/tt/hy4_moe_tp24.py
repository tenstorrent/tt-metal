# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""X40 TP32 routed experts, 24-core decode variant (3 cores per DRAM bank, experts a % 3 == group), layout v1, T = 1..16.
Same host contract as Tp32Experts (tt/hy4_moe_tp32.py); kernels tt/tp32_kernels/tp24_{reader,writer,compute}.cpp.
Core index c = 3 * bank + group; bank core (group 0, the LEADER) is the stock bank core; groups 1, 2 are the two next free cores of its row.
L1 per core (T = 16, MT = 2): ring 332 KB + x 24 + p 32 + pr 197 + act 88 + o 25 + lnd 49 + fin 25 + small ~ 0.78 MB (see cb_bytes)."""
import os

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import skinny as SK
from models.demos.blackhole.deepseek_v41_flash.tt.skinny import _acc, _core_set, _hash
from models.demos.blackhole.deepseek_v41_flash.tt.skinny_shared import _alias_cb, _fb
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import HIDDEN, KB, RING_TILES, TB4, Tp32Experts

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tp32_kernels")


def pick_cores(md, banks):
    """24 logical cores: [3 b + 0] = bank core b, [3 b + 1], [3 b + 2] = nearest free cores in the same row (env HY4_TP24_EXTRA='x.y/x.y/...' overrides all 16 extras in order)"""
    g = md.compute_with_storage_grid_size()
    used = {(c.x, c.y) for c in banks}
    ov = [e for e in os.environ.get("HY4_TP24_EXTRA", "").split("/") if e]
    out = []
    for b, c in enumerate(banks):
        out.append(ttnn.CoreCoord(c.x, c.y))
        for i in range(2):
            if ov:
                x, y = map(int, ov[2 * b + i].split("."))
            else:
                x = y = None
                for dx in (1, -1, 2, -2, 3, -3, 4, -4):
                    cx = c.x + dx
                    if 0 <= cx < g.x and (cx, c.y) not in used:
                        x, y = cx, c.y
                        break
                assert x is not None, f"no free core near bank core {(c.x, c.y)}"
            used.add((x, y))
            out.append(ttnn.CoreCoord(x, y))
    return out


class Tp24Experts(Tp32Experts):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cores8 = self.cores
        self.cores = pick_cores(self.md, self.cores8)
        self.phys = [self.md.worker_core_from_logical_core(ttnn.CoreCoord(c.x, c.y)) for c in self.cores]

    @staticmethod
    def cb_bytes(T):
        R = 2 if T <= 2 else 4 if T <= 4 else 8
        MT = 2 if T > 8 else 1
        nq = min(8 * T, 86)
        nj = (nq + 7) // 8
        return {
            "x": 24 * 64 * R * MT,
            "w": RING_TILES * TB4,
            "p": 32 * 512 * MT,
            "pr": nj * 32 * 512 * MT,
            "sc": nj * 512 * MT,
            "actl": 4 * 512 * MT,
            "act": 2 * nq * 64 * R * MT,
            "o": 24 * 512 * MT,
            "lnd": 48 * 512 * MT,
            "fin": 24 * 512 * MT,
            "meta": 64 + 64 * T,
            "rt": 64 * T,
        }

    def __call__(self, h, route, T, out=None):
        assert 1 <= T <= 16
        md = self.md
        MT = 2 if T > 8 else 1
        R = 2 if T <= 2 else 4 if T <= 4 else 8
        NQ = min(8 * T, 86)  # max experts of one group (e % 3 == g: at most 86 of the 256 ids)
        NJMAX = (NQ + 7) // 8
        cores = self.cores
        core_set = _core_set(cores)
        leaders = _core_set(cores[0::3])
        others = _core_set([c for i, c in enumerate(cores) if i % 3])
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, T, HIDDEN]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
            )
        (CB_X, CB_W, CB_P, CB_PR, CB_PR16, CB_SC, CB_SC16, CB_ACTL, CB_ACTL16, CB_ACT, CB_O, CB_META, CB_RT) = range(13)
        CB_O16, CB_LND, CB_LND16, CB_FIN, CB_FIN16 = range(13, 18)
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
            SK._tiny_cb(core_set, CB_ACT, 2 * NQ * MT, rows=R),
            _alias_cb(core_set, CB_O, CB_O16, 24 * MT),
            _alias_cb(core_set, CB_LND, CB_LND16, 48 * MT),
            _alias_cb(core_set, CB_FIN, CB_FIN16, 24 * MT),
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
            b, g = divmod(i, 3)
            rt_r[cc.x][cc.y] = [b, self.vcs[b], g]
            rt_w[cc.x][cc.y] = [b, KB * b, g]
        reader = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/tp24_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[CB_W, CB_RT, T, RING_TILES, TB4, 6, 0] + _acc(route),
            runtime_args=rt_r,
            common_runtime_args=[self.w01.buffer_address(), self.w2.buffer_address(), route.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        common_w = [h.buffer_address(), route.buffer_address(), out.buffer_address()]
        for q in self.phys:
            common_w += [q.x, q.y]
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/tp24_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set,
            compile_time_args=[
                CB_X,
                CB_P,
                CB_PR,
                CB_SC,
                CB_ACTL,
                CB_ACT,
                CB_O,
                CB_META,
                CB_RT,
                T,
                R,
                NQ,
                CB_LND,
                CB_FIN,
            ]
            + _acc(h)
            + _acc(route)
            + _acc(out),
            runtime_args=rt_w,
            common_runtime_args=common_w,
            config=ttnn.WriterConfigDescriptor(),
        )
        L = self.limit

        def compute(crs, leader):
            return ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/tp24_compute.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=crs,
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
                    0,
                    MT,
                    leader,
                    CB_O16,
                    CB_LND,
                    CB_LND16,
                    CB_FIN,
                    CB_FIN16,
                ],
                runtime_args=ttnn.RuntimeArgs(),
                common_runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=True
                ),
            )

        prog = ttnn.ProgramDescriptor(
            kernels=[reader, writer, compute(leaders, 1), compute(others, 0)],
            semaphores=[ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0) for i in range(3)],
            cbs=cbs,
        )
        prog.custom_program_hash = _hash(
            0x7C24,
            T,
            R,
            self.w01.buffer_address(),
            self.w2.buffer_address(),
            tuple(_acc(h)),
            tuple(_acc(route)),
            tuple(_acc(out)),
        )
        ttnn.generic_op([h, route, self.w01, self.w2, out], prog)
        return out
