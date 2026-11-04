# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 16 SP-row sparse attention (M-B2 section 2b), v1: the union GATHER over the fabric (kernels/spa/spa_{read,send}).

Each chip attends its own query rows with all 64 heads, so it needs the latents of the union of its rows' selected pools,
and the cache is token-sharded (the served GLM53_DSA_SHARD layout: 128-token units, owner = unit % N; the kernels only
see the entries, so any layout works). Owner push: every chip walks its request entries (src_row in its local row-major
store, dst_slot in the destination's buffer, dst_pos = the destination's position on the line), reads each pool's 4
rows and writes them into the destination's [U_max x 4, 512] bf16 ROW_MAJOR buffer (same address on every chip): locally
or as one 4 x 1 KiB fabric scatter packet. One arrival word per source on the lane core; the launch ends when every
peer's rows landed. The attention then runs on the compact buffer (v1: ttnn.transformer.sparse_sdpa with the ids
remapped to compact rows; the in-program flash phase follows).
Lines: FABRIC_1D, one instance per mesh axis (`axis`; default the one non-trivial axis of a 1 x n / n x 1 mesh); every
row (axis 1) / column (axis 0) is an independent line, chip (r, c) at position c / r. TP8 on the 2x4 = two instances:
axis 0 into a relay region appended to the local store (the pools a row needs from the other row), then axis 1 into the
compact buffer (spa_gather_unit.py). One lane core per chip (default (0, 0)), one fabric connection per direction."""

from __future__ import annotations

import os

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.fabric_stream import _crs, fabric_connect

KDIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "spa")
CB_STAGE, CB_META, CB_LIST = 0, 1, 2
EB = 64  # entries per list batch (768 B)


class SpaGather:
    def __init__(self, mesh, compact, *, lane=(0, 0), link: int = 0, depth: int = 8, axis: int | None = None):
        self.mesh, self.compact = mesh, compact
        self.R, self.C = int(mesh.shape[0]), int(mesh.shape[1])
        if axis is None:
            assert self.R == 1 or self.C == 1, "a 2-D mesh: pass axis="
            axis = 1 if self.C > 1 else 0
        assert axis in (0, 1), axis
        self.axis = int(axis)
        self.n = self.C if self.axis == 1 else self.R
        self.lane, self.link, self.depth = tuple(lane), int(link), int(depth)
        assert (
            tuple(compact.shape)[-1] == 512
            and compact.layout == ttnn.ROW_MAJOR_LAYOUT
            and compact.dtype == ttnn.bfloat16
        )
        words = 32  # GEN + ARR[n] (L1 words on the lane core, same address on every chip)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(_crs([self.lane]), [1, words], ttnn.ShardOrientation.ROW_MAJOR))  # fmt: skip
        self.sync = ttnn.from_torch(torch.zeros(1, words, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                    device=mesh, memory_config=mc, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
        self.calls = 0

    def _pos(self, r, c):
        return c if self.axis == 1 else r

    def program(self, lat, lst, n_entries: int, r: int, c: int):
        p = self._pos(r, c)
        up, dn = self.n - 1 - p, p
        lane = ttnn.CoreCoord(*self.lane)
        crs = _crs([self.lane])
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        cb = lambda idx, fmt, page, pages: ttnn.CBDescriptor(total_size=page * pages, core_ranges=crs, format_descriptors=[  # noqa: E731
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)])  # fmt: skip
        cbs = [cb(CB_STAGE, ttnn.bfloat16, 4096, self.depth), cb(CB_META, ttnn.uint32, 16, self.depth),
               cb(CB_LIST, ttnn.uint32, EB * 12 + 32, 1)]  # fmt: skip
        rt_r, rt_s = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rt_r[lane.x][lane.y] = [int(lat.buffer_address()), int(lst.buffer_address()), int(n_entries)]
        rt_s[lane.x][lane.y] = [int(self.sync.buffer_address()), p, self.n, up, dn, int(self.compact.buffer_address()),
                                int(n_entries)]  # fmt: skip
        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, src),  # noqa: E731
                                                            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                                            core_ranges=crs, compile_time_args=[int(v) for v in ct],
                                                            runtime_args=rt, config=cfg, defines=[])  # fmt: skip
        kernels = [kd("spa_send.cpp", [CB_STAGE, CB_META] + acc(self.compact), rt_s, ttnn.WriterConfigDescriptor()),
                   kd("spa_read.cpp", [CB_STAGE, CB_META, CB_LIST, EB] + acc(lat) + acc(lst), rt_r, ttnn.ReaderConfigDescriptor())]  # fmt: skip
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        me = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
        step = (lambda d: (r, c + d)) if self.axis == 1 else (lambda d: (r + d, c))  # noqa: E731
        for d, rng in ((1, up), (-1, dn)):  # up first, then down (spa_send.cpp's order)
            if rng:
                peer = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*step(d)))
                fab = fabric_connect(me, peer, self.link, prog, lane)
                prog.kernels[0].runtime_args[lane.x][lane.y].extend(fab)
        return prog

    def __call__(self, lat, lst, n_entries: list):
        """lat: each chip's local row-major latent store [1, 1, T_local, 512] bf16 (sharded); lst: each chip's entries
        uint32 [1, 1, 1, 3 L_max] ROW_MAJOR (sharded); n_entries[chip rank r C + c] -> the compact buffer"""
        mpd = ttnn.MeshProgramDescriptor()
        for r in range(self.R):
            for c in range(self.C):
                m = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(m, m)] = self.program(lat, lst, n_entries[r * self.C + c], r, c)
        ttnn.generic_op([lat, lst, self.compact, self.sync], mpd)
        self.calls += 1
        return self.compact
