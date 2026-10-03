# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device Engram n-gram hash (bit-exact vs the int64 torch reference), one generic_op / one data-movement kernel.

Input  hist [T,16] int32 row-major per device: hist[t][s] = RAW token id s steps back from user t's current token
       (s=0 current), -1 before the sequence start.  The compressed-token map lookup is done in the kernel from a
       device table [V,16] int32 (64 B rows).  Output [T,64] int32; columns [0, n_layers*(W-1)*H) =
       [layer][shift-1][head] ids (offsets already added); use ``DSV41EngramHash.ids(out)`` to view [T,NL,NCOL]."""

import numpy as np
import torch

import ttnn

KDIR = "models/demos/blackhole/deepseek_v41_flash/tt/engram_hash_kernels"
i32 = ttnn.int32


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


class DSV41EngramHash:
    def __init__(self, mesh, token_map, multipliers, primes, offsets, pad_id, T=4, mode=0):
        """token_map [V] int; multipliers [NL,W] int64; primes [NL,W-1,H] int; offsets [NL, (W-1)*H] int; pad_id compressed."""
        self.mesh, self.T, self.mode = mesh, T, mode
        self.NL, self.W = int(multipliers.shape[0]), int(multipliers.shape[1])
        self.H = int(primes.shape[2])
        self.NCOL = (self.W - 1) * self.H
        assert self.NL * self.NCOL <= 64 and 2 * self.NL * self.W + 2 * self.NL * self.NCOL + 1 <= 128
        mult = np.asarray(multipliers, dtype=np.int64).view(np.uint64)
        lo, hi = (mult & 0xFFFFFFFF).astype(np.uint32), (mult >> 32).astype(np.uint32)
        c = np.zeros(128, dtype=np.uint32)
        mh = np.stack([lo, hi], -1).reshape(-1)
        c[: mh.size] = mh
        o = mh.size
        pr = np.asarray(primes, dtype=np.int64).reshape(-1)
        assert pr.max() < 2**31
        c[o : o + pr.size] = pr.astype(np.uint32)
        o += pr.size
        of = np.asarray(offsets, dtype=np.int64).reshape(-1)
        c[o : o + of.size] = of.astype(np.uint32)
        o += of.size
        c[o] = np.uint32(pad_id)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        up = lambda t: ttnn.from_torch(
            t,
            device=mesh,
            dtype=i32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.consts = up(torch.from_numpy(c.view(np.int32)).reshape(1, 128))
        tm = np.zeros((len(token_map), 16), dtype=np.int32)
        tm[:, 0] = np.asarray(token_map, dtype=np.int32)
        self.table = up(torch.from_numpy(tm))
        self.out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([T, 64]), i32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )

    def upload_hist(self, hist):
        """hist torch int32 [n_dev*T, 16] -> sharded over the mesh (dim 0)."""
        return ttnn.from_torch(
            hist,
            device=self.mesh,
            dtype=i32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )

    def __call__(self, hist):
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        CB = 0
        cb = ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB, data_format=i32, page_size=4096)],
        )
        ct = (
            [self.mode, self.T, self.W, self.NL, self.H, CB]
            + _acc(hist)
            + _acc(self.table)
            + _acc(self.consts)
            + _acc(self.out)
        )
        k = ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/engram_hash.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core,
            compile_time_args=ct,
            runtime_args=ttnn.RuntimeArgs(),
            common_runtime_args=[
                hist.buffer_address(),
                self.table.buffer_address(),
                self.consts.buffer_address(),
                self.out.buffer_address(),
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
        prog.custom_program_hash = (0x3E6 << 40) | (
            hash((self.mode, self.T, self.W, self.NL, self.H, tuple(_acc(hist)))) & ((1 << 40) - 1)
        )
        ttnn.generic_op([hist, self.table, self.consts, self.out], prog)
        return self.out

    def ids(self, out_torch):
        """[n_dev*T, 64] -> [n_dev*T, NL, NCOL]"""
        return out_torch[:, : self.NL * self.NCOL].reshape(-1, self.NL, self.NCOL)
