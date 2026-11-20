# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The fused row matmul alone on one core: x [rows, 32 kt] (row tiles) times w [32 kt, 32 ct] (BFP8 / BFP4 / BF16
weight tiles), FP32 accumulation, the FP32 sums read back as they are.

Everything is resident in the core's L1 (the tensors back the kernel's circular buffers) and the program is one
compute kernel (kernels/fid_lab.cpp) with no data movement kernel, so it waits on nothing outside its own three
threads. It is the instrument of this piece: operand probes (one-hot inputs), accuracy on real weights, and the time
of unpack + multiply alone (``reps`` matmuls in one launch).

    lab = Lab(device, kt=8, ct=2, fmt="bfp8", passes=2)      # HiFi2
    lab.set_w(w); lab.set_x(x); lab(); out = lab.read()      # [rows, 64] fp32

``passes`` / ``step``: kernels/hifi_mm.h (1: LoFi, 2: HiFi2, 3: HiFi3, 4: HiFi4; passes 2 with step 2: phases 0
and 2). ``ref=True`` runs the tree's custom_mm instead (LoFi). ``layout``: the CustomMmDestLayout; the SplitAcc ones
cut the second K face's accumulator to BF16 when custom_mm's finalize merges it (tests/test_dense.py), the plain
ones have nothing to merge, and "<split layout>_exact" merges with hifi_mm_merge (FP32).

set_w / stored() convert float32 to the block format on the host in one rounding. A tensor created directly with a
SHARDED memory config is rounded to BF16 first (ttnn/core/tensor/py_to_tt_tensor.cpp compute_host_dtype), which is
what the streamed pieces' DRAM tensors hold: stored_sharded() is that value."""

from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn

KERNEL = Path(__file__).parent / "kernels" / "fid_lab.cpp"
MOCK = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))  # the mock Galaxy compiles and checks, runs nothing
TILE = 32
DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16, "fp32": ttnn.float32}
TILE_BYTES = {"bfp8": 1088, "bfp4": 576, "bf16": 2048, "fp32": 4096}
LAYOUTS = {"default": 0, "default_split": 1, "dense": 2, "dense_split": 3, "compact": 4}
# "<split layout>_exact": the SplitAcc layout with hifi_mm_merge (FP32, SFPU) instead of custom_mm's finalize
LAYOUTS.update({"default_split_exact": 1, "dense_split_exact": 3})
CB_X, CB_W, CB_OUT = 0, 1, 16
FIDELITIES = {"lofi": (1, 1), "hifi2": (2, 1), "hifi3": (3, 1), "hifi4": (4, 1), "p02": (2, 2)}


def stored(w, fmt):
    """w [32 a, 32 b] as the device holds it: ttnn's host rounding to BFP8 / BFP4 / BF16 tiles."""
    t = ttnn.from_torch(w.reshape(1, 1, *w.shape).float().contiguous(), dtype=DTYPES[fmt], layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).float().reshape(w.shape)


def stored_sharded(w, fmt):
    """w as a DRAM / L1 tensor created with a sharded memory config holds it: BF16 first, then the block format."""
    return stored(w.to(torch.bfloat16).float(), fmt)


class Lab:
    def __init__(
        self,
        device,
        kt,
        ct,
        fmt="bfp8",
        rows=8,
        passes=1,
        step=1,
        layout="dense",
        kb=None,
        reps=1,
        ref=False,
        column_major=False,
        x_fmt="bf16",
        core=(0, 0),
        full_sync=True,
        round_bf16=False,
        out_fmt="fp32",
    ):
        """full_sync: DEST in SyncFull (512 FP32 rows) or SyncHalf (256: what fused/moe runs with). round_bf16: with
        a "_exact" layout, the merged sums rounded to the nearest BF16 (still read back as FP32). out_fmt "bf16":
        the sums are packed into a BF16 buffer by the packer (what the packer does to an FP32 DEST value)."""
        assert rows in (1, 2, 4, 8, 16) and fmt in DTYPES and layout in LAYOUTS
        self.device, self.kt, self.ct, self.fmt, self.rows = device, kt, ct, fmt, rows
        self.kb = kt if kb is None else kb
        self.column_major = column_major
        self.x_dtype = DTYPES[x_fmt]
        core = ttnn.CoreCoord(*core)
        self.cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
        self.row_tile = ttnn.Tile([rows, TILE])
        self.w_tile = ttnn.Tile([TILE, TILE])

        def config(layout_, shape):
            return ttnn.MemoryConfig(
                layout_, ttnn.BufferType.L1, ttnn.ShardSpec(self.cores, shape, ttnn.ShardOrientation.ROW_MAJOR)
            )

        self.x_config = config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (rows, TILE * kt))
        self.w_shape = (TILE * ct, TILE * kt) if column_major else (TILE * kt, TILE * ct)
        self.w_config = config(ttnn.TensorMemoryLayout.WIDTH_SHARDED, self.w_shape)
        self.out_config = config(ttnn.TensorMemoryLayout.WIDTH_SHARDED, (rows, TILE * ct))
        self.x = self._x_tensor(torch.zeros(rows, TILE * kt), device)
        self.w = self._w_tensor(torch.zeros(TILE * kt, TILE * ct), device)
        self.out = ttnn.from_torch(
            torch.zeros(rows, TILE * ct),
            dtype=DTYPES[out_fmt],
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=self.out_config,
            tile=self.row_tile,
        )

        defines = [("LAB_LAYOUT", str(LAYOUTS[layout]))]
        if ref:
            assert passes == 1 and not layout.endswith("_exact")
            defines.append(("LAB_REF", "1"))
        if layout.endswith("_exact"):
            defines.append(("LAB_MERGE", "1"))
        if round_bf16:
            assert layout.endswith("_exact")
            defines.append(("LAB_ROUND", "1"))
        if column_major:
            defines.append(("LAB_COLUMN_MAJOR", "1"))
        if rows == 16:  # hifi_mm16: two faces of 16 rows, no split accumulators
            assert layout in ("dense", "default") and not ref, layout
            defines.append(("LAB_ROWS16", "1"))
        compute = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=full_sync,
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL),
            core_ranges=self.cores,
            compile_time_args=[CB_X, CB_W, CB_OUT, kt, ct, self.kb, passes, step, reps],
            runtime_args=ttnn.RuntimeArgs(),
            config=compute,
            defines=defines,
        )
        cbs = [
            ttnn.cb_descriptor_from_sharded_tensor(CB_X, self.x),
            ttnn.cb_descriptor_from_sharded_tensor(CB_W, self.w),
            ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, self.out),
        ]
        self.program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
        self.io = [self.x, self.w, self.out]

    def _x_tensor(self, x, device=None):
        kwargs = dict(dtype=self.x_dtype, layout=ttnn.TILE_LAYOUT, tile=self.row_tile)
        if device is not None:
            kwargs.update(device=device, memory_config=self.x_config)
        return ttnn.from_torch(x.float().contiguous(), **kwargs)

    def _w_tensor(self, w, device=None):
        """w [32 kt, 32 ct]. Row-major tiles are K-major (per K tile its ct tiles); column-major weights are stored
        as the transposed matrix's tiles would be ordered, each tile itself not transposed."""
        assert tuple(w.shape) == (TILE * self.kt, TILE * self.ct), tuple(w.shape)
        if self.column_major:
            w = w.reshape(self.kt, TILE, self.ct, TILE).permute(2, 1, 0, 3).reshape(TILE * self.ct, TILE * self.kt)
        kwargs = dict(dtype=DTYPES[self.fmt], layout=ttnn.TILE_LAYOUT, tile=self.w_tile)
        if device is not None:
            kwargs.update(device=device, memory_config=self.w_config)
        return ttnn.from_torch(w.float().contiguous(), **kwargs)

    def set_x(self, x):
        assert tuple(x.shape) == (self.rows, TILE * self.kt), tuple(x.shape)
        ttnn.copy_host_to_device_tensor(self._x_tensor(x), self.x)

    def set_w(self, w):
        ttnn.copy_host_to_device_tensor(self._w_tensor(w), self.w)

    def __call__(self):
        if MOCK:
            ttnn.experimental.prepare_generic_op(self.io, self.program)
        else:
            ttnn.generic_op(self.io, self.program)
        return self.out

    def read(self):
        return ttnn.to_torch(self.out).float().reshape(self.rows, TILE * self.ct)

    def close(self):
        """Free the three L1 tensors (a device shared by several tests has one core's L1 for all of them)."""
        for t in self.io:
            ttnn.deallocate(t)
