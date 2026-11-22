# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing of the Inkling-2-Mini fused residual kernels (ttnn.generic_op, JIT kernels, one Tensix core).

Dense form: a C-vector stored contiguously, padded with zeros to whole 1024-element pages. On device it is the raw
page content of a [1, 1, 32, 32 * P] TILE tensor (page p = tile p); the kernels view each page as one 32x32 tile.
Host helpers below permute a torch vector so that tilizing it yields exactly that raw page content.
"""

import os

import torch

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")  # not cwd-relative
PAGE = 1024  # elements per dense page
CORE = ttnn.CoreCoord(0, 0)
CORES = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])
ES = {ttnn.float32: 4, ttnn.bfloat16: 2}


def pages(c):
    return (c + PAGE - 1) // PAGE


def dense_matrix(vecs, n_pages=None):
    """[N, C] torch -> [1, 1, 32, 32 * N * P] torch whose TILE layout holds vector n in pages n*P .. n*P + P - 1."""
    vecs = vecs.reshape(-1, vecs.shape[-1]).float()
    n, c = vecs.shape
    p = n_pages or pages(c)
    v = torch.nn.functional.pad(vecs, (0, p * PAGE - c)).reshape(n * p, 2, 2, 16, 16)  # [page, fr, fc, r, c]
    m = v.permute(0, 1, 3, 2, 4).reshape(n * p, 32, 32)  # [page, row, col]
    return m.permute(1, 0, 2).reshape(1, 1, 32, n * p * 32)


def from_dense_matrix(m, c, n=1):
    """Inverse of dense_matrix: [.., 32, 32 * N * P] -> [N, C]."""
    m = m.reshape(32, -1, 32).permute(1, 0, 2)  # [page, row, col]
    v = m.reshape(-1, 2, 16, 2, 16).permute(0, 1, 3, 2, 4).reshape(n, -1)
    return v[:, :c]


def dense_tensor(device, vecs, dtype=ttnn.float32):
    return ttnn.from_torch(
        dense_matrix(vecs), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def zeros_row_form(device, rows, c, dtype):
    """Zeroed [1, 1, rows, C] TILE DRAM tensor (the kernels write only the logical rows)."""
    return ttnn.from_torch(
        torch.zeros(1, 1, rows, c),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def check_row_input(t, l1_ok=False):
    """l1_ok: also accept an L1-interleaved row tensor (only kernels that handle input kind 2, see input_args)."""
    assert t.layout == ttnn.TILE_LAYOUT and t.dtype in ES, (t.layout, t.dtype)
    kinds = (ttnn.BufferType.DRAM, ttnn.BufferType.L1) if l1_ok else (ttnn.BufferType.DRAM,)
    assert t.memory_config().buffer_type in kinds and not t.is_sharded(), t.memory_config()


def tile_bytes(dtype):
    return 1024 * ES[dtype]


def cb(index, dtype, n_pages, page_bytes=None, cores=None):
    page_bytes = page_bytes or tile_bytes(dtype)
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)
    return ttnn.CBDescriptor(total_size=n_pages * page_bytes, core_ranges=cores or CORES, format_descriptors=[fmt])


def stage_cb(index, ntc, cores=None):
    """L1 scratch for 64B-aligned DRAM reads (rowio::gather_row): 128 B per tile + alignment slack."""
    return cb(index, ttnn.bfloat16, 1, ntc * 128 + 64, cores)


def compute_config(fp32_cbs, full_sync=False):
    """fp32_cbs: CBs read by copy_tile that must reach DST unrounded (UnpackToDestFp32; FPU ops read via SrcA/B).
    full_sync: 8 fp32 DST tiles instead of 4."""
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=full_sync
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in fp32_cbs:
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    return cfg


def kernel(name, config, named, rt=None, cores=None):
    """rt: runtime args of the single CORE, or {CoreCoord: args} with cores = their CoreRangeSet."""
    args = ttnn.RuntimeArgs()
    if rt is not None:
        for core, vals in rt.items() if isinstance(rt, dict) else [(CORE, rt)]:
            args[core.x][core.y] = [int(v) for v in vals]
    return ttnn.KernelDescriptor(
        kernel_source=os.path.join(KDIR, name),
        core_ranges=cores or CORES,
        named_compile_time_args=list(named.items()),
        runtime_args=args,
        config=config,
    )


def fixed_rt(named, *rts):
    """Pad every core's runtime args of each {core: args} dict to one length and add those lengths to the named
    compile-time args. generic_op's program cache key ignores runtime-arg counts, so a module whose per-core
    counts depend on the row count (the multi-row prefill kernels) would otherwise reuse a cached program built
    for another count and fail its arg-count check. The kernels ignore the zero padding."""
    named = dict(named)
    for i, rt in enumerate(rts):
        n = max(len(v) for v in rt.values())
        for v in rt.values():
            v.extend([0] * (n - len(v)))
        named[f"rt_len{i}"] = n
    return named


def core_list(device, n, first_row=0):
    """n worker cores, row-major over the compute grid starting at first_row: (list, CoreRangeSet)."""
    gx = device.compute_with_storage_grid_size().x
    cores = [ttnn.CoreCoord(i % gx, first_row + i // gx) for i in range(n)]
    return cores, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def run(io_tensors, kernels, cbs):
    """io_tensors: every tensor the kernels touch (keeps them alive / tracked); the last one is returned."""
    return ttnn.generic_op(io_tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))


# Dense twins: every row-form output of a fused kernel also gets a dense copy (a few page writes). A fused
# kernel given that same tensor object as input reads the twin's pages instead of gathering the row (2 x 72 small
# NOC reads, ~6 us). Keyed by object identity, so a tensor that passed through any other op has no twin.
_TWIN = {}


def set_twin(row, dense, nvec):
    _TWIN[id(row)] = (row, dense, nvec)


def twin(t):
    """(dense tensor, number of vectors) of a fused-kernel output, else None."""
    e = _TWIN.get(id(t))
    return (e[1], e[2]) if e is not None and e[0] is t else None


def input_args(t, row=0):
    """Reader runtime args (addr, page bytes, elem bytes, row / first page, kind) of a row-form input t: its dense
    twin when it has one (kind 1), else its rows in DRAM (0) or, interleaved, in L1 (2: conv's x only)."""
    tw = twin(t)
    if tw is not None:
        dense, nvec = tw
        return [
            dense.buffer_address(),
            tile_bytes(dense.dtype),
            ES[dense.dtype],
            min(row, nvec - 1) * pages(t.shape[-1]),
            1,
        ]
    l1 = t.memory_config().buffer_type == ttnn.BufferType.L1
    return [t.buffer_address(), tile_bytes(t.dtype), ES[t.dtype], row, 2 if l1 else 0]


def input_dtype(t):
    tw = twin(t)
    return tw[0].dtype if tw is not None else t.dtype


class OutCache:
    """Per-module zeroed output buffers: an output stays valid until the same module is called again with the
    same key. First use must be eager (it writes zeros from host), as the model and tests do before capture."""

    def __init__(self, device):
        self.device = device
        self.bufs = {}

    def get(self, key, rows, c, dtype):
        if key not in self.bufs:
            self.bufs[key] = zeros_row_form(self.device, rows, c, dtype)
        return self.bufs[key]

    def get_twin(self, key, rows, c, dtype, nvec=None):
        """Row-form output buffer plus its dense twin (nvec vectors, default rows)."""
        out = self.get(key, rows, c, dtype)
        tkey = ("twin", key)
        if tkey not in self.bufs:
            nvec = nvec or rows
            self.bufs[tkey] = dense_tensor(self.device, torch.zeros(nvec, c), dtype)
            set_twin(out, self.bufs[tkey], nvec)
        return out, self.bufs[tkey]
