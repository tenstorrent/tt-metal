# SPDX-License-Identifier: Apache-2.0
"""Discover NoC coordinates of the compute grid by running a tiny generic_op (host has no direct binding)."""
import torch
import ttnn
from models.demos.deepseek_v3_d_p.dsa_single_chip.ops import _program as P

_CACHE = {}


def grid_noc_coords(device):
    """-> dict {(lx, ly): (noc_x, noc_y)} for the writer RISC's NoC."""
    key = id(device)
    if key in _CACHE:
        return _CACHE[key]
    g = device.compute_with_storage_grid_size()
    n = g.x * g.y
    work = P.split_work(n, device, n)
    cores = P.core_rect(work, device)
    out = P.alloc([n, 8], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    wr = P.writer(P.kernel_path("coord_probe.cpp"), cores, [g.y] + P.accessor_args(out), [], common=[out.buffer_address()])
    P.run([out], P.program([wr], [P.cb(0, ttnn.uint32, 1, cores, page_bytes=32)]))
    t = ttnn.to_torch(out).reshape(n, 8)
    res = {(int(r[2]), int(r[3])): (int(r[0]), int(r[1])) for r in t}
    _CACHE[key] = res
    return res
