# SPDX-License-Identifier: Apache-2.0
"""Multi-core gpt-oss decode QKV head split producing the STOCK packed layout.

See qkv_headsplit_reader.cpp for the layout and work-decomposition reasoning.
Work unit = one output tile-row, so max parallelism is
    ceil(num_q_heads/32) + ceil(num_kv/32) * 2   (= 4 for gpt-oss 64/8/8)
NOT one-per-head: heads 0..31 share a destination tile.

The ProgramDescriptor is cached keyed on buffer addresses. Rebuilding
ttnn.RuntimeArgs per call cost 19-97 us and previously masked all device time.
"""
import ttnn

_KDIR = "models/demos/gpt_oss/kernels"
_TILE = 32
_CACHE = {}
_PROG = {}


def _units(num_q_heads, num_kv_heads):
    qr = (num_q_heads + _TILE - 1) // _TILE
    kr = (num_kv_heads + _TILE - 1) // _TILE
    return qr + 2 * kr


def qkv_headsplit(xqkv, q, k, v, num_heads, num_kv_heads, head_dim, ncores=None):
    head_tiles = head_dim // _TILE
    elem = 2  # bfloat16
    subtile_line = 16 * elem
    n_units = _units(num_heads, num_kv_heads)
    NC = min(ncores or n_units, n_units)

    key = (id(xqkv.device()), num_heads, num_kv_heads, head_dim, NC)
    if key not in _CACHE:
        gx = min(8, NC)
        gy = (NC + gx - 1) // gx
        cl = [(x, y) for y in range(gy) for x in range(gx)][:NC]
        core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in cl])
        cb_q, cb_k, cb_v = 0, 1, 2
        rct = [cb_q, cb_k, cb_v, num_heads, num_kv_heads, head_tiles, subtile_line, elem]
        rct += ttnn.TensorAccessorArgs(xqkv).get_compile_time_args()
        wct = [cb_q, cb_k, cb_v, num_heads, num_kv_heads, head_tiles]
        for t in (q, k, v):
            wct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        # spread units round-robin over cores
        per = [[] for _ in range(NC)]
        for u in range(n_units):
            per[u % NC].append(u)
        _CACHE[key] = (core, cl, per, cb_q, cb_k, cb_v, rct, wct, head_tiles)

    core, cl, per, cb_q, cb_k, cb_v, rct, wct, ht = _CACHE[key]

    addrs = (xqkv.buffer_address(), q.buffer_address(), k.buffer_address(), v.buffer_address())
    pk = (key, addrs)
    if pk in _PROG:
        return ttnn.generic_op([xqkv, q, k, v], _PROG[pk])

    tile_bytes = _TILE * _TILE * 2

    def cb(i):
        fmt = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=tile_bytes)
        return ttnn.CBDescriptor(total_size=2 * ht * tile_bytes, core_ranges=core, format_descriptors=[fmt])

    cbs = [cb(cb_q), cb(cb_k), cb(cb_v)]

    rr = ttnn.RuntimeArgs()
    wr = ttnn.RuntimeArgs()
    for ci, (x, y) in enumerate(cl):
        us = per[ci]
        st = us[0] if us else 0
        rr[x][y] = [xqkv.buffer_address(), st, len(us)]
        wr[x][y] = [q.buffer_address(), k.buffer_address(), v.buffer_address(), st, len(us)]

    reader = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/qkv_headsplit_reader.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=rct,
        runtime_args=rr,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{_KDIR}/qkv_headsplit_writer.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=wct,
        runtime_args=wr,
        config=ttnn.WriterConfigDescriptor(),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)
    _PROG[pk] = prog
    return ttnn.generic_op([xqkv, q, k, v], prog)
