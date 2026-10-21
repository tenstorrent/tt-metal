# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Literal source chunk sequence, confined to Qwen's admitted head/chunk shapes.

Derived from Samuel Jett (sjettTT, sjett@tenstorrent.com), source cd9a117,
chunk_gdn_phased_program_factory.cpp. No shared operation default is changed.
"""
from pathlib import Path
import struct
import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

KERNELS = Path(__file__).parent / "kernels"
NAME = "gdn_source_chunk"
HEADS, DIM, CHUNK = 12, 128, 32


def allocate(shape, mesh):
    return fp.allocate(shape, ttnn.float32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)


def _cb(index, pages, cores, dtype=ttnn.float32):
    return fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, cores)


def _accessors(tensors):
    return [arg for tensor in tensors for arg in fp.accessor_args(tensor)]


def _work(mesh, units):
    grid = mesh.compute_with_storage_grid_size()
    count = min(units, grid.x * grid.y)
    base, rest = divmod(units, count)
    start = 0
    work = []
    for i in range(count):
        n = base + (i < rest)
        work.append(fp.CoreWork(ttnn.CoreCoord(i % grid.x, i // grid.x), start, n))
        start += n
    return work


def _check(tensor, shape, dtype, name):
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype or tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{name} requires TILE {dtype} {shape}, got {tensor.shape}/{tensor.dtype}/{tensor.layout}")


def prepare(q, k, v, beta, g, constants, *, rows_total, outputs=None):
    if type(rows_total) is not int or rows_total < CHUNK or rows_total % CHUNK or rows_total > 4096:
        raise ValueError("Source chunk rows must be a complete multiple of 32 in 32..4096")
    nc = rows_total // CHUNK
    mesh = q.device()
    _check(q, (HEADS, nc, CHUNK, DIM), ttnn.bfloat16, "q")
    _check(k, (HEADS, nc, CHUNK, DIM), ttnn.bfloat16, "k")
    _check(v, (1, 1, rows_total, HEADS * DIM), ttnn.bfloat16, "v")
    for tensor, name in [(beta, "beta"), (g, "g")]:
        _check(tensor, (HEADS, nc, CHUNK, 1), ttnn.float32, name)
    shapes = (
        [(HEADS, nc, CHUNK, DIM)] * 3
        + [(HEADS, nc, CHUNK, CHUNK), (HEADS, nc, DIM, CHUNK)]
        + [(HEADS, nc, 1, 1), (HEADS, nc, CHUNK, CHUNK)]
    )
    if outputs is None:
        outputs = tuple(allocate(shape, mesh) for shape in shapes)
    for tensor, shape in zip(outputs, shapes):
        _check(tensor, shape, ttnn.float32, "prep output")
    inputs = (q, k, ttnn.reshape(v, (1, rows_total, HEADS * DIM)), g, beta, *constants)
    work = _work(mesh, HEADS * nc)
    cores = fp.core_set(work)
    # Exact source allocation order and sizes: the inverse scratch's L1 layout is part of its control.
    sizes = [4, 4, 4, 1, 1, 1, 1, 1, 32, 1, 1, 1, 1, 1, 4, 4, 8, 4, 4, 4, 1, 32, 4, 4, 4, 16, 16, 16, 16, 16, 16, 32]
    cbs = [_cb(i, n, cores, ttnn.bfloat16 if i in (0, 1, 2, 16) else ttnn.float32) for i, n in enumerate(sizes)]
    dims = [1, 4, 4]
    addresses = [tensor.buffer_address() for tensor in inputs]
    result_addresses = [tensor.buffer_address() for tensor in outputs]
    scale = struct.unpack("<I", struct.pack("<f", 128**-0.5))[0]
    eps = struct.unpack("<I", struct.pack("<f", 1e-6))[0]
    kernels = [
        fp.reader_kernel(
            str(KERNELS / "reader_prep.cpp"),
            cores,
            [*dims, *_accessors(inputs), 1, 0],
            [(w.core, [w.start, w.count, *addresses, nc, HEADS, HEADS]) for w in work],
        ),
        fp.writer_kernel(
            str(KERNELS / "writer_prep.cpp"),
            cores,
            [*dims, *_accessors(outputs)],
            [(w.core, [w.start, w.count, *result_addresses]) for w in work],
        ),
        fp.compute_kernel(
            str(KERNELS / "compute_prep.cpp"),
            cores,
            [*dims, 0, scale, eps],
            [(w.core, [w.count]) for w in work],
            fp32_dest=True,
            approx=False,
            fidelity=ttnn.MathFidelity.HiFi4,
        ),
    ]
    meta = fp.program_meta(
        NAME,
        "prepare",
        rows_total,
        reads=inputs,
        writes=outputs,
        cores=len(work),
        outputs=tuple((output, q) for output in outputs),
    )
    fp.run_program([*inputs, *outputs], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs), meta=meta)
    return outputs


def scan(prepared, initial_state, *, rows_total, outputs=None):
    mesh = prepared[0].device()
    nc = rows_total // CHUNK
    state = ttnn.reshape(initial_state, (HEADS, DIM, DIM))
    _check(state, (HEADS, DIM, DIM), ttnn.float32, "initial_state")
    if outputs is None:
        outputs = (allocate((HEADS, nc, CHUNK, DIM), mesh), allocate((HEADS, DIM, DIM), mesh))
    grid = mesh.compute_with_storage_grid_size()
    nv = max(n for n in (1, 2, 4) if HEADS * n <= grid.x * grid.y)
    vt = 4 // nv
    work = _work(mesh, HEADS * nv)
    assert all(w.count == 1 for w in work)
    cores = fp.core_set(work)
    # Literal source scan CB map. Full-K and value-block scratch preserve pack/re-unpack boundaries.
    sizes = [
        (17, vt),
        (18, 4),
        (19, 4),
        (20, 1),
        (24, 4),
        (11, 1),
        (13, 1),
        (8, 4 * vt),
        (21, 4 * vt),
        (31, 4 * vt),
        (16, 2 * vt),
        (27, 4 * vt),
        (22, vt),
        (23, vt),
        (25, 4 * vt),
        (26, 4 * vt),
        (28, max(4, 4 * vt)),
    ]
    cbs = [_cb(i, n, cores) for i, n in sizes]
    inputs = (*prepared, state)
    dims = [1, 4, vt, 1, 4]
    addresses = [tensor.buffer_address() for tensor in inputs]
    result_addresses = [tensor.buffer_address() for tensor in outputs]
    kernels = [
        fp.reader_kernel(
            str(KERNELS / "reader_scan.cpp"),
            cores,
            [*dims, *_accessors(inputs)],
            [(w.core, [w.start // nv, w.start % nv, nc, *addresses]) for w in work],
        ),
        fp.writer_kernel(
            str(KERNELS / "writer_scan.cpp"),
            cores,
            [*dims, *_accessors(outputs)],
            [(w.core, [w.start // nv, w.start % nv, nc, *result_addresses]) for w in work],
        ),
        fp.compute_kernel(
            str(KERNELS / "compute_scan.cpp"),
            cores,
            dims,
            [(w.core, [nc]) for w in work],
            fp32_dest=True,
            approx=False,
            fidelity=ttnn.MathFidelity.HiFi4,
        ),
    ]
    meta = fp.program_meta(
        NAME,
        "scan",
        rows_total,
        reads=inputs,
        writes=outputs,
        cores=len(work),
        outputs=((outputs[0], prepared[0]), (outputs[1], state)),
    )
    fp.run_program([*inputs, *outputs], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs), meta=meta)
    return outputs


def chunk_source(q, k, v, beta, g, initial_state, constants, *, rows_total):
    prepared = prepare(q, k, v, beta, g, constants, rows_total=rows_total)
    try:
        result = scan(prepared, initial_state, rows_total=rows_total)
    finally:
        for tensor in prepared:
            ttnn.deallocate(tensor)
    return ttnn.reshape(result[0], (HEADS, rows_total, DIM)), ttnn.reshape(result[1], (1, HEADS, DIM, DIM))


def chunk_token_major(q, k, v, g, beta, initial_state, constants, *, rows_total, scale):
    """The composed rank-four producer, with the source's remaining query fold.

    No normalization occurs here. Flat raw q/k retain the existing public path.
    """
    _check(q, (1, rows_total, HEADS, DIM), ttnn.bfloat16, "token-major q")
    _check(k, (1, rows_total, HEADS, DIM), ttnn.bfloat16, "token-major k")
    temporary = []

    def keep(value):
        temporary.append(value)
        return value

    try:
        qc = keep(ttnn.reshape(keep(ttnn.permute(q, (0, 2, 1, 3))), (HEADS, rows_total, DIM)))
        qc = keep(ttnn.multiply(qc, scale))
        qc = keep(ttnn.reshape(qc, (HEADS, rows_total // CHUNK, CHUNK, DIM)))
        kc = keep(ttnn.reshape(keep(ttnn.permute(k, (0, 2, 1, 3))), (HEADS, rows_total // CHUNK, CHUNK, DIM)))
        gc = keep(ttnn.reshape(keep(ttnn.permute(g, (0, 2, 1))), (HEADS, rows_total // CHUNK, CHUNK, 1)))
        bc = keep(ttnn.reshape(keep(ttnn.permute(beta, (0, 2, 1))), (HEADS, rows_total // CHUNK, CHUNK, 1)))
        return chunk_source(
            qc,
            kc,
            ttnn.reshape(v, (1, 1, rows_total, HEADS * DIM)),
            bc,
            gc,
            initial_state,
            constants,
            rows_total=rows_total,
        )
    finally:
        # Views may alias each other or an input. Release only owned allocations;
        # original inputs and constants remain live through pending commands.
        input_addresses = {x.buffer_address() for x in (q, k, v, g, beta)}
        released = set()
        for value in reversed(temporary):
            if value.is_allocated():
                address = value.buffer_address()
                if address not in input_addresses and address not in released:
                    ttnn.deallocate(value)
                    released.add(address)
