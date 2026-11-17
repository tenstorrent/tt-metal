# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.copy_device_to_host_regions / ttnn.copy_host_to_device_regions: byte ranges of a mesh tensor's per-device
buffers to and from caller-owned host memory.

The tensors here are row-major uint32 in interleaved DRAM, where a page is one row (``width * 4`` bytes) and page i
of a device's buffer is that device's row i, so a region's bytes can be checked against ``ttnn.to_torch``. Every
device of the mesh gets different rows. Covered: reads and writes of several regions on every device in one call, a
write-then-read round trip, non-blocking reads (the bytes after a sync, and the host buffer kept alive until then),
and every refusal (misaligned, out of range, a wrapping offset + size, a bad device index, a region outside
``host_buffer``, a sharded tensor, a non-blocking read without ``host_buffer``), each of which must leave nothing
queued: the caller's memory untouched and the next read correct."""

import gc
import weakref

import numpy as np
import pytest
import torch

import ttnn

ROWS, WIDTH = 64, 256
PAGE = WIDTH * 4  # bytes of one row of uint32


def device_tensor(mesh_device, fill=None):
    """(torch int32 [devices, 1, ROWS, WIDTH], the device tensor holding it, one [1, 1, ROWS, WIDTH] per device)."""
    n = mesh_device.get_num_devices()
    data = torch.arange(n * ROWS * WIDTH, dtype=torch.int32).reshape(n, 1, ROWS, WIDTH)
    if fill is not None:
        data = torch.full_like(data, fill)
    tensor = ttnn.from_torch(
        data,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    assert tensor.buffer_page_size() == PAGE
    return data, tensor


def on_host(mesh_device, tensor):
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).to(torch.int32)


def row_regions(host, runs, devices):
    """Regions for ``runs`` = [(first row, rows)] on every device, packed one after the other into ``host`` (a
    C-contiguous uint32 array of ``devices * sum(rows)`` rows): [(device, offset, bytes, address)]."""
    regions, at = [], host.ctypes.data
    for device in range(devices):
        for first, count in runs:
            regions.append((device, first * PAGE, count * PAGE, at))
            at += count * PAGE
    assert at - host.ctypes.data == host.nbytes
    return regions


def expected_rows(data, runs):
    return torch.cat([data[d, 0, first : first + count] for d in range(data.shape[0]) for first, count in runs]).numpy()


RUNS = [(3, 5), (40, 1), (63, 1), (0, 2)]


@pytest.mark.parametrize("with_host_buffer", [False, True])
def test_read_regions_of_every_device(mesh_device, with_host_buffer):
    data, tensor = device_tensor(mesh_device)
    n = mesh_device.get_num_devices()
    host = np.zeros((n * sum(c for _, c in RUNS), WIDTH), dtype=np.uint32)
    kwargs = {"host_buffer": host} if with_host_buffer else {}
    ttnn.copy_device_to_host_regions(tensor, row_regions(host, RUNS, n), **kwargs)
    assert np.array_equal(host.astype(np.int32), expected_rows(data, RUNS))
    # The whole buffer of one device as a single region.
    whole = np.zeros((ROWS, WIDTH), dtype=np.uint32)
    ttnn.copy_device_to_host_regions(tensor, [(n - 1, 0, ROWS * PAGE, whole.ctypes.data)], host_buffer=whole)
    assert np.array_equal(whole.astype(np.int32), data[n - 1, 0].numpy())


@pytest.mark.parametrize("blocking", [True, False])
def test_write_regions_then_read_them_back(mesh_device, blocking):
    data, tensor = device_tensor(mesh_device, fill=7)
    n = mesh_device.get_num_devices()
    rows = sum(c for _, c in RUNS)
    source = np.arange(n * rows * WIDTH, dtype=np.uint32).reshape(n * rows, WIDTH) + 1000
    ttnn.copy_host_to_device_regions(tensor, row_regions(source, RUNS, n), blocking=blocking, host_buffer=source)
    written = source.copy()
    source[:] = 0  # the bytes were copied into the command queue before the call returned
    del source
    expect = data.clone()
    at = 0
    for d in range(n):
        for first, count in RUNS:
            expect[d, 0, first : first + count] = torch.from_numpy(written[at : at + count].astype(np.int32))
            at += count
    assert torch.equal(on_host(mesh_device, tensor), expect)  # (a read waits for the writes before it)
    back = np.zeros_like(written)
    ttnn.copy_device_to_host_regions(tensor, row_regions(back, RUNS, n), host_buffer=back)
    assert np.array_equal(back, written)


def test_non_blocking_read_lands_by_the_next_sync(mesh_device):
    data, tensor = device_tensor(mesh_device)
    n = mesh_device.get_num_devices()
    hosts = [np.zeros((n * sum(c for _, c in runs), WIDTH), dtype=np.uint32) for runs in (RUNS, RUNS[:1], RUNS[1:])]
    for host, runs in zip(hosts, (RUNS, RUNS[:1], RUNS[1:])):
        ttnn.copy_device_to_host_regions(tensor, row_regions(host, runs, n), blocking=False, host_buffer=host)
    ttnn.synchronize_device(mesh_device)
    for host, runs in zip(hosts, (RUNS, RUNS[:1], RUNS[1:])):
        assert np.array_equal(host.astype(np.int32), expected_rows(data, runs))


def test_non_blocking_read_keeps_its_host_buffer_alive(mesh_device):
    """The copy into the host memory happens later, on the queue's reader thread: the queue holds the buffer until
    then, also when the caller drops it right after the call, and lets go of it afterwards."""
    _, tensor = device_tensor(mesh_device)
    n = mesh_device.get_num_devices()
    refs = []
    for _ in range(8):
        host = np.zeros((n * ROWS, WIDTH), dtype=np.uint32)
        refs.append(weakref.ref(host))
        regions = [(d, 0, ROWS * PAGE, host.ctypes.data + d * ROWS * PAGE) for d in range(n)]
        ttnn.copy_device_to_host_regions(tensor, regions, blocking=False, host_buffer=host)
        del host, regions
    gc.collect()
    ttnn.synchronize_device(mesh_device)  # every read has landed in memory that was still there
    for _ in range(200):  # the reader thread releases a buffer right after its copy
        gc.collect()
        if all(ref() is None for ref in refs):
            break
        ttnn.synchronize_device(mesh_device)
    assert all(ref() is None for ref in refs), "the queue still holds host buffers after the reads completed"


def test_non_blocking_read_needs_a_host_buffer(mesh_device, expect_error):
    _, tensor = device_tensor(mesh_device)
    host = np.zeros((1, WIDTH), dtype=np.uint32)
    with expect_error(RuntimeError, "blocking=False needs host_buffer"):
        ttnn.copy_device_to_host_regions(tensor, [(0, 0, PAGE, host.ctypes.data)], blocking=False)
    assert not host.any()


def bad_regions(n, address):
    """(why, the error's text, regions): each list starts with a good region and holds one the API must refuse."""
    good = (0, 0, PAGE, address)
    second = address + PAGE
    aligned, inside = "must be multiples of the buffer's page size", "are outside the buffer"
    return [
        ("offset not page-aligned", aligned, [good, (0, PAGE // 2, PAGE, second)]),
        ("size not page-aligned", aligned, [good, (0, PAGE, PAGE + 4, second)]),
        ("past the end", inside, [good, (0, (ROWS - 1) * PAGE, 2 * PAGE, second)]),
        ("offset past the end", inside, [good, (0, (ROWS + 1) * PAGE, PAGE, second)]),
        ("offset + size wraps", inside, [good, (0, 2**64 - PAGE, PAGE, second)]),
        ("size wraps", inside, [good, (0, PAGE, 2**64 - PAGE, second)]),
        ("no such device", "out of range: the mesh has", [good, (n, 0, PAGE, second)]),
        ("empty", "empty transfer", [good, (0, 0, 0, second)]),
        ("null host", "empty transfer", [good, (0, 0, PAGE, 0)]),
    ]


@pytest.mark.parametrize("case", range(9))
@pytest.mark.parametrize("write", [False, True])
def test_a_bad_region_is_refused_before_anything_is_queued(mesh_device, case, write, expect_error):
    """Every region is checked before the first is issued: the good region in front of the bad one is neither read
    nor written, and the queue is as it was (the reads after it return the right bytes to the right buffers)."""
    data, tensor = device_tensor(mesh_device)
    n = mesh_device.get_num_devices()
    host = np.full((2, WIDTH), 0xABCD, dtype=np.uint32)
    why, message, regions = bad_regions(n, host.ctypes.data)[case]
    copy = ttnn.copy_host_to_device_regions if write else ttnn.copy_device_to_host_regions
    with expect_error(RuntimeError, message):
        copy(tensor, regions)
    assert (host == 0xABCD).all(), why
    assert torch.equal(on_host(mesh_device, tensor), data), why  # nothing was written
    for first in (5, 20):  # and nothing was left on the queue: later reads are their own
        row = np.zeros((1, WIDTH), dtype=np.uint32)
        ttnn.copy_device_to_host_regions(tensor, [(n - 1, first * PAGE, PAGE, row.ctypes.data)], host_buffer=row)
        assert np.array_equal(row[0].astype(np.int32), data[n - 1, 0, first].numpy()), why


def test_regions_must_lie_inside_host_buffer(mesh_device, expect_error):
    data, tensor = device_tensor(mesh_device)
    host = np.zeros((2, WIDTH), dtype=np.uint32)
    at = host.ctypes.data
    for regions in ([(0, 0, 3 * PAGE, at)], [(0, 0, PAGE, at + 2 * PAGE)], [(0, 0, PAGE, at - PAGE)]):
        with expect_error(RuntimeError, "outside host_buffer"):
            ttnn.copy_device_to_host_regions(tensor, regions, host_buffer=host)
        with expect_error(RuntimeError, "outside host_buffer"):
            ttnn.copy_host_to_device_regions(tensor, regions, host_buffer=host)
    assert not host.any() and torch.equal(on_host(mesh_device, tensor), data)
    readonly = bytes(2 * PAGE)  # a read needs a writable buffer; a write takes a read-only one
    with expect_error(BufferError, "not writable"):
        ttnn.copy_device_to_host_regions(tensor, [(0, 0, PAGE, at)], host_buffer=readonly)


def test_sharded_tensors_are_refused(mesh_device, expect_error):
    """A sharded buffer keeps its pages per core in host page order: a byte range of it names no device range."""
    data = torch.zeros(1, 1, 64, 64, dtype=torch.bfloat16)
    interleaved = ttnn.from_torch(data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    sharded = ttnn.to_memory_config(
        interleaved,
        ttnn.create_sharded_memory_config(
            shape=(32, 64),
            core_grid=ttnn.CoreGrid(y=1, x=2),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    page = sharded.buffer_page_size()
    host = np.zeros(page, dtype=np.uint8)
    with expect_error(RuntimeError, "need an interleaved buffer"):
        ttnn.copy_device_to_host_regions(sharded, [(0, 0, page, host.ctypes.data)], host_buffer=host)
    with expect_error(RuntimeError, "need an interleaved buffer"):
        ttnn.copy_host_to_device_regions(sharded, [(0, 0, page, host.ctypes.data)], host_buffer=host)
