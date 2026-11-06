# SPDX-License-Identifier: Apache-2.0
"""Experimental raw interleaved DRAM page transport; no model integration yet."""

import math
from pathlib import Path

import torch

import ttnn

_KERNEL = str(Path(__file__).with_name("kernels") / "private_cache_copy.cpp")


def page_geometry(value):
    if value.is_sharded() or value.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("Only unsharded interleaved DRAM tensors are supported")
    shape = tuple(value.padded_shape)
    if value.layout == ttnn.TILE_LAYOUT:
        sizes = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048, ttnn.float32: 4096}
        if value.dtype not in sizes:
            raise ValueError("Unsupported tiled format")
        return math.prod(shape) // 1024, sizes[value.dtype]
    if value.layout == ttnn.ROW_MAJOR_LAYOUT and value.dtype in (ttnn.bfloat16, ttnn.float32):
        width = shape[-1] * (2 if value.dtype == ttnn.bfloat16 else 4)
        if width % 32:
            raise ValueError("Row-major pages must already have32-byte aligned width")
        return math.prod(shape[:-1]), width
    raise ValueError("Unsupported raw page geometry")


class CopyWorkspace:
    """Setup-owned mapping storage; no device allocations in request handling."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.row_bytes = max(32, ttnn.get_dram_alignment())
        self.tables = {}
        self.inflight = set()
        self.reuse_fences = 0
        for capacity in (1, 8192):
            self.tables[capacity] = ttnn.from_torch(
                torch.zeros(capacity, self.row_bytes // 4, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

    def mapping(self, pairs, keepalive, *, large, prime):
        capacity = 8192 if large else 1
        if len(pairs) > capacity:
            raise ValueError("Mapping exceeds fixed prepared capacity")
        table = self.tables[capacity]
        if not prime:
            if capacity in self.inflight:
                # Conservative first mechanism: complete every earlier reader
                # before mutating this shared table, regardless of CQ details.
                ttnn.synchronize_device(self.mesh)
                self.inflight.clear()
                self.reuse_fences += 1
            host = torch.zeros(capacity, self.row_bytes // 4, dtype=torch.int32)
            host[: len(pairs), :2] = torch.tensor(pairs, dtype=torch.int32)
            host_tensor = ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            # Retain host storage until the caller's fence. Reuse above waits
            # for prior device readers; Python ownership alone is insufficient.
            ttnn.copy_host_to_device_tensor(host_tensor, table)
            keepalive.extend((host, host_tensor))
            self.inflight.add(capacity)
        return table


def copy_blocks(
    source, destination, block_pairs, *, pages_per_block, keepalive, workspace=None, large_mapping=False, prime=False
):
    """Enqueue independent block copies; caller retains keepalive until a fence.

    block_pairs is a host sequence of (source block,destination block). Blocks
    contain pages_per_block physical tensor pages. Duplicate destinations and
    aliasing are rejected. No dtype conversion, shape reinterpretation or reads
    back to the host occur. This is NOT validated until the device probe passes.
    """
    source_pages, source_bytes = page_geometry(source)
    destination_pages, destination_bytes = page_geometry(destination)
    if source.dtype != destination.dtype or source.layout != destination.layout or source_bytes != destination_bytes:
        raise ValueError("Raw copies require equal dtype, layout and physical page bytes")
    if source.buffer_address() == destination.buffer_address():
        raise ValueError("In-place/aliasing page copies are unsupported")
    if source_bytes > 8192:
        raise ValueError("Page exceeds bounded L1 staging")
    if not isinstance(pages_per_block, int) or pages_per_block < 1:
        raise ValueError("pages_per_block must be positive")
    pairs = [(int(s), int(d)) for s, d in block_pairs]
    if not pairs:
        return
    if prime and workspace is None:
        raise ValueError("Zero-work priming requires persistent workspace")
    if len({d for _, d in pairs}) != len(pairs):
        raise ValueError("Duplicate destination blocks would race")
    if any(
        s < 0 or d < 0 or (s + 1) * pages_per_block > source_pages or (d + 1) * pages_per_block > destination_pages
        for s, d in pairs
    ):
        raise ValueError("Block mapping exceeds source/destination tensor")
    mesh = source.device()
    # Each mapping row is one accessor page; its physical stride comes from the
    # actual TTNN tensor rather than an assumed8-byte row alignment.
    if workspace is not None:
        table = workspace.mapping(pairs, keepalive, large=large_mapping, prime=prime)
    else:
        mapping_row_bytes = max(32, ttnn.get_dram_alignment())
        mapping_host = torch.zeros(len(pairs), mapping_row_bytes // 4, dtype=torch.int32)
        mapping_host[:, :2] = torch.tensor(pairs, dtype=torch.int32)
        table = ttnn.from_torch(
            mapping_host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
    grid = mesh.compute_with_storage_grid_size()
    # Stable topology: record count and pages/block are runtime args only.
    count = min(16, grid.x * grid.y)
    coords = [ttnn.CoreCoord(i // grid.y, i % grid.y) for i in range(count)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    runtime = ttnn.RuntimeArgs()
    for worker, core in enumerate(coords):
        runtime[core.x][core.y] = [
            source.buffer_address(),
            destination.buffer_address(),
            table.buffer_address(),
            0 if prime else len(pairs),
            pages_per_block,
            worker,
            count,
        ]
    compile_args = []
    for value in (source, destination, table):
        compile_args.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL,
                core_ranges=cores,
                compile_time_args=compile_args,
                runtime_args=runtime,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        cbs=[
            ttnn.CBDescriptor(
                total_size=size,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint32, page_size=size)
                ],
            )
            for index, size in [(0, 8192), (1, 32)]
        ],
        semaphores=[],
    )
    # generic_op inputs retain native storage dependencies; keepalive explicitly
    # extends Python tensor ownership through subsequent queue operations/fence.
    keepalive.extend((source, destination, table))
    ttnn.generic_op([source, destination, table], descriptor)
