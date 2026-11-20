# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device side of the paged KV store: the pool / cache tensors, the frame-table tensor in L1, and the probe
program that reads through a table (``kernels/kv_probe_reader.cpp``). The copy program is in ``copy.py``.

Tensors of one kind (``frames.Kind``), per chip, all ND-sharded ``[1, 1, 32 rows, width]`` round-robin over the 8
DRAM banks exactly as the prefill stack allocates its caches (``utils/kv_cache_utils.py`` ``init_kvpe_cache``):

    pool   ``Geometry.pool(kind)``            [frames * layers, 1, 256, width]
    cache  ``Geometry.contiguous(kind, C)``   [users * layers, 1, C * 256, width]   (today's layout; the workbench)

Real dtypes: latent rows fp8_e4m3 ROW_MAJOR (656 B rows in 704 B pages), index keys bfloat8_b TILE (4 tiles of
1088 B per shard). For tests that need chosen bytes there is a "raw twin" of each: a uint8 ROW_MAJOR tensor with the
same pages, shards and bytes per shard (latent ``[.., 256, 656]`` in 32-row shards; keys ``[.., 32, 1088]`` in
4-row shards), which the address arithmetic cannot tell from the real one.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.experimental.hy4_preview.fused.kv_store import frames as fr

KERNEL_DIR = "models/experimental/hy4_preview/fused/kv_store/kernels/"
_HERE = Path(__file__).resolve().parent
INCLUDE = [str(_HERE / "kernels"), str(_HERE.parent / "sparse_mla" / "kernels")]
assert (_HERE / "kernels").is_dir()


def core_set(cores) -> ttnn.CoreRangeSet:
    """The cores as few rectangles as merge() finds (dispatch writes binaries and arguments per core range)."""
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores[:1]])
    for c in cores[1:]:
        ranges = ranges.merge(ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c))]))
    return ranges


def nd_memory_config(device, shard_rows: int, width: int):
    banks = device.dram_grid_size().x
    assert banks == fr.BANKS, f"{banks} DRAM banks: the frame arithmetic is written for {fr.BANKS}"
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, shard_rows, width],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def raw_shape(geometry: fr.Geometry, n: int) -> tuple:
    """Shape of the raw (uint8) twin of the tensor of ``n`` frames (pool) or users (cache)."""
    slots, _, rows, _ = geometry.tensor_shape(n)
    if geometry.kind.row_stride:
        return (slots, 1, rows, geometry.kind.width)
    return (slots, 1, rows // fr.SHARD_ROWS * 4, geometry.kind.shard_bytes // 4)  # 4 pages of 1088 B per shard


def check_layout(tensor, geometry: fr.Geometry, n: int) -> None:
    """The tensor has the bytes per shard and the shard order the frame arithmetic assumes."""
    kind = geometry.kind
    slots, _, rows, _ = geometry.tensor_shape(n)
    shards = slots * rows // fr.SHARD_ROWS
    pages_per_shard = tensor.buffer_num_pages() // shards
    assert tensor.buffer_num_pages() == shards * pages_per_shard, (tensor.buffer_num_pages(), shards)
    assert pages_per_shard * tensor.buffer_aligned_page_size() == kind.shard_bytes, (
        pages_per_shard,
        tensor.buffer_aligned_page_size(),
        kind.shard_bytes,
    )
    spec = tensor.memory_config().nd_shard_spec
    assert spec is not None and tuple(spec.shard_shape)[:2] == (1, 1), spec
    assert rows % fr.FRAME_ROWS == 0, "every slot must start on DRAM bank 0 (capacity a multiple of 8192 tokens)"


def allocate(device, geometry: fr.Geometry, n: int, raw: bool = False):
    """An uninitialised tensor of ``n`` frames (pool) or users (cache)."""
    kind = geometry.kind
    if raw:
        shape = raw_shape(geometry, n)
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)),
            ttnn.uint8,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            nd_memory_config(device, fr.SHARD_ROWS if kind.row_stride else 4, shape[3]),
        )
    else:
        dtype, layout = (
            (ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT) if kind.row_stride else (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        )
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(geometry.tensor_shape(n))),
            dtype,
            layout,
            device,
            nd_memory_config(device, fr.SHARD_ROWS, kind.width),
        )
    check_layout(t, geometry, n)
    return t


def raw_to_device(device, geometry: fr.Geometry, host: torch.Tensor):
    """Host bytes (``raw_shape``, uint8) -> the raw twin on the device."""
    kind = geometry.kind
    assert host.dtype == torch.uint8
    t = ttnn.from_torch(
        host.contiguous(),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=nd_memory_config(device, fr.SHARD_ROWS if kind.row_stride else 4, host.shape[3]),
    )
    return t


def frame_view(geometry: fr.Geometry, host: torch.Tensor, frame: int, layer: int) -> torch.Tensor:
    """The rows of (frame, layer) in a host image of the tensor (raw twin or real): a view of 256 stripe rows
    (raw key twin: 32 pages). ``frame`` as in the table: the pool frame, or ``contiguous_frame(user, k)``."""
    per = host.shape[2] // (geometry.slot_stride if geometry.frame_stride == 1 else 1)  # rows of one frame
    if geometry.frame_stride == 1:
        user_slots, k = divmod(frame, geometry.slot_stride)
        return host[user_slots + layer, 0, k * per : (k + 1) * per]
    return host[frame * geometry.kind.layers + layer, 0]


# ---------------------------------------------------------------------------------------------------------------
# The frame table in L1
# ---------------------------------------------------------------------------------------------------------------
class FrameTables:
    """An L1 tensor that holds, on each of ``cores``, ``tables`` tables of ``rows`` decode rows x TABLE_FRAMES
    uint32 frame bases (2 KiB per table of 4 rows, 4 KiB of 8, 8 KiB of 16). A kernel reaches it through a
    circular buffer laid over it (``cb(index)``): ``get_read_ptr(cb)`` is table 0, row 0. The host rewrites it
    when the manager's ``table_version`` changed; nothing is read at launch."""

    def __init__(self, device, cores, rows: int, tables: int = 1):
        self.device, self.cores, self.rows, self.tables = device, list(cores), rows, tables
        self.per_core = tables * rows
        self.memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_set(self.cores), (self.per_core, fr.TABLE_FRAMES), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.tensor = self._host_to_device(torch.full((self.per_core, fr.TABLE_FRAMES), fr.UNMAPPED, dtype=torch.int64))

    def _host(self, per_core: torch.Tensor):
        # uint32 values through int32 bits (UNMAPPED = -1)
        bits = per_core.to(torch.int64) & 0xFFFFFFFF
        bits = torch.where(bits >= 2**31, bits - 2**32, bits).to(torch.int32)
        return bits.repeat(len(self.cores), 1).reshape(1, 1, len(self.cores) * self.per_core, fr.TABLE_FRAMES)

    def _host_to_device(self, per_core: torch.Tensor):
        return ttnn.from_torch(
            self._host(per_core),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=self.memory_config,
        )

    def write(self, *tables) -> None:
        """``tables``: one flat sequence of rows * TABLE_FRAMES frame bases per table (``KvStore.table``)."""
        assert len(tables) == self.tables
        per_core = torch.tensor([list(t) for t in tables], dtype=torch.int64).reshape(self.per_core, fr.TABLE_FRAMES)
        host = ttnn.from_torch(self._host(per_core), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.tensor)

    def read(self, device_index: int = 0) -> torch.Tensor:
        """[cores, tables, rows, TABLE_FRAMES] as int64 holding the uint32 values."""
        t = ttnn.to_torch(ttnn.get_device_tensors(self.tensor)[device_index]).to(torch.int64) & 0xFFFFFFFF
        return t.reshape(len(self.cores), self.tables, self.rows, fr.TABLE_FRAMES)

    def cb(self, index: int):
        return ttnn.cb_descriptor_from_sharded_tensor(index, self.tensor)

    def table_words(self, table: int) -> int:
        return table * self.rows * fr.TABLE_FRAMES


# ---------------------------------------------------------------------------------------------------------------
# The probe reader
# ---------------------------------------------------------------------------------------------------------------
REQ_PER_PAGE = 64
CB_TABLE, CB_REQ, CB_DATA = 0, 1, 2
MODE_NO_READS, MODE_CONTIGUOUS, MODE_FLOOR = 1, 2, 4


class Probe:
    """Reads items of one cache tensor through a frame table on one core and writes them to ``out`` (tests)."""

    def __init__(
        self,
        device,
        geometry: fr.Geometry,
        tensor,
        tables: FrameTables,
        table: int,
        max_requests: int,
        out_like=None,
        core=(0, 0),
    ):
        kind = geometry.kind
        self.device, self.geometry, self.tensor, self.tables, self.table, self.core = (
            device,
            geometry,
            tensor,
            tables,
            table,
            core,
        )
        assert list(tables.cores) == [core]
        self.item_bytes = kind.row_stride or kind.shard_bytes
        self.item_pages = 1 if kind.row_stride else 4
        self.out_page = self.item_bytes // self.item_pages
        self.max_requests = max_requests
        pages = -(-max_requests // REQ_PER_PAGE)
        self.req = ttnn.from_torch(
            torch.zeros(1, 1, pages, REQ_PER_PAGE * 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if out_like == "bfp8":  # index-key blocks as BFP8 tiles: page = one tile, 4 per block
            self.out = ttnn.from_torch(
                torch.zeros(1, 1, max_requests * 32, kind.width),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        elif out_like == "fp8":
            self.out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, max_requests, kind.width]),
                ttnn.fp8_e4m3,
                ttnn.ROW_MAJOR_LAYOUT,
                device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            width = kind.width if kind.row_stride else self.out_page
            self.out = ttnn.from_torch(
                torch.zeros(1, 1, max_requests * self.item_pages, width, dtype=torch.uint8),
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        assert self.out.buffer_aligned_page_size() == self.out_page, (
            self.out.buffer_aligned_page_size(),
            self.out_page,
        )
        assert self.req.buffer_aligned_page_size() == REQ_PER_PAGE * 16
        self.count = 0

    def set_requests(self, requests) -> None:
        """requests: (decode row, layer, stripe row) triples."""
        assert len(requests) <= self.max_requests
        pages = -(-self.max_requests // REQ_PER_PAGE)
        host = torch.zeros(pages * REQ_PER_PAGE, 4, dtype=torch.int32)
        if requests:
            host[: len(requests), :3] = torch.tensor(requests, dtype=torch.int32)
        t = ttnn.from_torch(
            host.reshape(1, 1, pages, REQ_PER_PAGE * 4), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(t, self.req)
        self.count = len(requests)

    def program(self, mode: int = 0, repeat: int = 1, batch: int = 16, users_layers: int = 0):
        g, kind = self.geometry, self.geometry.kind
        cores = core_set([self.core])
        named = [
            ("cb_table", CB_TABLE),
            ("cb_req", CB_REQ),
            ("cb_data", CB_DATA),
            ("item_bytes", self.item_bytes),
            ("row_stride", kind.row_stride),
            ("out_page_bytes", self.out_page),
            ("item_pages", self.item_pages),
            ("batch", batch),
            ("requests_per_page", REQ_PER_PAGE),
            ("slot_bytes", g.slot_bytes(1)),
            ("table_words", self.tables.table_words(self.table)),
            ("mode", mode),
            ("repeat", repeat),
            # today's arithmetic (mode bit 1): a slot-major cache of `slot_stride` frames per slot
            ("rows_per_slot", g.slot_stride * fr.FRAME_ROWS),
            ("slot_stride", kind.layers),
        ]
        kernel = ttnn.KernelDescriptor(
            kernel_source=KERNEL_DIR + "kv_probe_reader.cpp",
            core_ranges=cores,
            compile_time_args=list(ttnn.TensorAccessorArgs(self.req).get_compile_time_args())
            + list(ttnn.TensorAccessorArgs(self.out).get_compile_time_args()),
            named_compile_time_args=named,
            common_runtime_args=[
                self.tensor.buffer_address(),
                self.req.buffer_address(),
                self.out.buffer_address(),
                self.count,
            ],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            compiler_include_paths=INCLUDE,
        )

        def cb(index, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.uint8, page_size=page)
                ],
            )

        cbs = [self.tables.cb(CB_TABLE), cb(CB_REQ, REQ_PER_PAGE * 16, 1), cb(CB_DATA, self.item_bytes, batch)]
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)

    def io(self):
        return [self.tensor, self.tables.tensor, self.req, self.out]

    def compile(self, **kw):
        ttnn.experimental.prepare_generic_op(self.io(), self.program(**kw))

    def __call__(self, **kw):
        return ttnn.generic_op(self.io(), self.program(**kw))

    def read(self, device_index: int = 0) -> torch.Tensor:
        """Output as the host sees it: uint8 [requests, item bytes] for the raw twins."""
        t = ttnn.to_torch(ttnn.get_device_tensors(self.out)[device_index])
        return t
