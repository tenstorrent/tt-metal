# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded, byte-preserving prefill transport with device-owned offsets.

All tensors are interleaved DRAM. This helper only moves replicated BF16 rows
and INT32 page IDs; it never computes an activation or reads one on the host.
Offsets/counts live in a persistent metadata tensor, independent of program
keys. Calls are serial on one command queue. Warm every source/destination
physical shape before the first live trace.
"""

import torch

import ttnn

from .slot_state import _program

_ROW_COPY = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr uint32_t doff=get_compile_time_arg_val(0), moff=get_compile_time_arg_val(1);
 const auto src=TensorAccessor(TensorAccessorArgs<2>(),get_arg_val<uint32_t>(0));
 const auto dst=TensorAccessor(TensorAccessorArgs<doff>(),get_arg_val<uint32_t>(1));
 const auto meta=TensorAccessor(TensorAccessorArgs<moff>(),get_arg_val<uint32_t>(2));
 const uint32_t worker=get_arg_val<uint32_t>(3), workers=get_arg_val<uint32_t>(4);
 const uint32_t bytes=get_arg_val<uint32_t>(5);
 Noc noc; DataflowBuffer scratch(0); scratch.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(0)&63))&63;
 auto* words=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias);
 noc.async_read(meta,scratch,32,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias});
 noc.async_read_barrier();
 const uint32_t from=words[0], to=words[1], count=words[2], extent=words[3];
 for(uint32_t i=worker;i<extent;i+=workers) {
  if(i<count) {
   noc.async_read(src,scratch,bytes,{.page_id=from+i,.offset_bytes=0},{.offset_bytes=bias});
   noc.async_read_barrier();
  } else {
   for(uint32_t j=0;j<bytes/4;++j) words[j]=0;
  }
  noc.async_write(scratch,dst,bytes,{.offset_bytes=bias},{.page_id=to+i,.offset_bytes=0});
  noc.async_write_barrier();
 }
}
"""

_PAGE_WINDOW = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr uint32_t doff=get_compile_time_arg_val(0), moff=get_compile_time_arg_val(1);
 const auto src=TensorAccessor(TensorAccessorArgs<2>(),get_arg_val<uint32_t>(0));
 const auto dst=TensorAccessor(TensorAccessorArgs<doff>(),get_arg_val<uint32_t>(1));
 const auto meta=TensorAccessor(TensorAccessorArgs<moff>(),get_arg_val<uint32_t>(2));
 const uint32_t width=get_arg_val<uint32_t>(3);
 Noc noc; DataflowBuffer scratch(0); scratch.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(0)&63))&63;
 auto* words=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias);
 noc.async_read(meta,scratch,32,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias});
 noc.async_read_barrier(); const uint32_t first=words[0];
 // Match low address bits for every small read, then pack eight IDs locally.
 uint32_t ids[8];
 for(uint32_t i=0;i<8;++i) {
  const uint32_t page=(first+i<width)?first+i:width-1;
  const uint32_t offset=page*4, lane=offset&63;
  noc.async_read(src,scratch,4,{.page_id=0,.offset_bytes=offset},{.offset_bytes=bias+lane});
  noc.async_read_barrier(); ids[i]=words[lane/4];
 }
 for(uint32_t i=0;i<8;++i) words[i]=ids[i];
 noc.async_write(scratch,dst,32,{.offset_bytes=bias},{.page_id=0,.offset_bytes=0});
 noc.async_write_barrier();
}
"""


def _accessor_args(source, destination, metadata):
    arrays = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in (source, destination, metadata)]
    a, b, c = arrays
    return [2 + len(a), 2 + len(a) + len(b)] + a + b + c


class PrefillTransport:
    def __init__(self, mesh):
        self.mesh = mesh
        self.metadata = ttnn.from_torch(
            torch.zeros((1, 8), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def _refresh(self, *values):
        host = torch.zeros((1, 8), dtype=torch.int32)
        host[0, : len(values)] = torch.tensor(values, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self.metadata
        )

    @staticmethod
    def _row_tensor(tensor):
        if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("Prefill row transport requires BF16 ROW_MAJOR tensors")
        if tuple(tensor.shape)[-1] != 2048 or len(tensor.shape) != 3 or tensor.shape[0] != 1:
            raise ValueError("Prefill row transport requires [1, rows, 2048]")
        if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("Prefill row transport requires interleaved DRAM")

    def rows(self, source, destination, *, source_row=0, destination_row=0, count, extent=None):
        """Copy count rows, then zero extent-count rows in the destination.

        Source and destination must not overlap. The metadata refresh is eager
        request-boundary work; the device kernel itself has no host dependence.
        """
        self._row_tensor(source)
        self._row_tensor(destination)
        extent = count if extent is None else extent
        if not 0 <= count <= extent or source_row < 0 or destination_row < 0:
            raise ValueError("Invalid prefill row window")
        if source_row + count > source.shape[1] or destination_row + extent > destination.shape[1]:
            raise ValueError("Prefill row window exceeds an allocation")
        if source.buffer_address() == destination.buffer_address():
            raise ValueError("Prefill row transport does not support overlapping buffers")
        self._refresh(source_row, destination_row, count, extent)
        return self.rows_device(source, destination)

    def rows_device(self, source, destination):
        """Device-only row copy consuming pre-refreshed metadata; capture-safe.

        Metadata is [source_row, destination_row, valid_count, extent, ...].
        The caller must validate bounds before refreshing it outside capture.
        """
        self._row_tensor(source)
        self._row_tensor(destination)
        if source.buffer_address() == destination.buffer_address():
            raise ValueError("Prefill row transport does not support overlapping buffers")
        grid = self.mesh.compute_with_storage_grid_size()
        workers = min(8, grid.x * grid.y)
        runtime = [
            (
                i % grid.x,
                i // grid.x,
                [
                    source.buffer_address(),
                    destination.buffer_address(),
                    self.metadata.buffer_address(),
                    i,
                    workers,
                    4096,
                ],
            )
            for i in range(workers)
        ]
        program = _program(self.mesh, _ROW_COPY, _accessor_args(source, destination, self.metadata), runtime, 4160)
        ttnn.generic_op([source, destination, self.metadata], program)
        return destination

    def page_window(self, source, destination, first_page):
        for tensor in (source, destination):
            if tensor.dtype != ttnn.int32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise ValueError("Fill-page views require INT32 ROW_MAJOR")
            if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG or tensor.shape[0] != 1:
                raise ValueError("Fill-page views require one interleaved DRAM row")
        if tuple(destination.shape) != (1, 8) or not 0 <= first_page < source.shape[-1]:
            raise ValueError("Invalid fill-page window")
        self._refresh(first_page)
        return self.page_window_device(source, destination)

    def page_window_device(self, source, destination):
        """Device-only fill view; metadata[0] holds the first page index."""
        runtime = [
            (
                0,
                0,
                [
                    source.buffer_address(),
                    destination.buffer_address(),
                    self.metadata.buffer_address(),
                    source.shape[-1],
                ],
            )
        ]
        program = _program(self.mesh, _PAGE_WINDOW, _accessor_args(source, destination, self.metadata), runtime, 128)
        ttnn.generic_op([source, destination, self.metadata], program)
        return destination
