# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-only fixed-slot state predicates, with exact byte-preserving commits.

Native ternary validation requires TILE inputs/output; request positions and
convolution state are ROW_MAJOR. These small data-movement operations avoid
FP32/BF16 conversion and never copy an inactive recurrent state. They are
optional: the accepted unmasked decoder does not call them. Warm every variant
before capture. The supplied scalar activity tensors must remain allocated.
"""

from math import prod

import ttnn

_EMPTY = '#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n'
_COPY = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr uint32_t doff=get_compile_time_arg_val(0), aoff=get_compile_time_arg_val(1);
 const auto src=TensorAccessor(TensorAccessorArgs<2>(),get_arg_val<uint32_t>(0));
 const auto dst=TensorAccessor(TensorAccessorArgs<doff>(),get_arg_val<uint32_t>(1));
 const auto activity=TensorAccessor(TensorAccessorArgs<aoff>(),get_arg_val<uint32_t>(2));
 const uint32_t start=get_arg_val<uint32_t>(3), count=get_arg_val<uint32_t>(4);
 const uint32_t bytes=get_arg_val<uint32_t>(5), zero_inactive=get_arg_val<uint32_t>(6);
 Noc noc; DataflowBuffer scratch(0); scratch.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(0)&63))&63;
 auto* words=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias);
 noc.async_read(activity,scratch,4,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias});
 noc.async_read_barrier();
 const bool active=words[0]!=0;
 if(!active && !zero_inactive) return;
 if(!active) for(uint32_t i=0;i<(bytes+3)/4;++i) words[i]=0;
 for(uint32_t page=start;page<start+count;++page) {
  if(active) {
   noc.async_read(src,scratch,bytes,{.page_id=page,.offset_bytes=0},{.offset_bytes=bias});
   noc.async_read_barrier();
  }
  noc.async_write(scratch,dst,bytes,{.offset_bytes=bias},{.page_id=page,.offset_bytes=0});
  noc.async_write_barrier();
 }
}
"""
_POSITION = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr uint32_t ao=get_compile_time_arg_val(0), ro=get_compile_time_arg_val(1);
 constexpr uint32_t co=get_compile_time_arg_val(2);
 const auto pos=TensorAccessor(TensorAccessorArgs<3>(),get_arg_val<uint32_t>(0));
 const auto activity=TensorAccessor(TensorAccessorArgs<ao>(),get_arg_val<uint32_t>(1));
 const auto rope=TensorAccessor(TensorAccessorArgs<ro>(),get_arg_val<uint32_t>(2));
 const auto cur=TensorAccessor(TensorAccessorArgs<co>(),get_arg_val<uint32_t>(3));
 const bool advance=get_arg_val<uint32_t>(4)!=0;
 Noc noc; DataflowBuffer scratch(0); scratch.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(0)&63))&63;
 auto* words=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias);
 noc.async_read(pos,scratch,4,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias});
 noc.async_read(activity,scratch,4,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias+64});
 noc.async_read_barrier();
 const uint32_t position=words[0]; const bool active=words[16]!=0;
 if(advance) {
  if(active) {
   words[0]=position+1;
   noc.async_write(scratch,pos,4,{.offset_bytes=bias},{.page_id=0,.offset_bytes=0});
   noc.async_write_barrier();
  }
 } else {
  words[0]=active?position:0;
  noc.async_write(scratch,rope,4,{.offset_bytes=bias},{.page_id=0,.offset_bytes=0});
  noc.async_write_barrier();
  words[0]=active?position:UINT32_MAX;
  noc.async_write(scratch,cur,4,{.offset_bytes=bias},{.page_id=0,.offset_bytes=0});
  noc.async_write_barrier();
 }
}
"""
_ROWS = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr uint32_t doff=get_compile_time_arg_val(0), aoff=get_compile_time_arg_val(1);
 const auto staging=TensorAccessor(TensorAccessorArgs<2>(),get_arg_val<uint32_t>(0));
 const auto destination=TensorAccessor(TensorAccessorArgs<doff>(),get_arg_val<uint32_t>(1));
 const auto active=TensorAccessor(TensorAccessorArgs<aoff>(),get_arg_val<uint32_t>(2));
 const bool advance=get_arg_val<uint32_t>(3)!=0;
 Noc noc; DataflowBuffer scratch(0); scratch.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(0)&63))&63;
 auto* words=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias);
 noc.async_read(staging,scratch,128,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias});
 noc.async_read(destination,scratch,128,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias+128});
 noc.async_read(active,scratch,128,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias+256});
 noc.async_read_barrier();
 for(uint32_t row=0;row<32;++row) {
  if(words[64+row]!=0) words[32+row]=advance?words[32+row]+1:words[row];
 }
 // All three inputs have been consumed before the write; input/output alias
 // is safe. Inactive words retain all their original bits, without casting.
 noc.async_write(scratch,destination,128,{.offset_bytes=bias+128},{.page_id=0,.offset_bytes=0});
 noc.async_write_barrier();
}
"""


def _scalar(value):
    if value.dtype not in (ttnn.int32, ttnn.uint32) or value.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("Slot activity and position must be INT32/UINT32 ROW_MAJOR")
    if prod(value.shape) != 1 or value.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError("Slot activity and position must be interleaved DRAM scalars")


def _program(mesh, source, compile_args, runtime_rows, scratch_bytes):
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y, _ in runtime_rows])
    runtime = ttnn.RuntimeArgs()
    for x, y, args in runtime_rows:
        runtime[x][y] = args

    def kernel(src, config, args=(), rt=()):
        return ttnn.KernelDescriptor(
            kernel_source=src,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            config=config,
            compile_time_args=list(args),
            runtime_args=rt,
        )

    # Only the reader accesses this private scratch. Other processors have no
    # CB producer/consumer protocol, so an inactive reader can safely return.
    size = (scratch_bytes + 31) // 32 * 32
    return ttnn.ProgramDescriptor(
        kernels=[
            kernel(source, ttnn.ReaderConfigDescriptor(), compile_args, runtime),
            kernel(_EMPTY, ttnn.WriterConfigDescriptor()),
            kernel('#include "api/compute/common.h"\nvoid kernel_main() {}\n', ttnn.ComputeConfigDescriptor()),
        ],
        cbs=[
            ttnn.CBDescriptor(
                total_size=size, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, size)]
            )
        ],
    )


def copy_if(source, destination, activity, mesh, *, zero_if_inactive=False):
    """Copy exact bits when active; otherwise retain destination (or zero it).

    Source and destination may alias. Only interleaved DRAM BF16/F32/I32/U32
    tensors are supported; no layout/dtype conversion is hidden in this op.
    """
    _scalar(activity)
    if source.dtype not in (ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32):
        raise ValueError("Conditional copy requires an uncompressed numeric dtype")
    if source.dtype != destination.dtype or source.layout != destination.layout:
        raise ValueError("Conditional copy must preserve dtype and layout")
    if source.shape != destination.shape or source.padded_shape != destination.padded_shape:
        raise ValueError("Conditional copy must preserve logical and padded shape")
    if any(t.memory_config() != ttnn.DRAM_MEMORY_CONFIG for t in (source, destination)):
        raise ValueError("Conditional copy requires interleaved DRAM")
    element_bytes = 2 if source.dtype == ttnn.bfloat16 else 4
    volume = prod(source.padded_shape)
    if source.layout == ttnn.TILE_LAYOUT:
        pages, page_bytes = volume // 1024, 1024 * element_bytes
    else:
        pages, page_bytes = volume // source.padded_shape[-1], source.padded_shape[-1] * element_bytes
    # Every page starts at the device allocator's DRAM alignment. Reserve an
    # extra 64 bytes so the L1 scratch can match source address bits 0..5.
    grid = mesh.compute_with_storage_grid_size()
    workers = min(pages, grid.x * grid.y)
    rows, start = [], 0
    for i in range(workers):
        count = pages // workers + (i < pages % workers)
        rows.append(
            (
                i % grid.x,
                i // grid.x,
                [
                    source.buffer_address(),
                    destination.buffer_address(),
                    activity.buffer_address(),
                    start,
                    count,
                    page_bytes,
                    int(zero_if_inactive),
                ],
            )
        )
        start += count
    sa, da, aa = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in (source, destination, activity)]
    args = [2 + len(sa), 2 + len(sa) + len(da)] + sa + da + aa
    ttnn.generic_op([source, destination, activity], _program(mesh, _COPY, args, rows, max(page_bytes, 4) + 64))
    return destination


def position_inputs(position, activity, mesh):
    """Create safe RoPE index and SDPA/cache index; inactive cache index is -1."""
    _scalar(position)
    _scalar(activity)
    rope = ttnn.empty_like(position)
    current = ttnn.empty_like(position)
    _position_op(position, activity, rope, current, mesh, advance=False)
    return rope, current


def advance_position(position, activity, mesh):
    """Advance active absolute positions on device; inactive bytes stay intact."""
    _scalar(position)
    _scalar(activity)
    _position_op(position, activity, position, position, mesh, advance=True)


def _position_op(position, activity, rope, current, mesh, *, advance):
    tensors = [position, activity, rope, current]
    accessors = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in tensors]
    offsets, offset = [], 3
    for args in accessors:
        offsets.append(offset)
        offset += len(args)
    args = offsets[1:] + sum(accessors, [])
    runtime = [(0, 0, [t.buffer_address() for t in tensors] + [int(advance)])]
    ttnn.generic_op(tensors, _program(mesh, _POSITION, args, runtime, 192))


def _rows_op(staging, destination, active_mask, mesh, *, advance):
    tensors = [staging, destination, active_mask]
    for tensor in tensors:
        if tensor.dtype not in (ttnn.int32, ttnn.uint32) or tensor.dtype != destination.dtype:
            raise ValueError("Slot vectors must have the same INT32/UINT32 dtype")
        if tensor.layout != ttnn.ROW_MAJOR_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("Slot vectors must be ROW_MAJOR interleaved DRAM")
        if prod(tensor.shape) != 32 or tensor.shape[-1] != 32 or prod(tensor.padded_shape) != 32:
            raise ValueError("Slot vectors must occupy one contiguous 32-entry row")
    sa, da, aa = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in tensors]
    args = [2 + len(sa), 2 + len(sa) + len(da)] + sa + da + aa
    runtime = [(0, 0, [t.buffer_address() for t in tensors] + [int(advance)])]
    ttnn.generic_op(tensors, _program(mesh, _ROWS, args, runtime, 448))
    return destination


def merge_rows(staging, destination, active_mask, mesh):
    """Merge active 32-entry integer rows, preserving inactive bits exactly."""
    return _rows_op(staging, destination, active_mask, mesh, advance=False)


def advance_seeds(seeds, active_mask, mesh):
    """Increment only active seeds on device, using uint32 wraparound."""
    return _rows_op(seeds, seeds, active_mask, mesh, advance=True)


def masked_convolution_reader(source, activity_offset):
    """Add only a runtime predicate around the selected conv history writes."""
    anchor = " auto* wide=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3)+bias);"
    assert source.count(anchor) == 1
    read = f"""
 const auto slot_mask=TensorAccessor(TensorAccessorArgs<{activity_offset}>(),get_arg_val<uint32_t>(8));
 noc.async_read(slot_mask,scratch,4,{{.page_id=0,.offset_bytes=0}},{{.offset_bytes=bias}});
 noc.async_read_barrier();
 const bool slot_active=wide[0]!=0;
"""
    source = source.replace(anchor, anchor + read)
    begin = "  // All source history words were fetched before any in-place carry writes."
    end = "  noc.async_write_barrier();"
    assert source.count(begin) == 1 and source.count(end) == 1
    return source.replace(begin, begin + "\n  if(slot_active) {").replace(end, end + "\n  }")
