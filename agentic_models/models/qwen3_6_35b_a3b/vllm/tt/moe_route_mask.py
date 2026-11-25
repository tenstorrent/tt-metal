# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Replay-time grouped-MoE route suppression for inactive fixed-slot rows.

Active UINT32 route IDs retain every bit. Inactive/padded rows use a valid
expert owned by another rank, so the existing local dispatch/reduce tables
skip them. The caller must use the returned IDs for counts AND both routing
consumers. This helper changes neither scores nor expert arithmetic.
"""

from math import prod

import ttnn

from .slot_state import _program

_MASK = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr uint32_t doff=get_compile_time_arg_val(0), aoff=get_compile_time_arg_val(1);
 constexpr uint32_t toff=get_compile_time_arg_val(2);
 const auto ids=TensorAccessor(TensorAccessorArgs<3>(),get_arg_val<uint32_t>(0));
 const auto out=TensorAccessor(TensorAccessorArgs<doff>(),get_arg_val<uint32_t>(1));
 const auto active=TensorAccessor(TensorAccessorArgs<aoff>(),get_arg_val<uint32_t>(2));
 const auto table=TensorAccessor(TensorAccessorArgs<toff>(),get_arg_val<uint32_t>(3));
 const uint32_t logical_rows=get_arg_val<uint32_t>(4);
 Noc noc; DataflowBuffer scratch(0); scratch.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(0)&63))&63;
 auto* routes=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias);
 auto* activity=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias+4096);
 auto* owners=reinterpret_cast<volatile tt_l1_ptr int32_t*>(get_write_ptr(0)+bias+4224);
 noc.async_read(ids,scratch,4096,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias});
 noc.async_read(active,scratch,128,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias+4096});
 noc.async_read(table,scratch,1024,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias+4224});
 noc.async_read_barrier();
 // The selected EP4 table owns r*64..(r+1)*64. Rank0 skips ID64;
 // ranks1..3 skip ID0. Both IDs are in range and table[skip] is -1.
 const uint32_t skip=owners[0]<0?0:64;
 for(uint32_t row=0;row<32;++row) {
  const bool live=row<logical_rows && activity[get_arg_val<uint32_t>(5+row)]!=0;
  if(!live) {
   // Standard 32x32 UINT32 tile: four 16x16 faces. Only top-eight
   // logical columns are routes; preserve all physical padding bits.
   const uint32_t offset=(row/16)*512+(row%16)*16;
   for(uint32_t k=0;k<8;++k) routes[offset+k]=skip;
  }
 }
 noc.async_write(scratch,out,4096,{.offset_bytes=bias},{.page_id=0,.offset_bytes=0});
 noc.async_write_barrier();
}
"""


_PLAN = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
void kernel_main() {
 constexpr uint32_t doff=get_compile_time_arg_val(0), aoff=get_compile_time_arg_val(1);
 constexpr uint32_t toff=get_compile_time_arg_val(2), coff=get_compile_time_arg_val(3);
 constexpr uint32_t poff=get_compile_time_arg_val(4);
 const auto ids=TensorAccessor(TensorAccessorArgs<5>(),get_arg_val<uint32_t>(0));
 const auto out=TensorAccessor(TensorAccessorArgs<doff>(),get_arg_val<uint32_t>(1));
 const auto active=TensorAccessor(TensorAccessorArgs<aoff>(),get_arg_val<uint32_t>(2));
 const auto table=TensorAccessor(TensorAccessorArgs<toff>(),get_arg_val<uint32_t>(3));
 const auto counts_out=TensorAccessor(TensorAccessorArgs<coff>(),get_arg_val<uint32_t>(4));
 const auto offsets_out=TensorAccessor(TensorAccessorArgs<poff>(),get_arg_val<uint32_t>(5));
 const uint32_t logical_rows=get_arg_val<uint32_t>(6);
 Noc noc; DataflowBuffer scratch(0); scratch.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(0)&63))&63;
 auto* routes=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias);
 auto* activity=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias+4096);
 auto* owners=reinterpret_cast<volatile tt_l1_ptr int32_t*>(get_write_ptr(0)+bias+4224);
 auto* counts=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias+5248);
 auto* offsets=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias+6272);
 noc.async_read(ids,scratch,4096,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias});
 noc.async_read(active,scratch,128,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias+4096});
 noc.async_read(table,scratch,1024,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias+4224});
 noc.async_read_barrier();
 for(uint32_t e=0;e<256;++e) counts[e]=0;
 const uint32_t skip=owners[0]<0?0:64;
 ASSERT(owners[skip]<0);
 for(uint32_t row=0;row<32;++row) {
  const bool live=row<logical_rows && activity[get_arg_val<uint32_t>(7+row)]!=0;
  const uint32_t offset=(row/16)*512+(row%16)*16;
  for(uint32_t k=0;k<8;++k) {
   if(!live) routes[offset+k]=skip;
   else {
    // TopK carries distinct column indices; each live expert occurs once per row.
    const uint32_t expert=routes[offset+k];
    ASSERT(expert<256);
    if(expert<256 && owners[expert]>=0) counts[expert]=counts[expert]+1;
   }
  }
 }
 uint32_t running=0;
 for(uint32_t e=0;e<256;++e) {
  offsets[e]=running;
  running+=((counts[e]+31)/32)*32;
 }
 // Complete processor stores before NoC reads scratch, following index_tile_dataflow.hpp.
 (void)ckernel::load_blocking(offsets+255);
 noc.async_write(scratch,out,4096,{.offset_bytes=bias},{.page_id=0,.offset_bytes=0});
 noc.async_write(scratch,counts_out,1024,{.offset_bytes=bias+5248},{.page_id=0,.offset_bytes=0});
 noc.async_write(scratch,offsets_out,1024,{.offset_bytes=bias+6272},{.page_id=0,.offset_bytes=0});
 noc.async_write_barrier();
}
"""


def _validate_route_inputs(ids, active_mask, dispatch_table, slots):
    slots = tuple(slots)
    if not 2 <= len(slots) <= 32 or len(set(slots)) != len(slots):
        raise ValueError("Grouped route masking requires 2..32 unique physical slots")
    if any(type(slot) is not int or not 0 <= slot < 32 for slot in slots):
        raise ValueError("Route slot is outside the persistent activity vector")
    if ids.dtype != ttnn.uint32 or ids.layout != ttnn.TILE_LAYOUT:
        raise ValueError("Grouped route masking requires UINT32 TILE IDs from the FP32 router")
    if tuple(ids.shape) != (1, 1, 32, 8) or tuple(ids.padded_shape) != (1, 1, 32, 32):
        raise ValueError("Grouped route masking requires exactly one top-eight route tile")
    if active_mask.dtype != ttnn.uint32 or active_mask.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("Route activity must be the persistent UINT32 ROW_MAJOR mask")
    if prod(active_mask.shape) != 32 or prod(active_mask.padded_shape) != 32 or active_mask.shape[-1] != 32:
        raise ValueError("Route activity must occupy one 32-word page")
    if dispatch_table.dtype != ttnn.int32 or dispatch_table.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("Route ownership table must be INT32 ROW_MAJOR")
    if tuple(dispatch_table.shape) != (1, 256) or tuple(dispatch_table.padded_shape) != (1, 256):
        raise ValueError("Route ownership table must be the rank-local EP4 table")
    if any(t.memory_config() != ttnn.DRAM_MEMORY_CONFIG for t in (ids, active_mask, dispatch_table)):
        raise ValueError("Route mask tensors must use interleaved DRAM")
    return slots


def mask_inactive_routes(ids, active_mask, dispatch_table, slots, mesh):
    """Return one masked route tile, reading persistent activity on every replay.

    Slots map the unpadded rows to physical request slots. Extra rows introduced
    by grouped-MoE tile padding are always inactive, even if another live slot
    happens to share their row number. All inputs stay device-resident.
    """
    slots = _validate_route_inputs(ids, active_mask, dispatch_table, slots)
    output = ttnn.empty_like(ids)
    tensors = [ids, output, active_mask, dispatch_table]
    accessors = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in tensors]
    offsets, offset = [], 3
    for args in accessors:
        offsets.append(offset)
        offset += len(args)
    compile_args = offsets[1:] + sum(accessors, [])
    runtime = [(0, 0, [t.buffer_address() for t in tensors] + [len(slots)] + list(slots) + [0] * (32 - len(slots)))]
    ttnn.generic_op(tensors, _program(mesh, _MASK, compile_args, runtime, 5248 + 64))
    return output


def plan_inactive_routes(ids, active_mask, dispatch_table, slots, mesh):
    """Return masked IDs plus UINT32 ROW_MAJOR (1, 256) counts and row offsets.

    Live rows must contain eight distinct in-range TopK expert IDs. The helper
    reads persistent activity on every replay and rewrites every output word.
    Ownership filtering matches the selected local EP4 dispatch table.
    """
    slots = _validate_route_inputs(ids, active_mask, dispatch_table, slots)
    output = ttnn.empty_like(ids)
    # empty_like preserves the ownership tensor's per-device (1, 256) shape
    # and mesh topology, while selecting unsigned count/offset representation.
    counts = ttnn.empty_like(dispatch_table, dtype=ttnn.uint32)
    offsets_out = ttnn.empty_like(dispatch_table, dtype=ttnn.uint32)
    tensors = [ids, output, active_mask, dispatch_table, counts, offsets_out]
    accessors = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in tensors]
    offsets, offset = [], 5
    for args in accessors:
        offsets.append(offset)
        offset += len(args)
    compile_args = offsets[1:] + sum(accessors, [])
    runtime = [(0, 0, [t.buffer_address() for t in tensors] + [len(slots)] + list(slots) + [0] * (32 - len(slots)))]
    ttnn.generic_op(tensors, _program(mesh, _PLAN, compile_args, runtime, 7296 + 64))
    return output, counts, offsets_out
