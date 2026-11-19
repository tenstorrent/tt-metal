# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact tiled BF16 row extraction with persistent fixed-slot activity masking."""

from math import prod

import ttnn

from .slot_state import _program

_EXTRACT = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
constexpr uint32_t face(uint32_t r,uint32_t c) {
 return (r/16)*512+(c/16)*256+(r%16)*16+c%16;
}
// Preserve the existing untilize/slice/tilize BF16 transport semantics,
// verified exhaustively over all 65536 input bit patterns on every rank.
constexpr uint32_t canonical_bf16(uint32_t bits) {
 const uint32_t exponent=bits&0x7f80;
 return exponent==0 ? 0 : (exponent==0x7f80 ? bits&0xff80 : bits);
}
constexpr uint32_t canonical_pair(uint32_t bits) {
 return canonical_bf16(bits&0xffff)|(canonical_bf16(bits>>16)<<16);
}
void kernel_main() {
 constexpr uint32_t ao=get_compile_time_arg_val(0), oo=get_compile_time_arg_val(1);
 const auto source=TensorAccessor(TensorAccessorArgs<2>(),get_arg_val<uint32_t>(0));
 const auto active=TensorAccessor(TensorAccessorArgs<ao>(),get_arg_val<uint32_t>(1));
 const uint32_t first=get_arg_val<uint32_t>(2), stride=get_arg_val<uint32_t>(3);
 const uint32_t pages=get_arg_val<uint32_t>(4), rows=get_arg_val<uint32_t>(5);
 Noc noc; DataflowBuffer scratch(0); scratch.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(0)&63))&63;
 auto* input=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias);
 auto* output=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias+2048);
 auto* activity=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias+4096);
 noc.async_read(active,scratch,128,{.page_id=0},{.offset_bytes=bias+4096});
 noc.async_read_barrier();
 // Every output has a single logical row. Padding stays zero and is never
 // overwritten; only the sixteen uint32 words of row zero change below.
 for(uint32_t word=0;word<512;++word) output[word]=0;
 (void)ckernel::load_blocking(output+511);
 for(uint32_t page=first;page<pages;page+=stride) {
  noc.async_read(source,scratch,2048,{.page_id=page},{.offset_bytes=bias});
  noc.async_read_barrier();
  for(uint32_t row=0;row<rows;++row) {
   const uint32_t slot=get_arg_val<uint32_t>(6+rows+row);
   const bool live=activity[slot]!=0;
   for(uint32_t col=0;col<32;col+=2)
    output[face(0,col)/2]=live?canonical_pair(input[face(row,col)/2]):0;
   // Complete processor stores before NoC consumes the scratch tile.
   (void)ckernel::load_blocking(output+face(0,30)/2);
   const auto destination=TensorAccessor(TensorAccessorArgs<oo>(),get_arg_val<uint32_t>(6+row));
   noc.async_write(scratch,destination,2048,{.offset_bytes=bias+2048},{.page_id=page});
   // This scratch tile is reused for the next row only after completion.
   noc.async_write_barrier();
  }
 }
}
"""


def split_masked_hidden_rows(source, active_mask, slots, mesh, *, outputs=None):
    """Copy logical row i into an independent singleton, or zero it if inactive.

    All padded rows are explicitly zero. Bit normalization matches the existing
    BF16 split chain, including zero, subnormal and NaN behavior. No host tensor
    refresh occurs here. Optional outputs support prepared probes;
    ordinary graph capture retains the independently allocated output buffers.
    """
    slots = tuple(slots)
    rows = len(slots)
    if rows not in (2, 4, 32) or len(set(slots)) != rows:
        raise ValueError("Hidden row extraction requires 2, 4 or 32 unique slots")
    if any(type(slot) is not int or not 0 <= slot < 32 for slot in slots):
        raise ValueError("Hidden row slot is outside the activity vector")
    width = source.shape[-1]
    if (
        tuple(source.shape) != (1, rows, width)
        or tuple(source.padded_shape) != (1, 32, width)
        or width <= 0
        or width % 32
    ):
        raise ValueError("Hidden source must be one tile-row with tile-aligned width")
    if source.dtype != ttnn.bfloat16 or source.layout != ttnn.TILE_LAYOUT:
        raise ValueError("Hidden row extraction requires BF16 TILE source")
    if (
        active_mask.dtype != ttnn.uint32
        or active_mask.layout != ttnn.ROW_MAJOR_LAYOUT
        or prod(active_mask.shape) != 32
        or prod(active_mask.padded_shape) != 32
        or active_mask.shape[-1] != 32
    ):
        raise ValueError("Activity must be one UINT32 ROW_MAJOR page of 32 entries")
    if any(t.memory_config() != ttnn.DRAM_MEMORY_CONFIG for t in (source, active_mask)):
        raise ValueError("Source and activity must use interleaved DRAM")
    if outputs is None:
        outputs = tuple(
            ttnn.empty(
                (1, 1, width),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in slots
        )
    else:
        outputs = tuple(outputs)
    if len(outputs) != rows:
        raise ValueError("One independent output is required per row")
    addresses = [source.buffer_address(), active_mask.buffer_address()]
    for output in outputs:
        if (
            tuple(output.shape) != (1, 1, width)
            or tuple(output.padded_shape) != (1, 32, width)
            or output.dtype != ttnn.bfloat16
            or output.layout != ttnn.TILE_LAYOUT
            or output.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("Hidden outputs must preserve singleton BF16 TILE DRAM geometry")
        addresses.append(output.buffer_address())
    if len(set(addresses)) != len(addresses):
        raise ValueError("Hidden extraction inputs and outputs must not alias")
    sa, aa, oa = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in (source, active_mask, outputs[0])]
    # Generic-op cache identity must distinguish the runtime-argument arity:
    # each bucket owns a different number of destination addresses and slots.
    compile_args = [2 + len(sa), 2 + len(sa) + len(aa)] + sa + aa + oa + [rows]
    pages = width // 32
    grid = mesh.compute_with_storage_grid_size()
    workers = min(pages, grid.x * grid.y)
    runtime = [
        (
            i % grid.x,
            i // grid.x,
            [source.buffer_address(), active_mask.buffer_address(), i, workers, pages, rows]
            + [t.buffer_address() for t in outputs]
            + list(slots),
        )
        for i in range(workers)
    ]
    ttnn.generic_op([source, active_mask, *outputs], _program(mesh, _EXTRACT, compile_args, runtime, 4288))
    return outputs
