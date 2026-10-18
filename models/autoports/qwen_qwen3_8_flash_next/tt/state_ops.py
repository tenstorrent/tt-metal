# SPDX-License-Identifier: Apache-2.0
"""Small native persistent-state operations; no host cache reconstruction."""

import math
import ttnn

_ZERO = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t bytes=get_compile_time_arg_val(0);
    constexpr auto ta=TensorAccessorArgs<1>();
    auto dst=TensorAccessor(ta,get_arg_val<uint32_t>(0));
    const uint32_t first=get_arg_val<uint32_t>(1), count=get_arg_val<uint32_t>(2);
    CircularBuffer cb(0); cb.reserve_back(1); uint32_t base=cb.get_write_ptr();
    auto w=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    for(uint32_t j=0;j<bytes/4;++j) w[j]=0;
    Noc noc;
    for(uint32_t i=first;i<first+count;++i) {
        noc.async_write(CoreLocalMem<uint32_t>(base),dst,bytes,{}, {.page_id=i});
        noc.async_write_barrier();
    }
}
"""

_ADVANCE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0);
    constexpr auto pa=TensorAccessorArgs<1>();
    constexpr auto aa=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    auto pos=TensorAccessor(pa,get_arg_val<uint32_t>(0));
    auto active=TensorAccessor(aa,get_arg_val<uint32_t>(1));
    CircularBuffer cb(0); cb.reserve_back(1); const uint32_t base=cb.get_write_ptr();
    auto w=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base); Noc noc;
    for(uint32_t b=0;b<B;++b) {
        noc.async_read(pos,CoreLocalMem<uint32_t>(base),128,{.page_id=b},{});
        noc.async_read(active,CoreLocalMem<uint32_t>(base+128),128,{.page_id=b},{});
        noc.async_read_barrier();
        w[0]=w[32] ? w[0]+1 : 0;
        noc.async_write(CoreLocalMem<uint32_t>(base),pos,128,{}, {.page_id=b});
        noc.async_write_barrier();
    }
}
"""


def zero_inplace(mesh, tensor):
    sizes = {ttnn.uint32: 4, ttnn.int32: 4, ttnn.float32: 4, ttnn.bfloat16: 2, ttnn.fp8_e4m3: 1}
    size = sizes[tensor.dtype]
    if tensor.layout == ttnn.TILE_LAYOUT:
        pagebytes = 1024 * size
        pages = math.prod(tuple(tensor.padded_shape)) // 1024
    else:
        pagebytes = int(tensor.padded_shape[-1]) * size
        pages = math.prod(tuple(tensor.padded_shape)[:-1])
    grid = mesh.compute_with_storage_grid_size()
    count = min(grid.x * grid.y, pages)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(count)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    args = [pagebytes, *ttnn.TensorAccessorArgs(tensor).get_compile_time_args()]
    runtime = []
    for i, c in enumerate(coords):
        first = pages * i // count
        stop = pages * (i + 1) // count
        runtime.append(((c.x, c.y), [tensor.buffer_address(), first, stop - first]))
    cb = ttnn.CBDescriptor(
        total_size=pagebytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=pagebytes)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=_ZERO,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=runtime,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([tensor, tensor], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))


def advance_positions(positions, active):
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    args = [positions.shape[0]]
    for t in (positions, active):
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    cb = ttnn.CBDescriptor(
        total_size=256,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=256)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=_ADVANCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[((0, 0), [positions.buffer_address(), active.buffer_address()])],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([positions, active], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))


_RECORD = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0), C=get_compile_time_arg_val(1);
    constexpr auto ta=TensorAccessorArgs<2>();
    constexpr auto pa=TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto ha=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    auto tok=TensorAccessor(ta,get_arg_val<uint32_t>(0));
    auto pos=TensorAccessor(pa,get_arg_val<uint32_t>(1));
    auto history=TensorAccessor(ha,get_arg_val<uint32_t>(2));
    CircularBuffer cb(0); cb.reserve_back(1); const uint32_t base=cb.get_write_ptr();
    auto w=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base); Noc noc;
    noc.async_read(tok,CoreLocalMem<uint32_t>(base),B*4,{.page_id=0},{});
    noc.async_read_barrier();
    for(uint32_t b=0;b<B;++b) {
        noc.async_read(pos,CoreLocalMem<uint32_t>(base+128),4,{.page_id=b},{});
        noc.async_read_barrier(); const uint32_t index=w[32]; ASSERT(index<C);
        w[48+(index%8)]=w[b];
        noc.async_write(CoreLocalMem<uint32_t>(base+192+(index%8)*4),history,4,{}, {.page_id=b,.offset_bytes=index*4});
        noc.async_write_barrier();
    }
}
"""


def record_tokens(tokens, positions, history):
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    args = [positions.shape[0], history.shape[-1]]
    tensors = [tokens, positions, history]
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    cb = ttnn.CBDescriptor(
        total_size=256,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=256)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=_RECORD,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[((0, 0), [t.buffer_address() for t in tensors])],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))


_SAMPLE_SEEDS = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0);
    constexpr auto sa=TensorAccessorArgs<1>();
    constexpr auto pa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    auto seeds=TensorAccessor(sa,get_arg_val<uint32_t>(0));
    auto pos=TensorAccessor(pa,get_arg_val<uint32_t>(1));
    auto out=TensorAccessor(oa,get_arg_val<uint32_t>(2));
    CircularBuffer cb(0); cb.reserve_back(1); const uint32_t base=cb.get_write_ptr();
    auto w=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base); Noc noc;
    noc.async_read(seeds,CoreLocalMem<uint32_t>(base),128,{.page_id=0},{});
    noc.async_read_barrier();
    for(uint32_t b=0;b<32;++b) w[64+b]=1;
    for(uint32_t b=0;b<B;++b) {
        noc.async_read(pos,CoreLocalMem<uint32_t>(base+128),4,{.page_id=b},{});
        noc.async_read_barrier();
        // Same SplitMix64 finalizer as common SamplingGenerator's
        // _hash_request_seed_to_device_seed(seed, counter, salt=row).
        // Absolute device position is the counter; no host token-step update.
        uint64_t value=uint64_t(w[b]) ^ (uint64_t(w[32])+0x9E3779B97F4A7C15ULL);
        value ^= uint64_t(b)*0xD1B54A32D192ED03ULL;
        value=(value^(value>>30))*0xBF58476D1CE4E5B9ULL;
        value=(value^(value>>27))*0x94D049BB133111EBULL;
        value=value^(value>>31);
        w[64+b]=uint32_t(value%1000000ULL)+1;
    }
    noc.async_write(CoreLocalMem<uint32_t>(base+256),out,128,{}, {.page_id=0});
    noc.async_write_barrier();
}
"""


def prepare_sample_seeds(request_seeds, positions, output):
    """Common seed derivation on device; Sampling1D still owns all sampling."""
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    tensors = [request_seeds, positions, output]
    args = [positions.shape[0]]
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    cb = ttnn.CBDescriptor(
        total_size=384,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=384)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=_SAMPLE_SEEDS,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[((0, 0), [t.buffer_address() for t in tensors])],
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
