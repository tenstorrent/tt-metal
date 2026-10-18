# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact FP32 token-loop prefill experiment, isolated from production runtime.

Each worker owns complete heads and processes logical tokens only. FP32 state
is transferred through disjoint DRAM head pages between tokens; output rows are
written once to row-major[B,S,48,128], then tilized by the public TTNN operation.
"""

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.gdn import GatedDeltaNet

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

uint32_t off(uint32_t r, uint32_t c) {
    return (r/16)*512 + (c/16)*256 + (r%16)*16 + c%16;
}
void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto fa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    const auto decay = TensorAccessor(da, get_arg_val<uint32_t>(3));
    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(4));
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(5));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(6));
    const auto final = TensorAccessor(fa, get_arg_val<uint32_t>(7));
    const uint32_t first = get_arg_val<uint32_t>(8), count = get_arg_val<uint32_t>(9);
    const uint32_t length=get_arg_val<uint32_t>(10), st=(length+31)/32;
    Noc noc;
    cb_reserve_back(14, 1);
    const uint32_t scratch = get_write_ptr(14);
    auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<6, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL>();
    for (uint32_t job = first; job < first + count; ++job) {
        const uint32_t b = job/48, h = job%48;
        for(uint32_t token=0;token<length;++token) {
        cb_reserve_back(0,16);
        for (uint32_t i=0;i<16;++i) {
            if(token==0) noc.async_read(state, CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096),4096,
                {.page_id=job*16+i},{});
            else noc.async_read(final, CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096),4096,
                {.page_id=job*16+i},{});
        }
        noc.async_read_barrier();
        cb_push_back(0,16);
        // Raw 32-bit replication: no floating point conversion on dataflow RISCs.
        for (uint32_t which=1;which<=3;++which) {
            cb_reserve_back(which,4);
            auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for (uint32_t tile=0;tile<4;++tile) {
                uint32_t page = which==3 ? ((b*length+token)*2+h/32)*4+tile : (b*length+token)*4+tile;
                if (which==1) noc.async_read(q,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==2) noc.async_read(k,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                noc.async_read_barrier();
                if (which==3) {
                    for(uint32_t c=0;c<32;++c) dst[tile*1024+off(0,c)]=src[off(h%32,c)];
                } else {
                    for(uint32_t r=0;r<32;++r) {
                        const uint32_t value=src[off(h/3,r)];
                        for(uint32_t c=0;c<16;++c) {
                            dst[tile*1024+off(r,c)]=value;
                            dst[tile*1024+off(r,c+16)]=value;
                        }
                    }
                }
            }
            cb_push_back(which,4);
        }
        for(uint32_t which=4;which<=5;++which) {
            cb_reserve_back(which,1);
            if(which==4) noc.async_read(decay,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=(b*st+token/32)*2+h/32},{});
            else noc.async_read(beta,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=(b*st+token/32)*2+h/32},{});
            noc.async_read_barrier();
            const uint32_t value=src[off(token%32,h%32)];
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for(uint32_t i=0;i<1024;++i) dst[i]=value;
            cb_push_back(which,1);
        }
        cb_wait_front(12,16);
        for(uint32_t i=0;i<16;++i)
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(12)+i*4096),final,4096,{},
                {.page_id=job*16+i});
        noc.async_write_barrier();
        // Final state is still read by compute until output has been published.
        cb_wait_front(13,4);
        auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(13));
        for(uint32_t t=0;t<4;++t) for(uint32_t c=0;c<32;++c)
            src[t*32+c]=out[t*1024+off(0,c)];
        noc.async_write(CoreLocalMem<uint32_t>(scratch),output,512,{},
            {.page_id=(b*length+token)*48+h});
        noc.async_write_barrier();
        cb_pop_front(13,4);
        cb_pop_front(12,16);
        }
    }
}
"""

TILED_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

uint32_t off(uint32_t r, uint32_t c) {
    return (r/16)*512 + (c/16)*256 + (r%16)*16 + c%16;
}
void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto fa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    const auto decay = TensorAccessor(da, get_arg_val<uint32_t>(3));
    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(4));
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(5));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(6));
    const auto final = TensorAccessor(fa, get_arg_val<uint32_t>(7));
    const uint32_t first = get_arg_val<uint32_t>(8), count = get_arg_val<uint32_t>(9);
    const uint32_t length=get_arg_val<uint32_t>(10), st=(length+31)/32;
    cb_reserve_back(15,4);
    auto assembled=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(15));
    Noc noc;
    cb_reserve_back(14, 1);
    const uint32_t scratch = get_write_ptr(14);
    auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<6, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL>();
    for (uint32_t job = first; job < first + count; ++job) {
        const uint32_t b = job/48, h = job%48;
        for(uint32_t token=0;token<length;++token) {
        cb_reserve_back(0,16);
        for (uint32_t i=0;i<16;++i) {
            if(token==0) noc.async_read(state, CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096),4096,
                {.page_id=job*16+i},{});
            else noc.async_read(final, CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096),4096,
                {.page_id=job*16+i},{});
        }
        noc.async_read_barrier();
        cb_push_back(0,16);
        // Raw 32-bit replication: no floating point conversion on dataflow RISCs.
        for (uint32_t which=1;which<=3;++which) {
            cb_reserve_back(which,4);
            auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for (uint32_t tile=0;tile<4;++tile) {
                uint32_t page = which==3 ? ((b*length+token)*2+h/32)*4+tile : (b*length+token)*4+tile;
                if (which==1) noc.async_read(q,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==2) noc.async_read(k,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                noc.async_read_barrier();
                if (which==3) {
                    for(uint32_t c=0;c<32;++c) dst[tile*1024+off(0,c)]=src[off(h%32,c)];
                } else {
                    for(uint32_t r=0;r<32;++r) {
                        const uint32_t value=src[off(h/3,r)];
                        for(uint32_t c=0;c<16;++c) {
                            dst[tile*1024+off(r,c)]=value;
                            dst[tile*1024+off(r,c+16)]=value;
                        }
                    }
                }
            }
            cb_push_back(which,4);
        }
        for(uint32_t which=4;which<=5;++which) {
            cb_reserve_back(which,1);
            if(which==4) noc.async_read(decay,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=(b*st+token/32)*2+h/32},{});
            else noc.async_read(beta,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=(b*st+token/32)*2+h/32},{});
            noc.async_read_barrier();
            const uint32_t value=src[off(token%32,h%32)];
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for(uint32_t i=0;i<1024;++i) dst[i]=value;
            cb_push_back(which,1);
        }
        cb_wait_front(12,16);
        for(uint32_t i=0;i<16;++i)
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(12)+i*4096),final,4096,{},
                {.page_id=job*16+i});
        noc.async_write_barrier();
        // Final state is still read by compute until output has been published.
        cb_wait_front(13,4);
        auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(13));
        if(token%32==0) for(uint32_t i=0;i<4096;++i) assembled[i]=0;
        for(uint32_t t=0;t<4;++t) for(uint32_t c=0;c<32;++c)
            assembled[t*1024+off(token%32,c)]=out[t*1024+off(0,c)];
        if(token%32==31 || token+1==length) {
            for(uint32_t t=0;t<4;++t)
                noc.async_write(CoreLocalMem<uint32_t>(get_write_ptr(15)+t*4096),output,4096,{},
                    {.page_id=(job*st+token/32)*4+t});
            noc.async_write_barrier();
        }
        cb_pop_front(13,4);
        cb_pop_front(12,16);
        }
    }
}
"""

ALIAS_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

uint32_t off(uint32_t r, uint32_t c) {
    return (r/16)*512 + (c/16)*256 + (r%16)*16 + c%16;
}
void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto fa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    const auto decay = TensorAccessor(da, get_arg_val<uint32_t>(3));
    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(4));
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(5));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(6));
    const auto final = TensorAccessor(fa, get_arg_val<uint32_t>(7));
    const uint32_t first = get_arg_val<uint32_t>(8), count = get_arg_val<uint32_t>(9);
    const uint32_t length=get_arg_val<uint32_t>(10), st=(length+31)/32;
    cb_reserve_back(15,4);
    auto assembled=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(15));
    Noc noc;
    cb_reserve_back(14, 1);
    const uint32_t scratch = get_write_ptr(14);
    auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<6, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL>();
    for (uint32_t job = first; job < first + count; ++job) {
        const uint32_t b = job/48, h = job%48;
        for(uint32_t token=0;token<length;++token) {
        cb_reserve_back(0,16);
        for (uint32_t i=0;i<16;++i) {
            if(token==0) noc.async_read(state, CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096),4096,
                {.page_id=job*16+i},{});
            // Later tokens reuse the FP32 bytes through aliased CB0/CB12.

        }
        noc.async_read_barrier();
        cb_push_back(0,16);
        // Raw 32-bit replication: no floating point conversion on dataflow RISCs.
        for (uint32_t which=1;which<=3;++which) {
            cb_reserve_back(which,4);
            auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for (uint32_t tile=0;tile<4;++tile) {
                uint32_t page = which==3 ? ((b*length+token)*2+h/32)*4+tile : (b*length+token)*4+tile;
                if (which==1) noc.async_read(q,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==2) noc.async_read(k,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                noc.async_read_barrier();
                if (which==3) {
                    for(uint32_t c=0;c<32;++c) dst[tile*1024+off(0,c)]=src[off(h%32,c)];
                } else {
                    for(uint32_t r=0;r<32;++r) {
                        const uint32_t value=src[off(h/3,r)];
                        for(uint32_t c=0;c<16;++c) {
                            dst[tile*1024+off(r,c)]=value;
                            dst[tile*1024+off(r,c+16)]=value;
                        }
                    }
                }
            }
            cb_push_back(which,4);
        }
        for(uint32_t which=4;which<=5;++which) {
            cb_reserve_back(which,1);
            if(which==4) noc.async_read(decay,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=(b*st+token/32)*2+h/32},{});
            else noc.async_read(beta,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=(b*st+token/32)*2+h/32},{});
            noc.async_read_barrier();
            const uint32_t value=src[off(token%32,h%32)];
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for(uint32_t i=0;i<1024;++i) dst[i]=value;
            cb_push_back(which,1);
        }
        cb_wait_front(12,16);
        if(token+1==length) for(uint32_t i=0;i<16;++i)
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(12)+i*4096),final,4096,{},
                {.page_id=job*16+i});
        noc.async_write_barrier();
        // Final state is still read by compute until output has been published.
        cb_wait_front(13,4);
        auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(13));
        if(token%32==0) for(uint32_t i=0;i<4096;++i) assembled[i]=0;
        for(uint32_t t=0;t<4;++t) for(uint32_t c=0;c<32;++c)
            assembled[t*1024+off(token%32,c)]=out[t*1024+off(0,c)];
        if(token%32==31 || token+1==length) {
            for(uint32_t t=0;t<4;++t)
                noc.async_write(CoreLocalMem<uint32_t>(get_write_ptr(15)+t*4096),output,4096,{},
                    {.page_id=(job*st+token/32)*4+t});
            noc.async_write_barrier();
        }
        cb_pop_front(13,4);
        cb_pop_front(12,16);
        }
    }
}
"""

COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void load(uint32_t cb,uint32_t tile,uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb,tile,dst);
}
void finish(uint32_t cb,uint32_t tile) {
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,cb,tile); tile_regs_release();
}
template<uint32_t Out> void reduce_product() {
    cb_wait_front(8,16);
    compute_kernel_lib::reduce<PoolType::SUM,ReduceDim::REDUCE_COL,8,6,Out,
        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Accurate>(compute_kernel_lib::ReduceInputBlockShape::of(4,4));
    cb_pop_front(8,16);
}
void product(uint32_t state,uint32_t column) {
    cb_wait_front(state,16); cb_wait_front(column,4); cb_reserve_back(8,16);
    pack_reconfig_data_format(8);
    for(uint32_t i=0;i<16;++i) {
        tile_regs_acquire(); load(state,i,0); load(column,i/4,1);
        mul_binary_tile_init(); mul_binary_tile(0,1,0); finish(8,i);
    }
    cb_push_back(8,16);
}
void kernel_main() {
    compute_kernel_hw_startup(0,4,7);
    const uint32_t count=get_arg_val<uint32_t>(0);
    for(uint32_t job=0;job<count;++job) {
        cb_wait_front(0,16); cb_wait_front(4,1); cb_reserve_back(7,16);
        pack_reconfig_data_format(7);
        for(uint32_t i=0;i<16;++i) {
            tile_regs_acquire(); load(0,i,0); load(4,0,1);
            mul_binary_tile_init(); mul_binary_tile(0,1,0); finish(7,i);
        }
        cb_push_back(7,16); cb_pop_front(0,16); cb_pop_front(4,1);
        product(7,2); reduce_product<9>();
        cb_wait_front(9,4); cb_wait_front(3,4); cb_wait_front(5,1);
        cb_reserve_back(11,4); pack_reconfig_data_format(11);
        for(uint32_t i=0;i<4;++i) {
            tile_regs_acquire(); load(3,i,0); load(9,i,1); load(5,0,2);
            sub_binary_tile_init(); sub_binary_tile(0,1,0);
            mul_binary_tile_init(); mul_binary_tile(0,2,0); finish(11,i);
        }
        cb_push_back(11,4); cb_pop_front(9,4); cb_pop_front(3,4); cb_pop_front(5,1);
        cb_wait_front(11,4); cb_reserve_back(12,16); pack_reconfig_data_format(12);
        for(uint32_t i=0;i<16;++i) {
            tile_regs_acquire(); load(2,i/4,0); load(11,i%4,1); load(7,i,2);
            sfpu_mul_bcast_row_init(); sfpu_mul_bcast_row(0,1);
            add_binary_tile_init(); add_binary_tile(0,2,0); finish(12,i);
        }
        cb_push_back(12,16); cb_pop_front(11,4); cb_pop_front(7,16); cb_pop_front(2,4);
        product(12,1); reduce_product<13>(); cb_pop_front(1,4);
    }
    cb_pop_front(6,1);
}
"""


class PrecisePrefillGDN(GatedDeltaNet):
    tiled_output = False
    alias_state = False

    def _recurrence(self, q, k, v, g, beta):
        batch, length = q.shape[0], q.shape[1]
        q = ttnn.multiply(q, 128**-0.5)
        decay = ttnn.exp(g)
        shape = (batch, 48, length, 128) if self.tiled_output else (batch, length, 48, 128)
        layout = ttnn.TILE_LAYOUT if self.tiled_output else ttnn.ROW_MAJOR_LAYOUT
        output = ttnn.empty(
            shape, dtype=ttnn.float32, layout=layout, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        final = ttnn.empty(
            (batch, 48, 128, 128),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tensors = [q, k, v, decay, beta, self.recurrent_state, output, final]
        grid = self.device.compute_with_storage_grid_size()
        workers = min(batch * 48, grid.x * grid.y)
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        args = []
        for tensor in tensors:
            args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        assignments = [
            (i * (batch * 48) // workers, (i + 1) * (batch * 48) // workers - i * (batch * 48) // workers)
            for i in range(workers)
        ]
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
        )
        config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=ALIAS_READER if self.alias_state else TILED_READER if self.tiled_output else READER,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=args,
                config=ttnn.ReaderConfigDescriptor(),
                runtime_args=[
                    ((c.x, c.y), [*[t.buffer_address() for t in tensors], *job, length])
                    for c, job in zip(coords, assignments)
                ],
            ),
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=config,
                runtime_args=[((c.x, c.y), [job[1] * length]) for c, job in zip(coords, assignments)],
            ),
        ]
        sizes = [16, 4, 4, 4, 1, 1, 1, 16, 16, 4, 1, 4, 16, 4, 1]
        if self.tiled_output:
            sizes.append(4)
        cbs = [
            ttnn.CBDescriptor(
                total_size=count * 4096,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)],
            )
            for i, count in enumerate(sizes)
            if not (self.alias_state and i in (0, 12))
        ]
        if self.alias_state:
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=16 * 4096,
                    core_ranges=cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)
                        for i in (0, 12)
                    ],
                )
            )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs))
        result = ttnn.permute(output, (0, 2, 1, 3)) if self.tiled_output else ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        return result, final


class PrecisePrefillTiledGDN(PrecisePrefillGDN):
    tiled_output = True


class PrecisePrefillAliasGDN(PrecisePrefillTiledGDN):
    alias_state = True
