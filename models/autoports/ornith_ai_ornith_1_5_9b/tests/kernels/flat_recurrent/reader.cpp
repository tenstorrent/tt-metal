// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// One independent value head/core; extract its one token from flat BF16 conv.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ca=TensorAccessorArgs<0>();
    constexpr auto ga=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto ba=TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    const uint32_t head=get_arg_val<uint32_t>(0), batch=head/8, hv=head%8, hk=hv/2;
    const auto conv=TensorAccessor(ca,get_arg_val<uint32_t>(1),2048);
    const auto g=TensorAccessor(ga,get_arg_val<uint32_t>(2),4096);
    const auto beta=TensorAccessor(ba,get_arg_val<uint32_t>(3),2048);
    const auto state=TensorAccessor(sa,get_arg_val<uint32_t>(4),4096);
    CircularBuffer cs(5);cs.reserve_back(16);
    for(uint32_t t=0;t<16;++t)noc_async_read(state.get_noc_addr(head*16+t),get_write_ptr(5)+t*4096,4096);
    noc_async_read_barrier();cs.push_back(16);
    CircularBuffer scratch(28);scratch.reserve_back(1);
    const uint32_t stage=get_write_ptr(28);
    const uint32_t row_offset=(batch/16)*1024+(batch%16)*32;
    for(uint32_t which=0;which<3;++which){
        const uint32_t cb=20+which;
        CircularBuffer c(cb);c.reserve_back(4);
        const uint32_t base=get_write_ptr(cb);
        volatile tt_l1_ptr uint32_t* zero=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
        for(uint32_t i=0;i<2048;++i)zero[i]=0;
        const uint32_t col=which==0?hk*4:(which==1?16+hk*4:32+hv*4);
        for(uint32_t t=0;t<4;++t){
            const uint64_t addr=conv.get_noc_addr(col+t);
            if constexpr (ca.is_dram) {
                // DRAM transfers require matching64-byte address alignment.
                // Read aligned neighboring row pairs, then extract our row.
                const uint32_t aligned=row_offset & ~63u, phase=row_offset & 63u;
                noc_async_read(addr+aligned,stage,64);
                noc_async_read(addr+aligned+512,stage+64,64);
                noc_async_read_barrier();
                for(uint32_t word=0;word<8;++word) {
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+t*2048)[word]=
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage+phase)[word];
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+t*2048+512)[word]=
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage+64+phase)[word];
                }
            } else {
                noc_async_read(addr+row_offset,base+t*2048,32);
                noc_async_read(addr+row_offset+512,base+t*2048+512,32);
            }
        }
        noc_async_read_barrier();c.push_back(4);
    }
    // Scalar head gates: only [0,0] is active; all other rows/columns stay zero.
    CircularBuffer cg(23),cb(24);cg.reserve_back(1);cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* gp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(23));
    volatile tt_l1_ptr uint32_t* bp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(24));
    for(uint32_t i=0;i<1024;++i)gp[i]=0;
    for(uint32_t i=0;i<512;++i)bp[i]=0;
    noc_async_read(g.get_noc_addr(batch),stage,4096);noc_async_read_barrier();
    gp[0]=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[hv];
    noc_async_read(beta.get_noc_addr(batch),stage,2048);noc_async_read_barrier();
    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(bp)[0]=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage)[hv];
    cg.push_back(1);cb.push_back(1);
    CircularBuffer unit(27);unit.reserve_back(1);
    volatile tt_l1_ptr uint32_t* u=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(27));
    for(uint32_t i=0;i<512;++i)u[i]=0;
    for(uint32_t f=0;f<4;++f)for(uint32_t j=0;j<8;++j)u[f*128+j]=0x3F803F80u;
    unit.push_back(1);
}
