# SPDX-License-Identifier: Apache-2.0
"""Pipelined expert dispatch and output compaction for the serial 1024-row prefill MoE group.

Same pages, bytes and output as prefill_hybrid's COMPACT1024: for every (route, 32-row output
tile) the BFP8 source tiles of all rows owned by this rank are read in one batch (one barrier per
output tile instead of one per row), then the rows are assembled exactly as before. The output tile
is cleared with word stores. Dispatch: one pass over (row, route) instead of one per owned expert, and
matched rows moved in batches behind one barrier. Toggles (default on): QWEN_PREFILL_COMPACT_PIPE=0 and
QWEN_PREFILL_DISPATCH_PIPE=0 restore the serial kernels.
"""

import os

from .multichip_decoder import _EXPERT_HEADER

SLOT_BYTES = 1088  # one BFP8 tile: 64 exponent bytes + 1024 mantissa bytes
EXTRA_CB_BYTES = 32 * SLOT_BYTES + 64


def compact_pipe_enabled():
    return os.environ.get("QWEN_PREFILL_COMPACT_PIPE", "1") != "0"


COMPACT1024_PIPE = (
    _EXPERT_HEADER
    + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0),T=(N+31)/32;
    constexpr auto da=TensorAccessorArgs<2>();
    constexpr auto ia=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ma=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    const auto down=TensorAccessor(da,get_arg_val<uint32_t>(0));
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto rank_tensor=TensorAccessor(ra,get_arg_val<uint32_t>(3));
    const auto mapping=TensorAccessor(ma,get_arg_val<uint32_t>(4));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(5));
    const uint32_t col=get_arg_val<uint32_t>(6);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    constexpr uint32_t ib=2176,sb=ib+N*128,mb=sb+N*64,rb=mb+N*128,slots=rb+64,SLOT=1088;
    for(uint32_t row=0;row<N;++row) {
        noc.async_read(ids,CoreLocalMem<uint32_t>(base+ib+row*128),128,{.page_id=row},{});
        noc.async_read(scores,CoreLocalMem<uint32_t>(base+sb+row*64),64,{.page_id=row},{});
        noc.async_read(mapping,CoreLocalMem<uint32_t>(base+mb+row*128),128,{.page_id=row},{});
    }
    noc.async_read(rank_tensor,CoreLocalMem<uint32_t>(base+rb),4,{.page_id=0},{});noc.async_read_barrier();
    const uint32_t rank=*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+rb);
    auto id=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+ib);
    auto score=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+sb);
    auto map=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+mb);
    auto out=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base+1088);
    auto out_words=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+1088);
    uint32_t rows[32],packs[32];
    for(uint32_t route=0;route<10;++route)for(uint32_t tile=0;tile<T;++tile) {
        uint32_t count=0;
        for(uint32_t row=tile*32;row<N && row<(tile+1)*32;++row) {
            const uint32_t e=id[row*32+route];ASSERT(e<512);
            if(e/128!=rank || score[row*32+route]==0)continue;
            const uint32_t packed=map[row*32+route];ASSERT(packed<1024);
            noc.async_read(down,CoreLocalMem<uint32_t>(base+slots+count*SLOT),1088,{.page_id=((e%128)*32+packed/32)*80+col},{});
            rows[count]=row;packs[count]=packed;++count;
        }
        for(uint32_t i=0;i<272;++i)out_words[i]=0;
        noc.async_read_barrier();
        for(uint32_t n=0;n<count;++n) {
            const uint32_t row=rows[n],packed=packs[n];
            auto src=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base+slots+n*SLOT);
            for(uint32_t side=0;side<2;++side) {
                const uint32_t sf=(packed%32/16)*2+side,df=(row%32/16)*2+side;
                out[df*16+row%16]=src[sf*16+packed%16];
                for(uint32_t j=0;j<16;++j)out[64+df*256+(row%16)*16+j]=src[64+sf*256+(packed%16)*16+j];
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(base+1088),output,1088,{}, {.page_id=(route*T+tile)*80+col});noc.async_write_barrier();
    }
}
"""
)


def dispatch_pipe_enabled():
    return os.environ.get("QWEN_PREFILL_DISPATCH_PIPE", "1") != "0"


DISPATCH_SLOTS = 8


def dispatch_cb_bytes(n):
    return DISPATCH_SLOTS * 5120 + n * 192 + 64 + DISPATCH_SLOTS * 16 + 128 * 16


# Same capacity rows, mapping words and counts as prefill_hybrid's DISPATCH1024. One pass over
# (row, route) in the original order assigns each expert's rows in the original order; matched rows
# are moved in batches of DISPATCH_SLOTS (one read barrier, one write barrier per batch).
DISPATCH1024_PIPE = (
    _EXPERT_HEADER
    + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0),W=get_compile_time_arg_val(1);
    constexpr auto xa=TensorAccessorArgs<2>();
    constexpr auto ia=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ca=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto ma=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0));
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto rank_tensor=TensorAccessor(ra,get_arg_val<uint32_t>(3));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(4));
    const auto mapping=TensorAccessor(ma,get_arg_val<uint32_t>(5));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(6));
    const uint32_t worker=get_arg_val<uint32_t>(7);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    constexpr uint32_t SL=8,RB=SL*5120,scratch=RB+N*192,maps=scratch+64,cnts=maps+SL*16,E=(128+W-1)/W;
    for(uint32_t row=0;row<N;++row) {
        noc.async_read(ids,CoreLocalMem<uint32_t>(base+RB+row*128),128,{.page_id=row},{});
        noc.async_read(scores,CoreLocalMem<uint32_t>(base+RB+N*128+row*64),64,{.page_id=row},{});
    }
    noc.async_read(rank_tensor,CoreLocalMem<uint32_t>(base+scratch),4,{.page_id=0},{});noc.async_read_barrier();
    const uint32_t rank=*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+scratch);
    auto id=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+RB);
    auto score=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+RB+N*128);
    uint32_t count[E];
    for(uint32_t i=0;i<E;++i)count[i]=0;
    uint32_t prow[SL],pexp[SL],pcnt[SL],proute[SL],n=0;
    auto flush=[&]() {
        noc.async_read_barrier();
        for(uint32_t i=0;i<n;++i) {
            noc.async_write(CoreLocalMem<uint32_t>(base+i*5120),output,5120,{}, {.page_id=pexp[i]*1024+pcnt[i]});
            const uint32_t map_lane=(proute[i]*4)&15;
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+maps+i*16)[map_lane/4]=pcnt[i];
            noc.async_write(CoreLocalMem<uint32_t>(base+maps+i*16+map_lane),mapping,4,{}, {.page_id=prow[i],.offset_bytes=proute[i]*4});
        }
        noc.async_write_barrier();
        n=0;
    };
    for(uint32_t row=0;row<N;++row)for(uint32_t route=0;route<10;++route) {
        const uint32_t v=id[row*32+route];
        if(v/128!=rank || score[row*32+route]==0)continue;
        const uint32_t e=v%128;
        if(e%W!=worker)continue;
        noc.async_read(x,CoreLocalMem<uint32_t>(base+n*5120),5120,{.page_id=row},{});
        prow[n]=row;pexp[n]=e;pcnt[n]=count[e/W]++;proute[n]=route;
        if(++n==SL)flush();
    }
    if(n)flush();
    for(uint32_t e=worker;e<128;e+=W) {
        const uint32_t count_lane=(e*4)&15;
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+cnts+(e/W)*16)[count_lane/4]=count[e/W];
        noc.async_write(CoreLocalMem<uint32_t>(base+cnts+(e/W)*16+count_lane),counts,4,{}, {.page_id=0,.offset_bytes=e*4});
    }
    noc.async_write_barrier();
}
"""
)
