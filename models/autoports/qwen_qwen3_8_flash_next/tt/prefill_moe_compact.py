# SPDX-License-Identifier: Apache-2.0
"""Pipelined expert dispatch and output compaction for the serial 1024-row prefill MoE group.

Dispatch: one pass over (row, route) instead of one per owned expert, matched rows moved in batches
behind one barrier, and an inverse map inverse[e][packed] = row * 16 + route written next to the
mapping. Compaction (80 column workers): reads counts and the inverse map instead of every token's
ids/scores/mapping (the per-row metadata reads of all 80 workers hit the same DRAM pages), reads each
touched BFP8 expert-output tile once instead of once per row, places the rows exactly where the
per-row kernel placed them (mantissa rows as word copies) into all 10*T output tiles held in L1, and
writes them at the end. Output bytes are unchanged. Toggles (default on):
QWEN_PREFILL_DISPATCH_PIPE=0 restores the serial dispatch (and with it the serial compaction);
QWEN_PREFILL_COMPACT_PIPE=0 restores the serial compaction.
"""

import os

from .multichip_decoder import _EXPERT_HEADER

SLOT_BYTES = 1088  # one BFP8 tile: 64 exponent bytes + 1024 mantissa bytes


def compact_pipe_enabled():
    return os.environ.get("QWEN_PREFILL_COMPACT_PIPE", "1") != "0"


COMPACT1024_PIPE = (
    _EXPERT_HEADER
    + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0),T=(N+31)/32;
    constexpr auto da=TensorAccessorArgs<2>();
    constexpr auto ca=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto va=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto ra=TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    const auto down=TensorAccessor(da,get_arg_val<uint32_t>(0));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(1));
    const auto inverse=TensorAccessor(va,get_arg_val<uint32_t>(2));
    const auto rank_tensor=TensorAccessor(ra,get_arg_val<uint32_t>(3));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(4));
    const uint32_t col=get_arg_val<uint32_t>(5);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    // [counts 512][offsets 512][inverse entries, each expert 64-byte aligned][ring R tiles][10*T output tiles]
    constexpr uint32_t R=8,TILE=1088,cnt=0,offs=512,inv=1024,ring=inv+(10*N+128*16)*4,outs=ring+R*TILE;
    noc.async_read(counts,CoreLocalMem<uint32_t>(base+cnt),512,{.page_id=0},{});noc.async_read_barrier();
    auto count=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+cnt);
    auto offset=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+offs);
    uint32_t total=0;
    for(uint32_t e=0;e<128;++e) {
        offset[e]=total;
        const uint32_t bytes=(count[e]*4+63)/64*64;
        if(bytes)noc.async_read(inverse,CoreLocalMem<uint32_t>(base+inv+total*4),bytes,{.page_id=e},{});
        total+=bytes/4;
    }
    ASSERT(total<=10*N+128*16);
    auto out_words=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+outs);
    for(uint32_t i=0;i<10*T*TILE/4;++i)out_words[i]=0;
    auto out=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base+outs);
    noc.async_read_barrier();
    // inverse[e][packed] = token row * 16 + route, written by the dispatch kernel for exactly the
    // (row, route) pairs the per-row kernel compacted. Every touched expert-output tile is read once.
    uint32_t pe[R],pb[R],n=0;
    auto drain=[&]() {
        noc.async_read_barrier();
        for(uint32_t k=0;k<n;++k) {
            auto src=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base+ring+k*TILE);
            auto entries=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+inv+offset[pe[k]]*4);
            const uint32_t first=pb[k]*32,last=first+32<count[pe[k]]?first+32:count[pe[k]];
            for(uint32_t packed=first;packed<last;++packed) {
                const uint32_t v=entries[packed],row=v/16,route=v%16,tile=route*T+row/32;
                for(uint32_t side=0;side<2;++side) {
                    const uint32_t sf=(packed%32/16)*2+side,df=(row%32/16)*2+side;
                    out[tile*TILE+df*16+row%16]=src[sf*16+packed%16];
                    // 16 mantissa bytes, 16-byte aligned on both sides: four word copies.
                    auto dw=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+outs+tile*TILE+64+df*256+(row%16)*16);
                    auto sw=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+ring+k*TILE+64+sf*256+(packed%16)*16);
                    dw[0]=sw[0];dw[1]=sw[1];dw[2]=sw[2];dw[3]=sw[3];
                }
            }
        }
        n=0;
    };
    for(uint32_t e=0;e<128;++e)for(uint32_t block=0;block*32<count[e];++block) {
        noc.async_read(down,CoreLocalMem<uint32_t>(base+ring+n*TILE),TILE,{.page_id=(e*32+block)*80+col},{});
        pe[n]=e;pb[n]=block;
        if(++n==R)drain();
    }
    if(n)drain();
    for(uint32_t route=0;route<10;++route)for(uint32_t tile=0;tile<T;++tile)
        noc.async_write(CoreLocalMem<uint32_t>(base+outs+(route*T+tile)*TILE),output,TILE,{}, {.page_id=(route*T+tile)*80+col});
    noc.async_write_barrier();
}
"""
)


def compact_cb_bytes(n):
    return 1024 + (10 * n + 128 * 16) * 4 + 8 * SLOT_BYTES + 10 * ((n + 31) // 32) * SLOT_BYTES


def dispatch_pipe_enabled():
    return os.environ.get("QWEN_PREFILL_DISPATCH_PIPE", "1") != "0"


DISPATCH_SLOTS = 8


def dispatch_cb_bytes(n):
    return DISPATCH_SLOTS * 5120 + n * 192 + 64 + DISPATCH_SLOTS * 16 + 128 * 16 + DISPATCH_SLOTS * 16


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
    constexpr auto va=TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0));
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto rank_tensor=TensorAccessor(ra,get_arg_val<uint32_t>(3));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(4));
    const auto mapping=TensorAccessor(ma,get_arg_val<uint32_t>(5));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(6));
    const auto inverse=TensorAccessor(va,get_arg_val<uint32_t>(7));
    const uint32_t worker=get_arg_val<uint32_t>(8);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    constexpr uint32_t SL=8,RB=SL*5120,scratch=RB+N*192,maps=scratch+64,cnts=maps+SL*16,invs=cnts+128*16,E=(128+W-1)/W;
    // ids (uint32) and scores (bf16) arrive in TILE layout: one 32-row tile per page.
    for(uint32_t t=0;t<(N+31)/32;++t) {
        noc.async_read(ids,CoreLocalMem<uint32_t>(base+RB+t*4096),4096,{.page_id=t},{});
        noc.async_read(scores,CoreLocalMem<uint32_t>(base+RB+N*128+t*2048),2048,{.page_id=t},{});
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
            // Inverse map for the compaction kernel: inverse[e][count] = row * 16 + route.
            const uint32_t inv_lane=(pcnt[i]*4)&15;
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+invs+i*16)[inv_lane/4]=prow[i]*16+proute[i];
            noc.async_write(CoreLocalMem<uint32_t>(base+invs+i*16+inv_lane),inverse,4,{}, {.page_id=pexp[i],.offset_bytes=pcnt[i]*4});
        }
        noc.async_write_barrier();
        n=0;
    };
    for(uint32_t row=0;row<N;++row)for(uint32_t route=0;route<10;++route) {
        const uint32_t at=(row/32)*1024+(row%32/16)*512+(route/16)*256+(row%16)*16+route%16;
        const uint32_t v=id[at];
        if(v/128!=rank || score[at]==0)continue;
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
