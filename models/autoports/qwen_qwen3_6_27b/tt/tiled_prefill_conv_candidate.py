# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in 32-time-row native-math convolution family; no default wiring.

CPU source generation only at import. Native compute is adapted from the local
source tree; the new reader accepts compact flat TILE packed inputs directly.
No sequence-parallel exchange: target mesh has SP axis size 1, TP axis size 4.
"""

from pathlib import Path
from functools import lru_cache

NATIVE = Path(__file__).resolve().parents[4] / "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device"

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t BC=get_compile_time_arg_val(0),MT=get_compile_time_arg_val(1),G=get_compile_time_arg_val(2);
    constexpr uint32_t S=get_compile_time_arg_val(3),STRIDE=get_compile_time_arg_val(4),WT=get_compile_time_arg_val(5);
    constexpr uint32_t CO=get_compile_time_arg_val(6),BLOCKS=80/BC;
    constexpr auto xa=TensorAccessorArgs<7>();
    constexpr auto ha=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto t0a=TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto t1a=TensorAccessorArgs<t0a.next_compile_time_args_offset()>();
    constexpr auto t2a=TensorAccessorArgs<t1a.next_compile_time_args_offset()>();
    constexpr auto t3a=TensorAccessorArgs<t2a.next_compile_time_args_offset()>();
    constexpr auto ca=TensorAccessorArgs<t3a.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const auto h=TensorAccessor(ha,get_arg_val<uint32_t>(1),5120);
    const auto t0=TensorAccessor(t0a,get_arg_val<uint32_t>(2),2048);
    const auto t1=TensorAccessor(t1a,get_arg_val<uint32_t>(3),2048);
    const auto t2=TensorAccessor(t2a,get_arg_val<uint32_t>(4),2048);
    const auto t3=TensorAccessor(t3a,get_arg_val<uint32_t>(5),2048);
    const auto carry=TensorAccessor(ca,get_arg_val<uint32_t>(6),5120);
    const uint32_t first=get_arg_val<uint32_t>(7),count=get_arg_val<uint32_t>(8);
    CircularBuffer act(0),taps(1),eye(2),tiles(20),rows(21),strips(22);
    const uint32_t tile_base=(tiles.get_write_ptr()+63u)&~63u;
    const uint32_t row_base=(rows.get_write_ptr()+63u)&~63u;
    const uint32_t strip_base=(strips.get_write_ptr()+63u)&~63u;
    Noc noc; UnicastEndpoint self;
    const uint32_t nx=my_x[noc.get_noc_id()],ny=my_y[noc.get_noc_id()];
    const auto local=[&](uint32_t a) { return noc_traits_t<UnicastEndpoint>::src_args_type{.noc_x=nx,.noc_y=ny,.addr=a}; };
    const auto strip=[&](uint32_t ct,uint32_t side) { return strip_base+(ct*2+side)*1152+32; };
    eye.reserve_back(1); noc.async_write_zeros(eye,2048); noc.write_zeros_l1_barrier();
    auto* identity=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(eye.get_write_ptr());
    for(uint32_t face : {0u,3u}) for(uint32_t i=0;i<16;i++) identity[(face*16+i)*16+i]=0x3f80;
    eye.push_back(1);
    uint32_t loaded=~0u;
    for(uint32_t work=first;work<first+count;work++) {
        const uint32_t block=work/MT,mt=work%MT,user=block/BLOCKS,ct0=(block%BLOCKS)*BC;
        if(block!=loaded) {
            taps.reserve_back(4*BC);
            for(uint32_t ct=0;ct<BC;ct++) {
                noc.async_read(t0,taps,1024,{.page_id=ct0+ct},{.offset_bytes=ct*2048});
                noc.async_read(t1,taps,1024,{.page_id=ct0+ct},{.offset_bytes=(BC+ct)*2048});
                noc.async_read(t2,taps,1024,{.page_id=ct0+ct},{.offset_bytes=(2*BC+ct)*2048});
                noc.async_read(t3,taps,1024,{.page_id=ct0+ct},{.offset_bytes=(3*BC+ct)*2048});
            }
            noc.async_read_barrier(); taps.push_back(4*BC); loaded=block;
        }
        // Each window contains 3 halo +32 real/dead time rows. Compact users
        // can start at any global row phase; read at most 3 complete tiles.
        // DRAM transfers are always 64-byte aligned and never cross a page.
        const int32_t window=int32_t(mt*32)-3;
        const uint32_t lo=window<0?0:uint32_t(window);
        const uint32_t hi=(mt*32+32<S)?mt*32+32:S;
        const uint32_t first_tile=(user*STRIDE+lo)/32;
        const uint32_t last_tile=(user*STRIDE+hi-1)/32;
        noc.async_write_zeros(CoreLocalMem<uint32_t>(strip_base),BC*2*1152);
        noc.write_zeros_l1_barrier();
        for(uint32_t ct=0;ct<BC;ct++) for(uint32_t tile=first_tile;tile<=last_tile;tile++) {
            noc.async_read(x,CoreLocalMem<uint32_t>(tile_base+(ct*3+tile-first_tile)*2048),2048,
                {.page_id=tile*WT+CO+ct0+ct},{});
        }
        if(mt==0) for(uint32_t row=0;row<3;row++)
            noc.async_read(h,CoreLocalMem<uint32_t>(row_base+row*BC*64),BC*64,
                {.page_id=user*3+row,.offset_bytes=ct0*64},{});
        noc.async_read_barrier();
        // Coalesce local copies up to a face boundary. Unlike the per-token
        // SFPU prototype, no input/tap is reread separately for each output row.
        for(uint32_t ct=0;ct<BC;ct++) {
            uint32_t wr=0;
            while(wr<35) {
                const int32_t logical=window+int32_t(wr);
                if(logical<0) {
                    const uint32_t hr=uint32_t(logical+3);
                    for(uint32_t side=0;side<2;side++)
                        noc.async_read(self,CoreLocalMem<uint32_t>(strip(ct,side)+wr*32),32,
                            local(row_base+hr*BC*64+ct*64+side*32),{});
                    wr++; continue;
                }
                if(uint32_t(logical)>=S) break; // dead strip stays zero
                const uint32_t global=user*STRIDE+uint32_t(logical),phase=global%32;
                uint32_t n=16-(phase%16);
                if(n>35-wr) n=35-wr;
                if(n>S-uint32_t(logical)) n=S-uint32_t(logical);
                for(uint32_t side=0;side<2;side++) {
                    const uint32_t src=tile_base+(ct*3+global/32-first_tile)*2048+
                        (phase/16*2+side)*512+(phase%16)*32;
                    noc.async_read(self,CoreLocalMem<uint32_t>(strip(ct,side)+wr*32),n*32,local(src),{});
                }
                wr+=n;
            }
        }
        noc.async_read_barrier();
        // Raw final three tokens are copied out separately. Input history is
        // immutable until the parent operation completes on every core.
        if(mt+1==MT) {
            auto* dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_base);
            for(uint32_t row=0;row<3;row++) for(uint32_t ct=0;ct<BC;ct++) for(uint32_t side=0;side<2;side++) {
                const uint32_t wr=S-mt*32+row;
                const auto* src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(strip(ct,side)+wr*32);
                for(uint32_t word=0;word<8;word++) dst[row*BC*16+ct*16+side*8+word]=src[word];
            }
            for(uint32_t row=0;row<3;row++)
                noc.async_write(CoreLocalMem<uint32_t>(row_base+row*BC*64),carry,BC*64,{},
                    {.page_id=user*3+row,.offset_bytes=ct0*64});
            noc.async_write_barrier();
        }
        for(uint32_t ct0local=0;ct0local<BC;ct0local+=G) {
            act.reserve_back(G*4);
            for(uint32_t i=0;i<G;i++) for(uint32_t tap=0;tap<4;tap++) for(uint32_t face=0;face<4;face++)
                noc.async_read(self,act,512,
                    local(strip(ct0local+i,face%2)+(tap+(face/2)*16)*32),
                    {.offset_bytes=(i*4+tap)*2048+face*512});
            noc.async_read_barrier(); act.push_back(G*4);
        }
    }
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t BC=get_compile_time_arg_val(0),MT=get_compile_time_arg_val(1),G=get_compile_time_arg_val(2),BLOCKS=80/BC;
    constexpr auto qa=TensorAccessorArgs<3>();
    constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va=TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(0),2048);
    const auto k=TensorAccessor(ka,get_arg_val<uint32_t>(1),2048);
    const auto v=TensorAccessor(va,get_arg_val<uint32_t>(2),2048);
    const uint32_t first=get_arg_val<uint32_t>(3),count=get_arg_val<uint32_t>(4);
    Noc noc; CircularBuffer out(4);
    for(uint32_t work=first;work<first+count;work++) {
        const uint32_t block=work/MT,user=block/BLOCKS,mt=work%MT,ct0=(block%BLOCKS)*BC;
        for(uint32_t c=0;c<BC;c++) {
            const uint32_t slot=c%G,ct=ct0+c;
            if(slot==0) out.wait_front(G);
            if(ct<16) noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),q,2048,
                {.offset_bytes=slot*2048},{.page_id=(user*MT+mt)*16+ct});
            else if(ct<32) noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),k,2048,
                {.offset_bytes=slot*2048},{.page_id=(user*MT+mt)*16+ct-16});
            else noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),v,2048,
                {.offset_bytes=slot*2048},{.page_id=(user*MT+mt)*48+ct-32});
            if(slot+1==G) { noc.async_write_barrier(); out.pop_front(G); }
        }
    }
}
"""


@lru_cache(maxsize=1)
def native_compute():
    """Retain native diagonal matmuls and pack-SiLU; adapt binding API only."""
    source = (NATIVE / "kernels/compute/qkv_causal_conv1d_silu.cpp").read_text()
    old = "template <uint32_t block_ct, uint32_t Mt, uint32_t group, uint32_t dest_limit, uint32_t defer_first_rows>\nTT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {"
    assert source.count(old) == 1
    source = source.replace(
        old,
        """void kernel_main() {
    constexpr uint32_t block_ct=get_compile_time_arg_val(0),Mt=get_compile_time_arg_val(1),group=get_compile_time_arg_val(2);
    constexpr uint32_t dest_limit=4,defer_first_rows=0;
    const uint32_t wi_start=get_arg_val<uint32_t>(0),wi_count=get_arg_val<uint32_t>(1);""",
    )
    source = source.replace('#include "experimental/kernel_args.h"', '#include "api/compile_time_args.h"')
    source = source.replace('#include "api/dataflow/dataflow_buffer.h"', '#include "api/dataflow/circular_buffer.h"')
    source = source.replace("DataflowBuffer", "CircularBuffer")
    for name, index in (("act_tile", 0), ("taps", 1), ("eye", 2), ("weights", 3), ("output", 4)):
        source = source.replace("dfb::" + name, str(index))
    assert "dfb::" not in source and "TT_KERNEL" not in source
    return source


def plan(batch, seq, block_ct=8, cores=110):
    assert 1 <= batch <= 32 and seq >= 1 and block_ct in (1, 2, 4, 8) and cores >= 1
    mt = (seq + 31) // 32
    group = min(4, block_ct)
    cbs = {
        0: 8 * group,
        1: 4 * block_ct,
        2: 1,
        3: 4 * block_ct,
        4: 2 * group,
        20: (3 * block_ct * 2048 + 64 + 2047) // 2048,
        21: (3 * block_ct * 64 + 64 + 2047) // 2048,
        22: (2 * block_ct * 1152 + 64 + 2047) // 2048,
    }
    tasks = batch * (80 // block_ct) * mt
    return {
        "batch": batch,
        "sequence": seq,
        "mt": mt,
        "block_ct": block_ct,
        "group": group,
        "tasks": tasks,
        "cores": min(cores, tasks),
        "cb_tiles": cbs,
        "l1_cb_bytes": sum(cbs.values()) * 2048,
    }


def tiled_prefill_conv(value, history, taps, *, batch, seq, block_ct=8, cores=None, column_offset=0):
    """Direct TILE input [B,S,W] or compact [1,B*S,W], W>=QKV width.

    Supports current TP4 only, no SP exchange. Preserves raw history and native
    arithmetic. Q/K/V are written directly into padded per-user TILE outputs.
    """
    import ttnn

    assert len(value.shape) == 3 and value.shape[-1] >= column_offset + 2560
    assert column_offset >= 0 and column_offset % 32 == 0
    assert value.layout == ttnn.TILE_LAYOUT and value.dtype == ttnn.bfloat16
    assert tuple(history.shape) == (batch, 3, 2560) and history.layout == ttnn.ROW_MAJOR_LAYOUT
    assert history.dtype == ttnn.bfloat16 and len(taps) == 4
    assert all(
        tuple(t.shape) == (1, 1, 2560) and t.layout == ttnn.TILE_LAYOUT and t.dtype == ttnn.bfloat16 for t in taps
    )
    assert all(t.memory_config() == ttnn.DRAM_MEMORY_CONFIG for t in (value, history, *taps))
    if tuple(value.shape)[:2] == (1, batch * seq):
        stride = seq
    else:
        assert tuple(value.shape)[:2] == (batch, seq)
        stride = (seq + 31) // 32 * 32
    grid = value.device().compute_with_storage_grid_size()
    cfg = plan(batch, seq, block_ct, cores or grid.x * grid.y)
    assert cfg["cores"] <= grid.x * grid.y

    def allocate(shape, layout):
        return ttnn.empty(
            shape, dtype=ttnn.bfloat16, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=value.device()
        )

    outputs = [allocate((batch, cfg["mt"] * 32, w), ttnn.TILE_LAYOUT) for w in (512, 512, 1536)]
    carry = allocate((batch, 3, 2560), ttnn.ROW_MAJOR_LAYOUT)
    inputs = [value, history, *taps, carry]
    rct = [block_ct, cfg["mt"], cfg["group"], seq, stride, value.padded_shape[-1] // 32, column_offset // 32]
    wct = [block_ct, cfg["mt"], cfg["group"]]
    for t in inputs:
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    for t in outputs:
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(cfg["cores"])]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    input_addresses = [t.buffer_address() for t in inputs]
    output_addresses = [t.buffer_address() for t in outputs]
    first = 0
    for i, c in enumerate(coords):
        count = cfg["tasks"] // cfg["cores"] + (i < cfg["tasks"] % cfg["cores"])
        rr[c.x][c.y] = input_addresses + [first, count]
        wr[c.x][c.y] = output_addresses + [first, count]
        cr[c.x][c.y] = [first, count]
        first += count

    def kernel(source, config, args, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=args,
            config=config,
        )

    kernels = [
        kernel(READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(WRITER, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(
            native_compute(),
            ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                math_approx_mode=False,
            ),
            cr,
            [block_ct, cfg["mt"], cfg["group"]],
        ),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=tiles * 2048,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for index, tiles in cfg["cb_tiles"].items()
    ]
    ttnn.generic_op(inputs + outputs, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    ttnn.copy(carry, history)
    return tuple(outputs)


def cpu_check():
    import json
    import sys

    compute = native_compute()
    assert (
        compute.count("matmul_tiles(") == 1 and "apply_activation_from_pack<KernelActivation::SILU>(group)" in compute
    )
    cases = 0
    for batch in (1, 3, 17, 32):
        for seq in (1, 2, 3, 17, 31, 32, 33, 65, 257, 512, 513):
            for compact in (False, True):
                cfg = plan(batch, seq)
                stride = seq if compact else cfg["mt"] * 32
                for user in range(batch):
                    for mt in range(cfg["mt"]):
                        window = mt * 32 - 3
                        lo = max(0, window)
                        hi = min(seq, mt * 32 + 32)
                        first = (user * stride + lo) // 32
                        last = (user * stride + hi - 1) // 32
                        assert 1 <= last - first + 1 <= 3
                        for wr in range(35):
                            logical = window + wr
                            if logical < 0:
                                assert mt == 0 and 0 <= logical + 3 < 3
                            elif logical < seq:
                                global_row = user * stride + logical
                                assert first <= global_row // 32 <= last
                                phase = global_row % 32
                                for side in range(2):
                                    offset = (phase // 16 * 2 + side) * 512 + (phase % 16) * 32
                                    assert 0 <= offset <= 2016
                            # logical>=seq is zero, never a neighbor user row.
                        if mt + 1 == cfg["mt"]:
                            for row in range(3):
                                wr = seq - mt * 32 + row
                                assert 0 <= wr < 35 and window + wr == seq - 3 + row
                cases += 1
    for bc in (1, 2, 4, 8):
        p = plan(32, 512, bc)
        assert p["l1_cb_bytes"] < 1024 * 1024
        assert 80 % bc == 0 and bc % p["group"] == 0
    assert "torch" not in sys.modules and "ttnn" not in sys.modules
    print(
        json.dumps(
            {
                "status": "pass",
                "scope": "CPU address/source only",
                "cases": cases,
                "plans": [plan(32, 512, b) for b in (1, 2, 4, 8)],
            }
        )
    )


if __name__ == "__main__":
    import sys

    assert sys.argv[1:] == ["--cpu-check"]
    cpu_check()
