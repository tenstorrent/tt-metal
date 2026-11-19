# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated two-launch TP4 native GDN candidate. No production registration.

Imports of TTNN are deferred. Native recurrence helper arithmetic is loaded
verbatim from the repository kernel; only T1 scheduling/persistence changes.
BF16 state expands locally in L1 and rounds only after FP32 live readout.
"""

from pathlib import Path

NATIVE_COMPUTE = (
    Path(__file__).resolve().parents[4]
    / "ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/compute/fused_recurrent_gated_delta_rule.cpp"
)


def head_partition(batch, cores=110):
    assert 1 <= batch <= 32 and cores >= 1
    heads = batch * 12
    count = min(heads, cores)
    q, r = divmod(heads, count)
    return [(i * q + min(i, r), q + (i < r)) for i in range(count)]


def memory_estimate(batch, cores=110):
    return {
        "partition": head_partition(batch, cores),
        "prepared_payload_bytes_per_rank": batch * 2592 * 4,
        "compact_output_bytes_per_rank": ((batch * 12 + 31) // 32) * 4 * 4096,
        "state_fp32_bytes_per_rank": batch * 12 * 128 * 128 * 4,
        "state_bf16_bytes_per_rank": batch * 12 * 128 * 128 * 2,
        "recurrence_cb_fp32_bytes_per_core": (130 + 2) * 4096,
        "recurrence_cb_bf16_bytes_per_core": (130 + 2) * 4096 - 32 * 2048 + 16 * 2048,
        "preparation_cb_bytes_per_core": (4 + 1 + 4 + 1 + 1 + 2 + 3) * 4096,
        "limitations": [
            "Payload/CB accounting excludes allocator bank padding, kernels and trace metadata.",
            "Preparation and recurrence execute sequentially; measure their physical allocation peak.",
        ],
    }


def cpu_geometry_check():
    for batch in range(1, 33):
        heads = batch * 12
        assert [h for start, count in head_partition(batch) for h in range(start, start + count)] == list(range(heads))
        segments = set()
        for h in range((heads + 31) // 32 * 32):
            r = h % 32
            for c in range(4):
                for f in range(2):
                    offset = (r // 16) * 2048 + (r % 16) * 64 + f * 1024
                    key = ((h // 32) * 4 + c, offset)
                    assert key not in segments and offset % 64 == 0 and offset + 64 <= 4096
                    segments.add(key)
        assert len(segments) == ((heads + 31) // 32) * 4 * 64
        assert 2592 * 4 % 64 == 0
        for h in range(heads):
            hv = h % 12
            offsets = ((hv // 3) * 512, 2048 + (hv // 3) * 512, 4096 + hv * 512)
            assert all(offset % 64 == 0 and offset + 512 <= 10240 for offset in offsets)
            assert (10240 + 128) == 10368
            assert hv + 12 < 24
        assert 5184 + 64 <= 2 * 4096 and 10368 <= 3 * 4096
        assert memory_estimate(batch)["recurrence_cb_fp32_bytes_per_core"] == 540672
        assert memory_estimate(batch)["recurrence_cb_bf16_bytes_per_core"] == 507904
    return {
        "batches": list(range(1, 33)),
        "unique_head_owners": True,
        "compact_output_segments_disjoint_complete_aligned": True,
        "native_final_state_drained_between_heads": True,
        "bf16_expansion_and_rounding_are_local_only": True,
    }


PREP_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t GF=get_compile_time_arg_val(0),BF=get_compile_time_arg_val(1);
    constexpr auto xa=TensorAccessorArgs<2>();
    constexpr auto ga=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto ba=TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),5120);
    const auto g=TensorAccessor(ga,get_arg_val<uint32_t>(1),GF?4096:2048);
    const auto b=TensorAccessor(ba,get_arg_val<uint32_t>(2),BF?4096:2048);
    const uint32_t user=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer raw(15),gate(1),in(0);
    raw.reserve_back(2);
    const uint32_t address=raw.get_write_ptr();
    noc.async_read(x,raw,5120,{.page_id=user},{.offset_bytes=0});
    noc.async_read(g,raw,64,{.page_id=user},{.offset_bytes=5120});
    noc.async_read(b,raw,64,{.page_id=user},{.offset_bytes=5184});
    noc.async_read_barrier();
    gate.reserve_back(1); noc.async_write_zeros(gate,4096); noc.write_zeros_l1_barrier();
    volatile tt_l1_ptr uint32_t* gp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gate.get_write_ptr());
    volatile tt_l1_ptr uint32_t* g32=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address+5120);
    volatile tt_l1_ptr uint16_t* g16=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(address+5120);
    for(uint32_t i=0;i<12;i++) gp[i]=GF?g32[i]:(uint32_t(g16[i])<<16);
    gate.push_back(1); raw.push_back(2);
    volatile tt_l1_ptr uint16_t* xp=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(address);
    for(uint32_t vector=0;vector<8;vector++) {
        in.reserve_back(4); noc.async_write_zeros(in,4*4096); noc.write_zeros_l1_barrier();
        volatile tt_l1_ptr uint32_t* dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in.get_write_ptr());
        for(uint32_t i=0;i<128;i++) dst[(i/32)*1024+((i%32)/16)*256+i%16]=uint32_t(xp[vector*128+i])<<16;
        in.push_back(4);
    }
}
"""

PREP_COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    compute_kernel_hw_startup(0,1,2);
    CircularBuffer input(0),gate(1),output(2),decay(3),norm(4);
    gate.wait_front(1); decay.reserve_back(1); tile_regs_acquire();
    copy_init(1); copy_tile(1,0,0); exp_tile_init<false>(); exp_tile<false>(0);
    tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,3,0); tile_regs_release();
    decay.push_back(1); gate.pop_front(1);
    for(uint32_t vector=0;vector<8;vector++) {
        input.wait_front(4); norm.reserve_back(1); tile_regs_acquire(); copy_init(0);
        for(uint32_t i=0;i<4;i++) {
            const uint32_t w=i?2:0;
            copy_tile(0,i,w); copy_tile(0,i,1);
            mul_binary_tile_init(); mul_binary_tile(w,1,w);
            if(i) { add_binary_tile_init(); add_binary_tile(0,2,0); }
        }
        sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_ROW>(0,1,1);
        binop_with_scalar_tile_init(); add_unary_tile(0,0x358637bdu);
        rsqrt_tile_init(); rsqrt_tile(0);
        tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,4,0); tile_regs_release();
        norm.push_back(1); norm.wait_front(1); output.reserve_back(4);
        for(uint32_t i=0;i<4;i++) {
            tile_regs_acquire(); copy_init(0); copy_tile(0,i,0); copy_init(4); copy_tile(4,0,1);
            sfpu_mul_bcast_col_init(); sfpu_mul_bcast_col(0,1);
            if(vector<4) { binop_with_scalar_tile_init(); mul_unary_tile(0,0x3db504f3u); }
            tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,2,i); tile_regs_release();
        }
        output.push_back(4); input.pop_front(4); norm.pop_front(1);
    }
}
"""

PREP_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t BF=get_compile_time_arg_val(0);
    constexpr auto pa=TensorAccessorArgs<1>();
    const auto prepared=TensorAccessor(pa,get_arg_val<uint32_t>(0),10368);
    const uint32_t user=get_arg_val<uint32_t>(1);
    Noc noc; CircularBuffer raw(15),normalized(2),decay(3),record(16);
    raw.wait_front(2); decay.wait_front(1); record.reserve_back(3);
    volatile tt_l1_ptr uint32_t* dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(record.get_write_ptr());
    volatile tt_l1_ptr uint16_t* xp=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(raw.get_read_ptr());
    volatile tt_l1_ptr uint32_t* dp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(decay.get_read_ptr());
    volatile tt_l1_ptr uint16_t* b16=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(raw.get_read_ptr()+5184);
    volatile tt_l1_ptr uint32_t* b32=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(raw.get_read_ptr()+5184);
    for(uint32_t i=0;i<1536;i++) dst[1024+i]=uint32_t(xp[1024+i])<<16;
    for(uint32_t i=0;i<12;i++) { dst[2560+i]=dp[i]; dst[2572+i]=BF?b32[i]:(uint32_t(b16[i])<<16); }
    for(uint32_t i=2584;i<2592;i++) dst[i]=0;
    decay.pop_front(1);
    for(uint32_t vector=0;vector<8;vector++) {
        normalized.wait_front(4);
        volatile tt_l1_ptr uint32_t* src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(normalized.get_read_ptr());
        for(uint32_t i=0;i<128;i++) dst[vector*128+i]=src[(i/32)*1024+((i%32)/16)*256+i%16];
        normalized.pop_front(4);
    }
    noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(record),prepared,10368,{.offset_bytes=0},{.page_id=user});
    noc.async_write_barrier(); raw.pop_front(2);
}
"""

REC_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t SB=get_compile_time_arg_val(0);
    constexpr auto pa=TensorAccessorArgs<1>();
    constexpr auto sa=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto p=TensorAccessor(pa,get_arg_val<uint32_t>(0),10368);
    const auto s=TensorAccessor(sa,get_arg_val<uint32_t>(1),SB);
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer scratch(15); scratch.reserve_back(1);
    const uint32_t scratch_addr=scratch.get_write_ptr();
    for(uint32_t h=start;h<start+count;h++) {
        const uint32_t user=h/12,hv=h%12,hk=hv/3;
        CircularBuffer state(SB==2048?14:5); state.reserve_back(16);
        for(uint32_t i=0;i<16;i++) noc.async_read(s,state,SB,{.page_id=h*16+i},{.offset_bytes=i*SB});
        noc.async_read_barrier(); state.push_back(16);
        noc.async_read(p,scratch,512,{.page_id=user,.offset_bytes=hk*512},{.offset_bytes=0});
        noc.async_read(p,scratch,512,{.page_id=user,.offset_bytes=2048+hk*512},{.offset_bytes=512});
        noc.async_read(p,scratch,512,{.page_id=user,.offset_bytes=4096+hv*512},{.offset_bytes=1024});
        noc.async_read(p,scratch,128,{.page_id=user,.offset_bytes=10240},{.offset_bytes=1536});
        noc.async_read_barrier();
        for(uint32_t id=0;id<5;id++) {
            const uint32_t n=id<3?4:1;
            CircularBuffer cb(id); cb.reserve_back(n); noc.async_write_zeros(cb,n*4096); noc.write_zeros_l1_barrier();
            volatile tt_l1_ptr uint32_t* dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
            volatile tt_l1_ptr uint32_t* src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr+(id<3?id*512:1536));
            if(id<3) for(uint32_t i=0;i<128;i++) dst[(i/32)*1024+((i%32)/16)*256+i%16]=src[i];
            else dst[0]=src[hv+(id==4?12:0)];
            cb.push_back(n);
        }
    }
}
"""

REC_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t SB=get_compile_time_arg_val(0),H=get_compile_time_arg_val(1);
    constexpr auto sa=TensorAccessorArgs<2>();
    constexpr auto oa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto state=TensorAccessor(sa,get_arg_val<uint32_t>(0),SB);
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(1),4096);
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer out(6),persist(7),zero(16); zero.reserve_back(1);
    volatile tt_l1_ptr uint32_t* zp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero.get_write_ptr());
    for(uint32_t i=0;i<16;i++) zp[i]=0;
    for(uint32_t h=start;h<start+count;h++) {
        out.wait_front(4);
        for(uint32_t c=0;c<4;c++) for(uint32_t f=0;f<2;f++) {
            const uint32_t r=h%32,offset=(r/16)*2048+(r%16)*64+f*1024;
            noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,64,
                {.offset_bytes=c*4096+f*1024},{.page_id=(h/32)*4+c,.offset_bytes=offset});
        }
        if(h+1==H && H%32) for(uint32_t r=H%32;r<32;r++) for(uint32_t c=0;c<4;c++) for(uint32_t f=0;f<2;f++)
            noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(zero),output,64,{.offset_bytes=0},
                {.page_id=(h/32)*4+c,.offset_bytes=(r/16)*2048+(r%16)*64+f*1024});
        noc.async_write_barrier(); out.pop_front(4);
        persist.wait_front(16);
        for(uint32_t i=0;i<16;i++) noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(persist),state,SB,
            {.offset_bytes=i*SB},{.page_id=h*16+i});
        noc.async_write_barrier(); persist.pop_front(16);
    }
}
"""

REC_MAIN = r"""
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/typecast.h"
void kernel_main() {
    constexpr bool BF16=get_compile_time_arg_val(0);
    const uint32_t count=get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_q,cb_v,cb_out);
    CircularBuffer q(cb_q),k(cb_k),v(cb_v),decay(cb_decay),beta(cb_beta);
    CircularBuffer state(cb_S),next(cb_s2),sd(cb_sd),vread(cb_vread),u(cb_u),kcol(cb_kcol);
    for(uint32_t head=0;head<count;head++) {
        if constexpr(BF16) {
            CircularBuffer raw(14); raw.wait_front(16); state.reserve_back(16);
            reconfig_data_format_srca(14); copy_init(14);
            for(uint32_t i=0;i<16;i++) {
                tile_regs_acquire(); copy_tile(14,i,0);
                tile_regs_commit(); tile_regs_wait(); pack_tile<true>(0,cb_S,i); tile_regs_release();
            }
            state.push_back(16); raw.pop_front(16);
            reconfig_data_format_srca(cb_q);
        }
        decay.wait_front(1); state.wait_front(16);
        bcast_scalar_mul(cb_S,cb_decay,cb_sd,16); decay.pop_front(1); state.pop_front(16); sd.wait_front(16);
        k.wait_front(4); mm(cb_k,cb_sd,cb_vread,1,4,4,false); vread.wait_front(4);
        v.wait_front(4); beta.wait_front(1);
        delta_rule_residual(cb_v,cb_vread,cb_beta,cb_u,4);
        v.pop_front(4); vread.pop_front(4); beta.pop_front(1); u.wait_front(4);
        transpose_block(cb_k,cb_kcol,4); k.pop_front(4); kcol.wait_front(4);
        rank1_update<4,4>(cb_sd,cb_kcol,cb_u,cb_s2,!BF16);
        kcol.pop_front(4); u.pop_front(4); sd.pop_front(16); next.wait_front(16);
        q.wait_front(4); mm(cb_q,cb_s2,cb_out,1,4,4,false); q.pop_front(4);
        if constexpr(BF16) {
            CircularBuffer persist(cb_state); persist.reserve_back(16);
            copy_init(cb_s2);
            for(uint32_t i=0;i<16;i++) {
                tile_regs_acquire(); copy_tile(cb_s2,i,0);
                typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
                typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
                tile_regs_commit(); tile_regs_wait();
                pack_reconfig_data_format(cb_out,cb_state); pack_tile<true>(0,cb_state,i);
                pack_reconfig_data_format(cb_state,cb_out); tile_regs_release();
            }
            persist.push_back(16);
        }
        // Independent heads cannot inherit the native T>1 loop's retained state.
        next.pop_front(16);
    }
}
"""


def recurrence_source():
    source = NATIVE_COMPUTE.read_text()
    assert source.count("void kernel_main()") == 1
    # Exact native mm/rank1/residual/broadcast/transpose functions, including
    # original fidelity and accumulation order. No replacement arithmetic.
    return source.split("void kernel_main()", 1)[0] + REC_MAIN


class PreparedNativeRecurrentCandidate:
    def __init__(self, initial_state):
        import ttnn

        self.mesh = initial_state.device()
        self.batch = initial_state.shape[0]
        assert tuple(self.mesh.shape) == (1, 4)
        assert tuple(initial_state.shape) == (self.batch, 12, 128, 128) and 1 <= self.batch <= 32
        assert tuple(initial_state.padded_shape) == tuple(initial_state.shape)
        assert initial_state.dtype in (ttnn.float32, ttnn.bfloat16)
        assert initial_state.layout == ttnn.TILE_LAYOUT and initial_state.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        self.state = ttnn.empty_like(initial_state)
        ttnn.copy(initial_state, self.state)
        self.output = ttnn.empty(
            (1, self.batch * 12, 128),
            device=self.mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.prepared = ttnn.empty(
            (self.batch, 1, 2592),
            device=self.mesh,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.bf16 = initial_state.dtype == ttnn.bfloat16
        self.addresses = tuple(t.buffer_address() for t in (self.state, self.prepared, self.output))
        self.source = recurrence_source()

    def step(self, packed, g, beta):
        import ttnn

        assert tuple(packed.shape) == tuple(packed.padded_shape) == (self.batch, 1, 2560)
        assert packed.dtype == ttnn.bfloat16 and packed.layout == ttnn.ROW_MAJOR_LAYOUT
        for gate in (g, beta):
            assert tuple(gate.shape) == (self.batch, 1, 12)
            assert tuple(gate.padded_shape) == (self.batch, 32, 32)
            assert gate.dtype in (ttnn.float32, ttnn.bfloat16) and gate.layout == ttnn.TILE_LAYOUT
        for tensor in (packed, g, beta, self.state, self.prepared, self.output):
            assert tensor.device() == self.mesh and tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tuple(t.buffer_address() for t in (self.state, self.prepared, self.output)) == self.addresses
        grid = self.mesh.compute_with_storage_grid_size()
        assert grid.x * grid.y >= self.batch

        def launch(coords, definitions, cb_specs, tensors):
            ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
            kernels = []
            for source, config, ct, per_core in definitions:
                runtime = ttnn.RuntimeArgs()
                for c, values in zip(coords, per_core):
                    runtime[c.x][c.y] = values
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=source,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=ranges,
                        compile_time_args=ct,
                        runtime_args=runtime,
                        config=config,
                    )
                )
            cbs = [
                ttnn.CBDescriptor(
                    total_size=n * size,
                    core_ranges=ranges,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=size)],
                )
                for i, n, size, dtype in cb_specs
            ]
            ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

        def accessors(tensors):
            result = []
            for tensor in tensors:
                result.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
            return result

        def compute_config(direct=()):
            cfg = ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
                math_approx_mode=False,
            )
            modes = [ttnn.UnpackToDestMode.Default] * 64
            for index in direct:
                modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
            cfg.unpack_to_dest_mode = modes
            return cfg

        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(self.batch)]
        gf, bf = int(g.dtype == ttnn.float32), int(beta.dtype == ttnn.float32)
        launch(
            coords,
            [
                (
                    PREP_READER,
                    ttnn.ReaderConfigDescriptor(),
                    [gf, bf] + accessors([packed, g, beta]),
                    [
                        [packed.buffer_address(), g.buffer_address(), beta.buffer_address(), u]
                        for u in range(self.batch)
                    ],
                ),
                (
                    PREP_WRITER,
                    ttnn.WriterConfigDescriptor(),
                    [bf] + accessors([self.prepared]),
                    [[self.prepared.buffer_address(), u] for u in range(self.batch)],
                ),
                (PREP_COMPUTE, compute_config((0, 1, 4)), [], [[] for _ in coords]),
            ],
            [(i, n, 4096, ttnn.float32) for i, n in {0: 4, 1: 1, 2: 4, 3: 1, 4: 1, 15: 2, 16: 3}.items()],
            [packed, g, beta, self.prepared],
        )

        partition = head_partition(self.batch, grid.x * grid.y)
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(partition))]
        sb = 2048 if self.bf16 else 4096
        sizes = {
            0: 4,
            1: 4,
            2: 4,
            3: 1,
            4: 1,
            5: 16,
            6: 8,
            7: 32,
            8: 16,
            9: 16,
            10: 16,
            11: 4,
            12: 4,
            13: 4,
            15: 1,
            16: 1,
        }
        cb_specs = [
            (i, n, 2048 if self.bf16 and i == 7 else 4096, ttnn.bfloat16 if self.bf16 and i == 7 else ttnn.float32)
            for i, n in sizes.items()
        ]
        if self.bf16:
            cb_specs.append((14, 16, 2048, ttnn.bfloat16))
        launch(
            coords,
            [
                (
                    REC_READER,
                    ttnn.ReaderConfigDescriptor(),
                    [sb] + accessors([self.prepared, self.state]),
                    [[self.prepared.buffer_address(), self.state.buffer_address(), s, n] for s, n in partition],
                ),
                (
                    REC_WRITER,
                    ttnn.WriterConfigDescriptor(),
                    [sb, self.batch * 12] + accessors([self.state, self.output]),
                    [[self.state.buffer_address(), self.output.buffer_address(), s, n] for s, n in partition],
                ),
                (self.source, compute_config((8,) if self.bf16 else ()), [int(self.bf16)], [[n] for _, n in partition]),
            ],
            cb_specs,
            [self.prepared, self.state, self.output],
        )
        return self.output
