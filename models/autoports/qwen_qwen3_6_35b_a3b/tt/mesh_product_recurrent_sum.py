# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qualified FP32 product plus exact recurrence contraction.

Literal reader/compute sources and descriptors match the stage7 component and
layer0/layer34 controls. q/k layout, FP32 products, q scaling and ordered sums
are unchanged; products remain in destination registers until contraction.
The caller retains its selected path for unsupported input contracts.
"""

from .mesh_direct_recurrent_sum import WRITER, DirectRecurrentSum

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
 constexpr uint32_t second=get_compile_time_arg_val(0);
 const auto state=TensorAccessor(TensorAccessorArgs<1>(),get_arg_val<uint32_t>(0));
 const auto key=TensorAccessor(TensorAccessorArgs<second>(),get_arg_val<uint32_t>(1));
 const uint32_t head=get_arg_val<uint32_t>(2),col=get_arg_val<uint32_t>(3);
 DataflowBuffer input(0),column(1);Noc noc;
 input.reserve_back(4);column.reserve_back(4);
 // Local CB allocation uses DRAM alignment (ProgramImpl::allocate_circular_buffers).
 ASSERT((get_write_ptr(0)&63)==0);ASSERT((get_write_ptr(1)&63)==0);
 for(uint32_t row=0;row<4;++row) {
  noc.async_read(state,input,4096,{.page_id=head*16+row*4+col},{.offset_bytes=row*4096});
  noc.async_read(key,column,4096,{.page_id=(head/2)*4+row},{.offset_bytes=row*4096});
 }
 noc.async_read_barrier();input.push_back(4);column.push_back(4);
}
"""

COMPUTE = r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/dataflow_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu.h"
#include "ckernel_sfpu_cumsum.h"
// The SFPU transpose produces four successive rows in LREG0..3. A
// second transpose preserves the running carry in the same representation
// used by the selected compensated scan. Unlike cumsum, each group starts
// with its first operand, with no initial addition of zero.
template<uint32_t GROUP,uint32_t PASS> inline void rows() {
 constexpr uint32_t tile=GROUP/2,half=GROUP%2;
 constexpr uint32_t row=tile*64+half*32+PASS*4;
 TTI_SFPLOAD(0,InstrModLoadStore::DEFAULT,ADDR_MOD_7,row);
 TTI_SFPLOAD(1,InstrModLoadStore::DEFAULT,ADDR_MOD_7,row+2);
 TTI_SFPLOAD(2,InstrModLoadStore::DEFAULT,ADDR_MOD_7,row+16);
 TTI_SFPLOAD(3,InstrModLoadStore::DEFAULT,ADDR_MOD_7,row+18);
 TTI_SFPTRANSP(0,0,0,0);
 if constexpr(PASS!=0) {TTI_SFPADD(10,7,0,0,0);TTI_SFPNOP;}
 TTI_SFPADD(10,0,1,1,0);TTI_SFPNOP;
 TTI_SFPADD(10,1,2,2,0);TTI_SFPNOP;
 TTI_SFPADD(10,2,3,3,0);TTI_SFPNOP;
 if constexpr(PASS==3) {
  if constexpr(GROUP!=0) {
   TTI_SFPLOAD(4,InstrModLoadStore::DEFAULT,ADDR_MOD_7,256);
   TTI_SFPADD(10,4,3,3,0);TTI_SFPNOP;
  }
  // Raw transposed scratch, not a packed tensor boundary.
  TTI_SFPSTORE(3,InstrModLoadStore::DEFAULT,ADDR_MOD_7,256);
 }
 TTI_SFPMOV(0,3,7,0);
 TTI_SFPTRANSP(0,0,0,0);
}
template<uint32_t GROUP> inline void group() {
 rows<GROUP,0>();rows<GROUP,1>();rows<GROUP,2>();rows<GROUP,3>();
}
inline void exact_sum() {
 sfpu::_init_sfpu_config_reg();
 // Explicit SFPU addressing must not inherit datacopy address modifiers.
 addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_7);
 math::reset_counters(p_setrwc::SET_ABD_F);
 _llk_math_eltwise_sfpu_start_(0);
 group<0>();group<1>();group<2>();group<3>();
 group<4>();group<5>();group<6>();group<7>();
 TTI_SFPLOAD(0,InstrModLoadStore::DEFAULT,ADDR_MOD_7,256);
 TTI_SFPMOV(0,0,1,0);TTI_SFPMOV(0,0,2,0);TTI_SFPMOV(0,0,3,0);
 TTI_SFPTRANSP(0,0,0,0);
 TTI_SFPSTORE(0,InstrModLoadStore::DEFAULT,ADDR_MOD_7,256);
 TTI_SFPSTORE(1,InstrModLoadStore::DEFAULT,ADDR_MOD_7,258);
 TTI_SFPSTORE(2,InstrModLoadStore::DEFAULT,ADDR_MOD_7,272);
 TTI_SFPSTORE(3,InstrModLoadStore::DEFAULT,ADDR_MOD_7,274);
 _llk_math_eltwise_sfpu_done_();
}
#endif
void kernel_main() {
 compute_kernel_hw_startup(0,16);copy_init(0);
 constexpr bool scaled=get_compile_time_arg_val(0)!=0;
 DataflowBuffer in(0),column(1),out(16);in.wait_front(4);column.wait_front(4);tile_regs_acquire();
 for(uint32_t tile=0;tile<4;++tile) {
  copy_init(0);copy_tile(0,tile,tile);
  copy_init(1);copy_tile(1,tile,4);
  if constexpr(scaled) {
   binop_with_scalar_tile_init();mul_unary_tile<true>(4,0x3db504f3);
  }
  sfpu_mul_bcast_col_init();sfpu_mul_bcast_col<true>(tile,4);
 }

#ifdef TRISC_MATH
 exact_sum();
#endif
 tile_regs_commit();tile_regs_wait();out.reserve_back(1);pack_tile(4,16);
 out.push_back(1);tile_regs_release();in.pop_front(4);column.pop_front(4);
}
"""


class ProductRecurrentSum(DirectRecurrentSum):
    def supports(self, state, column):
        t = self.t
        return (
            tuple(state.shape) == (1, 8, 128, 128)
            and tuple(column.shape) in ((1, 4, 128, 1), (1, 4, 1, 128, 1))
            and all(
                value.dtype == t.float32
                and value.layout == t.TILE_LAYOUT
                and value.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)
                for value in (state, column)
            )
        )

    def __call__(self, state, column, *, scale_q=False, terminal_rows=False):
        t = self.t
        if tuple(state.shape) != (1, 8, 128, 128):
            raise ValueError("expected local state [1,8,128,128]")
        if tuple(column.shape) not in ((1, 4, 128, 1), (1, 4, 1, 128, 1)):
            raise ValueError("expected existing q/k column geometry")
        for value in (state, column):
            if value.dtype != t.float32 or value.layout != t.TILE_LAYOUT:
                raise ValueError("FP32 tiled operands required")
            if value.memory_config() not in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG):
                raise ValueError("interleaved tiled operands required")
        output = t.empty(
            (1, 1, 8, 128) if terminal_rows else (1, 8, 1, 128),
            device=self.mesh,
            dtype=t.float32,
            layout=t.TILE_LAYOUT,
            memory_config=t.L1_MEMORY_CONFIG,
        )
        rd, wr = t.RuntimeArgs(), t.RuntimeArgs()
        for i, (x, y) in enumerate(self.coords):
            rd[x][y] = [state.buffer_address(), column.buffer_address(), i // 4, i % 4]
            wr[x][y] = [output.buffer_address(), i // 4, i % 4]
        sa = list(t.TensorAccessorArgs(state).get_compile_time_args())
        ka = list(t.TensorAccessorArgs(column).get_compile_time_args())
        config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        modes[0] = modes[1] = t.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes

        def kernel(source, cfg, args, runtime=None):
            kw = dict(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=self.cores,
                config=cfg,
                compile_time_args=args,
            )
            if runtime is not None:
                kw["runtime_args"] = runtime
            return t.KernelDescriptor(**kw)

        program = t.ProgramDescriptor(
            kernels=[
                kernel(READER, t.ReaderConfigDescriptor(), [1 + len(sa), *sa, *ka], rd),
                kernel(COMPUTE, config, [int(scale_q)]),
                kernel(
                    WRITER,
                    t.WriterConfigDescriptor(),
                    [int(terminal_rows), *t.TensorAccessorArgs(output).get_compile_time_args()],
                    wr,
                ),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=count * 4096,
                    core_ranges=self.cores,
                    format_descriptors=[t.CBFormatDescriptor(index, t.float32, 4096)],
                )
                for index, count in ((0, 4), (1, 4), (16, 1))
            ],
        )
        t.generic_op([state, column, output], program)
        return output
