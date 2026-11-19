# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in exact-order contraction of native FP32 recurrence products.

Not installed in production. Thirty-two workers each read four original tiles;
only local destination registers are transposed. No padded HC tensor is built.
Hardware equality and timing are mandatory before using this experimental op.
"""

HEADER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
"""
READER = (
    HEADER
    + r"""
void kernel_main() {
 const auto src=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));
 const uint32_t head=get_arg_val<uint32_t>(1), col=get_arg_val<uint32_t>(2);
 DataflowBuffer input(0); Noc noc; input.reserve_back(4);
 for(uint32_t row=0;row<4;++row)
  noc.async_read(src,input,4096,{.page_id=head*16+row*4+col},{.offset_bytes=row*4096});
 noc.async_read_barrier(); input.push_back(4);
}
"""
)
WRITER = (
    HEADER
    + r"""
void kernel_main() {
 constexpr bool terminal=get_compile_time_arg_val(0)!=0;
 const auto dst=TensorAccessor(TensorAccessorArgs<1>(),get_arg_val<uint32_t>(0));
 const uint32_t head=get_arg_val<uint32_t>(1),col=get_arg_val<uint32_t>(2);
 DataflowBuffer out(16);Noc noc;out.wait_front(1);
 if constexpr(terminal) {
  // Disjoint 64-byte half rows. Padding is deliberately unspecified; valid
  // rows are fully written by their owners without any inter-worker barrier.
  noc.async_write(out,dst,64,{.offset_bytes=0},{.page_id=col,.offset_bytes=head*64});
  noc.async_write(out,dst,64,{.offset_bytes=1024},{.page_id=col,.offset_bytes=1024+head*64});
 } else {
  noc.async_write(out,dst,4096,{}, {.page_id=head*4+col});
 }
 noc.async_write_barrier();out.pop_front(1);
}
"""
)
COMPUTE = r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
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
 DataflowBuffer in(0),out(16);in.wait_front(4);tile_regs_acquire();
 copy_tile(0,0,0);copy_tile(0,1,1);copy_tile(0,2,2);copy_tile(0,3,3);
#ifdef TRISC_MATH
 exact_sum();
#endif
 tile_regs_commit();tile_regs_wait();out.reserve_back(1);pack_tile(4,16);
 out.push_back(1);tile_regs_release();in.pop_front(4);
}
"""


class DirectRecurrentSum:
    def __init__(self, t, mesh):
        self.t, self.mesh = t, mesh
        grid = mesh.compute_with_storage_grid_size()
        if grid.x * grid.y < 32:
            raise ValueError("direct recurrent sum needs 32 workers")
        self.coords = [(i % grid.x, i // grid.x) for i in range(32)]
        self.cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y in self.coords])

    def __call__(self, products, *, terminal_rows=False):
        t = self.t
        if tuple(products.shape) != (1, 8, 128, 128) or products.dtype != t.float32:
            raise ValueError("expected local FP32 [1,8,128,128] recurrence products")
        if products.layout != t.TILE_LAYOUT or products.memory_config() != t.L1_MEMORY_CONFIG:
            raise ValueError("expected interleaved L1 tiled products")
        output = t.empty(
            (1, 1, 8, 128) if terminal_rows else (1, 8, 1, 128),
            device=self.mesh,
            dtype=t.float32,
            layout=t.TILE_LAYOUT,
            memory_config=products.memory_config(),
        )
        rd, wr = t.RuntimeArgs(), t.RuntimeArgs()
        for i, (x, y) in enumerate(self.coords):
            rd[x][y] = [products.buffer_address(), i // 4, i % 4]
            wr[x][y] = [output.buffer_address(), i // 4, i % 4]
        config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,
        )
        modes = [t.UnpackToDestMode.Default] * 64
        modes[0] = t.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes

        def kernel(source, cfg, args=(), runtime=None):
            kw = dict(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=self.cores,
                config=cfg,
                compile_time_args=list(args),
            )
            if runtime is not None:
                kw["runtime_args"] = runtime
            return t.KernelDescriptor(**kw)

        program = t.ProgramDescriptor(
            kernels=[
                kernel(READER, t.ReaderConfigDescriptor(), t.TensorAccessorArgs(products).get_compile_time_args(), rd),
                kernel(COMPUTE, config),
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
                for index, count in ((0, 4), (16, 1))
            ],
        )
        t.generic_op([products, output], program)
        return output


def install(dec):
    """Explicit experimental installation; no caller is changed by import."""
    from types import MethodType

    import ttnn as t

    if dec.kind != "linear_attention":
        return dec
    if hasattr(dec, "direct_recurrent_sum_probe"):
        raise ValueError("direct recurrent sum already installed")
    original = dec._recurrent_sum
    engine = DirectRecurrentSum(t, dec.mesh_device)

    def contract(self, products):
        if tuple(products.shape) != (1, 8, 128, 128):
            return original(products)
        return engine(products)

    dec.direct_recurrent_sum_probe = engine
    dec._recurrent_sum = MethodType(contract, dec)
    return dec
