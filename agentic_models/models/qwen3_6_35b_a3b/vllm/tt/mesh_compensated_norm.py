# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compensated FP32 normalization for singleton linear-attention rows.

The scalar kernel retains native sqrt/reciprocal refinement and BF16 high/low
rounding. The normalization helper preserves the ordered reduction and the
separate FP32 normalization and BF16 gamma multiplication.
"""

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 const auto src=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));
 const uint32_t start=get_arg_val<uint32_t>(1), count=get_arg_val<uint32_t>(2);
 Noc noc; DataflowBuffer input(0);
 for(uint32_t row=start;row<start+count;++row) {
  input.reserve_back(1);
  noc.async_read(src,input,4096,{.page_id=row},{});
  noc.async_read_barrier(); input.push_back(1);
 }
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 const auto dst=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));
 const uint32_t start=get_arg_val<uint32_t>(1), count=get_arg_val<uint32_t>(2);
 Noc noc; DataflowBuffer output(16);
 for(uint32_t row=start;row<start+count;++row) {
  output.wait_front(1);
  noc.async_write(output,dst,4096,{}, {.page_id=row});
  noc.async_write_barrier(); output.pop_front(1);
 }
}
"""

COMPUTE = r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/dataflow/dataflow_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu.h"
sfpi_inline sfpi::vFloat round_bf16(sfpi::vFloat x) {
 sfpi::vUInt bits=sfpi::as<sfpi::vUInt>(x);
 sfpi::vUInt tie=(bits>>16)&1;
 bits=bits+0x7fff; bits=bits+tie; bits=bits&0xffff0000;
 return sfpi::as<sfpi::vFloat>(bits);
}
template<uint32_t phase> inline void refine() {
 sfpu::_init_sfpu_config_reg();
 addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);
 math::reset_counters(p_setrwc::SET_ABD_F);
 _llk_math_eltwise_sfpu_start_(0);
 // FP32 tile registers0,1,2,3 are32 vector registers apart.
 for(uint32_t lane=0;lane<32;++lane) {
  if constexpr(phase==0) {
   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::dst_reg[0]=x*0x1p-11f; }
   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::dst_reg[0]=x+0x1.0c6f7a0b5ed8dp-20f; }
   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::dst_reg[32]=x; }
  } else if constexpr(phase==1) {
   sfpi::vFloat root=sfpi::dst_reg[32];
   sfpi::vFloat hi=round_bf16(root);
   sfpi::vFloat lo=root-hi;
   sfpi::dst_reg[64]=hi*hi;
   { sfpi::vFloat x=sfpi::dst_reg[0], y=sfpi::dst_reg[64]; sfpi::dst_reg[64]=x-y; }
   sfpi::dst_reg[96]=hi*2.0f;
   { sfpi::vFloat x=sfpi::dst_reg[96]; sfpi::dst_reg[96]=x*lo; }
   { sfpi::vFloat x=sfpi::dst_reg[64], y=sfpi::dst_reg[96]; sfpi::dst_reg[64]=x-y; }
   sfpi::dst_reg[96]=lo*lo;
   { sfpi::vFloat x=sfpi::dst_reg[64], y=sfpi::dst_reg[96]; sfpi::dst_reg[64]=x-y; }
   sfpi::dst_reg[96]=root;
  } else if constexpr(phase==2) {
   { sfpi::vFloat x=sfpi::dst_reg[96]; sfpi::dst_reg[96]=x*0.5f; }
   { sfpi::vFloat x=sfpi::dst_reg[64], y=sfpi::dst_reg[96]; sfpi::dst_reg[64]=x*y; }
   { sfpi::vFloat x=sfpi::dst_reg[32], y=sfpi::dst_reg[64]; sfpi::dst_reg[32]=x+y; }
   { sfpi::vFloat x=sfpi::dst_reg[32]; sfpi::dst_reg[96]=x; }
  } else {
   sfpi::vFloat root=sfpi::dst_reg[32], inv=sfpi::dst_reg[96];
   sfpi::vFloat rh=round_bf16(root), ih=round_bf16(inv);
   sfpi::vFloat rl=root-rh, il=inv-ih;
   sfpi::dst_reg[64]=rh*ih;
   { sfpi::vFloat x=sfpi::dst_reg[64]; sfpi::dst_reg[64]=1.0f-x; }
   sfpi::dst_reg[0]=rh*il;
   { sfpi::vFloat x=sfpi::dst_reg[64], y=sfpi::dst_reg[0]; sfpi::dst_reg[64]=x-y; }
   sfpi::dst_reg[0]=rl*ih;
   { sfpi::vFloat x=sfpi::dst_reg[64], y=sfpi::dst_reg[0]; sfpi::dst_reg[64]=x-y; }
   sfpi::dst_reg[0]=rl*il;
   { sfpi::vFloat x=sfpi::dst_reg[64], y=sfpi::dst_reg[0]; sfpi::dst_reg[64]=x-y; }
   { sfpi::vFloat x=sfpi::dst_reg[64]; sfpi::dst_reg[0]=inv*x; }
   { sfpi::vFloat x=sfpi::dst_reg[0]; sfpi::dst_reg[96]=inv+x; }
  }
  sfpi::dst_reg++;
 }
 _llk_math_eltwise_sfpu_done_();
}
#endif
void kernel_main() {
 const uint32_t count=get_arg_val<uint32_t>(0);
 DataflowBuffer input(0), output(16);
 compute_kernel_hw_startup(0,16); copy_init(0);
 for(uint32_t row=0;row<count;++row) {
  input.wait_front(1); tile_regs_acquire(); copy_tile(0,0,0);
#ifdef TRISC_MATH
  refine<0>();
#endif
  sqrt_tile_init(); sqrt_tile<false,true>(1);
#ifdef TRISC_MATH
  refine<1>();
#endif
  recip_tile_init<ReciprocalDestAcc::FP32>(); recip_tile<ReciprocalDestAcc::FP32>(3);
#ifdef TRISC_MATH
  refine<2>();
#endif
  // Re-initialize before every reciprocal, matching native separate dispatch.
  recip_tile_init<ReciprocalDestAcc::FP32>(); recip_tile<ReciprocalDestAcc::FP32>(3);
#ifdef TRISC_MATH
  refine<3>();
#endif
  tile_regs_commit(); input.pop_front(1); tile_regs_wait();
  output.reserve_back(1); pack_tile(3,16); output.push_back(1);
  tile_regs_release();
 }
}
"""


class CompensatedNormEpilogue:
    """Map an existing terminal sum [rows,1,1,1] to its corrected inverse."""

    def __init__(self, ttnn, mesh, epsilon):
        if epsilon != 1e-6:
            raise ValueError("compensated norm currently encodes the Qwen epsilon 1e-6 exactly")
        self.t, self.mesh = ttnn, mesh
        self.grid = mesh.compute_with_storage_grid_size()

    def __call__(self, total):
        t = self.t
        if tuple(total.shape)[1:] != (1, 1, 1) or total.dtype != t.float32:
            raise ValueError("expected FP32 terminal sum [rows,1,1,1]")
        if total.layout != t.TILE_LAYOUT or total.memory_config() not in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG):
            raise ValueError("terminal sum must be interleaved TILE")
        rows = total.shape[0]
        active = min(rows, self.grid.x * self.grid.y)
        coords = [(i % self.grid.x, i // self.grid.x) for i in range(active)]
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y in coords])
        output = t.empty(
            tuple(total.shape),
            dtype=t.float32,
            layout=t.TILE_LAYOUT,
            device=self.mesh,
            memory_config=total.memory_config(),
        )
        rd, wr, cr = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
        start = 0
        for i, (x, y) in enumerate(coords):
            count = rows // active + (i < rows % active)
            rd[x][y] = [total.buffer_address(), start, count]
            wr[x][y] = [output.buffer_address(), start, count]
            cr[x][y] = [count]
            start += count
        compute = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        )
        modes = [t.UnpackToDestMode.Default] * 64
        modes[0] = t.UnpackToDestMode.UnpackToDestFp32
        compute.unpack_to_dest_mode = modes

        def kernel(source, config, compile_args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=config,
                compile_time_args=compile_args,
                runtime_args=runtime,
            )

        program = t.ProgramDescriptor(
            kernels=[
                kernel(READER, t.ReaderConfigDescriptor(), t.TensorAccessorArgs(total).get_compile_time_args(), rd),
                kernel(COMPUTE, compute, [], cr),
                kernel(WRITER, t.WriterConfigDescriptor(), t.TensorAccessorArgs(output).get_compile_time_args(), wr),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=4096,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(index, t.float32, 4096)],
                )
                for index in (0, 16)
            ],
        )
        t.generic_op([total, output], program)
        return output


def normalized_singleton(dec, x, key, epilogue):
    """Normalize one linear-attention row with the existing reduction order.

    ``epilogue`` is a construction-time CompensatedNormEpilogue instance for
    this decoder's mesh and epsilon. The caller selects this singleton path.
    """
    import ttnn as t

    if (
        dec.kind != "linear_attention"
        or key not in ("input_layernorm.weight", "post_attention_layernorm.weight")
        or tuple(x.shape)[-2:] != (1, 2048)
    ):
        raise ValueError("expected a singleton 2048-wide linear-attention norm input")
    original_memory = x.memory_config()
    wide = t.to_memory_config(x, t.DRAM_MEMORY_CONFIG)
    squares = t.reshape(t.multiply(wide, wide, dtype=t.float32, fast_and_approximate_mode=False), (4, 16, 1, 32))
    total = t.empty(
        (1, 1, 1, 32),
        dtype=t.float32,
        layout=t.TILE_LAYOUT,
        device=dec.mesh_device,
        memory_config=t.DRAM_MEMORY_CONFIG,
    )
    total = t.experimental.fast_reduce_nc(squares, dims=[0, 1], output=total, compute_kernel_config=dec.compute)
    total = dec.norm_terminal_scan(t.reshape(total, (1, 1, 4, 8)))
    inverse = t.reshape(epilogue(total), (1, 1, 1))
    normalized = t.multiply(wide, inverse, dtype=t.float32, fast_and_approximate_mode=False)
    result = t.multiply(normalized, dec.w[key], dtype=t.bfloat16, fast_and_approximate_mode=False)
    return t.to_memory_config(result, original_memory)
