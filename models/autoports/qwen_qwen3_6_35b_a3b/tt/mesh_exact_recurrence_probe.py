# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in exact-operation FP32 recurrence fusion; hardware qualification required.

Two SFPU kernels fuse pointwise work. Both original _recurrent_sum calls remain,
including their sequential 16-then-8 reduction order and layout transforms.
No native matmul, changed reduction, or implicit production installation occurs.
"""

from types import MethodType

HEADER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
constexpr uint32_t face(uint32_t r,uint32_t c) {
 return (r/16)*512+(c/16)*256+(r%16)*16+c%16;
}
"""

WRITER = (
    HEADER
    + r"""
void kernel_main() {
 constexpr uint32_t second=get_compile_time_arg_val(0);
 const auto a=TensorAccessor(TensorAccessorArgs<1>(),get_arg_val<uint32_t>(0));
 const auto b=TensorAccessor(TensorAccessorArgs<second>(),get_arg_val<uint32_t>(1));
 const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
 Noc noc;DataflowBuffer first(16),products(17);
 for(uint32_t tile=start;tile<start+count;++tile) {
  first.wait_front(1);products.wait_front(1);
  noc.async_write(first,a,4096,{}, {.page_id=tile});
  noc.async_write(products,b,4096,{}, {.page_id=tile});
  noc.async_write_barrier();first.pop_front(1);products.pop_front(1);
 }
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
sfpi_inline sfpi::vFloat round_bf16(sfpi::vFloat x) {
 sfpi::vUInt b=sfpi::as<sfpi::vUInt>(x);
 sfpi::vUInt tie=(b>>16)&1;
 b=b+0x7fff;b=b+tie;b=b&0xffff0000;
 return sfpi::as<sfpi::vFloat>(b);
}
template<bool second,bool rounded> inline void pointwise() {
 sfpu::_init_sfpu_config_reg();
 addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);
 math::reset_counters(p_setrwc::SET_ABD_F);
 _llk_math_eltwise_sfpu_start_(0);
 for(uint32_t lane=0;lane<32;++lane) {
  if constexpr(!second) {
   if constexpr(rounded) {sfpi::vFloat s=sfpi::dst_reg[0];sfpi::dst_reg[0]=round_bf16(s);}
   {sfpi::vFloat s=sfpi::dst_reg[0],d=sfpi::dst_reg[64];sfpi::dst_reg[0]=s*d;}
   {sfpi::vFloat s=sfpi::dst_reg[0],k=sfpi::dst_reg[32];sfpi::dst_reg[32]=s*k;}
  } else {
   {sfpi::vFloat v=sfpi::dst_reg[96],m=sfpi::dst_reg[128];sfpi::dst_reg[96]=v-m;}
   {sfpi::vFloat d=sfpi::dst_reg[96],b=sfpi::dst_reg[160];sfpi::dst_reg[96]=d*b;}
   {sfpi::vFloat k=sfpi::dst_reg[32],d=sfpi::dst_reg[96];sfpi::dst_reg[32]=k*d;}
   {sfpi::vFloat s=sfpi::dst_reg[0],u=sfpi::dst_reg[32];sfpi::dst_reg[0]=s+u;}
   {sfpi::vFloat q=sfpi::dst_reg[64];sfpi::dst_reg[64]=q*0x1.6a09e6p-4f;}
   {sfpi::vFloat s=sfpi::dst_reg[0],q=sfpi::dst_reg[64];sfpi::dst_reg[32]=s*q;}
  }
  sfpi::dst_reg++;
 }
 _llk_math_eltwise_sfpu_done_();
}
#endif
void kernel_main() {
 constexpr bool second=get_compile_time_arg_val(0)!=0;
 constexpr bool rounded=get_compile_time_arg_val(1)!=0;
 constexpr uint32_t inputs=second?6:3;
 const uint32_t count=get_arg_val<uint32_t>(0);
 compute_kernel_hw_startup(0,16);
 DataflowBuffer first(16),products(17);
 for(uint32_t tile=0;tile<count;++tile) {
  for(uint32_t i=0;i<inputs;++i)cb_wait_front(i,1);
  tile_regs_acquire();
  // Explicit copies keep all FP32 inputs on SFPU; no TF32 FPU conversion.
  copy_init(0);copy_tile(0,0,0);
  copy_init(1);copy_tile(1,0,1);
  copy_init(2);copy_tile(2,0,2);
  if constexpr(second) {
   copy_init(3);copy_tile(3,0,3);
   copy_init(4);copy_tile(4,0,4);
   copy_init(5);copy_tile(5,0,5);
  }
#ifdef TRISC_MATH
  pointwise<second,rounded>();
#endif
  tile_regs_commit();tile_regs_wait();
  first.reserve_back(1);products.reserve_back(1);
  pack_tile(0,16);pack_tile(1,17);
  first.push_back(1);products.push_back(1);
  tile_regs_release();
  for(uint32_t i=0;i<inputs;++i)cb_pop_front(i,1);
 }
}
"""


def _reader(second):
    count = 6 if second else 3
    declarations = "\n".join(
        f"const auto a{i}=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val({i})>(),get_arg_val<uint32_t>({i}));"
        for i in range(count)
    )
    scratch = "\n".join(f"DataflowBuffer raw{i}({i+6});raw{i}.reserve_back(1);" for i in range(1, count))
    common = r"""
 const uint32_t head=tile/16,kh=head/2,row=(tile%16)/4,col=tile%4;
 DataflowBuffer state(0);state.reserve_back(1);
 noc.async_read(a0,state,4096,{.page_id=tile},{});
 noc.async_read(a1,raw1,4096,{.page_id=row},{});
"""
    reads = (
        r"""
 noc.async_read(a2,raw2,4096,{.page_id=row},{});
 noc.async_read(a3,raw3,2048,{.page_id=col},{});
 noc.async_read(a4,raw4,4096,{.page_id=head*4+col},{});
 noc.async_read(a5,raw5,4096,{.page_id=0},{});
"""
        if second
        else r"""
 noc.async_read(a2,raw2,4096,{.page_id=0},{});
"""
    )
    fill = (
        r"""
 const auto* k=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(7));
 const auto* q=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(8));
 const auto* v=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(9));
 const auto* m=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(10));
 const auto* b=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(11));
 for(uint32_t r=0;r<32;++r)for(uint32_t c=0;c<32;++c) {
  const uint32_t dst=face(r,c);
  reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1))[dst]=k[face(kh,r)];
  reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2))[dst]=q[face(kh,r)];
  reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3))[dst]=uint32_t(v[face(head,c)])<<16;
  reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(4))[dst]=m[face(0,c)];
  reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(5))[dst]=b[face(0,head)];
 }
"""
        if second
        else r"""
 const auto* k=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(7));
 const auto* d=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(8));
 for(uint32_t r=0;r<32;++r)for(uint32_t c=0;c<32;++c) {
  const uint32_t dst=face(r,c);
  reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1))[dst]=k[face(kh,r)];
  reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2))[dst]=d[face(0,head)];
 }
"""
    )
    return (
        HEADER
        + "void kernel_main() {\n"
        + declarations
        + f"""
 const uint32_t start=get_arg_val<uint32_t>({count}),count=get_arg_val<uint32_t>({count+1});
 Noc noc;
 {scratch}
 for(uint32_t tile=start;tile<start+count;++tile) {{
 {common}{reads}
 noc.async_read_barrier();
 for(uint32_t i=1;i<{count};++i)cb_reserve_back(i,1);
 {fill}
 for(uint32_t i=0;i<{count};++i)cb_push_back(i,1);
 }}
}}
"""
    )


class ExactRecurrencePointwise:
    """Two phase kernels with FP32 tensor boundaries between every arithmetic op."""

    def __init__(self, t, mesh):
        self.t, self.mesh = t, mesh
        grid = mesh.compute_with_storage_grid_size()
        self.coords = [(i % grid.x, i // grid.x) for i in range(min(128, grid.x * grid.y))]
        self.cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y in self.coords])

    def __call__(self, inputs, *, second, rounded=False, output_state=None):
        t = self.t
        for index, value in enumerate(inputs):
            expected = t.bfloat16 if second and index == 3 else t.float32
            if value.dtype != expected or value.layout != t.TILE_LAYOUT:
                raise ValueError("recurrence probe requires the original FP32/BF16 tiled input types")
            if value.memory_config() not in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG):
                raise ValueError("recurrence probe requires interleaved inputs")

        def allocate():
            return t.empty(
                (1, 8, 128, 128),
                device=self.mesh,
                dtype=t.float32,
                layout=t.TILE_LAYOUT,
                memory_config=t.L1_MEMORY_CONFIG,
            )

        state = allocate() if output_state is None else output_state
        products = allocate()
        rd, wr, cr = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
        start = 0
        for index, (x, y) in enumerate(self.coords):
            count = 128 // len(self.coords) + (index < 128 % len(self.coords))
            rd[x][y] = [value.buffer_address() for value in inputs] + [start, count]
            wr[x][y] = [state.buffer_address(), products.buffer_address(), start, count]
            cr[x][y] = [count]
            start += count
        accessors = [list(t.TensorAccessorArgs(value).get_compile_time_args()) for value in inputs]
        offsets = []
        offset = len(inputs)
        for args in accessors:
            offsets.append(offset)
            offset += len(args)
        reader_args = offsets + [item for args in accessors for item in args]
        oa = list(t.TensorAccessorArgs(state).get_compile_time_args())
        pa = list(t.TensorAccessorArgs(products).get_compile_time_args())
        config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for index in range(len(inputs)):
            modes[index] = t.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes

        def kernel(source, config, args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=self.cores,
                config=config,
                compile_time_args=args,
                runtime_args=runtime,
            )

        program = t.ProgramDescriptor(
            kernels=[
                kernel(_reader(second), t.ReaderConfigDescriptor(), reader_args, rd),
                kernel(COMPUTE, config, [int(second), int(rounded)], cr),
                kernel(WRITER, t.WriterConfigDescriptor(), [1 + len(oa)] + oa + pa, wr),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=4096,
                    core_ranges=self.cores,
                    format_descriptors=[t.CBFormatDescriptor(index, t.float32, 4096)],
                )
                for index in list(range(len(inputs))) + list(range(7, 6 + len(inputs))) + [16, 17]
            ],
        )
        t.generic_op([*inputs, state, products], program)
        return state, products


def install(dec):
    """Opt in only for cached singleton TP4 recurrence with an explicit decay LUT."""
    import ttnn as t

    if dec.kind != "linear_attention":
        return dec
    if hasattr(dec, "exact_recurrence_probe"):
        raise ValueError("exact recurrence probe already installed")
    original = dec._recurrent_step
    engine = ExactRecurrencePointwise(t, dec.mesh_device)

    def step(self, q, k, v, g, beta, state, *, decay_value=None, round_state=False, output_state=None):
        if decay_value is None or tuple(q.shape) != (1, 1, 4, 128):
            return original(
                q, k, v, g, beta, state, decay_value=decay_value, round_state=round_state, output_state=output_state
            )
        if (
            tuple(k.shape) != (1, 1, 4, 128)
            or tuple(v.shape) != (1, 1, 8, 128)
            or tuple(state.shape) != (1, 8, 128, 128)
        ):
            raise ValueError("unexpected singleton TP4 recurrence geometry")
        if output_state is not None and (
            tuple(output_state.shape) != (1, 8, 128, 128) or output_state.dtype != t.float32
        ):
            raise ValueError("recurrence output state must retain FP32 geometry")
        decayed, products = engine([state, k, decay_value], second=False, rounded=round_state)
        memory = self._recurrent_sum(products)
        new, products = engine([decayed, k, q, v, memory, beta], second=True, output_state=output_state)
        output = self._recurrent_sum(products)
        return t.permute(output, (0, 2, 1, 3)), new

    dec.exact_recurrence_probe = engine
    dec._recurrent_step = MethodType(step, dec)
    dec.precision_policy["exact_recurrence_probe"] = {
        "experimental": True,
        "math": "FP32 SFPU with original reduction calls",
        "state_rounding": "BF16 on consumption only when round_state; live new state FP32",
    }
    return dec
