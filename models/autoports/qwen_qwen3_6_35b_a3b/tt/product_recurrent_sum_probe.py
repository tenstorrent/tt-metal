# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experimental FP32 product plus exact recurrence contraction, opt-in only.

Reuses the independently validated direct contraction source unchanged. Existing
q/k column-layout transformations remain. Native SFPU column broadcast avoids
scalar BRISC expansion; all products round to FP32 in destination registers.
"""
from types import MethodType

from .direct_recurrent_sum_probe import COMPUTE as SUM_COMPUTE
from .direct_recurrent_sum_probe import HEADER, WRITER, DirectRecurrentSum

READER = (
    HEADER
    + r"""
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
)

# The accepted reduction is imported verbatim. Only the input/product stage and
# balanced column CB lifetime differ; no edits to DirectRecurrentSum are needed.
COMPUTE = (
    SUM_COMPUTE.replace(
        '#include "api/compute/pack.h"',
        '#include "api/compute/pack.h"\n#include "api/compute/sfpu_binary_bcast.h"\n'
        '#include "api/compute/eltwise_unary/binop_with_scalar.h"',
    )
    .replace(
        "DataflowBuffer in(0),out(16);in.wait_front(4);tile_regs_acquire();\n"
        " copy_tile(0,0,0);copy_tile(0,1,1);copy_tile(0,2,2);copy_tile(0,3,3);",
        """constexpr bool scaled=get_compile_time_arg_val(0)!=0;
 DataflowBuffer in(0),column(1),out(16);in.wait_front(4);column.wait_front(4);tile_regs_acquire();
 for(uint32_t tile=0;tile<4;++tile) {
  copy_init(0);copy_tile(0,tile,tile);
  copy_init(1);copy_tile(1,tile,4);
  if constexpr(scaled) {
   binop_with_scalar_tile_init();mul_unary_tile<true>(4,0x3db504f3);
  }
  sfpu_mul_bcast_col_init();sfpu_mul_bcast_col<true>(tile,4);
 }
""",
    )
    .replace("in.pop_front(4);", "in.pop_front(4);column.pop_front(4);")
)
assert "sfpu_mul_bcast_col<true>(tile,4)" in COMPUTE


class ProductRecurrentSum(DirectRecurrentSum):
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


def install(dec):
    """Experimental complete recurrence adapter; unchanged pointwise state work."""
    import ttnn as t

    from .fused_decoder import reshape

    if dec.kind != "linear_attention":
        return dec
    if hasattr(dec, "product_recurrent_sum_probe"):
        raise ValueError("product recurrence sum already installed")
    original = dec._recurrent_step
    engine = ProductRecurrentSum(t, dec.mesh_device)

    def step(self, q, k, v, g, beta, state, *, decay_value=None, round_state=False, output_state=None):
        if decay_value is None or tuple(q.shape) != (1, 1, 4, 128):
            return original(
                q, k, v, g, beta, state, decay_value=decay_value, round_state=round_state, output_state=output_state
            )
        q = t.reshape(t.permute(q, (0, 2, 3, 1)), (1, 4, 1, 128, 1))
        k = t.reshape(t.permute(k, (0, 2, 3, 1)), (1, 4, 1, 128, 1))
        v = t.permute(t.typecast(v, t.float32), (0, 2, 1, 3))
        beta = reshape(beta, (1, 8, 1, 1))
        decayed = t.multiply(
            state,
            reshape(decay_value, (1, 8, 1, 1)),
            input_tensor_a_activations=self.round_widen_chain if round_state else [],
            memory_config=t.L1_MEMORY_CONFIG,
        )
        memory = engine(decayed, k)
        delta = t.multiply(t.subtract(v, memory, memory_config=t.L1_MEMORY_CONFIG), beta)
        update = t.multiply(k, t.reshape(delta, (1, 4, 2, 1, 128)), memory_config=t.L1_MEMORY_CONFIG)
        new = t.add(decayed, t.reshape(update, (1, 8, 128, 128)), output_tensor=output_state)
        output = engine(new, q, scale_q=True, terminal_rows=True)
        return output, new

    dec.product_recurrent_sum_probe = engine
    dec._recurrent_step = MethodType(step, dec)
    return dec
