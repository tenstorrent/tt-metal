# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated exact-scan candidates. No default installation or precision change.

Narrow activation reads and terminal DST stores are independent switches.
The baseline reader/compute/writer and schedule come from current production.
"""

import math

from .fused_decoder import (
    DIRECT_COEFFICIENT_READER,
    STREAMING_SCAN_COMPUTE,
    STREAMING_SCAN_PRODUCT_WRITER,
    STREAMING_SCAN_WRITER,
    terminal_projection_schedule,
)


def reader_source(read_bytes):
    if read_bytes == 2048:
        return DIRECT_COEFFICIENT_READER
    if read_bytes not in (32, 64):
        raise ValueError("read_bytes must be 2048, 32 or 64")
    old = "   noc.async_read(activation,raw,2048,{.page_id=column},{.offset_bytes=0});"
    new = f"""   // BH requires source/destination low six bits to agree.
   const uint32_t raw_bias=(64-(get_write_ptr(2)&63))&63;
   noc.async_read(activation,raw,{read_bytes},{{.page_id=column,.offset_bytes=0}},{{.offset_bytes=raw_bias}});
   noc.async_read(activation,raw,{read_bytes},{{.page_id=column,.offset_bytes=512}},{{.offset_bytes=raw_bias+512}});"""
    assert DIRECT_COEFFICIENT_READER.count(old) == 1
    result = DIRECT_COEFFICIENT_READER.replace(old, new)
    old_pointer = "get_read_ptr(2));"
    assert result.count(old_pointer) == 1
    return result.replace(old_pointer, "get_read_ptr(2)+raw_bias);")


TERMINAL_SFPU = r"""
#ifdef TRISC_MATH
namespace ckernel { namespace sfpu {
template<uint32_t PASS, bool NOP_STORES>
inline void stage7_terminal_pass(bool emit) {
 constexpr uint32_t LREG=(PASS%2)*4;
 constexpr uint32_t row=(PASS%4)*4+(PASS/4)*32;
 TTI_SFPLOAD(LREG, InstrModLoadStore::DEFAULT, ADDR_MOD_7, row);
 TTI_SFPLOAD(LREG+1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, row+2);
 TTI_SFPLOAD(LREG+2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, row+16);
 TTI_SFPLOAD(LREG+3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, row+18);
 TTI_SFPTRANSP(0,0,0,0);
 lltt::replay((PASS%2)*cumsum_step,cumsum_step);
 TTI_SFPTRANSP(0,0,0,0);
 if constexpr(PASS==7) {
  if(emit) {
   TTI_SFPSTORE(LREG, InstrModLoadStore::DEFAULT, ADDR_MOD_7, row);
   TTI_SFPSTORE(LREG+1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, row+2);
   TTI_SFPSTORE(LREG+2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, row+16);
   TTI_SFPSTORE(LREG+3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, row+18);
  } else if constexpr(NOP_STORES) { TTI_SFPNOP; TTI_SFPNOP; TTI_SFPNOP; TTI_SFPNOP; }
 } else if constexpr(NOP_STORES) { TTI_SFPNOP; TTI_SFPNOP; TTI_SFPNOP; TTI_SFPNOP; }
}
template<bool APPROXIMATION_MODE, bool NOP_STORES>
inline void stage7_terminal_cumsum(bool first,bool emit) {
 if(first) set_carried_lregs<p_sfpu::LCONST_0>();
 stage7_terminal_pass<0,NOP_STORES>(emit);
 stage7_terminal_pass<1,NOP_STORES>(emit);
 stage7_terminal_pass<2,NOP_STORES>(emit);
 stage7_terminal_pass<3,NOP_STORES>(emit);
 stage7_terminal_pass<4,NOP_STORES>(emit);
 stage7_terminal_pass<5,NOP_STORES>(emit);
 stage7_terminal_pass<6,NOP_STORES>(emit);
 stage7_terminal_pass<7,NOP_STORES>(emit);
}
}}
#endif
namespace ckernel {
inline void stage7_terminal_tile(uint32_t idst,bool first,bool emit) {
 MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,stage7_terminal_cumsum,
                     (false,STAGE7_NOP_STORES),idst,VectorMode::RC_custom,first,emit));
}
}
"""


def compute_source(stores):
    if stores == "baseline":
        return STREAMING_SCAN_COMPUTE
    if stores not in ("nop", "elide"):
        raise ValueError("stores must be baseline, nop or elide")
    # ADDR_MOD_7 is zero-increment. Preserve all loads/transposes/replay ADDs
    # and arithmetic NOPs. Elision remains an experimental hazard hypothesis;
    # validate NOP control and exact repeated replay before using its timings.
    prefix = "#define STAGE7_NOP_STORES " + ("true" if stores == "nop" else "false") + "\n"
    old = "    cumsum_tile<CumsumMode::Default>(0,h==0);"
    assert STREAMING_SCAN_COMPUTE.count(old) == 1
    result = STREAMING_SCAN_COMPUTE.replace(old, "    stage7_terminal_tile(0,h==0,h+1==ht);")
    assert result.count("void kernel_main() {") == 1
    return prefix + result.replace("void kernel_main() {", TERMINAL_SFPU + "\nvoid kernel_main() {")


class ScanTerminalProbe:
    """Same streaming FPU/cumsum, with raw BF16 original-input coefficient construction."""

    def __init__(self, ttnn, mesh, *, read_bytes=2048, stores="baseline"):
        self.ttnn, self.mesh = (ttnn, mesh)
        self.grid = mesh.compute_with_storage_grid_size()
        self.reader_source = reader_source(read_bytes)
        self.compute_source = compute_source(stores)

    def __call__(self, weight, activation, *, products_only=False, coefficients_only=False):
        t = self.ttnn
        shape = tuple(weight.shape)
        assert shape[-2] in (512, 1024) and shape[-1] % 32 == 0 and (len(shape) == 4) and (shape[1] == 1)
        total_k = math.prod(shape[:-2]) * shape[-2]
        assert tuple(activation.shape) == (1, 1, total_k)
        assert not (products_only and coefficients_only)
        assert weight.dtype == activation.dtype == t.bfloat16 and weight.layout == activation.layout == t.TILE_LAYOUT
        assert weight.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)
        ht, wt = (shape[-2] // 32, shape[-1] // 32)
        partition = terminal_projection_schedule(math.prod(shape[:-2]) * wt, self.grid.x, self.grid.y)
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partition])
        output_shape = (
            shape[:-1] + (1,) if coefficients_only else shape if products_only else shape[:-2] + (1, shape[-1])
        )
        output = t.empty(
            output_shape,
            device=self.mesh,
            dtype=t.bfloat16 if coefficients_only else t.float32,
            layout=t.TILE_LAYOUT,
            memory_config=t.DRAM_MEMORY_CONFIG if products_only or coefficients_only else t.L1_MEMORY_CONFIG,
        )
        rd, wr, cr = (t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs())
        for x, y, start, count in partition:
            rd[x][y] = [
                weight.buffer_address(),
                activation.buffer_address(),
                start,
                count,
                ht,
                wt,
                output.buffer_address(),
            ]
            wr[x][y] = [output.buffer_address(), start, count, ht, wt]
            cr[x][y] = [count]
        config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
        )

        def kernel(source, conf, args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=conf,
                compile_time_args=args,
                runtime_args=runtime,
            )

        accessors = [t.TensorAccessorArgs(value) for value in (weight, activation, output)]
        assert all((not a.get_common_runtime_args() for a in accessors))
        wa, xa, oa = [a.get_compile_time_args() for a in accessors]
        kernels = [
            kernel(
                self.reader_source,
                t.ReaderConfigDescriptor(),
                [3 + len(wa), 3 + len(wa) + len(xa), int(coefficients_only)] + wa + xa + oa,
                rd,
            )
        ]
        if not coefficients_only:
            kernels += [
                kernel(self.compute_source, config, [ht, int(products_only)], cr),
                kernel(
                    STREAMING_SCAN_PRODUCT_WRITER if products_only else STREAMING_SCAN_WRITER,
                    t.WriterConfigDescriptor(),
                    oa,
                    wr,
                ),
            ]
        program = t.ProgramDescriptor(
            kernels=kernels,
            cbs=[
                t.CBDescriptor(
                    total_size=pages * page,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(index, dtype, page)],
                )
                for index, pages, page, dtype in [
                    (0, 2, 2048, t.bfloat16),
                    (1, 2, 2048, t.bfloat16),
                    (2, 1, 2048, t.bfloat16),
                    (16, 1, 4096, t.float32),
                    (17, 1, 4096, t.float32),
                ]
            ],
        )
        t.generic_op([weight, activation, output], program)
        return output
