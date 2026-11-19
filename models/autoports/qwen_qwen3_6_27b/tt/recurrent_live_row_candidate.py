# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated Blackhole live-band Q/K preparation; no production registration.

Only prepare_vector pointwise work changes. The native row reduction, pack,
transpose, column replication and recurrence remain byte-for-byte source equal.
The minimum unpredicated SFPLOAD band is four rows, not one logical row.
"""

BASELINE = "cumulative_exp_scalar"
VARIANT = "cumulative_qk_live_band"
PREP_START = "void prepare_vector(uint32_t input,uint32_t output,bool is_q) {"
PREP_END = "\n\n\n// Per value-column full-DST allocation:"
HELPER_START = "inline void _gdn_mul_scalar_(uint32_t data, uint32_t scalar, uint32_t output) {"
HELPER_END = "\n\n// Replicate row0 without loading a tile of ones and multiplying by it."
WRAPPER_ANCHOR = "template<bool INIT=true>\nALWI void gdn_mul_scalar"

WRAPPERS = r"""
// VectorMode::R chooses two top faces. ITERATIONS=2 explicitly restricts
// each face to its first two 32-lane vectors (rows0..3, columns0..15).
// The underlying native arithmetic and separate DST stores are unchanged.
ALWI void gdn_live_mul(uint32_t a,uint32_t b,uint32_t o) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,(APPROX,ckernel::BinaryOp::MUL,2,DST_ACCUM_MODE),a,b,o,VectorMode::R)));
}
ALWI void gdn_live_add(uint32_t a,uint32_t b,uint32_t o) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,
        calculate_sfpu_binary,(APPROX,ckernel::BinaryOp::ADD,2,DST_ACCUM_MODE),a,b,o,VectorMode::R)));
}
ALWI void gdn_live_epsilon(uint32_t d) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,calculate_binop_with_scalar,
        (APPROX,ADD_UNARY,1,DST_ACCUM_MODE),d,VectorMode::None,0x358637bdu));
}
ALWI void gdn_live_rsqrt(uint32_t d) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,calculate_rsqrt,
        (APPROX,1,DST_ACCUM_MODE,false),d,VectorMode::None));
}
ALWI void gdn_live_scale(uint32_t d) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,DST_ACCUM_MODE,calculate_binop_with_scalar,
        (APPROX,MUL_UNARY,2,DST_ACCUM_MODE),d,VectorMode::R,0x3db504f3u));
}
ALWI void gdn_live_normalize(uint32_t d,uint32_t scalar) {
    sfpu_normalize_bcast_scalar_init();
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_mul_scalar_live_band_,d,scalar,d,VectorMode::None)));
}

"""

REPLACEMENTS = (
    ("mul_binary_tile(work,1,work);", "gdn_live_mul(work,1,work);"),
    ("add_binary_tile(0,2,0);", "gdn_live_add(0,2,0);"),
    ("add_unary_tile(0,0x358637bdu);", "gdn_live_epsilon(0);"),
    ("rsqrt_tile(0);", "gdn_live_rsqrt(0);"),
    ("gdn_mul_scalar(0,1);", "gdn_live_normalize(0,1);"),
    ("mul_unary_tile(0,0x3db504f3u);", "gdn_live_scale(0);"),
)


def _once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Live-band source drift: expected one {old[:100]!r}, found {source.count(old)}")
    return source.replace(old, new, 1)


def live_row_sources(reader, compute):
    """Transform cumulative-exp source without changing reader/CB/ownership."""
    if "gdn_live_mul(" in compute:
        raise ValueError("Live-band transform already applied")
    start, end = compute.index(HELPER_START), compute.index(HELPER_END)
    original = compute[start:end]
    helper = _once(original, "_gdn_mul_scalar_", "_gdn_mul_scalar_live_band_")
    helper = _once(
        helper,
        "for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {",
        "for(uint32_t half=0;half<1;half++) for(uint32_t band=0;band<1;band++) {",
    )
    compute = _once(compute, HELPER_END, "\n\n" + helper + HELPER_END)
    compute = _once(compute, WRAPPER_ANCHOR, WRAPPERS + WRAPPER_ANCHOR)
    start, end = compute.index(PREP_START), compute.index(PREP_END)
    old = compute[start:end]
    new = old
    for before, after in REPLACEMENTS:
        new = _once(new, before, after)
    compute = compute[:start] + new + compute[end:]
    return reader, compute


PREP_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto qa=TensorAccessorArgs<0>();
    constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(0),4096);
    const auto k=TensorAccessor(ka,get_arg_val<uint32_t>(1),4096);
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer qbuf(0),kbuf(1);
    for(uint32_t h=start;h<start+count;h++) {
        qbuf.reserve_back(4); kbuf.reserve_back(4);
        for(uint32_t i=0;i<4;i++) {
            noc.async_read(q,qbuf,4096,{.page_id=h*4+i},{.offset_bytes=i*4096});
            noc.async_read(k,kbuf,4096,{.page_id=h*4+i},{.offset_bytes=i*4096});
        }
        noc.async_read_barrier(); qbuf.push_back(4); kbuf.push_back(4);
    }
}
"""
PREP_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto qa=TensorAccessorArgs<0>();
    constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(0),4096);
    const auto k=TensorAccessor(ka,get_arg_val<uint32_t>(1),4096);
    const uint32_t start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer qbuf(13),kbuf(14);
    for(uint32_t h=start;h<start+count;h++) {
        qbuf.wait_front(1); kbuf.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(qbuf),q,4096,{},{.page_id=h});
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(kbuf),k,4096,{},{.page_id=h});
        noc.async_write_barrier(); qbuf.pop_front(1); kbuf.pop_front(1);
    }
}
"""
PREP_MAIN = r"""
void kernel_main() {
    compute_kernel_hw_startup(0,1,13);
    copy_init(0);
    const uint32_t count=get_arg_val<uint32_t>(0);
    for(uint32_t h=0;h<count;h++) {
        prepare_vector(0,13,true);
        prepare_vector(1,14,false);
    }
}
"""


def prepared_compute(compute):
    assert compute.count("void kernel_main() {") == 1
    return compute.split("void kernel_main() {", 1)[0] + PREP_MAIN


def prepare_qk(q, k, output_q, output_k, compute_source, *, cores=None):
    """Diagnostic only: expose exact packed prepared-column tiles.

    Inputs are [tasks,32,128] FP32 tiled values, normally BF16-exact row0 plus
    deliberate finite/NaN dead-row contents. Outputs [tasks,32,32] are supplied
    before capture. No internal retained tensor allocation.
    """
    import ttnn
    from .optimized_decoder import _fused_delta_row_compact_fp32_compute_config

    mesh = q.device()
    tasks = q.shape[0]
    for t in (q, k, output_q, output_k):
        assert t.device() == mesh and t.dtype == ttnn.float32
        assert t.layout == ttnn.TILE_LAYOUT and t.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(q.shape) == tuple(k.shape) == (tasks, 32, 128)
    assert tuple(output_q.shape) == tuple(output_k.shape) == (tasks, 32, 32)
    grid = mesh.compute_with_storage_grid_size()
    ncores = min(tasks, grid.x * grid.y if cores is None else cores)
    assert ncores >= 1 and ncores <= grid.x * grid.y
    partition = [(i * tasks // ncores, (i + 1) * tasks // ncores - i * tasks // ncores) for i in range(ncores)]
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])

    def accessors(ts):
        result = []
        for t in ts:
            result.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        return result

    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for coord, (start, count) in zip(coords, partition):
        rr[coord.x][coord.y] = [q.buffer_address(), k.buffer_address(), start, count]
        wr[coord.x][coord.y] = [output_q.buffer_address(), output_k.buffer_address(), start, count]
        cr[coord.x][coord.y] = [count]
    configs = [
        ttnn.ReaderConfigDescriptor(),
        ttnn.WriterConfigDescriptor(),
        _fused_delta_row_compact_fp32_compute_config(False),
    ]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=s,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )
        for s, ct, rt, cfg in zip(
            [PREP_READER, PREP_WRITER, prepared_compute(compute_source)],
            [accessors([q, k]), accessors([output_q, output_k]), [0, 0, 0, 1, 1, 1, 1, 1]],
            [rr, wr, cr],
            configs,
        )
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=n * 4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)],
        )
        for i, n in ((0, 8), (1, 8), (13, 1), (14, 1), (16, 1), (17, 4))
    ]
    ttnn.generic_op([q, k, output_q, output_k], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output_q, output_k
