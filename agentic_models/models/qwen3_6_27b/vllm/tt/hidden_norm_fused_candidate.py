# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in exact-tree hidden norm: local fold/scatter/inverse, unchanged output.

No device imports at module load. Thirty-two equal-layout workers own the fold;
workers [0,B) also own inverse rows. Dispatch resets program-local semaphores on
every enqueue/replay. Kernels never reset them. All remote atomics drain before
kernel exit. The public output and optional inverse buffers are independently
rebound generic-op inputs, and the final output program is the frozen native one.
"""

import struct

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const uint32_t block=get_arg_val<uint32_t>(1), batch=get_arg_val<uint32_t>(2);
    Noc noc; CircularBuffer data(0),dense(4);
    // All 32 cores have identical local CB allocations and reserve this slot.
    // READY is sent only after initialization and source publication.
    dense.reserve_back(1); data.reserve_back(5);
    for(uint32_t chunk=0;chunk<5;chunk++)
        noc.async_read(x,data,2048,{.page_id=chunk*32+block},{.offset_bytes=chunk*2048});
    noc.async_read_barrier(); data.push_back(5);
    const uint32_t ready=get_semaphore(0),go=get_semaphore(1),done=get_semaphore(2);
    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(3),get_arg_val<uint32_t>(4),ready),1);
    noc_async_atomic_barrier();
    if(block==0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready),32);
        for(uint32_t peer=0;peer<32;peer++)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(3+2*peer),
                get_arg_val<uint32_t>(4+2*peer),go),1);
        noc_async_atomic_barrier();
    }
    if(block<batch) {
        // Every sender drains its data writes before publishing DONE.
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done),32);
        dense.push_back(1);
    }
}
"""

WRITER_PREFIX = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ia=TensorAccessorArgs<0>();
    const auto inverse=TensorAccessor(ia,get_arg_val<uint32_t>(0),4096);
    const uint32_t block=get_arg_val<uint32_t>(1),batch=get_arg_val<uint32_t>(2);
    Noc noc; UnicastEndpoint remote; CircularBuffer folded(3),dense(4);
    const uint32_t dense_addr=dense.get_write_ptr(),go=get_semaphore(1),done=get_semaphore(2);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go),1);
    folded.wait_front(1);
    for(uint32_t user=0;user<batch;user++) for(uint32_t face=0;face<2;face++) {
        const uint32_t source=((user/16)*512+face*256+(user%16)*16)*4;
        const uint32_t target=((block/16)*512+face*256+(block%16)*16)*4;
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(folded),remote,64,
            {.offset_bytes=source},{.noc_x=get_arg_val<uint32_t>(3+2*user),
            .noc_y=get_arg_val<uint32_t>(4+2*user),.addr=dense_addr+target});
    }
    noc.async_write_barrier();
    for(uint32_t user=0;user<batch;user++)
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(3+2*user),
            get_arg_val<uint32_t>(4+2*user),done),1);
    noc_async_atomic_barrier(); folded.pop_front(1);
    if(block>=batch) return;
    const uint32_t user=block;
"""


def admission_sources(reader, writer, *, admission="software", defer_receiver_reserve=False):
    """Select startup admission without changing arithmetic, data or DONE order.

    Software/default with no delay returns byte-identical old kernel strings.
    Dispatch admission relies on the config-write fence/prior-worker completion
    before GO and metadata-only local CB initialization. Dense CB4 is one-shot,
    identically addressed on all32 workers, and is never cleared by its receiver.
    A correctness-only control pauses reader0 until31 peer DONE credits arrive
    before reserving dense/raw. Their data writes must already be complete.
    """
    assert admission in ("software", "dispatch")
    assert isinstance(defer_receiver_reserve, bool)
    assert not defer_receiver_reserve or admission == "dispatch"

    def replace_once(source, old, new):
        assert source.count(old) == 1, (old, source.count(old))
        return source.replace(old, new, 1)

    if admission == "dispatch":
        reader = replace_once(
            reader,
            """    // All 32 cores have identical local CB allocations and reserve this slot.
    // READY is sent only after initialization and source publication.
""",
            """    // Dispatch config-write/GO admission owns the one-shot dense allocation.
    // CB interface initialization/reservation never clears its receive payload.
""",
        )
        reader = replace_once(
            reader,
            """    const uint32_t ready=get_semaphore(0),go=get_semaphore(1),done=get_semaphore(2);
    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(3),get_arg_val<uint32_t>(4),ready),1);
    noc_async_atomic_barrier();
    if(block==0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready),32);
        for(uint32_t peer=0;peer<32;peer++)
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(3+2*peer),
                get_arg_val<uint32_t>(4+2*peer),go),1);
        noc_async_atomic_barrier();
    }
""",
            "    const uint32_t done=get_semaphore(2);\n",
        )
        writer = replace_once(
            writer,
            """    const uint32_t dense_addr=dense.get_write_ptr(),go=get_semaphore(1),done=get_semaphore(2);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go),1);
""",
            "    const uint32_t dense_addr=dense.get_write_ptr(),done=get_semaphore(2);\n",
        )
    if defer_receiver_reserve:
        reader = replace_once(
            reader,
            "    dense.reserve_back(1); data.reserve_back(5);",
            """    // Correctness-only skew: every peer has acknowledged writes into this
    // dense slot before reader0 reserves it. Own writer cannot publish DONE
    // until this reader provides raw input, so the count cannot exceed31 yet.
    if(block==0)
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2)),31);
    dense.reserve_back(1); data.reserve_back(5);""",
        )
    return reader, writer


def fused_sources(fold_compute, inverse_compute, inverse_writer, *, admission="software", defer_receiver_reserve=False):
    """Reuse the exact frozen arithmetic bodies; fail closed on source drift."""
    fold_start = "    CircularBuffer x(0),out(3);"
    inverse_start = "    CircularBuffer dense(0),inverse(2);"
    write_start = "    Noc noc; CircularBuffer data(2),zero(4);"
    assert (
        fold_compute.count(fold_start) == inverse_compute.count(inverse_start) == inverse_writer.count(write_start) == 1
    )
    fold_body = fold_compute.split(fold_start, 1)[1].rsplit("}", 1)[0]
    inverse_body = inverse_compute.split(inverse_start, 1)[1].rsplit("}", 1)[0]
    inverse_body = inverse_body.replace("copy_tile(0,0,0)", "copy_tile(4,0,0)").replace(
        "pack_tile(0,2)", "pack_tile(0,5)"
    )
    headers = fold_compute.split("void kernel_main()", 1)[0] + inverse_compute.split("void kernel_main()", 1)[0]
    headers += '#include "api/compute/reconfig_data_format.h"\n'
    compute = (
        headers
        + """
void kernel_main() {
    constexpr uint32_t EPS_BITS=get_compile_time_arg_val(0), RECIPROCAL_BITS=get_compile_time_arg_val(1);
    const uint32_t block=get_arg_val<uint32_t>(0),batch=get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(0,0,3); copy_init(0);
    CircularBuffer x(0),out(3);"""
        + fold_body
        + """
    if(block>=batch) return;
    reconfig_data_format_srca(4); copy_init(4);
    CircularBuffer dense(4),inverse(5);"""
        + inverse_body
        + "}\n"
    )
    writer_body = inverse_writer.split(write_start, 1)[1].rsplit("}", 1)[0]
    writer = WRITER_PREFIX + "    CircularBuffer data(5),zero(6);" + writer_body + "}\n"
    reader, writer = admission_sources(
        READER, writer, admission=admission, defer_receiver_reserve=defer_receiver_reserve
    )
    return reader, writer, compute


def hidden_norm_fused(
    x, gamma, *, epsilon=1e-6, output=None, inverse=None, admission="software", defer_receiver_reserve=False
):
    """BF16[1,B,5120] and FP32 gamma[1,1,5120], B1..32; two programs.

    Inputs/output remain DRAM interleaved TILE. Optional FP32 inverse[1,32,32]
    is L1 interleaved TILE. No material folded tensor or inter-chip collective.
    Software admission remains the default/control. Dispatch admission omits
    READY/GO, retaining DONE and all data/atomic completion barriers. Enabled
    defer_receiver_reserve is a correctness-only skew, never a timing path.
    """
    import ttnn
    from .optimized_decoder import (
        _norm_zero_meanFOLD_COMPUTE,
        _norm_zero_meanINVERSE_COMPUTE,
        _norm_zero_meanINVERSE_WRITER,
        _norm_zero_meanOUTPUT_READER,
        _norm_zero_meanOUTPUT_WRITER,
        _norm_zero_meanOUTPUT_COMPUTE,
    )

    assert admission in ("software", "dispatch")
    assert isinstance(defer_receiver_reserve, bool)
    assert not defer_receiver_reserve or admission == "dispatch"
    assert len(x.shape) == 3 and x.shape[0] == 1 and 1 <= x.shape[1] <= 32 and x.shape[2] == 5120
    assert x.dtype == ttnn.bfloat16 and tuple(x.padded_shape) == (1, 32, 5120)
    assert tuple(gamma.shape) == (1, 1, 5120) and gamma.dtype == ttnn.float32 and epsilon > 0
    device = x.device()
    batch = x.shape[1]
    if output is None:
        output = ttnn.empty(
            tuple(x.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if inverse is None:
        inverse = ttnn.empty(
            (1, 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
    assert tuple(output.shape) == tuple(x.shape) and output.dtype == ttnn.bfloat16
    assert tuple(inverse.shape) == (1, 32, 32) and inverse.dtype == ttnn.float32
    for t in (x, gamma, output, inverse):
        assert t.layout == ttnn.TILE_LAYOUT and t.device() == device
    for t in (x, gamma, output):
        assert t.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert inverse.memory_config() == ttnn.L1_MEMORY_CONFIG
    assert len({t.buffer_address() for t in (x, gamma, output, inverse)}) == 4
    grid = device.compute_with_storage_grid_size()
    assert grid.x * grid.y >= 32
    output_cores = min(80, grid.x * grid.y)
    output_tiles = (160 + output_cores - 1) // output_cores
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(output_cores)]
    physical = [device.worker_core_from_logical_core(c) for c in coords[:32]]
    physical_args = [v for c in physical for v in (c.x, c.y)]
    bits = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]

    def accessor_args(*tensors):
        result = []
        for t in tensors:
            result.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        return result

    def launch(sources, runtime, ct, sizes, active, tensors, synchronized=False):
        selected = coords[:active]
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in selected])
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for index, count, dtype in sizes:
            if dtype == ttnn.float32:
                modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
        kernels = []
        for source, args, compile_args, kernel_config in zip(
            sources, runtime, ct, [ttnn.ReaderConfigDescriptor(), ttnn.WriterConfigDescriptor(), config]
        ):
            rt = ttnn.RuntimeArgs()
            for index, c in enumerate(selected):
                rt[c.x][c.y] = args(index)
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=source,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ranges,
                    compile_time_args=compile_args,
                    runtime_args=rt,
                    config=kernel_config,
                )
            )
        cbs = []
        for index, count, dtype in sizes:
            page = 2048 if dtype == ttnn.bfloat16 else 4096
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=count * page,
                    core_ranges=ranges,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
                )
            )
        # Keep descriptor IDs/config footprint identical across admission controls;
        # dispatch kernels never touch the two unused READY/GO semaphore words.
        sems = (
            [ttnn.SemaphoreDescriptor(id=i, core_ranges=ranges, initial_value=0) for i in range(3)]
            if synchronized
            else []
        )
        ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))

    launch(
        fused_sources(
            _norm_zero_meanFOLD_COMPUTE,
            _norm_zero_meanINVERSE_COMPUTE,
            _norm_zero_meanINVERSE_WRITER,
            admission=admission,
            defer_receiver_reserve=defer_receiver_reserve,
        ),
        [
            lambda c: [x.buffer_address(), c, batch, *physical_args],
            lambda c: [inverse.buffer_address(), c, batch, *physical_args],
            lambda c: [c, batch],
        ],
        [accessor_args(x), accessor_args(inverse), [bits(epsilon), bits(1 / 5120)]],
        [(0, 5, ttnn.bfloat16), (3, 1, ttnn.float32), (4, 1, ttnn.float32), (5, 1, ttnn.float32), (6, 1, ttnn.float32)],
        32,
        [x, inverse],
        synchronized=True,
    )

    def output_count(c):
        return 160 // output_cores + (c < 160 % output_cores)

    def output_start(c):
        return c * (160 // output_cores) + min(c, 160 % output_cores)

    launch(
        [_norm_zero_meanOUTPUT_READER, _norm_zero_meanOUTPUT_WRITER, _norm_zero_meanOUTPUT_COMPUTE],
        [
            lambda c: [
                x.buffer_address(),
                gamma.buffer_address(),
                inverse.buffer_address(),
                output_start(c),
                output_count(c),
            ],
            lambda c: [output.buffer_address(), output_start(c), batch, output_count(c)],
            lambda c: [output_count(c)],
        ],
        [accessor_args(x, gamma, inverse), accessor_args(output), []],
        [
            (0, output_tiles, ttnn.bfloat16),
            (1, output_tiles, ttnn.float32),
            (2, 1, ttnn.float32),
            (3, output_tiles, ttnn.bfloat16),
        ],
        output_cores,
        [x, gamma, inverse, output],
    )
    return output
