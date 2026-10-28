# SPDX-License-Identifier: Apache-2.0
"""Compact four-way V-column B1 recurrence with unchanged FP32 compute.

Reader gathers compact head rows into the baseline kernel operands. Writer scatters
only disjoint valid head rows. State layout and compute arithmetic are unchanged.
"""

from pathlib import Path

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.fp32_recurrence import _KERNEL

_DIR = Path(__file__).with_name("kernels")


def fp32_recurrence(
    q,
    k,
    v,
    g,
    beta,
    *,
    initial_state,
    scale=None,
    output_final_state=True,
    compute_kernel_config=None,
    inplace_state=False,
    debug_operands=False,
    precomputed_decay=False,
):
    """Experimental compact B1/T1 dataflow; unchanged HiFi4/FP32 compute."""
    batch, length, qheads, key_dim = q.shape
    if batch != 1 or length != 1:
        raise ValueError("Compact recurrence is B1/T1 only")
    if not output_final_state or initial_state is None:
        raise ValueError("An explicit initial state and final state output are required")
    for x in (q, k, v, g, beta, initial_state):
        if x.layout != ttnn.TILE_LAYOUT or x.is_sharded():
            raise ValueError("Compact recurrence requires interleaved tiled inputs")
    if list(g.shape) != [1, 1, 12] or list(beta.shape) != [1, 1, 12]:
        raise ValueError("Compact scalar gates must have shape [1,1,12]")
    if list(k.shape) != list(q.shape) or list(v.shape) != [1, 1, 12, 128]:
        raise ValueError("Compact q/k/v head shapes do not match")
    if list(initial_state.shape) != [1, 12, 128, 128]:
        raise ValueError("Compact state must have shape [1,12,128,128]")
    value_heads, value_dim = v.shape[2], v.shape[3]
    if key_dim != 128 or value_dim != 128 or value_heads != 12 or qheads not in (4, 12):
        raise ValueError("FP32 recurrence supports the Qwen3.8-27B head dimensions only")
    device = q.device()
    grid = device.compute_with_storage_grid_size()
    heads = batch * value_heads
    if heads > grid.x * grid.y:
        raise ValueError("Split recurrence batches to at most the available cores / 12")
    for x in (q, k, v, g, beta, initial_state):
        if x.dtype != ttnn.float32:
            raise ValueError("Recurrence inputs and state must be FP32")

    # Keep the public FP32 scale/exp operations exactly as in the baseline.
    q = ttnn.multiply(q, key_dim**-0.5 if scale is None else scale)
    decay = g if precomputed_decay else ttnn.exp(g)
    state = ttnn.reshape(initial_state, [heads, key_dim, value_dim])
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, value_heads, value_dim]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    # Each head reads its entire initial state once, before computing final state.
    # Final-only writer writes that same disjoint head after its computation.
    final = (
        state
        if inplace_state
        else ttnn.allocate_tensor_on_device(
            ttnn.Shape([heads, key_dim, value_dim]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
    )
    workers = heads * 4
    if workers > grid.x * grid.y:
        raise ValueError("Four-way V partition needs48 worker cores")
    coords = [ttnn.CoreCoord(h // grid.y, h % grid.y) for h in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    kt, vt, tile_bytes = 4, 1, 4096
    cb_sizes = [
        kt,
        kt,
        vt,
        2 if length > 1 else 1,
        2 if length > 1 else 1,
        kt * vt,
        2 * vt,
        2 * kt * vt,
        kt * vt,
        kt * vt,
        kt * vt,
        vt,
        vt,
        kt,
        kt * vt,
        kt * vt,
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=tiles * tile_bytes,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=tile_bytes)
            ],
        )
        for index, tiles in enumerate(cb_sizes)
    ]
    reader_ct = [kt, vt, qheads]
    # Compact dataflow has no ring-index accessor or new persistent state.
    for tensor in (q, k, v, decay, beta, state):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    writer_ct = [kt, vt]
    for tensor in (out, final):
        writer_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for worker, core in enumerate(coords):
        h, part = divmod(worker, 4)
        rr[core.x][core.y] = [
            h,
            part,
            q.buffer_address(),
            k.buffer_address(),
            v.buffer_address(),
            decay.buffer_address(),
            beta.buffer_address(),
            state.buffer_address(),
        ]
        wr[core.x][core.y] = [h, part, out.buffer_address(), final.buffer_address()]
        cr[core.x][core.y] = [length]
    if debug_operands:
        outputs = [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([workers, 1, width]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
            for width in (128, 128, 32, 1, 1)
        ]
        outputs.append(
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([workers, 128, 32]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
        )
        debug_ct = [kt, vt]
        for tensor in outputs:
            debug_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        debug_rt = ttnn.RuntimeArgs()
        for h, core in enumerate(coords):
            debug_rt[core.x][core.y] = [h] + [tensor.buffer_address() for tensor in outputs]
        descriptor = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=str(_DIR / "reader_parallel_compact_b1_recurrence.cpp"),
                    core_ranges=cores,
                    compile_time_args=reader_ct,
                    runtime_args=rr,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(_DIR / "writer_parallel_compact_b1_operand_debug.cpp"),
                    core_ranges=cores,
                    compile_time_args=debug_ct,
                    runtime_args=debug_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
            ],
            cbs=cbs,
            semaphores=[],
        )
        ttnn.generic_op([q, k, v, decay, beta, state] + outputs, descriptor)
        return outputs
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    # Only copy/SFPU consumers get direct-to-DST FP32. CB 14/15 are separate
    # packed copies of state/sd for matmul; q/k/u/kcol remain matrix operands.
    for index in (2, 3, 4, 5, 8, 9, 10, 11):
        modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(_DIR / "reader_parallel_compact_b1_recurrence.cpp"),
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=rr,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(_DIR / "writer_parallel_compact_b1_recurrence.cpp"),
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=wr,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL, core_ranges=cores, compile_time_args=[kt, vt, 0], runtime_args=cr, config=config
            ),
        ],
        cbs=cbs,
        semaphores=[],
    )
    ttnn.generic_op([q, k, v, decay, beta, state, out, final], descriptor)
    return out, ttnn.reshape(final, [batch, value_heads, key_dim, value_dim])
