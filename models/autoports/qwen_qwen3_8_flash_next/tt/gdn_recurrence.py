# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-local FP32-preserving GDN recurrence for target16/48heads, K=V=128.

Public contract matches the native fused recurrent op used by GatedDeltaNet:
q/k [1,T,16,128] are already L2-normalized, v [1,T,48,128], g/beta
[1,T,48], initial_state [1,48,128,128] FP32. g is log decay; beta is
already sigmoid. Returns FP32 output [1,T,48,128] and final state. Initial
state is mandatory, so no uncached creation/fallback occurs during traces.
All transformations and compute remain on device. State copy-back is caller
owned; the input state is read-only. Every call consumes actual logical T.

The functional default retains48-core/head parallelism and loads every
arithmetic operand directly into FP32 DST. State decay, delta subtraction,
dot products and rank-one updates use SFPU arithmetic without TF32 operand
truncation. This is a numerical repair, not a new model stage. The earlier
FPU-dot candidate remains selectable with precise_dot_products=False solely
for diagnostics; native recurrence is also separately available.
"""

from pathlib import Path

_KERNEL_ROOT = Path(__file__).parent / "kernels"


def fp32_recurrent_gated_delta_rule(
    q, k, v, g, beta, *, initial_state, output_final_state=True, compute_kernel_config=None, precise_dot_products=True
):
    import ttnn as t

    count = q.shape[1]
    if tuple(q.shape) != (1, count, 16, 128) or tuple(k.shape) != tuple(q.shape):
        raise ValueError("Expected target q/k [1,T,16,128]")
    if tuple(v.shape) != (1, count, 48, 128) or tuple(g.shape) != (1, count, 48) or tuple(beta.shape) != tuple(g.shape):
        raise ValueError("Expected target v [1,T,48,128], g/beta [1,T,48]")
    if tuple(initial_state.shape) != (1, 48, 128, 128) or initial_state.dtype != t.float32:
        raise ValueError("A persistent FP32 [1,48,128,128] initial state is mandatory")
    if count < 1:
        raise ValueError("At least one actual token is required")
    device = q.device()

    def fp32(x):
        return x if x.dtype == t.float32 else t.typecast(x, t.float32)

    def reshape_rm(x, shape):
        return t.to_layout(t.reshape(t.to_layout(x, t.ROW_MAJOR_LAYOUT), shape), t.TILE_LAYOUT)

    def heads(x, number):
        return t.reshape(t.permute(fp32(x), (0, 2, 1, 3)), (number, count, 128))

    qh = heads(t.multiply(fp32(q), 128**-0.5), 16)
    kh = heads(k, 16)
    qh = t.repeat_interleave(qh, repeats=3, dim=0)
    kh = t.repeat_interleave(kh, repeats=3, dim=0)
    vh = heads(v, 48)
    if count > 1:
        qh, kh, vh = (reshape_rm(x, (48 * count, 1, 128)) for x in (qh, kh, vh))
    decay = t.exp(t.reshape(t.permute(fp32(g), (0, 2, 1)), (48, count)))
    betah = t.reshape(t.permute(fp32(beta), (0, 2, 1)), (48, count))
    decay = reshape_rm(decay, (48 * count, 1, 1))
    betah = reshape_rm(betah, (48 * count, 1, 1))
    state = t.reshape(initial_state, (48, 128, 128))
    output = t.empty(
        (48 * count, 1, 128), dtype=t.float32, layout=t.TILE_LAYOUT, device=device, memory_config=t.DRAM_MEMORY_CONFIG
    )
    final = t.empty(
        (48, 128, 128), dtype=t.float32, layout=t.TILE_LAYOUT, device=device, memory_config=t.DRAM_MEMORY_CONFIG
    )
    tensors = [qh, kh, vh, decay, betah, state, output, final]
    for tensor in tensors:
        if (
            tensor.layout != t.TILE_LAYOUT
            or tensor.dtype != t.float32
            or tensor.memory_config() != t.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("Recurrence requires interleaved DRAM FP32 TILE operands")
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y < 48:
        raise ValueError("Target recurrence requires48 compute cores")
    cores = [t.CoreCoord(h // grid.y, h % grid.y) for h in range(48)]
    ranges = t.CoreRangeSet([t.CoreRange(core, core) for core in cores])
    reader_rt, writer_rt, compute_rt = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
    addresses = [value.buffer_address() for value in tensors]
    for h, core in enumerate(cores):
        reader_rt[core.x][core.y] = [h, count, *addresses[:6], 0, 0, 48]
        writer_rt[core.x][core.y] = [h, count, *addresses[6:], 48]
        compute_rt[core.x][core.y] = [count]
    reader_ct = [4, 4, 0]
    for value in tensors[:6]:
        reader_ct.extend(t.TensorAccessorArgs(value).get_compile_time_args())
    writer_ct = [4, 4, 0]
    for value in tensors[6:]:
        writer_ct.extend(t.TensorAccessorArgs(value).get_compile_time_args())
    compute_config = t.ComputeConfigDescriptor(
        math_fidelity=t.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=False, math_approx_mode=False
    )
    modes = [t.UnpackToDestMode.Default] * 64
    # Direct copies in decay, residual and rank1 seeding must retain all FP32
    # mantissa bits. q/k/matmul operands remain native unless explicitly copied.
    # State CBs8/9/10 are also FPU operands: direct-DST modes cannot be
    # applied to those indices. Separate aliases share SRAM but expose an
    # exact-copy view; their read pointers are synchronized in the kernel.
    exact_aliases = {8: 17, 9: 18, 10: 19}
    for cb in (2, 3, 4, 5, 11, *exact_aliases.values()):
        modes[cb] = t.UnpackToDestMode.UnpackToDestFp32
    if precise_dot_products:
        # The SFPU variant has no FPU source consumers: all arithmetic inputs
        # load FP32 directly into DST, including q/k, state and delta.
        exact_aliases = {}
        for cb in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12):
            modes[cb] = t.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = modes
    specs = [
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 2 if count > 1 else 1),
        (4, 2 if count > 1 else 1),
        (5, 16),
        (6, 8),
        (7, 32),
        (8, 16),
        (9, 16),
        (10, 16),
        (11, 4),
        (12, 4),
        (13, 4),
    ]
    cbs = [
        t.CBDescriptor(
            total_size=tiles * 4096,
            core_ranges=ranges,
            format_descriptors=[
                t.CBFormatDescriptor(buffer_index=view, data_format=t.float32, page_size=4096)
                for view in ([index, exact_aliases[index]] if index in exact_aliases else [index])
            ],
        )
        for index, tiles in specs
    ]
    descriptors = []
    for filename, ct, rt, config in (
        ("gdn_reader.cpp", reader_ct, reader_rt, t.ReaderConfigDescriptor()),
        ("gdn_writer.cpp", writer_ct, writer_rt, t.WriterConfigDescriptor()),
        ("gdn_compute_sfpu.cpp" if precise_dot_products else "gdn_compute.cpp", [4, 4, 0], compute_rt, compute_config),
    ):
        descriptors.append(
            t.KernelDescriptor(
                kernel_source=str(_KERNEL_ROOT / filename),
                source_type=t.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )
        )
    program = t.ProgramDescriptor(kernels=descriptors, cbs=cbs, semaphores=[])
    t.generic_op(tensors, program)
    result = t.to_layout(
        t.permute(t.reshape(t.to_layout(output, t.ROW_MAJOR_LAYOUT), (1, 48, count, 128)), (0, 2, 1, 3)), t.TILE_LAYOUT
    )
    return result, t.reshape(final, (1, 48, 128, 128)) if output_final_state else None
