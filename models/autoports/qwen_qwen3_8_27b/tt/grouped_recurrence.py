# SPDX-License-Identifier: Apache-2.0
"""Grouped TP4 FP32 recurrent decode update, with disjoint per-head state.

Derived from tt.multichip_decoder.fp32_recurrence. It retains all FP32 arithmetic
and assigns sequential heads to each core, so batch16/32 can avoid splitting.
"""
from pathlib import Path

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.fp32_recurrence import _KERNEL, _UPSTREAM
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import reshape


def fp32_recurrence(
    q,
    k,
    v,
    g,
    beta,
    *,
    initial_state,
    active_mask=None,
    scale=None,
    output_final_state=True,
    compute_kernel_config=None,
    inplace_state=False,
    heads_per_core=None,
):
    """[B,T,Hq,128], [B,T,12,128], log g/beta [B,T,12] -> output, FP32 state.

    Only final-state mode is supported. The compute policy is fixed HiFi4/FP32;
    the optional compute_kernel_config argument preserves call-site compatibility.
    """
    batch, length, qheads, key_dim = q.shape
    masked = active_mask is not None
    if masked:
        if length != 1 or not inplace_state:
            raise ValueError("Masked recurrence requires one decode step and in-place state")
        if (
            active_mask.dtype != ttnn.bfloat16
            or active_mask.layout != ttnn.TILE_LAYOUT
            or list(active_mask.shape) != [batch, 1, 1]
            or list(active_mask.padded_shape) != [batch, 32, 32]
            or active_mask.is_sharded()
        ):
            raise ValueError("Masked recurrence requires an interleaved BF16 [B,1,1] tile mask")
    value_heads, value_dim = v.shape[2], v.shape[3]
    if key_dim != 128 or value_dim != 128 or value_heads != 12 or qheads not in (4, 12):
        raise ValueError("FP32 recurrence supports the Qwen3.8-27B head dimensions only")
    if not output_final_state or initial_state is None:
        raise ValueError("An explicit initial state and final state output are required")
    device = q.device()
    grid = device.compute_with_storage_grid_size()
    heads = batch * value_heads
    if heads_per_core is None:
        heads_per_core = (heads + grid.x * grid.y - 1) // (grid.x * grid.y)
    if type(heads_per_core) is not int or not 1 <= heads_per_core <= 4:
        raise ValueError("Grouped recurrence requires one through four heads per core")
    if heads_per_core > 1 and length != 1:
        raise ValueError("Grouped recurrence is a decode-only candidate")
    core_count = (heads + heads_per_core - 1) // heads_per_core
    if core_count > grid.x * grid.y:
        raise ValueError("Split recurrence batches to at most the available cores / 12")
    for x in (q, k, v, g, beta, initial_state):
        if x.dtype != ttnn.float32:
            raise ValueError("Recurrence inputs and state must be FP32")

    def split(x, h, width):
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), [batch * h, length, width])
        if h != value_heads:
            x = ttnn.repeat_interleave(x, value_heads // h, dim=0)
        return reshape(x, [heads * length, 1, width]) if length > 1 else x

    q = split(ttnn.multiply(q, key_dim**-0.5 if scale is None else scale), qheads, key_dim)
    k = split(k, qheads, key_dim)
    v = split(v, value_heads, value_dim)

    def split_scalar(x):
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1)), [heads, length])
        return reshape(x, [heads * length, 1, 1])

    # Exponentiation remains the public FP32 TTNN op, as in the upstream wrapper.
    decay = split_scalar(ttnn.exp(g))
    beta = split_scalar(beta)
    state = ttnn.reshape(initial_state, [heads, key_dim, value_dim])
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([heads * length, 1, value_dim]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
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
    coords = [ttnn.CoreCoord(h // grid.y, h % grid.y) for h in range(core_count)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    kt, vt, tile_bytes = 4, 4, 4096
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
    if masked:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=32,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.uint32, page_size=32)],
            )
        )
    reader_ct = [kt, vt, 0]
    # Ring indexing is disabled. A valid dummy accessor keeps the unused template
    # argument well-formed without any scratch buffer or new persistent state.
    for tensor in (q, k, v, decay, beta, state, state):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    writer_ct = [kt, vt, 0]
    for tensor in (out, final):
        writer_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    if masked:
        writer_ct.extend(ttnn.TensorAccessorArgs(active_mask).get_compile_time_args())
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core_index, core in enumerate(coords):
        h = core_index * heads_per_core
        count = min(heads_per_core, heads - h)
        reader_args = [
            h,
            length,
            q.buffer_address(),
            k.buffer_address(),
            v.buffer_address(),
            decay.buffer_address(),
            beta.buffer_address(),
            state.buffer_address(),
            0,
            0,
            heads,
        ]
        writer_args = [h, length, out.buffer_address(), final.buffer_address(), heads]
        if masked:
            writer_args.extend([count, active_mask.buffer_address()])
        compute_args = [length]
        if heads_per_core > 1:
            reader_args.append(count)
            if not masked:
                writer_args.append(count)
            compute_args.append(count)
        rr[core.x][core.y] = reader_args
        wr[core.x][core.y] = writer_args
        cr[core.x][core.y] = compute_args
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    # Only copy/SFPU consumers get direct-to-DST FP32. CB 14/15 are separate
    # packed copies of state/sd for matmul; q/k/u/kcol remain matrix operands.
    for index in (2, 3, 4, 5, 8, 9, 10, 11):
        modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    kernel_dir = Path(__file__).with_name("kernels")
    reader_source = (
        str(kernel_dir / "reader_grouped_recurrence.cpp")
        if heads_per_core > 1
        else _UPSTREAM + "reader_fused_recurrent_gated_delta_rule.cpp"
    )
    writer_source = (
        str(kernel_dir / "writer_masked_grouped_recurrence.cpp")
        if masked
        else (
            str(kernel_dir / "writer_grouped_recurrence.cpp")
            if heads_per_core > 1
            else _UPSTREAM + "writer_fused_recurrent_gated_delta_rule.cpp"
        )
    )
    compute_source = str(kernel_dir / "compute_grouped_recurrence.cpp") if heads_per_core > 1 else _KERNEL
    descriptor = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=reader_source,
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=rr,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=writer_source,
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=wr,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=compute_source,
                core_ranges=cores,
                compile_time_args=[kt, vt, 0],
                runtime_args=cr,
                config=config,
            ),
        ],
        cbs=cbs,
        semaphores=[],
    )
    inputs = [q, k, v, decay, beta, state]
    if masked:
        inputs.append(active_mask)
    # generic_op declares its last tensor as the output; keep final state last.
    inputs.extend([out, final])
    ttnn.generic_op(inputs, descriptor)
    out = ttnn.permute(ttnn.reshape(out, [batch, value_heads, length, value_dim]), (0, 2, 1, 3))
    return out, ttnn.reshape(final, [batch, value_heads, key_dim, value_dim])
