# SPDX-License-Identifier: Apache-2.0
"""Decoder-local recurrence with FP32 state copy, decay and residual arithmetic.

This adapts the checkout's fused recurrence descriptor, retaining its dataflow
kernels and tensor layout, adding two per-core matrix staging CBs. It changes compute precision: FP32 unpack-to-DST
for copies and SFPU arithmetic for state decay and the value residual. Matmul
still uses HiFi4 with FP32 accumulation. Callers normalize q/k and split B so
B * 48 does not exceed the physical core count. No host tensor conversion occurs.
"""
from pathlib import Path

import ttnn


_KERNEL = str(Path(__file__).with_name("kernels") / "fp32_recurrence.cpp")
_UPSTREAM = "ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/dataflow/"


def _reshape_rm(x, shape):
    return ttnn.to_layout(ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), shape), ttnn.TILE_LAYOUT)


def fp32_recurrence(q, k, v, g, beta, *, initial_state, scale=None, output_final_state=True,
                    compute_kernel_config=None):
    """[B,T,Hq,128], [B,T,48,128], log g/beta [B,T,48] -> output, FP32 state.

    Only final-state mode is supported. The compute policy is fixed HiFi4/FP32;
    the optional compute_kernel_config argument preserves call-site compatibility.
    """
    batch, length, qheads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    if key_dim != 128 or value_dim != 128 or value_heads != 48 or qheads not in (16, 48):
        raise ValueError("FP32 recurrence supports the Qwen3.8-27B head dimensions only")
    if not output_final_state or initial_state is None:
        raise ValueError("An explicit initial state and final state output are required")
    device = q.device()
    grid = device.compute_with_storage_grid_size()
    heads = batch * value_heads
    if heads > grid.x * grid.y:
        raise ValueError("Split recurrence batches to at most the available cores / 48")
    for x in (q, k, v, g, beta, initial_state):
        if x.dtype != ttnn.float32:
            raise ValueError("Recurrence inputs and state must be FP32")

    def split(x, h, width):
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), [batch * h, length, width])
        if h != value_heads:
            x = ttnn.repeat_interleave(x, value_heads // h, dim=0)
        return _reshape_rm(x, [heads * length, 1, width]) if length > 1 else x

    q = split(ttnn.multiply(q, key_dim ** -0.5 if scale is None else scale), qheads, key_dim)
    k = split(k, qheads, key_dim)
    v = split(v, value_heads, value_dim)

    def split_scalar(x):
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1)), [heads, length])
        return _reshape_rm(x, [heads * length, 1, 1])

    # Exponentiation remains the public FP32 TTNN op, as in the upstream wrapper.
    decay = split_scalar(ttnn.exp(g))
    beta = split_scalar(beta)
    state = ttnn.reshape(initial_state, [heads, key_dim, value_dim])
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([heads * length, 1, value_dim]), ttnn.float32,
                                        ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    final = ttnn.allocate_tensor_on_device(ttnn.Shape([heads, key_dim, value_dim]), ttnn.float32,
                                          ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    coords = [ttnn.CoreCoord(h // grid.y, h % grid.y) for h in range(heads)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    kt, vt, tile_bytes = 4, 4, 4096
    cb_sizes = [kt, kt, vt, 2 if length > 1 else 1, 2 if length > 1 else 1,
                kt * vt, 2 * vt, 2 * kt * vt, kt * vt, kt * vt, kt * vt, vt, vt, kt, kt * vt, kt * vt]
    cbs = [ttnn.CBDescriptor(total_size=tiles * tile_bytes, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index,
                                     data_format=ttnn.float32, page_size=tile_bytes)])
           for index, tiles in enumerate(cb_sizes)]
    reader_ct = [kt, vt, 0]
    # Ring indexing is disabled. A valid dummy accessor keeps the unused template
    # argument well-formed without any scratch buffer or new persistent state.
    for tensor in (q, k, v, decay, beta, state, state):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    writer_ct = [kt, vt, 0]
    for tensor in (out, final):
        writer_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for h, core in enumerate(coords):
        rr[core.x][core.y] = [h, length, q.buffer_address(), k.buffer_address(), v.buffer_address(),
                              decay.buffer_address(), beta.buffer_address(), state.buffer_address(), 0, 0, heads]
        wr[core.x][core.y] = [h, length, out.buffer_address(), final.buffer_address(), heads]
        cr[core.x][core.y] = [length]
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    # Only copy/SFPU consumers get direct-to-DST FP32. CB 14/15 are separate
    # packed copies of state/sd for matmul; q/k/u/kcol remain matrix operands.
    for index in (2, 3, 4, 5, 8, 9, 10, 11):
        modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    descriptor = ttnn.ProgramDescriptor(kernels=[
        ttnn.KernelDescriptor(kernel_source=_UPSTREAM + "reader_fused_recurrent_gated_delta_rule.cpp",
                             core_ranges=cores, compile_time_args=reader_ct, runtime_args=rr,
                             config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=_UPSTREAM + "writer_fused_recurrent_gated_delta_rule.cpp",
                             core_ranges=cores, compile_time_args=writer_ct, runtime_args=wr,
                             config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=_KERNEL, core_ranges=cores, compile_time_args=[kt, vt, 0],
                             runtime_args=cr, config=config),
    ], cbs=cbs, semaphores=[])
    ttnn.generic_op([q, k, v, decay, beta, state, out, final], descriptor)
    out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.reshape(out, [batch, value_heads, length, value_dim])
    out = ttnn.to_layout(ttnn.permute(out, (0, 2, 1, 3)), ttnn.TILE_LAYOUT)
    return out, ttnn.reshape(final, [batch, value_heads, key_dim, value_dim])
