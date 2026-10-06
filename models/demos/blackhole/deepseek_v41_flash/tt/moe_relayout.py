# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-side copy between the unified-prefill and the moe_compute (decode ring) layout of the bfp8 routed experts of one layer.
Pure JIT (ttnn.generic_op + tt/kernels/moe_relayout.cpp), no C++ rebuild. See the kernel header for the exact page mappings."""
import ttnn

KERNEL = "models/demos/blackhole/deepseek_v41_flash/tt/kernels/moe_relayout.cpp"
TB = 1088  # bfp8 tile bytes
NB = 12  # units per batch (kernel constant)
E = 12
UNITS = {0: 8 * E * 5 * 161, 1: 8 * E * 5 * 77}


def decode_shapes():
    return {0: (8, 1, E, 5, 161 * 32, 128), 1: (8, 1, E, 5, 77 * 32, 128)}


def decode_mem_configs(md):
    w = ttnn.experimental.get_weight_mem_configs(
        md, num_layers=1, experts_per_device=E, hidden_size=5120, intermediate_size=2304, has_bias=False
    )
    return {0: w.w0_w1, 1: w.w2}


def alloc_decode(md):
    mc = decode_mem_configs(md)
    sh = decode_shapes()
    return [
        ttnn.empty(sh[m], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=md, memory_config=mc[m]) for m in (0, 1)
    ]


def _program(md, mode, direction, srcs, dec):
    """srcs: [gate_list, up_list] (mode 0) or [down_list] (mode 1)."""
    grid = md.compute_with_storage_grid_size()
    cores = [(x, y) for y in range(grid.y) for x in range(grid.x)]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    total = UNITS[mode]
    per = -(-total // len(cores))
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        u0 = min(i * per, total)
        rt[x][y] = [u0, min(per, total - u0)]
    ct = [mode, direction, TB, E]
    common = [dec.buffer_address()]
    for lst in srcs:
        ct += ttnn.TensorAccessorArgs(lst[0]).get_compile_time_args()
        common += [t.buffer_address() for t in lst]
    ct += ttnn.TensorAccessorArgs(dec).get_compile_time_args()
    cb = ttnn.CBDescriptor(
        total_size=NB * 4 * TB,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat8_b, page_size=TB)],
    )
    k = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    p = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])
    p.custom_program_hash = (0x7E1A << 8) | (mode << 4) | direction
    return p


def relayout(md, direction, gate, up, down, dec_w0_w1, dec_w2):
    """direction 0: unified (gate/up/down lists of E per-device tensors) -> decode tensors; 1: decode -> unified lists."""
    ttnn.generic_op([gate[0], up[0], dec_w0_w1], _program(md, 0, direction, [gate, up], dec_w0_w1))
    ttnn.generic_op([down[0], dec_w2], _program(md, 1, direction, [down], dec_w2))
