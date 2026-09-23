# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer weight staging on normalization workers, inside the resident loop."""

from pathlib import Path
import ttnn
from .mlp import _grid


def append_prefetch(body, program):
    """Reserve static helper L1 and attach reads to the two norm-reader phases.

    Gate/up staging precedes the MLP norm's activation wait. Down staging
    follows the attention norm reader's publication of its reduction scalar;
    its writer can publish the QKV input while staging proceeds. The global
    layer boundaries protect reuse of both staging allocations and mailboxes.
    """
    projections = [body.mesh.worker_core_from_logical_core(c) for c in body.projection_cores]
    coordinates = [v for c in projections for v in (c.x, c.y)]
    kernels, cbs = list(program.kernels), list(program.cbs)
    for role, blocks, cores, weight_role, block_bytes in (
        (0, body.tuning.prefetch_gu_blocks, body.norm_cores, "gate_up", 8 * 28 * 576),
        (1, body.tuning.prefetch_down_blocks, body.preparation.norm_cores, "down", 7 * 16 * 1088),
    ):
        if not blocks:
            continue
        grid = _grid(cores)
        storage_bytes = ((blocks * block_bytes + 4095) // 4096) * 4096
        cbs.append(ttnn.CBDescriptor(
            total_size=storage_bytes, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=32, data_format=ttnn.uint32, page_size=4096)],
        ))
        matches = 0
        for kernel in kernels:
            if (Path(kernel.kernel_source).name != "norm_dataflow.cpp"
                    or kernel.core_ranges != grid or "READER" not in dict(kernel.defines)):
                continue
            matches += 1
            rt = kernel.runtime_args
            offsets = set()
            for core in cores:
                args = list(rt[core.x][core.y])
                offsets.add(len(args))
                rt[core.x][core.y] = [*args, *coordinates]
            if len(offsets) != 1:
                raise ValueError("Unexpected norm prefetch runtime layout")
            kernel.runtime_args = rt
            kernel.defines = [*kernel.defines, ("PREFETCH_ROLE", str(role)),
                              ("PREFETCH_BLOCKS", str(blocks)),
                              ("PREFETCH_RT_OFFSET", str(offsets.pop())),
                              ("PREFETCH_CT_OFFSET", str(len(kernel.compile_time_args)))]
            kernel.compile_time_args = [*kernel.compile_time_args,
                *ttnn.TensorAccessorArgs(body.layers[0].decode_weights[weight_role]).get_compile_time_args()]
        if matches != 1:
            raise ValueError(f"Expected one norm reader for prefetch role {role}, got {matches}")
    program.kernels, program.cbs = kernels, cbs
    return program
