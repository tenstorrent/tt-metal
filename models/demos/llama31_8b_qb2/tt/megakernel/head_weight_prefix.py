# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trigger the head's own weight-ring refill during the final layer."""
from pathlib import Path
import ttnn


def append_trigger(loop, program):
    physical = [loop.body.mesh.worker_core_from_logical_core(c) for c in loop.head.cores]
    extra = [ttnn.get_global_semaphore_address(loop.head.weight_ready),
             loop.first + loop.count - 1, *[v for c in physical for v in (c.x, c.y)]]
    found = False
    kernels = list(program.kernels)
    for kernel in kernels:
        if Path(kernel.kernel_source).name != "qkv.cpp" or "WRITER" not in dict(kernel.defines):
            continue
        found = True
        rt, offsets = kernel.runtime_args, set()
        for core in loop.body.preparation.projection_cores:
            args = list(rt[core.x][core.y]); offsets.add(len(args))
            rt[core.x][core.y] = [*args, *extra]
        assert len(offsets) == 1
        kernel.runtime_args = rt
        kernel.defines = [*kernel.defines, ("HEAD_WEIGHT_TRIGGER_RT", str(offsets.pop()))]
    if not found:
        raise ValueError("Head prefix requires the separate QKV writer")
    program.kernels = kernels
    return program
