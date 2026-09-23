# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Describe the constructed resident program outside warmed trace timing."""
from pathlib import Path
import ttnn


def describe_program(loop, program):
    def runtime_words(kernel, core):
        try:
            return len(kernel.runtime_args[core.x][core.y])
        except IndexError:
            return 0

    core_records = {}
    cb_records, kernel_records = [], []
    for cb in program.cbs:
        cores = ttnn.corerange_to_cores(cb.core_ranges, row_wise=True)
        pinned = cb.has_buffer()
        record = {"cores": [[c.x, c.y] for c in cores], "bytes_per_core": cb.total_size,
                  "pinned_tensor": pinned, "formats": [
                      {"index": f.buffer_index, "page_bytes": f.page_size}
                      for f in cb.format_descriptors]}
        cb_records.append(record)
        for c in cores:
            v = core_records.setdefault((c.x, c.y), {"static_cb_bytes": 0, "pinned_cb_view_bytes": 0})
            v["pinned_cb_view_bytes" if pinned else "static_cb_bytes"] += cb.total_size
    for kernel in program.kernels:
        cores = ttnn.corerange_to_cores(kernel.core_ranges, row_wise=True)
        definitions = dict(kernel.defines)
        kernel_records.append({"source": kernel.kernel_source,
            "phase_source": definitions.get("LOOP_SOURCE", kernel.kernel_source),
            "roles": [name for name in ("READER", "WRITER", "COMPUTE", "PROJECTION", "SWIGLU", "QUERY", "KEY", "VALUE") if definitions.get(name) == "1"],
            "cores": [[c.x, c.y] for c in cores],
            "compile_time_words": len(kernel.compile_time_args),
            "max_runtime_words": max(runtime_words(kernel, c) for c in cores)})
    body = loop.body
    roles = {"o_gate_up_down": body.projection_cores, "swiglu": body.sfpu_cores,
             "collectives": body.communication_cores, "mlp_norm": body.norm_cores,
             "attention": body.attention_stage.cores,
             "qkv": body.preparation.projection_cores, "pre_attention_norm": body.preparation.norm_cores,
             "rope": body.preparation.rope_cores, "kv_update": body.preparation.cache_cores}
    if loop.head is not None:
        roles.update(head=loop.head.cores, final_norm=loop.head.norm.cores)
    return {"scope": "Per-chip descriptor geometry, identical program on four TP chips. Static CB bytes count each alias allocation once. Pinned CB views may overlap allocator tensors; do not add them to allocator usage. Firmware, kernel code, config, semaphores and stacks are separate.",
            "layers": loop.count, "layer_workers": len(loop.cores),
            "state_bytes_per_core": loop.state_rows * 32 * 4,
            "roles": {name: [[c.x, c.y] for c in cores] for name, cores in roles.items()},
            "max_static_cb_bytes_per_core": max(v["static_cb_bytes"] for v in core_records.values()),
            "core_buffers": [{"core": list(c), **v} for c, v in sorted(core_records.items())],
            "circular_buffers": cb_records, "kernels": kernel_records}
