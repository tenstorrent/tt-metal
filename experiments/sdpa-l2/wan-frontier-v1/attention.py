# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Wan SP4/TP2 adapter; quantize before gathering KV, preserving the tail mask."""

import functools
import importlib.util
from pathlib import Path

import ttnn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "wan_frontier_kernel", HERE.parent / "flux2-frontier-v1/device_attention.py"
)
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)


class WanAttentionAdapter:
    def __init__(self, variant):
        assert variant in kernel.VARIANTS
        self.variant = variant
        self.transport = {}

    def install(self, pipeline):
        for expert, state in enumerate(pipeline.transformer_states):
            for index, block in enumerate(state.model.blocks):
                block.attn1._attention_override = functools.partial(self.run, f"expert{expert}.block{index}")

    def run(self, name, attn, q, k, v, logical_n):
        sp = attn.parallel_config.sequence_parallel
        assert tuple(attn.mesh_device.shape) == (2, 4) and sp.factor == 4 and sp.mesh_axis == 1
        assert q.shape == k.shape == v.shape and q.shape[0] == 1
        assert 0 <= q.shape[2] * sp.factor - logical_n < 32
        device = attn.mesh_device
        hw = device.compute_with_storage_grid_size()
        cores = hw.x * (hw.y - 1)
        q, k, v = [kernel.prepare(device, x, self.variant, is_q=i == 0, cores=cores) for i, x in enumerate((q, k, v))]
        k, v = [attn.ccl_manager.all_gather(x, dim=2, mesh_axis=sp.mesh_axis, use_hyperparams=True) for x in (k, v)]
        self.transport[name] = dict(
            q_dtype=str(q.dtype),
            k_dtype=str(k.dtype),
            v_dtype=str(v.dtype),
            q_shape=list(q.shape),
            kv_shape=list(k.shape),
            logical_k=logical_n,
        )
        return kernel.attention(device, q, k, v, self.variant, logical_k=logical_n, max_cores=cores)
