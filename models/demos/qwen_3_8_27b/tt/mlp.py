# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dense SwiGLU MLP, TP-sharded: gate/up column-parallel, down row-parallel, TP all-reduce.

Structure from minimax_m3/tt/dense_mlp.py; the activation is plain SiLU (Qwen), not M3's clamped
swigluoai — the structure is ported, not the math. Every matmul HiFi4 + fp32 accumulation.
"""

import ttnn
from models.demos.qwen_3_8_27b.tt.common import hifi4_fp32, residual_dtype, upload


class TtMLP:
    def __init__(self, mesh_config, sd, *, dtypes: dict, cache=None, prefix=""):
        """sd: {gate_proj.weight, up_proj.weight, down_proj.weight} (HF [out, in]) or None on a cache hit.
        dtypes: {"gate", "up", "down"} -> ttnn dtype."""
        self.mc = mesh_config
        mesh = mesh_config.mesh_device
        col = mesh_config.shard(None, 3)
        row = mesh_config.shard(None, 2)

        def w(name):
            return None if sd is None else sd[f"{name}.weight"].T.contiguous()[None, None]

        self.w_gate = upload(
            w("gate_proj"), mesh, dtype=dtypes["gate"], mapper=col, cache=cache, name=f"{prefix}w_gate"
        )
        self.w_up = upload(w("up_proj"), mesh, dtype=dtypes["up"], mapper=col, cache=cache, name=f"{prefix}w_up")
        self.w_down = upload(
            w("down_proj"), mesh, dtype=dtypes["down"], mapper=row, cache=cache, name=f"{prefix}w_down"
        )
        self.ckc = hifi4_fp32()

    def __call__(self, x):
        mc = dict(compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = ttnn.linear(x, self.w_gate, **mc)
        u = ttnn.linear(x, self.w_up, **mc)
        h = ttnn.multiply(
            g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = ttnn.linear(h, self.w_down, dtype=residual_dtype(), **mc)
        ttnn.deallocate(h)
        return self.mc.all_reduce_tp(out)
