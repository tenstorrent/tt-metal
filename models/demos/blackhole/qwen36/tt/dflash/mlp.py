# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M5: the drafter's SwiGLU MLP -- ``down(silu(gate(x)) * up(x))``.

Dimensionally identical to the target's (5120 -> 17408 -> 5120), but deliberately *not*
built on ``tt/mlp.py``. That module needs a fully-constructed ``Qwen36ModelArgs`` -- swept
matmul program configs, ``prefill_tuning``, DRAM-shard memory configs, the fused
``all_gather_swiglu_prefill`` AGMM path -- which would drag the target's GDN and vision
config into the drafter for no correctness benefit. Those are perf levers earned by
measurement on 2048-row prefill chunks; the drafter runs 16 rows, where none of them apply.
Adopting the tuned paths is a later, separately-measured step.

TP: ``gate``/``up`` column-parallel (2176 out per device), ``down`` row-parallel over the
17408 axis, then one all-reduce to restore the replicated residual stream.
"""

from __future__ import annotations

import ttnn
from models.demos.blackhole.qwen36.tt.dflash.ccl import all_reduce_replicated
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig


class DFlashMLP:
    """SwiGLU feed-forward, column-parallel in and row-parallel out."""

    def __init__(
        self, mesh_device, cfg: DFlashDrafterConfig, weights, tt_ccl, topology=None, compute_kernel_config=None
    ):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.gate_proj = weights.gate_proj
        self.up_proj = weights.up_proj
        self.down_proj = weights.down_proj
        self.tt_ccl = tt_ccl
        self.topology = topology
        self.compute_kernel_config = compute_kernel_config or ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def forward(self, x):
        """``x`` [1, 1, S, 5120] replicated -> [1, 1, S, 5120] replicated."""
        ckc = self.compute_kernel_config
        mc = ttnn.DRAM_MEMORY_CONFIG

        # SILU fused into the gate matmul; `up` is a plain projection.
        gate = ttnn.linear(x, self.gate_proj, activation="silu", compute_kernel_config=ckc, memory_config=mc)
        up = ttnn.linear(x, self.up_proj, compute_kernel_config=ckc, memory_config=mc)
        hidden = ttnn.mul(gate, up, memory_config=mc)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        partial = ttnn.linear(hidden, self.down_proj, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(hidden)
        out = all_reduce_replicated(partial, self.mesh_device, self.tt_ccl, self.topology, dim=3)
        ttnn.deallocate(partial)
        return out
