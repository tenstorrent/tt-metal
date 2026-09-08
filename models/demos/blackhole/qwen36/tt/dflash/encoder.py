# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M4: the DFlash context encoder -- ``hidden_norm(fc(target_hidden))``.

This is what makes the drafter a *drafter* rather than a small LM: instead of embedding
tokens, it consumes the target's residual stream at ``target_layer_ids`` = [1, 16, 31, 46,
61], concatenated to 25600 wide, and projects it down to 5120. EAGLE-3-style multi-layer
fusion.

Computed **once per forward** and shared by all 5 layers, which every layer then uses as the
source of its context K/V (see ``attention.py``). It is not recomputed per layer.

``fc`` is row-parallel over the 25600 input axis, because that is the axis the target
already has sharded -- chip ``d`` holds ``tap_i[d*640:(d+1)*640]`` for every tap. The
weight's rows are permuted at load time to match (``weights.fc_input_permutation``), so each
chip's contiguous 3200-row slice is exactly the rows it holds activations for. The partials
are then summed by an all-reduce, leaving ``target_hidden`` replicated for the per-layer
norms and column-parallel K/V projections.
"""

from __future__ import annotations

import ttnn
from models.demos.blackhole.qwen36.tt.dflash.ccl import all_reduce_replicated
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig
from models.demos.blackhole.qwen36.tt.dflash.rms_norm import rms_norm


class DFlashContextEncoder:
    """``fc`` + ``hidden_norm``, fracturing over the tap axis."""

    def __init__(
        self, mesh_device, cfg: DFlashDrafterConfig, weights, tt_ccl, topology=None, compute_kernel_config=None
    ):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.fc = weights.fc
        self.hidden_norm = weights.hidden_norm
        self.tt_ccl = tt_ccl
        self.topology = topology
        self.compute_kernel_config = compute_kernel_config or ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def forward(self, target_hidden):
        """``target_hidden`` [1, 1, ctx, 25600/tp] sharded -> [1, 1, ctx, 5120] replicated."""
        partial = ttnn.linear(
            target_hidden,
            self.fc,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        summed = all_reduce_replicated(partial, self.mesh_device, self.tt_ccl, self.topology, dim=3)
        ttnn.deallocate(partial)
        out = rms_norm(summed, self.hidden_norm, self.cfg.rms_norm_eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(summed)
        return out
