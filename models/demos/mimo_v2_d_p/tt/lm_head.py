# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Final RMSNorm + lm_head on device, for the one token whose next-token logits prefill hands out.

The vocab (152576) is split over the TP cols (2x2: 76288, 2x4 / Galaxy: 38144 per chip) and replicated over the SP
rows. The token's row lives on one SP row (block-cyclic order); every row runs the same 32-row tile slice, the host
reads back only the owning row's TP shards (V fp32 values).
"""

import torch

import ttnn
from models.demos.mimo_v2_d_p.tt.ffn import TtRMSNorm
from models.demos.mimo_v2_d_p.tt.weight_cache import cache_name


class TtLMHead:
    def __init__(self, mesh_device, norm_w: torch.Tensor, lm_head_w: torch.Tensor, eps: float, *, options=None):
        """``norm_w`` [H] (``model.norm.weight``), ``lm_head_w`` [V, H] (HF layout)."""
        self.mesh_device = mesh_device
        self.sp, self.tp = tuple(mesh_device.shape)
        self.vocab, self.hidden = lm_head_w.shape
        assert self.vocab % (ttnn.TILE_SIZE * self.tp) == 0, (self.vocab, self.tp)
        self.norm = TtRMSNorm(mesh_device, norm_w, eps)
        # bf16 weights (1.25 GB / TP per chip): one row of logits is ~0.3 GFLOP, precision is free here
        self.w = ttnn.as_tensor(
            lm_head_w.T.contiguous()[None, None],
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(self.sp, self.tp), dims=(None, 3)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(mesh_device, "G", "lm_head", options),
        )
        self.cfg = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def __call__(self, x: ttnn.Tensor, local_row: int) -> ttnn.Tensor:
        """x [1,1,S_local,H] -> fp32 logits [1,1,32,V/TP] of the 32-row tile holding ``local_row`` (every chip)."""
        r0 = local_row // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        xs = ttnn.slice(x, [0, 0, r0, 0], [1, 1, r0 + ttnn.TILE_SIZE, self.hidden])
        h = self.norm(xs)
        xs.deallocate(True)
        out = ttnn.linear(h, self.w, dtype=ttnn.float32, compute_kernel_config=self.cfg)
        h.deallocate(True)
        return out

    def logits(self, x: ttnn.Tensor, sp_row: int, local_row: int) -> torch.Tensor:
        """Host [V] fp32 logits of the token at (``sp_row``, ``local_row``) of x."""
        out = self(x, local_row)
        dts = ttnn.get_device_tensors(out)
        r = local_row % ttnn.TILE_SIZE
        logits = torch.cat([ttnn.to_torch(dts[sp_row * self.tp + c]).float()[0, 0, r] for c in range(self.tp)])
        out.deallocate(True)
        return logits
