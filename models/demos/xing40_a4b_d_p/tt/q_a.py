# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 q_a stem on the 4x2 mesh: q_resid = q_a_layernorm(q_a_proj(attn_norm)), plain w, eps 1e-6 (rms_norm_eps).

Adapted from models/demos/hy4_preview_d_p/tt/q_a.py:TtQa (itself from deepseek_v3_d_p/tt/mla/mla.py:
ttMLA._q_a_latent). Per chip (r, c) the input is attn_norm's column split [1, 1, S/4, 1792] (tt/norm.py output, bf16):

    ttnn.linear                x_c @ W_c, W_c = q_a_proj^T[c*1792:(c+1)*1792, :] [1792, 768] bf16, HiFi4 + fp32 dest,
                               fp32 partial out (no bf16 rounding before the reduce)
    ttnn.all_reduce axis 1     sum the two K partials -> [1, 1, S/4, 768] fp32, identical on both chips of a row
                               (payload 1280 x 768 fp32 = 3.9 MB at chunk 5120)
    ttnn.bringup.rms_norm      q_a_layernorm, eps 1e-6, fp32 in, fp32 row-major weight, HiFi4 + fp32 dest -> fp32,
                               then typecast to ``dtype`` (bf16)

``norm_impl="native"`` runs ttnn.rms_norm for comparison (it scales rows ~0.1% low on fp32 input, known issues).
Output q_resid [1, 1, S/4, 768] per chip, split by rows over axis 0, replicated over axis 1 (plan.md).
No host work in __call__: weight and norm weight built at load.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.xing40_a4b_d_p.tt.settings import settings

TILE = 32


class TtQa:
    """q_a_proj + q_a_layernorm of one layer. Call with the per-chip attn_norm [1, 1, S/4, H/2] TILE (bf16 or fp32);
    returns q_resid [1, 1, S/4, q_lora_rank] TILE ``dtype``, replicated over ``cluster_axis``."""

    def __init__(
        self,
        mesh,
        q_a_proj: torch.Tensor,  # [q_lora_rank, H] (HF nn.Linear layout)
        q_a_norm: torch.Tensor,  # [q_lora_rank]
        eps: float,
        cluster_axis: int = 1,
        dtype=ttnn.bfloat16,
        norm_impl: str = "bringup",
    ):
        assert norm_impl in ("bringup", "native"), norm_impl
        self.norm_op = ttnn.bringup.rms_norm if norm_impl == "bringup" else ttnn.rms_norm
        rank, hidden = q_a_proj.shape
        tp = mesh.shape[cluster_axis]
        assert hidden % (TILE * tp) == 0 and rank % TILE == 0, (rank, hidden, tp)
        assert q_a_norm.numel() == rank
        self.mesh, self.rank, self.eps, self.cluster_axis, self.dtype = mesh, rank, float(eps), cluster_axis, dtype
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, settings.get("MATMUL_FIDELITY")),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # W^T [H, rank], K (dim 2) split over the cluster axis: chip column c holds rows c*H/tp .., the K slice that
        # matches its attn_norm columns; replicated over the other axis.
        dims = (None, 2) if cluster_axis == 1 else (2, None)
        self.weight = ttnn.from_torch(
            q_a_proj.t().contiguous().reshape(1, 1, hidden, rank),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )
        self.norm_w = ttnn.from_torch(
            q_a_norm.float().reshape(1, 1, rank // TILE, TILE),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        part = ttnn.linear(x, self.weight, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        qr = ttnn.all_reduce(part, cluster_axis=self.cluster_axis, memory_config=dram)
        ttnn.deallocate(part)
        y = self.norm_op(qr, weight=self.norm_w, epsilon=self.eps, memory_config=dram, compute_kernel_config=self.ckc)
        ttnn.deallocate(qr)
        if y.dtype != self.dtype:
            y2 = ttnn.typecast(y, self.dtype, memory_config=dram)
            ttnn.deallocate(y)
            y = y2
        return y


def build_q_a(mesh, loader, cfg, layer: int, dtype=ttnn.bfloat16) -> TtQa:
    p = f"model.layers.{layer}.self_attn."
    return TtQa(
        mesh,
        loader.get(p + "q_a_proj.weight").float(),
        loader.get(p + "q_a_layernorm.weight").float(),
        cfg.rms_norm_eps,
        cluster_axis=1,
        dtype=dtype,
    )
