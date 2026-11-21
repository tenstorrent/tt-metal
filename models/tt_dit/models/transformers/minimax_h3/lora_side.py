# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unmerged LoRA side branch for the DiT block linears: `y = W x + B (A x)`.

The rank activations ride along as extra K columns of the linear's own input, and B^T is stacked under W in the same
weight, so the existing fused ops (all-gather matmul with fused SwiGLU / head split / QK norm, matmul + reduce-scatter
+ gated residual) run unchanged and add the adapter before any activation, norm or reduction:

- column-parallel (to_qkv, ff1; input K-fractured on TP): `u = reduce_scatter(x_d @ A_d^T)` gives each device a slice
  of `A x`, which is concatenated to `x_d`; the all-gather then assembles `[x_0, u_0, x_1, u_1, ...]` and the weight
  rows are stacked in that order.
- row-parallel (to_out, ff2; input K-local): `p_d = x_d @ A_d^T` is the device's partial of `A x` and goes in as the
  matmul's fused-concat second input; each device's weight shard is `[W_d; B^T]`, so the reduce-scatter that sums the
  `W_d x_d` partials sums the `p_d B^T` ones too.

A is its own bf16 weight; B shares W's tensor (and so its dtype, fidelity and weight rounding).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

import ttnn

from ....layers.linear import _linear
from ....layers.module import Module, Parameter
from ....utils.matmul import _resolve_fused_mmrs_config

_TILE = 32
# MINIMAX_H3_LORA_SIDE_PROBE="qkv_tiles,ff1_tiles,fidelity" (perf probe, default unset): per-device rank-column tiles of
# the q/k/v and ff1 inputs, and the A matmuls' fidelity (hifi2 / lofi).
_PROBE = (
    os.environ.get("MINIMAX_H3_LORA_SIDE_PROBE", "").split(",") if os.environ.get("MINIMAX_H3_LORA_SIDE_PROBE") else []
)


def col_suffix_cols(k_local: int, rank_total: int, tp: int, site: str = "") -> int:
    """Per-device width of the rank columns appended to a column-parallel input: the fewest tiles holding
    `rank_total / tp`, padded until the per-device K tile count is a multiple of 3 (every swept blocking then keeps a
    K block near its own; 42 -> 45 tiles for the H3 shapes)."""
    if _PROBE and site in ("qkv", "ff1"):
        return int(_PROBE[0 if site == "qkv" else 1]) * _TILE
    k_tiles = k_local // _TILE
    tiles = -(-rank_total // (tp * _TILE))
    while (k_tiles + tiles) % 3:
        tiles += 1
    return tiles * _TILE


def fit_k_block(k_block: int, *tile_counts: int) -> int:
    """The largest K block not above `k_block` that divides every tile count."""
    return max(d for d in range(1, k_block + 1) if all(n % d == 0 for n in tile_counts))


def _wide(*tensors: torch.Tensor) -> list[torch.Tensor]:
    """The tensors in one dtype of at least float32."""
    dtype = torch.float32
    for t in tensors:
        dtype = torch.promote_types(dtype, t.dtype)
    return [t.to(dtype) for t in tensors]


def augment_col_weight(weight: torch.Tensor, b: torch.Tensor, tp: int) -> torch.Tensor:
    """[out, K] W and [out, R_pad] B (columns in `u`'s order) -> [out, tp * (K / tp + R_pad / tp)], the input columns
    in the gathered order `[x_0, u_0, x_1, u_1, ...]`."""
    out = weight.shape[0]
    w, b = _wide(weight, b)
    return torch.cat([w.reshape(out, tp, -1), b.reshape(out, tp, -1)], dim=2).reshape(out, -1)


def augment_row_weight(weight: torch.Tensor, b: torch.Tensor, tp: int) -> torch.Tensor:
    """[out, K] W and [out, r] B -> [out, tp * (K / tp + r)]: each device's K slice followed by all of B, so the K-row
    split hands device d `[W_d; B^T]` (after the row-parallel linear's transpose)."""
    out = weight.shape[0]
    w, b = _wide(weight, b)
    return torch.cat([w.reshape(out, tp, -1), b.unsqueeze(1).expand(out, tp, b.shape[1])], dim=2).reshape(out, -1)


def rank_columns(b: torch.Tensor, offset: int, width: int) -> torch.Tensor:
    """[out, r] B placed at rank columns [offset, offset + r) of a zero [out, width]."""
    return F.pad(_wide(b)[0], (offset, width - offset - b.shape[1]))


def agmm_side_kwargs(kwargs: dict, k_tiles: int) -> dict:
    """An all-gather matmul's blocking kwargs with the K block refitted to `k_tiles` per-device tiles."""
    block = kwargs.get("default_block_size")
    if block is not None:
        m, k, n, *rest = block
        return {**kwargs, "default_block_size": (m, fit_k_block(k, k_tiles), n, *rest)}
    config = kwargs.get("matmul_config")
    if config is not None:
        config = ttnn.MinimalMatmulConfig(
            M_block_size=config.M_block_size,
            K_block_size=fit_k_block(config.K_block_size, k_tiles),
            N_block_size=config.N_block_size,
            subblock_h=config.subblock_h,
            subblock_w=config.subblock_w,
            compute_with_storage_grid_size=config.compute_with_storage_grid_size,
        )
        return {**kwargs, "matmul_config": config}
    return kwargs


def mmrs_side_config(config, m: int, k: int, n: int, core_grid: ttnn.CoreCoord, rank: int):
    """The fused MM + RS blocking of the base `(m, k, n)` shape (`config` if given, else the table / rule pick) with
    its K block refitted to `k + rank` and the concat seam at `k`; None when the base shape resolves none."""
    if config is None:
        config = _resolve_fused_mmrs_config(m, k, n, core_grid, log=False)
    if config is None:
        return None
    return config._replace(K_block_size=fit_k_block(config.K_block_size, k // _TILE, (k + rank) // _TILE))


class LowRankDown(Module):
    """`x @ A^T` for a K-fractured `x`: A^T is [in, width] sharded on K over TP, so each device computes its partial
    of `A x`; `reduce` reduce-scatters the partials on the rank axis (column-parallel sites), otherwise they stay
    per-device partials (row-parallel sites). Rows of A beyond the file's rank are zero."""

    def __init__(
        self,
        in_features: int,
        width: int,
        *,
        reduce: bool,
        mesh_device: ttnn.MeshDevice,
        mesh_axis: int,
        ccl_manager,
    ) -> None:
        super().__init__()
        self.width = width
        self.reduce = reduce
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.ccl_manager = ccl_manager
        self.weight = Parameter(total_shape=[in_features, width], mesh_axes=[mesh_axis, None], device=mesh_device)
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi if _PROBE[2:] == ["lofi"] else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        a = state.pop("weight", None)
        if a is not None:
            state["weight"] = F.pad(a.float(), (0, 0, 0, self.width - a.shape[0])).T

    def forward(self, x: ttnn.Tensor, *, dtype=None, use_persistent_buffer: bool = True) -> ttnn.Tensor:
        """`dtype`: the output dtype; the reduce-scatter itself always moves bf16 partials."""
        p = _linear(
            x,
            self.weight.data,
            bias=None,
            mesh_device=self.mesh_device,
            compute_kernel_config=self.compute_config,
            dtype=ttnn.bfloat16 if self.reduce else dtype,
        )
        if self.reduce:
            p = self.ccl_manager.reduce_scatter(
                p, dim=-1, mesh_axis=self.mesh_axis, use_persistent_buffer=use_persistent_buffer
            )
            if dtype is not None and dtype != p.dtype:
                p = ttnn.typecast(p, dtype)
        return p
