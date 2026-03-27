# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SwiGLU Image Projector for Molmo2 Vision Adapter.

Projects pooled image features from the ViT hidden dimension (1152)
to the language model hidden dimension (4096) using a SwiGLU MLP:
    output = w2(silu(w1(x)) * w3(x))

Dimensions:
    - input_dim: 1152 (adapter hidden size)
    - intermediate_dim: 12288
    - output_dim: 4096 (text model hidden size)

All linear layers have bias=False.

**Tracy / T3K profiling — all projector matmuls** (``w1``, ``w3``, ``w2``):

- ``MOLMO2_IMAGE_PROJECTOR_MATMUL_MODE`` (preferred) or ``MOLMO2_IMAGE_PROJECTOR_W2_MODE`` (alias) or
  constructor ``matmul_mode`` / ``w2_matmul_mode``:

  - ``0`` — Default ``ttnn.linear`` (interleaved DRAM; implicit multi-core matmul).
  - ``1`` — Pass ``core_grid`` (from device grid + per-K snap of ``grid_y``).
  - ``2`` — Explicit ``MatmulMultiCoreReuseMultiCastProgramConfig`` (2D multicast reuse).
  - ``3`` — Combine (1) + (2).

- Optional grid override for modes 1–3::

    export MOLMO2_IMAGE_PROJECTOR_MATMUL_GRID=8,8

  Fallback env: ``MOLMO2_IMAGE_PROJECTOR_W2_GRID``. If unset, uses ``device.compute_with_storage_grid_size()``.

  For ``w1``/``w3`` (``K=1152``), ``grid_y`` is snapped down so ``K`` divides ``32×grid_y`` (e.g. 8→6 for
  an 8×8 request); ``w2`` (``K=12288``) usually keeps the full grid.

Example Tracy sweeps on T3K::

    MOLMO2_IMAGE_PROJECTOR_MATMUL_MODE=0 python -m tracy ...   # baseline
    MOLMO2_IMAGE_PROJECTOR_MATMUL_MODE=3 python -m tracy ...   # all linears optimized
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

_TILE = 32


def _largest_divisor_at_most(n: int, max_d: int = 8) -> int:
    for d in range(max_d, 0, -1):
        if n % d == 0:
            return d
    return 1


def _out_subblock_w(per_core_N: int, out_subblock_h: int) -> int:
    w = 4
    while w > 1:
        if w * out_subblock_h <= 4 and per_core_N % w == 0:
            return w
        w -= 1
    return 1


def _parse_matmul_grid_env() -> Optional[Tuple[int, int]]:
    for key in ("MOLMO2_IMAGE_PROJECTOR_MATMUL_GRID", "MOLMO2_IMAGE_PROJECTOR_W2_GRID"):
        raw = os.environ.get(key)
        if not raw:
            continue
        parts = raw.strip().split(",")
        if len(parts) != 2:
            continue
        return int(parts[0].strip()), int(parts[1].strip())
    return None


def _resolve_matmul_mode(matmul_mode: Optional[int], w2_matmul_mode: Optional[int]) -> int:
    if matmul_mode is not None:
        return int(matmul_mode)
    if w2_matmul_mode is not None:
        return int(w2_matmul_mode)
    if "MOLMO2_IMAGE_PROJECTOR_MATMUL_MODE" in os.environ:
        return int(os.environ["MOLMO2_IMAGE_PROJECTOR_MATMUL_MODE"])
    return int(os.environ.get("MOLMO2_IMAGE_PROJECTOR_W2_MODE", "0"))


def _device_compute_grid(mesh_device) -> Tuple[int, int]:
    try:
        gs = mesh_device.compute_with_storage_grid_size()
        return int(gs.x), int(gs.y)
    except (AttributeError, TypeError):
        return 8, 8


def _snap_grid_y_for_k(k: int, grid_y: int) -> int:
    """
    Pick grid_y <= requested so K % (32 * grid_y) == 0 (required for multicast program config).

    K is in elements (e.g. 1152 or 12288). Example: K=1152, kt=36 tiles, grid_y=8 → use 6.
    """
    if k % _TILE != 0:
        return 1
    kt = k // _TILE
    for d in range(min(grid_y, kt), 0, -1):
        if kt % d == 0:
            return d
    return 1


def _linear_multicast_program_config(
    m: int,
    k: int,
    n: int,
    grid_x: int,
    grid_y: int,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """Build a 2D multicast reuse config for interleaved DRAM matmul (matches tt_transformers-style tuning)."""
    if k % (_TILE * grid_y) != 0:
        raise ValueError(
            f"MatmulMultiCoreReuseMultiCastProgramConfig: K={k} must be divisible by "
            f"32 * grid_y ({_TILE * grid_y}) for grid ({grid_x}, {grid_y}). "
            "Try a different MOLMO2_IMAGE_PROJECTOR_MATMUL_GRID or rely on automatic grid_y snapping."
        )
    per_core_M = math.ceil(m / (_TILE * grid_y))
    per_core_N = math.ceil(n / (_TILE * grid_x))
    in0_block_w = _largest_divisor_at_most(k // (_TILE * grid_y))
    out_subblock_h = 1
    out_subblock_w = _out_subblock_w(per_core_N, out_subblock_h)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


class ImageProjector(LightweightModule):
    """
    SwiGLU projector for mapping vision features to language model space.

    Architecture:
        - w1: Linear(input_dim, intermediate_dim, bias=False) - gate projection
        - w3: Linear(input_dim, intermediate_dim, bias=False) - up projection
        - w2: Linear(intermediate_dim, output_dim, bias=False) - down projection
        - Output: w2(silu(w1(x)) * w3(x))
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        input_dim: int = 1152,
        intermediate_dim: int = 12288,
        output_dim: int = 4096,
        weight_cache_path=None,
        state_dict_prefix: str = "model.vision_backbone.image_projector",
        dtype=ttnn.bfloat8_b,
        matmul_mode: Optional[int] = None,
        w2_matmul_mode: Optional[int] = None,
    ):
        """
        Initialize ImageProjector.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing weights
            input_dim: Input dimension (1152)
            intermediate_dim: Hidden dimension (12288)
            output_dim: Output dimension (4096)
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
            matmul_mode: Optional override for ``MOLMO2_IMAGE_PROJECTOR_MATMUL_MODE`` (0–3).
            w2_matmul_mode: Deprecated alias for ``matmul_mode`` (``MOLMO2_IMAGE_PROJECTOR_W2_MODE``).
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.dtype = dtype
        self.matmul_mode = _resolve_matmul_mode(matmul_mode, w2_matmul_mode)

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

        # Load w1 (gate): input_dim -> intermediate_dim (no bias)
        w1_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w1.weight"], -2, -1)

        self.w1_weight = ttnn.as_tensor(
            w1_weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w1.weight"),
        )

        # Load w3 (up): input_dim -> intermediate_dim (no bias)
        w3_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w3.weight"], -2, -1)

        self.w3_weight = ttnn.as_tensor(
            w3_weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w3.weight"),
        )

        # Load w2 (down): intermediate_dim -> output_dim (no bias)
        w2_weight = torch.transpose(state_dict[f"{state_dict_prefix}.w2.weight"], -2, -1)

        self.w2_weight = ttnn.as_tensor(
            w2_weight.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("w2.weight"),
        )

        # Compute kernel configs
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    @property
    def w2_matmul_mode(self) -> int:
        """Alias for ``matmul_mode`` (backward compatibility)."""
        return self.matmul_mode

    def _matmul_grid_xy(self) -> Tuple[int, int]:
        g = _parse_matmul_grid_env()
        return g if g is not None else _device_compute_grid(self.mesh_device)

    def _linear_with_matmul_mode(self, x: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        """
        Shared path for ``w1``, ``w3``, ``w2`` linears. Mode from ``matmul_mode`` / env (see module docstring).

        ``grid_y`` is snapped per matmul so ``K`` divides ``32×grid_y`` (needed for mode 2/3 on ``w1``/``w3``).
        """
        mode = self.matmul_mode
        gx, gy = self._matmul_grid_xy()
        k = int(x.shape[-1])
        gy_eff = _snap_grid_y_for_k(k, gy)

        kwargs = {
            "compute_kernel_config": self.compute_kernel_config,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }

        if mode in (1, 3):
            kwargs["core_grid"] = ttnn.CoreGrid(x=gx, y=gy_eff)

        if mode in (2, 3):
            m = int(x.shape[-2])
            n = int(weight.shape[-1])
            kwargs["program_config"] = _linear_multicast_program_config(m, k, n, gx, gy_eff)

        return ttnn.linear(x, weight, **kwargs)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through SwiGLU projector.

        Args:
            x: Input tensor of shape [1, 1, num_tokens, input_dim]

        Returns:
            Output tensor of shape [1, 1, num_tokens, output_dim]
        """
        seq_len = x.shape[-2]

        # Reshape for long sequences
        if seq_len > 1024:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        # w1 (gate projection) with SiLU activation
        gate = self._linear_with_matmul_mode(x, self.w1_weight)
        gate = ttnn.silu(gate)

        # w3 (up projection)
        up = self._linear_with_matmul_mode(x, self.w3_weight)

        # Element-wise multiply: silu(w1(x)) * w3(x)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # w2 down projection — M×K×N ≈ (padded_seq)×12288×4096 (e.g. 736×12288×4096 for 729 tokens).
        output = self._linear_with_matmul_mode(hidden, self.w2_weight)
        ttnn.deallocate(hidden)

        # Reshape back if needed
        if seq_len > 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        return output
