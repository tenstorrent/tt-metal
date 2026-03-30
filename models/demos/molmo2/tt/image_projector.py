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

**Matmul:** w1, w3, and w2 use tuned explicit multicast
(``MatmulMultiCoreReuseMultiCastProgramConfig``) when it fits device L1; w2 also applies
narrow-``M`` grid caps, optional LoFi in a band of ``M``, and implicit linear for very small ``M``.
Constants live in this module only (no separate config type or env vars).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

_TILE = 32

# --- Tuned matmul (fixed defaults; all logic stays in this file) ---
_GRID_X_OVERRIDE: Optional[int] = None
_GRID_Y_OVERRIDE: Optional[int] = None

_W2_MAX_PROGRAM_GRID_Y: Optional[int] = 4
_W2_MAX_GRID_Y_M_LE = 512
_W2_NARROW_M_OPTIM = True
_W2_NARROW_M_MAX = 512
_W2_NARROW_GRID_Y_CAP = 4

_W2_LOFI = True
_W2_LOFI_M_LE = 512
_W2_LOFI_ALWAYS = False
_W2_LOFI_ON_MICRO = False
_W2_MICRO_M_MAX = 160

_MULTICAST_ENFORCE_L1_LIMITS = True
_MULTICAST_MAX_PER_CORE_M_TILES = 24
_MULTICAST_MAX_PER_CORE_N_TILES = 16


def _largest_divisor_at_most(n: int, max_d: int = 8) -> int:
    for d in range(max_d, 0, -1):
        if n % d == 0:
            return d
    return 1


def _out_subblock_w(per_core_n: int, out_subblock_h: int) -> int:
    w = 4
    while w > 1:
        if w * out_subblock_h <= 4 and per_core_n % w == 0:
            return w
        w -= 1
    return 1


def _device_compute_grid(mesh_device) -> Tuple[int, int]:
    try:
        gs = mesh_device.compute_with_storage_grid_size()
        return int(gs.x), int(gs.y)
    except (AttributeError, TypeError):
        return 8, 8


def _snap_grid_y_for_k(k: int, grid_y: int) -> int:
    if k % _TILE != 0:
        return 1
    kt = k // _TILE
    for d in range(min(grid_y, kt), 0, -1):
        if kt % d == 0:
            return d
    return 1


def _multicast_per_core_tile_counts(m: int, k: int, n: int, grid_x: int, grid_y: int) -> Tuple[int, int, bool]:
    if k % (_TILE * grid_y) != 0:
        return 0, 0, False
    per_core_m = math.ceil(m / (_TILE * grid_y))
    per_core_n = math.ceil(n / (_TILE * grid_x))
    return per_core_m, per_core_n, True


def _explicit_multicast_within_l1_budget(
    m: int,
    k: int,
    n: int,
    grid_x: int,
    grid_y: int,
    max_m_tiles: int,
    max_n_tiles: int,
) -> bool:
    pm, pn, ok = _multicast_per_core_tile_counts(m, k, n, grid_x, grid_y)
    if not ok:
        return False
    return pm <= max_m_tiles and pn <= max_n_tiles


def _linear_multicast_program_config(
    m: int,
    k: int,
    n: int,
    grid_x: int,
    grid_y: int,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    if k % (_TILE * grid_y) != 0:
        raise ValueError(
            f"MatmulMultiCoreReuseMultiCastProgramConfig: K={k} must be divisible by "
            f"32 * grid_y ({_TILE * grid_y}) for grid ({grid_x}, {grid_y})."
        )
    per_core_m = math.ceil(m / (_TILE * grid_y))
    per_core_n = math.ceil(n / (_TILE * grid_x))
    in0_block_w = _largest_divisor_at_most(k // (_TILE * grid_y))
    out_subblock_h = 1
    out_subblock_w = _out_subblock_w(per_core_n, out_subblock_h)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
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
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.dtype = dtype

        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

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

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_w2_lofi = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _matmul_grid_xy(self) -> Tuple[int, int]:
        if _GRID_X_OVERRIDE is not None and _GRID_Y_OVERRIDE is not None:
            return int(_GRID_X_OVERRIDE), int(_GRID_Y_OVERRIDE)
        return _device_compute_grid(self.mesh_device)

    def _w2_effective_grid_y(self, k: int, gy_eff: int, m: int) -> int:
        gy = gy_eff
        if _W2_MAX_PROGRAM_GRID_Y is not None and m <= _W2_MAX_GRID_Y_M_LE:
            gy = _snap_grid_y_for_k(k, min(gy, _W2_MAX_PROGRAM_GRID_Y))
        if _W2_NARROW_M_OPTIM and m <= _W2_NARROW_M_MAX:
            gy = _snap_grid_y_for_k(k, min(gy, _W2_NARROW_GRID_Y_CAP))
        return gy

    def _w2_use_implicit_matmul_program(self, m: int) -> bool:
        return _W2_MICRO_M_MAX > 0 and m <= _W2_MICRO_M_MAX

    def _w2_use_lofi(self, m: int) -> bool:
        if self._w2_use_implicit_matmul_program(m) and not _W2_LOFI_ON_MICRO:
            return False
        if not _W2_LOFI:
            return False
        if _W2_LOFI_ALWAYS:
            return True
        return m <= _W2_LOFI_M_LE

    def _linear_tuned(self, x: ttnn.Tensor, weight: ttnn.Tensor, *, is_w2: bool = False) -> ttnn.Tensor:
        gx, gy = self._matmul_grid_xy()
        k = int(x.shape[-1])
        m = int(x.shape[-2])
        gy_eff = _snap_grid_y_for_k(k, gy)
        gy_use = self._w2_effective_grid_y(k, gy_eff, m) if is_w2 else gy_eff

        compute_cfg = (
            self.compute_kernel_config_w2_lofi if is_w2 and self._w2_use_lofi(m) else self.compute_kernel_config
        )

        kwargs = {
            "compute_kernel_config": compute_cfg,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }

        # w2: very small M uses implicit linear only (no explicit multicast program).
        if is_w2 and self._w2_use_implicit_matmul_program(m):
            return ttnn.linear(x, weight, **kwargs)

        n = int(weight.shape[-1])
        use_multicast = (not _MULTICAST_ENFORCE_L1_LIMITS) or _explicit_multicast_within_l1_budget(
            m,
            k,
            n,
            gx,
            gy_use,
            _MULTICAST_MAX_PER_CORE_M_TILES,
            _MULTICAST_MAX_PER_CORE_N_TILES,
        )
        if use_multicast:
            kwargs["program_config"] = _linear_multicast_program_config(m, k, n, gx, gy_use)

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

        if seq_len > 1024:
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])

        gate = self._linear_tuned(x, self.w1_weight)
        gate = ttnn.silu(gate)

        up = self._linear_tuned(x, self.w3_weight)

        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        output = self._linear_tuned(hidden, self.w2_weight, is_w2=True)
        ttnn.deallocate(hidden)

        if seq_len > 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        return output
