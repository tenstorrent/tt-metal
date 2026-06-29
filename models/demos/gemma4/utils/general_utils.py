# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the Gemma4 demo.
"""

import math


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def build_matmul_program_config(M: int, K: int, N: int, grid_x: int, grid_y: int):
    """Build a MatmulMultiCoreReuseMultiCastProgramConfig for a (M, K, N) linear on a given grid."""
    import ttnn

    tile = ttnn.TILE_SIZE
    m_t = math.ceil(M / tile)
    k_t = math.ceil(K / tile)
    n_t = math.ceil(N / tile)
    per_core_m = math.ceil(m_t / grid_y)
    per_core_n = math.ceil(n_t / grid_x)
    in0_block_w = max(1, k_t // grid_x)
    while k_t % in0_block_w != 0:
        in0_block_w -= 1
    out_subblock_w = max(i for i in range(1, 9) if per_core_n % i == 0)
    out_subblock_h = max(i for i in range(1, 9) if per_core_m % i == 0 and i * out_subblock_w <= 8)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


def cast_host_for_ttnn(torch_tensor, ttnn_dtype):
    """Cast a host torch tensor to the torch dtype matching ``ttnn_dtype``.

    ``ttnn.as_tensor``/``from_torch`` run a C++ ``to_dtype`` conversion whenever the
    source dtype differs from the requested one. That conversion queries tile metadata
    on a ROW_MAJOR host intermediate and emits the #18536 "extract tile information out
    of a ROW MAJOR layout" warning. Matching the host dtype up front lets ``to_dtype``
    short-circuit, so no warning is emitted.

    Only float targets representable in torch are handled; block formats (bfloat8_b /
    bfloat4_b) have no torch equivalent and are returned unchanged.
    """
    import ttnn

    try:
        import torch
    except ImportError:
        return torch_tensor

    mapping = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
    target = mapping.get(ttnn_dtype)
    if target is not None and torch_tensor.dtype != target:
        return torch_tensor.to(target)
    return torch_tensor
