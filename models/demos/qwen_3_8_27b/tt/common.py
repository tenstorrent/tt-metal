# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Small shared pieces: compute configs, weight upload (with optional on-disk tensor cache)."""

from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn

# Bring-up default for every projection matmul (recipe §2.3): HiFi4 + fp32 accumulation.
HIFI4_FP32 = dict(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)


def hifi4_fp32():
    return ttnn.WormholeComputeKernelConfig(**HIFI4_FP32)


def residual_dtype():
    """dtype of the residual stream and of each block's (TP-partial-summed) output.

    Correctness knob (default on, ``QWEN38_FP32_RESIDUAL=0`` restores bf16): the stream accumulates 64
    layers of block outputs and this model amplifies small errors from ~layer 26 on, so the residual adds
    and the TP all-reduce partial sums are kept fp32. Every matmul / SDPA / GDN *input* stays bf16 (the
    spec's activation dtype); norms cast back to bf16 on the way into a block.
    """
    return ttnn.bfloat16 if os.environ.get("QWEN38_FP32_RESIDUAL", "1") == "0" else ttnn.float32


def weight_cache_dir(mesh_device) -> Path | None:
    """Tilized weight cache root (outside the package). ``QWEN38_TT_CACHE=off`` disables caching."""
    root = os.environ.get("QWEN38_TT_CACHE", os.path.expanduser("~/.cache/tt_qwen_3_8_27b"))
    if root == "off":
        return None
    shape = "x".join(str(d) for d in mesh_device.shape)
    p = Path(root) / f"tensor_cache_{shape}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def upload(
    t: torch.Tensor | None,
    mesh_device,
    *,
    dtype,
    mapper,
    layout=ttnn.TILE_LAYOUT,
    cache: Path | None = None,
    name: str | None = None,
    memory_config=None,
):
    """``ttnn.as_tensor`` with an optional cache file; ``t`` may be None on a guaranteed cache hit."""
    cache_file = str(cache / name) if (cache is not None and name is not None) else None
    return ttnn.as_tensor(
        t,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=mapper,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_file,
    )


def per_tp_concat(blocks: list[torch.Tensor], dim: int) -> torch.Tensor:
    """Concatenate per-TP-column blocks so that a ShardTensor2dMesh over ``dim`` hands block c to column c."""
    return torch.cat(blocks, dim=dim)
