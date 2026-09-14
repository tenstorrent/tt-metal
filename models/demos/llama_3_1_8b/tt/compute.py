# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compute-kernel and SDPA program configs, re-derived for THIS model's shapes.

Recipe §2.3: compute configs are *shape-tuned*, not structural. A borrowed constant encodes the
source's hidden size, head_dim and chunk — so the numbers below are justified against Llama-3.1-8B's
own dimensions rather than inherited.

Bring-up default (recipe §2.3): **HiFi4 with ``fp32_dest_acc_en=True``** for every projection matmul
and the plain SDPA. The one documented exception is the ring cache-read SDPA, where the constraint is
local to that op, not to SDPA in general — :func:`ring_sdpa_compute_config` carries it with the
reason attached and ``LLAMA_RING_FP32_ACC=1`` re-enables it for an A/B.
"""

from __future__ import annotations

import os

import ttnn

# Per-device sequence rows the SDPA q-loop walks. chunk 5120 / sp 8 = 640, one-shot 10240 / sp 8 =
# 1280; q_chunk 128 divides both (and 64 would too, at more loop overhead). k_chunk 512 spans the
# gathered ring buffer in whole steps for every cache capacity this package runs (10240 / 512 = 20).
RING_Q_CHUNK = 128
RING_K_CHUNK = 512


def matmul_compute_config(mesh_device=None):
    """HiFi4 + fp32 accumulate — the bring-up default for every projection matmul."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def sdpa_compute_config():
    """HiFi4 + fp32 accumulate for the plain (non-ring) causal SDPA used by the unit tests."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def ring_sdpa_compute_config():
    """Compute config for ``ring_joint_scaled_dot_product_attention``.

    ``fp32_dest_acc_en`` defaults to **False** here and only here. The ring path keeps the online
    softmax running statistics plus the gathered K/V slab resident, and the fp32 dest doubles the
    accumulator footprint in the same L1 budget; the op's validated configuration on this galaxy
    (the source this was taken from, measured at head_dim 128 / sp8) runs it off. It is a local
    constraint, not a global one — the projection matmuls above keep fp32 accumulate. Set
    ``LLAMA_RING_FP32_ACC=1`` to measure the difference.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=os.getenv("LLAMA_RING_FP32_ACC") == "1",
        packer_l1_acc=False,
    )


def ring_sdpa_program_config(mesh_device):
    """SDPA program config for the ring path: the compute grid MINUS the CCL column.

    ``ring_joint`` asserts its CCL core-grid offset is at or past the SDPA grid's x extent, and the
    CCL manager puts the ring workers in the last column — so the SDPA grid must be one column
    narrower than the device's real compute grid.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=RING_Q_CHUNK,
        k_chunk_size=RING_K_CHUNK,
        exp_approx_mode=False,
    )


def plain_sdpa_program_config(mesh_device, seq_len: int):
    """Program config for ``ttnn.transformer.scaled_dot_product_attention`` (single-device tests).

    q/k chunks are clamped to the sequence so a short unit-test sequence does not ask the kernel for
    a chunk larger than the tensor it is walking.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    q_chunk = min(RING_Q_CHUNK, max(32, seq_len))
    k_chunk = min(RING_K_CHUNK, max(32, seq_len))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
