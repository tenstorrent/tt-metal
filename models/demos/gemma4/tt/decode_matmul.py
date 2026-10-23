# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Explicit 1D-multicast program configs for the dense decode matmuls (opt-in).

A plain ``ttnn.linear`` on a one-tile-high activation picks the 1D systolic config with ``in0_block_w=1``:
the activation is multicast one K tile at a time, so a K=2816 projection runs 88 multicast rounds and
reads its weight at ~15% of the DRAM peak (26B-A4B TP=4: gate_up 43 us for 6.1 MB, wqkv 52 us for 11.5 MB,
router 35 us for 0.7 MB, measured on a P150 with trace replay). The same kernel with a wider K block
(``in0_block_w`` = largest divisor of the K tiles up to ``GEMMA4_DECODE_MM_IN0_BLOCK_W``, default 8) and
one N tile per core over the whole worker grid halves those times.

A matmul with an explicit program config silently drops the default math fidelity from HiFi2 to LoFi
(``create_matmul_attributes``), so the config comes with an explicit HiFi2 compute config. The K blocks
accumulate in fp32 Dest (``fp32_dest_acc_en``): with the auto path's packer-L1 accumulation a wide block
sums its tiles in 16-bit Dest and the per-op PCC vs an fp32 reference falls from 0.99995 (auto) to
0.99976 (22-tile block); with fp32 Dest it is 0.99999 for every block width at the same speed.
``GEMMA4_DECODE_MM_FP32_ACC=0`` selects the packer-L1 accumulation instead.

Per-site opt-in flags (all default off): ``GEMMA4_MLP_DECODE_1D`` (shared MLP gate_up / down),
``GEMMA4_ATTN_DECODE_1D`` (wqkv / o_proj), ``GEMMA4_ROUTER_DECODE_1D`` (router projection).
"""

import math
import os

import ttnn

TILE = ttnn.TILE_SIZE


def flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def is_decode_rows(x) -> bool:
    """True for a ``[1, 1, M<=32, K]`` activation (one tile of rows): the only shape the 1D decode config serves."""
    shape = x.shape
    return len(shape) == 4 and int(shape[0]) == 1 and int(shape[1]) == 1 and 1 <= int(shape[2]) <= TILE


def _largest_divisor(n: int, max_div: int) -> int:
    for d in range(max(1, min(n, max_div)), 0, -1):
        if n % d == 0:
            return d
    return 1


def decode_in0_block_w(k: int, max_block_w=None) -> int:
    """K-tile block width: the largest divisor of the K tiles not above ``max_block_w``
    (``GEMMA4_DECODE_MM_IN0_BLOCK_W``, default 8)."""
    if max_block_w is None:
        max_block_w = max(1, int(os.environ.get("GEMMA4_DECODE_MM_IN0_BLOCK_W", "8")))
    return _largest_divisor(math.ceil(k / TILE), max_block_w)


def decode_1d_progcfg(mesh_device, k: int, n: int, max_block_w=None):
    """1D mcast_in0 config for ``[1, 1, <=32, k] @ [k, n]``: one N tile per core across the device's
    worker grid (more when N has more tiles than cores), K streamed in ``decode_in0_block_w`` blocks."""
    grid = mesh_device.compute_with_storage_grid_size()
    n_tiles = math.ceil(n / TILE)
    per_core_n = max(1, math.ceil(n_tiles / (grid.x * grid.y)))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=decode_in0_block_w(k, max_block_w),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def decode_compute_kernel_config(mesh_device):
    """HiFi2 (the auto path's fidelity) with fp32 Dest accumulation across the K blocks (default); the
    auto path's packer-L1 accumulation on ``GEMMA4_DECODE_MM_FP32_ACC=0``."""
    fp32_acc = os.environ.get("GEMMA4_DECODE_MM_FP32_ACC", "1") == "1"
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=not fp32_acc,
    )


class DecodeLinear:
    """``ttnn.linear`` on an interleaved weight with the 1D decode config for one-tile activations and the
    op default for everything else (prefill rows, batched leading dims)."""

    def __init__(self, weight, mesh_device, k: int, n: int, enabled: bool, max_block_w=None):
        self.weight = weight
        self.k = k
        self.n = n
        self.program_config = decode_1d_progcfg(mesh_device, k, n, max_block_w) if enabled else None
        self.compute_kernel_config = decode_compute_kernel_config(mesh_device) if enabled else None

    def __call__(self, x, memory_config=None, compute_kernel_config=None):
        kwargs = {}
        if memory_config is not None:
            kwargs["memory_config"] = memory_config
        if self.program_config is not None and is_decode_rows(x):
            kwargs["program_config"] = self.program_config
            kwargs["compute_kernel_config"] = (
                compute_kernel_config if compute_kernel_config is not None else self.compute_kernel_config
            )
        elif compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = compute_kernel_config
        return ttnn.linear(x, self.weight, **kwargs)
