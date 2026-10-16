# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM-streaming expert path for MiMo MoE decode (a ``ttnn.generic_op`` program; not wired in yet).

Blackhole streams DRAM fastest when each bank is read by the worker core next to it, in large
contiguous blocks with several reads in flight. These helpers are the starting point for a program that
computes every (user, expert) pair of a decode step on the eight bank cores; its device kernels live in
``tt/kernels/`` and are JIT-compiled from the checkout.
"""

import torch

import ttnn

from .common import TILE

KERNEL_DIR = "models/autoports/xiaomimimo_mimo_v2_6_flash_rl/tt/kernels"


def dram_bank_cores(device):
    """One worker core per DRAM bank (NOC 0 optimal placement): core ``i`` streams bank ``i``."""
    return device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)


def zero_output(plan, hidden, rows=TILE):
    """Persistent zero ``[1, 1, rows, hidden]`` bf16 tile tensor on the row, for a kernel that writes only user rows."""
    return ttnn.from_torch(
        torch.zeros(1, 1, rows, hidden),
        device=plan.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=plan.replicate(),
    )
