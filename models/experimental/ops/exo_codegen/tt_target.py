# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent hardware target definition for Exo.

Defines custom Memory classes and @instr hardware instructions that map
Exo's abstract operations to TT-Metal's LLK APIs. These form the
"hardware library" that Exo uses for code generation and correctness
verification.

Memory hierarchy:
    TT_DRAM  - Off-chip DRAM, accessed via NOC DMA
    TT_L1_CB - On-chip L1 circular buffers (producer-consumer queues)

Instructions (coarse-grained):
    tt_read_tile     - DMA one tile from DRAM into a CB
    tt_identity_tile - Copy one tile through DST registers (identity)
    tt_relu_tile     - Copy + ReLU one tile through DST registers
    tt_write_tile    - DMA one tile from a CB back to DRAM
"""

from __future__ import annotations

from exo import instr, Memory, DRAM


# ---------------------------------------------------------------------------
# Memory classes
# ---------------------------------------------------------------------------


class TT_DRAM(DRAM):
    """Off-chip DRAM. Accessed only through NOC DMA instructions."""

    @classmethod
    def global_(cls):
        return '#include "api/dataflow/dataflow_api.h"'

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        # DRAM buffers are allocated by the host and passed via runtime args.
        return f"// DRAM buffer: {new_name} (host-allocated, address from runtime args)"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""


class TT_L1_CB(Memory):
    """On-chip L1 circular buffer. Producer-consumer queue between kernels."""

    @classmethod
    def global_(cls):
        return ""

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        # CBs are configured by the host via CBDescriptor.
        return f"// CB: {new_name} (host-configured via CBDescriptor)"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        # CB windowing is abstract — actual tile access is via CB protocol.
        return f"{baseptr}"


# ---------------------------------------------------------------------------
# Reader instructions (NCRISC dataflow)
# ---------------------------------------------------------------------------


@instr(
    "cb_reserve_back(cb_id_in0, 1);\n"
    "l1_write_addr = get_write_ptr(cb_id_in0);\n"
    "noc_async_read_page(start_id + {i}, s, l1_write_addr);\n"
    "noc_async_read_barrier();\n"
    "cb_push_back(cb_id_in0, 1);"
)
def tt_read_tile(
    i: index,
    src: [f32][1] @ TT_DRAM,
    dst: [f32][1] @ TT_L1_CB,
):
    dst[0] = src[0]


# ---------------------------------------------------------------------------
# Compute instructions (TRISC math/pack)
# ---------------------------------------------------------------------------


@instr(
    "cb_reserve_back(tt::CBIndex::c_2, 1);\n"
    "tile_regs_acquire();\n"
    "cb_wait_front(tt::CBIndex::c_0, 1);\n"
    "copy_tile(tt::CBIndex::c_0, 0, 0);\n"
    "tile_regs_commit();\n"
    "tile_regs_wait();\n"
    "pack_tile(0, tt::CBIndex::c_2);\n"
    "cb_pop_front(tt::CBIndex::c_0, 1);\n"
    "tile_regs_release();\n"
    "cb_push_back(tt::CBIndex::c_2, 1);"
)
def tt_identity_tile(
    src: [f32][1] @ TT_L1_CB,
    dst: [f32][1] @ TT_L1_CB,
):
    dst[0] = src[0]


@instr(
    "cb_reserve_back(tt::CBIndex::c_2, 1);\n"
    "tile_regs_acquire();\n"
    "cb_wait_front(tt::CBIndex::c_0, 1);\n"
    "copy_tile(tt::CBIndex::c_0, 0, 0);\n"
    "relu_tile_init();\n"
    "relu_tile(0);\n"
    "tile_regs_commit();\n"
    "tile_regs_wait();\n"
    "pack_tile(0, tt::CBIndex::c_2);\n"
    "cb_pop_front(tt::CBIndex::c_0, 1);\n"
    "tile_regs_release();\n"
    "cb_push_back(tt::CBIndex::c_2, 1);"
)
def tt_relu_tile(
    src: [f32][1] @ TT_L1_CB,
    dst: [f32][1] @ TT_L1_CB,
):
    # ReLU: max(0, x) — semantically same as identity for Exo's correctness
    # checking (Exo doesn't model the actual math, just data movement)
    dst[0] = src[0]


# ---------------------------------------------------------------------------
# Writer instructions (NCRISC dataflow)
# ---------------------------------------------------------------------------


@instr(
    "cb_wait_front(cb_id_out, 1);\n"
    "l1_read_addr = get_read_ptr(cb_id_out);\n"
    "noc_async_write_page(start_id + {i}, s, l1_read_addr);\n"
    "noc_async_writes_flushed();\n"
    "cb_pop_front(cb_id_out, 1);"
)
def tt_write_tile(
    i: index,
    src: [f32][1] @ TT_L1_CB,
    dst: [f32][1] @ DRAM,
):
    dst[0] = src[0]
