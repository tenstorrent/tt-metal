# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GDN verify rows with fused input preparation and gated output programs.

The recurrence is the public chunk_gated_delta_rule on the normalized, scaled producer buffers (gdn_public_adapter);
forward and masked commits share it. Persistent buffers belong to the rows state; the body releases only its own
intermediates.
"""

from __future__ import annotations

import os

import ttnn

from . import gdn_post_rows as post
from . import gdn_pre_rows as pre
from . import program as fp
from .gdn_public_adapter import chunk_public
from .registry import BITWISE, FusedKernel, enabled, register

NAME = "gdn_rows_wrap"

TILE = pre.TILE  # 32: the verify tile, one chunk, NC = 1
HEADS = pre.HEADS  # 12 value heads per device
HEAD_DIM = pre.HEAD_DIM  # 128
VALUE_WIDTH = pre.VALUE_WIDTH  # 1536
QKV_WIDTH = pre.QKV_WIDTH  # 2560
PROJECTION_WIDTH = pre.PROJECTION_WIDTH  # 4160
BF16, FP32 = ttnn.bfloat16, ttnn.float32


def _release(*tensors) -> None:
    """Free the body's own intermediates once each; a view whose buffer another handle already freed is skipped."""

    for tensor in tensors:
        if tensor is not None and tensor.is_allocated():
            ttnn.deallocate(tensor)


# ------------------------------------------------------------------------------- the per-layer buffers and constants


class Buffers:
    """The wrap's per-layer tensors, allocated with the rows state and freed with it.

    Persistent because the commit's masked re-run reads ``q_c`` / ``k_c`` / ``beta_c`` / ``g_c`` in a different trace
    from the forward pass that wrote them. ``gdn_pre_rows``'s ``v`` output is ``rows_state.v``.
    """

    __slots__ = (
        "q_c",
        "k_c",
        "beta_c",
        "g_c",
        "sig",
        "pre_constants",
        "pre_selects",
        "pre_scalars",
        "post_scalars",
        "gated_memory_config",
    )

    def __init__(self, *, q_c, k_c, beta_c, g_c, sig, pre_constants, pre_selects, pre_scalars, post_scalars, gated_memory_config):  # fmt: skip
        self.q_c, self.k_c, self.beta_c, self.g_c, self.sig = q_c, k_c, beta_c, g_c, sig
        self.pre_constants, self.pre_selects, self.pre_scalars = pre_constants, pre_selects, pre_scalars
        self.post_scalars = post_scalars
        self.gated_memory_config = gated_memory_config

    @property
    def tensors(self) -> tuple:
        return (
            self.q_c,
            self.k_c,
            self.beta_c,
            self.g_c,
            self.sig,
            self.pre_constants,
            self.pre_selects,
            self.pre_scalars,
            self.post_scalars,
        )

    def deallocate(self) -> None:
        _release(*self.tensors)


def qualifies(rows_state) -> bool:
    """Whether a rows state is the 32-row verify tile in head-major layouts (not a shared slab body); shapes only."""

    constants = getattr(rows_state, "constants", None)
    if constants is None or not getattr(rows_state, "owns_body", True):
        return False
    # The programs read inputs by ``buffer_address()``; host fakes (torch-backed test tensors) have none and keep the
    # chain.
    if not callable(getattr(getattr(rows_state, "v", None), "buffer_address", None)):
        return False
    return getattr(constants, "tile_rows", 0) == TILE and 1 <= getattr(constants, "rows", 0) <= TILE


def attach(gdn, rows_state, environ=None) -> Buffers | None:
    """Allocate the wrap's buffers onto a new rows state; None when the wrap is off or the state does not qualify.

    Called from ``allocate_rows_state`` so the warm pass, the capture and the commit take the same branch as the
    forward pass that wrote the buffers.
    """

    environ = os.environ if environ is None else environ
    if not enabled(NAME, environ) or not qualifies(rows_state):
        return None
    mesh = gdn.mesh_device
    # The pre program's own helper allocates its outputs so specs match its writers; its ``v`` is replaced by the rows
    # state's ``v`` (same spec, already head-shard tagged).
    q_c, k_c, spare_v, beta_c, g_c, sig = pre.allocate_outputs(mesh, TILE)
    _release(spare_v)
    constants, selects, scalars = pre.upload_constants(mesh, gdn.weights.dt_bias, gdn.weights.neg_exp_A)
    buffers = Buffers(
        q_c=q_c,
        k_c=k_c,
        beta_c=beta_c,
        g_c=g_c,
        sig=sig,
        pre_constants=constants,
        pre_selects=selects,
        pre_scalars=scalars,
        post_scalars=post.scalar_tensor(mesh),
        gated_memory_config=gdn.out_proj_act_memory_config,
    )
    rows_state.wrap_buffers = buffers
    return buffers


def buffers_of(rows_state) -> Buffers | None:
    """The wrap's buffers of a rows state, or None when it runs the chain."""

    return getattr(rows_state, "wrap_buffers", None)


# ---------------------------------------------------------------------------------------------- the recurrence


def chunk(gdn, rows_state, buffers, initial_state, committed_mask_c=None):
    """Run the public recurrence on the producer's pages, masking beta and decay for a commit.

    The producer already applies normalization and both query scale folds. Only the temporary masks are released.
    """
    beta_c, g_c, masked = buffers.beta_c, buffers.g_c, ()
    if committed_mask_c is not None:
        dram = ttnn.DRAM_MEMORY_CONFIG
        beta_c = ttnn.multiply(buffers.beta_c, committed_mask_c, memory_config=dram)
        g_c = ttnn.multiply(buffers.g_c, committed_mask_c, memory_config=dram)
        masked = (beta_c, g_c)
    constants = rows_state.constants
    try:
        return chunk_public(
            buffers.q_c,
            buffers.k_c,
            rows_state.v,
            beta_c,
            g_c,
            initial_state,
            (constants.eye, constants.tril, constants.ones, constants.masks),
            rows_total=TILE,
        )
    finally:
        _release(*masked)


# ------------------------------------------------------------------------------------------------- the two bodies


def rows_body_composed(gdn, full_hidden, rows_state, state, *, full_tile: bool = False):
    """The composed rows body from ``ttnn/gdn.py``; every row count the wrap does not admit takes it."""

    z, a, b = gdn._project_rows(full_hidden, rows_state)
    conv = gdn._causal_conv_rows(rows_state)
    gdn._make_chunk_inputs(conv, a, b, rows_state)
    recurrent_output, final_state = gdn._chunk_rows(rows_state, initial_state=state.recurrent)
    output = gdn._gate_and_project_rows(recurrent_output, z, full_hidden, rows_state, full_tile=full_tile)
    return output, final_state


def rows_body_wrap(gdn, full_hidden, rows_state, state, *, full_tile: bool = False):
    """The fused rows body: projection, qkv landing, pre program, the public recurrence (its fused or phased path,
    chosen by the op's cost model), cast, gated norm, out-projection.

    ``rows_state.q`` / ``k`` / ``beta`` / ``g`` stay untouched; the prim layouts live in the wrap's buffers.
    """

    buffers = buffers_of(rows_state)
    rows = rows_state.constants.rows
    projected = gdn._project_rows_linear(full_hidden, rows_state)
    gdn._land_rows_qkv(projected, rows_state)
    pre.run(
        projected,
        rows_state.history,
        gdn.weights.conv_taps,
        buffers.pre_constants,
        buffers.pre_selects,
        buffers.pre_scalars,
        buffers.q_c,
        buffers.k_c,
        rows_state.v,
        buffers.beta_c,
        buffers.g_c,
        buffers.sig,
        rows=rows,
    )
    # Through ``_chunk_rows``: it dispatches back here and applies the head-shard tags and shape checks of both forms.
    recurrent_output, final_state = gdn._chunk_rows(rows_state, initial_state=state.recurrent)
    o16 = fp.allocate((HEADS, TILE, HEAD_DIM), BF16, ttnn.TILE_LAYOUT, gdn.mesh_device)
    post.post_cast(recurrent_output, o16)
    _release(recurrent_output)
    gated = fp.allocate((1, 1, TILE, VALUE_WIDTH), BF16, ttnn.TILE_LAYOUT, gdn.mesh_device, buffers.gated_memory_config)
    post.post_norm(
        o16,
        buffers.sig,
        gdn.weights.norm,
        buffers.post_scalars,
        projected,
        gated,
        None,
        rows=rows,
        history=False,
    )
    _release(o16, projected)
    fp.stamp_topology(gated, rows_state.v, shard_dim=3)
    output = gdn._out_proj_tile(gated, full_hidden)  # consumes the gated tile
    _release(full_hidden)
    return gdn._rows_output_tile(output, rows_state, full_tile=full_tile), final_state


def admits(gdn, full_hidden, rows_state, state, *, full_tile: bool = False) -> bool:
    """Whether the rows state carries the wrap's buffers and the state is the single-lane fp32 recurrent tensor.

    Shapes only; host fakes and every other row count take the composed chain."""

    if buffers_of(rows_state) is None or not qualifies(rows_state):
        return False
    recurrent = getattr(state, "recurrent", None)
    if recurrent is None:
        return False
    try:
        return tuple(recurrent.shape) == (1, HEADS, HEAD_DIM, HEAD_DIM) and recurrent.dtype == FP32
    except (AttributeError, TypeError):
        return False


register(
    FusedKernel(
        name=NAME,
        replaces="the GDN verify-rows preparation and epilogue with gdn_pre_rows, the public chunk recurrence, "
        "post_cast and post_norm",
        tolerance=BITWISE,
        fused=rows_body_wrap,
        composed=rows_body_composed,
        admits=admits,
        gate=None,
    )
)
