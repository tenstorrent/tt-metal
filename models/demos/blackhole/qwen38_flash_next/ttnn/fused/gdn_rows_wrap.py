# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gdn_rows_wrap``: the MTP verify rows' body as the prefill lane's two programs around the two chunk prims.

Stage A of the verify-rows scan design: WRAP the unchanged ``prim::chunk_gdn_prep`` / ``prim::chunk_gdn_scan`` in
``gdn_pre_rows`` (the projection to the prims' inputs) and ``gdn_post_rows`` (the scan's output to the gated tile),
both of which are the chain's own LLK sequences in the chain's DEST widths and are proven BITWISE against the chain
on one die.  One GDN layer's verify body becomes

    projection linear + S2I -> qkv landing slice -> gdn_pre_rows -> chunk_gdn_prep -> chunk_gdn_scan
                            -> post_cast -> post_norm -> out-projection linear + reduce-scatter

The window between the S2I and the out-projection matmul is **6 programs** (the qkv landing slice, ``gdn_pre_rows``,
the two prims, ``post_cast``, ``post_norm``) where the chain runs **53** (the design's programs 3..55: the landing and
z/a/b slices, the FIR row selects and the conv, the GQA expands, both norms, the masks, the beta and decay gates, the
composite's own seven relayout programs, the two prims and the six epilogue programs), so the layer's verify segment
from the all-gather to the reduce-scatter is 11 programs against 58.  The two prims are
the same two device operations the composite launches and they are handed the layouts ``gdn_pre_rows`` already wrote,
so the composite's relayout disappears with the rest of the glue rather than moving to Python as it does in
``gdn_rows_prims_direct`` (step 1 of the same lane, which this form subsumes when both are on).

Class BITWISE, by construction and by each program's own die proof: every arithmetic step is the chain op's LLK
sequence in the chain op's destination width with one pack per op, the row shifts and page maps are data movement,
and the recurrence is the identical pair of prims on identical bytes.

A production default (the registry's ``DEFAULT_ON``) since the served pass pair of 2026-09-25 on the 1x4 line: the
gated rows, o and the final state bitwise the composite's on the die, the pass 7.3-9.8 ms shorter with tokens per pass
and the accept histograms identical.  ``QWEN38_FUSED_OFF=gdn_rows_wrap`` restores the chain (every rows state
allocated under the switch keeps the chain's layouts).  ``QWEN38_FUSED_GDN_ROWS_WRAP_GATED_DRAM=1`` writes
the gated tile to DRAM and moves it into the out-projection's activation shard with one extra program, the fallback
if a sharded output ever refuses the post writer's accessor (the default writes that shard directly, as the chain's
gate multiply and the fused ``gdn_step`` both do).

Admission is a property of the rows state, decided once when it is allocated (``attach``): the 32-row verify form
with the wrap's own buffers attached.  The 128-row long chunk, the prefill slab and the slab's flat-q/k form keep the
chain, so the warm pass and the capture take the same branch on the served tensors.

What the wrap owns and the chain no longer writes: with the wrap on, ``rows_state.q`` / ``k`` / ``beta`` / ``g`` (the
chain's layouts) are never filled -- the prims' layouts live in this module's buffers instead, and ``rows_state.v``
(already the composite's flat-v form) and ``rows_state.qkv`` (the history advance reads it) are shared with the
chain.  The commit's masked re-run therefore runs here too (``chunk``), against the prim-layout committed mask
``selectors.committed_mask_c``; the two 1-row-step anchor switches of ``commit_rows`` read the chain's layouts and
are refused while the wrap owns the buffers.
"""

from __future__ import annotations

import os

import ttnn

from . import gdn_post_rows as post
from . import gdn_pre_rows as pre
from . import program as fp
from .registry import BITWISE, FusedKernel, enabled, register

NAME = "gdn_rows_wrap"
# The fallback switch: write the gated tile to DRAM and move it into the out-projection's activation shard.
GATED_DRAM_ENV = "QWEN38_FUSED_GDN_ROWS_WRAP_GATED_DRAM"

TILE = pre.TILE  # 32: the verify tile, one chunk, NC = 1
HEADS = pre.HEADS  # 12 value heads per device
HEAD_DIM = pre.HEAD_DIM  # 128
VALUE_WIDTH = pre.VALUE_WIDTH  # 1536
QKV_WIDTH = pre.QKV_WIDTH  # 2560
PROJECTION_WIDTH = pre.PROJECTION_WIDTH  # 4160
# The composite's ``scale``; ``_chunk_rows`` passes the same python float.  The prep ignores it unless ``qk_norm``
# (q/k arrive flat), and here they do not: ``gdn_pre_rows`` already folded both scales into q.  It is a COMPILE-TIME
# argument of the prep kernel (``SCALE_BITS``), so the composite's value is what keeps the compiled program the
# composite's; the call is spelled as ``gdn_prefill_rows.chunk_prims`` spells it and the two are pinned together.
SCALE = HEAD_DIM**-0.5
BF16, FP32 = ttnn.bfloat16, ttnn.float32
# The GDN layer's verify segment with the wrap on, for the micro-test's report: the all-gather, the projection
# linear, its S2I, the qkv landing slice, gdn_pre_rows, the two prims, post_cast, post_norm, the out-projection
# linear and the reduce-scatter.  The chain runs 58 of them (the same 5 either side of the 53 it replaces).
PROGRAMS_PER_LAYER = 11
CHAIN_PROGRAMS_PER_LAYER = 58


def _release(*tensors) -> None:
    """Free the body's own intermediates once each; a view whose buffer another handle already freed is skipped."""

    for tensor in tensors:
        if tensor is not None and tensor.is_allocated():
            ttnn.deallocate(tensor)


# ------------------------------------------------------------------------------- the per-layer buffers and constants


class Buffers:
    """The wrap's per-layer tensors, allocated with the rows state and freed with it.

    The five prim-layout outputs of ``gdn_pre_rows`` that are not already rows-state buffers (its ``v`` output IS
    ``rows_state.v``, the composite's flat-v form both paths use), and the per-layer constants of the two programs.
    They are persistent because the commit's masked re-run reads ``q_c`` / ``k_c`` / ``beta_c`` / ``g_c`` in a
    DIFFERENT trace from the forward pass that wrote them; ``o16`` and the gated tile do not outlive one body and are
    allocated per call, as the fused ``gdn_step`` allocates its output.
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
    """Whether a rows state is the form this wrap serves: the 32-row verify tile in the head-major layouts, not the
    slab's flat q/k and not a shared slab body.  Shapes only, read once when the state is allocated."""

    constants = getattr(rows_state, "constants", None)
    if constants is None or getattr(rows_state, "flat_qk", False) or not getattr(rows_state, "owns_body", True):
        return False
    # The two programs read every input by ``buffer_address()``: a rows state whose tensors have none is a host fake
    # (the no-device tests' torch-backed tensors) and keeps the chain, whatever the switch says.
    if not callable(getattr(getattr(rows_state, "v", None), "buffer_address", None)):
        return False
    return getattr(constants, "tile_rows", 0) == TILE and 1 <= getattr(constants, "rows", 0) <= TILE


def gated_memory_config(gdn, environ=None) -> object | None:
    """The memory config the post program writes the gated tile into: the out-projection's activation shard (what the
    chain's gate multiply targets), or None under ``QWEN38_FUSED_GDN_ROWS_WRAP_GATED_DRAM`` -- DRAM plus one
    ``to_memory_config`` into that shard."""

    environ = os.environ if environ is None else environ
    if environ.get(GATED_DRAM_ENV, "") not in ("", "0"):
        return None
    return gdn.out_proj_act_memory_config


def attach(gdn, rows_state, environ=None) -> Buffers | None:
    """Allocate the wrap's buffers onto a newly allocated rows state and return them, or None when the wrap is switched
    off (``QWEN38_FUSED_OFF=gdn_rows_wrap``) or the state is not the form it serves (another row count, the slab's
    forms, a host fake).

    Called from ``allocate_rows_state``, so whether a rows state runs the wrap is fixed before the warm pass: the
    warm rounds and the capture take the same branch, and the commit that reads these buffers cannot find itself on
    the other side of the switch from the forward pass that wrote them.
    """

    environ = os.environ if environ is None else environ
    if not enabled(NAME, environ) or not qualifies(rows_state):
        return None
    mesh = gdn.mesh_device
    # The six output buffers of the pre program at one tile row, allocated by the program's own helper so the shapes,
    # dtypes and memory configs are the ones its writers were proven against; its ``v`` is dropped for the rows
    # state's own ``v``, which is the same tensor spec and already carries the mesh contract's head-shard tag.
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
        gated_memory_config=gated_memory_config(gdn, environ),
    )
    rows_state.wrap_buffers = buffers
    return buffers


def buffers_of(rows_state) -> Buffers | None:
    """The wrap's buffers of a rows state, or None when it runs the chain."""

    return getattr(rows_state, "wrap_buffers", None)


# ---------------------------------------------------------------------------------------------- the recurrence


def chunk(gdn, rows_state, buffers, initial_state, committed_mask_c=None):
    """The two prims on the layouts ``gdn_pre_rows`` wrote: the verify's call, or the commit's masked re-run when
    ``committed_mask_c`` is given (the prim-layout ``[12, 1, 32, 1]`` fp32 mask, so the beta and decay multiplies are
    element for element and no broadcast rule enters a committed state).

    Returns the head-major ``o`` ``[12, 32, 128]`` and the final state ``[1, 12, 128, 128]``, both fp32 TILE in new
    buffers -- the shapes ``_chunk_rows`` checks and retags, so both call sites keep the chain's own validation.
    """

    beta_c, g_c, masked = buffers.beta_c, buffers.g_c, ()
    if committed_mask_c is not None:
        dram = ttnn.DRAM_MEMORY_CONFIG
        beta_c = ttnn.multiply(buffers.beta_c, committed_mask_c, memory_config=dram)
        g_c = ttnn.multiply(buffers.g_c, committed_mask_c, memory_config=dram)
        masked = (beta_c, g_c)
    # [1, 1, 32, 1536] -> the composite's rank-3 flat v (leading unit dims only: a metadata view, no program).
    v_flat = ttnn.reshape(rows_state.v, (1, TILE, VALUE_WIDTH))
    s0 = ttnn.reshape(initial_state, (HEADS, HEAD_DIM, HEAD_DIM))
    constants = rows_state.constants
    prep = ttnn.prim.chunk_gdn_prep(
        buffers.q_c,
        buffers.k_c,
        v_flat,
        g_c,
        beta_c,
        eye=constants.eye,
        tril=constants.tril,
        ones=constants.ones,
        masks=constants.masks,
        chunk_size=TILE,
        scale=SCALE,
        v_flat=True,
        HV=HEADS,
        qk_flat=False,
        Hk=HEADS,
        qk_norm=False,
    )
    scan = ttnn.prim.chunk_gdn_scan(*prep, s0, chunk_size=TILE, output_final_state=True)
    output = ttnn.reshape(scan[0], (HEADS, TILE, HEAD_DIM))
    final_state = ttnn.reshape(scan[1], (1, HEADS, HEAD_DIM, HEAD_DIM))
    _release(*masked, *prep)
    return output, final_state


# ------------------------------------------------------------------------------------------------- the two bodies


def rows_body_composed(gdn, full_hidden, rows_state, state, *, full_tile: bool = False):
    """Today's rows body: the landing slices and the z/a/b slices, the FIR conv, the chunk inputs, the recurrence and
    the gated epilogue with the out-projection (``ttnn/gdn.py``).  The registry's composed chain, and the body every
    row count outside the wrap's admission takes."""

    z, a, b = gdn._project_rows(full_hidden, rows_state)
    conv = gdn._causal_conv_rows(rows_state)
    gdn._make_chunk_inputs(conv, a, b, rows_state)
    recurrent_output, final_state = gdn._chunk_rows(rows_state, initial_state=state.recurrent)
    output = gdn._gate_and_project_rows(recurrent_output, z, full_hidden, rows_state, full_tile=full_tile)
    return output, final_state


def rows_body_wrap(gdn, full_hidden, rows_state, state, *, full_tile: bool = False):
    """The wrapped body: the projection's linear and S2I, the qkv landing the history advance reads, the pre program
    at ``rows`` real rows, the two prims, the cast and the gated norm, then the chain's own out-projection.

    Every program after the S2I and before the out-projection is one of the four this form runs; ``rows_state.q`` /
    ``k`` / ``beta`` / ``g`` stay untouched (the prim layouts are the wrap's buffers) and ``rows_state.v``,
    ``rows_state.qkv`` and ``rows_state.history`` are the chain's own tensors.
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
    # through the layer's own ``_chunk_rows``: it dispatches straight back here (the rows state carries the wrap's
    # buffers) and applies the head-shard tags and the shape checks both rows forms return with.
    recurrent_output, final_state = gdn._chunk_rows(rows_state, initial_state=state.recurrent)
    o16 = fp.allocate((HEADS, TILE, HEAD_DIM), BF16, ttnn.TILE_LAYOUT, gdn.mesh_device)
    post.post_cast(recurrent_output, o16)
    _release(recurrent_output)
    shard = buffers.gated_memory_config
    gated = fp.allocate(
        (1, 1, TILE, VALUE_WIDTH),
        BF16,
        ttnn.TILE_LAYOUT,
        gdn.mesh_device,
        ttnn.DRAM_MEMORY_CONFIG if shard is None else shard,
    )
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
    if shard is None:  # the fallback: one more program moves the tile into the out-projection's activation shard
        interleaved, gated = gated, ttnn.to_memory_config(gated, gdn.out_proj_act_memory_config)
        _release(interleaved)
    fp.stamp_topology(gated, rows_state.v, shard_dim=3)
    output = gdn._out_proj_tile(gated, full_hidden)  # consumes the gated tile
    _release(full_hidden)
    return gdn._rows_output_tile(output, rows_state, full_tile=full_tile), final_state


def admits(gdn, full_hidden, rows_state, state, *, full_tile: bool = False) -> bool:
    """The wrap's input contract for one ``forward_rows`` body: the rows state carries the wrap's buffers (which
    ``attach`` gives only the 32-row verify form, and only when the kernel is on) and the call's state is the
    single-lane fp32 recurrent tensor.  Shapes only: host fakes and every other row count take the chain."""

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
        replaces="the GDN verify-rows body between the projection's sharded-to-interleaved and the out-projection "
        "matmul (53 programs: the landing and z/a/b slices, the FIR row selects, conv and SiLU, the GQA expands, both "
        "q/k norms and scales, the v/beta/g row masks and gates, the composite's seven relayout programs, the two "
        "chunk prims and the six epilogue programs) as gdn_pre_rows + chunk_gdn_prep + chunk_gdn_scan + post_cast + "
        "post_norm",
        tolerance=BITWISE,
        fused=rows_body_wrap,
        composed=rows_body_composed,
        admits=admits,
        # no captured-input gate: the proof is each program's own device test against the chain on one die, the rows
        # micro-test's wrap arm (the gated tile, o and the final state bitwise the composite's, eager and replayed, at
        # every commit mask; 2026-09-25 on the line) and the served pass pair (tokens per pass and the accept
        # histograms identical, the pass 7.3-9.8 ms shorter)
        gate=None,
    )
)
