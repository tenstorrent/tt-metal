# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GDN prefill slab with fused input preparation and gated output programs.

Samuel Jett's producer preserves the source normalization, query scaling and
chunk-major bytes. The model-owned recurrence retains the source intermediate rounding sequence
through the current public generic_op API.
The existing projection, output projection, reduce-scatter and state ownership
remain in Qwen38TTNNGDN. Source-era timing and program counts in docs/PREFILL.md
are historical measurements, not measurements of this adapter.

The registry admits this body only for a slab carrying the fused buffers.
The composed control and every other row shape retain their normal dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ttnn

from . import gdn_post_rows, gdn_pre_rows
from . import program as fp
from .gdn_rows_reference import A_COLUMN, HEAD_DIM, HEADS, PROJECTION_WIDTH, QKV_WIDTH, TILE, VALUE_WIDTH
from .gdn_source_chunk import chunk_source
from .registry import BITWISE, FusedKernel, register

NAME = "gdn_prefill_rows"
CHUNK = TILE  # the chunk kernel's C: the rows path runs one tile row per chunk
FP32, BF16 = ttnn.float32, ttnn.bfloat16
DRAM = ttnn.DRAM_MEMORY_CONFIG
# The slab row counts this form serves (ttnn/contracts.is_slab_rows, repeated here so the module stays free of the
# model package's imports -- ttnn/gdn.py imports this one).
MIN_SLAB_ROWS, MAX_SLAB_ROWS, SLAB_ROW_STEP = 256, 4096, 128


def is_slab_rows(rows) -> bool:
    """A prefill slab row count: a multiple of 128 in 256 .. 4096 (never 32 or 128)."""

    return (
        not isinstance(rows, bool)
        and type(rows) is int
        and MIN_SLAB_ROWS <= rows <= MAX_SLAB_ROWS
        and rows % SLAB_ROW_STEP == 0
    )


def buffer_layouts(tile_rows: int) -> dict[str, tuple[tuple[int, ...], Any, int]]:
    """The per-device local shape, dtype and TP shard dim of every buffer the form adds or re-shapes, by rows-state
    field name.

    ``q`` / ``k`` / ``beta`` / ``g`` take the prims' pad-free per-chunk pages in place of the chain's padded
    token-major head tensors (``[1, T, 12, 128]``, 16.8 MB each) and the token-major ``[1, 1, T, 12]`` gate columns;
    ``sig``, ``o16``, ``gated`` and ``history_next`` are new.  ``v`` and ``output`` are unchanged and not listed.
    """

    chunks = tile_rows // CHUNK
    return {
        "q": ((HEADS, chunks, CHUNK, HEAD_DIM), BF16, 0),
        "k": ((HEADS, chunks, CHUNK, HEAD_DIM), BF16, 0),
        "beta": ((HEADS, chunks, CHUNK, 1), FP32, 0),
        "g": ((HEADS, chunks, CHUNK, 1), FP32, 0),
        "sig": ((1, 1, tile_rows, VALUE_WIDTH), BF16, 3),
        "o16": ((HEADS, tile_rows, HEAD_DIM), BF16, 0),
        "gated": ((1, 1, tile_rows, VALUE_WIDTH), BF16, 3),
        "history_next": ((1, 1, TILE, QKV_WIDTH), BF16, 3),
    }


# ------------------------------------------------------------------------------------ the layer's constant tiles


@dataclass(frozen=True)
class RowsConstants:
    """The constant tiles the two programs read, uploaded once per layer and resident before a trace capture (the
    ``gdn_step`` pattern: built on the first eager call and cached on the module).

    ``gates`` carries this device's own ``dt_bias`` / ``neg_exp_A`` row as two full fp32 tiles; ``selects`` the nine
    0/1 row-shift tiles; ``scalars`` the pre program's three small bf16 tiles (the layernorm reduce scaler, the
    truncated epsilon, the host-rounded q/k scale); ``norm_scalars`` the post program's two (the scaler and the
    gated norm's epsilon).
    """

    gates: Any
    selects: Any
    scalars: Any
    norm_scalars: Any


def upload_constants(mesh, dt_bias, neg_exp_A) -> RowsConstants:
    """Every constant tile the pair reads, for one layer's device tensors."""

    gates, selects, scalars = gdn_pre_rows.upload_constants(mesh, dt_bias, neg_exp_A)
    return RowsConstants(gates, selects, scalars, gdn_post_rows.scalar_tensor(mesh))


def constants(gdn) -> RowsConstants:
    """The layer's cached constants, built on the first (eager) call.  A trace capture replays a program whose
    constants are already resident, never an upload."""

    tiles = getattr(gdn, "_fused_gdn_prefill_rows_constants", None)
    if tiles is None:
        tiles = upload_constants(gdn.mesh_device, gdn.weights.dt_bias, gdn.weights.neg_exp_A)
        gdn._fused_gdn_prefill_rows_constants = tiles
    return tiles


# ------------------------------------------------------------------------------------------------- the two prims


def chunk_prims(q_c, k_c, v, beta_c, g_c, initial_state, chunk_tiles, *, rows_total: int):
    """Run the source numerical sequence on pre-normalized, twice-scaled pages.

    Persistent producers/constants/state stay caller-owned. Intermediate pack
    boundaries retain Samuel Jett's frozen source words on the current runtime.
    """
    return chunk_source(q_c, k_c, v, beta_c, g_c, initial_state, chunk_tiles, rows_total=rows_total)


# -------------------------------------------------------------------------------------------- the slab body


def restamp_written(buffers, reference, names: tuple[str, ...]) -> None:
    """Give back their declared distributed topology to the persistent buffers a program has just written.

    ``ttnn.generic_op`` leaves its output tensors with the allocation's default topology (``PlacementShard(0)`` on
    the four-die line) instead of the buffer's declared one, exactly as ``fp.allocate`` does; the chain's ops with an
    ``output_tensor`` keep the target's topology, so the rows state's placement checks (``validate``, the mesh contract)
    hold for the chain and, on one die where every check is vacuous, for the programs too.  On the line they failed on
    the first buffer declared on dim 3 that a program wrote (``sig``: "head_sharded tensor shards dim 0, expected 3").
    Every buffer a program writes is re-stamped after that program from ``buffer_layouts`` (``v`` keeps the chain's
    token-major dim 3), the way the other fused kernels stamp their outputs.
    """

    layouts = buffer_layouts(int(reference.shape[-2]))
    for name in names:
        dim = 3 if name == "v" else layouts[name][2]
        fp.stamp_topology(getattr(buffers, name), reference, dim)


PRE_WRITES = ("q", "k", "v", "beta", "g", "sig")
CAST_WRITES = ("o16",)
NORM_WRITES = ("gated", "history_next")


def slab_body(projected, history, taps, norm, chunk_tiles, tiles: RowsConstants, buffers, initial_state, *, rows: int):
    """The fused body on explicit tensors: projection -> gated rows, history tile and final recurrent state.

    ``projected`` ``[1, 1, T, 4160]`` bf16 (the slab's whole projection, read in place), ``history``
    ``[1, 1, 32, 2560]`` bf16, ``taps`` the four ``[1, 1, 1, 2560]`` conv weights, ``norm`` ``[1, 1, 1, 128]`` bf16,
    ``chunk_tiles`` the prims' ``(eye, tril, ones, masks)``, ``buffers`` an object with the fields
    :func:`buffer_layouts` names plus ``v``, ``initial_state`` ``[1, 12, 128, 128]`` fp32.

    Returns ``(gated [1, 1, T, 1536] bf16, final_state [1, 12, 128, 128] fp32, history_next [1, 1, 32, 2560] bf16)``;
    ``gated`` and ``history_next`` are the caller's persistent buffers, ``final_state`` a new one.
    """

    if tuple(projected.shape)[:2] != (1, 1) or int(projected.shape[-1]) != PROJECTION_WIDTH:
        raise ValueError(
            f"gdn_prefill_rows reads the whole projection [1, 1, T, {PROJECTION_WIDTH}], got {tuple(projected.shape)}"
        )
    rows_total = int(projected.shape[-2])
    gdn_pre_rows.run(
        projected,
        history,
        taps,
        tiles.gates,
        tiles.selects,
        tiles.scalars,
        buffers.q,
        buffers.k,
        buffers.v,
        buffers.beta,
        buffers.g,
        buffers.sig,
        rows=rows,
    )
    restamp_written(buffers, projected, PRE_WRITES)
    o, final_state = chunk_prims(
        buffers.q, buffers.k, buffers.v, buffers.beta, buffers.g, initial_state, chunk_tiles, rows_total=rows_total
    )
    gdn_post_rows.post_cast(o, buffers.o16)
    restamp_written(buffers, projected, CAST_WRITES)
    ttnn.deallocate(o)
    gated, history_next = gdn_post_rows.post_norm(
        buffers.o16,
        buffers.sig,
        norm,
        tiles.norm_scalars,
        projected,
        buffers.gated,
        buffers.history_next,
        rows=rows,
        history=True,
    )
    restamp_written(buffers, projected, NORM_WRITES)
    return gated, final_state, history_next


def slab_body_composed(projected, history, taps, norm, chunk_tiles, dt_bias, neg_exp_A, initial_state, *, rows: int):
    """Today's chain over the same span, op for op -- the device test's bitwise oracle, not a production path.

    The two programs' own chain transcriptions (``gdn_pre_rows.chain_on_device`` and
    ``gdn_post_rows.chain_on_device``, each taken op for op from ``ttnn/gdn.py``) around :func:`chunk_prims`.  The
    prims are the same two calls in both arms because this form does not change them: their equivalence to
    ``ttnn.transformer.chunk_gated_delta_rule`` -- the argument set, the flags and ``scale`` -- is the separate pin
    of the prims' device test (a dev test) and of this form's own device test, on the composite itself.
    What this oracle isolates is the wiring: that the pre program's pages are the bits the chain's relayout writes
    and that the post programs consume the scan's output the way the chain does.

    Returns the same triple as :func:`slab_body`, all in new buffers.
    """

    mesh = projected.device()
    rows_total = int(projected.shape[-2])
    q_c, k_c, v, beta_c, g_c, _sig = gdn_pre_rows.chain_on_device(
        mesh, projected, history, taps, dt_bias, neg_exp_A, rows=rows
    )
    o, final_state = chunk_prims(q_c, k_c, v, beta_c, g_c, initial_state, chunk_tiles, rows_total=rows_total)
    z = ttnn.slice(projected, (0, 0, 0, QKV_WIDTH), (1, 1, rows_total, A_COLUMN), memory_config=DRAM)
    gated, history_next, sigmoid_bf16 = gdn_post_rows.chain_on_device(mesh, o, z, norm, projected)
    for tensor in (q_c, k_c, v, beta_c, g_c, _sig, o, z, sigmoid_bf16):
        ttnn.deallocate(tensor)
    return gated, final_state, history_next


# ------------------------------------------------------------------------------------------- the registry entry


def admits(gdn, full_hidden, rows_state, initial_state, *, full_tile: bool = False) -> bool:
    """The form's input contract for one call: a slab rows state whose pass buffers are the fused ones, the chunk
    constant tiles present, ``flat_qk`` off and no ``full_tile`` (a 32-row option).

    Read as a few shape and flag tests, so a traced capture keeps whichever branch its real tensors took; host fakes
    outside the contract take the composed chain (``registry.resolve_admitted``).
    """

    if full_tile:
        return False
    constants_of = getattr(rows_state, "constants", None)
    if constants_of is None or not is_slab_rows(getattr(constants_of, "rows", None)):
        return False
    if getattr(rows_state, "flat_qk", False) or not getattr(rows_state, "fused_prefill", False):
        return False
    if any(getattr(constants_of, name, None) is None for name in ("eye", "tril", "ones", "masks")):
        return False
    tile_rows = getattr(constants_of, "tile_rows", None)
    if tile_rows != constants_of.rows:
        return False
    layouts = buffer_layouts(tile_rows)
    layouts["v"] = ((1, 1, tile_rows, VALUE_WIDTH), BF16, 3)
    for name, (shape, dtype, _shard) in layouts.items():
        tensor = getattr(rows_state, name, None)
        if tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype:
            return False
    return (
        initial_state is not None
        and tuple(initial_state.shape) == (1, HEADS, HEAD_DIM, HEAD_DIM)
        and initial_state.dtype == FP32
    )


def gdn_prefill_rows(gdn, full_hidden, rows_state, initial_state, *, full_tile: bool = False):
    """The slab GDN body: the chain's projection linear, the fused pair around the prims, the chain's out-projection.

    Returns ``(rows_state.output, final_state)`` -- what ``forward_rows``' slab branch returns.  ``history_next``
    stays in the rows state for ``commit_rows_full`` to copy into the shared history buffer (``forward_rows`` must
    leave that buffer alone: the slab rows state shares it with the layer's 32-row chunk state).
    """

    constants_of = rows_state.constants
    projected = gdn._slab_projection(full_hidden, rows_state)
    gated, final_state, _history_next = slab_body(
        projected,
        rows_state.history,
        gdn.weights.conv_taps,
        gdn.weights.norm,
        (constants_of.eye, constants_of.tril, constants_of.ones, constants_of.masks),
        constants(gdn),
        rows_state,
        initial_state,
        rows=constants_of.rows,
    )
    ttnn.deallocate(projected)
    fp.stamp_topology(final_state, initial_state, 1)
    fp.stamp_topology(gated, full_hidden, 3)
    output = gdn._slab_out_projection(gated, full_hidden, rows_state)
    return output, final_state


def gdn_prefill_rows_composed(gdn, full_hidden, rows_state, initial_state, *, full_tile: bool = False):
    """Today's chain: the four ``ttnn/gdn.py`` calls of ``forward_rows``, in their order and with their arguments."""

    z, a, b = gdn._project_rows(full_hidden, rows_state)
    conv = gdn._causal_conv_rows(rows_state)
    gdn._make_chunk_inputs(conv, a, b, rows_state)
    recurrent_output, final_state = gdn._chunk_rows(rows_state, initial_state=initial_state)
    output = gdn._gate_and_project_rows(recurrent_output, z, full_hidden, rows_state, full_tile=full_tile)
    return output, final_state


register(
    FusedKernel(
        name=NAME,
        replaces="the prefill slab's GDN glue: the projection's landing slices, the FIR row shifts, the causal "
        "convolution and SiLU, the q/k GQA expand, both rms_norms and scales, the v row mask, the beta and "
        "log-decay gates, the composite's head-major relayout and q scale, the gated epilogue's typecast, weighted "
        "norm, head fold and sigmoid gate, and the next pass's history tile (about 150 programs per slab layer)",
        tolerance=BITWISE,
        fused=gdn_prefill_rows,
        composed=gdn_prefill_rows_composed,
        admits=admits,
        # The component gate's capture carries no GDN recurrent state, so this form's proofs are the two programs'
        # one-die microtests (bitwise against the chain run on the same die) and the line's slab gates.
        gate=None,
    )
)
