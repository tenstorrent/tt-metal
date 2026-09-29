# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gdn_rows_prims_direct``: the GDN rows recurrence as the two chunk phase prims called directly (opt-in, BITWISE).

``Qwen38TTNNGDN._chunk_rows`` (``forward_rows`` and the commit's masked re-run, ttnn/gdn.py) runs the chunked gated
delta rule over one 32-row tile through the composite ``ttnn.transformer.chunk_gated_delta_rule``.  Inside, the
composite lays the inputs of its two device operations out itself (``chunk_gated_delta_rule.cpp``: ``head_split_tile``
on q and k, ``headvec_split_tile`` on g and beta, the ``scale`` fold into q, the per-chunk reshapes, the state reshape)
and launches ``prim::chunk_gdn_prep`` then ``prim::chunk_gdn_scan``.  This form makes the same calls from Python, one
for one: the composite's relayout ops in its order on the tensors ``_chunk_rows`` hands over, then the bound prims
``ttnn.prim.chunk_gdn_prep`` / ``ttnn.prim.chunk_gdn_scan`` with the tensors, constants and keyword arguments the
composite passes (its DRAM output and its HiFi4 / fp32-accumulate kernel config are the bindings' defaults), so o and
the final state are the composite's bytes.  It adds no kernel and removes no program: the seven relayout programs the
composite ran stand in Python, where the verify-rows programs of the design's Stage A (the prims' layouts written by the
rows pre-program) take them over one at a time and are proven against this seam.  Off by default;
``QWEN38_FUSED=gdn_rows_prims_direct`` switches it on.

Admission is shape-only (the warm pass and the capture take one branch; host fakes take the composite): the head-major
rows form at one 32-row tile as ``_chunk_rows`` builds it (q / k ``[1, 32, 12, 128]`` bf16, v flat ``[1, 32, 1536]``
bf16, g / beta ``[1, 32, 12]`` fp32, the state ``[1, 12, 128, 128]`` fp32, the four constant tiles).  The slab's flat
q/k form (the composite's in-kernel norm) and the 128-row long chunk keep the composite.
"""

from __future__ import annotations

import ttnn

from .registry import BITWISE, FusedKernel, register

NAME = "gdn_rows_prims_direct"
HEADS = 12  # HV, the value heads per device; q/k arrive GQA-expanded to them, so H = HV and BH = B * HV with B = 1
HEAD_DIM = 128  # K = V
TILE = ttnn.TILE_SIZE  # C: the chunk is the one 32-row tile, NC = 1, no time padding
VALUE_WIDTH = HEADS * HEAD_DIM
# the composite's ``scale``: ``_chunk_rows`` passes HEAD_DIM**-0.5 (the same python float, so the same fp32 scalar)
SCALE = HEAD_DIM**-0.5
BF16, FP32 = ttnn.bfloat16, ttnn.float32


def _is(tensor, shape: tuple[int, ...], dtype) -> bool:
    try:
        return tuple(tensor.shape) == shape and tensor.dtype == dtype and tensor.layout == ttnn.TILE_LAYOUT
    except (AttributeError, TypeError):
        return False


def admits(gdn, q_rows, k_rows, v_rows, g_rows, beta_rows, initial_state, constants) -> bool:
    """The form's input contract for one ``_chunk_rows`` call: the head-major q/k of one 32-row tile, the flat v, the
    fp32 g/beta columns, the fp32 state of one lane and the four constant tiles (what the composite's flat-v,
    head-major, ``qk_norm`` off branch consumes at NC = 1).  Shapes and dtypes only: the served rows tensors satisfy it,
    the slab's rank-3 flat q/k and the 128-row long chunk do not and keep the composite."""

    return (
        _is(q_rows, (1, TILE, HEADS, HEAD_DIM), BF16)
        and _is(k_rows, (1, TILE, HEADS, HEAD_DIM), BF16)
        and _is(v_rows, (1, TILE, VALUE_WIDTH), BF16)
        and _is(g_rows, (1, TILE, HEADS), FP32)
        and _is(beta_rows, (1, TILE, HEADS), FP32)
        and _is(initial_state, (1, HEADS, HEAD_DIM, HEAD_DIM), FP32)
        and all(getattr(constants, name, None) is not None for name in ("eye", "tril", "ones", "masks"))
    )


def chunk_rows_composite(gdn, q_rows, k_rows, v_rows, g_rows, beta_rows, initial_state, constants):
    """Today's call: the composite over the one chunk (ttnn/gdn.py ``_chunk_rows_composite``)."""

    return gdn._chunk_rows_composite(q_rows, k_rows, v_rows, g_rows, beta_rows, initial_state, constants)


def _release(*tensors) -> None:
    """Free the form's own intermediates once each: a view whose buffer another handle already freed is skipped."""

    for tensor in tensors:
        if tensor.is_allocated():
            ttnn.deallocate(tensor)


def chunk_rows_prims(gdn, q_rows, k_rows, v_rows, g_rows, beta_rows, initial_state, constants):
    """The composite's call as its own ops from Python (``chunk_gated_delta_rule.cpp``, ``chunk_gated_delta_rule``
    with B = 1, T = C = 32, H = HV = 12, K = V = 128, flat v, head-major q/k, so ``qk_norm`` off and no time padding):

    * ``head_split_tile(q_in, B, T, H, K)``: q is bf16 already; ``permute {0, 2, 1, 3}`` on TILE, then the view
      ``[B * H, T, K]``; the same for k.
    * v arrives rank 3 (the flat token-major form) and bf16: it passes as it is.
    * ``headvec_split_tile(g_in, B, T, HV)``: g is fp32 already; ``permute {0, 2, 1}`` on TILE, then the view
      ``[B * HV, T]``; the same for beta.
    * ``G = HV / H = 1``: no ``repeat_interleave``.  ``qk_norm`` is false (q/k are not flat): ``q = multiply(q, scale)``,
      a bf16 pack; the composite's reassignment releases the permuted q there.
    * ``pad = 0``: no time padding.  ``to_chunks_tile``: the views ``[BH, NC, C, D]`` of q and k; g and beta to
      ``[BH, NC, C, 1]`` (one column tile per head: a relayout program each).
    * the initial state is fp32 already: the view ``[BH, K, V]`` of the committed state (never freed here).
    * ``prim::chunk_gdn_prep(q_c, k_c, v_c, g_c, beta_c, eye, tril, ones, masks, C, out_mem, kernel_cfg, flat_v, HV,
      qk_norm, scale, flat_qk, H)``: the binding's ``memory_config`` / ``compute_kernel_config`` left unset ARE the
      composite's ``out_mem`` (DRAM) and ``kernel_cfg`` (HiFi4, approximations off, fp32 accumulate, L1 accumulate off).
    * ``scale`` is a COMPILE-TIME argument of the prep kernel (``SCALE_BITS``, read only under ``QK_NORM``): with
      ``qk_norm`` off it multiplies nothing, and only the composite's value keeps the compiled program the composite's
      (another value would compile a different program).  The prefill slab's ``gdn_prefill_rows.chunk_prims`` spells
      this call the same way -- positional q, k, v, g, beta, the seven hand-offs then the state view positional to
      the scan -- and the two are pinned against each other.
    * ``prim::chunk_gdn_scan(prep[0..6], s0, C, output_final_state = true, out_mem, kernel_cfg)``.
    * ``output_head_major`` with ``pad = 0``: the fold ``NC, C -> T`` is the view ``[BH, T, V]``; the final state's
      view ``[B, HV, K, V]``.

    Returns the head-major o ``[12, 32, 128]`` and the final state ``[1, 12, 128, 128]``, fp32 TILE in new buffers,
    as the composite returns them; ``_chunk_rows`` retags and checks both."""

    q_head_major = ttnn.permute(q_rows, (0, 2, 1, 3))
    q_heads = ttnn.reshape(q_head_major, (HEADS, TILE, HEAD_DIM))
    k_head_major = ttnn.permute(k_rows, (0, 2, 1, 3))
    k_heads = ttnn.reshape(k_head_major, (HEADS, TILE, HEAD_DIM))
    g_head_major = ttnn.permute(g_rows, (0, 2, 1))
    g_heads = ttnn.reshape(g_head_major, (HEADS, TILE))
    beta_head_major = ttnn.permute(beta_rows, (0, 2, 1))
    beta_heads = ttnn.reshape(beta_head_major, (HEADS, TILE))
    q_scaled = ttnn.multiply(q_heads, SCALE)
    ttnn.deallocate(q_head_major)
    q_c = ttnn.reshape(q_scaled, (HEADS, 1, TILE, HEAD_DIM))
    k_c = ttnn.reshape(k_heads, (HEADS, 1, TILE, HEAD_DIM))
    g_c = ttnn.reshape(g_heads, (HEADS, 1, TILE, 1))
    beta_c = ttnn.reshape(beta_heads, (HEADS, 1, TILE, 1))
    s0 = ttnn.reshape(initial_state, (HEADS, HEAD_DIM, HEAD_DIM))
    prep = ttnn.prim.chunk_gdn_prep(
        q_c,
        k_c,
        v_rows,
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
    _release(k_head_major, q_scaled, g_head_major, beta_head_major, g_c, beta_c, *prep)
    return output, final_state


register(
    FusedKernel(
        name=NAME,
        replaces="the GDN rows recurrence's composite ttnn.transformer.chunk_gated_delta_rule call (forward_rows and the "
        "commit's masked re-run, one 32-row tile): the composite's own relayout ops and its two phase prims, called "
        "directly as ttnn.prim.chunk_gdn_prep / chunk_gdn_scan",
        tolerance=BITWISE,
        fused=chunk_rows_prims,
        composed=chunk_rows_composite,
        admits=admits,
        # no captured-input gate: the proof is the rows micro-test's prims arm (o and the final state bitwise the
        # composite's on the die, unmasked and at every commit mask, eager and replayed)
        gate=None,
    )
)
