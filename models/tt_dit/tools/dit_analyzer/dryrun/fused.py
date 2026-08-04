# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Fused-kernel internal stages, as data (roadmap blocker 18).

A fused ttnn kernel runs a collective *inside* a compute op: an AGMM gathers its
activation before the matmul; an MMRS reduce-scatters the partial sums after; a
ring joint SDPA gathers K/V mid-kernel. The dry run must expand each into its
stages as separate IR nodes (tagged with a shared ``fused_in``), or the hidden
collective is invisible to the analyzer and a redundant gather inside a kernel is
never flagged.

Keeping *which* kernels do this and *what* they hide as a table -- rather than
buried in imperative shape rules -- buys three things:

* a new fused kernel of a known shape is one table entry driving a shared builder;
* the set of collective-hiding kernels is inspectable (``ditcheck ops --fused``);
* an *unregistered* op whose name matches the pattern (:func:`looks_fused`) can be
  flagged as a likely collective-hiding kernel instead of silently passing through
  as "output equals input 0".

This module is pure data (no ttnn, no shim), so the CLI can read it without a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class FusedKernel:
    """One fused ttnn kernel and the stages it expands into."""

    call: str  # ttnn call name (leaf)
    tag: str  # short prefix for the shared fused_in tag ("agmm", "mmrs", ...)
    stages: Tuple[str, ...]  # ordered canonical stage ops, e.g. ("all_gather", "matmul")
    collective: str  # the internal collective it hides (all_gather / reduce_scatter / ...)
    order: str = ""  # builder shape: "gather_then_matmul" | "matmul_then_scatter" | "" (custom)
    chunked: bool = False  # the matmul is split into `chunks` output-column blocks
    epilogue: bool = False  # an optional fused pointwise (addcmul) tail
    doc: str = ""


FUSED_KERNELS = {
    k.call: k
    for k in (
        FusedKernel(
            call="all_gather_minimal_matmul_async",
            tag="agmm",
            stages=("all_gather", "matmul"),
            collective="all_gather",
            order="gather_then_matmul",
            chunked=True,
            epilogue=True,
            doc="AGMM: gather the activation over cluster_axis, then (chunked) matmul.",
        ),
        FusedKernel(
            call="minimal_matmul_strided_reduce_scatter_async",
            tag="mmrs",
            stages=("matmul", "reduce_scatter"),
            collective="reduce_scatter",
            order="matmul_then_scatter",
            epilogue=True,
            doc="MMRS: matmul, then reduce-scatter the partial sums over cluster_axis.",
        ),
        # Custom builders (order=""), listed here so they are inspectable and so
        # looks_fused's guard is calibrated against the real set.
        FusedKernel(
            call="ring_joint_scaled_dot_product_attention",
            tag="ring_sdpa",
            stages=("all_gather", "sdpa"),
            collective="all_gather",
            doc="Ring joint SDPA: gather K/V over cluster_axis inside the kernel, then attend.",
        ),
        FusedKernel(
            call="exp_ring_joint_scaled_dot_product_attention",
            tag="ring_sdpa",
            stages=("all_gather", "sdpa"),
            collective="all_gather",
            doc="Experimental entry point for ring joint SDPA; same internal stages.",
        ),
    )
}


def looks_fused(call: str) -> Optional[str]:
    """For an *unregistered* op: does the name look like a collective-hiding kernel?

    A heuristic on the call name -- a compute word (matmul / sdpa / attention)
    next to a collective word -- so ``ditcheck ops --missing`` can warn that an
    unmodelled op probably hides a collective, rather than letting it pass through
    as an ordinary node. Returns the suspected collective, or ``None``.
    """
    name = call.rsplit(".", 1)[-1]
    if name in FUSED_KERNELS:
        return FUSED_KERNELS[name].collective
    compute = any(w in name for w in ("matmul", "linear", "sdpa", "attention"))
    if not compute:
        return None
    if "reduce_scatter" in name:
        return "reduce_scatter"
    if "all_reduce" in name:
        return "all_reduce"
    if "all_gather" in name or "ring" in name:
        return "all_gather"
    return None


def describe() -> str:
    lines = ["Fused kernels (a collective runs inside the compute op):", ""]
    for name in sorted(FUSED_KERNELS):
        k = FUSED_KERNELS[name]
        lines.append("  %-46s hides %-15s stages: %s" % (k.call, k.collective, " -> ".join(k.stages)))
        if k.doc:
            lines.append("  %-46s %s" % ("", k.doc))
    return "\n".join(lines)
