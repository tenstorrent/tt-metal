# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Concat-experts MoE for the DiffusionGemma denoise step (``DG_MOE_CONCAT``, default off).

## Why this exists

The shipped denoise MoE is a token-gather "sparse" path (:mod:`tt.sparse_moe`): build a
capacity dispatch matrix, gather each expert's tokens with ``disp^T @ hidden``, run a batched
per-expert matmul, then scatter back with ``comb @ down_flat``. That shape only pays for itself
when the capacity is much smaller than the token count. It is not, any more — the capacity
default moved to the canvas length (256) on 2026-07-15 because anything smaller silently dropped
41-84% of the active routes. At ``C = S = 256`` the arithmetic is:

    E=128, C=256, H=2816, I_dev=192          (config.py; weights.py pads I/tp up to a tile)

    expert gate+up+down MACs   5.31e10   — identical to computing every expert densely
    gather   [EC,S] @ [S,H]    2.36e10
    combine  [S,EC] @ [EC,H]   2.36e10
    ------------------------------------
    total                      1.00e11   = +89% over the dense-equivalent 5.31e10

plus two ~184 MiB intermediates (``dispatched``, ``down_flat``) that are each written to DRAM and
read back. The gathered ``[1,E,C,H]`` is ~94% zero rows, because ``build_capacity_dispatch`` places
token *t* of expert *e* at column ``e*C + slot`` and only ~top_k/E of the slots are ever filled.

So the "sparse" path pays the full dense expert cost and then adds the dispatch on top.

## What this does instead

Relayout the per-expert weights ONCE so all experts are one wide matmul, and fold the routing
weights into the GeGLU output so the down projection is also one matmul:

    gate_cat / up_cat : [1,E,H,I]  -> [1,1,H,E*I]      (expert-major along N)
    down_cat          : [1,E,I,H]  -> [1,1,E*I,H]      (expert-major along K)

    routing = router(x)                       [1,1,S,E]  dense, top-k masked, zeros elsewhere
    g       = geglu(x @ gate_cat, x @ up_cat) [1,1,S,E*I]
    rexp    = routing @ expand                [1,1,S,E*I]  expand = repeat_interleave(I(E), I)
    out     = (g * rexp) @ down_cat           [1,1,S,H]

The down fold is exact because the projection is linear in its input:
``sum_e W_down_e @ (r_e * g_e) == (r ⊙ g) @ down_cat``. It also avoids ever materializing the
``[1,E,S,H]`` per-expert output.

``expand`` is a static ``[1,1,E,E*I]`` block matrix (row *e* is 1 across its own I columns) that
broadcasts each expert's scalar routing weight across that expert's intermediate block with a
cheap matmul — the alternative is a reshape of a very wide tensor, which is a tile repack.

## What it costs, and why it is default-off

The concatenated gate/up are a SECOND copy of those weights: 2 * H * E*I * 2 B = 132 MiB per layer
per device at bf16, i.e. **~7.7 GiB across 30 layers**. The originals cannot simply be freed —
prefill and commit still run the ragged top-8 path over them. ``down_cat`` is free: at bf16 in TILE
layout ``[1,E,I,H] -> [1,1,E*I,H]`` is the same byte order (expert *e* occupies row-blocks
``[6e, 6e+6)`` either way), so it is a metadata reshape. :func:`verify_down_concat_is_free` checks
that on device rather than trusting the argument.

7.7 GiB does not fit next to a 12 GiB trace reservation, and it does not have to: the 48 resident
traces measure ~1.44 GiB (doc/vllm_integration/traced_serving.md), so ~10 GiB of that reservation
is unusable slack. Right-size ``DG_TRACE_REGION_SIZE`` first
(``doc/optimize_perf/bisect_trace_region.sh``), then enable this.

This path is **not bit-identical** to the gather path — the routing weight is applied to the GeGLU
output in bf16 before a single 24576-long reduction, where the gather path accumulates the down
projection per expert and applies the routing weight in the combine matmul. Gate it on absolute
quality (the GPQA arm), not on ``committed_sha256``.
"""

from __future__ import annotations

import os

from loguru import logger
import torch
import ttnn

from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.experimental.diffusion_gemma.tt.expert_operations import apply_geglu

TILE = 32

# Block formats reject permute/reshape, so a non-bf16 weight has to round-trip through bf16 to be
# relaid out. The relayout is exact for bf16; for a block format the requant is lossy in the same
# way the original quantization was, which is why this reports rather than hides it.
_RELAYOUT_SAFE_DTYPES = (ttnn.bfloat16, ttnn.float32)


def concat_moe_enabled() -> bool:
    """``DG_MOE_CONCAT`` (default off): run the denoise MoE as concat-experts matmuls."""
    return os.environ.get("DG_MOE_CONCAT", "0").strip().lower() not in ("0", "false", "no", "off")


def _relayout(tensor, fn):
    """Apply ``fn`` to ``tensor``, round-tripping through bf16 when the dtype rejects it."""
    if tensor.dtype in _RELAYOUT_SAFE_DTYPES:
        return fn(tensor)
    wide = ttnn.typecast(tensor, ttnn.bfloat16)
    out = fn(wide)
    wide.deallocate(True)
    requant = ttnn.typecast(out, tensor.dtype)
    out.deallocate(True)
    return requant


def build_gate_up_concat(weight):
    """``[1,E,H,I] -> [1,1,H,E*I]`` (expert-major along N), so all experts are one wide matmul."""

    def _fn(t):
        permuted = ttnn.permute(t, (0, 2, 1, 3))  # [1,H,E,I]
        e_i = permuted.shape[2] * permuted.shape[3]
        out = ttnn.reshape(permuted, (1, 1, permuted.shape[1], e_i))
        permuted.deallocate(True)
        return out

    return _relayout(weight, _fn)


def build_down_concat(weight):
    """``[1,E,I,H] -> [1,1,E*I,H]`` (expert-major along K). A metadata reshape at bf16 TILE."""

    def _fn(t):
        e_i = t.shape[1] * t.shape[2]
        return ttnn.reshape(t, (1, 1, e_i, t.shape[3]))

    return _relayout(weight, _fn)


def build_route_expand(device, num_experts: int, intermediate: int, mesh_mapper=None):
    """Static ``[1,1,E,E*I]`` block matrix: row *e* is 1 across expert *e*'s I columns.

    ``routing @ expand`` broadcasts each expert's scalar routing weight across its intermediate
    block. bf16 is exact here — the entries are 0 and 1.
    """
    expand = torch.repeat_interleave(torch.eye(num_experts), intermediate, dim=1)
    expand = expand.unsqueeze(0).unsqueeze(0)  # [1,1,E,E*I]
    return ttnn.from_torch(
        expand,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=mesh_mapper,
    )


class ConcatExpertWeights:
    """Per-layer concat weights plus the shared expand matrix.

    Built lazily on the first denoise forward of each layer and cached on the layer, so a run
    that never enables this path pays nothing and a reduced-layer bench pays only for the layers
    it runs.
    """

    __slots__ = ("gate_cat", "up_cat", "down_cat", "num_experts", "intermediate")

    def __init__(self, weights):
        self.num_experts = int(weights.gate_proj.shape[1])
        self.intermediate = int(weights.gate_proj.shape[3])
        self.gate_cat = build_gate_up_concat(weights.gate_proj)
        self.up_cat = build_gate_up_concat(weights.up_proj)
        self.down_cat = build_down_concat(weights.down_proj)

    def deallocate(self):
        for name in ("gate_cat", "up_cat", "down_cat"):
            tensor = getattr(self, name, None)
            if tensor is not None:
                tensor.deallocate(True)
                setattr(self, name, None)


_EXPAND_CACHE = {}


def _route_expand_for(device, num_experts, intermediate):
    key = (id(device), num_experts, intermediate)
    cached = _EXPAND_CACHE.get(key)
    if cached is None:
        mapper = ttnn.ReplicateTensorToMesh(device) if hasattr(device, "shape") else None
        cached = build_route_expand(device, num_experts, intermediate, mesh_mapper=mapper)
        _EXPAND_CACHE[key] = cached
    return cached


def concat_weights_for(experts):
    """Return (building if needed) the concat weights cached on an experts module."""
    cached = getattr(experts, "_dg_concat_weights", None)
    if cached is None:
        cached = ConcatExpertWeights(experts.weights)
        experts._dg_concat_weights = cached
        logger.info(
            f"[concat-moe] built concat expert weights: E={cached.num_experts} "
            f"I_dev={cached.intermediate} N={cached.num_experts * cached.intermediate}"
        )
    return cached


def concat_experts_forward(experts, expert_input, dense_routing, *, compute_kernel_config=None):
    """All-experts MoE as three wide matmuls with the routing folded into the GeGLU output.

    ``expert_input``: ``[1,1,S,H]``. ``dense_routing``: ``[1,1,S,E]``, top-k masked (zero for
    unselected experts, which is what makes the fold exact). Returns ``[1,1,S,H]``, all-reduced
    across TP when the down projection is row-parallel.
    """
    from models.experimental.diffusion_gemma.tt.sparse_moe import (
        default_sparse_moe_compute_kernel_config,
        expert_compute_kernel_config,
    )

    concat = concat_weights_for(experts)
    ckcfg = compute_kernel_config or default_sparse_moe_compute_kernel_config()
    dram = ttnn.DRAM_MEMORY_CONFIG

    gate = ttnn.matmul(
        expert_input,
        concat.gate_cat,
        memory_config=dram,
        compute_kernel_config=expert_compute_kernel_config(expert_input, ckcfg),
    )
    up = ttnn.matmul(
        expert_input,
        concat.up_cat,
        memory_config=dram,
        compute_kernel_config=expert_compute_kernel_config(expert_input, ckcfg),
    )
    activated = apply_geglu(gate, up)  # DiffusionGemma's tanh GeLU, not the gemma4 default
    gate.deallocate(True)
    up.deallocate(True)

    expand = _route_expand_for(expert_input.device(), concat.num_experts, concat.intermediate)
    routing_expanded = ttnn.matmul(dense_routing, expand, memory_config=dram, compute_kernel_config=ckcfg)
    weighted = ttnn.mul(activated, routing_expanded)
    activated.deallocate(True)
    routing_expanded.deallocate(True)

    out = ttnn.matmul(
        weighted,
        concat.down_cat,
        memory_config=dram,
        compute_kernel_config=expert_compute_kernel_config(weighted, ckcfg),
    )
    weighted.deallocate(True)

    mesh_config = experts.mesh_config
    if mesh_config is not None and mesh_config.tp > 1:
        out = ccl_allreduce(out, mesh_config, experts.ccl_manager)
    return out


def verify_down_concat_is_free(weights) -> dict:
    """Check the claim that ``[1,E,I,H] -> [1,1,E*I,H]`` is a byte-order-preserving reshape.

    The concat MoE's memory budget rests on the down concat costing nothing. Returns a dict with
    the two buffer addresses and whether the reshaped values match a host-side reference, so the
    claim is measured rather than asserted. Host-side comparison, so only call it off the hot path.
    """
    source = weights.down_proj
    reshaped = build_down_concat(source)
    try:
        result = {
            "source_address": source.buffer_address(),
            "reshaped_address": reshaped.buffer_address(),
            "aliases_source": source.buffer_address() == reshaped.buffer_address(),
        }
        host_source = ttnn.to_torch(source, mesh_composer=None) if not hasattr(source, "shape") else None
        if host_source is None:
            host_source = ttnn.to_torch(source)
        host_reshaped = ttnn.to_torch(reshaped)
        expected = host_source.reshape(host_reshaped.shape)
        result["values_match"] = bool(torch.equal(expected, host_reshaped))
        result["max_abs_diff"] = float((expected.float() - host_reshaped.float()).abs().max().item())
    finally:
        if reshaped.buffer_address() != source.buffer_address():
            reshaped.deallocate(True)
    return result
