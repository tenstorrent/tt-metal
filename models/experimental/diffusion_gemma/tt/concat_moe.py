# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Concat-experts MoE for the DiffusionGemma denoise step — the only denoise MoE path.

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

## What it costs

The concatenated gate/up are a SECOND copy of those weights: ~264 MiB per layer per device at
bf16, ~7.7 GiB across 30 layers. The originals cannot simply be freed — prefill still runs the
ragged top-8 path over them. ``down_cat`` is free at bf16: in TILE layout ``[1,E,I,H] ->
[1,1,E*I,H]`` is the same byte order, so it is a metadata reshape — a **view**, which is why
:meth:`ConcatExpertWeights.deallocate` must not force-free it. :func:`verify_down_concat_is_free`
checks that on device rather than trusting the argument.

That accounting is the **bf16** case, and it is the only case where ``down_cat`` is free.
Quantized experts (``DG_EXPERTS_DTYPE`` / ``DG_EXPERTS_BFP8`` in :mod:`tt.precision_build`)
quantize at build time and therefore apply to these concat weights too. At a block format the
relayout has to round-trip through bf16 (``_RELAYOUT_SAFE_DTYPES``) and ``down_cat`` becomes a
real third tensor rather than a view: re-measure with :func:`verify_down_concat_is_free` instead
of scaling the bf16 figure.

**Blast radius.** This is not denoise-only. The batched commit runs the same layer body and calls
the same ``_denoise_moe_forward`` seam (``tt/commit_batched.py``), so this folds the **commit**
MoE too — deliberate, since commit is meant to be numerically the same body as denoise. Prefill
is untouched — it has its own ragged top-8 path in :mod:`tt.sparse_moe`, which is why the
original ``[1,E,H,I]`` gate/up weights must stay live. ``DG_TRACE_REGION_SIZE`` must be
right-sized to leave room for the concat weights — an oversized trace reservation is an OOM.

Numerics: the routing weight is applied to the GeGLU output in bf16 before a single long
reduction, not per expert in the down accumulation, so outputs are not bit-identical to a
per-expert-accumulated MoE.
"""

from __future__ import annotations

import os

from loguru import logger
import torch
import ttnn

from models.experimental.diffusion_gemma.tt.ccl import ccl_allreduce
from models.experimental.diffusion_gemma.tt.expert_operations import apply_geglu

TILE = 32

# Block formats reject permute/reshape, so a non-bf16 weight has to round-trip through bf16 to be
# relaid out. The relayout is exact for bf16; for a block format the requant is lossy in the same
# way the original quantization was, which is why this reports rather than hides it.
_RELAYOUT_SAFE_DTYPES = (ttnn.bfloat16, ttnn.float32)

_EXPERT_FP32_FULL_SYNC_CFG_CACHE = {}


def default_expert_compute_kernel_config():
    """HiFi2 for the expert matmuls. ``DG_SPARSE_MOE_HIFI4=1`` raises it to HiFi4, which is what
    the gemma4 dense reference (``models/demos/gemma4/tt/experts/prefill.py``) uses; the
    ``DG_SPARSE_*`` flag names are legacy naming. A fidelity change moves block convergence in
    BOTH directions, so flipping the default needs a fresh paired run, not a single-prompt result.
    ``fp32_dest_acc_en`` stays False: flipping it changes expert numerics and owes its own paired
    run."""
    fidelity = ttnn.MathFidelity.HiFi4 if os.environ.get("DG_SPARSE_MOE_HIFI4", "0") != "0" else ttnn.MathFidelity.HiFi2
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def expert_compute_kernel_config(tensor, fallback):
    # Blackhole half-DST only exposes four FP32 tiles and corrupts the 8-tile subblocks the expert
    # matmuls emit. Full-DST synchronization preserves the FP32 accumulation fidelity without
    # changing the BF16 expert outputs. Keep an escape hatch for architecture/performance bisects.
    arch = tensor.device().arch()
    if os.environ.get("DG_SPARSE_EXPERT_FP32_FULL_SYNC", "0") != "1" or arch != ttnn.Arch.BLACKHOLE:
        return fallback
    key = (id(tensor.device()), arch)
    config = _EXPERT_FP32_FULL_SYNC_CFG_CACHE.get(key)
    if config is None:
        config = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            dst_full_sync_en=True,
        )
        _EXPERT_FP32_FULL_SYNC_CFG_CACHE[key] = config
    return config


def _free_if_distinct(candidate, source) -> None:
    """Free ``candidate`` only when it does not alias ``source``.

    ``ttnn.reshape`` returns a **view** when the last dim is unchanged and the second-last dims are
    tile-aligned, and a view carries its own ``MeshTensorHolder`` — so ``is_allocated()`` on the view
    stays true after the root is freed. Force-freeing the root and then touching the view therefore
    reads DRAM the allocator has already handed back, silently, with no validation error. Comparing
    buffer addresses is the only reliable test; when it cannot be taken we leak rather than risk it,
    the same discipline ``diffusion_attention._is_distinct_buffer`` uses.
    """
    if candidate is source:
        return
    try:
        distinct = candidate.buffer_address() != source.buffer_address()
    except Exception:
        return
    if distinct:
        source.deallocate(True)


def _relayout(tensor, fn):
    """Apply ``fn`` to ``tensor``, round-tripping through bf16 when the dtype rejects it."""
    if tensor.dtype in _RELAYOUT_SAFE_DTYPES:
        return fn(tensor)
    wide = ttnn.typecast(tensor, ttnn.bfloat16)
    out = fn(wide)
    _free_if_distinct(out, wide)
    requant = ttnn.typecast(out, tensor.dtype)
    _free_if_distinct(requant, out)
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

    Built lazily on the first denoise forward of each layer and cached on the layer, so a
    reduced-layer bench pays the ~264 MiB/layer relayout only for the layers it actually runs.

    The ~7.7 GiB is **process-lifetime**: nothing reserves or pre-checks it before the lazy build,
    and :meth:`deallocate` currently has no caller (no teardown path frees these), so the only
    release is process exit. A run whose ``DG_TRACE_REGION_SIZE`` is not right-sized for it OOMs,
    and the reservation is validated nowhere (``tt/generator_vllm.py`` only checks that it parses
    as > 0).
    """

    __slots__ = ("gate_cat", "up_cat", "down_cat", "num_experts", "intermediate")

    def __init__(self, weights):
        self.num_experts = int(weights.gate_proj.shape[1])
        self.intermediate = int(weights.gate_proj.shape[3])
        self.gate_cat = build_gate_up_concat(weights.gate_proj)
        self.up_cat = build_gate_up_concat(weights.up_proj)
        self.down_cat = build_down_concat(weights.down_proj)

    def deallocate(self):
        """Release the concat weights **without** freeing anything they alias.

        ``down_cat`` is a *view* of ``experts.weights.down_proj`` at bf16.
        ``deallocate(True)`` bypasses the not-sole-owner guard
        and reaches the root holder, so force-freeing it would free the live row-parallel down
        weights that the ragged prefill path still reads — and the failure would surface inside
        prefill, far from here. ``deallocate(False)`` is correct in both cases: the aliasing bf16
        view is not the sole owner and is skipped, while a non-aliasing bfp8 copy is freed normally.
        """
        for name in ("gate_cat", "up_cat", "down_cat"):
            tensor = getattr(self, name, None)
            if tensor is not None:
                tensor.deallocate(False)
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
    concat = concat_weights_for(experts)
    ckcfg = compute_kernel_config or default_expert_compute_kernel_config()
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
