# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Concat-experts MoE for the DiffusionGemma denoise step — the only denoise MoE path.

## Why this exists

The denoise MoE used to be a token-gather "sparse" path in :mod:`tt.sparse_moe`: build a
capacity dispatch matrix, gather each expert's tokens with ``disp^T @ hidden``, run a batched
per-expert matmul, then scatter back with ``comb @ down_flat``. That path — and the dense-128
reference that sat behind it — were deleted on 2026-07-29, together with the ``DG_MOE_CONCAT``
selector that chose between them, for two independent reasons.

**The gather path did not converge.** Same 2-question smoke, same everything else, the selector the
only difference:

    token-gather   halted  0/9   min halt_entropy_final 0.44021   2 degenerate, 2 guard kills
    concat         halted 19/19  min halt_entropy_final 0.000588  0 degenerate, 0 kills

At 0.44 the mean entropy sits ~100x above the 0.005 halt threshold, so the early halt cannot fire,
every block runs the full 48 steps and commits an unsettled canvas, and the degeneracy guard ends
roughly two thirds of requests. With concat the trajectory reaches ~6e-4 and halts in 8-27 steps.
The fold itself is numerically validated at PCC 0.99992 on device (``213ac50f221``).

**And it was slower.** The gather shape only pays for itself when the capacity is much smaller
than the token count. It was not, any more — the capacity default moved to the canvas length (256)
on 2026-07-15 because anything smaller silently dropped 41-84% of the active routes. At
``C = S = 256`` the arithmetic was:

    E=128, C=256, H=2816, I_dev=192          (config.py; weights.py pads I/tp up to a tile)

    expert gate+up+down MACs   5.31e10   — identical to computing every expert densely
    gather   [EC,S] @ [S,H]    2.36e10
    combine  [S,EC] @ [EC,H]   2.36e10
    ------------------------------------
    total                      1.00e11   = +89% over the dense-equivalent 5.31e10

plus two ~184 MiB intermediates (``dispatched``, ``down_flat``) that were each written to DRAM and
read back. The gathered ``[1,E,C,H]`` was ~94% zero rows, because the capacity dispatch placed
token *t* of expert *e* at column ``e*C + slot`` and only ~top_k/E of the slots are ever filled.

So the "sparse" path paid the full dense expert cost and then added the dispatch on top.

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

## What it costs

The concatenated gate/up are a SECOND copy of those weights: ``H * E*I * 2 B`` = **132 MiB each**,
so ``2 * 132 = 264 MiB per layer per device`` at bf16, i.e. **~7.7 GiB across 30 layers** (measured
7.773 GiB). The originals cannot simply be freed — prefill still runs the ragged top-8 path over
them. ``down_cat`` is free: at bf16 in TILE layout ``[1,E,I,H] -> [1,1,E*I,H]`` is the same byte
order (expert *e* occupies row-blocks ``[6e, 6e+6)`` either way), so it is a metadata reshape —
a **view**, which is why :meth:`ConcatExpertWeights.deallocate` must not force-free it.
:func:`verify_down_concat_is_free` checks that on device rather than trusting the argument.

That accounting is the **bf16** case, and it is the only case where ``down_cat`` is free. Quantized
experts come from ``DG_EXPERTS_DTYPE`` / ``DG_EXPERTS_BFP8`` in :mod:`tt.precision_build`, which
quantize at build time and therefore apply to these concat weights too — that is the whole supported
route now that the sparse path's runtime ``DG_MOE_EXPERT_BFP8`` is gone. At a block format the
relayout has to round-trip through bf16 (``_RELAYOUT_SAFE_DTYPES``) and ``down_cat`` becomes a real
third tensor rather than a view, so the shape of the cost changes, not just its size: re-measure with
:func:`verify_down_concat_is_free` instead of scaling the 7.7 GiB figure.

**Blast radius.** This is not denoise-only. The batched commit runs the same layer body and calls
the same ``_denoise_moe_forward`` seam (``tt/commit_batched.py``), and batched commit is the shipped
default — so this folds the **commit** MoE too, and commit hidden states are what the
committed-prefix KV is built from, so it compounds across blocks. That is deliberate: commit is
meant to be numerically the same body as denoise. Prefill is genuinely untouched — it has its own
ragged top-8 path in :mod:`tt.sparse_moe`, which is why the original ``[1,E,H,I]`` gate/up weights
must stay live.

7.7 GiB does not fit next to a 12 GiB trace reservation, and it does not have to: the 48 resident
traces measure ~1.44 GiB (doc/vllm_integration/traced_serving.md), so ~10 GiB of that reservation
is unusable slack. Since this path is no longer optional, ``DG_TRACE_REGION_SIZE`` has to be
right-sized for it (``doc/optimize_perf/bisect_trace_region.sh``) — an oversized trace reservation
is now an OOM, not a forgone optimization.

This path is **not bit-identical** to the deleted gather path — the routing weight is applied to
the GeGLU output in bf16 before a single 24576-long reduction, where the gather path accumulated
the down projection per expert and applied the routing weight in the combine matmul. So a
``committed_sha256`` recorded before 2026-07-29 cannot be compared against one recorded after;
quality across that boundary is an absolute measurement (the GPQA arm), not a hash match.
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


def default_expert_compute_kernel_config():
    """HiFi2 for the expert matmuls. ``DG_SPARSE_MOE_HIFI4=1`` raises it to HiFi4, which is what
    the gemma4 dense reference (``models/demos/gemma4/tt/experts/prefill.py``) uses.

    On the flag name: ``DG_SPARSE_MOE_HIFI4`` says "sparse" because it predates this module — it was
    the token-gather MoE's knob and moved here with the expert matmuls when that path was deleted on
    2026-07-29. The name is deliberately kept so existing launcher scripts
    (``doc/optimize_perf/sweep_denoise_arms.sh``) and the recorded runs still address the same thing.
    Its sibling ``DG_SPARSE_EXPERT_FP32_FULL_SYNC`` was deleted 2026-08-03: nothing in the tree set
    it, and the +0.4pp it recorded is void twice over (measured on the retired token-gather MoE, and
    against a seed-0 baseline of 0.9296875 that the tanh-GeLU fix has since moved to 0.99609375).

    The fidelity gap against that reference is real and is a contributor to the pcc-vs-dense gap,
    but raising it is NOT a net win. **Scope caveat: the numbers below were measured on the retired
    token-gather MoE.** They are why the default is HiFi2 and why flipping it needs a fresh paired
    run; they are not evidence about this MoE body. Measured over paired 16K runs on the same
    prompts, seed, span, traced path and noise mode, with only the fidelity differing:

        HiFi2   2/13 collapsed   1 empty output   11 answered   10 correct
        HiFi4   4/13 collapsed   0 empty output    9 answered    9 correct

    HiFi4 won only the empty-output case: q007's block 0 goes from 48 steps / not halted /
    109 of 256 positions still flipping to 30 steps / halted / 0, so that request stops emitting
    nothing. But collapses double and one fewer answer is correct, because many blocks converge
    right AT the 48-step cap -- q012's block 0 halted on step 45 of 48 under HiFi2 and tips over
    under HiFi4 -- so a numerical change moves blocks in BOTH directions.

    Cost was not the obstacle -- it is small, because the expert matmuls are weight-bound and these
    blocks are dominated by attention and DRAM traffic, so the extra math passes hide behind the
    DRAM read. Per-block latency at 16K, on the two questions with a clean before/after:

        q012   HiFi2 24.112 s/blk (10.62 tok/s)   HiFi4 25.556 s/blk (10.02 tok/s)   +6.0%
        q013   HiFi2 27.786 s/blk ( 9.21 tok/s)   HiFi4 27.685 s/blk ( 9.25 tok/s)   -0.4%

    So if a paired run on THIS path ever shows HiFi4 net-positive, the throughput will not stand in
    the way. What is needed is that measurement, not another single-prompt result: this default was
    briefly flipped to HiFi4 on three data points and had to be reverted, which is the same mistake
    that shipped a corrupting DG_VLLM_GUMBEL_MODE default for two weeks.

    fp32_dest_acc_en stays False. It was originally False so the retired path's tuned out_subblock
    (product up to 8) stayed legal — fp32_dest_acc caps it at 4 — and it stays False now because
    every quality number on record was taken with it False; flipping it changes expert numerics and
    owes its own paired run."""
    fidelity = ttnn.MathFidelity.HiFi4 if os.environ.get("DG_SPARSE_MOE_HIFI4", "0") != "0" else ttnn.MathFidelity.HiFi2
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


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
    reduced-layer bench pays the 264 MiB/layer relayout only for the layers it actually runs.

    The 7.7 GiB is **process-lifetime**: nothing reserves or pre-checks it before the lazy build,
    and :meth:`deallocate` currently has no caller (no teardown path frees these), so the only
    release is process exit. That was tolerable while this path was opt-in; now that it is the only
    denoise MoE, a run whose ``DG_TRACE_REGION_SIZE`` is not right-sized for it OOMs, and the
    reservation is validated nowhere (``tt/generator_vllm.py`` only checks that it parses as > 0).
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

        ``down_cat`` is a *view* of ``experts.weights.down_proj`` at bf16 (that is exactly why the
        relayout costs 7.7 GiB and not 11.6). ``deallocate(True)`` bypasses the not-sole-owner guard
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
        compute_kernel_config=ckcfg,
    )
    up = ttnn.matmul(
        expert_input,
        concat.up_cat,
        memory_config=dram,
        compute_kernel_config=ckcfg,
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
        compute_kernel_config=ckcfg,
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
