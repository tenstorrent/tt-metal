# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-ttnn flash-attention partial merge for DiffusionGemma (design task T7).

This is the **Phase-2 companion** to the ``return_lse`` SDPA kernel extension
described in ``doc/optimize_perf/paged_prefix_denoise_design.md`` §1a ("FULL-
ATTENTION 5 layers — paged chunked SDPA + LSE-merge") and enumerated as task T7
in that doc's §6 table.

On the 5 full-attention denoise layers the canvas query attends
``[prefix(committed) ++ canvas]``. Rather than materialize the full C×(P+C) score
matrix, that context is split into two independent SDPA partials — a paged causal
read over the committed prefix (partial *a*) and the existing non-causal C×C
canvas SDPA (partial *b*) — each already softmax-normalized over **its own** key
group. To recombine them into the single-softmax output over the concatenation of
both key groups we need each partial's flash log-sum-exp statistic
``lse = m + log(l)`` (running max ``m`` + log of the running exp-sum ``l``), which
the ``return_lse=True`` kernel extension (task T6) emits as an fp32 second output.

This module is the ``merge_attention_partials`` half of that pair: a *pure-ttnn*
online-softmax combine (max / sub / exp / mul / add / reciprocal, no kernel).
It is **algebraically exact** — given exact ``out_a/out_b`` and ``lse_a/lse_b`` it
reconstructs ``softmax(concat(scores)) @ concat(V)`` bit-for-bit in fp32. The
weight computation (max-shift, exp, sum, reciprocal) is done in fp32 for numeric
stability; only the final rescale of the bf16 partial outputs runs in bf16, so
the merged output carries bf16 rescale drift. Per the design doc, that drift is
**gated on diffusion decision-agreement (zero argmax flips), not on bitwise
equality** (§1a merge note + §3.3): "Algebraically exact; bf16 rescale drift →
gate on decision-agreement, not bitwise."

ttnn idioms (dtype / DRAM / deallocate) follow ``tt/diffusion_attention.py`` and
``tt/sampling.py`` (fp32-compute-then-typecast-back, ``memory_config`` on every op,
``.deallocate(True)`` on every intermediate).
"""

from __future__ import annotations

import ttnn


def merge_attention_partials(out_a, lse_a, out_b, lse_b):
    """Merge two per-key-group softmax-normalized attention partials into one.

    Args:
        out_a: ``[1, H, C, vhd]`` attention output softmax-normalized over key
            group A (the committed-prefix keys). Activation dtype (bf16).
        lse_a: ``[1, H, C, 1]`` fp32 flash log-sum-exp ``m + log(l)`` of group A.
        out_b: ``[1, H, C, vhd]`` attention output softmax-normalized over key
            group B (the canvas keys). Same dtype/shape family as ``out_a``.
        lse_b: ``[1, H, C, 1]`` fp32 flash log-sum-exp ``m + log(l)`` of group B.

    Returns:
        ``[1, H, C, vhd]`` = the exact single-softmax output over the
        concatenation of both key groups (in ``out_a``'s dtype).

    Math (numerically stable, fp32 weights):
        ``m     = max(lse_a, lse_b)``          — per-(head, query) running max
        ``wa    = exp(lse_a - m)``             — group-A weight in (0, 1]
        ``wb    = exp(lse_b - m)``             — group-B weight in (0, 1]
        ``denom = wa + wb``                    — combined normalizer in [1, 2]
        ``out   = (out_a*wa + out_b*wb) * reciprocal(denom)``

    The ``[1, H, C, 1]`` weights broadcast across the ``vhd`` dim in the elementwise
    ``mul`` (last-dim / "column" broadcast, the same shape pattern as the
    ``ttnn.subtract(z, max)`` shift in ``tt/sampling.py``). The weights are
    typecast to ``out_a``'s dtype before that broadcast-mul so both operands of the
    eltwise op share a dtype.
    """
    dram = ttnn.DRAM_MEMORY_CONFIG
    out_dtype = out_a.get_dtype()

    # --- fp32 weight computation (numerically stable, tiny [1,H,C,1] tensors) ---
    m = ttnn.maximum(lse_a, lse_b, memory_config=dram)

    shifted_a = ttnn.subtract(lse_a, m, memory_config=dram)
    wa = ttnn.exp(shifted_a, memory_config=dram)
    shifted_a.deallocate(True)

    shifted_b = ttnn.subtract(lse_b, m, memory_config=dram)
    wb = ttnn.exp(shifted_b, memory_config=dram)
    shifted_b.deallocate(True)
    m.deallocate(True)

    denom = ttnn.add(wa, wb, memory_config=dram)
    inv_denom = ttnn.reciprocal(denom, memory_config=dram)
    denom.deallocate(True)

    # --- apply the weights to the (bf16) partial outputs -----------------------
    # Cast the fp32 weights to the output dtype so each broadcast-mul has matching
    # operand dtypes; this is the only place bf16 rescale drift enters (gated on
    # decision-agreement per the design doc, not on bitwise equality).
    wa_cast = ttnn.typecast(wa, out_dtype)
    wa.deallocate(True)
    wb_cast = ttnn.typecast(wb, out_dtype)
    wb.deallocate(True)
    inv_denom_cast = ttnn.typecast(inv_denom, out_dtype)
    inv_denom.deallocate(True)

    weighted_a = ttnn.mul(out_a, wa_cast, memory_config=dram)  # [1,H,C,1] -> vhd bcast
    wa_cast.deallocate(True)
    weighted_b = ttnn.mul(out_b, wb_cast, memory_config=dram)  # [1,H,C,1] -> vhd bcast
    wb_cast.deallocate(True)

    numerator = ttnn.add(weighted_a, weighted_b, memory_config=dram)
    weighted_a.deallocate(True)
    weighted_b.deallocate(True)

    out = ttnn.mul(numerator, inv_denom_cast, memory_config=dram)  # [1,H,C,1] -> vhd bcast
    numerator.deallocate(True)
    inv_denom_cast.deallocate(True)
    return out
