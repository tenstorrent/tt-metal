# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced microbenchmark for the DiffusionGemma terminal per-step sampling chain.

The measured unit is the DENOISE-STEP terminal path over the 256-token canvas at
real production vocab (262144): gumbel-max / argmax / entropy over the vocab axis,
the entropy-budget accept chain (sort -> cumsum -> subtract -> le -> scatter) over
the 256 canvas axis, and renoise. This is the net-new terminal path that replaces
the autoregressive LM-head/argmax apparatus; it has no generic autoregressive
tuning guidance, so this harness produces the before/after candidate evidence.

Run for e2e device timing (fast A/B):
    python models/experimental/diffusion_gemma/doc/optimize_perf/bench_sampling_step.py \
        --variant baseline --iters 16

Run under tracy for a tt-perf-report op table (signposts SAMP_START..SAMP_END):
    python -m tracy -r -p -v -m \
        models/experimental/diffusion_gemma/doc/optimize_perf/bench_sampling_step.py \
        --variant baseline --iters 16
"""

from __future__ import annotations

import argparse
import time

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.experimental.diffusion_gemma.tt import denoise_loop as DL
from models.experimental.diffusion_gemma.tt import sampling as TS

VOCAB = 262144
CANVAS = 256
TEMP = 0.6
BUDGET = 0.1


def _replicate(mesh):
    return ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None


def make_inputs(mesh, use_gumbel: bool):
    torch.manual_seed(0)
    logits_t = (torch.randn(1, 1, CANVAS, VOCAB) * 4.0).to(torch.float32)
    logits = ttnn.from_torch(
        logits_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=_replicate(mesh)
    )
    noise = None
    if use_gumbel:
        u = torch.rand(1, 1, CANVAS, VOCAB).clamp_min(1e-10)
        g = (-torch.log((-torch.log(u)).clamp_min(1e-10))).to(torch.float32)
        noise = ttnn.from_torch(
            g, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=_replicate(mesh)
        )
    nt = torch.randint(0, VOCAB, (1, 1, CANVAS, 1)).to(torch.int32)
    noise_tokens = ttnn.from_torch(
        nt, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, mesh_mapper=_replicate(mesh)
    )
    return logits, noise, noise_tokens


def _dealloc(res: "DL.TtDenoiseStepResult"):
    res.canvas.deallocate(True)
    res.accept_mask.deallocate(True)
    res.entropy.deallocate(True)
    res.sampled.deallocate(True)
    res.argmax.deallocate(True)


# ---------------------------------------------------------------------------
# Variants. Each takes (logits, noise, noise_tokens) and runs one full terminal
# step, deallocating everything it produces (logits/noise/noise_tokens are fixed
# inputs owned by the harness and must NOT be deallocated).
# ---------------------------------------------------------------------------


def variant_baseline(logits, noise, noise_tokens, consts):
    res = DL.denoise_step(
        logits,
        temperature=TEMP,
        entropy_budget=BUDGET,
        gumbel_noise=noise,
        noise_tokens=noise_tokens,
        constants=consts,
    )
    _dealloc(res)


VARIANTS = {
    "baseline": variant_baseline,
}


def register_variant(name):
    def deco(fn):
        VARIANTS[name] = fn
        return fn

    return deco


# --- candidate helpers (preserve diffusion decision semantics exactly) ---


def _token_entropy_from_scaled(z):
    """token_entropy but taking the already-temperature-scaled z (avoids re-scaling)."""
    zmax = ttnn.max(z, dim=-1, keepdim=True)
    shifted = ttnn.subtract(z, zmax, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    exp_shifted = ttnn.exp(shifted, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sum_exp = ttnn.sum(exp_shifted, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    log_sum_exp = ttnn.log(sum_exp)
    expected_terms = ttnn.multiply(exp_shifted, shifted, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sum_weighted_shifted = ttnn.sum(expected_terms, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    expected_shifted = ttnn.div(sum_weighted_shifted, sum_exp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    entropy = ttnn.subtract(log_sum_exp, expected_shifted)
    for t in (zmax, shifted, exp_shifted, sum_exp, log_sum_exp, expected_terms, sum_weighted_shifted, expected_shifted):
        t.deallocate(True)
    return entropy


def _token_entropy_chunked(logits, temperature, chunk=32768):
    """Streaming entropy over vocab chunks: H = logsumexp(z) - E_p[z].

    Two-pass over chunks (pass1 = global max; pass2 = accumulate sum_exp and
    sum(exp*z)). Keeps each intermediate to [1,1,256,chunk] instead of a full
    [1,1,256,262144] DRAM tensor. inv_t folds temperature into the pass.
    """
    inv_t = 1.0 / float(temperature)
    vocab = logits.shape[-1]

    def zslice(start, end):
        s = ttnn.slice(
            logits,
            [0, 0, 0, start],
            [logits.shape[0], logits.shape[1], logits.shape[2], end],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if temperature == 1.0:
            return s
        zs = ttnn.multiply(s, inv_t)
        s.deallocate(True)
        return zs

    # pass 1: global max over vocab
    zmax = None
    for start in range(0, vocab, chunk):
        end = min(start + chunk, vocab)
        zc = zslice(start, end)
        cmax = ttnn.max(zc, dim=-1, keepdim=True)
        zc.deallocate(True)
        if zmax is None:
            zmax = cmax
        else:
            nm = ttnn.maximum(zmax, cmax)
            zmax.deallocate(True)
            cmax.deallocate(True)
            zmax = nm
    # pass 2: accumulate
    sum_exp = None
    sum_wexp = None
    for start in range(0, vocab, chunk):
        end = min(start + chunk, vocab)
        zc = zslice(start, end)
        shifted = ttnn.subtract(zc, zmax)
        zc.deallocate(True)
        expc = ttnn.exp(shifted)
        se = ttnn.sum(expc, dim=-1, keepdim=True)
        wexp = ttnn.multiply(expc, shifted)
        expc.deallocate(True)
        shifted.deallocate(True)
        swe = ttnn.sum(wexp, dim=-1, keepdim=True)
        wexp.deallocate(True)
        if sum_exp is None:
            sum_exp, sum_wexp = se, swe
        else:
            n1 = ttnn.add(sum_exp, se)
            sum_exp.deallocate(True)
            se.deallocate(True)
            sum_exp = n1
            n2 = ttnn.add(sum_wexp, swe)
            sum_wexp.deallocate(True)
            swe.deallocate(True)
            sum_wexp = n2
    zmax.deallocate(True)
    lse = ttnn.log(sum_exp)
    e_shift = ttnn.div(sum_wexp, sum_exp)
    entropy = ttnn.subtract(lse, e_shift)
    sum_exp.deallocate(True)
    sum_wexp.deallocate(True)
    lse.deallocate(True)
    e_shift.deallocate(True)
    return entropy


def _accept_from_entropy(entropy_flat, budget, *, budget_t=None, zeros=None, mem=None):
    kw = {} if mem is None else {"memory_config": mem}
    sorted_vals, sorted_idx = ttnn.sort(entropy_flat, dim=-1)
    cum = ttnn.cumsum(sorted_vals, dim=-1)
    excl = ttnn.subtract(cum, sorted_vals)
    own_budget = budget_t is None
    if own_budget:
        budget_t = ttnn.full(
            list(entropy_flat.shape),
            float(budget),
            dtype=entropy_flat.get_dtype(),
            layout=ttnn.TILE_LAYOUT,
            device=entropy_flat.device(),
        )
    accept_sorted = ttnn.le(excl, budget_t)
    accept_sorted_bf = ttnn.typecast(accept_sorted, ttnn.bfloat16)
    own_zeros = zeros is None
    if own_zeros:
        zeros = ttnn.typecast(ttnn.zeros_like(entropy_flat), ttnn.bfloat16)
    accept = ttnn.scatter(zeros, -1, sorted_idx, accept_sorted_bf, **kw)
    for t in (sorted_vals, sorted_idx, cum, excl, accept_sorted, accept_sorted_bf):
        t.deallocate(True)
    if own_budget:
        budget_t.deallocate(True)
    if own_zeros:
        zeros.deallocate(True)
    return accept


def _terminal_from_entropy_and_samples(
    entropy,
    sampled_u32,
    argmax_u32,
    noise_tokens,
    *,
    budget,
    budget_t=None,
    zeros=None,
    accept_mem=None,
    renoise_ones=None,
):
    entropy_flat = ttnn.reshape(entropy, (entropy.shape[0] * entropy.shape[1], entropy.shape[2]))
    accept_flat = _accept_from_entropy(entropy_flat, budget, budget_t=budget_t, zeros=zeros, mem=accept_mem)
    accept_mask = ttnn.reshape(accept_flat, (entropy.shape[0], entropy.shape[1], 1, entropy.shape[2]))
    accept_for_where = ttnn.reshape(accept_mask, sampled_u32.shape)
    canvas = DL.renoise(accept_for_where, sampled_u32, noise_tokens, ones=renoise_ones)
    entropy_flat.deallocate(True)
    accept_flat.deallocate(True)
    accept_for_where.deallocate(True)
    canvas.deallocate(True)
    accept_mask.deallocate(True)


@register_variant("share_z")
def variant_share_z(logits, noise, noise_tokens, consts):
    """Compute z = logits/T once; reuse for gumbel argmax, clean argmax, entropy.

    Eliminates 2 redundant full-vocab temperature_scale passes (argmax(logits)==
    argmax(z) for T>0; softmax(logits/T)==softmax(z)). ROW_MAJOR multi-core argmax.
    """
    z = TS.temperature_scale(logits, TEMP)
    if noise is not None:
        perturbed = ttnn.add(z, noise)
        sampled = TS.argmax_last_dim(perturbed)
        perturbed.deallocate(True)
    else:
        sampled = TS.argmax_last_dim(z)
    sampled = ttnn.typecast(sampled, ttnn.uint32)
    argmax = TS.argmax_last_dim(z)
    argmax = ttnn.typecast(argmax, ttnn.uint32)
    entropy = _token_entropy_from_scaled(z)
    if z is not logits:
        z.deallocate(True)
    _terminal_from_entropy_and_samples(
        entropy,
        sampled,
        argmax,
        noise_tokens,
        budget=BUDGET,
        budget_t=consts.budget_t,
        zeros=consts.accept_zeros,
        renoise_ones=consts.renoise_ones,
    )
    sampled.deallocate(True)
    argmax.deallocate(True)
    entropy.deallocate(True)


@register_variant("chunked_entropy")
def variant_chunked_entropy(logits, noise, noise_tokens, consts):
    """share_z + chunked (L1-resident) entropy reduction."""
    z = TS.temperature_scale(logits, TEMP)
    if noise is not None:
        perturbed = ttnn.add(z, noise)
        sampled = TS.argmax_last_dim(perturbed)
        perturbed.deallocate(True)
    else:
        sampled = TS.argmax_last_dim(z)
    sampled = ttnn.typecast(sampled, ttnn.uint32)
    argmax = TS.argmax_last_dim(z)
    argmax = ttnn.typecast(argmax, ttnn.uint32)
    if z is not logits:
        z.deallocate(True)
    entropy = _token_entropy_chunked(logits, TEMP)
    _terminal_from_entropy_and_samples(
        entropy,
        sampled,
        argmax,
        noise_tokens,
        budget=BUDGET,
        budget_t=consts.budget_t,
        zeros=consts.accept_zeros,
        renoise_ones=consts.renoise_ones,
    )
    sampled.deallocate(True)
    argmax.deallocate(True)
    entropy.deallocate(True)


def run(variant: str, iters: int, use_gumbel: bool):
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=900000000)
    try:
        logits, noise, noise_tokens = make_inputs(mesh, use_gumbel)
        fn = VARIANTS[variant]
        # Preallocate accept/renoise constants OUTSIDE the trace (ttnn.full/zeros_like
        # are host writes and are illegal during trace capture).
        consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=CANVAS, budget=BUDGET)

        # 1) eager compile run to warm program cache
        fn(logits, noise, noise_tokens, consts)
        ttnn.synchronize_device(mesh)
        logger.info("compiled")

        # 2) capture ONE iteration in the trace (avoids holding N copies of the
        #    heavy full-vocab intermediates), then replay it `iters` times.
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        fn(logits, noise, noise_tokens, consts)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        logger.info("captured")

        # 3) warmup replay
        ttnn.execute_trace(mesh, tid, blocking=False)
        ttnn.synchronize_device(mesh)

        # 4) measured replays with signposts
        signpost("SAMP_START")
        t0 = time.perf_counter()
        for _ in range(iters):
            ttnn.execute_trace(mesh, tid, blocking=False)
        ttnn.synchronize_device(mesh)
        t1 = time.perf_counter()
        signpost("SAMP_END")

        ttnn.release_trace(mesh, tid)
        e2e_ms = (t1 - t0) * 1e3 / iters
        logger.info(f"VARIANT={variant} use_gumbel={use_gumbel} iters={iters} e2e_ms_per_step={e2e_ms:.4f}")
        print(f"RESULT variant={variant} use_gumbel={use_gumbel} e2e_ms_per_step={e2e_ms:.4f}")
    finally:
        ttnn.close_mesh_device(mesh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline", choices=sorted(VARIANTS.keys()))
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--argmax", action="store_true", help="argmax path (no gumbel noise)")
    args = ap.parse_args()
    run(args.variant, args.iters, use_gumbel=not args.argmax)


if __name__ == "__main__":
    main()
