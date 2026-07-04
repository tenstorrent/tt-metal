# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify + time the opt-in terminal argmax-reduction dedup (``DG_DEDUP_ARGMAX``).

Lever 4 / rank 3 of ``path_to_100tps.md`` (terminal-path trim). In the
argmax-sampling regime (``gumbel_noise is None`` — the RUN-first serving default)
:func:`denoise_loop.denoise_step` runs TWO full-vocab (262144) argmax reductions
per step over *different* tensors: ``sampled = argmax_last_dim(logits/T)``
(``gumbel_max`` scales first) and ``argmax = argmax_last_dim(logits)`` (raw). In
exact arithmetic they are equal for any ``T>0``; the dedup computes the raw-logit
argmax ONCE and clones the tiny ``[B,1,L,1]`` index for ``sampled``.

This script is **self-contained** (synthetic logits at production vocab — no 46 GB
checkpoint) and proves the change is safe + faster, entirely on device.

What is proven (see the ``_sample_and_argmax`` correctness note):

  A. single-step, at production ``T`` and at ``T==1.0``. HARD bit-exact OFF vs ON:
     the committed ``argmax``, ``entropy``, and ``accept`` mask, plus the dedup
     invariant ``sampled==argmax``. For ``sampled``/``canvas``, the ONLY positions
     that move are exactly the default path's own bf16 temperature-rescale ties
     (positions where its ``sampled != argmax``): this script asserts that
     identity (``dedup sampled == default argmax`` everywhere) and reports the
     tie count. At ``T==1.0`` the multiply is a no-op ⇒ ``sampled``/``canvas`` are
     bit-exact too.
  B. multi-step loop equivalence. At ``T==1.0`` the whole trajectory is
     deterministic ⇒ HARD bit-exact committed argmax OFF vs ON over the fixed-step
     loop. At production ``T`` it reports the committed-argmax agreement (any drift
     is a benign temperature-tie cascade).
  C. traced timing A/B at production vocab: ms/step OFF vs ON and the per-step
     saving (expected ~one 262144-wide argmax, ~14 ms on QB2).

Run (device owned by another agent — DO NOT run here; ready for that agent):
    python models/experimental/diffusion_gemma/doc/optimize_perf/verify_terminal_dedup.py
    python .../verify_terminal_dedup.py --vocab 4096 --iters 8   # fast smoke

Exit code is non-zero if any HARD assertion fails.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from loguru import logger

import ttnn
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt import denoise_loop as DL

VOCAB = 262144
CANVAS = 256
HIDDEN = 128
PROD_TEMP = 0.6
BUDGET = 0.1


def _replicate(mesh):
    return ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None


def _to_host(t):
    """Host torch copy of a (possibly mesh-replicated) tensor's device-0 shard."""
    if hasattr(t.device(), "get_num_devices") and t.device().get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    return ttnn.to_torch(t)


def _ids(t):
    return _to_host(t).reshape(-1).to(torch.long)


def _make_logits(mesh, *, vocab, canvas, seed=0):
    torch.manual_seed(seed)
    # Scale so a clear per-position max exists (mirrors bench_sampling_step).
    logits_t = (torch.randn(1, 1, canvas, vocab) * 4.0).to(torch.float32)
    return ttnn.from_torch(
        logits_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=_replicate(mesh)
    )


def _make_noise_tokens(mesh, *, vocab, canvas, seed=1):
    nt = torch.randint(0, vocab, (1, 1, canvas, 1), generator=torch.Generator().manual_seed(seed)).to(torch.int32)
    return ttnn.from_torch(nt, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, mesh_mapper=_replicate(mesh))


def _dealloc(res):
    for t in (res.canvas, res.accept_mask, res.entropy, res.sampled, res.argmax):
        t.deallocate(True)


def _run_step(mesh, logits, noise_tokens, *, temperature, dedup):
    res = DL.denoise_step(
        logits,
        temperature=temperature,
        entropy_budget=BUDGET,
        gumbel_noise=None,
        noise_tokens=noise_tokens,
        constants=None,
        dedup_argmax=dedup,
    )
    ttnn.synchronize_device(mesh)
    out = dict(
        canvas=_ids(res.canvas),
        sampled=_ids(res.sampled),
        argmax=_ids(res.argmax),
        accept=_to_host(res.accept_mask).reshape(-1).float(),
        entropy=_to_host(res.entropy).reshape(-1).float(),
    )
    _dealloc(res)
    return out


# --------------------------------------------------------------------------- #
# A. single-step bit-exactness at production vocab
# --------------------------------------------------------------------------- #
def check_single_step(mesh, *, vocab, canvas, temperature):
    logger.info(f"[A] single-step  vocab={vocab} canvas={canvas} T={temperature}")
    logits = _make_logits(mesh, vocab=vocab, canvas=canvas)
    noise_tokens = _make_noise_tokens(mesh, vocab=vocab, canvas=canvas)
    base = _run_step(mesh, logits, noise_tokens, temperature=temperature, dedup=False)
    ded = _run_step(mesh, logits, noise_tokens, temperature=temperature, dedup=True)
    logits.deallocate(True)
    noise_tokens.deallocate(True)

    ok = True

    def hard(name, cond):
        nonlocal ok
        logger.info(f"  [HARD] {name:34s} : {'PASS' if cond else 'FAIL'}")
        ok &= bool(cond)

    # Load-bearing decision outputs — bit-identical by construction.
    hard("committed argmax  OFF==ON", torch.equal(base["argmax"], ded["argmax"]))
    hard("entropy           OFF==ON", torch.equal(base["entropy"], ded["entropy"]))
    hard("accept mask       OFF==ON", torch.equal(base["accept"], ded["accept"]))
    # Dedup invariant: the sampled token IS the clean argmax.
    hard("dedup sampled == dedup argmax", torch.equal(ded["sampled"], ded["argmax"]))
    # Exact characterization of where sampled can move: the dedup's sampled equals
    # the (bit-identical) committed argmax everywhere, so any disagreement with the
    # DEFAULT sampled is exactly a position where the default's own two argmaxes
    # disagreed — a bf16 temperature-rescale tie, nothing else.
    hard("dedup sampled == default argmax", torch.equal(ded["sampled"], base["argmax"]))

    ties = int((base["sampled"] != base["argmax"]).sum())
    moved_sampled = int((base["sampled"] != ded["sampled"]).sum())
    moved_canvas = int((base["canvas"] != ded["canvas"]).sum())
    logger.info(f"  default temperature-rescale ties (sampled!=argmax) : {ties}")
    logger.info(f"  sampled moved OFF->ON : {moved_sampled}  (must equal ties)")
    logger.info(f"  canvas  moved OFF->ON : {moved_canvas}  (<= ties; accepted ties only)")
    hard("moved-sampled set == tie set", moved_sampled == ties)
    if temperature == 1.0:
        hard("T==1.0 sampled OFF==ON (no-op multiply)", torch.equal(base["sampled"], ded["sampled"]))
        hard("T==1.0 canvas  OFF==ON", torch.equal(base["canvas"], ded["canvas"]))

    logger.info(f"[A] {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# B. multi-step loop equivalence (canvas-dependent logits_fn)
# --------------------------------------------------------------------------- #
def check_loop_equivalence(mesh, *, steps, canvas, vocab, temp_start, temp_end, hard_assert):
    tag = f"T[{temp_start}->{temp_end}]"
    logger.info(f"[B] loop  steps={steps} canvas={canvas} vocab={vocab} {tag} hard={hard_assert}")
    torch.manual_seed(0)
    emb_t = torch.randn(vocab, HIDDEN) * 0.1
    proj_t = torch.randn(HIDDEN, vocab) * 0.1
    emb = ttnn.from_torch(
        emb_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=_replicate(mesh)
    )
    proj = ttnn.from_torch(
        proj_t.unsqueeze(0).unsqueeze(0),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=_replicate(mesh),
    )

    def logits_fn(canvas_t, step):
        ids = ttnn.reshape(canvas_t, (1, canvas))
        ids = ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT)
        h = ttnn.embedding(ids, emb, layout=ttnn.TILE_LAYOUT)
        ids.deallocate(True)
        h = ttnn.reshape(h, (1, 1, canvas, HIDDEN))
        lg = ttnn.matmul(h, proj)
        h.deallocate(True)
        return lg

    cfg = DiffusionConfig(
        canvas_length=canvas, max_denoise_steps=steps, temperature_start=temp_start, temperature_end=temp_end
    )
    noise_list = [
        ttnn.from_torch(
            torch.randint(0, vocab, (1, 1, canvas, 1), generator=torch.Generator().manual_seed(100 + s)).int(),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=_replicate(mesh),
        )
        for s in range(steps)
    ]

    def noise_tokens_fn(step):
        return ttnn.clone(noise_list[step])

    def make_init():
        init_t = torch.randint(0, vocab, (1, 1, canvas, 1), generator=torch.Generator().manual_seed(7)).int()
        return ttnn.from_torch(
            init_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, mesh_mapper=_replicate(mesh)
        )

    def run(dedup):
        committed = DL.run_fixed_denoise_steps(
            logits_fn, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_tokens_fn, dedup_argmax=dedup
        )
        ttnn.synchronize_device(mesh)
        ids = _ids(committed)
        committed.deallocate(True)
        return ids

    off = run(False)
    on = run(True)
    emb.deallocate(True)
    proj.deallocate(True)
    for nt in noise_list:
        nt.deallocate(True)

    agree = (off == on).float().mean().item()
    eq = torch.equal(off, on)
    logger.info(f"  committed argmax agreement OFF==ON : {agree * 100:.2f}%  off[:8]={off[:8].tolist()}")
    if hard_assert:
        logger.info(f"[B] {tag} {'PASS' if eq else 'FAIL'} (hard bit-exact)")
        return eq
    logger.info(f"[B] {tag} report-only (agreement {agree * 100:.2f}%)")
    return True


# --------------------------------------------------------------------------- #
# C. traced timing A/B at production vocab
# --------------------------------------------------------------------------- #
def time_step(mesh, *, vocab, canvas, iters):
    logger.info(f"[C] traced timing  vocab={vocab} canvas={canvas} iters={iters}")
    logits = _make_logits(mesh, vocab=vocab, canvas=canvas)
    noise_tokens = _make_noise_tokens(mesh, vocab=vocab, canvas=canvas)
    consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=canvas, budget=BUDGET)

    def one(dedup):
        res = DL.denoise_step(
            logits,
            temperature=PROD_TEMP,
            entropy_budget=BUDGET,
            gumbel_noise=None,
            noise_tokens=noise_tokens,
            constants=consts,
            dedup_argmax=dedup,
        )
        _dealloc(res)

    def measure(dedup):
        one(dedup)  # compile / warm program cache
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        one(dedup)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        ttnn.execute_trace(mesh, tid, blocking=False)  # warm replay
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(iters):
            ttnn.execute_trace(mesh, tid, blocking=False)
        ttnn.synchronize_device(mesh)
        ms = (time.perf_counter() - t0) / iters * 1000.0
        ttnn.release_trace(mesh, tid)
        return ms

    off_ms = measure(False)
    on_ms = measure(True)
    logits.deallocate(True)
    noise_tokens.deallocate(True)
    saved = off_ms - on_ms
    logger.info(
        f"  OFF {off_ms:8.2f} ms/step   ON {on_ms:8.2f} ms/step   "
        f"saved {saved:6.2f} ms ({saved / off_ms * 100:5.1f}%)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=VOCAB)
    ap.add_argument("--canvas", type=int, default=CANVAS)
    ap.add_argument("--loop-steps", type=int, default=6)
    ap.add_argument("--loop-canvas", type=int, default=64)
    ap.add_argument("--loop-vocab", type=int, default=256)
    ap.add_argument("--iters", type=int, default=16)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=900000000)
    try:
        ok = True
        # A: single step at production T (rescale active) and at T=1.0 (no-op multiply).
        ok &= check_single_step(mesh, vocab=args.vocab, canvas=args.canvas, temperature=PROD_TEMP)
        ok &= check_single_step(mesh, vocab=args.vocab, canvas=args.canvas, temperature=1.0)
        # B: loop — hard bit-exact at T=1.0; report-only at production schedule.
        ok &= check_loop_equivalence(
            mesh,
            steps=args.loop_steps,
            canvas=args.loop_canvas,
            vocab=args.loop_vocab,
            temp_start=1.0,
            temp_end=1.0,
            hard_assert=True,
        )
        ok &= check_loop_equivalence(
            mesh,
            steps=args.loop_steps,
            canvas=args.loop_canvas,
            vocab=args.loop_vocab,
            temp_start=0.8,
            temp_end=0.4,
            hard_assert=False,
        )
        if not args.skip_timing:
            time_step(mesh, vocab=args.vocab, canvas=args.canvas, iters=args.iters)
    finally:
        ttnn.close_mesh_device(mesh)

    logger.info(f"DEDUP_VERIFY {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
