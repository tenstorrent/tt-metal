# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""#48291 — how much does the top-8 expert SELECTION diverge under bf16?

Per-layer / per-step overlap of the router's top-8 expert INDEX SET between the
HF backbone run in fp32 and in bf16, with identical seeded noise / canvas /
prompt and ZERO TT kernels. TT's router is bf16 (ttnn topk on a bf16 softmax),
and the 5-seed floor control shows TT tracks the HF-bf16 floor, so HF-fp32 vs
HF-bf16 top-8 overlap is a CPU-only, device-free proxy for "how much does TT's
top-8 selection differ from HF's" at the router level.

Fairness note: at STEP 0 both runs see the *identical* initial canvas (seeded
noise + prompt), so the step-0 overlap is a clean isolate of router precision
(bf16 rounding of the softmax) — the only divergence is the bf16 backbone drift
*inside that one forward*. At later steps the two trajectories have already
committed different tokens, so the overlap there conflates router precision with
trajectory divergence; it is reported as the compounding curve, not the isolate.

Usage (DiffusionGemma venv + ~104 GB host RAM for fp32; NO TT device):

    PYTHONPATH=$TT_METAL_HOME python \
        models/experimental/diffusion_gemma/doc/decision_fidelity/measure_topk_overlap.py \
        --stage-artifact /tmp/dg48291_tanh_seed1.pt --checkpoint $DG_CKPT
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import torch


def _dense_stack(routing: list[torch.Tensor]) -> torch.Tensor:
    """routing: list of [1,1,S,E] dense weight tensors (one per router call, in
    (step, layer) order). Returns a [n_calls, S, E] float tensor of the dense
    per-expert routing WEIGHTS (0 for non-selected experts)."""
    return torch.stack([d.reshape(d.shape[-2], d.shape[-1]).float() for d in routing], dim=0)


def _sets(dense: torch.Tensor, k: int) -> torch.Tensor:
    """[..., E] dense weights -> [..., k] selected expert indices (top-k largest)."""
    return dense.topk(k, dim=-1).indices


def _overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> tuple[float, float]:
    """a,b: [..., k] index sets. Returns (mean intersection fraction, exact-set-match rate)."""
    inter = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).sum(-1).float()  # [...]
    return (inter / k).mean().item(), (inter == k).float().mean().item()


def _weight_metrics(da: torch.Tensor, db: torch.Tensor) -> tuple[float, float]:
    """da,db: [..., E] dense routing WEIGHTS. Returns:
    - weight-mass overlap = mean over tokens of sum_e min(w_a, w_b) after
      renormalizing each row to sum 1 (= 1 - total-variation distance; the
      fraction of routing weight that lands on shared experts). This is the
      FUNCTIONAL overlap: a flip on a tiny-weight tail expert barely moves it.
    - top-1 (dominant expert) agreement rate."""
    na = da / da.sum(-1, keepdim=True).clamp_min(1e-12)
    nb = db / db.sum(-1, keepdim=True).clamp_min(1e-12)
    mass = torch.minimum(na, nb).sum(-1).mean().item()
    top1 = (da.argmax(-1) == db.argmax(-1)).float().mean().item()
    return mass, top1


def _flip_rank_hist(da: torch.Tensor, db: torch.Tensor, k: int) -> list[float]:
    """For each token, take fp32's ranked top-k experts; report the fraction of
    fp32-picks that fall OUT of bf16's top-k, resolved by fp32 rank (0=dominant
    .. k-1=weakest). Confirms whether flips are tail-dominated."""
    a = da.topk(k, dim=-1).indices  # [..., k], fp32 ranked
    b = db.topk(k, dim=-1).indices  # [..., k], bf16 set
    inb = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1)  # [..., k] True if fp32-pick r is in bf16 set
    out = (~inb).reshape(-1, k).float().mean(0)  # per-rank drop fraction
    return [round(x, 4) for x in out.tolist()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-artifact", required=True, help="replay_hf_tt.py .pt (fixes prompt/seed/config/canvas).")
    ap.add_argument("--checkpoint", default=os.getenv("DG_CKPT"), help="HF checkpoint dir/id.")
    ap.add_argument("--k", type=int, default=8, help="top-k experts (default 8).")
    ap.add_argument("--content-only", action="store_true", help="restrict to fp32 final committed non-EOS positions.")
    ap.add_argument("--output", default=None, help="optional .pt to save routing tensors + per-layer table.")
    args = ap.parse_args()

    from models.experimental.diffusion_gemma.demo.replay_hf_tt import (
        _hf_text_vocab_size,
        _load_hf_model,
        _make_replay_noise,
        _run_hf_reference,
    )

    torch.set_num_threads(os.cpu_count() or 1)
    art = torch.load(args.stage_artifact, map_location="cpu", weights_only=False)
    prompt, seed, config = art["prompt"], art["seed"], art["config"]
    host_canvas = art["host_canvas"]
    n_steps, k = config.max_denoise_steps, args.k
    print(f"[cfg] prompt={prompt!r} seed={seed} steps={n_steps} canvas={config.canvas_length} k={k}", flush=True)

    def run(dtype):
        tok, model = _load_hf_model(args.checkpoint, local_files_only=True, dtype=dtype)
        vocab = _hf_text_vocab_size(model, tok)
        gumbel, renoise = _make_replay_noise(
            seed=seed, steps=n_steps, canvas_length=config.canvas_length, vocab_size=vocab, mode="seeded"
        )
        t0 = time.time()
        ret = _run_hf_reference(
            model, tok, prompt, host_canvas, config, capture_routing=True, gumbel_noise=gumbel, renoise_tokens=renoise
        )
        traj, routing = ret[1], ret[6]
        print(f"[run] {dtype} HF done in {time.time() - t0:.1f}s, {len(routing)} router calls", flush=True)
        dense = _dense_stack(routing)  # [n_calls, S, E]
        del model
        gc.collect()
        return traj, dense

    traj_fp32, dense_fp32 = run(torch.float32)
    traj_bf16, dense_bf16 = run(torch.bfloat16)

    n_calls = min(dense_fp32.shape[0], dense_bf16.shape[0])
    S = dense_fp32.shape[1]
    n_layers = n_calls // n_steps if n_steps else 0
    if n_layers == 0:
        print("[warn] could not infer layer count; treating all calls as one group")
        n_layers = n_calls
        n_steps_eff = 1
    else:
        n_steps_eff = n_calls // n_layers
    print(f"[shape] router calls={n_calls} layers={n_layers} steps={n_steps_eff} tokens/call={S}", flush=True)

    # optional: restrict to fp32 final committed non-EOS content positions
    pos = None
    if args.content_only:
        committed = traj_fp32.committed.flatten()
        # EOS heuristic: exclude the most common trailing token id (pad/eos)
        eos = committed.mode().values.item()
        pos = (committed != eos).nonzero().flatten()
        print(f"[content] {pos.numel()}/{S} positions kept (excluding id={eos})", flush=True)

    da = dense_fp32[:n_calls].reshape(n_steps_eff, n_layers, S, -1)
    db = dense_bf16[:n_calls].reshape(n_steps_eff, n_layers, S, -1)
    if pos is not None:
        da, db = da[:, :, pos, :], db[:, :, pos, :]
    a, b = _sets(da, k), _sets(db, k)

    def block(name, da_, db_, a_, b_):
        frac, exact = _overlap(a_, b_, k)
        mass, top1 = _weight_metrics(da_, db_)
        print(f"\n===== {name} =====")
        print(
            f"  index-set:  {frac*k:.2f}/{k} experts shared/token (frac {frac:.4f}); "
            f"exact-set match {exact:.4f}  ({(1-exact)*100:.1f}% flip >=1)"
        )
        print(
            f"  FUNCTIONAL: weight-mass overlap {mass:.4f}  ({(1-mass)*100:.1f}% of routing weight on flipped experts); "
            f"top-1 (dominant expert) agree {top1:.4f}"
        )

    block("OVERALL (all steps, all layers)", da, db, a, b)
    block("STEP 0 (identical initial canvas — clean router-precision isolate)", da[0], db[0], a[0], b[0])
    print(f"  rank-resolved drop @step0 (fp32 rank0..{k-1} -> out of bf16 top-{k}): {_flip_rank_hist(da[0], db[0], k)}")

    print(f"\n===== PER-LAYER @ step 0  (frac / exact / massOverlap / top1) =====")
    print(f"{'layer':>5} {'frac':>7} {'exact':>7} {'mass':>7} {'top1':>7}")
    for L in range(n_layers):
        fL, eL = _overlap(a[0, L], b[0, L], k)
        mL, t1L = _weight_metrics(da[0, L], db[0, L])
        print(f"{L:>5} {fL:>7.4f} {eL:>7.4f} {mL:>7.4f} {t1L:>7.4f}")

    print(f"\n===== PER-STEP (all layers)  (frac / exact / massOverlap / top1) =====")
    print(f"{'step':>5} {'frac':>7} {'exact':>7} {'mass':>7} {'top1':>7}")
    for s in range(n_steps_eff):
        fs, es = _overlap(a[s], b[s], k)
        ms, t1s = _weight_metrics(da[s], db[s])
        print(f"{s:>5} {fs:>7.4f} {es:>7.4f} {ms:>7.4f} {t1s:>7.4f}")

    if args.output:
        torch.save(
            {
                "dense_fp32": dense_fp32.to(torch.float16),
                "dense_bf16": dense_bf16.to(torch.float16),
                "n_layers": n_layers,
                "n_steps": n_steps_eff,
                "prompt": prompt,
                "seed": seed,
                "k": k,
            },
            args.output,
        )
        print(f"\n[saved] {args.output}", flush=True)
    print("\n[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
