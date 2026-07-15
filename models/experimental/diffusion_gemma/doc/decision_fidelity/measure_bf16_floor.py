# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""#48291 bf16-floor self-consistency control.

Isolates the INTRINSIC bf16 chaos floor of the DiffusionGemma denoise trajectory
with ZERO TT kernels involved: run the HF backbone in fp32 and in bf16 with the
*same* injected noise / canvas / prompt, then compare the two REFERENCE
trajectories to each other. The block-diffusion loop commits the clean argmax and
has no temperature cushion, so a bf16-scale perturbation of the model logits can
bifurcate the trajectory into a *different but equally valid paraphrase*.

Key result (see README.md / work_log.md): running the same HF model, fp32 vs
bf16, seeded-8-step, committed_match is 0.8633 (seed 0) / 0.9141 (seed 1) — BELOW
the 0.95 gate — and the per-step entropy PCC collapses (even goes negative) at
converged steps. So no bf16 implementation, TP or single-device, can match the
fp32 ideal to the strict gate: the reference cannot pass it against itself.

Usage (needs the DiffusionGemma venv + ~104 GB host RAM for the fp32 model; NO TT
device — runs HF on CPU only):

    PYTHONPATH=$TT_METAL_HOME python \
        models/experimental/diffusion_gemma/doc/decision_fidelity/measure_bf16_floor.py \
        --stage-artifact /tmp/dg48291_tanh_seed1.pt \
        --checkpoint $DG_CKPT

The stage artifact is any `demo/replay_hf_tt.py` output that already holds the
bf16 `hf_traj` and device `tt_traj` for a seed (its prompt / seed / config /
host_canvas fix the inputs so the fp32 run is apples-to-apples).
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import torch


def _table(name: str, ref, cand) -> None:
    from models.experimental.diffusion_gemma.tests.trajectory_pcc import _pearson, sound_entropy_step_fidelity

    print(f"\n===== {name} =====", flush=True)
    cm = (ref.committed == cand.committed).float().mean().item()
    ndiff = int((ref.committed != cand.committed).sum())
    print(f"committed_match = {cm:.6f}   ({ndiff} of {ref.committed.numel()} positions differ)")
    diff = (ref.committed != cand.committed).flatten().nonzero().flatten().tolist()
    print(f"differing positions: {diff}")
    n = min(len(ref.per_step), len(cand.per_step))
    header = f"{'step':>4} {'refAcc':>6} {'candAcc':>7} {'iou':>6} {'argAgr':>7} {'refStd':>8} {'entPCC':>8} {'entMaxAbs':>9} {'sound':>6}"
    print(header)
    for i in range(n):
        h, t = ref.per_step[i], cand.per_step[i]
        he, te = h.entropy.flatten().float(), t.entropy.flatten().float()
        a, b = h.accept_mask.bool(), t.accept_mask.bool()
        u = int((a | b).sum())
        iou = (int((a & b).sum()) / u) if u else 1.0
        v = sound_entropy_step_fidelity(he, te)
        print(
            f"{i:>4} {int(h.accept_mask.sum()):>6} {int(t.accept_mask.sum()):>7} {iou:>6.3f} "
            f"{(h.argmax == t.argmax).float().mean().item():>7.4f} {he.std().item():>8.4f} "
            f"{_pearson(he, te):>8.4f} {(he - te).abs().max().item():>9.4f} "
            f"{('P' if v.passed else 'F') + '/' + v.mode:>6}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-artifact", required=True, help="replay_hf_tt.py .pt with hf_traj (bf16) + tt_traj.")
    ap.add_argument("--checkpoint", default=os.getenv("DG_CKPT"), help="HF checkpoint dir/id for the fp32 run.")
    ap.add_argument("--output", default=None, help="Optional path to save the fp32 HF trajectory.")
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
    host_canvas, hf_bf16, tt = art["host_canvas"], art["hf_traj"], art["tt_traj"]
    print(
        f"[cfg] prompt={prompt!r} seed={seed} steps={config.max_denoise_steps} canvas={config.canvas_length}",
        flush=True,
    )

    tok, model = _load_hf_model(args.checkpoint, local_files_only=True, dtype=torch.float32)
    vocab = _hf_text_vocab_size(model, tok)
    gumbel, renoise = _make_replay_noise(
        seed=seed, steps=config.max_denoise_steps, canvas_length=config.canvas_length, vocab_size=vocab, mode="seeded"
    )
    t0 = time.time()
    hf_fp32 = _run_hf_reference(model, tok, prompt, host_canvas, config, gumbel_noise=gumbel, renoise_tokens=renoise)[1]
    print(f"[run] fp32 HF trajectory done in {time.time() - t0:.1f}s", flush=True)
    if args.output:
        torch.save({"hf_fp32_traj": hf_fp32, "prompt": prompt, "seed": seed, "config": config}, args.output)
    del model
    gc.collect()

    _table("HF-fp32 (ref) vs HF-bf16 (cand)  [INTRINSIC bf16 FLOOR — zero TT]", hf_fp32, hf_bf16)
    _table("HF-fp32 (ref) vs TT (cand)  [TT fidelity to the fp32 IDEAL]", hf_fp32, tt)
    _table("HF-bf16 (ref) vs TT (cand)  [the current #48291 gate]", hf_bf16, tt)
    print("\n[done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
