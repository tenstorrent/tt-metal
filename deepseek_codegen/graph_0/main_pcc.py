# SPDX-License-Identifier: Apache-2.0
"""Full-graph PCC gate for the moe_compute integration into main.py.

Runs main._main once (eager) and saves the computed live-outs (post-lm_head) as
per-device-stacked torch tensors, tagged sparse|moe. When both tags exist, prints
the self-consistency PCC (sparse path == original golden at PCC 1.0, so sparse-vs-moe
PCC measures the moe_compute bf4 deviation through the full graph incl lm_head).

Usage:
  PCC_TAG=sparse python3 main_pcc.py     # run BEFORE the swap (sparse main.py)
  PCC_TAG=moe    python3 main_pcc.py     # run AFTER  the swap (moe_compute main.py)
"""
import os
import sys
import torch
import ttnn
import utils
import main as M

TAG = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PCC_TAG", "run")).strip()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_io")
# computed live-outs of _main (indices into the return list); 0..5 are kv-cache passthroughs
LIVE_OUTS = {6: "typecast_106", 7: "all_gather_33", 8: "add_29", 9: "to_layout_267"}


def _to_stack(t):
    """Per-device shards -> stacked torch [num_dev, ...] float (composer-agnostic; same for both runs)."""
    return torch.stack([ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(t)])


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main():
    weights = M.load_weights_for__main()
    acts = M.load_activations_for__main()
    outs = M._main(acts, weights)
    saved = {name: _to_stack(outs[i]) for i, name in LIVE_OUTS.items()}
    path = os.path.join(OUT, f"main_outs_{TAG}.pt")
    torch.save(saved, path)
    print(
        f"[main_pcc] saved {TAG} -> {path}: " + ", ".join(f"{n}{tuple(v.shape)}" for n, v in saved.items()), flush=True
    )
    if utils.DeviceGetter._instance is not None:
        ttnn.close_mesh_device(utils.DeviceGetter._instance)
        utils.DeviceGetter._instance = None

    sp, mo = os.path.join(OUT, "main_outs_sparse.pt"), os.path.join(OUT, "main_outs_moe.pt")
    if os.path.exists(sp) and os.path.exists(mo):
        A, B = torch.load(sp), torch.load(mo)
        print("=== PCC sparse-vs-moe (full-graph live-outs; gate >= 0.99) ===", flush=True)
        worst = 1.0
        for n in A:
            p = _pcc(A[n], B[n])
            worst = min(worst, p)
            print(
                f"  {n:16s} PCC={p:.6f}  sparse_norm={float(A[n].norm()):.3f} moe_norm={float(B[n].norm()):.3f} shape={tuple(A[n].shape)}",
                flush=True,
            )
        print(
            f"=== WORST live-out PCC = {worst:.6f} -> {'PASS' if worst >= 0.99 else 'FAIL'} (floor 0.99) ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
