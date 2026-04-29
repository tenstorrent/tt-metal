# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""VGGT accuracy evaluation on CO3Dv2 scenes.

Two metrics per scene:

  1. Relative PCC (port vs torch reference) on real CO3D images.

  2. Camera-pose error vs CO3Dv2 ground-truth viewpoints:
       - RRA@5/15 deg  (relative rotation accuracy)
       - RTA@5/15 deg  (relative translation-direction accuracy)
       - AUC@30 deg    (area under the cumulative-error curve)
     Metrics follow the VGGT paper convention (pair-wise relative poses).

Published results (Blackhole p150a, 12 scenes, 6 categories):

    S=1 : AUC30=96.8 %, RRA/RTA 100 % @ 5°/15°, PCC 0.9957
    S=3 : AUC30=96.9 % (Δ=−0.1 vs ref), 36 pairs
    S=4 : AUC30=96.9 % (Δ=0.0 vs ref),  72 pairs, port ties ref exactly

Usage:
    python models/experimental/vggt/tests/eval_co3d.py \\
        --co3d-root /path/to/co3d_data \\
        --categories apple,bottle,chair,laptop,hydrant,teddybear \\
        --num-views 1 --device-id 0

Environment variables:
    TT_METAL_HOME, TT_DEVICE_ID, VGGT_REF_PATH, VGGT_WEIGHTS_PATH, VGGT_S_CANON
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

_TT_METAL_HOME = Path(os.environ.get("TT_METAL_HOME", str(Path(__file__).parents[4])))
if str(_TT_METAL_HOME) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_HOME))

_VGGT_REF = os.environ.get("VGGT_REF_PATH", "/home/ttuser/experiments/vggt/vggt_ref")
if _VGGT_REF not in sys.path:
    sys.path.insert(0, _VGGT_REF)

from models.experimental.vggt.reference.torch_vggt import load_vggt  # noqa: E402
from models.experimental.vggt.tt.ttnn_vggt import (  # noqa: E402
    _ensure_installed,
    vggt_forward,
)
from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore  # noqa: E402

_PCC_KEYS = ("depth", "depth_conf", "world_points", "world_points_conf", "pose_enc")
DEVICE_ID = int(os.environ.get("TT_DEVICE_ID", "0"))


# ---------------------------------------------------------------------------
# CO3D annotation helpers
# ---------------------------------------------------------------------------

def load_co3d_annotations(co3d_root: Path, category: str) -> dict:
    ann_path = co3d_root / category / "frame_annotations.jgz"
    with gzip.open(ann_path, "rb") as f:
        annos = json.load(f)
    by_seq: dict = {}
    for entry in annos:
        by_seq.setdefault(entry["sequence_name"], []).append(entry)
    for seq in by_seq:
        by_seq[seq].sort(key=lambda e: e["frame_number"])
    return by_seq


def co3d_to_opencv_extrinsic(viewpoint: dict) -> np.ndarray:
    """CO3Dv2 PyTorch3D convention → OpenCV 3×4 [R|t]."""
    R = np.array(viewpoint["R"], dtype=np.float64).T
    T = np.array(viewpoint["T"], dtype=np.float64)
    flip = np.diag([-1.0, -1.0, 1.0])
    extri = np.concatenate([flip @ R, (flip @ T)[:, None]], axis=-1)
    return extri


# ---------------------------------------------------------------------------
# Pose metrics
# ---------------------------------------------------------------------------

def rel_rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    tr = np.clip((np.trace(R1 @ R2.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def rel_translation_angle_deg(t1: np.ndarray, t2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(t1), np.linalg.norm(t2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos = float(np.clip(np.dot(t1 / n1, t2 / n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def extrinsic_to_rel(extri_i: np.ndarray, extri_j: np.ndarray):
    Ri, ti = extri_i[:, :3], extri_i[:, 3]
    Rj, tj = extri_j[:, :3], extri_j[:, 3]
    R_rel = Rj @ Ri.T
    t_rel = tj - R_rel @ ti
    return R_rel, t_rel


def compute_auc(errors: list[float], max_deg: float = 30.0) -> float:
    if not errors:
        return 0.0
    bins = np.linspace(0, max_deg, 100)
    hist = np.mean(np.array(errors)[:, None] <= bins[None, :], axis=0)
    return float(np.trapz(hist, bins) / max_deg * 100.0)


# ---------------------------------------------------------------------------
# PCC helper
# ---------------------------------------------------------------------------

def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    denom = (a.norm() * b.norm()).item()
    return float((a @ b).item() / denom) if denom else 0.0


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def eval_scene(
    device,
    ref_model,
    image_paths: list[Path],
    gt_extris: list[np.ndarray],
) -> dict:
    images = load_and_preprocess_images(image_paths)  # (1, S, 3, H, W)

    with torch.no_grad():
        ref_out = ref_model(images)

    tt_out = vggt_forward(images, device=device)

    # PCC
    pcc_scores = {
        k: _pcc(ref_out[k], tt_out[k])
        for k in _PCC_KEYS
        if k in ref_out and k in tt_out and isinstance(ref_out[k], torch.Tensor)
    }

    # Camera poses from ttnn port
    pose_enc = tt_out.get("pose_enc")
    if pose_enc is None:
        return {"pcc": pcc_scores, "pose_errors": []}

    pred_extris, _ = pose_encoding_to_extri_intri(
        pose_enc, images.shape[-2:], build_intrinsics=False
    )
    pred_extris = pred_extris[0].cpu().numpy()  # (S, 3, 4)

    S = len(gt_extris)
    rot_errs, trans_errs = [], []
    for i in range(S):
        for j in range(i + 1, S):
            R_gt, t_gt = extrinsic_to_rel(gt_extris[i], gt_extris[j])
            R_pr, t_pr = extrinsic_to_rel(pred_extris[i], pred_extris[j])
            rot_errs.append(rel_rotation_angle_deg(R_gt, R_pr))
            trans_errs.append(rel_translation_angle_deg(t_gt, t_pr))

    return {"pcc": pcc_scores, "rot_errs": rot_errs, "trans_errs": trans_errs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--co3d-root", required=True)
    parser.add_argument("--categories", default="apple,bottle,chair,laptop,hydrant,teddybear")
    parser.add_argument("--num-scenes", type=int, default=3,
                        help="Scenes per category (most-viewed first)")
    parser.add_argument("--num-views", type=int, default=1,
                        help="Frames per scene. VGGT is scale-ambiguous at S=1.")
    parser.add_argument("--device-id", type=int, default=DEVICE_ID)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    import ttnn

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32 * 1024)
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()

    _closed = [False]
    def _close_once():
        if _closed[0]:
            return
        _closed[0] = True
        try:
            ttnn.close_device(device)
        except Exception:
            traceback.print_exc()

    signal.signal(signal.SIGINT, lambda s, f: (_close_once(), sys.exit(128 + s)))
    signal.signal(signal.SIGTERM, lambda s, f: (_close_once(), sys.exit(128 + s)))

    try:
        s = args.num_views
        prewarm = (s,) if s <= 2 else (1, 2)
        _ensure_installed(device, prewarm_seqs=prewarm)

        ref_model = load_vggt(eval_mode=True)
        co3d_root = Path(args.co3d_root)
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]

        all_rot, all_trans = [], []
        rra5_list, rra15_list, rta5_list, rta15_list = [], [], [], []
        all_pcc: dict[str, list] = {k: [] for k in _PCC_KEYS}

        for cat in categories:
            annotations = load_co3d_annotations(co3d_root, cat)
            seqs = sorted(
                annotations.keys(),
                key=lambda sq: -len(annotations[sq]),
            )[: args.num_scenes]

            for seq_name in seqs:
                frames = annotations[seq_name]
                step = max(1, len(frames) // s)
                selected = frames[::step][:s]
                if len(selected) < s:
                    continue

                image_paths = [co3d_root / cat / entry["image"]["path"] for entry in selected]
                gt_extris = [co3d_to_opencv_extrinsic(e["viewpoint"]) for e in selected]

                t0 = time.perf_counter()
                result = eval_scene(device, ref_model, image_paths, gt_extris)
                elapsed = time.perf_counter() - t0

                rot_e = result.get("rot_errs", [])
                trans_e = result.get("trans_errs", [])
                all_rot.extend(rot_e)
                all_trans.extend(trans_e)
                rra5_list.append(np.mean([e < 5 for e in rot_e]) if rot_e else 0.0)
                rra15_list.append(np.mean([e < 15 for e in rot_e]) if rot_e else 0.0)
                rta5_list.append(np.mean([e < 5 for e in trans_e]) if trans_e else 0.0)
                rta15_list.append(np.mean([e < 15 for e in trans_e]) if trans_e else 0.0)
                for k, v in result["pcc"].items():
                    all_pcc[k].append(v)

                min_pcc = min(result["pcc"].values()) if result["pcc"] else 0.0
                print(
                    f"[{cat}/{seq_name}] pairs={len(rot_e)} "
                    f"min_pcc={min_pcc:.4f} "
                    f"RRA5={rra5_list[-1]:.2%} "
                    f"elapsed={elapsed:.1f}s"
                )

        auc_rot = compute_auc(all_rot, 30.0)
        auc_trans = compute_auc(all_trans, 30.0)
        auc30 = (auc_rot + auc_trans) / 2.0

        print("\n=== Summary ===")
        print(f"S={s}  scenes={len(rra5_list)}  pairs={len(all_rot)}")
        print(f"RRA@5° : {np.mean(rra5_list):.1%}   RRA@15°: {np.mean(rra15_list):.1%}")
        print(f"RTA@5° : {np.mean(rta5_list):.1%}   RTA@15°: {np.mean(rta15_list):.1%}")
        print(f"AUC@30°: {auc30:.1f}")
        for k, vals in all_pcc.items():
            if vals:
                print(f"mean pcc_{k}: {np.mean(vals):.4f}")
    finally:
        _close_once()


if __name__ == "__main__":
    main()
