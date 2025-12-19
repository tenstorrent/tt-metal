# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from models.common.utility_functions import comp_pcc

LOG = logging.getLogger(__name__)


def compute_mae(pred: np.ndarray, ref: np.ndarray) -> float:
    pred_flat = pred.reshape(-1)
    ref_flat = ref.reshape(-1)
    return float(np.mean(np.abs(pred_flat - ref_flat)))


def compute_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    pred_flat = pred.reshape(-1)
    ref_flat = ref.reshape(-1)
    return float(np.sqrt(np.mean((pred_flat - ref_flat) ** 2)))


def evaluate_tt_vs_cpu(
    images: Iterable[str],
    cpu_pipeline,
    tt_pipeline,
    normalize: bool = True,
) -> Tuple[List[float], float]:
    pccs: List[float] = []
    for img in images:
        cpu_depth = cpu_pipeline.run_depth_cpu(img, normalize=normalize)
        tt_depth = tt_pipeline.forward(img, normalize=normalize)
        # Use shared PCC utility from tt-metal
        passing, pcc = comp_pcc(cpu_depth, tt_depth, 0.99)
        pccs.append(pcc)
        LOG.info("PCC for %s: %.4f", img, pcc)
    mean_pcc = float(np.mean(pccs)) if pccs else 0.0
    return pccs, mean_pcc


def dump_pcc_report(path: str, pccs: List[float], mean_pcc: float):
    Path(path).write_text(json.dumps({"pcc": pccs, "mean_pcc": mean_pcc}, indent=2))


def dump_full_metrics_report(
    path: str,
    images: List[str],
    pccs: List[float],
    maes: List[float],
    rmses: List[float],
) -> None:
    """
    Dump a richer TT vs CPU metrics report for offline analysis.

    The format is:
        {
          "images": [...],
          "pcc": [...],
          "mean_pcc": ...,
          "mae": [...],
          "mean_mae": ...,
          "rmse": [...],
          "mean_rmse": ...
        }
    """
    report: Dict[str, object] = {
        "images": images,
        "pcc": pccs,
        "mean_pcc": float(np.mean(pccs)) if pccs else 0.0,
        "mae": maes,
        "mean_mae": float(np.mean(maes)) if maes else 0.0,
        "rmse": rmses,
        "mean_rmse": float(np.mean(rmses)) if rmses else 0.0,
    }
    Path(path).write_text(json.dumps(report, indent=2))


def _discover_images(root: Path, limit: int = 20) -> List[str]:
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in exts)
    return images[:limit]


def zero_shot_eval(dataset_root: str, pipeline, limit: int = 20):
    """
    Minimal zero-shot evaluation helper.

    For now this simply walks `dataset_root`, runs the given pipeline on the
    first `limit` RGB images, and logs how many were processed. It returns the
    list of depth maps for callers that want to perform additional analysis.

    A more advanced version can wrap NYU/KITTI-specific logic and metrics
    without changing the CLI surface.
    """
    if dataset_root is None:
        LOG.warning("Dataset root is None; skipping zero-shot eval.")
        return []

    root_path = Path(dataset_root)
    if not root_path.exists():
        LOG.warning("Dataset root %s not found; skipping zero-shot eval.", dataset_root)
        return []

    images = _discover_images(root_path, limit=limit)
    preds = []
    for img in images:
        depth = pipeline.forward(img, normalize=True)
        preds.append(depth)
    LOG.info("Zero-shot processed %d images from %s", len(preds), dataset_root)
    return preds


def run_zero_shot_dataset(
    dataset_name: str,
    dataset_root: str,
    pipeline,
    ref_pipeline=None,
    num_images: int = 20,
    output_dir: str | None = None,
):
    """
    Dataset-aware zero-shot evaluation helper.

    Args:
        dataset_name: Informational tag, e.g. \"nyu\" or \"kitti\".
        dataset_root: Root directory containing RGB images.
        pipeline: Primary pipeline (TT or CPU) exposing `.forward(image_path, normalize=True)`.
        ref_pipeline: Optional reference pipeline (usually CPU) exposing the same API.
        num_images: Maximum number of images to evaluate.
        output_dir: If provided, raw depths (.npy), colorized PNGs, and a metrics JSON are written here.
    """
    if dataset_root is None:
        LOG.warning("Dataset root is None for %s; skipping.", dataset_name)
        return {}

    root_path = Path(dataset_root)
    if not root_path.exists():
        LOG.warning("Dataset root %s not found for %s; skipping.", dataset_root, dataset_name)
        return {}

    images = _discover_images(root_path, limit=num_images)
    if not images:
        LOG.warning("No images discovered under %s for %s; skipping.", dataset_root, dataset_name)
        return {}

    out_dir_path = Path(output_dir) if output_dir is not None else None
    if out_dir_path is not None:
        out_dir_path.mkdir(parents=True, exist_ok=True)

    pccs: List[float] = []
    maes: List[float] = []
    rmses: List[float] = []

    for idx, img in enumerate(images):
        depth = pipeline.forward(img, normalize=True)

        if out_dir_path is not None:
            base = Path(img).stem
            depth_array = np.asarray(depth)
            np.save(out_dir_path / f"{dataset_name}_{idx:04d}_{base}_depth.npy", depth_array)

            # Simple colorization similar to runner helper.
            dmin = depth_array.min()
            dmax = depth_array.max()
            norm = (depth_array - dmin) / (dmax - dmin + 1e-8)
            color = (norm.squeeze() * 255).astype(np.uint8)
            Image.fromarray(color).save(out_dir_path / f"{dataset_name}_{idx:04d}_{base}_depth.png")

        if ref_pipeline is not None:
            ref_depth = ref_pipeline.forward(img, normalize=True)
            pccs.append(compute_pcc(depth, ref_depth))
            maes.append(compute_mae(depth, ref_depth))
            rmses.append(compute_rmse(depth, ref_depth))

    if out_dir_path is not None and ref_pipeline is not None:
        metrics_path = out_dir_path / f"{dataset_name}_metrics.json"
        dump_full_metrics_report(str(metrics_path), images, pccs, maes, rmses)

    LOG.info(
        "Zero-shot %s: processed %d images from %s%s",
        dataset_name,
        len(images),
        dataset_root,
        "" if ref_pipeline is None else f", mean PCC={float(np.mean(pccs)):.4f}",
    )

    return {
        "dataset": dataset_name,
        "images": images,
        "pcc": pccs,
        "mae": maes,
        "rmse": rmses,
    }


# ============================================================================
# Standard Depth Estimation Metrics
# ============================================================================


def compute_depth_metrics(
    pred: np.ndarray, gt: np.ndarray, min_depth: float = 1e-3, max_depth: float = 10.0
) -> Dict[str, float]:
    """
    Compute standard depth estimation metrics.

    Args:
        pred: Predicted depth map (H, W) in meters.
        gt: Ground truth depth map (H, W) in meters.
        min_depth: Minimum valid depth (default 1e-3).
        max_depth: Maximum valid depth (default 10.0 for NYU, 80.0 for KITTI).

    Returns:
        Dictionary with metrics: abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3.
    """
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)

    # Mask out invalid regions
    mask = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt)
    if mask.sum() == 0:
        return {"abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0, "d1": 0.0, "d2": 0.0, "d3": 0.0}

    pred_valid = pred[mask]
    gt_valid = gt[mask]

    # Scale prediction to match GT median (affine-invariant evaluation)
    scale = np.median(gt_valid) / (np.median(pred_valid) + 1e-8)
    pred_valid = pred_valid * scale

    # Clamp predictions
    pred_valid = np.clip(pred_valid, min_depth, max_depth)

    # Compute metrics
    thresh = np.maximum(gt_valid / (pred_valid + 1e-8), pred_valid / (gt_valid + 1e-8))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25**2).mean()
    d3 = (thresh < 1.25**3).mean()

    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))

    # Log RMSE with epsilon for numerical stability
    log_diff = np.log(pred_valid + 1e-8) - np.log(gt_valid + 1e-8)
    rmse_log = np.sqrt(np.mean(log_diff**2))

    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "d1": float(d1),
        "d2": float(d2),
        "d3": float(d3),
    }


def load_nyu_depth(path: str, scale: float = 1000.0) -> np.ndarray:
    """
    Load NYU Depth V2 ground truth depth (16-bit PNG in millimeters).

    Args:
        path: Path to depth PNG file.
        scale: Division factor to convert to meters (1000.0 for mm -> m).

    Returns:
        Depth map as float32 numpy array in meters.
    """
    img = Image.open(path)
    depth = np.asarray(img, dtype=np.float32) / scale
    return depth


def load_kitti_depth(path: str, scale: float = 256.0) -> np.ndarray:
    """
    Load KITTI depth ground truth (16-bit PNG, depth = value / 256.0).

    Args:
        path: Path to depth PNG file.
        scale: Division factor (256.0 for KITTI convention).

    Returns:
        Depth map as float32 numpy array in meters.
    """
    img = Image.open(path)
    depth = np.asarray(img, dtype=np.float32) / scale
    return depth


def _find_paired_images(rgb_dir: Path, depth_dir: Path, limit: int) -> List[Tuple[str, str]]:
    """Find matching RGB/depth pairs by filename stem."""
    rgb_exts = {".jpg", ".jpeg", ".png"}
    rgb_files = {p.stem: p for p in rgb_dir.glob("*") if p.suffix.lower() in rgb_exts}
    depth_files = {p.stem: p for p in depth_dir.glob("*.png")}

    pairs = []
    for stem in sorted(rgb_files.keys()):
        if stem in depth_files:
            pairs.append((str(rgb_files[stem]), str(depth_files[stem])))
            if len(pairs) >= limit:
                break
    return pairs


def zero_shot_eval_nyu(
    dataset_root: str,
    pipeline,
    num_images: int = 654,
    output_path: str | None = None,
    flags: List[str] | None = None,
) -> Dict[str, object]:
    """
    Full NYU Depth V2 zero-shot evaluation.

    Expected dataset layout:
        nyu_root/
          rgb/*.png (or .jpg)
          depth/*.png (16-bit depth in mm)

    Args:
        dataset_root: Root directory of NYU dataset.
        pipeline: Pipeline with .forward(image_path, normalize=True) method.
        num_images: Maximum images to evaluate (default 654 = full test set).
        output_path: Optional JSON output path for results.
        flags: Optional list of flags used for the run.

    Returns:
        Dictionary with dataset info and aggregated metrics.
    """
    root = Path(dataset_root)
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"

    if not rgb_dir.exists() or not depth_dir.exists():
        LOG.warning("NYU dataset not found at %s (need rgb/ and depth/ subdirs)", dataset_root)
        return {"error": "dataset_not_found"}

    pairs = _find_paired_images(rgb_dir, depth_dir, num_images)
    if not pairs:
        LOG.warning("No matching RGB/depth pairs found in %s", dataset_root)
        return {"error": "no_pairs_found"}

    all_metrics: List[Dict[str, float]] = []
    for rgb_path, depth_path in pairs:
        pred = pipeline.forward(rgb_path, normalize=True)
        gt = load_nyu_depth(depth_path)

        # Resize pred to GT shape if needed
        pred = np.squeeze(pred)
        if pred.shape != gt.shape:
            from PIL import Image as PILImage

            pred_img = PILImage.fromarray(pred.astype(np.float32), mode="F")
            pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), PILImage.BILINEAR)
            pred = np.asarray(pred_img)

        metrics = compute_depth_metrics(pred, gt, min_depth=1e-3, max_depth=10.0)
        all_metrics.append(metrics)

    # Aggregate metrics
    agg = {}
    for key in ["abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]:
        values = [m[key] for m in all_metrics]
        agg[key] = float(np.mean(values))

    result = {
        "dataset": "nyu",
        "num_images": len(pairs),
        "metrics": agg,
        "flags": flags or [],
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        LOG.info("NYU eval results written to %s", output_path)

    LOG.info("NYU Depth V2: %d images, abs_rel=%.4f, d1=%.4f", len(pairs), agg["abs_rel"], agg["d1"])
    return result


def zero_shot_eval_kitti(
    dataset_root: str,
    pipeline,
    num_images: int = 697,
    output_path: str | None = None,
    flags: List[str] | None = None,
) -> Dict[str, object]:
    """
    KITTI depth zero-shot evaluation.

    Expected dataset layout:
        kitti_root/
          image/*.png
          depth/*.png (16-bit depth, value / 256.0 = meters)

    Args:
        dataset_root: Root directory of KITTI dataset.
        pipeline: Pipeline with .forward(image_path, normalize=True) method.
        num_images: Maximum images to evaluate.
        output_path: Optional JSON output path for results.
        flags: Optional list of flags used for the run.

    Returns:
        Dictionary with dataset info and aggregated metrics.
    """
    root = Path(dataset_root)
    rgb_dir = root / "image"
    depth_dir = root / "depth"

    if not rgb_dir.exists() or not depth_dir.exists():
        LOG.warning("KITTI dataset not found at %s (need image/ and depth/ subdirs)", dataset_root)
        return {"error": "dataset_not_found"}

    pairs = _find_paired_images(rgb_dir, depth_dir, num_images)
    if not pairs:
        LOG.warning("No matching image/depth pairs found in %s", dataset_root)
        return {"error": "no_pairs_found"}

    all_metrics: List[Dict[str, float]] = []
    for rgb_path, depth_path in pairs:
        pred = pipeline.forward(rgb_path, normalize=True)
        gt = load_kitti_depth(depth_path)

        # Resize pred to GT shape if needed
        pred = np.squeeze(pred)
        if pred.shape != gt.shape:
            from PIL import Image as PILImage

            pred_img = PILImage.fromarray(pred.astype(np.float32), mode="F")
            pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), PILImage.BILINEAR)
            pred = np.asarray(pred_img)

        metrics = compute_depth_metrics(pred, gt, min_depth=1e-3, max_depth=80.0)
        all_metrics.append(metrics)

    # Aggregate metrics
    agg = {}
    for key in ["abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]:
        values = [m[key] for m in all_metrics]
        agg[key] = float(np.mean(values))

    result = {
        "dataset": "kitti",
        "num_images": len(pairs),
        "metrics": agg,
        "flags": flags or [],
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        LOG.info("KITTI eval results written to %s", output_path)

    LOG.info("KITTI: %d images, abs_rel=%.4f, d1=%.4f", len(pairs), agg["abs_rel"], agg["d1"])
    return result
