# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
UNet Deep Dive Analysis Utilities

This module provides comprehensive tensor statistics, comparison metrics,
and analysis utilities for investigating SSIM degradation in the SDXL pipeline.

Key Functions:
- compute_tensor_stats(): Full statistical analysis of a tensor
- compute_comparison_metrics(): PCC, MSE, and other comparison metrics
- find_divergence_onset(): Detect when PCC drops below threshold
- analyze_spatial_error(): Generate per-pixel error analysis
- save_step_data(): Write step data to JSON file
- load_step_data(): Read step data from JSON file
- aggregate_run_data(): Merge all step files into summary
"""

import json
import os
import sys
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union


def compute_tensor_stats(tensor: Union[torch.Tensor, np.ndarray], name: str = "tensor") -> Dict[str, Any]:
    """
    Compute comprehensive statistics of a tensor.
    
    Args:
        tensor: PyTorch tensor or numpy array to analyze
        name: Descriptive name for the tensor
        
    Returns:
        Dictionary containing:
        - shape: tensor dimensions
        - dtype: data type
        - min, max, mean, std, median
        - percentiles: p1, p5, p10, p25, p50, p75, p90, p95, p99
        - l1_norm, l2_norm: vector norms
        - nan_count, inf_count: numerical stability indicators
        - histogram: 20-bin histogram of values
    """
    # Convert to numpy for consistent computation
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)
    
    # Flatten for statistics
    flat = arr.flatten()
    
    # Basic statistics
    stats = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "size": int(flat.size),
        "min": float(np.nanmin(flat)),
        "max": float(np.nanmax(flat)),
        "mean": float(np.nanmean(flat)),
        "std": float(np.nanstd(flat)),
        "median": float(np.nanmedian(flat)),
        "abs_mean": float(np.nanmean(np.abs(flat))),
        "abs_max": float(np.nanmax(np.abs(flat))),
    }
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        stats[f"p{p}"] = float(np.nanpercentile(flat, p))
    
    # Norms
    valid_flat = flat[np.isfinite(flat)]
    if len(valid_flat) > 0:
        stats["l1_norm"] = float(np.sum(np.abs(valid_flat)))
        stats["l2_norm"] = float(np.sqrt(np.sum(valid_flat ** 2)))
    else:
        stats["l1_norm"] = float("nan")
        stats["l2_norm"] = float("nan")
    
    # Numerical stability checks
    stats["nan_count"] = int(np.sum(np.isnan(flat)))
    stats["inf_count"] = int(np.sum(np.isinf(flat)))
    stats["zero_count"] = int(np.sum(flat == 0))
    
    # Histogram (20 bins)
    try:
        hist_values, hist_edges = np.histogram(valid_flat, bins=20)
        stats["histogram"] = {
            "counts": hist_values.tolist(),
            "edges": hist_edges.tolist()
        }
    except Exception:
        stats["histogram"] = None
    
    return stats


def compute_comparison_metrics(
    reference: Union[torch.Tensor, np.ndarray],
    candidate: Union[torch.Tensor, np.ndarray],
    name: str = "comparison"
) -> Dict[str, Any]:
    """
    Compute comparison metrics between reference (PyTorch) and candidate (TT) tensors.
    
    Args:
        reference: Ground truth tensor (PyTorch output)
        candidate: Tensor to compare (TT output)
        name: Descriptive name for this comparison
        
    Returns:
        Dictionary containing:
        - pcc: Pearson correlation coefficient
        - mse: Mean squared error
        - mae: Mean absolute error
        - max_abs_error: Maximum absolute difference
        - rmse: Root mean squared error
        - relative_error: Mean relative error
        - cosine_similarity: Cosine similarity score
    """
    # Convert to numpy
    if isinstance(reference, torch.Tensor):
        ref = reference.detach().cpu().float().numpy().flatten()
    else:
        ref = np.asarray(reference, dtype=np.float32).flatten()
        
    if isinstance(candidate, torch.Tensor):
        cand = candidate.detach().cpu().float().numpy().flatten()
    else:
        cand = np.asarray(candidate, dtype=np.float32).flatten()
    
    # Ensure same size
    if ref.size != cand.size:
        return {
            "name": name,
            "error": f"Size mismatch: reference={ref.size}, candidate={cand.size}",
            "pcc": float("nan"),
            "mse": float("nan"),
        }
    
    metrics = {"name": name}
    
    # PCC (Pearson Correlation Coefficient)
    try:
        ref_centered = ref - np.mean(ref)
        cand_centered = cand - np.mean(cand)
        
        numerator = np.sum(ref_centered * cand_centered)
        denominator = np.sqrt(np.sum(ref_centered ** 2) * np.sum(cand_centered ** 2))
        
        if denominator > 1e-10:
            metrics["pcc"] = float(numerator / denominator)
        else:
            metrics["pcc"] = 1.0 if np.allclose(ref, cand) else 0.0
    except Exception as e:
        metrics["pcc"] = float("nan")
        metrics["pcc_error"] = str(e)
    
    # MSE (Mean Squared Error)
    diff = ref - cand
    metrics["mse"] = float(np.mean(diff ** 2))
    
    # MAE (Mean Absolute Error)
    metrics["mae"] = float(np.mean(np.abs(diff)))
    
    # Max Absolute Error
    metrics["max_abs_error"] = float(np.max(np.abs(diff)))
    
    # RMSE (Root Mean Squared Error)
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    
    # Relative Error (where reference is non-zero)
    non_zero_mask = np.abs(ref) > 1e-10
    if np.any(non_zero_mask):
        rel_errors = np.abs(diff[non_zero_mask]) / np.abs(ref[non_zero_mask])
        metrics["mean_relative_error"] = float(np.mean(rel_errors))
        metrics["max_relative_error"] = float(np.max(rel_errors))
    else:
        metrics["mean_relative_error"] = float("nan")
        metrics["max_relative_error"] = float("nan")
    
    # Cosine Similarity
    try:
        ref_norm = np.linalg.norm(ref)
        cand_norm = np.linalg.norm(cand)
        if ref_norm > 1e-10 and cand_norm > 1e-10:
            metrics["cosine_similarity"] = float(np.dot(ref, cand) / (ref_norm * cand_norm))
        else:
            metrics["cosine_similarity"] = float("nan")
    except Exception:
        metrics["cosine_similarity"] = float("nan")
    
    # Error distribution statistics
    metrics["error_mean"] = float(np.mean(diff))
    metrics["error_std"] = float(np.std(diff))
    metrics["error_skewness"] = float(_compute_skewness(diff))
    
    return metrics


def _compute_skewness(arr: np.ndarray) -> float:
    """Compute skewness of an array."""
    n = len(arr)
    if n < 3:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-10:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def find_divergence_onset(
    pcc_values: List[float],
    threshold: float = 0.99,
    sustained_drop_steps: int = 3
) -> Dict[str, Any]:
    """
    Detect the onset of divergence in PCC progression.
    
    Args:
        pcc_values: List of PCC values for each step
        threshold: PCC threshold below which divergence is considered
        sustained_drop_steps: Number of consecutive steps below threshold to confirm
        
    Returns:
        Dictionary containing:
        - divergence_step: First step where sustained drop begins (-1 if none)
        - divergence_pcc: PCC value at divergence point
        - is_monotonic: Whether degradation is monotonic
        - is_sudden: Whether there's a sudden drop (>0.05 in one step)
        - degradation_pattern: "monotonic", "sudden", "oscillating", or "stable"
        - min_pcc: Minimum PCC observed
        - final_pcc: Final PCC value
    """
    result = {
        "threshold": threshold,
        "num_steps": len(pcc_values),
        "divergence_step": -1,
        "divergence_pcc": None,
        "is_monotonic": False,
        "is_sudden": False,
        "degradation_pattern": "stable",
        "min_pcc": float(min(pcc_values)) if pcc_values else None,
        "final_pcc": pcc_values[-1] if pcc_values else None,
    }
    
    if not pcc_values:
        return result
    
    # Find first step where PCC drops below threshold
    below_threshold_count = 0
    for i, pcc in enumerate(pcc_values):
        if pcc < threshold:
            below_threshold_count += 1
            if below_threshold_count >= sustained_drop_steps:
                result["divergence_step"] = i - sustained_drop_steps + 1
                result["divergence_pcc"] = pcc_values[result["divergence_step"]]
                break
        else:
            below_threshold_count = 0
    
    # Check for sudden drops
    for i in range(1, len(pcc_values)):
        drop = pcc_values[i-1] - pcc_values[i]
        if drop > 0.05:  # More than 5% drop in one step
            result["is_sudden"] = True
            if result["divergence_step"] == -1:
                result["divergence_step"] = i
                result["divergence_pcc"] = pcc_values[i]
            break
    
    # Check if degradation is monotonic
    if len(pcc_values) > 1:
        diffs = [pcc_values[i+1] - pcc_values[i] for i in range(len(pcc_values)-1)]
        negative_diffs = sum(1 for d in diffs if d < -0.001)
        positive_diffs = sum(1 for d in diffs if d > 0.001)
        
        if negative_diffs > 0.7 * len(diffs) and positive_diffs < 0.1 * len(diffs):
            result["is_monotonic"] = True
            result["degradation_pattern"] = "monotonic"
        elif result["is_sudden"]:
            result["degradation_pattern"] = "sudden"
        elif negative_diffs > 0.3 * len(diffs) and positive_diffs > 0.3 * len(diffs):
            result["degradation_pattern"] = "oscillating"
    
    return result


def analyze_spatial_error(
    reference: Union[torch.Tensor, np.ndarray],
    candidate: Union[torch.Tensor, np.ndarray],
    expected_shape: Tuple[int, ...] = None
) -> Dict[str, Any]:
    """
    Analyze spatial distribution of errors for 4D tensors (B, C, H, W).
    
    Args:
        reference: Ground truth tensor
        candidate: Comparison tensor
        expected_shape: Expected (B, C, H, W) shape to reshape to
        
    Returns:
        Dictionary containing:
        - error_heatmap_stats: Statistics of error per spatial location
        - channel_errors: Per-channel error statistics
        - quadrant_errors: Error statistics per image quadrant
        - edge_vs_center: Comparison of edge vs center errors
    """
    # Convert to numpy
    if isinstance(reference, torch.Tensor):
        ref = reference.detach().cpu().float().numpy()
    else:
        ref = np.asarray(reference, dtype=np.float32)
        
    if isinstance(candidate, torch.Tensor):
        cand = candidate.detach().cpu().float().numpy()
    else:
        cand = np.asarray(candidate, dtype=np.float32)
    
    # Try to reshape to (B, C, H, W) if needed
    if expected_shape is not None and ref.shape != expected_shape:
        try:
            ref = ref.reshape(expected_shape)
            cand = cand.reshape(expected_shape)
        except Exception:
            pass
    
    result = {"shape": list(ref.shape)}
    
    # Compute absolute error map
    abs_error = np.abs(ref - cand)
    
    if len(ref.shape) == 4:
        B, C, H, W = ref.shape
        result["dimensions"] = {"B": B, "C": C, "H": H, "W": W}
        
        # Spatial error map (mean over batch and channels)
        spatial_error = np.mean(abs_error, axis=(0, 1))
        result["spatial_error_stats"] = {
            "mean": float(np.mean(spatial_error)),
            "std": float(np.std(spatial_error)),
            "max": float(np.max(spatial_error)),
            "max_location": [int(x) for x in np.unravel_index(np.argmax(spatial_error), spatial_error.shape)]
        }
        
        # Per-channel errors
        channel_errors = []
        for c in range(C):
            ch_err = abs_error[:, c, :, :]
            channel_errors.append({
                "channel": c,
                "mean": float(np.mean(ch_err)),
                "max": float(np.max(ch_err)),
                "std": float(np.std(ch_err))
            })
        result["channel_errors"] = channel_errors
        
        # Quadrant analysis
        mid_h, mid_w = H // 2, W // 2
        quadrants = {
            "top_left": abs_error[:, :, :mid_h, :mid_w],
            "top_right": abs_error[:, :, :mid_h, mid_w:],
            "bottom_left": abs_error[:, :, mid_h:, :mid_w],
            "bottom_right": abs_error[:, :, mid_h:, mid_w:]
        }
        result["quadrant_errors"] = {
            name: {"mean": float(np.mean(q)), "max": float(np.max(q))}
            for name, q in quadrants.items()
        }
        
        # Edge vs center analysis (10% border)
        border = max(1, min(H, W) // 10)
        center = abs_error[:, :, border:-border, border:-border] if border < H//2 and border < W//2 else abs_error
        edge_mask = np.ones_like(abs_error, dtype=bool)
        if border < H//2 and border < W//2:
            edge_mask[:, :, border:-border, border:-border] = False
        edge = abs_error[edge_mask]
        
        result["edge_vs_center"] = {
            "edge_mean": float(np.mean(edge)),
            "center_mean": float(np.mean(center)),
            "edge_max": float(np.max(edge)),
            "center_max": float(np.max(center)),
            "ratio": float(np.mean(edge) / np.mean(center)) if np.mean(center) > 1e-10 else float("nan")
        }
    
    return result


def save_step_data(
    step_index: int,
    output_dir: str,
    pytorch_stats: Dict[str, Any],
    tt_stats: Dict[str, Any],
    comparison: Dict[str, Any],
    metadata: Dict[str, Any] = None
) -> str:
    """
    Save step analysis data to JSON file.
    
    Args:
        step_index: Current denoising step (0-indexed)
        output_dir: Directory to save files
        pytorch_stats: Statistics from PyTorch reference
        tt_stats: Statistics from TT implementation
        comparison: Comparison metrics between the two
        metadata: Additional metadata (timestep, etc.)
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    data = {
        "step_index": step_index,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
        "pytorch": pytorch_stats,
        "tt": tt_stats,
        "comparison": comparison
    }
    
    filepath = os.path.join(output_dir, f"step_{step_index:03d}.json")
    
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
            # Ensure write is flushed
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        # Fallback to stderr
        print(f"ERROR saving step {step_index}: {e}", file=sys.stderr)
        raise
    
    return filepath


def load_step_data(filepath: str) -> Dict[str, Any]:
    """
    Load step data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed JSON data as dictionary
    """
    with open(filepath, "r") as f:
        return json.load(f)


def aggregate_run_data(output_dir: str) -> Dict[str, Any]:
    """
    Aggregate all step files into a summary.
    
    Args:
        output_dir: Directory containing step_XXX.json files
        
    Returns:
        Summary dictionary with aggregated statistics and analysis
    """
    step_files = sorted(Path(output_dir).glob("step_*.json"))
    
    if not step_files:
        return {"error": "No step files found", "output_dir": output_dir}
    
    steps_data = []
    pcc_values = []
    mse_values = []
    max_errors = []
    
    for filepath in step_files:
        data = load_step_data(str(filepath))
        steps_data.append(data)
        
        if "comparison" in data:
            comp = data["comparison"]
            if "pcc" in comp and not np.isnan(comp["pcc"]):
                pcc_values.append(comp["pcc"])
            if "mse" in comp and not np.isnan(comp["mse"]):
                mse_values.append(comp["mse"])
            if "max_abs_error" in comp and not np.isnan(comp["max_abs_error"]):
                max_errors.append(comp["max_abs_error"])
    
    summary = {
        "output_dir": output_dir,
        "num_steps": len(steps_data),
        "timestamp": datetime.now().isoformat(),
        "pcc_progression": pcc_values,
        "mse_progression": mse_values,
        "max_error_progression": max_errors,
    }
    
    # Compute summary statistics
    if pcc_values:
        summary["pcc_stats"] = {
            "initial": pcc_values[0],
            "final": pcc_values[-1],
            "min": min(pcc_values),
            "max": max(pcc_values),
            "mean": float(np.mean(pcc_values)),
            "degradation": pcc_values[0] - pcc_values[-1]
        }
        
        # Find divergence
        summary["divergence_analysis"] = find_divergence_onset(pcc_values)
    
    if mse_values:
        summary["mse_stats"] = {
            "initial": mse_values[0],
            "final": mse_values[-1],
            "min": min(mse_values),
            "max": max(mse_values),
            "mean": float(np.mean(mse_values)),
            "growth_factor": mse_values[-1] / mse_values[0] if mse_values[0] > 1e-10 else float("nan")
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    summary["summary_path"] = summary_path
    
    return summary


def generate_analysis_csv(output_dir: str) -> str:
    """
    Generate a CSV file with per-step analysis data.
    
    Args:
        output_dir: Directory containing step_XXX.json files
        
    Returns:
        Path to generated CSV file
    """
    step_files = sorted(Path(output_dir).glob("step_*.json"))
    
    csv_lines = ["step,timestep,pcc,mse,max_abs_error,mae,pytorch_mean,pytorch_std,tt_mean,tt_std"]
    
    for filepath in step_files:
        data = load_step_data(str(filepath))
        step = data.get("step_index", -1)
        timestep = data.get("metadata", {}).get("timestep", -1)
        
        comp = data.get("comparison", {})
        pcc = comp.get("pcc", "nan")
        mse = comp.get("mse", "nan")
        max_err = comp.get("max_abs_error", "nan")
        mae = comp.get("mae", "nan")
        
        pt = data.get("pytorch", {})
        pt_mean = pt.get("mean", "nan")
        pt_std = pt.get("std", "nan")
        
        tt = data.get("tt", {})
        tt_mean = tt.get("mean", "nan")
        tt_std = tt.get("std", "nan")
        
        csv_lines.append(f"{step},{timestep},{pcc},{mse},{max_err},{mae},{pt_mean},{pt_std},{tt_mean},{tt_std}")
    
    csv_path = os.path.join(output_dir, "analysis.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    
    return csv_path


def identify_component_contribution(
    raw_unet_comparison: Dict[str, Any],
    post_guidance_comparison: Dict[str, Any],
    scheduler_output_comparison: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Determine which component contributes most to the error.
    
    Args:
        raw_unet_comparison: Comparison metrics for raw UNet output
        post_guidance_comparison: Comparison metrics after guidance computation
        scheduler_output_comparison: Comparison metrics after scheduler step
        
    Returns:
        Analysis of which component introduces the most error
    """
    components = {
        "raw_unet": raw_unet_comparison,
        "post_guidance": post_guidance_comparison,
        "scheduler_output": scheduler_output_comparison
    }
    
    result = {
        "component_pcc": {},
        "component_mse": {},
        "error_introduction": {}
    }
    
    prev_mse = 0.0
    for name, comp in components.items():
        pcc = comp.get("pcc", float("nan"))
        mse = comp.get("mse", float("nan"))
        
        result["component_pcc"][name] = pcc
        result["component_mse"][name] = mse
        
        # Calculate error introduced by this component
        if not np.isnan(mse):
            result["error_introduction"][name] = mse - prev_mse
            prev_mse = mse
    
    # Identify primary culprit
    if result["error_introduction"]:
        max_contributor = max(result["error_introduction"].items(), key=lambda x: x[1])
        result["primary_error_source"] = max_contributor[0]
        result["primary_error_contribution"] = max_contributor[1]
    
    # Check if UNet is maintaining good accuracy
    unet_pcc = result["component_pcc"].get("raw_unet", 0)
    if unet_pcc > 0.99:
        result["unet_accuracy"] = "excellent"
    elif unet_pcc > 0.95:
        result["unet_accuracy"] = "good"
    elif unet_pcc > 0.90:
        result["unet_accuracy"] = "moderate"
    else:
        result["unet_accuracy"] = "poor"
    
    return result
