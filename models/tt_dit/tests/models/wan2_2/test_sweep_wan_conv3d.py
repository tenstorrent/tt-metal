# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Generalized Wan Conv3d blocking sweep harness.

Sweeps candidate blockings for production-relevant conv3d shapes across
mesh families, resolutions, and cache modes.  Results are persisted to
JSON under .cache/wan_conv3d_blocking_sweeps/<stage>/<mesh_id>/.

Usage examples:

    # Validate harness on 1x8, 480p, all 13 shapes (cheapest)
    FAKE_DEVICE=N300 pytest test_sweep_wan_conv3d.py -k "validate_1x8_480p" -x -s

    # Production 2x4, 480p, decoder only
    pytest test_sweep_wan_conv3d.py -k "prod_2x4_480p and dec" -x -s

    # Single case debug
    pytest test_sweep_wan_conv3d.py -k "validate_1x8_480p and dec_conv_in_480p__cache_none" -x -s

The harness monkey-patches get_conv3d_config() to inject candidate blockings,
runs timed iterations, and records per-candidate latency statistics.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from loguru import logger

import ttnn

from ....parallel.config import ParallelFactor, VaeHWParallelConfig
from ....parallel.manager import CCLManager
from ....utils.conv3d import aligned_channels, conv_pad_height, conv_pad_in_channels
from ....utils.tensor import typed_tensor_2dshard
from .wan_conv3d_cases import (
    SWEEP_STAGES,
    SWEEP_STAGES_BY_ID,
    MeshTarget,
    SweepStage,
    build_sweep_manifest,
    generate_candidates,
    get_current_default,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parents[5] / ".cache" / "wan_conv3d_blocking_sweeps"
WARMUP_ITERS = 2
MEASURE_ITERS = 5
TILE_WIDTH = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(blocking: tuple, weights_dtype, grid_size) -> ttnn.Conv3dConfig:
    """Build a Conv3dConfig from a blocking tuple."""
    C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = blocking
    return ttnn.Conv3dConfig(
        weights_dtype=weights_dtype,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=T_out_block,
        W_out_block=W_out_block,
        H_out_block=H_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        compute_with_storage_grid_size=grid_size,
    )


def _run_timed_conv3d(
    tt_model,
    tt_input_tensor,
    tt_cache_tensor,
    logical_h: int,
    n_warmup: int = WARMUP_ITERS,
    n_measure: int = MEASURE_ITERS,
) -> dict:
    """
    Run a conv3d model with warmup + timed iterations.

    Returns dict with timing stats and status.
    """
    try:
        # Warmup
        for _ in range(n_warmup):
            _ = tt_model(tt_input_tensor, cache_x_BTHWC=tt_cache_tensor, logical_h=logical_h)
            ttnn.synchronize_device(tt_model.mesh_device)

        # Timed runs
        times_ms = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            _ = tt_model(tt_input_tensor, cache_x_BTHWC=tt_cache_tensor, logical_h=logical_h)
            ttnn.synchronize_device(tt_model.mesh_device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        return dict(
            status="ok",
            times_ms=times_ms,
            median_ms=statistics.median(times_ms),
            mean_ms=statistics.mean(times_ms),
            stdev_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            min_ms=min(times_ms),
            max_ms=max(times_ms),
        )
    except Exception as e:
        logger.warning(f"Conv3d run failed: {e}")
        return dict(status="error", error=str(e), times_ms=[], median_ms=float("inf"))


def _save_result(stage_id: str, mesh_id: str, sweep_id: str, result: dict):
    """Persist a sweep result to JSON."""
    out_dir = RESULTS_DIR / stage_id / mesh_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sweep_id}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved result to {out_path}")


def _blocking_label(blocking: tuple) -> str:
    C_in, C_out, T, H, W = blocking
    return f"Cin_blk={C_in:>3} Cout_blk={C_out:>3} T={T:>2} H={H:>3} W={W:>3}"


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def _create_tt_model_and_inputs(
    mesh_device,
    case: dict,
    mesh_target: MeshTarget,
    cache_len: int | None,
    dtype=ttnn.DataType.BFLOAT16,
):
    """
    Create a TT WanCausalConv3d model and matching input tensors for a sweep case.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d as TorchWanCausalConv3d

    from ....models.vae.vae_wan2_1 import WanCausalConv3d

    B = case["B"]
    C_in = case["C_in"]
    C_out = case["C_out"]
    T = case["T"]
    H = case["H"]
    W = case["W"]
    kernel_size = case["kernel_size"]
    stride = case["stride"]
    padding = case["padding"]

    h_axis = mesh_target.h_axis
    w_axis = mesh_target.w_axis

    tt_input_dtype = ttnn.bfloat16 if dtype == ttnn.DataType.BFLOAT16 else ttnn.float32
    torch_dtype = torch.float32

    # Create torch reference model to get weights
    torch_model = TorchWanCausalConv3d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    torch_model.eval()

    # Create CCL manager and parallel config
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )

    # Create TT model
    tt_model = WanCausalConv3d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        mesh_device=mesh_device,
        stride=stride,
        padding=padding,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        dtype=dtype,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    # Create input tensor
    torch_input = torch.randn(B, C_in, T, H, W, dtype=torch_dtype)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # BCTHW -> BTHWC
    tt_input = conv_pad_in_channels(tt_input)
    tt_input, logical_h = conv_pad_height(tt_input, parallel_config.height_parallel.factor)
    tt_input = typed_tensor_2dshard(
        tt_input,
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={h_axis: 2, w_axis: 3},
        dtype=tt_input_dtype,
    )

    # Create cache tensor if needed
    tt_cache = None
    if cache_len is not None:
        torch_cache = torch.randn(B, C_in, cache_len, H, W, dtype=torch_dtype)
        tt_cache = torch_cache.permute(0, 2, 3, 4, 1)
        tt_cache = conv_pad_in_channels(tt_cache)
        tt_cache, _ = conv_pad_height(tt_cache, parallel_config.height_parallel.factor)
        tt_cache = typed_tensor_2dshard(
            tt_cache,
            mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={h_axis: 2, w_axis: 3},
            dtype=tt_input_dtype,
        )

    return tt_model, tt_input, tt_cache, logical_h


# ---------------------------------------------------------------------------
# Staged sweep logic
# ---------------------------------------------------------------------------


def run_staged_sweep(
    mesh_device,
    case: dict,
    mesh_target: MeshTarget,
    cache_len: int | None,
    dtype=ttnn.DataType.BFLOAT16,
) -> dict:
    """
    Run a staged blocking sweep for a single case.

    Strategy:
    1. Run baseline (current default or fallback)
    2. Sweep spatial (H_out_block, W_out_block) with baseline C_in/C_out
    3. Pick best spatial, then sweep C_out_block
    4. Pick best C_out, then sweep C_in_block
    5. Record all results and the overall winner

    Returns a result dict suitable for JSON serialization.
    """
    candidates = generate_candidates(case, mesh_target)
    current_default = get_current_default(case)
    C_in_aligned = aligned_channels(case["C_in"])

    # Determine baseline blocking
    if current_default is not None:
        baseline = current_default
    else:
        # Fallback: full C_in, 32 C_out, 1x1 spatial
        baseline = (C_in_aligned, 32, 1, 1, 1)

    logger.info(f"Baseline blocking: {_blocking_label(baseline)}")

    tt_model, tt_input, tt_cache, logical_h = _create_tt_model_and_inputs(
        mesh_device,
        case,
        mesh_target,
        cache_len,
        dtype,
    )

    all_results = []
    best_blocking = baseline
    best_ms = float("inf")
    best_stage = "baseline"

    def _try_blocking(blocking: tuple, stage_name: str) -> dict:
        """Monkey-patch get_conv3d_config and run timed conv3d."""
        nonlocal best_blocking, best_ms, best_stage

        label = _blocking_label(blocking)

        def patched_get_conv3d_config(in_channels, out_channels, kernel_size, weights_dtype, grid_size):
            return _make_config(blocking, weights_dtype, grid_size)

        with patch("models.tt_dit.utils.conv3d.get_conv3d_config", patched_get_conv3d_config):
            # Re-create TT model with patched config
            tt_model_patched, _, _, _ = _create_tt_model_and_inputs(
                mesh_device,
                case,
                mesh_target,
                cache_len,
                dtype,
            )
            timing = _run_timed_conv3d(tt_model_patched, tt_input, tt_cache, logical_h)

        entry = dict(
            stage=stage_name,
            blocking=list(blocking),
            label=label,
            ms=timing["median_ms"],
            status=timing["status"],
        )
        if timing["status"] == "error":
            entry["error"] = timing.get("error", "unknown")
        all_results.append(entry)

        if timing["status"] == "ok" and timing["median_ms"] < best_ms:
            best_ms = timing["median_ms"]
            best_blocking = blocking
            best_stage = stage_name
            logger.info(f"  NEW BEST [{stage_name}]: {label} -> {timing['median_ms']:.3f} ms")
        else:
            logger.info(f"  [{stage_name}]: {label} -> {timing['median_ms']:.3f} ms ({timing['status']})")

        return entry

    # --- Stage 0: baseline ---
    _try_blocking(baseline, "baseline")

    # --- Stage 1: spatial sweep ---
    # Use baseline C_in/C_out, sweep H and W
    C_in_blk, C_out_blk = baseline[0], baseline[1]
    T_blk = candidates["T_out_block"]

    for h in candidates["h_candidates"]:
        for w in candidates["w_candidates"]:
            blocking = (C_in_blk, C_out_blk, T_blk, h, w)
            if blocking == baseline:
                continue  # already tested
            _try_blocking(blocking, "spatial")

    # --- Stage 2: C_out sweep ---
    # Use best spatial from stage 1
    best_spatial_H = best_blocking[3]
    best_spatial_W = best_blocking[4]

    for c_out in candidates["c_out_candidates"]:
        blocking = (C_in_blk, c_out, T_blk, best_spatial_H, best_spatial_W)
        if blocking == best_blocking:
            continue
        _try_blocking(blocking, "c_out")

    # --- Stage 3: C_in sweep ---
    best_C_out = best_blocking[1]
    for c_in in candidates["c_in_candidates"]:
        blocking = (c_in, best_C_out, T_blk, best_spatial_H, best_spatial_W)
        if blocking == best_blocking:
            continue
        _try_blocking(blocking, "c_in")

    return dict(
        stage=case.get("stage_id", "unknown"),
        case_id=case["case_id"],
        path=case["path"],
        resolution=case["resolution"],
        kind=case["kind"],
        cache_len=cache_len,
        mesh_id=mesh_target.mesh_id,
        mesh_shape=list(mesh_target.mesh_shape),
        h_axis=mesh_target.h_axis,
        w_axis=mesh_target.w_axis,
        baseline_blocking=list(baseline),
        best_blocking=list(best_blocking),
        best_stage=best_stage,
        best_ms=best_ms,
        results=all_results,
    )


# ---------------------------------------------------------------------------
# Pytest parametrization
# ---------------------------------------------------------------------------


def _build_param_list():
    """Build pytest parametrize list from all stages and cases."""
    params = []
    ids = []
    for stage in SWEEP_STAGES:
        manifest = build_sweep_manifest(stage)
        for entry in manifest:
            params.append((stage, entry))
            ids.append(f"{stage.stage_id}/{entry['sweep_id']}")
    return params, ids


_PARAMS, _IDS = _build_param_list()


@pytest.mark.parametrize("stage, entry", _PARAMS, ids=_IDS)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 8),
        (2, 4),
        (4, 8),
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_sweep_conv3d(mesh_device, device_params, stage: SweepStage, entry: dict):
    """
    Sweep blocking candidates for a single conv3d case.

    Automatically skips if the mesh_device shape doesn't match the stage's mesh target.
    Results are saved to .cache/wan_conv3d_blocking_sweeps/.
    """
    mesh_target = stage.mesh_target
    actual_shape = tuple(mesh_device.shape)

    # Skip if mesh doesn't match this stage's target
    if actual_shape != mesh_target.mesh_shape:
        pytest.skip(f"Mesh shape {actual_shape} doesn't match stage target {mesh_target.mesh_shape}")

    case_id = entry["case_id"]
    cache_len = entry["cache_len"]
    sweep_id = entry["sweep_id"]

    logger.info(f"=== Sweep: {stage.stage_id} / {sweep_id} ===")
    logger.info(f"  Case: {case_id} | Path: {entry['path']}")
    logger.info(
        f"  Shape: B={entry['B']} C_in={entry['C_in']} C_out={entry['C_out']} "
        f"T={entry['T']} H={entry['H']} W={entry['W']}"
    )
    logger.info(f"  Kernel: {entry['kernel_size']} | Mesh: {mesh_target.mesh_id}")
    logger.info(f"  Cache: {cache_len}")

    # Run the staged sweep
    result = run_staged_sweep(
        mesh_device=mesh_device,
        case=entry,
        mesh_target=mesh_target,
        cache_len=cache_len,
    )

    # Save results
    _save_result(stage.stage_id, mesh_target.mesh_id, sweep_id, result)

    # Log summary
    logger.info(
        f"  Winner: {_blocking_label(tuple(result['best_blocking']))} "
        f"({result['best_ms']:.3f} ms, stage={result['best_stage']})"
    )

    baseline_ms = result["results"][0]["ms"] if result["results"] else float("inf")
    if result["best_ms"] < baseline_ms:
        improvement = (1 - result["best_ms"] / baseline_ms) * 100
        logger.info(f"  Improvement over baseline: {improvement:.1f}%")
    else:
        logger.info(f"  Baseline was already optimal")


# ---------------------------------------------------------------------------
# Convenience: run a single stage from the command line
# ---------------------------------------------------------------------------


def _build_single_stage_params(stage_id: str):
    """Build params for a single stage — used for focused runs."""
    stage = SWEEP_STAGES_BY_ID[stage_id]
    manifest = build_sweep_manifest(stage)
    return [(stage, entry) for entry in manifest]
