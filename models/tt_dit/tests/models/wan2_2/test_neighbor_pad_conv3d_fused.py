# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for halo-only NeighborPad + Conv3d correctness.

Tests the production BH Loud Box 2×4 configuration (H_AXIS=0, W_AXIS=1, NUM_LINKS=2).
Shapes are taken from the VAE decoder layers that trigger the halo path (T_out_block > 1).

test_old_halo_vs_full_padded:
  Validates that the old halo path (fabric_only NP + conv3d reading from compact
  halo buffer) produces output identical to the full-padded path (NP writes full
  output tensor, conv3d reads from it).  Any difference is a bug in the halo buffer
  layout, NP fabric_only addressing, or conv3d gather_rows_halo indexing.

test_neighbor_pad_conv3d_fused_changing_inputs:
  Tests the fused NP+Conv3d pipelined path across multiple seeds with a shared
  CCLManager.  Stale W-halo data from seed N would corrupt seed N+1.
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.vae_wan2_1 import WanCausalConv3d
from ....parallel.config import ParallelFactor, VaeHWParallelConfig
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.conv3d import conv_pad_height, conv_pad_in_channels
from ....utils.tensor import typed_tensor_2dshard

HALO_VS_FULL_PCC = 0.99999
TORCH_REF_PCC = 0.99999
MAX_RMSE = 0.010


def _make_model_and_manager(mesh_device, B, C_in, C_out, kernel_size, padding, h_axis, w_axis, num_links, dtype):
    """Create ONE CCLManager + WanCausalConv3d pair (matches production usage)."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d as TorchWanCausalConv3d

    h_factor = tuple(mesh_device.shape)[h_axis]
    w_factor = tuple(mesh_device.shape)[w_axis]
    stride = 1

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=h_factor, mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=w_factor, mesh_axis=w_axis),
    )

    torch.manual_seed(42)
    torch_model = TorchWanCausalConv3d(
        in_channels=C_in, out_channels=C_out, kernel_size=kernel_size, stride=stride, padding=padding
    )
    torch_model.eval()

    return torch_model, ccl_manager, parallel_config, h_factor, w_factor


def _run_one_seed(
    mesh_device,
    torch_model,
    ccl_manager,
    parallel_config,
    B,
    C_in,
    C_out,
    T,
    H,
    W,
    kernel_size,
    padding,
    h_axis,
    w_axis,
    h_factor,
    w_factor,
    dtype,
    seed,
    force_no_fused=False,
    force_old_halo=False,
):
    """Run one inference call with a specific seed; return (torch_output, tt_output_torch)."""
    tt_input_dtype = ttnn.bfloat16 if dtype == ttnn.DataType.BFLOAT16 else ttnn.float32
    torch_dtype = torch.float32
    stride = 1

    H_out_per_device = H // h_factor
    W_out_per_device = W // w_factor

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
        H_out=H_out_per_device,
        W_out=W_out_per_device,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    if force_no_fused:
        tt_model.conv_config.use_h_halo_buffer = False
        tt_model.conv_config.input_progress_t_batch_size = 0
    if force_old_halo:
        tt_model.conv_config.use_h_halo_buffer = True
        tt_model.conv_config.input_progress_t_batch_size = 0

    logger.info(
        f"seed={seed} use_h_halo_buffer={tt_model.conv_config.use_h_halo_buffer} "
        f"T_out_block={tt_model.conv_config.T_out_block} "
        f"force_no_fused={force_no_fused} force_old_halo={force_old_halo}"
    )
    if not force_no_fused and not force_old_halo:
        assert tt_model.conv_config.use_h_halo_buffer, (
            f"Fused path not enabled for this shape (T_out_block={tt_model.conv_config.T_out_block}). "
            "Adjust T so that T_out >= T_out_block > 1."
        )

    torch.manual_seed(seed + 999)
    torch_input = torch.randn(B, C_in, T, H, W, dtype=torch_dtype)

    tt_input = torch_input.permute(0, 2, 3, 4, 1)
    tt_input = conv_pad_in_channels(tt_input)
    tt_input, logical_h = conv_pad_height(tt_input, parallel_config.height_parallel.factor)
    tt_input = typed_tensor_2dshard(
        tt_input,
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={h_axis: 2, w_axis: 3},
        dtype=tt_input_dtype,
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_output = tt_model(tt_input, logical_h=logical_h)

    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
    )
    if logical_h != tt_output_torch.shape[2]:
        tt_output_torch = tt_output_torch[:, :logical_h, :, :, :]
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)
    if tt_output_torch.shape[1] != C_out:
        tt_output_torch = tt_output_torch[:, :C_out]

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)
    return torch_output, tt_output_torch


def _log_boundary_diagnostics(label, a, b, h_factor, w_factor):
    """Log per-device-boundary and regional error diagnostics between two [B,C,T,H,W] tensors."""
    diff = (a - b).abs()
    H, W = a.shape[3], a.shape[4]
    H_dev, W_dev = H // h_factor, W // w_factor

    logger.info(
        f"[{label}] max_diff={diff.max():.6g}  mean_diff={diff.mean():.6g}  "
        f"RMSE/sigma={diff.pow(2).mean().sqrt() / a.std():.6f}  "
        f"n_nonzero(>1e-3)={(diff > 1e-3).sum().item()}"
    )

    # Top-5 error positions
    flat_idx = diff.flatten().topk(min(5, diff.numel())).indices
    for rank, idx in enumerate(flat_idx):
        coords = []
        tmp = idx.item()
        for dim_size in reversed(diff.shape):
            coords.append(tmp % dim_size)
            tmp //= dim_size
        bi, c, t, h, w = list(reversed(coords))
        dev_h, dev_w = h // H_dev, w // W_dev
        val_a = a[bi, c, t, h, w].item()
        val_b = b[bi, c, t, h, w].item()
        logger.info(
            f"  #{rank} B={bi} C={c} T={t} H={h} W={w} "
            f"(dev[{dev_h},{dev_w}] local h={h % H_dev} w={w % W_dev})  "
            f"diff={diff[bi, c, t, h, w]:.6g}  a={val_a:.6g}  b={val_b:.6g}"
        )

    # H-boundary analysis: rows adjacent to device H splits
    for hi in range(1, h_factor):
        boundary_h = hi * H_dev
        for offset in [-1, 0]:
            row = boundary_h + offset
            if 0 <= row < H:
                mean_err = diff[:, :, :, row, :].mean().item()
                max_err = diff[:, :, :, row, :].max().item()
                logger.info(f"  H-boundary row {row}: mean_diff={mean_err:.6g}  max_diff={max_err:.6g}")

    # W-boundary analysis: cols adjacent to device W splits
    for wi in range(1, w_factor):
        boundary_w = wi * W_dev
        for offset in [-1, 0]:
            col = boundary_w + offset
            if 0 <= col < W:
                mean_err = diff[:, :, :, :, col].mean().item()
                max_err = diff[:, :, :, :, col].max().item()
                logger.info(f"  W-boundary col {col}: mean_diff={mean_err:.6g}  max_diff={max_err:.6g}")

    # Edge rows/cols (mesh boundary)
    for tag, sl in [
        ("H=0", diff[:, :, :, 0, :]),
        (f"H={H-1}", diff[:, :, :, -1, :]),
        ("W=0", diff[:, :, :, :, 0]),
        (f"W={W-1}", diff[:, :, :, :, -1]),
    ]:
        logger.info(f"  Edge {tag}: mean_diff={sl.mean():.6g}  max_diff={sl.max():.6g}")

    # Interior-only vs boundary-only PCC
    interior_mask = torch.ones_like(diff, dtype=torch.bool)
    for hi in range(h_factor + 1):
        row = hi * H_dev
        for offset in [-1, 0]:
            r = row + offset
            if 0 <= r < H:
                interior_mask[:, :, :, r, :] = False
    for wi in range(w_factor + 1):
        col = wi * W_dev
        for offset in [-1, 0]:
            c = col + offset
            if 0 <= c < W:
                interior_mask[:, :, :, :, c] = False

    interior_a = a[interior_mask].flatten().to(torch.float64)
    interior_b = b[interior_mask].flatten().to(torch.float64)
    boundary_a = a[~interior_mask].flatten().to(torch.float64)
    boundary_b = b[~interior_mask].flatten().to(torch.float64)

    if interior_a.numel() > 1:
        cov_int = torch.cov(torch.stack([interior_a, interior_b]))
        pcc_int = cov_int[0, 1] / (cov_int[0, 0].sqrt() * cov_int[1, 1].sqrt())
        logger.info(f"  Interior PCC = {pcc_int.item() * 100:.6f} %  ({interior_a.numel()} elements)")
    if boundary_a.numel() > 1:
        cov_bnd = torch.cov(torch.stack([boundary_a, boundary_b]))
        pcc_bnd = cov_bnd[0, 1] / (cov_bnd[0, 0].sqrt() * cov_bnd[1, 1].sqrt())
        logger.info(f"  Boundary PCC = {pcc_bnd.item() * 100:.6f} %  ({boundary_a.numel()} elements)")


# ---------------------------------------------------------------------------
# Test: old halo path vs full-padded path (and both vs PyTorch reference)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "B, C_in, C_out, T, H, W, kernel_size, padding",
    [
        (1, 96, 96, 14, 480, 832, 3, 1),
        (1, 192, 192, 18, 240, 416, 3, 1),
    ],
    ids=["up3_res_T14", "up2_res_T18"],
)
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis, num_links",
    [((2, 4), 0, 1, 2)],
    ids=["bh_lb_2x4"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_old_halo_vs_full_padded(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype
):
    """Verify that the old halo path produces output identical to the full-padded path.

    Both paths execute the same mathematical operation (NP + conv3d).  The only
    difference is WHERE conv3d reads boundary data from:
      - Full-padded: conv3d reads everything from the NP output tensor.
      - Old halo: conv3d reads interior from the original input tensor, boundaries
        from the compact halo buffer.

    If the halo buffer values and conv3d gather indexing are correct, the two outputs
    must be identical (bit-for-bit in the ideal case, within floating-point noise in
    practice).  Any systematic difference at device boundaries indicates a bug.
    """
    torch_model, ccl_manager, parallel_config, h_factor, w_factor = _make_model_and_manager(
        mesh_device, B, C_in, C_out, kernel_size, padding, h_axis, w_axis, num_links, dtype
    )

    logger.info("--- Running FULL-PADDED path (golden baseline) ---")
    torch_ref, tt_full = _run_one_seed(
        mesh_device,
        torch_model,
        ccl_manager,
        parallel_config,
        B,
        C_in,
        C_out,
        T,
        H,
        W,
        kernel_size,
        padding,
        h_axis,
        w_axis,
        h_factor,
        w_factor,
        dtype,
        seed=0,
        force_no_fused=True,
    )

    logger.info("--- Running OLD HALO path ---")
    _, tt_halo = _run_one_seed(
        mesh_device,
        torch_model,
        ccl_manager,
        parallel_config,
        B,
        C_in,
        C_out,
        T,
        H,
        W,
        kernel_size,
        padding,
        h_axis,
        w_axis,
        h_factor,
        w_factor,
        dtype,
        seed=0,
        force_old_halo=True,
    )

    # 1) Full-padded vs Old halo — should be near-identical
    logger.info("=" * 72)
    logger.info("FULL-PADDED vs OLD HALO (same seed, same input)")
    logger.info("=" * 72)
    _log_boundary_diagnostics("full_vs_halo", tt_full, tt_halo, h_factor, w_factor)
    assert_quality(tt_full, tt_halo, pcc=HALO_VS_FULL_PCC, relative_rmse=MAX_RMSE)
    logger.info("PASS: full-padded vs old halo")

    # 2) Full-padded vs PyTorch reference
    logger.info("=" * 72)
    logger.info("FULL-PADDED vs PYTORCH REFERENCE")
    logger.info("=" * 72)
    _log_boundary_diagnostics("full_vs_torch", tt_full, torch_ref, h_factor, w_factor)
    assert_quality(torch_ref, tt_full, pcc=TORCH_REF_PCC, relative_rmse=MAX_RMSE)
    logger.info("PASS: full-padded vs pytorch")

    # 3) Old halo vs PyTorch reference
    logger.info("=" * 72)
    logger.info("OLD HALO vs PYTORCH REFERENCE")
    logger.info("=" * 72)
    _log_boundary_diagnostics("halo_vs_torch", tt_halo, torch_ref, h_factor, w_factor)
    assert_quality(torch_ref, tt_halo, pcc=TORCH_REF_PCC, relative_rmse=MAX_RMSE)
    logger.info("PASS: old halo vs pytorch")


# ---------------------------------------------------------------------------
# Test: fused NP+Conv3d with changing inputs (stale W-halo detection)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "B, C_in, C_out, T, H, W, kernel_size, padding",
    [
        (1, 96, 96, 14, 480, 832, 3, 1),
        (1, 192, 192, 18, 240, 416, 3, 1),
    ],
    ids=["up3_res_T14", "up2_res_T18"],
)
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis, num_links",
    [((2, 4), 0, 1, 2)],
    ids=["bh_lb_2x4"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_neighbor_pad_conv3d_fused_changing_inputs(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype
):
    """Test fused NP+Conv3d correctness across N different random inputs.

    Uses ONE shared CCLManager so the W-halo buffer is reused across calls.
    Each seed runs both old_halo (correct baseline) and fused, then compares
    them directly.  Any difference isolates the pipelining race from bf16
    accumulation noise.
    """
    N_SEEDS = 3

    torch_model, ccl_manager, parallel_config, h_factor, w_factor = _make_model_and_manager(
        mesh_device, B, C_in, C_out, kernel_size, padding, h_axis, w_axis, num_links, dtype
    )

    for seed in range(N_SEEDS):
        logger.info(f"=== seed {seed} / {N_SEEDS} ===")

        # Old halo baseline (sequential dispatch — no race)
        torch_output, tt_baseline = _run_one_seed(
            mesh_device,
            torch_model,
            ccl_manager,
            parallel_config,
            B,
            C_in,
            C_out,
            T,
            H,
            W,
            kernel_size,
            padding,
            h_axis,
            w_axis,
            h_factor,
            w_factor,
            dtype,
            seed,
            force_old_halo=True,
        )

        # Fused path (pipelined — may race)
        _, tt_fused = _run_one_seed(
            mesh_device,
            torch_model,
            ccl_manager,
            parallel_config,
            B,
            C_in,
            C_out,
            T,
            H,
            W,
            kernel_size,
            padding,
            h_axis,
            w_axis,
            h_factor,
            w_factor,
            dtype,
            seed,
        )

        # Fused vs old_halo — isolates pipelining race from bf16 noise
        logger.info(f"seed={seed}: FUSED vs OLD_HALO (same seed)")
        _log_boundary_diagnostics("fused_vs_baseline", tt_fused, tt_baseline, h_factor, w_factor)
        assert_quality(tt_baseline, tt_fused, pcc=HALO_VS_FULL_PCC, relative_rmse=MAX_RMSE)

        # Both vs PyTorch
        assert_quality(torch_output, tt_fused, pcc=TORCH_REF_PCC, relative_rmse=MAX_RMSE)
        logger.info(f"seed={seed} PASSED")


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis, num_links",
    [((2, 4), 0, 1, 2)],
    ids=["bh_lb_2x4"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
def test_fused_debug_minimal(mesh_device, h_axis, w_axis, num_links, dtype):
    """Minimal single-seed fused test for hang debugging (no baselines)."""
    B, C_in, C_out, T, H, W, kernel_size, padding = 1, 96, 96, 14, 480, 832, 3, 1
    torch_model, ccl_manager, parallel_config, h_factor, w_factor = _make_model_and_manager(
        mesh_device, B, C_in, C_out, kernel_size, padding, h_axis, w_axis, num_links, dtype
    )
    logger.info("Running fused seed=0 (no baselines)...")
    torch_out, tt_out = _run_one_seed(
        mesh_device,
        torch_model,
        ccl_manager,
        parallel_config,
        B,
        C_in,
        C_out,
        T,
        H,
        W,
        kernel_size,
        padding,
        h_axis,
        w_axis,
        h_factor,
        w_factor,
        dtype,
        seed=0,
    )
    logger.info(f"DONE torch={torch_out.shape} tt={tt_out.shape}")
    assert_quality(torch_out, tt_out, pcc=TORCH_REF_PCC, relative_rmse=MAX_RMSE)
    logger.info("PASSED")
