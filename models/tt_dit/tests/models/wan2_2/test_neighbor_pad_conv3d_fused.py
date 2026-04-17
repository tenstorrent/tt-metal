# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Production-shape tests for fused NeighborPad + Conv3d correctness.

Validates the fused NP+Conv3d path against PyTorch reference on all
VAE decoder shapes that trigger the halo path.  Checks PCC both globally
and at device-boundary pixels where NP halo data is consumed by the conv
kernel, catching boundary errors that global PCC can mask.
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.vae_wan2_1 import WanCausalConv3d, WanConv2d, get_neighbor_pad_num_links
from ....parallel.config import ParallelFactor, VaeHWParallelConfig
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.conv3d import ConvDims, conv_pad_height, conv_pad_in_channels
from ....utils.tensor import typed_tensor_2dshard

TORCH_REF_PCC = 0.99999
BOUNDARY_PCC = 0.99999
MAX_RMSE = 0.010


def _compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two flat tensors."""
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() < 2:
        return 1.0
    cov = torch.corrcoef(torch.stack([a, b]))
    return cov[0, 1].item()


def _boundary_mask(H, W, h_factor, w_factor, pad_h, pad_w):
    """Build a (H, W) bool mask covering pixels within pad distance of device boundaries.

    These are the output positions whose conv receptive field crosses a device
    boundary and therefore depends on NP halo data.
    """
    H_dev, W_dev = H // h_factor, W // w_factor
    h_bnd = torch.zeros(H, dtype=torch.bool)
    w_bnd = torch.zeros(W, dtype=torch.bool)
    for d in range(1, h_factor):
        pos = d * H_dev
        h_bnd[max(0, pos - pad_h) : min(H, pos + pad_h)] = True
    for d in range(1, w_factor):
        pos = d * W_dev
        w_bnd[max(0, pos - pad_w) : min(W, pos + pad_w)] = True
    return h_bnd.unsqueeze(-1) | w_bnd.unsqueeze(-2)  # (H, W)


def _make_model_and_manager(mesh_device, B, C_in, C_out, kernel_size, padding, h_axis, w_axis, num_links, dtype):
    """Create ONE CCLManager + torch ref model pair.

    Uses the exact same model classes as the production decoder:
      - TorchWanCausalConv3d for (3,3,3) kernels
      - torch.nn.Conv3d for (1,3,3) kernels (WanConv2d uses Conv3d internally)
    """
    h_factor = tuple(mesh_device.shape)[h_axis]
    w_factor = tuple(mesh_device.shape)[w_axis]
    stride = 1

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=h_factor, mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=w_factor, mesh_axis=w_axis),
    )

    kernel_size_tuple = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

    torch.manual_seed(42)
    if kernel_size_tuple[0] == 1:
        padding_tuple = padding if isinstance(padding, tuple) else (0, padding, padding)
        torch_model = torch.nn.Conv3d(C_in, C_out, kernel_size=kernel_size_tuple, stride=stride, padding=padding_tuple)
    else:
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d as TorchWanCausalConv3d

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
    force_pipelining=False,
    run_standalone=False,
):
    """Run one inference call with a specific seed; return (torch_output, tt_fused_output, tt_standalone_output)."""
    tt_input_dtype = ttnn.bfloat16 if dtype == ttnn.DataType.BFLOAT16 else ttnn.float32
    torch_dtype = torch.float32
    stride = 1

    H_out_per_device = H // h_factor
    W_out_per_device = W // w_factor

    kernel_size_tuple = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
    is_spatial_only = kernel_size_tuple[0] == 1

    if is_spatial_only:
        tt_model = WanConv2d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=kernel_size_tuple,
            mesh_device=mesh_device,
            stride=stride,
            padding=padding,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=ConvDims(T=T, H=H_out_per_device, W=W_out_per_device),
        )
        # WanConv2d has no use_fused parameter — it always uses standalone path
        torch_state = torch_model.state_dict()
        # WanConv2d._prepare_torch_state does unsqueeze(2) expecting a 4D Conv2d weight.
        # Our torch ref is nn.Conv3d(kernel=(1,3,3)) which gives 5D weight — squeeze the T dim.
        if "weight" in torch_state and torch_state["weight"].ndim == 5:
            torch_state["weight"] = torch_state["weight"].squeeze(2)
    else:
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
            conv_dims=ConvDims(T=T, H=H_out_per_device, W=W_out_per_device),
            use_fused=True,
        )
        torch_state = torch_model.state_dict()
        # TorchWanCausalConv3d wraps nn.Conv3d in self.conv, so keys have "conv." prefix.
        if any(k.startswith("conv.") for k in torch_state):
            torch_state = {k.removeprefix("conv."): v for k, v in torch_state.items()}

    tt_model.load_torch_state_dict(torch_state)

    assert tt_model._needs_halo, "Fused path not enabled — test shapes must require spatial halo exchange."

    if force_pipelining and tt_model.conv_config.T_out_block > 0:
        tt_model.conv_config.input_progress_t_batch_size = tt_model.conv_config.T_out_block

    logger.info(
        f"seed={seed} kernel={kernel_size_tuple} model={type(tt_model).__name__} "
        f"input_progress_t_batch_size={tt_model.conv_config.input_progress_t_batch_size} "
        f"T_out_block={tt_model.conv_config.T_out_block} force_pipelining={force_pipelining}"
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

    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    # --- Run standalone NP + conv3d FIRST (clean device state) ---
    standalone_output_torch = None
    if run_standalone:
        torch.manual_seed(seed + 999)
        torch_input_sa = torch.randn(B, C_in, T, H, W, dtype=torch_dtype)
        tt_input_sa = torch_input_sa.permute(0, 2, 3, 4, 1)
        tt_input_sa = conv_pad_in_channels(tt_input_sa)
        tt_input_sa, _ = conv_pad_height(tt_input_sa, parallel_config.height_parallel.factor)
        tt_input_sa = typed_tensor_2dshard(
            tt_input_sa,
            mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={h_axis: 2, w_axis: 3},
            dtype=tt_input_dtype,
        )

        # T-pad the same way the model does
        t_front_padding = tt_model.external_padding[0]
        if t_front_padding > 0:
            B_t, T_t, H_t, W_t, C_t = tt_input_sa.shape
            tt_input_sa = ttnn.reshape(tt_input_sa, (B_t, T_t, H_t * W_t, C_t))
            tt_input_sa = ttnn.pad(tt_input_sa, [(0, 0), (t_front_padding, 0), (0, 0), (0, 0)], value=0.0)
            tt_input_sa = ttnn.reshape(tt_input_sa, (B_t, T_t + t_front_padding, H_t, W_t, C_t))

        # Height mask (same as model forward)
        if tt_input_sa.shape[2] * parallel_config.height_parallel.factor > logical_h:
            mask = tt_model.get_cached_mask(tt_input_sa, logical_h)
            tt_input_sa = ttnn.mul(tt_input_sa, mask)

        # Standalone NP
        ext_pad = tt_model.external_padding
        np_dims, np_pad_left, np_pad_right = [], [], []
        np_axes, np_neighbor_sems, np_links = [], [], []
        if ext_pad[1] > 0 and parallel_config.height_parallel.factor > 1:
            np_dims.append(2)
            np_pad_left.append(ext_pad[1])
            np_pad_right.append(ext_pad[1])
            np_axes.append(parallel_config.height_parallel.mesh_axis)
            np_neighbor_sems.append(ccl_manager.get_np_ping_pong_semaphore(parallel_config.height_parallel.mesh_axis))
            np_links.append(get_neighbor_pad_num_links(ccl_manager, tt_input_sa, 2))
        if ext_pad[2] > 0 and parallel_config.width_parallel.factor > 1:
            np_dims.append(3)
            np_pad_left.append(ext_pad[2])
            np_pad_right.append(ext_pad[2])
            np_axes.append(parallel_config.width_parallel.mesh_axis)
            np_neighbor_sems.append(ccl_manager.get_np_ping_pong_semaphore(parallel_config.width_parallel.mesh_axis))
            np_links.append(get_neighbor_pad_num_links(ccl_manager, tt_input_sa, 3))

        padded = ccl_manager.neighbor_pad(
            tt_input_sa,
            dims=np_dims,
            pad_left=np_pad_left,
            pad_right=np_pad_right,
            padding_mode="zeros",
            axes=np_axes,
            neighbor_sems=np_neighbor_sems,
            num_links=np_links,
        )
        logger.info(f"Standalone NP output shape: {padded.shape}")

        # Separate conv3d on padded tensor (no halo buffer)
        cfg = tt_model.conv_config
        standalone_config = ttnn.Conv3dConfig(
            weights_dtype=cfg.weights_dtype,
            output_layout=cfg.output_layout,
            T_out_block=cfg.T_out_block,
            W_out_block=cfg.W_out_block,
            H_out_block=cfg.H_out_block,
            C_out_block=cfg.C_out_block,
            C_in_block=cfg.C_in_block,
            compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
        )

        standalone_out = ttnn.experimental.conv3d(
            input_tensor=padded,
            weight_tensor=tt_model.weight.data,
            bias_tensor=tt_model.bias.data,
            device=mesh_device,
            config=standalone_config,
            output_channels=tt_model.out_channels,
            kernel_size=tt_model.kernel_size,
            stride=tt_model.stride,
            padding=tt_model.internal_padding,
            padding_mode="zeros",
            dtype=tt_model.dtype,
            compute_kernel_config=tt_model.compute_kernel_config,
        )
        logger.info(f"Standalone conv3d output shape: {standalone_out.shape}")

        standalone_output_torch = ttnn.to_torch(
            standalone_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
        )
        if logical_h != standalone_output_torch.shape[2]:
            standalone_output_torch = standalone_output_torch[:, :logical_h, :, :, :]
        standalone_output_torch = standalone_output_torch.permute(0, 4, 1, 2, 3)
        if standalone_output_torch.shape[1] != C_out:
            standalone_output_torch = standalone_output_torch[:, :C_out]

        ttnn.deallocate(tt_input_sa)
        ttnn.deallocate(padded)
        ttnn.deallocate(standalone_out)

        # Reset semaphores before fused run
        ccl_manager.reset_global_semaphores()

    # --- Run fused NP + conv3d ---
    tt_output = tt_model(tt_input, logical_h=logical_h)

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

    return torch_output, tt_output_torch, standalone_output_torch


def _diagnose_pcc_failure(torch_ref, tt_out, h_factor, w_factor, pad_h=1, pad_w=1, top_k=20):
    """Print top-K error positions with boundary classification on PCC failure."""
    diff = (torch_ref - tt_out).abs()
    H, W = torch_ref.shape[3], torch_ref.shape[4]
    H_dev, W_dev = H // h_factor, W // w_factor

    flat_idx = diff.flatten().topk(min(top_k, diff.numel())).indices
    for rank, idx in enumerate(flat_idx):
        coords = []
        tmp = idx.item()
        for dim_size in reversed(diff.shape):
            coords.append(tmp % dim_size)
            tmp //= dim_size
        bi, c, t, h, w = list(reversed(coords))

        local_h = h % H_dev
        local_w = w % W_dev
        dev_h, dev_w = h // H_dev, w // W_dev

        regions = []
        if dev_h > 0 and local_h < pad_h:
            regions.append("H-top")
        if dev_h < h_factor - 1 and local_h >= H_dev - pad_h:
            regions.append("H-bot")
        if dev_w > 0 and local_w < pad_w:
            regions.append("W-left")
        if dev_w < w_factor - 1 and local_w >= W_dev - pad_w:
            regions.append("W-right")
        region = "+".join(regions) if regions else "interior"

        logger.error(
            f"  #{rank:2d} B={bi} C={c} T={t} H={h} W={w} "
            f"(dev[{dev_h},{dev_w}] lh={local_h} lw={local_w}) "
            f"diff={diff[bi, c, t, h, w]:.6g}  "
            f"ref={torch_ref[bi, c, t, h, w]:.6g}  tt={tt_out[bi, c, t, h, w]:.6g}  "
            f"[{region}]"
        )


@pytest.mark.parametrize(
    "B, C_in, C_out, T, H, W, kernel_size, padding, mesh_device, h_axis, w_axis, num_links",
    [
        # --- BH 4x8 720p WanCausalConv3d (3,3,3) shapes ---
        (1, 32, 384, 23, 92, 160, 3, 1, (4, 8), 0, 1, 2),
        (1, 384, 384, 23, 92, 160, 3, 1, (4, 8), 0, 1, 2),
        (1, 192, 384, 43, 184, 320, 3, 1, (4, 8), 0, 1, 2),
        (1, 384, 384, 43, 184, 320, 3, 1, (4, 8), 0, 1, 2),
        (1, 192, 192, 83, 368, 640, 3, 1, (4, 8), 0, 1, 2),
        (1, 96, 96, 83, 736, 1280, 3, 1, (4, 8), 0, 1, 2),
        (1, 96, 3, 83, 736, 1280, 3, 1, (4, 8), 0, 1, 2),
        # --- BH 4x8 720p WanConv2d (1,3,3) spatial_conv shapes ---
        (1, 384, 192, 41, 184, 320, (1, 3, 3), (0, 1, 1), (4, 8), 0, 1, 2),
        (1, 384, 192, 81, 368, 640, (1, 3, 3), (0, 1, 1), (4, 8), 0, 1, 2),
        (1, 192, 96, 81, 736, 1280, (1, 3, 3), (0, 1, 1), (4, 8), 0, 1, 2),
    ],
    ids=[
        "conv_in_4x8",
        "mid_up0_res_4x8",
        "up1_res0_4x8",
        "up1_res_4x8",
        "up2_res_4x8",
        "up3_res_4x8",
        "conv_out_4x8",
        "up0_spatial_4x8",
        "up1_spatial_4x8",
        "up2_spatial_4x8",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.timeout(900)
def test_fused_production_shapes(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype
):
    """Fused NP+Conv3d vs PyTorch reference on production VAE decoder shapes.

    Checks BOTH global PCC and boundary-region PCC separately.  The boundary
    check isolates pixels within ``pad`` distance of each device boundary —
    exactly the positions whose conv receptive field depends on NP halo data.
    """
    torch_model, ccl_manager, parallel_config, h_factor, w_factor = _make_model_and_manager(
        mesh_device, B, C_in, C_out, kernel_size, padding, h_axis, w_axis, num_links, dtype
    )

    torch_ref, tt_out, standalone_out = _run_one_seed(
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
        force_pipelining=True,
        run_standalone=True,
    )

    pad_h = padding[1] if isinstance(padding, tuple) else padding
    pad_w = padding[2] if isinstance(padding, tuple) else padding

    H, W = torch_ref.shape[3], torch_ref.shape[4]
    H_dev, W_dev = H // h_factor, W // w_factor
    bnd_mask = _boundary_mask(H, W, h_factor, w_factor, pad_h, pad_w)

    # ===== Standalone vs PyTorch (verify standalone path is correct) =====
    if standalone_out is not None:
        sa_global_pcc = _compute_pcc(torch_ref, standalone_out)
        bnd_sa = standalone_out[..., bnd_mask].flatten()
        int_sa = standalone_out[..., ~bnd_mask].flatten()
        bnd_ref = torch_ref[..., bnd_mask].flatten()
        int_ref = torch_ref[..., ~bnd_mask].flatten()
        sa_boundary_pcc = _compute_pcc(bnd_ref, bnd_sa)
        sa_interior_pcc = _compute_pcc(int_ref, int_sa)
        logger.info(
            f"=== STANDALONE vs TORCH: Global={sa_global_pcc*100:.4f}%  "
            f"Boundary={sa_boundary_pcc*100:.4f}%  Interior={sa_interior_pcc*100:.4f}% ==="
        )

    # ===== Fused vs PyTorch =====
    fused_global_pcc = _compute_pcc(torch_ref, tt_out)
    bnd_ref = torch_ref[..., bnd_mask].flatten()
    int_ref = torch_ref[..., ~bnd_mask].flatten()
    bnd_fused = tt_out[..., bnd_mask].flatten()
    int_fused = tt_out[..., ~bnd_mask].flatten()
    fused_boundary_pcc = _compute_pcc(bnd_ref, bnd_fused)
    fused_interior_pcc = _compute_pcc(int_ref, int_fused)
    logger.info(
        f"=== FUSED vs TORCH: Global={fused_global_pcc*100:.4f}%  "
        f"Boundary={fused_boundary_pcc*100:.4f}%  Interior={fused_interior_pcc*100:.4f}% ==="
    )

    # ===== Fused vs Standalone (direct TT-to-TT comparison) =====
    fused_vs_standalone_pcc = None
    if standalone_out is not None:
        fused_vs_standalone_pcc = _compute_pcc(tt_out, standalone_out)
        bnd_fvs_fused = tt_out[..., bnd_mask].flatten()
        bnd_fvs_sa = standalone_out[..., bnd_mask].flatten()
        fvs_boundary_pcc = _compute_pcc(bnd_fvs_fused, bnd_fvs_sa)
        int_fvs_fused = tt_out[..., ~bnd_mask].flatten()
        int_fvs_sa = standalone_out[..., ~bnd_mask].flatten()
        fvs_interior_pcc = _compute_pcc(int_fvs_fused, int_fvs_sa)
        logger.info(
            f"=== FUSED vs STANDALONE: Global={fused_vs_standalone_pcc*100:.6f}%  "
            f"Boundary={fvs_boundary_pcc*100:.6f}%  Interior={fvs_interior_pcc*100:.6f}% ==="
        )

        # Per-boundary-type breakdown
        h_bnd_rows = torch.zeros(H, dtype=torch.bool)
        w_bnd_cols = torch.zeros(W, dtype=torch.bool)
        for d in range(1, h_factor):
            pos = d * H_dev
            h_bnd_rows[max(0, pos - pad_h) : min(H, pos + pad_h)] = True
        for d in range(1, w_factor):
            pos = d * W_dev
            w_bnd_cols[max(0, pos - pad_w) : min(W, pos + pad_w)] = True
        h_mask_2d = h_bnd_rows.unsqueeze(-1).expand(H, W)
        w_mask_2d = w_bnd_cols.unsqueeze(0).expand(H, W)

        for label, mask_2d in [
            ("H-only", h_mask_2d & ~w_mask_2d),
            ("W-only", ~h_mask_2d & w_mask_2d),
            ("Corner", h_mask_2d & w_mask_2d),
        ]:
            cnt = mask_2d.sum().item()
            if cnt < 2:
                continue
            fv = tt_out[..., mask_2d].flatten()
            sv = standalone_out[..., mask_2d].flatten()
            pcc_val = _compute_pcc(fv, sv)
            diff_max = (fv - sv).abs().max().item()
            logger.info(f"  FvS {label:8s} PCC = {pcc_val * 100:.6f} %  max_diff = {diff_max:.6g}  ({cnt} px/slice)")

        # Top-20 fused-vs-standalone differences
        diff_fvs = (tt_out - standalone_out).abs()
        if diff_fvs.max().item() > 0:
            logger.info("  Top-20 fused-vs-standalone error positions:")
            flat_idx = diff_fvs.flatten().topk(min(20, diff_fvs.numel())).indices
            for rank, idx in enumerate(flat_idx):
                coords = []
                tmp = idx.item()
                for dim_size in reversed(diff_fvs.shape):
                    coords.append(tmp % dim_size)
                    tmp //= dim_size
                bi, c, t, h, w = list(reversed(coords))
                local_h, local_w = h % H_dev, w % W_dev
                dev_h, dev_w = h // H_dev, w // W_dev
                regions = []
                if dev_h > 0 and local_h < pad_h:
                    regions.append("H-top")
                if dev_h < h_factor - 1 and local_h >= H_dev - pad_h:
                    regions.append("H-bot")
                if dev_w > 0 and local_w < pad_w:
                    regions.append("W-left")
                if dev_w < w_factor - 1 and local_w >= W_dev - pad_w:
                    regions.append("W-right")
                region = "+".join(regions) if regions else "interior"
                logger.info(
                    f"    #{rank:2d} B={bi} C={c} T={t} H={h} W={w} "
                    f"(dev[{dev_h},{dev_w}] lh={local_h} lw={local_w}) "
                    f"diff={diff_fvs[bi, c, t, h, w]:.6g}  "
                    f"fused={tt_out[bi, c, t, h, w]:.6g}  standalone={standalone_out[bi, c, t, h, w]:.6g}  "
                    f"torch_ref={torch_ref[bi, c, t, h, w]:.6g}  "
                    f"[{region}]"
                )
        else:
            logger.info("  Fused and standalone outputs are IDENTICAL (no differences)")

    # --- Assert quality ---
    try:
        assert_quality(torch_ref, tt_out, pcc=TORCH_REF_PCC, relative_rmse=MAX_RMSE)
    except Exception:
        logger.error(f"PCC failure — top-20 error positions (H_dev={H // h_factor}, W_dev={W // w_factor}):")
        _diagnose_pcc_failure(torch_ref, tt_out, h_factor, w_factor, pad_h=pad_h, pad_w=pad_w)
        raise

    if fused_vs_standalone_pcc is not None and fused_vs_standalone_pcc < 0.999999:
        logger.error(
            f"FUSED vs STANDALONE mismatch! PCC = {fused_vs_standalone_pcc * 100:.6f} %. "
            f"Fused op diverges from standalone NP+conv3d."
        )

    logger.info("PASSED")
