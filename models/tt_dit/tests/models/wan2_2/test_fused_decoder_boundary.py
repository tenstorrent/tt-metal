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
    use_ones_input=False,
    use_deterministic_weights=False,
):
    """Run one inference call with a specific seed; return (torch_output, tt_fused_output, tt_standalone_output).

    When ``use_ones_input=True`` the input is all ones (deterministic, value-invariant
    interior). When ``use_deterministic_weights=True`` the model weights are set to a
    single constant and bias to zero so the conv output at any position is
    ``constant * (# non-padded receptive-field elements)``. Combined, fused-vs-standalone
    divergence directly reveals which halo positions received the wrong (or no) data.
    """
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

    if use_deterministic_weights:
        # Constant weight + zero bias: output at position p = weight_val * C_in * (# valid RF taps at p).
        # With all-ones input, interior output = weight_val * C_in * kernel_volume everywhere;
        # boundary output differs only by the number of padding-zero taps in the receptive field.
        WEIGHT_VAL = 1.0 / 1000.0  # keep BF16 precision reasonable; avoids overflow on large C_in

        def _det_state(sd):
            out = {}
            for k, v in sd.items():
                if k.endswith("weight"):
                    out[k] = torch.full_like(v, WEIGHT_VAL)
                elif k.endswith("bias"):
                    out[k] = torch.zeros_like(v)
                else:
                    out[k] = v
            return out

        torch_state = _det_state(torch_state)
        torch_model.load_state_dict(_det_state(torch_model.state_dict()))

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
    if use_ones_input:
        torch_input = torch.ones(B, C_in, T, H, W, dtype=torch_dtype)
    else:
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
        if use_ones_input:
            torch_input_sa = torch.ones(B, C_in, T, H, W, dtype=torch_dtype)
        else:
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


# =============================================================================
# SYSTEMATIC DEBUG TEST — all-ones input, deterministic weights
# =============================================================================
# Purpose: isolate fused NP+Conv3d divergence from standalone NP+Conv3d with a
# fully deterministic input (all ones) and constant weights (zero bias). With
# this setup:
#   - Interior output is a constant (weight_val * kernel_volume) everywhere.
#   - Boundary output depends only on how many padding-zero kernel taps fall
#     into the receptive field.
#   - ANY fused-vs-standalone difference points directly at wrong halo data.
# Run:
#   pytest -xvs models/tt_dit/tests/models/wan2_2/test_neighbor_pad_conv3d_fused.py::test_fused_vs_standalone_ones_4x8


def _dump_ones_boundary_diff(tt_out, standalone_out, torch_ref, h_factor, w_factor, pad_h, pad_w, tag=""):
    """With all-ones input + constant weights, every pixel should equal a known value.
    Print a compact per-(device_row, device_col) table of max-diff between fused
    and standalone at each boundary row/column, so you can SEE which boundary is wrong.
    """
    H, W = tt_out.shape[3], tt_out.shape[4]
    H_dev, W_dev = H // h_factor, W // w_factor
    diff = (tt_out - standalone_out).abs()
    diff_t = (tt_out - torch_ref).abs()
    diff_s = (standalone_out - torch_ref).abs()
    logger.info(f"[{tag}] Output shape: {tuple(tt_out.shape)}  H_dev={H_dev}  W_dev={W_dev}")
    logger.info(f"[{tag}] GLOBAL max |fused-standalone| = {diff.max().item():.6g}")
    logger.info(f"[{tag}] GLOBAL max |fused-torch|      = {diff_t.max().item():.6g}")
    logger.info(f"[{tag}] GLOBAL max |standalone-torch| = {diff_s.max().item():.6g}")

    # Per-boundary-row max-diff (H direction)
    if h_factor > 1:
        logger.info(f"[{tag}] H-boundary per-row max|fused-standalone|:")
        for d in range(1, h_factor):
            for offset in range(-pad_h, pad_h):
                row = d * H_dev + offset
                if 0 <= row < H:
                    row_diff = diff[..., row, :].max().item()
                    logger.info(f"  H=[{row:4d}] (dev_boundary {d - 1}|{d} offset={offset:+d}) max_diff={row_diff:.6g}")

    # Per-boundary-col max-diff (W direction)
    if w_factor > 1:
        logger.info(f"[{tag}] W-boundary per-col max|fused-standalone|:")
        for d in range(1, w_factor):
            for offset in range(-pad_w, pad_w):
                col = d * W_dev + offset
                if 0 <= col < W:
                    col_diff = diff[..., :, col].max().item()
                    logger.info(f"  W=[{col:4d}] (dev_boundary {d - 1}|{d} offset={offset:+d}) max_diff={col_diff:.6g}")


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
def test_fused_vs_standalone_ones_4x8(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype
):
    """Systematic debug: all-ones input + constant weights + zero bias.

    With this setup interior should be a constant scalar everywhere, and boundary
    output differs from interior ONLY based on how many padding-zero kernel taps
    fall in the receptive field. Any fused-vs-standalone divergence is an NP/halo bug.

    Does NOT assert strict PCC (ones input produces low-variance outputs where
    PCC is numerically ill-conditioned). Instead asserts max-abs-diff between
    fused and standalone outputs.
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
        use_ones_input=True,
        use_deterministic_weights=True,
    )

    assert standalone_out is not None, "Standalone run was skipped — cannot compare"

    pad_h = padding[1] if isinstance(padding, tuple) else padding
    pad_w = padding[2] if isinstance(padding, tuple) else padding

    # Full per-boundary dump
    _dump_ones_boundary_diff(tt_out, standalone_out, torch_ref, h_factor, w_factor, pad_h, pad_w, tag="ones_debug")

    # Expected interior output magnitude (used to scale tolerances).
    kernel_tuple = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
    kernel_volume = kernel_tuple[0] * kernel_tuple[1] * kernel_tuple[2]
    WEIGHT_VAL = 1.0 / 1000.0  # must match _run_one_seed
    interior_expected = WEIGHT_VAL * C_in * kernel_volume
    # BF16 has ~7 mantissa bits → ~0.78% relative precision per accumulation.
    # Standalone (BF16 conv) vs torch (FP32) allow ~1.5% relative + small absolute floor.
    sa_tol = 0.015 * abs(interior_expected) + 1e-3
    # Fused vs standalone should be MUCH closer since both go through identical BF16 math.
    # Allow a bit for potential CB-reuse ordering differences.
    fvs_tol = 0.005 * abs(interior_expected) + 1e-4

    # Hard threshold: with deterministic input + weights, fused must match standalone
    # up to a small BF16 accumulation noise.
    fvs_max = (tt_out - standalone_out).abs().max().item()
    sa_vs_torch_max = (standalone_out - torch_ref).abs().max().item()
    logger.info(
        f"=== SUMMARY: interior_expected={interior_expected:.4g}  " f"sa_tol={sa_tol:.4g}  fvs_tol={fvs_tol:.4g}"
    )
    logger.info(f"=== SUMMARY: max|fused - standalone| = {fvs_max:.6g}  (tol {fvs_tol:.4g})")
    logger.info(f"=== SUMMARY: max|standalone - torch| = {sa_vs_torch_max:.6g}  (tol {sa_tol:.4g})")

    # Spatial distribution of standalone-vs-torch error: uniform (BF16 noise) or boundary-concentrated (bug)?
    sa_diff = (standalone_out - torch_ref).abs()
    sa_bnd_mask = _boundary_mask(H, W, h_factor, w_factor, pad_h, pad_w)
    sa_bnd_max = sa_diff[..., sa_bnd_mask].max().item() if sa_bnd_mask.any() else 0.0
    sa_int_max = sa_diff[..., ~sa_bnd_mask].max().item() if (~sa_bnd_mask).any() else 0.0
    logger.info(
        f"=== SA-vs-TORCH spatial: boundary_max={sa_bnd_max:.6g}  interior_max={sa_int_max:.6g}  "
        f"ratio={sa_bnd_max / max(sa_int_max, 1e-12):.2f}x"
    )

    # Standalone should be very close to torch reference (baseline).
    assert sa_vs_torch_max < sa_tol, (
        f"Standalone NP+conv3d itself diverges from torch reference "
        f"(max_diff={sa_vs_torch_max:.6g} > tol={sa_tol:.6g}). The fused path cannot be "
        f"trusted until the standalone path is correct. interior_expected={interior_expected:.4g}"
    )

    if fvs_max > fvs_tol:
        FUSED_VS_STANDALONE_TOL = fvs_tol  # for error message below
        # Locate worst-offending device-boundary column/row
        diff = (tt_out - standalone_out).abs()
        max_idx = diff.flatten().argmax().item()
        coords = []
        tmp = max_idx
        for dim_size in reversed(diff.shape):
            coords.append(tmp % dim_size)
            tmp //= dim_size
        bi, c, t, h, w = list(reversed(coords))
        H_dev_local = H // h_factor
        W_dev_local = W // w_factor
        logger.error(
            f"Worst fused-vs-standalone diff at B={bi} C={c} T={t} H={h} W={w} "
            f"(dev_row={h // H_dev_local} dev_col={w // W_dev_local} "
            f"local_h={h % H_dev_local} local_w={w % W_dev_local})  "
            f"fused={tt_out[bi, c, t, h, w]:.6g}  standalone={standalone_out[bi, c, t, h, w]:.6g}  "
            f"torch={torch_ref[bi, c, t, h, w]:.6g}  diff={diff[bi, c, t, h, w]:.6g}"
        )
        pytest.fail(
            f"Fused diverges from standalone by max={fvs_max:.6g} "
            f"(tol={FUSED_VS_STANDALONE_TOL}) with all-ones input — halo data is wrong."
        )

    logger.info("PASSED ones-debug")


# =============================================================================
# TWO-LAYER CHAIN DEBUG TEST — reproduce cross-layer state pollution
# =============================================================================
# Purpose: single-layer fused == standalone bit-exactly. If the bug is cross-layer
# pollution (halo-buffer aliasing, progress-sem reuse, fabric-CB leak), chaining
# two fused conv3d calls with the SAME shape should expose it.
#   conv_a(x) -> x1
#   conv_b(x1) -> y_fused
# vs standalone:
#   np_a + conv_a(x) -> x1_sa
#   np_b + conv_b(x1_sa) -> y_sa
# A non-zero |y_fused - y_sa| with deterministic input would implicate cross-layer state.


@pytest.mark.parametrize(
    "B, C_in, C_out, T, H, W, kernel_size, padding, mesh_device, h_axis, w_axis, num_links",
    [
        # Same C_in=C_out so we can chain two conv3d calls with one model pair.
        (1, 384, 384, 23, 92, 160, 3, 1, (4, 8), 0, 1, 2),
        (1, 384, 384, 43, 184, 320, 3, 1, (4, 8), 0, 1, 2),
        (1, 192, 192, 83, 368, 640, 3, 1, (4, 8), 0, 1, 2),
        (1, 96, 96, 83, 736, 1280, 3, 1, (4, 8), 0, 1, 2),
    ],
    ids=["mid_up0_res_chain", "up1_res_chain", "up2_res_chain", "up3_res_chain"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
# Each strategy describes what happens BETWEEN conv_a and conv_b in the chain. The
# strategy is also applied to the reference run (which always uses full sync), so
# we compare each candidate against the full-sync reference.
#
#   none        → nothing — back-to-back, the original failing path
#   sem_reset   → only ccl_manager.reset_global_semaphores() (no device sync)
#   device_sync → only ttnn.synchronize_device() (no sem reset)
#   sleep       → host-side time.sleep(2.0) (no sync, no reset) — pure timing test
#   full_sync   → device sync + sem reset (the known-good reference)
@pytest.mark.parametrize(
    "between_strategy",
    ["none", "sem_reset", "device_sync", "sleep", "full_sync"],
    ids=["none", "sem_reset", "device_sync", "sleep", "full_sync"],
)
@pytest.mark.timeout(900)
def test_fused_two_layer_chain_ones(
    mesh_device,
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
    num_links,
    dtype,
    between_strategy,
):
    """Chain two fused conv3d calls; vary what happens BETWEEN them.

    Each strategy is run for the test arm AND for a reference arm that ALWAYS
    uses full_sync. Comparison: |arm - full_sync_reference|. The strategy that
    matches full_sync identifies which mechanism (semaphore / device sync /
    timing) is the cause of cross-layer pollution.

    Decision table:
      strategy=full_sync    must match (sanity)
      strategy=none         large diff — original bug
      strategy=sem_reset    matches → progress / NP semaphore is the culprit
      strategy=device_sync  matches → timing / DMA pipeline ordering is the culprit
      strategy=sleep        matches → pure timing race (no shared state corruption)
    """
    import time as _time

    assert C_in == C_out, "chain test needs C_in == C_out to compose two convs"

    tt_input_dtype = ttnn.bfloat16 if dtype == ttnn.DataType.BFLOAT16 else ttnn.float32
    torch_dtype = torch.float32
    stride = 1

    h_factor = tuple(mesh_device.shape)[h_axis]
    w_factor = tuple(mesh_device.shape)[w_axis]
    H_out_per_device = H // h_factor
    W_out_per_device = W // w_factor

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=h_factor, mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=w_factor, mesh_axis=w_axis),
    )

    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d as TorchWanCausalConv3d

    WEIGHT_VAL = 1.0 / 1000.0

    def _make_det_conv(seed_offset):
        torch.manual_seed(42 + seed_offset)
        m = TorchWanCausalConv3d(
            in_channels=C_in, out_channels=C_out, kernel_size=kernel_size, stride=stride, padding=padding
        )
        m.eval()
        m.load_state_dict(
            {
                k: torch.full_like(v, WEIGHT_VAL) if k.endswith("weight") else torch.zeros_like(v)
                for k, v in m.state_dict().items()
            }
        )
        return m

    torch_conv_a = _make_det_conv(0)
    torch_conv_b = _make_det_conv(1)

    def _make_tt_conv(torch_m, use_fused):
        t = WanCausalConv3d(
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
            use_fused=use_fused,
        )
        sd = torch_m.state_dict()
        if any(k.startswith("conv.") for k in sd):
            sd = {k.removeprefix("conv."): v for k, v in sd.items()}
        t.load_torch_state_dict(sd)
        if t.conv_config.T_out_block > 0:
            t.conv_config.input_progress_t_batch_size = t.conv_config.T_out_block
        return t

    torch_input = torch.ones(B, C_in, T, H, W, dtype=torch_dtype)
    with torch.no_grad():
        torch_mid = torch_conv_a(torch_input)
        torch_out = torch_conv_b(torch_mid)

    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3

    def _prepare_tt_input():
        tt_x = torch_input.permute(0, 2, 3, 4, 1)
        tt_x = conv_pad_in_channels(tt_x)
        tt_x, logical_h = conv_pad_height(tt_x, parallel_config.height_parallel.factor)
        tt_x = typed_tensor_2dshard(
            tt_x,
            mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={h_axis: 2, w_axis: 3},
            dtype=tt_input_dtype,
        )
        return tt_x, logical_h

    def _apply_between(strategy):
        if strategy == "none":
            return
        if strategy == "sem_reset":
            ccl_manager.reset_global_semaphores()
            return
        if strategy == "device_sync":
            ttnn.synchronize_device(mesh_device)
            return
        if strategy == "sleep":
            _time.sleep(2.0)
            return
        if strategy == "full_sync":
            ttnn.synchronize_device(mesh_device)
            ccl_manager.reset_global_semaphores()
            return
        raise ValueError(f"unknown strategy {strategy}")

    def _run_chain(strategy):
        # Build fresh tt models each time so we don't carry over per-model state.
        conv_a = _make_tt_conv(torch_conv_a, use_fused=True)
        conv_b = _make_tt_conv(torch_conv_b, use_fused=True)
        ccl_manager.reset_global_semaphores()
        tt_in, logical_h = _prepare_tt_input()
        tt_mid = conv_a(tt_in, logical_h=logical_h)
        _apply_between(strategy)
        tt_out = conv_b(tt_mid, logical_h=logical_h)
        out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
        )
        if logical_h != out.shape[2]:
            out = out[:, :logical_h, :, :, :]
        out = out.permute(0, 4, 1, 2, 3)
        if out.shape[1] != C_out:
            out = out[:, :C_out]
        ttnn.deallocate(tt_in)
        ttnn.deallocate(tt_mid)
        ttnn.deallocate(tt_out)
        return out, logical_h

    # Reference: always full_sync.
    logger.info(f"--- Running reference chain (full_sync) ---")
    out_ref, _ = _run_chain("full_sync")

    # Test arm: the strategy under evaluation.
    logger.info(f"--- Running candidate chain (strategy={between_strategy}) ---")
    out_arm, _ = _run_chain(between_strategy)

    pad_h = padding if not isinstance(padding, tuple) else padding[1]
    pad_w = padding if not isinstance(padding, tuple) else padding[2]

    diff = (out_arm - out_ref).abs()
    arm_vs_ref = diff.max().item()
    ref_vs_torch = (out_ref - torch_out).abs().max().item()
    arm_vs_torch = (out_arm - torch_out).abs().max().item()

    logger.info(
        f"=== STRATEGY={between_strategy:<12s} | max|arm - full_sync_ref| = {arm_vs_ref:.6g}  "
        f"max|arm - torch| = {arm_vs_torch:.6g}  max|full_sync_ref - torch| = {ref_vs_torch:.6g}"
    )

    # Per-W / per-H boundary breakdown
    _dump_ones_boundary_diff(
        out_arm, out_ref, torch_out, h_factor, w_factor, pad_h, pad_w, tag=f"chain_{between_strategy}"
    )

    # Tolerance: with deterministic ones input + zero bias + constant weights and identical
    # device-side BF16 math, two correctly-synced runs should be bit-exact (or very close).
    # Any non-trivial diff implicates the strategy under test as missing a critical sync.
    CHAIN_TOL = 5e-3 * abs(
        WEIGHT_VAL
        * C_in
        * (kernel_size if isinstance(kernel_size, int) else kernel_size[0])
        * (kernel_size if isinstance(kernel_size, int) else kernel_size[1])
        * (kernel_size if isinstance(kernel_size, int) else kernel_size[2])
        * WEIGHT_VAL
        * C_in
        * (kernel_size if isinstance(kernel_size, int) else kernel_size[0])
        * (kernel_size if isinstance(kernel_size, int) else kernel_size[1])
        * (kernel_size if isinstance(kernel_size, int) else kernel_size[2])
    )

    if between_strategy == "full_sync":
        # Sanity: full_sync vs full_sync (different model objects but same strategy) should
        # be very close — small CB-allocation differences allowed.
        assert arm_vs_ref < max(CHAIN_TOL, 1e-3), (
            f"full_sync vs full_sync differs by {arm_vs_ref:.6g} — non-determinism in the "
            f"reference path itself (CB allocation jitter? buffer aliasing?). Tolerance was {CHAIN_TOL:.4g}."
        )
        logger.info(f"PASSED chain-debug strategy={between_strategy}")
        return

    # For non-reference strategies, log the verdict but do NOT fail — we want to see ALL
    # strategies' results in one pytest run for the decision table.
    if arm_vs_ref < max(CHAIN_TOL, 1e-3):
        logger.info(
            f"VERDICT [{between_strategy}]: ✓ matches full_sync — this strategy is sufficient "
            f"to prevent cross-layer pollution."
        )
    else:
        logger.error(
            f"VERDICT [{between_strategy}]: ✗ DIVERGES from full_sync by {arm_vs_ref:.6g} — "
            f"this strategy is NOT sufficient. Bug source remains."
        )

    logger.info(f"PASSED chain-debug strategy={between_strategy}  (diagnostic only)")


# =============================================================================
# FULL-DECODER FUSED vs NON-FUSED TEST — the decisive check
# =============================================================================
# Runs the entire WanDecoder3d twice with identical input/weights:
#   (a) with use_fused=True  everywhere (production path)
#   (b) with use_fused=False everywhere (standalone NP + conv3d per layer)
# Both paths call the same standalone conv3d op with a host-side NP;
# the difference IS the fused op interaction with all other layers.
#
# Any divergence pinpoints which stage of the decoder breaks under fusion.


def _get_wan_classes():
    """Import tt_dit + torch diffusers classes lazily."""
    from ....models.vae.vae_wan2_1 import WanCausalConv3d, WanDecoder3d
    from ....utils.conv3d import conv_pad_in_channels, conv_unpad_height, count_convs

    try:
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanDecoder3d as TorchWanDecoder3d
    except ImportError:
        TorchWanDecoder3d = None
    return WanCausalConv3d, WanDecoder3d, TorchWanDecoder3d, conv_pad_in_channels, conv_unpad_height, count_convs


@pytest.mark.parametrize(
    "B, C, T, H, W",
    [
        (1, 16, 1, 60, 104),  # 480p
        (1, 16, 1, 90, 160),  # 720p
    ],
    ids=["480p", "720p"],
)
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis, num_links",
    [
        ((4, 8), 0, 1, 2),
    ],
    ids=["bh_4x8_h0_w1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.timeout(1800)
def test_full_decoder_fused_vs_nonfused(mesh_device, B, C, T, H, W, h_axis, w_axis, num_links, dtype, monkeypatch):
    """Full WanDecoder3d: fused vs non-fused with deterministic input.

    This is the most direct reproducer of the vertical-line artifact visible in
    generated videos. Any fused-vs-non-fused divergence localizes the bug to a
    specific conv layer / pixel pattern in the decoder.
    """
    WanCausalConv3d, WanDecoder3d, _TorchDec, _conv_pad_in, _conv_unpad, count_convs = _get_wan_classes()
    from ....parallel.config import ParallelFactor, VaeHWParallelConfig
    from ....parallel.manager import CCLManager
    from ....utils.conv3d import conv_pad_in_channels, conv_unpad_height
    from ....utils.tensor import typed_tensor_2dshard

    torch.manual_seed(0)
    tt_input_dtype = ttnn.bfloat16 if dtype == ttnn.DataType.BFLOAT16 else ttnn.float32
    torch_dtype = torch.float32

    base_dim = 96
    z_dim = 16
    dim_mult = [1, 2, 4, 4]
    num_res_blocks = 2
    attn_scales = []
    temperal_upsample = [True, True, False]
    out_channels = 3
    is_residual = False

    h_factor = tuple(mesh_device.shape)[h_axis]
    w_factor = tuple(mesh_device.shape)[w_axis]

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=h_factor, mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=w_factor, mesh_axis=w_axis),
    )

    # Prepare deterministic input ONCE (so both runs see identical data).
    # Ones input makes interior output a constant, so any divergence highlights boundaries.
    torch_input = torch.ones(B, C, T, H, W, dtype=torch_dtype)
    tt_input_host = torch_input.permute(0, 2, 3, 4, 1)
    tt_input_host = conv_pad_in_channels(tt_input_host)
    tt_input_host, logical_h = conv_pad_height(tt_input_host, h_factor)

    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3

    def _run_decoder(force_use_fused: bool) -> torch.Tensor:
        """Build and run a WanDecoder3d with use_fused forced to the given value on
        every WanCausalConv3d instance."""
        orig_init = WanCausalConv3d.__init__

        def patched_init(self, *args, **kwargs):
            kwargs["use_fused"] = force_use_fused
            return orig_init(self, *args, **kwargs)

        monkeypatch.setattr(WanCausalConv3d, "__init__", patched_init)

        torch.manual_seed(0)
        tt_model = WanDecoder3d(
            dim=base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=temperal_upsample,
            out_channels=out_channels,
            is_residual=is_residual,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )
        # Random weights OK — we just need both runs to use SAME weights, so fix seed.
        torch.manual_seed(0)
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanDecoder3d as TorchWanDecoder3d

        torch_model = TorchWanDecoder3d(
            dim=base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=temperal_upsample,
            dropout=0.0,
            out_channels=out_channels,
            is_residual=is_residual,
        )
        torch_model.eval()
        tt_model.load_torch_state_dict(torch_model.state_dict())

        num_convs = count_convs(tt_model)
        tt_feat_cache = [None for _ in range(num_convs)]
        tt_feat_idx = [0]

        tt_input_tensor = typed_tensor_2dshard(
            tt_input_host,
            mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={h_axis: 2, w_axis: 3},
            dtype=tt_input_dtype,
        )

        logger.info(f"running tt decoder with use_fused={force_use_fused}")
        tt_output, new_logical_h = tt_model(
            tt_input_tensor,
            logical_h,
            feat_cache=tt_feat_cache,
            feat_idx=tt_feat_idx,
        )
        out_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
        )
        out_torch = conv_unpad_height(out_torch, new_logical_h)
        out_torch = out_torch.permute(0, 4, 1, 2, 3)

        # Free device state before next run
        ttnn.deallocate(tt_input_tensor)
        ttnn.deallocate(tt_output)
        for fc in tt_feat_cache:
            if fc is not None and not isinstance(fc, str):
                try:
                    ttnn.deallocate(fc)
                except Exception:
                    pass
        ccl_manager.reset_global_semaphores()
        ttnn.synchronize_device(mesh_device)

        monkeypatch.undo()
        return out_torch

    # ---- Run both paths ----
    out_fused = _run_decoder(force_use_fused=True)
    out_nonfused = _run_decoder(force_use_fused=False)

    assert (
        out_fused.shape == out_nonfused.shape
    ), f"Output shape mismatch: fused {out_fused.shape} vs non-fused {out_nonfused.shape}"

    # ---- Compare ----
    diff = (out_fused - out_nonfused).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    logger.info(
        f"=== DECODER FUSED vs NON-FUSED: max_diff={max_diff:.6g}  mean_diff={mean_diff:.6g}  "
        f"out_shape={tuple(out_fused.shape)}"
    )

    # Boundary vs interior breakdown on the output spatial dims.
    Ho, Wo = out_fused.shape[3], out_fused.shape[4]
    # Output device boundaries are at Ho // h_factor, Wo // w_factor granularity.
    bnd_mask = _boundary_mask(Ho, Wo, h_factor, w_factor, pad_h=2, pad_w=2)
    bnd_max = diff[..., bnd_mask].max().item() if bnd_mask.any() else 0.0
    int_max = diff[..., ~bnd_mask].max().item() if (~bnd_mask).any() else 0.0
    logger.info(
        f"=== Spatial: boundary_max={bnd_max:.6g}  interior_max={int_max:.6g}  "
        f"ratio={bnd_max / max(int_max, 1e-12):.2f}x  (>>1 ⇒ boundary localization)"
    )

    # Per-W-column + per-H-row max diff (identifies which specific seams are wrong).
    W_dev = Wo // w_factor
    H_dev = Ho // h_factor
    logger.info("Per-W-boundary column max_diff:")
    for d in range(1, w_factor):
        for offset in (-1, 0, 1):
            col = d * W_dev + offset
            if 0 <= col < Wo:
                logger.info(
                    f"  W=[{col:5d}] (seam {d-1}|{d} offset={offset:+d}) max={diff[..., :, col].max().item():.6g}"
                )
    logger.info("Per-H-boundary row max_diff:")
    for d in range(1, h_factor):
        for offset in (-1, 0, 1):
            row = d * H_dev + offset
            if 0 <= row < Ho:
                logger.info(
                    f"  H=[{row:5d}] (seam {d-1}|{d} offset={offset:+d}) max={diff[..., row, :].max().item():.6g}"
                )

    # Top-K worst positions
    top_k = 15
    flat = diff.flatten().topk(min(top_k, diff.numel())).indices
    logger.info(f"Top-{top_k} fused-vs-nonfused error positions:")
    for rank, idx in enumerate(flat):
        coords = []
        tmp = idx.item()
        for dim_size in reversed(diff.shape):
            coords.append(tmp % dim_size)
            tmp //= dim_size
        bi, c, t, h, w = list(reversed(coords))
        dev_h, dev_w = h // H_dev, w // W_dev
        local_h, local_w = h % H_dev, w % W_dev
        logger.info(
            f"  #{rank:2d} B={bi} C={c} T={t} H={h} W={w}  "
            f"(dev[{dev_h},{dev_w}] lh={local_h} lw={local_w})  "
            f"fused={out_fused[bi, c, t, h, w]:.6g}  nonfused={out_nonfused[bi, c, t, h, w]:.6g}  "
            f"diff={diff[bi, c, t, h, w]:.6g}"
        )

    # Tolerance: with deterministic ones input, fused should match non-fused to BF16 noise.
    # The decoder has ~20+ convs with varying C, so use a loose absolute tolerance.
    FVS_TOL = 0.05
    if max_diff > FVS_TOL:
        pytest.fail(
            f"FULL-DECODER fused-vs-nonfused divergence max_diff={max_diff:.6g} > tol={FVS_TOL}. "
            f"Bug is in fused-path interaction with other layers. "
            f"Boundary ratio = {bnd_max / max(int_max, 1e-12):.1f}x."
        )

    logger.info("PASSED full-decoder fused==nonfused")
