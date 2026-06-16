# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Production-shape tests for fused NeighborPad + Conv3d correctness.

Validates the fused NP+Conv3d path against PyTorch reference on all
VAE decoder shapes that trigger the halo path.  Checks PCC both globally
and at device-boundary pixels where NP halo data is consumed by the conv
kernel, catching boundary errors that global PCC can mask.
"""

import os

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
    scheme=None,
):
    """Run one inference call with a specific seed; return (torch_output, tt_fused_output, tt_standalone_output).

    scheme: None keeps the production-routed scheme (env vars may still force one); "halo_last" or
    "force_spatial" pins that overlap scheme directly so CI gates both without env vars.
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

    tt_model.load_torch_state_dict(torch_state)

    assert tt_model._needs_halo, "Fused path not enabled — test shapes must require spatial halo exchange."

    # Hybrid dispatch may have disabled fused for small-T shapes in production, but the
    # correctness test must exercise the fused kernel for ALL shapes to catch regressions.
    # Bypass the threshold here.
    if not is_spatial_only and not tt_model._use_fused:
        tt_model._use_fused = True

    if os.environ.get("NP_FORCE_SPATIAL"):
        tt_model.conv_config.force_spatial_parallel = True

    if os.environ.get("NP_HALO_LAST"):
        tt_model.conv_config.halo_last = True

    # Explicit scheme pin (takes priority over env). The reduced-T correctness shapes do not match
    # the production routing keys, so without this CI would only ever exercise the default scheme;
    # pinning lets a no-env run PCC-gate halo_last AND force_spatial directly.
    if scheme == "halo_last":
        tt_model.conv_config.halo_last = True
        tt_model.conv_config.force_spatial_parallel = False
    elif scheme == "force_spatial":
        tt_model.conv_config.force_spatial_parallel = True
        tt_model.conv_config.halo_last = False

    # NP_BLK="Cin,Cout,T,H,W": override blocking to validate the fine-block s4 sweep configs against
    # the PyTorch reference (the perf test only checks device time, not PCC). halo_last is position-
    # independent so PCC must hold at any blocking; this catches matmul-subblock / vol2col regressions.
    _blk = os.environ.get("NP_BLK")
    if _blk and isinstance(tt_model, WanCausalConv3d):
        ci, co, t, h, w = (int(x) for x in _blk.split(","))
        tt_model.conv_config.C_in_block = ci
        tt_model.conv_config.C_out_block = co
        tt_model.conv_config.T_out_block = t
        tt_model.conv_config.H_out_block = h
        tt_model.conv_config.W_out_block = w

    logger.info(
        f"seed={seed} kernel={kernel_size_tuple} model={type(tt_model).__name__} "
        f"T_out_block={tt_model.conv_config.T_out_block} force_pipelining={force_pipelining}"
    )

    torch.manual_seed(seed + 999)
    if use_ones_input:
        torch_input = torch.ones(B, C_in, T, H, W, dtype=torch_dtype)
        logger.info("Using all-ones input (replicates decoder bug-triggering condition)")
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
        if use_ones_input:
            torch_input_sa = torch.ones(B, C_in, T, H, W, dtype=torch_dtype)
        else:
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


# The 4x8 (32-device) entries cannot run on an 8-chip BH-LB. They are marked with an explicit skip
# so the coverage gap is visible in the CI report rather than silently dropped by the mesh_device
# fixture's generic "requested more devices than available" skip.
_SKIP_4X8 = pytest.mark.skip(reason="4x8 (32-device) shape — needs a 32-chip mesh, not available on 8-chip BH-LB")


@pytest.mark.parametrize(
    "B, C_in, C_out, T, H, W, kernel_size, padding, mesh_device, h_axis, w_axis, num_links",
    [
        # --- BH 4x8 720p WanCausalConv3d (3,3,3) shapes ---
        pytest.param(1, 32, 384, 23, 92, 160, 3, 1, (4, 8), 0, 1, 2, id="conv_in_4x8", marks=_SKIP_4X8),
        pytest.param(1, 384, 384, 23, 92, 160, 3, 1, (4, 8), 0, 1, 2, id="mid_up0_res_4x8", marks=_SKIP_4X8),
        pytest.param(1, 192, 384, 43, 184, 320, 3, 1, (4, 8), 0, 1, 2, id="up1_res0_4x8", marks=_SKIP_4X8),
        pytest.param(1, 384, 384, 43, 184, 320, 3, 1, (4, 8), 0, 1, 2, id="up1_res_4x8", marks=_SKIP_4X8),
        pytest.param(1, 192, 192, 83, 368, 640, 3, 1, (4, 8), 0, 1, 2, id="up2_res_4x8", marks=_SKIP_4X8),
        pytest.param(1, 96, 96, 83, 736, 1280, 3, 1, (4, 8), 0, 1, 2, id="up3_res_4x8", marks=_SKIP_4X8),
        pytest.param(1, 96, 3, 83, 736, 1280, 3, 1, (4, 8), 0, 1, 2, id="conv_out_4x8", marks=_SKIP_4X8),
        # --- BH 4x8 720p WanConv2d (1,3,3) spatial_conv shapes ---
        pytest.param(
            1, 384, 192, 41, 184, 320, (1, 3, 3), (0, 1, 1), (4, 8), 0, 1, 2, id="up0_spatial_4x8", marks=_SKIP_4X8
        ),
        pytest.param(
            1, 384, 192, 81, 368, 640, (1, 3, 3), (0, 1, 1), (4, 8), 0, 1, 2, id="up1_spatial_4x8", marks=_SKIP_4X8
        ),
        pytest.param(
            1, 192, 96, 81, 736, 1280, (1, 3, 3), (0, 1, 1), (4, 8), 0, 1, 2, id="up2_spatial_4x8", marks=_SKIP_4X8
        ),
        # --- BH 2x4 480p WanCausalConv3d (3,3,3) shapes (requires 8-device 2x4 mesh) ---
        # mid_block resnet: C_in=384, C_in_block=96 → 4 C_in blocks, H_out_block=32 (exact table)
        pytest.param(1, 384, 384, 7, 60, 104, 3, 1, (2, 4), 0, 1, 1, id="mid_res_2x4_480p"),
        # up1_res (stage1): C_in=384, T_res=16
        pytest.param(1, 384, 384, 14, 120, 208, 3, 1, (2, 4), 0, 1, 1, id="up1_res_2x4_480p"),
        # up1_res num_links=2: production target; nl2 is where the H/W T-frame partition bites.
        pytest.param(1, 384, 384, 14, 120, 208, 3, 1, (2, 4), 0, 1, 2, id="up1_res_2x4_nl2"),
        # ltx s4_out (128->48, asymmetric channels). Reduced T keeps the CPU torch ref fast;
        # force_spatial/halo_last are position-independent so PCC holds at any T/blocking.
        pytest.param(1, 128, 48, 21, 136, 240, 3, 1, (2, 4), 0, 1, 2, id="ltx_s4_out_2x4"),
        # ltx s4_res (128->128, per-dev 136x120 = full 272x480): the largest 2x4 halo_last shape (was
        # log-only in the perf test). Small T keeps the large-spatial CPU torch ref tractable;
        # halo_last is position-independent so PCC holds at any T.
        pytest.param(1, 128, 128, 7, 272, 480, 3, 1, (2, 4), 0, 1, 2, id="ltx_s4_res_2x4"),
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

    # Product-never-regresses gate: the fused op must match the two-dispatch standalone NP+conv3d to
    # >= 0.999 (mirrors the LTX fused-vs-standalone gate). bf16 accumulation-order rounding keeps this
    # below an exact 1.0, but any structural seam/staleness regression drops it well past 0.999.
    if fused_vs_standalone_pcc is not None:
        assert fused_vs_standalone_pcc >= 0.999, (
            f"FUSED vs STANDALONE mismatch! PCC = {fused_vs_standalone_pcc * 100:.6f} % < 99.9%. "
            f"Fused op diverges from standalone NP+conv3d."
        )

    logger.info("PASSED")


# ---------------------------------------------------------------------------
# Program-hash distinctness across schemes
# halo_last / force_spatial select structurally different programs for the same base blocking;
# compute_program_hash folds those flags in, so dispatching the same shape under different schemes
# must allocate distinct program-cache entries (and re-dispatching the same scheme must NOT).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "B, C_in, C_out, T, H, W, kernel_size, padding, mesh_device, h_axis, w_axis, num_links",
    [
        pytest.param(1, 128, 128, 14, 136, 240, 3, 1, (2, 4), 0, 1, 2, id="ltx_s4_res_2x4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.timeout(900)
def test_fused_scheme_program_hash_distinct(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype
):
    """The same shape under halo_last vs force_spatial must hash to distinct cached programs.

    Verifies the compute_program_hash fix: the scheme flags are folded into the hash so a cached
    program is never reused across structurally different schemes, while re-running the same scheme
    re-hits its entry.
    """
    h_factor = tuple(mesh_device.shape)[h_axis]
    w_factor = tuple(mesh_device.shape)[w_axis]
    H_out_per_device = H // h_factor
    W_out_per_device = W // w_factor

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=h_factor, mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=w_factor, mesh_axis=w_axis),
    )

    torch.manual_seed(42)
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d as TorchWanCausalConv3d

    torch_model = TorchWanCausalConv3d(in_channels=C_in, out_channels=C_out, kernel_size=3, stride=1, padding=1)
    torch_model.eval()

    tt_model = WanCausalConv3d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=3,
        mesh_device=mesh_device,
        stride=1,
        padding=1,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        dtype=dtype,
        conv_dims=ConvDims(T=T, H=H_out_per_device, W=W_out_per_device),
        use_fused=True,
    )
    torch_state = torch_model.state_dict()
    if any(k.startswith("conv.") for k in torch_state):
        torch_state = {k.removeprefix("conv."): v for k, v in torch_state.items()}
    tt_model.load_torch_state_dict(torch_state)
    if not tt_model._use_fused:
        tt_model._use_fused = True

    torch_input = torch.randn(B, C_in, T, H, W, dtype=torch.float32)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)
    tt_input = conv_pad_in_channels(tt_input)
    tt_input, logical_h = conv_pad_height(tt_input, parallel_config.height_parallel.factor)

    def _shard():
        return typed_tensor_2dshard(
            tt_input,
            mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={h_axis: 2, w_axis: 3},
            dtype=ttnn.bfloat16,
        )

    def _set_scheme(name):
        tt_model.conv_config.halo_last = name == "halo_last"
        tt_model.conv_config.force_spatial_parallel = name == "force_spatial"

    def _dispatch():
        out = tt_model(_shard(), logical_h=logical_h)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(out)

    base = mesh_device.num_program_cache_entries()
    _set_scheme("halo_last")
    _dispatch()
    after_halo = mesh_device.num_program_cache_entries()
    _set_scheme("force_spatial")
    _dispatch()
    after_fs = mesh_device.num_program_cache_entries()
    _set_scheme("halo_last")
    _dispatch()
    after_halo2 = mesh_device.num_program_cache_entries()

    logger.info(
        f"program-cache entries: base={base} after_halo_last={after_halo} "
        f"after_force_spatial={after_fs} after_halo_last_again={after_halo2}"
    )
    assert after_halo > base, "halo_last dispatch did not allocate a program-cache entry"
    assert after_fs > after_halo, (
        "force_spatial did NOT allocate a new program-cache entry — the scheme flags are not in the "
        "program hash (halo_last and force_spatial would silently share one cached program)."
    )
    assert after_halo2 == after_fs, (
        "re-dispatching halo_last allocated a NEW entry instead of re-hitting its cached program — "
        "the hash is unstable across identical scheme dispatches."
    )
    logger.info("PASSED: scheme flags produce distinct, stable program-cache entries")


# ---------------------------------------------------------------------------
# Per-scheme PCC gate (no env vars)
# The reduced-T correctness shapes do not match the production routing keys in
# get_conv3d_config, so without an explicit pin a no-env CI run only ever exercises the
# default scheme — leaving halo_last and force_spatial (the real production schemes)
# unverified. This test pins each scheme directly so both are PCC-gated by default.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "B, C_in, C_out, T, H, W, kernel_size, padding, mesh_device, h_axis, w_axis, num_links",
    [
        # ltx s4_res (128->128): production-routed to halo_last. ltx s4_out (128->48): routed to
        # force_spatial. Reduced T/spatial keeps the CPU torch ref fast; both schemes are
        # position-independent so PCC holds at any T/blocking.
        pytest.param(1, 128, 128, 14, 136, 240, 3, 1, (2, 4), 0, 1, 2, id="ltx_s4_res_2x4"),
        pytest.param(1, 128, 48, 14, 136, 240, 3, 1, (2, 4), 0, 1, 2, id="ltx_s4_out_2x4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("scheme", ["halo_last", "force_spatial"])
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.timeout(900)
def test_fused_scheme_pcc(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype, scheme
):
    """Fused NP+Conv3d with halo_last AND force_spatial explicitly pinned, vs PyTorch + standalone.

    Gates BOTH overlap schemes with no env vars, so CI verifies the real production schemes (the
    program-hash fix means halo_last and force_spatial now get distinct cached programs).
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
        scheme=scheme,
    )

    fused_global_pcc = _compute_pcc(torch_ref, tt_out)
    logger.info(f"=== scheme={scheme} FUSED vs TORCH Global PCC = {fused_global_pcc*100:.4f}% ===")
    try:
        assert_quality(torch_ref, tt_out, pcc=TORCH_REF_PCC, relative_rmse=MAX_RMSE)
    except Exception:
        logger.error(f"PCC failure (scheme={scheme}) — top-20 error positions:")
        pad_h = padding[1] if isinstance(padding, tuple) else padding
        pad_w = padding[2] if isinstance(padding, tuple) else padding
        _diagnose_pcc_failure(torch_ref, tt_out, h_factor, w_factor, pad_h=pad_h, pad_w=pad_w)
        raise

    if standalone_out is not None:
        fvs = _compute_pcc(tt_out, standalone_out)
        logger.info(f"=== scheme={scheme} FUSED vs STANDALONE PCC = {fvs*100:.6f}% ===")
        assert fvs >= 0.999, (
            f"scheme={scheme}: fused vs standalone PCC = {fvs*100:.6f}% < 99.9%; "
            f"fused op diverges from standalone NP+conv3d."
        )
    logger.info(f"PASSED scheme={scheme}")


# ---------------------------------------------------------------------------
# Repeat-invocation PCC gate (program-cache hit + ping-pong halo-buffer RTA refresh)
# Dispatching the SAME fused layer >= 2 times exercises the program-cache-hit path where
# override_runtime_arguments must re-supply the per-call semaphore addresses and swap the
# ping-pong halo buffer. An unrefreshed RTA regression would leave the second call reading a
# stale semaphore/buffer — caught by asserting PCC on every call.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "B, C_in, C_out, T, H, W, kernel_size, padding, mesh_device, h_axis, w_axis, num_links",
    [
        pytest.param(1, 128, 128, 14, 136, 240, 3, 1, (2, 4), 0, 1, 2, id="ltx_s4_res_2x4"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("scheme", ["halo_last"])
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.timeout(900)
def test_fused_repeat_invocation(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype, scheme
):
    """Dispatch the SAME fused layer twice; PCC must hold on BOTH calls.

    The second call hits the program cache, so it goes through override_runtime_arguments — which
    must refresh the per-call progress/region semaphore addresses and swap the ping-pong halo
    buffer. A stale RTA would make the second call read a half-reset semaphore or the wrong buffer
    half; asserting PCC per call catches that without a perf run.
    """
    n_calls = 2
    tt_input_dtype = ttnn.bfloat16
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

    torch.manual_seed(42)
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d as TorchWanCausalConv3d

    torch_model = TorchWanCausalConv3d(in_channels=C_in, out_channels=C_out, kernel_size=3, stride=stride, padding=1)
    torch_model.eval()

    tt_model = WanCausalConv3d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=3,
        mesh_device=mesh_device,
        stride=stride,
        padding=1,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        dtype=dtype,
        conv_dims=ConvDims(T=T, H=H_out_per_device, W=W_out_per_device),
        use_fused=True,
    )
    torch_state = torch_model.state_dict()
    if any(k.startswith("conv.") for k in torch_state):
        torch_state = {k.removeprefix("conv."): v for k, v in torch_state.items()}
    tt_model.load_torch_state_dict(torch_state)

    assert tt_model._needs_halo, "Fused path not enabled for repeat test shape."
    if not tt_model._use_fused:
        tt_model._use_fused = True
    if scheme == "halo_last":
        tt_model.conv_config.halo_last = True
        tt_model.conv_config.force_spatial_parallel = False

    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3

    for call_idx in range(n_calls):
        torch.manual_seed(call_idx + 100)  # distinct input per call so a stale buffer mismatches
        torch_input = torch.randn(B, C_in, T, H, W, dtype=torch_dtype)
        with torch.no_grad():
            torch_output = torch_model(torch_input)

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

        pcc = _compute_pcc(torch_output, tt_output_torch)
        cache_state = "miss (first)" if call_idx == 0 else "hit"
        logger.info(f"=== call {call_idx} (program-cache {cache_state}) PCC = {pcc*100:.4f}% ===")
        assert_quality(torch_output, tt_output_torch, pcc=TORCH_REF_PCC, relative_rmse=MAX_RMSE)

        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_output)

    logger.info(f"PASSED repeat-invocation ({n_calls} calls)")


# ---------------------------------------------------------------------------
# All-ones input seam test
# Reproduces the exact bug condition: all-ones input exposes fused vs standalone
# boundary divergence that random input masks (PCC averages it out).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "B, C_in, C_out, T, H, W, kernel_size, padding, mesh_device, h_axis, w_axis, num_links",
    [
        # --- 2x4 num_links=2: the seam detector for BH-LB.
        # nl2 is where the per-link T-frame partition can leave a corner stale (Attempt 1's bug). ---
        (1, 384, 384, 7, 60, 104, 3, 1, (2, 4), 0, 1, 2),
    ],
    ids=["mid_res_2x4_nl2_ones"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.timeout(300)
def test_fused_ones_input_seam(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype
):
    """Fused NP+Conv3d with all-ones input vs standalone: boundary PCC must be 100%.

    With all-ones input any seam artifact manifests as a detectable value
    discontinuity at W-boundaries (rather than being buried in random noise).
    This test reproduces the full-decoder bug condition in a single-layer context.
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
    )

    pad_h = padding[1] if isinstance(padding, tuple) else padding
    pad_w = padding[2] if isinstance(padding, tuple) else padding
    bnd_mask = _boundary_mask(H, W, h_factor, w_factor, pad_h, pad_w)

    if standalone_out is not None:
        fvs_global_pcc = _compute_pcc(standalone_out, tt_out)
        bnd_fused = tt_out[..., bnd_mask].flatten() if bnd_mask.any() else None
        bnd_sa = standalone_out[..., bnd_mask].flatten() if bnd_mask.any() else None
        fvs_bnd_pcc = _compute_pcc(bnd_sa, bnd_fused) if bnd_fused is not None and bnd_fused.numel() > 1 else 1.0
        max_diff_bnd = (tt_out - standalone_out).abs()[..., bnd_mask].max().item() if bnd_mask.any() else 0.0
        logger.info(
            f"=== ONES INPUT — FUSED vs STANDALONE: Global={fvs_global_pcc*100:.4f}%  "
            f"Boundary={fvs_bnd_pcc*100:.4f}%  max_diff_bnd={max_diff_bnd:.6g} ==="
        )
        # Halo staleness reads uninitialized DRAM → an order-of-magnitude diff (~the value itself).
        # A correct fused op still differs from the two-dispatch standalone by bf16 accumulation-order
        # rounding (~1 ULP), so the bar is a few bf16 ULP relative to the output magnitude, not exact 0.
        out_mag = max(1.0, standalone_out.abs().max().item())
        seam_tol = 4.0 * out_mag / 128.0  # 4 bf16 ULP (mantissa 2^-7); staleness is far larger
        if max_diff_bnd > seam_tol:
            logger.error(
                f"SEAM DETECTED with all-ones input! max_diff at boundary = {max_diff_bnd:.6g} "
                f"> tol {seam_tol:.6g}. Fused op produces boundary errors invisible to random-input PCC."
            )
        assert max_diff_bnd <= seam_tol, (
            f"Fused vs standalone boundary max_diff={max_diff_bnd:.6g} > {seam_tol:.6g} (4 bf16 ULP) "
            f"with all-ones input. Seam artifact confirmed in fused path."
        )
    logger.info("PASSED")
