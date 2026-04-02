# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_mochi import MochiDecoder3D, MochiResnetBlock3D, MochiUpBlock3D
from loguru import logger

import ttnn

from ....models.vae.vae_mochi import CausalUpsampleBlock as TtCausalUpsampleBlock
from ....models.vae.vae_mochi import Conv1x1 as TtConv1x1
from ....models.vae.vae_mochi import MochiVAEDecoder as TtDecoder
from ....models.vae.vae_mochi import ResBlock as TtResBlock
from ....parallel.config import MochiVAEParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import cache
from ....utils.check import assert_quality


def get_padded_size(numerator, denominator):
    return ((numerator + denominator - 1) // denominator) * denominator


def make_mochi_vae_parallel_config(mesh_device):
    """Build VAE parallel config based on mesh topology.

    2D mesh (Galaxy: both dims > 1): H on axis 0, W on axis 1, no time parallelism.
    1D mesh (T3K/N300/N150): time parallelism on the multi-device axis, no spatial parallelism.
    """
    if mesh_device.shape[0] > 1 and mesh_device.shape[1] > 1:
        return MochiVAEParallelConfig(
            time_parallel=ParallelFactor(factor=1, mesh_axis=1),
            h_parallel=ParallelFactor(factor=mesh_device.shape[0], mesh_axis=0),
            w_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
        )
    else:
        t_axis = 1 if mesh_device.shape[1] > 1 else 0
        return MochiVAEParallelConfig(
            time_parallel=ParallelFactor(factor=mesh_device.shape[t_axis], mesh_axis=t_axis),
            h_parallel=ParallelFactor(factor=1, mesh_axis=0),
            w_parallel=ParallelFactor(factor=1, mesh_axis=1),
        )


def get_shard_dims(parallel_config):
    """Get ShardTensor2dMesh/ConcatMesh2dToTensor dims based on parallel config.

    For axes with no parallelism (factor=1), uses dummy dims.
    These are safe because those axes have mesh_shape=1 (no actual sharding/concat).
    Dims must be unique (required by the framework).
    """
    dims = [0, 1]
    if parallel_config.h_parallel.factor > 1:
        dims[parallel_config.h_parallel.mesh_axis] = 2
    if parallel_config.w_parallel.factor > 1:
        dims[parallel_config.w_parallel.mesh_axis] = 3
    if parallel_config.time_parallel.factor > 1:
        dims[parallel_config.time_parallel.mesh_axis] = 1
    return dims


# Custom pytest mark for shared VAE device configuration
def vae_device_config(func):
    """Decorator to apply standard VAE device configuration to tests"""
    func = pytest.mark.parametrize(
        "mesh_device",
        [
            {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "T3K_2D": (2, 4), "TG": (8, 4)}.get(
                os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
            )
        ],
        indirect=True,
    )(func)
    func = pytest.mark.parametrize(
        "device_params",
        [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 20000000}],
        indirect=True,
    )(func)
    return func


class Conv3d1x1(nn.Conv3d):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=(1, 1, 1), bias=bias)


def create_random_conv3d_models(mesh_device, in_channels, out_channels, bias=True):
    """Initialize both reference Conv3d and TT models."""
    # Create reference model
    reference_model = Conv3d1x1(in_channels, out_channels, bias=bias)

    # Create TT model
    tt_model = TtConv1x1(
        mesh_device=mesh_device,
        in_channels=in_channels,
        out_channels=out_channels,
        bias=bias,
    )
    tt_model.load_torch_state_dict(reference_model.state_dict())

    return reference_model, tt_model


@pytest.mark.parametrize(
    "N, C_in, C_out, T, H, W",
    [
        (1, 12, 768, 28, 60, 106),
    ],
    ids=["large_latent"],
)
@vae_device_config
def test_tt_conv3d_1x1x1(mesh_device, N, C_in, C_out, T, H, W, reset_seeds):
    """Test forward pass of TtConv1x1 against Conv3d with 1x1x1 kernel."""
    reference_model, tt_model = create_random_conv3d_models(mesh_device, C_in, C_out)

    vae_parallel_config = make_mochi_vae_parallel_config(mesh_device)
    shard_dims = get_shard_dims(vae_parallel_config)

    # Create input tensor
    torch_input = torch.randn(N, C_in, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]

    if vae_parallel_config.time_parallel.factor > 1 and T % vae_parallel_config.time_parallel.factor:
        tt_input = torch.nn.functional.pad(
            tt_input, pad=(0, 0, 0, 0, 0, 0, 0, get_padded_size(T, vae_parallel_config.time_parallel.factor) - T)
        )
    if vae_parallel_config.w_parallel.factor > 1 and W % vae_parallel_config.w_parallel.factor:
        tt_input = torch.nn.functional.pad(
            tt_input, pad=(0, 0, 0, get_padded_size(W, vae_parallel_config.w_parallel.factor) - W)
        )
    if vae_parallel_config.h_parallel.factor > 1 and H % vae_parallel_config.h_parallel.factor:
        tt_input = torch.nn.functional.pad(
            tt_input, pad=(0, 0, 0, 0, 0, get_padded_size(H, vae_parallel_config.h_parallel.factor) - H)
        )

    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    logger.info("Run TtConv1x1 forward (Conv3d mode)")
    tt_output = tt_model(tt_input)
    logger.info("End TtConv1x1 forward (Conv3d mode)")

    # Convert TT output to torch tensor
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
    tt_output_torch = tt_output_torch[0:N, 0:C_out, 0:T, 0:H, 0:W]

    # Get reference output
    with torch.no_grad():
        ref_output = reference_model(torch_input)

    assert_quality(ref_output, tt_output_torch, pcc=0.999_500)


resblock_args = {
    "affine": True,
    "attn_block": None,
    "causal": True,
    "prune_bottleneck": False,
    "padding_mode": "replicate",
    "bias": True,
}


def create_random_resblock_models(mesh_device, parallel_config, ccl_manager, in_channels, nonlinearity):
    """Initialize both reference and TT models."""
    # Create reference model
    reference_model = MochiResnetBlock3D(in_channels=in_channels, act_fn=nonlinearity)

    # Create TT model
    tt_model = TtResBlock(
        reference_model.in_channels,
        reference_model.out_channels,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_model.load_torch_state_dict(reference_model.state_dict())

    return reference_model, tt_model


@torch.no_grad()
@pytest.mark.parametrize(
    ("N", "C", "T", "H", "W"),
    [
        # small latent
        pytest.param(1, 768, 28, 40, 50, id="s768"),
        pytest.param(1, 512, 84, 80, 100, id="s512"),
        pytest.param(1, 256, 168, 160, 200, id="s256"),
        pytest.param(1, 128, 168, 320, 400, id="s128"),
        # large latent
        pytest.param(1, 768, 28, 60, 106, id="l768"),
        pytest.param(1, 512, 84, 120, 212, id="l512"),
        pytest.param(1, 256, 168, 240, 424, id="l256"),
        pytest.param(1, 128, 168, 480, 848, id="l128"),
    ],
)
@pytest.mark.parametrize(
    "num_links",
    [
        pytest.param(4, id="4links"),
        pytest.param(1, id="1link"),
    ],
)
@vae_device_config
def test_tt_resblock_forward(mesh_device, N, C, T, H, W, reset_seeds, num_links):
    """Test complete forward pass of TtResBlock."""
    block_args = resblock_args.copy()
    block_args["channels"] = C
    block_args["nonlinearity"] = "silu"

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)

    vae_parallel_config = make_mochi_vae_parallel_config(mesh_device)
    shard_dims = get_shard_dims(vae_parallel_config)

    reference_model, tt_model = create_random_resblock_models(
        mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        in_channels=block_args["channels"],
        nonlinearity=block_args["nonlinearity"],
    )

    # Create input tensor
    torch_input = torch.randn(N, C, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]

    if vae_parallel_config.time_parallel.factor > 1 and T % vae_parallel_config.time_parallel.factor:
        tt_input = torch.nn.functional.pad(
            tt_input, pad=(0, 0, 0, 0, 0, 0, 0, get_padded_size(T, vae_parallel_config.time_parallel.factor) - T)
        )
    if vae_parallel_config.w_parallel.factor > 1 and W % vae_parallel_config.w_parallel.factor:
        tt_input = torch.nn.functional.pad(
            tt_input, pad=(0, 0, 0, get_padded_size(W, vae_parallel_config.w_parallel.factor) - W)
        )
    if vae_parallel_config.h_parallel.factor > 1 and H % vae_parallel_config.h_parallel.factor:
        tt_input = torch.nn.functional.pad(
            tt_input, pad=(0, 0, 0, 0, 0, get_padded_size(H, vae_parallel_config.h_parallel.factor) - H)
        )

    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    # Pass original (unpadded) spatial dims so ResBlock can slice padding
    # before GroupNorm and re-pad after, preventing zero contamination.
    logical_h = H if vae_parallel_config.h_parallel.factor > 1 else 0
    logical_w = W if vae_parallel_config.w_parallel.factor > 1 else 0

    logger.info(f"TT input shape: {tt_input.shape}")

    # ── Round-trip diagnostics (only meaningful when spatial-parallel) ──
    h_factor = vae_parallel_config.h_parallel.factor
    w_factor = vae_parallel_config.w_parallel.factor
    if h_factor > 1 or w_factor > 1:
        from ....parallel.config import vae_all_gather

        # 1) Check all-gather round-trip: shard → gather → compare against original
        x_4d = ttnn.reshape(tt_input, [N * T, tt_input.shape[2], tt_input.shape[3], C])
        x_4d = ttnn.to_layout(x_4d, ttnn.TILE_LAYOUT)
        if h_factor > 1:
            x_4d = vae_all_gather(
                ccl_manager,
                x_4d,
                cluster_axis=vae_parallel_config.h_parallel.mesh_axis,
                dim=1,
                reshape=False,
            )
        if w_factor > 1:
            x_4d = vae_all_gather(
                ccl_manager,
                x_4d,
                cluster_axis=vae_parallel_config.w_parallel.mesh_axis,
                dim=2,
                reshape=False,
            )
        x_4d_rm = ttnn.to_layout(x_4d, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x_4d)
        # Read back via ConcatMesh2dToTensor on the fractured (pre-gather) input for reference
        tt_input_ref = ttnn.to_torch(
            tt_input,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        )  # (N, T_padded, H_padded, W_padded, C)
        # Read gathered tensor from device 0
        gathered_torch = ttnn.to_torch(ttnn.get_device_tensors(x_4d_rm)[0]).float()
        ttnn.deallocate(x_4d_rm)
        # Compare valid region
        padded_H = tt_input.shape[2] * h_factor
        padded_W = tt_input.shape[3] * w_factor
        ref_valid = tt_input_ref[0, :, :H, :W, :].float()  # (T, H, W, C) — original valid data
        gathered_valid = gathered_torch[:, :H, :W, :]  # (N*T, H, W, C) — gathered valid data
        pcc_ag = torch.corrcoef(torch.stack([ref_valid.flatten(), gathered_valid.flatten()]))[0, 1].item()
        logger.info(f"[DIAG] all-gather round-trip PCC (valid {H}x{W}): {pcc_ag*100:.4f}%")
        logger.info(f"[DIAG] ref_valid: mean={ref_valid.mean():.6f} std={ref_valid.std():.6f}")
        logger.info(f"[DIAG] gathered_valid: mean={gathered_valid.mean():.6f} std={gathered_valid.std():.6f}")

        # 2) Check GroupNorm+silu: run torch reference on same data, compare against _norm_silu_spatial
        # Re-do the gather (input was not consumed)
        x_4d2 = ttnn.reshape(tt_input, [N * T, tt_input.shape[2], tt_input.shape[3], C])
        x_4d2 = ttnn.to_layout(x_4d2, ttnn.TILE_LAYOUT)
        if h_factor > 1:
            x_4d2 = vae_all_gather(
                ccl_manager,
                x_4d2,
                cluster_axis=vae_parallel_config.h_parallel.mesh_axis,
                dim=1,
                reshape=False,
            )
        if w_factor > 1:
            x_4d2 = vae_all_gather(
                ccl_manager,
                x_4d2,
                cluster_axis=vae_parallel_config.w_parallel.mesh_axis,
                dim=2,
                reshape=False,
            )
        tt_normed = tt_model._norm_silu_spatial(
            x_4d2, tt_model.norm1, N, T, padded_H, padded_W, C, logical_h, logical_w
        )
        tt_normed_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_normed)[0]).float()
        ttnn.deallocate(tt_normed)
        # Torch reference: GroupNorm + silu on the valid (unpadded) input
        ref_norm_input = ref_valid.reshape(N * T, H * W, C)  # (N*T, H*W, C)
        # Extract norm weights from the TT model
        norm1_weight = tt_model.norm1.weight.data
        norm1_bias = tt_model.norm1.bias.data
        norm1_w_torch = ttnn.to_torch(ttnn.get_device_tensors(norm1_weight)[0]).flatten().float()
        norm1_b_torch = ttnn.to_torch(ttnn.get_device_tensors(norm1_bias)[0]).flatten().float()
        torch_gn = torch.nn.GroupNorm(32, C, affine=True)
        torch_gn.weight.data = norm1_w_torch
        torch_gn.bias.data = norm1_b_torch
        # GroupNorm expects (N, C, *) — reshape to (N*T, C, H*W)
        ref_norm_in_nchw = ref_valid.reshape(N * T, H, W, C).permute(0, 3, 1, 2)  # (N*T, C, H, W)
        with torch.no_grad():
            ref_normed = torch_gn(ref_norm_in_nchw)
        ref_normed = torch.nn.functional.silu(ref_normed)
        ref_normed = ref_normed.permute(0, 2, 3, 1)  # (N*T, H, W, C)
        # Compare valid region of tt_normed (which is N, T, padded_H, padded_W, C)
        tt_normed_valid = tt_normed_torch[:, :, :H, :W, :].reshape(N * T, H, W, C)
        pcc_norm = torch.corrcoef(torch.stack([ref_normed.flatten(), tt_normed_valid.flatten()]))[0, 1].item()
        logger.info(f"[DIAG] norm1+silu PCC (valid {H}x{W}): {pcc_norm*100:.4f}%")
        logger.info(f"[DIAG] ref_normed: mean={ref_normed.mean():.6f} std={ref_normed.std():.6f}")
        logger.info(f"[DIAG] tt_normed_valid: mean={tt_normed_valid.mean():.6f} std={tt_normed_valid.std():.6f}")

        # 3) Check re-pad correctness: verify valid region preserved and padding == last valid col/row
        # tt_normed_torch is (N, T, padded_H, padded_W, C) from _norm_silu_spatial
        tt_repad = tt_normed_torch.reshape(N, T, padded_H, padded_W, C)
        # Check valid region is bit-identical to pre-pad (no corruption from concat)
        repad_valid = tt_repad[:, :, :H, :W, :]
        repad_valid_flat = repad_valid.flatten()
        normed_valid_flat = tt_normed_valid.reshape(N, T, H, W, C).flatten()
        valid_match = torch.allclose(repad_valid_flat, normed_valid_flat, atol=0, rtol=0)
        logger.info(f"[DIAG] re-pad valid region bit-identical: {valid_match}")
        if not valid_match:
            diff = (repad_valid_flat - normed_valid_flat).abs()
            logger.info(f"[DIAG]   max diff in valid region: {diff.max():.6f}, num nonzero: {(diff > 0).sum()}")
        # Check W padding columns (H:padded_H should be copies of last valid col)
        if padded_W > W:
            last_valid_col = tt_repad[:, :, :H, W - 1 : W, :]  # (N, T, H, 1, C)
            pad_cols = tt_repad[:, :, :H, W:padded_W, :]  # (N, T, H, pad_w, C)
            for pc in range(padded_W - W):
                col_match = torch.allclose(pad_cols[:, :, :, pc : pc + 1, :], last_valid_col, atol=0, rtol=0)
                logger.info(f"[DIAG] W pad col {W + pc} matches last valid col {W - 1}: {col_match}")
        # Check H padding rows
        if padded_H > H:
            last_valid_row = tt_repad[:, :, H - 1 : H, :padded_W, :]  # (N, T, 1, padded_W, C)
            pad_rows = tt_repad[:, :, H:padded_H, :padded_W, :]
            for pr in range(padded_H - H):
                row_match = torch.allclose(pad_rows[:, :, pr : pr + 1, :, :], last_valid_row, atol=0, rtol=0)
                logger.info(f"[DIAG] H pad row {H + pr} matches last valid row {H - 1}: {row_match}")

    # Dump intermediate tensors for cross-topology comparison
    dump_dir = os.environ.get("DUMP_TENSORS")
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        # Also save the torch input so we can verify both runs got the same data
        torch.save(torch_input, os.path.join(dump_dir, "torch_input.pt"))

    logger.info("Run TtResBlock forward")
    tt_output = tt_model(tt_input, logical_h, logical_w, dump_dir=dump_dir)
    logger.info("End TtResBlock forward")

    # Convert TT output to torch tensor
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
    tt_output_torch = tt_output_torch[0:N, 0:C, 0:T, 0:H, 0:W]

    # Get reference output
    logger.info("Run RefResBlock forward")
    with torch.no_grad():
        ref_output = reference_model(torch_input)[0]
    logger.info("End RefResBlock forward")

    logger.info("assert quality")
    for i in range(T):
        ref_output_slice = ref_output[:, :, i, :, :]
        tt_output_torch_slice = tt_output_torch[:, :, i, :, :]
        assert_quality(ref_output_slice, tt_output_torch_slice, pcc=0.9998)


def create_random_causalupsampleblock_models(
    mesh_device,
    in_channels,
    out_channels,
    num_layers,
    temporal_expansion,
    spatial_expansion,
    temporal_offset,
    parallel_config,
    ccl_manager,
):
    """Initialize both reference and TT models with optional real weights."""
    # Create reference model
    reference_model = MochiUpBlock3D(
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        temporal_expansion=temporal_expansion,
        spatial_expansion=spatial_expansion,
    )

    # Create TT model with same weights
    tt_model = TtCausalUpsampleBlock(
        mesh_device=mesh_device,
        in_channels=in_channels,
        out_channels=out_channels,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        num_res_blocks=num_layers,
        temporal_expansion=temporal_expansion,
        spatial_expansion=spatial_expansion,
        temporal_offset=temporal_offset,
    )
    tt_model.load_torch_state_dict(reference_model.state_dict())

    return reference_model, tt_model


# Test case configurations for different input sizes
@pytest.mark.parametrize(
    "config",
    [
        # large latent
        # First upsample block (768->512)
        {
            "name": "block1_768-512",
            "in_channels": 768,
            "out_channels": 512,
            "num_res_blocks": 6,
            "temporal_expansion": 3,
            "spatial_expansion": 2,
            "input_shape": [1, 768, 28, 60, 106],
            "expected_output_shape": (1, 512, 84, 120, 212),
        },
        # Second upsample block (512->256)
        {
            "name": "block2_512-256",
            "in_channels": 512,
            "out_channels": 256,
            "num_res_blocks": 4,
            "temporal_expansion": 2,
            "spatial_expansion": 2,
            "input_shape": [1, 512, 84, 120, 212],
            "expected_output_shape": (1, 256, 168, 240, 424),
        },
        # Third upsample block (256->128)
        {
            "name": "block3_256-128",
            "in_channels": 256,
            "out_channels": 128,
            "num_res_blocks": 3,
            "temporal_expansion": 1,
            "spatial_expansion": 2,
            "input_shape": [1, 256, 168, 240, 424],
            "expected_output_shape": (1, 128, 168, 480, 848),
        },
        # small latent
        # First upsample block (768->512)
        {
            "name": "block1_768-512",
            "in_channels": 768,
            "out_channels": 512,
            "num_res_blocks": 6,
            "temporal_expansion": 3,
            "spatial_expansion": 2,
            "input_shape": [1, 768, 28, 40, 50],
            "expected_output_shape": (1, 512, 84, 80, 100),
        },
        # Second upsample block (512->256)
        {
            "name": "block2_512-256",
            "in_channels": 512,
            "out_channels": 256,
            "num_res_blocks": 4,
            "temporal_expansion": 2,
            "spatial_expansion": 2,
            "input_shape": [1, 512, 84, 80, 100],
            "expected_output_shape": (1, 256, 168, 160, 200),
        },
        # Third upsample block (256->128)
        {
            "name": "block3_256-128",
            "in_channels": 256,
            "out_channels": 128,
            "num_res_blocks": 3,
            "temporal_expansion": 1,
            "spatial_expansion": 2,
            "input_shape": [1, 256, 168, 160, 200],
            "expected_output_shape": (1, 128, 168, 320, 400),
        },
    ],
    ids=["l768", "l512", "l256", "s768", "s512", "s256"],
)
@pytest.mark.parametrize(
    "num_links",
    [
        pytest.param(4, id="4links"),
        pytest.param(1, id="1link"),
    ],
)
@vae_device_config
def test_tt_upsample_forward(mesh_device, config, reset_seeds, num_links):
    """Test TtCausalUpsampleBlock against reference implementation."""
    in_channels = config["in_channels"]
    out_channels = config["out_channels"]
    num_res_blocks = config["num_res_blocks"]
    temporal_expansion = config["temporal_expansion"]
    spatial_expansion = config["spatial_expansion"]
    input_shape = config["input_shape"]
    expected_output_shape = config["expected_output_shape"]
    temporal_offset = 0  # temporal_expansion-1
    N, C, T, H, W = input_shape

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)

    vae_parallel_config = make_mochi_vae_parallel_config(mesh_device)
    shard_dims = get_shard_dims(vae_parallel_config)

    reference_model, tt_model = create_random_causalupsampleblock_models(
        mesh_device,
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_res_blocks,
        temporal_expansion=temporal_expansion,
        spatial_expansion=spatial_expansion,
        temporal_offset=temporal_offset,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
    )

    # Create input tensor
    torch_input = torch.randn(N, C, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]

    if vae_parallel_config.time_parallel.factor > 1 and T % vae_parallel_config.time_parallel.factor:
        tt_input = torch.nn.functional.pad(
            tt_input, pad=(0, 0, 0, 0, 0, 0, 0, get_padded_size(T, vae_parallel_config.time_parallel.factor) - T)
        )
    if vae_parallel_config.w_parallel.factor > 1 and W % vae_parallel_config.w_parallel.factor:
        tt_input = torch.nn.functional.pad(
            tt_input, pad=(0, 0, 0, get_padded_size(W, vae_parallel_config.w_parallel.factor) - W)
        )
    if vae_parallel_config.h_parallel.factor > 1 and H % vae_parallel_config.h_parallel.factor:
        tt_input = torch.nn.functional.pad(
            tt_input, pad=(0, 0, 0, 0, 0, get_padded_size(H, vae_parallel_config.h_parallel.factor) - H)
        )

    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    # Pass original (unpadded) spatial dims so ResBlocks inside the upsample
    # block can slice padding before GroupNorm.
    logical_h = H if vae_parallel_config.h_parallel.factor > 1 else 0
    logical_w = W if vae_parallel_config.w_parallel.factor > 1 else 0

    logger.info(f"Input shape: {torch_input.shape}")
    logger.info("Run TtCausalUpsampleBlock forward")
    tt_output = tt_model(tt_input, logical_h, logical_w)
    logger.info("End TtCausalUpsampleBlock forward")

    # Convert TT output to torch tensor
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
    if mesh_device.get_num_devices() > 1:
        expected_T = T * temporal_expansion - temporal_offset
        tt_output_torch = tt_output_torch[
            0:N,
            :,
            0:expected_T,
            0 : H * spatial_expansion,
            0 : W * spatial_expansion,
        ]

    # Get reference output
    logger.info("Run RefCausalUpsampleBlock forward")
    with torch.no_grad():
        ref_output = reference_model(torch_input)[0]
    logger.info("End RefCausalUpsampleBlock forward")

    # ── Detailed diagnostics for multi-device spatial parallelism ──
    h_factor = vae_parallel_config.h_parallel.factor
    w_factor = vae_parallel_config.w_parallel.factor
    if h_factor > 1 or w_factor > 1:
        sexp = spatial_expansion
        expected_T = T * temporal_expansion - temporal_offset

        # 1) Read per-device tensors and manually concatenate
        device_tensors = ttnn.get_device_tensors(tt_output)
        per_device_list = []
        for d_idx in range(len(device_tensors)):
            dt = ttnn.to_torch(device_tensors[d_idx]).float()
            per_device_list.append(dt)
            logger.info(f"[DIAG] device {d_idx}: shape={list(dt.shape)} mean={dt.mean():.6f} std={dt.std():.6f}")

        # Manual 2D concat matching ConcatMesh2dToTensor logic
        # For (2,4) mesh with dims=[2,3]: axis 0 concats on dim 2 (H), axis 1 on dim 3 (W)
        mesh_rows, mesh_cols = mesh_device.shape
        # First concat along W (axis 1) within each row
        row_tensors = []
        for r in range(mesh_rows):
            cols = [per_device_list[r * mesh_cols + c] for c in range(mesh_cols)]
            row_tensor = torch.cat(cols, dim=3)  # concat on W dim (dim 3 of NTHWC)
            row_tensors.append(row_tensor)
        # Then concat along H (axis 0) across rows
        manual_concat = torch.cat(row_tensors, dim=2)  # concat on H dim (dim 2 of NTHWC)
        logger.info(f"[DIAG] manual_concat shape: {list(manual_concat.shape)}")

        # Compare manual concat vs ConcatMesh2dToTensor
        concat2d = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        ).float()
        logger.info(f"[DIAG] ConcatMesh2dToTensor shape: {list(concat2d.shape)}")

        # Check bit-for-bit identity
        if torch.equal(manual_concat, concat2d):
            logger.info("[DIAG] manual_concat == ConcatMesh2dToTensor: IDENTICAL")
        else:
            diff = (manual_concat - concat2d).abs()
            logger.info(
                f"[DIAG] manual_concat vs ConcatMesh2dToTensor: DIFFER "
                f"max_diff={diff.max():.6f} num_diff={(diff > 0).sum().item()}/{diff.numel()}"
            )
            # Check if manual concat gives better PCC
            manual_out = manual_concat.permute(0, 4, 1, 2, 3)[:N, :, :expected_T, : H * sexp, : W * sexp]
            manual_pcc = torch.corrcoef(torch.stack([ref_output.flatten().double(), manual_out.flatten().double()]))[
                0, 1
            ].item()
            concat2d_out = concat2d.permute(0, 4, 1, 2, 3)[:N, :, :expected_T, : H * sexp, : W * sexp]
            c2d_pcc = torch.corrcoef(torch.stack([ref_output.flatten().double(), concat2d_out.flatten().double()]))[
                0, 1
            ].item()
            logger.info(f"[DIAG] PCC manual_concat: {manual_pcc*100:.4f}%  ConcatMesh2dToTensor: {c2d_pcc*100:.4f}%")

        # 2) Per-device comparison against reference spatial chunks
        # ref_output is (N, C_out, T_out, H_out, W_out) in NCTHW
        ref_nthwc = ref_output.permute(0, 2, 3, 4, 1).float()  # (N, T_out, H_out, W_out, C_out)
        per_dev_H = per_device_list[0].shape[2]  # H per device after d2s
        per_dev_W = per_device_list[0].shape[3]  # W per device after d2s
        logger.info(f"[DIAG] Per-device d2s shape: T={per_device_list[0].shape[1]} H={per_dev_H} W={per_dev_W}")

        for r in range(mesh_rows):
            for c in range(mesh_cols):
                d_idx = r * mesh_cols + c
                dev_data = per_device_list[d_idx]  # (N, T_d2s, H_local, W_local, C_out)
                # Extract corresponding chunk from reference
                h_start = r * per_dev_H
                h_end = min(h_start + per_dev_H, H * sexp)
                w_start = c * per_dev_W
                w_end = min(w_start + per_dev_W, W * sexp)
                # Valid region for this device
                valid_h = h_end - h_start
                valid_w = w_end - w_start
                ref_chunk = ref_nthwc[:, :expected_T, h_start:h_end, w_start:w_end, :]
                dev_valid = dev_data[:, :expected_T, :valid_h, :valid_w, :]
                if ref_chunk.numel() > 0 and dev_valid.numel() > 0:
                    pcc_dev = torch.corrcoef(torch.stack([ref_chunk.flatten().double(), dev_valid.flatten().double()]))[
                        0, 1
                    ].item()
                    mdiff = (ref_chunk - dev_valid).abs().max().item()
                    logger.info(
                        f"[DIAG] device ({r},{c}) vs ref chunk "
                        f"H[{h_start}:{h_end}] W[{w_start}:{w_end}]: "
                        f"PCC={pcc_dev*100:.4f}% max_diff={mdiff:.6f}"
                    )

        # 3) Check per-frame PCC of first few frames for gathered output
        gathered_nthwc = manual_concat[:N, :expected_T, : H * sexp, : W * sexp, :]
        for t in range(min(5, expected_T)):
            frame_ref = ref_nthwc[:, t, :, :, :]
            frame_tt = gathered_nthwc[:, t, :, :, :]
            pcc_frame = torch.corrcoef(torch.stack([frame_ref.flatten().double(), frame_tt.flatten().double()]))[
                0, 1
            ].item()
            logger.info(f"[DIAG] frame {t} PCC (manual gather): {pcc_frame*100:.4f}%")

    logger.info("assert quality")
    for i in range(T * temporal_expansion - temporal_offset):
        ref_output_slice = ref_output[:, :, i, :, :]
        tt_output_torch_slice = tt_output_torch[:, :, i, :, :]
        assert_quality(ref_output_slice, tt_output_torch_slice, pcc=0.9995)


def create_decoder_models(
    mesh_device,
    parallel_config,
    ccl_manager,
    latent_dim,
    out_channels,
    base_channels,
    channel_multipliers,
    temporal_expansions,
    spatial_expansions,
    num_res_blocks,
    nonlinearity,
    output_nonlinearity,
):
    """Initialize both reference and TT decoder models with optional real weights."""
    # Create reference model
    reference_model = MochiDecoder3D(
        in_channels=latent_dim,
        out_channels=out_channels,
        block_out_channels=[base_channels * multiplier for multiplier in channel_multipliers],
        layers_per_block=num_res_blocks,
        temporal_expansions=temporal_expansions,
        spatial_expansions=spatial_expansions,
        act_fn=nonlinearity,
    )

    # Create TT model with same weights
    tt_model = TtDecoder(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        temporal_expansions=temporal_expansions,
        spatial_expansions=spatial_expansions,
        num_res_blocks=num_res_blocks,
        latent_dim=latent_dim,
        nonlinearity=nonlinearity,
        output_nonlinearity=output_nonlinearity,
    )
    tt_model.load_torch_state_dict(reference_model.state_dict())

    return reference_model, tt_model


# Test case configurations for different input sizes
decoder_test_configs = [
    {
        "name": "small_latent",
        "input_shape": [1, 12, 28, 40, 50],
        "out_channels": 3,
        "base_channels": 128,
        "channel_multipliers": [1, 2, 4, 6],
        "temporal_expansions": [1, 2, 3],
        "spatial_expansions": [2, 2, 2],
        "num_res_blocks": [3, 3, 4, 6, 3],
        "latent_dim": 12,
        "has_attention": [False, False, False, False, False],
        "output_norm": False,
        "nonlinearity": "silu",
        "output_nonlinearity": "silu",
        "causal": True,
        # Expected output will be approximately: (1, 3, 168, 320, 400)
    },
    {
        "name": "large_latent",
        "input_shape": [1, 12, 28, 60, 106],
        "out_channels": 3,
        "base_channels": 128,
        "channel_multipliers": [1, 2, 4, 6],
        "temporal_expansions": [1, 2, 3],
        "spatial_expansions": [2, 2, 2],
        "num_res_blocks": [3, 3, 4, 6, 3],
        "latent_dim": 12,
        "has_attention": [False, False, False, False, False],
        "output_norm": False,
        "nonlinearity": "silu",
        "output_nonlinearity": "silu",
        "causal": True,
        # Expected output will be approximately: (1, 3, 168, 480, 848)
    },
]


def load_dit(
    mesh_device: ttnn.MeshDevice,
    ccl_manager: CCLManager,
    use_cache: bool,
    model_name: str = "genmo/mochi-1-preview",
):
    # Load pretrained Mochi Transformer
    # First load the torch version to get the config and state dict
    from diffusers import MochiTransformer3DModel as TorchMochiTransformer3DModel

    from ....models.transformers.transformer_mochi import MochiTransformer3DModel
    from ....parallel.config import DiTParallelConfig

    torch_transformer = TorchMochiTransformer3DModel.from_pretrained(
        model_name, subfolder="transformer", torch_dtype=torch.float32
    )

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=mesh_device.shape[1], mesh_axis=1),
        sequence_parallel=ParallelFactor(factor=mesh_device.shape[0], mesh_axis=0),
    )

    # Create TT version with the same config
    transformer = MochiTransformer3DModel(
        patch_size=torch_transformer.config.patch_size,
        num_attention_heads=torch_transformer.config.num_attention_heads,
        attention_head_dim=torch_transformer.config.attention_head_dim,
        num_layers=torch_transformer.config.num_layers,
        pooled_projection_dim=torch_transformer.config.pooled_projection_dim,
        in_channels=torch_transformer.config.in_channels,
        text_embed_dim=torch_transformer.config.text_embed_dim,
        time_embed_dim=torch_transformer.config.time_embed_dim,
        activation_fn=torch_transformer.config.activation_fn,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=True,
    )

    # Load state dict into TT transformer
    if use_cache:
        cache.load_model(
            transformer,
            model_name="mochi-1-preview",
            subfolder="transformer",
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            dtype="bf16",
        )
    else:
        transformer.load_torch_state_dict(torch_transformer.state_dict())

    return transformer


@pytest.mark.parametrize(
    "config",
    decoder_test_configs,
    ids=[cfg["name"] for cfg in decoder_test_configs],
)
@pytest.mark.parametrize(
    "load_dit_weights",
    [
        pytest.param(False, id="no_dit"),
        pytest.param(True, id="load_dit"),
    ],
)
@pytest.mark.parametrize(
    "num_links",
    [
        pytest.param(4, id="4links"),
        pytest.param(1, id="1link"),
    ],
)
@vae_device_config
def test_tt_decoder_forward(mesh_device, config, reset_seeds, load_dit_weights, num_links):
    input_shape = config["input_shape"]
    N, C, T, H, W = input_shape

    logger.info(
        f"Testing decoder with latent_dim={config['latent_dim']}, "
        f"base_channels={config['base_channels']}, "
        f"channel_multipliers={config['channel_multipliers']}, "
    )

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)

    if load_dit_weights:
        # Load DiT weights to device to account for real world DRAM usage, checking for OOM.
        logger.info("Loading DiT weights")
        tt_model_dit = load_dit(mesh_device, ccl_manager, use_cache=False)

    # Create models
    logger.info("Creating VAE decoder models")

    vae_parallel_config = make_mochi_vae_parallel_config(mesh_device)
    shard_dims = get_shard_dims(vae_parallel_config)

    reference_model, tt_model = create_decoder_models(
        mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        latent_dim=config["latent_dim"],
        out_channels=config["out_channels"],
        base_channels=config["base_channels"],
        channel_multipliers=config["channel_multipliers"],
        num_res_blocks=config["num_res_blocks"],
        temporal_expansions=config["temporal_expansions"],
        spatial_expansions=config["spatial_expansions"],
        nonlinearity=config["nonlinearity"],
        output_nonlinearity=config["output_nonlinearity"],
    )

    # Create input tensor (latent representation)
    torch_input = torch.randn(N, C, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]

    if vae_parallel_config.time_parallel.factor > 1 and T % vae_parallel_config.time_parallel.factor:
        tt_input = torch.nn.functional.pad(
            tt_input, pad=(0, 0, 0, 0, 0, 0, 0, get_padded_size(T, vae_parallel_config.time_parallel.factor) - T)
        )
    if vae_parallel_config.w_parallel.factor > 1 and W % vae_parallel_config.w_parallel.factor:
        tt_input = torch.nn.functional.pad(
            tt_input, pad=(0, 0, 0, get_padded_size(W, vae_parallel_config.w_parallel.factor) - W)
        )
    if vae_parallel_config.h_parallel.factor > 1 and H % vae_parallel_config.h_parallel.factor:
        tt_input = torch.nn.functional.pad(
            tt_input, pad=(0, 0, 0, 0, 0, get_padded_size(H, vae_parallel_config.h_parallel.factor) - H)
        )

    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    # Pass original (unpadded) spatial dims so ResBlocks can slice padding
    # before GroupNorm.
    logical_h = H if vae_parallel_config.h_parallel.factor > 1 else 0
    logical_w = W if vae_parallel_config.w_parallel.factor > 1 else 0

    logger.info(f"Input shape: {torch_input.shape}")
    logger.info("Run TtDecoder forward")
    tt_output = tt_model(tt_input, logical_h, logical_w)
    logger.info("End TtDecoder forward")

    # Convert TT output to torch tensor
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    logger.info(f"TT Output shape {tt_output_torch.shape}")

    # Get reference output
    logger.info("Run RefDecoder forward")
    with torch.no_grad():
        ref_output = reference_model(torch_input)[0]
    logger.info("End RefDecoder forward")

    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]

    logger.info("assert quality")
    for i in range(ref_output.shape[2]):
        ref_output_slice = ref_output[:, :, i, :, :]
        tt_output_torch_slice = tt_output_torch[:, :, i, :, :]
        assert_quality(ref_output_slice, tt_output_torch_slice, pcc=0.995)
