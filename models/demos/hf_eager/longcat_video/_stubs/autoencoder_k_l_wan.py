# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `autoencoder_k_l_wan` (meituan-longcat/LongCat-Video's
VAE, `vae/config.json` -> `_class_name: AutoencoderKLWan`, the diffusers Wan2.1 video VAE).

This component graduates DIRECTLY tensor-parallel (no single-device phase): it adapts the
production `WanEncoder`/`WanDecoder` implementation in `models/tt_dit/models/vae/vae_wan2_1.py`
-- already validated against this exact architecture (identical class names and state-dict
layout: `WanCausalConv3d`, `WanResidualBlock`, `WanMidBlock`, `WanResample`, `WanAttentionBlock`,
`WanEncoder3D`/`WanDecoder3d`) -- to this bring-up harness's `build(device, torch_module) ->
callable` contract. The TP scheme is height-sharded activations (not weight-sharded): the video
tensor's H dimension is split across the mesh, convs exchange halos across the split via CCL, and
attention gathers/scatters around a replicated softmax -- matching the `((1, 4), 1, 0)`
height-parallel convention validated in `models/tt_dit/tests/models/wan2_2/test_vae_wan2_1.py`
for a 1x4 mesh.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanDecoder, WanEncoder
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_pad_width
from models.tt_dit.utils.tensor import typed_tensor_2dshard


def _hw_parallel_config(mesh_device: ttnn.MeshDevice) -> VaeHWParallelConfig:
    """A 1xN (or Nx1) TP mesh has exactly one axis with factor>1; shard height on that axis and
    leave width replicated (factor=1), matching the `((1, 4), 1, 0)` / `((2, 1), 0, 1)`-style
    convention validated in test_vae_wan2_1.py."""
    shape = tuple(mesh_device.shape)
    h_axis, w_axis = (0, 1) if shape[0] >= shape[1] else (1, 0)
    return VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=shape[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=shape[w_axis], mesh_axis=w_axis),
    )


def _replicated_ttnn_to_torch(t: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """Read back a tensor that is REPLICATED (not sharded) across every device in the mesh. Every
    per-device shard holds an identical replica, so reading back just ONE shard
    (`ttnn.get_device_tensors`) is correct and avoids the host-side concat+slice a
    `ConcatMeshToTensor` readback would need."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


class TtAutoencoderKLWan:
    """Height-sharded (TP) native ttnn port of diffusers' `AutoencoderKLWan` (Wan2.1 VAE)."""

    def __init__(self, mesh_device: ttnn.MeshDevice, torch_module) -> None:
        cfg = torch_module.config
        self.mesh_device = mesh_device
        self.parallel_config = _hw_parallel_config(mesh_device)
        self.ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        self.dtype = ttnn.bfloat16

        dim_mult = list(cfg.dim_mult)
        attn_scales = list(cfg.attn_scales)
        temperal_downsample = list(cfg.temperal_downsample)
        is_residual = getattr(cfg, "is_residual", False)
        decoder_base_dim = getattr(cfg, "decoder_base_dim", None) or cfg.base_dim

        # `height`/`width` (default 0) only tune per-stage conv3d blocking configs; correctness
        # doesn't depend on them (validated by test_wan_encoder/test_wan_decoder, which omit
        # them entirely), so weights can be loaded here without knowing the runtime input shape.
        self.encoder = WanEncoder(
            base_dim=cfg.base_dim,
            in_channels=cfg.in_channels,
            z_dim=cfg.z_dim,
            dim_mult=dim_mult,
            num_res_blocks=cfg.num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            is_residual=is_residual,
            mesh_device=mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            dtype=self.dtype,
        )
        self.decoder = WanDecoder(
            base_dim=cfg.base_dim,
            decoder_base_dim=decoder_base_dim,
            z_dim=cfg.z_dim,
            dim_mult=dim_mult,
            num_res_blocks=cfg.num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            out_channels=cfg.out_channels,
            is_residual=is_residual,
            mesh_device=mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            dtype=self.dtype,
        )

        state = torch_module.state_dict()
        self.encoder.load_torch_state_dict(state)
        self.decoder.load_torch_state_dict(state)

    def __call__(
        self,
        sample: ttnn.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator=None,
    ):
        assert not sample_posterior, (
            "sample_posterior=True draws a random latent (posterior.sample()); the golden "
            "reference uses the deterministic default (posterior.mode()), so this stays off."
        )

        h_axis = self.parallel_config.height_parallel.mesh_axis
        w_axis = self.parallel_config.width_parallel.mesh_axis
        h_factor = self.parallel_config.height_parallel.factor
        w_factor = self.parallel_config.width_parallel.factor

        # `sample` arrives replicated across the mesh (the harness uploads the primary input via
        # ReplicateTensorToMesh); this stub does its own shard placement, so read it back once.
        sample_torch = _replicated_ttnn_to_torch(sample, self.mesh_device).to(torch.float32)

        x_BTHWC = sample_torch.permute(0, 2, 3, 4, 1)  # BCTHW -> BTHWC
        x_BTHWC = conv_pad_in_channels(x_BTHWC)
        # *8: 3 downsample-by-2 stages, so each device's local H/W must stay even through every
        # stage (matches the `factor * 8` padding in test_vae_wan2_1.py::test_wan_encoder).
        x_BTHWC, logical_h = conv_pad_height(x_BTHWC, h_factor * 8)
        x_BTHWC, logical_w = conv_pad_width(x_BTHWC, w_factor * 8)
        x_BTHWC = typed_tensor_2dshard(
            x_BTHWC,
            self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={h_axis: 2, w_axis: 3},
            dtype=self.dtype,
        )

        # encoder_t_chunk_size left at its default (4): that's the mode validated by
        # test_vae_wan2_1.py::test_wan_encoder; the full-single-pass (None) path is less exercised.
        z_BCTHW, lat_logical_h, lat_logical_w = self.encoder(x_BTHWC, logical_h, logical_w=logical_w)

        # `WanEncoder.forward` trims its quant_conv output down to the true `z_dim` channels (the
        # golden's deterministic `posterior.mode()` mean-only latent) -- but `WanDecoder`'s
        # `post_quant_conv` Linear was built with an `aligned_channels(z_dim)` (tile-padded) input
        # width, so this un-padded boundary tensor can't feed straight back into the decoder (raises
        # `K == K_w` in the post_quant_conv matmul). Round-trip through host and reuse
        # `WanDecoder.prepare_input()` -- the exact torch-space channel/height/width padding helper
        # the production decoder entry point uses for an externally-supplied latent -- then reshard.
        z_concat_dims = [None, None]
        z_concat_dims[h_axis] = 3
        z_concat_dims[w_axis] = 4
        z_full_torch = ttnn.to_torch(
            z_BCTHW,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=z_concat_dims
            ),
        )
        z_full_torch = z_full_torch[:, :, :, :lat_logical_h, :lat_logical_w]
        z_BTHWC, dec_logical_h, dec_logical_w = self.decoder.prepare_input(z_full_torch)
        z_BTHWC = typed_tensor_2dshard(
            z_BTHWC,
            self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={h_axis: 2, w_axis: 3},
            dtype=self.dtype,
        )

        out_BCTHW, out_logical_h, out_logical_w = self.decoder(
            z_BTHWC, dec_logical_h, t_chunk_size=None, logical_w=dec_logical_w
        )

        concat_dims = [None, None]
        concat_dims[h_axis] = 3
        concat_dims[w_axis] = 4
        out_torch = ttnn.to_torch(
            out_BCTHW,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=concat_dims
            ),
        )
        out_torch = out_torch[:, :, :, :out_logical_h, :out_logical_w]

        if not return_dict:
            return (out_torch,)
        return out_torch


def build(device: ttnn.MeshDevice, torch_module) -> TtAutoencoderKLWan:
    return TtAutoencoderKLWan(device, torch_module)
