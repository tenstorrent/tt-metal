# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `wan_encoder3d`
(meituan-longcat/LongCat-Video's `vae.encoder`, a real
`diffusers.models.autoencoders.autoencoder_kl_wan.WanEncoder3d`):

    forward(x):     # x: raw (B, 3, T, H, W) RGB video
        x = conv_in(x); x = down_blocks(x); x = mid_block(x); x = conv_out(silu(norm_out(x)))

Adapts the already-validated `WanEncoder3D` (the INNER class) in
`models/tt_dit/models/vae/vae_wan2_1.py` -- the SAME class the graduated
`autoencoder_k_l_wan` uses internally via its `WanEncoder` wrapper -- same
rationale/precedent as `wan_decoder3d` (which uses `WanDecoder3d` directly
rather than the `WanEncoder`/`WanDecoder` outer wrappers, since those also
run a `quant_conv`/`post_quant_conv` Linear step that lives on the OUTER
`AutoencoderKLWan` in diffusers, one level above this resolved submodule).

Config values (`dim_mult`, `num_res_blocks`, `attn_scales`,
`temperal_downsample`, `is_residual`) match this checkpoint's real
`vae/config.json` (verified against the loaded model directly, same values
`autoencoder_k_l_wan.py` reads from `cfg`); `in_channels`/`z_dim`/`dim` are
derived from the resolved submodule's own `conv_in`/`conv_out` instead of
hardcoding them.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.hf_eager.longcat_video._stubs.autoencoder_k_l_wan import (
    _hw_parallel_config,
    _replicated_ttnn_to_torch,
)
from models.tt_dit.models.vae.vae_wan2_1 import WanEncoder3D
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_pad_width
from models.tt_dit.utils.tensor import typed_tensor_2dshard

_DIM_MULT = [1, 2, 4, 4]
_NUM_RES_BLOCKS = 2
_ATTN_SCALES = []
_TEMPERAL_DOWNSAMPLE = [False, True, True]
_IS_RESIDUAL = False


class TtWanEncoder3D:
    def __init__(self, mesh_device: ttnn.MeshDevice, torch_module) -> None:
        self.mesh_device = mesh_device
        self.parallel_config = _hw_parallel_config(mesh_device)
        self.ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        self.dtype = ttnn.bfloat16

        in_channels = torch_module.conv_in.in_channels
        z_dim = torch_module.conv_out.out_channels
        dim = torch_module.conv_in.out_channels

        self.module = WanEncoder3D(
            in_channels=in_channels,
            dim=dim,
            z_dim=z_dim,
            dim_mult=_DIM_MULT,
            num_res_blocks=_NUM_RES_BLOCKS,
            attn_scales=_ATTN_SCALES,
            temperal_downsample=_TEMPERAL_DOWNSAMPLE,
            is_residual=_IS_RESIDUAL,
            mesh_device=mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            dtype=self.dtype,
        )
        self.module.load_torch_state_dict(torch_module.state_dict())

    def __call__(self, x: ttnn.Tensor) -> torch.Tensor:
        h_axis = self.parallel_config.height_parallel.mesh_axis
        w_axis = self.parallel_config.width_parallel.mesh_axis
        h_factor = self.parallel_config.height_parallel.factor
        w_factor = self.parallel_config.width_parallel.factor

        x_torch = _replicated_ttnn_to_torch(x, self.mesh_device).to(torch.float32)
        x_BTHWC = x_torch.permute(0, 2, 3, 4, 1)  # BCTHW -> BTHWC
        x_BTHWC = conv_pad_in_channels(x_BTHWC)
        x_BTHWC, logical_h = conv_pad_height(x_BTHWC, h_factor * 8)
        x_BTHWC, logical_w = conv_pad_width(x_BTHWC, w_factor * 8)
        x_BTHWC = typed_tensor_2dshard(
            x_BTHWC,
            self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={h_axis: 2, w_axis: 3},
            dtype=self.dtype,
        )

        out_BTHWC, out_logical_h, out_logical_w = self.module(x_BTHWC, logical_h, logical_w=logical_w)

        concat_dims = [None, None]
        concat_dims[h_axis] = 2
        concat_dims[w_axis] = 3
        out_torch = ttnn.to_torch(
            out_BTHWC,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=concat_dims
            ),
        )
        out_torch = out_torch[:, :, :out_logical_h, :out_logical_w, :]
        return out_torch.permute(0, 4, 1, 2, 3)  # BTHWC -> BCTHW


def build(mesh_device: ttnn.MeshDevice, torch_module) -> TtWanEncoder3D:
    return TtWanEncoder3D(mesh_device, torch_module)
