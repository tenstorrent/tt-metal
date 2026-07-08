# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `wan_mid_block`
(meituan-longcat/LongCat-Video's `vae.encoder.mid_block`, a real
`diffusers.models.autoencoders.autoencoder_kl_wan.WanMidBlock` -- the VAE
bottleneck: `resnet -> (attention -> resnet) x num_layers`, constant
resolution throughout, no up/downsampling).

Adapts the already-validated `WanMidBlock` in
`models/tt_dit/models/vae/vae_wan2_1.py` -- the SAME class the graduated
`autoencoder_k_l_wan` uses internally in both its encoder and decoder (and
which itself composes the already-graduated `wan_attention_block`'s
`WanAttentionBlock`) -- same rationale/precedent as `wan_attention_block`/
`wan_causal_conv3d`. `dim` (384 for this checkpoint) is read from the
resolved submodule's own `attentions[0].dim` rather than hardcoded, matching
`wan_attention_block.py`'s pattern.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.hf_eager.longcat_video._stubs.autoencoder_k_l_wan import (
    _hw_parallel_config,
    _replicated_ttnn_to_torch,
)
from models.tt_dit.models.vae.vae_wan2_1 import WanMidBlock
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_width
from models.tt_dit.utils.tensor import typed_tensor_2dshard


class TtWanMidBlock:
    def __init__(self, mesh_device: ttnn.MeshDevice, torch_module) -> None:
        self.mesh_device = mesh_device
        self.parallel_config = _hw_parallel_config(mesh_device)
        self.ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        self.dtype = ttnn.bfloat16

        self.module = WanMidBlock(
            dim=torch_module.attentions[0].dim,
            num_layers=len(torch_module.attentions),
            mesh_device=mesh_device,
            parallel_config=self.parallel_config,
            ccl_manager=self.ccl_manager,
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
        x_BTHWC, logical_h = conv_pad_height(x_BTHWC, h_factor * 8)
        x_BTHWC, logical_w = conv_pad_width(x_BTHWC, w_factor * 8)
        x_BTHWC = typed_tensor_2dshard(
            x_BTHWC,
            self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            shard_mapping={h_axis: 2, w_axis: 3},
            dtype=self.dtype,
        )

        out_BTHWC = self.module(x_BTHWC, logical_h, logical_w=logical_w)

        concat_dims = [None, None]
        concat_dims[h_axis] = 2
        concat_dims[w_axis] = 3
        out_torch = ttnn.to_torch(
            out_BTHWC,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=concat_dims
            ),
        )
        out_torch = out_torch[:, :, :logical_h, :logical_w, :]
        return out_torch.permute(0, 4, 1, 2, 3)  # BTHWC -> BCTHW


def build(mesh_device: ttnn.MeshDevice, torch_module) -> TtWanMidBlock:
    return TtWanMidBlock(mesh_device, torch_module)
