# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `wan_attention_block`
(meituan-longcat/LongCat-Video's `vae.encoder.mid_block.attentions.0`, a real
`diffusers.models.autoencoders.autoencoder_kl_wan.WanAttentionBlock` --
single-head causal self-attention over the VAE's spatial bottleneck):

    forward(x):                              # x: (B, C, T, H, W)
        x = norm(x); qkv = to_qkv(x); attn = sdpa(q, k, v); return proj(attn) + x

Adapts the already-validated `WanAttentionBlock` in
`models/tt_dit/models/vae/vae_wan2_1.py` -- the SAME class the graduated
`autoencoder_k_l_wan` uses internally for its mid_block attention -- same
rationale/precedent as `u_m_t5_block` adapting `model_t5.py`. Its TP scheme
(already implemented, not re-derived) is height/width-SHARDED ACTIVATIONS,
not weight-sharded: `to_qkv`/`proj` stay replicated `Linear`s, the attention
itself gathers H/W to a replicated softmax and scatters back afterward --
matching `autoencoder_k_l_wan.py`'s `_hw_parallel_config` convention (H
sharded on the mesh's larger axis, W replicated, for a 1x4 mesh).

Reuses `autoencoder_k_l_wan.py`'s `_hw_parallel_config`/`_replicated_ttnn_to_torch`
helpers directly rather than re-deriving them.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.hf_eager.longcat_video._stubs.autoencoder_k_l_wan import (
    _hw_parallel_config,
    _replicated_ttnn_to_torch,
)
from models.tt_dit.models.vae.vae_wan2_1 import WanAttentionBlock
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_width
from models.tt_dit.utils.tensor import typed_tensor_2dshard


class TtWanAttentionBlock:
    def __init__(self, mesh_device: ttnn.MeshDevice, torch_module) -> None:
        self.mesh_device = mesh_device
        self.parallel_config = _hw_parallel_config(mesh_device)
        self.ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        self.dtype = ttnn.bfloat16

        self.module = WanAttentionBlock(
            dim=torch_module.dim,
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


def build(mesh_device: ttnn.MeshDevice, torch_module) -> TtWanAttentionBlock:
    return TtWanAttentionBlock(mesh_device, torch_module)
