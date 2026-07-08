# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `wan_decoder3d`
(meituan-longcat/LongCat-Video's `vae.decoder`, a real
`diffusers.models.autoencoders.autoencoder_kl_wan.WanDecoder3d`):

    forward(x):     # x: raw (B, z_dim, T, H, W) latent, NO quant/post_quant conv here
        x = conv_in(x); x = mid_block(x); x = up_blocks(x); x = conv_out(silu(norm_out(x)))

Adapts the already-validated `WanDecoder3d` (the INNER class) in
`models/tt_dit/models/vae/vae_wan2_1.py` -- the SAME class the graduated
`autoencoder_k_l_wan` uses internally via its `WanDecoder` wrapper -- same
rationale/precedent as `wan_attention_block`/`wan_causal_conv3d`. Uses
`WanDecoder3d` DIRECTLY rather than the `WanDecoder` outer wrapper:
`WanDecoder` also runs a `post_quant_conv` step first, but that lives on the
OUTER `AutoencoderKLWan.decode()` in diffusers, one level above this
resolved submodule (`vae.decoder` IS `WanDecoder3d`, called directly on the
raw z_dim-channel latent, no post_quant_conv) -- so wrapping `WanDecoder`
here would compute an extra op the golden reference never applies.

Config values (`dim_mult`, `num_res_blocks`, `attn_scales`,
`temperal_downsample`, `is_residual`) match this checkpoint's real
`vae/config.json` (verified against the loaded model directly, same values
`autoencoder_k_l_wan.py` reads from `cfg`); `z_dim`/`out_channels`/`dim` are
derived from the resolved submodule's own `conv_in`/`conv_out` instead of
hardcoding them, so a channel-count typo would surface as a shape error
rather than a silent mismatch.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.hf_eager.longcat_video._stubs.autoencoder_k_l_wan import (
    _hw_parallel_config,
    _replicated_ttnn_to_torch,
)
from models.tt_dit.models.vae.vae_wan2_1 import WanDecoder3d
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import aligned_channels, conv_pad_height, conv_pad_width
from models.tt_dit.utils.tensor import typed_tensor_2dshard

_DIM_MULT = [1, 2, 4, 4]
_NUM_RES_BLOCKS = 2
_ATTN_SCALES = []
_TEMPERAL_UPSAMPLE = [True, True, False]  # reverse of vae/config.json's temperal_downsample
_IS_RESIDUAL = False


class TtWanDecoder3d:
    def __init__(self, mesh_device: ttnn.MeshDevice, torch_module) -> None:
        self.mesh_device = mesh_device
        self.parallel_config = _hw_parallel_config(mesh_device)
        self.ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        self.dtype = ttnn.bfloat16

        z_dim = torch_module.conv_in.in_channels
        out_channels = torch_module.conv_out.out_channels
        dim = torch_module.conv_in.out_channels // _DIM_MULT[-1]

        self.module = WanDecoder3d(
            dim=dim,
            z_dim=z_dim,
            dim_mult=_DIM_MULT,
            num_res_blocks=_NUM_RES_BLOCKS,
            attn_scales=_ATTN_SCALES,
            temperal_upsample=_TEMPERAL_UPSAMPLE,
            out_channels=out_channels,
            is_residual=_IS_RESIDUAL,
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

        x_dev_BTHWC = ttnn.permute(x, (0, 2, 3, 4, 1))  # BCTHW -> BTHWC, on-device
        x_BTHWC = _replicated_ttnn_to_torch(x_dev_BTHWC, self.mesh_device).to(torch.float32)
        x_BTHWC, logical_h = conv_pad_height(x_BTHWC, h_factor)
        x_BTHWC, logical_w = conv_pad_width(x_BTHWC, w_factor)
        x_BTHWC = typed_tensor_2dshard(
            x_BTHWC,
            self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={h_axis: 2, w_axis: 3},
            dtype=self.dtype,
        )
        # Channel-count padding (e.g. 16 -> 32) is identical zero-fill on every device's local
        # shard (only H/W are sharded here, never channels) -- pad on-device with ttnn.pad AFTER
        # upload instead of torch.nn.functional.pad on the host tensor, so this real per-call
        # data path (padding depends on the actual latent's channel count) stays on TT.
        C_in = x_BTHWC.shape[-1]
        padded_C_in = aligned_channels(C_in)
        if padded_C_in != C_in:
            x_BTHWC = ttnn.pad(x_BTHWC, [(0, 0), (0, 0), (0, 0), (0, 0), (0, padded_C_in - C_in)], 0.0)

        out_BTHWC, out_logical_h, out_logical_w = self.module(x_BTHWC, logical_h, logical_w=logical_w)

        out_BCTHW = ttnn.permute(out_BTHWC, (0, 4, 1, 2, 3))  # BTHWC -> BCTHW, on-device
        # H/W moved from dims (2, 3) to (3, 4) after inserting the channel axis at dim 1.
        concat_dims = [None, None]
        concat_dims[h_axis] = 3
        concat_dims[w_axis] = 4
        out_torch = ttnn.to_torch(
            out_BCTHW,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=concat_dims
            ),
        )
        return out_torch[:, :, :, :out_logical_h, :out_logical_w]


def build(mesh_device: ttnn.MeshDevice, torch_module) -> TtWanDecoder3d:
    return TtWanDecoder3d(mesh_device, torch_module)
