"""tt_hw_planner: native TTNN port of the Qwen-Image VAE causal 3D conv (`encoder.conv_in`).

Component: qwen_image_causal_conv3d  (torch reference: diffusers QwenImageCausalConv3d)

Reuses tt_dit's native `WanCausalConv3d` (the op WanEncoder3D uses for every causal conv).

Tensor-parallel scheme (TP over the 1xN mesh): a conv's cost is spatial, and splitting its output
channels would leave per-chip widths below a tile for the narrow VAE stages, so -- as in
encoder_stack -- the parallel axis is SPATIAL. The activation is partitioned along W across the
mesh, the conv exchanges one-column halos with its neighbours (neighbor_pad), weights stay
replicated, and the output is all-gathered along W. The gathered output equals the single-device
result.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanCausalConv3d
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._resident import ResidentPort
from models.tt_dit.utils.conv3d import aligned_channels

# Mesh axis that carries the W partition; the other axis carries H.
_H_AXIS = 0
_W_AXIS = 1


def _mesh_shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return (1, 1)
    return shape if len(shape) == 2 else (1, 1)


class TtQwenImageCausalConv3d(ResidentPort):
    BODY_ATTR = "conv"  # inside encoder3d/decoder3d the port is entered via forward_sharded

    def __init__(self, device, torch_module):
        self.device = device
        mesh_shape = _mesh_shape(device)
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=mesh_shape[_H_AXIS], mesh_axis=_H_AXIS),
            width_parallel=ParallelFactor(factor=mesh_shape[_W_AXIS], mesh_axis=_W_AXIS),
        )
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1)

        self.in_channels = torch_module.in_channels
        self.out_channels = torch_module.out_channels
        self.conv = WanCausalConv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=torch_module.kernel_size,
            stride=torch_module.stride,
            padding=_symmetric_padding(torch_module),
            mesh_device=device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            dtype=ttnn.bfloat16,
        )
        state = {k: v.detach().float() for k, v in torch_module.state_dict().items()}
        c_pad = aligned_channels(self.in_channels) - self.in_channels
        if c_pad:
            # Zero input channels contribute nothing -- matches the zero-padded activation below.
            state["weight"] = torch.nn.functional.pad(state["weight"], (0, 0, 0, 0, 0, 0, 0, c_pad))
        self.conv.load_torch_state_dict(state)

    def __call__(self, x, cache_x=None, **_ignored):
        # x: replicated TILE [B, C, T, H, W] (BCTHW, like the torch reference). The captured call has
        # no cache (first chunk), i.e. causal zero front-padding in T.
        assert cache_x is None, "only the first (cache-free) chunk is ported; cache_x is not supported"
        B, C, T, H, W = x.shape
        pc = self.parallel_config
        assert (
            H % pc.height_parallel.factor == 0 and W % pc.width_parallel.factor == 0
        ), f"image {H}x{W} must divide the {pc.height_parallel.factor}x{pc.width_parallel.factor} spatial mesh"

        x = ttnn.permute(x, (0, 2, 3, 4, 1))  # BTHWC
        c_pad = aligned_channels(C) - C
        if c_pad:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, 0), (0, 0), (0, c_pad)], value=0.0)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if pc.height_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
        if pc.width_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=3, cluster_axis=pc.width_parallel.mesh_axis)

        out = self.conv(x, H, logical_w=W)

        if pc.width_parallel.factor > 1:
            out = self.ccl_manager.all_gather(out, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=False)
        if pc.height_parallel.factor > 1:
            out = self.ccl_manager.all_gather(out, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=False)

        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        out = ttnn.permute(out, (0, 4, 1, 2, 3))  # BCTHW
        if out.shape[1] != self.out_channels:
            ob, _, ot, oh, ow = out.shape
            out = ttnn.slice(out, (0, 0, 0, 0, 0), (ob, self.out_channels, ot, oh, ow))
        return out


def _symmetric_padding(torch_module):
    # QwenImageCausalConv3d stores F.pad order (w, w, h, h, 2*t, 0); WanCausalConv3d takes (t, h, w).
    p = torch_module._padding
    return (p[4] // 2, p[2], p[0])


def build(device, torch_module=None):
    return TtQwenImageCausalConv3d(device, torch_module)
