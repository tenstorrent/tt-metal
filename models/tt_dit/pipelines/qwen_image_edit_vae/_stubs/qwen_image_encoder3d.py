"""tt_hw_planner: native TTNN port of the Qwen-Image VAE encoder (`AutoencoderKLQwenImage.encoder`).

Component: qwen_image_encoder3d  (torch reference: diffusers QwenImageEncoder3d)

QwenImageEncoder3d is architecturally identical to the Wan2.1 VAE encoder (causal 3D convs,
RMS norms, resample down-blocks, mid-block attention), so this reuses tt_dit's native
`WanEncoder3D` rather than the llama-vision-encoder scaffold this stub was seeded with.

Tensor-parallel scheme (TP over the 1xN mesh): a conv VAE has no large matmul to column/row split
-- its cost is the spatial convolutions -- so the parallel axis is SPATIAL. The image is
partitioned along W across the mesh (each chip convolves its own W-slice; WanCausalConv3d
exchanges halos with its neighbours), conv/norm weights stay replicated, and the latent is
all-gathered along W. The math is unchanged: the gathered output equals the single-device encode.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanEncoder3D
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import aligned_channels, count_convs

# Mesh axis that carries the W partition; the other axis (size 1 on a 1xN mesh) carries H.
_H_AXIS = 0
_W_AXIS = 1


def _mesh_shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return (1, 1)
    return shape if len(shape) == 2 else (1, 1)


class TtQwenImageEncoder3d:
    def __init__(self, device, torch_module, batch_parallel=False):
        """batch_parallel: split the batch over mesh axis 0 (DP) instead of partitioning H there. The
        Wan causal conv fuses its temporal front pad into the H halo exchange, and that fused
        neighbor_pad only takes B=1; with H unsplit the pad is a plain ttnn.pad and B > 1 runs."""
        self.device = device
        mesh_shape = _mesh_shape(device)
        self.batch_axis = _H_AXIS if (batch_parallel and mesh_shape[_H_AXIS] > 1) else None
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(
                factor=1 if self.batch_axis is not None else mesh_shape[_H_AXIS], mesh_axis=_H_AXIS
            ),
            width_parallel=ParallelFactor(factor=mesh_shape[_W_AXIS], mesh_axis=_W_AXIS),
        )
        self.batch_factor = mesh_shape[_H_AXIS] if self.batch_axis is not None else 1
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1)
        self.out_channels = torch_module.conv_out.out_channels

        self.encoder = WanEncoder3D(
            in_channels=torch_module.conv_in.in_channels,
            dim=torch_module.dim,
            z_dim=torch_module.z_dim,
            dim_mult=list(torch_module.dim_mult),
            num_res_blocks=torch_module.num_res_blocks,
            attn_scales=list(torch_module.attn_scales),
            temperal_downsample=list(torch_module.temperal_downsample),
            mesh_device=device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            dtype=ttnn.bfloat16,
        )
        self.encoder.load_torch_state_dict(torch_module.state_dict())
        self.num_convs = count_convs(self.encoder)
        # Every child of the Wan stack runs as its graduated port (causal conv, RMS norm, residual /
        # mid / attention / up block, resample with the zero_pad2d / upsample op ports) on the
        # partitioned activation; see _resident.py.
        from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._resident import attach_block_ports, ports_of

        self.ports = attach_block_ports(
            self.encoder, device, self.parallel_config, self.ccl_manager, torch_stack=torch_module
        )
        # the repeated stack (residual blocks + resamples, common base ResidentPort), as a plain list
        self.down_blocks = ports_of(self.ports, self.encoder.down_blocks)
        # the mid block's residual blocks (a second repeated stack, HF encoder.mid_block.resnets)
        self.mid_block_resnets = ports_of(self.ports, self.encoder.mid_block.resnets)

    def __call__(self, x, feat_cache=None, feat_idx=None, **_ignored):
        # x: replicated TILE [B, C=3, T, H, W] (BCTHW, like the torch reference).
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
        if self.batch_axis is not None:
            assert B % self.batch_factor == 0, f"batch {B} must divide the {self.batch_factor}-way batch axis"
            x = ttnn.mesh_partition(x, dim=0, cluster_axis=self.batch_axis)
        if pc.height_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
        if pc.width_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=3, cluster_axis=pc.width_parallel.mesh_axis)

        # Fresh causal-conv cache, exactly as the reference's first (and only) chunk sees it.
        tt_feat_cache = [None] * self.num_convs
        tt_feat_idx = [0]
        out, _logical_h, _logical_w = self.encoder(x, H, feat_cache=tt_feat_cache, feat_idx=tt_feat_idx, logical_w=W)

        out = self.ccl_manager.all_gather(out, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=False)
        if pc.height_parallel.factor > 1:
            out = self.ccl_manager.all_gather(out, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=False)
        if self.batch_axis is not None:
            out = self.ccl_manager.all_gather(out, dim=0, mesh_axis=self.batch_axis, use_hyperparams=False)

        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        out = ttnn.permute(out, (0, 4, 1, 2, 3))  # BCTHW
        if out.shape[1] != self.out_channels:
            ob, _, ot, oh, ow = out.shape
            out = ttnn.slice(out, (0, 0, 0, 0, 0), (ob, self.out_channels, ot, oh, ow))
        return out


def build(device, torch_module, **kwargs):
    return TtQwenImageEncoder3d(device, torch_module, **kwargs)
