"""tt_hw_planner: native TTNN port of the Qwen-Image VAE decoder up block (`decoder.up_blocks.0`).

Component: qwen_image_up_block  (torch reference: diffusers QwenImageUpBlock)

QwenImageUpBlock (num_res_blocks+1 residual blocks, then an optional upsample2d/3d resample) is
architecturally identical to the Wan2.1 VAE up block, so this reuses tt_dit's native `WanUpBlock`.

Tensor-parallel scheme (TP over the 1xN mesh): same as decoder_head, which contains this block --
the parallel axis is SPATIAL. The activation is partitioned along W across the mesh (the convs
exchange halos with their neighbours; nearest-upsampling a W shard yields exactly the matching
shard of the upsampled image), weights stay replicated, and the output is all-gathered along W.
The gathered output equals the single-device result.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanUpBlock
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.qwen_image_edit_vae._stubs._resident import ResidentPort

# Mesh axis that carries the W partition; the other axis (size 1 on a 1xN mesh) carries H.
_H_AXIS = 0
_W_AXIS = 1


def _mesh_shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return (1, 1)
    return shape if len(shape) == 2 else (1, 1)


class TtQwenImageUpBlock(ResidentPort):
    BODY_ATTR = "block"  # inside encoder3d/decoder3d the port is entered via forward_sharded

    def __init__(self, device, torch_module):
        self.device = device
        mesh_shape = _mesh_shape(device)
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=mesh_shape[_H_AXIS], mesh_axis=_H_AXIS),
            width_parallel=ParallelFactor(factor=mesh_shape[_W_AXIS], mesh_axis=_W_AXIS),
        )
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1)

        resnets = torch_module.resnets
        upsampler = torch_module.upsamplers[0] if torch_module.upsamplers is not None else None
        # Two causal convs per residual block, plus one cache slot for the resample's time conv.
        self.num_convs = 2 * len(resnets) + (1 if upsampler is not None else 0)
        self.block = WanUpBlock(
            in_dim=resnets[0].in_dim,
            out_dim=resnets[0].out_dim,
            num_res_blocks=len(resnets) - 1,
            upsample_mode=upsampler.mode if upsampler is not None else None,
            mesh_device=device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            dtype=ttnn.bfloat16,
        )
        self.block.load_torch_state_dict(torch_module.state_dict())

    def __call__(self, x, feat_cache=None, feat_idx=None, **_ignored):
        # x: replicated TILE [B, C, T, H, W] (BCTHW, like the torch reference).
        B, C, T, H, W = x.shape
        pc = self.parallel_config
        assert (
            H % pc.height_parallel.factor == 0 and W % pc.width_parallel.factor == 0
        ), f"activation {H}x{W} must divide the {pc.height_parallel.factor}x{pc.width_parallel.factor} spatial mesh"

        x = ttnn.permute(x, (0, 2, 3, 4, 1))  # BTHWC
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if pc.height_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
        if pc.width_parallel.factor > 1:
            x = ttnn.mesh_partition(x, dim=3, cluster_axis=pc.width_parallel.mesh_axis)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Fresh causal-conv cache, exactly as the reference's first (and only) chunk sees it.
        tt_feat_cache = [None] * self.num_convs
        tt_feat_idx = [0]
        out, _logical_h, _logical_w = self.block(x, H, feat_cache=tt_feat_cache, feat_idx=tt_feat_idx, logical_w=W)

        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = self.ccl_manager.all_gather(out, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=False)
        out = self.ccl_manager.all_gather(out, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=False)

        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return ttnn.permute(out, (0, 4, 1, 2, 3))  # BCTHW


def build(device, torch_module):
    return TtQwenImageUpBlock(device, torch_module)
