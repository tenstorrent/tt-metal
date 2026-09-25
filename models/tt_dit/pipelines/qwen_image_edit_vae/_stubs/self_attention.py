"""tt_hw_planner: native TTNN port of the Qwen-Image VAE mid-block attention.

Component: self_attention  (torch reference: diffusers QwenImageAttentionBlock,
`encoder.mid_block.attentions.0`)

QwenImageAttentionBlock is architecturally identical to the Wan2.1 VAE attention block (RMS norm,
1x1 to_qkv, single-head spatial SDPA per frame, 1x1 proj, residual), so this reuses tt_dit's native
`WanAttentionBlock`.

Tensor-parallel scheme (TP over the 1xN mesh): the block is single-head (no heads to split) and a
column split of the 384 -> 1152 qkv projection is not tile-aligned per chip, so -- as in
encoder_stack, which contains this block -- the parallel axis is SPATIAL. The activation arrives
W-partitioned across the mesh; the block all-gathers H/W for the (replicated) SDPA and re-partitions
its output, weights stay replicated, the residual is added per shard, and the output is
all-gathered along W. The gathered output equals the single-device result.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanAttentionBlock
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager

# Mesh axis that carries the W partition; the other axis (size 1 on a 1xN mesh) carries H.
_H_AXIS = 0
_W_AXIS = 1


def _mesh_shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return (1, 1)
    return shape if len(shape) == 2 else (1, 1)


class TtQwenImageAttentionBlock:
    def __init__(self, device, torch_module):
        self.device = device
        mesh_shape = _mesh_shape(device)
        self.parallel_config = VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=mesh_shape[_H_AXIS], mesh_axis=_H_AXIS),
            width_parallel=ParallelFactor(factor=mesh_shape[_W_AXIS], mesh_axis=_W_AXIS),
        )
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1)

        self.block = WanAttentionBlock(
            dim=torch_module.dim,
            mesh_device=device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            dtype=ttnn.bfloat16,
        )
        self.block.load_torch_state_dict(torch_module.state_dict())

    def __call__(self, x, **_ignored):
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

        out = self.block(x, H, logical_w=W)

        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = self.ccl_manager.all_gather(out, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=False)
        out = self.ccl_manager.all_gather(out, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=False)

        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return ttnn.permute(out, (0, 4, 1, 2, 3))  # BCTHW


def build(device, torch_module):
    return TtQwenImageAttentionBlock(device, torch_module)
