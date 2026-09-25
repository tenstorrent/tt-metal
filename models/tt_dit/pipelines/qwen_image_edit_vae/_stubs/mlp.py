"""tt_hw_planner: native TTNN port of the Qwen-Image VAE's pointwise projection (`quant_conv`).

Component: mlp  (torch reference: diffusers QwenImageCausalConv3d, kernel 1x1x1)

The VAE has no transformer MLP; its standalone feed-forward projection is `quant_conv`, a 1x1x1
causal conv (no temporal/spatial padding) that maps the encoder latent 2*z_dim -> 2*z_dim channels
per voxel. A 1x1x1 conv is exactly a per-voxel linear over channels, so this is one `ttnn.linear`
on the channels-last token matrix. This replaces the tt_transformers SwiGLU MLP scaffold.

Tensor-parallel scheme (TP over the 1xN mesh): the weight is only 32x32 -- splitting its output
features gives 4 per chip, below one tile -- so, like encoder_stack / layer, the parallel axis is
SPATIAL. The flattened voxel (token) axis is partitioned across the mesh, the weight/bias stay
replicated, each chip projects its own voxels, and the tokens are all-gathered back in order. The
gathered output equals the single-device result.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.parallel.manager import CCLManager

_TOKEN_DIM = 2


def _mesh_shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return (1, 1)
    return shape if len(shape) == 2 else (1, 1)


class TtQwenImagePointwiseProj:
    def __init__(self, device, torch_module):
        self.device = device
        mesh_shape = _mesh_shape(device)
        # Tokens are split across the larger mesh axis (the N of a 1xN mesh).
        self.tp_axis = 1 if mesh_shape[1] >= mesh_shape[0] else 0
        self.tp = mesh_shape[self.tp_axis]
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1) if self.tp > 1 else None

        weight = torch_module.weight.detach().float()
        assert tuple(weight.shape[2:]) == (1, 1, 1), f"expected a 1x1x1 conv, got kernel {tuple(weight.shape[2:])}"
        self.out_channels = weight.shape[0]
        is_mesh = device.__class__.__name__ == "MeshDevice"
        replicate = ttnn.ReplicateTensorToMesh(device) if is_mesh else None

        self.weight = ttnn.from_torch(
            weight.reshape(self.out_channels, -1).t().contiguous(),  # [C_in, C_out]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=replicate,
        )
        self.bias = None
        if torch_module.bias is not None:
            self.bias = ttnn.from_torch(
                torch_module.bias.detach().float().reshape(1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=replicate,
            )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x, cache_x=None, **_ignored):
        # x: replicated TILE [B, C, T, H, W] (BCTHW, like the torch reference). A 1x1x1 conv has no
        # temporal padding, so the causal cache never enters the math.
        B, C, T, H, W = x.shape
        n_tokens = B * T * H * W
        assert n_tokens % self.tp == 0, f"{n_tokens} voxels must divide the TP={self.tp} mesh"

        x = ttnn.permute(x, (0, 2, 3, 4, 1))  # BTHWC
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (1, 1, n_tokens, C))
        if self.tp > 1:
            x = ttnn.mesh_partition(x, dim=_TOKEN_DIM, cluster_axis=self.tp_axis)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        out = ttnn.linear(x, self.weight, bias=self.bias, compute_kernel_config=self.compute_kernel_config)

        if self.tp > 1:
            out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
            out = self.ccl_manager.all_gather(out, dim=_TOKEN_DIM, mesh_axis=self.tp_axis, use_hyperparams=False)
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.reshape(out, (B, T, H, W, self.out_channels))
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return ttnn.permute(out, (0, 4, 1, 2, 3))  # BCTHW


def build(device, torch_module=None):
    return TtQwenImagePointwiseProj(device, torch_module)
