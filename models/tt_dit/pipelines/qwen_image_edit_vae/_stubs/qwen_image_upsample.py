"""tt_hw_planner: native TTNN port of the Qwen-Image VAE spatial upsample.

Component: qwen_image_upsample  (torch reference: diffusers QwenImageUpsample,
`decoder.up_blocks.0.upsamplers.0.resample.0`)

QwenImageUpsample is nn.Upsample(scale_factor=2, mode="nearest-exact") on NCHW frames. For an
integer scale, nearest-exact (src = floor((dst + 0.5) / s)) picks the same source pixel as nearest
(src = floor(dst / s)), so this is `ttnn.upsample` (nearest) on the NHWC tensor -- the same op
WanResample uses for its upsample.

Tensor-parallel scheme (TP over the 1xN mesh): the op has no weights and is pixel-local, so the
parallel axis is SPATIAL, as in decoder_head. The frame is partitioned along W across the mesh;
nearest-upsampling a W shard yields exactly the matching shard of the upsampled frame (no halo
needed), and the output is all-gathered along W. The gathered output equals the single-device result.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.parallel.manager import CCLManager

# NHWC layout: W is dim 2.
_W_DIM = 2


def _mesh_shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return (1, 1)
    return shape if len(shape) == 2 else (1, 1)


class TtQwenImageUpsample:
    def __init__(self, device, torch_module):
        assert torch_module.mode in ("nearest", "nearest-exact"), f"unsupported upsample mode {torch_module.mode}"
        assert torch_module.size is None, "only scale_factor upsampling is used by this VAE"
        scale = torch_module.scale_factor
        scale = tuple(scale) if isinstance(scale, (tuple, list)) else (scale, scale)
        assert all(
            float(s).is_integer() for s in scale
        ), f"nearest-exact == nearest only for integer scales, got {scale}"
        self.scale = tuple(int(s) for s in scale)

        self.device = device
        mesh_shape = _mesh_shape(device)
        # W is split across the larger mesh axis (the N of a 1xN mesh).
        self.tp_axis = 1 if mesh_shape[1] >= mesh_shape[0] else 0
        self.tp = mesh_shape[self.tp_axis]
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1) if self.tp > 1 else None

    @classmethod
    def resident(cls, device, torch_module):
        """The op port used inside the decoder (forward_sharded only): no CCLManager / L1 semaphores."""
        self = cls.__new__(cls)
        assert torch_module.mode in ("nearest", "nearest-exact") and torch_module.size is None
        scale = torch_module.scale_factor
        scale = tuple(scale) if isinstance(scale, (tuple, list)) else (scale, scale)
        assert all(float(s).is_integer() for s in scale)
        self.scale = tuple(int(s) for s in scale)
        self.device, self.tp, self.ccl_manager = device, 1, None
        return self

    def forward_sharded(self, x_NHWC):
        """Inside the decoder: nearest x2 of a ROW_MAJOR NHWC W-shard is exactly the matching shard of the
        upsampled frame (no halo), so the op runs on the shard as is."""
        return ttnn.upsample(x_NHWC, scale_factor=self.scale)

    def __call__(self, x, **_ignored):
        # x: replicated TILE [N, C, H, W] (NCHW, like the torch reference).
        N, C, H, W = x.shape
        assert W % self.tp == 0, f"width {W} must divide the TP={self.tp} mesh"

        x = ttnn.permute(x, (0, 2, 3, 1))  # NHWC
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if self.tp > 1:
            x = ttnn.mesh_partition(x, dim=_W_DIM, cluster_axis=self.tp_axis)

        out = ttnn.upsample(x, scale_factor=self.scale)

        if self.tp > 1:
            out = self.ccl_manager.all_gather(out, dim=_W_DIM, mesh_axis=self.tp_axis, use_hyperparams=False)
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return ttnn.permute(out, (0, 3, 1, 2))  # NCHW


def build(device, torch_module=None):
    return TtQwenImageUpsample(device, torch_module)
