"""tt_hw_planner: native TTNN port of the Qwen-Image VAE downsample zero-pad.

Component: zero_pad2d  (torch reference: nn.ZeroPad2d((0, 1, 0, 1)), `encoder.down_blocks.2.resample.0`)

nn.ZeroPad2d(left, right, top, bottom) zero-pads the last two (H, W) axes of an NCHW frame; here it
adds one zero row at the bottom and one zero column on the right before the stride-2 downsample conv.
This is a single `ttnn.pad` on the row-major NCHW tensor.

Tensor-parallel scheme (TP over the 1xN mesh): the op has no weights and pads every channel
identically, so the parallel axis is the CHANNEL axis -- a W split would put the right-hand pad on
the last chip only. Channels are partitioned across the mesh, each chip pads its own channels, and
the output is all-gathered along C. The gathered output equals the single-device result.
"""

from __future__ import annotations

import ttnn
from models.tt_dit.parallel.manager import CCLManager

# NCHW layout: C is dim 1.
_C_DIM = 1


def _mesh_shape(device):
    try:
        shape = tuple(device.shape)
    except (AttributeError, TypeError):
        return (1, 1)
    return shape if len(shape) == 2 else (1, 1)


class TtZeroPad2d:
    def __init__(self, device, torch_module):
        left, right, top, bottom = torch_module.padding
        assert float(torch_module.value) == 0.0, f"expected zero padding, got value={torch_module.value}"
        self.pad_h = (top, bottom)
        self.pad_w = (left, right)

        self.device = device
        mesh_shape = _mesh_shape(device)
        # Channels are split across the larger mesh axis (the N of a 1xN mesh).
        self.tp_axis = 1 if mesh_shape[1] >= mesh_shape[0] else 0
        self.tp = mesh_shape[self.tp_axis]
        self.ccl_manager = CCLManager(device, topology=ttnn.Topology.Linear, num_links=1) if self.tp > 1 else None

    @classmethod
    def resident(cls, device, torch_module):
        """The op port used inside the encoder (forward_sharded only): no CCLManager / L1 semaphores."""
        self = cls.__new__(cls)
        left, right, top, bottom = torch_module.padding
        assert float(torch_module.value) == 0.0, f"expected zero padding, got value={torch_module.value}"
        self.pad_h, self.pad_w = (top, bottom), (left, right)
        self.device, self.tp, self.ccl_manager = device, 1, None
        return self

    def forward_sharded(self, x_BTHWC):
        """Inside the encoder: x is a ROW_MAJOR BTHWC tensor with H whole on every chip and W split over
        the mesh. The bottom zero row is padded here, locally. The right zero column falls on the global W
        edge, which the following W-sharded conv's halo exchange zero-fills; that conv samples odd columns,
        so its left halo column is never read."""
        top, bottom = self.pad_h
        assert tuple(self.pad_w) == (0, 1), f"downsample pad expected (0, 1) on W, got {self.pad_w}"
        B, T, H, W, C = x_BTHWC.shape
        x = ttnn.reshape(x_BTHWC, (B * T, H, W, C))
        x = ttnn.pad(x, [(0, 0), (top, bottom), (0, 0), (0, 0)], value=0.0)
        return ttnn.reshape(x, (B, T, H + top + bottom, W, C))

    def __call__(self, x, **_ignored):
        # x: replicated TILE [N, C, H, W] (NCHW, like the torch reference).
        N, C, H, W = x.shape
        assert C % self.tp == 0, f"{C} channels must divide the TP={self.tp} mesh"

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if self.tp > 1:
            x = ttnn.mesh_partition(x, dim=_C_DIM, cluster_axis=self.tp_axis)

        out = ttnn.pad(x, [(0, 0), (0, 0), self.pad_h, self.pad_w], value=0.0)

        if self.tp > 1:
            out = self.ccl_manager.all_gather(out, dim=_C_DIM, mesh_axis=self.tp_axis, use_hyperparams=False)
        return out


def build(device, torch_module=None):
    return TtZeroPad2d(device, torch_module)
