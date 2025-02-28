import ttnn
import torch
from models.common.lightweightmodule import LightweightModule

from models.experimental.mochi.common import (
    to_tt_tensor,
)
from loguru import logger


class TtGroupNorm(LightweightModule):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        state_dict_prefix: str,
        num_groups: int,
        channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        self.mesh_device = mesh_device
        self.channels = channels
        self.affine = affine

        # NOTE: fallback to host until ttnn.group_norm is functional
        self.norm = torch.nn.GroupNorm(num_groups, channels, eps=eps, affine=affine)
        partial_state_dict = {
            k[len(state_dict_prefix) :]: v for k, v in state_dict.items() if k.startswith(state_dict_prefix)
        }
        self.norm.load_state_dict(partial_state_dict, strict=True)

    def forward(self, x_NTHWC):
        logger.warning("GroupNorm is not functional in TTNN, fallback to host")
        N, T, H, W, C = x_NTHWC.shape
        torch_x_NTHWC = ttnn.to_torch(x_NTHWC, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[0:1]
        torch_x_NT_CHW = torch_x_NTHWC.permute(0, 1, 4, 2, 3).reshape(N * T, C, H, W)
        torch_x_NT_CHW = self.norm(torch_x_NT_CHW)
        torch_x_NTCHW = torch_x_NT_CHW.reshape(N, T, C, H, W)
        torch_x_NTHWC = torch_x_NTCHW.permute(0, 1, 3, 4, 2)
        x_NTHWC = ttnn.from_torch(
            torch_x_NTHWC,
            device=self.mesh_device,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return x_NTHWC
