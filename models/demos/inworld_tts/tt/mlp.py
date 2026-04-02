"""TTNN implementation of SiLU MLP for VocosBackbone.

Simple two-layer MLP: Linear(d, 4d) -> SiLU -> Linear(4d, d).
All activations kept in L1, using full core grid for matmuls.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.inworld_tts.tt.model_config import VOCOS_DIM, VOCOS_MLP_DIM, get_compute_kernel_config_hifi4

L1 = ttnn.L1_MEMORY_CONFIG


class TtMLP(LightweightModule):
    """SiLU MLP: fc1(d->4d) -> silu -> fc2(4d->d). All ops in L1, full core grid."""

    def __init__(
        self,
        device,
        state_dict,
        layer_num,
        dim=VOCOS_DIM,
        mlp_dim=VOCOS_MLP_DIM,
        dtype=ttnn.bfloat16,
        state_dict_prefix="",
    ):
        super().__init__()
        self.device = device

        # Get full compute grid from device
        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        prefix = f"{state_dict_prefix}transformers.{layer_num}.mlp."

        self.fc1 = ttnn.from_torch(
            state_dict[prefix + "fc1.weight"].T.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.fc2 = ttnn.from_torch(
            state_dict[prefix + "fc2.weight"].T.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.compute_kernel_config = get_compute_kernel_config_hifi4()

    def forward(self, x):
        """Forward pass. Full core grid matmuls, all in L1.

        Args:
            x: [1, 1, T, dim] in TILE_LAYOUT
        Returns:
            [1, 1, T, dim] in L1
        """
        h = ttnn.linear(
            x, self.fc1, core_grid=self.core_grid, memory_config=L1, compute_kernel_config=self.compute_kernel_config
        )
        h = ttnn.silu(h, memory_config=L1)
        h = ttnn.linear(
            h, self.fc2, core_grid=self.core_grid, memory_config=L1, compute_kernel_config=self.compute_kernel_config
        )
        return h
