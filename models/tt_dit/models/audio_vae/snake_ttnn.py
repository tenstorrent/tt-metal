import torch

import ttnn
from models.tt_dit.layers.module import Module, Parameter


class Snake(Module):
    def __init__(
        self,
        *,
        channels: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.initial_alpha = alpha

        # TTNN Conv1d uses NLC, so alpha broadcasts over B and T.
        self.alpha = Parameter(
            total_shape=[1, 1, channels],
            device=mesh_device,
            dtype=dtype,
            pad_value=1.0,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        state["alpha"] = torch.full(
            (1, 1, self.channels),
            self.initial_alpha,
            dtype=torch.float32,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        xa = ttnn.multiply(x, self.alpha.data)
        sin_xa = ttnn.sin(xa)
        sin_sq = ttnn.pow(sin_xa, 2)

        denom = ttnn.add(self.alpha.data, 1e-9)
        inv_alpha = ttnn.reciprocal(denom)

        periodic = ttnn.multiply(sin_sq, inv_alpha)
        result = ttnn.add(x, periodic)

        if result.layout != x.layout:
            result = ttnn.to_layout(result, x.layout)

        return result
