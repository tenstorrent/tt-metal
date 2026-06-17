import ttnn
from models.common.lightweightmodule import LightweightModule


class Qwen35RMSNorm(LightweightModule):
    """ttnn port of transformers' Qwen3_5RMSNorm (modeling_qwen3_5.py)."""

    def __init__(self, weight: ttnn.Tensor, eps: float = 1e-6, scale: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = ttnn.add(weight, 1.0) if scale else weight

    def forward(self, x: ttnn.Tensor):
        output = ttnn.rms_norm(x, epsilon=self.eps)
        output = output * self.weight
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
