import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.activation import TTNNGelu
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.experimental.tt_symbiote.modules.tensor import (
    TTNNPermute,
    TTNNAdd,
)


class TTNNConvNeXtBlock(TTNNModule):
    def __init__(self, dim, conv1d_module):
        super().__init__()

        # Depthwise causal conv (your TT implementation)
        self.dwconv = conv1d_module

        # Tensor ops
        self.permute1 = TTNNPermute()
        self.permute2 = TTNNPermute()
        self.add = TTNNAdd()

        # Norm
        self.norm = TTNNLayerNorm.from_torch(torch.nn.LayerNorm(dim, eps=1e-6))

        # Pointwise MLP
        self.pwconv1 = TTNNLinear(dim, 4 * dim)
        self.act = TTNNGelu()
        self.pwconv2 = TTNNLinear(4 * dim, dim)

        # Learnable scale (gamma)
        self.gamma = torch.nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: ttnn.Tensor):
        """
        Input:  [B, C, T]
        Output: [B, C, T]
        """

        identity = x

        # -----------------------------
        # 1. Depthwise Conv1D (TT-native)
        # -----------------------------
        B, C, T = x.shape
        x = self.dwconv(x, batch_size=B, input_length=T)

        # -----------------------------
        # 2. [B, C, T] → [B, T, C]
        # -----------------------------
        x = self.permute1(x, (0, 2, 1))

        # -----------------------------
        # 3. LayerNorm
        # -----------------------------
        x = self.norm(x)

        # -----------------------------
        # 4. Pointwise MLP
        # -----------------------------
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # -----------------------------
        # 5. Gamma scaling (TT-native)
        # -----------------------------
        gamma_tt = ttnn.from_torch(
            self.gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        gamma_tt = ttnn.to_device(gamma_tt, x.device())

        x = ttnn.multiply(x, gamma_tt)

        # -----------------------------
        # 6. Back to [B, C, T]
        # -----------------------------
        x = self.permute2(x, (0, 2, 1))

        # -----------------------------
        # 7. Residual connection
        # -----------------------------
        x = self.add(identity, x)

        return x
