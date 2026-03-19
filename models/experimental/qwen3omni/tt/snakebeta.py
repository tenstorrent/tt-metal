import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule

# On Wormhole, several unary ops are wrong for some shapes/sizes (verified vs torch):
# - exp: wrong on [1, C, 1] for moderate C; wrong on 1-D when length > 1024
# - reciprocal: wrong on [1, C, 1] for C like 768/1536; wrong on 1-D when length > 1024
# Fix: apply exp/reciprocal on rank-1 [C] only, chunking when C > 1024, then reshape.
_TTNN_RANK1_UNARY_SAFE_MAX = 1024
_TTNN_RANK1_UNARY_CHUNK = 512


def _ttnn_exp_rank1(t):
    """Element-wise exp on rank-1 tensor; chunked when length > 1024."""
    shape = t.shape
    if len(shape) != 1:
        return ttnn.exp(t)
    n = int(shape[0])
    if n <= _TTNN_RANK1_UNARY_SAFE_MAX:
        return ttnn.exp(t)
    parts = []
    for start in range(0, n, _TTNN_RANK1_UNARY_CHUNK):
        end = min(start + _TTNN_RANK1_UNARY_CHUNK, n)
        chunk = ttnn.slice(t, [start], [end], [1])
        parts.append(ttnn.exp(chunk))
    return ttnn.concat(parts, dim=0) if len(parts) > 1 else parts[0]


def _ttnn_reciprocal_rank1(t):
    """Element-wise reciprocal on rank-1 tensor; chunked when length > 1024."""
    shape = t.shape
    if len(shape) != 1:
        return ttnn.reciprocal(t)
    n = int(shape[0])
    if n <= _TTNN_RANK1_UNARY_SAFE_MAX:
        return ttnn.reciprocal(t)
    parts = []
    for start in range(0, n, _TTNN_RANK1_UNARY_CHUNK):
        end = min(start + _TTNN_RANK1_UNARY_CHUNK, n)
        chunk = ttnn.slice(t, [start], [end], [1])
        parts.append(ttnn.reciprocal(chunk))
    return ttnn.concat(parts, dim=0) if len(parts) > 1 else parts[0]


class TTNNSnakeBeta(TTNNModule):
    def __init__(self, device, alpha, beta, epsilon=1e-9):
        super().__init__()

        self.to_device(device)
        self.epsilon = epsilon

        # Checkpoints store alpha/beta as bf16; PyTorch forward promotes against float32
        # activations. Keeping weights in bf16 on device makes exp/sin/mul too lossy vs
        # the float32 reference — promote to float32 for parity with SnakeBeta + float x.
        if alpha.dtype != torch.float32:
            alpha = alpha.to(torch.float32)
        if beta.dtype != torch.float32:
            beta = beta.to(torch.float32)

        # Store parameters as TT tensors
        self.alpha = ttnn.from_torch(alpha, device=device)
        self.beta = ttnn.from_torch(beta, device=device)

    def forward(self, hidden_states):
        """
        hidden_states: TT tensor of shape [B, C, T]
        """

        # Match PT: exp on the per-channel vector, then [1, C, 1] for broadcast.
        # Do not exp on [1, C, 1] directly — ttnn.exp is wrong for large C in that layout,
        # and exp on long 1-D tensors needs chunking (see _ttnn_exp_rank1).
        c = int(self.alpha.shape[0])
        alpha = ttnn.reshape(_ttnn_exp_rank1(self.alpha), (1, c, 1))
        beta = ttnn.reshape(_ttnn_exp_rank1(self.beta), (1, c, 1))

        # x * alpha
        x_alpha = ttnn.mul(hidden_states, alpha)

        # sin(x * alpha)
        sin_term = ttnn.sin(x_alpha)

        # sin^2
        sin_sq = ttnn.mul(sin_term, sin_term)

        # 1 / (beta + eps) on rank-1 [C] then [1, C, 1] — reciprocal([1, C, 1]) is wrong on WH.
        beta_vec = ttnn.reshape(beta, (c,))
        inv_vec = _ttnn_reciprocal_rank1(ttnn.add(beta_vec, self.epsilon))
        inv_beta = ttnn.reshape(inv_vec, (1, c, 1))

        # final term
        periodic = ttnn.mul(inv_beta, sin_sq)

        # output
        output = ttnn.add(hidden_states, periodic)

        return output
