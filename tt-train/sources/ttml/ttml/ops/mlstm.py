# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""mLSTM (Matrix LSTM) parallel implementation.

This module implements the mLSTM cell from the xLSTM paper (Beck et al., 2024).
The mLSTM uses a matrix memory with covariance update rule, enabling
fully parallelizable computation.

Reference:
    - Paper: "xLSTM: Extended Long Short-Term Memory" (https://arxiv.org/abs/2405.04517)
    - JAX reference: https://github.com/NX-AI/mlstm_kernels
"""

import math
from typing import Tuple

import ttnn

from ..autograd import Function, FunctionContext
from .. import _ttml as cpp


def _create_causal_mask(seq_len: int, device) -> ttnn.Tensor:
    """Create a lower triangular mask of ones.

    Args:
        seq_len: Sequence length S
        device: Device to create tensor on

    Returns:
        Tensor of shape (1, 1, S, S) with lower triangular ones
    """
    shape = (1, 1, seq_len, seq_len)
    ones = cpp.core.ones(shape, device)
    return ttnn.tril(ones, 0)


def mlstm_parallel_forward(
    matQ: ttnn.Tensor,
    matK: ttnn.Tensor,
    matV: ttnn.Tensor,
    vecI: ttnn.Tensor,
    vecF: ttnn.Tensor,
    eps: float = 1e-6,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """mLSTM parallel forward pass.

    Implements the mLSTM forward computation from the xLSTM paper.

    Args:
        matQ: Query tensor of shape (B, NH, S, DHQK)
        matK: Key tensor of shape (B, NH, S, DHQK)
        matV: Value tensor of shape (B, NH, S, DHV)
        vecI: Input gate pre-activation tensor of shape (B, NH, S)
        vecF: Forget gate pre-activation tensor of shape (B, NH, S)
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (matH, vecN, vecM):
            - matH: Output tensor of shape (B, NH, S, DHV)
            - vecN: Normalizer state of shape (B, NH, S)
            - vecM: Max state for stabilization of shape (B, NH, S)
    """
    # Get dimensions from shapes
    shape_q = matQ.shape
    B, NH, S, DHQK = shape_q[0], shape_q[1], shape_q[2], shape_q[3]

    # Step 1: log_sigmoid of forget gate
    # vecLogSigF = log_sigmoid(vecF)  shape: (B, NH, S)
    vecLogSigF = ttnn.log_sigmoid(vecF)

    # Step 2: Cumulative sum along sequence dimension (dim 2 for 3D tensor)
    vecLogSigF_cumsum = ttnn.cumsum(vecLogSigF, dim=2)

    # Step 3: Compute the log forget gate matrix
    # matLogSigF = vecLogSigF_cumsum[:,:,:,None] - vecLogSigF_cumsum[:,:,None,:]
    # Reshape for broadcasting: (B, NH, S, 1) - (B, NH, 1, S) -> (B, NH, S, S)
    cumsum_col = ttnn.reshape(vecLogSigF_cumsum, (B, NH, S, 1))
    cumsum_row = ttnn.reshape(vecLogSigF_cumsum, (B, NH, 1, S))
    matLogSigF = ttnn.subtract(cumsum_col, cumsum_row)

    # Step 4: Apply lower triangular mask (causal mask)
    # matLogSigF_mask = where(ltr, matLogSigF, -inf)
    device = cpp.autograd.AutoContext.get_instance().get_device()
    causal_mask = _create_causal_mask(S, device)
    NEG_INF = -1e9  # Use large negative value for numerical stability
    matLogSigF_masked = ttnn.where(causal_mask, matLogSigF, NEG_INF)

    # Step 5: Add input gate to create log D matrix
    # matLogD = matLogSigF_mask + vecI[:,:,None,:]
    vecI_expanded = ttnn.reshape(vecI, (B, NH, 1, S))
    matLogD = ttnn.add(matLogSigF_masked, vecI_expanded)

    # Step 6: Stabilization - compute row-wise max
    # vecM = max(matLogD, dim=-1, keepdim=True)  shape: (B, NH, S, 1)
    vecM = ttnn.max(matLogD, dim=3, keepdim=True)

    # Step 7: Stabilized D matrix
    # matD = exp(matLogD - vecM)  shape: (B, NH, S, S)
    matLogD_stabilized = ttnn.subtract(matLogD, vecM)
    matD = ttnn.exp(matLogD_stabilized)

    # Step 8: Compute scaled dot product
    # matS = (Q @ K^T) / sqrt(d)  shape: (B, NH, S, S)
    scale = 1.0 / math.sqrt(float(DHQK))
    matS = ttnn.matmul(matQ, ttnn.transpose(matK, -2, -1))
    matS = ttnn.multiply(matS, scale)

    # Step 9: Gated attention
    # matCtilde = matS * matD  shape: (B, NH, S, S)
    matCtilde = ttnn.multiply(matS, matD)

    # Step 10: Compute normalizer
    # vecN = max(|sum(matCtilde, dim=-1)|, exp(-vecM))  shape: (B, NH, S, 1)
    sumCtilde = ttnn.sum(matCtilde, dim=3, keepdim=True)
    absSumCtilde = ttnn.abs(sumCtilde)
    expNegM = ttnn.exp(ttnn.neg(vecM))
    vecN = ttnn.maximum(absSumCtilde, expNegM)

    # Step 11: Normalize
    # matC = matCtilde / (vecN + eps)  shape: (B, NH, S, S)
    vecN_eps = ttnn.add(vecN, eps)
    matC = ttnn.divide(matCtilde, vecN_eps)

    # Step 12: Compute output
    # matH = matC @ V  shape: (B, NH, S, DHV)
    matH = ttnn.matmul(matC, matV)

    # Squeeze vecN and vecM for return (B, NH, S)
    vecN_squeezed = ttnn.reshape(vecN, (B, NH, S))
    vecM_squeezed = ttnn.reshape(vecM, (B, NH, S))

    return matH, vecN_squeezed, vecM_squeezed


def mlstm_parallel_backward(
    matDeltaHtilde: ttnn.Tensor,
    matQ: ttnn.Tensor,
    matK: ttnn.Tensor,
    matV: ttnn.Tensor,
    vecI: ttnn.Tensor,
    vecF: ttnn.Tensor,
    vecN: ttnn.Tensor,
    vecM: ttnn.Tensor,
    eps: float = 1e-6,
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """mLSTM parallel backward pass.

    Computes gradients for the mLSTM operation.

    Args:
        matDeltaHtilde: Gradient of output, shape (B, NH, S, DHV)
        matQ: Query tensor of shape (B, NH, S, DHQK)
        matK: Key tensor of shape (B, NH, S, DHQK)
        matV: Value tensor of shape (B, NH, S, DHV)
        vecI: Input gate pre-activation tensor of shape (B, NH, S)
        vecF: Forget gate pre-activation tensor of shape (B, NH, S)
        vecN: Normalizer state of shape (B, NH, S)
        vecM: Max state of shape (B, NH, S)
        eps: Epsilon for numerical stability

    Returns:
        Tuple of gradients (matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF)
    """
    # Get dimensions
    shape_q = matQ.shape
    B, NH, S, DHQK = shape_q[0], shape_q[1], shape_q[2], shape_q[3]

    # Recompute forward intermediates
    vecLogSigF = ttnn.log_sigmoid(vecF)
    vecLogSigF_cumsum = ttnn.cumsum(vecLogSigF, dim=2)

    cumsum_col = ttnn.reshape(vecLogSigF_cumsum, (B, NH, S, 1))
    cumsum_row = ttnn.reshape(vecLogSigF_cumsum, (B, NH, 1, S))
    matLogSigF = ttnn.subtract(cumsum_col, cumsum_row)

    device = cpp.autograd.AutoContext.get_instance().get_device()
    causal_mask = _create_causal_mask(S, device)
    NEG_INF = -1e9
    matLogSigF_masked = ttnn.where(causal_mask, matLogSigF, NEG_INF)

    vecI_expanded = ttnn.reshape(vecI, (B, NH, 1, S))
    matLogD = ttnn.add(matLogSigF_masked, vecI_expanded)

    vecM_expanded = ttnn.reshape(vecM, (B, NH, S, 1))
    matLogD_stabilized = ttnn.subtract(matLogD, vecM_expanded)
    matD = ttnn.exp(matLogD_stabilized)

    scale = 1.0 / math.sqrt(float(DHQK))
    matS = ttnn.matmul(matQ, ttnn.transpose(matK, -2, -1))
    matS = ttnn.multiply(matS, scale)

    # Intermediate delta-errors
    vecN_expanded = ttnn.reshape(vecN, (B, NH, S, 1))
    vecN_eps = ttnn.add(vecN_expanded, eps)
    matDeltaC = ttnn.divide(
        ttnn.matmul(matDeltaHtilde, ttnn.transpose(matV, -2, -1)), vecN_eps
    )

    matDeltaDtilde = ttnn.multiply(ttnn.multiply(matDeltaC, matD), matS)

    # vecDeltaI = sum(matDeltaDtilde, dim=-2)
    vecDeltaI = ttnn.sum(matDeltaDtilde, dim=2)
    vecDeltaI = ttnn.reshape(vecDeltaI, (B, NH, S))

    # Output delta-errors / gradients
    matP = ttnn.multiply(matDeltaC, matD)

    matDeltaQ = ttnn.multiply(ttnn.matmul(matP, matK), scale)
    matDeltaK = ttnn.multiply(ttnn.matmul(ttnn.transpose(matP, -2, -1), matQ), scale)

    matCtilde = ttnn.multiply(matS, matD)
    matDeltaV = ttnn.matmul(
        ttnn.transpose(matCtilde, -2, -1), ttnn.divide(matDeltaHtilde, vecN_eps)
    )

    # Compute vecDeltaF using the formula from the paper:
    # vecDeltaFbar = rev_cumsum((q*dq - k*dk).sum(-1))
    # vecDeltaF = vecDeltaFbar * sigmoid(-vecF)
    q_times_dq = ttnn.multiply(matQ, matDeltaQ)
    k_times_dk = ttnn.multiply(matK, matDeltaK)
    diff = ttnn.subtract(q_times_dq, k_times_dk)

    # Sum over embedding dimension (dim 3)
    vecDeltaFbar_acc = ttnn.sum(diff, dim=3)
    vecDeltaFbar_acc = ttnn.reshape(vecDeltaFbar_acc, (B, NH, S))

    # Reverse cumsum using cumsum with reverse_order=True
    vecDeltaFbar = ttnn.cumsum(vecDeltaFbar_acc, dim=2, reverse_order=True)

    # vecDeltaF = vecDeltaFbar * sigmoid(-vecF)
    neg_vecF = ttnn.neg(vecF)
    sig_neg_vecF = ttnn.sigmoid(neg_vecF)
    vecDeltaF = ttnn.multiply(vecDeltaFbar, sig_neg_vecF)

    return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF


class MLSTMParallel(Function):
    """mLSTM (Matrix LSTM) parallel operation with autograd support.

    This implements the mLSTM cell from the xLSTM paper using the parallel
    formulation that enables efficient GPU computation.

    Example:
        >>> import ttml
        >>> import numpy as np
        >>>
        >>> # Create inputs
        >>> B, NH, S, D = 1, 2, 32, 64
        >>> q = ttml.autograd.Tensor.from_numpy(np.random.randn(B, NH, S, D).astype(np.float32))
        >>> k = ttml.autograd.Tensor.from_numpy(np.random.randn(B, NH, S, D).astype(np.float32))
        >>> v = ttml.autograd.Tensor.from_numpy(np.random.randn(B, NH, S, D).astype(np.float32))
        >>> i = ttml.autograd.Tensor.from_numpy(np.random.randn(B, NH, S).astype(np.float32))
        >>> f = ttml.autograd.Tensor.from_numpy(np.random.randn(B, NH, S).astype(np.float32))
        >>>
        >>> # Forward pass
        >>> output = MLSTMParallel.apply(q, k, v, i, f)
        >>>
        >>> # Backward pass
        >>> output.backward()
    """

    @staticmethod
    def forward(
        ctx: FunctionContext,
        query,
        key,
        value,
        input_gate,
        forget_gate,
        eps: float = 1e-6,
    ):
        """Forward pass of mLSTM.

        Args:
            ctx: Context for saving tensors for backward
            query: Query tensor of shape (B, NH, S, DHQK)
            key: Key tensor of shape (B, NH, S, DHQK)
            value: Value tensor of shape (B, NH, S, DHV)
            input_gate: Input gate pre-activation of shape (B, NH, S)
            forget_gate: Forget gate pre-activation of shape (B, NH, S)
            eps: Epsilon for numerical stability

        Returns:
            Output tensor of shape (B, NH, S, DHV)
        """
        # Get raw ttnn tensors
        matQ = query.get_value()
        matK = key.get_value()
        matV = value.get_value()
        vecI = input_gate.get_value()
        vecF = forget_gate.get_value()

        # Run forward pass
        matH, vecN, vecM = mlstm_parallel_forward(matQ, matK, matV, vecI, vecF, eps)

        # Save tensors for backward
        ctx.save_for_backward(query, key, value, input_gate, forget_gate)
        ctx.vecN = vecN
        ctx.vecM = vecM
        ctx.eps = eps

        return matH

    @staticmethod
    def backward(ctx: FunctionContext, grad_output):
        """Backward pass of mLSTM.

        Args:
            ctx: Context with saved tensors
            grad_output: Gradient of output tensor (B, NH, S, DHV)

        Returns:
            Tuple of gradients for (query, key, value, input_gate, forget_gate)
        """
        # Retrieve saved tensors
        query, key, value, input_gate, forget_gate = ctx.saved_tensors
        vecN = ctx.vecN
        vecM = ctx.vecM
        eps = ctx.eps

        # Get raw ttnn tensors
        matQ = query.get_value()
        matK = key.get_value()
        matV = value.get_value()
        vecI = input_gate.get_value()
        vecF = forget_gate.get_value()

        # Run backward pass
        matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF = mlstm_parallel_backward(
            grad_output, matQ, matK, matV, vecI, vecF, vecN, vecM, eps
        )

        return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF


def mlstm_parallel(
    query,
    key,
    value,
    input_gate,
    forget_gate,
    eps: float = 1e-6,
):
    """mLSTM (Matrix LSTM) parallel forward pass with autograd support.

    This is the main entry point for the mLSTM operation.

    Args:
        query: Query tensor of shape (B, NH, S, DHQK)
        key: Key tensor of shape (B, NH, S, DHQK)
        value: Value tensor of shape (B, NH, S, DHV)
        input_gate: Input gate pre-activation of shape (B, NH, S)
        forget_gate: Forget gate pre-activation of shape (B, NH, S)
        eps: Epsilon for numerical stability (default: 1e-6)

    Returns:
        Output tensor of shape (B, NH, S, DHV)

    Example:
        >>> import ttml
        >>> from ttml.ops import mlstm_parallel
        >>> import numpy as np
        >>>
        >>> # Initialize
        >>> auto_ctx = ttml.autograd.AutoContext.get_instance()
        >>> auto_ctx.open_device()
        >>>
        >>> # Create inputs
        >>> B, NH, S, D = 1, 2, 32, 64
        >>> q = ttml.autograd.Tensor.from_numpy(np.random.randn(B, NH, S, D).astype(np.float32))
        >>> k = ttml.autograd.Tensor.from_numpy(np.random.randn(B, NH, S, D).astype(np.float32))
        >>> v = ttml.autograd.Tensor.from_numpy(np.random.randn(B, NH, S, D).astype(np.float32))
        >>> i = ttml.autograd.Tensor.from_numpy(np.random.randn(B, NH, S).astype(np.float32))
        >>> f = ttml.autograd.Tensor.from_numpy(np.random.randn(B, NH, S).astype(np.float32))
        >>>
        >>> # Forward + backward
        >>> output = mlstm_parallel(q, k, v, i, f)
        >>> output.backward()
        >>>
        >>> # Check gradients
        >>> print("q gradient initialized:", q.is_grad_initialized())
        >>>
        >>> auto_ctx.close_device()
    """
    return MLSTMParallel.apply(query, key, value, input_gate, forget_gate, eps)
