# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Roofline performance model for attention operations."""

from __future__ import annotations
from typing import Optional, List, TYPE_CHECKING
import math

from ..hardware import HardwareSpec, DataType, MathFidelity
from .matmul import matmul_roofline
from .softmax import softmax_roofline

if TYPE_CHECKING:
    from .roofline import RooflineEstimate


def attention_roofline(
    hw: HardwareSpec,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "ScaledDotProductAttention",
    phase: str = "forward",
) -> List["RooflineEstimate"]:
    """
    Performance model for scaled dot-product attention.

    SDPA computes: softmax(Q @ K^T / sqrt(d_k)) @ V

    Forward:
        1. Q @ K^T: matmul [B*H, S, d] x [B*H, d, S] -> [B*H, S, S]
        2. Scale by 1/sqrt(d_k): elementwise
        3. Softmax over last dim
        4. Attn @ V: matmul [B*H, S, S] x [B*H, S, d] -> [B*H, S, d]

    Backward:
        1. grad_V = Attn^T @ grad_output
        2. grad_Attn = grad_output @ V^T
        3. grad_softmax -> grad_scores (softmax backward)
        4. grad_Q = grad_scores @ K
        5. grad_K = grad_scores^T @ Q

    Args:
        hw: Hardware specification
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension per head
        dtype: Data type for operands
        fidelity: Math fidelity level
        num_cores: Number of cores to use (default: all)
        operation: Base name for the operation
        phase: "forward" or "backward"

    Returns:
        List of RooflineEstimate for each sub-operation
    """
    from .roofline import RooflineEstimate
    from .elementwise import elementwise_roofline

    estimates = []
    B = batch_size
    H = num_heads
    S = seq_len
    d = head_dim

    if phase == "forward":
        # Q @ K^T: [B*H, S, d] x [B*H, d, S] -> [B*H, S, S]
        estimates.append(
            matmul_roofline(
                hw,
                B * H * S,
                d,
                S,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.QK",
                phase="forward",
            )
        )

        # Scale by 1/sqrt(d_k) - elementwise
        estimates.append(
            elementwise_roofline(
                hw,
                B * H * S * S,
                num_inputs=1,
                sfpu_ops_per_element=1.0,
                fpu_ops_per_element=0.0,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.scale",
                phase="forward",
            )
        )

        # Softmax
        estimates.append(
            softmax_roofline(
                hw,
                B * H * S,
                S,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.softmax",
                phase="forward",
            )
        )

        # Attn @ V: [B*H, S, S] x [B*H, S, d] -> [B*H, S, d]
        estimates.append(
            matmul_roofline(
                hw,
                B * H * S,
                S,
                d,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.AV",
                phase="forward",
            )
        )
    else:
        # grad_V = Attn^T @ grad_output: [B*H, S, S]^T x [B*H, S, d] -> [B*H, S, d]
        estimates.append(
            matmul_roofline(
                hw,
                B * H * S,
                S,
                d,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.backward.grad_V",
                phase="backward",
            )
        )

        # grad_Attn = grad_output @ V^T: [B*H, S, d] x [B*H, d, S] -> [B*H, S, S]
        estimates.append(
            matmul_roofline(
                hw,
                B * H * S,
                d,
                S,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.backward.grad_Attn",
                phase="backward",
            )
        )

        # Softmax backward
        estimates.append(
            softmax_roofline(
                hw,
                B * H * S,
                S,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.backward.softmax",
                phase="backward",
            )
        )

        # Scale backward - elementwise
        # note: mul uses fpu
        estimates.append(
            elementwise_roofline(
                hw,
                B * H * S * S,
                num_inputs=1,
                sfpu_ops_per_element=0.0,
                fpu_ops_per_element=1.0,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.backward.scale",
                phase="backward",
            )
        )

        # grad_Q = grad_scores @ K: [B*H, S, S] x [B*H, S, d] -> [B*H, S, d]
        estimates.append(
            matmul_roofline(
                hw,
                B * H * S,
                S,
                d,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.backward.grad_Q",
                phase="backward",
            )
        )

        # grad_K = grad_scores^T @ Q: [B*H, S, S]^T x [B*H, S, d] -> [B*H, S, d]
        estimates.append(
            matmul_roofline(
                hw,
                B * H * S,
                S,
                d,
                dtype=dtype,
                fidelity=fidelity,
                num_cores=num_cores,
                operation=f"{operation}.backward.grad_K",
                phase="backward",
            )
        )

    return estimates


def heads_creation_roofline(
    hw: HardwareSpec,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_tensors: int = 3,  # Q, K, V
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "HeadsCreation",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for heads creation (split QKV into heads).

    This is a reshape/transpose operation - pure memory movement.
    Input: [B, 1, S, 3*H*d] -> Output: 3x [B, H, S, d]

    Args:
        hw: Hardware specification
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_tensors: Number of output tensors (3 for QKV)
        dtype: Data type
        fidelity: Math fidelity level
        num_cores: Number of cores to use
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate with performance metrics
    """
    from .roofline import RooflineEstimate

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    bytes_per_elem = dtype.value

    # Total elements: B * S * num_tensors * H * d
    num_elements = batch_size * seq_len * num_tensors * num_heads * head_dim

    # Pure memory movement: read input, write output
    total_bytes = int(2 * num_elements * bytes_per_elem)
    total_flops = 0  # No compute, just reshape/transpose

    ideal_compute_ns = 0
    ideal_memory_ns = total_bytes / hw.dram_bw_gb_s

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=total_flops,
        total_bytes=total_bytes,
        ideal_compute_ns=ideal_compute_ns,
        ideal_memory_ns=ideal_memory_ns,
        hw=hw,
    )


def grouped_heads_creation_roofline(
    hw: HardwareSpec,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_groups: int,
    head_dim: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "GroupedHeadsCreation",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for grouped heads creation (split Q and KV into heads for GQA).

    This is a reshape/transpose operation - pure memory movement.
    For Grouped Query Attention:
    - Q: [B, 1, S, E] -> [B, num_heads, S, head_dim]
    - KV: [B, 1, S, 2*num_groups*head_dim] -> K [B, num_groups, S, head_dim], V [B, num_groups, S, head_dim]

    Args:
        hw: Hardware specification
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads for Q
        num_groups: Number of KV groups (num_heads for MHA, fewer for GQA)
        head_dim: Dimension per head
        dtype: Data type
        fidelity: Math fidelity level
        num_cores: Number of cores to use
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate with performance metrics
    """
    from .roofline import RooflineEstimate

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    bytes_per_elem = dtype.value

    # Total elements: Q has num_heads, K and V have num_groups each
    q_elements = batch_size * seq_len * num_heads * head_dim
    kv_elements = batch_size * seq_len * 2 * num_groups * head_dim
    num_elements = q_elements + kv_elements

    # Pure memory movement: read input, write output
    total_bytes = int(2 * num_elements * bytes_per_elem)
    total_flops = 0  # No compute, just reshape/transpose

    ideal_compute_ns = 0
    ideal_memory_ns = total_bytes / hw.dram_bw_gb_s

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=total_flops,
        total_bytes=total_bytes,
        ideal_compute_ns=ideal_compute_ns,
        ideal_memory_ns=ideal_memory_ns,
        hw=hw,
    )


def heads_fusion_roofline(
    hw: HardwareSpec,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: DataType = DataType.BFLOAT16,
    fidelity: MathFidelity = MathFidelity.HiFi4,
    num_cores: Optional[int] = None,
    operation: str = "HeadsFusion",
    phase: str = "forward",
) -> "RooflineEstimate":
    """
    Performance model for heads fusion (merge heads back).

    This is a reshape/transpose operation - pure memory movement.
    Input: [B, H, S, d] -> Output: [B, 1, S, H*d]

    Args:
        hw: Hardware specification
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dtype: Data type
        fidelity: Math fidelity level
        num_cores: Number of cores to use
        operation: Name for the operation
        phase: "forward" or "backward"

    Returns:
        RooflineEstimate with performance metrics
    """
    from .roofline import RooflineEstimate

    if num_cores is None:
        num_cores = hw.tensix_cores_per_chip

    bytes_per_elem = dtype.value

    # Total elements: B * H * S * d
    num_elements = batch_size * num_heads * seq_len * head_dim

    # Pure memory movement: read input, write output
    total_bytes = int(2 * num_elements * bytes_per_elem)
    total_flops = 0  # No compute, just reshape/transpose

    ideal_compute_ns = 0
    ideal_memory_ns = total_bytes / hw.dram_bw_gb_s

    return RooflineEstimate(
        operation=operation,
        phase=phase,
        total_flops=total_flops,
        total_bytes=total_bytes,
        ideal_compute_ns=ideal_compute_ns,
        ideal_memory_ns=ideal_memory_ns,
        hw=hw,
    )
