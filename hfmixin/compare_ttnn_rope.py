"""Compare TTNN rotary embedding implementations: HF-style vs Llama-style.

This script demonstrates that ttnn.experimental.rotary_embedding (HF-style) and
ttnn.experimental.rotary_embedding_llama (Llama-style) are mathematically equivalent
when element reordering is properly handled.

Key insight: The two implementations rotate different element pairs:
- HF-style: rotates (first_half, second_half) pairs
- Llama-style: rotates (even_idx, odd_idx) pairs

To achieve equivalence, we reorder elements before/after Llama-style rotation.
"""

import torch
import torch.nn as nn

import ttnn
from models.tt_transformers.tt.common import get_rot_transformation_mat

# Shared configuration
hidden_size = 512
num_heads = 4
num_kv_heads = 2
head_dim = hidden_size // num_heads  # 128
rope_theta = 500000.0
max_seq_len = 64

# Shared projection layers - SAME weights used by both paths
wq = nn.Linear(hidden_size, num_heads * head_dim)
wk = nn.Linear(hidden_size, num_kv_heads * head_dim)


def pre_meta_to_hf(x: torch.Tensor) -> torch.Tensor:
    """Reorder elements BEFORE Llama rotation so it rotates same pairs as HF.

    This function interleaves the first and second halves of the last dimension.

    Example:
        Input:  [x1, x2, x3, x4]  (original order)
        Output: [x1, x3, x2, x4]  (interleave first/second halves)

    Now when Llama-style rotates adjacent pairs (x1,x3) and (x2,x4), it matches
    HF-style behavior of rotating (first_half, second_half) pairs.

    Args:
        x: Tensor with shape (..., head_dim) where head_dim is even

    Returns:
        Reordered tensor with same shape, where elements are interleaved
    """
    dim = x.shape[-1]
    half_dim = dim // 2
    x_shape = x.shape
    # Interleave first half with second half: [x1...x_half, x_half+1...x_dim] -> [x1, x_half+1, x2, x_half+2, ...]
    return torch.cat(
        [x[..., :half_dim, None], x[..., half_dim:, None]], dim=-1  # [..., half_dim, 1]  # [..., half_dim, 1]
    ).view(x_shape)


def post_hf_to_meta(x: torch.Tensor) -> torch.Tensor:
    """Reorder elements AFTER Llama rotation back to original order.

    This function de-interleaves elements that were interleaved by pre_meta_to_hf.

    Example:
        Input:  [y1, y3, y2, y4]  (interleaved from rotation)
        Output: [y1, y2, y3, y4]  (de-interleave back to original)

    Args:
        x: Tensor with shape (batch_size, num_heads, seq_len, head_dim) where head_dim is even

    Returns:
        Reordered tensor with same shape, de-interleaved to original order
    """
    x_shape = x.shape  # (batch_size, num_heads, seq_len, head_dim)
    # De-interleave: [x1, x_half+1, x2, x_half+2, ...] -> [x1...x_half, x_half+1...x_dim]
    # View creates 5D tensor: (batch, num_heads, seq_len, head_dim//2, 2)
    # Transpose swaps last two dims: (batch, num_heads, seq_len, 2, head_dim//2)
    # Reshape back to original: (batch, num_heads, seq_len, head_dim)
    return x.view((*x_shape[:-1], -1, 2)).transpose(-2, -1).reshape(x_shape)


def generate_unified_cos_sin(head_dim: int, seq_len: int, theta: float):
    """Generate cos/sin in both HF and Meta formats from same base frequencies.

    This ensures mathematical consistency - both formats use identical frequency values,
    just arranged differently.

    Args:
        head_dim: Dimension of each attention head
        seq_len: Sequence length
        theta: Base frequency parameter (rope_theta)

    Returns:
        Tuple of ((cos_hf, sin_hf), (cos_meta, sin_meta))
        - cos_hf, sin_hf: HF format [1, 1, seq_len, head_dim] with pattern [c1, c2, ..., c1, c2, ...]
        - cos_meta, sin_meta: Meta format [1, 1, seq_len, head_dim] with pattern [c1, c1, c2, c2, ...]
    """
    # Base frequencies - single source of truth
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim/2]

    # HF format: [c1, c2, c3, ..., c1, c2, c3, ...] (concat freqs with itself)
    emb_hf = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
    cos_hf = emb_hf.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin_hf = emb_hf.sin().unsqueeze(0).unsqueeze(0)

    # Meta format: [c1, c1, c2, c2, c3, c3, ...] (interleave each freq with itself)
    cos_meta = torch.stack([freqs.cos(), freqs.cos()], dim=-1).flatten(-2)
    sin_meta = torch.stack([freqs.sin(), freqs.sin()], dim=-1).flatten(-2)
    cos_meta = cos_meta.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin_meta = sin_meta.unsqueeze(0).unsqueeze(0)

    return (cos_hf, sin_hf), (cos_meta, sin_meta)


def ttnn_hf_rotate_prefill(hidden_states, position_ids, device):
    """HF-style RoPE using ttnn.experimental.rotary_embedding (no element reordering).

    This path applies rotation directly without any element reordering.

    Args:
        hidden_states: (batch_size, seq_len, hidden_size)
        position_ids: (batch_size, seq_len)
        device: TTNN device

    Returns:
        Tuple of (q_rotated, k_rotated) tensors in torch format
    """
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]

    # Project to Q/K using shared weights
    xq, xk = wq(hidden_states), wk(hidden_states)

    # Reshape: (batch, seq_len, hidden) -> (batch, num_heads, seq_len, head_dim)
    xq = xq.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    xk = xk.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Generate HF-format cos/sin
    (cos_hf, sin_hf), _ = generate_unified_cos_sin(head_dim, seq_len, rope_theta)

    # For prefill, process first batch item (can extend to full batch if needed)
    xq_batch = xq[0:1]  # (1, num_heads, seq_len, head_dim)
    xk_batch = xk[0:1]  # (1, num_kv_heads, seq_len, head_dim)

    # Convert to TTNN tensors
    xq_tt = ttnn.from_torch(xq_batch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    xk_tt = ttnn.from_torch(xk_batch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cos_tt = ttnn.from_torch(cos_hf, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin_hf, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Apply HF-style rotary embedding (no transformation matrix)
    xq_rotated = ttnn.experimental.rotary_embedding(xq_tt, cos_tt, sin_tt)
    xk_rotated = ttnn.experimental.rotary_embedding(xk_tt, cos_tt, sin_tt)

    # Convert back to torch
    q_out = ttnn.to_torch(xq_rotated)
    k_out = ttnn.to_torch(xk_rotated)

    return q_out, k_out


def ttnn_hf_rotate_decode(hidden_states, position_ids, device):
    """HF-style RoPE for decode mode using ttnn.experimental.rotary_embedding.

    Args:
        hidden_states: (batch_size, 1, hidden_size)
        position_ids: (batch_size, 1)
        device: TTNN device

    Returns:
        Tuple of (q_rotated, k_rotated) tensors in torch format
    """
    batch_size = hidden_states.shape[0]
    seq_len = 1  # Decode mode

    # Project to Q/K using shared weights
    xq, xk = wq(hidden_states), wk(hidden_states)

    # Reshape: (batch, 1, hidden) -> (batch, num_heads, 1, head_dim)
    xq = xq.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    xk = xk.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Generate HF-format cos/sin for decode positions
    (cos_hf, sin_hf), _ = generate_unified_cos_sin(head_dim, max_seq_len, rope_theta)

    # Process each batch item separately (decode mode supports different positions per batch)
    xq_outputs = []
    xk_outputs = []

    for b in range(batch_size):
        pos = int(position_ids[b, 0])
        xq_b = xq[b : b + 1]  # (1, num_heads, 1, head_dim)
        xk_b = xk[b : b + 1]  # (1, num_kv_heads, 1, head_dim)
        cos_b = cos_hf[:, :, pos : pos + 1, :]  # (1, 1, 1, head_dim)
        sin_b = sin_hf[:, :, pos : pos + 1, :]  # (1, 1, 1, head_dim)

        # Convert to TTNN tensors
        xq_tt = ttnn.from_torch(xq_b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        xk_tt = ttnn.from_torch(xk_b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        cos_tt = ttnn.from_torch(cos_b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        sin_tt = ttnn.from_torch(sin_b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Apply HF-style rotary embedding
        xq_rotated = ttnn.experimental.rotary_embedding(xq_tt, cos_tt, sin_tt)
        xk_rotated = ttnn.experimental.rotary_embedding(xk_tt, cos_tt, sin_tt)

        # Convert back to torch
        xq_out_b = ttnn.to_torch(xq_rotated)
        xk_out_b = ttnn.to_torch(xk_rotated)

        xq_outputs.append(xq_out_b[:, :, :1])
        xk_outputs.append(xk_out_b[:, :, :1])

    # Concatenate batch items
    q_out = torch.cat(xq_outputs, dim=0)  # (batch, num_heads, 1, head_dim)
    k_out = torch.cat(xk_outputs, dim=0)  # (batch, num_kv_heads, 1, head_dim)

    return q_out, k_out


def ttnn_llama_rotate_prefill(hidden_states, position_ids, device):
    """Llama-style RoPE using ttnn.experimental.rotary_embedding_llama WITH element reordering.

    CRITICAL: This function applies element reordering before and after rotation to ensure
    mathematical equivalence with HF-style rotation.

    Args:
        hidden_states: (batch_size, seq_len, hidden_size)
        position_ids: (batch_size, seq_len)
        device: TTNN device

    Returns:
        Tuple of (q_rotated, k_rotated) tensors in torch format, equivalent to HF-style
    """
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]

    # Project to Q/K using shared weights
    xq, xk = wq(hidden_states), wk(hidden_states)

    # Reshape: (batch, seq_len, hidden) -> (batch, num_heads, seq_len, head_dim)
    xq = xq.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    xk = xk.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # CRITICAL: Reorder BEFORE rotation so Llama rotates same pairs as HF
    xq_reordered = pre_meta_to_hf(xq)
    xk_reordered = pre_meta_to_hf(xk)

    # Generate Meta-format cos/sin
    _, (cos_meta, sin_meta) = generate_unified_cos_sin(head_dim, seq_len, rope_theta)

    # Get cos/sin for the sequence positions
    start_pos = int(position_ids[0, 0])
    position_range = torch.arange(start_pos, start_pos + seq_len)
    cos = cos_meta[:, :, position_range, :]  # (1, 1, seq_len, head_dim)
    sin = sin_meta[:, :, position_range, :]  # (1, 1, seq_len, head_dim)

    # Create transformation matrix (required for Llama-style)
    trans_mat = get_rot_transformation_mat(dhead=head_dim)

    # For prefill, process first batch item
    xq_batch = xq_reordered[0:1]  # (1, num_heads, seq_len, head_dim)
    xk_batch = xk_reordered[0:1]  # (1, num_kv_heads, seq_len, head_dim)

    # Convert to TTNN tensors
    xq_tt = ttnn.from_torch(xq_batch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    xk_tt = ttnn.from_torch(xk_batch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cos_tt = ttnn.from_torch(cos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    trans_mat_tt = ttnn.from_torch(trans_mat, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Apply Llama-style rotary embedding
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=(True if head_dim <= 128 else False),
        packer_l1_acc=True,
    )

    xq_rotated = ttnn.experimental.rotary_embedding_llama(
        xq_tt, cos_tt, sin_tt, trans_mat_tt, is_decode_mode=False, compute_kernel_config=compute_kernel_config
    )
    xk_rotated = ttnn.experimental.rotary_embedding_llama(
        xk_tt, cos_tt, sin_tt, trans_mat_tt, is_decode_mode=False, compute_kernel_config=compute_kernel_config
    )

    # Convert back to torch
    q_rot = ttnn.to_torch(xq_rotated)
    k_rot = ttnn.to_torch(xk_rotated)

    # CRITICAL: Reorder AFTER rotation back to original order
    q_rot = post_hf_to_meta(q_rot)
    k_rot = post_hf_to_meta(k_rot)

    return q_rot, k_rot


def ttnn_llama_rotate_decode(hidden_states, position_ids, device):
    """Llama-style RoPE for decode mode WITH element reordering.

    For decode mode, rotary_embedding_llama requires:
    - Input shape: (1, batch, num_heads, head_dim)
    - HEIGHT_SHARDED memory layout
    - cos/sin from RotarySetup.get_rot_mats()

    Args:
        hidden_states: (batch_size, 1, hidden_size)
        position_ids: (batch_size, 1)
        device: TTNN device

    Returns:
        Tuple of (q_rotated, k_rotated) tensors in torch format
    """
    from models.tt_transformers.tt.rope import RotarySetup

    batch_size = hidden_states.shape[0]
    seq_len = 1  # Decode mode

    # Project to Q/K using shared weights
    xq, xk = wq(hidden_states), wk(hidden_states)

    # Reshape: (batch, 1, hidden) -> (1, batch, num_heads, head_dim) for decode
    xq = xq.view(batch_size, seq_len, num_heads, head_dim).transpose(0, 1)
    xk = xk.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 1)

    # CRITICAL: Reorder BEFORE rotation
    xq_reordered = pre_meta_to_hf(xq)
    xk_reordered = pre_meta_to_hf(xk)

    # Use RotarySetup for decode mode (handles position-specific cos/sin)
    rope_setup = RotarySetup(
        device=device,
        batch_size=batch_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len * 2,
        rope_theta=rope_theta,
        rope_scaling=None,
        datatype=ttnn.bfloat16,
    )

    # Get position IDs as tensor and get cos/sin matrices
    position_ids_tensor = position_ids[:, 0]  # (batch_size,)
    cos, sin = rope_setup.get_rot_mats(position_ids_tensor)
    trans_mat = rope_setup.transformation_mat

    # Convert inputs to TTNN tensors with HEIGHT_SHARDED memory config
    grid = ttnn.num_cores_to_corerangeset(batch_size, rope_setup.core_grid, row_wise=True)
    input_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    xq_tt = ttnn.from_torch(
        xq_reordered, device=device, dtype=ttnn.bfloat16, memory_config=input_mem_config, layout=ttnn.TILE_LAYOUT
    )
    xk_tt = ttnn.from_torch(
        xk_reordered, device=device, dtype=ttnn.bfloat16, memory_config=input_mem_config, layout=ttnn.TILE_LAYOUT
    )

    # Apply rotary embedding
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=(True if head_dim <= 128 else False),
        packer_l1_acc=True,
    )

    xq_rotated = ttnn.experimental.rotary_embedding_llama(
        xq_tt, cos, sin, trans_mat, is_decode_mode=True, compute_kernel_config=compute_kernel_config
    )
    xk_rotated = ttnn.experimental.rotary_embedding_llama(
        xk_tt, cos, sin, trans_mat, is_decode_mode=True, compute_kernel_config=compute_kernel_config
    )

    # Convert back to torch
    xq_out = ttnn.to_torch(xq_rotated)
    xk_out = ttnn.to_torch(xk_rotated)

    # Reshape back: (1, batch, num_heads, head_dim) -> (batch, num_heads, 1, head_dim)
    xq_out = xq_out.transpose(0, 1).transpose(1, 2)
    xk_out = xk_out.transpose(0, 1).transpose(1, 2)

    # CRITICAL: Reorder AFTER rotation back to original order
    q_out = post_hf_to_meta(xq_out)
    k_out = post_hf_to_meta(xk_out)

    return q_out, k_out


def comp_pcc(torch_tensor, tt_tensor, pcc_threshold=0.9997):
    """Compute Pearson Correlation Coefficient between two tensors.

    Args:
        torch_tensor: Reference tensor
        tt_tensor: Comparison tensor
        pcc_threshold: Minimum PCC value to consider as passing

    Returns:
        Tuple of (pass_bool, pcc_value)
    """
    torch_flat = torch_tensor.flatten().float()
    tt_flat = tt_tensor.flatten().float()

    # Compute PCC
    mean_torch = torch_flat.mean()
    mean_tt = tt_flat.mean()

    centered_torch = torch_flat - mean_torch
    centered_tt = tt_flat - mean_tt

    numerator = (centered_torch * centered_tt).sum()
    denominator = torch.sqrt((centered_torch**2).sum() * (centered_tt**2).sum())

    if denominator == 0:
        return False, 0.0

    pcc_value = numerator / denominator
    return pcc_value.item() >= pcc_threshold, pcc_value.item()


if __name__ == "__main__":
    # Test parameters
    batch_size = 4
    seq_len_prefill = 32
    seq_len_decode = 1

    assert seq_len_prefill <= max_seq_len

    # Initialize device
    device_id = 1
    device = ttnn.open_device(device_id=device_id)

    try:
        # Test Prefill Mode
        print("=" * 80)
        print("Testing Prefill Mode...")
        print("=" * 80)

        position_ids_prefill = torch.LongTensor(batch_size * [list(range(seq_len_prefill))])
        hidden_states_prefill = torch.rand(batch_size, seq_len_prefill, hidden_size)

        # HF-style reference
        q_hf_prefill, k_hf_prefill = ttnn_hf_rotate_prefill(hidden_states_prefill, position_ids_prefill, device)

        # Llama-style with reordering
        q_llama_prefill, k_llama_prefill = ttnn_llama_rotate_prefill(
            hidden_states_prefill, position_ids_prefill, device
        )

        print(f"q_hf_prefill.shape: {q_hf_prefill.shape}")
        print(f"q_llama_prefill.shape: {q_llama_prefill.shape}")

        # Compare (only first batch item for prefill since we process one at a time)
        q_pass, q_pcc = comp_pcc(q_hf_prefill, q_llama_prefill)
        k_pass, k_pcc = comp_pcc(k_hf_prefill, k_llama_prefill)

        print(f"Prefill Q PCC: {q_pcc:.6f}, Pass: {q_pass}")
        print(f"Prefill K PCC: {k_pcc:.6f}, Pass: {k_pass}")

        assert q_pass, f"Prefill Q PCC {q_pcc} below threshold 0.9997"
        assert k_pass, f"Prefill K PCC {k_pcc} below threshold 0.9997"

        print("✓ Prefill mode PASSED")

        # Test Decode Mode
        print("\n" + "=" * 80)
        print("Testing Decode Mode...")
        print("=" * 80)

        # For decode, each batch item can have different positions
        position_ids_decode = torch.LongTensor([[i] for i in range(batch_size)])
        hidden_states_decode = torch.rand(batch_size, seq_len_decode, hidden_size)

        # HF-style reference
        q_hf_decode, k_hf_decode = ttnn_hf_rotate_decode(hidden_states_decode, position_ids_decode, device)

        # Llama-style with reordering
        q_llama_decode, k_llama_decode = ttnn_llama_rotate_decode(hidden_states_decode, position_ids_decode, device)

        print(f"q_hf_decode.shape: {q_hf_decode.shape}")
        print(f"q_llama_decode.shape: {q_llama_decode.shape}")

        # Compare
        q_pass_decode, q_pcc_decode = comp_pcc(q_hf_decode, q_llama_decode)
        k_pass_decode, k_pcc_decode = comp_pcc(k_hf_decode, k_llama_decode)

        print(f"Decode Q PCC: {q_pcc_decode:.6f}, Pass: {q_pass_decode}")
        print(f"Decode K PCC: {k_pcc_decode:.6f}, Pass: {k_pass_decode}")

        assert q_pass_decode, f"Decode Q PCC {q_pcc_decode} below threshold 0.9997"
        assert k_pass_decode, f"Decode K PCC {k_pcc_decode} below threshold 0.9997"

        print("✓ Decode mode PASSED")

        print("\n" + "=" * 80)
        print("All tests PASSED!")
        print("=" * 80)
        print("\nConclusion: Both TTNN rotary embedding implementations are")
        print("mathematically equivalent when element reordering is properly handled.")
        print("\nImplications for attention.py:")
        print("  - ttnn.experimental.rotary_embedding can replace rotary_embedding_llama")
        print("  - Requires different cos/sin format (HF vs Meta)")
        print("  - No transformation matrix needed for HF-style")
        print("  - No element reordering needed if model uses HF ordering natively")

    finally:
        ttnn.close_device(device)
