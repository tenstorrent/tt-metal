"""Compare Meta-style rotary embeddings (mllama_model) with ttnn.experimental.rotary_embedding_llama.
The model used as reference will be Llama in both cases"""

import torch
import torch.nn as nn

import ttnn
from models.tt_transformers.tt.common import get_rot_transformation_mat, gather_cos_sin
from models.tt_transformers.tt.rope import RotaryEmbedding
import mllama_model

hidden_size = 512
num_heads = 4
num_kv_heads = 2
rope_theta = 500000.0  # Use the same theta as Meta
head_dim = hidden_size // num_heads
max_seq_len = 32

# Initialize linear projections
wq = q_proj = nn.Linear(hidden_size, num_heads * head_dim)
wk = k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)

# Precompute Meta-style frequencies
m_freq_cis = mllama_model.precompute_freqs_cis(dim=head_dim, end=max_seq_len * 2, theta=rope_theta)


def mllama_rotate(hidden_states, position_ids):
    """Meta-style RoPE using mllama_model.apply_rotary_emb"""
    start_pos = int(position_ids[0, 0])
    seq_len = len(position_ids[0])
    freq_cis = m_freq_cis[start_pos : start_pos + seq_len]

    batch_size = hidden_states.shape[0]
    xq, xk = wq(hidden_states), wk(hidden_states)
    print("xq.shape: ", xq.shape)
    xq = xq.view(batch_size, seq_len, num_heads, head_dim)
    xk = xk.view(batch_size, seq_len, num_kv_heads, head_dim)

    xq, xk = mllama_model.apply_rotary_emb(xq, xk, freq_cis)

    return xq, xk, freq_cis


def ttnn_llama_rotate_prefill(hidden_states, position_ids, device):
    """TTNN rotary_embedding_llama for prefill mode"""
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]

    # Project to Q/K
    xq, xk = wq(hidden_states), wk(hidden_states)

    # Reshape: (batch, seq_len, hidden) -> (batch, num_heads, seq_len, head_dim)
    xq = xq.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    xk = xk.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Generate cos/sin using RotaryEmbedding (which handles permute_to_meta_format)
    rotary_emb = RotaryEmbedding(dim=head_dim, max_position_embeddings=max_seq_len * 2, base=rope_theta, device=None)

    # Get cos/sin for the sequence positions
    start_pos = int(position_ids[0, 0])
    position_range = torch.arange(start_pos, start_pos + seq_len)
    cos_full = rotary_emb.cos_cached  # Get the buffer as tensor
    sin_full = rotary_emb.sin_cached  # Get the buffer as tensor
    cos = cos_full[:, :, position_range, :]  # (1, 1, seq_len, head_dim)
    sin = sin_full[:, :, position_range, :]  # (1, 1, seq_len, head_dim)

    # Create transformation matrix
    trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)

    # Convert to TTNN tensors
    xq_tt = ttnn.from_torch(xq, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    xk_tt = ttnn.from_torch(xk, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cos_tt = ttnn.from_torch(cos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    sin_tt = ttnn.from_torch(sin, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    trans_mat_tt = ttnn.from_torch(trans_mat, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Apply rotary embedding
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
    xq_out = ttnn.to_torch(xq_rotated)
    xk_out = ttnn.to_torch(xk_rotated)

    return xq_out, xk_out


def ttnn_llama_rotate_decode(hidden_states, position_ids, device):
    """TTNN rotary_embedding_llama for decode mode"""
    batch_size = hidden_states.shape[0]
    seq_len = 1  # Decode mode

    # Project to Q/K
    xq, xk = wq(hidden_states), wk(hidden_states)

    # Reshape: (batch, 1, hidden) -> (1, batch, num_heads, head_dim) for decode
    xq = xq.view(batch_size, seq_len, num_heads, head_dim).transpose(0, 1)
    xk = xk.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(0, 1)

    # Use RotarySetup for decode mode (handles position-specific cos/sin)
    from models.tt_transformers.tt.rope import RotarySetup

    rope_setup = RotarySetup(
        device=device,
        batch_size=batch_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len * 2,
        rope_theta=rope_theta,
        rope_scaling=None,
        datatype=ttnn.bfloat16,
    )

    # Get position IDs as tensor
    position_ids_tensor = position_ids[:, 0]  # (batch_size,)
    print("position_ids_tensor.shape", position_ids_tensor.shape)
    cos, sin = rope_setup.get_rot_mats(position_ids_tensor)
    print("cos.shape", cos.shape)
    trans_mat = rope_setup.transformation_mat
    print("trans_mat.shape", trans_mat.shape)

    # Convert inputs to TTNN tensors with proper memory config
    grid = ttnn.num_cores_to_corerangeset(batch_size, rope_setup.core_grid, row_wise=True)
    input_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    xq_tt = ttnn.from_torch(
        xq, device=device, dtype=ttnn.bfloat16, memory_config=input_mem_config, layout=ttnn.TILE_LAYOUT
    )
    xk_tt = ttnn.from_torch(
        xk, device=device, dtype=ttnn.bfloat16, memory_config=input_mem_config, layout=ttnn.TILE_LAYOUT
    )

    # Apply rotary embedding
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=(True if head_dim <= 128 else False),
        packer_l1_acc=True,
    )
    print("xq_tt.shape", xq_tt.shape)
    xq_rotated = ttnn.experimental.rotary_embedding_llama(
        xq_tt, cos, sin, trans_mat, is_decode_mode=True, compute_kernel_config=compute_kernel_config
    )
    print("xk_tt.shape", xk_tt.shape)

    xk_rotated = ttnn.experimental.rotary_embedding_llama(
        xk_tt, cos, sin, trans_mat, is_decode_mode=True, compute_kernel_config=compute_kernel_config
    )

    # Convert back to torch
    xq_out = ttnn.to_torch(xq_rotated)
    xk_out = ttnn.to_torch(xk_rotated)

    # Reshape back: (1, batch, num_heads, head_dim) -> (batch, num_heads, 1, head_dim)
    xq_out = xq_out.transpose(0, 1).transpose(1, 2)
    xk_out = xk_out.transpose(0, 1).transpose(1, 2)

    return xq_out, xk_out


def comp_pcc(torch_tensor, tt_tensor, pcc_threshold=0.9997):
    """Compute Pearson Correlation Coefficient"""
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
    seq_len_prefill = 16
    seq_len_decode = 1

    assert seq_len_prefill <= max_seq_len

    # Initialize device
    device_id = 1
    device = ttnn.open_device(device_id=device_id)

    try:
        # Test Prefill Mode
        print("Testing Prefill Mode...")
        position_ids_prefill = torch.LongTensor(batch_size * [list(range(seq_len_prefill))])
        hidden_states_prefill = torch.rand(batch_size, seq_len_prefill, hidden_size)

        # Meta reference
        rot_xq_meta, rot_xk_meta, freq_cis = mllama_rotate(hidden_states_prefill, position_ids_prefill)
        rot_xq_meta = rot_xq_meta.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        rot_xk_meta = rot_xk_meta.transpose(1, 2)  # (batch, num_kv_heads, seq_len, head_dim)

        # TTNN implementation
        rot_xq_ttnn, rot_xk_ttnn = ttnn_llama_rotate_prefill(hidden_states_prefill, position_ids_prefill, device)

        print("rot_xq_meta.shape", rot_xq_meta.shape)
        print("rot_xq_ttnn.shape", rot_xq_ttnn.shape)
        # Compare
        q_pass, q_pcc = comp_pcc(rot_xq_meta, rot_xq_ttnn)
        k_pass, k_pcc = comp_pcc(rot_xk_meta, rot_xk_ttnn)

        print(f"Prefill Q PCC: {q_pcc:.6f}, Pass: {q_pass}")
        print(f"Prefill K PCC: {k_pcc:.6f}, Pass: {k_pass}")

        assert q_pass, f"Prefill Q PCC {q_pcc} below threshold"
        assert k_pass, f"Prefill K PCC {k_pcc} below threshold"

        # Test Decode Mode
        print("\nTesting Decode Mode...")
        position_ids_decode = torch.LongTensor([list[int](range(batch_size))])  # Different positions for each batch
        position_ids_decode = torch.LongTensor(batch_size * [list(range(seq_len_decode))])

        hidden_states_decode = torch.rand(batch_size, seq_len_decode, hidden_size)

        # Meta reference
        rot_xq_meta_decode, rot_xk_meta_decode, _ = mllama_rotate(hidden_states_decode, position_ids_decode)
        rot_xq_meta_decode = rot_xq_meta_decode.transpose(1, 2)  # (batch, num_heads, 1, head_dim)
        rot_xk_meta_decode = rot_xk_meta_decode.transpose(1, 2)  # (batch, num_kv_heads, 1, head_dim)

        # TTNN implementation
        rot_xq_ttnn_decode, rot_xk_ttnn_decode = ttnn_llama_rotate_decode(
            hidden_states_decode, position_ids_decode, device
        )

        print("rot_xk_meta_decode.shape", rot_xk_meta_decode.shape)
        print("rot_xq_ttnn_decode.shape", rot_xq_ttnn_decode.shape)

        # Compare
        q_pass_decode, q_pcc_decode = comp_pcc(rot_xq_meta_decode, rot_xq_ttnn_decode)
        k_pass_decode, k_pcc_decode = comp_pcc(rot_xk_meta_decode, rot_xk_ttnn_decode)

        print(f"Decode Q PCC: {q_pcc_decode:.6f}, Pass: {q_pass_decode}")
        print(f"Decode K PCC: {k_pcc_decode:.6f}, Pass: {k_pass_decode}")

        assert q_pass_decode, f"Decode Q PCC {q_pcc_decode} below threshold"
        assert k_pass_decode, f"Decode K PCC {k_pcc_decode} below threshold"

        print("\nAll tests PASSED!!")

    finally:
        ttnn.close_device(device)
