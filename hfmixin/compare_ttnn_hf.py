"""Compare HuggingFace-style rotary embeddings with ttnn.experimental.rotary_embedding.
The model used as reference will be Llama in both cases"""

import os
import re
import torch
import torch.nn as nn

import ttnn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig
from models.tt_transformers.tt.rope import RotarySetup2

hidden_size = 512
num_heads = 4
num_kv_heads = 2
rope_theta = 500000.0  # Use the same theta as Meta
head_dim = hidden_size // num_heads
max_seq_len = 64

# Create HF config
hf_config = LlamaConfig(
    hidden_size=hidden_size,
    num_attention_heads=num_heads,
    num_key_value_heads=num_kv_heads,
    rope_theta=rope_theta,
    rope_scaling=None,  # This forces "default" RoPE type (vanilla)
    max_position_embeddings=max_seq_len * 2,
)

# Initialize HF rotary embedding
hf_rotary_emb = LlamaRotaryEmbedding(hf_config)

# Initialize linear projections
wq = q_proj = nn.Linear(hidden_size, num_heads * head_dim)
wk = k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)


def hf_rotate(hidden_states, position_ids):
    """HuggingFace-style RoPE using apply_rotary_pos_emb"""
    # hidden_states (batch_size, seq_len, hidden_size)
    # position_ids (batch_size, seq_len)

    position_embeddings = hf_rotary_emb(hidden_states, position_ids)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = q_proj(hidden_states).view(hidden_shape)  # (batch, seq_len, num_heads, head_dim)
    query_states = query_states.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
    key_states = k_proj(hidden_states).view(hidden_shape)  # (batch, seq_len, num_kv_heads, head_dim)
    key_states = key_states.transpose(1, 2)  # (batch, num_kv_heads, seq_len, head_dim)

    cos, sin = position_embeddings
    # cos.shape: (batch_size, seq_len, head_dim)
    # sin.shape: (batch_size, seq_len, head_dim)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # query_states.shape: (batch_size, num_heads, seq_len, head_dim)
    # key_states.shape: (batch_size, num_kv_heads, seq_len, head_dim)

    return query_states, key_states, position_embeddings


def ttnn_hf_rotate_prefill(hidden_states, position_ids, device):
    """TTNN rotary_embedding for prefill mode (HF-style)"""
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]

    # Project to Q/K
    xq, xk = wq(hidden_states), wk(hidden_states)

    # Reshape: (batch, seq_len, hidden) -> (batch, num_heads, seq_len, head_dim)
    xq = xq.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    xk = xk.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Get HF cos/sin embeddings
    cos, sin = hf_rotary_emb(hidden_states, position_ids)
    # cos, sin shape: (batch_size, seq_len, head_dim)

    # For prefill, typically batch=1, so we use the first batch item
    # Reshape cos/sin to TTNN format: (1, 1, seq_len, head_dim)
    # Take first batch item since HF gives (batch, seq_len, head_dim)
    cos_ttnn = cos[0:1].unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin_ttnn = sin[0:1].unsqueeze(0)  # (1, 1, seq_len, head_dim)

    # Process first batch item only for prefill
    xq = xq[0:1]  # (1, num_heads, seq_len, head_dim)
    xk = xk[0:1]  # (1, num_kv_heads, seq_len, head_dim)

    print("TILING??")
    print("xq.shape", xq.shape)
    # Convert to TTNN tensors
    xq_tt = ttnn.from_torch(xq, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    print("xq_tt.shape", xq_tt.shape)

    xk_tt = ttnn.from_torch(xk, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cos_tt = ttnn.from_torch(cos_ttnn, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    print("cos_tt.shape", cos_tt.shape)

    sin_tt = ttnn.from_torch(sin_ttnn, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Apply rotary embedding (HF-style, no transformation matrix)
    xq_rotated = ttnn.experimental.rotary_embedding(xq_tt, cos_tt, sin_tt)
    xk_rotated = ttnn.experimental.rotary_embedding(xk_tt, cos_tt, sin_tt)
    print("xq_rotated.shape", xq_rotated.shape)

    # Convert back to torch
    xq_out = ttnn.to_torch(xq_rotated)
    xk_out = ttnn.to_torch(xk_rotated)

    return xq_out, xk_out


def ttnn_hf_rotate_decode(hidden_states, position_ids, device):
    """TTNN rotary_embedding for decode mode (HF-style)"""
    batch_size = hidden_states.shape[0]
    seq_len = 1  # Decode mode

    # Project to Q/K
    xq, xk = wq(hidden_states), wk(hidden_states)

    # Reshape: (batch, 1, hidden) -> (batch, num_heads, 1, head_dim)
    xq = xq.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    xk = xk.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Get HF cos/sin embeddings for decode positions
    # position_ids shape: (batch_size, 1) for decode
    cos, sin = hf_rotary_emb(hidden_states, position_ids)
    # cos, sin shape: (batch_size, 1, head_dim)
    cos_cache_tt = ttnn.from_torch(cos, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Process each batch item separately since ttnn.experimental.rotary_embedding
    # expects cos/sin to match the input batch dimension
    # For decode, input shape is (1, batch, num_heads, head_dim) or (batch, num_heads, 1, head_dim)
    # cos/sin should be (1, 1, 1, head_dim) for a single position

    # We'll process batch items one at a time and concatenate
    xq_outputs = []
    xk_outputs = []

    for b in range(batch_size):
        xq_b = xq[b : b + 1]  # (1, num_heads, 1, head_dim)
        xk_b = xk[b : b + 1]  # (1, num_kv_heads, 1, head_dim)
        cos_b = cos[b : b + 1].unsqueeze(0).unsqueeze(0)  # (1, 1, 1, head_dim)
        sin_b = sin[b : b + 1].unsqueeze(0).unsqueeze(0)  # (1, 1, 1, head_dim)

        # Convert to TTNN tensors
        xq_tt = ttnn.from_torch(xq_b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        xk_tt = ttnn.from_torch(xk_b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        cos_tt = ttnn.from_torch(cos_b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        sin_tt = ttnn.from_torch(sin_b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Apply rotary embedding (HF-style, no transformation matrix)
        xq_rotated = ttnn.experimental.rotary_embedding(xq_tt, cos_tt, sin_tt)
        xk_rotated = ttnn.experimental.rotary_embedding(xk_tt, cos_tt, sin_tt)

        # Convert back to torch
        xq_out_b = ttnn.to_torch(xq_rotated)
        xk_out_b = ttnn.to_torch(xk_rotated)

        xq_outputs.append(xq_out_b)
        xk_outputs.append(xk_out_b)

    # Concatenate batch items
    xq_out = torch.cat(xq_outputs, dim=0)  # (batch, num_heads, 1, head_dim)
    xk_out = torch.cat(xk_outputs, dim=0)  # (batch, num_kv_heads, 1, head_dim)

    return xq_out[:, :, :1], xk_out[:, :, :1], (xq_out, xk_out)


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


def test_prefill_random_data(batch_size, seq_len_prefill, hidden_size, max_seq_len, device):
    # Test Prefill Mode
    print("Testing Prefill Mode...")
    position_ids_prefill = torch.LongTensor(batch_size * [list(range(seq_len_prefill))])
    hidden_states_prefill = torch.rand(batch_size, seq_len_prefill, hidden_size)

    # HF reference
    rot_q_hf, rot_k_hf, _ = hf_rotate(hidden_states_prefill, position_ids_prefill)

    # TTNN implementation
    rot_q_ttnn, rot_k_ttnn = ttnn_hf_rotate_prefill(hidden_states_prefill, position_ids_prefill, device)

    print("rot_q_hf.shape", rot_q_hf.shape)
    print("rot_q_ttnn.shape", rot_q_ttnn.shape)
    # Compare (only first batch item for prefill since we process one at a time)
    q_pass, q_pcc = comp_pcc(rot_q_hf[0:1], rot_q_ttnn)
    k_pass, k_pcc = comp_pcc(rot_k_hf[0:1], rot_k_ttnn)

    print(f"Prefill Q PCC: {q_pcc:.6f}, Pass: {q_pass}")
    print(f"Prefill K PCC: {k_pcc:.6f}, Pass: {k_pass}")

    assert q_pass, f"Prefill Q PCC {q_pcc} below threshold"


def test_decode_random_data(batch_size, seq_len_decode, hidden_size, max_seq_len, device):
    """Test Decode Mode"""
    print("\nTesting Decode Mode...")
    # For decode, each batch item can have different positions
    position_ids_decode = torch.LongTensor([[i] for i in range(batch_size)])  # Different positions
    hidden_states_decode = torch.rand(batch_size, seq_len_decode, hidden_size)

    # HF reference
    rot_q_hf_decode, rot_k_hf_decode, _ = hf_rotate(hidden_states_decode, position_ids_decode)

    # TTNN implementation
    rot_q_ttnn_decode, rot_k_ttnn_decode, (raw_rot_q_ttnn_decode, raw_rot_k_ttnn_decode) = ttnn_hf_rotate_decode(
        hidden_states_decode, position_ids_decode, device
    )

    print("rot_q_hf_decode.shape", rot_q_hf_decode.shape)
    print("rot_q_ttnn_decode.shape", rot_q_ttnn_decode.shape)

    print("raw_rot_q_ttnn_decode", raw_rot_q_ttnn_decode.shape)
    print("raw_rot_k_ttnn_decode.shape", raw_rot_k_ttnn_decode.shape)

    # Compare
    q_pass_decode, q_pcc_decode = comp_pcc(rot_q_hf_decode, rot_q_ttnn_decode)
    k_pass_decode, k_pcc_decode = comp_pcc(rot_k_hf_decode, rot_k_ttnn_decode)

    print(f"Decode Q PCC: {q_pcc_decode:.6f}, Pass: {q_pass_decode}")
    print(f"Decode K PCC: {k_pcc_decode:.6f}, Pass: {k_pass_decode}")

    assert q_pass_decode, f"Decode Q PCC {q_pcc_decode} below threshold"
    assert k_pass_decode, f"Decode K PCC {k_pcc_decode} below threshold"


def test_tensors_from_folder(folder_path, device, batch_size=1, is_q=True):
    """Test rotary embedding on tensors loaded from a folder.

    Args:
        folder_path: Path to folder containing tensors named q0.pt, q1.pt, ... or k0.pt, k1.pt, ...
        device: TTNN device
        batch_size: Batch size (default 1)
        is_q: True for Q tensors, False for K tensors

    Returns:
        List of PCC values for each tensor
    """
    prefix = "q" if is_q else "k"
    tensor_type = "Q" if is_q else "K"

    # Get all tensor files matching the pattern
    tensor_files = []
    if os.path.exists(folder_path):
        for filename in sorted(os.listdir(folder_path)):
            match = re.match(rf"{prefix}(\d+)\.pt", filename)
            if match:
                tensor_files.append((int(match.group(1)), os.path.join(folder_path, filename)))

    # Sort by number
    tensor_files.sort(key=lambda x: x[0])

    if not tensor_files:
        raise ValueError(f"No {prefix}*.pt files found in {folder_path}")

    print(f"\nTesting {tensor_type} tensors from {folder_path}")
    print(f"Found {len(tensor_files)} tensors")

    # Setup RotarySetup2 for TTNN
    rope_setup = RotarySetup2(
        device=device,
        batch_size=batch_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rope_scaling=None,
        datatype=ttnn.bfloat16,
    )
    rot_mats = [rope_setup.cos_matrix, rope_setup.sin_matrix]

    pcc_list = []

    for pos_idx, filepath in tensor_files:
        print(f"\nProcessing {os.path.basename(filepath)} (position {pos_idx})...")

        # Load tensor
        tensor = torch.load(filepath)
        print(f"Loaded tensor shape: {tensor.shape}")

        # Reshape tensor to expected format [batch, num_heads, seq_len, head_dim]
        # Handle different possible input formats
        original_shape = tensor.shape
        if len(tensor.shape) == 4:
            batch, dim1, dim2, dim3 = tensor.shape

            # Case 1: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
            if dim2 == num_heads and dim3 == head_dim:
                tensor = tensor.transpose(1, 2)
            # Case 2: [batch, num_heads, seq_len, head_dim] - already correct
            elif dim1 == num_heads and dim3 == head_dim:
                pass
            # Case 3: [batch, seq_len, num_heads, num_heads*head_dim] or [batch, seq_len, num_heads, hidden_size]
            elif dim2 == num_heads:
                # Reshape last dimension: if it's num_heads*head_dim, split it
                if dim3 == num_heads * head_dim:
                    tensor = tensor.view(batch, dim1, num_heads, head_dim).transpose(1, 2)
                elif dim3 == hidden_size:
                    # Project back: hidden_size -> num_heads * head_dim, then reshape
                    # Actually, if it's hidden_size, we need to use the projection
                    # But for now, let's assume it needs reshaping
                    tensor = tensor.view(batch, dim1, num_heads, head_dim).transpose(1, 2)
                else:
                    raise ValueError(f"Cannot infer tensor format from shape {original_shape}")
            # Case 4: [batch, 1, num_heads, something] - common for decode tensors
            elif dim1 == 1 and dim2 == num_heads:
                if dim3 == head_dim:
                    # [batch, 1, num_heads, head_dim] -> [batch, num_heads, 1, head_dim]
                    tensor = tensor.squeeze(1).unsqueeze(2)  # [batch, num_heads, 1, head_dim]
                elif dim3 == num_heads * head_dim or dim3 == hidden_size:
                    # [batch, 1, num_heads, num_heads*head_dim] -> [batch, num_heads, 1, head_dim]
                    # The last dimension (num_heads*head_dim) needs to be interpreted correctly
                    # If it's [batch, 1, num_heads, num_heads*head_dim], we can view as [batch, 1, num_heads*num_heads, head_dim]
                    # Then reshape to [batch, num_heads, 1, head_dim] by taking every num_heads-th element
                    # Actually, simpler: view as [batch, 1, num_heads, num_heads, head_dim], then take diagonal
                    # Or: [batch, 1, num_heads, num_heads*head_dim] -> flatten last two dims -> [batch, 1, num_heads*num_heads*head_dim]
                    # -> view as [batch, num_heads, 1, head_dim] assuming data is interleaved
                    # Let's try a simpler approach: assume the data is [batch, seq_len, num_heads, num_heads*head_dim]
                    # where the last dim contains all heads' data concatenated. We need to split it.
                    # View as [batch, 1, num_heads, num_heads, head_dim] and take first slice of last dim
                    tensor_reshaped = tensor.view(batch, dim1, dim2, num_heads, head_dim)
                    # Now [batch, 1, num_heads, num_heads, head_dim]
                    # Take the "diagonal": [batch, 1, num_heads, head_dim] by taking [:, :, i, i, :] for each i
                    # Actually, let's just take the first slice: [:, :, :, 0, :]
                    tensor = tensor_reshaped[:, :, :, 0, :]  # [batch, 1, num_heads, head_dim]
                    tensor = tensor.squeeze(1).unsqueeze(2)  # [batch, num_heads, 1, head_dim]
                else:
                    raise ValueError(
                        f"Cannot infer tensor format from shape {original_shape}, dim3={dim3}, expected {head_dim} or {num_heads*head_dim} or {hidden_size}"
                    )
            else:
                raise ValueError(f"Cannot infer tensor format from shape {original_shape}")
        else:
            raise ValueError(f"Expected 4D tensor, got shape {tensor.shape}")

        # Ensure tensor is in [batch, num_heads, seq_len, head_dim] format
        if len(tensor.shape) != 4:
            tensor = tensor.unsqueeze(2) if len(tensor.shape) == 3 else tensor

        if tensor.shape[1] != num_heads or tensor.shape[3] != head_dim:
            raise ValueError(
                f"After reshaping, tensor shape is {tensor.shape}, expected [batch, {num_heads}, seq_len, {head_dim}]"
            )

        batch_size_actual = tensor.shape[0]
        seq_len_actual = tensor.shape[2]

        # Create position_ids for HF
        position_ids = torch.LongTensor([[pos_idx] for _ in range(batch_size_actual)])

        # Create dummy hidden_states for HF rotary embedding (we only need cos/sin)
        hidden_states_dummy = torch.rand(batch_size_actual, seq_len_actual, hidden_size)

        # Get HF cos/sin
        cos_hf, sin_hf = hf_rotary_emb(hidden_states_dummy, position_ids)
        # cos_hf, sin_hf shape: (batch_size, seq_len, head_dim)

        # Apply HF rotary embedding directly to Q/K tensors
        # tensor is in [batch, num_heads, seq_len, head_dim] format
        if is_q:
            # For Q: [batch, num_heads, seq_len, head_dim]
            tensor_hf_rotated, _ = apply_rotary_pos_emb(tensor, tensor, cos_hf, sin_hf)
        else:
            # For K: [batch, num_kv_heads, seq_len, head_dim]
            # Note: tensor might have num_heads dimension, but K should have num_kv_heads
            # If tensor has num_heads but we need num_kv_heads, take first num_kv_heads
            if tensor.shape[1] == num_heads:
                tensor_k = tensor[:, :num_kv_heads, :, :]
            else:
                tensor_k = tensor
            _, tensor_hf_rotated = apply_rotary_pos_emb(tensor_k, tensor_k, cos_hf, sin_hf)

        # For TTNN, use the full cos/sin matrices from rot_mats and pass token_index
        cos_cached = rot_mats[0]  # Full cos matrix
        sin_cached = rot_mats[1]  # Full sin matrix
        token_index = pos_idx  # Position index as int (same for the whole batch)

        # For K tensors, we might need to handle num_kv_heads
        tensor_for_ttnn = tensor
        if not is_q and tensor.shape[1] == num_heads:
            tensor_for_ttnn = tensor[:, :num_kv_heads, :, :]

        # Convert tensor to TTNN format
        tensor_ttnn = ttnn.from_torch(tensor_for_ttnn, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Apply TTNN rotary embedding with token_index
        tensor_ttnn_rotated = ttnn.experimental.rotary_embedding(tensor_ttnn, cos_cached, sin_cached, token_index)

        # Convert back to torch
        tensor_ttnn_rotated_torch = ttnn.to_torch(tensor_ttnn_rotated)

        # Compare - ensure shapes match
        if tensor_hf_rotated.shape != tensor_ttnn_rotated_torch.shape:
            print(f"Warning: Shape mismatch - HF: {tensor_hf_rotated.shape}, TTNN: {tensor_ttnn_rotated_torch.shape}")
            # Try to reshape if possible
            if tensor_hf_rotated.numel() == tensor_ttnn_rotated_torch.numel():
                tensor_ttnn_rotated_torch = tensor_ttnn_rotated_torch.reshape(tensor_hf_rotated.shape)

        pass_check, pcc_value = comp_pcc(tensor_hf_rotated, tensor_ttnn_rotated_torch)

        print(f"Position {pos_idx} {tensor_type} PCC: {pcc_value:.6f}, Pass: {pass_check}")

        pcc_list.append(pass_check)

    return pcc_list


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
        test_prefill_random_data(batch_size, seq_len_prefill, hidden_size, max_seq_len, device)
        # Test Decode Mode
        test_decode_random_data(batch_size, seq_len_decode, hidden_size, max_seq_len, device)

        # Test tensors from folders
        print("\n" + "=" * 60)
        pcc_list_k = test_tensors_from_folder("temp/k", device, batch_size=batch_size, is_q=False)
        print("\n" + "=" * 60)
        pcc_list_q = test_tensors_from_folder("temp/q", device, batch_size=batch_size, is_q=True)

        # Assert all PCCs pass
        assert all(pcc_list_k), f"Some K tensor PCCs failed: {pcc_list_k}"
        assert all(pcc_list_q), f"Some Q tensor PCCs failed: {pcc_list_q}"

        print("\nAll tests PASSED!!")

    finally:
        ttnn.close_device(device)
