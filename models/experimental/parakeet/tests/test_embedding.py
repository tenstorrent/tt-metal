import torch
from models.common.metrics import compute_pcc, compute_max_abs_error, compute_mean_abs_error
import ttnn


# -----------------------------
# Config
# -----------------------------
vocab_size = 1024
embedding_dim = 512
batch_size = 4
seq_len = 16

device = ttnn.open_device(device_id=0)


# -----------------------------
# Create identical weights
# -----------------------------
torch_embedding = torch.nn.Embedding(vocab_size, embedding_dim)

# Clone weights for TTNN
weight_torch = torch_embedding.weight.data.clone()

# -----------------------------
# Create random input tokens
# -----------------------------
input_tokens = torch.randint(
    low=0,
    high=vocab_size,
    size=(batch_size, seq_len),
    dtype=torch.int64,
)

print("Input shape:", input_tokens.shape)


# -----------------------------
# PyTorch forward
# -----------------------------
torch_output = torch_embedding(input_tokens)

print("PyTorch output shape:", torch_output.shape)


# -----------------------------
# Convert weights to TTNN
# -----------------------------
weight_tt = ttnn.from_torch(
    weight_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
)

input_tt = ttnn.from_torch(
    input_tokens,
    dtype=ttnn.uint32,
    layout=ttnn.TILE_LAYOUT,
    device=device,
)


# -----------------------------
# TTNN forward
# -----------------------------
tt_output = ttnn.embedding(
    input_tt,
    weight_tt,
)

# Convert back to torch
tt_output_torch = ttnn.to_torch(tt_output)


print("TTNN output shape:", tt_output_torch.shape)


# -----------------------------
# Compare
# -----------------------------

diff = torch_output - tt_output_torch

max_error = torch.max(torch.abs(diff)).item()

print("\nMax Error:", max_error)

print("\n6.3. Comparing outputs...")
pcc_value = compute_pcc(tt_output_torch, torch_output)
max_error = compute_max_abs_error(tt_output_torch, torch_output)
mean_error = compute_mean_abs_error(tt_output_torch, torch_output)

print(f"   📊 Metrics:")
print(f"      - PCC: {pcc_value:.6f}")
print(f"      - Max Abs Error: {max_error:.6f}")
print(f"      - Mean Abs Error: {mean_error:.6f}")


# -----------------------------
# Cleanup
# -----------------------------

ttnn.close_device(device)
