import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# S non-aligned: S_q=S_kv=47, D=64
torch.manual_seed(42)
Q = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
K = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
V = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)

tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Check what _make_padding_mask produces
from ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention import _make_padding_mask

effective_mask, is_per_head = _make_padding_mask(tQ, tK, None, is_causal=False)
print(f"effective_mask is None: {effective_mask is None}")
if effective_mask is not None:
    print(f"effective_mask shape: {list(effective_mask.shape)}")
    mask_torch = ttnn.to_torch(effective_mask)
    print(f"mask has inf: {torch.isinf(mask_torch).any().item()}")
    print(f"mask unique values: {mask_torch.unique()}")
    # Check the valid region [0, 47) - should all be 0
    print(f"mask valid region (0-46) max: {mask_torch[0,0,:47,:47].abs().max().item()}")
    print(f"mask padded region (47-63) min: {mask_torch[0,0,0,47:].min().item()}")

# Run the op and check the output
out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tQ, tK, tV)
result = ttnn.to_torch(out)

# Check which elements are inf
inf_mask = torch.isinf(result.float())
if inf_mask.any():
    # Check the first inf location
    inf_locs = inf_mask.nonzero()
    print(f"Num inf elements: {inf_locs.numel()}")
    if inf_locs.numel() > 0:
        print(f"First inf at: {inf_locs[0].tolist()}")
        print(f"Result[0,0,0,:5] = {result.float()[0,0,0,:5].tolist()}")
        print(f"Expected[0,0,0,:5] = {expected[0,0,0,:5].tolist()}")

ttnn.close_device(device)
