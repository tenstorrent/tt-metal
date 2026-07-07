import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.CreateDevice(0)

for S in [512, 640, 768, 896, 1024]:
    torch.manual_seed(42)
    B, H, D = 1, 1, 64
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q_t = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_t = ttnn.from_torch(
        k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_t = ttnn.from_torch(
        v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output = scaled_dot_product_attention(q_t, k_t, v_t)
    result = ttnn.to_torch(output)

    has_nan = result.float().isnan().any()
    has_inf = result.float().isinf().any()
    max_val = result.float().abs().max()
    print(f"S={S}: nan={has_nan}, inf={has_inf}, max_abs={max_val}")

ttnn.CloseDevice(device)
