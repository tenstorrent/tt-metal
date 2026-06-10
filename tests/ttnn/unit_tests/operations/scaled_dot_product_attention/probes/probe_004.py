import torch
import ttnn

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)

    # 1) D non-aligned: shape (1,1,32,50)
    t_torch = torch.arange(32 * 50, dtype=torch.float32).reshape(1, 1, 32, 50).to(torch.bfloat16)
    t_ttnn = ttnn.from_torch(t_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print("D non-aligned (1,1,32,50):")
    print("  shape:", t_ttnn.shape)
    try:
        print("  padded_shape:", t_ttnn.padded_shape)
    except Exception as e:
        print("  padded_shape: <err>", e)
    print("  buffer_aligned_page_size():", t_ttnn.buffer_aligned_page_size())
    print("  tile_size(bf16):", ttnn.tile_size(ttnn.bfloat16))

    # Round-trip
    t_back = ttnn.to_torch(t_ttnn)
    print("  round-trip shape:", t_back.shape)
    print("  max diff:", (t_back.float() - t_torch.float()).abs().max().item())

    # 2) S_q non-aligned (1,1,47,64)
    t2 = torch.arange(47 * 64, dtype=torch.float32).reshape(1, 1, 47, 64).to(torch.bfloat16)
    t2_ttnn = ttnn.from_torch(t2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print("\nS non-aligned (1,1,47,64):")
    print("  shape:", t2_ttnn.shape)
    try:
        print("  padded_shape:", t2_ttnn.padded_shape)
    except Exception as e:
        print("  padded_shape: <err>", e)
    print("  buffer_aligned_page_size():", t2_ttnn.buffer_aligned_page_size())
    t2_back = ttnn.to_torch(t2_ttnn)
    print("  round-trip shape:", t2_back.shape)
    print("  max diff:", (t2_back.float() - t2.float()).abs().max().item())

    # 3) Check what's in the padding region — build a tensor with constant 7.0
    # everywhere logical, then check whether the padded region reads back as 0.
    # We do this by going to physical layout via RM (which exposes the padded shape)
    const_t = torch.full((1, 1, 32, 50), 7.0, dtype=torch.bfloat16)
    const_ttnn = ttnn.from_torch(const_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # Get logical content
    rt = ttnn.to_torch(const_ttnn)
    print("\nLogical round-trip of all-7s (1,1,32,50):")
    print("  shape:", rt.shape)
    print("  unique values:", torch.unique(rt).tolist())
    # Try to expose physical
    try:
        from ttnn import physical_volume
    except ImportError:
        pass
finally:
    ttnn.close_device(device)
