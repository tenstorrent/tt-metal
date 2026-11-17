import torch, ttnn
from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_weights

device = ttnn.open_device(device_id=0)
try:
    H, W, c_in, c_out = 16, 16, 8, 64
    kw = dict(
        stride=1,
        padding=1,
        dilation=1,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weight_dtype=ttnn.bfloat16,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
    torch.manual_seed(0)
    w = torch.randn(c_out, c_in, 3, 3).bfloat16()
    tw = prepare_conv2d_weights(w, **kw)
    for trial in range(2):
        # Poison: a NaN RM tensor whose page stride (64 B = 32 bf16) equals the C_in=8 input's aligned page stride.
        poison = ttnn.from_torch(
            torch.full((1, 1, H * W, 32), float("nan")).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        paddr = poison.buffer_address()
        ttnn.deallocate(poison)
        x = torch.randn(1, 1, H * W, c_in).bfloat16()
        tx = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        print("poison addr", paddr, "input addr", tx.buffer_address(), "aligned page", tx.buffer_aligned_page_size())
        out = ttnn.to_torch(conv2d_3x3(tx, tw, input_height=H, input_width=W, padding=1)).float()
        ref = torch.nn.functional.conv2d(x.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2), w.float(), padding=1)
        ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
        print(
            "nan count",
            torch.isnan(out).sum().item(),
            "of",
            out.numel(),
            "max abs err (finite)",
            (out - ref)[torch.isfinite(out)].abs().max().item(),
        )
finally:
    ttnn.close_device(device)
