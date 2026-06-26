import torch
import ttnn

# Test multiple RM V2 shapes
shapes_and_dims = [
    ((1, 1, 32, 4096), -1, "fp32"),
    ((1, 1, 32, 4096), -1, "bf16"),
    ((1, 1, 32, 8192), -1, "fp32"),
    ((1, 1, 2048, 256), -2, "fp32"),
    ((1, 1, 4096, 128), -2, "fp32"),
    ((1024, 1024), -1, "fp32"),
    ((1024, 1024), -2, "fp32"),
]

device = ttnn.open_device(device_id=0)
for shape, dim, dtype_str in shapes_and_dims:
    dtype = torch.float32 if dtype_str == "fp32" else torch.bfloat16
    ttnn_dtype = ttnn.float32 if dtype_str == "fp32" else ttnn.bfloat16
    torch.manual_seed(42)
    x = torch.randn(*shape, dtype=dtype)
    expected = torch.softmax(x, dim=dim)

    ttnn_input = ttnn.from_torch(x, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output = ttnn.operations.softmax.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(output)

    max_diff = (result.float() - expected.float()).abs().max().item()
    pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    status = "PASS" if pcc >= 0.999 and not (max_diff == float("inf") or max_diff == float("nan")) else "FAIL"
    print(f"{status} shape={shape} dim={dim} {dtype_str}: max_diff={max_diff:.6f} PCC={pcc:.6f}")

ttnn.close_device(device)
