import torch
import ttnn


def test_shape(device, shape, dim, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    torch_dtype = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}[dtype]
    torch_input = torch.randn(*shape, dtype=torch_dtype)
    expected = torch.softmax(torch_input, dim=dim)
    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device)
    ttnn_output = ttnn.softmax(ttnn_input, dim=dim)
    result = ttnn.to_torch(ttnn_output)
    max_diff = (result.float() - expected.float()).abs().max().item()
    pcc_val = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    status = "PASS" if pcc_val >= 0.999 else "FAIL"
    print(f"{status} shape={shape} dim={dim}: max_diff={max_diff:.6f}, PCC={pcc_val:.6f}")
    return pcc_val >= 0.999


device = ttnn.open_device(device_id=0)
try:
    all_pass = True
    all_pass &= test_shape(device, (1, 1, 32, 4096), -1)
    all_pass &= test_shape(device, (1, 1, 32, 8192), -1)
    all_pass &= test_shape(device, (1, 1, 128, 4096), -1)
    all_pass &= test_shape(device, (1, 1, 2048, 256), -2)
    all_pass &= test_shape(device, (1, 1, 4096, 128), -2)
    all_pass &= test_shape(device, (1, 1, 32, 4096), -2)
    all_pass &= test_shape(device, (1, 1, 32, 8192), -2)
    all_pass &= test_shape(device, (2, 1, 64, 4096), -2)
    all_pass &= test_shape(device, (1, 1, 1024, 1024), -1)
    all_pass &= test_shape(device, (1, 1, 1024, 1024), -2)
    all_pass &= test_shape(device, (1, 1, 32, 64), -1)
    all_pass &= test_shape(device, (1, 1, 32, 64), -2)
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
finally:
    ttnn.close_device(device)
