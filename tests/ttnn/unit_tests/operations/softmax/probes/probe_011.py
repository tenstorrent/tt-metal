import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 32, 4096)
    torch_input = torch.randn(*shape, dtype=torch.float32)
    expected = torch.softmax(torch_input, dim=-1)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.softmax(ttnn_input, dim=-1)
    result = ttnn.to_torch(ttnn_output)
    max_diff = (result.float() - expected.float()).abs().max().item()
    pcc = torch.corrcoef(torch.stack([result.float().flatten(), expected.float().flatten()]))[0, 1].item()
    print(f"shape={shape} dim=-1: max_diff={max_diff:.6f}, PCC={pcc:.6f} {'PASS' if pcc>=0.999 else 'FAIL'}")
finally:
    ttnn.close_device(device)
