import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    shape, G = (1, 1, 32, 320), 32
    x = torch.randn(shape, dtype=torch.bfloat16)
    ref = (
        torch.nn.functional.group_norm(x.float().permute(0, 3, 1, 2).reshape(shape[0], shape[3], -1), G)
        .reshape(shape[0], shape[3], shape[1], shape[2])
        .permute(0, 2, 3, 1)
    )
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ty = groupnorm_sc_N_1_HW_C(tx, G)
    y = ttnn.to_torch(ty).float()
    diff = (y - ref).abs()
    pcc = torch.corrcoef(torch.stack([y.flatten(), ref.flatten()]))[0, 1].item()
    print(f"{shape} G={G}: max_diff={diff.max():.5f} pcc={pcc:.6f}", flush=True)
finally:
    ttnn.close_device(device)
