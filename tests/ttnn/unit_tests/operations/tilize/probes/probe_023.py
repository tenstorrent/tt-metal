import torch, ttnn
from ttnn.operations.tilize import tilize
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
for ind in [ttnn.bfloat16, ttnn.float32]:
    x = torch.randn(1, 1, 32, 64).to(torch.float32 if ind == ttnn.float32 else torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ind, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    for bf in [ttnn.bfloat8_b, ttnn.bfloat4_b]:
        for th in [32, 16, 8, 4, 2, 1]:
            y = tilize(t, dtype=bf, tile=ttnn.Tile([th, 32]) if th != 32 else None)
            print("RES", ind, bf, th, comp_pcc(x, ttnn.to_torch(y), 0.98 if bf == ttnn.bfloat4_b else 0.99)[1][-30:])
ttnn.close_device(dev)
