import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:
    for C in (1024, 2048):
        shape = (1, 1, 32, C)
        x = torch.randn(shape, dtype=torch.bfloat16)
        gm = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
        bt = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tg = ttnn.from_torch(gm, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        tb = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        try:
            out = groupnorm_sc_N_1_HW_C(tx, 1, gamma=tg, beta=tb)
            print(f"G1 C={C}: ran OK")
        except Exception as e:
            print(f"G1 C={C}: {type(e).__name__}: {str(e)[:140]}")
finally:
    ttnn.close_device(device)
