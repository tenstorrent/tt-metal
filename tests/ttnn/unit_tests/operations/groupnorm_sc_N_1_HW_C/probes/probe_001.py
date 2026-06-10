import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:
    for shape, G in [((1, 1, 64, 128), 4), ((2, 1, 64, 288), 9)]:
        torch.manual_seed(0)
        N, _, HW, C = shape
        x = torch.randn(shape, dtype=torch.bfloat16)
        gm = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
        bt = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
        # gamma/beta in TILE layout (the candidate cell)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tg = ttnn.from_torch(gm, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tb = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # bypass validate() gate by calling internals? No — just check current rejection, then golden math via RM path for reference
        try:
            out = groupnorm_sc_N_1_HW_C(tx, G, gamma=tg, beta=tb)
            print(shape, "ran (unexpected — gate should reject)")
        except NotImplementedError as e:
            print(shape, "gated:", e)
finally:
    ttnn.close_device(device)
