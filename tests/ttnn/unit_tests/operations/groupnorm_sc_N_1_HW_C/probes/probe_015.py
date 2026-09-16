import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:
    # A) 20-core ROOT + lane finalize: all-ones -> sum=640, sumsq=640 per group, mean=1, var=0 -> rstd=rsqrt(1e-5)=316.2
    x = torch.ones((1, 1, 64, 320), dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tx, 32))
    print("A max |y| =", out.abs().max().item(), "(expected 0)")
    # B) 8-core ALL_GATHER unicast: (1,1,64,128) G=4
    x = torch.ones((1, 1, 64, 128), dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tx, 4))
    print("B max |y| =", out.abs().max().item(), "(expected 0)")
finally:
    ttnn.close_device(device)
