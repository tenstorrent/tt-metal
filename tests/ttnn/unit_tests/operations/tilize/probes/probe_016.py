import torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize.tilize import tilize
from models.common.utility_functions import comp_pcc

dev = ttnn.open_device(device_id=0)
orig = pd._compute_config


def precise(i, o):
    c = orig(i, o)
    c.bfp8_pack_precise = True
    return c


for name, fn in [("default", orig), ("precise", precise)]:
    pd._compute_config = fn
    for di, mk in [
        (ttnn.bfloat16, lambda s: torch.randn(s).bfloat16()),
        (ttnn.float32, lambda s: torch.randn(s)),
        (ttnn.fp8_e4m3, lambda s: torch.randn(s).to(torch.float8_e4m3fn).float()),
    ]:
        for do in [ttnn.bfloat4_b, ttnn.bfloat8_b]:
            for pv in [None, -7.0, 3.0]:
                ps = []
                for seed in range(6):
                    torch.manual_seed(seed)
                    x = mk((1, 1, 50, 50) if pv is not None else (1, 1, 64, 128))
                    t = ttnn.from_torch(x, dtype=di, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
                    y = ttnn.to_torch(tilize(t, dtype=do, pad_value=pv)).float()
                    ps.append(comp_pcc(x.float(), y, 0.5)[1])
                print("RES", name, di, do, pv, "min", round(min(ps), 5), "mean", round(sum(ps) / len(ps), 5))
ttnn.close_device(dev)
