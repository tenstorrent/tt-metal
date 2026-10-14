import torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize.tilize import tilize
from models.common.utility_functions import comp_pcc

dev = ttnn.open_device(device_id=0)
dev.disable_and_clear_program_cache() if hasattr(dev, "disable_and_clear_program_cache") else None
orig = pd._compute_config


def precise(i, o):
    c = orig(i, o)
    c.bfp8_pack_precise = True
    return c


pd._compute_config = precise
for di, mk in [(ttnn.bfloat16, lambda s: torch.randn(s).bfloat16()), (ttnn.float32, lambda s: torch.randn(s))]:
    for pv in [None, -7.0]:
        ps = []
        hs = []
        for seed in range(6):
            torch.manual_seed(seed)
            x = mk((1, 1, 50, 50) if pv is not None else (1, 1, 64, 128))
            t = ttnn.from_torch(x, dtype=di, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            y = ttnn.to_torch(tilize(t, dtype=ttnn.bfloat4_b, pad_value=pv)).float()
            ps.append(comp_pcc(x.float(), y, 0.5)[1])
            xp = x.float() if pv is None else torch.nn.functional.pad(x.float(), (0, 14, 0, 14), value=pv)
            h = ttnn.to_torch(ttnn.from_torch(xp, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()[
                ..., : x.shape[-2], : x.shape[-1]
            ]
            hs.append(comp_pcc(x.float(), h, 0.5)[1])
        print("RES precise", di, pv, "dev min", round(min(ps), 5), "host min", round(min(hs), 5))
ttnn.close_device(dev)
