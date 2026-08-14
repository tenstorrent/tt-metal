import torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
shape = [1, 1, 16, 32]
x = torch.arange(16 * 32).reshape(shape).float()


def go(tag, th=16, **cfg):
    saved = {}
    for k, v in cfg.items():
        saved[k] = getattr(pd, k, None)
    try:
        for k, v in cfg.items():
            setattr(pd, k, v)
        tt = ttnn.from_torch(
            x.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = tilize(
            tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            use_multicore=False,
            tile=ttnn.Tile([th, 32]),
        )
        got = ttnn.to_torch(out).float().flatten()
        exp = x.flatten()
        # map each got value back to the source position it most resembles
        pos = [int(round(v.item())) for v in got[:12]]
        print(f"{tag}: maxdiff={(got-exp).abs().max().item():7.1f} got_pos[:12]={pos}")
    except Exception as e:
        print(f"{tag}: EXC {type(e).__name__}: {str(e)[:160]}")
    finally:
        for k, v in saved.items():
            setattr(pd, k, v)


try:
    go("baseline th=16")
    go("baseline th=8", th=8)
    go("pack_precise", LEVERS={**pd.LEVERS, "pack_fast": 0})
finally:
    ttnn.close_device(device)
