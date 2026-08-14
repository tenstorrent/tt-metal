import torch, ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
x16 = torch.arange(16 * 32).reshape(1, 1, 16, 32).float()


def go(tag, shape, x, th, **exp_cfg):
    pd.EXPERIMENT = dict(exp_cfg)
    try:
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
        e = x.flatten()
        print(f"{tag}: maxdiff={(got-e).abs().max().item():8.1f} got[:8]={[round(v.item(),1) for v in got[:8]]}")
    except Exception as ex:
        print(f"{tag}: EXC {type(ex).__name__}: {str(ex)[:150]}")
    finally:
        pd.EXPERIMENT = {}


try:
    go("th16 base", [1, 1, 16, 32], x16, 16)
    go("th16 fp32dest", [1, 1, 16, 32], x16, 16, fp32_dest_acc_en=True)
    go("th16 dstfullsync", [1, 1, 16, 32], x16, 16, dst_full_sync_en=True)
    go("th16 both", [1, 1, 16, 32], x16, 16, fp32_dest_acc_en=True, dst_full_sync_en=True)
    x2 = torch.arange(16 * 64).reshape(1, 1, 16, 64).float()
    go("th16 2tiles base", [1, 1, 16, 64], x2, 16)
finally:
    ttnn.close_device(device)
