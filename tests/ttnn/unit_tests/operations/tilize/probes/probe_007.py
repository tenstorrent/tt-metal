import torch, ttnn
from ttnn.operations.tilize import SUPPORTED, tilize

dev = ttnn.open_device(device_id=0)
try:
    SUPPORTED["rank"] = [0, 1, 2, 3, 4, 5, 6]
    SUPPORTED["low_l1"] = [False, True]

    cases = [
        ((32, 64), "rank2", False),
        ((2, 32, 64), "rank3", False),
        ((2, 1, 3, 32, 64), "rank5", False),
        ((2, 1, 1, 3, 32, 64), "rank6", False),
        ((1, 1, 32, 4096), "low_l1_short_wide", True),
        ((1, 1, 32, 8192), "low_l1_forcing_width", True),
        ((1, 1, 2048, 2048), "low_l1_square_large", True),
    ]
    for shape, name, low in cases:
        try:
            torch.manual_seed(3)
            ti = torch.randn(shape, dtype=torch.float32).bfloat16()
            tt = ttnn.from_torch(
                ti, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            out = tilize(tt, low_l1=low)
            got = ttnn.to_torch(out).float()
            ok = torch.equal(got, ti.float())
            print(
                f"PROBE {name} shape={shape} low_l1={low}: layout={out.layout} exact={ok} shape_ok={list(out.shape)==list(shape)}"
            )
        except Exception as e:
            print(f"PROBE {name} shape={shape} low_l1={low}: RAISED {type(e).__name__}: {str(e)[:160]}")

    torch.manual_seed(3)
    ti = torch.randn((1, 1, 32, 8192), dtype=torch.float32).bfloat16()
    tt = ttnn.from_torch(
        ti, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    a = ttnn.to_torch(tilize(tt, low_l1=False))
    b = ttnn.to_torch(tilize(tt, low_l1=True))
    print("PROBE low_l1 A/B bit-identical:", torch.equal(a, b))
finally:
    ttnn.close_device(dev)
