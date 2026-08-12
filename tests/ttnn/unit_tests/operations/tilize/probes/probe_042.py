import torch, ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize.tilize import _dispatch

device = ttnn.open_device(device_id=0)
DR = ttnn.DRAM_MEMORY_CONFIG
try:
    t = torch.randn((1, 1, 128, 256))
    for th in [16, 8]:
        for indt, name in ((ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")):
            for lev in ({}, {"bfp8_precise": 1}, {"bfp8_precise": 0}):
                try:
                    tt_in = ttnn.from_torch(
                        t, dtype=indt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DR
                    )
                    out = _dispatch(
                        tt_in, memory_config=DR, dtype=ttnn.bfloat8_b, tile=ttnn.Tile([th, 32]), levers=lev or None
                    )
                    got = ttnn.to_torch(out).float()
                    print(f"case th={th} in={name} lev={lev}: maxdiff={(got - t).abs().max().item():.5f}")
                except Exception as e:
                    print(f"case th={th} in={name} lev={lev}: FAIL {type(e).__name__}: {str(e)[:150]}")
finally:
    ttnn.close_device(device)
