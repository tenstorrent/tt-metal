import importlib, torch, ttnn

g6 = importlib.import_module("ttnn.operations.g6_slice.g6_slice")
from ttnn.operations.g6_slice import g6_slice, describe_plan

g6.SUPPORTED["dtype"] = [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.bfloat8_b, ttnn.bfloat4_b]
device = ttnn.open_device(device_id=0)
try:
    cases = [
        ("fp32_page_copy", ttnn.float32, (2, 128, 256), (0, 0, 64), (2, 128, 256)),
        ("fp32_row_runs", ttnn.float32, (1, 1, 128, 64), (0, 0, 17, 0), (1, 1, 82, 64)),
        ("fp32_col_realign", ttnn.float32, (1, 1, 96, 128), (0, 0, 17, 20), (1, 1, 90, 120)),
        ("fp32_narrow_last_col", ttnn.float32, (2, 64, 256), (0, 0, 0), (2, 64, 196)),
        ("int32_row_runs", ttnn.int32, (1, 1, 128, 64), (0, 0, 17, 0), (1, 1, 82, 64)),
        ("bf8b_page_copy", ttnn.bfloat8_b, (1, 32, 64, 2048), (0, 0, 0, 1024), (1, 32, 64, 2048)),
        ("bf4b_page_copy", ttnn.bfloat4_b, (512, 2048), (0, 512), (512, 1024)),
        ("bf8b_row_runs", ttnn.bfloat8_b, (1, 1, 128, 64), (0, 0, 17, 0), (1, 1, 82, 64)),
    ]
    for name, dt, shape, s, e in cases:
        if dt in (ttnn.int32,):
            x = torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32)
        else:
            x = torch.randn(shape, dtype=torch.float32)
        xh = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT)
        xr = ttnn.to_torch(xh)
        xd = ttnn.to_device(xh, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        exp = xr[tuple(slice(b, en) for b, en in zip(s, e))]
        try:
            plan = describe_plan(xd, list(s), list(e))
            out = ttnn.to_torch(g6_slice(xd, list(s), list(e)))
            if dt in (ttnn.bfloat8_b, ttnn.bfloat4_b):
                ok = torch.allclose(out.float(), exp.float(), atol=0.25, rtol=0.1)
                maxd = (out.float() - exp.float()).abs().max().item()
            else:
                ok = torch.equal(out, exp)
                maxd = (out.double() - exp.double()).abs().max().item()
            print(f"PROBE {name}: regime={plan['regime']} ok={ok} max_diff={maxd}")
        except Exception as ex:
            print(f"PROBE {name}: EXC {type(ex).__name__}: {str(ex)[:200]}")
finally:
    ttnn.close_device(device)
