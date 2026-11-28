import sys, importlib.util, torch, ttnn

p = "/localdev/pmilojevic/campaign/bh-lb-18/slots/chip3/ttnn/ttnn/operations/g6_permute/perf_experiments/tensix_transpose/tensix_transpose.py"
spec = importlib.util.spec_from_file_location("ttx", p)
tt = importlib.util.module_from_spec(spec)
sys.modules["ttx"] = tt
spec.loader.exec_module(tt)
device = ttnn.open_device(device_id=0)
try:
    idx = torch.arange(1024, dtype=torch.int32).to(torch.int16)  # value = w*32 + c
    x = idx.view(torch.bfloat16).reshape(1, 1, 32, 32)
    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    y = ttnn.to_torch(tt.permute(xt, tt.Cfg(variant="fused_raw", fmt="uint16")))
    got = y.contiguous().view(torch.int16).to(torch.int32).reshape(32, 32)  # [c][w]
    for c in range(32):
        print("ROW", c, " ".join(f"{int(v)//32:2d}.{int(v)%32:2d}" for v in got[c]))
finally:
    ttnn.close_device(device)
