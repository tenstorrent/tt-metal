import ttnn, torch

device = ttnn.open_device(device_id=0)
try:
    t = ttnn.from_torch(
        torch.randn(1, 1, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    a = ttnn.TensorAccessorArgs(t)
    ct = list(a.get_compile_time_args())
    print(f"REAL_COUNT={len(ct)} REAL_ARGS={ct}")
    e = ttnn.TensorAccessorArgs()
    ect = list(e.get_compile_time_args())
    print(f"EMPTY_COUNT={len(ect)} EMPTY_ARGS={ect}")
finally:
    ttnn.close_device(device)
