import time, torch, ttnn
from ttnn.operations.g6_slice.g6_slice import validate, _plan
from ttnn.operations.g6_slice.g6_slice_program_descriptor import create_program_descriptor

device = ttnn.open_device(device_id=0)
try:
    x = ttnn.from_torch(
        torch.randn(16, 4096, 1280, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    s = validate(x, [0, 0, 0], [16, 4096, 1024])
    t0 = time.perf_counter()
    for _ in range(20):
        g, b = _plan(x, s)
    t1 = time.perf_counter()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([16, 4096, 1024]), x.dtype, x.layout, device, x.memory_config())
    for _ in range(20):
        d = create_program_descriptor(x, out, g, b)
    t2 = time.perf_counter()
    print(f"PROBE plan {1e3*(t1-t0)/20:.3f} ms  descriptor {1e3*(t2-t1)/20:.3f} ms")
    d.kernels[0].common_runtime_args = [1, 2, 3]
    print("PROBE after in-place set via d.kernels[0]:", list(d.kernels[0].common_runtime_args)[:3])
    ks = d.kernels
    ks[0].common_runtime_args = [7, 8, 9]
    print("PROBE list handle:", list(d.kernels[0].common_runtime_args)[:3])
finally:
    ttnn.close_device(device)
