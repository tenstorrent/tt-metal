import time, statistics, torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

device = ttnn.open_device(device_id=0)
try:
    tensors = make_inputs(((1, 1, 2560), (1, 12, 128, 128)))
    dev = [
        ttnn.from_torch(
            tensors[n],
            dtype=ttnn.float32 if n == "recurrent_state" else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for n in ORDER
    ]
    cfg = compute_config()
    for _ in range(5):
        gated_delta_net(*dev, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    times = []
    for _ in range(200):
        t = time.perf_counter()
        gated_delta_net(*dev, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        times.append((time.perf_counter() - t) * 1e6)
    print("HOST median %.1f us p10 %.1f" % (statistics.median(times), sorted(times)[20]))
finally:
    ttnn.close_device(device)
