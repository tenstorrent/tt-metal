import time, statistics, torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

device = ttnn.open_device(device_id=0)
try:
    for label, shapes in (
        ("decode_b1", ((1, 1, 2560), (1, 12, 128, 128))),
        ("prefill_t128", ((1, 128, 2560), (1, 12, 128, 128))),
    ):
        torch.manual_seed(0)
        t = make_inputs(shapes)
        dev = [
            ttnn.from_torch(
                t[n],
                dtype=ttnn.float32 if n == "recurrent_state" else ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            for n in ORDER
        ]
        cfg = compute_config()
        for _ in range(5):
            gated_delta_net(*dev, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            outs = gated_delta_net(*dev, compute_kernel_config=cfg)
            times.append(time.perf_counter() - t0)
            for o in outs:
                o.deallocate()
        ttnn.synchronize_device(device)
        print(f"HOST {label}: median {statistics.median(times)*1e6:.1f} us")
finally:
    ttnn.close_device(device)
