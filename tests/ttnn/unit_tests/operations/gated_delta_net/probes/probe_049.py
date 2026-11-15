import time, statistics, torch, ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
import ttnn.operations.gated_delta_net.gated_delta_net as gmod
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

acc = {"cpd": [], "gop": []}
orig_cpd = gmod.cached_program_descriptor
orig_gop = ttnn.generic_op


def cpd(*a, **k):
    t = time.perf_counter()
    r = orig_cpd(*a, **k)
    acc["cpd"].append((time.perf_counter() - t) * 1e6)
    return r


def gop(*a, **k):
    t = time.perf_counter()
    r = orig_gop(*a, **k)
    acc["gop"].append((time.perf_counter() - t) * 1e6)
    return r


gmod.cached_program_descriptor = cpd
gmod.ttnn.generic_op = gop
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
    acc = {"cpd": [], "gop": []}
    times = []
    for _ in range(100):
        t = time.perf_counter()
        gated_delta_net(*dev, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        times.append((time.perf_counter() - t) * 1e6)
    print(
        "HOST median %.1f cpd %.1f gop %.1f"
        % (statistics.median(times), statistics.median(acc["cpd"]), statistics.median(acc["gop"]))
    )
finally:
    ttnn.close_device(device)
