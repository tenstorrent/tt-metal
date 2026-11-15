import time, statistics, torch, ttnn
import ttnn.operations.gated_delta_net.gated_delta_net as gm
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

device = ttnn.open_device(device_id=0)
device.enable_program_cache()

# Phase split via wrapping the module functions
phase = {"validate": 0.0, "build": 0.0, "dispatch": 0.0}
orig_validate, orig_build, orig_generic = gm.validate, gm.create_program_descriptor, ttnn.generic_op


def wrap(name, fn):
    def inner(*a, **k):
        t = time.perf_counter()
        r = fn(*a, **k)
        phase[name] += time.perf_counter() - t
        return r

    return inner


gm.validate = wrap("validate", orig_validate)
gm.create_program_descriptor = wrap("build", orig_build)
gm.ttnn.generic_op = wrap("dispatch", orig_generic)

for label, inputs in (
    ("prefill_T128", ((1, 128, 2560), (1, 12, 128, 128))),
    ("decode_B1", ((1, 1, 2560), (1, 12, 128, 128))),
):
    torch.manual_seed(0)
    tensors = make_inputs(inputs)
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
    outs = gated_delta_net(*dev, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    del outs
    walls, nosync = [], []
    for k in phase:
        phase[k] = 0.0
    N = 200
    for _ in range(N):
        t = time.perf_counter()
        outs = gated_delta_net(*dev, compute_kernel_config=compute_config())
        ttnn.synchronize_device(device)
        walls.append(time.perf_counter() - t)
        del outs
    split = {k: v / N * 1e6 for k, v in phase.items()}
    t = time.perf_counter()
    for _ in range(N):
        outs = gated_delta_net(*dev, compute_kernel_config=compute_config())
    ttnn.synchronize_device(device)
    host_only = (time.perf_counter() - t) / N * 1e6
    print(
        f"RESULT {label}: wall median {statistics.median(walls)*1e6:.1f} us, pipelined per call {host_only:.1f} us, split(us) "
        + ", ".join(f"{k}={v:.1f}" for k, v in split.items())
    )
ttnn.close_device(device)
