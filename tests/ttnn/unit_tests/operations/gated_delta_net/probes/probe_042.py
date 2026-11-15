# Host wall per repeat call (perf_counter around call + synchronize_device, median of 200, first excluded): HEAD vs now.
import importlib.util, statistics, sys, time, torch, ttnn
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs
import ttnn.operations.gated_delta_net

gm = sys.modules["ttnn.operations.gated_delta_net.gated_delta_net"]
cur_cached = gm.cached_program_descriptor
spec = importlib.util.spec_from_file_location("gdn_head_pd", "/tmp/gdn_head/gated_delta_net_program_descriptor.py")
head_pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(head_pd)
device = ttnn.open_device(device_id=0)
cfg = compute_config()
for label, shape in (
    ("decode_b1", ((1, 1, 2560), (1, 12, 128, 128))),
    ("prefill_t128", ((1, 128, 2560), (1, 12, 128, 128))),
):
    torch.manual_seed(0)
    tensors = make_inputs(shape)
    dev = [
        ttnn.from_torch(
            tensors[n],
            dtype=ttnn.float32 if n == "recurrent_state" else ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        for n in ORDER
    ]
    for impl, fn in (("head", head_pd.cached_program_descriptor), ("now", cur_cached)) * 2:
        gm.cached_program_descriptor = fn
        gm.gated_delta_net(*dev, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            gm.gated_delta_net(*dev, compute_kernel_config=cfg)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
        print(f"HOST {label} {impl}: median {statistics.median(times)*1e6:.1f} us")
ttnn.close_device(device)
