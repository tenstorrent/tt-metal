import time, statistics, torch, ttnn, importlib
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs
from ttnn.operations.gated_delta_net import gated_delta_net

pd = importlib.import_module("ttnn.operations.gated_delta_net.gated_delta_net_program_descriptor")
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
    for split in (False, True, False, True):
        pd.SPLIT_BCAST_GROUPS = split
        pd.clear_program_cache()
        gated_delta_net(*dev, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            gated_delta_net(*dev, compute_kernel_config=cfg)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
        print(f"HOST {label} split={split}: median {statistics.median(times)*1e6:.1f} us")
ttnn.close_device(device)
