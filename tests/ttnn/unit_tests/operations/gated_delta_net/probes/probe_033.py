import time, torch, ttnn, importlib, cProfile, pstats
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

gm = importlib.import_module("ttnn.operations.gated_delta_net.gated_delta_net")
pd = importlib.import_module("ttnn.operations.gated_delta_net.gated_delta_net_program_descriptor")
device = ttnn.open_device(device_id=0)
tensors = make_inputs(((1, 128, 2560), (1, 12, 128, 128)))
dev = tuple(
    ttnn.from_torch(
        tensors[n],
        dtype=ttnn.float32 if n == "recurrent_state" else ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    for n in ORDER
)
cfg = compute_config()
gm._validate(dev, 1e-6, cfg, None, None)


def bench(name, fn, n=2000):
    fn()
    s = time.perf_counter()
    for _ in range(n):
        fn()
    print(f"COST {name}: {(time.perf_counter()-s)/n*1e6:.2f} us")


bench("_validate", lambda: gm._validate(dev, 1e-6, cfg, None, None))
bench("_validate fresh cfg", lambda: gm._validate(dev, 1e-6, compute_config(), None, None))
bench("compute_config()", compute_config)
bench("cfg key", lambda: pd.compute_config_key(cfg))
bench("knobs", pd.knob_values)
bench("tensor keys", lambda: [gm._tensor_key(t) for t in dev])
bench("full key", lambda: gm._cache_key(dev, None, None, 1e-6, cfg))
bench("hash key", lambda: hash(gm._cache_key(dev, None, None, 1e-6, cfg)))
print("CFG modes len", len(cfg.unpack_to_dest_mode))
ttnn.close_device(device)
