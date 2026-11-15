# Host wall per repeat call: HEAD vs now vs now without the gdn_role named CT arg.
import importlib.util, statistics, sys, time, torch, ttnn
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs
import ttnn.operations.gated_delta_net

gm = sys.modules["ttnn.operations.gated_delta_net.gated_delta_net"]
pd = sys.modules["ttnn.operations.gated_delta_net.gated_delta_net_program_descriptor"]
cur_cached = gm.cached_program_descriptor
spec = importlib.util.spec_from_file_location("gdn_head_pd", "/tmp/gdn_head/gated_delta_net_program_descriptor.py")
head_pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(head_pd)
RealKD = ttnn.KernelDescriptor


def NoLabelKD(*a, **kw):
    kw.pop("named_compile_time_args", None)
    return RealKD(*a, **kw)


device = ttnn.open_device(device_id=0)
cfg = compute_config()
for label, shape in (("decode_b1", ((1, 1, 2560), (1, 12, 128, 128))),):
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
    for impl in ("head", "now", "nolabel") * 3:
        ttnn.KernelDescriptor = NoLabelKD if impl == "nolabel" else RealKD
        pd.clear_program_cache()
        gm.cached_program_descriptor = head_pd.cached_program_descriptor if impl == "head" else cur_cached
        gm.gated_delta_net(*dev, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        times = []
        for _ in range(300):
            t0 = time.perf_counter()
            gm.gated_delta_net(*dev, compute_kernel_config=cfg)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
        print(f"HOST {label} {impl}: median {statistics.median(times)*1e6:.1f} us")
ttnn.KernelDescriptor = RealKD
ttnn.close_device(device)
