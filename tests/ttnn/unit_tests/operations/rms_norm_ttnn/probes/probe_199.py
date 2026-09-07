import torch, ttnn
from ttnn.operations.rms_norm_ttnn.perf_experiments.rw_overlap.rms_norm_ttnn import rms_norm_ttnn
from ttnn.operations.rms_norm_ttnn.perf_experiments.rw_overlap import rms_norm_ttnn_program_descriptor as PD

dev = ttnn.open_device(device_id=0)
print("avail L1", ttnn.get_max_worker_l1_unreserved_size())
cfg = ttnn.ComputeConfigDescriptor()
cfg.math_fidelity = ttnn.MathFidelity.HiFi2
cfg.fp32_dest_acc_en = False
cfg.math_approx_mode = False
CASES = {
    "FOCUS": (8192, 2304),
    "W1024": (8192, 1024),
    "W5120": (8192, 5120),
    "W7168": (8192, 7168),
    "SMALL": (32, 1024),
}
VAR = {
    "base": {},
    "depth3": {"CB_DEPTH_CANDIDATES": (3, 2)},
    "depth4": {"CB_DEPTH_CANDIDATES": (4, 3, 2)},
    "depth3l90": {"CB_DEPTH_CANDIDATES": (3, 2), "L1_SAFETY_FRACTION": 0.92},
    "depth4l90": {"CB_DEPTH_CANDIDATES": (4, 3, 2), "L1_SAFETY_FRACTION": 0.92},
}
for name, (H, W) in CASES.items():
    tx = torch.randn(1, 1, H, W, dtype=torch.float32).to(torch.bfloat16)
    tg = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
    x = ttnn.from_torch(
        tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    g = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    for label, knobs in VAR.items():
        saved = {k: getattr(PD, k) for k in knobs}
        for k, v in knobs.items():
            setattr(PD, k, v)
        print(f"VARIANT {name} {label}")
        try:
            rms_norm_ttnn(x, epsilon=1e-12, weight=g, compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        except Exception as e:
            print("PROBE err", type(e).__name__, str(e)[:120])
        for k, v in saved.items():
            setattr(PD, k, v)
    ttnn.deallocate(x)
    ttnn.deallocate(g)
ttnn.close_device(dev)
