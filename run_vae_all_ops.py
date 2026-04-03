"""
Conv3d device kernel duration measurement — BH 4x32 720p per-device shapes.
Run with:
    TT_METAL_DEVICE_PROFILER=1 python run_vae_all_ops.py
Then check: generated/profiler/.logs/profile_log_device.csv
DEVICE KERNEL DURATION column summed = total conv3d device time.
"""
import os
import time
import torch
import ttnn
from models.tt_dit.utils.conv3d import aligned_channels, get_conv3d_config

torch.manual_seed(42)
device = ttnn.open_device(device_id=0)
GRID = ttnn.CoreCoord(12, 10)
CKC = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def prep_w(cout, cin, kernel, cin_b):
    pc = aligned_channels(cin)
    tw = ttnn.from_torch(
        torch.randn(cout, pc, *kernel), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )
    return ttnn.experimental.prepare_conv3d_weights(weight_tensor=tw, C_in_block=cin_b, device=device)


def prep_b(cout):
    return ttnn.from_torch(
        torch.randn(1, cout), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )


def mk(B, T, H, W, C):
    return ttnn.from_torch(
        torch.randn(B, T, H, W, aligned_channels(C)), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def c3d(x, w, b, cout, kernel, pad, cfg):
    return ttnn.experimental.conv3d(
        input_tensor=x,
        weight_tensor=w,
        bias_tensor=b,
        config=cfg,
        output_channels=cout,
        kernel_size=kernel,
        stride=(1, 1, 1),
        padding=pad,
        padding_mode="zeros",
        dtype=ttnn.bfloat16,
        compute_kernel_config=CKC,
        device=device,
    )


def cfg(cin, cout, kernel, H, W):
    c = get_conv3d_config(cin, cout, kernel, ttnn.bfloat16, GRID, h_factor=4, w_factor=32, H_out=H, W_out=W)
    c.compute_with_storage_grid_size = GRID
    return c


print("Preparing weights and inputs...")
W, B_ = {}, {}
W["cin"] = prep_w(384, 32, (3, 3, 3), 32)
B_["cin"] = prep_b(384)
W["lat"] = prep_w(384, 384, (3, 3, 3), 128)
B_["lat"] = prep_b(384)
W["tc0"] = prep_w(768, 384, (3, 1, 1), 128)
B_["tc0"] = prep_b(768)
W["sp0"] = prep_w(192, 384, (1, 3, 3), 96)
B_["sp0"] = prep_b(192)
W["r10c1"] = prep_w(384, 192, (3, 3, 3), 96)
B_["r10c1"] = prep_b(384)
W["r1"] = prep_w(384, 384, (3, 3, 3), 128)
B_["r1"] = prep_b(384)
W["tc1"] = prep_w(768, 384, (3, 1, 1), 192)
B_["tc1"] = prep_b(768)
W["sp1"] = prep_w(192, 384, (1, 3, 3), 192)
B_["sp1"] = prep_b(192)
W["r2"] = prep_w(192, 192, (3, 3, 3), 96)
B_["r2"] = prep_b(192)
W["sp2"] = prep_w(96, 192, (1, 3, 3), 192)
B_["sp2"] = prep_b(96)
W["r3"] = prep_w(96, 96, (3, 3, 3), 96)
B_["r3"] = prep_b(96)
W["cout"] = prep_w(3, 96, (3, 3, 3), 96)
B_["cout"] = prep_b(3)

I = {
    "conv_in": mk(1, 23, 23, 5, 32),
    "lat": mk(1, 23, 23, 5, 384),
    "tc0": mk(1, 22, 23, 5, 384),
    "sp0": mk(1, 41, 46, 10, 384),
    "r10c1": mk(1, 43, 46, 10, 192),
    "r1": mk(1, 43, 46, 10, 384),
    "tc1": mk(1, 42, 46, 10, 384),
    "sp1": mk(1, 81, 92, 20, 384),
    "r2": mk(1, 83, 92, 20, 192),
    "sp2": mk(1, 81, 184, 40, 192),
    "r3": mk(1, 83, 184, 40, 96),
    "conv_out": mk(1, 83, 184, 40, 96),
}

# Configs: use get_conv3d_config with actual 4x32 blocking keys
CFG = {
    "cin": cfg(32, 384, (3, 3, 3), 23, 5),
    "lat": cfg(384, 384, (3, 3, 3), 23, 5),
    "tc0": cfg(384, 768, (3, 1, 1), 23, 5),
    "sp0": cfg(384, 192, (1, 3, 3), 46, 10),
    "r10c1": cfg(192, 384, (3, 3, 3), 46, 10),
    "r1": cfg(384, 384, (3, 3, 3), 46, 10),
    "tc1": cfg(384, 768, (3, 1, 1), 46, 10),
    "sp1": cfg(384, 192, (1, 3, 3), 92, 20),
    "r2": cfg(192, 192, (3, 3, 3), 92, 20),
    "sp2": cfg(192, 96, (1, 3, 3), 184, 40),
    "r3": cfg(96, 96, (3, 3, 3), 184, 40),
    "cout": cfg(96, 3, (3, 3, 3), 184, 40),
}

# 35-op sequence matching the full uncached decoder
OPS = []
OPS.append(("conv_in", I["conv_in"], W["cin"], B_["cin"], 384, (3, 3, 3), (0, 1, 1), CFG["cin"]))
for i in range(5):
    OPS.append((f"lat_{i}a", I["lat"], W["lat"], B_["lat"], 384, (3, 3, 3), (0, 1, 1), CFG["lat"]))
    OPS.append((f"lat_{i}b", I["lat"], W["lat"], B_["lat"], 384, (3, 3, 3), (0, 1, 1), CFG["lat"]))
OPS.append(("tc0", I["tc0"], W["tc0"], B_["tc0"], 768, (3, 1, 1), (0, 0, 0), CFG["tc0"]))
OPS.append(("sp0", I["sp0"], W["sp0"], B_["sp0"], 192, (1, 3, 3), (0, 1, 1), CFG["sp0"]))
OPS.append(("up1_r0c1", I["r10c1"], W["r10c1"], B_["r10c1"], 384, (3, 3, 3), (0, 1, 1), CFG["r10c1"]))
OPS.append(("up1_r0c2", I["r1"], W["r1"], B_["r1"], 384, (3, 3, 3), (0, 1, 1), CFG["r1"]))
for i in range(2):
    OPS.append((f"up1_r{i+1}c1", I["r1"], W["r1"], B_["r1"], 384, (3, 3, 3), (0, 1, 1), CFG["r1"]))
    OPS.append((f"up1_r{i+1}c2", I["r1"], W["r1"], B_["r1"], 384, (3, 3, 3), (0, 1, 1), CFG["r1"]))
OPS.append(("tc1", I["tc1"], W["tc1"], B_["tc1"], 768, (3, 1, 1), (0, 0, 0), CFG["tc1"]))
OPS.append(("sp1", I["sp1"], W["sp1"], B_["sp1"], 192, (1, 3, 3), (0, 1, 1), CFG["sp1"]))
for i in range(3):
    OPS.append((f"up2_{i}a", I["r2"], W["r2"], B_["r2"], 192, (3, 3, 3), (0, 1, 1), CFG["r2"]))
    OPS.append((f"up2_{i}b", I["r2"], W["r2"], B_["r2"], 192, (3, 3, 3), (0, 1, 1), CFG["r2"]))
OPS.append(("sp2", I["sp2"], W["sp2"], B_["sp2"], 96, (1, 3, 3), (0, 1, 1), CFG["sp2"]))
for i in range(3):
    OPS.append((f"up3_{i}a", I["r3"], W["r3"], B_["r3"], 96, (3, 3, 3), (0, 1, 1), CFG["r3"]))
    OPS.append((f"up3_{i}b", I["r3"], W["r3"], B_["r3"], 96, (3, 3, 3), (0, 1, 1), CFG["r3"]))
OPS.append(("conv_out", I["conv_out"], W["cout"], B_["cout"], 3, (3, 3, 3), (0, 1, 1), CFG["cout"]))

assert len(OPS) == 35, f"Expected 35 ops, got {len(OPS)}"

# Warmup — not measured
print("Warmup...")
for _, *args in OPS:
    c3d(*args)
ttnn.synchronize_device(device)

# Profiled timed run — device profiler captures this if TT_METAL_DEVICE_PROFILER=1
print("Timed run (profiled)...")
t0 = time.perf_counter()
for _, *args in OPS:
    c3d(*args)
ttnn.synchronize_device(device)
wall_ms = (time.perf_counter() - t0) * 1e3

print(f"\nWall time (35 ops, single sync): {wall_ms:.1f} ms")

# Close device first — profiler CSV is written on device close
ttnn.close_device(device)

if os.environ.get("TT_METAL_DEVICE_PROFILER"):
    FREQ_MHZ = 1350
    starts, ends = {}, {}
    with open("generated/profiler/.logs/profile_log_device.csv") as f:
        next(f)
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) < 12:
                continue
            if "KERNEL" not in p[10]:
                continue
            try:
                cycles = int(p[5])
                run_id = int(p[7])
            except ValueError:
                continue
            starts[run_id] = min(starts.get(run_id, cycles), cycles)
            ends[run_id] = max(ends.get(run_id, cycles), cycles)

    dispatches = sorted(
        [(rid, (ends[rid] - starts[rid]) / FREQ_MHZ) for rid in starts if rid in ends],
        key=lambda x: x[0],
    )
    # Last 35 dispatches = the timed run (warmup dispatches have lower run_ids)
    timed_dispatches = dispatches[-35:]
    total_device_ms = sum(d for _, d in timed_dispatches) / 1000
    print(f"Device kernel time (last 35 dispatches): {total_device_ms:.1f} ms")
    print(f"  (matches python3 -m tracy DEVICE KERNEL DURATION sum)")
