"""Conv3d single-device benchmark — 35 conv layers with 4x32 Galaxy per-device shapes.

Shapes: BH 4x32 720p, inputs pre-padded (T+2 causal), ROW_MAJOR.
"""
import time
import torch
import ttnn
from models.tt_dit.utils.conv3d import aligned_channels

torch.manual_seed(42)
device = ttnn.open_device(device_id=0)
# 12x10 = 120 cores to match simulated Galaxy constraints
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
    wt = torch.randn(cout, pc, *kernel)
    tw = ttnn.from_torch(wt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0)
    return ttnn.experimental.prepare_conv3d_weights(weight_tensor=tw, C_in_block=cin_b, device=device)


def prep_b(cout):
    return ttnn.from_torch(
        torch.randn(1, cout), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0
    )


def mk(B, T, H, W, C):
    return ttnn.from_torch(
        torch.randn(B, T, H, W, aligned_channels(C)), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def c3d(x, w, b, cout, kernel, pad, cin_b, cout_b, t_b, h_b, w_b):
    cfg = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=t_b,
        W_out_block=w_b,
        H_out_block=h_b,
        C_out_block=cout_b,
        C_in_block=cin_b,
        compute_with_storage_grid_size=GRID,
    )
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
    )


print("Preparing weights...")
W, B_ = {}, {}
W["cin"] = prep_w(384, 32, (3, 3, 3), 32)
B_["cin"] = prep_b(384)
W["lat"] = prep_w(384, 384, (3, 3, 3), 96)
B_["lat"] = prep_b(384)
W["tc0"] = prep_w(768, 384, (3, 1, 1), 128)
B_["tc0"] = prep_b(768)
W["sp0"] = prep_w(192, 384, (1, 3, 3), 96)
B_["sp0"] = prep_b(192)
W["r10c1"] = prep_w(384, 192, (3, 3, 3), 96)
B_["r10c1"] = prep_b(384)
W["r1"] = prep_w(384, 384, (3, 3, 3), 96)
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

# Uncached T values: full T=81 at latent, doubles at each temporal upsample stage
# lat: T=81+2=83 (causal pad)
# tc0 input: T=81+1=82 (pre-upsample, +1 for causal conv cache slot)
# up0_spatial: T=81*2-1=161 (after temporal upsample)
# up1: T=161+2=163
# tc1 input: T=161+1=162
# up1_spatial: T=161*2-1=321
# up2/up3: T=321+2=323
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

CONVS = []
CONVS.append(("conv_in", (I["conv_in"], W["cin"], B_["cin"], 384, (3, 3, 3), (0, 1, 1), 32, 128, 1, 16, 2)))
for i in range(5):
    CONVS.append((f"lat_r{i}_c1", (I["lat"], W["lat"], B_["lat"], 384, (3, 3, 3), (0, 1, 1), 96, 96, 1, 32, 1)))
    CONVS.append((f"lat_r{i}_c2", (I["lat"], W["lat"], B_["lat"], 384, (3, 3, 3), (0, 1, 1), 96, 96, 1, 32, 1)))
CONVS.append(("tc0", (I["tc0"], W["tc0"], B_["tc0"], 768, (3, 1, 1), (0, 0, 0), 128, 256, 3, 8, 4)))
CONVS.append(("sp0", (I["sp0"], W["sp0"], B_["sp0"], 192, (1, 3, 3), (0, 1, 1), 96, 96, 1, 16, 4)))
CONVS.append(("up1_r0_c1", (I["r10c1"], W["r10c1"], B_["r10c1"], 384, (3, 3, 3), (0, 1, 1), 96, 128, 3, 8, 4)))
CONVS.append(("up1_r0_c2", (I["r1"], W["r1"], B_["r1"], 384, (3, 3, 3), (0, 1, 1), 96, 128, 1, 16, 2)))
for i in range(2):
    CONVS.append((f"up1_r{i+1}_c1", (I["r1"], W["r1"], B_["r1"], 384, (3, 3, 3), (0, 1, 1), 96, 128, 1, 16, 2)))
    CONVS.append((f"up1_r{i+1}_c2", (I["r1"], W["r1"], B_["r1"], 384, (3, 3, 3), (0, 1, 1), 96, 128, 1, 16, 2)))
CONVS.append(("tc1", (I["tc1"], W["tc1"], B_["tc1"], 768, (3, 1, 1), (0, 0, 0), 192, 384, 3, 16, 2)))
CONVS.append(("sp1", (I["sp1"], W["sp1"], B_["sp1"], 192, (1, 3, 3), (0, 1, 1), 192, 96, 1, 32, 4)))
for i in range(3):
    CONVS.append((f"up2_r{i}_c1", (I["r2"], W["r2"], B_["r2"], 192, (3, 3, 3), (0, 1, 1), 96, 96, 3, 8, 4)))
    CONVS.append((f"up2_r{i}_c2", (I["r2"], W["r2"], B_["r2"], 192, (3, 3, 3), (0, 1, 1), 96, 96, 3, 8, 4)))
CONVS.append(("sp2", (I["sp2"], W["sp2"], B_["sp2"], 96, (1, 3, 3), (0, 1, 1), 192, 96, 1, 4, 8)))
for i in range(3):
    CONVS.append((f"up3_r{i}_c1", (I["r3"], W["r3"], B_["r3"], 96, (3, 3, 3), (0, 1, 1), 96, 96, 6, 8, 4)))
    CONVS.append((f"up3_r{i}_c2", (I["r3"], W["r3"], B_["r3"], 96, (3, 3, 3), (0, 1, 1), 96, 96, 6, 8, 4)))
CONVS.append(("conv_out", (I["conv_out"], W["cout"], B_["cout"], 3, (3, 3, 3), (0, 1, 1), 96, 32, 9, 8, 4)))

assert len(CONVS) == 35, f"Expected 35 convs, got {len(CONVS)}"
print(f"Conv count: {len(CONVS)}")

# Warmup
print("Warming up...")
for label, args in CONVS:
    c3d(*args)
ttnn.synchronize_device(device)

# Timed run
print("\nRunning 35 convs...")
for i, (label, args) in enumerate(CONVS):
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    c3d(*args)
    ttnn.synchronize_device(device)
    ms = (time.perf_counter() - t0) * 1e3
    print(f"  [{i+1:02d}/35] {label:<20} {ms:.2f} ms")

ttnn.close_device(device)
