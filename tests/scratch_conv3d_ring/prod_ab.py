#!/usr/bin/env python3
# Production-shape A/B for the conv3d trid ring: every ring-enabled Blackhole production entry (from the
# census) at its table shape, on the full grid, timed with the streaming profiler.
#
# SCRATCH CI copy (WH run). usage: prod_ab.py <census.jsonl> <out_prefix> [warmup] [n_timed]
#   The arm (kernel variant, TT_CONV3D_FORCE_GATHER_TRIDS) is the caller's business, like perf.py.
#   Writes <out_prefix>.meta.json: the ordered list of entries run, so the profiler CSV rows can be mapped.
import json
import os
import sys
import time
import traceback

import torch
import ttnn

census_path, out_prefix = sys.argv[1], sys.argv[2]
WARMUP = int(sys.argv[3]) if len(sys.argv) > 3 else 2
N_TIMED = int(sys.argv[4]) if len(sys.argv) > 4 else 8
ONLY = os.environ.get("PROD_AB_ONLY")  # optional comma-separated list of entry names

# Representative input sizes for tables whose keys carry no (T, H, W). Marked synthetic in the output.
SYNTHETIC_DIMS = {
    "minimax_h3": (6, 64, 64),
    "mochi.layers_conv3d": (4, 60, 106),
    "unet_3d": (8, 64, 64),
    "tt_dit._DEFAULT_BLOCKINGS": (6, 64, 64),
    "tt_dit._FP32_BLOCKINGS": (512, 1, 1),  # audio: (k,1,1) kernels on (B, T, 1, 1, C)
    "hunyuan.channel": (4, 64, 64),
    "tt_dit._SPLIT_BLOCKINGS": (1, 128, 128),  # conv2d-as-conv3d, T = 1
}


def entries():
    out = []
    for line in open(census_path):
        r = json.loads(line)
        if r.get("gather_trids", 0) == 0:
            continue
        kT, kH, kW = r["kernel"]
        # One run per distinct (source, dims); several table rows can share a probe.
        seen = set()
        for src, note in r["users"]:
            dims = None
            synthetic = False
            if note and " T" in " " + note:
                parts = {p[0]: int(p[1:]) for p in note.split() if p[0] in "THW" and p[1:].isdigit()}
                if {"T", "H", "W"} <= parts.keys():
                    dims = (parts["T"], parts["H"] + kH - 1, parts["W"] + kW - 1)  # sweep-script convention
            if dims is None:
                dims = SYNTHETIC_DIMS[src]
                synthetic = True
            key = (src, dims)
            if key in seen:
                continue
            seen.add(key)
            name = f"{src.split('.')[-1]}_{r['dtype']}_{r['C_in']}x{r['C_out']}_k{kT}{kH}{kW}_blk{'-'.join(map(str, r['blocking']))}_T{dims[0]}H{dims[1]}W{dims[2]}"
            out.append(
                dict(
                    name=name,
                    src=src,
                    note=note,
                    dtype=r["dtype"],
                    C_in=r["C_in"],
                    C_out=r["C_out"],
                    kernel=tuple(r["kernel"]),
                    blocking=tuple(r["blocking"]),
                    dims=dims,
                    synthetic=synthetic,
                    ring=r["gather_trids"],
                    bytes_per_tile=r["bytes_per_tile"],
                    inner_burst=r["inner_burst"],
                )
            )
    out.sort(key=lambda e: e["name"])
    if ONLY:
        keep = set(ONLY.split(","))
        out = [e for e in out if e["name"] in keep]
    return out


def run_entry(device, grid, e):
    cin_blk, cout_blk, t_blk, h_blk, w_blk = e["blocking"]
    kT, kH, kW = e["kernel"]
    T, H, W = e["dims"]
    c_pad = (e["C_in"] + 31) // 32 * 32
    ddt = ttnn.float32 if e["dtype"] == "fp32" else ttnn.bfloat16
    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn(1, T, H, W, c_pad, dtype=torch.float32), device=device, dtype=ddt, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    w = ttnn.from_torch(torch.randn(e["C_out"], c_pad, kT, kH, kW, dtype=torch.float32) * 0.05, dtype=ddt, pad_value=0)
    w = ttnn.experimental.prepare_conv3d_weights(weight_tensor=w, C_in_block=cin_blk, alignment=32, device=device)
    b = ttnn.from_torch(
        torch.randn(1, e["C_out"], dtype=torch.float32), device=device, dtype=ddt, layout=ttnn.TILE_LAYOUT, pad_value=0
    )
    cfg = ttnn.Conv3dConfig(
        weights_dtype=ddt,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=t_blk,
        W_out_block=w_blk,
        H_out_block=h_blk,
        C_out_block=cout_blk,
        C_in_block=cin_blk,
        compute_with_storage_grid_size=grid,
    )
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def call():
        return ttnn.experimental.conv3d(
            input_tensor=x,
            weight_tensor=w,
            device=device,
            bias_tensor=b,
            dtype=ddt,
            output_channels=e["C_out"],
            kernel_size=e["kernel"],
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
            padding_mode="zeros",
            config=cfg,
            compute_kernel_config=ck,
        )

    ttnn.synchronize_device(device)
    out = None
    t0 = time.time()
    for _ in range(WARMUP + N_TIMED):
        if out is not None:
            out.deallocate()
        out = call()
    ttnn.synchronize_device(device)
    wall = time.time() - t0
    chk = float(ttnn.to_torch(out).float().abs().sum())
    for t in (x, w, b, out):
        t.deallocate()
    return wall, chk


device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
grid = (grid.x, grid.y)


def sentinel():
    """One 1-core conv3d (a single output position) after every entry: its (prelude, main) row pair
    marks the end of the entry's rows in the profiler CSV."""
    x = ttnn.from_torch(torch.zeros(1, 3, 3, 3, 32), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    w = ttnn.from_torch(torch.zeros(32, 32, 3, 3, 3), dtype=ttnn.bfloat16, pad_value=0)
    w = ttnn.experimental.prepare_conv3d_weights(weight_tensor=w, C_in_block=32, alignment=32, device=device)
    cfg = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=1,
        W_out_block=1,
        H_out_block=1,
        C_out_block=32,
        C_in_block=32,
        compute_with_storage_grid_size=(1, 1),
    )
    ttnn.synchronize_device(device)
    outs = []
    for _ in range(3):  # three back-to-back 1-core calls: an unmistakable (prelude, 1-core main) x 3 pattern
        outs.append(
            ttnn.experimental.conv3d(
                input_tensor=x,
                weight_tensor=w,
                device=device,
                dtype=ttnn.bfloat16,
                output_channels=32,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                padding_mode="zeros",
                config=cfg,
            )
        )
    ttnn.synchronize_device(device)
    for t in (x, w, *outs):
        t.deallocate()


ran = []
for i, e in enumerate(entries()):
    try:
        wall, chk = run_entry(device, grid, e)
        e.update(status="ok", wall_s=round(wall, 2), checksum=chk)
    except Exception as ex:  # noqa: BLE001
        e.update(status=f"fail: {type(ex).__name__}: {str(ex).splitlines()[0][:200]}")
        traceback.print_exc(limit=1, file=sys.stderr)
    finally:
        sentinel()
    e["kernel"], e["blocking"], e["dims"] = list(e["kernel"]), list(e["blocking"]), list(e["dims"])
    ran.append(e)
    per_call = (
        f" wall_us_per_call={e['wall_s'] * 1e6 / (WARMUP + N_TIMED):.1f} checksum={e.get('checksum')}"
        if e["status"] == "ok"
        else ""
    )
    print(f"[{i + 1}] {e['name']} -> {e['status']}{per_call}", file=sys.stderr, flush=True)
    with open(out_prefix + ".meta.json", "w") as f:
        json.dump({"warmup": WARMUP, "n_timed": N_TIMED, "grid": grid, "entries": ran}, f, indent=1)
ttnn.close_device(device)
print("prod_ab done", file=sys.stderr)
