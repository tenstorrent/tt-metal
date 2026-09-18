"""conv3d blocking sweep for the SD3.5 spatial-parallel VAE decoder shapes (1024x1024, width split
x4 => per-device W = 256, halo-padded to 258; conv's own H padding 1). Reuses the Wan brute-force
sweep runner on a single device of the Galaxy. Results: sweep_results_sd35_w4_1024/<layer>.json."""

import os, sys, json, pathlib

sys.path.insert(0, os.getcwd())
import ttnn
from models.tt_dit.tests.models.wan2_2.bruteforce_conv3d_sweep import run_sweep

# (name, C_in, C_out, H, W_padded, count_in_decoder)
LAYERS = [
    ("s3_res128", 128, 128, 1024, 258, 5),
    ("s3_res256_128", 256, 128, 1024, 258, 1),
    ("s3_conv_out", 128, 3, 1024, 258, 1),
    ("s2_ups256", 256, 256, 1024, 258, 1),
    ("s2_res256", 256, 256, 512, 130, 5),
    ("s2_res512_256", 512, 256, 512, 130, 1),
    ("s1_ups512", 512, 512, 512, 130, 1),
    ("s1_res512", 512, 512, 256, 66, 6),
    ("s0_ups512", 512, 512, 256, 66, 1),
    ("s0_res512", 512, 512, 128, 34, 10),
]
ONLY = os.environ.get("SWEEP_LAYERS")
if ONLY:
    LAYERS = [l for l in LAYERS if l[0] in ONLY.split(",")]
MAX_COMBOS = int(os.environ.get("SWEEP_MAX_COMBOS", "250"))
full = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8), l1_small_size=65536, trace_region_size=64 * 1024 * 1024)
try:
    dev = full.create_submeshes(ttnn.MeshShape(1, 1))[0]
    outdir = pathlib.Path("sweep_results_sd35_w4_1024")
    outdir.mkdir(exist_ok=True)
    for name, cin, cout, H, W, count in LAYERS:
        out = outdir / f"{name}_{cin}x{cout}.json"
        if out.exists():
            print(f"SKIP {name} (exists)")
            continue
        print(f"\n===== SWEEP {name}: {cin}->{cout} @ H={H} W={W} (x{count} per decode) =====", flush=True)
        try:
            run_sweep(
                dev,
                cin,
                cout,
                (1, 3, 3),
                1,
                H,
                W,
                str(out),
                stride=(1, 1, 1),
                padding=(0, 1, 0),
                h_factor=1,
                w_factor=4,
                max_combos=MAX_COMBOS,
            )
        except Exception as e:  # noqa: BLE001
            print(f"SWEEP {name} FAILED: {e}", flush=True)
    print("\n===== SUMMARY =====")
    for name, cin, cout, H, W, count in LAYERS:
        out = outdir / f"{name}_{cin}x{cout}.json"
        if out.exists():
            d = json.load(open(out))
            print(
                f"SWEEPSUM {name:16s} {cin:3d}->{cout:3d} @{H}x{W} x{count}: table {d.get('table_blocking')} {d.get('table_us')} us -> best {d.get('best_blocking')} {d.get('best_us')} us"
            )
finally:
    for s_ in full.get_submeshes():
        ttnn.close_mesh_device(s_)
    ttnn.close_mesh_device(full)
print("CONV3D_SWEEP_DONE")
