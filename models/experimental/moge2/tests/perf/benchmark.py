# SPDX-License-Identifier: Apache-2.0
"""MoGe-2 benchmark harness (STABLE — do not modify per experiment protocol).

Runs the current `TtMoGe` model end-to-end on chip 3 and reports the three keys
the experiment greps for:

    inference_speed: <frames/sec>   (median over N warm device-synchronized iters)
    accuracy:        <percent>      (100 * PCC of the point map vs fp32 torch reference)
    peak_dram:       <MiB>          (device DRAM peak, best-effort)

Also logs per-output PCC (points / depth-z / normal / mask / metric_scale).
Accuracy (point-map PCC) is the hard gate (>= 99.0); inference_speed is the
optimization objective. Always imports models.experimental.moge2.tt.ttnn_moge,
so model optimizations are picked up without touching this file.

Usage: python models/experimental/moge2/tests/perf/benchmark.py [--device-id N] [--iters N]
"""
import argparse, os, sys, glob, statistics, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "reference"))  # vendored moge + utils3d

import ttnn
from moge.model.v2 import MoGeModel
from models.experimental.moge2.tt.ttnn_moge import TtMoGe

CKPT = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Ruicheng--moge-2-vitl-normal/snapshots/*/model.pt"))[0]
IMAGE = os.environ.get("MOGE_IMAGE", "/home/ttuser/img.png")
NUM_TOKENS = int(os.environ.get("MOGE_NUM_TOKENS", "1800"))

# Device deployment config (not measurement logic). l1_small_size required by
# ttnn.conv2d (future on-device decoder); trace_region_size enables metal-trace.
DEVICE_PARAMS = dict(l1_small_size=32768, trace_region_size=1500000000,
                     num_command_queues=2 if int(os.environ.get("MOGE_2CQ", "0")) else 1)


def pcc(golden, calc):
    a = golden.flatten().float()
    b = calc.flatten().float()
    m = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[m], b[m]
    if a.numel() < 2:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def load_image(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=int(os.environ.get("MOGE_DEVICE_ID", "0")))
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(0)
    ref = MoGeModel.from_pretrained(CKPT).eval()
    image = load_image(IMAGE).float()

    with torch.inference_mode():
        golden = ref.forward(image, num_tokens=NUM_TOKENS)

    device = ttnn.open_device(device_id=args.device_id, **DEVICE_PARAMS)
    try:
        model = TtMoGe(ref, device)

        # ---- accuracy (vs fp32 reference raw forward) ----
        out = model(image, num_tokens=NUM_TOKENS)
        pcc_points = pcc(golden["points"], out["points"])
        pcc_depth = pcc(golden["points"][..., 2], out["points"][..., 2])
        pcc_normal = pcc(golden["normal"], out["normal"]) if "normal" in out else float("nan")
        pcc_mask = pcc(golden["mask"], out["mask"]) if "mask" in out else float("nan")
        pcc_scale = pcc(golden["metric_scale"], out["metric_scale"]) if "metric_scale" in out else float("nan")
        accuracy = 100.0 * pcc_points

        # ---- speed (warm, device-synchronized, median) ----
        for _ in range(3):
            model(image, num_tokens=NUM_TOKENS)
            ttnn.synchronize_device(device)
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            model(image, num_tokens=NUM_TOKENS)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        fps = 1.0 / med if med > 0 else 0.0

        try:
            mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
            peak_dram = mv.total_bytes_allocated_per_bank * mv.num_banks / (1024 * 1024)
        except Exception:
            peak_dram = -1.0

        print(f"pcc_points: {pcc_points:.6f}")
        print(f"pcc_depth_z: {pcc_depth:.6f}")
        print(f"pcc_normal: {pcc_normal:.6f}")
        print(f"pcc_mask: {pcc_mask:.6f}")
        print(f"pcc_metric_scale: {pcc_scale:.6f}")
        print(f"median_latency_ms: {med * 1000:.3f}")
        print(f"num_tokens: {NUM_TOKENS}")
        print(f"inference_speed: {fps:.4f}")
        print(f"accuracy: {accuracy:.4f}")
        print(f"peak_dram: {peak_dram:.2f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
