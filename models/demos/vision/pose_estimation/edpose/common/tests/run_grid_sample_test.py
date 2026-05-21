# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone grid_sample compatibility test for ED-Pose shapes.
Run inside container:
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_grid_sample_test.py
"""

import torch
import torch.nn.functional as F
import ttnn
import time


def pytorch_ref(input_nhwc, grid):
    input_nchw = input_nhwc.permute(0, 3, 1, 2).float()
    ref_nchw = F.grid_sample(input_nchw, grid.float(), mode="bilinear", padding_mode="zeros", align_corners=False)
    return ref_nchw.permute(0, 2, 3, 1).to(torch.bfloat16)


def compute_pcc(a, b):
    fa = a.flatten().float()
    fb = b.flatten().float()
    return torch.corrcoef(torch.stack([fa, fb]))[0, 1].item()


def run_test(device, name, N, C, H_in, W_in, H_grid, W_grid):
    torch.manual_seed(42)
    input_nhwc = torch.randn(N, H_in, W_in, C, dtype=torch.bfloat16)
    grid = torch.rand(N, H_grid, W_grid, 2, dtype=torch.float32) * 2.0 - 1.0

    ref = pytorch_ref(input_nhwc, grid)

    tt_input = ttnn.from_torch(input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_grid = ttnn.from_torch(grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32)

    t0 = time.time()
    tt_output = ttnn.grid_sample(tt_input, tt_grid, mode="bilinear", align_corners=False)
    t1 = time.time()

    tt_result = ttnn.to_torch(tt_output)
    pcc = compute_pcc(ref, tt_result)
    status = "PASS" if pcc > 0.98 else "FAIL"
    elapsed = (t1 - t0) * 1000

    print(f"  {name:30s} | in=[{N},{H_in},{W_in},{C}] grid=[{N},{H_grid},{W_grid},2] | "
          f"PCC={pcc:.5f} | {elapsed:7.1f}ms | {status}")

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_grid)
    ttnn.deallocate(tt_output)
    return pcc > 0.98


def main():
    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    results = []

    # --- Small sanity checks ---
    print("=== Small sanity checks ===")
    results.append(run_test(device, "small_1", 1, 32, 8, 8, 16, 4))
    results.append(run_test(device, "small_2", 2, 32, 16, 16, 64, 4))
    results.append(run_test(device, "small_3", 8, 32, 25, 38, 100, 4))

    # --- Decoder box queries (900 points) ---
    print("\n=== Decoder box queries (900 pts) ===")
    results.append(run_test(device, "dec_box_level0", 8, 32, 200, 304, 900, 4))
    results.append(run_test(device, "dec_box_level1", 8, 32, 100, 152, 900, 4))
    results.append(run_test(device, "dec_box_level2", 8, 32, 50, 76, 900, 4))
    results.append(run_test(device, "dec_box_level3", 8, 32, 25, 38, 900, 4))
    results.append(run_test(device, "dec_box_level4", 8, 32, 13, 19, 900, 4))

    # --- Decoder pose queries (1800 points) ---
    print("\n=== Decoder pose queries (1800 pts) ===")
    results.append(run_test(device, "dec_pose_level0", 8, 32, 200, 304, 1800, 4))
    results.append(run_test(device, "dec_pose_level1", 8, 32, 100, 152, 1800, 4))
    results.append(run_test(device, "dec_pose_level2", 8, 32, 50, 76, 1800, 4))
    results.append(run_test(device, "dec_pose_level3", 8, 32, 25, 38, 1800, 4))
    results.append(run_test(device, "dec_pose_level4", 8, 32, 13, 19, 1800, 4))

    # --- Encoder (80997 query points) - may OOM ---
    print("\n=== Encoder queries (80997 pts) - may OOM ===")
    for level_name, H, W in [("level4", 13, 19), ("level3", 25, 38), ("level2", 50, 76)]:
        try:
            results.append(run_test(device, f"enc_{level_name}", 8, 32, H, W, 80997, 4))
        except Exception as e:
            print(f"  enc_{level_name:24s} | SKIPPED: {e}")
            results.append(None)

    for level_name, H, W in [("level1", 100, 152), ("level0", 200, 304)]:
        try:
            results.append(run_test(device, f"enc_{level_name}", 8, 32, H, W, 80997, 4))
        except Exception as e:
            print(f"  enc_{level_name:24s} | SKIPPED: {e}")
            results.append(None)

    # --- Summary ---
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
