# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# validate.py -- Stage 1 validation for Depth-Anything-V2-Large on N300
#
# Produces:
#   - Depth map PNGs (input / torch-ref / ttnn side-by-side)
#   - PCC (Pearson Correlation Coefficient) per image
#   - Throughput measurement (FPS) with warmup
#   - CSV summary

import os
import sys
import time
import traceback

import numpy as np
import torch

try:
    from scipy.stats import pearsonr
except ImportError:
    print("WARNING: scipy not installed, PCC will use numpy fallback")
    def pearsonr(x, y):
        r = np.corrcoef(x, y)[0, 1]
        return r, 0.0

import ttnn
from PIL import Image

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
INPUT_SIZE = (518, 518)
WARMUP_ITERS = 10
BENCH_ITERS = 50
OUTPUT_DIR = "validation"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def depth_to_colormap(depth_np):
    """Normalize depth to 0-255 and apply a simple colormap."""
    d = depth_np.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8) * 255
    return d.astype(np.uint8)


def save_side_by_side(input_img, ref_depth, tt_depth, path):
    """Save input / reference / ttnn depth maps side by side."""
    input_np = np.array(input_img.resize((256, 256)))
    ref_vis = depth_to_colormap(ref_depth)
    tt_vis = depth_to_colormap(tt_depth)

    # Resize all to same height
    from PIL import Image as PILImage
    ref_pil = PILImage.fromarray(ref_vis).resize((256, 256))
    tt_pil = PILImage.fromarray(tt_vis).resize((256, 256))
    inp_pil = PILImage.fromarray(input_np)

    combined = PILImage.new("RGB", (256 * 3, 256))
    combined.paste(inp_pil, (0, 0))
    combined.paste(ref_pil.convert("RGB"), (256, 0))
    combined.paste(tt_pil.convert("RGB"), (512, 0))
    combined.save(path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(os.path.join(OUTPUT_DIR, "depth_maps"))

    print(f"Loading model: {MODEL_ID}")
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    # 1. Load PyTorch reference model
    torch_model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    ).eval()
    image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    print("PyTorch model loaded.")

    # 2. Open device
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    print(f"Device opened.")

    # 3. Convert weights & init TT model
    sys.path.insert(0, "/workdir/tt-metal")
    from models.experimental.depth_anything_v2.tt.model_def import (
        TtDepthAnythingV2, custom_preprocessor
    )

    print("Converting weights...")
    parameters = custom_preprocessor(torch_model, "depth_anything_v2")
    tt_model = TtDepthAnythingV2(torch_model.config, parameters, device)
    print("TT Model initialized.")

    # 4. Prepare test images (use dummy images for now)
    test_images = []
    for i in range(5):
        np.random.seed(42 + i)
        arr = np.random.randint(0, 255, (INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.uint8)
        test_images.append(Image.fromarray(arr))

    # 5. Run PyTorch reference
    print("\nRunning PyTorch reference...")
    ref_depths = []
    for img in test_images:
        inputs = image_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = torch_model(**inputs)
        ref_depth = outputs.predicted_depth.squeeze().cpu().numpy()
        ref_depths.append(ref_depth)
    print(f"  Reference done. Shape: {ref_depths[0].shape}")

    # 6. Run TT inference + PCC
    print("\nRunning TT inference...")
    pcc_results = []
    tt_depths = []

    for idx, img in enumerate(test_images):
        inputs = image_processor(images=img, return_tensors="pt")
        pv = inputs["pixel_values"]
        tt_pv = ttnn.from_torch(
            pv, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )

        try:
            tt_out = tt_model(tt_pv)
            tt_depth_torch = ttnn.to_torch(tt_out).squeeze()

            # Interpolate to match reference size
            tt_interp = torch.nn.functional.interpolate(
                tt_depth_torch.unsqueeze(0).unsqueeze(0).float(),
                size=ref_depths[idx].shape,
                mode="bicubic",
                align_corners=False,
            ).squeeze().numpy()

            tt_depths.append(tt_interp)

            # PCC
            corr, pval = pearsonr(ref_depths[idx].ravel(), tt_interp.ravel())
            pcc_results.append({"image": f"img_{idx}", "pcc": corr, "pval": pval})
            print(f"  Image {idx}: PCC = {corr:.6f}")

            # Save side-by-side
            save_side_by_side(
                img, ref_depths[idx], tt_interp,
                os.path.join(OUTPUT_DIR, "depth_maps", f"comparison_{idx}.png")
            )

        except Exception as e:
            print(f"  Image {idx}: FAILED - {e}")
            traceback.print_exc()
            pcc_results.append({"image": f"img_{idx}", "pcc": 0.0, "pval": 1.0})
            tt_depths.append(None)

    # 7. Throughput benchmark
    print(f"\nBenchmarking ({WARMUP_ITERS} warmup + {BENCH_ITERS} timed)...")
    sample_inputs = image_processor(images=test_images[0], return_tensors="pt")
    sample_pv = sample_inputs["pixel_values"]
    sample_tt = ttnn.from_torch(
        sample_pv, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    # Warmup
    for _ in range(WARMUP_ITERS):
        try:
            _ = tt_model(sample_tt)
        except Exception:
            break

    # Timed run
    start = time.perf_counter()
    successful_iters = 0
    for _ in range(BENCH_ITERS):
        try:
            _ = tt_model(sample_tt)
            successful_iters += 1
        except Exception:
            break
    elapsed = time.perf_counter() - start

    fps = successful_iters / elapsed if elapsed > 0 else 0
    print(f"  Elapsed: {elapsed:.3f}s, Successful: {successful_iters}/{BENCH_ITERS}, FPS: {fps:.2f}")

    # 8. Save CSV summary
    csv_path = os.path.join(OUTPUT_DIR, "pcc_results.csv")
    with open(csv_path, "w") as f:
        f.write("image,pcc,pval\n")
        for r in pcc_results:
            f.write(f"{r['image']},{r['pcc']:.6f},{r['pval']:.6e}\n")

    bench_csv = os.path.join(OUTPUT_DIR, "benchmark.csv")
    with open(bench_csv, "w") as f:
        f.write("device,model,resolution,warmup_iters,bench_iters,elapsed_s,fps\n")
        f.write(f"N300,DepthAnythingV2Large,{INPUT_SIZE[0]}x{INPUT_SIZE[1]},{WARMUP_ITERS},{successful_iters},{elapsed:.3f},{fps:.2f}\n")

    # 9. Save numeric outputs
    npz_path = os.path.join(OUTPUT_DIR, "depth_outputs.npz")
    save_dict = {}
    for idx, (ref, tt) in enumerate(zip(ref_depths, tt_depths)):
        save_dict[f"ref_{idx}"] = ref
        if tt is not None:
            save_dict[f"tt_{idx}"] = tt
    np.savez(npz_path, **save_dict)

    # 10. Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    valid_pccs = [r["pcc"] for r in pcc_results if r["pcc"] > 0]
    if valid_pccs:
        mean_pcc = np.mean(valid_pccs)
        print(f"  Mean PCC: {mean_pcc:.6f} (target > 0.99)")
        print(f"  Min PCC:  {min(valid_pccs):.6f}")
        print(f"  Max PCC:  {max(valid_pccs):.6f}")
    else:
        print("  No valid PCC results (inference failed)")
    print(f"  FPS:      {fps:.2f} (target >= 15)")
    print(f"  Artifacts saved to: {OUTPUT_DIR}/")
    print("=" * 60)

    # Cleanup
    ttnn.close_device(device)
    print("Device closed.")


if __name__ == "__main__":
    main()
