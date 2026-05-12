"""Per-stage PCC debug: backbone -> reassembly -> fusion -> head."""

import torch
import ttnn
import numpy as np
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from PIL import Image
from models.experimental.depth_anything_v2.tt.model_def import (
    custom_preprocessor, TtDepthAnythingV2,
)


def pcc(a, b):
    a = a.flatten().astype(float)
    b = b.flatten().astype(float)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def move(p, dev):
    if isinstance(p, ttnn.Tensor):
        return ttnn.to_device(p, dev)
    elif isinstance(p, dict):
        return {k: move(v, dev) for k, v in p.items()}
    elif isinstance(p, list):
        return [move(v, dev) for v in p]
    return p


def main():
    torch_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Large-hf", torch_dtype=torch.float32
    ).eval()
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")

    np.random.seed(42)
    img = Image.fromarray(np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8))
    pv = processor(images=img, return_tensors="pt")["pixel_values"]

    # --- PyTorch full forward ---
    with torch.no_grad():
        pt_out = torch_model(pv)
        pt_depth = pt_out.predicted_depth.squeeze().numpy()
    print(f"PT depth shape: {pt_depth.shape}")

    # --- TT full forward ---
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    params = custom_preprocessor(torch_model, "test")
    tt_model = TtDepthAnythingV2(torch_model.config, params, device)

    tt_pv = ttnn.from_torch(pv, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_out = tt_model(tt_pv)
    tt_depth_raw = ttnn.to_torch(tt_out).float().squeeze()
    print(f"TT depth raw shape: {tt_depth_raw.shape}")

    # Check shapes and compare
    if tt_depth_raw.dim() == 2:
        # Direct comparison if same size
        if tt_depth_raw.shape == torch.Size(list(pt_depth.shape)):
            p = pcc(pt_depth, tt_depth_raw.numpy())
            print(f"Direct PCC: {p:.6f}")
        else:
            # Interpolate
            tt_interp = torch.nn.functional.interpolate(
                tt_depth_raw.unsqueeze(0).unsqueeze(0),
                size=pt_depth.shape,
                mode="bicubic",
                align_corners=False,
            ).squeeze().numpy()
            p = pcc(pt_depth, tt_interp)
            print(f"Interpolated PCC: {p:.6f}")
    elif tt_depth_raw.dim() == 3:
        # (C, H, W) - take channel 0
        print(f"TT has {tt_depth_raw.shape[0]} channels")
        tt_ch0 = tt_depth_raw[0].numpy()
        tt_interp = torch.nn.functional.interpolate(
            torch.tensor(tt_ch0).unsqueeze(0).unsqueeze(0).float(),
            size=pt_depth.shape,
            mode="bicubic",
            align_corners=False,
        ).squeeze().numpy()
        p = pcc(pt_depth, tt_interp)
        print(f"Channel 0 PCC: {p:.6f}")
    else:
        print(f"Unexpected TT output dims: {tt_depth_raw.dim()}")
        # Flatten approach
        tt_flat = tt_depth_raw.numpy()
        p = pcc(pt_depth, tt_flat[:pt_depth.size])
        print(f"Flat PCC: {p:.6f}")

    # --- Check TT output statistics ---
    print(f"\nPT depth stats: min={pt_depth.min():.4f}, max={pt_depth.max():.4f}, mean={pt_depth.mean():.4f}")
    if tt_depth_raw.dim() <= 2:
        tt_np = tt_depth_raw.numpy()
    else:
        tt_np = tt_depth_raw[0].numpy()
    print(f"TT depth stats: min={tt_np.min():.4f}, max={tt_np.max():.4f}, mean={tt_np.mean():.4f}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
