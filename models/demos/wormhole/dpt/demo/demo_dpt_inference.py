from __future__ import annotations

import argparse
import json
from typing import Dict

import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor

from models.common.utility_functions import comp_pcc


def parse_args():
    p = argparse.ArgumentParser(description="DPT-Large Demo (Wormhole)")
    p.add_argument("--image", type=str, help="Path to input image")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--tt-run", action="store_true")
    p.add_argument("--cpu-run", action="store_true")
    p.add_argument("--traced", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--pcc-eval", action="store_true")
    p.add_argument("--dump-perf", type=str)
    p.add_argument("--output", type=str)
    return p.parse_args()


def load_model():
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model.eval()
    return model, processor


def main():
    args = parse_args()
    model, processor = load_model()
    state = model.state_dict()

    device = None
    if args.tt_run or args.benchmark or args.pcc_eval:
        import ttnn
        device = ttnn.open_device(device_id=args.device_id)

    try:
        if args.pcc_eval:
            torch.manual_seed(0)
            x = torch.randn(1, 3, args.height, args.width)
            with torch.no_grad():
                ref = model(x).predicted_depth
            from models.experimental.dpt_large.tt_traced_pipeline import TracedDPTFull
            pipe = TracedDPTFull(state, device, batch_size=1, image_size=args.height)
            tt_out = pipe.forward_untraced(x)
            import ttnn
            tt_torch = ttnn.to_torch(tt_out).float()
            if tt_torch.shape != ref.shape:
                tt_torch = torch.nn.functional.interpolate(tt_torch.unsqueeze(1), size=ref.shape[-2:], mode="bilinear", align_corners=True).squeeze(1)
            passing, pcc = comp_pcc(ref, tt_torch, 0.99)
            print(f"PCC: {pcc}")
            print("PASS" if passing else "FAIL")
            return

        if args.image:
            from PIL import Image
            img = Image.open(args.image).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            x = inputs.pixel_values
            if x.shape[2:] != (args.height, args.width):
                x = torch.nn.functional.interpolate(x, size=(args.height, args.width), mode="bilinear", align_corners=False)
            if args.cpu_run or not args.tt_run:
                with torch.no_grad():
                    depth = model(x).predicted_depth
            else:
                from models.experimental.dpt_large.tt_traced_pipeline import TracedDPTFull
                import ttnn
                pipe = TracedDPTFull(state, device, batch_size=x.shape[0], image_size=args.height)
                pipe.compile(x)
                out = pipe.forward(x)
                depth = ttnn.to_torch(out)
            if args.output:
                from PIL import Image
                import numpy as np
                d = depth.squeeze().detach().cpu().numpy()
                d = (d - d.min()) / (d.max() - d.min() + 1e-6)
                Image.fromarray((d * 255).astype('uint8')).save(args.output)
                print(f"Saved to {args.output}")
            else:
                print("Depth shape:", tuple(depth.shape))
    finally:
        if device is not None:
            import ttnn
            ttnn.close_device(device)


if __name__ == "__main__":
    main()

