# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
generate a visual computation graph for rt-detr
output is a pdf — useful for understanding the full op graph before porting to ttnn
"""

import argparse
import os
import sys

import torch
from torchviz import make_dot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RT-DETR", "rtdetr_pytorch"))
from src.core import YAMLConfig


def generate_graph(config_path, out_path="docs/rtdetr_model_graph"):
    cfg = YAMLConfig(config_path)
    model = cfg.model
    model.eval()

    dummy = torch.randn(1, 3, 640, 640)
    print(f"input shape: {dummy.shape}")

    with torch.no_grad():
        out = model(dummy)

    if isinstance(out, dict):
        # torchviz needs a single tensor; sum scalars from all outputs
        out_tensor = sum(v.sum() for v in out.values() if isinstance(v, torch.Tensor))
    else:
        out_tensor = out

    dot = make_dot(out_tensor, params=dict(model.named_parameters()))
    dot.format = "pdf"
    dot.render(out_path, cleanup=True)
    print(f"graph saved to {out_path}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml")
    parser.add_argument("--output", default="docs/rtdetr_model_graph")
    args = parser.parse_args()
    generate_graph(args.config, args.output)
