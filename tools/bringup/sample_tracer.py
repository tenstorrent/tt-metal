#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
sample_tracer.py

Runs a forward pass under TTNN tracer and dumps a graph visualization.

Key fix:
- --no-fast-runtime works reliably by setting TTNN_CONFIG_OVERRIDES
  BEFORE importing ttnn (TTNN reads config at import/init time).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Callable, Optional, Tuple

import torch


def _import_symbol(dotted: str) -> Any:
    """
    Accepts:
      - "pkg.module:Symbol" (recommended)
      - "pkg.module.Symbol"
    Returns the imported symbol.
    """
    if ":" in dotted:
        mod, sym = dotted.split(":", 1)
    else:
        parts = dotted.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid --model '{dotted}'. Use 'pkg.module:Symbol' or 'pkg.module.Symbol'")
        mod, sym = ".".join(parts[:-1]), parts[-1]
    m = importlib.import_module(mod)
    return getattr(m, sym)


def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _apply_ttnn_overrides_before_import(no_fast_runtime: bool) -> None:
    """
    Ensure TTNN sees overrides at import time.
    If user already has TTNN_CONFIG_OVERRIDES set, we merge/override the enable_fast_runtime_mode key.
    """
    if not no_fast_runtime:
        return

    existing = os.environ.get("TTNN_CONFIG_OVERRIDES")
    overrides = {}
    if existing:
        try:
            overrides = json.loads(existing)
            if not isinstance(overrides, dict):
                overrides = {}
        except Exception:
            overrides = {}

    overrides["enable_fast_runtime_mode"] = False
    os.environ["TTNN_CONFIG_OVERRIDES"] = json.dumps(overrides)


def build_model_unet_vgg19_from_ckpt(
    ckpt_path: str,
    num_classes: int = 1,
    pretrained_encoder: bool = True,
    decoder_upsample: str = "bilinear",
    gn_groups: int = 16,
    bridge_kernel_size: int = 1,
) -> torch.nn.Module:
    """
    Loads YOUR UNetVGG19 (unet_vgg19.py) and loads checkpoint with key "model" (as in your train.py).
    """
    # Local import to avoid forcing this module path for users who want --model
    from unet_vgg19 import UNetVGG19

    bilinear = decoder_upsample != "transpose"

    model = UNetVGG19(
        num_classes=int(num_classes),
        pretrained=bool(pretrained_encoder),
        bilinear=bool(bilinear),
        use_checkpoint=False,
        norm="group",
        gn_groups=int(gn_groups),
        bridge_kernel_size=int(bridge_kernel_size),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]  # <-- your train.py saves under "model"
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-shape", nargs=4, type=int, required=True, metavar=("B", "C", "H", "W"))

    # Choose ONE of these:
    ap.add_argument("--ckpt", default=None, help="Path to checkpoint (expects key 'model' by default).")
    ap.add_argument(
        "--model",
        default=None,
        help="Optional model factory/class import path, e.g. mypkg.models:build_model. "
        "If provided, we call it to create the model. If omitted, we use UNetVGG19 builder.",
    )

    # If using default UNetVGG19 builder:
    ap.add_argument("--num-classes", type=int, default=1)
    ap.add_argument("--pretrained-encoder", action="store_true", default=True)
    ap.add_argument("--no-pretrained-encoder", action="store_false", dest="pretrained_encoder")
    ap.add_argument("--decoder-upsample", choices=["bilinear", "transpose"], default="bilinear")
    ap.add_argument("--gn-groups", type=int, default=16)
    ap.add_argument("--bridge-kernel-size", type=int, default=1)

    # TTNN knobs
    ap.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="Optional: open a TT device (useful if tracing TTNN ops). Not required for pure torch forward.",
    )
    ap.add_argument("--report-name", default=None, help="Optional: ttnn.CONFIG.report_name")
    ap.add_argument(
        "--no-fast-runtime", action="store_true", help="Disable TTNN fast runtime mode (required for tracer)."
    )

    args = ap.parse_args()

    # IMPORTANT: apply env overrides BEFORE importing ttnn
    _apply_ttnn_overrides_before_import(no_fast_runtime=bool(args.no_fast_runtime))

    # Import TTNN only after overrides are in place
    import ttnn
    from ttnn.tracer import trace, visualize

    if args.report_name:
        ttnn.CONFIG.report_name = args.report_name

    # Build model
    if args.model:
        sym = _import_symbol(args.model)
        model = sym() if callable(sym) else sym
        model.eval()
    else:
        if not args.ckpt:
            raise SystemExit("Either --model or --ckpt must be provided.")
        model = build_model_unet_vgg19_from_ckpt(
            ckpt_path=args.ckpt,
            num_classes=args.num_classes,
            pretrained_encoder=args.pretrained_encoder,
            decoder_upsample=args.decoder_upsample,
            gn_groups=args.gn_groups,
            bridge_kernel_size=args.bridge_kernel_size,
        )

    # Load weights if user passed --model and --ckpt
    if args.model and args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        state = _strip_module_prefix(state)
        model.load_state_dict(state, strict=False)
        model.eval()

    x = torch.randn(tuple(args.input_shape), dtype=torch.float32)

    device = None
    try:
        if args.device_id is not None:
            device = ttnn.open_device(device_id=args.device_id)

        # Helpful sanity print (so you can confirm the override actually applied)
        print(f"[TTNN] enable_fast_runtime_mode = {ttnn.CONFIG.enable_fast_runtime_mode}")

        with torch.no_grad():
            with trace():
                y = model(x)

        visualize(y)
        print("✅ Trace complete. Graph visualization dumped by tracer.")
    finally:
        if device is not None:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
