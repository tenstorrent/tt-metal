import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

# Ensure sibling tool modules resolve regardless of cwd (this file may be run as
# a script from anywhere).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trace_manifest_validation import format_report, validate_manifest
from unet_vgg19 import UNetVGG19


@dataclass
class OpRecord:
    idx: int
    name: str
    kind: str
    # minimal “params” we care about for generating tests
    params: Dict[str, Any]
    # tensor meta + saved sample tensors
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]
    in_path: str
    out_path: str
    w_path: str | None = None
    b_path: str | None = None


def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def build_model_from_ckpt(ckpt_path: Path | None) -> nn.Module:
    # Matches your train.py model construction: norm="group", gn_groups=16, bridge_kernel_size=1, bilinear upsample by default.
    model = UNetVGG19(
        num_classes=1,
        pretrained=True,  # you train with pretrained_encoder=True in config :contentReference[oaicite:2]{index=2}
        bilinear=True,  # decoder_upsample="bilinear" in config :contentReference[oaicite:3]{index=3}
        use_checkpoint=False,
        norm="group",
        gn_groups=16,
        bridge_kernel_size=1,
    )

    if ckpt_path is not None:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        state = strip_module_prefix(state)
        model.load_state_dict(state, strict=True)
    else:
        print(
            "No --ckpt given: using torchvision-downloaded VGG19 encoder weights; "
            "decoder/bridge/head are randomly initialized."
        )

    model.eval()
    return model


def module_params(m: nn.Module) -> Dict[str, Any]:
    # capture enough to recreate TTNN op call signature later
    if isinstance(m, nn.Conv2d):
        return dict(
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            kernel_size=tuple(m.kernel_size),
            stride=tuple(m.stride),
            padding=tuple(m.padding),
            dilation=tuple(m.dilation),
            groups=m.groups,
            bias=(m.bias is not None),
        )
    if isinstance(m, nn.ConvTranspose2d):
        return dict(
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            kernel_size=tuple(m.kernel_size),
            stride=tuple(m.stride),
            padding=tuple(m.padding),
            output_padding=tuple(m.output_padding),
            dilation=tuple(m.dilation),
            groups=m.groups,
            bias=(m.bias is not None),
        )
    if isinstance(m, nn.GroupNorm):
        return dict(num_groups=m.num_groups, num_channels=m.num_channels, eps=m.eps, affine=m.affine)
    if isinstance(m, nn.BatchNorm2d):
        return dict(num_features=m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine)
    if isinstance(m, nn.MaxPool2d):
        return dict(
            kernel_size=m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, ceil_mode=m.ceil_mode
        )
    if isinstance(m, nn.Upsample):
        return dict(scale_factor=m.scale_factor, mode=m.mode, align_corners=m.align_corners)
    if isinstance(m, nn.ReLU):
        return dict(inplace=m.inplace)
    return dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--input-shape", nargs=4, type=int, required=True, metavar=("B", "C", "H", "W"))
    ap.add_argument("--out", default="bringup/artifacts/phase1")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    tensors_dir = out_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_ckpt(Path(args.ckpt) if args.ckpt else None)

    records: List[OpRecord] = []
    idx = 0

    def hook(name: str):
        def _h(mod: nn.Module, inp, out):
            nonlocal idx
            # keep it simple: assume single-tensor input/output for module-level ops
            x = inp[0].detach().cpu()
            y = out.detach().cpu()

            # Artifact paths are stored manifest-relative (see tools/bringup/README.md).
            # The manifest lives at out_dir/manifest.json, so "tensors/<file>" resolves
            # against the manifest directory. Tensors are saved to the real location.
            in_path = f"tensors/{idx:05d}_{name}_in.pt"
            out_path = f"tensors/{idx:05d}_{name}_out.pt"
            torch.save(x, out_dir / in_path)
            torch.save(y, out_dir / out_path)

            w_path = b_path = None
            if hasattr(mod, "weight") and isinstance(getattr(mod, "weight"), torch.Tensor):
                w_path = f"tensors/{idx:05d}_{name}_w.pt"
                torch.save(mod.weight.detach().cpu(), out_dir / w_path)
            if hasattr(mod, "bias") and isinstance(getattr(mod, "bias"), torch.Tensor) and mod.bias is not None:
                b_path = f"tensors/{idx:05d}_{name}_b.pt"
                torch.save(mod.bias.detach().cpu(), out_dir / b_path)

            rec = OpRecord(
                idx=idx,
                name=name,
                kind=type(mod).__name__,
                params=module_params(mod),
                in_shape=tuple(x.shape),
                out_shape=tuple(y.shape),
                in_path=in_path,
                out_path=out_path,
                w_path=w_path,
                b_path=b_path,
            )
            records.append(rec)
            idx += 1

        return _h

    # Register hooks on “leaf” modules we want tests for
    interesting = (nn.Conv2d, nn.ConvTranspose2d, nn.GroupNorm, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Upsample)
    for name, m in model.named_modules():
        if isinstance(m, interesting):
            m.register_forward_hook(hook(name))

    x = torch.randn(tuple(args.input_shape), dtype=torch.float32)
    with torch.no_grad():
        _ = model(x)

    # Save JSON manifest
    manifest = {
        "input_shape": list(args.input_shape),
        "num_records": len(records),
        "records": [asdict(r) for r in records],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved Phase 1 manifest: {manifest_path}")
    print(f"Saved tensors under: {tensors_dir}")

    # Self-validate the manifest we just wrote using the same rules the Phase-2
    # harness enforces, so a faulty/inconsistent manifest is never handed off.
    report = validate_manifest(manifest_path, check_shapes=True)
    print("")
    print(format_report(manifest_path, report), end="")
    if report.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
