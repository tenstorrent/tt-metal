#!/usr/bin/env python3
"""Run per-module PCC comparisons between PyTorch pretrained blocks and TT shim implementations.

This script runs a small set of module-level comparisons (stem, inverted residual,
classification head, regression head) using pretrained weights downloaded via
torchvision. It uses the local `ttnn` shim and `models.common.utility_functions`
for conversions and PCC computation.

Run with PYTHONPATH pointing to `experimental/ssd` and the venv python.
"""
import sys
import os

# Add repo root and experimental/ssd to PYTHONPATH so imports like
# `models.experimental` and `experimental.ssd.reference` resolve correctly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large as pretrained,
)

import ttnn

# Import TT module classes
from models.common.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor, comp_pcc

# Import TT module classes from local `tt` package inside experimental/ssd
from tt.ssd_mobilenetv3_stemlayer import TtMobileNetV3Stem
from tt.ssd_mobilenetv3_inverted_residual import TtMobileNetV3InvertedResidual
from tt.ssd_classification_head import TtSSDclassificationhead
from tt.ssd_regression_head import TtSSDregressionhead


def compare_module(torch_module, tt_module, input_tensor, name: str, pcc_threshold: float = 0.97):
    torch_module.eval()
    with torch.no_grad():
        torch_out = torch_module(input_tensor)

    # Debug: print torch output shape(s)
    if isinstance(torch_out, (list, tuple)):
        print(
            f"[DEBUG] {name} torch_out types/shapes: {[x.shape if hasattr(x, 'shape') else type(x) for x in torch_out]}"
        )
    else:
        print(f"[DEBUG] {name} torch_out shape: {torch_out.shape}")

    tt_input = torch_to_tt_tensor_rm(input_tensor, device=device)
    tt_out = tt_module(tt_input)
    # Debug: print tt output shapes
    if isinstance(tt_out, list):
        try:
            print(
                f"[DEBUG] {name} tt_out types/shapes: {[tt_to_torch_tensor(x).shape if hasattr(x, 'tt_tensor') or hasattr(x, 'shape') else type(x) for x in tt_out]}"
            )
        except Exception:
            print(f"[DEBUG] {name} tt_out (list) type: {type(tt_out)}")
    else:
        try:
            print(f"[DEBUG] {name} tt_out shape: {tt_to_torch_tensor(tt_out).shape}")
        except Exception:
            print(f"[DEBUG] {name} tt_out type: {type(tt_out)}")
    # Convert tt_out to torch for comparison; TT modules may return ttnn.Tensor
    if isinstance(tt_out, list):
        # some modules may return a list of outputs - take first
        tt_out_torch = tt_to_torch_tensor(tt_out[0])
    else:
        try:
            tt_out_torch = tt_to_torch_tensor(tt_out)
        except Exception:
            # try utility to convert
            tt_out_torch = tt_to_torch_tensor(tt_out)

    # Ensure torch tensors
    if isinstance(torch_out, tuple) or isinstance(torch_out, list):
        torch_out_t = torch_out[0]
    else:
        torch_out_t = torch_out

    passed, pcc = comp_pcc(torch_out_t, tt_out_torch, threshold=pcc_threshold)
    print(f"{name}: PCC={pcc:.6f} -> {'PASS' if passed else 'FAIL'}")
    return passed, pcc


if __name__ == "__main__":
    print("Loading pretrained PyTorch SSDLite320 MobileNetV3-Large...")
    tv_model = pretrained(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    tv_model.eval()
    state_dict = tv_model.state_dict()

    # initialize shim device
    device = ttnn.open_device(0)
    ttnn.SetDefaultDevice(device)

    print("Device:", device)

    # 1) Stem layer: torch model reference is tv_model.backbone.features[0][1]
    torch_stem = tv_model.backbone.features[0][1]
    stem_base = "backbone.features.0.1"
    # Set padding to preserve spatial dims for 3x3 conv (same padding)
    tt_stem = TtMobileNetV3Stem(
        {},
        in_channels=16,
        expanded_channels=16,
        out_channels=16,
        kernel_size=3,
        padding=1,
        stride=1,
        state_dict=state_dict,
        base_address=stem_base,
        device=device,
    )

    inp = torch.randn(1, 16, 112, 112)
    compare_module(torch_stem, tt_stem, inp, "stem", pcc_threshold=0.97)

    # 2) Inverted residual (example index): tv_model.backbone.features[0][2]
    torch_inv = tv_model.backbone.features[0][2]
    inv_base = "backbone.features.0.2"
    tt_inv = TtMobileNetV3InvertedResidual(
        {},
        in_channels=16,
        expanded_channels=64,
        out_channels=24,
        kernel_size=3,
        stride=2,
        padding=1,
        use_activation=True,
        state_dict=state_dict,
        base_address=inv_base,
        device=device,
    )
    inp2 = torch.randn(1, 16, 112, 112)
    compare_module(torch_inv, tt_inv, inp2, "inverted_residual", pcc_threshold=0.97)

    # 3) Classification head: use the classification head from reference model
    # construct TT classification head with proper in_channels and state_dict
    # The SSDLite reference in our repo uses six feature maps; we'll retrieve channel sizes
    # Prepare feature maps as list of tensors matching sizes by running the backbone
    sample_input = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        features = tv_model.backbone(sample_input)
        if isinstance(features, torch.Tensor):
            features = [features]
        else:
            features = list(features.values())

    # derive out_channels from the produced features to match exact shapes
    out_channels = [f.shape[1] for f in features]
    print(f"[DEBUG] out_channels derived from backbone run: {out_channels}")

    tt_cls = TtSSDclassificationhead(
        {},
        in_channels=out_channels,
        num_classes=91,
        state_dict=state_dict,
        base_address="head.classification_head.module_list",
        device=device,
    )
    # Convert each to TT tensor
    tt_features = [torch_to_tt_tensor_rm(f, device) for f in features]
    # Run torch classification head for comparison
    # Torch head: tv_model.head.classification_head
    torch_cls = tv_model.head.classification_head
    # For per-feature test, run the head on features and compare the concat result
    with torch.no_grad():
        torch_cls_out = torch_cls(features)
    tt_cls_out = tt_cls(tt_features)
    # Convert and compare
    tt_cls_out_torch = tt_to_torch_tensor(tt_cls_out)
    # Debug shapes
    print(f"[DEBUG] features lengths: torch={len(features)}, tt={len(tt_features)}")
    print(f"[DEBUG] per-feature shapes (torch): {[f.shape for f in features]}")
    print(f"[DEBUG] torch_cls_out shape: {torch_cls_out.shape}")
    print(f"[DEBUG] tt_cls_out_torch shape: {tt_cls_out_torch.shape}")
    # Debug: inspect TT classification head projection configs to compute
    # expected total anchors/locations
    try:
        proj_infos = []
        for i, proj in enumerate(tt_cls.projections):
            out_ch = proj["out_channels"]
            anchors_per_loc = out_ch // 91
            h = features[i].shape[2]
            w = features[i].shape[3]
            proj_infos.append((i, anchors_per_loc, h, w, anchors_per_loc * h * w))
        print(f"[DEBUG] TT projection infos (idx, anchors_per_loc, H, W, boxes): {proj_infos}")
        print(f"[DEBUG] Sum boxes TT: {sum([p[-1] for p in proj_infos])}")
    except Exception:
        pass
    passed, pcc = comp_pcc(torch_cls_out, tt_cls_out_torch, threshold=0.97)
    print(f"classification_head: PCC={pcc:.6f} -> {'PASS' if passed else 'FAIL'}")

    # 4) Regression head
    # derive num_anchors per feature from classification projection configs
    try:
        num_anchors = [proj["out_channels"] // 91 for proj in tt_cls.projections]
    except Exception:
        num_anchors = [6 for _ in out_channels]

    tt_reg = TtSSDregressionhead(
        {},
        in_channels=out_channels,
        num_anchors=num_anchors,
        num_columns=4,
        state_dict=state_dict,
        base_address="head.regression_head.module_list",
        device=device,
    )
    torch_reg = tv_model.head.regression_head
    with torch.no_grad():
        torch_reg_out = torch_reg(features)
    tt_reg_out = tt_reg(tt_features)
    tt_reg_out_torch = tt_to_torch_tensor(tt_reg_out)
    passed, pcc = comp_pcc(torch_reg_out, tt_reg_out_torch, threshold=0.97)
    print(f"regression_head: PCC={pcc:.6f} -> {'PASS' if passed else 'FAIL'}")

    print("\nPer-module PCC comparisons finished.")
    ttnn.close_device(device)

    # ===== End-to-end network comparison =====
    print("\nRunning end-to-end network comparison...")
    # prepare a sample image (use same size as model expects)
    sample_input = torch.randn(1, 3, 320, 320)

    # Torch reference: run backbone -> head to get raw head outputs
    with torch.no_grad():
        torch_features = tv_model.backbone(sample_input)
        if isinstance(torch_features, torch.Tensor):
            torch_features = [torch_features]
        else:
            torch_features = list(torch_features.values())
        torch_cls_out_full = tv_model.head.classification_head(torch_features)
        torch_reg_out_full = tv_model.head.regression_head(torch_features)

    # TT network: instantiate full TtSSD and run its internal backbone+head
    from tt.ssd import TtSSD

    tt_model = TtSSD({}, (320, 320), num_classes=91, state_dict=state_dict, base_address="", device=device)
    # run TT backbone and head
    tt_image = torch_to_tt_tensor_rm(sample_input, device)
    tt_features = tt_model.Ttbackbone(tt_image)
    if isinstance(tt_features, ttnn.Tensor):
        tt_features_list = [tt_features]
    else:
        tt_features_list = list(tt_features.values())

    tt_head_outputs = tt_model.Ttssdhead(tt_features_list)
    tt_cls_out_full = tt_to_torch_tensor(tt_head_outputs["cls_logits"])
    tt_reg_out_full = tt_to_torch_tensor(tt_head_outputs["bbox_regression"])

    # Convert TT outputs into the same shape semantics as torch outputs
    # Torch outputs are already PyTorch tensors shaped (N, total_anchors, C)
    # TT outputs returned by our shim are (Tensor) with padding similar to torch
    tt_cls_out_full = tt_cls_out_full
    tt_reg_out_full = tt_reg_out_full

    # Compute PCC for classification and regression head full outputs
    passed_cls, pcc_cls = comp_pcc(torch_cls_out_full, tt_cls_out_full, threshold=0.97)
    passed_reg, pcc_reg = comp_pcc(torch_reg_out_full, tt_reg_out_full, threshold=0.97)
    print(f"End-to-end classification head PCC={pcc_cls:.6f} -> {'PASS' if passed_cls else 'FAIL'}")
    print(f"End-to-end regression head PCC={pcc_reg:.6f} -> {'PASS' if passed_reg else 'FAIL'}")

    if passed_cls and passed_reg:
        print("\nEnd-to-end PCC checks: PASS")
    else:
        print("\nEnd-to-end PCC checks: FAIL")
