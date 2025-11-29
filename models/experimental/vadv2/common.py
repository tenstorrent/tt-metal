# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch


def load_torch_model(torch_model, layer="", model_location_generator=None):
    # Load VADv2 transformer weights
    # Note: VAD_base.pth is incompatible (requires model architecture changes)
    # See CHECKPOINT_COMPARISON.md for details
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        # Force use of vadv2_weights_1.pth for consistency in weight verification
        # Note: VAD_tiny.pth and vadv2_weights_1.pth are identical, but we use
        # vadv2_weights_1.pth to ensure both PyTorch and TT-NN use the exact same file
        weights_path = "models/experimental/vadv2/vadv2_weights_1.pth"

        if not os.path.exists(weights_path):
            os.system("bash models/experimental/vadv2/weights_download.sh")
        backbone_path = "ckpts/fcos3d.pth"
        print(f"Loading weights from: {weights_path} (forced for consistency)")
    else:
        weights_path = (
            model_location_generator("vision-models/vad_v2", model_subdir="", download_if_ci_v2=True)
            / "vadv2_weights_1.pth"
        )
        backbone_path = "ckpts/fcos3d.pth"

    checkpoint = torch.load(weights_path, map_location="cpu")

    # Extract state_dict if checkpoint contains metadata
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        print(f"✓ Using state_dict from checkpoint (trained for {checkpoint.get('epoch', '?')} epochs)")
        torch_dict = checkpoint["state_dict"]
    else:
        torch_dict = checkpoint

    if layer == "":
        new_state_dict = {}
        for k, v in torch_dict.items():
            # Strip .0. from img_neck keys to match model's expected format
            # Checkpoint: img_neck.lateral_convs.0.conv.weight
            # Model expects: img_neck.lateral_convs.conv.weight
            k_new = k.replace("lateral_convs.0.", "lateral_convs.")
            k_new = k_new.replace("fpn_convs.0.", "fpn_convs.")
            new_state_dict[k_new] = v
    else:
        state_dict = {k: v for k, v in torch_dict.items() if (k.startswith(layer))}
        new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict, strict=False)
    torch_model.eval()

    return torch_model
